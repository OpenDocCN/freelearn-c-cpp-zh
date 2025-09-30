

# 第五章：使用标准智能指针

C++ 强调使用值进行编程。默认情况下，你的代码使用对象，而不是对象的间接引用（引用和指针）。当然，对象的间接访问是允许的，而且很少有程序从不使用这种语义，但这是一种可选的，并且需要额外的语法。*第四章* 探讨了通过析构函数和 RAII 习语将资源管理与对象生命周期相关联，展示了 C++在该方面的主要优势，即基本上所有资源（包括内存）都可以通过语言的机制隐式地处理。

C++ 允许在代码中使用原始指针，但并不积极鼓励这样做。事实上，恰恰相反——原始指针是一种低级设施，效率极高但容易误用，并且从源代码中直接推断出对*指针所指内容*的责任并不容易。从几十年前的（现在已移除的）`auto_ptr<T>`设施开始，C++社区一直在努力定义围绕低级设施（如原始指针）的抽象，通过提供清晰、定义良好的语义的类型来减少编程错误的风险。这一努力取得了显著的成功，这在很大程度上得益于 C++语言的丰富性和其创建强大且高效的抽象的能力，而不会在运行时损失速度或使用更多内存。因此，在当代 C++中，原始指针通常封装在更难误用的抽象之下，例如标准容器和智能指针，这些内容我们将在本章中探讨；未封装的原始指针主要用于表示“*这里有一个你可以使用但* *不拥有*的资源。”

本章将探讨如何使用 C++的标准智能指针类型。我们首先将了解它们是什么，然后深入探讨如何有效地使用主要智能指针类型。最后，我们将探讨那些需要“亲自动手”（如此说法）并使用原始指针的时刻，理想情况下（但不仅限于此）通过智能指针的介来实现。这应该会引导我们学习如何为特定用例选择标准智能指针，如何适当地使用它们，以及如何处理必须通过自定义机制释放的资源。在整个过程中，我们将牢记并解释我们所做选择的开销。

在本章中，我们将做以下几件事：

+   快速了解一下标准智能指针的一般概念，以形成它们存在原因的认识

+   更仔细地看看 `std::unique_ptr`，包括它是如何被用来处理标量、数组和以非典型方式分配的资源

+   查看 `std::shared_ptr` 以及这种基本但成本更高的类型的用例，以便了解何时应优先考虑替代方案

+   快速看一下`std::weak_ptr`，它是`std::shared_ptr`的伴侣，当需要模拟临时共享所有权时非常有用

+   看看哪些情况下应该使用原始指针，因为它们在 C++生态系统中仍有其位置

准备好了吗？让我们深入探讨！

# 技术要求

您可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter5`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter5).

# 标准智能指针

C++的智能指针种类相对较少。在查看标准提供的选项集之前，让我们花点时间展示我们试图解决的问题。考虑以下（故意不完整）的程序。你看到它有什么问题吗？

```cpp
class X {
   // ...
};
X *f();
void g(X *p);
void h() {
   X *p = f();
   g(p);
   delete p;
}
```

这段代码在语法上是合法的，但你不希望在当代程序中看到它。这里可能出错的地方太多了，以下是一个潜在问题的非详尽列表：

+   我们不知道`g()`是否会调用`delete p`，这可能导致在`h()`之后的第二次`delete`（在已销毁的对象上！）

+   我们不知道`g()`是否可能会抛出异常，在这种情况下，`h()`中的`delete p;`指令将永远不会被执行

+   我们不知道是否应该假设`h()`拥有`p`，也就是说，我们不知道它是否应该负责在`p`上调用`operator delete()`（也许它应该是`g()`或其他函数的责任）

+   我们不知道`p`指向的内容是否是用`new`、`new[]`或其他方式（如`malloc()`、来自其他语言的某些设施、代码库中的某些自定义实用工具等）分配的

+   我们甚至不知道`p`指向的内容是否已经动态分配；例如，`p`可能指向在`f()`中声明的全局或`static`变量（这是一个坏主意，但有些人确实这样做——例如，以非惯用方式在 C++中实现单例设计模式）

例如，比较`f()`的两种可能实现（我们可以考虑的还有很多，但这里这些就足够了）：

```cpp
X *f() { // here’s one possibility
   return new X;
}
X *f() { // here’s another
   static X x;
   return &x;
}
```

在第一种情况下，调用返回指针上的`delete`可能是有意义的，但在第二种情况下，这样做将是灾难性的。函数签名中没有任何内容明确告知客户端代码我们面临的是这种情况，还是另一种情况，甚至完全是其他情况。

作为某种“奖励”，如果有人调用`f()`而没有使用返回值会怎样？如果`f()`实现为`return new X;`或类似的内容，那么代码将发生泄漏——这确实是一个不愉快的视角。请注意，自 C++17 以来，您可以通过在`f()`的返回类型上使用`[[nodiscard]]`属性来减轻这个问题，但您仍然应该注意。从函数返回原始指针是我们主要试图避免的，尽管有时我们不得不这样做。

这里还有其他可能的陷阱，它们都有一个共同的主题——使用原始指针，我们传统上无法从源代码中知道语义是什么。更具体地说，我们无法确定谁负责指针及其指向的对象。原始指针不提供清晰的所有权信息，这多年来一直是 C++中 bug 的反复来源。

现在，考虑另一种情况，以下代码片段：

```cpp
// ...
void f() {
   X *p = new X;
   thread th0{ [p] { /* use *p */ };
   thread th1{ [p] { /* use *p */ };
   th0.detach();
   th1.detach();
}
```

在这种情况下，`f()` 分配了一个由`p`指向的`X`对象，之后两个线程`th0`和`th1`复制`p`（从而共享`p`指向的`X`对象）。最后，`th0`和`th1`被分离，这意味着线程将一直运行到完成，即使`f()`已经执行完毕。如果我们不知道`th0`和`th1`将如何结束，我们就不能明确地说哪个线程应该负责在`p`上调用`operator delete()`。这是关于指针所指对象责任不明确的问题，但与我们的第一个例子不同，因此需要不同的解决方案。

对于有明确标识的最后所有者的指针所指对象的情况，无论指针之间是否共享指针所指对象，你可能希望使用`std::unique_ptr`。在（更专业，但非常真实且相当微妙）指针所指对象至少由两个“共同所有者”共享，并且这些所有者将销毁的顺序是先验未知的情况下，`std::shared_ptr`是首选工具。以下几节将更详细地介绍这些类型的作用和意义，希望有助于你在选择智能指针类型时做出明智的选择。

## 在通过函数签名表达意图的解释中

尽管我们还没有详细研究标准智能指针，但可能适当地提供一些关于它们含义的说明，特别是对于`std::unique_ptr`和`std::shared_ptr`。这两种类型传达*所有权语义*——`std::unique_ptr`代表*唯一所有权*，而`std::shared_ptr`代表*共同所有权*（或*共享所有权*）。

理解拥有（特别是共同拥有）指针所指对象与共享指针所指对象之间的区别非常重要。考虑以下示例，该示例使用`std::unique_ptr`（尽管我们尚未介绍它，但我们正在接近这个目标）和原始指针*一起*来在类型系统中记录所有权语义：

```cpp
#include <memory>
#include <iostream>
// print_pointee() shares a pointer with the caller
// but does not take ownership
template <class T> void print_pointee(T *p) {
   if (p) std::cout << *p << ‘\n’;
}
std::unique_ptr<T> make_one(const T &arg) {
   return std::make_unique<T>(arg);
}
int main() {
   auto p = make_one(3); // p is a std::unique_ptr<int>
   print_pointee(p.get()); // caller and callee share the
                           // pointer during this call
}
```

如在介绍此示例时提到的，我们使用 `std::unique_ptr` 对象来表示所有权——`make_one()` 构造 `std::unique_ptr<T>` 并将所有权转让给调用者；然后，该调用者保持对该对象的所有权，并与他人（此处为 `print_pointee()`）共享基础指针，但不放弃对指向对象的所有权。使用但不拥有是通过原始指针来模拟的。这在一个高度简化的设置中向我们展示了拥有和共享资源之间的区别——`main()` 中的 `p` 拥有资源，但与非所有者 `p` 在 `print_pointee()` 中共享资源。这全部都是安全且符合 C++ 习惯的代码。

了解标准智能指针类型模型表示所有权的知识，我们知道只要有一个明确的最后用户使用资源，`std::unique_ptr` 往往是首选类型；它比 `std::shared_ptr`（我们将会看到）轻量得多，并且提供了适当的所有权语义。

当然，有些情况下 `std::unique_ptr` 不是一个好的选择。考虑以下简化、非线程安全且不完整的代码片段：

```cpp
class entity {
   bool taken{ false };
public:
   void take() { taken = true; }
   void release() { taken = false; }
   bool taken() const { return taken; }
   // ...
};
constexpr int N = ...;
// entities is where the entity objects live. We did
// not allocate them dynamically, but if we had we would
// have used unique_ptr<entity> as this will be the
// single last point of use for these objects
array<entity,N> entities;
class nothing_left{};
// this function returns a non-owning pointer (Chapter 6
// will cover more ergonomic options than a raw pointer)
entity * borrow_one() {
   if(auto p = find_if(begin(entities), end(entities),
               [](auto && e) { return !e.taken(); };
      p != end(entities)) {
      p->take();
      return &(*p); // non-owning pointer
   }
   throw nothing_left{};
}
```

注意，`borrow_one()` 与调用代码共享一个指针，但不共享 *所有权* ——在这种情况下，`entity` 对象的提供者仍然对这些对象的生存期负责。这既不是 `std::unique_ptr`（资源的唯一所有者）的情况，也不是 `std::shared_ptr`（资源的共同所有者）的情况。我们将看到，有其他方法可以使用原始指针来表示非所有权的指针，正如我们在 *第六章* 中将看到的那样。

这里的重要点是 *函数签名传达了意义*，并且使用传达我们意图的类型是很重要的。为了做到这一点，我们必须理解这个意图。让我们在以下章节中探索如何使用标准智能指针来发挥优势时记住这一点。

# 输入 unique_ptr

如其名称所示，`unique_ptr<T>` 对象表示对指向对象的唯一（独特）所有权。这恰好是处理动态分配内存时所有权语义的常见情况——甚至可能是最常见的情况。

考虑本章的第一个（仍然故意不完整）示例，其中我们无法从源代码中确定 *指向对象的所有权，并让我们用 `unique_ptr` 对象而不是原始指针来重写它：

```cpp
#include <memory>
class X {
   // ...
};
std::unique_ptr<X> f();
void g(std::unique_ptr<X>&);
void h() {
   // we could write std::unique_ptr<X> instead of auto
   auto p = f();
   g(p);
} f() is responsible for the lifetime of the X object it points to, and it’s also clear that g() uses the enclosed X* without becoming responsible for the pointed-to X object. Add to this the fact that p is an object and, as such, will be destroyed if g() throws or if f() is called in such a way that the calling code forgets to use the return value, and you get an exception-safe program – one that’s shorter and simpler than the original one!
Murphy and Machiavelli
You might be thinking, “*But I’m sure I could steal the pointer managed by the* `std::unique_ptr` *in* `g()`,” and you would be correct. Not only is it possible but also easy, as `unique_ptr` gives you direct access to the underlying pointer in more than one way. However, the type system is designed to protect us from accidents and make reasonable well-written code work well. It will protect you from Murphy, the accidents that happen, not from Machiavelli, the deliberately hostile code.
If you write deliberately broken code, you will end up with a deliberately broken program. It’s pretty much what you would expect.
In terms of semantics, you could tell a story just with function signatures, using `std::unique_ptr` objects. Note that in the following example, the functions have been left deliberately incomplete to make it clear that we are concerned with their signatures only:

```

// ...

// 动态创建一个 X 或其派生类

// X 并返回它，而不会存在泄漏的风险

unique_ptr<X> factory(args);

// 值传递，这在实践中意味着移动传递

// 由于 unique_ptr 不可复制

unique_ptr<X> borrowing(unique_ptr<X>);

// 通过引用传递以允许修改指向对象。在

// 实践中，X* 会是一个更好的选择

void possible_mutation(unique_ptr<X>&);

// 通过引用到 const 传递以查询指向对象，但

// 不要修改它。在实践中，这里更喜欢 const X*

void consult(const unique_ptr<X>&);

// sink() 消耗作为参数传递的对象 : 获取

// in, never gets out. This could use pass-by-value but

// 用右值引用可能更清晰

void sink(unique_ptr<X> &&);

// ...

```cpp

 As we can see, function signatures talk to us. It’s better if we pay attention.
Handling objects
The `unique_ptr` type is a remarkable tool, one you should strive to get acquainted with if you have not done so already. Here are some interesting facts about that type and how it can be used to manage pointers to objects.
A `unique_ptr<T>` object is non-copyable, as its copy constructor and copy assignment member functions are marked as deleted. That’s why `g()` in the first example of the *Type unique_ptr* section takes its argument by reference – `g()` shares the pointee with the caller but does not take ownership of it. We could also have expressed `g()` as taking `X*` as an argument, with the contemporary acceptance that function arguments that are raw pointers are meant to model using a pointer but without owning it:

```

#include <memory>

class X {

// ...

};

std::unique_ptr<X> f();

void g(X*);

void h() {

// 我们可以写 std::unique_ptr<X> 而不是 auto

auto p = f();

g(p.get());

} // p 在这里隐式释放了指向的 X 对象

```cpp

 `unique_ptr<T>` is also movable – a moved-from `unique_ptr<T>` behaves like a null pointer, as the movement for this type semantically implements a transfer of ownership. This makes it simpler to implement various types that need to manage resources indirectly.
Consider, for example, the following `solar_system` class, which supposes a hypothetical `Planet` type as well as a hypothetical implementation for `create_planet()`:

```

#include “planet.h”

#include <memory>

#include <string>

#include <vector>

std::unique_ptr<Planet>

create_planet(std::string_view name);

class solar_system {

std::vector<std::unique_ptr<Planet>> planets {

create_planet(“mercury.data”),

create_planet(“venus.data”), // 等。

};

public:

// solar_system 默认是不可复制的

// solar_system 默认是可移动的

// 无需写 ~solar_system，因为行星

// 管理其资源隐式

};

```cpp

 If we had decided to implement `solar_system` with `vector<Planet*>` or as `Planet*` instead, then the memory management of our type would have to be performed by `solar_system` itself, adding to the complexity of that type. Since we used a `vector<unique_ptr<Planet>>`, everything is implicitly correct by default. Of course, depending on what we are doing, `vector<Planet>` might be even better, but let’s suppose we need pointers for the sake of the example.
A `unique_ptr<T>` offers most of the same operations as `T*`, including `operator*()` and `operator->()`, as well as the ability to compare them with `==` or `!=` to see whether two `unique_ptr<T>` objects point to the same `T` object. The latter two might seem strange, as the type represents sole ownership of the *pointee*, but you could use references to `unique_ptr<T>`, in which case these functions make sense:

```

#include <memory>

template <class T>

bool point_to_same(const std::unique_ptr<T> &p0,

const std::unique_ptr<T> &p1) {

return p0 == p1;

}

template <class T>

bool have_same_value(const std::unique_ptr<T> &p0,

const std::unique_ptr<T> &p1) {

return p0 && p1 && *p0 == *p1;

}

#include <cassert>

int main() {

// 两个指向具有相同值的对象的独立指针

std::unique_ptr<int> a{ new int { 3 } };

std::unique_ptr<int> b{ new int { 3 } };

assert(point_to_same(a, a) && have_same_value(a, a));

assert(!point_to_same(a, b) && have_same_value(a, b));

}

```cpp

 For good reasons, you cannot do pointer arithmetic on `unique_ptr<T>`. If you need to do pointer arithmetic (and we sometimes will – for example, when we write our own containers in *Chapter 13*), it’s always possible to get to the raw pointer owned by a `unique_pointer<T>` through its `get()` member function. This is often useful when interfacing with C libraries, making system calls, or calling functions that use a raw pointer without taking ownership of it.
Oh, and here’s a fun fact – `sizeof(unique_ptr<T>)==sizeof(T*)` with a few exceptions that will be discussed later in this chapter. This means that there’s generally no cost in terms of memory space to using a smart pointer instead of a raw pointer. In other words, by default, the only state found in a `unique_ptr<T>` object is `T*`.
Handling arrays
A nice aspect of `unique_ptr` is that it offers a specialization to handle arrays. Consider the following:

```

void f(int n) {

// p 指向一个值为 3 的 int

std::unique_ptr<int> p{ new int{ 3 } };

// q 指向一个 n 个 int 对象的数组

// 初始化为零

std::unique_ptr<int[]> q{ new int[n] {} };

// 示例用法

std::cout << *p << ‘\n’; // 显示 3

for(int i = 0; i != n; ++i) {

// operator[] 对 unique_ptr<T[]> 支持操作

q[i] = i + 1;

}

// ...

} // q 的析构函数在其指针上调用 delete []

// p 的析构函数在其指针上调用 delete

```cpp

 What, you might think, is the use case for this? Well, it all depends on your needs. For example, if you require a variable-sized array of `T` that grows as needed, use `vector<T>`. It’s a wonderful tool and extremely efficient if used well.
If you want a fixed-sized array that’s small enough to fit on your execution stack where the number of elements, `N`, is known at compile time, use a raw array of `T` or an object of type `std::array<T,N>`.
If you want a fixed-sized array that’s either not small enough to fit on your execution stack or where the number of elements, `n`, is known at runtime, you can use `vector<T>`, but you’ll pay for facilities you might not require (`vector<T>` remains an awesome choice, that being said), or you could use `unique_ptr<T[]>`. Note that if you go for this latter option, you will end up having to track the size yourself, separately from the actual array, since `unique_ptr` does no such tracking. Alternatively, of course, you can wrap it in your own abstraction, such as `fixed_size_array<T>`, as follows:

```

#include <cstddef>

#include <memory>

template <class T>

class fixed_size_array {

std::size_t nelems{};

std::unique_ptr<T[]> elems {};

public:

fixed_size_array() = default;

auto size() const { return nelems; }

bool empty() const { return size() == 0; }

fixed_size_array(std::size_t n)

: nelems { n }, elems{ new T[n] {} } {

}

T& operator[](int n) { return elems[n]; }

const T& operator[](int n) const { return elems[n]; }

// 等。

};

```cpp

 This is a naïve implementation that brings together knowledge of the number of elements with implicit ownership of the resource. Note that we don’t have to write the copy operations (unless we want to implement them!), the move operations, or the destructor, as they all implicitly do something reasonable. Also, this type will be relatively efficient if type `T` is trivially constructible but will (really) not be as efficient as `vector<T>` for numerous use cases. Why is that? Well, it so happens that `vector` does significantly better memory management than we do… but we’ll get there.
Note that, as with scalar types, the fact that `sizeof(unique_ptr<T[]>)` is equal to `sizeof(T*)` is also true, which I’m sure we can all appreciate.
Custom deleters
You might think, “*Well, in my code base, we don’t use* `delete` *to deallocate objects because [insert your favorite reason here], so I cannot use* `unique_ptr`.” There are indeed many situations where applying `operator delete` on a pointer to destroy the pointed-to object is not an option:

*   Sometimes, `T::~T()` is `private` or `protected`, making it inaccessible to other classes such as `unique_ptr<T>`.
*   Sometimes, the finalization semantics require doing something else than calling `delete` – for example, calling a `destroy()` or `release()` member function
*   Sometimes, the expectation is to call a free function that will perform auxiliary work in addition to freeing a resource.

No matter what the reasons are for freeing a resource in an unconventional manner, `unique_ptr<T>` can take a `T*` stored within `unique_ptr<T>` when the destructor of that smart pointer is called. Indeed, the actual signature of the `unique_ptr` template is as follows:

```

template<class T, class D = std::default_delete<T>>

class unique_ptr {

// ...

};

```cpp

 Here, `default_delete<T>` itself is essentially the following:

```

template<class T> struct default_delete {

constexpr default_delete() noexcept = default;

// ...

constexpr void operator()(T *p) const { delete p; }

};

```cpp

 The presence of a default type for `D` is what usually allows us to write code that ignores that parameter. The `D` parameter in the `unique_ptr<T,D>` signature is expected to be stateless, as it’s not stored within the `unique_ptr` object but instantiated as needed, and then it’s used as a function that takes the pointer and does whatever is required to finalize the *pointee*.
As such, imagine the following class with a `private` destructor, a common technique if you seek to prevent instantiation through other means than dynamic allocation (you cannot use an automatic or a static object of that type, since it cannot be implicitly destroyed):

```

#include <memory>

class requires_dynamic_alloc {

~requires_dynamic_alloc() = default; // 私有

// ...

friend struct cleaner;

};

// ...

struct cleaner {

template <class T>

void operator()(T *p) const { delete p; }

};

int main() {

using namespace std;

// requires_dynamic_alloc r0; // 不行

//auto p0 = unique_ptr<requires_dynamic_alloc>{

//   new requires_dynamic_alloc

//}; // 不行，因为默认删除器无法访问 delete

auto p1 = unique_ptr<requires_dynamic_alloc, cleaner>{

new requires_dynamic_alloc

}; // 好的，将使用 cleaner::operator() 来删除指针

}

```cpp

 Note that by making the `cleaner` functor its friend, the `requires_dynamic_alloc` class lets `cleaner` specifically access both its `protected` and `private` members, which includes access to its `private` destructor.
Imagine now that we are using an object through an interface that hides from client code information on whether we are the sole owner of the pointed-to resource, or whether we share that resource with others. Also, imagine that the potential sharing of that resource is done through intrusive means, as is done on many platforms, such that the way to signal that we are disconnecting from that resource is to call its `release()` member function, which will, in turn, either take into account that we have disconnected or free the resource if we were its last users. To simplify client code, our code base has a `release()` free function that calls the `release()` member function on such a pointer if it is non-null.
We can still use `unique_ptr` for this, but note the syntax, which is slightly different, as we will need to pass the function pointer as an argument to the constructor, since that pointer will be stored within. Thus, this specialization of `unique_ptr` with a function pointer as a *deleter* leads to a slight size increase:

```

#include <memory>

struct releasable {

void release() {

// 为了本例简化过度

delete this;

}

protected:

~releasable() = default;

};

class important_resource : public releasable {

// ...

};

void release(releasable *p) {

if(p) p->release();

}

int main() {

using namespace std;

auto p = unique_ptr<important_resource,

void(*)(releasable*)>{

new important_resource, release

}; // 好的，将使用 release() 来删除指针

}

```cpp

 If the extra cost of a function pointer’s size (plus alignment) in the size of `unique_ptr` is unacceptable (for example, because you are on a resource-constrained platform or because you have a container with many `unique_ptr` objects, which makes the costs increase significantly faster), there’s a neat trick you can play by pushing the runtime use of the `deleter` function into the wonderful world of the type system:

```

#include <memory>

struct releasable {

void release() {

// 为了本例简化过度

delete this;

}

protected:

~releasable() = default;

};

class important_resource : public releasable {

// ...

};

void release(releasable *p) {

if(p) p->release();

}

int main() {

using namespace std;

auto p = unique_ptr<important_resource,

void(*)(releasable*)>{

new important_resource, release

}; // 好的，将使用 release() 来删除指针

static_assert(sizeof(p) > sizeof(void*));

auto q = unique_ptr<

important_resource,

decltype([](auto p) { release(p); })}{

new important_resource

};

static_assert(sizeof(q) == sizeof(void*));

}

```cpp

 As you can see, in the case of `p`, we used a function pointer as a deleter, which requires storing the address of the function, whereas with `q`, we replaced the function pointer with the *type of a hypothetical lambda*, which will, when instantiated, call that function, passing the pointer as an argument. It’s simple and can save space if used judiciously!
make_unique
Since C++14, `unique_ptr<T>` has been accompanied by a factory function that perfectly forwards its arguments to a constructor of `T`, allocates and constructs the `T` as well as `unique_ptr<T>` to hold it, and returns the resulting object. That function is `std::make_unique<T>(args...)`, and a naïve implementation would be as follows:

```

template <class T, class ... Args>

std::unique_ptr<T> make_unique(Args &&... args) {

return std::unique_ptr<T>{

new T(std::forward<Args>(args)...);

}

}

```cpp

 There are also variants to create a `T[]`, of course. You might wonder what the point of such a function is, and indeed, that function was not shipped along with `unique_ptr` initially (`unique_ptr` is a C++11 type), but consider the following (contrived) example:

```

template <class T>

class pair_with_alloc {

T *p0, *p1;

public:

pair_with_alloc(const T &val0, const T &val1)

: p0{ new T(val0) }, p1{ new T(val1) } {

}

~pair_with_alloc() {

delete p1; delete p0;

}

// 复制和移动操作留给你的想象

};

```cpp

 We can suppose from this example that this class is used when, for some reason, client code prefers to dynamically allocate the `T` objects (in practice, using objects rather than pointers to objects makes your life simpler). Knowing that subobjects in a C++ object are constructed in order of declaration, we know that `p0` will be constructed before `p1`:

```

// ...

T *p0, *p1; // p0 在 p1 之前声明

public:

// 下面：

// - new T(val0) 将在 p0 构造之前发生

// - new T(val1) 将在 p1 构造之前发生

// - p0 的构造将在 p1 的构造之前发生

pair_with_alloc(const T &val0, const T &val1)

: p0{ new T(val0) }, p1{ new T(val1) } {

}

// ...

```cpp

 However, suppose that the order of operations is `new T(val0)`, the construction of `p0`, `new T(val1)`, and the construction of `p1`. What happens then if `new T(val1)` throws an exception, either because `new` fails to allocate sufficient memory or because the constructor of `T` fails? You might be tempted to think that the destructor of `pair_with_alloc` will clean up, but that will not be the case – for a destructor to be called, the corresponding constructor must have completed first; otherwise, there is no object to destroy!
There are ways around this ,of course. One of them might be to use `unique_ptr<T>` instead of `T*`, which would be wonderful, given that this is what we’re currently discussing! Let’s rewrite `pair_with_alloc` that way:

```

#include <memory>

template <class T>

class pair_with_alloc {

std::unique_ptr<T> p0, p1;

public:

pair_with_alloc(const T &val0, const T &val1)

: p0{ new T(val0) }, p1{ new T(val1) } {

}

// 析构函数隐式正确

// 复制和移动操作隐式工作

// 或留给你的想象

};

```cpp

 With this version, if the order of operations is `new T(val0)`, the construction of `p0`, `new T(val1)`, the construction of `p1`, then if `new T(val1)` throws an exception, the `pair_with_alloc` object will still not be destroyed (it has not been constructed). However, `p0` itself *has* been constructed by that point, and as such, it will be destroyed. Our code has suddenly become simpler and safer!
What then has that to do with `make_unique<T>()`? Well, there’s a hidden trap here. Let’s look closer at the order of operations in our constructor:

```

// ...

std::unique_ptr<T> p0, p1; // p0 在 p1 之前声明

public:

// 下面，假设我们按照以下方式识别操作：

// A: new T(val0)

// B: p0 的构造

// C: new T(val1)

// D: p1 的构造

// 我们知道：

// - A 在 B 之前

// - C 在 D 之前

// - B 在 D 之前

pair_with_alloc(const T &val0, const T &val1)

: p0{ new T(val0) }, p1{ new T(val1) } {

}

// ...

```cpp

 If you look at the rules laid out in the comments, you will see that we could have the operations in the following order, A→B→C→D, but we could also have them ordered as A→C→B→D or C→A→B→D, in which case the two calls to `new T(...)` would occur, followed by the two `unique_ptr<T>` constructors. If this happens, then an exception thrown by the second call to `new` or the associated constructor of `T` would still lead to a resource leak.
Now, that’s a shame. But that’s also the point of `make_unique<T>()` – with a factory function, client code never finds itself with “floating results from calls to `new`”; it either has a complete `unique_ptr<T>` object or not:

```

#include <memory>

template <class T>

class pair_with_alloc {

std::unique_ptr<T> p0, p1;

public:

pair_with_alloc(const T &val0, const T &val1)

: p0{ std::make_unique<T>(val0) },

p1{ std::make_unique<T>(val1) } {

}

// 析构函数隐式正确

// 复制和移动操作隐式工作

// 或留给你的想象

};

#include <string>

#include <random>

#include <iostream>

class risky {

shared_ptr<X> p{ new X };

std::uniform_int_distribution<int> penny{ 0,1 };

public:

risky() = default;

if(full()) {

if(penny(prng)) throw 3; // throws 50% of the time

}

~risky() {

std::cout << “~risky()\n”;

}

};

std::begin(resources), std::end(resources),

pair_with_alloc a{ s0, s1 };

// an exception is thrown

std::weak_ptr<Resource> obtain(Resource::id_type id){

class Cache {

shared_ptr<X> p{ new X(args) };

pair_with_alloc b{ risky{}, risky{} };

} catch(...) {

std::cout << std::format(“*sh == {}\n”, *sh);

}

}

```cpp

 As you can see, `make_unique<T>()` is a security feature, mostly useful to avoid exposing ownerless resources in client code. As a bonus, `make_unique<T>()` allows us to limit how we repeat ourselves in source code. Check the following:

```

{

auto p1 = unique_ptr<some_type> { new some_type{ args } };

auto p2 = make_unique<some_type>(args);

```cpp

 As you can see, `p0` and `p1` require you to spell the name of the pointed-to type twice whereas `p2` only requires you to write it once. That’s always nice.
Types shared_ptr and weak_ptr
In most cases, `unique_ptr<T>` will be your smart pointer of choice. It’s small, fast, and does what most code requires. There are some specialized but important use cases where `unique_ptr<T>` is not what you need, and these have in common the following:

*   The semantics being conveyed is the *shared ownership* of the resource
*   The last owner of the resource is not known a priori (which mostly happens in concurrent code)

Note that if the execution is not concurrent, you will, in general, know who the last owner of the resource is – it’s the last object to observe the resource that will be destroyed in the program. This is an important point – you can have concurrent code that shares resources and still uses `unique_ptr` to manage the resource. Non-owning users of the resource, such as raw pointers, can access it without taking ownership (more on that later in this chapter), and this approach is sufficient.
You can, of course, have non-concurrent code where the last owner of a resource is not known a priori. An example might involve a protocol where the provider of the resource still holds on to it after returning it to the client, but they might be asked to release it at a later point while client code retains it, making the client the last owner from that point on, or they might never be asked to release it, in which case the provider might be the last owner of the resource. Such situations are highly specific, obviously, but they show that there might be reasons to use shared ownership semantics as expressed through `std::shared_ptr`, even in non-concurrent code.
Since concurrent code remains the posterchild for situations where the last owner of a shared resource is not known a priori, we will use this as a basis for our investigation. Remember this example from the beginning of this chapter:

```

std::cout << “Using resource “ << q->id() << ‘\n’;

void f() {

[](auto && a, auto && b) {

);

thread th1{ [p] { /* use *p */ };

// w points to an expired shared_ptr<int> here

// ...

t, std::shared_ptr<Resource>{ p }

```cpp

 Here, `p` in `f()` does not own the `X` it points to, being a raw pointer, and both `th0` and `th1` copy that raw pointer, so neither is responsible for the pointee (at least on the basis of the rules enforced by the type system; you could envision acrobatics to make this work, but it’s involved, tricky, and bug-prone).
This example can be amended to have clear ownership semantics by shifting `p` from `X*` to `shared_ptr<X>`. Indeed, let’s consider the following:

```

thread th0{ [p] { /* use *p */ };

void f() {

std::shared_ptr<X> p { new X };

}

thread th1{ [p] { /* use *p */ };

th0.detach();

try {

}

```cpp

 In `f()`, the `p` object is initially the sole owner of the `X` it points to. When `p` is copied, as it is in the capture blocks of the lambdas executed by `th0` and `th1`, the mechanics of `shared_ptr` ensure that `p` and its two copies share both `X*` and an integral counter, used to determine how many shared owners there are for the resource.
The key functions of `shared_ptr` are its copy constructor (shares the resource and increments the counter), copy assignment (disconnects from the original resource, decrementing its counter, and then connects to the new resource, incrementing its counter), and the destructor (decrements the counter and destroys the resource if there’s no owner left). Each of these functions is subtle to implement; to help understand what the stakes are, we will provide simplified implementation examples in *Chapter 6*. Move semantics, unsurprisingly, implement transfer of ownership semantics for `shared_ptr`.
Note that `shared_ptr<T>` implements extrusive (non-intrusive) shared ownership semantics. Type `T` could be a fundamental type and does not need to implement a particular interface for this type to work. This differs from the intrusive shared semantics that were mentioned earlier in this chapter, with the `releasable` type an example.
Usefulness and costs
There are intrinsic costs to the `shared_ptr<T>` model. The most obvious one is that `sizeof(shared_ptr<T>)>sizeof(unique_ptr<T>)` for any type `T`, since `shared_ptr<T>` needs to handle both a pointer to the shared resource and a pointer to the shared counter.
Another cost is that copying a `shared_ptr<T>` is not a cheap operation. Remember that `shared_ptr<T>` makes sense mostly in concurrent code, where you do not know a priori the last owner of a resource. For that reason, the increments and decrements of the shared counter require synchronization, meaning that the counter is typically an `atomic` integer, and mutating an `atomic<int>` object (for example) costs more than mutating an `int`.
Another non-negligible cost is the following:

```

void observe(std::weak_ptr<int> w) {

```cpp

 An instruction such as this one will lead to *two* allocations, not one – there will be one for the `X` object and another one (performed internally by the `shared_ptr`) for the counter. Since these two allocations will be done separately, one by the client code and one by the constructor itself, the two allocated objects might find themselves in distinct cache lines, potentially leading to a loss of efficiency when accessing the `shared_ptr` object.
make_shared()
There is a way to alleviate the latter cost, and that is to make the same entity perform both allocations, instead of letting the client code do one and the constructor do the other. The standard tool to achieve this is the `std::make_shared<T>()` factory function.
Compare the following two instructions:

```

assert(p != std::end(resources));

auto q = make_shared<X>(args);

```cpp

 When constructing `p`, `shared_ptr<X>` is provided an existing `X*` to manage, so it has no choice but to perform a second, separate allocation for the shared counter. Conversely, the call expressed as `make_shared<X>(args)` specifies the type `X` to construct along with the arguments `args` to forward directly to the constructor. It falls upon that function to create `shared_ptr<X>`, `X`, and the shared counter, which lets us put both `X` and the counter in the same contiguous space (the **control block**), using mechanisms such as a *union* or the *placement new* mechanism, which will be explored in *Chapter 7*.
Clearly, given the same arguments used for construction, the preceding `p` and `q` will be equivalent `shared_ptr<X>` objects, but in general, `q` will perform better than `p`, as its two key components will be organized in a more cache-friendly manner.
What about weak_ptr?
If `shared_ptr<T>` is a type with a narrower (yet essential) niche than `unique_ptr<T>`, `weak_ptr<T>` occupies an even narrower (but still essential) niche. The role of `weak_ptr<T>` is to model the *temporary* ownership of `T`. Type `weak_ptr<T>` is meant to interact with `shared_ptr<T>` in a way that makes the continued existence of the *pointee* testable from client code.
A good example of `weak_ptr` usage, inspired by the excellent `cppreference` website ([`en.cppreference.com/w/cpp/memory/weak_ptr`](https://en.cppreference.com/w/cpp/memory/weak_ptr)), is as follows:

```

// inspired from a cppreference example

risky(const risky &) {

#include <memory>

#include <format>

return p.second->id() == id;

X *p = new X;

th1.detach();

// ...

std::cout << “w is expired\n”;

}

int main() {

std::weak_ptr<int> w;

}

auto sh = std::make_shared<int>(3);

thread th0{ [p] { /* use *p */ };

// w points to a live shared_ptr<int> here

unique_ptr<some_type> p0 { new some_type{ args } };

>> resources;

if(auto q = p.lock(); q)

observe(w);

}

```cpp

 As this example shows, you can make `weak_ptr<T>` from `shared_ptr<T>`, but `weak_ptr` does not own the resource until you call `lock()` on it, yielding `shared_ptr<T>`, from which you can safely use the resource after having verified that it does not model an empty pointer.
Another use case for `std::weak_ptr` and `std::shared_ptr` would be a cache of resources such that the following occurs:

*   The data in a `Resource` object is sufficiently big or costly to duplicate that it’s preferable to share it than to copy it
*   A `Cache` object shares the objects it stores, but it needs to invalidate them before replacing them when its capacity is reached

In such a situation, a `Cache` object could hold `std::shared_ptr<Resource>` objects but provide its client code, `std::weak_ptr<Resource>`, on demand, such that the `Resource` objects can be disposed of when the `Cache` needs to do so, but the client code needs to be able to verify that the objects it points to have not yet been invalidated.
A full (simplified) example would be the following (see the GitHub repository for this book to get the full example):

```

}

template <auto Cap>

for(int i = 0; i != 5; ++i)

else

// a cache of capacity Cap that keeps the

// most recently used Resource objects

std::cerr << “Something was thrown...\n”;

decltype(clock::now()),

std::shared_ptr<Resource>

cache.add(new Resource{ i + 1 });

bool full() const { return resources.size() == Cap; }

// precondition: !resources.empty()

void expunge_one() {

auto p = std::min_element(

return p->second; // make weak_ptr from shared_ptr

if(std::string s0, s1; std::cin >> s0 >> s1)

return a.first < b.first;

}

th1.detach();

}

p->second.reset(); // relinquish ownership

resources.erase(p);

using clock = std::chrono::system_clock;

public:

void add(Resource *p) {

const auto t = clock::now();

};

expunge_one();

);

resources.emplace_back(

id {

std::vector<std::pair<

}

}

const auto t = clock::now();

auto p = std::find_if(

std::begin(resources),

std::end(resources),

}

// the following objects do not leak even if

// ...

th0.detach();

if(p == std::end(resources))

if (std::shared_ptr<int> sh = w.lock())

p->first = t;

#include <iostream>

);

std::mt19937 prng{ std::random_device{}() };

int main() {

Cache<5> cache;

return {};

cache.add(new Resource{ i + 1 });

// let’s take a pointer to resource 3

auto p = cache.obtain(3);

w = sh; // weak_ptr made from shared_ptr

observe(w);

// things happen, resources get added, used, etc.

for(int i = 6; i != 15; ++i)

int main() {

if(auto q = p.lock(); q)

std::cout << “使用资源 “ << q->id() << ‘\n’;

else

std::cout << “资源不可用 ...\n”;

}

```cpp

 After a sufficient number of additions to the cache, the object pointed to by `p` in `main()` becomes invalidated and erased from the set of resources, one of our requirements for this example (without that requirement, we could have simply used `std::shared_ptr` objects in this case). Yet, `main()` can test for the validity of the object pointed to by `p` through the construction of `std::shared_ptr` from the `std::weak_ptr` it holds.
In practice, `weak_ptr` is sometimes used to break cycles when `shared_ptr` objects refer to each other in some way. If you have two types whose objects mutually refer to one another (say, `X` and `Y`) and do not know which one will be destroyed first, then consider making one of them the owner (`shared_ptr`) and the other one the non-owner in a verifiable manner (`weak_ptr`), which will ensure that they will not keep each other alive forever. For example, this will conclude, but the `X` and `Y` destructors will never be called:

```

#include <memory>

#include <iostream>

struct Y;

struct X {

std::shared_ptr<Y> p;

~X() { std::cout << “~X()\n”; }

};

struct Y {

std::shared_ptr<X> p;

~Y() { std::cout << “~Y()\n”; }

};

void oops() {

auto x = std::make_shared<X>();

auto y = std::make_shared<Y>();

x->p = y;

y->p = x;

}

int main() {

oops();

std::cout << “完成\n”;

}

```cpp

 If you change either `X::p` or `Y::p` to `weak_ptr`, you will see both the `X` and `Y` destructors being called:

```

#include <memory>

#include <iostream>

struct Y;

struct X {

std::weak_ptr<Y> p;

~X() { std::cout << “~X()\n”; }

};

struct Y {

std::shared_ptr<X> p;

~Y() { std::cout << “~Y()\n”; }

};

void oops() {

auto x = std::make_shared<X>();

auto y = std::make_shared<Y>();

x->p = y;

y->p = x;

}

int main() {

oops();

std::cout << “完成\n”;

}

```cpp

 Of course, the easiest way not to get to the point where you face a cycle of `shared_ptr<T>` objects is to not build such a cycle, but when faced with external libraries and third-party tools, that’s sometimes easier said than done.
When to use raw pointers
We have seen that smart pointer types such as `unique_ptr<T>` and `shared_ptr<T>` shine when there is a need to describe ownership of a type `T` resource through the type system. Does that mean that `T*` has become useless?
No, of course not. The trick is to use it in controlled situations. The first is that for a function, being passed a `T*` as an argument should mean the function is *an observer, not an owner*, of that `T`. If your code base used raw pointers in that sense, you will most probably not run into trouble.
Secondly, you can use a raw pointer inside a class that implements your preferred ownership semantics. It’s fine to implement a container that manipulates objects through raw pointers (for example, a tree-like structure meant for various traversal orders), as long as that container implements clear copy and move semantics. What you don’t want to do is expose pointers to the internal nodes of your container to external code. Pay attention to the container’s interface.
Indeed, consider this single-linked list of (excerpt):

```

template <class T>

class single_linked_list {

struct node {

T value;

node *next = nullptr;

node(const T &val) : value { val } {

};

node *head = nullptr;

// ...

public:

// ...

~single_linked_list() {

for(auto p = head; p;) {

auto q = p->next;

delete p;

p = q;

}

}

};

```cpp

 We will explore this example in greater detail in *Chapter 13*. The destructor works fine and (supposing the rest of the class is reasonably well-written) the class is usable and useful. Now, suppose we decide to use `unique_ptr<node>` instead of `node*` as the `head` data member for `single_linked_list`, and as a replacement for the `next` member of the node. This seems like a good idea, except when you consider the consequences:

```

template <class T>

class single_linked_list {

struct node {

T value;

unique_ptr<node> next; // 好主意？

node(const T &val) : value { val } {

};

unique_ptr<node> head; // 好主意？

// ...

public:

// ...

~single_linked_list() = default;

};

```cpp

 This seems like a good idea on the surface, but it does not convey the proper semantics – it’s *not* true that a node *owns* and *is responsible for* the next node. We don’t want to make the removal of a node destroy the node that follows (and so on, recursively) and if that looks like a simplification in the destructor of `single_linked_list`, think about the consequences – this strategy leads to as many destructors recursively called as there are nodes in the list, which is a very good way to achieve a stack overflow!
Use a smart pointer when the use case matches the semantics it models. Of course, when the relationship modeled by your pointers is neither unique ownership nor shared ownership, you probably do not want smart pointer types that provide these semantics, resorting instead to either nonstandard and non-owning smart pointers or, simply, raw pointers.
Finally, you often need raw pointers to use lower-level interfaces – for example, when performing system calls. That does not disqualify higher-level abstractions, such as `vector<T>` or `unique_ptr<T>`, when writing system-level code – you can get access to the underlying array of `vector<T>` through its `data()` member function, just as you can get access to the underlying raw pointer of `unique_ptr<T>` through its `get()` member function. As long as it makes sense, see the called code as borrowing the pointer from the caller code for the duration of the call.
And if you have no other choice, use raw pointers. They exist, after all, and they work. Simply remember to use higher-level abstractions wherever possible – it will make your code simpler, safer, and (more often than you would think) faster. If you cannot define the higher-level semantics, maybe it’s still a bit early to write that part of the code, and you’ll get better results if you spend more time thinking about these semantics.
Summary
In this chapter, we saw how to use standard smart pointers. We discussed the ownership semantics they implement (sole ownership, shared co-ownership, and temporary co-ownership), saw examples of how they can be used, and discussed some ways in which they can be used while acknowledging that other, more appropriate options exist.
In the next chapter, we’ll take this a step further and write our own (usable, if naïve) versions of `unique_ptr<T>` and `shared_ptr<T>`, in order to get an intuitive grasp of what this entails, and we will write some nonstandard but useful smart pointers too. This will help us build a nicer, more interesting resource management toolset.

```
