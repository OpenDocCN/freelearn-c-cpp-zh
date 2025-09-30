

# 第十一章：延迟回收

在*第九章*中，我们展示了某些不寻常的内存分配机制的示例以及如何使用它们，包括如何响应错误以给我们的程序一种“第二次机会”来继续执行，以及如何通过 C++语言设施使用非典型或异国内存。然后，在第*第十章*中，我们考察了基于竞技场的分配及其一些变体，重点关注速度、确定性和对资源消耗的控制问题。

在当前章节中，我们将进行一些在 C++中不常见但在许多其他语言中是常见做法的操作：我们将在程序执行过程中选择特定时刻延迟动态分配对象的销毁。

我们将*不会*编写一个真正的垃圾回收器，因为这会涉及到对编译器内部工作的更深入参与，并影响使 C++成为如此美妙工具的编程模型。然而，我们将组装**延迟回收**的机制，即选择性地在特定时刻销毁选定的对象，并一起释放其底层存储，但不保证销毁顺序。当然，我们不会提供实现这一目标的技术的详尽概述，但我们希望给你，亲爱的读者，足够的“思考材料”，以便在需要时构建自己的延迟回收机制。

本章中的技术可以与第*第十章*中看到的技术相结合，以使程序运行更快并减少内存碎片，但我们将单独介绍延迟回收，以使我们的论述更加清晰。阅读本章后，你将能够做到以下几件事情：

+   理解与延迟回收相关的权衡，因为虽然可以取得收益，但也涉及成本（这并非万能药！）

+   实现一个几乎透明的外部包装器，以跟踪需要收集的内存

+   实现一个几乎透明的外部包装器，以帮助最终确定那些受到延迟回收的对象

+   实现一个类似于`std::shared_ptr`对象的引用计数的计数指针，以识别在所选作用域结束时可以回收的对象

我们需要采取的第一步是尝试理解一些可以从中受益的延迟回收问题领域，包括它与（不同但并非完全不相似的）垃圾收集问题的关系。

最终化？回收？

您会注意到，在本章中，我们经常使用“finalization”（最终化）这个词而不是“destruction”（销毁）这个词，因为我们试图强调这样一个事实：在对象生命周期的末尾（其析构函数）执行的代码与释放其底层存储的代码是不同的。作为额外的好处，**finalization**在垃圾回收语言中更为常见，而垃圾回收是接下来几节讨论的技术的一个近亲。将 finalization（不进行回收）视为调用对象析构函数（不释放底层存储）的等价物。

如本章前面所述，我们将**回收**定义为在选定的时刻（例如，作用域结束时或程序执行结束时）释放一个或多个对象的内存。再次强调，这个术语在垃圾回收语言中比在 C++中更为常见，但本章的主题在某种程度上更接近这些语言所做的工作，因此，使用类似的术语可能有助于形成对涉及的思想和技术的共同理解。

# 技术要求

您可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter11`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter11)。

# 我们所说的延迟回收是什么意思？

为什么会有人想要求助于延迟回收？这确实是一个合理的问题，所以感谢您提问！

简短的回答是，它解决了实际问题。确实，有些程序在对象停止被客户端代码引用后不立即收集对象是有意义的，或者在我们确定可能使用它们的代码结束之前，不清楚它们是否可以被收集。由于我们用 C++语言思考代码的方式，这些程序在 C++中相对较少，但从一般编程世界的角度来看，它们并不罕见。

例如，考虑一个函数，其中一些局部分配的对象之间存在循环引用，或者一个可以从根节点导航到其叶节点的树，但在这个树中，叶节点也指向其根节点。有时，我们可以确定如何销毁一组对象：例如，在树的情况下，我们可以决定从根节点开始，沿着分支向下进行。在其他情况下，如果我们知道一组对象不会逃离给定的函数，我们也可以利用在函数结束时它们都可以作为一个组回收的知识。

如果你熟悉垃圾回收语言，你可能知道在大多数语言中，回收器“回收字节”，释放回收对象的基础存储（有时在执行过程中还会压缩内存），但不会最终化对象。其中一个原因是，在这样的语言中，一个对象很难（在某些情况下，甚至不可能）知道程序中还存在哪些其他对象，因为没有最终化顺序的保证……如果垃圾回收器需要处理相互引用的对象循环，又怎么可能存在这样的顺序呢？当对象达到其生命周期结束时，不知道哪些其他对象仍然存在，这严重限制了最终化代码能做的事情。

在许多语言中，回收并不意味着最终化，这简化了收集对象的任务：从概念上讲，可以调用`std::free()`或一些等效函数来释放内存，而无需担心其中的对象。在那些在回收之前保证最终化的语言中，通常会发现一个以单个公共基类（通常称为`object`或`Object`）为根的类层次结构，这使得可以在每个对象上调用等效的`virtual`析构函数，并多态地最终化它。当然，在这种情况下，在最终化对象时能做的事情是有限的，因为对象最终化的顺序通常是未知的。

在当代垃圾回收语言中，更常见的是将最终化的责任交给客户端代码，而将回收工作留给语言本身。这样的语言通常使用一个特殊的接口（例如 C#中的`IDisposable`和 Java 中的`Closeable`），由需要最终化的类实现（通常是管理外部资源的类），客户端代码将明确地放置所需的机制以实现对象的有序最终化。这将从对象本身（如 C++中的 RAII 习语所述，见*第四章*）将部分资源管理责任转移到使用它的代码上，这提醒我们，垃圾回收器倾向于简化内存管理，但同时也倾向于使其他资源的管理复杂化。

这样的客户端代码驱动的资源管理示例包括一个带有`finally`块的`try`块，无论`try`块是否正常结束或进入了一些`catch`块，它都作为应用清理代码的焦点。还有一些简化语法，以更轻松的方式为客户代码执行相同的事情。例如，Java 使用 try-with 块，并在作用域结束时隐式调用所选`Closeable`对象的`close()`，而 C#使用`using`块以类似的方式隐式调用所选`IDisposable`对象的`Dispose()`。

C++ 没有提供 `finally` 块，也不使用侵入式技术，例如语言已知并给予特殊处理或作为所有类型公共基类的特殊接口。在 C++ 中，对象通常通过 RAII 习语来负责管理自己的资源；与其他流行语言相比，这导致了一种不同的思维方式和不同的编程技术。

在这一章中，我们将面临与垃圾回收语言中遇到的情况相似但不同的情况：如果我们想使用对象的延迟回收，我们不能保证在销毁过程中，回收的对象之一能够访问同一组中回收的其他对象，因此不应该尝试这样做。另一方面，我们选择将延迟回收应用于 *选定* 对象（而不是对所有对象都这样做）的事实意味着，不属于该组且已知能够存活到该组回收的对象，在回收对象的最终化过程中仍然可以访问。这确实是拥有一种一刀切解决方案的好处：如果你在开始阅读这本书之前就知道，C++ 如果不是多才多艺的，那将什么都不是。

没有所有类型的公共基类意味着我们可能不得不放弃最终化（如果我们限制自己分配具有平凡析构类型的对象，这可以在编译时验证，那么这可以工作）或者我们必须找到其他方法来记住我们分配的对象的类型，并在适当的时候调用相应的析构函数。在这一章中，我们将展示如何实现这两种方法。

与流行观点相反，一些垃圾回收器已经为 C++ 实现。其中最著名的一个（由 Hans Boehm、Alan Demers 和 Mark Weiser 制作的 Boehm-Demers-Weiser 收集器）在一般情况下不终结对象，但允许从用户代码中注册选定的终结器。这是通过名为 `GC_register_finalizer` 的功能完成的，但作者警告用户，这种终结器能做的事情是有限的，就像垃圾回收语言中（以及在本节中之前讨论过）的情况一样。

进一步阅读

要进一步探索，请查看[`www.hboehm.info/gc/`](https://www.hboehm.info/gc/).

在这一章中，我们将使用其他技术。正如本书中始终所做的那样，我们的意图是展示你可以从中实验并构建你代码需要的解决方案的想法。我们将展示三个不同的示例：

+   在程序执行结束时回收选定对象但不会终结它们的代码，将延迟回收限制为具有平凡析构类型的对象

+   在程序执行结束时回收和终结选定对象的代码

+   在选定作用域结束时回收和终结选定对象的代码

在每种情况下，我们将采取不同的方法，以给你更广泛的视角，了解可以做什么。在所有三种情况下，我们将在全局可访问的对象中存储指针。是的，这是一个单例，但在这里这是正确的工具，因为我们正在讨论影响整个程序的功能。准备好了吗？我们开始了！

我们有时会做一些事情来使示例更易于阅读…

下文中的代码可能会让一些读者感到奇怪。为了专注于代码的延迟回收方面，并保持整体演示的可读性，我选择不深入探讨线程安全方面，尽管这在当代代码中是至关重要的。然而，在本章的 GitHub 仓库中，你可以找到本书中展示的代码以及每个示例的线程安全等效代码。

# 程序结束时的回收（不进行最终化）

我们的第一种实现将在程序执行结束时提供回收，但不提供最终化。因此，它不会接受管理类型 `T` 的对象，如果 `T` 不是平凡可析构的，因为该类型的对象可能需要执行析构函数以避免泄漏或其他问题。

就像本章中的其他示例一样，我们将从我们的测试代码开始，然后继续了解回收机制是如何实现的。我们的测试代码如下：

+   我们将声明两个类型，`NamedThing` 和 `Identifier`。前者不会是平凡可析构的，因为它的析构函数将包含打印调试信息的用户代码，而后者将是平凡的，因为它只包含平凡可析构的非静态数据成员，并且不提供用户提供的析构函数。

+   我们将提供两个 `g()` 函数。第一个将被注释掉，因为它试图通过我们的回收系统分配 `NamedThing` 对象，这不会编译，因为 `NamedThing` 类型不符合我们平凡可析构的要求。第二个将被使用，因为它分配的对象符合那些要求。

+   `f()`、`g()` 和 `main()` 函数将在我们程序调用栈的各个级别构造对象。然而，可回收的对象将仅在程序执行结束时存在。

在这种情况下，客户端代码如下：

```cpp
// ...
// note: not trivially destructible
struct NamedThing {
   const char *name;
   NamedThing(const char *name) : name{ name } {
      std::print("{} ctor\n", name);
   }
   ~NamedThing() {
      std::print("{} dtor\n", name);
   }
};
struct Identifier {
   int value;
};
// would not compile
/*
void g() {
   [[maybe_unused]] auto p = gcnew<NamedThing>("hi");
   [[maybe_unused]] auto q = gcnew<NamedThing>("there");
}
*/
void g() {
   [[maybe_unused]] auto p = gcnew<Identifier>(2);
   [[maybe_unused]] auto q = gcnew<Identifier>(3);
}
auto h() {
   struct X {
      int m() const { return 123; }
   };
   return gcnew<X>();
}
auto f() {
   g();
   return h();
}
int main() {
   std::print("Pre\n");
   std::print("{}\n", f()->m());
   std::print("Post\n");
}
```

使用这段代码和（到目前为止缺失的）延迟回收代码，这个程序将打印以下内容：

```cpp
Pre
123
Post
~GC with 3 objects to deallocate
```

注意，`f()` 函数分配并返回一个对象，`main()` 函数通过该对象调用 `m()` 成员函数，而不需要明确使用智能指针，但这个程序并没有内存泄漏。通过 `gcnew<T>()` 函数分配的对象被注册在 `GC` 对象中，`GC` 对象的析构函数将确保注册的内存块将被释放。

那么`gcnew<T>()`是如何工作的，为什么要写这样一个函数而不是简单地重载`operator new()`呢？记住，`operator new()`作为一个分配函数介入了整体分配过程——它交换的是原始内存，而不是知道要创建的对象的类型。在这个例子中，我们想要（a）为新对象分配内存，（b）构造对象（因此需要类型和传递给构造函数的参数），以及（c）拒绝不是简单可销毁的类型。我们需要知道要构造的对象的类型，这是`operator new()`所不知道的。

为了能够在程序执行结束时回收这些对象的内存，我们需要一种全局可用的存储形式，我们将把分配的指针放在那里。我们将这样的指针称为`roots`，并将它们存储在`GC`类型的单例中（受垃圾收集器通常使用的昵称的启发，尽管这并不是我们正在实现的功能——这个名称将很好地传达意图，而且足够短，不会妨碍使用）。

`GC::add_root<T>(args...)`成员函数将确保`T`是一个简单可销毁的类型，分配一个`sizeof(T)`字节的块，在该位置构造`T(args...)`，在`roots`中存储对该对象的抽象指针（一个`void*`），并返回一个指向新创建对象的`T*`对象。`gcnew<T>()`函数将允许用户代码以简化的方式与`GC::add_root<T>()`接口；由于我们希望用户代码使用`gcnew<T>()`，我们将`GC::add_root<T>()`标记为`private`，并将`gcnew<T>()`作为`GC`类的`friend`。

注意，`GC`类本身不是一个泛型类（它不是一个模板）。它公开了模板成员函数，但在结构上只存储原始地址（`void*`对象），这使得这个类在类型上大部分是无知的。所有这些都导致了以下代码：

```cpp
#include <vector>
#include <memory>
#include <string>
#include <print>
#include <type_traits>
class GC {
   std::vector<void*> roots;
   GC() = default;
   static auto &get() {
      static GC gc;
      return gc;
   }
   template <class T, class ... Args>
      T *add_root(Args &&... args) {
         // there will be no finalization
         static_assert(
            std::is_trivially_destructible_v<T>
         );
         return static_cast<T*>(
            roots.emplace_back(
               new T(std::forward<Args>(args)...)
            )
         );
      }
   // provide access privileges to gcnew<T>()
   template <class T, class ... Args>
      friend T* gcnew(Args&&...);
public:
   ~GC() {
      std::print("~GC with {} objects to deallocate",
                 std::size(roots));
      for(auto p : roots) std::free(p);
   }
   GC(const GC &) = delete;
   GC& operator=(const GC &) = delete;
};
template <class T, class ... Args>
   T *gcnew(Args &&...args) {
      return GC::get().add_root<T>(
         std::forward<Args>(args)...
      );
GC::~GC() calls std::free() but invokes no destructor, as this implementation reclaims memory but does not finalize objects.
This example shows a way to group memory reclamation as a single block to be executed at the end of a program. In code where there is more available memory than what the program requires, this can lead to a more streamlined program execution, albeit at the cost of a slight slowdown at program termination (of course, if you want to try this, please measure to see whether there are actual benefits for your code base!). It can also help us write analysis tools that examine how memory has been allocated throughout program execution and can be enhanced to collate additional information such as memory block size and alignment: we simply would need to keep pairs – or tuples, depending on the needs – instead of single `void*` objects in the `roots` container to aggregate the desired data.
Of course, not being able to finalize objects allocated through this mechanism can be a severe limitation, as no non-trivially destructible type can benefit from our efforts. Let’s see how we could add finalization support to our design.
Reclamation and finalization at the end of the program
Our second implementation will not only free the underlying storage for the objects allocated through our deferred reclamation system but will also finalize them by calling their destructors. To do so, we will need to remember the type of each object that goes through our system. There are, of course, many ways to achieve this, and we will see one of them.
By ensuring the finalization of reclaimed objects, we can get rid of the trivially destructible requirement of our previous implementation. We still will not guarantee the order in which objects are finalized, so it’s important that reclaimed objects do not refer to each other during finalization if we are to have sound programs, but that’s a constraint many other popular programming languages also share. This implementation will, however, keep the singleton approach and finalize and then deallocate objects and their underlying storage at the end of program execution.
As in the previous section, we will first look at client code. In this case, we will be using (and benefitting from) non-trivially destructible objects and use them to print out information during finalization: this will simplify the task of tracing program execution. Of course, we will also use trivially destructible types (such as `struct X`, local to the `h()` function) as there is no reason not to support these too. Note that, often (but not always), non-trivially destructible types will be RAII types (see *Chapter 4*) whose objects need to free resources before their life ends, but we just want a simple example here so doing anything non-trivial such as printing out some value (which is what we are doing with `NamedThing`) will suffice in demonstrating that we handle non-trivially-destructible types correctly.
We will use nested function calls to highlight the local aspect of construction and allocation, as well as the non-local aspect of object destruction and deallocation since these will happen at program termination time. Our example code will be as follows:

```

// ...

// note: not trivially destructible

struct NamedThing {

const char *name;

NamedThing(const char *name) : name{ name } {

std::print("{} ctor\n", name);

}

~NamedThing() {

std::print("{} dtor\n", name);

}

};

void g() {

[[maybe_unused]] auto p = gcnew<NamedThing>("hi");

[[maybe_unused]] auto q = gcnew<NamedThing>("there");

}

auto h() {

struct X {

int m() const { return 123; }

};

return gcnew<X>();

}

auto f() {

g();

return h();

}

int main() {

std::print("Pre\n");

std::print("{}\n", f()->m());

std::print("Post\n");

}

```cpp

 When executed, you should expect the following information to be printed on the screen:

```

Pre

hi ctor

there ctor

123

Post

hi dtor

there dtor

```cpp

 As can be seen, the constructors happen when invoked in the source code, but the destructors are called at program termination (after the end of `main()`) as we had announced we would do.
On the importance of interfaces
You might notice that user code essentially did not change between the non-object-finalizing implementation and this one. The beauty here is that our upgrade, or so to say, is completely achieved in the implementation, leaving the interface stable and, as such, the differences transparent to client code. Being able to change the implementation without impacting interfaces is a sign of low coupling and is a noble objective for one to seek to attain.
How did we get from a non-finalizing implementation to a finalizing one? Well, this implementation will also use a singleton named `GC` where “object roots” will be stored. In this case, however, we will store semantically enhanced objects, not just raw addresses (`void*` objects) as we did in the previous implementation.
We will achieve this objective through a set of old yet useful tricks:

*   Our `GC` class will not be a generic class, as it would force us to write `GC<T>` instead of just `GC` in our code, and find a way to have a distinct `GC<T>` object for each `T` type. What we want is for a single `GC` object to store the required information for all objects that require deferred reclamation, regardless of type.
*   In `GC`, instead of storing objects of the `void*` type, we will store objects of the `GC::GcRoot*` type. These objects will not be generic either but will be polymorphic, exposing a `destroy()` service to destroy (call the destructor, then free the underlying storage) objects.
*   There will be classes that derive from `GC::GcRoot`. We will call such classes `GC::GcNode<T>` and there will be one for each type `T` in a program that is involved in our deferred reclamation mechanism. These are where the type-specific code will be “hidden.”
*   By keeping `GC::GcRoot*` objects as roots but storing `GC::GcNode<T>*` in practice, we will be able to deallocate and finalize the `T` object appropriately.

The code for this implementation follows:

```

#include <vector>

#include <memory>

#include <print>

class GC {

class GcRoot {

void *p;

public:

auto get() const noexcept { return p; }

GcRoot(void *p) : p{ p } {

}

GcRoot(const GcRoot &) = delete;

GcRoot& operator=(const GcRoot &) = delete;

virtual void destroy(void *) const noexcept = 0;

virtual ~GcRoot() = default;

};

// ...

```cpp

 As can be seen, `GC::GcRoot` is an abstraction that trades in raw pointers (objects of the `void*` type) and contains no type-specific information, per se.
The type-specific information is held in derived classes of the `GcNode<T>` type:

```

// ...

template <class T> class GcNode : public GcRoot {

void destroy(void* q) const noexcept override {

delete static_cast<T*>(q);

}

public:

template <class T, class ... Args>

GcNode(Args &&... args) :

GcRoot(new T(std::forward<Args>(args)...)) {

}

~GcNode() {

destroy(get());

}

};

// ...

```cpp

 As we can see, a `GcNode<T>` object can be constructed with any sequence of arguments suitable for type `T`, perfectly forwarding them to the constructor of a `T` object. The actual (raw) pointers are stored in the base class part of the object (the `GcRoot` but the destructor of a `GcNode<T>` invokes `destroy()` on that raw pointer, which casts the `void*` to the appropriate `T*` type before invoking `operator delete()`.
Through the `GcRoot` abstraction, a `GC` object is kept apart from type-specific details of the objects it needs to reclaim at a later point. This implementation can be seen as a form of **external polymorphism**, where we use a polymorphic hierarchy “underneath the covers” to implement functionality in such a way as to keep client code unaware.
Given what we have written so far, our work is almost done:

*   Lifetime management can be delegated to smart pointers, as the finalization code is found in the destructor of `GcNode<T>` objects. Here, we will be using `std::unique_ptr<GcRoot>` objects (simple and efficient).
*   The `add_root()` function will create `GcNode<T>` objects, store them in the `roots` container as pointers to their base class, `GcRoot`, and return the `T*` pointing to the newly constructed object. Thus, it installs lifetime management mechanisms while exposing pointers in ways that look natural to users of `operator new()`.

That part of the code follows:

```

// ...

std::vector<std::unique_ptr<GcRoot>> roots;

GC() = default;

static auto &get() {

static GC gc;

return gc;

}

template <class T, class ... Args>

T *add_root(Args &&... args) {

return static_cast<T*>(roots.emplace_back(

std::make_unique<GcNode<T>>(

std::forward<Args>(args)...)

)->get());

}

template <class T, class ... Args>

friend T* gcnew(Args&&...);

public:

GC(const GC &) = delete;

GC& operator=(const GC &) = delete;

};

template <class T, class ... Args>

T *gcnew(Args &&...args) {

return GC::get().add_root<T>(

std::forward<Args>(args)...

);

}

// ...

```cpp

 So, there we have it: a way to create objects at selected points, and destroy and reclaim them all at program termination, with the corresponding upsides and downsides, of course. These tools are useful, but they are also niche tools that you should use (and customize to your needs) if there is indeed a need to do so.
So far, we have seen deferred reclamation facilities that terminate (and finalize, depending on the tool) at program termination. We still need a mechanism for reclamation at the end of selected scopes.
Reclamation and finalization at the end of the scope
Our third and last implementation for this chapter will ensure reclamation and finalization at the end of the scope, but only on demand. By this, we mean that if a user wants to reclaim unused objects that are subject to deferred reclamation at the end of a scope, it will be possible to do so. Objects subject to deferred reclamation that are still considered in use will not be reclaimed, and objects that are not in use will not be reclaimed if the user code does not ask for it. Of course, at program termination, all remaining objects that are subject to deferred reclamation will be claimed, as we want to avoid leaks.
This implementation will be more subtle than the previous ones, as we will need to consider (a) whether an object is still being referred to at a given point in program execution and (b) whether there is a need to collect objects that are not being referred to at that time.
To get to that point, we will inspire ourselves from `std::shared_ptr`, a type we provided an academic and simplified version of in *Chapter 6*, and will write a `counting_ptr<T>` type that, instead of destroying the pointee when its last client disconnects, will mark it as ready to be reclaimed.
The client code for this example follows. Pay attention to the presence of objects of the `scoped_collect` type in some scopes. These represent requests made by client code to reclaim objects not in use anymore at the end of that scope:

```

// ...

// 注意：不是简单可销毁的

struct NamedThing {

const char *name;

NamedThing(const char *name) : name{ name } {

std::cout << name << " ctor" << std::endl;

}

~NamedThing() {

std::cout << name << " dtor" << std::endl;

}

};

auto g() {

auto _ = scoped_collect{};

[[maybe_unused]] auto p = gcnew<NamedThing>("hi");

auto q = gcnew<NamedThing>("there");

return q;

} // 在这里将发生回收

auto h() {

struct X {

int m() const { return 123; }

};

return gcnew<X>();

}

auto f() {

auto _ = scoped_collect{};

auto p = g();

std::cout << '\"' << p->name << '\"' << std::endl;

} // 在这里将发生回收

int main() {

using namespace std;

cout << "Pre" << endl;

f();

cout << h()->m() << endl;

cout << "Post" << endl;

} scoped_collect 对象的生命周期结束将导致通过 gcnew<T>()分配且在该点不再被引用的所有对象的回收；这无论它们是在该作用域内还是程序的其他地方分配的都成立。这里的意图是，作用域的结束是一个我们愿意“付出”收集一组对象所需的时间和精力的点。不要在速度或确定性行为至关重要的作用域中使用 scoped_collect 对象！

执行此代码，我们最终得到以下结果：

```cpp
Pre
hi ctor
there ctor
hi dtor
"there"
there dtor
123
Post
```

如我们所见，仍然被引用的对象仍然可用，而不再被引用的对象要么在`scoped_collect`对象的析构函数被调用时回收，要么在程序终止时回收，如果此时程序中仍有可回收的对象。

`scoped_collect`类型本身非常简单，其主要作用是与`GC`全局对象交互。它是一个不可复制、不可移动的 RAII 对象，在其生命周期结束时执行回收：

```cpp
// ...
struct scoped_collect {
   scoped_collect() = default;
   scoped_collect(const scoped_collect &) = delete;
   scoped_collect(scoped_collect &&) = delete;
   scoped_collect&
      operator=(const scoped_collect &) = delete;
   scoped_collect &operator=(scoped_collect &&) = delete;
   ~scoped_collect() {
      GC::get().collect();
   }
};
// ...
```

整个基础设施是如何工作的？让我们一步一步来。我们将从本章前面的部分汲取灵感，在那里我们最初在程序执行结束时收集所有对象，然后为这些对象添加终结。本节的新颖之处在于，我们将添加在程序执行过程中的不同时间收集对象的可能性，并实现跟踪对象引用所需的代码。

为了跟踪对象的引用，我们将使用 `counting_ptr<T>` 类型的对象：

```cpp
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <atomic>
#include <functional>
#include <utility>
```

如所见，我们可以（并且确实！）仅通过标准工具实现此类。请注意，`count` 数据成员是一个指针，因为它可能在 `counting_ptr<T>` 的实例之间共享：

```cpp
template <class T>
   class counting_ptr {
      using count_type = std::atomic<int>;
      T *p;
      count_type *count;
      std::function<void()> mark;
   public:
      template <class M>
         constexpr counting_ptr(T *p, M mark) try :
            p{ p }, mark{ mark } {
               count = new count_type{ 1 };
         } catch(...) {
            delete p;
            throw;
         }
      T& operator*() noexcept {
         return *p;
      }
      const T& operator*() const noexcept {
         return *p;
      }
      T* operator->() noexcept {
         return p;
      }
      const T* operator->() const noexcept {
         return p;
      }
      constexpr bool
         operator==(const counting_ptr &other) const {
         return p == other.p;
      }
      // operator!= can be omitted since C++20
      constexpr bool
         operator!=(const counting_ptr &other) const {
         return !(*this == other);
      }
      // we allow comparing counting_ptr<T> objects
      // to objects of type U* or counting_ptr<U> to
      // simplify the handling of types in a class
      // hierarchy
      template <class U>
         constexpr bool
           operator==(const counting_ptr<U> &other) const {
            return p == &*other;
         }
      template <class U>
         constexpr bool
           operator!=(const counting_ptr<U> &other) const {
            return !(*this == other);
         }
      template <class U>
         constexpr bool operator==(const U *q) const {
            return p == q;
         }
      template <class U>
         constexpr bool operator!=(const U *q) const {
            return !(*this == q);
         }
       // ...
```

现在关系运算符已经就位，我们可以为我们的类型实现拷贝和移动语义：

```cpp
      // ...
      void swap(counting_ptr &other) {
         using std::swap;
         swap(p, other.p);
         swap(count, other.count);
         swap(mark, other.mark);
      }
      constexpr operator bool() const noexcept {
         return p != nullptr;
      }
      counting_ptr(counting_ptr &&other) noexcept
         : p{ std::exchange(other.p, nullptr) },
           count{ std::exchange(other.count, nullptr) },
           mark{ other.mark } {
      }
      counting_ptr &
         operator=(counting_ptr &&other) noexcept {
         counting_ptr{ std::move(other) }.swap(*this);
         return *this;
      }
      counting_ptr(const counting_ptr &other)
         : p{ other.p }, count{ other.count },
           mark{ other.mark } {
         if (count) ++(*count);
      }
      counting_ptr &operator=(const counting_ptr &other) {
         counting_ptr{ other }.swap(*this);
         return *this;
      }
      ~counting_ptr() {
         if (count) {
            if ((*count)-- == 1) {
               mark();
               delete count;
            }
         }
      }
   };
namespace std {
   template <class T, class M>
      void swap(counting_ptr<T> &a, counting_ptr<T> &b) {
         a.swap(b);
      }
}
// ...
```

与 `shared_ptr<T>` 相似，`counting_ptr<T>` 不会像销毁计数器和指针一样销毁，而是删除计数器但“标记”指针，使其成为后续回收的候选对象。

上节中提到的通用 `GC`、`GC::GcRoot` 和 `GC::GcNode<T>` 方法仍然保留，但如下进行了增强：

+   `roots` 容器将 `unique_ptr<GcRoot>` 与一个类型为 `bool` 的“标记”数据成员相结合

+   `make_collectable(p)` 成员函数将 `p` 指针关联的根标记为可回收

+   `collect()` 成员函数回收所有标记为可回收的根

此实现所做的（a）是给每个可回收指针关联一个布尔标记（回收或不回收），（b）使用 `counting_ptr<T>` 对象与每个 `T*` 一起跟踪每个指针的使用情况，以及（c）每当收到回收请求时，将可回收指针作为一组进行收集。请求此类收集的最简单方法是通过 `scoped_collect` 对象的析构函数。

这个稍微复杂一些的版本的代码如下：

```cpp
// ...
class GC {
   class GcRoot {
      void *p;
   public:
      auto get() const noexcept { return p; }
      GcRoot(void *p) : p{ p } {
      }
      GcRoot(const GcRoot&) = delete;
      GcRoot& operator=(const GcRoot&) = delete;
      virtual void destroy(void*) const noexcept = 0;
      virtual ~GcRoot() = default;
   };
   template <class T> class GcNode : public GcRoot {
      void destroy(void *q) const noexcept override {
         delete static_cast<T*>(q);
      }
   public:
      template <class ... Args>
         GcNode(Args &&... args)
            : GcRoot(new T(std::forward<Args>(args)...)) {
         }
      ~GcNode() {
         destroy(get());
      }
   };
   std::vector<
      std::pair<std::unique_ptr<GcRoot>, bool>
   > roots;
   GC() = default;
   static auto &get() {
      static GC gc;
      return gc;
   }
```

在这种情况下，收集函数如下：

```cpp
   void make_collectable(void *p) {
      for (auto &[q, coll] : roots)
         if (static_cast<GcRoot*>(p) == q.get())
            coll = true;
   }
   void collect() {
      for (auto p = std::begin(roots);
           p != std::end(roots); ) {
         if (auto &[ptr, collectible] = *p; collectible) {
            ptr = nullptr;
            p = roots.erase(p);
         } else {
            ++p;
         }
      }
   }
   template <class T, class ... Args>
      auto add_root(Args &&... args) {
         auto q = static_cast<T*>(roots.emplace_back(
            std::make_unique<GcNode<T>>(
               std::forward<Args>(args)...
            ), false
         ).first->get());
         // the marking function is implemented as
         // a lambda expression that iterates through
         // the roots, then finds and marks for
         // reclamation pointer q. It is overly
         // simplified (linear search) and you are
         // welcome to do something better!
         return counting_ptr{
            q, [&,q]() {
               for (auto &[p, coll] : roots)
                  if (static_cast<void*>(q) ==
                      p.get()->get()) {
                     coll = true;
                     return;
                  }
            }
         };
      }
   template <class T, class ... Args>
      friend counting_ptr<T> gcnew(Args&&...);
   friend struct scoped_collect;
public:
   GC(const GC &) = delete;
   GC& operator=(const GC &) = delete;
};
// ...
template <class T, class ... Args>
   counting_ptr<T> gcnew(Args &&... args) {
      return GC::get().add_root<T>(
         std::forward<Args>(args)...
      );
   }
// ...
```

正如您所看到的，亲爱的读者，这个最后的例子可以从几个优化中受益，但它可以工作，并且旨在足够简单，以便理解和改进。

现在我们知道，在 C++ 中，像在其他流行语言中一样，可以以组的形式回收对象。这可能不是典型的 C++ 代码，但通过合理的努力，可以以可选的方式实现延迟回收。还不错！

摘要

本章带我们进入了延迟回收的领域，这对许多 C++ 程序员来说是不熟悉的。我们看到了在程序中的特定点以组的形式回收对象的方法，讨论了在回收此类对象时可能进行的限制，并检查了在释放相关内存存储之前最终化对象的各种技术。

现在，我们可以看看内存管理如何与 C++ 容器交互，这是一个重要的主题，将在接下来的三章中占据我们的注意力。

事实上，我们可以编写处理内存的容器，但通常这会适得其反（例如，如果我们把 `std::vector<T>` 与 `new` 和 `delete` 绑定，`std::vector<T>` 如何处理需要通过其他方式分配和释放的类型 `T`？）。

当然，到达那里的方法有很多。想知道一些吗？让我们深呼吸，深入探讨…

```cpp

```

# 第四部分：编写泛型容器（以及更多内容）

在本部分中，我们将专注于编写高效的泛型容器，通过显式内存管理来实现，然后通过隐式内存管理，最后通过分配器，在多年来的各种形式下。利用我们对内存管理技术和设施的更深入理解，我们将以比简单、更直观的实现方式更有效的方式表达两种类型的容器（一种使用连续内存，另一种使用链式节点）。我们以对 C++内存管理近未来展望结束本部分。

本部分包含以下章节：

+   *第十二章*, *使用显式内存管理编写泛型容器*

+   *第十三章*, *使用隐式内存管理编写泛型容器*

+   *第十四章*, *使用分配器支持的泛型容器编写*

+   *第十五章*, *当代问题*
