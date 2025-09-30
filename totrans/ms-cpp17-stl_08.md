# 分配器

我们在前面的章节中看到，C++对动态内存分配有着爱恨交加的关系。

一方面，从堆中进行动态内存分配是一种“代码异味”；追逐指针可能会损害程序的性能，堆可能会意外耗尽（导致`std::bad_alloc`类型的异常），手动内存管理是如此微妙地困难，以至于 C++11 引入了多种不同的“智能指针”类型来管理复杂性（参见第六章，*智能指针*）。2011 年之后的 C++连续版本也添加了大量的非分配代数数据类型，如`tuple`、`optional`和`variant`（参见第五章，*词汇类型*），这些类型可以在不接触堆的情况下表达所有权或包含关系。

另一方面，新的智能指针类型确实有效地管理了内存管理的复杂性；在现代 C++中，你可以安全地分配和释放内存，而无需使用原始的`new`或`delete`，也无需担心内存泄漏。并且堆分配在许多新的 C++特性（`any`、`function`、`promise`）的“幕后”使用，就像它继续被许多旧特性（`stable_partition`、`vector`）使用一样。

因此，这里存在冲突：如果我们被告知好的 C++代码应避免堆分配，我们如何使用这些伟大的新特性（以及旧特性）呢，这些特性依赖于堆分配？

在大多数情况下，你应该偏向于*使用 C++提供的特性*。如果你想有一个可调整大小的元素向量，你应该使用默认的`std::vector`，除非你在你的情况下测量了使用它的实际性能问题。但也存在一类程序员——在非常受限的环境中工作，如飞行软件——他们必须避免接触堆，原因很简单：“堆”在他们的平台上不存在！在这些嵌入式环境中，整个程序的整个占用空间必须在编译时确定。有些这样的程序简单地避免任何类似于堆分配的算法——如果你从未动态分配任何类型的资源，你永远不会遇到意外的资源耗尽！其他这样的程序虽然使用类似于堆分配的算法，但要求在他们的程序中显式表示“堆”（比如说，通过一个非常大的`char`数组以及用于“保留”和“返回”该数组连续块的功能）。

如果这类程序无法使用 C++提供的功能，如`std::vector`和`std::any`，那将极其不幸。因此，自从 1998 年的原始标准以来，标准库就提供了一种称为“分配器感知”的功能。当一个类型或算法是“分配器感知”的，它为程序员提供了一种指定类型或算法应该如何保留和返回动态内存的方法。这个“如何”被具体化为一个称为“分配器”的对象。

在本章中，我们将学习：

+   “分配器”和“内存资源”的定义

+   如何创建自己的内存资源，该资源从静态缓冲区中分配

+   如何使自己的容器“分配器感知”

+   命名空间`std::pmr`中的标准内存资源类型及其令人惊讶的陷阱

+   C++11 分配器模型中的许多奇怪特性纯粹是为了支持`scoped_allocator_adaptor`

+   什么使一个类型成为“花哨指针”类型，以及这种类型可能在何处有用

# 分配器是内存资源的句柄

在阅读本章时，你必须牢记两个基本概念之间的区别，我将称它们为“内存资源”和“分配器”。一个“内存资源”（一个受标准自身术语启发的名字——你可能更愿意称它为“堆”）是一个长期存在的对象，可以在请求时分配内存块（通常是通过从内存资源本身拥有的一个大内存块中切割出来）。内存资源具有经典的面向对象语义（参见第一章，*经典多态和泛型编程*）：你创建一个内存资源一次，永远不会移动或复制它，内存资源的相等性通常由*对象身份*定义。另一方面，一个“分配器”是一个指向内存资源的短暂句柄。分配器具有指针语义：你可以复制它们，移动它们，并且通常可以随意操作它们，分配器的相等性通常由它们是否指向相同的内存资源来定义。我们可以说分配器“指向”特定的内存资源，我们也可以说分配器“由”那个内存资源“支持”；这两个术语可以互换使用。

当我在本章中谈论“内存资源”和“分配器”时，我将谈论前面的概念。标准库还有一些名为`memory_resource`和`allocator`的类型；每当我谈论这些类型时，我会小心地使用`打字机文本`。这不应该太令人困惑。情况与第二章，“迭代器和范围”相似，在那里我们谈论了“迭代器”以及`std::iterator`。当然，那更容易，因为我只提到`std::iterator`是为了告诉你永远不要使用它；它在良好的 C++代码中没有任何位置。在本章中，我们将了解到`std::pmr::memory_resource`在特定的 C++程序中确实有它的位置！

尽管我描述了分配器为一个“指向”内存资源的句柄，但你应该注意到，有时所涉及的内存资源是一个全局单例——这类单例的一个例子是全局堆，其访问器是全局的`operator new`和`operator delete`。就像一个“捕获”全局变量的 lambda 实际上并没有捕获任何东西一样，由全局堆支持的分配器实际上不需要任何状态。事实上，`std::allocator<T>`就是这样一种无状态的分配器类型——但我们在这里跑题了！

# 复习 - 接口与概念

从第一章，“经典多态与泛型编程”中回忆起，C++提供了两种主要不兼容的处理多态的方法。静态、编译时多态被称为*泛型编程*；它依赖于将多态接口表达为一个*概念*，具有许多可能的*模型*，与接口交互的代码以*模板*的形式表达。动态、运行时多态被称为*经典多态*；它依赖于将多态接口表达为一个*基类*，具有许多可能的*派生类*，与接口交互的代码以对*虚函数*的调用形式表达。

在本章中，我们将第一次（也是最后一次）真正接近泛型编程。除非你能够同时记住两个想法，否则无法理解 C++的分配器：一方面是定义接口的*概念* `Allocator`，另一方面是某些特定的*模型*，例如`std::allocator`，它实现了符合`Allocator`概念的行为。

为了进一步复杂化问题，`Allocator`概念实际上是一个模板化的概念家族！更准确地说，我们应该谈论概念家族`Allocator<T>`；例如，`Allocator<int>`将是定义“分配`int`对象的分配器”的概念，而`Allocator<char>`将是“分配`char`对象的分配器”，等等。例如，具体的类`std::allocator<int>`是概念`Allocator<int>`的一个模型，但它不是`Allocator<char>`的模型。

每个类型`T`的分配器（每个`Allocator<T>`）都必须提供一个名为`allocate`的成员函数，以便`a.allocate(n)`返回足够内存的指针，用于存储类型为`T`的`n`个对象的数组。（该指针将来自支持分配器实例的内存资源。）没有指定`allocate`成员函数应该是静态的还是非静态的，也没有指定它应该恰好接受一个参数（`n`）或者可能接受一些具有默认值的附加参数。因此，以下两种类类型在这一点上都是`Allocator<int>`的可接受模型：

```cpp
    struct int_allocator_2014 {
      int *allocate(size_t n, const void *hint = nullptr);
    };

    struct int_allocator_2017 {
      int *allocate(size_t n);
    };
```

类别名为`int_allocator_2017`显然是建模`Allocator<int>`的*更简单*方法，但`int_allocator_2014`也是一个正确的模型，因为在两种情况下，表达式`a.allocate(n)`都将被编译器接受；这就是我们在谈论*泛型编程*时所要求的一切。

相比之下，当我们进行经典多态时，我们为基类的每个方法指定一个固定的签名，并且不允许派生类偏离该签名：

```cpp
    struct classical_base {
      virtual int *allocate(size_t n) = 0;
    };

    struct classical_derived : public classical_base {
      int *allocate(size_t n) override;
    };
```

派生类`classical_derived`不允许在`allocate`方法的签名上添加任何额外的参数；不允许更改返回类型；不允许使方法`static`。与泛型编程相比，接口在经典多态中更加“锁定”。

由于“锁定”的经典接口自然比开放的抽象接口更容易描述，因此我们将从 C++17 的全新、经典多态的`memory_resource`开始我们的分配器库之旅。

# 使用`memory_resource`定义堆

回想一下，在资源受限的平台，我们可能不允许使用“堆”（例如通过`new`和`delete`），因为平台的运行时可能不支持动态内存分配。但我们可以创建自己的小堆——不是“堆”，而是“一个堆”——并通过编写几个函数`allocate`和`deallocate`来模拟动态内存分配的效果，这些函数保留了一个大静态分配的`char`数组的一部分，类似于这样：

```cpp
    static char big_buffer[10000];
    static size_t index = 0;

    void *allocate(size_t bytes) {
      if (bytes > sizeof big_buffer - index) {
        throw std::bad_alloc();
      }
      index += bytes;
      return &big_buffer[index - bytes];
    }

    void deallocate(void *p, size_t bytes) {
      // drop it on the floor
    }
```

为了使代码尽可能简单，我将`deallocate`设为无操作。这个小堆允许调用者分配最多 10,000 字节的内存，然后从此开始抛出`bad_alloc`异常。

通过在代码上投入更多，我们可以允许调用者无限次地分配和释放内存，只要分配的内存总量不超过 10,000 字节，并且只要调用者始终遵循“最后分配的先释放”的协议：

```cpp
    void deallocate(void *p, size_t bytes) {
      if ((char*)p + bytes == &big_buffer[index]) {
        // aha! we can roll back our index!
        index -= bytes;
      } else {
        // drop it on the floor
      }
    }
```

这里突出的点是，我们的堆有一些*状态*（在这种情况下，`big_buffer`和`index`），以及一些操作这个状态的函数。我们已经看到了`deallocate`的两种不同可能的实现——还有其他可能性，有额外的共享状态，不会那么“漏斗”——然而，接口，`allocate`和`deallocate`函数签名的本身，却保持不变。这表明我们可以将我们的状态和访问函数包装到一个 C++对象中；广泛的实现可能性加上我们函数签名的恒定性表明，我们可以使用一些经典的多态。

C++17 分配器模型正是如此。标准库提供了一个经典多态的基类定义，`std::pmr::memory_resource`，然后我们实现我们自己的小堆作为派生类。（在实践中，我们可能会使用标准库提供的派生类之一，但在讨论这些之前，让我们完成我们的小例子。）基类`std::pmr::memory_resource`在标准头文件`<memory_resource>`中定义：

```cpp
    class memory_resource {
      virtual void *do_allocate(size_t bytes, size_t align) = 0;
      virtual void do_deallocate(void *p, size_t bytes, size_t align) = 0;
      virtual bool do_is_equal(const memory_resource& rhs) const = 0;
    public:
      void *allocate(size_t bytes, size_t align) {
        return do_allocate(bytes, align);
      }
      void deallocate(void *p, size_t bytes, size_t align) {
        return do_deallocate(p, bytes, align);
      }
      bool is_equal(const memory_resource& rhs) const {
        return do_is_equal(rhs);
      }
    };
```

注意到类`public`接口和`virtual`实现之间的奇特间接层。通常当我们进行经典的多态时，我们只有一组既是`public`又是`virtual`的方法；但在这个例子中，我们有一个`public`非虚拟接口，它调用到私有的虚拟方法。这种将接口从实现中分离出来的做法带来了一些微妙的好处——例如，它防止任何子类使用“直接调用虚拟方法非虚拟”的语法来调用`this->SomeBaseClass::allocate()`——但老实说，它对我们来说的主要好处是，当我们定义一个派生类时，我们根本不需要使用`public`关键字。因为我们只指定了*实现*，而不是接口，所以我们写的所有代码都可以是`private`的。这就是我们的微不足道的漏斗堆：

```cpp
    class example_resource : public std::pmr::memory_resource {
      alignas(std::max_align_t) char big_buffer[10000];
      size_t index = 0;
      void *do_allocate(size_t bytes, size_t align) override {
        if (align > alignof(std::max_align_t) ||
            (-index % align) > sizeof big_buffer - index ||
            bytes > sizeof big_buffer - index - (-index % align))
        {
            throw std::bad_alloc();
        }
        index += (-index % align) + bytes;
        return &big_buffer[index - bytes];
      }
      void do_deallocate(void *, size_t, size_t) override {
        // drop it on the floor
      }
      bool do_is_equal(const memory_resource& rhs) const override {
        return this == &rhs;
      }
    };
```

注意到标准库的`std::pmr::memory_resource::allocate`不仅接受字节数，还接受对齐方式。我们需要确保从`do_allocate`返回的任何指针都适当地对齐；例如，如果我们的调用者计划在我们的提供的内存中存储`int`，他可能会要求四字节对齐。

关于我们的派生类`example_resource`的最后一点要注意的是，它代表了由我们的“堆”实际控制的资源；也就是说，它实际上包含、拥有和管理从其中分配内存的`big_buffer`。对于任何给定的`big_buffer`，在我们的程序中将有且只有一个`example_resource`对象来操作该缓冲区。正如我们之前所说的：`example_resource`类型的对象是“内存资源”，因此它们*不*打算被复制或移动；它们是经典面向对象的，而不是值语义的。

标准库提供了几种内存资源类型，它们都源自 `std::pmr::memory_resource`。让我们看看其中的一些。

# 使用标准内存资源

标准库中的内存资源有两种类型。其中一些是实际类类型，你可以创建其实例；还有一些是“匿名”类类型，只能通过单例函数访问。通常，你可以通过思考两个对象是否可能“不同”，或者类型本质上是否是单例来预测它们是哪一种。

在 `<memory_resource>` 头文件中最简单的内存资源是通过 `std::pmr::null_memory_resource()` 访问的“匿名”单例。这个函数的定义可能类似于以下内容：

```cpp
    class UNKNOWN : public std::pmr::memory_resource {
      void *do_allocate(size_t, size_t) override {
        throw std::bad_alloc();
      }
      void do_deallocate(void *, size_t, size_t) override {}
      bool do_is_equal(const memory_resource& rhs) const override {
        return this == &rhs;
      }
    };

    std::pmr::memory_resource *null_memory_resource() noexcept {
      static UNKNOWN singleton;
      return &singleton;
    }
```

注意，该函数返回单例实例的指针。通常，`std::pmr::memory_resource` 对象将通过指针进行操作，因为`memory_resource` 对象本身无法移动。

`null_memory_resource` 似乎相当无用；它所做的只是在你尝试从中分配时抛出异常。然而，当你开始使用我们稍后将看到的更复杂的内存资源时，它可能很有用。

下一个最复杂的内存资源是通过 `std::pmr::new_delete_resource()` 访问的单例；它使用 `::operator new` 和 `::operator delete` 来分配和释放内存。

现在我们来谈谈命名类类型。这些是在单个程序中拥有多个相同类型资源是有意义的资源。例如，有 `class std::pmr::monotonic_buffer_resource`。这种内存资源基本上与之前我们的 `example_resource` 相同，除了两点不同：它不是将其大缓冲区作为成员数据（`std::array` 风格）持有，而是只持有从别处分配的大缓冲区的指针（`std::vector` 风格）。当其第一个大缓冲区耗尽时，它不会立即开始抛出 `bad_alloc` 异常，而是会尝试分配第二个大缓冲区，并从这个缓冲区中分配块，直到它全部用完；此时，它将分配第三个大缓冲区……以此类推，直到最终它甚至无法再分配任何大缓冲区。与我们的 `example_resource` 一样，直到资源对象本身被销毁，所有已释放的内存都不会被释放。有一个有用的出口：如果你调用 `a.release()` 方法，`monotonic_buffer_resource` 将释放它当前持有的所有缓冲区，有点像在向量上调用 `clear()`。

当你构造一个 `std::pmr::monotonic_buffer_resource` 类型的资源时，你需要告诉它两件事：它的第一个大缓冲区在哪里？当该缓冲区耗尽时，它应该向谁请求另一个缓冲区？第一个问题的答案是提供一个 `void*, size_t` 的参数对，它描述了第一个大缓冲区（可选 `nullptr`）；第二个问题的答案是提供一个指向此资源“上游”资源的 `std::pmr::memory_resource*`。对于“上游”资源，一个合理的传递方式是 `std::pmr::new_delete_resource()`，以便使用 `::operator new` 分配新的缓冲区。或者，另一个合理的传递方式是 `std::pmr::null_memory_resource()`，以便对特定资源的内存使用设置一个硬限制。以下是一个后者的示例：

```cpp
    alignas(16) char big_buffer[10000];

    std::pmr::monotonic_buffer_resource a(
      big_buffer, sizeof big_buffer,
      std::pmr::null_memory_resource()
    );

    void *p1 = a.allocate(100);
    assert(p1 == big_buffer + 0);

    void *p2 = a.allocate(100, 16); // alignment
    assert(p1 == big_buffer + 112);

    // Now clear everything allocated so far and start over.
    a.release();
    void *p3 = a.allocate(100);
    assert(p3 == big_buffer + 0);

    // When the buffer is exhausted, a will go upstream
    // to look for more buffers... and not find any.
    try {
      a.allocate(9901);
    } catch (const std::bad_alloc&) {
      puts("The null_memory_resource did its job!");
    }
```

如果你忘记了特定的 `monotonic_buffer_resource` 正在使用哪个上游资源，你可以通过调用 `a.upstream_resource()` 来找出；该方法返回一个指向提供给构造函数的上游资源的指针。

# 从资源池中分配

C++17 标准库提供的最后一种内存资源被称为“池资源”。池资源不仅仅管理一个大的缓冲区，例如 `example_resource`；甚至不是一个单调递增的缓冲区链，例如 `monotonic_buffer_resource`。相反，它管理各种大小的“块”。给定大小的所有块都存储在“池”中，因此我们可以谈论“大小为 4 的块池”、“大小为 16 的块池”等等。当一个请求到来，需要分配大小为 *k* 的资源时，池资源会在大小为 *k* 的块池中查找，取出一个并返回。如果大小为 *k* 的池为空，那么池资源将尝试从其上游资源中分配更多的块。此外，如果一个请求到来，需要分配的块大小如此之大，以至于我们甚至没有该大小的块池，那么池资源允许直接将请求传递给其上游资源。

池资源有两种类型：*同步* 和 *异步*，也就是说，线程安全和线程不安全。如果你将同时从两个不同的线程访问池，那么你应该使用 `std::pmr::synchronized_pool_resource`，如果你肯定永远不会这样做，并且想要原始速度，那么你应该使用 `std::pmr::unsynchronized_pool_resource`。（顺便说一下，`std::pmr::monotonic_buffer_resource` 总是线程不安全的；而 `new_delete_resource()` 实际上是线程安全的，因为它只是调用 `new` 和 `delete`。）

当你构建一个类型为 `std::pmr::synchronized_pool_resource` 的资源时，你需要告诉它三件事情：它应该在池中保留哪些块大小；当它从上游资源获取更多块时，应该将多少块组合成一个“块组”；以及它的上游资源是谁。不幸的是，标准接口在这里留下了很多遗憾——如此之多，以至于坦白地说，我建议如果你真正关心这些参数，你应该实现自己的派生 `memory_resource`，而不要触及标准库的版本。表达这些选项的语法也相当复杂：

```cpp
    std::pmr::pool_options options;
    options.max_blocks_per_chunk = 100;
    options.largest_required_pool_block = 256;

    std::pmr::synchronized_pool_resource a(
      options,
      std::pmr::new_delete_resource()
    );
```

注意，无法指定确切的块大小；这留给供应商对 `synchronized_pool_resource` 的实现。如果你很幸运，它可能会选择适合你用例的合理块大小；但个人来说，我不会依赖这个假设。注意，也无法为不同的块大小使用不同的上游资源，也没有为当调用者请求异常大小的分配时使用的“后备”资源使用不同的上游资源。

简而言之，在可预见的未来，我会避开内置的 `pool_resource` 派生类。但从 `memory_resource` 派生自己的类的根本思想是稳固的。如果你担心内存分配和管理你自己的小堆，我建议将 `memory_resource` 纳入你的代码库。

现在，到目前为止，我们一直在谈论各种分配策略，这些策略由不同的 `memory_resource` 派生类“体现”。我们仍然需要看看如何将 `memory_resource` 集成到标准模板库的算法和容器中。为此，我们必须从 `memory_resource` 的经典多态世界过渡到 C++03 STL 的值语义世界。

# 标准分配器的 500 顶帽子

标准分配器模型在 2011 年看起来很神奇。我们将看到，仅使用一种 C++ 类型，我们就可以完成以下所有任务：

+   指定用于分配内存的内存资源。

+   在每个分配的指针上注解一些将随指针一起传递的元数据

    在其整个生命周期内，一直到释放时间。

+   将一个容器对象与特定的内存资源关联起来，并确保

    这种关联是“粘性的”——这个容器对象将始终使用给定的

    为其分配使用堆。

+   将一个容器 *值* 与特定的内存资源关联起来，这意味着

    容器可以使用值语义高效地移动，而无需

    忘记如何释放其内容。

+   在上述两种互斥行为之间进行选择。

+   在多级结构的所有级别上指定分配内存的策略

    容器，例如向量中的向量。

+   重新定义“构造”容器内容的意义，以便

    例如，`vector<int>::resize` 可以被定义为对新元素进行默认初始化，而不是零初始化。

这对于任何单个类类型来说都是一个**疯狂**的帽子数量——这是对单一责任原则的严重违反。尽管如此，这正是标准分配器模型所做的事情；所以让我们尝试解释所有这些特性。

记住，“标准分配器”只是任何满足某些类型 `T` 的概念 `Allocator<T>` 的类类型。标准库提供了三种标准分配器类型：`std::allocator<T>`、`std::pmr::polymorphic_allocator<T>` 和 `std::scoped_allocator_adaptor<A...>`。

让我们先看看 `std::allocator<T>`：

```cpp
    template<class T>
    struct allocator {
      using value_type = T;

      T *allocate(size_t n) {
        return static_cast<T *>(::operator new(n * sizeof (T)));
      }
      void deallocate(T *p, size_t) {
        ::operator delete(static_cast<void *>(p));
      }

      // NOTE 1
      template<class U>
      explicit allocator(const allocator<U>&) noexcept {}

      // NOTE 2
      allocator() = default;
      allocator(const allocator&) = default;
    };
```

`std::allocator<T>` 有 `allocate` 和 `deallocate` 成员函数，这些函数是 `Allocator<T>` 概念所要求的。记住，我们现在处于基于概念泛型编程的世界！经典的多态 `memory_resource` 也**同样**有名为 `allocate` 和 `deallocate` 的成员函数，但它们总是返回 `void*`，而不是 `T*`。（此外，`memory_resource::allocate()` 接受两个参数——`bytes` 和 `align`——而 `allocator<T>::allocate()` 只接受一个参数。第一个原因是 `allocator<T>` 产生于对对齐重要性的主流理解之前；记住，`sizeof` 操作符是从 20 世纪 80 年代的 C 语言继承而来的，但 `alignof` 操作符只出现在 C++11 中。第二个原因是，在 `std::allocator<T>` 的上下文中，我们知道正在分配的对象类型是 `T`，因此请求的对齐必须是 `alignof(T)`。`std::allocator<T>` 不使用这个信息，因为它早于 `alignof`；但原则上它可以，这就是为什么 `Allocator<T>` 概念只要求 `a.allocate(n)` 的签名，而不是 `a.allocate(n, align)` 的原因。）

标记为 `NOTE 1` 的构造函数很重要；每个分配器都需要一个模板构造函数，其模式与此类似。标记为 `NOTE 2` 的后续构造函数并不重要；我们之所以在代码中明确写出它们，仅仅是因为如果我们没有写出它们，由于存在用户定义的构造函数（即 `NOTE 1` 构造函数），它们将被隐式删除。

任何标准分配器的想法是，我们可以将其作为任何标准容器（第四章，*容器动物园*）的最后一个模板类型参数插入，然后容器将在需要为任何原因分配内存时使用该分配器而不是其通常的机制。让我们看一个例子：

```cpp
    template<class T>
    struct helloworld {
      using value_type = T;

      T *allocate(size_t n) {
        printf("hello world %zu\n", n);
        return static_cast<T *>(::operator new(n * sizeof (T)));
      }
      void deallocate(T *p, size_t) {
        ::operator delete(static_cast<void *>(p));
      }
    };

    void test() {
      std::vector<int, helloworld<int>> v;
      v.push_back(42); // prints "hello world 1"
      v.push_back(42); // prints "hello world 2"
      v.push_back(42); // prints "hello world 4"
    }
```

在这里，我们的类 `helloworld<int>` 模拟 `Allocator<int>`；但我们省略了模板构造函数。如果我们只处理 `vector`，这是可以的，因为 `vector` 只会为其元素类型分配数组。然而，如果我们改变测试用例以使用 `list` 代替，看看会发生什么：

```cpp
    void test() {
      std::list<int, helloworld<int>> v;
      v.push_back(42);
    }
```

在 libc++ 下，这段代码会输出几十行错误信息，归结为基本抱怨：“没有已知从 `helloworld<int>` 转换到 `helloworld<std::__1::__list_node<int, void *>>` 的转换。”回想一下 第四章 中的图，“容器动物园”，`std::list<T>` 存储其元素在比 `T` 本身更大的节点中。因此，`std::list<T>` 不打算尝试分配任何 `T` 对象；它想要分配 `__list_node` 类型的对象。为了为 `__list_node` 对象分配内存，它需要一个模拟 `Allocator<__list_node>` 概念的分配器，而不是 `Allocator<int>`。

在内部，`std::list<int>` 的构造函数尝试将我们的 `helloworld<int>` “重新绑定”为分配 `__list_node` 对象而不是 `int` 对象。这是通过一个 *特性类--* 一个我们在 第二章，“迭代器和范围”中首次遇到的 C++ 习语来实现的：

```cpp
    using AllocOfInt = helloworld<int>;

    using AllocOfChar =
      std::allocator_traits<AllocOfInt>::rebind_alloc<char>;

    // Now alloc_of_char is helloworld<char>
```

标准类模板 `std::allocator_traits<A>` 将关于分配器类型 `A` 的许多信息封装在一个地方，因此很容易访问。例如，`std::allocator_traits<A>::value_type` 是一个别名，表示由 `A` 分配的内存的类型 `T`；而 `std::allocator_traits<A>::pointer` 是对应指针类型的别名（通常是 `T*`）。

嵌套别名模板 `std::allocator_traits<A>::rebind_alloc<U>` 是一种将分配器从一种类型 `T` 转换为另一种类型 `U` 的方法。这种类型特性使用元编程来打开类型 `A` 并查看：首先，`A` 是否有一个嵌套模板别名 `A::rebind<U>::other`（这种情况很少见），其次，类型 `A` 是否可以表示为 `Foo<Bar,Baz...>` 的形式（其中 `Baz...` 是一些类型列表，可能是一个空列表）。如果 `A` 可以以这种方式表示，那么 `std::allocator_traits<A>::rebind_alloc<U>` 将是 `Foo<U,Baz...>` 的同义词。从哲学上讲，这是完全任意的；但在实践中，它适用于你将看到的每个分配器类型。特别是，它适用于 `helloworld<int>`--这也解释了为什么我们不需要在我们的 `helloworld` 类中提供嵌套别名 `rebind<U>::other`。通过提供合理的默认行为，`std::allocator_traits` 模板为我们节省了一些样板代码。这就是 `std::allocator_traits` 存在的原因。

你可能会想知道为什么 `std::allocator_traits<Foo<Bar,Baz...>>::value_type` 不默认为 `Bar`。坦白说，我也不知道。这似乎是一个显而易见的事情；但标准库没有这样做。因此，你必须为每个你编写的分配器类型（记住我们现在在谈论模拟 `Allocator<T>` 的类，而不是从 `memory_resource` 派生的类）提供一个嵌套 typedef `value_type`，它是一个 `T` 的别名。

然而，一旦你为 `value_type` 定义了嵌套类型别名，你就可以依赖 `std::allocator_traits` 来推断其嵌套类型别名 `pointer`（即 `T*`）、`const_pointer`（即 `const T*`）、`void_pointer`（即 `void*`）等的正确定义。如果你跟随了之前关于 `rebind_alloc` 的讨论，你可能会猜测将指针类型如 `T*` 转换为 `void*` 与将分配器类型 `Foo<T>` 转换为 `Foo<void>` 一样困难或容易；你是对的！这些指针相关类型别名的值都是通过第二个标准特性类 `std::pointer_traits<P>` 计算得出的：

```cpp
    using PtrToInt = int*;

    using PtrToChar =
      std::pointer_traits<PtrToInt>::rebind<char>;

    // Now PtrToChar is char*

    using PtrToConstVoid =
      std::pointer_traits<PtrToInt>::rebind<const void>;

    // Now PtrToConstVoid is const void*
```

当我们讨论 `Allocator<T>` 的下一个职责时，这个特性类变得非常重要，即“为每个分配的指针添加一些将在其整个生命周期中携带的元数据。”

# 与花哨指针一起携带元数据

考虑以下内存资源的高级设计，这应该会让你非常想起 `std::pmr::monotonic_buffer_resource`：

+   维护一个我们从系统获取的内存块的列表。对于每个块，还存储从块开始分配的字节数的 `index`；并存储从该特定块中已释放的字节数的计数 `freed`。

+   当有人调用 `allocate(n)` 时，如果可能，增加我们任何一个块的 `index` 以适当的字节数，或者在绝对必要时从上游资源获取一个新的块。

+   当有人调用 `deallocate(p, n)` 时，找出 `p` 来自我们的哪个块，并增加其 `freed += n`。如果 `freed == index`，则整个块为空，因此将 `freed = index = 0`。

将上述描述转换为代码相当直接。唯一的问题在于：在 `deallocate(p, n)` 中，我们如何确定 `p` 来自我们的哪个块？

如果我们只是在“指针”本身中记录块的标识符，这将很容易：

```cpp
    template<class T>
    class ChunkyPtr {
      T *m_ptr = nullptr;
      Chunk *m_chunk = nullptr;
    public:
      explicit ChunkyPtr(T *p, Chunk *ch) :
      m_ptr(p), m_chunk(ch) {}

      T& operator *() const {
        return *m_ptr;
      }
      explicit operator T *() const {
        return m_ptr;
      }
      // ... and so on ...

      // ... plus this extra accessor:
      auto chunk() const {
        return m_chunk;
      }
    };
```

然后在我们的 `deallocate(p, n)` 函数中，我们只需查看 `p.chunk()`。但要使这生效，我们需要更改 `allocate(n)` 和 `deallocate(p, n)` 函数的签名，使 `deallocate` 接受 `ChunkyPtr<T>` 而不是 `T*`，并且 `allocate` 返回 `ChunkyPtr<T>` 而不是 `T*`。

幸运的是，C++ 标准库为我们提供了一种方法来做这件事！我们只需要定义自己的类型来模拟 `Allocator<T>`，并给它一个成员类型别名 `pointer`，其值为 `ChunkyPtr<T>`：

```cpp
    template<class T>
    struct ChunkyAllocator {
      using value_type = T;
      using pointer = ChunkyPtr<T>;

      ChunkyAllocator(ChunkyMemoryResource *mr) :
        m_resource(mr) {}

      template<class U>
      ChunkyAllocator(const ChunkyAllocator<U>& rhs) :
        m_resource(rhs.m_resource) {}

      pointer allocate(size_t n) {
        return m_resource->allocate(
          n * sizeof(T), alignof(T));
      } 
      void deallocate(pointer p, size_t n) {
        m_resource->deallocate(
          p, n * sizeof(T), alignof(T));
      }
    private:
      ChunkyMemoryResource *m_resource;

      template<class U>
      friend struct ChunkyAllocator;
    };
```

特性类 `std::allocator_traits` 和 `std::pointer_traits` 将会负责推断其他类型别名--例如 `void_pointer`，它通过 `pointer_traits::rebind` 的魔法最终会成为 `ChunkyPtr<void>` 的别名。

我在这里省略了 `allocate` 和 `deallocate` 函数的实现，因为它们将依赖于 `ChunkyMemoryResource` 的接口。我们可能会像这样实现 `ChunkyMemoryResource`：

```cpp
    class Chunk {
      char buffer[10000];
      int index = 0;
      int freed = 0;
    public:
      bool can_allocate(size_t bytes) {
        return (sizeof buffer - index) >= bytes;
      }
      auto allocate(size_t bytes) {
        index += bytes;
        void *p = &buffer[index - bytes];
        return ChunkyPtr<void>(p, this);
      }
      void deallocate(void *, size_t bytes) {
        freed += bytes;
        if (freed == index) {
            index = freed = 0;
        }
      }
    };

    class ChunkyMemoryResource {
      std::list<Chunk> m_chunks;
    public:
      ChunkyPtr<void> allocate(size_t bytes, size_t align) {
        assert(align <= alignof(std::max_align_t));
        bytes += -bytes % alignof(std::max_align_t);
        assert(bytes <= 10000);

        for (auto&& ch : m_chunks) {
          if (ch.can_allocate(bytes)) {
            return ch.allocate(bytes);
          }
        }
        return m_chunks.emplace_back().allocate(bytes);
      }
      void deallocate(ChunkyPtr<void> p, size_t bytes, size_t) {
        bytes += -bytes % alignof(std::max_align_t);
        p.chunk()->deallocate(static_cast<void*>(p), bytes);
      }
    };
```

现在我们可以使用我们的`ChunkyMemoryResource`为像这样的标准分配器感知容器分配内存：

```cpp
    ChunkyMemoryResource mr;
    std::vector<int, ChunkyAllocator<int>> v{&mr};
    v.push_back(42);
    // All the memory for v's underlying array
    // is coming from blocks owned by "mr".
```

现在，我选择这个例子是为了让它看起来非常简单直接；并且我省略了`ChunkyPtr<T>`类型本身的许多细节。如果你尝试自己复制这段代码，你会发现你需要为`ChunkyPtr`提供许多重载运算符，例如`==`, `!=`, `<`, `++`, `--`, 和`-`；你还需要为`ChunkyPtr<void>`提供一个特化，该特化省略了重载的`operator*`。大部分的细节与我们在第二章，“迭代器和范围”，当我们实现自己的迭代器类型时所覆盖的内容相同。实际上，每个“花哨指针”类型都必须能够作为*随机访问迭代器*使用——这意味着你必须提供第二章，“迭代器和范围”末尾列出的五个嵌套 typedef：`iterator_category`, `difference_type`, `value_type`, `pointer`, 和`reference`。

最后，如果你想使用某些容器，例如`std::list`和`std::map`，你需要实现一个名为`pointer_to(r)`的静态成员函数：

```cpp
    static ChunkyPtr<T> pointer_to(T &r) noexcept {
      return ChunkyPtr<T>(&r, nullptr);
    }
```

这是因为——正如你可能从第四章，“容器动物园”中回忆起来——一些容器，例如`std::list`，将它们的数据存储在节点中，这些节点的`prev`和`next`指针需要能够指向*任意*一个已分配的节点*或者*指向包含在`std::list`对象本身成员数据中的节点。有两种明显的方法可以实现这一点：要么每个`next`指针都必须存储在一个带有花哨指针和原始指针（可能是一个`std::variant`，如第五章，“词汇类型”中描述的）的标记联合体中，要么我们必须找到一种方法将原始指针*编码*为花哨指针。标准库选择了后者。所以，每当您编写一个花哨指针类型时，它不仅必须完成分配器要求的所有事情，而且它必须满足随机访问迭代器的需求，而且它还必须*也有*一种表示程序地址空间中任何任意指针的方法——至少如果您想使用您的分配器与基于节点的容器，如`std::list`。

即使跳过了所有这些障碍，你也会发现（截至出版时间），libc++和 libstdc++都无法处理比`std::vector`更复杂的任何容器中的花哨指针。它们只支持足够的操作与单个花哨指针类型一起工作——`boost::interprocess::offset_ptr<T>`，它不携带元数据。而且标准仍在不断发展；`std::pmr::memory_resource`是在 C++17 中新引入的，截至本文撰写时，它还没有被 libc++和 libstdc++实现。

你可能也注意到了缺少任何使用花哨指针的内存资源的标准基类。幸运的是，这很容易自己编写：

```cpp
    namespace my {

      template<class VoidPtr>
      class fancy_memory_resource {
      public:
        VoidPtr allocate(size_t bytes,
          size_t align = alignof(std::max_align_t)) {
          return do_allocate(bytes, align);
        }
        void deallocate(VoidPtr p, size_t bytes,
          size_t align = alignof(std::max_align_t)) {
          return do_deallocate(p, bytes, align);
        }
        bool is_equal(const fancy_memory_resource& rhs) const noexcept {
          return do_is_equal(rhs);
        }
        virtual ~fancy_memory_resource() = default;
      private:
        virtual VoidPtr do_allocate(size_t bytes, size_t align) = 0;
        virtual void do_deallocate(VoidPtr p, size_t bytes,
          size_t align) = 0;
        virtual bool do_is_equal(const fancy_memory_resource& rhs)
          const noexcept = 0;
      };

      using memory_resource = fancy_memory_resource<void*>;

    } // namespace my
```

标准库不提供使用花哨指针的分配器；每个库提供的分配器类型都使用原始指针。

# 将容器固定到单个内存资源上

标准分配器模型戴上的下一个帽子——由`std::allocator_traits`控制的下一个特性——是能够将特定的容器对象与特定的堆关联起来。我们之前用三个项目符号描述了这一特性：

+   将容器对象与特定的内存资源关联起来，并确保

    这种关联是“粘性的”——这个容器对象将始终使用给定的

    使用堆进行分配

+   将容器*值*与特定的内存资源关联起来，意味着

    容器可以使用值语义有效地移动，而无需

    忘记如何释放其内容

+   在上述两种互斥行为之间进行选择。

让我们看看一个例子，使用`std::pmr::monotonic_buffer_resource`作为我们的资源，但使用手写的类类型作为我们的分配器类型。（只是为了让你放心，你确实没有错过任何东西：实际上，我们仍然没有涵盖任何标准库提供的分配器类型——除了`std::allocator<T>`，这是一个平凡的无状态分配器，它是`new`和`delete`管理的全局堆的句柄。）

```cpp
    template<class T>
    struct WidgetAlloc {
      std::pmr::memory_resource *mr;

      using value_type = T;

      WidgetAlloc(std::pmr::memory_resource *mr) : mr(mr) {}

      template<class U>
      WidgetAlloc(const WidgetAlloc<U>& rhs) : mr(rhs.mr) {}

      T *allocate(size_t n) {
        return (T *)mr->allocate(n * sizeof(T), alignof(T));
      }
      void deallocate(void *p, size_t n) {
        mr->deallocate(p, n * sizeof(T), alignof(T));
      }
    };

    class Widget {
      char buffer[10000];
      std::pmr::monotonic_buffer_resource mr {buffer, sizeof buffer};
      std::vector<int, WidgetAlloc<int>> v {&mr};
      std::list<int, WidgetAlloc<int>> lst {&mr};
    public:
      static void swap_elems(Widget& a, Widget& b) {
        std::swap(a.v, b.v);
      }
    };
```

在这里，我们的`Widget`是一个经典的面向对象类类型；我们期望它在整个生命周期中存在于特定的内存地址。然后，为了减少堆碎片或提高缓存局部性，我们在每个`Widget`对象内部放置了一个大缓冲区，并使`Widget`使用该缓冲区作为其数据成员`v`和`lst`的后备存储。

现在看看`Widget::swap_elems(a, b)`函数。它交换了`Widget a`和`Widget b`的`v`数据成员。你可能还记得第四章，“容器动物园”，其中`std::vector`不过是一个指向动态分配数组的指针，因此通常库可以通过简单地交换它们的底层指针来交换两个`std::vector`实例，而不需要移动任何底层数据——使得向量交换成为 O(1)操作而不是 O(*n*)操作。

此外，`vector`足够智能，知道如果它交换指针，它还需要交换分配器——这样关于如何释放的信息就会随着最终需要释放的指针一起传递。

但在这种情况下，如果库只是交换了指针和分配器，那将是一场灾难！我们会有一个向量 `a.v`，其底层数组现在“属于”`b.mr`，反之亦然。如果我们销毁 `Widget b`，那么下次我们访问 `a.v` 的元素时，我们将访问已释放的内存。而且更进一步，即使我们以后再也不访问 `a.v`，当 `a.v` 的析构函数尝试调用早已死亡的 `b.mr` 的 `deallocate` 方法时，我们的程序很可能会崩溃！

幸运的是，标准库救了我们于水火。一个分配器感知容器的一个责任是在复制赋值、移动赋值和交换时适当地*传播*其分配器。由于历史原因，这由 `allocator_traits` 类模板中的大量 typedef 处理，但为了*正确使用*分配器传播，你只需要知道几件事情：

+   分配器是否传播自身，或者是否坚定地粘附在特定的容器上，是*分配器类型*的一个属性。如果你想使一个分配器“粘性”而另一个分配器传播，你必须使它们成为不同的类型。

+   当一个分配器“粘性”时，它会粘附在特定的（经典、面向对象的）

    容器对象。在非粘性分配器类型下原本是 O(1) 指针交换的操作可能会变成 O(*n*)，因为“采用”来自某个其他分配器内存空间中的元素到我们自己的内存空间中，需要在我们自己的内存空间中为它们分配空间。

+   粘性有明确的用例（正如我们刚刚用 `Widget` 展示的那样），并且

    非粘性的影响可能是灾难性的（再次，参见 `Widget`）。因此，`std::allocator_traits` 默认假设分配器类型是粘性的，除非它能判断出分配器类型是*空的*，因此绝对是*无状态的*。对于*空的*分配器类型，默认实际上是粘性。

+   作为程序员，你基本上总是想要默认状态：无状态的分配器可以传播，而状态性的分配器*可能*在需要粘性的 `Widget` 类似场景之外没有太多用途。

# 使用标准分配器类型

让我们谈谈标准库提供的分配器类型。

`std::allocator<T>` 是默认的分配器类型；它是每个标准容器模板类型参数的默认值。所以例如，当你代码中写 `std::vector<T>` 时，这实际上是 `std::vector<T, std::allocator<T>>` 的完全相同类型。正如我们本章前面提到的，`std::allocator<T>` 是一个无状态的空类型；它是 `new` 和 `delete` 管理的全局堆的“句柄”。由于 `std::allocator` 是一个无状态类型，`allocator_traits` 假定（正确地）它应该是非粘性的。这意味着操作如 `std::vector<T>::swap` 和 `std::vector<T>::operator=` 保证是非常高效的指针交换操作——因为任何 `std::vector<T, std::allocator<T>>` 类型的对象总是知道如何释放由任何其他 `std::vector<T, std::allocator<T>>` 分配的内存。

`std::pmr::polymorphic_allocator<T>` 是 C++17 中新增的一个类型。它是一个有状态的、非空类型；它有一个数据成员，是一个指向 `std::pmr::memory_resource` 的指针。（实际上，它与本章早期示例中的 `WidgetAlloc` 几乎相同！）两个不同的 `std::pmr::polymorphic_allocator<T>` 实例不一定可以互换，因为它们的指针可能指向完全不同的 `memory_resource`；这意味着 `std::vector<T, std::pmr::polymorphic_allocator<T>>` 类型的对象不一定知道如何释放由其他 `std::vector<T, std::pmr::polymorphic_allocator<T>>` 分配的内存。这反过来意味着 `std::pmr::polymorphic_allocator<T>` 是一个“粘性”分配器类型；这意味着操作如 `std::vector<T, std::pmr::polymorphic_allocator<T>>::operator=` 可能会导致大量的复制。

顺便说一下，反复写出 `std::vector<T, std::pmr::polymorphic_allocator<T>>` 这个类型名称相当繁琐。幸运的是，标准库实现者得出了相同的认识，因此标准库在 `std::pmr` 命名空间中提供了类型别名：

```cpp
    namespace std::pmr {

      template<class T>
      using vector = std::vector<T,
        polymorphic_allocator<T>>;

      template<class K, class V, class Cmp = std::less<K>>
      using map = std::map<K, V, Cmp,
        polymorphic_allocator<typename std::map<K, V>::value_type>>;

      // ...

    } // namespace std::pmr
```

# 设置默认内存资源

标准库中的 `polymorphic_allocator` 与我们的示例 `WidgetAlloc` 之间最大的区别是 `polymorphic_allocator` 可以默认构造。默认构造性是一个分配器的有吸引力的特性；这意味着我们可以写出这两行中的第二行而不是第一行：

```cpp
    std::pmr::vector<int> v2({1, 2, 3}, std::pmr::new_delete_resource());
        // Specifying a specific memory resource

    std::pmr::vector<int> v1 = {1, 2, 3};
        // Using the default memory resource
```

另一方面，当你看到第二行时，你可能会想，“底层数组实际上是在哪里被分配的？”毕竟，指定分配器的关键点是我们想知道我们的字节是从哪里来的！这就是为什么构建标准`polymorphic_allocator`的*正常*方式是传递一个指向`memory_resource`的指针——实际上，这个习惯用法预计会非常常见，以至于从`std::pmr::memory_resource*`到`std::pmr::polymorphic_allocator`的转换是一个隐式转换。但是`polymorphic_allocator`也有一个默认的无参数构造函数。当你默认构造一个`polymorphic_allocator`时，你得到一个指向“默认内存资源”的句柄，默认情况下是`new_delete_resource()`。然而，你可以改变这个！默认内存资源指针存储在一个全局原子（线程安全）变量中，可以使用库函数`std::pmr::get_default_resource()`（返回指针）和`std::pmr::set_default_resource()`（将新值赋给指针并返回旧值）来操作。

如果你完全想避免通过`new`和`delete`进行堆分配，那么在程序开始时调用`std::pmr::set_default_resource(std::pmr::null_memory_resource())`可能是有意义的。当然，你无法阻止程序的其他部分变得混乱并自行调用`set_default_resource`；并且由于相同的全局变量被程序中的每个线程共享，如果在程序执行期间*尝试*修改默认资源，你可能会遇到一些非常奇怪的行为。例如，无法说“只为我的当前线程设置默认资源”。此外，调用`get_default_resource()`（例如从`polymorphic_allocator`的默认构造函数中）执行原子访问，这通常会比如果可以避免原子访问而稍微慢一些。因此，你最好的行动方案是避免`polymorphic_allocator`的默认构造函数；始终明确你正在尝试使用哪种内存资源。为了绝对的安全，你可能考虑简单地使用上述`WidgetAlloc`而不是`polymorphic_allocator`；由于`WidgetAlloc`没有默认构造函数，它根本不可能被误用。

# 使容器具有分配器意识

在覆盖了内存资源（堆）和分配器（堆的句柄）之后，现在让我们转向三脚架的第三条腿：容器类。在每一个具有分配器意识的容器内部，至少必须发生以下四件事情：

+   容器实例必须将分配器实例作为成员数据存储。（因此，容器必须将分配器的类型作为模板参数；否则，它不知道为该成员变量预留多少空间。）

+   容器必须提供接受分配器参数的构造函数。

+   容器实际上必须使用其分配器来分配和释放内存；任何使用`new`或`delete`的操作都必须被禁止。

+   容器的移动构造函数、移动赋值运算符和`swap`函数都必须根据其`allocator_traits`传播分配器。

这里有一个非常简单的感知分配器的容器——一个只包含一个对象的容器，在堆上分配。这类似于第六章中*智能指针*的分配器感知版本`std::unique_ptr<T>`：

```cpp
    template<class T, class A = std::allocator<T>>
    class uniqueish {
      using Traits = std::allocator_traits<A>;
      using FancyPtr = typename Traits::pointer;

      A m_allocator;
      FancyPtr m_ptr = nullptr;

    public:
      using allocator_type = A;

      uniqueish(A a = {}) : m_allocator(a) {
        this->emplace();
      }

      ~uniqueish() {
        clear();
      }

      T& value() { return *m_ptr; }
      const T& value() const { return *m_ptr; }

      template<class... Args>
      void emplace(Args&&... args) {
        clear();
        m_ptr = Traits::allocate(m_allocator, 1);
        try {
          T *raw_ptr = static_cast<T *>(m_ptr);
          Traits::construct(m_allocator, raw_ptr,
              std::forward<Args>(args)...
          );
        } catch (...) {
          Traits::deallocate(m_allocator, m_ptr, 1);
          throw;
        }
      }

      void clear() noexcept {
        if (m_ptr) {
          T *raw_ptr = static_cast<T *>(m_ptr);
          Traits::destroy(m_allocator, raw_ptr);
          Traits::deallocate(m_allocator, m_ptr, 1);
          m_ptr = nullptr;
        }
      }
    };
```

注意，`unique_ptr`使用`T*`的地方，我们当前的代码使用`allocator_traits<A>::pointer`；而`make_unique`使用`new`和`delete`的地方，我们当前的代码使用`allocator_traits<A>::allocate`/`construct`和`allocator_traits<A>::destroy`/`deallocate`的一击两式。我们已经讨论了`allocate`和`deallocate`的目的——它们处理从适当的内存资源获取内存。但是，这些内存块只是原始的字节；为了将一块内存转换成一个可用的对象，我们必须在那个地址构造一个`T`的实例。我们可以使用“placement `new`”语法来达到这个目的；但我们将看到在下一节中为什么使用`construct`和`destroy`是重要的。

最后，在我们继续之前，请注意`uniqueish`析构函数在尝试释放分配之前会检查是否存在分配。这很重要，因为它给我们一个代表“空对象”的`uniqueish`值——一个可以在不分配任何内存的情况下构造的值，并且是我们类型的一个合适的“移动后”表示。

现在我们来实现我们类型的移动操作。我们希望确保在从`uniqueish<T>`对象中移动之后，移动后的对象是“空的”。此外，如果左侧对象和右侧对象使用相同的分配器，或者如果分配器类型是“非粘性的”，那么我们希望根本不调用`T`的移动构造函数——我们希望将分配的指针的所有权从右侧对象转移到左侧对象：

```cpp
    uniqueish(uniqueish&& rhs) : m_allocator(rhs.m_allocator) 
    {
      m_ptr = std::exchange(rhs.m_ptr, nullptr);
    }

    uniqueish& operator=(uniqueish&& rhs)
    {
      constexpr bool pocma =
        Traits::propagate_on_container_move_assignment::value;
      if constexpr (pocma) {
        // We can adopt the new allocator, since
        // our allocator type is not "sticky".
        this->clear(); // using the old allocator
        this->m_allocator = rhs.m_allocator;
        this->m_ptr = std::exchange(rhs.m_ptr, nullptr);
      } else if (m_allocator() == rhs.m_allocator()) {
        // Our allocator is "stuck" to this container;
        // but since it's equivalent to rhs's allocator,
        // we can still adopt rhs's memory.
        this->clear();
        this->m_ptr = std::exchange(rhs.m_ptr, nullptr);
      } else {
        // We must not propagate this new allocator
        // and thus cannot adopt its memory.
        if (rhs.m_ptr) {
          this->emplace(std::move(rhs.value()));
          rhs.clear();
        } else {
          this->clear();
        }
      }
      return *this;
    }
```

移动构造函数就像它曾经一样简单。唯一的细微差别是我们必须记住将我们的`m_allocator`构造为右侧对象分配器的副本。

我们可以使用`std::move`来移动分配器而不是复制它，但我觉得在这个例子中这样做不值得。记住，分配器只是一个指向实际内存资源的薄“句柄”，并且许多分配器类型，如`std::allocator<T>`，实际上都是空的。复制分配器类型应该总是相对便宜的。尽管如此，在这里使用`std::move`并不会造成伤害。

另一方面，移动 *赋值运算符* 非常复杂！我们首先需要做的是检查我们的分配器类型是否是“粘性”的。非粘性通过 `propagate_on_container_move_assignment::value` 的真值表示，我们将其缩写为 "`pocma`"。（实际上，标准说 `propagate_on_container_move_assignment` 应该是 `std::true_type` 类型；GNU 的 libstdc++ 会严格遵循这一要求。所以当定义自己的分配器类型时要小心。）如果分配器类型是非粘性的，那么我们移动赋值的最高效做法是销毁我们的当前值（如果有的话）——确保使用我们的旧 `m_allocator`——然后采用右手对象的指针及其分配器。因为我们同时采用指针和分配器，我们可以确信将来我们会知道如何释放那个指针。

另一方面，如果我们的分配器类型 *是* “粘性”的，那么我们就不能采用右手对象的分配器。如果我们的当前（“卡住”）的分配器实例恰好等于右手对象的分配器实例，那么我们无论如何都可以采用右手对象的指针；我们已经知道如何处理由这个特定分配器实例分配的指针。

最后，如果我们不能采用右手对象的分配器实例，并且我们的当前分配器实例不等于右手对象的，那么我们就不能采用右手对象的指针——因为将来某个时候我们得释放那个指针，而唯一释放那个指针的方法是使用右手对象的分配器实例，但我们不允许采用右手对象的分配器实例，因为我们的实例是“卡住”的。在这种情况下，我们实际上必须使用自己的分配器实例分配一个全新的指针，然后通过调用 `T` 的移动构造函数将数据从 `rhs.value()` 复制到我们的值。这种情况是唯一一个我们实际上调用 `T` 的移动构造函数的情况！

复制赋值在传播右手分配器实例的逻辑上遵循类似的逻辑，除了它查看特性 `propagate_on_container_copy_assignment`，或称为 "`pocca`"。

交换特别有趣，因为它的最终情况（当分配器类型是“粘性”且分配器实例不相等时）需要额外的分配：

```cpp
    void swap(uniqueish& rhs) noexcept {
      constexpr bool pocs =
        Traits::propagate_on_container_swap::value;
      using std::swap;
      if constexpr (pocs) {
        // We can swap allocators, since
        // our allocator type is not "sticky".
        swap(this->m_allocator, rhs.m_allocator);
        swap(this->m_ptr, rhs.m_ptr);
      } else if (m_allocator == rhs.m_allocator) {
        // Our allocator is "stuck" to this container;
        // but since it's equivalent to rhs's allocator,
        // we can still adopt rhs's memory and vice versa.
        swap(this->m_ptr, rhs.m_ptr);
      } else {
        // Neither side can adopt the other's memory, and
        // so one side or the other must allocate.
        auto temp = std::move(*this);
        *this = std::move(rhs); // might throw
        rhs = std::move(temp); // might throw
      }
    }
```

在标记为“可能抛出异常”的两行中，我们正在调用移动赋值运算符，在这种情况下可能会调用`emplace`，这将要求分配器分配内存。如果底层内存资源已经耗尽，那么`Traits::allocate(m_allocator, 1)`可能会抛出异常--然后我们就会遇到麻烦，原因有两个。首先，我们已经开始移动状态并释放旧内存，我们可能发现无法“回滚”到一个合理的状态。其次，更重要的是，`swap`是那些非常基础和原始的函数之一，标准库没有为其失败提供任何处理--例如，`std::swap`算法（第三章，*迭代器对算法*）被声明为`noexcept`，这意味着它*必须*成功；它不允许抛出异常。

因此，如果在我们的`noexcept`交换函数中发生分配失败，我们将在调用栈中看到`bad_alloc`异常逐层上升，直到它达到我们的`noexcept`交换函数声明；此时，C++运行时会停止回滚并调用`std::terminate`，除非程序员通过`std::set_terminate`更改其行为，否则这将导致我们的程序崩溃并终止。

C++17 标准在规范标准容器类型交换过程中应该发生的事情方面比这更进一步。首先，标准不是说明`swap`过程中的分配失败将导致调用`std::terminate`，而是简单地说明`swap`过程中的分配失败将导致*未定义行为*。其次，标准并没有将这种未定义行为限制在分配失败上！根据 C++17 标准，仅仅对任何分配器不平等比较的标准库容器实例调用`swap`将导致未定义行为，无论是否遇到分配失败！

事实上，libc++利用这个优化机会为所有标准容器`swap`函数生成代码，其大致形式如下：

```cpp
    void swap(uniqueish& rhs) noexcept {
      constexpr bool pocs =
        Traits::propagate_on_container_swap::value;
      using std::swap;
      if constexpr (pocs) {
        swap(this->m_allocator, rhs.m_allocator);
      }
      // Don't even check that we know how to free
      // the adopted pointer; just assume that we can.
      swap(this->m_ptr, rhs.m_ptr);
    }
```

注意，如果你使用这个代码（如 libc++所做）来交换具有不等价分配器的两个容器，你最终会在指针和它们的分配器之间出现不匹配，然后你的程序可能会崩溃--或者更糟--在你下次尝试使用不匹配的分配器释放这些指针时。在处理 C++17 的“便利”类型，如`std::pmr::vector`时，记住这个陷阱至关重要！

```cpp
    char buffer[100];
    auto mr = std::pmr::monotonic_buffer_resource(buffer, 100);

    std::pmr::vector<int> a {1,2,3};
    std::pmr::vector<int> b({4,5,6}, &mr);

    std::swap(a, b);
      // UNDEFINED BEHAVIOR

    a.reserve(a.capacity() + 1);
      // this line will undoubtedly crash, as
      // it tries to delete[] a stack pointer
```

如果你的代码设计允许不同内存资源支持的容器之间相互交换，那么你必须避免使用`std::swap`，而应使用这个安全的惯用语：

```cpp
    auto temp = std::move(a); // OK
    a = std::move(b); // OK
    b = std::move(temp); // OK
```

当我说“避免`std::swap`”时，我的意思是“避免 STL 中的任何排列算法”，包括像`std::reverse`和`std::sort`这样的算法。这将是一项相当大的工作，我不建议尝试这样做！

如果你的代码设计允许不同内存资源支持的容器之间可以互换，那么实际上，你可能真的需要重新考虑你的设计。如果你能够修复它，使得你只能交换共享相同内存资源的容器，或者如果你可以完全避免有状态的和/或粘性的分配器，那么你就永远不需要考虑这个特定的陷阱。

# 通过 scoped_allocator_adaptor 向下传播

在前面的章节中，我们介绍了`std::allocator_traits<A>::construct(a, ptr, args...)`，并将其描述为比 placement-`new`语法`::new ((void*)ptr) T(args...)`更可取的替代方案。现在我们将看到为什么某个特定分配器的作者可能希望给它不同的语义。

改变我们自己的分配器类型`construct`的语义的一个可能明显的方法是，对于原始类型，使其以默认方式初始化，而不是零初始化。代码看起来会是这样：

```cpp
    template<class T>
    struct my_allocator : std::allocator<T> 
    {
      my_allocator() = default;

      template<class U>
      my_allocator(const my_allocator<U>&) {}

      template<class... Args>
      void construct(T *p, Args&&... args) {
        if (sizeof...(Args) == 0) {
          ::new ((void*)p) T;
        } else {
          ::new ((void*)p) T(std::forward<Args>(args)...);
        }
      }
    };
```

现在，你可以使用`std::vector<int, my_allocator<int>>`作为一个“类似向量”的类型，满足`std::vector<int>`的所有常用不变性，除了当你通过`v.resize(n)`或`v.emplace_back()`隐式创建新元素时，新元素是未初始化的，就像栈变量一样，而不是被零初始化。

在某种意义上，我们在这里设计的是一个“适配器”，它覆盖在`std::allocator<T>`之上，并以一种有趣的方式修改其行为。如果我们能够以同样的方式修改或“适配”任何任意的分配器那就更好了；为了做到这一点，我们只需将我们的`template<class T>`更改为`template<class A>`，并在旧代码继承自`std::allocator<T>`的地方从`A`继承。当然，我们新的适配器的模板参数列表不再以`T`开头，因此我们不得不自己实现`rebind`；这条路径很快就会进入深层次的元编程，所以我就不展开说明了。

然而，我们还可以用另一种有用的方法来调整我们自己的分配器类型的`construct`方法。考虑以下代码示例，它创建了一个`int`类型的向量向量：

```cpp
    std::vector<std::vector<int>> vv;
    vv.emplace_back();
    vv.emplace_back();
    vv[0].push_back(1);
    vv[1].push_back(2);
    vv[1].push_back(3);
```

假设我们想要将这个容器“粘”到我们自己设计的内存资源上，比如我们最喜欢的`WidgetAlloc`。我们不得不写一些重复性的代码，如下所示：

```cpp
    char buffer[10000];
    std::pmr::monotonic_buffer_resource mr {buffer, sizeof buffer};

    using InnerAlloc = WidgetAlloc<int>;
    using InnerVector = std::vector<int, InnerAlloc>;
    using OuterAlloc = WidgetAlloc<InnerVector>;

    std::vector<InnerVector, OuterAlloc> vv(&mr);
    vv.emplace_back(&mr);
    vv.emplace_back(&mr);
    vv[0].push_back(1);
    vv[1].push_back(2);
    vv[1].push_back(3);
```

注意分配器对象的初始化器`&mr`在两个级别上的重复。需要重复`&mr`使得在泛型上下文中使用我们的向量`vv`变得困难；例如，我们无法轻易将其传递给一个函数模板以填充数据，因为每次被调用者想要`emplace_back`一个新的`int`向量时，它都需要知道只有调用者知道的地址`&mr`。我们想要做的是封装并具体化“每次你构造向量向量的元素时，你都需要将`&mr`附加到参数列表的末尾”的概念。而标准库已经为我们提供了解决方案！

自从 C++11 以来，标准库在名为`<scoped_allocator>`的头文件中提供了一个名为`scoped_allocator_adaptor<A>`的类模板。就像我们的默认初始化“适配器”一样，`scoped_allocator_adaptor<A>`继承自`A`，从而继承了`A`的所有行为；然后它重写了`construct`方法以执行不同的操作。具体来说，它试图弄清楚它当前正在构建的`T`对象是否“使用分配器”，如果是的话，它将把自己作为额外的参数传递给`T`的构造函数。

要决定类型`T`是否“使用分配器”，`scoped_allocator_adaptor<A>::construct`会委托给类型特性`std::uses_allocator_v<T,A>`，除非你已对其进行特化（你很可能不应该这样做），否则当且仅当`A`可以隐式转换为`T::allocator_type`时，它将为真。如果`T`没有`allocator_type`，那么库将假设`T`不关心分配器，除了`pair`和`tuple`的特殊情况（它们都有针对特定于成员的分配器传播的构造函数的重载）以及`promise`的特殊情况（即使它没有提供引用该分配器对象的方法，它也可以使用分配器分配其共享状态；我们说`promise`的分配器支持比我们在第五章，*词汇类型*中看到的类型擦除示例更彻底地“类型擦除”）。 

由于历史原因，分配器感知类型的构造函数可以遵循两种不同的模式，而`scoped_allocator_adaptor`足够智能，可以知道它们两个。较旧且简单的类型（即除了`tuple`和`promise`之外的所有类型）通常具有形式为`T(args..., A)`的构造函数，其中分配器`A`位于末尾。对于`tuple`和`promise`，标准库引入了一种新的模式：`T(std::allocator_arg, A, args...)`，其中分配器`A`位于开头，但前面有一个特殊的标记值`std::allocator_arg`，其唯一目的是指示参数列表中的下一个参数代表一个分配器，类似于标记`std::nullopt`的唯一目的是指示`optional`没有值（参见第五章，*词汇类型*）。就像标准禁止创建类型`std::optional<std::nullopt_t>`一样，如果你尝试创建`std::tuple<std::allocator_arg_t>`，你也会发现自己陷入麻烦。

使用`scoped_allocator_adaptor`，我们可以以前一种稍微不那么繁琐的方式重写我们之前繁琐的例子：

```cpp
    char buffer[10000];
    std::pmr::monotonic_buffer_resource mr {buffer, sizeof buffer};

    using InnerAlloc = WidgetAlloc<int>;
    using InnerVector = std::vector<int, InnerAlloc>;
    using OuterAlloc = std::scoped_allocator_adaptor<WidgetAlloc<InnerVector>>;

    std::vector<InnerVector, OuterAlloc> vv(&mr);
    vv.emplace_back();
    vv.emplace_back();
    vv[0].push_back(1);
    vv[1].push_back(2);
    vv[1].push_back(3);
```

注意到分配器类型变得更加繁琐，但重要的是 `emplace_back` 的 `&mr` 参数已经消失了；我们现在可以在期望能够以自然方式推送元素的环境中使用 `vv`，而无需记住到处添加 `&mr`。在我们的情况下，因为我们使用的是我们的 `WidgetAlloc`，它不是默认可构造的，所以忘记 `&mr` 的症状是一系列编译时错误。但你可能还记得，在本章前面的部分中，`std::pmr::polymorphic_allocator<T>` 会愉快地允许你默认构造它，这可能会产生灾难性的后果；所以如果你计划使用 `polymorphic_allocator`，那么查看 `scoped_allocator_adaptor` 也可能是明智的，只是为了限制你可能忘记指定分配策略的地方数量。

# 传播不同的分配器

在我介绍 `scoped_allocator_adaptor<A>` 时，遗漏了一个更复杂的点。模板参数列表不仅限于只有一个分配器类型参数！实际上，你可以创建一个具有多个分配器类型参数的 scoped-allocator 类型，如下所示：

```cpp
    using InnerAlloc = WidgetAlloc<int>;
    using InnerVector = std::vector<int, InnerAlloc>;

    using MiddleAlloc = std::scoped_allocator_adaptor<
      WidgetAlloc<InnerVector>,
      WidgetAlloc<int>
    >;
    using MiddleVector = std::vector<InnerVector, MiddleAlloc>;

    using OuterAlloc = std::scoped_allocator_adaptor<
      WidgetAlloc<MiddleVector>,
      WidgetAlloc<InnerVector>,
      WidgetAlloc<int>
    >;
    using OuterVector = std::vector<MiddleVector, OuterAlloc>;
```

在设置这些 `typedef` 之后，我们继续设置三个不同的内存资源，并构造一个能够记住所有三个内存资源的 `scoped_allocator_adaptor` 实例（因为它包含三个不同的 `WidgetAlloc` 实例，每个“级别”一个）：

```cpp
    char bi[1000];
    std::pmr::monotonic_buffer_resource mri {bi, sizeof bi};
    char bm[1000];
    std::pmr::monotonic_buffer_resource mrm {bm, sizeof bm};
    char bo[1000];
    std::pmr::monotonic_buffer_resource mro {bo, sizeof bo};

    OuterAlloc saa(&mro, &mrm, &mri);
```

最后，我们可以构造一个 `OuterVector` 的实例，传入我们的 `scoped_allocator_adaptor` 参数；这就全部完成了！我们精心设计的分配器类型中隐藏的 `construct` 方法会负责将 `&bm` 或 `&bi` 参数传递给需要其中一个的任何构造函数：

```cpp
    OuterVector vvv(saa);

    vvv.emplace_back();
      // This allocation comes from buffer "bo".

    vvv[0].emplace_back();
      // This allocation comes from buffer "bm".

    vvv[0][0].emplace_back(42);
      // This allocation comes from buffer "bi".
```

如你所见，一个深度嵌套的 `scoped_allocator_adaptor` 并不是为胆小的人准备的；而且它们只有在沿途创建了很多“辅助” `typedef` 的情况下才能使用，就像我们在本例中所做的那样。

关于 `std::scoped_allocator_adaptor<A...>` 的最后一点说明：如果容器的嵌套深度超过了模板参数列表中分配器类型的数量，那么 `scoped_allocator_adaptor` 将会像其参数列表中的最后一个分配器类型无限重复一样行事。例如：

```cpp
    using InnerAlloc = WidgetAlloc<int>;
    using InnerVector = std::vector<int, InnerAlloc>;

    using MiddleAlloc = std::scoped_allocator_adaptor<
      WidgetAlloc<InnerVector>
    >;
    using MiddleVector = std::vector<InnerVector, MiddleAlloc>;

    using TooShortAlloc = std::scoped_allocator_adaptor<
      WidgetAlloc<MiddleVector>,
      WidgetAlloc<InnerVector>
    >;
    using OuterVector = std::vector<MiddleVector, TooShortAlloc>;

    TooShortAlloc tsa(&mro, WidgetAlloc<InnerVector>(&mri));
    OuterVector tsv(tsa);

    tsv.emplace_back();
      // This allocation comes from buffer "bo".

    tsv[0].emplace_back();
      // This allocation comes from buffer "bi".

    tsv[0][0].emplace_back(42);
      // This allocation AGAIN comes from buffer "bi"!
```

实际上，我们在第一个 `scoped_allocator_adaptor` 示例中就依赖了这种行为，即涉及 `vv` 的那个示例，尽管当时我没有提到。现在你知道了这一点，你可能想回去研究那个示例，看看“无限重复”的行为在哪里被使用，如果你想要为 `int` 的内部数组使用不同于外部 `InnerVector` 数组的内存资源，你应该如何修改那段代码。

# 摘要

分配器是 C++ 中一个基本深奥的话题，主要由于历史原因。几个不同的接口，具有不同的晦涩用途，层层叠叠；所有这些都涉及强烈的元编程；并且许多这些特性（即使是相对较旧的 C++11 特性，如花哨的指针）的供应商支持仍然不足。

C++17 提供了标准库类型 `std::pmr::memory_resource` 来阐明现有 *内存资源*（即 *堆*）和 `allocators`（即 *堆的句柄*）之间的区别。内存资源提供 `allocate` 和 `deallocate` 方法；分配器提供这些方法以及 `construct` 和 `destroy` 方法。

如果你实现了自己的分配器类型 `A`，它必须是一个模板；它的第一个模板参数应该是它期望`分配`的类型 `T`。你的分配器类型 `A` 还必须有一个模板构造函数来支持从 `A<U>` 到 `A<T>` 的“重新绑定”。就像任何其他类型的指针一样，分配器类型必须支持 `==` 和 `!=` 操作符。

堆的 `deallocate` 方法允许要求附加到传入指针的额外元数据。C++ 通过 *花哨的指针* 来处理这一点。C++17 的 `std::pmr::memory_resource` 不支持花哨的指针，但实现自己的并不困难。

花哨指针类型必须满足所有随机访问迭代器的需求，并且必须是可空的，并且必须可转换为普通原始指针。如果你想使用你的花哨指针类型与基于节点的容器，如 `std::list`，你必须给它一个静态的 `pointer_to` 成员函数。

C++17 区分了“粘性”和“非粘性”分配器类型。无状态分配器类型，如 `std::allocator<T>`，是非粘性的；有状态分配器类型，如 `std::pmr::polymorphic_allocator<T>`，默认是粘性的。创建一个非默认粘性的自定义分配器类型需要设置三个成员类型别名，通常称为“POCCA”、“POCMA”和“POCS”。粘性分配器类型，如 `std::pmr::polymorphic_allocator<T>`，主要用于——或许仅用于——经典面向对象的情况，其中容器对象被固定在特定的内存地址上。面向值的编程（涉及大量移动和交换操作）需要无状态分配器类型，或者程序中的每个人都要使用相同的堆和单个粘性但*实际上无状态*的分配器类型。

`scoped_allocator_adaptor<A...>` 可以帮助简化使用自定义分配器或内存资源的深层嵌套容器的使用。几乎任何使用非默认分配器类型的深层嵌套容器都需要大量的辅助类型别名来保持可读性。

交换两个具有不同粘性分配器的容器：在理论上这会引发未定义的行为，在实践中会破坏内存并导致段错误。不要这样做！
