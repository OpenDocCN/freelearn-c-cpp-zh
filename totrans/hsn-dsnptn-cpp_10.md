# 10

# 本地缓冲区优化

并非所有设计模式都关注于设计类层次结构。对于常见问题，软件设计模式是最通用和可重用的解决方案，而对于使用 C++ 的程序员来说，最常见的问题之一是性能不足。这种糟糕性能的最常见原因是不高效的内存管理。模式是为了解决这些问题而开发的。在本章中，我们将探讨一种特别针对小型、频繁内存分配开销的模式。

本章将涵盖以下主题：

+   小型内存分配的开销是什么，如何进行测量？

+   本地缓冲区优化是什么，它如何提高性能，以及如何测量这些改进？

+   在什么情况下可以有效地使用本地缓冲区优化模式？

+   使用本地缓冲区优化模式可能有哪些潜在缺点和限制？

# 技术要求

你需要安装和配置 Google Benchmark 库，详细信息可以在以下链接找到：[`github.com/google/benchmark`](https://github.com/google/benchmark)（参见 *第四章*，*交换 – 从简单到微妙*，有关安装说明）。

示例代码可以在以下链接找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/main/Chapter10`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/main/Chapter10)。

# 小型内存分配的开销

本地缓冲区优化仅仅是优化。它是一个面向性能的模式，因此我们必须牢记性能的第一规则——永远不要对性能做出任何猜测。性能，以及任何优化的影响，都必须进行测量。

## 内存分配的成本

由于我们正在探索内存分配的开销及其降低方法，我们必须回答的第一个问题是内存分配有多昂贵。毕竟，没有人想优化一个如此快速以至于不需要优化的东西。我们可以使用 Google Benchmark（或任何其他微基准测试，如果你更喜欢）来回答这个问题。测量内存分配成本的最简单基准可能看起来像这样：

```cpp
void BM_malloc(benchmark::State& state) {
  for (auto _ : state) {
    void* p = malloc(64);
    benchmark::DoNotOptimize(p);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_malloc_free);
```

`benchmark::DoNotOptimize` 包装器阻止编译器优化掉未使用的变量。唉，这个实验可能不会有一个好结果；微基准测试库需要多次运行测试，通常是数百万次，以积累足够准确的平均运行时间。在基准测试完成之前，机器很可能耗尽内存。修复方法是足够的简单，我们必须释放我们分配的内存：

```cpp
// Example 01
void BM_malloc_free(benchmark::State& state) {
  const size_t S = state.range(0);
  for (auto _ : state) {
    void* p = malloc(S);
    benchmark::DoNotOptimize(p); free(p);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_malloc_free)->Arg(64);
```

我们必须注意，我们现在测量的是分配和释放的成本，这反映在函数名称的改变上。这种改变并不不合理；任何分配的内存最终都需要释放，因此成本必须在某个时候支付。我们还改变了基准测试，使其由分配大小参数化。如果你运行这个基准测试，你应该得到类似以下的结果：

```cpp
Benchmark                 Time   Items per second
BM_malloc_free/64        19.2 ns 52.2041M/s
```

这告诉我们，在这个特定的机器上，分配和释放`64`字节内存的成本大约是`19`纳秒，这意味着每秒可以完成 5200 万次分配/释放。如果你对`64`字节的大小是否以某种方式特别感兴趣，你可以改变基准中参数的尺寸值，或者为一系列尺寸运行基准测试：

```cpp
void BM_malloc_free(benchmark::State& state) {
  const size_t S = state.range(0);
  for (auto _ : state) {
    void* p = malloc(S);
    benchmark::DoNotOptimize(p); free(p);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_malloc_free)->
  RangeMultiplier(2)->Range(32,   256);
```

你可能还会注意到，到目前为止，我们只测量了程序中第一次内存分配所需的时间，因为我们还没有分配其他任何东西。C++运行时系统可能在程序启动时进行了一些动态分配，但这仍然不是一个非常现实的基准。我们可以通过重新分配一些内存来使测量更加相关：

```cpp
// Example 02
#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)
void BM_malloc_free(benchmark::State& state) {
  const size_t S = state.range(0);
  const size_t N = state.range(1);
  std::vector<void*> v(N);
  for (size_t i = 0; i < N; ++i) v[i] = malloc(S);
  for (auto _ : state) {
    REPEAT({
      void* p = malloc(S);
      benchmark::DoNotOptimize(p);
      free(p);
    });
  }
  state.SetItemsProcessed(32*state.iterations());
  for (size_t i = 0; i < N; ++i) free(v[i]);
}
BENCHMARK(BM_malloc_free)->
  RangeMultiplier(2)->Ranges({{32, 256}, {1<<15, 1<<15}});
```

在这里，我们在开始基准测试之前调用`malloc` `N`次。通过在重新分配期间改变分配大小，我们可以实现进一步的改进。我们还使用 C 预处理器宏将基准循环的主体复制了`32`次，以减少循环本身在测量中的开销。基准测试报告的时间现在是进行`32`次分配和释放所需的时间，这不太方便，但分配率仍然是有效的，因为我们已经考虑了循环展开，并在设置处理项目数量时将迭代次数乘以`32`（在 Google Benchmark 中，项目是你想要它成为的任何东西，每秒的项目数量在基准测试结束时报告，因此我们声明一次分配/释放为一个项目）。

即使经过所有这些修改和改进，最终结果也将非常接近我们最初的每秒`54`百万次分配的测量值。这似乎非常快，只有`18`纳秒。然而，请记住，现代 CPU 可以在这么短的时间内执行数十条指令。由于我们处理的是小分配，因此每个分配的内存片段的处理时间也很小，分配的开销也不可忽视。这当然是对性能的猜测，也是我之前警告过你的，因此我们将通过直接实验来验证这一说法。

然而，首先我想向你展示另一个原因，为什么小内存分配特别低效。到目前为止，我们只探索了在单个线程上内存分配的成本。如今，大多数有性能要求的程序都是并发的，C++支持并发和多线程。让我们看看当我们在多个线程上同时进行内存分配时，成本是如何变化的：

```cpp
// Example 03
void BM_malloc_free(benchmark::State& state) {
  const size_t S = state.range(0);
  const size_t N = state.range(1);
  std::vector<void*> v(N);
  for (size_t i = 0; i < N; ++i) v[i] = malloc(S);
  for (auto _ : state) {
    REPEAT({
      void* p = malloc(S);
      benchmark::DoNotOptimize(p);
      free(p);
    });
  }
  state.SetItemsProcessed(32*state.iterations());
  for (size_t i = 0; i < N; ++i) free(v[i]);
}
BENCHMARK(BM_malloc_free)->
  RangeMultiplier(2)->Ranges({{32, 256}, {1<<15, 1<<15}})
  ->ThreadRange(1, 2);
```

结果很大程度上取决于硬件和系统使用的`malloc`版本。此外，在拥有许多 CPU 的大机器上，你可以有超过两个线程。

尽管如此，整体趋势应该看起来像这样：

```cpp
Benchmark                          Time   Items per second
BM_malloc_free/32/32768/threads:1  778 ns 41.1468M/s
BM_malloc_free/32/32768/threads:2  657 ns 24.3749M/s
BM_malloc_free/32/32768/threads:4  328 ns 24.3854M/s
BM_malloc_free/32/32768/threads:8  242 ns 16.5146M/s
```

这相当令人沮丧；当我们从单个线程增加到两个线程（在更大的机器上，类似的增加将会发生，但可能涉及超过两个线程）时，分配的成本增加了几倍。系统内存分配器似乎成为了有效并发的祸害。有更好的分配器可以用来替换默认的`malloc()`分配器，但它们也有自己的缺点。此外，如果我们的 C++程序不依赖于特定、非标准的系统库替换以获得性能，那就更好了。我们需要一种更好的内存分配方式。让我们来看看。

# 引入局部缓冲区优化

程序完成特定任务所需做的最少工作就是什么都不做。免费的东西很棒。同样，分配和释放内存最快的方式就是——不做。局部缓冲区优化是一种不劳而获的方式；在这种情况下，不增加额外的计算成本就能获得一些内存。

## 主要思想

要理解局部缓冲区优化，你必须记住内存分配并不是孤立发生的。通常情况下，如果需要少量内存，分配的内存会被用作某些数据结构的一部分。例如，让我们考虑一个非常简单的字符串：

```cpp
// Example 04
class simple_string {
  public:
  simple_string() = default;
  explicit simple_string(const char* s) : s_(strdup(s)) {}
  simple_string(const simple_string& s)
    : s_(strdup(s.s_)) {}
  simple_string& operator=(const char* s) {
    free(s_);
    s_ = strdup(s);
    return *this;
  }
  simple_string& operator=(const simple_string& s) {
    if (this == &s) return *this;
    free(s_);
    s_ = strdup(s.s_);
    return *this;
  }
  bool operator==(const simple_string& rhs) const {
    return strcmp(s_, rhs.s_) == 0;
  }
  ~simple_string() { free(s_); }
  private:
  char* s_ = nullptr;
};
```

字符串通过`strdup()`调用从`malloc()`分配内存，并通过调用`free()`返回它。为了在任何程度上有用，字符串需要更多的成员函数，但就现在而言，这些已经足够探索内存分配的开销了。说到分配，每次字符串被构造、复制或赋值时，都会发生分配。更准确地说，每次字符串被构造时，都会发生额外的分配；字符串对象本身必须被分配到某个地方，这可能是在栈上的局部变量，或者如果字符串是某些动态分配的数据结构的一部分，则是在堆上。此外，还会为字符串数据发生分配，内存总是从`malloc()`中获取。

这就是局部缓冲区优化的想法——我们为什么不将字符串对象做得更大，以便它可以包含自己的数据呢？这实际上真的是不劳而获；字符串对象的内存无论如何都需要分配，但我们可以得到额外的字符串数据内存，而无需额外成本。当然，字符串可以任意长，所以我们事先不知道需要将字符串对象做得多大才能存储程序可能遇到的任何字符串。即使我们知道，总是分配那么大的对象，即使是对于非常短的字符串，也会造成巨大的内存浪费。

然而，我们可以观察到——字符串越长，处理它所需的时间就越长（复制、搜索、转换或我们需要对其进行的任何操作）。

对于非常长的字符串，分配的成本与处理成本相比将非常小。另一方面，对于短字符串，分配的成本可能会很大。因此，通过在对象本身存储短字符串，而将任何太长无法放入对象的字符串存储在分配的内存中，我们可以获得最大的性能提升。简而言之，这就是局部缓冲区优化，对于字符串来说也被称为**短字符串优化**；对象（字符串）包含一个特定大小的本地缓冲区，任何适合该缓冲区的字符串都直接存储在对象内部：

```cpp
// Example 04
class small_string {
  public:
  small_string() = default;
  explicit small_string(const char* s) :
    s_((strlen(s) + 1 < sizeof(buf_)) ? strcpy(buf_, s)
                                      : strdup(s)) {}
  small_string(const small_string& s) :
    s_((s.s_ == s.buf_) ? strcpy(buf_, s.buf_)
                        : strdup(s.s_)) {}
  small_string& operator=(const char* s) {
    if (s_ != buf_) free(s_);
    s_ = (strlen(s) + 1 < sizeof(buf_)) ? strcpy(buf_, s)
                                        : strdup(s);
    return *this;
  }
  small_string& operator=(const small_string& s) {
    if (this == &s) return *this;
    if (s_ != buf_) free(s_);
    s_ = (s.s_ == s.buf_) ? strcpy(buf_, s.buf_)
                          : strdup(s.s_);
    return *this;
  }
  bool operator==(const small_string& rhs) const {
    return strcmp(s_, rhs.s_) == 0;
  }
  ~small_string() {
    if (s_ != buf_) free(s_);
  }
  private:
  char* s_ = nullptr;
  char buf_[16];
};
```

在前面的代码示例中，缓冲区大小被静态设置为`16`个字符，包括用于终止字符串的空字符。任何长度超过`16`的字符串都将从`malloc()`分配。在分配或销毁字符串对象时，我们必须检查是否进行了分配或使用了内部缓冲区，以便适当地释放字符串使用的内存。

## 局部缓冲区优化的效果

`small_string`与`simple_string`相比快多少？这当然取决于你需要用它做什么。让我们从仅仅创建和删除字符串开始。为了避免两次输入相同的基准代码，我们可以使用模板基准，如下所示：

```cpp
// Example 04
template <typename T>
void BM_string_create_short(benchmark::State& state) {
  const char* s = "Simple string";
  for (auto _ : state) {
    REPEAT({
      T S(s);
      benchmark::DoNotOptimize(S);
    })
  }
  state.SetItemsProcessed(32*state.iterations());
}
BENCHMARK_TEMPLATE1(BM_string_create_short, simple_string);
BENCHMARK_TEMPLATE1(BM_string_create_short, small_string);
```

结果相当令人印象深刻：

```cpp
Benchmark                                Time Items per sec
BM_string_create_short<simple_string>     835 ns 38.34M/s
BM_string_create_short<small_string>     18.7 ns 1.71658G/s
```

当我们在多个线程上尝试相同的测试时，情况甚至更好：

```cpp
Benchmark                                Time Items per sec
BM_create<simple_string>/threads:2        743 ns 21.5644M/s
BM_create<simple_string>/threads:4        435 ns 18.4288M/s
BM_create<small_string>/threads:2        9.34 ns 1.71508G/s
BM_create<small_string>/threads:4        4.77 ns 1.67998G/s
```

在两个线程上，常规字符串创建稍微快一些，但创建短字符串几乎正好快两倍（在四个线程上再次快两倍）。当然，这几乎是短字符串优化的最佳情况——首先是因为我们做的只是创建和删除字符串，这正是我们优化的部分，其次是因为字符串是局部变量，其内存作为栈帧的一部分分配，因此没有额外的分配成本。

然而，这并不是一个不合理的情况；毕竟，局部变量并不罕见，如果字符串是某个大型数据结构的一部分，那么该结构的分配成本无论如何都必须支付，因此同时分配其他任何东西而无需额外成本实际上是免费的。

尽管如此，我们不太可能只分配字符串然后立即释放它们，因此我们应该考虑其他操作的成本。只要它们保持较短，我们可以期望复制或分配字符串会有类似的改进：

```cpp
template <typename T>
void BM_string_copy_short(benchmark::State& state) {
  const T s("Simple string");
  for (auto _ : state) {
    REPEAT({
      T S(s);
      benchmark::DoNotOptimize(S);
    })
  }
  state.SetItemsProcessed(32*state.iterations());
}
template <typename T>
void BM_string_assign_short(benchmark::State& state) {
  const T s("Simple string");
  T S;
  for (auto _ : state) {
    REPEAT({ benchmark::DoNotOptimize(S = s); })
  }
  state.SetItemsProcessed(32*state.iterations());
}
BENCHMARK_TEMPLATE1(BM_string_copy_short, simple_string);
BENCHMARK_TEMPLATE1(BM_string_copy_short, small_string);
BENCHMARK_TEMPLATE1(BM_string_assign_short, simple_string);
BENCHMARK_TEMPLATE1(BM_string_assign_short, small_string);
```

事实上，也观察到了类似的戏剧性的性能提升：

```cpp
Benchmark                                Time Items per sec
BM_string_copy_short<simple_string>       786 ns 40.725M/s
BM_string_copy_short<small_string>       53.5 ns 598.847M/s
BM_string_assign_short<simple_string>     770 ns 41.5977M/s
BM_string_assign_short<small_string>     46.9 ns 683.182M/s
```

我们还可能需要至少读取字符串中的数据一次，以比较它们或搜索特定的字符串或字符，或者计算一些派生值。当然，我们并不期望这些操作会有类似的规模改进，因为它们都不涉及任何分配或释放。那么，为什么我们仍然期望有任何改进呢？

事实上，一个简单的字符串比较测试，例如，显示两个字符串版本之间没有差异。为了看到任何好处，我们必须创建许多字符串对象并将它们进行比较：

```cpp
template <typename T>
void BM_string_compare_short(benchmark::State& state) {
  const size_t N = state.range(0);
  const T s("Simple string");
  std::vector<T> v1, v2;
  ... populate the vectors with strings ...
  for (auto _ : state) {
    for (size_t i = 0; i < N; ++i) {
      benchmark::DoNotOptimize(v1[i] == v2[i]);
    }
  }
  state.SetItemsProcessed(N*state.iterations());
}
BENCHMARK_TEMPLATE1(BM_string_compare_short,
                    simple_string)->Arg(1<<22);
BENCHMARK_TEMPLATE1(BM_string_compare_short,
                    small_string)->Arg(1<<22);
```

对于`N`值较小的情况（字符串的总数较少），优化不会带来任何显著的好处。但是，当我们必须处理许多字符串时，使用小字符串优化比较字符串可以大约快两倍：

```cpp
Benchmark                                Time Items per sec
BM_compare<simple_string>/4194304    30230749 ns 138.855M/s
BM_compare<small_string>/4194304     15062582 ns 278.684M/s
```

如果没有任何分配，为什么会发生这种情况？这个实验显示了局部缓冲区优化的第二个、非常重要的好处——提高了缓存局部性。在读取字符串数据之前，必须访问字符串对象本身；它包含数据的指针。对于常规字符串，访问字符串字符涉及两次不同的、通常无关的地址的内存访问。如果数据总量很大，那么第二次访问，即访问字符串数据，很可能会错过缓存并等待数据从主内存中传输过来。另一方面，优化的字符串将数据保持得靠近字符串对象，因此一旦字符串本身在缓存中，数据也在缓存中。我们需要足够多的不同字符串来看到这种好处的原因是，当字符串很少时，所有字符串对象及其数据可以永久地驻留在缓存中。只有当字符串的总大小超过缓存大小时，性能好处才会显现出来。现在，让我们更深入地探讨一些额外的优化。

## 额外的优化

我们实现的`small_string`类存在一个明显的低效之处——当字符串存储在本地缓冲区时，我们实际上并不需要数据的指针。我们确切地知道数据在哪里，就在本地缓冲区中。我们确实需要以某种方式知道数据是存储在本地缓冲区还是外部分配的内存中，但仅仅为了存储这一点，我们并不需要使用 8 个字节（在 64 位机器上）。当然，我们仍然需要指针来存储较长的字符串，但当我们处理短字符串时，我们可以重复使用那段内存：

```cpp
// Example 05
class small_string {
  ...
  private:
  union {
    char* s_;
    struct {
      char buf[15];
      char tag;
    } b_;
  };
};
```

在这里，我们使用最后一个字节作为`tag`来指示字符串是存储在本地（`tag == 0`）还是单独的分配中（`tag == 1`）。请注意，总缓冲区大小仍然是`16`个字符，`15`个用于字符串本身，`1`个用于 tag，如果字符串需要所有`16`个字节，这个 tag 也会变成尾随的零（这就是为什么我们必须使用`tag == 0`来指示本地存储，否则会多浪费一个字节）。指针覆盖在字符缓冲区的第一个`8`个字节上。在这个例子中，我们选择优化字符串占用的总内存；这个字符串仍然有 16 个字符的本地缓冲区，就像之前的版本一样，但对象本身现在只有 16 个字节，而不是 24 个。如果我们愿意保持对象大小不变，我们可以使用更大的缓冲区并本地存储更长的字符串。一般来说，随着字符串变长，小字符串优化的好处会逐渐减少。从本地到远程字符串的最佳转换点取决于特定的应用程序，并且当然必须通过基准测试来确定。

# 超越字符串的本地缓冲区优化

本地缓冲区优化可以有效地用于比短字符串更多的情况。实际上，任何需要运行时确定大小的动态小分配时，都应该考虑这种优化。在本节中，我们将考虑几个这样的数据结构。

## 小向量

另一个非常常见的、经常从本地缓冲区优化中受益的数据结构是向量。向量本质上是由指定类型的数据元素组成的动态连续数组（在这个意义上，字符串是字节的向量，尽管空终止符赋予了字符串其特定的特性）。一个基本的向量，如 C++标准库中找到的`std::vector`，需要两个数据成员，一个数据指针和一个数据大小：

```cpp
// Example 06
class simple_vector {
  public:
  simple_vector() = default;
  simple_vector(std::initializer_list<int> il) :
    n_(il.size()),
    p_(static_cast<int*>(malloc(sizeof(int)*n_)))
  {
    int* p = p_;
    for (auto x : il) *p++ = x;
  }
  ~simple_vector() { free(p_); }
  size_t size() const { return n_; }
  private:
  size_t n_ = 0;
  int* p_ = nullptr;
};
```

向量通常是模板，就像标准`std::vector`一样，但我们将这个例子简化了，以展示一个整数向量（将这个向量类转换为模板留给你作为练习，并且不会以任何方式改变本地缓冲区优化模式的应用）。只要足够小，我们就可以应用*小向量优化*并将向量数据存储在向量对象的主体中：

```cpp
// Example 06
class small_vector {
  public:
  small_vector() = default;
  small_vector(std::initializer_list<int> il) :
    n_(il.size()), p_((n_ < sizeof(buf_)/sizeof(buf_[0]))
      ? buf_ : static_cast<int*>(malloc(sizeof(int)*n_)))
  {
    int* p = p_;
    for (auto x : il) *p++ = x;
  }
  ~small_vector() {
    if (p_ != buf_) free(p_);
  }
  private:
  size_t n_ = nullptr;
  int* p_ = nullptr;
  int buf_[16];
};
```

我们可以用类似字符串的方法进一步优化向量，并将局部缓冲区与指针叠加。我们不能像之前那样使用最后一个字节作为 `tag`，因为向量的任何元素都可以有任意值，而零值在一般情况下并不特殊。然而，我们无论如何都需要存储向量的大小，因此我们可以随时用它来确定是否使用了局部缓冲区。我们可以进一步利用这样一个事实，即如果使用局部缓冲区优化，向量的大小不可能非常大，所以我们不需要 `size_t` 类型的字段来存储它：

```cpp
// Example 07
class small_vector {
  public:
  small_vector() = default;
  small_vector(std::initializer_list<int> il) {
    int* p;
    if (il.size() < sizeof(short_.buf)/
                    sizeof(short_.buf[0])) {
      short_.n = il.size();
      p = short_.buf;
    } else {
      short_.n = UCHAR_MAX;
      long_.n = il.size();
      p = long_.p = static_cast<int*>(
        malloc(sizeof(int)*long_.n));
    }
    for (auto x : il) *p++ = x;
  }
  ~small_vector() {
    if (short_.n == UCHAR_MAX) free(long_.p);
  }
  private:
  union {
    struct {
      int buf[15];
      unsigned char n;
    } short_ = { {}, '\0' };
    struct {
      size_t n;
      int* p;
    } long_;
  };
};
```

在这里，我们根据是否使用局部缓冲区来存储向量大小，要么在 `size_t` `long_.n` 中，要么在 `unsigned` `char` `short_.n` 中。远程缓冲区通过在短大小中存储 `UCHAR_MAX`（即 255）来表示。由于这个值大于局部缓冲区的大小，这个 `tag` 是明确的（如果局部缓冲区增加到可以存储超过 255 个元素，那么 `short_.n` 的类型就需要更改为更长的整数）。

我们可以使用与字符串相同的基准测试来衡量小向量优化的性能提升。根据向量的实际大小，在创建和复制向量时可以期望大约 10 倍的性能提升，如果基准测试在多线程上运行，则提升更多。当然，当它们存储少量动态分配的数据时，其他数据结构也可以以类似的方式优化。这些数据结构的优化在本质上相似，但有一个值得注意的变体我们应该强调。

## 小队列

我们刚才看到的小向量使用局部缓冲区来存储向量元素的小数组。这是在元素数量经常很少时优化存储可变数量元素的数据结构的标准方式。对于基于队列的数据结构，这种优化有一个特定的版本，其中缓冲区在一端增长，在另一端消耗。如果队列在任何时候只有少数几个元素，则可以使用局部缓冲区来优化队列。这里常用的技术是 `buffer[N]`，因此，当元素被添加到队列的末尾时，我们将达到数组的末尾。到那时，一些元素已经被从队列中取出，所以数组的前几个元素不再使用。当我们到达数组的末尾时，下一个入队的值将进入数组的第一个元素，`buffer[0]`。数组被当作环形处理，在 `buffer[N-1]` 元素之后是 `buffer[0]` 元素（因此这种技术的另一个名字是 *环形缓冲区*）。

环形缓冲区技术通常用于队列和其他数据结构，在这些数据结构中，数据被多次添加和移除，而在任何给定时间存储的数据总量是有限的。下面是环形缓冲区队列的一种可能的实现：

```cpp
// Example 08
class small_queue {
  public:
  bool push(int i) {
    if (front_ - tail_ > buf_size_) return false;
    buf_[(++front_) & (buf_size_ - 1)] = i;
    return true;
  }
  int front() const {
    return buf_[tail_ & (buf_size_ - 1)];
  }
  void pop() { ++tail_; }
  size_t size() const { return front_ - tail_; }
  bool empty() const { return front_ == tail_; }
  private:
  static constexpr size_t buf_size_ = 16;
  static_assert((buf_size_ & (buf_size_ - 1)) == 0,
                "Buffer size must be a power of 2");
  int buf_[buf_size_];
  size_t front_ = 0;
  size_t tail_ = 0;
};
```

在这个例子中，我们只支持局部缓冲区；如果队列必须保留的元素数量超过了缓冲区的大小，`push()`调用将返回`false`。我们本可以切换到堆分配的数组，就像我们在`Example 07`中为`small_vector`所做的那样。

在这个实现中，我们无限制地增加`front_`和`tail_`索引，但当这些值用作局部缓冲区的索引时，我们取索引值对缓冲区大小的模。值得注意的是，当处理循环缓冲区时，这种优化非常常见：缓冲区的大小是 2 的幂（由 assert 强制）。这允许我们用更快的位运算来替换一般的（并且较慢的）模运算，例如`front_ % buf_size_`。我们也不必担心整数溢出：即使我们调用`push()`和`pop()`超过`2⁶⁴`次，无符号整数索引值将溢出并回到零，队列仍然可以正常工作。

如预期的那样，具有局部缓冲区优化的队列远远优于一般的队列，例如`std::queue<int>`（当然，只要优化仍然有效且队列中的元素数量较少，当然是这样）：

```cpp
Benchmark                         Time   items_per_second
BM_queue<std::queue<int>>       472 ns          67.787M/s
BM_queue<small_queue>           100 ns         319.857M/s
```

循环局部缓冲区可以非常有效地用于许多需要处理大量数据但一次只保留少量元素的情况。可能的应用包括网络和 I/O 缓冲区、在并发程序中交换线程间数据的管道等。

让我们现在看看局部缓冲区优化在常见数据结构之外的用途。

## 类型擦除和可调用对象

另有一种非常不同的应用类型，其中可以使用局部缓冲优化来非常有效地存储可调用对象，这些对象可以作为函数调用。许多模板类提供了一种使用可调用对象自定义其行为一部分的选项。例如，`std::shared_ptr`是 C++中的标准共享指针，允许用户指定一个自定义的`deleter`。这个`deleter`将使用要删除的对象的地址被调用，因此它是一个具有`void*`类型一个参数的可调用对象。它可以是一个函数指针、成员函数指针或函数对象（定义了`operator()`的对象）——任何可以在`p`指针上调用的类型；也就是说，任何可以在`callable(p)`函数调用语法中编译的类型都可以使用。然而，`deleter`不仅仅是一个类型；它是一个对象，并在运行时指定，因此需要存储在一个共享指针可以访问到它的位置。如果`deleter`是共享指针类型的一部分，我们可以在共享指针对象中声明该类型的数据成员（或者在 C++共享指针的情况下，在其引用对象中声明，该引用对象在所有共享指针副本之间共享）。你可以将其视为局部缓冲优化的简单应用，如下面的智能指针，当指针超出作用域时自动删除对象（就像`std::unique_ptr`一样）：

```cpp
// Example 09
template <typename T, typename Deleter> class smartptr {
  public:
  smartptr(T* p, Deleter d) : p_(p), d_(d) {}
  ~smartptr() { d_(p_); }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; } private:
  T* p_;
  Deleter d_;
};
```

然而，我们追求的是更有趣的事情，当我们处理类型擦除对象时，我们可以找到这样一件事。这类对象的细节在专门讨论类型擦除的章节中已经讨论过，但简而言之，它们是可调用不是类型本身的一部分（也就是说，它被从包含对象的类型中*擦除*）。可调用存储在一个多态对象中，并通过一个虚函数在运行时调用正确类型的对象。多态对象反过来又通过基类指针进行操作。

现在，我们面临一个问题，从某种意义上讲，与前面的小向量类似——我们需要存储一些数据，在我们的例子中是可调用对象，其类型和因此的大小不是静态已知的。一般的解决方案是动态分配这样的对象，并通过基类指针访问它们。在智能指针`deleter`的情况下，我们可以这样做：

```cpp
// Example 09
template <typename T> class smartptr_te {
  struct deleter_base {
    virtual void apply(void*) = 0;
    virtual ~deleter_base() {}
  };
  template <typename Deleter>
  struct deleter : public deleter_base {
    deleter(Deleter d) : d_(d) {}
    void apply(void* p) override {
      d_(static_cast<T*>(p));
    }
    Deleter d_;
  };
  public:
  template <typename Deleter>
  smartptr_te(T* p, Deleter d) : p_(p),
    d_(new deleter<Deleter>(d)) {}
  ~smartptr_te() {
    d_->apply(p_);
    delete d_;
  }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
  private:
  T* p_;
  deleter_base* d_;
};
```

注意，`Deleter`类型不再是智能指针类型的一部分；它已经被*擦除*。对于相同的`T`对象类型，所有智能指针都有相同的类型，`smartptr_te<T>`（在这里，`te`代表*类型擦除*）。然而，我们必须为此语法便利付出高昂的代价——每次创建智能指针时，都会进行额外的内存分配。高昂到什么程度？我们必须再次记住性能的第一规则——*高昂*只是一个猜测，直到通过实验得到证实，如下面的基准测试：

```cpp
// Example 09
struct deleter {    // Very simple deleter for operator new
  template <typename T> void operator()(T* p) { delete p; }
};
void BM_smartptr(benchmark::State& state) {
  deleter d;
  for (auto _ : state) {
    smartptr<int, deleter> p(new int, d);
  }
  state.SetItemsProcessed(state.iterations());
}
void BM_smartptr_te(benchmark::State& state) {
  deleter d;
  for (auto _ : state) {
    smartptr_te<int> p(new int, d);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_smartptr);
BENCHMARK(BM_smartptr_te);
BENCHMARK_MAIN();
```

对于具有静态定义的删除器的智能指针，我们可以预期每次迭代的成本与之前测量的`malloc()`和`free()`的成本非常相似：

```cpp
Benchmark                  Time Items per second
BM_smartptr             21.0 ns 47.5732M/s
BM_smartptr_te          44.2 ns 22.6608M/s
```

对于类型擦除智能指针，有两个分配而不是一个，因此创建指针对象所需的时间加倍。顺便说一下，我们还可以测量原始指针的性能，它应该与智能指针在测量精度内相同（这实际上是一个针对`std::unique_ptr`标准的明确设计目标）。

我们可以在这里应用相同的局部缓冲区优化理念，并且它可能比字符串中的效果还要好；毕竟，大多数可调用对象都很小。然而，我们并不能完全依赖这一点，必须处理大于局部缓冲区的可调用对象的情况：

```cpp
// Example 09
template <typename T> class smartptr_te_lb {
  struct deleter_base {
    virtual void apply(void*) = 0;
    virtual ~deleter_base() {}
  };
  template <typename Deleter>
    struct deleter : public deleter_base {
    deleter(Deleter d) : d_(d) {}
    void apply(void* p) override {
      d_(static_cast<T*>(p));
    }
    Deleter d_;
  };
  public:
  template <typename Deleter>
    smartptr_te_lb(T* p, Deleter d) : p_(p),
      d_((sizeof(Deleter) > sizeof(buf_))
         ? new deleter<Deleter>(d)
         : new (buf_) deleter<Deleter>(d)) {}
  ~smartptr_te_lb() {
    d_->apply(p_);
    if ((void*)(d_) == (void*)(buf_)) {
      d_->~deleter_base();
    } else {
      delete d_;
    }
  }
  T* operator->() { return p_; }
  const T* operator->() const { return p_; }
  private:
  T* p_;
  deleter_base* d_;
  char buf_[16];
};
```

使用之前的相同基准测试，我们可以测量具有局部缓冲区优化的类型擦除智能指针的性能：

```cpp
Benchmark                  Time Items per second
BM_smartptr             21.0 ns 47.5732M/s
BM_smartptr_te          44.2 ns 22.6608M/s
BM_smartptr_te_lb       22.3 ns 44.8747M/s
```

虽然没有类型擦除的智能指针的构建和删除需要 21 纳秒，而有类型擦除的需要 44 纳秒，但优化后的类型擦除共享指针测试在同一台机器上只需要 22 纳秒。轻微的额外开销来自检查`deleter`是存储在本地还是远程。

## 标准库中的局部缓冲区优化

我们应该注意，局部缓冲区优化的最后一种应用，为类型擦除对象存储可调用对象，在 C++标准模板库中被广泛使用。例如，`std::shared_ptr`有一个类型擦除删除器，并且大多数实现都使用局部缓冲区优化；当然，删除器是与引用对象一起存储，而不是与共享指针的每个副本一起存储。另一方面，`std::unique_pointer`标准根本不进行类型擦除，以避免任何小的开销，或者如果删除器不适合局部缓冲区，可能是一个更大的开销。

C++标准库中的“*终极*”类型擦除对象`std::function`通常也使用局部缓冲区来存储小型可调用对象，而不需要额外的分配开销。任何类型的通用容器对象`std::any`（自 C++17 起）在可能的情况下也通常不进行动态分配。

# 局部缓冲区优化的详细说明

我们已经看到了局部缓冲区优化的应用；为了简单起见，我们坚持最基本实现。这种简单实现遗漏了几个重要细节，我们现在将突出显示。

首先，我们完全忽略了缓冲区的对齐。我们用来在对象内部预留空间的类型是 `char`；因此，我们的缓冲区是字节对齐的。大多数数据类型有更高的对齐要求：确切的要求是平台特定的，但大多数内置类型都对其自身大小进行对齐（在 64 位平台如 x86 上，`double` 是 8 字节对齐的）。对于一些特定于机器的类型，如用于 AVX 指令的打包整数或浮点数组，需要更高的对齐。

对齐很重要：根据处理器和编译器生成的代码，如果访问的数据类型未按要求对齐，可能会导致性能下降或内存访问违规（崩溃）。例如，大多数 AVX 指令需要 16 或 32 字节的对齐，而这些指令的非对齐版本要慢得多。另一个例子是原子操作，如用于互斥锁和其他并发数据结构的操作：如果数据类型没有正确对齐，它们也无法工作（例如，原子 `long` 必须对齐在 8 字节边界上）。

指定缓冲区的最小对齐要求并不困难，至少如果我们知道我们想在缓冲区中存储的类型。例如，如果我们有一个任意类型 `T` 的小向量，我们可以简单地写出：

```cpp
template <typename T>
class small_vector {
  alignas(T) char buffer_[buffer_size_];
  …
};
```

如果缓冲区用于存储几种类型之一的对象，我们必须使用所有可能类型中的最高对齐。最后，如果要存储的对象类型未知——这是类型擦除实现的典型情况——我们必须选择一个“*足够高*”的对齐，并在缓冲区中构造特定对象的位置添加编译时检查。

需要记住的第二个重要细节是如何定义缓冲区。通常，它是一个字符（或 `std::byte_t`）的对齐数组。在前一节中，我们使用 `int` 数组来表示小整数向量。同样，这里也有一个细节：将缓冲区声明为对象或正确类型的对象数组，当包含缓冲区的对象被销毁时，这些对象将自动被销毁。对于像整数这样的简单可销毁类型，这根本无关紧要——它们的析构函数什么也不做。

通常情况下并非如此，只有当在此位置构造了对象时，才会调用任意析构函数。对于我们的小型向量，这并不总是成立：向量可能为空或包含的对象少于缓冲区能容纳的数量。这可能是最常见的情况：通常，如果我们采用本地缓冲区优化，我们无法确定对象是否在缓冲区中构造。在这种情况下，将缓冲区声明为具有非平凡析构对象的数组将会是一个错误。然而，如果你有保证，在你的特定情况下，缓冲区总是包含一个对象（或多个对象，对于数组），使用相应的类型声明将大大简化析构函数的实现，以及复制/移动操作。

到现在为止，你应该已经注意到，一个典型的本地缓冲区的实现需要大量的模板代码。到处都是 `reinterpret_cast` 转换，你必须记得添加对齐，还有一些编译时检查你应该始终添加，以确保只有合适的类型存储在缓冲区中，等等。将这些细节组合成一个单一的可重用实现是很好的。不幸的是，正如通常情况下那样，重用性和复杂性之间存在矛盾，所以我们只能满足于几个可重用的通用实现。

如果我们将关于本地缓冲区所学的所有内容综合起来，我们可以得出如下结论：

```cpp
// Example 10
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  constexpr static auto size = S, alignment = A;
  alignas(alignment) char space_[size];
  …
};
```

这里我们有一个任意大小和对齐（两者都是模板参数）的缓冲区。现在我们有了存储对象的空间，我们必须确保我们想要擦除的类型适合这个空间。为此，添加一个 `constexpr` 验证函数是方便的（它仅在编译时语法检查中使用）：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  template <typename T> static constexpr bool valid_type()
  {
    return sizeof(T) <= S && (A % alignof(T)) == 0;
  }
  …
};
```

缓冲区可以通过调用成员函数 `as<T>()` 来使用，仿佛它包含了一个类型为 `T` 的对象：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  template <typename T> requires(valid_type<T>())
    T* as() noexcept {
    return reinterpret_cast<T*>(&space_);
  }
  template <typename T> requires(valid_type<T>())
    const T* as() const noexcept {
    return const_cast<Buffer*>(this)->as<T>();
  }
};
```

缓冲区可以构造为空（默认构造）或带有立即构造的对象。在前一种情况下，对象可以在稍后放置。无论哪种方式，我们都验证类型是否适合缓冲区并满足对齐要求（如果 C++20 和概念不可用，可以使用 SFINAE）。默认构造函数是平凡的，但放置构造函数和 `emplace()` 方法对类型和构造函数参数有约束：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  Buffer() = default;
  template <typename T, typename... Args>
    requires(valid_type<T>() &&
             std::constructible_from<T, Args...>)
  Buffer(std::in_place_type_t<T>, Args&& ...args)
    noexcept(std::is_nothrow_constructible_v<T, Args...>)
  {
    ::new (static_cast<void*>(as<T>()))
      T(std::forward<Args>(args)...);
  }
  template<typename T, typename... Args>
    requires(valid_type<T>() &&
             std::constructible_from<T, Args...>)
  T* emplace(Args&& ...args)
    noexcept(std::is_nothrow_constructible_v<T, Args...>)
  {
    return ::new (static_cast<void*>(as<T>()))
      T(std::forward<Args>(args)...);
  }
};
```

注意，我们确实检查了请求的类型是否可以存储在缓冲区中，但在运行时没有进行检查以确保缓冲区确实包含这样的对象。这种检查可以通过增加额外的空间和运行时计算来实现，并且可能作为调试工具是有意义的。我们对于复制、移动或删除缓冲区没有做任何特殊处理。目前，这个实现适用于简单可复制的和简单可破坏的对象。在这种情况下，当在缓冲区中构造对象时（在构造函数和 `emplace()` 方法中），我们希望断言这些限制：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  template <typename T>
  Buffer(std::in_place_type_t<T>, Args&& ...args) … {
    static_assert(std::is_trivially_destructible_v<T>, "");
    static_assert(std::is_trivially_copyable_v<T>, "");
    …
  }
};
```

在这种情况下，给类添加一个 `swap()` 方法也是有意义的：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  void swap(Buffer& that) noexcept {
    alignas(alignment) char tmp[size];
    ::memcpy(tmp, this->space_, size);
    ::memcpy(this->space_, that.space_, size);
    ::memcpy(that.space_, tmp, size);
  }
};
```

另一方面，如果我们使用这个缓冲区来存储单个已知类型的对象，并且该类型不是简单可破坏的，我们就会一直写类似这样的代码：

```cpp
buffer_.as<T>()->~T();
```

我们可以通过添加另一个通用的方法来简化客户端代码：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  template <typename T> void destroy() {
    this->as<T>()->~T();
  }
};
```

我们可以添加类似的方法来复制和移动存储在缓冲区中的对象，或者让客户端来处理：

我们的一般本地缓冲区实现适用于所有简单可复制的和可破坏的类型，以及所有已知类型的情况，其中客户端处理存储在缓冲区中的对象的复制和销毁。有一个特殊情况被省略了，但仍然值得考虑：当在类型擦除类中使用本地缓冲区时，存储（擦除）的类型可能需要非简单复制或删除，但客户端无法执行这些操作，因为类型擦除的整个目的就是客户端代码在对象放入缓冲区后不知道擦除的类型。在这种情况下，我们需要在存储类型时捕获该类型，并生成相应的复制、移动和删除操作。换句话说，我们必须将我们的本地缓冲区与我们在*第六章*中学习的技术结合起来，*理解类型擦除*。在这种情况下，最合适的类型擦除变体是 `vtable` —— 我们使用模板生成的函数指针表。`vtable` 本身是一个包含将执行删除、复制或移动的函数指针的聚合（`struct`）：

```cpp
// Example 11
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  struct vtable_t {
    using deleter_t = void(Buffer*);
    using copy_construct_t = void(Buffer*, const Buffer*);
    using move_construct_t = void(Buffer*, Buffer*);
    deleter_t*  deleter_;
    copy_construct_t* copy_construct_;
    move_construct_t* move_construct_;
  };
  const vtable_t* vtable_ = nullptr;
};
```

我们需要一个类成员 `vtable_` 来存储对 `vtable` 的指针。当然，我们将要指向的对象需要由构造函数或 `emplace()` 方法创建——这是唯一我们知道实际类型以及如何删除或复制它的时候。但是，我们不会为它进行动态内存分配。相反，我们创建一个静态模板变量，并用指向静态成员函数（也是模板）的指针来初始化它。编译器会为我们在缓冲区中存储的每个类型创建这个静态变量的实例。当然，我们还需要静态模板函数（一个指向静态成员函数的指针与一个常规函数指针相同，而不是成员函数指针）。这些函数由编译器使用存储在缓冲区中的对象的相同类型 `T` 实例化：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  template <typename U, typename T>
  constexpr static vtable_t vtable = {
    U::template deleter<T>,
    U::template copy_construct<T>,
    U::template move_construct<T>
  };
  template <typename T>
    requires(valid_type<T>() &&
    std::is_nothrow_destructible_v<T>)
  static void deleter(Buffer* space) {
    space->as<T>()->~T();
  }
  template <typename T>
    requires(valid_type<T>())
  static void copy_construct(Buffer* to,
                             const Buffer* from)
    noexcept(std::is_nothrow_copy_constructible_v<T>)
  {
    ::new (static_cast<void*>(to->as<T>()))
      T(*from->as<T>());
    to->vtable_ = from->vtable_;
  }
  template <typename T>
    requires(valid_type<T>())
    static void move_construct(Buffer* to, Buffer* from)
    noexcept(std::is_nothrow_move_constructible_v<T>)
  {
    ::new (static_cast<void*>(to->as<T>()))
      T(std::move(*from->as<T>()));
    to->vtable_ = from->vtable_;
  }
};
```

如*第六章* *理解类型擦除*所示，我们首先使用模板静态函数为任何类型 `T` 生成复制、移动和删除操作。我们将这些函数的指针存储在一个静态模板变量 `vtable` 的实例中，并将该实例的指针存储在一个（非静态）数据成员 `vtable_` 中。后者是我们唯一的成本，从大小上来说（其余的是编译器为存储在缓冲区中的每个类型生成一次的静态变量和函数）。

这个 `vtable_` 必须在对象放入缓冲区时初始化，因为这是我们最后一次明确知道存储对象的类型：

```cpp
// Example 11
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  template <typename T, typename... Args>
    requires(valid_type<T>() &&
    std::constructible_from<T, Args...>)
  Buffer(std::in_place_type_t<T>, Args&& ...args)
    noexcept(std::is_nothrow_constructible_v<T, Args...>)
    : vtable_(&vtable<Buffer, T>)
  {
    ::new (static_cast<void*>(as<T>()))
      T(std::forward<Args>(args)...);
  }
  template<typename T, typename... Args>
    requires(valid_type<T>() &&
    std::constructible_from<T, Args...>)
  T* emplace(Args&& ...args)
    noexcept(std::is_nothrow_constructible_v<T, Args...>)
  {
    if (this->vtable_) this->vtable_->deleter_(this);
    this->vtable_ = &vtable<Buffer, T>;
    return ::new (static_cast<void*>(as<T>()))
      T(std::forward<Args>(args)...);
  }
  …
};
```

注意构造函数中 `vtable_` 成员的初始化。在 `emplace()` 方法中，我们还需要删除缓冲区中先前构造的对象，如果有的话。

在类型擦除机制到位后，我们最终可以实现析构函数和复制/移动操作。它们都使用类似的方法——调用 `vtable` 中的相应函数。以下是复制操作：

```cpp
// Example 11
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  Buffer(const Buffer& that) {
    if (that.vtable_)
      that.vtable_->copy_construct_(this, &that);
  }
  Buffer& operator=(const Buffer& that) {
    if (this == &that) return *this;
    if (this->vtable_) this->vtable_->deleter_(this);
    if (that.vtable_)
      that.vtable_->copy_construct_(this, &that);
    else this->vtable_ = nullptr;
    return *this;
  }
};
```

移动操作类似，只是它们使用 `move_construct_` 函数：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  Buffer(Buffer&& that) {
    if (that.vtable_)
      that.vtable_->move_construct_(this, &that);
  }
  Buffer& operator=(Buffer&& that) {
    if (this == &that) return *this;
    if (this->vtable_) this->vtable_->deleter_(this);
    if (that.vtable_)
      that.vtable_->move_construct_(this, &that);
    else this->vtable_ = nullptr;
    return *this;
  }
};
```

注意移动赋值运算符不需要检查自赋值，但这样做也没有错。移动操作最好是 `noexcept`；不幸的是，我们无法保证这一点，因为我们不知道擦除类型。我们可以做出设计选择，并声明它们为 `noexcept`。如果我们这样做，我们还可以在编译时断言我们存储在缓冲区中的对象是 `noexcept` 可移动的。

最后，我们有销毁操作。由于我们允许调用者销毁包含的对象而不销毁缓冲区本身（通过调用 `destroy()`），我们必须确保对象只被销毁一次：

```cpp
template<size_t S, size_t A = alignof(void*)>
struct Buffer {
  …
  ~Buffer() noexcept {
    if (this->vtable_) this->vtable_->deleter_(this);
  }
  // Destroy the object stored in the aligned space.
  void destroy() noexcept {
    if (this->vtable_) this->vtable_->deleter_(this);
    this->vtable_ = nullptr;
  }
};
```

类型擦除的`vtable`允许我们在运行时重建缓冲区中存储的类型（它嵌入为静态函数（如`copy_construct()`）生成的代码中）。当然，这也有成本；我们已注意到额外的数据成员`vtable_`，但还有一些由于间接函数调用而产生的运行时成本。我们可以通过使用本地缓冲区的两种实现（带有和没有类型擦除）来存储和复制一些简单的可复制对象来估计它，例如，一个捕获引用的 lambda 表达式：

```cpp
Benchmark                        Time
BM_lambda_copy_trivial          5.45 ns
BM_lambda_copy_typeerased       4.02 ns
```

（良好实现的）类型擦除的开销不可忽视，但还算适度。一个额外的优势是，我们可以在运行时验证我们的`as<T>()`调用是否引用了一个有效的类型，并且对象确实被构造。相对于未经检查的方法的非常便宜的实现，这将增加显著的开销，因此可能应该限制在调试构建中使用。

我们已经看到，本地缓冲区优化对许多不同的数据结构和类性能的显著甚至戏剧性的改进。有了我们刚刚学到的易于使用的通用实现，为什么你不会一直使用这种优化呢？正如任何设计模式一样，我们的探索如果没有提到权衡和缺点，就不算完整。

# 本地缓冲区优化的缺点

本地缓冲区优化并非没有缺点。最明显的一个缺点是，所有带有本地缓冲区的对象都比没有缓冲区时更大。如果缓冲区中存储的典型数据小于所选的缓冲区大小，那么每个对象都会浪费一些内存，但至少优化是有回报的。更糟糕的是，如果我们选择的缓冲区大小不合适，并且大多数数据实际上比本地缓冲区大，那么数据将存储在远程位置，但每个对象内部仍然会创建本地缓冲区，所有这些内存都浪费了。

在我们愿意浪费的内存量与优化有效的数据大小范围之间存在明显的权衡。本地缓冲区的大小应该根据应用进行仔细选择。

更微妙的问题是这样的——以前存储在对象外部的数据现在存储在对象内部。这有几个后果，除了我们如此关注的性能优势之外。首先，只要数据适合本地缓冲区，每个对象的副本都包含其自己的数据副本。这阻止了像数据引用计数这样的设计；例如，一个**写时复制**（**COW**）字符串，只要所有字符串副本都保持相同，数据就不会被复制，不能使用小字符串优化。

其次，如果对象本身被移动，则必须移动数据。这与`std::vector`的情况形成对比，`std::vector`被移动或交换，本质上就像一个指针——数据指针被移动，但数据本身保持不变。对于`std::any`内部包含的对象也存在类似的考虑。你可以认为这种担忧是微不足道的；毕竟，局部缓冲区优化主要用于少量数据，移动它们的成本应该与复制指针的成本相当。然而，这里不仅仅是性能问题——移动`std::vector`（或`std::any`，无论如何）的实例保证不会抛出异常。然而，在移动任意对象时没有这样的保证。因此，只有当`std::any`包含的对象是`std::is_nothrow_move_constructible`时，`std::any`才能使用局部缓冲区优化。

然而，即使有这样的保证，对于`std::vector`的情况也不够；标准明确指出，移动或交换向量不会使指向向量任何元素的迭代器失效。显然，这个要求与局部缓冲区优化不相容，因为移动一个小向量会将所有元素重新定位到内存的不同区域。因此，许多高效库提供了一种定制的类似向量的容器，它支持小向量优化，但牺牲了标准迭代器失效的保证。

# 摘要

我们刚刚介绍了一种旨在提高性能的设计模式。效率是 C++语言的一个重要考虑因素；因此，C++社区开发了模式来解决最常见的低效问题。重复或浪费的内存分配可能是所有问题中最常见的。我们刚刚看到的设计模式——局部缓冲区优化——是一种强大的工具，可以大大减少这种分配。我们已经看到它如何应用于紧凑的数据结构，以及存储小对象，如可调用对象。我们还回顾了使用此模式可能存在的缺点。

在下一章，*第十一章*，*ScopeGuard*，我们将继续研究更复杂的模式，这些模式解决更广泛的设计问题。我们迄今为止学到的惯用用法通常用于这些模式的实现。

# 问题

1.  我们如何衡量一小段代码的性能？

1.  为什么小而频繁的内存分配对性能尤其不利？

1.  局部缓冲区优化是什么，它是如何工作的？

1.  为什么在对象内部分配一个额外的缓冲区实际上是*免费*的？

1.  短字符串优化是什么？

1.  小向量优化是什么？

1.  为什么局部缓冲区优化对可调用对象特别有效？

1.  使用局部缓冲区优化时需要考虑哪些权衡？

1.  何时不应该将对象放入局部缓冲区？
