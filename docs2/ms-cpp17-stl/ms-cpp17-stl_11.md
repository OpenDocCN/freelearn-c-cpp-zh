# 随机数

在上一章中，你学习了正则表达式，这是一个自 C++11 以来一直是 C++标准库的一部分的功能，但许多程序员仍然不太了解。你看到正则表达式在 C++光谱的两端都很有用——在需要对复杂输入格式进行坚如磐石解析的复杂程序中，以及在需要可读性和开发速度的简单脚本中。

另一个位于这两个类别中的库特性是*随机数生成*。许多脚本程序需要一点随机性，但几十年来，C++程序员一直被告知经典的 libc `rand()` 函数已经过时。在光谱的另一端，`rand()` 对于密码学和复杂的数值模拟来说都是极其不合适的。然而，C++11 `<random>` 库却成功地实现了这三个目标。

在本章中，我们将涵盖以下主题：

+   真正随机数序列与伪随机数序列之间的区别

+   随机比特生成器与产生数据值的分布之间的区别

+   为随机数生成器设置种子的三种策略

+   几种标准库生成器和分布，以及它们的用例

+   如何在 C++17 中洗牌一副牌

# 随机数与伪随机数

在计算机编程的语境中谈论随机数时，我们必须小心地区分真正随机的数，这些数来自物理上非确定性的来源，以及*伪随机*数，这些数来自一个算法，该算法以确定性的方式产生一系列“看起来随机”的数。这样的算法被称为**伪随机数生成器**（**PRNG**）。每个 PRNG 在概念上都以相同的方式工作——它有一些内部*状态*，并且有一些方式让用户请求*下一个输出*。每次我们请求下一个输出时，PRNG 都会根据某种确定性的算法打乱其内部状态，并返回该状态的一部分。以下是一个例子：

```cpp
    template<class T>
    class SimplePRNG {
      uint32_t state = 1;
    public:
      static constexpr T min() { return 0; }
      static constexpr T max() { return 0x7FFF; }

      T operator()() {
        state = state * 1103515245 + 12345;
        return (state >> 16) & 0x7FFF;
      }
    };
```

这个 `SimplePRNG` 类实现了一个*线性同余生成器*，这可能与你的标准库中 `rand()` 的实现非常相似。请注意，`SimplePRNG::operator()` 产生 `[0, 32767]` 15 位范围内的整数，但其内部 `state` 有 32 位范围。这种模式在现实世界的 PRNG 中也是成立的。

例如，标准的梅森旋转算法几乎保持 20 千字节的状态！保持如此多的内部状态意味着有很多位可以混淆，并且每次生成时只有 PRNG 内部状态的一小部分泄露出来。这使得人类（或计算机）在只有少量先前输出的情况下难以预测 PRNG 的下一个输出。预测其输出的难度使我们称这个为*伪随机*数生成器。如果其输出充满了明显的模式和易于预测，我们可能会称其为*非随机*数生成器！

尽管具有伪随机的特性，PRNG 的行为始终是完美的*确定性*；它严格遵循其编码的算法。如果我们运行一个使用 PRNG 的程序并连续运行几次，我们期望每次都能得到完全相同的伪随机数序列。它的严格确定性使我们称这个为*伪*-随机数生成器。

假随机数生成器的一个方面是，两个运行相同算法但初始状态有微小差异的生成器会迅速放大这些差异，*发散*彼此，并产生看起来完全不同的输出序列——就像两滴水被放在你手背上的不同位置，会向完全不同的方向流去。这意味着如果我们想在每次运行程序时得到不同的伪随机数序列，我们只需确保我们为我们的 PRNG 使用不同的*初始状态*。设置 PRNG 的初始状态被称为*播种*PRNG。

我们至少有三种为 PRNG 播种的策略：

+   使用从外部提供的种子*——*来自调用者或最终用户。这对于需要可重复性的任何事物都最合适，例如蒙特卡洛模拟或任何需要进行单元测试的事物。

+   使用可预测但可变的种子，例如当前时间戳。在 C++11 之前，这是最常见的策略，因为 C 标准库提供了一个便携且方便的`time`函数，但它不提供任何真正的随机位源。基于像`time`这样可预测的东西进行播种不适合任何与安全相关的事物。从 C++11 开始，你不应该再使用这种策略。

+   使用从某些特定平台来源直接获得的*真正随机*种子。

*真正随机*的位是通过操作系统基于各种随机事件收集的；一个经典的方法是对于每个系统调用，收集硬件周期计数器的低阶位，并将它们通过 XOR 操作合并到操作系统的*熵池*中。内核内部的伪随机数生成器（PRNG）会定期用熵池中的位重新初始化；该 PRNG 的输出序列被暴露给应用程序开发者。在 Linux 上，原始的熵池作为`/dev/random`暴露，PRNG 的输出序列作为`/dev/urandom`暴露。幸运的是，你永远不需要直接处理这些设备；C++标准库已经为你解决了这个问题。请继续阅读。

# `rand()`的问题

传统的 C 语言生成*随机*数的方法是调用`rand()`函数。这个`rand()`函数仍然是 C++的一部分，它不接受任何参数，并在`[0, RAND_MAX]`范围内产生一个单一、均匀分布的整数。内部状态可以通过调用库函数`void srand(unsigned int seed_value)`来*初始化*。

自 1980 年代以来，生成`[0, x)`范围内的*随机*数的经典代码没有变化，如下所示：

```cpp
    #include <stdlib.h>

    int randint0(int x) {
      return rand() % x;
    }
```

然而，这段代码有几个问题。第一个也是最明显的问题是它没有以相等的可能性生成所有的`x`输出。假设为了论证，`rand()`返回一个在`[0, 32767]`范围内的均匀分布值，那么`randint0(10)`将比返回`8`或`9`更频繁地返回`[0, 7]`范围内的每个值，频率是 1/3276。

第二个问题是`rand()`访问全局状态；在 C++程序中的每个线程都共享同一个随机数生成器。这不是线程安全的问题--`rand()`自 C++11 以来被保证是线程安全的。然而，这是一个性能问题（因为每次调用`rand()`都必须获取全局互斥锁），这也是一个可重复性问题（因为如果你从多个线程并发使用`rand()`，不同的程序运行可能会得到不同的结果）。

`rand()`函数的第三个问题，也是与其全局状态相关的问题，是任何程序中的函数都可以通过调用`rand()`来修改该状态。这使得在单元测试驱动的环境中使用`rand()`变得实际上是不可能的。考虑以下代码片段：

```cpp
    int heads(int n) {
      DEBUG_LOG("heads");
      int result = 0;
      for (int i = 0; i < n; ++i) {
        result += (rand() % 2);
      }
      return result;
    }

    void test_heads() {
      srand(17); // nail down the seed
      int result = heads(42);
      assert(result == 27);
    }
```

显然，单元测试 `test_heads` 将在开始并行化单元测试时立即中断（因为来自其他线程对 `rand()` 的调用将干扰这个测试的微妙工作）。然而，更微妙的是，它也可能因为有人更改了 `DEBUG_LOG` 的实现，添加或删除对 `rand()` 的调用而中断！这种 *遥远的神秘作用* 是任何依赖于全局变量的架构的问题。我们在第八章 分配器中看到了类似的危险。在每种情况下，我强烈推荐的治疗方法都是相同的--*不要使用全局变量。不要使用全局状态*。

因此，C 库有两个问题--它没有提供生成真正均匀分布的伪随机数的方法，并且它从根本上依赖于全局变量。让我们看看 C++ 标准库的 `<random>` 头文件是如何解决这两个问题的。

# 使用 `<random>` 解决问题

`<random>` 头文件提供了两个核心概念--*生成器* 和 *分布*。一个 *生成器*（一个模拟 `UniformRandomBitGenerator` 概念的类）将 PRNG 的内部状态封装到一个 C++ 对象中，并提供了一个以函数调用操作符 `operator()(void)` 形式的下一个输出成员函数。一个 *分布*（一个模拟 `RandomNumberDistribution` 的类）是你可以在生成器的输出上放置的一种过滤器，这样你得到的不是像从 `rand()` 得到的均匀分布的随机位，而是根据指定的数学分布实际数据值分布，并限制在特定范围内，如 `rand() % n`，但更数学上合适且具有更大的灵活性。

`<random>` 头文件包含总共七种 *生成器* 类型以及二十种 *分布* 类型。其中大部分是模板，需要很多参数。这些生成器中大多数比实际应用更有历史意义，而大多数分布只对数学家感兴趣。因此，在本章中，我们将专注于几个标准的生成器和分布，每个都展示了标准库的一些有趣之处。

# 处理生成器

对于任何 *生成器* 对象 `g`，你可以对其执行以下操作：

+   `g()`: 这会打乱生成器的内部状态并产生下一个输出。

+   `g.min()`: 这告诉你 `g()` 的最小可能输出（通常是 `0`）。

+   `g.max()`: 这告诉你 `g()` 的最大可能输出。也就是说，`g()` 的可能输出范围是从 `g.min()` 到 `g.max()`，包括两端。

    可能的输出范围是 `g.min()` 到 `g.max()`，包括 `g.max()`。

+   `g.discard(n)`: 这实际上是对 `g()` 进行 `n` 次调用并丢弃这些结果。

    结果。在一个好的库实现中，你将支付打乱生成器内部状态 `n` 次的费用，但节省与从状态计算下一个输出相关的任何成本。

# 使用 std::random_device 的真正随机位

`std::random_device` 是一个 *生成器*。它的接口极其简单；它甚至不是一个类模板，而是一个普通的类。一旦你使用其默认构造函数构造了一个 `std::random_device` 的实例，你就可以使用其重载的调用操作符来获取类型为 `unsigned int` 的值，这些值在闭区间 `[rd.min(), rd.max()]` 内均匀分布。

一个需要注意的地方是，`std::random_device` 并不完全符合 `UniformRandomBitGenerator` 的概念。最重要的是，它既不可复制也不可移动。在实践中，这并不是一个大问题，因为你通常不会长时间保留一个 *真正* 随机的生成器。相反，你会使用一个短暂的 `std::random_device` 实例来为某种类型的长期伪随机生成器生成一个 *种子*，如下所示：

```cpp
    std::random_device rd;
    unsigned int seed = rd();
    assert(rd.min() <= seed && seed <= rd.max());
```

现在我们来看看你唯一需要了解的伪随机生成器。

# 使用 std::mt19937 的伪随机位

你唯一需要了解的伪随机生成器被称为 *梅森旋转器* 算法。这个算法自 1997 年以来就为人所知，在任何编程语言中的高质量实现都很容易找到。从技术上讲，梅森旋转器算法定义了一个相关 PRNG 的整个家族--它是 C++模板的算法等价物--但这个家族中最常用的成员被称为 **MT19937**。这一串数字可能看起来像时间戳，但并非如此；它是旋转器内部状态的大小（以位为单位）。因为梅森旋转器的下一个输出函数完美地打乱了其状态，它最终会达到（除了一个之外）所有可能的状态，然后再回到开始--MT19937 生成器的 *周期* 是 2¹⁹⁹³⁷-1。与此相比，我们本章开头的 `SimplePRNG` 只有一个 32 位的内部状态和一个周期为 2³¹。（我们的 `SimplePRNG` 生成器有 2³² 种可能的内部状态，但在它再次循环之前，只有一半的状态被达到。例如，`state=3` 从初始的 `state=1` 是无法到达的。）

理论已经足够。让我们看看梅森旋转器在实际中的应用！对应于梅森旋转器 *算法模板* 的 C++类模板是 `std::mersenne_twister_engine<...>`，但你不会直接使用它；你将使用便利的 typedef `std::mt19937`，如下所示：

```cpp
    std::mt19937 g;
    assert(g.min() == 0 && g.max() == 4294967295);

    assert(g() == 3499211612);
    assert(g() == 581869302);
    assert(g() == 3890346734);
```

`std::mt19937` 的默认构造函数将其内部状态设置为众所周知的标准值。这确保了从默认构造的 `mt19937` 对象获得的输出序列在所有平台上都是相同的--与 `rand()` 相比，`rand()` 在不同平台上往往给出不同的输出序列。

要获得不同的输出序列，你需要向`std::mt19937`的构造函数提供一个**种子**。 在 C++17 中有两种方法--繁琐的方法和简单的方法。 繁琐的方法是构建一个真正的 19937 位种子，并将其通过一个**种子序列**复制到`std::mt19937`对象中，如下所示：

```cpp
    std::random_device rd;

    uint32_t numbers[624];
    std::generate(numbers, std::end(numbers), std::ref(rd));
      // Generate initial state.

    SeedSeq sseq(numbers, std::end(numbers));
      // Copy our state into a heap-allocated "seed sequence".

    std::mt19937 g(sseq);
      // Initialize a mt19937 generator with our state.
```

在这里，`SeedSeq`类型可以是`std::seed_seq`（一个被美化的`std::vector`；它使用堆分配）或者一个正确编写的“种子序列”类，如下所示：

```cpp
    template<class It>
    struct SeedSeq {
      It begin_;
      It end_;
    public:
      SeedSeq(It begin, It end) : begin_(begin), end_(end) {}

      template<class It2>
      void generate(It2 b, It2 e) {
        assert((e - b) <= (end_ - begin_));
        std::copy(begin_, begin_ + (e - b), b);
      }
    };
```

当然，仅仅为了构建一个单一的 PRNG 对象就需要写这么多代码！ (我告诉过你，这是**繁琐**的方法。) 简单的方法，也是你将在实践中看到的方法，是将 MT19937 用单个真正的**32 位整数**进行初始化，如下所示：

```cpp
    std::random_device rd;

    std::mt19937 g(rd());
      // 32 bits of randomness ought to be enough for anyone!
      // ...Right?
```

警惕！32 比 19937 小得多！这种简单的初始化方法只能产生 40 亿种不同的输出序列，**永远**；这意味着如果你用随机种子反复运行你的程序，你可以在运行了几十万次之后看到一些重复。 (这是著名的**生日悖论**的应用。) 然而，如果你认为这种可预测性很重要，你可能还应该知道，梅森旋转器**不是密码学安全的**。 这意味着即使你用真正的 19937 位种子序列初始化它，恶意攻击者也可以逆向工程你的原始种子中的所有 19937 位，并在只看到输出序列的几百项之后，完美准确地预测后续的每个输出。 如果你需要一个**密码学安全的伪随机数生成器**（**CSPRNG**），你应该使用类似 AES-CTR 或 ISAAC 的东西，这两种东西都不是 C++标准库提供的。 你仍然应该将你的 CSPRNG 实现包装在一个模拟`UniformRandomBitGenerator`的类中，这样它就可以与标准算法一起使用，我们将在本章末尾讨论这一点。

# 使用适配器过滤生成器输出

我们提到，生成器的原始输出通常需要通过单个**分布**进行过滤，以便将生成器的原始比特转换为可用的数据值。有趣的是，也可以将生成器的输出通过一个**生成器适配器**发送，该适配器可以以各种可能有用的方式重新格式化原始比特。标准库提供了三种适配器--`std::discard_block_engine`、`std::shuffle_order_engine`和`std::independent_bits_engine`。这些适配器类型的工作方式与我们在第四章“容器动物园”中讨论的**容器适配器**（如`std::stack`）类似--它们提供一定的接口，但将大部分实现细节委托给其他某个类。

`std::discard_block_engine<Gen, p, r>`的一个实例保留了一个类型为`Gen`的*底层生成器*，并将所有操作委托给该底层生成器，除了`discard_block_engine::operator()`将只返回底层生成器每`p`个输出中的前`r`个。例如，考虑以下示例：

```cpp
    std::vector<uint32_t> raw(10), filtered(10);

    std::discard_block_engine<std::mt19937, 3, 2> g2;
    std::mt19937 g1 = g2.base();

    std::generate(raw.begin(), raw.end(), g1);
    std::generate(filtered.begin(), filtered.end(), g2);

    assert(raw[0] == filtered[0]);
    assert(raw[1] == filtered[1]);
      // raw[2] doesn't appear in filtered[]
    assert(raw[3] == filtered[2]);
    assert(raw[4] == filtered[3]);
      // raw[5] doesn't appear in filtered[]
```

注意，可以通过`g2.base()`检索底层生成器的引用。在上面的示例中，`g1`被初始化为`g2.base()`的一个副本；这解释了为什么调用`g1()`不会影响`g2`的状态，反之亦然。

`std::shuffle_order_engine<Gen, k>`的一个实例保留其底层生成器最后*k*个输出的缓冲区，以及一个额外的整数`Y`。每次调用

`shuffle_order_engine::operator()`将`Y = buffer[Y % k]`设置为`buffer[Y] = base()()`。（从`Y`计算缓冲区索引的公式实际上比简单的模运算更复杂，但它基本上有相同的效果。）值得注意的是，`std::shuffle_order_engine`并不使用`std::uniform_int_distribution`将`Y`映射到`0, k)`范围。这不会影响生成器输出的*随机性*——如果底层生成器已经是伪随机的话，稍微打乱其输出并不会使它们变得更加或更少随机，无论我们使用什么算法来进行打乱。因此，`shuffle_order_engine`使用的算法是专门挑选的，因为它具有历史兴趣——它是唐纳德·克努特在《计算机程序设计艺术》中描述的经典算法的一个构建块：

```cpp
    using knuth_b = std::shuffle_order_engine<
      std::linear_congruential_engine<
        uint_fast32_t, 16807, 0, 2147483647
      >,
      256
    >;
```

`std::independent_bits_engine<Gen, w, T>`的一个实例除了其底层生成器类型为`Gen`之外，不保留任何状态。`independent_bits_engine::operator()`函数调用`base()()`足够多次以计算至少`w`个随机位；然后，它通过一个比实际应用更有历史意义的算法精确地拼接这些位，并将它们作为类型为`T`的无符号整数提供。 （如果`T`不是无符号整数类型，或者`T`的位数少于`w`位，则是一个错误。）

以下是一个`independent_bits_engine`从多个`base()()`调用中拼接位的示例：

```cpp
    std::independent_bits_engine<std::mt19937, 40, uint64_t> g2;
    std::mt19937 g1 = g2.base();

    assert(g1() == 0xd09'1bb5c); // Take "1bb5c"...
    assert(g1() == 0x22a'e9ef6); // and "e9ef6"...
    assert(g2() == 0x1bb5c'e9ef6); // Paste and serve!
```

以下是一个使用`independent_bits_engine`从`mt19937`的输出中移除所有但最低有效位（创建一个*翻转生成器*）的示例，然后，将这个生成器的 32 个输出拼接起来，以重建一个 32 位生成器：

```cpp
    using coinflipper = std::independent_bits_engine<
      std::mt19937, 1, uint8_t>;

    coinflipper onecoin;
    std::array<int, 64> results;
    std::generate(results.begin(), results.end(), onecoin);
    assert((results == std::array<int, 64>{{
      0,0,0,1, 0,1,1,1, 0,1,1,1, 0,0,1,0,
      1,0,1,0, 1,1,1,1, 0,0,0,1, 0,1,0,1,
      1,0,0,1, 1,1,1,0, 0,0,1,0, 1,0,1,0,
      1,0,0,1, 0,0,0,0, 0,1,0,0, 1,1,0,0,
    }}));

    std::independent_bits_engine<coinflipper, 32, uint32_t> manycoins;
    assert(manycoins() == 0x1772af15);
    assert(manycoins() == 0x9e2a904c);
```

注意，`independent_bits_engine`对其底层生成器的位不执行任何复杂的操作；特别是，它假设其底层生成器没有偏差。如果`WeightedCoin`生成器倾向于偶数。你将看到这种偏差也会在`independent_bits_engine<WeightedCoin, w, T>`的输出中体现出来。

尽管我们花费了数页的篇幅来讨论这些生成器，但请记住，在你的代码中没有任何理由使用这些神秘的类！如果你需要一个伪随机数生成器，请使用 `std::mt19937`；如果你需要一个加密安全的伪随机数生成器，请使用类似 AES-CTR 或 ISAAC 的东西；如果你需要相对较少的真正随机位来为你的伪随机数生成器设置种子，请使用 `std::random_device`。这些是你将在实践中唯一使用的生成器。

# 处理分布

现在我们已经看到了如何按需生成随机位，让我们看看如何将这些随机位转换为匹配特定 *分布* 的数值。这个两步过程--生成原始位，然后将它们格式化为数据值--与我们前面在 [第九章 中介绍的缓冲和解析的两步过程非常相似，即 *Iostreams*。首先，获取原始位和字节，然后执行某种操作将这些位和字节转换为类型化的数据值。

对于任何分布对象 `dist`，你可以对其执行以下操作：

+   `dist(g)`: 这将根据适当的数学分布产生下一个输出。这可能需要多次调用 `g()`，或者根本不需要，这取决于 `dist` 对象的内部状态。

+   `dist.reset()`: 这将清除 `dist` 对象的内部状态（如果有的话）。你永远不会需要使用这个成员函数。

+   `dist.min()` 和 `dist.max()`: 这些告诉你 `dist(g)` 对于任何随机位生成器 `g` 的最小和最大可能输出。通常，这些值要么是显而易见的，要么是没有意义的；例如，`std::normal_distribution<float>().max()` 是 `INFINITY`。

让我们看看几个分布类型在实际中的应用。

# 使用 uniform_int_distribution 投掷骰子

`std::uniform_int_distribution` 方法是标准库中最简单的分布类型。它执行的操作与我们本章前面尝试用 `randint0` 执行的操作相同--将一个随机无符号整数映射到给定的范围中--但它没有任何偏差。`uniform_int_distribution` 的最简单实现看起来可能像这样：

```cpp
    template<class Int>
    class uniform_int_distribution {
      using UInt = std::make_unsigned_t<Int>;
      UInt m_min, m_max;
    public:
      uniform_int_distribution(Int a, Int b) :
        m_min(a), m_max(b) {}

      template<class Gen>
      Int operator()(Gen& g) {
        UInt range = (m_max - m_min);
        assert(g.max() - g.min() >= range);
        while (true) {
          UInt r = g() - g.min();
          if (r <= range) {
            return Int(m_min + r);
          }
        }
      }
    };
```

实际的标准库实现必须做一些事情来消除那个 `assert`。通常，他们会使用类似 `independent_bits_engine` 的东西来一次生成正好 `ceil(log2(range))` 个随机位，从而最小化 `while` 循环需要运行的次数。

如前例所示，`uniform_int_distribution` 是无状态的（尽管这并不是 *技术上* 保证的），因此最常见的使用方式是在每次生成数字时创建一个新的分布对象。因此，我们可以像这样实现我们的 `randint0` 函数：

```cpp
    int randint0(int x) {
      static std::mt19937 g;
      return std::uniform_int_distribution<int>(0, x-1)(g);
    }
```

现在可能是时候指出 `<random>` 库的一些奇怪之处了。一般来说，每次你向这些函数或构造函数提供一个 *整数数值范围* 时，它被视为一个 *闭区间*。这与 C 和 C++ 中范围通常的工作方式形成鲜明对比；我们甚至在 第三章，*迭代器对算法* 中看到，偏离 *半开区间* 规则通常是代码有问题的标志。然而，在 C++ 随机数库的情况下，有一条新的规则--*闭区间* 规则。为什么？

好吧，半开区间的关键优势是它可以轻松地表示一个 *空区间*。另一方面，半开区间不能表示一个 *完全满的区间*，也就是说，一个覆盖整个域的区间。（我们在 第四章，*容器动物园* 的实现中看到了这个问题。）假设我们想要表达在整个 `long long` 范围上均匀分布的概念。我们不能将其表示为半开区间 `[LLONG_MIN, LLONG_MAX+1)`，因为 `LLONG_MAX+1` 会溢出。然而，我们可以将其表示为闭区间 `[LLONG_MIN, LLONG_MAX]`--因此，这就是 `<random>` 库的函数和类（如 `uniform_int_distribution`）所做的事情。《uniform_int_distribution<int>(0,6)` 方法是在 `[0,6]` 七个数范围内的分布，而 `uniform_int_distribution<int>(42,42)` 是一个完全有效的分布，总是返回 `42`。

另一方面，`std::uniform_real_distribution<double>(a, b)` *确实* 在一个半开区间上操作！`std::uniform_real_distribution<double>(0, 1)` 方法产生类型为 `double` 的值，在 `0, 1)` 范围内均匀分布。在浮点数域中，没有溢出问题--`[0, INFINITY)` 的半开区间实际上是可以表示的，尽管当然，在无限范围内不存在 *均匀分布*。浮点数也使得很难区分半开区间和闭区间；例如，`std::uniform_real_distribution<float>(0, 1)(g)` 可以合法地返回 `float(1.0)`，只要它生成的随机实数足够接近 1，以至于每 2²⁵ 个结果中大约有一个会被四舍五入。 (在出版时，libc++ 的行为如上所述。GNU 的 libstdc++ 应用了一个补丁，使得接近 1 的实数向下而不是向上舍入，因此略低于 1.0 的浮点数出现的频率略高于随机预测。)

# 使用 normal_distribution 生成种群

实值分布最有用的例子可能是**正态分布**，也称为**钟形曲线**。在现实世界中，正态分布无处不在，尤其是在一个群体中物理特征的分布中。例如，成年人类身高的直方图往往会呈现出正态分布——许多个体围绕着平均身高聚集，其他人则向两边延伸。反过来，这意味着你可能想要使用正态分布来为游戏中的模拟个体分配身高、体重等。

`std::normal_distribution<double>(m, sd)` 方法构建了一个具有均值（`m`）和标准差（`sd`）的 `normal_distribution<double>` 实例。（如果你没有提供这些参数，这些参数默认为 `m=0` 和 `sd=1`，所以要注意拼写错误！）以下是一个使用 `normal_distribution` 创建 10,000 个正态分布样本的“人口”，然后通过数学方法验证其分布的示例：

```cpp
    double mean = 161.8;
    double stddev = 6.8;
    std::normal_distribution<double> dist(mean, stddev);

      // Initialize our generator.
    std::mt19937 g(std::random_device{}());

      // Fill a vector with 10,000 samples.
    std::vector<double> v;
    for (int i=0; i < 10000; ++i) {
      v.push_back( dist(g) );
    }
    std::sort(v.begin(), v.end());

      // Compare expectations with reality.
    auto square = [ { return x*x; };
    double mean_of_values = std::accumulate(
      v.begin(), v.end(), 0.0) / v.size();
    double mean_of_squares = std::inner_product(
      v.begin(), v.end(), v.begin(), 0.0) / v.size();
    double actual_stddev =
      std::sqrt(mean_of_squares - square(mean_of_values));
    printf("Expected mean and stddev: %g, %g\n", mean, stddev);
    printf("Actual mean and stddev: %g, %g\n",
           mean_of_values, actual_stddev);
```

与本章中（或将要看到的）的其他分布不同，`std::normal_distribution` 是有状态的。虽然为每个生成的值构造一个新的 `std::normal_distribution` 实例是可以的，但如果你这样做，实际上会减半你程序的效率。这是因为生成正态分布值的最流行算法每次产生两个独立值；`std::normal_distribution` 不能一次给你两个值，所以它会将其中一个值保留在成员变量中，以便下次请求时提供给你。可以使用 `dist.reset()` 成员函数清除这个保存的状态，尽管你永远不会想这样做。

# 使用 `discrete_distribution` 进行加权选择

`std::discrete_distribution<int>(wbegin, wend)` 方法在 `[0, wend - wbegin)` 的半开区间上构建一个离散的或加权的分布。以下示例可以最容易地解释这一点：

```cpp
    template<class Values, class Weights, class Gen>
    auto weighted_choice(const Values& v, const Weights& w, Gen& g)
    {
      auto dist = std::discrete_distribution<int>(
        std::begin(w), std::end(w));
      int index = dist(g);
      return v[index];
    }

    void test() {
      auto g = std::mt19937(std::random_device{}());
      std::vector<std::string> choices =
        { "quick", "brown", "fox" };
      std::vector<int> weights = { 1, 7, 2 };
      std::string word = weighted_choice(choices, weights, g);
        // 7/10 of the time, we expect word=="brown".
    }
```

`std::discrete_distribution<int>` 方法会将其传入的权重在自己的私有成员变量 `std::vector<double>` 中创建一个内部副本（并且，像 `<random>` 中的常规操作一样，它不是分配器感知的）。你可以通过调用 `dist.probabilities()` 来获取这个向量的副本，如下所示：

```cpp
    int w[] = { 1, 0, 2, 1 };
    std::discrete_distribution<int> dist(w, w+4);
    std::vector<double> v = dist.probabilities();
    assert((v == std::vector{ 0.25, 0.0, 0.50, 0.25 }));
```

你可能不想直接在自己的代码中使用 `discrete_distribution`；最好的办法是将它的使用封装在类似前面的 `weighted_choice` 函数中。然而，如果你需要避免堆分配或浮点运算，使用一个更简单的不分配函数可能更有利，如下所示：

```cpp
    template<class Values, class Gen>
    auto weighted_choice(
      const Values& v, const std::vector<int>& w,
      Gen& g)
    {
      int sum = std::accumulate(w.begin(), w.end(), 0);
      int cutoff = std::uniform_int_distribution<int>(0, sum - 1)(g);
      auto vi = v.begin();
      auto wi = w.begin();
      while (cutoff > *wi) {
        cutoff -= *wi++;
        ++vi;
      }
      return *vi;
    }
```

然而，`discrete_distribution` 的默认库实现之所以将其所有数学运算作为浮点数进行，是因为它为你节省了担心整数溢出的麻烦。如果 `sum` 超出了 `int` 的范围，前面的代码将会有不良行为。

# 使用 `std::shuffle` 洗牌

让我们通过查看`std::shuffle(a,b,g)`来结束这一章，这是唯一一个接受随机数生成器作为输入的标准算法。根据第三章的定义，它是一个*排列算法*--它接受一个元素范围 `[a,b)` 并对其进行洗牌，保留其值但不保留其位置。

`std::shuffle(a,b,g)`方法是在 C++11 中引入的，用于取代旧的`std::random_shuffle(a,b)`算法。那个旧的算法“随机”地洗牌 `[a,b)` 范围，但没有指定随机性的来源；在实践中，这意味着它将使用全局 C 库的`rand()`，并带来所有相关问题。一旦 C++11 通过`<random>`引入了关于随机数生成器的标准化方法，就到了摆脱基于旧`rand()`的`random_shuffle`的时候了；并且，截至 C++17，`std::random_shuffle(a,b)`不再是 C++标准库的一部分。

这是我们可以如何使用 C++11 的`std::shuffle`来洗牌一副扑克牌的方法：

```cpp
    std::vector<int> deck(52);
    std::iota(deck.begin(), deck.end(), 1);
      // deck now contains ints from 1 to 52.

    std::mt19937 g(std::random_device{}());
    std::shuffle(deck.begin(), deck.end(), g);
      // The deck is now randomly shuffled.
```

回想一下，`<random>`中的每个*生成器*都是完全指定的，例如，使用固定值初始化的`std::mt19937`实例将在每个平台上产生完全相同的输出。对于像`uniform_real_distribution`这样的*分布*，以及`shuffle`算法，情况并非如此。从 libc++切换到 libstdc++，或者只是升级编译器，可能会导致你的`std::shuffle`行为发生变化。

```cpp
9 different shuffles--out of the 8 × 1067 ways, you can shuffle a deck of cards by hand! If you were shuffling cards for a real casino game, you'd certainly want to use the "tedious" method of seeding, described earlier in this chapter, or--simpler, if performance isn't a concern--just use std::random_device directly:
```

```cpp
    std::random_device rd;
    std::shuffle(deck.begin(), deck.end(), rd);
    // The deck is now TRULY randomly shuffled.
```

无论你使用什么生成器和初始化方法，你都可以直接将其插入到`std::shuffle`中。这是标准库对随机数生成可组合方法的好处。

# 摘要

标准库提供了两个与随机数相关的概念--*生成器*和*分布*。生成器是有状态的，必须进行初始化，并通过`operator()(void)`产生无符号整数输出（原始比特）。两种重要的生成器类型是`std::random_device`，它产生真正的随机比特，以及`std::mt19937`，它产生伪随机比特。

分布通常是*无状态的*，并通过`operator()(Gen&)`产生数值数据。对于大多数程序员来说，最重要的分布类型将是`std::uniform_int_distribution<int>(a,b)`，它产生闭区间 `[a,b]` 内的整数。标准库还提供了其他分布，例如`std::uniform_real_distribution`、`std::normal_distribution`和`std::discrete_distribution`，以及许多对数学家和统计学家有用的神秘分布。

使用随机性的唯一标准算法是`std::shuffle`，它取代了旧式的`std::random_shuffle`。不要在新代码中使用`random_shuffle`。

注意，`std::mt19937`在所有平台上具有完全相同的行为，但任何*分布*类型，以及`std::shuffle`，情况并非如此。
