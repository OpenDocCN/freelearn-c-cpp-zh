# 第十四章：14

# 并行算法

之前的章节着重介绍了如何通过使用线程和协程在我们的程序中引入并发和异步性。本章侧重于独立任务的并行执行，这与并发相关但又不同。

在之前的章节中，我强调我更喜欢使用标准库算法而不是手工制作的`for`循环。在本章中，您将看到使用 C++17 引入的执行策略与标准库算法的一些巨大优势。

本章不会深入探讨并行算法或并行编程的理论，因为这些主题太复杂，无法在一章中涵盖。此外，关于这个主题有很多书籍。相反，本章将采取更实际的方法，演示如何扩展当前的 C++代码库以利用并行性，同时保持代码库的可读性。换句话说，我们不希望并行性影响可读性；相反，我们希望将并行性抽象出来，使得并行化代码只是改变算法的一个参数。

在本章中，您将学到：

+   实现并行算法的各种技术

+   如何评估并行算法的性能

+   如何调整代码库以使用标准库算法的并行扩展

并行编程是一个复杂的话题，因此在开始之前，您需要了解引入并行性的动机。

# 并行性的重要性

从程序员的角度来看，如果今天的计算机硬件是一个 100 GHz 的单核 CPU 而不是一个 3 GHz 的多核 CPU，那将非常方便；我们就不需要关心并行性。不幸的是，使单核 CPU 变得越来越快已经达到了物理极限。因此，随着计算机硬件的发展朝着多核 CPU 和可编程 GPU 的方向发展，程序员必须使用高效的并行模式来充分利用硬件。

并行算法允许我们通过在多核 CPU 或 GPU 上同时执行多个单独的任务或子任务来优化我们的程序。

# 并行算法

正如在*第十一章*，*并发*中提到的，*并发*和*并行*这两个术语有时很难区分。作为提醒，如果程序在重叠的时间段内具有多个单独的控制流，则称该程序在并发运行。另一方面，并行程序同时执行多个任务或子任务（在完全相同的时间），这需要具有多个核心的硬件。我们使用并行算法来优化延迟或吞吐量。如果没有硬件可以同时执行多个任务以实现更好的性能，那么并行化算法就毫无意义。现在将介绍一些简单的公式，以帮助您了解在评估并行算法时需要考虑哪些因素。

## 评估并行算法

在本章中，**加速比**被定义为顺序算法和并行算法之间的比率，如下所示：

![](img/B15619_14_001.png)

*T*[1]是使用顺序算法在一个核心上执行解决问题所需的时间，*T*[n]是使用*n*个核心解决相同问题所需的时间。*时间*指的是挂钟时间（而不是 CPU 时间）。

与其顺序等效物相比，并行算法通常更复杂，需要更多的计算资源（例如 CPU 时间）。并行版本的好处来自于能够将算法分布到多个处理单元上。

在这种情况下，值得注意的是，并非所有算法在并行运行时都能获得相同的性能提升。并行算法的**效率**可以通过以下公式计算：

![](img/B15619_14_002.png)

在这个公式中，*n*是执行算法的核心数。由于*T*[1]/*T*[n]表示加速比，效率也可以表示为*加速比*/*n*。

如果效率为*1.0*，则算法并行化完美。例如，这意味着在具有八个核心的计算机上执行并行算法时，我们可以实现 8 倍的加速。但在实践中，有许多参数限制了并行执行，比如创建线程、内存带宽和上下文切换，正如*第十一章*，*并发*中所述。因此，通常效率远低于 1.0。

并行算法的效率取决于每个工作块的独立处理程度。例如，`std::transform()`在某种意义上是非常容易并行化的，因为每个元素的处理完全独立于其他元素。这将在本章后面进行演示。

效率还取决于问题的规模和核心数量。例如，由于并行算法的复杂性增加而导致的开销，一个并行算法在小数据集上可能表现非常糟糕。同样，在计算机上执行程序时，可能会遇到其他瓶颈，比如内存带宽，这可能会导致在大量核心上执行程序时性能下降。我们说一个并行算法是可扩展的，如果效率在改变核心数量和/或输入规模时保持不变。

同样重要的是要记住，并非程序的所有部分都可以并行化。即使我们有无限数量的核心，这个事实也限制了程序的理论最大加速。我们可以使用**阿姆达尔定律**来计算最大可能的加速，这是在*第三章*，*分析和测量性能*中介绍的。

## 阿姆达尔定律的再审视

在这里，我们将阿姆达尔定律应用于并行程序。它的工作原理是：程序的总运行时间可以分为两个不同的部分或*比例*：

+   *F*[seq]是程序中只能按顺序执行的部分的比例

+   *F*[par]是程序中可以并行执行的部分的比例

由于这两个比例加起来构成了整个程序，这意味着*F*[seq] = 1 - *F*[par]。现在，阿姆达尔定律告诉我们，程序在*n*个核心上执行的**最大加速**是：

![](img/B15619_14_003.png)

为了可视化这个定律的效果，下面的图像显示了一个程序的执行时间，其中底部是顺序部分，顶部是并行部分。增加核心数量只会影响并行部分，这限制了最大加速比。

![](img/B15619_14_01.png)

图 14.1：阿姆达尔定律定义了最大加速；在这种情况下是 2 倍

在上图中，当在单个 CPU 上运行时，顺序部分占执行时间的 50%。因此，当执行这样的程序时，通过增加更多核心，我们可以实现的最大加速是 2 倍。

为了让您了解并行算法是如何实现的，我们现在将通过一些示例来说明。我们将从`std::transform()`开始，因为它相对容易分成多个独立的部分。

## 实现并行的 std::transform()

尽管从算法上来说`std::transform()`很容易实现，但在实践中，实现一个初级的并行版本比看上去更复杂。

算法`std::transform()`为序列中的每个元素调用一个函数，并将结果存储在另一个序列中。`std::transform()`的顺序版本的一个可能实现可能看起来像这样：

```cpp
template<class SrcIt, class DstIt, class Func>
auto transform(SrcIt first, SrcIt last, DstIt dst, Func func) {
  while (first != last) {
      *dst++ = func(*first++);
  }
} 
```

标准库版本也返回`dst`迭代器，但在我们的示例中将忽略它。为了理解`std::transform()`并行版本的挑战，让我们从一个天真的方法开始。

### 天真的实现

`std::transform()`的一个天真的并行实现可能看起来像这样：

+   将元素分成与计算机核心数相对应的块

+   在单独的任务中处理每个块

+   等待所有任务完成

使用 `std::thread::hardware_concurrency()` 来确定支持的硬件线程数量，可能的实现如下：

```cpp
template <typename SrcIt, typename DstIt, typename Func>
auto par_transform_naive(SrcIt first, SrcIt last, DstIt dst, Func f) {
  auto n = static_cast<size_t>(std::distance(first, last));
  auto n_cores = size_t{std::thread::hardware_concurrency()};
  auto n_tasks = std::max(n_cores, size_t{1});
  auto chunk_sz = (n + n_tasks - 1) / n_tasks;
  auto futures = std::vector<std::future<void>>{};
  // Process each chunk on a separate
  for (auto i = 0ul; i < n_tasks; ++i) {
    auto start = chunk_sz * i;
    if (start < n) {
      auto stop = std::min(chunk_sz * (i + 1), n);
      auto fut = std::async(std::launch::async,
         [first, dst, start, stop, f]() {
          std::transform(first + start, first + stop, dst + start, f);
      });
      futures.emplace_back(std::move(fut));
    }
  }
  // Wait for each task to finish
  for (auto&& fut : futures) {
    fut.wait();
  }
} 
```

请注意，如果 `hardware_concurrency()` 由于某种原因无法确定，可能会返回 `0`，因此会被夹制为至少为 1。

`std::transform()` 和我们的并行版本之间的一个细微差别是它们对迭代器有不同的要求。`std::transform()` 可以操作输入和输出迭代器，比如绑定到 `std::cin` 的 `std::istream_iterator<>`。这对于 `par_transform_naive()` 是不可能的，因为迭代器被复制并且从多个任务中使用。正如你将看到的，本章中没有呈现可以操作输入和输出迭代器的并行算法。相反，并行算法至少需要允许多次遍历的前向迭代器。

#### 性能评估

继续使用朴素实现，让我们通过与在单个 CPU 核心上执行的顺序版本 `std::transform()` 的简单性能评估来测量其性能。

在这个测试中，我们将测量数据的输入大小变化时的时间（挂钟时间）和在 CPU 上花费的总时间。

我们将使用在 *第三章* *分析和测量性能* 中介绍的 Google Benchmark 来设置这个基准。为了避免重复代码，我们将实现一个函数来为我们的基准设置一个测试夹具。夹具需要一个包含一些示例值的源范围，一个用于结果的目标范围，以及一个转换函数：

```cpp
auto setup_fixture(int n) {
  auto src = std::vector<float>(n);
  std::iota(src.begin(), src.end(), 1.0f); // Values from 1.0 to n
  auto dst = std::vector<float>(src.size());
  auto transform_function = [](float v) { 
    auto sum = v;
    for (auto i = 0; i < 500; ++i) {
      sum += (i * i * i * sum);
    }
    return sum;
  };
  return std::tuple{src, dst, transform_function};
} 
```

现在我们已经设置好了我们的夹具，是时候实现实际的基准了。将会有两个版本：一个用于顺序的 `std::transform()`，一个用于我们的并行版本 `par_transform_naive()`：

```cpp
void bm_sequential(benchmark::State& state) {
  auto [src, dst, f] = setup_fixture(state.range(0));
  for (auto _ : state) {
    std::transform(src.begin(), src.end(), dst.begin(), f);
  }
}
void bm_parallel(benchmark::State& state) {
  auto [src, dst, f] = setup_fixture(state.range(0));
  for (auto _ : state) {
    par_transform_naive(src.begin(), src.end(), dst.begin(), f);
  }
} 
```

只有 `for`-循环内的代码将被测量。通过使用 `state.range(0)` 作为输入大小，我们可以通过将一系列值附加到每个基准来生成不同的值。实际上，我们需要为每个基准指定一些参数，因此我们创建一个帮助函数，应用我们需要的所有设置：

```cpp
void CustomArguments(benchmark::internal::Benchmark* b) {
  b->Arg(50)->Arg(10'000)->Arg(1'000'000)  // Input size
      ->MeasureProcessCPUTime()            // Measure all threads
      ->UseRealTime()                      // Clock on the wall 
      ->Unit(benchmark::kMillisecond);     // Use ms
} 
```

关于自定义参数的一些注意事项：

+   我们将值 50、10,000 和 1,000,000 作为基准的参数传递。它们在创建 `setup_fixture()` 函数中的向量时用作输入大小。在测试函数中使用 `state.range(0)` 访问这些值。

+   默认情况下，Google Benchmark 只在主线程上测量 CPU 时间。但由于我们对所有线程上的 CPU 时间总量感兴趣，我们使用 `MeasureProcessCPUTime()`。

+   Google Benchmark 决定每个测试需要重复多少次，直到达到统计上稳定的结果。我们希望库在这方面使用挂钟时间而不是 CPU 时间，因此我们应用设置 `UseRealTime()`。

这几乎就是了。最后，注册基准并调用 main：

```cpp
BENCHMARK(bm_sequential)->Apply(CustomArguments);
BENCHMARK(bm_parallel)->Apply(CustomArguments);
BENCHMARK_MAIN(); 
```

在使用优化后的代码（使用 gcc 和 -O3）编译后，我在一台具有八个核心的笔记本电脑上执行了这个基准。以下表格显示了使用 50 个元素时的结果：

| 算法 | CPU | 时间 | 加速比 |
| --- | --- | --- | --- |
| `std::transform()` | 0.02 毫秒 | 0.02 毫秒 | 0.25x |
| `par_transform_naive()` | 0.17 毫秒 | 0.08 毫秒 |

*CPU* 是在 CPU 上花费的总时间。*时间* 是挂钟时间，这是我们最感兴趣的。*加速比* 是比较顺序版本的经过时间和并行版本的相对加速比（在这种情况下为 0.02/0.08）。

显然，对于只有 50 个元素的小数据集，顺序版本的性能优于并行算法。但是，当有 10,000 个元素时，我们真的开始看到并行化的好处：

| 算法 | CPU | 时间 | 加速比 |
| --- | --- | --- | --- |
| `std::transform()` | 0.89 毫秒 | 0.89 毫秒 | 4.5x |
| `par_transform_naive()` | 1.95 毫秒 | 0.20 毫秒 |

最后，使用 1,000,000 个元素给我们带来了更高的效率，如下表所示：

| 算法 | CPU | 时间 | 加速比 |
| --- | --- | --- | --- |
| `std::transform()` | 9071 ms | 9092 ms | 7.3x |
| `par_transform_naive()` | 9782 ms | 1245 ms |

在这次运行中，并行算法的效率非常高。它在八个核心上执行，因此效率为 7.3x/8 = 0.925。这里呈现的结果（绝对执行时间和相对加速比）不应过分依赖。结果取决于计算机架构、操作系统调度程序以及在执行测试时当前机器上运行的其他工作量。尽管如此，基准测试结果证实了前面讨论的一些重要观点：

+   对于小数据集，由于创建线程等产生的开销，顺序版本`std::transform()`比并行版本快得多。

+   与`std::transform()`相比，并行版本总是使用更多的计算资源（CPU 时间）。

+   对于大数据集，当测量挂钟时间时，并行版本的性能优于顺序版本。在具有八个核心的机器上运行时，加速比超过 7 倍。

我们算法效率高的一个原因（至少对于大数据集来说）是计算成本均匀分布，每个子任务高度独立。然而，并非总是如此。

#### 朴素实现的缺点

如果每个工作块的计算成本相同，并且算法在没有其他应用程序利用硬件的环境中执行，那么朴素实现可能会做得很好。然而，这种情况很少发生；相反，我们希望有一个既高效又可扩展的通用并行实现。

以下插图显示了我们要避免的问题。如果每个块的计算成本不相等，实现将受限于花费最长时间的块：

![](img/B15619_14_02.png)

图 14.2：计算时间与块大小不成比例的可能场景

如果应用程序和/或操作系统有其他进程需要处理，操作将无法并行处理所有块：

![](img/B15619_14_03.png)

图 14.3：计算时间与块大小成比例的可能场景

如您在*图 14.3*中所见，将操作分割成更小的块使并行化适应当前条件，避免了使整个操作停滞的单个任务。

还要注意，对于小数据集，朴素实现是不成功的。有许多方法可以调整朴素实现以获得更好的性能。例如，我们可以通过将核心数乘以大于 1 的某个因子来创建更多任务和更小的任务。或者，为了避免在小数据集上产生显着的开销，我们可以让块大小决定要创建的任务数量等。

现在您已经知道如何实现和评估简单的并行算法。我们不会对朴素实现进行任何微调；相反，我将展示在实现并行算法时使用的另一种有用技术。

### 分而治之

将问题分解为较小子问题的算法技术称为**分而治之**。我们将在这里实现另一个使用分而治之的并行转换算法版本。它的工作原理如下：如果输入范围小于指定的阈值，则处理该范围；否则，将范围分成两部分：

+   第一部分在新分支的任务上处理

+   另一部分在调用线程中进行递归处理

以下插图显示了如何使用以下数据和参数递归地转换范围的分治算法：

+   范围大小：16

+   源范围包含从 1.0 到 16.0 的浮点数

+   块大小：4

+   转换函数：`[](auto x) { return x*x; }`

![](img/B15619_14_04.png)

图 14.4：一个范围被递归地分割以进行并行处理。源数组包含从 1.0 到 8.0 的浮点值。目标数组包含转换后的值。

在*图 14.4*中，您可以看到主任务生成了两个异步任务（**任务 1**和**任务 2**），最后转换了范围中的最后一个块。**任务 1**生成了**任务 3**，然后转换了包含值 5.0、6.0、7.0 和 8.0 的剩余元素。让我们来看看实现。

#### 实施

在实施方面，这是一小段代码。输入范围被递归地分成两个块；第一个块被调用为一个新任务，第二个块在同一个任务上被递归处理：

```cpp
template <typename SrcIt, typename DstIt, typename Func>
auto par_transform(SrcIt first, SrcIt last, DstIt dst,
                   Func func, size_t chunk_sz) {
  const auto n = static_cast<size_t>(std::distance(first, last));
  if (n <= chunk_sz) {
    std::transform(first, last, dst, func);
    return;
  }
  const auto src_middle = std::next(first, n / 2);
  // Branch of first part to another task
  auto future = std::async(std::launch::async, [=, &func] {
    par_transform(first, src_middle, dst, func, chunk_sz);
  });
  // Recursively handle the second part
  const auto dst_middle = std::next(dst, n / 2);
  par_transform(src_middle, last, dst_middle, func, chunk_sz);
  future.wait(); 
} 
```

将递归与多线程结合起来可能需要一段时间才能理解。在以下示例中，您将看到这种模式在实现更复杂的算法时可以使用。但首先，让我们看看它的性能如何。

#### 性能评估

为了评估我们的新版本，我们将通过更新 transform 函数来修改基准测试装置，使其根据输入值的不同需要更长的时间。通过使用`std::iota()`填充范围来增加输入值的范围。这样做意味着算法需要处理不同大小的作业。以下是新的`setup_fixture()`函数：

```cpp
auto setup_fixture(int n) {
  auto src = std::vector<float>(n);
  std::iota(src.begin(), src.end(), 1.0f);  // From 1.0 to n
  auto dst = std::vector<float>(src.size());
  auto transform_function = [](float v) { 
    auto sum = v;
    auto n = v / 20'000;                  // The larger v is, 
    for (auto i = 0; i < n; ++i) {        // the more to compute
      sum += (i * i * i * sum);
    }
    return sum;
  };
  return std::tuple{src, dst, transform_function};
} 
```

现在我们可以尝试通过使用递增的参数来找到分而治之算法的最佳块大小。看看我们的分而治之算法在这个新的装置上与朴素版本相比的表现，这需要处理不同大小的作业。以下是完整的代码：

```cpp
// Divide and conquer version
void bm_parallel(benchmark::State& state) {
  auto [src, dst, f] = setup_fixture(10'000'000);
  auto n = state.range(0);        // Chunk size is parameterized
  for (auto _ : state) {
    par_transform(src.begin(), src.end(), dst.begin(), f, n);
  }
}
// Naive version
void bm_parallel_naive(benchmark::State& state) {
  auto [src, dst, f] = setup_fixture(10'000'000);
  for (auto _ : state) {
    par_transform_naive(src.begin(), src.end(), dst.begin(), f);
  }
}
void CustomArguments(benchmark::internal::Benchmark* b) {
  b->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
}
BENCHMARK(bm_parallel)->Apply(CustomArguments)
  ->RangeMultiplier(10)           // Chunk size goes from 
  ->Range(1000, 10'000'000);      // 1k to 10M
BENCHMARK(bm_parallel_naive)->Apply(CustomArguments);
BENCHMARK_MAIN(); 
```

下图显示了我在 macOS 上运行测试时所获得的结果，使用了一个拥有八个核心的英特尔 Core i7 CPU：

![](img/B15619_14_05.png)

图 14.5：比较我们的朴素算法和使用不同块大小的分而治之算法

当使用大约 10,000 个元素的块大小时，可以实现最佳效率，这将创建 1,000 个任务。使用更大的块时，性能会在处理最终块所需的时间上受到瓶颈，而使用太小的块会导致在创建和调用任务方面产生过多的开销，与计算相比。

从这个例子中可以得出的结论是，调度 1,000 个较小的任务而不是几个大任务所带来的性能惩罚在这里并不是一个问题。可以通过使用线程池来限制线程的数量，但在这种情况下`std::async()`似乎运行得相当好。通用的实现会选择使用相当大数量的任务，而不是试图匹配确切的核心数量。

在实现并行算法时，找到最佳的块大小和任务数量是一个真正的问题。如您所见，这取决于许多变量，也取决于您是优化延迟还是吞吐量。获得洞察力的最佳方法是在您的算法应该运行的环境中进行测量。

现在您已经学会了如何使用分而治之来实现并行转换算法，让我们看看相同的技术如何应用到其他问题上。

## 实现并行 std::count_if()

分而治之的好处是它可以应用到许多问题上。我们可以很容易地使用相同的技术来实现`std::count_if()`的并行版本，唯一的区别是我们需要在函数末尾累加返回的值，就像这样：

```cpp
template <typename It, typename Pred> 
auto par_count_if(It first, It last, Pred pred, size_t chunk_sz) { 
  auto n = static_cast<size_t>(std::distance(first, last)); 
  if (n <= chunk_sz) 
    return std::count_if(first, last, pred);
  auto middle = std::next(first, n/2); 
  auto fut = std::async(std::launch::async, [=, &pred] { 
    return par_count_if(first, middle, pred, chunk_sz); 
  }); 
  auto num = par_count_if(middle, last, pred, chunk_sz); 
  return num + fut.get(); 
} 
```

如您所见，这里唯一的区别是我们需要在函数末尾对结果进行求和。如果您希望块大小取决于核心数量，您可以很容易地将`par_count_if()`包装在一个外部函数中：

```cpp
template <typename It, typename Pred> 
auto par_count_if(It first, It last, Pred pred) { 
  auto n = static_cast<size_t>(std::distance(first, last));
  auto n_cores = size_t{std::thread::hardware_concurrency()};
  auto chunk_sz = std::max(n / n_cores * 32, size_t{1000});

  return par_count_if(first, last, pred, chunk_sz);
} 
```

这里的神奇数字 32 是一个相当任意的因子，如果我们有一个大的输入范围，它将给我们更多的块和更小的块。通常情况下，我们需要测量性能来得出一个好的常数。现在让我们继续尝试解决一个更复杂的并行算法。

## 实现并行 std::copy_if()

我们已经研究了`std::transform()`和`std::count_if()`，它们在顺序和并行实现上都相当容易。如果我们再考虑另一个在顺序中容易实现的算法`std::copy_if()`，在并行中执行起来就会变得更加困难。

顺序地，实现`std::copy_if()`就像这样简单：

```cpp
template <typename SrcIt, typename DstIt, typename Pred> 
auto copy_if(SrcIt first, SrcIt last, DstIt dst, Pred pred) { 
  for (auto it = first; it != last; ++it) { 
    if (pred(*it)) { 
      *dst = *it; 
      ++dst;
    }
  }
  return dst;
} 
```

为了演示如何使用它，考虑以下示例，其中我们有一个包含整数序列的范围，我们想要将奇数整数复制到另一个范围中：

```cpp
const auto src = {1, 2, 3, 4}; 
auto dst = std::vector<int>(src.size(), -1); 
auto new_end = std::copy_if(src.begin(), src.end(), dst.begin(), 
                            [](int v) { return (v % 2) == 1; }); 
// dst is {1, 3, -1, -1}
dst.erase(new_end, dst.end()); // dst is now {1, 3} 
```

现在，如果我们想要制作`copy_if()`的并行版本，我们立即遇到问题，因为我们不能同时向目标迭代器写入。这是一个失败的尝试，具有未定义的行为，因为两个任务将同时写入目标范围中的相同位置：

```cpp
// Warning: Undefined behavior
template <typename SrcIt, typename DstIt, typename Func> 
auto par_copy_if(SrcIt first, SrcIt last, DstIt dst, Func func) { 
  auto n = std::distance(first, last);
  auto middle = std::next(first, n / 2); 
  auto fut0 = std::async([=]() { 
    return std::copy_if(first, middle, dst, func); }); 
  auto fut1 = std::async([=]() { 
    return std::copy_if(middle, last, dst, func); });
  auto dst0 = fut0.get();
  auto dst1 = fut1.get();
  return *std::max(dst0, dst1); // Just to return something...
} 
```

现在我们有了两种简单的方法：要么我们同步我们写入的索引（使用原子/无锁变量），要么我们将算法分成两部分。接下来我们将探索这两种方法。

### 方法 1：使用同步的写入位置

我们可能考虑的第一种方法是使用原子`size_t`和`fetch_add()`成员函数来同步写入位置，就像你在*第十一章* *并发*中学到的那样。每当一个线程尝试写入一个新元素时，它原子地获取当前索引并添加一个；因此，每个值都被写入到一个唯一的索引。

在我们的代码中，我们将算法分成两个函数：一个内部函数和一个外部函数。原子写入索引将在外部函数中定义，而算法的主要部分将在内部函数中实现。

#### 内部函数

内部函数需要一个同步写入位置的原子`size_t`。由于算法是递归的，它不能自己存储原子`size_t`；它需要一个外部函数来调用算法：

```cpp
template <typename SrcIt, typename DstIt, typename Pred>
void inner_par_copy_if_sync(SrcIt first, SrcIt last, DstIt dst,
                            std::atomic_size_t& dst_idx,
                            Pred pred, size_t chunk_sz) {
  const auto n = static_cast<size_t>(std::distance(first, last));
  if (n <= chunk_sz) {
    std::for_each(first, last, & {
      if (pred(v)) {
        auto write_idx = dst_idx.fetch_add(1);
        *std::next(dst, write_idx) = v;
      }
    });
    return;
  }
  auto middle = std::next(first, n / 2);
  auto future = std::async([first, middle, dst, chunk_sz, &pred, &dst_idx] {
    inner_par_copy_if_sync(first, middle, dst, dst_idx, pred, chunk_sz);
  });
  inner_par_copy_if_sync(middle, last, dst, dst_idx, pred, chunk_sz);
  future.wait();
} 
```

这仍然是一个分而治之的算法，希望你现在开始看到我们正在使用的模式。写入索引`dst_idx`的原子更新确保多个线程永远不会写入相同的目标序列中的索引。

#### 外部函数

从客户端代码调用的外部函数只是原子`size_t`的占位符，它被初始化为零。然后函数初始化内部函数，进一步并行化代码：

```cpp
template <typename SrcIt, typename DstIt, typename Pred>
auto par_copy_if_sync(SrcIt first,SrcIt last,DstIt dst,
                      Pred p, size_t chunk_sz) {
  auto dst_write_idx = std::atomic_size_t{0};
  inner_par_copy_if_sync(first, last, dst, dst_write_idx, p, chunk_sz);
  return std::next(dst, dst_write_idx);
} 
```

内部函数返回后，我们可以使用`dst_write_idx`来计算目标范围的结束迭代器。现在让我们来看看解决相同问题的另一种方法。

### 方法 2：将算法分成两部分

第二种方法是将算法分成两部分。首先，在并行块中执行条件复制，然后将结果稀疏范围压缩为连续范围。

#### 第一部分 - 并行复制元素到目标范围

第一部分将元素分块复制，得到了在*图 14.6*中说明的稀疏目标数组。每个块都是以并行方式有条件地复制的，结果范围迭代器存储在`std::future`对象中以供以后检索：

![](img/B15619_14_06.png)

图 14.6：第一步条件复制后的稀疏目标范围

以下代码实现了算法的前半部分：

```cpp
template <typename SrcIt, typename DstIt, typename Pred>
auto par_copy_if_split(SrcIt first, SrcIt last, DstIt dst, 
                       Pred pred, size_t chunk_sz) -> DstIt {
  auto n = static_cast<size_t>(std::distance(first, last));
  auto futures = std::vector<std::future<std::pair<DstIt, DstIt>>>{};
  futures.reserve(n / chunk_sz);
  for (auto i = size_t{0}; i < n; i += chunk_sz) {
    const auto stop_idx = std::min(i + chunk_sz, n);
    auto future = std::async([=, &pred] {
      auto dst_first = dst + i;
      auto dst_last = std::copy_if(first+i, first+stop_idx,                                   dst_first, pred);
      return std::make_pair(dst_first, dst_last);
    });
    futures.emplace_back(std::move(future));
  }
  // To be continued ... 
```

现在我们已经将（应该被复制的）元素复制到了稀疏的目标范围中。现在是时候通过将元素向左移动到范围中来填补空白了。

#### 第二部分 - 将稀疏范围顺序地移动到连续范围

当创建稀疏范围时，它使用每个`std::future`的结果值进行合并。合并是顺序执行的，因为部分重叠：

```cpp
 // ...continued from above... 
  // Part #2: Perform merge of resulting sparse range sequentially 
  auto new_end = futures.front().get().second; 
  for (auto it = std::next(futures.begin()); it != futures.end(); ++it)  { 
    auto chunk_rng = it->get(); 
    new_end = std::move(chunk_rng.first, chunk_rng.second, new_end);
  } 
  return new_end; 
} // end of par_copy_if_split 
```

将所有子范围移动到范围开始的算法的第二部分如下图所示：

![](img/B15619_14_07.png)

图 14.7：将稀疏范围合并到连续范围中

有两个解决同一个问题的算法，现在是时候看看它们的表现如何了。

### 性能评估

使用这个并行化版本的`copy_if()`的性能提升严重依赖于谓词的昂贵程度。因此，在我们的基准测试中，我们使用了两个不同计算成本的谓词。这是*廉价*的谓词：

```cpp
auto is_odd = [](unsigned v) { 
  return (v % 2) == 1; 
}; 
```

更*昂贵*的谓词检查其参数是否为质数：

```cpp
auto is_prime = [](unsigned v) {
  if (v < 2) return false;
  if (v == 2) return true;
  if (v % 2 == 0) return false;
  for (auto i = 3u; (i * i) <= v; i+=2) {
    if ((v % i) == 0) {
      return false; 
     }
  }
  return true;
}; 
```

请注意，这不是实现`is_prime()`的特别优化的方式，仅仅是为了基准测试的目的而使用。

基准测试代码没有在这里详细说明，但包含在附带的源代码中。比较了三个算法：`std::copy_if()`、`par_copy_if_split()`和`par_copy_if_sync()`。下图显示了在使用英特尔 Core i7 CPU 进行测量时的结果。这个基准测试中，并行算法使用了一个大小为 100,000 的块。

![](img/B15619_14_08.png)

图 14.8：条件复制策略与计算时间

在测量性能时最明显的观察是，当使用廉价的`is_odd()`谓词时，同步版本`par_copy_if_sync()`的性能是多么慢。灾难性的性能实际上并不是由于原子写入索引，而是因为硬件的缓存机制由于多个线程写入同一缓存行而被破坏（正如你在*第七章*，*内存管理*中学到的）。

因此，有了这个知识，我们现在明白了为什么`par_copy_if_split()`的性能更好。在廉价的谓词`is_odd()`上，`par_copy_if_split()`比`std::copy_if()`快大约 2 倍，但在昂贵的`is_prime()`上，效率增加到了近 5 倍。增加的效率是由于大部分计算在算法的第一部分中并行执行。

现在你应该掌握了一些用于并行化算法的技术。这些新的见解将帮助你理解使用标准库中的并行算法时的要求和期望。

# 并行标准库算法

从 C++17 开始，标准库已经扩展了大多数算法的并行版本，但并非所有算法都有。将你的算法更改为允许并行执行只是添加一个参数，告诉算法使用哪个并行执行策略。

本书早些时候强调过，如果你的代码基于标准库算法，或者至少习惯于使用算法编写 C++，那么通过在适当的地方添加执行策略，你几乎可以免费获得即时的性能提升：

```cpp
auto v = std::vector<std::string>{ 
  "woody", "steely", "loopy", "upside_down" 
};
// Parallel sort
std::sort(std::execution::par, v.begin(), v.end()); 
```

一旦指定了执行策略，你就进入了并行算法的领域，这些算法与它们原始的顺序版本有一些显著的区别。首先，最小的迭代器类别要求从输入迭代器变为前向迭代器。其次，你的代码抛出的异常（从复制构造函数或传递给算法的函数对象）永远不会到达你。相反，算法要求调用`std::terminate()`。第三，由于并行实现的复杂性增加，算法的复杂性保证（时间和内存）可能会放宽。

在使用标准库算法的并行版本时，你需要指定一个执行策略，该策略规定了算法允许并行执行的方式。但是，实现可能会决定按顺序执行算法。如果你比较不同标准库实现中并行算法的效率和可伸缩性，你会发现巨大的差异。

## 执行策略

**执行策略**通知算法执行是否可以并行化以及如何并行化。标准库的并行扩展中包括四种默认执行策略。编译器和第三方库可以为特定的硬件和条件扩展这些策略。例如，已经可以使用特定供应商的策略从标准库算法中使用现代图形卡的并行能力。

执行策略在头文件`<execution>`中定义，并驻留在命名空间`std::execution`中。目前有四种不同的标签类型，每种执行策略对应一种。这些类型不能由您实例化；相反，每种类型有一个预定义对象。例如，并行执行策略有一个名为`std::execution::parallel_policy`的类型，该类型的预定义实例名为`std::execution::par`。每种策略有一个*类型*（而不是具有多个预定义实例的一个类型）的原因是，您提供的策略可以在库中在编译时区分。

### 顺序策略

顺序执行策略`std::execution::seq`使算法以顺序方式执行，没有并行性，类似于没有额外执行策略参数的算法将运行的方式。然而，每当您指定执行策略时，这意味着您正在使用具有放宽的复杂性保证和更严格的迭代器要求的算法的版本；它还假定您提供的代码不会抛出异常，否则算法将调用`std::terminate()`。

### 并行策略

并行执行策略`std::execution::par`可以被认为是并行算法的标准执行策略。您提供给算法的代码需要是线程安全的。理解这一要求的一种方法是考虑您将要使用的算法的顺序版本中的循环主体。例如，考虑我们在本章前面这样拼写出来的`copy_if()`的顺序版本：

```cpp
template <typename SrcIt, typename DstIt, typename Pred> 
auto copy_if(SrcIt first, SrcIt last, DstIt dst, Pred pred) { 
  for (auto it = first; it != last; ++it) 
  {                            // Start of loop body
    if (pred(*it)) {           // Call predicate
      *dst = *it;              // Copy construct 
      ++dst;
    }
  }                            // End of loop body 
  return dst;
} 
```

在这个算法中，循环主体内的代码将调用您提供的谓词，并在范围内的元素上调用复制赋值运算符。如果您将`std::execution::par`传递给`copy_if()`，则您有责任保证这些部分是线程安全的，并且可以安全地并行执行。

让我们看一个例子，我们提供不安全的代码，然后看看我们能做些什么。假设我们有一个字符串向量：

```cpp
auto v = std::vector<std::string>{"Ada", "APL" /* ... */ }; 
```

如果我们想要使用并行算法计算向量中所有字符串的总大小，一个不足的方法是使用`std::for_each()`，就像这样：

```cpp
auto tot_size = size_t{0};
std::for_each(std::execution::par, v.begin(), v.end(),
              & { 
  tot_size += s.size(); // Undefined behavior, data race!
}); 
```

由于函数对象的主体不是线程安全的（因为它从多个线程更新共享变量），这段代码表现出未定义的行为。当然，我们可以使用`std::mutex`保护`tot_size`变量，但这将破坏以并行方式执行此代码的整个目的，因为互斥锁只允许一个线程一次进入主体。使用`std::atomic`数据类型是另一种选择，但这也可能降低效率。

这里的解决方案是*不*使用`std::for_each()`来解决这个问题。相反，我们可以使用`std::transform_reduce()`或`std::reduce()`，这些都是专门为这种工作量身定做的。以下是使用`std::reduce()`的方法：

```cpp
auto tot_size = std::reduce(std::execution::par, v.begin(), v.end(),                             size_t{0}, [](auto i, const auto& s) { 
  return i + s.size();   // OK! Thread safe
}); 
```

通过消除 lambda 内部的可变引用，lambda 的主体现在是线程安全的。对`std::string`对象的`const`引用是可以的，因为它从不改变任何字符串对象，因此不会引入任何数据竞争。

通常，您传递给算法的代码是线程安全的，除非您的函数对象通过引用捕获对象或具有其他诸如写入文件的副作用。

### 非顺序策略

无序策略是在 C++20 中添加的。它告诉算法，循环允许使用例如 SIMD 指令进行矢量化。实际上，这意味着您不能在传递给算法的代码中使用任何同步原语，因为这可能导致死锁。

为了理解死锁是如何发生的，我们将回到之前不足的例子，当计算向量中所有字符串的总大小时。假设，我们不是使用`std::reduce()`，而是通过添加互斥锁来保护`tot_size`变量，像这样：

```cpp
auto v = std::vector<std::string>{"Ada", "APL" /* ... */ };
auto tot_size = size_t{0};
auto mut = std::mutex{};
std::for_each(std::execution::par, v.begin(), v.end(),
              & { 
    auto lock = std::scoped_lock{mut}; // Lock
    tot_size += s.size(); 
  }                                    // Unlock
); 
```

现在，使用`std::execution::par`执行此代码是安全的，但效率很低。如果我们将执行策略更改为`std::execution::unseq`，结果不仅是一个低效的程序，还是一个有死锁风险的程序！

无序执行策略告诉算法，它可以重新排序我们的代码的指令，这通常是优化编译器不允许的。

为了使算法受益于矢量化，它需要从输入范围中读取多个值，然后一次应用 SIMD 指令于多个值。让我们分析一下`for_each()`循环中的两次迭代可能是什么样子，有无重新排序。以下是两次迭代没有任何重新排序的情况：

```cpp
{ // Iteration 1
  const auto& s = *it++;
  mut.lock();
  tot_size += s.size();
  mut.unlock();
}
{ // Iteration 2
  const auto& s = *it++;
  mut.lock();
  tot_size += s.size();
  mut.unlock();
} 
```

算法允许以以下方式合并这两次迭代：

```cpp
{ // Iteration 1 & 2 merged
  const auto& s1 = *it++;
  const auto& s2 = *it++;
  mut.lock();
  mut.lock();                // Deadlock!
  tot_size += s1.size();     // Replace these operations
  tot_size += s2.size();     // with vectorized instructions
  mut.unlock();
  mut.unlock();
} 
```

尝试在同一线程上执行此代码将导致死锁，因为我们试图连续两次锁定同一个互斥锁。换句话说，当使用`std::execution::unseq`策略时，您必须确保您提供给算法的代码不会获取任何锁。

请注意，优化编译器随时可以对您的代码进行矢量化。然而，在这些情况下，由编译器来保证矢量化不会改变程序的含义，就像编译器和硬件允许执行的任何其他优化一样。在这里，当显式地为算法提供`std::execute::unseq`策略时，*您*保证您提供的代码是安全的可矢量化的。

### 并行无序策略

并行无序策略`std::execution::par_unseq`像并行策略一样并行执行算法，另外还可以对循环进行矢量化。

除了四种标准执行策略之外，标准库供应商可以为您提供具有自定义行为的其他策略，并对输入施加其他约束。例如，英特尔并行 STL 库定义了四种只接受随机访问迭代器的自定义执行策略。

## 异常处理

如果您为算法提供了四种标准执行策略中的一种，您的代码不能抛出异常，否则算法将调用`std::terminate()`。这与正常的单线程算法有很大的不同，后者总是将异常传播回调用者：

```cpp
auto v = {1, 2, 3, 4};
auto f = [](auto) { throw std::exception{}; };
try {
  std::for_each(v.begin(), v.end(), f);
} catch (...) {
  std::cout << "Exception caught\n";
} 
```

使用执行策略运行相同的代码会导致调用`std::terminate()`：

```cpp
try {
  std::for_each(std::execution::seq, v.begin(), v.end(), f);
} catch (...) {
  // The thrown std:::exception never reaches us.
  // Instead, std::terminate() has been called 
} 
```

您可能认为这意味着并行算法声明为`noexcept`，但事实并非如此。许多并行算法需要分配内存，因此标准并行算法本身可以抛出`std::bad_alloc`。

还应该说，其他库提供的执行策略可能以不同的方式处理异常。

现在，我们将继续讨论在 C++17 中首次引入并行算法时添加和修改的一些算法。

## 并行算法的添加和更改

标准库中的大多数算法都可以直接作为并行版本使用。但是，也有一些值得注意的例外，包括`std::accumulate()`和`std::for_each()`，因为它们的原始规范要求按顺序执行。

### std::accumulate()和 std::reduce()

`std::accumulate()`算法不能并行化，因为它必须按元素的顺序执行，这是不可能并行化的。相反，已添加了一个新算法叫做`std::reduce()`，它的工作方式与`std::accumulate()`相同，只是它是无序执行的。

对于可交换的操作，它们的结果是相同的，因为累积的顺序无关紧要。换句话说，给定一个整数范围：

```cpp
const auto r = {1, 2, 3, 4}; 
```

通过加法或乘法累积它们：

```cpp
auto sum = 
  std::accumulate(r.begin(), r.end(), 0, std::plus<int>{});

auto product = 
  std::accumulate(r.begin(), r.end(), 1, std::multiplies<int>{}); 
```

将产生与调用`std::reduce()`而不是`std::accumulate()`相同的结果，因为整数的加法和乘法都是可交换的。例如：

![](img/B15619_14_004.png)

但是，如果操作不是可交换的，结果是*不确定的*，因为它取决于参数的顺序。例如，如果我们要按如下方式累积字符串列表：

```cpp
auto v = std::vector<std::string>{"A", "B", "C"};
auto acc = std::accumulate(v.begin(), v.end(), std::string{});
std::cout << acc << '\n'; // Prints "ABC" 
```

这段代码将始终产生字符串`"ABC"`。但是，通过使用`std::reduce()`，结果字符串中的字符可能以任何顺序出现，因为字符串连接不是可交换的。换句话说，字符串`"A" + "B"`不等于`"B" + "A"`。因此，使用`std::reduce()`的以下代码可能产生不同的结果：

```cpp
auto red = std::reduce(v.begin(), v.end(), std::string{}); 
std::cout << red << '\n'; 
// Possible output: "CBA" or "ACB" etc 
```

与性能相关的一个有趣点是浮点数运算不是可交换的。通过在浮点值上使用`std::reduce()`，结果可能会有所不同，但这也意味着`std::reduce()`可能比`std::accumulate()`快得多。这是因为`std::reduce()`允许重新排序操作并利用 SIMD 指令，而在使用严格的浮点数运算时，`std::accumulate()`是不允许这样做的。

#### std::transform_reduce()

作为标准库算法的补充，`std::transform_reduce()`也已添加到`<numeric>`头文件中。它确切地做了它所说的：它将一个元素范围转换为`std::transform()`，然后应用一个函数对象。这样累积它们是无序的，就像`std::reduce()`一样：

```cpp
auto v = std::vector<std::string>{"Ada","Bash","C++"}; 
auto num_chars = std::transform_reduce( 
  v.begin(), v.end(), size_t{0}, 
  [](size_t a, size_t b) { return a + b; },     // Reduce
  [](const std::string& s) { return s.size(); } // Transform 
); 
// num_chars is 10 
```

当并行算法被引入时，`std::reduce()`和`std::transform_reduce()`也被添加到 C++17 中。另一个必要的更改是调整`std::for_each()`的返回类型。

### `std::for_each()`

`std::for_each()`的一个相对不常用的特性是它返回传递给它的函数对象。这使得可以使用`std::for_each()`在有状态的函数对象内累积值。以下示例演示了可能的用例：

```cpp
struct Func {
  void operator()(const std::string& s) {
    res_ += s;
  };
  std::string res_{};    // State
};
auto v = std::vector<std::string>{"A", "B", "C"};
auto s = std::for_each(v.begin(), v.end(), Func{}).res_;
// s is "ABC" 
```

这种用法类似于使用`std::accumulate()`可以实现的用法，因此在尝试并行化时也会出现相同的问题：无序执行函数对象将产生不确定的结果，因为调用顺序是未定义的。因此，`std::for_each()`的并行版本简单地返回`void`。

## 基于索引的 for 循环的并行化

尽管我建议使用算法，但有时特定任务需要原始的基于索引的`for`循环。标准库算法通过在库中包含算法`std::for_each()`提供了等效于基于范围的`for`循环。

然而，并没有基于索引的`for`循环的等效算法。换句话说，我们不能简单地通过向其添加并行策略来轻松并行化这样的代码：

```cpp
auto v = std::vector<std::string>{"A", "B", "C"};
for (auto i = 0u; i < v.size(); ++i) { 
  v[i] += std::to_string(i+1); 
} 
// v is now { "A1", "B2", "C3" } 
```

但让我们看看如何通过组合算法来构建一个。正如您已经得出的结论，实现并行算法是复杂的。但在这种情况下，我们将使用`std::for_each()`作为构建块构建一个`parallel_for()`算法，从而将复杂的并行性留给`std::for_each()`。

### 结合 std::for_each()和 std::views::iota()

基于标准库算法的基于索引的`for`循环可以通过将标准库算法`std::for_each()`与范围库中的`std::views::iota()`结合使用来创建（见*第六章*，*范围和视图*）。它看起来是这样的：

```cpp
auto v = std::vector<std::string>{"A", "B", "C"};
auto r = std::views::iota(size_t{0}, v.size()); 
std::for_each(r.begin(), r.end(), &v { 
  v[i] += std::to_string(i + 1); 
}); 
// v is now { "A1", "B2", "C3" } 
```

然后可以通过使用并行执行策略进一步并行化：

```cpp
std::for_each(std::execution::par, r.begin(), r.end(), &v { 
  v[i] += std::to_string(i + 1); 
}); 
```

正如前面所述，我们在像这样从多个线程调用的 lambda 中传递引用时必须非常小心。通过仅通过唯一索引`i`访问向量元素，我们避免了在向量中突变字符串时引入数据竞争。

### 通过包装简化构造

为了以简洁的语法迭代索引，先前的代码被封装到一个名为`parallel_for()`的实用函数中，如下所示：

```cpp
template <typename Policy, typename Index, typename F>
auto parallel_for(Policy&& p, Index first, Index last, F f) {
  auto r = std::views::iota(first, last);
  std::for_each(p, r.begin(), r.end(), std::move(f));
} 
```

然后可以直接使用`parallel_for()`函数模板，如下所示：

```cpp
auto v = std::vector<std::string>{"A", "B", "C"};
parallel_for(std::execution::par, size_t{0}, v.size(),
              & { v[i] += std::to_string(i + 1); }); 
```

由于`parallel_for()`是建立在`std::for_each()`之上的，它接受`std::for_each()`接受的任何策略。

我们将用一个简短的介绍性概述来总结本章，介绍 GPU 以及它们如何在现在和将来用于并行编程。

# 在 GPU 上执行算法

**图形** **处理单元**（**GPU**）最初是为了处理计算机图形渲染中的点和像素而设计和使用的。简而言之，GPU 所做的是检索像素数据或顶点数据的缓冲区，对每个缓冲区进行简单操作，并将结果存储在新的缓冲区中（最终将被显示）。

以下是一些可以在早期阶段在 GPU 上执行的简单独立操作的示例：

+   将点从世界坐标转换为屏幕坐标

+   在特定点执行光照计算（通过光照计算，我指的是计算图像中特定像素的颜色）

由于这些操作可以并行执行，GPU 被设计用于并行执行小操作。后来，这些图形操作变得可编程，尽管程序是以计算机图形的术语编写的（也就是说，内存读取是以从纹理中读取颜色的形式进行的，结果总是以颜色写入纹理）。这些程序被称为**着色器**。

随着时间的推移，引入了更多的着色器类型程序，着色器获得了越来越多的低级选项，例如从缓冲区中读取和写入原始值，而不是从纹理中读取颜色值。

从技术上讲，CPU 通常由几个通用缓存核心组成，而 GPU 由大量高度专门化的核心组成。这意味着良好扩展的并行算法非常适合在 GPU 上执行。

GPU 有自己的内存，在算法可以在 GPU 上执行之前，CPU 需要在 GPU 内存中分配内存，并将数据从主内存复制到 GPU 内存。接下来发生的事情是 CPU 在 GPU 上启动例程（也称为内核）。最后，CPU 将数据从 GPU 内存复制回主内存，使其可以被在 CPU 上执行的“正常”代码访问。在 CPU 和 GPU 之间来回复制数据所产生的开销是 GPU 更适合批处理任务的原因之一，其中吞吐量比延迟更重要。

今天有几个库和抽象层可用，使得从 C++进行 GPU 编程变得容易。然而，标准 C++在这方面几乎没有提供任何东西。但是，并行执行策略`std::execution::par`和`std::execution::par_unseq`允许编译器将标准算法的执行从 CPU 移动到 GPU。其中一个例子是 NVC++，NVIDIA HPC 编译器。它可以配置为将标准 C++算法编译为在 NVIDIA GPU 上执行。

如果您想了解 C++和 GPU 编程的当前状态，我强烈推荐 Michael Wong 在 ACCU 2019 年会议上的演讲*使用现代 C++进行 GPU 编程*（[`accu.org/video/spring-2019-day-3/wong/`](https://accu.org/video/spring-2019-day-3/wong/)）。

## 总结

在本章中，您已经了解了手工编写并行算法的复杂性。您现在也知道如何分析、测量和调整并行算法的效率。在学习并行算法时获得的见解将加深您对 C++标准库中并行算法的要求和行为的理解。C++带有四种标准执行策略，可以由编译器供应商进行扩展。这为利用 GPU 执行标准算法打开了大门。下一个 C++标准，C++23，很可能会增加对 GPU 并行编程的支持。

您现在已经到达了本书的结尾。恭喜！性能是代码质量的重要方面。但往往性能是以牺牲其他质量方面（如可读性、可维护性和正确性）为代价的。掌握编写高效和干净代码的艺术需要实际训练。我希望您从本书中学到了一些东西，可以将其融入到您的日常生活中，创造出令人惊叹的软件。

解决性能问题通常需要愿意进一步调查事情。往往需要足够了解硬件和底层操作系统，以便能够从测量数据中得出结论。在这本书中，我在这些领域只是浅尝辄止。在第二版中写了关于 C++20 特性之后，我现在期待着开始在我的职业作为软件开发人员中使用这些特性。正如我之前提到的，这本书中呈现的许多代码今天只有部分得到编译器的支持。我将继续更新 GitHub 存储库，并添加有关编译器支持的信息。祝你好运！

**分享您的经验**

感谢您抽出时间阅读本书。如果您喜欢这本书，帮助其他人找到它。在[`www.amazon.com/dp/1839216549`](https://www.amazon.com/dp/1839216549)留下评论。
