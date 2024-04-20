# 第十章：性能优化

性能是选择 C++作为项目编程语言的关键驱动因素之一。现在是讨论如何在以函数式风格构建代码时改善性能的时候了。

虽然性能是一个庞大的主题，显然我们无法在一个章节中完全覆盖，但我们将探讨改善性能的关键思想，纯函数式语言如何优化性能，以及如何将这些优化转化为 C++。

本章将涵盖以下主题：

+   交付性能的流程

+   如何使用并行/异步来提高性能

+   理解什么是尾递归以及如何激活它

+   如何在使用函数式构造时改善内存消耗

+   功能性异步代码

# 技术要求

您需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.3.0。

代码可以在 GitHub 上找到，位于[https:/​/​github.​com/​PacktPublishing/​Hands-​On-​Functional-Programming-​with-​Cpp](https://github.%E2%80%8Bcom/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)的`Chapter10`文件夹中。它包括并使用`doctest`，这是一个单头文件的开源单元测试库。您可以在其 GitHub 存储库上找到它，网址为[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# 性能优化

谈论性能优化就像谈论披萨。有些人喜欢和寻找菠萝披萨。其他人只吃传统的意大利披萨（或来自特定地区的披萨）。有些人只吃素食披萨，而其他人喜欢各种披萨。关键是，性能优化是与您的代码库和产品相关的。您正在寻找什么样的性能？对于您的用户来说，性能的最有价值的部分是什么？您需要考虑哪些约束？

我与合作的客户通常有一些性能要求，取决于主题：

+   *嵌入式产品*（例如汽车、能源或电信）通常需要在内存限制内工作。堆栈和堆通常很小，因此限制了长期存在的变量数量。增加内存的成本可能是禁止性的（一位客户告诉我们，他们需要超过 1000 万欧元才能在所有设备上增加 1MB 的额外内存）。因此，程序员需要通过尽可能避免不必要的内存分配来解决这些限制。这可能包括初始化、通过复制传递参数（特别是较大的结构）以及避免需要内存消耗的特定算法，等等。

+   *工程应用*（例如计算机辅助设计或 CAD）需要在非常大的数据集上使用从数学、物理和工程中衍生出的特定算法，并尽快返回结果。处理通常在现代 PC 上进行，因此 RAM 不是问题；然而，CPU 是问题。随着多核 CPU 的出现，专用 GPU 可以接管部分处理工作以及允许在多个强大或专用服务器之间分配工作负载的云技术的出现，开发人员的工作往往变成了在并行和异步世界中优化速度。

+   *桌面游戏和游戏引擎*有它们自己特别的关注点。图形必须尽可能好看，以便在中低端机器上优雅地缩放，并避免延迟。游戏通常会占据它们运行的机器，因此它们只需要与操作系统和系统应用程序（如防病毒软件或防火墙）争夺资源。它们还可以假定特定级别的 GPU、CPU 和 RAM。优化变得关于并行性（因为预期有多个核心）以及避免浪费，以保持整个游戏过程中的流畅体验。

+   *游戏服务器*，然而，是一个不同的问题。例如暴雪的战网（我作为*星际争霸 II*玩家经常使用的一个）需要快速响应，即使在压力下也是如此。在云计算时代，使用的服务器数量和性能并不重要；我们可以轻松地扩展或缩减它们。主要问题是尽可能快地响应大多数 I/O 工作负载。

+   *未来令人兴奋*。游戏的趋势是将处理移动到服务器，从而使玩家甚至可以在低端机器上玩游戏。这将为未来的游戏开辟令人惊人的机会。（如果你有 10 个 GPU，你能做什么？如果有 100 个呢？）但也将导致需要优化游戏引擎以进行服务器端、多机器、并行处理。远离游戏，物联网行业为嵌入式软件和可扩展的服务器端处理提供了更多机会。

考虑到所有这些可能性，我们可以在代码库中做些什么来提供性能？

# 提供性能的流程

正如您所看到的，性能优化在很大程度上取决于您要实现的目标。下一步可以快速总结如下：

1.  为性能设定明确的目标，包括指标和如何测量它们。

1.  为性能定义一些编码准则。保持它们清晰并针对代码的特定部分进行调整。

1.  使代码工作。

1.  在需要的地方测量和改进性能。

1.  监控和改进。

在我们更详细地了解这些步骤之前，重要的是要理解性能优化的一个重要警告——有两种优化类型。第一种来自清晰的设计和清晰的代码。例如，通过从代码中删除某些相似性，您可能会减少可执行文件的大小，从而为数据提供更多空间；数据可能会通过代码传输得更少，从而避免不必要的复制或间接；或者，它将允许编译器更好地理解代码并为您进行优化。根据我的经验，将代码重构为简单设计也经常提高了性能。

改进性能的第二种方法是使用点优化。这些是非常具体的方式，我们可以重写函数或流程，使代码能够更快地工作或消耗更少的内存，通常适用于特定的编译器和平台。结果代码通常看起来很聪明，但很难理解和更改。

点优化与编写易于更改和维护的代码存在天然冲突。这导致了唐纳德·克努斯说*过早优化是万恶之源*。这并不意味着我们应该编写明显缓慢的代码，比如通过复制大型集合。然而，这意味着我们应该首先优化设计以便更易更改，然后测量性能，然后优化它，并且只在绝对必要时使用点优化。平台的怪癖、特定的编译器版本或使用的库可能需要不时进行点优化；将它们分开并节制使用。

现在让我们来看看我们的性能优化流程。

# 为性能设定明确的目标，包括指标和如何测量它们

如果我们不知道我们要去哪里，那么我们去哪个方向都无所谓——我是从《爱丽丝梦游仙境》中引用的。因此，我们应该知道我们要去哪里。我们需要一个适合我们产品需求的性能指标列表。此外，对于每个性能指标，我们需要一个定义该指标的*好*值和*可接受*值的范围。让我们看几个例子。

如果您正在为具有 4MB 内存的设备构建*嵌入式产品*，您可能会关注诸如：

+   内存消耗：

+   很好：1-3 MB

+   好：3-4 MB

+   设备启动时间：

+   很好：<1 秒

+   好：1-3 秒

如果你正在构建一个*桌面 CAD 应用程序*，用于模拟建筑设计中的声波，其他指标也很有趣。

模拟声波建模的计算时间：

+   对于一个小房间：

+   很好：<1 分钟

+   好：<5 分钟

+   对于一个中等大小的房间：

+   很好：<2 分钟

+   好：<10 分钟

这里的数字仅供参考；你需要为你的产品找到自己的度量标准。

有了这些度量标准和好/很好的范围，我们可以在添加新功能后测量性能并进行相应的优化。它还可以让我们向利益相关者或业务人员简单地解释产品的性能。

# 为性能定义一些编码准则-保持清晰，并针对代码的特定部分进行定制

如果你问 50 个不同的 C++程序员关于优化性能的建议，你很快就会被淹没在建议中。如果你开始调查这些建议，结果会发现其中一些已经过时，一些非常具体，一些很好。

因此，对性能有编码准则是很重要的，但有一个警告。C++代码库往往很庞大，因为它们已经发展了很多年。如果你对你的代码库进行批判性审视，你会意识到只有一部分代码是性能瓶颈。举个例子，如果一个数学运算快了 1 毫秒，只有当这个运算会被多次调用时才有意义；如果它只被调用一两次，或者很少被调用，就没有必要进行优化。事实上，下一个版本的编译器或 CPU 可能会比你更擅长优化它。

由于这个事实，你应该了解你的代码的哪些部分对你定义的性能标准至关重要。找出哪种设计最适合这个特定的代码片段；制定清晰的准则，并遵循它们。虽然`const&`在任何地方都很有用，也许你可以避免浪费开发人员的时间对一个只做一次的非常小的集合进行排序。

# 让代码工作

牢记这些准则，并有一个新功能要实现，第一步应该始终是让代码工作。此外，结构化使其易于在你的约束条件内进行更改。不要试图在这里优化性能；再次强调，编译器和 CPU 可能比你想象的更聪明，做的工作也比你期望的多。要知道是否是这种情况，唯一的办法就是测量性能。

# 在需要的地方测量和改进性能

你的代码可以按照你的准则工作和结构化，并且为变更进行了优化。现在是时候写下一些关于优化它的假设，然后进行测试了。

由于你对性能有明确的度量标准，验证它们相对容易。当然，这需要正确的基础设施和适当的测量过程。有了这些，你就可以测量你在性能指标上的表现。

在这里应该欢迎额外的假设。比如- *如果我们像这样重构这段代码，我期望指标 X 会有所改善*。然后你可以继续测试你的假设-开始一个分支，改变代码，构建产品，经过性能指标测量过程，看看结果。当然，实际情况可能比我说的更复杂-有时可能需要使用不同的编译器进行构建，使用不同的优化选项，或者统计数据。如果你想做出明智的决定，这些都是必要的。投入一些时间来进行度量，而不是改变代码并使其更难理解会更好。否则，你最终会得到一笔技术债务，你将长期支付利息。

然而，如果你必须进行点优化，没有变通的办法。只需确保尽可能详细地记录它们。因为你之前已经测试过你的假设，你会有很多东西要写，对吧？

# 监控和改进

我们通过定义性能指标来开始循环。现在是时候结束了，我们需要监控这些指标（可能还有其他指标），并根据我们所学到的知识调整我们的间隔和编码准则。性能优化是一个持续的过程，因为目标设备也在不断发展。

我们已经看过了交付性能的流程，但这与函数式编程有什么关系呢？哪些用例使函数式代码结构发光，哪些又效果不佳？现在是时候深入研究我们的代码结构了。

# 并行性-利用不可变性

编写并行运行的代码一直是软件开发中的一大痛点。多线程、多进程或多服务器环境带来的问题似乎根本难以解决。死锁、饥饿、数据竞争、锁或调试多线程代码等术语让我们这些见过它们的人害怕再次遇到它们。然而，由于多核 CPU、GPU 和多个服务器，我们不得不面对并行代码。函数式编程能帮助解决这个问题吗？

每个人都同意这是函数式编程的一个强项，特别是源自不可变性。如果你的数据从不改变，就不会有锁，同步也会变得非常简单并且可以泛化。如果你只使用纯函数和函数转换（当然除了 I/O），你几乎可以免费获得并行化。

事实上，C++ 17 标准包括 STL 高级函数的执行策略，允许我们通过一个参数将算法从顺序改为并行。让我们来检查向量中是否所有数字都大于`5`的并行执行。我们只需要将`execution::par`作为`all_of`的执行策略即可：

```cpp
auto aVector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
auto all_of_parallel = [&aVector](){
    return all_of(execution::par, aVector.begin(), aVector.end(),  
        [](auto value){return value > 5;});
};
```

然后，我们可以使用`chrono`命名空间的高分辨率计时器来衡量使用顺序和并行版本算法的差异，就像这样：

```cpp
auto measureExecutionTimeForF = [](auto f){
    auto t1 = high_resolution_clock::now();
    f();
    auto t2 = high_resolution_clock::now();
    chrono::nanoseconds duration = t2 - t1;
    return duration;
};
```

通常情况下，我现在会展示基于我的实验的执行差异。不幸的是，在这种情况下，我不能这样做。在撰写本文时，唯一实现执行策略的编译器是 MSVC 和英特尔 C++，但它们都不符合标准。然而，如下代码段所示，我在`parallelExecution.cpp`源文件中编写了代码，当你的编译器支持标准时，你可以通过取消注释一行来启用它：

```cpp
// At the time when I created this file, only MSVC had implementation  
    for execution policies.
// Since you're seeing this in the future, you can enable the parallel 
    execution code by uncommenting the following line 
//#define PARALLEL_ENABLED
```

当你运行这段代码时，它将显示顺序和并行运行`all_of`的比较持续时间，就像这样：

```cpp
TEST_CASE("all_of with sequential execution policy"){
    auto aVector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto all_of_sequential = [&aVector](){
        return all_of(execution::seq, aVector.begin(), aVector.end(), 
            [](auto value){return value > 5;});
    };

    auto sequentialDuration = 
        measureExecutionTimeForF(all_of_sequential);
        cout << "Execution time for sequential policy:" << 
            sequentialDuration.count() << " ns" << endl;

    auto all_of_parallel = [&aVector](){
        return all_of(execution::par, aVector.begin(), aVector.end(), 
            [](auto value){return value > 5;});
    };

    auto parallelDuration = measureExecutionTimeForF(all_of_parallel);
    cout << "Execution time for parallel policy:" <<   
        parallelDuration.count() << " ns" << endl;
}
```

虽然我很想在这里分析一些执行数据，但也许最好的是我不能，因为这一章最重要的信息是要衡量、衡量、衡量，然后再优化。希望在合适的时候你也能进行一些衡量。

C++ 17 标准支持许多 STL 函数的执行策略，包括`sort`、`find`、`copy`、`transform`和`reduce`。也就是说，如果你在这些函数上进行链式调用并使用纯函数，你只需要为所有调用传递一个额外的参数（或者将高级函数绑定），就可以实现并行执行！我敢说，对于那些尝试自己管理线程或调试奇怪同步问题的人来说，这几乎就像魔法一样。事实上，在前几章中我们为井字棋和扑克牌手写的所有代码都可以很容易地切换到并行执行，只要编译器支持完整的 C++ 17 标准。

但是这是如何工作的？对于`all_of`来说，运行在多个线程中是相当容易的；每个线程在集合中的特定元素上执行谓词，返回一个布尔值，并且当第一个谓词返回`False`时，进程停止。只有当谓词是纯函数时才可能发生这种情况；以任何方式修改结果或向量都会创建竞争条件。文档明确指出程序员有责任保持谓词函数的纯净性——不会有警告或编译错误。除了是纯函数外，你的谓词不能假设元素被处理的顺序。

如果并行执行策略无法启动（例如，由于资源不足），执行将回退到顺序调用。在测量性能时，这是一个需要记住的有用事情：如果性能远低于预期，请首先检查程序是否可以并行执行。

这个选项对于使用多个 CPU 的计算密集型应用程序非常有用。如果你对它的内存消耗感兴趣，你需要测量一下，因为它取决于你使用的编译器和标准库。

# 记忆化

纯函数具有一个有趣的特性。对于相同的输入值，它们返回相同的输出。这使它们等同于一个大表格的值，其中每个输入参数的组合都对应一个输出值。有时，记住这个表格的部分比进行计算更快。这种技术称为**记忆化**。

纯函数式编程语言以及诸如 Python 和 Groovy 之类的语言，都有办法在特定函数调用上启用记忆化，从而提供了高度的控制。不幸的是，C++没有这个功能，所以我们必须自己编写它。

# 实现记忆化

要开始我们的实现，我们需要一个函数；最好是计算昂贵的。让我们选择`power`函数。一个简单的实现只是标准`pow`函数的包装器，如下面的代码片段所示：

```cpp
function<long long(int, int)> power = [](auto base, auto exponent){
    return pow(base, exponent);
};
```

我们如何开始实现记忆化？嗯，在其核心，记忆化就是缓存。每当一个函数第一次被调用时，它会正常运行，但同时也将结果与输入值组合存储起来。在后续的调用中，函数将搜索映射以查看值是否被缓存，并在有缓存时返回它。

这意味着我们需要一个缓存，其键是参数，值是计算结果。为了将参数组合在一起，我们可以简单地使用一对或元组：

```cpp
tuple<int, int> parameters
```

因此，缓存将是：

```cpp
    map<tuple<int, int>, long long> cache;
```

让我们改变我们的`power`函数以使用这个缓存。首先，我们需要在缓存中查找结果：

```cpp
    function<long long(int, int)> memoizedPower = &cache{
            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);

```

如果没有找到任何东西，我们计算结果并将其存储在缓存中。如果找到了某些东西，那就是我们要返回的值：

```cpp
        if(valueIterator == cache.end()){
            result = pow(base, exponent);
            cache[parameters] = result;
        } else{
            result = valueIterator -> second;
        }
        return result; 
```

为了检查这种方法是否正常工作，让我们运行一些测试：

```cpp
    CHECK_EQ(power(1, 1), memoizedPower(1, 1));
    CHECK_EQ(power(3, 19), memoizedPower(3, 19));
    CHECK_EQ(power(2, 25), memoizedPower(2, 25));
```

一切都很顺利。现在让我们比较 power 的两个版本，在下面的代码片段中有和没有记忆化。下面的代码显示了我们如何提取一种更通用的方法来记忆化函数：

```cpp
    function<long long(int, int)> power = [](int base, int exponent){
        return pow(base, exponent);
    };

    map<tuple<int, int>, long long> cache;

    function<long long(int, int)> memoizedPower = &cache{
            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);
            long long result;
            if(valueIterator == cache.end()){
 result = pow(base, exponent);
            cache[parameters] = result;
        } else{
            result = valueIterator -> second;
        }
        return result; 
    };
```

第一个观察是我们可以用原始 power 函数的调用替换粗体行，所以让我们这样做：

```cpp
    function<long long(int, int)> memoizedPower = &cache, &power{
            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);
            long long result;
            if(valueIterator == cache.end()){
 result = power(base, exponent);
            cache[parameters] = result;
        } else{
            result = valueIterator -> second;
        }
        return result; 
    };
```

如果我们传入我们需要在记忆化期间调用的函数，我们将得到一个更通用的解决方案：

```cpp
    auto memoize = &cache{
            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);
            long long result;
            if(valueIterator == cache.end()){
            result = functionToMemoize(base, exponent);
            cache[parameters] = result;
        } else{
            result = valueIterator -> second;
        }
        return result; 
    };

    CHECK_EQ(power(1, 1), memoize(1, 1, power));
    CHECK_EQ(power(3, 19), memoize(3, 19, power));
    CHECK_EQ(power(2, 25), memoize(2, 25, power));
```

但是返回一个记忆化的函数不是很好吗？我们可以修改我们的`memoize`函数，使其接收一个函数并返回一个记忆化的函数，该函数接收与初始函数相同的参数：

```cpp
    auto memoize = [](auto functionToMemoize){
        map<tuple<int, int>, long long> cache;
 return & {
            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);
            long long result;
            if(valueIterator == cache.end()){
                result = functionToMemoize(base, exponent);
                cache[parameters] = result;
            } else{
                result = valueIterator -> second;
            }
            return result; 
            };
    };
    auto memoizedPower = memoize(power);
```

这个改变最初不起作用——我得到了一个分段错误。原因是我们在 lambda 内部改变了缓存。为了使它工作，我们需要使 lambda 可变，并按值捕获：

```cpp
    auto memoize = [](auto functionToMemoize){
        map<tuple<int, int>, long long> cache;
 return = mutable {
            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);
            long long result;
            if(valueIterator == cache.end()){
                result = functionToMemoize(base, exponent);
                cache[parameters] = result;
            } else{
                result = valueIterator -> second;
            }
            return result; 
            };
    };
```

现在我们有一个可以对任何带有两个整数参数的函数进行记忆化的函数。通过使用一些类型参数，很容易使它更通用。我们需要一个返回值的类型，第一个参数的类型和第二个参数的类型：

```cpp
template<typename ReturnType, typename FirstArgType, typename 
    SecondArgType>
auto memoizeTwoParams = [](function<ReturnType(FirstArgType, SecondArgType)> functionToMemoize){
    map<tuple<FirstArgType, SecondArgType>, ReturnType> cache;
    return = mutable {
        tuple<FirstArgType, SecondArgType> parameters(firstArg, 
    secondArg);
        auto valueIterator = cache.find(parameters);
        ReturnType result;
        if(valueIterator == cache.end()){
            result = functionToMemoize(firstArg, secondArg);
            cache[parameters] = result;
        } else{
            result = valueIterator -> second;
        }
        return result; 
    };
};
```

我们已经实现了一个对具有两个参数的任何函数进行记忆化的函数。我们可以做得更好。C++允许我们使用具有未指定数量类型参数的模板，即所谓的**可变参数模板**。通过使用它们的魔力，我们最终得到了一个可以处理任何数量参数的函数的记忆化实现：

```cpp
template<typename ReturnType, typename... Args>
function<ReturnType(Args...)> memoize(function<ReturnType(Args...)> f){
    map<tuple<Args...>, ReturnType> cache;
    return (= mutable  {
            tuple<Args...> theArguments(args...);
            auto cached = cache.find(theArguments);
            if(cached != cache.end()) return cached -> second;
            auto result = f(args...);
            cache[theArguments] = result;
            return result;
    });
};
```

这个函数对缓存任何其他函数都有帮助；然而，有一个问题。到目前为止，我们使用了`power`的包装实现。以下是一个示例，如果我们自己编写它会是什么样子：

```cpp
function<long long(int, int)> power = & 
{
    return (exponent == 0) ? 1 : base * power(base, exponent - 1);
};
```

对这个函数进行记忆化只会缓存最终结果。然而，这个函数是递归的，我们的`memoize`函数调用不会记忆递归的中间结果。为了做到这一点，我们需要告诉我们的记忆化幂函数不要调用`power`函数，而是调用记忆化的`power`函数。

不幸的是，没有简单的方法可以做到这一点。我们可以将递归调用的函数作为参数传递，但这会因为实现原因改变原始函数的签名。或者我们可以重写函数以利用记忆化。

最终，我们得到了一个相当不错的解决方案。让我们来测试一下。

# 使用记忆化

让我们使用我们的`measureExecutionTimeForF`函数来测量调用我们的`power`函数所需的时间。现在也是时候考虑我们期望的结果了。我们确实缓存了重复调用的值，但这需要在每次调用函数时进行自己的处理和内存。所以，也许它会有所帮助，也许不会。在尝试之前，我们不会知道。

```cpp
TEST_CASE("Pow vs memoized pow"){
    function<int(int, int)> power = [](auto first, auto second){
        return pow(first, second);
    };

    cout << "Computing pow" << endl;
    printDuration("First call no memoization: ",  [&](){ return 
        power(5, 24);});
    printDuration("Second call no memoization: ", [&](){return power(3, 
        1024);});
    printDuration("Third call no memoization: ", [&](){return power(9, 
        176);});
    printDuration("Fourth call no memoization (same as first call): ", 
        [&](){return power(5, 24);});

    auto powerWithMemoization = memoize(power);
    printDuration("First call with memoization: ",  [&](){ return 
        powerWithMemoization(5, 24);});
    printDuration("Second call with memoization: ", [&](){return 
        powerWithMemoization(3, 1024);});
    printDuration("Third call with memoization: ", [&](){return 
        powerWithMemoization(9, 176);});
    printDuration("Fourth call with memoization (same as first call): 
        ", [&](){return powerWithMemoization(5, 24);});
    cout << "DONE computing pow" << endl;

    CHECK_EQ(power(5, 24),  powerWithMemoization(5, 24));
    CHECK_EQ(power(3, 1024),  powerWithMemoization(3, 1024));
    CHECK_EQ(power(9, 176),  powerWithMemoization(9, 176));
}
```

这段代码使用相同的值调用`power`函数，最后一次调用返回到第一次调用的值。然后继续做同样的事情，但在创建`power`的记忆化版本之后。最后，一个健全性检查——`power`函数的结果和记忆化的`power`函数的结果进行比较，以确保我们的`memoize`函数没有错误。

问题是——记忆化是否改善了执行系列中最后一个调用所需的时间（与系列中第一个调用完全相同）？在我的配置中，结果是混合的，如下面的片段所示：

```cpp
Computing pow
First call no memoization: 26421 ns
Second call no memoization: 5207 ns
Third call no memoization: 2058 ns
Fourth call no memoization (same as first call): 179 ns
First call with memoization: 2380 ns
Second call with memoization: 2207 ns
Third call with memoization: 1539 ns
Fourth call with memoization (same as first call): 936 ns
DONE computing pow

```

或者，为了更好地查看（首先是没有记忆化的调用），有以下内容：

```cpp
First call: 26421 ns > 2380 ns
Second call: 5207 ns > 2207 ns
Third call: 2058 ns > 1539 ns
Fourth call: 179 ns < 936 ns
```

总的来说，使用记忆化的调用更好，除非我们重复第一个调用。当然，重复运行测试时结果会有所不同，但这表明提高性能并不像只是使用缓存那么简单。背后发生了什么？我认为最有可能的解释是另一个缓存机制启动了——CPU 或其他机制。

无论如何，这证明了测量的重要性。不出乎意料的是，CPU 和编译器已经做了相当多的优化，我们在代码中能做的也有限。

如果我们尝试递归记忆化呢？我重写了`power`函数以递归使用记忆化，并将缓存与递归调用混合在一起。以下是代码：

```cpp
    map<tuple<int, int>, long long> cache;
    function<long long(int, int)> powerWithMemoization = & -> long long{
            if(exponent == 0) return 1;
            long long value;

            tuple<int, int> parameters(base, exponent);
            auto valueIterator = cache.find(parameters);
            if(valueIterator == cache.end()){
            value = base * powerWithMemoization(base, exponent - 1);
            cache[parameters] = value;
            } else {
            value = valueIterator->second;
        };
        return value;
    };
```

当我们运行它时，结果如下：

```cpp
Computing pow
First call no memoization: 1761 ns
Second call no memoization: 106994 ns
Third call no memoization: 8718 ns
Fourth call no memoization (same as first call): 1395 ns
First call with recursive memoization: 30921 ns
Second call with recursive memoization: 2427337 ns
Third call with recursive memoization: 482062 ns
Fourth call with recursive memoization (same as first call): 1721 ns
DONE computing pow
```

另外，以压缩视图（首先是没有记忆化的调用），有以下内容：

```cpp
First call: 1761 ns < 30921 ns
Second call: 106994 ns < 2427337 ns
Third call: 8718 ns < 482062 ns
Fourth call: 1395 ns < 1721 ns
```

正如你所看到的，构建缓存的时间是巨大的。然而，对于重复调用来说是值得的，但在这种情况下仍然无法击败 CPU 和编译器的优化。

那么，备忘录有帮助吗？当我们使用更复杂的函数时，它确实有帮助。接下来让我们尝试计算两个数字的阶乘之间的差异。我们将使用阶乘的一个简单实现，并首先尝试对阶乘函数进行备忘录，然后再对计算差异的函数进行备忘录。为了保持一致，我们将使用与之前相同的数字对。让我们看一下以下代码：

```cpp
TEST_CASE("Factorial difference vs memoized"){
    function<int(int)> fact = &fact{
        if(n == 0) return 1;
        return n * fact(n-1);
    };

    function<int(int, int)> factorialDifference = &fact{
            return fact(second) - fact(first);
    };
    cout << "Computing factorial difference" << endl;
    printDuration("First call no memoization: ",  [&](){ return 
        factorialDifference(5, 24);});
    printDuration("Second call no memoization: ", [&](){return 
        factorialDifference(3, 1024);});
    printDuration("Third call no memoization: ", [&](){return 
        factorialDifference(9, 176);});
    printDuration("Fourth call no memoization (same as first call): ", 
        [&](){return factorialDifference(5, 24);});

    auto factWithMemoization = memoize(fact);
    function<int(int, int)> factorialMemoizedDifference = 
        &factWithMemoization{
        return factWithMemoization(second) - 
            factWithMemoization(first);
    };
    printDuration("First call with memoized factorial: ",  [&](){ 
        return factorialMemoizedDifference(5, 24);});
    printDuration("Second call with memoized factorial: ", [&](){return 
        factorialMemoizedDifference(3, 1024);});
    printDuration("Third call with memoized factorial: ", [&](){return 
        factorialMemoizedDifference(9, 176);});
    printDuration("Fourth call with memoized factorial (same as first 
        call): ", [&](){return factorialMemoizedDifference(5, 24);});

    auto factorialDifferenceWithMemoization = 
        memoize(factorialDifference);
    printDuration("First call with memoization: ",  [&](){ return 
        factorialDifferenceWithMemoization(5, 24);});
    printDuration("Second call with memoization: ", [&](){return 
        factorialDifferenceWithMemoization(3, 1024);});
    printDuration("Third call with memoization: ", [&](){return 
        factorialDifferenceWithMemoization(9, 176);});
    printDuration("Fourth call with memoization (same as first call): 
        ", [&](){return factorialDifferenceWithMemoization(5, 24);});

    cout << "DONE computing factorial difference" << endl;

    CHECK_EQ(factorialDifference(5, 24),  
        factorialMemoizedDifference(5, 24));
    CHECK_EQ(factorialDifference(3, 1024),  
        factorialMemoizedDifference(3, 1024));
    CHECK_EQ(factorialDifference(9, 176),        
        factorialMemoizedDifference(9, 176));

    CHECK_EQ(factorialDifference(5, 24),  
        factorialDifferenceWithMemoization(5, 24));
    CHECK_EQ(factorialDifference(3, 1024),  
        factorialDifferenceWithMemoization(3, 1024));
    CHECK_EQ(factorialDifference(9, 176),  
        factorialDifferenceWithMemoization(9, 176));
}
```

结果是什么？让我们先看一下普通函数和使用备忘录阶乘函数之间的差异：

```cpp
Computing factorial difference
First call no memoization: 1727 ns
Second call no memoization: 79908 ns
Third call no memoization: 8037 ns
Fourth call no memoization (same as first call): 1539 ns
First call with memoized factorial: 4672 ns
Second call with memoized factorial: 41183 ns
Third call with memoized factrorial: 10029 ns
Fourth call with memoized factorial (same as first call): 1105 ns
```

让我们再次并排比较它们：

```cpp
First call: 1727 ns < 4672 ns
Second call: 79908 ns > 41183 ns
Third call: 8037 ns < 10029 ns
Fourth call: 1539 ns > 1105 ns
```

尽管其他调用的结果是混合的，但在命中缓存值时，备忘录函数比非备忘录函数有约 20%的改进。这似乎是一个小的改进，因为阶乘是递归的，所以理论上，备忘录应该会有很大的帮助。然而，*我们没有对递归进行备忘录*。相反，阶乘函数仍然递归调用非备忘录版本。我们稍后会回到这个问题；现在，让我们来看一下在备忘录`factorialDifference`函数时会发生什么：

```cpp
First call no memoization: 1727 ns
Second call no memoization: 79908 ns
Third call no memoization: 8037 ns
Fourth call no memoization (same as first call): 1539 ns
First call with memoization: 2363 ns
Second call with memoization: 39700 ns
Third call with memoization: 8678 ns
Fourth call with memoization (same as first call): 704 ns
```

让我们并排看一下结果：

```cpp
First call: 1727 ns < 2363 ns
Second call: 79908 ns > 39700 ns
Third call: 8037 ns < 8678 ns
Fourth call: 1539 ns > 704 ns
```

备忘录版本比非备忘录版本在缓存值上快两倍！这太大了！然而，当我们没有缓存值时，我们会因此而付出性能损失。而且，在第二次调用时出现了一些奇怪的情况；某种缓存可能会干扰我们的结果。

我们能通过优化阶乘函数的所有递归来使其更好吗？让我们看看。我们需要改变我们的阶乘函数，使得缓存适用于每次调用。为了做到这一点，我们需要递归调用备忘录阶乘函数，而不是普通的阶乘函数，如下所示：

```cpp
    map<int, int> cache;
    function<int(int)> recursiveMemoizedFactorial = 
        &recursiveMemoizedFactorial, &cache mutable{
        auto value = cache.find(n); 
        if(value != cache.end()) return value->second;
        int result;

        if(n == 0) 
            result = 1;
        else 
            result = n * recursiveMemoizedFactorial(n-1);

        cache[n] = result;
        return result;
    };
```

我们使用差异函数，递归地对阶乘的两次调用进行备忘录：

```cpp
    function<int(int, int)> factorialMemoizedDifference =  
        &recursiveMemoizedFactorial{
                return recursiveMemoizedFactorial(second) -  
                    recursiveMemoizedFactorial(first);
    };
```

通过并排运行初始函数和先前函数的相同数据，我得到了以下输出：

```cpp
Computing factorial difference
First call no memoization: 1367 ns
Second call no memoization: 58045 ns
Third call no memoization: 16167 ns
Fourth call no memoization (same as first call): 1334 ns
First call with recursive memoized factorial: 16281 ns
Second call with recursive memoized factorial: 890056 ns
Third call with recursive memoized factorial: 939 ns
Fourth call with recursive memoized factorial (same as first call): 798 ns 
```

我们可以并排看一下：

```cpp
First call: 1,367 ns < 16,281 ns
Second call: 58,045 ns < 890,056 ns Third call: 16,167 ns > 939 ns Fourth call: 1,334 ns > 798 ns
```

正如我们所看到的，缓存正在累积，对于第一个大计算来说惩罚很大；第二次调用涉及 1024！然而，由于缓存命中，随后的调用速度要快得多。

总之，我们可以说，当有足够的内存可用时，备忘录对于加速重复的复杂计算是有用的。它可能需要一些调整，因为缓存大小和缓存命中取决于对函数的调用次数和重复调用次数。因此，不要认为这是理所当然的——要进行测量，测量，测量。

# 尾递归优化

递归算法在函数式编程中非常常见。实际上，我们的命令式循环中的许多循环可以使用纯函数重写为递归算法。

然而，在命令式编程中，递归并不是很受欢迎，因为它有一些问题。首先，开发人员往往对递归算法的练习比起命令式循环要少。其次，可怕的堆栈溢出——递归调用默认情况下会被放到堆栈上，如果迭代次数太多，堆栈就会溢出并出现一个丑陋的错误。

幸运的是，编译器很聪明，可以为我们解决这个问题，同时优化递归函数。进入尾递归优化。

让我们来看一个简单的递归函数。我们将重用前一节中的阶乘，如下所示：

```cpp
    function<int(int)> fact = &fact{
        if(n == 0) return 1;
        return n * fact(n-1);
    };
```

通常，每次调用都会被放在堆栈上，因此每次调用堆栈都会增长。让我们来可视化一下：

```cpp
Stack content fact(1024)
1024 * fact(1023)
1023 * fact(1022)
...
1 * fact(0)
fact(0) = 1 => unwind the stack
```

我们可以通过重写代码来避免堆栈。我们注意到递归调用是在最后进行的；因此，我们可以将函数重写为以下伪代码：

```cpp
    function<int(int)> fact = &fact{
        if(n == 0) return 1;
        return n * (n-1) * (n-1-1) * (n-1-1-1) * ... * fact(0);
    };
```

简而言之，如果我们启用正确的优化标志，编译器可以为我们做的事情。这个调用不仅占用更少的内存，避免了堆栈溢出，而且速度更快。

到现在为止，你应该知道不要相信任何人的说法，包括我的，除非经过测量。所以，让我们验证这个假设。

首先，我们需要一个测试，用于测量对阶乘函数的多次调用的时间。我选择了一些值来进行测试：

```cpp
TEST_CASE("Factorial"){
    function<int(int)> fact = &fact{
        if(n == 0) return 1;
        return n * fact(n-1);
    };

    printDuration("Duration for 0!: ", [&](){return fact(0);});
    printDuration("Duration for 1!: ", [&](){return fact(1);});
    printDuration("Duration for 10!: ", [&](){return fact(10);});
    printDuration("Duration for 100!: ", [&](){return fact(100);});
    printDuration("Duration for 1024!: ", [&](){return fact(1024);});
}
```

然后，我们需要编译此函数，分别禁用和启用优化。**GNU 编译器集合**（**GCC**）优化尾递归的标志是`-foptimize-sibling-calls`；该名称指的是该标志同时优化了兄弟调用和尾调用。我不会详细介绍兄弟调用优化的作用；让我们只说它不会以任何方式影响我们的测试。

运行这两个程序的时间。首先，让我们看一下原始输出：

+   这是没有优化的程序：

```cpp
Duration for 0!: 210 ns
Duration for 1!: 152 ns
Duration for 10!: 463 ns
Duration for 100!: 10946 ns
Duration for 1024!: 82683 ns
```

+   这是带有优化的程序：

```cpp
Duration for 0!: 209 ns
Duration for 1!: 152 ns
Duration for 10!: 464 ns
Duration for 100!: 6455 ns
Duration for 1024!: 75602 ns
```

现在让我们一起看一下结果；没有优化的持续时间在左边：

```cpp
Duration for 0!: 210 ns > 209 ns
Duration for 1!: 152 ns  = 152 ns
Duration for 10!: 463 ns < 464 ns
Duration for 100!: 10946 ns > 6455 ns
Duration for 1024!: 82683 ns > 75602 ns
```

看起来在我的机器上，优化确实对较大的值起作用。这再次证明了在性能要求时度量的重要性。

在接下来的几节中，我们将以各种方式对代码进行实验，并测量结果。

# 完全优化的调用

出于好奇，我决定运行相同的程序，并打开所有安全优化标志。在 GCC 中，这个选项是`-O3`。结果令人震惊，至少可以这么说：

```cpp
Duration for 0!: 128 ns
Duration for 1!: 96 ns
Duration for 10!: 96 ns
Duration for 100!: 405 ns
Duration for 1024!: 17249 ns
```

让我们比较启用所有优化标志的结果（下一段代码中的第二个值）与仅尾递归优化的结果：

```cpp
Duration for 0!: 209 ns > 128 ns
Duration for 1!: 152 ns > 96 ns
Duration for 10!: 464 ns > 96 ns
Duration for 100!: 6455 ns > 405 ns
Duration for 1024!: 75602 ns > 17249 ns
```

差异是巨大的，正如你所看到的。结论是，尽管尾递归优化很有用，但启用编译器的 CPU 缓存命中和所有优化功能会更好。

但是我们使用了`if`语句；当我们使用`?:`运算符时，这会有不同的效果吗？

# if vs ?:

出于好奇，我决定使用`?:`运算符重写代码，而不是`if`语句，如下所示：

```cpp
    function<int(int)> fact = &fact{
        return (n == 0) ? 1 : (n * fact(n-1));
    };
```

我不知道会有什么结果，结果很有趣。让我们看一下原始输出：

+   没有优化标志：

```cpp
Duration for 0!: 633 ns
Duration for 1!: 561 ns
Duration for 10!: 1441 ns
Duration for 100!: 20407 ns
Duration for 1024!: 215600 ns
```

+   打开尾递归标志：

```cpp
Duration for 0!: 277 ns
Duration for 1!: 214 ns
Duration for 10!: 578 ns
Duration for 100!: 9573 ns
Duration for 1024!: 81182 ns
```

让我们比较一下结果；没有优化的持续时间首先出现：

```cpp
Duration for 0!: 633 ns > 277 ns
Duration for 1!: 561 ns > 214 ns
Duration for 10!: 1441 ns > 578 ns
Duration for 100!: 20407 ns > 9573 ns
Duration for 1024!: 75602 ns > 17249 ns
```

两个版本之间的差异非常大，这是我没有预料到的。像往常一样，这很可能是 GCC 编译器的结果，你应该自己测试一下。然而，看起来这个版本对于我的编译器来说更适合尾部优化，这是一个非常有趣的结果。

# 双递归

尾递归对双递归有效吗？我们需要想出一个例子，将递归从一个函数传递到另一个函数，以检查这一点。我决定编写两个函数，`f1`和`f2`，它们互相递归调用。`f1`将当前参数与`f2(n - 1 )`相乘，而`f2`将`f1(n)`添加到`f1(n-1)`。以下是代码：

```cpp
    function<int(int)> f2;
    function<int(int)> f1 = &f2{
        return (n == 0) ? 1 : (n * f2(n-1));
    };

    f2 = &f1{
        return (n == 0) ? 2 : (f1(n) + f1(n-1));
    };
```

让我们检查对`f1`的调用的时间，值从`0`到`8`：

```cpp
    printDuration("Duration for f1(0): ", [&](){return f1(0);});
    printDuration("Duration for f1(1): ", [&](){return f1(1);});
    printDuration("Duration for f1(2): ", [&](){return f1(2);});
    printDuration("Duration for f1(3): ", [&](){return f1(3);});
    printDuration("Duration for f1(4): ", [&](){return f1(4);});
    printDuration("Duration for f1(5): ", [&](){return f1(5);});
    printDuration("Duration for f1(6): ", [&](){return f1(6);});
    printDuration("Duration for f1(7): ", [&](){return f1(7);});
    printDuration("Duration for f1(8): ", [&](){return f1(8);});
```

这是我们得到的结果：

+   没有尾调用优化：

```cpp
Duration for f1(0): 838 ns
Duration for f1(1): 825 ns
Duration for f1(2): 1218 ns
Duration for f1(3): 1515 ns
Duration for f1(4): 2477 ns
Duration for f1(5): 3919 ns
Duration for f1(6): 5809 ns
Duration for f1(7): 9354 ns
Duration for f1(8): 14884 ns
```

+   使用调用优化：

```cpp
Duration for f1(0): 206 ns
Duration for f1(1): 327 ns
Duration for f1(2): 467 ns
Duration for f1(3): 642 ns
Duration for f1(4): 760 ns
Duration for f1(5): 1155 ns
Duration for f1(6): 2023 ns
Duration for f1(7): 3849 ns
Duration for f1(8): 4986 ns
```

让我们一起看一下结果；没有尾优化的调用持续时间在左边：

```cpp
f1(0): 838 ns > 206 ns
f1(1): 825 ns > 327 ns
f1(2): 1218 ns > 467 ns
f1(3): 1515 ns > 642 ns
f1(4): 2477 ns > 760 ns
f1(5): 3919 ns > 1155 ns
f1(6): 5809 ns > 2023 ns
f1(7): 9354 ns > 3849 ns
f1(8): 14884 ns > 4986 ns
```

差异确实非常大，显示代码得到了很大的优化。但是，请记住，对于 GCC，我们使用的是`-foptimize-sibling-calls`优化标志。该标志执行两种优化：尾调用和兄弟调用。兄弟调用是指对返回类型和参数列表总大小相同的函数的调用，因此允许编译器以与尾调用类似的方式处理它们。在我们的情况下，很可能两种优化都被应用了。

# 使用异步代码优化执行时间

当我们有多个线程时，我们可以使用两种近似技术来优化执行时间：并行执行和异步执行。我们已经在前一节中看到了并行执行的工作原理；异步调用呢？

首先，让我们回顾一下异步调用是什么。我们希望进行一次调用，然后在主线程上继续正常进行，并在将来的某个时候获得结果。对我来说，这听起来像是函数的完美工作。我们只需要调用函数，让它们执行，然后在一段时间后再与它们交谈。

既然我们已经谈到了 future，让我们来谈谈 C++中的`future`构造。

# Futures

我们已经确定，在程序中避免管理线程是理想的，除非进行非常专业化的工作，但我们需要并行执行，并且通常需要同步以从另一个线程获取结果。一个典型的例子是一个长时间的计算，它会阻塞主线程，除非我们在自己的线程中运行它。我们如何知道计算何时完成，以及如何获得计算的结果？

在 1976 年至 1977 年，计算机科学中提出了两个概念来简化解决这个问题的方法——futures 和 promises。虽然这些概念在各种技术中经常可以互换使用，在 C++中它们有特定的含义：

+   一个 future 可以从提供者那里检索一个值，同时进行同步处理

+   promise 存储了一个未来的值，并提供了一个同步点

由于它的性质，`future`对象在 C++中有一些限制。它不能被复制，只能被移动，并且只有在与共享状态相关联时才有效。这意味着我们只能通过调用`async`、`promise.get_future()`或`packaged_task.get_future()`来创建一个有效的 future 对象。

值得一提的是，promises 和 futures 在它们的实现中使用了线程库；因此，您可能需要添加对另一个库的依赖。在我的系统（Ubuntu 18.04，64 位）上，使用 g++编译时，我不得不添加一个对`pthread`库的链接依赖；如果您在 mingw 或 cygwin 配置上使用 g++，我希望您也需要相同的依赖。

让我们首先看看如何在 C++中同时使用`future`和`promise`。首先，我们将为一个秘密消息创建一个`promise`：

```cpp
    promise<string> secretMessagePromise;
```

接下来，让我们创建一个`future`并使用它启动一个新的线程。线程将使用一个 lambda 函数简单地打印出秘密消息：

```cpp
    future<string> secretMessageFuture = 
        secretMessagePromise.get_future();
    thread isPrimeThread(printSecretMessage, ref(secretMessageFuture));
```

注意我们需要避免复制`future`；在这种情况下，我们使用一个对`future`的引用包装器。

现在我们暂时只讨论这个线程；下一步是实现承诺，也就是设置一个值：

```cpp
    secretMessagePromise.set_value("It's a secret");
    isPrimeThread.join();
```

与此同时，另一个线程将做一些事情，然后要求我们信守诺言。嗯，不完全是；它将要求`promise`的值，这将阻塞它，直到调用`join()`：

```cpp
auto printSecretMessage = [](future<string>& secretMessageFuture) {
    string secretMessage = secretMessageFuture.get();
    cout << "The secret message: " << secretMessage << '\n';
};
```

正如您可能注意到的，这种方法将计算值的责任放在了主线程上。如果我们希望它在辅助线程上完成呢？我们只需要使用`async`。

假设我们想要检查一个数字是否是质数。我们首先编写一个 lambda 函数，以一种天真的方式检查这一点，对`2`到`x-1`之间的每个可能的除数进行检查，并检查`x`是否可以被它整除。如果它不能被任何值整除，那么它是一个质数：

```cpp
auto is_prime = [](int x) {
    auto xIsDivisibleBy = bind(isDivisibleBy, x, _1);
    return none_of_collection(
            rangeFrom2To(x - 1), 
            xIsDivisibleBy
        );
};
```

使用了一些辅助的 lambda 函数。一个用于生成这样的范围：

```cpp
auto rangeFromTo = [](const int start, const int end){
    vector<int> aVector(end);
    iota(aVector.begin(), aVector.end(), start);
    return aVector;
};
```

这是专门用于生成以`2`开头的范围：

```cpp
auto rangeFrom2To = bind(rangeFromTo, 2, _1);
```

然后，一个检查两个数字是否可被整除的谓词：

```cpp
auto isDivisibleBy = [](auto value, auto factor){
    return value % factor == 0;
};
```

要在主线程之外的一个单独线程中运行这个函数，我们需要使用`async`声明一个`future`：

```cpp
    future<bool> futureIsPrime(async(is_prime, 2597));
```

`async`的第二个参数是我们函数的输入参数。允许多个参数。

然后，我们可以做其他事情，最后，要求结果：

```cpp
TEST_CASE("Future with async"){
    future<bool> futureIsPrime(async(is_prime, 7757));
    cout << "doing stuff ..." << endl;
 bool result = futureIsPrime.get();

    CHECK(result);
}
```

粗体代码行标志着主线程停止等待来自辅助线程的结果的点。

如果您需要多个`future`，您可以使用它们。在下面的示例中，我们将使用四个不同的值在四个不同的线程中运行`is_prime`，如下所示：

```cpp
TEST_CASE("more futures"){
    future<bool> future1(async(is_prime, 2));
    future<bool> future2(async(is_prime, 27));
    future<bool> future3(async(is_prime, 1977));
    future<bool> future4(async(is_prime, 7757));

    CHECK(future1.get());
    CHECK(!future2.get());
    CHECK(!future3.get());
    CHECK(future4.get());
}
```

# 功能性异步代码

我们已经看到，线程的最简单实现是一个 lambda，但我们可以做得更多。最后一个示例使用多个线程异步地在不同的值上运行相同的操作，可以转换为一个功能高阶函数。

但让我们从一些简单的循环开始。首先，我们将输入值和预期结果转换为向量：

```cpp
    vector<int> values{2, 27, 1977, 7757};
    vector<bool> expectedResults{true, false, false, true};
```

然后，我们需要一个`for`循环来创建 futures。重要的是不要调用`future()`构造函数，因为这样做会由于尝试将新构造的`future`对象复制到容器中而失败。相反，将`async()`的结果直接添加到容器中：

```cpp
    vector<future<bool>> futures;
    for(auto value : values){
        futures.push_back(async(is_prime, value));
    }
```

然后，我们需要从线程中获取结果。再次，我们需要避免复制`future`，因此在迭代时将使用引用：

```cpp
    vector<bool> results;
    for(auto& future : futures){
        results.push_back(future.get());
    }
```

让我们来看看整个测试：

```cpp
TEST_CASE("more futures with loops"){
    vector<int> values{2, 27, 1977, 7757};
    vector<bool> expectedResults{true, false, false, true};

    vector<future<bool>> futures;
    for(auto value : values){
        futures.push_back(async(is_prime, value));
    }

    vector<bool> results;
    for(auto& future : futures){
        results.push_back(future.get());
    }

    CHECK_EQ(results, expectedResults);
}
```

很明显，我们可以将这些转换成几个 transform 调用。然而，我们需要特别注意避免复制 futures。首先，我创建了一个帮助创建`future`的 lambda：

```cpp
    auto makeFuture = [](auto value){
        return async(is_prime, value);
    };
```

第一个`for`循环然后变成了一个`transformAll`调用：

```cpp
    vector<future<bool>> futures = transformAll<vector<future<bool>>>
       (values, makeFuture);
```

第二部分比预期的要棘手。我们的`transformAll`的实现不起作用，所以我将内联调用`transform`：

```cpp
    vector<bool> results(values.size());
    transform(futures.begin(), futures.end(), results.begin(), []
        (future<bool>& future){ return future.get();});
```

我们最终得到了以下通过的测试：

```cpp
TEST_CASE("more futures functional"){
    vector<int> values{2, 27, 1977, 7757};

    auto makeFuture = [](auto value){
        return async(is_prime, value);
    };

    vector<future<bool>> futures = transformAll<vector<future<bool>>>
        (values, makeFuture);
    vector<bool> results(values.size());
    transform(futures.begin(), futures.end(), results.begin(), []
        (future<bool>& future){ return future.get();});

    vector<bool> expectedResults{true, false, false, true};

    CHECK_EQ(results, expectedResults);
}
```

我必须对你诚实，这是迄今为止实现起来最困难的代码。在处理 futures 时，有很多事情可能会出错，而且原因并不明显。错误消息相当没有帮助，至少对于我的 g++版本来说是这样。我成功让它工作的唯一方法是一步一步地进行，就像我在本节中向你展示的那样。

然而，这个代码示例展示了一个重要的事实；通过深思熟虑和测试使用 futures，我们可以并行化高阶函数。因此，如果您需要更好的性能，可以使用多个核心，并且不能等待标准中并行运行策略的实现，这是一个可能的解决方案。即使只是为了这一点，我认为我的努力是有用的！

由于我们正在谈论异步调用，我们也可以快速浏览一下响应式编程的世界。

# 响应式编程的一点体验

**响应式编程**是一种编写代码的范式，专注于处理数据流。想象一下需要分析一系列温度值的数据流，来自安装在自动驾驶汽车上的传感器的值，或者特定公司的股票值。在响应式编程中，我们接收这个连续的数据流并运行分析它的函数。由于新数据可能会不可预测地出现在流中，编程模型必须是异步的；也就是说，主线程不断等待新数据，当数据到达时，处理被委托给次要流。结果通常也是异步收集的——要么推送到用户界面，保存在数据存储中，要么传递给其他数据流。

我们已经看到，函数式编程的主要重点是数据。因此，函数式编程是处理实时数据流的良好选择并不足为奇。高阶函数的可组合性，如`map`、`reduce`或`filter`，以及并行处理的机会，使得函数式设计风格成为响应式编程的一个很好的解决方案。

我们不会详细讨论响应式编程。通常使用特定的库或框架来简化这种数据流处理的实现，但是根据我们目前拥有的元素，我们可以编写一个小规模的示例。

我们需要几样东西。首先，一个数据流；其次，一个接收数据并立即将其传递到处理管道的主线程；第三，一种获取输出的方式。

对于本例的目标，我将简单地使用标准输入作为输入流。我们将从键盘输入数字，并以响应式的方式检查它们是否是质数，从而始终保持主线程的响应。这意味着我们将使用`async`函数为我们从键盘读取的每个数字创建一个`future`。输出将简单地写入输出流。

我们将使用与之前相同的`is_prime`函数，但添加另一个函数，它将打印到标准输出该值是否是质数。

```cpp
auto printIsPrime = [](int value){
    cout << value << (is_prime(value) ? " is prime" : " is not prime")  
    << endl;
};
```

`main`函数是一个无限循环，它从输入流中读取数据，并在每次输入新值时启动一个`future`：

```cpp
int main(){
    int number;

    while(true){
        cin >> number;
        async(printIsPrime, number);
    }
}
```

使用一些随机输入值运行此代码会产生以下输出：

```cpp
23423
23423 is not prime
453576
453576 is not prime
53
53 is prime
2537
2537 is not prime
364544366
5347
54
534532
436
364544366 is not prime
5347 is prime
54 is not prime
534532 is not prime
436 is not prime
```

正如你所看到的，结果会尽快返回，但程序允许随时引入新数据。

我必须提到，为了避免每次编译本章的代码时都出现无限循环，响应式示例可以通过`make reactive`编译和运行。你需要用中断来停止它，因为它是一个无限循环。

这是一个基本的响应式编程示例。显然，它可以随着数据量的增加、复杂的流水线和每个流水线的并行化等变得更加复杂。然而，我们已经实现了本节的目标——让你了解响应式编程以及我们如何使用函数构造和异步调用使其工作。

我们已经讨论了如何优化执行时间，看了各种帮助我们实现更快性能的方法。现在是时候看一个情况，我们想要减少程序的内存使用。

# 优化内存使用

到目前为止，我们讨论的用于以函数方式构造代码的方法涉及多次通过被视为不可变的集合。因此，这可能会导致集合的复制。例如，让我们看一个简单的代码示例，它使用`transform`来增加向量的所有元素：

```cpp
template<typename DestinationType>
auto transformAll = [](const auto source, auto lambda){
    DestinationType result;
    transform(source.begin(), source.end(), back_inserter(result), 
        lambda);
    return result;
};

TEST_CASE("Memory"){
    vector<long long> manyNumbers(size);
    fill_n(manyNumbers.begin(), size, 1000L);

    auto result = transformAll<vector<long long>>(manyNumbers, 
        increment);

    CHECK_EQ(result[0], 1001);
}
```

这种实现会导致大量的内存分配。首先，`manyNumbers`向量被复制到`transformAll`中。然后，`result.push_back()`会自动调用，可能导致内存分配。最后，`result`被返回，但初始的`manyNumbers`向量仍然被分配。

我们可以立即改进其中一些问题，但讨论它们与其他可能的优化方法的比较也是值得的。

为了进行测试，我们需要处理大量的集合，并找到一种测量进程内存分配的方法。第一部分很容易——只需分配大量 64 位值（在我的编译器上是长长类型）；足够分配 1GB 的 RAM：

```cpp
const long size_1GB_64Bits = 125000000;
TEST_CASE("Memory"){
    auto size = size_1GB_64Bits;
    vector<long long> manyNumbers(size);
    fill_n(manyNumbers.begin(), size, 1000L);

    auto result = transformAll<vector<long long>>(manyNumbers, 
        increment);

    CHECK_EQ(result[0], 1001);
}
```

第二部分有点困难。幸运的是，在我的 Ubuntu 18.04 系统上，我可以在`/proc/PID/status`文件中监视进程的内存，其中 PID 是进程标识符。通过一些 Bash 魔法，我可以创建一个`makefile`配方，将每 0.1 秒获取的内存值输出到一个文件中，就像这样：

```cpp
memoryConsumptionNoMoveIterator: .outputFolder 
    g++ -DNO_MOVE_ITERATOR -std=c++17 memoryOptimization.cpp -Wall -
        Wextra -Werror -o out/memoryOptimization
    ./runWithMemoryConsumptionMonitoring memoryNoMoveIterator.log
```

你会注意到`-DNO_MOVE_ITERATOR`参数；这是一个编译指令，允许我为不同的目标编译相同的文件，以检查多个解决方案的内存占用。这意味着我们之前的测试是在`#if NO_MOVE_ITERATOR`指令内编写的。

只有一个注意事项——因为我使用了 bash `watch`命令来生成输出，你需要在运行`make memoryConsumptionNoMoveIterator`后按下一个键，以及对每个其他内存日志配方也是如此。

有了这个设置，让我们改进`transformAll`以减少内存使用，并查看输出。我们需要使用引用类型，并从一开始就为结果分配内存，如下所示：

```cpp
template<typename DestinationType>
auto transformAll = [](const auto& source, auto lambda){
    DestinationType result;
    result.resize(source.size());
    transform(source.begin(), source.end(), result.begin(), lambda);
    return result;
};
```

预期的结果是，改进的结果是最大分配从 0.99 GB 开始，但跳到 1.96 GB，大致翻了一番。

我们需要将这个值放在上下文中。让我们先测量一下一个简单的`for`循环能做什么，并将结果与使用`transform`实现的相同算法进行比较。

# 测量简单 for 循环的内存

使用`for`循环的解决方案非常简单：

```cpp
TEST_CASE("Memory"){
    auto size = size_1GB_64Bits;
    vector<long long> manyNumbers(size);
    fill_n(manyNumbers.begin(), size, 1000L);

    for(auto iter = manyNumbers.begin(); iter != manyNumbers.end(); 
        ++iter){
            ++(*iter);
    };

    CHECK_EQ(manyNumbers[0], 1001);
}
```

在测量内存时，没有什么意外——整个过程中占用的内存保持在 0.99 GB。我们能用`transform`也实现这个结果吗？嗯，有一个版本的`transform`可以就地修改集合。让我们来测试一下。

# 测量就地 transform 的内存

要就地使用`transform`，我们需要提供目标迭代器参数`source.begin()`，如下所示：

```cpp
auto increment = [](const auto value){
    return value + 1;
};

auto transformAllInPlace = [](auto& source, auto lambda){
    transform(source.begin(), source.end(), source.begin(), lambda);
};

TEST_CASE("Memory"){
    auto size = size_1GB_64Bits;
    vector<long long> manyNumbers(size);
    fill_n(manyNumbers.begin(), size, 1000L);

    transformAllInPlace(manyNumbers, increment);

    CHECK_EQ(manyNumbers[0], 1001);
}
```

根据文档，这应该在同一集合中进行更改；因此，它不应该分配更多的内存。如预期的那样，它具有与简单的`for`循环相同的行为，内存占用在整个程序运行期间保持在 0.99 GB。

然而，您可能会注意到我们现在不返回值以避免复制。我喜欢返回值，但我们还有另一个选择，使用移动语义：

```cpp
template<typename SourceType>
auto transformAllInPlace = [](auto& source, auto lambda) -> SourceType&& {
    transform(source.begin(), source.end(), source.begin(), lambda);
    return move(source);
};
```

为了使调用编译通过，我们需要在调用`transformAllInPlace`时传递源的类型，因此我们的测试变成了：

```cpp
TEST_CASE("Memory"){
    auto size = size_1GB_64Bits;
    vector<long long> manyNumbers(size);
    fill_n(manyNumbers.begin(), size, 1000L);

    auto result = transformAllInPlace<vector<long long>>(manyNumbers, 
        increment);

    CHECK_EQ(result[0], 1001);
}
```

让我们测量一下移动语义是否有所帮助。结果如预期；内存占用在整个运行时保持在 0.99 GB。

这引出了一个有趣的想法。如果我们在调用`transform`时使用移动语义呢？

# 使用移动迭代器进行 transform

我们可以将我们的`transform`函数重写为使用移动迭代器，如下所示：

```cpp
template<typename DestinationType>
auto transformAllWithMoveIterator = [](auto& source, auto lambda){
    DestinationType result(source.size());
    transform(make_move_iterator(source.begin()), 
        make_move_iterator(source.end()), result.begin(), lambda);
    source.clear();
    return result;
};
```

理论上，这应该是将值移动到目标而不是复制它们，从而保持内存占用低。为了测试一下，我们运行相同的测试并记录内存：

```cpp
TEST_CASE("Memory"){
    auto size = size_1GB_64Bits;
    vector<long long> manyNumbers(size);
    fill_n(manyNumbers.begin(), size, 1000L);

    auto result = transformAllWithMoveIterator<vector<long long>>
        (manyNumbers, increment);

    CHECK_EQ(result[0], 1001);
}
```

结果出乎意料；内存从 0.99 GB 开始上升到 1.96 GB（可能是在`transform`调用之后），然后又回到 0.99 GB（很可能是`source.clear()`的结果）。我尝试了多种变体来避免这种行为，但找不到保持内存占用在 0.99 GB 的解决方案。这似乎是移动迭代器实现的问题；我建议您在您的编译器上测试一下它是否有效。

# 比较解决方案

使用就地或移动语义的解决方案，虽然减少了内存占用，但只有在不需要源数据进行其他计算时才有效。如果您计划重用数据进行其他计算，那么保留初始集合是不可避免的。此外，不清楚这些调用是否可以并行运行；由于 g++尚未实现并行执行策略，我无法测试它们，因此我将把这个问题留给读者作为练习。

但是函数式编程语言为了减少内存占用做了什么呢？答案非常有趣。

# 不可变数据结构

纯函数式编程语言使用不可变数据结构和垃圾回收的组合。修改数据结构的每次调用都会创建一个似乎是初始数据结构的副本，只有一个元素被改变。初始结构不会受到任何影响。然而，这是使用指针来完成的；基本上，新的数据结构与初始数据结构相同，只是有一个指向改变值的指针。当丢弃初始集合时，旧值不再被使用，垃圾收集器会自动将其从内存中删除。

这种机制充分利用了不可变性，允许了 C++无法实现的优化。此外，实现通常是递归的，这也利用了尾递归优化。

然而，可以在 C++中实现这样的数据结构。一个例子是一个名为**immer**的库，你可以在 GitHub 上找到它，网址是[`github.com/arximboldi/immer`](https://github.com/arximboldi/immer)。Immer 实现了许多不可变的集合。我们将看看`immer::vector`；每当我们调用通常会修改向量的操作（比如`push_back`）时，`immer::vector`会返回一个新的集合。每个返回的值都可以是常量，因为它永远不会改变。我在本章的代码中使用 immer 0.5.0 编写了一个小测试，展示了`immer::vector`的用法，你可以在下面的代码中看到：

```cpp
TEST_CASE("Check immutable vector"){
    const auto empty = immer::vector<int>{};
    const auto withOneElement = empty.push_back(42);

    CHECK_EQ(0, empty.size());
    CHECK_EQ(1, withOneElement.size());
    CHECK_EQ(42, withOneElement[0]);
}
```

我不会详细介绍不可变数据结构；但是，我强烈建议你查看*immer*网站上的文档（[`sinusoid.es/immer/introduction.html`](https://sinusoid.es/immer/introduction.html)）并尝试使用该库。

# 总结

我们已经看到，性能优化是一个复杂的话题。作为 C++程序员，我们需要从我们的代码中获得更多的性能；本章中我们提出的问题是：是否可能优化以函数式风格编写的代码？

答案是——是的，如果你进行了测量，并且有一个明确的目标。我们需要特定的计算更快完成吗？我们需要减少内存占用吗？应用程序的哪个领域需要最大程度的性能改进？我们想要进行怪异的点优化吗，这可能需要在下一个编译器、库或平台版本中进行重写？这些都是你在优化代码之前需要回答的问题。

然而，我们已经看到，当涉及到利用计算机上的所有核心时，函数式编程有巨大的好处。虽然我们正在等待高阶函数的标准实现并行执行，但我们可以通过编写自己的并行算法来利用不可变性。递归是函数式编程的另一个基本特征，每当使用它时，我们都可以利用尾递归优化。

至于内存消耗，实现在第三方库中的不可变数据结构，以及根据目标谨慎优化我们使用的高阶函数，都可以帮助我们保持代码的简单性，而复杂性发生在代码的特定位置。当我们丢弃源集合时，可以使用移动语义，但记得检查它是否适用于并行执行。

最重要的是，我希望你已经了解到，测量是性能优化中最重要的部分。毕竟，如果你不知道自己在哪里，也不知道自己需要去哪里，你怎么能进行旅行呢？

我们将继续通过利用数据生成器来进行测试来继续我们的函数式编程之旅。现在是时候看看基于属性的测试了。
