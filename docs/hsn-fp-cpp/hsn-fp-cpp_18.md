# 第十四章：使用 ranges 库进行懒惰评估

在本书中，我们详细讨论了如何以函数的方式思考，以及函数链接和组合如何帮助创建模块化和可组合的设计。然而，我们遇到了一个问题——根据我们当前的方法，需要将大量数据从一个集合复制到另一个集合。

幸运的是，Eric Niebler 自己着手开发了一个库，使纯函数式编程语言中的解决方案——懒惰评估成为可能。该库名为**ranges**，随后被正式纳入 C++ 20 标准。在本章中，我们将看到如何利用它。

本章将涵盖以下主题：

+   为什么以及何时懒惰评估是有用的

+   ranges 库的介绍

+   如何使用 ranges 库进行懒惰评估

# 技术要求

你需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.4.0。

该代码可以在 GitHub 上找到，网址为[https:/​/​github.​com/​PacktPublishing/​Hands-​On-​Functional-Programming-​with-​Cpp](https://github.%E2%80%8Bcom/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)，在`Chapter14`文件夹中。它包括并使用了`doctest`，这是一个单头文件的开源单元测试库。你可以在它的 GitHub 仓库上找到它，网址为[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# ranges 库概述

ranges 库为 C++程序员提供了各种有用的新工具。它们都很有用，但对于我们的函数式编程需求来说，许多工具尤其如此。

但首先，让我们看看如何设置它。要在 C++ 17 中使用 ranges 库，你需要使用来自[`ericniebler.github.io/range-v3/`](https://ericniebler.github.io/range-v3/)的指示。然后，你只需要包含`all.hpp`头文件：

```cpp
#include <range/v3/all.hpp>
```

至于 C++ 20，你只需要包含`<ranges>`头文件，因为该库已包含在标准中：

```cpp
#include <ranges>
```

然而，如果你在尝试上一行代码时遇到编译错误，不要感到惊讶。在撰写本文时，最新版本的 g++是 9.1，但 ranges 库尚未包含在标准中。由于其规模，实现预计会相当晚。在那之前，如果你想尝试它，你仍然可以使用 Eric Niebler 的版本。

那么，ranges 库提供了什么？嗯，一切都始于范围的概念。一个范围由一个起始迭代器和一个结束迭代器组成。这使我们首先可以在现有集合的基础上添加一个范围。然后，我们可以将一个范围传递给需要起始和结束迭代器的算法（如`transform`、`sort`或`accumulate`），从而消除了对`begin()`和`end()`的不便调用。

使用 ranges，我们可以构建视图。视图指定我们对部分或全部集合感兴趣，通过两个迭代器，但也允许懒惰评估和可组合性。由于视图只是集合的轻量级包装器，我们可以声明一系列操作，而不实际执行它们，直到需要结果。我们将在下一节详细介绍这是如何工作的，但这里有一个简单的示例，组合两个操作，将过滤出集合中所有的倍数为六的数字，首先通过过滤*所有的偶数*，然后再过滤出*是 3 的倍数*的数字：

```cpp
numbers | ranges::view::filter(isEven) | ranges::view::filter(isMultipleOf3)
```

在 ranges 上也可以进行突变，借助于操作。操作类似于视图，只是它们会就地改变底层容器，而不是创建副本。正如我们之前多次讨论过的那样，在函数式编程中，我们更喜欢不改变数据；然而，在某些情况下，我们可以通过这种解决方案优化性能，因此值得一提。下面是一个操作的示例...嗯，在操作中：

```cpp
numbers |= action::sort | action::take(5);
```

`|`运算符对于函数式编程者来说非常有趣，因为它是一种函数组合运算符。对于 Unix/Linux 用户来说，使用它也很自然，他们非常习惯组合操作。正如我们在第四章中所看到的，*函数组合的概念*，这样的运算符将非常有用。不幸的是，它还不支持任意两个函数的组合，只支持视图和操作的组合。

最后，ranges 库支持自定义视图。这打开了诸如数据生成之类的可能性，这对许多事情都很有用，特别是第十一章中的*基于属性的测试*。

让我们更详细地访问范围库的特性，并举例说明。

# 惰性求值

在过去的章节中，我们已经看到了如何以函数式的方式构造代码，通过对数据结构进行小的转换来利用。让我们举一个简单的例子——计算列表中所有偶数的和。结构化编程方法是编写一个循环，遍历整个结构，并添加所有偶数元素：

```cpp
int sumOfEvenNumbersStructured(const list<int>& numbers){
    int sum = 0;
    for(auto number : numbers){
        if(number % 2 == 0) sum += number;
    }
    return sum;
};
```

这个函数的测试在一个简单的例子上运行正确：

```cpp
TEST_CASE("Run events and get the user store"){
    list<int> numbers{1, 2, 5, 6, 10, 12, 17, 25};

    CHECK_EQ(30, sumOfEvenNumbersStructured(numbers));
}
```

当然，这种方法会改变数据，我们已经知道这不总是一个好主意。它也一次做了太多的事情。我们宁愿组合更多的函数。第一个函数需要决定一个数字是否是偶数：

```cpp
auto isEven = [](const auto number){
    return number % 2 == 0;
};
```

第二个函数从集合中挑选满足谓词的数字：

```cpp
auto pickNumbers  = [](const auto& numbers, auto predicate){
    list<int> pickedNumbers;
    copy_if(numbers.begin(), numbers.end(), 
        back_inserter(pickedNumbers), predicate);
    return pickedNumbers;
};
```

第三个计算集合中所有元素的和：

```cpp
auto sum = [](const auto& numbers){
    return accumulate(numbers.begin(), numbers.end(), 0);
};
```

这将我们带到了最终的实现，它包括所有这些函数：

```cpp
auto sumOfEvenNumbersFunctional = [](const auto& numbers){
    return sum(pickNumbers(numbers, isEven));
};
```

然后它通过了测试，就像结构化的解决方案一样：

```cpp
TEST_CASE("Run events and get the user store"){
    list<int> numbers{1, 2, 5, 6, 10, 12, 17, 25};

    CHECK_EQ(30, sumOfEvenNumbersStructured(numbers));
    CHECK_EQ(30, sumOfEvenNumbersFunctional(numbers));
}
```

函数式解决方案有明显的优势——它简单，由可以重新组合的小函数组成，而且它是不可变的，这也意味着它可以并行运行。然而，它也有一个缺点——它会复制数据。

我们已经在第十章中看到了如何处理这个问题，但事实上，最简单的解决方案是惰性求值。想象一下，如果我们可以链接函数调用，但是在我们需要其结果的时刻之前，代码实际上并没有执行，那将意味着什么。这个解决方案打开了编写我们需要编写的代码以及我们需要的方式的可能性，编译器最大限度地优化了函数链。

这就是 ranges 库正在做的事情，以及其他一些额外的功能。

# 使用 ranges 库进行惰性求值

ranges 库提供了一个名为**views**的工具。视图允许从迭代器构造不可变且廉价的数据范围。它们不会复制数据，只是引用数据。我们可以使用`view`来过滤我们的集合中的所有偶数：

```cpp
ranges::view::filter(numbers, isEven)
```

视图可以在不复制任何内容的情况下进行组合，并使用组合运算符`|`。例如，我们可以通过组合两个过滤器来获得能被`6`整除的数字列表：第一个是偶数，第二个是能被`3`整除的数字。给定一个新的谓词，检查一个数字是否是`3`的倍数，我们使用以下方法：

```cpp
auto isMultipleOf3 = [](const auto number){
    return number % 3 == 0;
};
```

我们通过以下组合获得能被`6`整除的数字列表：

```cpp
numbers | ranges::view::filter(isEven) | ranges::view::filter(isMultipleOf3)
```

重要的是要注意，当编写这段代码时实际上没有计算任何东西。视图已经初始化，并且正在等待命令。所以，让我们计算视图中元素的和：

```cpp
auto sumOfEvenNumbersLazy = [](const auto& numbers){
    return ranges::accumulate(ranges::view::
        filter(numbers, isEven), 0);
};
TEST_CASE("Run events and get the user store"){
    list<int> numbers{1, 2, 5, 6, 10, 12, 17, 25};

    CHECK_EQ(30, sumOfEvenNumbersLazy(numbers));
}
```

`ranges::accumulate`函数是 accumulate 的一个特殊实现，它知道如何与视图一起工作。只有在调用`accumulate`时，视图才会起作用；此外，实际上没有数据被复制——相反，ranges 使用智能迭代器来计算结果。

让我们也看看组合视图的结果。如预期的那样，向量中所有能被`6`整除的数字的和是`18`：

```cpp
auto sumOfMultiplesOf6 = [](const auto& numbers){
    return ranges::accumulate(
            numbers | ranges::view::filter(isEven) | 
                ranges::view::filter(isMultipleOf3), 0);
};
TEST_CASE("Run events and get the user store"){
    list<int> numbers{1, 2, 5, 6, 10, 12, 17, 25};

    CHECK_EQ(18, sumOfMultiplesOf6(numbers));
}
```

写代码的方式真好！它比以前的两种选项都要容易得多，同时内存占用也很低。

但这还不是 ranges 能做的全部。

# 使用操作进行可变更改

除了视图，范围库还提供了操作。操作允许急切的、可变的操作。例如，要对同一个向量中的值进行排序，我们可以使用以下语法：

```cpp
TEST_CASE("Sort numbers"){
    vector<int> numbers{1, 12, 5, 20, 2, 10, 17, 25, 4};
    vector<int> expected{1, 2, 4, 5, 10, 12, 17, 20, 25};

    numbers |= ranges::action::sort;

    CHECK_EQ(expected, numbers);
}
```

`|=`运算符类似于`ranges::action::sort(numbers)`调用，原地对向量进行排序。操作也是可组合的，可以通过直接方法调用或使用`|`运算符进行组合。这使我们能够编写代码，通过`sort`和`unique`操作的组合来对容器进行排序并保留唯一项：

```cpp
TEST_CASE("Sort numbers and pick unique"){
    vector<int> numbers{1, 1, 12, 5, 20, 2, 10, 17, 25, 4};
    vector<int> expected{1, 2, 4, 5, 10, 12, 17, 20, 25};

    numbers |= ranges::action::sort | ranges::action::unique;

    CHECK_EQ(expected, numbers);
}
```

然而，这还不是范围可以做的一切。

# 无限序列和数据生成

由于视图是惰性评估的，它们允许我们创建无限序列。例如，要生成一系列整数，我们可以使用`view::ints`函数。然后，我们需要限制序列，所以我们可以使用`view::take`来保留序列的前五个元素：

```cpp
TEST_CASE("Infinite series"){
    vector<int> values = ranges::view::ints(1) | ranges::view::take(5);
    vector<int> expected{1, 2, 3, 4, 5};

    CHECK_EQ(expected, values);
}
```

可以使用`view::iota`来进行额外的数据生成，例如对于`chars`类型，只要允许增量即可：

```cpp
TEST_CASE("Infinite series"){
    vector<char> values = ranges::view::iota('a') | 
        ranges::view::take(5);
    vector<char> expected{'a', 'b', 'c', 'd', 'e'};

    CHECK_EQ(expected, values);
}
```

此外，您可以使用`linear_distribute`视图生成线性分布的值。给定一个值间隔和要包含在线性分布中的项目数，该视图包括间隔边界以及足够多的内部值。例如，从[`1`，`10`]区间中取出五个线性分布的值会得到这些值：`{1, 3, 5, 7, 10}`：

```cpp
TEST_CASE("Linear distributed"){
    vector<int> values = ranges::view::linear_distribute(1, 10, 5);
    vector<int> expected{1, 3, 5, 7, 10};

    CHECK_EQ(expected, values);
}
```

如果我们需要更复杂的数据生成器怎么办？幸运的是，我们可以创建自定义范围。假设我们想要创建从`1`开始的每个`2`的十次幂的列表（即*2¹*，*2¹¹*，*2²¹*等）。我们可以使用 transform 调用来做到这一点；然而，我们也可以使用`yield_if`函数结合`for_each`视图来实现。下面代码中的粗体行显示了如何将这两者结合使用：

```cpp
TEST_CASE("Custom generation"){
    using namespace ranges;
    vector<long> expected{ 2, 2048, 2097152, 2147483648 };

 auto everyTenthPowerOfTwo = view::ints(1) | view::for_each([](int 
        i){ return yield_if(i % 10 == 1, pow(2, i)); });
    vector<long> values = everyTenthPowerOfTwo | view::take(4);

    CHECK_EQ(expected, values);
}
```

首先，我们生成从`1`开始的无限整数序列。然后，对于每个整数，我们检查该值除以`10`的余数是否为`1`。如果是，我们返回`2`的幂。为了获得有限的向量，我们将前面的无限序列传递给`take`视图，它只保留前四个元素。

当然，这种生成方式并不是最佳的。对于每个有用的数字，我们需要访问`10`，最好是从`1`，`11`，`21`等开始。

值得在这里提到的是，编写这段代码的另一种方法是使用 stride 视图。`stride`视图从序列中取出每个 n^(th)元素，正好符合我们的需求。结合`transform`视图，我们可以实现完全相同的结果：

```cpp
TEST_CASE("Custom generation"){
    using namespace ranges;
    vector<long> expected{ 2, 2048, 2097152, 2147483648 };

 auto everyTenthPowerOfTwo = view::ints(1) | view::stride(10) | 
        view::transform([](int i){ return pow(2, i); });
    vector<long> values = everyTenthPowerOfTwo | view::take(4);

    CHECK_EQ(expected, values);
}
```

到目前为止，您可能已经意识到数据生成对于测试非常有趣，特别是基于属性的测试（正如我们在第十一章中讨论的那样，*基于属性的测试*）。然而，对于测试，我们经常需要生成字符串。让我们看看如何做到这一点。

# 生成字符串

要生成字符串，首先我们需要生成字符。对于 ASCII 字符，我们可以从`32`到`126`的整数范围开始，即有趣的可打印字符的 ASCII 代码。我们取一个随机样本并将代码转换为字符。我们如何取一个随机样本呢？好吧，有一个叫做`view::sample`的视图，它可以从范围中取出指定数量的随机样本。最后，我们只需要将其转换为字符串。这就是我们如何得到一个由 ASCII 字符组成的长度为`10`的随机字符串：

```cpp
TEST_CASE("Generate chars"){
    using namespace ranges;

    vector<char> chars = view::ints(32, 126) | view::sample(10) | 
        view::transform([](int asciiCode){ return char(asciiCode); });
    string aString(chars.begin(), chars.end()); 

    cout << aString << endl;

    CHECK_EQ(10, aString.size());
}
```

以下是运行此代码后得到的一些样本：

```cpp
%.0FL[cqrt
#0bfgiluwy
4PY]^_ahlr
;DJLQ^bipy
```

正如你所看到的，这些是我们测试中使用的有趣字符串。此外，我们可以通过改变`view::sample`的参数来改变字符串的大小。

这个例子仅限于 ASCII 字符。然而，由于 UTF-8 现在是 C++标准的一部分，扩展以支持特殊字符应该很容易。

# 总结

Eric Niebler 的 ranges 库在软件工程中是一个罕见的成就。它成功地简化了现有 STL 高阶函数的使用，同时添加了惰性评估，并附加了数据生成。它不仅是 C++ 20 标准的一部分，而且也适用于较旧版本的 C++。

即使您不使用函数式的代码结构，无论您喜欢可变的还是不可变的代码，ranges 库都可以让您的代码变得优雅和可组合。因此，我建议您尝试一下，看看它如何改变您的代码。这绝对是值得的，也是一种愉快的练习。

我们即将结束本书。现在是时候看看 STL 和语言标准对函数式编程的支持，以及我们可以从 C++ 20 中期待什么，这将是下一章的主题。
