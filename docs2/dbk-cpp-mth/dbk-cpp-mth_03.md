

# 只有一个 C++，它是面向对象的

*只有当你忽略所有* *其他所有内容*

C++ 是在 C 的基础上加入了对象特性而诞生的，这使得许多开发者仍然认为它是一种面向对象编程（OOP）语言。在本章中，我们将看到 C++ 允许多种编程范式，并且可以安全地将其描述为一种包含多种编程语言的单一语言。我们将探讨 C++ 支持的几种范式，包括结构化编程、面向对象编程、函数式编程和元编程，以及强类型与准可选类型的选择。

在本章中，我们将涵盖以下主要主题：

+   C++ 的多重面相

+   C++ 中的函数式编程

+   元编程

+   极端强类型

+   那么，忽略类型呢？

# 技术要求

本章的代码可以从 GitHub 仓库 [`github.com/PacktPublishing/Debunking-CPP-Myths`](https://github.com/PacktPublishing/Debunking-CPP-Myths) 中的 **ch3** 文件夹获取。它使用 Makefile、g++ 和 doctest 库（[`github.com/doctest/doctest`](https://github.com/doctest/doctest)）进行单元测试。代码是为 C++20 编译的。

# C++ 的多重面相

如果你像我一样，经常在不同的组织、团队和技术会议上穿梭，你很快就会注意到两件事：与其他开发者相比，C++ 程序员有独特的兴趣，C++ 社区更准确地描述为小型的、专业的 C++ 开发者群体。这与其他社区不同；如果你讨论 Java，你可能会最终谈到 Spring 框架和 REST API 或 Android 工具包。C# 主要围绕 Microsoft 库进行标准化，而 JavaScript 主要与 React 相关。但是，如果你把来自不同组织的 100 名 C++ 程序员召集到一起，你很快就会注意到差异。嵌入式 C++ 专注于控制所有资源，因为为销售数百万台的设备额外增加 1 MB 的内存会迅速推高成本。游戏开发者处于光谱的另一端，他们关注如何从下一代 GPU 和 CPU 中挤出额外的帧率。高频交易人士对避免 CPU 缓存未命中以及如何从自动化交易算法中消除皮秒级的延迟了如指掌，因为最小的时分数值可能意味着数百万欧元。工程软件开发者更为轻松，但仍担心复杂渲染模型中变更的有效性。然后你还会发现处理铁路、汽车或工厂自动化系统的程序员，他们的主要关注点是弹性和健壮性。

这张图片虽然远非完整，但足以展示 C++程序员巨大的多样性，与使用其他任何语言的同行相比。我们几乎可以说，从某个角度来看，C++是最后剩下的实际通用语言，因为其他主流语言在实践中主要用于特定类型的程序：Java 用于企业后端服务和 Android 开发，C#用于 Web 和 Windows 应用程序和服务，JavaScript 用于丰富的 Web 前端和无服务器后端，Python 用于脚本、数据科学和 DevOps。但 C++用于嵌入式软件、工厂系统、交易、模拟、工程工具、操作系统等等。

旧话“形式追随功能”是关于设计适用于人们建造的每一件事物，包括编程语言，对 C++同样适用。项目类型和程序员的大幅变化，以及斯特劳斯特鲁普希望使其尽可能强大的愿望，都融入了 C++语言。C++不是一种单一的语言；每个程序员使用的 C++子集通常与他们所在组织中的同事不同。

是的，C++最初是以具有对象的 C 语言为基础发展起来的，那时面向对象编程（OOP）正处于兴起阶段。但是，与此同时，C++与 C 语言向后兼容，这意味着你仍然可以在 C++中编写结构化编程。然后，模板变得必要。接着，lambda 表达式变得有用。虽然 C++始终是一系列不同语言的集合，但如今这种趋势更加明显。为了证明这一点，让我们看看你可以在 C++中使用的一些范式，从函数式编程开始。

# C++中的函数式编程

我记得在大学时，对编程非常着迷，并且已经相当擅长编写 BASIC、Pascal、Logo 和简单的 C++。我认为是在我二年级的时候，我选修了一门关于函数式编程的课程。老师非常热情，渴望向我们展示这种范式的奇妙之处，解释了许多我无法完全理解的概念。这门课程对我来说完全是一次失败，因为我唯一学到的是如何在 Lisp 中编写命令式代码，以及如何将我已知的习惯用法翻译成在这个奇怪的语言中可以工作，其括号位于表达式外部的东西。

我在作为软件工程师开始职业生涯后，试图回归函数式编程。网上有很多资源，但它们解释范式的方式并没有帮助。“这基本上是范畴论，”他们说。一切皆函数，甚至数字（查看 Church 编码）。由于它们是端内函子的范畴中的幺半群，你可以轻松理解单子。这种解释风格使用更复杂的概念来解释实际的概念，并不利于理解。

这也是为什么我花了好几年时间才理解函数式编程是什么以及它如何帮助软件开发。我成为了这个范式的粉丝，但不是狂热者。像任何工程师一样，我喜欢解决问题，在我的情况下，我通常用代码来解决。拥有更简单的代码总是很好的，尽管通常更简单并不意味着更熟悉。

如果我今天要解释函数式编程，我会关注三个重要的事情：*不可变性*、*纯函数*和*函数操作*。也许出乎意料的是，C++非常适合所有这些特性。与其它主流编程语言相比（尽管不如 Rust，但我们在最后一章会谈到这一点），C++在不可变性方面表现出色。

然而，有一个问题：函数式编程是一个不同的范式，有其自身的权衡。我发现 C++程序员发现思考 lambda 表达式很困难，因为他们将 lambda 表达式视为不是基本概念，而是建立在现有语言之上的东西。这是公平的，因为 lambda 表达式是对象，而不是 C++中的第一级设计元素。然而，以函数式范式思考要求程序员暂时忘记这些知识，并接受函数式设计元素。当你实现了一些有效的东西并寻求改进时，你可以回到这些知识。

让我们更详细地解释这三个特性，然后讨论使用函数式编程对我们软件架构的影响。

## 不可变性

不可变性从根本上意味着每个变量都初始化为一个值，但无法将新值赋给变量。在 C++中，这可以通过**const**或**constexpr**来实现，具体取决于我们希望值在运行时还是编译时不可变。

虽然不可变性对于简单类型来说容易理解，但集合和对象引入了挑战。一个不可变集合是在每次更改时返回一个新的集合。例如，以下代码展示了一个可变集合：

```cpp
vector<int> numbers {1, 2, 3};
numbers.push_back(4);
assert(numbers == vector<int> {1, 2, 3, 4});
```

将这个例子与下一个代码示例中的假设不可变集合进行对比，该集合在添加元素时返回一个新的集合：

```cpp
immutable_vector<int> numbers {1, 2, 3};
immutable_vector<int> moreNumbers = numbers.push_back(4);
assert(numbers == immutable_vector<int> {1, 2, 3});
assert(moreNumbers == immutable_vector<int> {1, 2, 3, 4});
```

这个特性保证了你使用的是所需数据结构的正确版本。但 C++大脑中的内存优化警钟可能会响起。似乎为不可变集合分配了大量的内存！这不是一种浪费吗？

在不可变集合中执行更改时，确实可能会暂时使用比预期更多的内存。然而，函数式语言已经找到了避免这种情况的智能方法，C++也完全有能力使用相同的机制。这取决于实现方式。

优化不可变集合内存的方法是使用*智能指针*。记住，一旦值被分配给变量，它就是不可变的。因此，当集合首次初始化时，为集合的每个元素分配内存，并将每个内存区域分配给特定的值。当添加新元素时，每个元素的指针被复制，并为新值分配新的内存区域。如果从集合中删除元素，除了指向被删除元素的指针外，所有指向现有元素的指针都被复制。一旦内存区域不再被任何指针引用，它就会被删除。

虽然 STL 中没有实现不可变集合，但像 immer（[`github.com/arximboldi/immer`](https://github.com/arximboldi/immer)）这样的库允许你使用这种模式，而不必过多担心内部细节。

好的，但不可变对象怎么办？面向对象的全部目的不是将行为与数据混合吗？

关于这一点，我有三件事情要说。

首先，好问题！

其次，面向对象编程被误解为关于封装、继承和多态，而实际上它是关于消息传递。不幸的是，C++是我喜欢称之为“面向类编程”的趋势的引领者：一种关注类及其关系而不是对象及其关系的编程风格。

第三，函数式编程实际上并不排斥对象。实现不可变对象非常简单：要么我们使用**const**实现不可变的数据结构，要么每个改变数据的函数返回一个包含修改后数据的新的对象。

在这里值得提一下的是，你不需要在程序中完全使用不可变性来从函数式编程中受益。我写的代码足够多，最大化了常量性，但仍然使用了标准 STL 集合和会改变其内部数据的对象。然而，你需要意识到，之前描述的不可变性的水平使得你更容易将并行性引入到你的程序中。如果值不能改变，你将不会有临界区的问题。每个线程都使用自己的值，改变值只会对特定的线程产生影响。实际上，这是不可变性的一个附带好处。我说附带好处，因为不可变性结合纯函数和良好的命名，一旦你习惯了构建块，程序就更容易理解。所以，让我们看看纯函数。

## 纯函数

纯函数是一个对于相同的输入返回相同输出且不会改变任何上下文值的函数。根据定义，纯函数不能进行**输入/输出**（**I/O**）操作。然而，任何非平凡的程序都可以写成纯函数和 I/O 函数的组合。

纯函数是你能想到的最简单的函数类型。它们易于理解，非常可预测，并且由于缺乏副作用，可以缓存。这导致数据驱动的单元测试变得容易，以及可能的优化，例如在第一次调用时缓存特定输入的函数结果，并在以后重用。

纯函数是函数式编程的核心。在 C++中，它们可以通过对不可变性的支持轻松实现。

在纯函数式语言中编写函数的原始方式是 lambda 表达式。自 C++11 以来，lambda 表达式已经成为了标准的一部分。然而，C++中的 lambda 表达式可以是可变的，因为它们可以改变它们在上下文中捕获的变量。因此，在 C++中编写纯函数，即使使用 lambda 表达式，也需要你确保所有涉及的变量的 const 属性。

在函数式范式下，一切要么是函数要么是数据结构，而在纯函数式语言中，这两者是可以互换的。那么，我们如何从简单的函数中创建复杂的行为呢？当然是通过使用各种操作来组合函数。

## 函数上的操作

由于函数是函数式编程的主要设计元素，思考函数如何通过操作进行变化是理所当然的。最常见的函数式操作是偏应用和组合。

偏应用指的是通过将函数的一个参数的值绑定到特定值来创建一个新的函数。例如，如果我们有一个函数**add(const int first, const int second)**，我们可以通过将**second**参数绑定到值**1**来获得**increment(const int)**函数。让我们花点时间考虑一下后果：无论函数接收多少参数，都可以通过后续的偏应用减少到不接受任何参数的函数。这为我们提供了一个通用的语言，可以用来在代码中表达任何事物。

要在 C++中实现偏应用，我们可以使用来自**<functional>**头文件的**std::bind**函数。让我们看看如何通过绑定**add**函数的第二个参数到值**1**来从**add**函数获得**increment**函数：

```cpp
#include <functional>
auto add = [](const int first, const int second){ return first + second; };
auto increment = std::bind(add, std::placeholders::_1, 1);
TEST_CASE("add"){
        CHECK_EQ(10, add(4, 6));
}
TEST_CASE("increment"){
        CHECK_EQ(10, increment(9));
}
```

这是从函数式编程的角度来看的一个整洁的解决方案。然而，返回值很复杂，近似于一个函数而不是一个函数。这是 C++程序员在尝试函数式编程时遇到的心理障碍之一。我已经远离这个语言足够长的时间，可以让自己用高级概念来思考，而不是总是分析实现。所以，当我使用**std::bind**进行偏应用时，我会把结果当作一个函数，并希望实现者已经完成了优化并提供必要的功能。

函数的另一个基本操作是函数式组合。你可能已经在数学中遇到过这个结构。函数式组合指的是从两个函数，*g* 和 *h*，创建一个函数 *f*，使得对于任何值 *x*，*f(x) = g(h(x))*。在数学中，这通常表示为 *f = g∘h*。

不幸的是，C++ 标准中没有函数或操作来实现函数式组合，但使用模板很容易实现这个操作。再次强调，C++ 中这个操作的结果可能很复杂，但我鼓励你将其视为一个函数，而不是实际的数据结构。

让我们看看 C++ 中函数式组合的一个可能实现。`compose` 函数接受两个类型参数，**F** 和 **G**，分别表示要组合的函数 **f** 和 **g** 的类型。`compose` 函数返回一个 lambda 表达式，它接受一个参数 **value**，并返回 **f(g(value)**：

```cpp
template <class F, class G>
auto compose(F f, G g){
  return ={return f(g(value));};
}
```

注意

上述例子是从 Alex 的另一本关于该主题的 Packt 出版物书籍中借用的，书名为 *Hands-On Functional Programming* *in C++* 。

让我们通过一个简单的例子来看看如何使用这个函数。让我们实现一个价格计算器，它接受价格、折扣、服务费和税费作为参数，并返回最终价格。我们先来看一个命令式实现，使用一个函数来直接计算所有内容。`computePriceImperative` 函数接受价格，减去折扣，加上服务费，然后再加上税费百分比：

```cpp
double computePriceImperative(const int taxPercentage, const int serviceFee, const double price, const int discount){
return (price - discount + serviceFee) * (1 + (static_cast<double>(taxPercentage) / 100));
}
TEST_CASE("compute price imperative"){
        int taxPercentage = 18;
        int serviceFee = 10;
        double price = 100;
        int discount = 10;
                                   double result = computePriceImperative(taxPercentage, serviceFee, price, discount);
        CHECK_EQ(118, result);
}
```

这是一个简单的实现，足以给出结果。当需要添加更多类型的折扣、根据项目修改税费或更改折扣的顺序时，这类代码通常会出现挑战。当然，当需要时，我们可以应用命令式或面向对象风格，并提取多个函数，每个操作一个函数，然后按需组合它们。

但现在让我们看看函数式风格。我们可以做的第一件事是使用 lambda 表达式来表示每个操作，并为最终计算使用另一个 lambda 表达式。我们实现了一些 lambda 表达式：一个用于从价格中减去折扣，第二个用于应用服务费，第三个用于应用税费，最后一个通过链式调用之前定义的所有 lambda 表达式来计算价格。最终我们得到了以下代码：

```cpp
auto discountPrice = [](const double price, const int discount){return price - discount;};
auto addServiceFee = [](const double price, const int serviceFee){ return price + serviceFee; };
auto applyTax = [](const double price, const int taxPercentage){ return price * (1 + static_cast<double>(taxPercentage)/100); };
auto computePriceLambda = [](const int taxPercentage, const int serviceFee, const double price, const int discount){
return applyTax(addServiceFee(discountPrice(price, discount), serviceFee), taxPercentage);
};
TEST_CASE("compute price with lambda"){
int taxPercentage = 18;
int serviceFee = 10;
double price = 100;
int discount = 10;
double result = computePriceLambda(taxPercentage, serviceFee, price, discount);
CHECK_EQ(118, result);
}
```

这段代码更好吗？好吧，这取决于。一个因素是对这种范式的熟悉程度，但不要让这阻止你；正如我之前说的，熟悉性常常被误认为是简单性，但这两者并不相同。另一个因素是将 lambda 表达式视为函数而不是数据结构。一旦你克服这两个挑战，我们会注意到一些事情：lambda 表达式非常小，易于理解，并且是纯函数，这在客观上是最简单的函数类型。我们可以以多种方式链式调用，例如，在含税价格上应用折扣，因此我们有了更多的选择。尽管如此，我们仍然可以用命令式编程做到现在为止我们能做的任何事情。

那么，让我们再进一步，使其完全功能化。我们将使用我们创建的 lambda 表达式，但不是返回一个值，我们的实现将使用部分应用和函数组合来返回一个函数，该函数能给出我们想要的答案。由于前面的 lambda 表达式有两个参数，在应用函数组合之前，我们需要将其中一个参数绑定到相应的输入。因此，对于**discountPrice** lambda 表达式，我们将折扣参数绑定到传递给**computePriceFunctional**函数的值，并得到一个只接受一个参数（初始价格）的 lambda 表达式，返回带有折扣的价格。对于**addServiceFee** lambda 表达式，我们将**serviceFee**参数绑定到传递给**computePriceFunctional**函数的值，并得到一个只接受一个参数（服务前的价格）的函数，返回带有服务费的价格。对于**applyTax** lambda 表达式，我们将**taxPercentage**参数绑定到传递给**computePriceFunctional**函数的值，并得到一个只接受一个参数（不含税的价格）的函数，返回带有税的价格。一旦我们得到这些只接受一个参数的函数，我们就可以使用之前展示的**compose**函数将它们组合起来，从而得到一个只接受一个参数（价格）的函数，当调用时，计算正确的最终价格。以下是结果：

```cpp
auto computePriceFunctional(const int taxPercentage, const int serviceFee, const double price, const int discount){
using std::bind;
using std::placeholders::_1;
auto discountLambda = bind(discountPrice, _1, discount);
auto serviceFeeLambda = bind(addServiceFee, _1, serviceFee);
auto applyTaxLambda = bind(applyTax, _1, taxPercentage);
return compose( applyTaxLambda, compose(serviceFeeLambda, discountLambda));
}
TEST_CASE("compute price functional"){
int taxPercentage = 18;
int serviceFee = 10;
double price = 100;
int discount = 10;
auto computePriceLambda = computePriceFunctional(taxPercentage, serviceFee, price, discount);
double result = computePriceLambda(price);
CHECK_EQ(118, result);
}
```

这种编程风格乍一看与面向对象编程（OOP）或结构化编程截然不同。但如果你稍微思考一下，你会意识到一个对象仅仅是一组紧密相连、部分应用的函数集合。如果你从对象中提取函数，你需要传递对象中使用的成员数据，这对于那些曾经用 C 语言编程的人来说是一种熟悉的风格。因此，将方法包含在对象中相当于将一些参数绑定到由构造函数初始化的对象数据成员上。因此，面向对象编程和函数式编程并不是真正的敌人，只是表达相同行为的不同且等效的方式，各有不同的权衡。

作为后续“元编程”部分的序言，让我们先看看如何在编译时使所有这些函数可用。我们需要用模板做一点魔法，并将值参数作为模板参数传递，还需要添加很多**constexpr**，但以下代码同样有效：

```cpp
template <class F, class G>
  constexpr auto compose(F f, G g){
    return ={return f(g(value));};
  }
constexpr auto discountPriceCompile = [](const double price,   const int discount){return price - discount;};
  constexpr auto addServiceFeeCompile = [](const double price,   const int serviceFee){ return price + serviceFee; };
  constexpr auto applyTaxCompile = [](const double price, cons  t int taxPercentage){ return price * (1 + static_cast<double  >(taxPercentage)/100); };
  template<int taxPercentage, int serviceFee, double price, in  t discount>
  constexpr auto computePriceFunctionalCompile() {
          using std::bind;
          using std::placeholders::_1;
          constexpr auto discountLambda = bind(discountPrice,   _1, discount);
          constexpr auto serviceFeeLambda = bind(addServiceFee  , _1, serviceFee);
          constexpr auto applyTaxLambda = bind(applyTax, _1, t  axPercentage);
          return compose( applyTaxLambda, compose(serviceFeeLa  mbda, discountLambda));
  }
TEST_CASE("compute price functional compile"){
        constexpr int taxPercentage = 18;
        constexpr int serviceFee = 10;
        constexpr double price = 100;
        constexpr int discount = 10;
        constexpr auto computePriceLambda = computePriceFunctionalCompile<taxPercentage, serviceFee, price, discount>();
        double result = computePriceLambda(price);
        CHECK_EQ(118, result);
}
```

通过这种方式，我们已经看到了 C++中函数式编程的基本块。现在让我们看看它们在哪里以及为什么有用。

## 函数式风格的架构模式

让我们先看看如何实现一个完全采用函数式风格的程序。我们无法讨论这种应用程序的所有可能的设计模式，但我们可以展示一些示例。

我们首先注意到，函数式编程对我们的设计提出了一些约束。我们倾向于不可变性和纯函数。我们使用数据结构，但它们是不可变的，这意味着对数据结构的任何更改都会给我们一个新的版本。最后，I/O 部分需要尽可能分离和瘦，因为它需要进行更改。

使用这些约束的一个简单设计模式是管道模式。让我们想象我们收到一个 XML 格式的文件，并使用其中的数据调用 Web 服务。我们有一个输入层读取 XML 文件，一个输出层写入 Web 服务，以及一个中间层使用函数式风格。我们现在可以考虑输入和输出数据，并在输入上实施后续转换，以产生所需的输出。这些转换中的每一个都是对不可变数据结构工作的纯函数。

由于缺乏更改，这种过程高度可并行化。事实上，C++17 引入了**<execution>**头文件，它允许并行运行常见的 STL 算法。类似的模式在数据转换架构（如**提取、转换、加载**（**ETL**））和由 Hadoop 普及的 MapReduce 架构中使用。

这种模式可以扩展到数据转换之外，到更宽松定义的**功能核心，命令式外壳**架构，由 Gary Bernhardt 恰当地命名。如果您想了解更多具体细节，可以查看具有功能核心的六边形架构。

这不仅表明我们可以使用函数式范式在 C++中设计程序，而且还表明在某些情况下这种架构是合适的。它还表明我们可以采用这种编程风格的一些部分，并将其应用于我们的实现中。

# 元编程

似乎有一件事将程序员团结在一起，无论他们如何不同：对递归笑话的喜爱。程序员心中有一种欣赏某种类型对称性的东西。当涉及到编程语言和编程范式时，你很难找到一个比能够理解自己的语言更对称的类型。

对应的编程范式被称为元编程，而将这一理念推向极限的编程语言被称为同构语言，这意味着一个程序可以操作另一个程序或其自身的表示或数据。具有这种特性的编程语言包括 Lisp 及其衍生方言，最新的是 Clojure。

元编程非常强大，但也非常难以掌握，并且可能会在大型项目中引入许多问题。与元编程相关的一些功能在现代语言中可用，例如工具、反射或指令的动态执行。但除了使用注解之外，实践中很少使用所有这些功能。

然而，C++却有所不同。元编程的一个特性是将计算从运行时移动到编译时，C++通过模板元编程完全接受了这一点。在语言更近的版本中，通过引入**constexpr**和**consteval**的泛化常量表达式，编译时计算的实现已经得到了简化。

这种技术的典型例子是阶乘实现。在运行时计算的递归阶乘实现看起来是这样的：

```cpp
int factorial(const int number){
    if(number == 0) return 1;
    return number * factorial(number – 1);
}
```

同样的实现可以使用模板元编程来完成。C++模板的一个可能不太为人所知的特性是它们可以接受一个值作为参数，而不仅仅是类型。此外，既可以是泛型模板，例如，接受任何整数值作为参数的模板，也可以是特化，它只接受特定的值，都可以提供。在我们的例子中，我们可以实现一个接受整数和针对值**0**的特化的阶乘模板，从而得到以下代码：

```cpp
template<int number>
struct Factorial {
enum { value = number * Factorial<number – 1>::value};
};
template<>
struct Factorial<0>{
enum {value = 1};
};
```

这种实现与之前的一个实现达到相同的目标，唯一的区别是对于**Factorial<25>**这样的调用，将在编译时而不是运行时进行计算。从 C++11 开始，随着泛化常量表达式的引入，我们可以完全避免使用模板，而是使用**constexpr**和**consteval**来告诉编译器哪些值需要在编译时计算。以下是对同一代码的简化实现，使用了常量表达式：

```cpp
constexpr int factorial(const int number) {
return (number == 0) ? 1 : (number * factorial(number - 1));
}
```

可供 C++程序员使用的这些元编程技术使得与编译时和运行时发生的事情相关的决策更加灵活。它们在 CPU 周期与可执行文件大小之间提供了一个权衡。如果你有大量的内存可用，但计算需要非常快，那么在可执行文件中缓存结果可能是可行的，而**constexpr**和**consteval**将成为你的朋友。

但可能性并不止于此。我们可以在 C++程序中创建从编译时就可以验证的有效程序。我们只需要将强类型推向极限。

# 强类型推向极限

软件开发中最大的挑战之一是避免错误。这是一个普遍存在的问题，以至于我们习惯于用暗示代码出了问题的名称来称呼它。然而，实际上，我们应该称之为“错误”，因为这就是它们的本质。

既然我们有编译器，为什么不能对代码施加足够的限制，以便它们在出现错误时告诉我们？我们可能能够做到这一点，但不是免费的。我们在上一节讨论了模板元编程，但我们遗漏了一个重要的特性：模板元编程是图灵完备的。这意味着对于我们可以用常规方式编写的任何程序，我们也可以使用模板元编程来编写。

这个想法非常强大，并且随着时间的推移在各个环境中都有所讨论。如果你想尝试一个完全围绕这个概念构建的编程语言，可以尝试 Idris（[`www.idris-lang.org/`](https://www.idris-lang.org/)）。许多程序员可能熟悉 Haskell 在编译时验证方面的支持。但我的第一次接触这个想法是在 2001 年安德烈·亚历山德鲁斯库的奠基性著作《现代 C++设计：泛型编程和设计模式应用》中。

让我们考虑一个简单的问题。常见的错误和代码恶臭的来源之一是所谓的**原始类型迷恋**，即使用原始类型来表示复杂数据的迷恋。原始类型迷恋的一个典型例子是将长度、金钱、温度或重量表示为数字，完全忽略它们的计量单位。与其这样做，不如为金钱使用一个值，它允许根据上下文具有特定的精度，例如会计和银行以及货币的七位小数。即使在程序只处理单一货币的情况下，这在软件开发中通常也很有用，因为你可以肯定的是，在功能方面，最终总会有一些变化——会有一个时间点，你的客户会要求你添加第二种货币。

与原始类型迷恋相关的一个典型挑战是限制原始类型。例如，考虑一个可以存储一天中小时数的类型。这个值不仅是一个无符号整数，而且只能是从 0 到 23，假设为了简单起见采用 24 小时制。如果能告诉编译器，0-23 范围之外的任何值都不应被视为小时数，并在传递例如 27 这样的值时给出相关错误，那就太好了。

在这种情况下，枚举可以是一个解决方案，因为值的数量很少。但我们将忽略这个选项，首先考虑如何在运行时实现它。我们可以想象一个名为**Hour**的类，如果传递给构造函数的值不在 0 到 23 之间，它会抛出一个异常：

```cpp
class Hour{
private:
int theValue = 0;
void setValue(int candidateValue) {
if(candidateValue >= 0 && candidateValue <= 23){
theValue = candidateValue;
}
else{
throw std::out_of_range("Value out of range");
}
}
public:
Hour(int theValue){
setValue(theValue);
}
int value() const {
return theValue;
}
};
TEST_CASE("Valid hour"){
Hour hour(10);
CHECK_EQ(10, hour.value());
}
TEST_CASE("Invalid hour"){
CHECK_THROWS(Hour(30));
}
```

如果我们想在编译时进行检查呢？好吧，是时候使用**constexpr**的力量来告诉编译器哪些值在编译时定义，以及使用**static_assert**来验证范围了：

```cpp
template <int Min, int Max>
class RangedInteger{
private:
int theValue;
constexpr RangedInteger(int theValue) : theValue(theValue) {}
public:
template <int CandidateValue>
static constexpr RangedInteger make() {
static_assert(CandidateValue >= Min && CandidateValue <= Max, "Value out of range.");
return CandidateValue;
}
constexpr int value() const {
return theValue;
}
};
using Hour = RangedInteger<0, 23>;
```

在前面的实现中，以下代码运行完美：

```cpp
TEST_CASE("Valid hour"){
constexpr Hour h = Hour::make<10>();
CHECK_EQ(10, h.value());
}
```

但如果我们尝试传递一个超出范围的值，我们会得到一个编译错误：

```cpp
TEST_CASE("Invalid hour"){
constexpr Hour h2 = Hour::make<30>();
}
Hour.h: In instantiation of 'static constexpr RangedInteger<Min, Max> RangedInteger<Min, Max>::make() [with int CandidateValue = 30; int Min = 0; int Max = 23]':
Hour.h:11:87: error: static assertion failed: Value out of range.
   11 |                                 static_assert(CandidateValue >= Min && CandidateValue <= Max, "Value out of range.");
      |                ~~~~~~~~~~~~~~~^~~~~~
Hour.h:11:87: note: '(30 <= 23)' evaluates to false
```

这个错误告诉我们，我们不能有一个值为 30 的小时，这正是我们所需要的！

这只是 C++程序员工具箱中的一种技术，他们希望创建在编译时可以证明有效的程序。正如我们提到的，模板元编程是图灵完备的，这意味着我们可以在理论上在编译时实现任何我们可以在运行时实现的程序。但总是有权衡。注意，**小时**值必须是**constexpr**，这意味着值将被存储在可执行文件中。这是设计上的考虑，因为将类型约束到最大值唯一的方法是将它们编译到单元中。

在实践中，我发现这种技术很容易导致难以理解和修改的代码。修改这种代码需要很强的纪律性，因为修改现有代码仍然可能引入我们通过强类型否则已经排除的 bug。基本技术始终是添加，而不是修改，除非是为了修复问题。我们一直保持代码的整洁，但类型可以很快变得非常抽象，这使得在六个月之后重建导致它们的推理变得非常困难。从积极的一面来看，这种技术在创建专注于非常特定领域的库时效果最好。

虽然我发现这种技术很有趣，但我倾向于在编程时更喜欢自由。我在编码时使用自己的纪律——测试驱动开发、无情重构、极端关注命名和简单设计。我更希望有一种编写代码的方式，让编译器处理细节，这就是为什么我将要讨论的最后一种范式尽可能地忽略类型。

# 那么忽略类型呢？

几年前，我领导了一个团队，使用名为 Groovy 的语言和名为 Grails 的框架构建了一些 Web 应用程序。Groovy 是一种可选类型和动态语言，这意味着它在运行时分配类型，但你可以为编译器提供类型提示。它也可以静态编译，并且由于它是建立在 JVM 上的，代码最终会变成 Java 单元。

我在之前的 Web 项目中注意到，类型在系统的边缘很有用，用于检查请求参数、与数据库交互以及其他 I/O 操作。但在 Web 应用的核心中，类型往往会使事情变得更复杂。我们经常不得不更改代码或编写额外的代码来适应已经实现的行为的新用法，因为 Web 应用的用户通常会注意到一个有用的场景，并希望它在其他上下文或其他类型的数据中也能工作。因此，我从一开始就决定，我们将使用类型进行请求验证，以确保安全和正确性，以及与外部系统的交互，以确保简单性。但我们没有在核心中使用类型。

计划一直是使用一种合理的自动化测试策略，以便通过测试证明所有代码的有效性。我预计没有类型会使我们编写更多的测试，但我遇到了一个巨大的惊喜：测试的数量与之前相对相同，但我们有更少的代码。此外，我们编写的代码，因为不涉及类型，迫使我们非常小心地命名事物，因为名称是我们作为程序员了解函数或变量正在做什么的唯一线索。

这至今仍是我最喜欢的编程风格。我想按自己的意愿编写代码，尽可能表达清晰，然后让编译器处理类型。你可以将这种方法视为极端的多态：如果你传递一个具有所需方法的类型的变量，代码应该能够根据你传递的类型正常工作。这不是一个我会推荐给每个人的风格，因为它是否有效并不明显，仅与特定的设计经验相结合，但它是一种你可以尝试的风格。然而，第一个挑战是放弃控制编译器做什么，这对非常注重细节的 C++程序员来说是一个更难达成的成就。

这在 C++中是如何工作的呢？幸运的是，自从 C++11 以来，C++引入了**auto**关键字，并在后续的标准中逐渐改进了其功能。然而，不利的是，C++在动态类型方面不如 Groovy 那么宽容，所以我偶尔需要使用模板。

首先，让我用你可以编写的最多态的函数来让你惊叹一下：

```cpp
auto identity(auto value){ return value;}
TEST_CASE("Identity"){
CHECK_EQ(1, identity(1));
CHECK_EQ("asdfasdf", identity("asdfasdf"));
CHECK_EQ(vector{1, 2, 3}, identity(vector{1, 2, 3}));
}
```

这个函数无论我们传递给它什么都能正常工作。这不是很酷吗？想象一下，你有一堆这样的函数可以用在系统的核心部分，而且不需要修改它们。这听起来就像是我心目中的理想编程环境。然而，现实生活比这要复杂得多，程序需要的不仅仅是恒等函数。

让我们来看一个稍微复杂一点的例子。我们将首先检查一个字符串是否是回文，也就是说，它正向和反向读起来都一样。在 C++中一个简单的实现是取一个字符串，使用**std::reverse_copy**来反转它，然后比较原始字符串与其反转：

```cpp
bool isStringPalindrome(std::string value){
std::vector<char> characters(value.begin(), value.end());
std::vector<char> reversedCharacters;
std::reverse_copy(characters.begin(), characters.end(), std::back_insert_iterator(reversedCharacters));
return characters == reversedCharacters;
}
TEST_CASE("Palindrome"){
CHECK(isStringPalindrome("asddsa"));
CHECK(isStringPalindrome("12321"));
CHECK_FALSE(isStringPalindrome("123123"));
CHECK_FALSE(isStringPalindrome("asd"));
}
```

如果我们使这段代码对类型的兴趣减少呢？首先，我们会将参数类型改为 **auto**。然后，我们需要一种方法来反转它，而不会限制我们只能使用字符串输入。幸运的是，**ranges** 库有一个 **reverse_view** 我们可以使用。最后，我们需要比较初始值和反转后的值，再次不太多地限制类型。C++ 为我们提供了 **std::equal**。因此，我们最终得到以下代码，我们可以用它不仅用于字符串，还可以用于表示短语的一个 **vector<string**>，或者用于在枚举中定义的标记。让我们看看极端多态的实际应用：

```cpp
bool isPalindrome(auto value){
auto tokens = value | std::views::all;
auto reversedTokens = value | std::views::reverse;
return std::equal(tokens.begin(), tokens.end(), reversedTokens.begin());
};
enum Token{
X, Y
};
TEST_CASE("Extreme polymorphic palindrome"){
CHECK(isPalindrome(string("asddsa")));
CHECK(isPalindrome(vector<string>{"asd", "dsa", "dsa", "asd"}));
CHECK(isPalindrome(vector<Token>{Token::X, Token::Y, Token::Y, Token::X}));
}
```

也许我现在已经向你展示了为什么我觉得这种编程风格非常吸引人。如果我们忽略类型，或者使我们的函数具有极端的多态性，我们可以编写适用于未来情况的代码，而无需进行更改。权衡是代码在其推导出的类型中有限制，并且参数和函数的名称非常重要。例如，如果我向 **isPalindrome** 传递一个整数值，我将得到一个复杂的错误，而不是一个简单的错误，告诉我参数类型不正确。这是在我尝试传递整数时，我的计算机上 g++ 编译器输出的开始：

```cpp
In file included from testPalindrome.cpp:3:
Palindrome.h: In instantiation of 'bool isPalindrome(auto:21)
[with auto:21 = int]':
testPalindrome.cpp:30:2:   required from here
Palindrome.h:14:29: error: no match for 'operator|' (operand t
ypes are 'int' and 'const std::ranges::views::_All')
   14 |         auto tokens = value | std::views::all;
      |                       ~~~~~~^~~~~~~~~~~~~~~~~
```

现在取决于你：你更喜欢强类型还是极端多态的行为？两者都有其权衡和各自的应用领域。

# 摘要

在本章中，我们了解到我们可以使用多种范式来用 C++ 编程。我们简要地看了一些：函数式编程、元编程、确保编译时验证的类型以及极端多态。所有这些方法，以及标准的面向对象和结构化编程，在构建库或特定程序的各种情况下都是有用的。它们各自都有提供给那些想要尽可能多地了解自己技艺的程序员的东西。它们各自都有其权衡和软件开发世界中的实现。

我们已经表明，C++ 程序员可能只使用了语言的一个子集，而且不一定是面向对象的。相反，最好是尝试所有这些，充分利用 C++ 足够强大，可以提供如此多的选项的事实，并根据手头的任务进行选择和选择。

在下一章中，我们将看到 **main()** 函数可能实际上并不是我们应用程序的入口点。
