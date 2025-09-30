

# 第十三章：如何测试浮点数和自定义值

我们第一次遇到测试浮点值的需求是在*第五章*中，*添加更多确认类型*，我们创建了一个简单的解决方案，允许我们在误差范围内比较浮点值。我们需要小的误差范围，因为接近且可能看起来相同的浮点值几乎总是不完全相等。这些小的差异使得验证测试结果变得困难。

本章的主要内容包括：

+   更精确的浮点数比较

+   添加浮点数 Hamcrest 匹配器

+   编写自定义 Hamcrest 匹配器

我们将改进之前开发的简单解决方案，使其成为一种更好的比较浮点数的方法，这种方法更精确，适用于小数和大数。我们将使用更好的比较方法来处理早期的经典风格确认和新 Hamcrest 风格确认。

你还将在本章中学习如何创建自己的 Hamcrest 匹配器。我们将创建一个新的匹配器来测试不等式，而不是始终测试相等性，你将看到如何将一个匹配器包含在另一个匹配器中，这样你就可以更好地重用匹配器，而无需重复所有匹配器模板特化。

最后，你将学习如何创建另一个自定义简单匹配器，它将与其他匹配器略有不同，因为新的匹配器不需要预期值。

# 技术要求

本章中所有代码都使用基于任何现代 C++ 20 或更高版本编译器和标准库的标准 C++。代码基于并继续增强本书第一部分*测试 MVP*中的测试库。

你可以在此 GitHub 仓库中找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP

)

# 更精确的浮点数比较

当需要改进时，首先要寻找的是衡量当前设计的方法。在*第五章*中，*添加更多确认类型*，我们探讨了浮点数，我解释说，直接将任何浮点类型值（float、double 或 long double）与另一个浮点值进行比较是一个糟糕的想法。这种比较对小的舍入误差过于敏感，通常会导致两个值比较不相等。

在*第五章*中，我向你展示了如何给比较添加一个小范围，这样只要被比较的两个数值足够接近，误差的累积就不会影响比较。换句话说，只要两个值足够接近，它们就可以比较相等。

但应该使用多大的容差？我们只是挑选了一些小的数字，这个解决方案就有效了。我们将改进这个解决方案。现在，你已经熟悉了`Hamcrest.cpp`。

第一个函数将通过除以一个常数将浮点数转换为分数。我们将除以`10`，如下所示：

```cpp
template <typename T>
T calculateFraction (T input)
{
    T denominator {10};
    return input / denominator;
}
```

这是一个模板，所以它适用于 float、double 和 long double 类型。意图是输入是一个整数，这个函数将数字转换为十分之一。记得从*第五章*中，十分之一在二进制中没有精确的表示。将引入一点误差，但不会太多，因为我们只做一次除法计算。

我们需要另一个函数，通过做更多的工作来生成更大的误差范围，如下所示：

```cpp
template <typename T>
T accumulateError (T input)
{
    // First add many small amounts.
    T partialAmount {0.1};
    for (int i = 0; i < 10; ++i)
    {
        input += partialAmount;
    }
    // Then subtract to get back to the original.
    T wholeAmount {1};
    input -= wholeAmount;
    return input;
}
```

这个函数先加`1`然后减`1`，所以输入应该保持不变。但由于我们添加了许多等于`1`的小量，这个函数在计算过程中引入了许多错误。返回的结果应该接近原始的`input`，但并不相同。

最后的辅助函数将多次调用前两个函数，对许多不同的值进行计数，以查看结果相等多少次。函数看起来像这样：

```cpp
template <typename T>
int performComparisons (int totalCount)
{
    int passCount {0};
    for (int i = 0; i < totalCount; ++i)
    {
        T expected = static_cast<T>(i);
        expected = calculateFraction(expected);
        T actual = accumulateError(expected);
        if (actual == expected)
        {
            ++passCount;
        }
    }
    return passCount;
}
```

函数使用分数作为`预期`值，因为它应该有最少的误差。`预期`值与从累积许多小误差中得到的`实际`值进行比较。这两个值应该很接近，但并不完全相等。尽管如此，它们应该足够接近，以至于可以被认为是相等的。

谁定义了“足够接近”是什么意思？这完全取决于你自己的决定。在这本书中我们创建的测试可能允许比你的应用程序可以容忍的更多错误。阅读这一节后，你会了解如果需要更多或更少的容忍度，如何修改你的代码。对于如何比较浮点值，没有适用于所有应用程序的正确答案。你能做的最好的事情就是意识到自己的需求，并调整代码以适应这些需求。

`performComparisons`函数也使用`==`运算符而不带任何类型的容差。结果应该有很多不相等的结果。但有多少呢？让我们写一个测试来找出答案！

将此测试添加到`Hamcrest.cpp`的末尾：

```cpp
TEST("Test many float comparisons")
{
    int totalCount {1'000};
    int passCount = performComparisons<float>(totalCount);
    CONFIRM_THAT(passCount, Equals(totalCount));
}
```

测试将循环通过`1,000`个值，将每个值转换为十分之一，引入错误，并计算有多少个比较相等。结果真的很糟糕：

```cpp
------- Test: Test many float comparisons
Failed confirm on line 125
    Expected: 1000
    Actual  : 4
```

只有四个值足够接近，可以用标准相等运算符被认为是相等的。你可能根据你的计算机和编译器得到略微不同的结果。如果你得到不同的结果，那么这应该更有力地证明浮点比较是多么不可靠。那么双精度和长双精度类型呢？添加这两个测试来找出答案：

```cpp
TEST("Test many double comparisons")
{
    int totalCount {1'000};
    int passCount = performComparisons<double>(totalCount);
    CONFIRM_THAT(passCount, Equals(totalCount));
}
TEST("Test many long double comparisons")
{
    int totalCount {1'000};
    int passCount = performComparisons<long                     double>(totalCount);
    CONFIRM_THAT(passCount, Equals(totalCount));
}
```

结果同样糟糕，看起来像这样：

```cpp
------- Test: Test many double comparisons
Failed confirm on line 132
    Expected: 1000
    Actual  : 4
------- Test: Test many long double comparisons
Failed confirm on line 139
    Expected: 1000
    Actual  : 0
```

让我们在相等比较中添加一个边缘值，看看比较会变得多好。我们将从`Test.h`中现有的`confirm`重载中使用的值开始。其中一个重载看起来像这样：

```cpp
inline void confirm (
    float expected,
    float actual,
    int line)
{
    if (actual < (expected - 0.0001f) ||
        actual > (expected + 0.0001f))
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

我们感兴趣的是硬编码的浮点字面值。在这种情况下，它是`0.0001f`。我们只需要创建三个额外的辅助函数来返回这些值。请注意，double 和 long double 的重载与 float 类型有不同的值。将这些三个辅助函数放在`Hamcrest.cpp`中，在`performComparisons`函数之前，如下所示：

```cpp
constexpr float getMargin (float)
{
    return 0.0001f;
}
constexpr double getMargin (double)
{
    return 0.000001;
}
constexpr long double getMargin (long double)
{
    return 0.000001L;
}
```

这三个辅助函数将使我们能够为每种类型定制边缘值。它们各自接受一个浮点类型参数，该参数仅用于确定要调用的函数。我们实际上不需要或使用传递给函数的参数值。我们将在`performComparisons`模板内部调用这些辅助函数，该模板将根据模板的构建方式知道要使用哪种类型。

我们还将稍微改变带有边缘值的比较方式。以下是一个确认函数如何比较的示例：

```cpp
    if (actual < (expected - 0.0001f) ||
        actual > (expected + 0.0001f))
```

而不是这样，我们将从`expected`值中减去`actual`值，然后比较这个减法结果的绝对值与边缘值。我们需要在`Hamcrest.cpp`的顶部包含`cmath`以使用`abs`函数，并且我们很快也需要`limits`，如下所示：

```cpp
#include "../Test.h"
#include <cmath>
#include <limits>
```

现在，我们可以将`performComparisons`函数更改为使用边缘值，如下所示：

```cpp
template <typename T>
int performComparisons (int totalCount)
{
    int passCount {0};
    for (int i = 0; i < totalCount; ++i)
    {
        T expected = static_cast<T>(i);
        expected = calculateFraction(expected);
        T actual = accumulateError(expected);
        if (std::abs(actual - expected) < getMargin(actual))
        {
            ++passCount;
        }
    }
    return passCount;
}
```

在做出这些更改后，所有的测试都通过了，如下所示：

```cpp
------- Test: Test many float comparisons
Passed
------- Test: Test many double comparisons
Passed
------- Test: Test many long double comparisons
Passed
```

这意味着现在所有`1,000`个值都在一个很小的误差范围内匹配。这是在*第五章*中解释的相同解决方案。我们应该没问题，对吧？并不完全是这样。

问题在于，对于小数，边缘值很大，而对于大数，边缘值又太小。所有的测试都通过了，但这仅仅是因为我们有一个足够大的边缘值，使得很多比较都被视为相等。

为了看到这一点，让我们将比较从`performComparisons`函数中重构出来，使其检查在自己的函数中，如下所示：

```cpp
template <typename T>
bool compareEq (T lhs, T rhs)
{
    return std::abs(lhs - rhs) < getMargin(lhs);
}
template <typename T>
int performComparisons (int totalCount)
{
    int passCount {0};
    for (int i = 0; i < totalCount; ++i)
    {
        T expected = static_cast<T>(i);
        expected = calculateFraction(expected);
        T actual = accumulateError(expected);
        if (compareEq(actual, expected))
        {
            ++passCount;
        }
    }
    return passCount;
}
```

然后，我们可以编写一些测试来直接调用`compareEq`，如下所示：

```cpp
TEST("Test small float values")
{
    // Based on float epsilon = 1.1920928955078125e-07
    bool result = compareEq(0.000001f, 0.000002f);
    CONFIRM_FALSE(result);
}
TEST("Test large float values")
{
    // Based on float epsilon = 1.1920928955078125e-07
    bool result = compareEq(9'999.0f, 9'999.001f);
    CONFIRM_TRUE(result);
}
```

对于小浮点数的测试比较了两个明显不同的数字，但比较函数会将它们视为相等，测试失败。固定的边缘值将任何在`0.0001f`内的浮点值视为相等。我们希望这两个值比较时不相等，但我们的边缘值足够大，以至于它们被视为相等。

注释中提到的 *epsilon* 值是多少？我们很快就会开始使用实际的 epsilon 值，这就是为什么我建议你包含 `limits`。浮点数有一个称为 epsilon 的概念，这是为每种浮点类型在 `limits` 中定义的值。epsilon 值表示在 1.0 和 2.0 之间的相邻浮点值之间的最小距离。记住，浮点数不能表示每个可能的分数数，因此在可以表示的数之间有间隙。

如果你将只有固定小数位数的数字写在纸上，你也能看到相同的情况。比如说，你限制自己只使用小数点后两位数字。你可以写 `1.00`、`1.01` 和 `1.02`。这些都是相邻的值。实际上，`1.00` 和 `1.02` 是你只能使用小数点后两位数字表示的，最接近 `1.01` 的数值。那么一个像 `1.011` 这样的数呢？它肯定比 `1.02` 更接近 `1.01`，但我们不能写 `1.011`，因为它需要小数点后三位数字。我们实验中的 epsilon 值是 `0.01`。浮点数也有类似的问题，只是 epsilon 的值更小，不是一个简单的值如 `0.01`。

另一个复杂的问题是，随着数值的增大，相邻浮点数之间的距离增加，而随着数值的减小，距离减小。小浮点数的测试使用的是小数值，但数值比 epsilon 大得多。因为数值比 epsilon 大得多，我们希望测试失败。测试通过是因为我们的固定容限甚至比 epsilon 还大。

大浮点数的测试也失败了。它使用了两个相差 `0.001f` 的值，如果我们比较 `1.0f` 和 `1.001f`，这将是一个很大的差异。在小数值的情况下，`0.001f` 的差异足以使值比较结果不相等。但我们处理的是大数值——我们处理的是几乎 10,000 的数值！现在我们希望较大的数值被认为是相等的，因为小数部分在较大的数值中占的比例更小。测试失败是因为我们的固定容限没有考虑到数值较大，只关注了差异，而这个差异大于固定容限允许的范围。

我们也可以测试其他浮点类型。在为小浮点数和大浮点数添加的两个测试之后，立即添加这两个类似的测试，如下所示：

```cpp
TEST("Test small double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    bool result = compareEq(0.000000000000001,                   0.000000000000002);
    CONFIRM_FALSE(result);
}
TEST("Test large double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    bool result = compareEq(1'500'000'000'000.0,                   1'500'000'000'000.0003);
    CONFIRM_TRUE(result);
}
```

对于双精度类型，我们有一个比浮点数 epsilon 值小得多的不同 epsilon 值，并且我们可以使用更多的有效数字进行操作，因此我们可以使用更多位数的数字。当我们使用浮点数时，我们仅限于大约 7 位数字。使用双精度，我们可以使用大约 16 位数字的数字。请注意，使用双精度时，我们需要一个以万亿为单位的较大值，才能看到应该视为相等的 `0.0003` 的差异。

如果你想知道我是如何得到这些测试数字的，我只是选择了比 epsilon 大一个小数位的较小值测试的小数。对于较大值，我选择了一个较大的数字，并将其乘以（1 + epsilon）以得到要比较的另一个数字。然后我对另一个数字进行了一些四舍五入，使其更接近一些。我必须选择一个较大的起始数字，以确保它保持在每种类型允许的位数内。

由于我们正在使用长双精度 epsilon 值，因此小长双精度和大长双精度的测试看起来与双精度的测试相似。长双精度的测试如下：

```cpp
TEST("Test small long double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    bool result = compareEq(0.000000000000001L,                   0.000000000000002L);
    CONFIRM_FALSE(result);
}
TEST("Test large long double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    bool result = compareEq(1'500'000'000'000.0L,                   1'500'000'000'000.0003L);
    CONFIRM_TRUE(result);
}
```

双精度测试和长双精度测试之间的唯一区别是长双精度字面值末尾的 `L` 后缀。

在添加了所有六个针对小浮点数和大型浮点数类型测试之后，当运行时它们都失败了。

失败的原因对每种类型都是相同的。小值测试全部失败，因为固定的边距将值视为相等，而实际上它们不应该相等，并且大值测试在考虑大值时将值视为不相等，而实际上它们非常接近。事实上，大值彼此之间只有一个 epsilon 值的距离。大值尽可能接近，但不是完全相等。当然——长双精度的大值可能更接近，但我们通过使用来自双精度类型的较大 epsilon 来简化长双精度。

我们需要增强 `compareEq` 函数，以便对于小值，边距可以更小，对于大值，边距可以更大。当我们承担比较浮点数值的责任时，有很多细节需要处理。我们在 *第五章* 中跳过了额外的细节。我们甚至在这里也会跳过一些细节。如果你还没有意识到，处理浮点数值真的很困难。当你认为一切都在正常工作时，另一个细节就会出现，从而改变一切。

让我们先修复 `getMargin` 函数，使其返回每种类型的修改后的真实 epsilon 值，如下所示：

```cpp
constexpr float getMargin (float)
{
    // 4 is chosen to pass a reasonable amount of error.
    return std::numeric_limits<float>::epsilon() * 4;
}
constexpr double getMargin (double)
{
    // 4 is chosen to pass a reasonable amount of error.
    return std::numeric_limits<double>::epsilon() * 4;
}
constexpr long double getMargin (long double)
{
    // Use double epsilon instead of long double epsilon.
    // Double epsilon is already much bigger than
    // long double epsilon so we don't need to multiply it.
    return std::numeric_limits<double>::epsilon();
}
```

`getMargin`函数现在使用在`numeric_limits`中定义的类型`epsilon`值。边缘被调整以满足我们的需求。你可能想要乘以不同的数字，你可能想要为长双精度使用实际的`epsilon`值。我们想要比`epsilon`本身更大的边缘的原因是我们想要考虑那些彼此之间相差不止一个`epsilon`值的数值相等。我们想要为至少几个计算误差的累积留出更多空间。我们将`epsilon`乘以`4`以提供额外的空间，并且对于长双精度，我们使用双倍的`epsilon`，这可能已经足够了。但这些边缘对我们来说是有效的。

我们将在新的`compareEq`函数中使用更精确的边缘值，该函数看起来像这样：

```cpp
template <typename T>
bool compareEq (T lhs, T rhs)
{
    // Check for an exact match with operator == first.
    if (lhs == rhs)
    {
        return true;
    }
    // Subnormal diffs near zero are treated as equal.
    T diff = std::abs(lhs - rhs);
    if (diff <= std::numeric_limits<T>::min())
    {
        return true;
    }
    // The margin should get bigger with bigger absolute values.
    // We scale the margin up by the larger value or
    // leave the margin unchanged if larger is less than 1.
    lhs = std::abs(lhs);
    rhs = std::abs(rhs);
    T larger = (lhs > rhs) ? lhs : rhs;
    larger = (larger < 1.0) ? 1.0 : larger;
    return diff <= getMargin(lhs) * larger;
}
```

我喜欢为像这样的操作符类型函数使用参数名称`lhs`和`rhs`。这些缩写分别代表左端和右端。

考虑这两个数字：

```cpp
3 == 4
```

当进行这些比较时，`3`位于操作符的左侧，将是`lhs`参数，而`4`位于右侧，将是`rhs`参数。

总是有可能被比较的两个数值完全相等。所以，我们首先使用`==`操作符检查是否完全匹配。

`compareEq`函数继续检查两个数值之间的差异以获得一个*异常值*结果。记得我说过浮点数很复杂吗？可能有一整本书是关于浮点数学的，可能已经有几本书写过了。我不会过多解释异常值，只是说这是当浮点值非常接近零时如何表示的。我们将认为任何两个异常值都是相等的。

异常值也是用*比较你的数值之间*而不是*比较它们的差值与零*的一个很好的理由。你可能想知道问题是什么。`compareEq`函数中的代码不是从另一个值中减去一个值来得到差值吗？是的，它是这样做的。但我们的`compareEq`函数并不试图直接将差值与零进行比较。我们找出两个值中哪个更大，然后通过将边缘与较大的值相乘来缩放边缘。我们还在比较小于`1.0`的值时避免缩小边缘。

如果你有两个值需要比较，并且不是将它们传递给`compareEq`函数，而是传递它们的差值，并将差值与零进行比较，那么你就移除了`compareEq`函数进行缩放的能力，因为`compareEq`函数将只会看到一个很小的差值和与零的比较。

这里的教训是始终直接将你要比较的数值传递给`compareEq`函数，并让它通过考虑数值的大小来确定两个数值之间的差异。你会得到更准确的比较。

我们甚至可以使 `compareEq` 函数更加精细。也许我们可以考虑次正常值的符号，而不是将它们都视为相等，或者我们可以将边界值缩小更多，以便在处理次正常值时非常精确。这不是一本关于数学的书，所以我们将在 `compareEq` 函数中停止添加更多内容。

在对 `compareEq` 进行更改后，所有测试都通过了。我们现在有一个解决方案，允许少量的累积误差，并且当两个数字足够接近时，它们可以比较相等。该解决方案适用于非常小的数字和非常大的数字。下一节将把在这里探索的代码转换成一个更好的 Hamcrest 等价匹配器。

# 添加浮点数 Hamcrest 匹配器

我们在上一节中探讨了更好的浮点数比较，现在是时候在单元测试库中使用比较代码了。一些代码应该移动到 `Test.h` 中，那里更适合，然后可以被测试库使用。其余的代码应该保留在 `Hamcrest.cpp` 中，因为它是支持测试的代码。

需要移动的代码是 `compareEq` 函数和 `compareEq` 调用来获取边界的三个 `getMargin` 函数。我们还需要将 `cmath` 和 `limits` 的包含文件移动到 `Test.h` 中，如下所示：

```cpp
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <vector>
```

三个 `getMargin` 函数和 `compareEq` 函数可以被移动到 `Test.h` 文件中，紧接在第一个接受布尔值的 `confirm` 函数重写之前。移动的函数中的代码无需更改。只需从 `Hamcrest.cpp` 中剪切包含和函数，然后将代码粘贴到 `Test.h` 中。

我们不妨修复现有的浮点数经典 `confirm` 函数。这就是为什么我让你立即将 `compareEq` 函数移动到 `Test.h` 中，紧接在第一个 `confirm` 函数之前。对现有浮点数 `confirm` 函数的更改很简单。它们需要调用 `compareEq` 而不是使用硬编码的边界值，这些边界值不会缩放。更改后的浮点类型 `confirm` 函数如下所示：

```cpp
inline void confirm (
    float expected,
    float actual,
    int line)
{
    if (not compareEq(actual, expected))
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

接受双精度浮点数和长双精度浮点数的其他两个 `confirm` 函数应该被修改得相似。所有三个 `confirm` 函数都将根据 `expected` 和 `actual` 参数类型创建正确的 `compareEq` 模板。

我们应该构建并运行测试应用程序，以确保这次小的重构没有破坏任何东西。并且所有测试都通过了。我们现在有了更新的经典风格 `confirm` 函数，它们将更好地与浮点数比较一起工作。

尽管如此，我们可以使代码稍微好一些。我们有三个几乎完全相同的函数，它们的不同之处仅在于它们的参数类型。这三个函数的唯一原因是我们想要覆盖浮点类型的 `confirm` 函数。但是，由于我们正在使用 C++20，让我们使用 *concepts* 代替！Concepts 是一个新特性，我们在上一章专门化 `Equals` 匹配器以与字符数组和字符指针一起使用时已经开始使用它。Concepts 允许我们告诉编译器哪些类型是模板参数和函数参数的可接受类型。在上一章中，我们只使用 `requires` 关键字对模板参数施加一些限制。在本章中，我们将使用更多知名的概念。

我们需要在 `Test.h` 中包含这样的 `concepts`：

```cpp
#include <concepts>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <vector>
```

然后，我们可以用单个模板替换接受 float、double 和 long double 类型的三个 `confirm` 函数，这个模板使用 `floating_point` 概念，如下所示：

```cpp
template <std::floating_point T>
void confirm (
    T expected,
    T actual,
    int line)
{
    if (not compareEq(actual, expected))
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

这个新模板将只接受浮点类型，通过使 `expected` 和 `actual` 共享相同的类型 `T`，那么这两种类型必须相同。`floating_point` 的定义是 `concepts` 头文件中定义的已知概念之一。

现在我们已经使经典风格的确认工作正常，让我们让 Hamcrest 的 `Equals` 匹配器为浮点值工作。我们首先可以将 `Hamcrest.cpp` 中的三个大型浮点测试更改，停止直接调用 `compareEq`，而是使用 `CONFIRM_THAT` 宏，使它们看起来像这样：

```cpp
TEST("Test large float values")
{
    // Based on float epsilon = 1.1920928955078125e-07
    CONFIRM_THAT(9'999.0f, Equals(9'999.001f));
}
TEST("Test large double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    CONFIRM_THAT(1'500'000'000'000.0,                 Equals(1'500'000'000'000.0003));
}
TEST("Test large long double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    CONFIRM_THAT(1'500'000'000'000.0L,             Equals(1'500'000'000'000.0003L));
}
```

我们现在不会更改小浮点值的测试，因为我们还没有一个可以进行不等式比较的匹配器。解决方案可能很简单，只需在 `Equals` 前面加上 `not` 关键字，但让我们稍后再考虑这一点，因为我们将探索我们的选项在下一段中。

随着测试的更改，它们应该会失败，因为我们还没有将 `Equals` 匹配器专门化以对浮点类型执行不同的操作。构建和运行测试应用程序显示，这三个测试确实失败了，如下所示：

```cpp
------- Test: Test large float values
Failed confirm on line 152
    Expected: 9999.000977
    Actual  : 9999.000000
------- Test: Test small double values
Passed
------- Test: Test large double values
Failed confirm on line 165
    Expected: 1500000000000.000244
    Actual  : 1500000000000.000000
------- Test: Test small long double values
Passed
------- Test: Test large long double values
Failed confirm on line 178
    Expected: 1500000000000.000300
    Actual  : 1500000000000.000000
```

注意，总结报告中打印的预期值并不完全匹配测试中给出的浮点类型和双精度类型的字面值。长双精度确实显示了一个与测试中给出的值匹配的值。这种差异是因为浮点变量无法始终匹配精确值。差异在浮点数中更为明显，在双精度数中稍微不明显，而在长双精度数中则更接近期望值。

我们刚才采取的步骤遵循 TDD（测试驱动开发）。我们修改了现有测试而不是创建新测试，因为我们不期望调用者直接使用 `compareEq`。测试最初是编写来直接调用 `compareEq` 以表明我们为浮点类型提供了一个解决方案。将测试修改为期望的使用方法是正确的事情，然后，通过运行测试，我们可以看到失败。这是好的，因为我们预计测试会失败。如果测试通过了，那么我们就需要找到意外成功的原因。

让我们再次通过测试！我们需要一个能够处理浮点类型的 `Equals` 版本。我们将使用之前用于经典风格确认的 `floating_point` 概念来创建另一个版本的 `Equals`，该版本将为浮点类型调用 `compareEq`。将这个新的 `Equals` 特化版本放在 `Test.h` 中，紧接在处理字符指针的 `Equals` 之后，如下所示：

```cpp
template <std::floating_point T>
class Equals<T> : public Matcher
{
public:
    Equals (T const & expected)
    : mExpected(expected)
    { }
    bool pass (T const & actual) const
    {
        return compareEq(actual, mExpected);
    }
    std::string to_string () const override
    {
        return std::to_string(mExpected);
    }
private:
    T mExpected;
};
```

这就是我们需要的所有更改，以使测试再次通过。新的 `Equals` 特化版本接受任何浮点类型，并且编译器将优先选择它而不是通用 `Equals` 模板来处理浮点类型。`Equals` 的浮点版本调用 `compareEq` 来进行比较。我们也不必担心将传递给 `to_string` 的类型，因为我们知道我们将有一个内置的浮点类型。如果用户传递了一个其他类型，该类型已被创建为 `floating_point` 概念类型，则 `to_string` 假设可能会失败，但让我们现在保持代码尽可能简单，不要担心自定义浮点类型。

下一节将首先创建一个用于测试不等式的解决方案。我们将使用下一节中创建的解决方案来修改小的浮点 Hamcrest 测试。

# 编写自定义的 Hamcrest 匹配器

上一节以将 `Equals` 匹配器特化为调用 `compareEq` 以处理浮点类型结束。我们还修改了大的浮点值测试，因为它们可以使用 Hamcrest 风格和 `Equals` 匹配器。

我们没有改变小的浮点值测试，因为这些测试需要确保实际值和预期值不相等。

我们想要更新小的浮点值测试，并需要一个方法来测试不等值。也许我们可以创建一个新的匹配器，称为 `NotEquals`，或者我们可以在 `Equals` 匹配器前面放置 `not` 关键字。

如果可能，我想避免需要一个新的匹配器。我们并不真的需要任何新的行为——我们只需要翻转现有 `Equals` 匹配器的结果。让我们尝试修改小的浮点值测试，使其在 `Hamcrest.cpp` 中看起来像这样：

```cpp
TEST("Test small float values")
{
    // Based on float epsilon = 1.1920928955078125e-07
    CONFIRM_THAT(0.000001f, not Equals(0.000002f));
}
TEST("Test small double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    CONFIRM_THAT(0.000000000000001,             not Equals(0.000000000000002));
}
TEST("Test small long double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    CONFIRM_THAT(0.000000000000001L,             not Equals(0.000000000000002L));
}
```

唯一的改变是停止直接调用 `compareEq` 并使用 `CONFIRM_THAT` 宏和 `Equals` 匹配器。注意，我们通过在前面放置 `not` 关键字来反转 `Equals` 匹配器的结果。

它能构建吗？不。我们得到了类似的编译错误：

```cpp
MereTDD/tests/Hamcrest.cpp:145:29: error: no match for 'operator!' (operand type is 'MereTDD::Equals<float>')
  145 |     CONFIRM_THAT(0.000001f, not Equals(0.000002f));
      |                             ^~~~~~~~~~~~~~~~~~~~~
```

C++ 中的 `not` 关键字是 `operator !` 的快捷方式。通常在使用 TDD（测试驱动开发）时，下一步是修改代码以便测试可以构建。但我们遇到了一个问题。`not` 关键字期望类有一个 `operator !` 方法或者某种将类转换为布尔值的方式。这两种选择都需要类能够生成布尔值，但这并不是匹配器的工作方式。为了使匹配器知道结果是否应该通过，它需要知道 `actual`（实际）值。`confirm_that` 函数通过将所需的 `actual` 值作为参数传递给 `pass` 方法，将匹配器传递给 `pass` 方法。我们不能仅仅将匹配器本身转换为布尔结果。

我们将不得不创建一个 `NotEquals` 匹配器。虽然这不是我的首选，但从测试的角度来看，一个新的匹配器是可以接受的。让我们将测试改为如下所示：

```cpp
TEST("Test small float values")
{
    // Based on float epsilon = 1.1920928955078125e-07
    CONFIRM_THAT(0.000001f, NotEquals(0.000002f));
}
TEST("Test small double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    CONFIRM_THAT(0.000000000000001,             NotEquals(0.000000000000002));
}
TEST("Test small long double values")
{
    // Based on double epsilon = 2.2204460492503130808e-16
    CONFIRM_THAT(0.000000000000001L,             NotEquals(0.000000000000002L));
}
```

我想要避免创建一个新的匹配器另一个原因是避免像为 `Equals` 匹配器所做的那样专门化新的匹配器，但有一种方法可以创建一个名为 `NotEquals` 的匹配器，并基于我们为 `Equals` 匹配器所做的所有工作来实现它。我们只需要包含 `Equals` 匹配器并反转 `pass` 结果，如下所示：

```cpp
template <typename T>
class NotEquals : public Matcher
{
public:
    NotEquals (T const & expected)
    : mExpected(expected)
    { }
    template <typename U>
    bool pass (U const & actual) const
    {
        return not mExpected.pass(actual);
    }
    std::string to_string () const override
    {
        return "not " + mExpected.to_string();
    }
private:
    Equals<T> mExpected;
};
```

在 `Test.h` 中，在 `Equals` 匹配器的所有模板专门化之后添加 `NotEquals` 匹配器。

`NotEquals` 匹配器是一个包含 `Equals` 匹配器作为其 `mExpected` 数据成员的新匹配器类型。这将给我们所有为 `Equals` 匹配器所做的专门化。每当调用 `NotEquals::pass` 方法时，我们只需调用 `mExpected.pass` 方法并反转结果。每当调用 `to_string` 方法时，我们只需将 `"not "` 添加到 `mExpected` 提供的任何字符串中。

一个有趣的现象是，`pass` 方法本身就是一个基于类型 `U` 的模板。这将使我们能够根据一个字符串字面量构建一个 `NotEquals` 匹配器，然后使用 `std::string` 调用 `pass` 方法。

我们应该添加一个测试来使用 `NotEquals` 匹配器与字符串字面量和 `std::string`，或者更好的是，扩展现有的测试。我们已经有两个测试与字符串、字符串字面量和字符指针一起工作。这两个测试都在 `Hamcrest.cpp` 中。第一个测试应该看起来像这样：

```cpp
TEST("Test hamcrest style string confirms")
{
    std::string s1 = "abc";
    std::string s2 = "abc";
    CONFIRM_THAT(s1, Equals(s2));       // string vs. string
    CONFIRM_THAT(s1, Equals("abc"));    // string vs. literal
    CONFIRM_THAT("abc", Equals(s1));    // literal vs. string
    // Probably not needed, but this works too.
    CONFIRM_THAT("abc", Equals("abc")); // literal vs. literal
    std::string s3 = "def";
    CONFIRM_THAT(s1, NotEquals(s3));       // string vs. string
    CONFIRM_THAT(s1, NotEquals("def"));    // string vs. literal
    CONFIRM_THAT("def", NotEquals(s1));    // literal vs. string
}
```

第二个测试应该修改为如下所示：

```cpp
TEST("Test hamcrest style string pointer confirms")
{
    char const * sp1 = "abc";
    std::string s1 = "abc";
    char const * sp2 = s1.c_str();    // avoid sp1 and sp2 being same
    CONFIRM_THAT(sp1, Equals(sp2));   // pointer vs. pointer
    CONFIRM_THAT(sp2, Equals("abc")); // pointer vs. literal
    CONFIRM_THAT("abc", Equals(sp2)); // literal vs. pointer
    CONFIRM_THAT(sp1, Equals(s1));    // pointer vs. string
    CONFIRM_THAT(s1, Equals(sp1));    // string vs. pointer
    char const * sp3 = "def";
    CONFIRM_THAT(sp1, NotEquals(sp3));   // pointer vs. pointer
    CONFIRM_THAT(sp1, NotEquals("def")); // pointer vs. literal
    CONFIRM_THAT("def", NotEquals(sp1)); // literal vs. pointer
    CONFIRM_THAT(sp3, NotEquals(s1));    // pointer vs. string
    CONFIRM_THAT(s1, NotEquals(sp3));    // string vs. pointer
}
```

构建和运行测试应用程序显示所有测试都通过了。我们没有添加新的测试，而是能够修改现有的测试，因为这两个现有的测试都集中在字符串和字符指针类型上。`NotEquals` 匹配器完美地融入了现有的测试中。

拥有`Equals`和`NotEquals`匹配器比经典风格的确认方式给我们提供了更多，我们可以通过创建另一个匹配器来更进一步。你还可以创建匹配器来在你的测试项目中执行任何你想要的操作。我们将在`MereTDD`命名空间中创建一个新的匹配器，但你也可以将你的匹配器放在你自己的命名空间中。我们将创建的匹配器将测试以确保一个整数是偶数。我们将称这个匹配器为`IsEven`，我们可以在`Hamcrest.cpp`中编写几个测试，如下所示：

```cpp
TEST("Test even integral value")
{
    CONFIRM_THAT(10, IsEven<int>());
}
TEST("Test even integral value confirm failure")
{
    CONFIRM_THAT(11, IsEven<int>());
}
```

你会注意到`IsEven`匹配器的一个不同之处：它不需要预期的值。匹配器只需要传递给它的实际值，以确认实际值是否为偶数。因为创建测试中的`IsEven`匹配器时没有东西可以传递给构造函数，我们需要指定类型，如下所示：

```cpp
IsEven<int>()
```

第二次测试应该失败，我们将利用这个失败来获取确切的错误信息，以便我们可以将测试转换为预期的失败。但首先我们需要创建一个`IsEven`匹配器。`IsEven`类可以放在`Test.h`中，紧随`NotEquals`匹配器之后，如下所示：

```cpp
template <std::integral T>
class IsEven : public Matcher
{
public:
    IsEven ()
    { }
    bool pass (T const & actual) const
    {
        return actual % 2 == 0;
    }
    std::string to_string () const override
    {
        return "is even";
    }
};
```

我想给你展示一个真正简单的自定义匹配器的例子，这样你就会知道它们并不都需要复杂或者有多个模板特化。`IsEven`匹配器只测试`pass`方法中的`actual`值以确保它是偶数，而`to_string`方法返回一个固定的字符串。

构建和运行显示，偶数值测试通过，而预期失败的测试失败，如下所示：

```cpp
------- Test: Test even integral value
Passed
------- Test: Test even integral value confirm failure
Failed confirm on line 185
    Expected: is even
    Actual  : 11
```

有错误信息，我们可以修改偶数确认失败测试，使其以预期的失败通过，如下所示：

```cpp
TEST("Test even integral value confirm failure")
{
    std::string reason = "    Expected: is even\n";
    reason += "    Actual  : 11";
    setExpectedFailureReason(reason);
    CONFIRM_THAT(11, IsEven<int>());
}
```

现在构建和运行显示，两个测试都通过了。一个成功通过，另一个以预期的失败通过，如下所示：

```cpp
------- Test: Test even integral value
Passed
------- Test: Test even integral value confirm failure
Expected failure
    Expected: is even
    Actual  : 11
```

这就是制作自定义匹配器的全部内容！你可以为你的类创建匹配器，或者为新的行为添加自定义匹配器。也许你想要验证一个数字只有一定数量的数字，一个字符串以某个给定的文本前缀开始，或者一个日志消息包含某个特定的标签。你还记得在*第十章*，*深入理解 TDD 过程*中，我们不得不通过写入文件然后扫描文件来验证标签吗？我们可以有一个自定义匹配器来查找标签。

# 摘要

Hamcrest 风格确认的主要优点之一是它们可以通过自定义匹配器进行扩展。还有什么比通过浮点数确认来探索这种能力更好的方法呢？因为比较浮点值没有唯一最佳的方式，你可能需要一个针对你特定需求进行调优的解决方案。在本章中，你了解了一种良好的通用浮点数比较技术，它将小的误差范围进行缩放，使得较大的浮点值在值变大时可以允许有更大的差异，但仍被视为相等。

如果这个通用解决方案不能满足你的需求，你现在知道如何创建自己的匹配器，使其正好满足你的需求。

而扩展匹配器的功能并不仅限于浮点值。你可能有自己的自定义行为需要确认，在阅读本章之后，你现在知道如何创建一个自定义匹配器来完成你所需要的工作。

并非所有匹配器都需要庞大且复杂，以及拥有多个模板特化。你看到了一个非常简单的自定义匹配器的例子，它确认一个数字是否为偶数。

我们还很好地利用了 C++20 中新引入的概念特性，该特性允许你轻松地指定对模板类型的约束。我们在本章中很好地利用了概念，以确保浮点数匹配器仅适用于浮点类型，并且 `IsEven` 匹配器仅适用于整型类型。你同样可以在你的匹配器中使用概念，这有助于你控制匹配器的使用方式。

下一章将探讨如何测试服务，并介绍一个使用本书迄今为止开发的所有代码的新服务项目。
