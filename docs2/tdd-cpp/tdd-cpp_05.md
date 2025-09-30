

# 添加更多确认类型

上一章介绍了确认，并展示了如何使用它们来验证测试中的布尔值是否与预期相符。这一章通过基于学校评分示例的探索性代码来完成。我们将更改评分示例以更好地适应测试库，并添加你可以在确认中使用的一些附加类型。

在本章中，我们将涵盖以下主要主题：

+   修复布尔确认

+   确认相等性

+   修改代码以修复行号导致测试失败的问题

+   添加更多确认类型

+   确认字符串字面量

+   确认浮点值

+   如何编写确认

这些附加类型为确认添加了一些新的变化，在本章中，你将学习如何应对。到本章结束时，你将能够编写可以验证任何需要测试的结果的测试。

# 技术要求

本章中的所有代码都使用基于任何现代 C++ 17 或更高版本编译器和标准库的标准 C++。代码基于前几章并继续发展。

你可以在这个 GitHub 仓库中找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

# 修复布尔确认

上一章探讨了确认一个值的意义。然而，它留下了一些我们需要修复的临时代码。让我们首先修复`Confirm.cpp`中的代码，使其不再引用学校评分。我们希望确认可以与如 bool 这样的类型一起工作。这就是为什么我们现在的确认宏被称为`CONFIRM_TRUE`和`CONFIRM_FALSE`。宏名称中提到的 true 和 false 是预期值。此外，这些宏接受一个参数，即实际值。

我们可以不用关于通过成绩的测试，而是用关于布尔值的测试来替换它：

```cpp
TEST("Test bool confirms")
{
    bool result = isNegative(0);
    CONFIRM_FALSE(result);
    result = isNegative(-1);
    CONFIRM_TRUE(result);
}
```

新的测试清楚地说明了它测试的内容，需要一个名为`isNegative`的新辅助函数，而不是之前的确定成绩是否通过的功能。我想找到一个简单且可以生成具有明显预期值的结果的函数。`isNegative`函数替换了之前的`isPassingGrade`函数，其外观如下：

```cpp
bool isNegative (int value)
{
    return value < 0;
}
```

这是一个简单的更改，移除了基于成绩的探索性代码，现在它适合测试库。现在，在下一节中，我们可以继续使用测试相等性的确认。

# 确认相等性

从某种意义上说，布尔确认确实是在测试相等性。它们确保实际布尔值等于预期值。这也是本章引入的新确认将要做的。唯一的区别是，`CONFIRM_TRUE`和`CONFIRM_FALSE`确认不需要接受预期值参数。它们的预期值隐含在它们的名称中。我们可以为布尔类型做这件事，因为只有两种可能的值。

然而，假设我们想要验证实际整数值是否等于 1。我们真的想要一个名为`CONFIRM_1`的宏吗？我们需要数十亿个宏来为每个可能的 32 位整型值创建宏，对于 64 位整型值则需要更多。使用这种方法验证文本字符串以确保它们与预期值匹配变得不可能。

相反，我们只需要修改其他类型的宏，以便接受预期值和实际值。如果这两个值不相等，则宏应该导致测试失败，并显示适当的错误消息，解释期望值和实际接收到的值。

宏不是用来解析不同类型的。它们只执行简单的文本替换。我们需要真正的 C++函数才能正确地与我们将要检查的不同类型一起工作。此外，我们还可以将现有的布尔宏更改为调用函数，而不是直接在宏中定义代码。以下是我们在上一章中定义的现有布尔宏：

```cpp
#define CONFIRM_FALSE( actual ) \
if (actual) \
{ \
    throw MereTDD::BoolConfirmException(false, __LINE__); \
}
#define CONFIRM_TRUE( actual ) \
if (not actual) \
{ \
    throw MereTDD::BoolConfirmException(true, __LINE__); \
}
```

我们需要做的是将`if`和`throw`语句移动到函数中。我们只需要一个函数来处理真和假，它将看起来像这样：

```cpp
inline void confirm (
    bool expected,
    bool actual,
    int line)
{
    if (actual != expected)
    {
        throw BoolConfirmException(expected, line);
    }
}
```

这个函数可以放在`MereTDD`命名空间内的`Test.h`文件中，在`TestBase`定义之前。该函数需要是内联的，并且由于它现在位于同一命名空间中，因此不再需要使用命名空间来限定异常。

此外，你可以更清楚地看到，即使是对于布尔值，这也是一个相等比较。该函数检查确保实际值等于预期值，如果不等于，则抛出异常。宏可以简化为调用新函数，如下所示：

```cpp
#define CONFIRM_FALSE( actual ) \
    MereTDD::confirm(false, actual, __LINE__)
#define CONFIRM_TRUE( actual ) \
    MereTDD:: confirm(true, actual, __LINE__)
```

构建和运行结果显示所有测试都通过了，我们现在可以添加额外的类型来确认。让我们从`Confirm.cpp`中的新测试开始，用于整型值，如下所示：

```cpp
TEST("Test int confirms")
{
    int result = multiplyBy2(0);
    CONFIRM(0, result);
    result = multiplyBy2(1);
    CONFIRM(2, result);
    result = multiplyBy2(-1);
    CONFIRM(-2, result);
}
```

与布尔值不同，此代码测试整数值。它使用一个新的辅助函数，这个函数应该很容易理解，它只是将一个值乘以 2。我们需要在文件顶部声明这个新辅助函数，如下所示：

```cpp
int multiplyBy2 (int value)
{
    return value * 2;
}
```

测试目前还不能构建。这是可以接受的，因为当我们使用 TDD 方法时，我们希望首先关注使用。这种使用看起来很好。它将使我们能够确认任何整数值都等于我们期望它成为的值。让我们创建`CONFIRM`宏，并将其放置在两个现有的确认真和假的宏之后，如下所示：

```cpp
#define CONFIRM_FALSE( actual ) \
    MereTDD::confirm(false, actual, __LINE__)
#define CONFIRM_TRUE( actual ) \
    MereTDD:: confirm(true, actual, __LINE__)
#define CONFIRM( expected, actual ) \
    MereTDD::confirm(expected, actual, __LINE__)
```

将宏更改为调用函数现在真的很有成效。`CONFIRM`宏需要一个额外的参数来传递预期值，并且可以调用相同的函数名。然而，它是如何调用相同的函数呢？嗯，那是因为我们将要重载函数。我们现在拥有的只适用于布尔值。这就是为什么我们转向了一个可以利用数据类型的设计。我们只需要提供另一个`confirm`的实现，使其可以重载以处理整数，如下所示：

```cpp
inline void confirm (
    int expected,
    int actual,
    int line)
{
    if (actual != expected)
    {
        throw ActualConfirmException(expected, actual, line);
    }
}
```

这几乎与现有的`confirm`函数相同。它接受预期和实际参数为整数，而不是布尔值，并将抛出一个新的异常类型。引入新异常类型的原因是我们可以格式化一个将显示预期和实际值的失败消息。`BoolConfirmException`类型将仅用于布尔值，并将格式化一个只提及预期的消息。此外，新的`ActualConfirmException`类型将格式化一个提及预期和实际值的消息。

新的异常类型如下：

```cpp
class ActualConfirmException : public ConfirmException
{
public:
    ActualConfirmException (int expected, int actual, int line)
    : mExpected(std::to_string(expected)),
      mActual(std::to_string(actual)),
      mLine(line)
    {
        formatReason();
    }
private:
    void formatReason ()
    {
        mReason =  "Confirm failed on line ";
        mReason += std::to_string(mLine) + "\n";
        mReason += "    Expected: " + mExpected + "\n";
        mReason += "    Actual  : " + mActual;
    }
    std::string mExpected;
    std::string mActual;
    int mLine;
};
```

你可能想知道为什么新的异常类型将预期和实际值存储为字符串。构造函数接受整数，然后在格式化原因之前将整数转换为字符串。这是因为我们将添加多个数据类型，我们实际上不需要做任何不同的事情。每种类型只需要在测试失败时根据字符串显示描述性消息。

我们不需要使用预期或实际值进行任何计算。它们只需要被格式化为可读的消息。此外，这种设计将使我们能够使用单个异常处理所有除了布尔值之外的数据类型。我们也可以为布尔值使用这个新异常，但对于布尔值，消息不需要提及实际值。因此，我们将保留现有的布尔值异常，并使用这个新的异常类型来处理其他所有情况。

通过将预期和实际值存储为字符串，我们需要的只是为每个我们想要支持的新数据类型提供一个重载构造函数。每个构造函数都可以将预期和实际值转换为字符串，然后可以将其格式化为可读的消息。这比有一个`IntActualConfirmException`类、一个`StringActualConfirmException`类等等要好。

我们可以再次构建和运行测试。布尔和整数测试的结果如下：

```cpp
---------------
Test bool confirms
Passed
---------------
Test int confirms
Passed
---------------
```

那么，如果确认失败会发生什么？嗯，我们在上一章已经看到了失败的布尔确认是什么样子。但我们还没有任何针对失败情况的测试。我们应该添加它们，并使它们成为预期失败，以便可以捕获行为。即使是失败也应该进行测试，以确保它仍然是失败。如果将来我们对代码进行了某些更改，将失败变成了成功，那将是一个破坏性的变化，因为失败应该是预期的。让我们向`Confirm.cpp`添加几个新的测试，如下所示：

```cpp
TEST("Test bool confirm failure")
{
    bool result = isNegative(0);
    CONFIRM_TRUE(result);
}
TEST("Test int confirm failure")
{
    int result = multiplyBy2(1);
    CONFIRM(0, result);
}
```

我们获取预期的失败，它们看起来像这样：

```cpp
---------------
Test bool confirm failure
Failed
Confirm failed on line 41
    Expected: true
---------------
Test int confirm failure
Failed
Confirm failed on line 47
    Expected: 0
    Actual  : 2
---------------
```

下一步是设置预期的错误消息，以便这些测试通过而不是失败。然而，有一个问题。行号是错误消息的一部分。我们希望行号在测试结果中显示。但这意味着我们也必须在预期的失败消息中包含行号，以便将失败视为通过。这为什么会成为问题呢？嗯，那是因为每次测试被移动，甚至当其他测试被添加或删除时，行号都会改变。我们不想不得不更改预期的错误消息，因为这不是错误真正的一部分。行号告诉我们错误发生的位置，不应该成为错误发生原因的一部分。

在下一节中，我们将通过一些重构来修复行号。

# 将测试失败与行号解耦

我们需要从确认失败原因中删除行号，以便测试可以给出一个不会随着测试移动或转移到源代码文件的不同位置而改变的预期失败原因。

这种类型的更改被称为*重构*。我们不会做出导致代码中出现不同或新行为的更改。至少，这是目标。使用 TDD 将帮助你重构代码，因为你应该已经为所有重要方面都设置了测试。

使用适当的测试进行重构可以让你验证没有任何东西发生变化。很多时候，为了避免引入新的错误，人们会避免在没有 TDD 的情况下进行重构。这往往会使问题变得更严重，因为重构被推迟或完全避免。

我们在行号上遇到了问题。我们本可以忽略这个问题，并在任何更改发生时只需更新测试中的预期失败消息中的新行号。但这是不正确的，只会导致更多的工作和脆弱的测试。随着测试的增加，问题只会变得更糟。我们真的应该现在解决这个问题。因为我们遵循 TDD，我们可以确信我们即将做出的更改不会破坏已经测试过的任何东西。或者，至少，如果它确实破坏了，我们会知道并立即修复任何破坏。

第一步是在`Test.cpp`中的`ConfirmException`基类中添加行号信息：

```cpp
class ConfirmException
{
public:
    ConfirmException (int line)
    : mLine(line)
    { }
    virtual ~ConfirmException () = default;
    std::string_view reason () const
    {
        return mReason;
    }
    int line () const
    {
        return mLine;
    }
protected:
    std::string mReason;
    int mLine;
};
```

然后，在`runTests`函数中，我们可以从确认异常中获取行号，并使用它来设置测试中的失败位置，如下所示：

```cpp
        try
        {
            test->runEx();
        }
        catch (ConfirmException const & ex)
        {
            test->setFailed(ex.reason(), ex.line());
        }
```

即使我们没有从测试开始，请注意我仍然在遵循 TDD 方法来编写代码，因为我希望在完全实现之前使用它。这是一个很好的例子，因为我最初考虑向测试类添加一个新方法。它被称为`setFailedLocation`。但这样做让现有的`setFailed`方法看起来很奇怪。我几乎将`setFailed`重命名为`setFailedReason`，这将意味着它需要在其他被调用的地方进行更改。相反，我决定向现有的`setFailed`方法添加一个额外的行号参数。我还决定给参数一个默认值，这样其他代码就不需要更改。这很有意义，并允许调用者自行设置失败原因，或者如果知道行号，则可以同时设置。

我们需要向`TestBase`类添加一个行号数据成员。行号将仅适用于确认，因此它将被称为`mConfirmLocation`，如下所示：

```cpp
    std::string mName;
    bool mPassed;
    std::string mReason;
    std::string mExpectedReason;
    int mConfirmLocation;
};
```

新的数据成员需要在`TestBase`构造函数中初始化。我们将使用-1 的值来表示行号位置不适用：

```cpp
    TestBase (std::string_view name)
    : mName(name), mPassed(true), mConfirmLocation(-1)
    { }
```

我们需要像这样向`setFailed`方法添加行号参数：

```cpp
    void setFailed (std::string_view reason,          int confirmLocation = -1)
    {
        mPassed = false;
        mReason = reason;
        mConfirmLocation = confirmLocation;
    }
```

此外，我们还需要为确认位置添加一个新的 getter 方法，如下所示：

```cpp
    int confirmLocation () const
    {
        return mConfirmLocation;
    }
```

这将允许`runTests`函数在捕获到确认异常时设置行号，并且测试将能够记住行号。在`runTests`的末尾，当将失败消息发送到输出时，我们需要测试`confirmLocation`，并根据是否有行号来更改输出，如下所示：

```cpp
        else
        {
            ++numFailed;
            if (test->confirmLocation() != -1)
            {
                output << "Failed confirm on line "
                    << test->confirmLocation() << "\n";
            }
            else
            {
                output << "Failed\n";
            }
            output << test->reason()
                << std::endl;
        }
```

这也将修复确认中的一个小问题。之前，测试结果打印了一条说测试失败的行，然后又打印了一条说确认失败的行。新的代码将只显示一个通用的失败消息或带有行号的确认失败消息。

我们还没有完成。我们需要更改派生异常类构造函数，以初始化基类行号，并停止将行号作为原因的一部分。`BoolConfirmException`的构造函数如下所示：

```cpp
    BoolConfirmException (bool expected, int line)
    : ConfirmException(line)
    {
        mReason += "    Expected: ";
        mReason += expected ? "true" : "false";
    }
```

此外，`ActualConfirmException`类需要在整个文件中进行更改。构造函数需要使用行号初始化基类，格式需要更改，并且可以删除行号数据成员，因为它现在在基类中。类看起来如下所示：

```cpp
class ActualConfirmException : public ConfirmException
{
public:
    ActualConfirmException (int expected, int actual, int line)
    : ConfirmException(line),
      mExpected(std::to_string(expected)),
      mActual(std::to_string(actual))
    {
        formatReason();
    }
private:
    void formatReason ()
    {
        mReason += "    Expected: " + mExpected + "\n";
        mReason += "    Actual  : " + mActual;
    }
    std::string mExpected;
    std::string mActual;
};
```

我们可以再次构建并运行，仍然显示预期的失败。失败原因的格式与之前略有不同，如下所示：

```cpp
---------------
Test bool confirm failure
Failed confirm on line 41
    Expected: true
---------------
Test int confirm failure
Failed confirm on line 47
    Expected: 0
    Actual  : 2
---------------
```

它看起来几乎一样，这是好的。现在我们可以设置预期的失败消息，而不用担心行号，如下所示：

```cpp
TEST("Test bool confirm failure")
{
    std::string reason = "    Expected: true";
    setExpectedFailureReason(reason);
    bool result = isNegative(0);
    CONFIRM_TRUE(result);
}
TEST("Test int confirm failure")
{
    std::string reason = "    Expected: 0\n";
    reason += "    Actual  : 2";
    setExpectedFailureReason(reason);
    int result = multiplyBy2(1);
    CONFIRM(0, result);
}
```

注意，预期的失败原因需要格式化，以与测试失败时显示的内容完全匹配。这包括用于缩进的空格和新行。一旦设置了预期的失败原因，所有的测试就会再次通过，如下所示：

```cpp
---------------
Test bool confirm failure
Expected failure
    Expected: true
---------------
Test int confirm failure
Expected failure
    Expected: 0
    Actual  : 2
---------------
```

这两个测试都预期会失败，并且被视为通过。现在我们可以继续添加更多确认类型。

# 添加更多确认类型

目前，我们可以在测试中确认 bool 和 int 值。我们需要更多，所以下一步应该添加什么？让我们添加对 long 类型的支持。它与 int 类似，在许多平台上将有效地相同。即使它可能或可能不使用与 int 相同数量的位，对于 C++ 编译器来说，它是一个不同的类型。我们可以通过在 `Confirm.cpp` 中添加一个基本的测试来开始，这个测试像这样测试 long 类型：

```cpp
TEST("Test long comfirms")
{
    long result = multiplyBy2(0L);
    CONFIRM(0L, result);
    result = multiplyBy2(1L);
    CONFIRM(2L, result);
    result = multiplyBy2(-1L);
    CONFIRM(-2L, result);
}
```

测试调用相同的 `multiplyBy2` 辅助函数，因为它不是在整个过程中使用 long 类型。我们通过添加 `L` 后缀以 long 文字值开始。这些值被转换为 int 以传递给 `multiplyBy2`。返回值也是一个 int，它被转换为 long 以分配给 `result`。让我们通过创建一个接受 long 类型并返回 long 类型的重载 `multiplyBy2` 版本来防止所有这些额外的转换：

```cpp
long multiplyBy2 (long value)
{
    return value * 2L;
}
```

如果我们现在尝试构建，将会出现错误，因为编译器不知道应该调用哪个重载的 `confirm` 函数。唯一可用的选择是将预期的长值和实际值转换为 int 或 bool。这两种选择都不匹配，编译器将调用视为模糊的。记住，`CONFIRM` 宏会被转换成对重载的 `confirm` 函数的调用。

我们可以通过添加一个新的重载 `confirm` 版本，该版本使用 long 参数来解决这个问题。然而，更好的解决方案是将现有的使用 int 参数的 `confirm` 版本改为模板，如下所示：

```cpp
template <typename T>
void confirm (
    T const & expected,
    T const & actual,
    int line)
{
    if (actual != expected)
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

我们仍然有使用 bool 参数的 `confirm` 版本。模板将匹配 int 和 long 类型。此外，模板还将匹配我们尚未测试的类型。新的模板 `confirm` 方法在创建要抛出的异常时也会将类型转换为 `std::string`。在 *第十二章*，*创建更好的测试确认*，你会看到我们在将预期值和实际值转换为字符串的方式上存在问题。或者至少，有更好的方法。我们目前的方法是可行的，但仅适用于可以传递给 `std::to_string` 的数值类型。

让我们更新 `ActualConfirmException` 构造函数，使其使用字符串，我们将在 `confirm` 函数内部调用 `std::to_string`。构造函数看起来像这样：

```cpp
    ActualConfirmException (
        std::string_view expected,
        std::string_view actual,
        int line)
    : ConfirmException(line),
      mExpected(expected),
      mActual(actual)
    {
        formatReason();
    }
```

一切构建正常，所有测试都通过了。我们可以在 `Confirm.cpp` 中添加一个新的测试，用于测试 long 失败，如下所示：

```cpp
TEST("Test long confirm failure")
{
    std::string reason = "    Expected: 0\n";
    reason += "    Actual  : 2";
    setExpectedFailureReason(reason);
    long result = multiplyBy2(1L);
    CONFIRM(0L, result);
}
```

失败原因字符串与 int 相同，即使我们正在测试 long 类型。新测试的测试结果如下：

```cpp
---------------
Test long confirm failure
Expected failure
    Expected: 0
    Actual  : 2
---------------
```

让我们尝试一个会显示不同结果的类型。`long long` 类型可以肯定地存储比 int 更大的数值。下面是 `Confirm.cpp` 中的一个新测试，用于测试 `long long` 值：

```cpp
TEST("Test long long confirms")
{
    long long result = multiplyBy2(0LL);
    CONFIRM(0LL, result);
    result = multiplyBy2(10'000'000'000LL);
    CONFIRM(20'000'000'000LL, result);
    result = multiplyBy2(-10'000'000'000LL);
    CONFIRM(-20'000'000'000LL, result);
}
```

对于`long long`类型，我们可以有大于最大 32 位有符号值的值。代码使用单引号来使较大的数字更容易阅读。编译器忽略单引号，但它们帮助我们视觉上分隔每一组千位数。此外，后缀`LL`告诉编译器将字面值视为`long long`类型。

这个通过测试的结果看起来和其他的一样：

```cpp
---------------
Test long long confirms
Passed
---------------
```

我们需要查看一个`长长`的失败测试结果来查看更大的数字。这里是一个失败测试：

```cpp
TEST("Test long long confirm failure")
{
    std::string reason = "    Expected: 10000000000\n";
    reason += "    Actual  : 20000000000";
    setExpectedFailureReason(reason);
    long long result = multiplyBy2(10'000'000'000LL);
    CONFIRM(10'000'000'000LL, result);
}
```

由于我们不使用分隔符格式化输出，我们需要使用不带逗号的纯数字文本格式。这可能是最好的方式，因为一些地区使用逗号，而一些地区使用点。注意，我们不做任何格式化尝试，所以期望的失败消息也不使用任何格式化。

现在，我们可以看到失败描述确实与较大的数字匹配，看起来像这样：

```cpp
---------------
Test long long confirm failure
Expected failure
    Expected: 10000000000
    Actual  : 20000000000
---------------
```

我想强调关于失败测试的一个重要观点。它们故意使用不正确的期望值来强制失败。你不会在测试中这样做。但你也无需编写你希望失败的测试。我们希望这些测试失败，以便我们可以验证测试库能够正确地检测和处理任何失败。因此，我们将这些失败视为通过。

我们可以继续添加对短整型、字符和所有无符号版本的测试。然而，在这个点上，这变得不再有趣，因为我们只是在测试模板函数是否正常工作。相反，让我们专注于使用非模板代码的类型，这些代码已经被编写来正常工作。

这里是对字符串类型的一个简单测试：

```cpp
TEST("Test string confirms")
{
    std::string result = "abc";
    std::string expected = "abc";
    CONFIRM(expected, result);
}
```

而不是编写一个返回字符串的假辅助方法，这个测试只是声明了两个字符串，并将使用一个作为实际值，另一个作为期望值。通过将两个字符串都初始化为相同的文本，我们期望它们相等，所以我们调用`CONFIRM`来确保它们相等。

当你编写测试时，你将想要给`result`分配一个从你正在测试的函数或方法中获得的值。我们的目标是测试`CONFIRM`宏和底层测试库代码是否正常工作。因此，我们可以跳过被测试的函数，直接使用两个字符串值进行宏测试，其中我们知道期望的结果。

这看起来像是一个合理的测试。而且确实是。但它无法编译。问题是`confirm`模板函数试图在提供的值上调用`std::to_string`。当值已经是字符串时，这没有意义。

我们需要的是一个新的`confirm`重载，它使用字符串。我们实际上会创建两个重载，一个用于字符串视图，一个用于字符串。第一个重载函数看起来像这样：

```cpp
inline void confirm (
    std::string_view expected,
    std::string_view actual,
    int line)
{
    if (actual != expected)
    {
        throw ActualConfirmException(
            expected,
            actual,
            line);
    }
}
```

这个第一个函数接受字符串视图，与模板方法相比，在处理字符串视图时将是一个更好的匹配。然后，它将给定的字符串传递给`ActualConfirmException`构造函数，而不尝试调用`std::to_string`，因为它们已经是字符串。

第二个重载函数看起来像这样：

```cpp
inline void confirm (
    std::string const & expected,
    std::string const & actual,
    int line)
{
    confirm(
        std::string_view(expected),
        std::string_view(actual),
        line);
}
```

这个第二个函数接受常量字符串引用，与模板方法相比，在处理字符串时将是一个更好的匹配。然后，它将字符串转换为字符串视图并调用第一个函数。

现在，我们可以添加一个字符串失败测试，如下所示：

```cpp
TEST("Test string confirm failure")
{
    std::string reason = "    Expected: def\n";
    reason += "    Actual  : abc";
    setExpectedFailureReason(reason);
    std::string result = "abc";
    std::string expected = "def";
    CONFIRM(expected, result);
}
```

构建和运行测试后的测试结果如下：

```cpp
---------------
Test string confirm failure
Expected failure
    Expected: def
    Actual  : abc
---------------
```

关于字符串，还有一个重要的方面需要考虑。我们需要考虑真正的常量字符指针的字符串字面量。我们将在下一节中探讨跟随字符串字面量的指针。 

# 确认字符串字面量

字符串字面量可能看起来像字符串，但 C++编译器将字符串字面量视为指向一组常量字符的第一个字符的指针。常量字符集以空字符值终止，这是零的数值。这就是编译器知道字符串有多长的方式。它只是继续进行，直到找到空字符。字符是常量的原因在于数据通常存储在写保护的内存中，因此不能被修改。

当我们尝试确认一个字符串字面量时，编译器看到的是一个指针，必须决定调用哪个重载的`confirm`函数。在我们深入探索字符串字面量之前，我们可能会遇到哪些与指针相关的问题？

让我们从简单的 bool 类型开始，看看如果我们尝试确认 bool 指针时会遇到什么问题。这将帮助你通过首先理解一个简单的 bool 指针示例测试来理解字符串字面量指针。你不需要将此测试添加到项目中。它被包含在这里只是为了解释当我们尝试确认指针时会发生什么。测试看起来是这样的：

```cpp
TEST("Test bool pointer confirms")
{
    bool result1 = true;
    bool result2 = false;
    bool * pResult1 = &result1;
    bool * pResult2 = &result2;
    CONFIRM_TRUE(pResult1);
    CONFIRM_FALSE(pResult2);
}
```

前面的测试实际上是可以编译和运行的。但它以以下结果失败：

```cpp
---------------
Test bool pointer confirms
Failed confirm on line 86
    Expected: false
---------------
```

第 86 行是测试中的第二个确认。那么，发生了什么？为什么确认认为`pResult2`指向一个真值？

好吧，记住，confirm 宏只是被替换为对`confirm`方法之一的调用。第二个确认处理以下宏：

```cpp
#define CONFIRM_FALSE( actual ) \
    confirm(false, actual, __LINE__)
```

然后它尝试使用硬编码的假 bool 值、传递给宏的 bool 指针和整行号调用`confirm`。对于任何版本的`confirm`，都没有 bool、bool 指针或 int 的确切匹配，所以要么必须进行转换，否则编译器将生成错误。我们知道没有错误，因为代码编译并运行了。那么，转换了什么？

这是对 TDD 过程的一个很好的例子，如第三章*“TDD 过程”*中所述，首先编写你希望它使用的代码，即使你预期构建会失败也要编译它。在这种情况下，构建没有失败，这让我们得到了我们可能错过的洞察。

编译器能够将指针值转换为布尔值，并且这被视为最佳选择。实际上，我甚至没有收到关于转换的警告。编译器默默地做出了将指针转换为布尔值的决定，并将其转换为布尔值。这几乎从来不是你想要发生的事情。

那么，将指针转换为布尔值究竟是什么意思呢？任何具有有效非零地址的指针都会转换为 true。此外，任何具有零地址的空指针都会转换为 false。因为我们已经将`result2`的实际地址存储在`pResult2`指针中，所以转换成了真实的布尔值。

你可能想知道第一个确认发生了什么，为什么它没有失败。为什么测试在失败之前继续进行到第二个确认？嗯，第一个确认对布尔值、布尔指针和整型进行了相同的转换。两种转换都产生了真实的布尔值，因为两个指针都持有有效的地址。

第一次确认调用`confirm`时传递了 true、true 和行号，这通过了。但第二次确认调用`confirm`时传递了 false、true 和行号，这失败了。

为了解决这个问题，我们或者需要添加对所有类型指针的支持，或者记得在确认之前解引用指针。添加对指针的支持可能看起来像是一个简单的解决方案，直到我们到达字符串字面量，它们也是指针。这并不像看起来那么简单，而且现在我们不需要这样做。让我们保持测试库尽可能简单。以下是如何修复前面显示的布尔确认测试的方法：

```cpp
TEST("Test bool pointer dereference confirms")
{
    bool result1 = true;
    bool result2 = false;
    bool * pResult1 = &result1;
    bool * pResult2 = &result2;
    CONFIRM_TRUE(*pResult1);
    CONFIRM_FALSE(*pResult2);
}
```

注意，测试解引用了指针而不是直接将指针传递给宏。这意味着测试实际上只是在测试布尔值，这就是为什么我说你实际上不需要添加测试。

字符串字面量在源代码中很常见。它们是表示预期字符串值的一种简单方法。字符串字面量的问题是它们不是字符串。它们是一个指向常量字符的指针。我们无法像对布尔指针那样解引用字符串字面量指针。那将导致一个单独的字符。我们想要确认整个字符串。

这里有一个测试，展示了字符串字面量可能的主要用法。最常见的使用是将字符串字面量与字符串进行比较。测试看起来是这样的：

```cpp
TEST("Test string and string literal confirms")
{
    std::string result = "abc";
    CONFIRM("abc", result);
}
```

这之所以有效，是因为最终传递给`confirm`函数的参数类型之一是`std::string`。编译器没有找到两个参数的精确匹配；然而，因为一个是字符串，它决定将字符串字面量也转换为字符串。

我们遇到问题的地方在于当我们尝试确认预期值和实际值的两个字符串字面量时。编译器看到两个指针，并不知道它们都应该被转换为字符串。这不是你需要在测试中验证的正常情况。另外，如果你确实需要比较两个字符串字面量，在确认之前将其中一个包裹成`std::string`参数类型很容易。

此外，在*第十二章* *创建更好的测试确认方法*中，你会看到如何解决确认两个字符串字面量的问题。我们将改进用于确认测试结果的整体设计。我们现在所使用的设计通常被称为确认值的经典方法。*第十二章*将介绍一种更可扩展、更易读、更灵活的新方法。

在支持不同类型方面我们已经取得了长足的进步，你也理解了如何处理字符串字面量。然而，我避开浮点型和双精度浮点型，因为它们需要特别的考虑。它们将在下一节中解释。

# 确认浮点值

在最基本层面上，确认工作是通过比较预期值与实际值，并在它们不同时抛出异常来完成的。这对于所有整型，如 int 和 long，布尔类型，甚至是字符串都适用。值要么匹配，要么不匹配。

对于浮点型和双精度浮点型，事情变得困难，因为并不总是能够准确比较两个浮点值。

即使在我们从小学就熟悉的十进制系统中，我们也知道存在一些无法准确表示的分数值。例如，1/3 很容易表示为分数。但是，以浮点十进制格式书写时，看起来像 0.33333，数字 3 无限循环。我们可以接近 1/3 的真实值，但在某个点上，我们必须在书写 0.333333333...时停止。无论我们包含多少个 3，总是还有更多。

在 C++中，浮点值使用具有类似精度问题的二进制数系统。但二进制中的精度问题比十进制中更为常见。

我不会深入所有细节，因为它们并不重要。然而，二进制中额外问题的主要原因是 2 的因子比 10 的因子少。在十进制系统中，因子是 1、2、5 和 10。而在二进制中，2 的因子只有 1 和 2。

那么，为什么因子很重要呢？嗯，这是因为它们决定了哪些分数可以准确描述，哪些不能。例如，1/3 这个分数对两个系统都造成麻烦，因为 3 在两个系统中都不是因子。另一个例子是 1/7。这些分数并不常见。1/10 的分数在十进制中非常常见。因为 10 是一个因子，这意味着像 0.1、0.2、0.3 等值都可以在十进制中准确表示。

此外，由于 10 不是二进制基数 2 的因子，这些广泛使用的相同值在十进制中没有用固定数字表示的表示，就像它们在十进制中那样。

所以，这一切意味着，如果你有一个看起来像 0.1 的二进制浮点值，它接近实际值，但无法完全精确。它可能在转换为字符串时显示为 0.1，但这也涉及一点舍入。

通常，我们不会担心计算机无法准确表示我们从小学就习惯于精确表示的值——也就是说，直到我们需要测试一个浮点值以查看它是否等于另一个值。

即使像 0.1 + 0.2 这样看起来等于 0.3 的简单运算，也可能不等于 0.3。

当比较计算机浮点值时，我们总是必须允许一定量的误差。只要值接近，我们就可以假设它们相等。

然而，最终的问题是，没有好的单一解决方案可以确定两个值是否接近。我们可以表示的误差量取决于值的大小。当浮点值接近 0 时，它们会急剧变化。并且当它们变大时，它们失去了表示小值的能力。因为浮点值可以变得非常大，所以大值丢失的精度也可能很大。

让我们想象一下，如果一家银行使用浮点值来跟踪你的钱。如果你有数十亿美元，但银行却无法跟踪低于一千美元的任何金额，你会高兴吗？我们不再谈论丢失几美分的问题。或者，也许你的账户里只有 30 美分，你想取出所有的 30 美分。你会期望银行拒绝你的取款，因为它认为 30 美分比你账户里的 30 美分多吗？这些问题就是浮点值可能导致的。

由于我们正在遵循 TDD（测试驱动开发）流程，我们将从简单的浮点值开始，并在比较浮点、双精度或长双精度值时包含一个小的误差范围，以查看它们是否相等。我们不会变得复杂，试图根据值的的大小调整误差范围。

这里是一个我们将用于浮点值的测试：

```cpp
TEST("Test float confirms")
{
    float f1 = 0.1f;
    float f2 = 0.2f;
    float sum = f1 + f2;
    float expected = 0.3f;
    CONFIRM(expected, sum);
}
```

浮点类型的测试实际上在我的电脑上通过了。

那么，如果我们为双精度类型创建另一个测试会发生什么呢？新的双精度测试看起来像这样：

```cpp
TEST("Test double confirms")
{
    double d1 = 0.1;
    double d2 = 0.2;
    double sum = d1 + d2;
    double expected = 0.3;
    CONFIRM(expected, sum);
}
```

这个测试几乎相同，但在我的电脑上失败了。而且，奇怪的是，除非你理解值可以以文本形式打印出来，并且已经被调整为看起来像是一个很好的圆整数，否则失败描述是没有意义的。以下是我电脑上的失败信息：

```cpp
---------------
Test double confirms
Failed confirm on line 122
    Expected: 0.300000
    Actual  : 0.300000
---------------
```

看到这条消息，你可能会问，为什么 0.300000 不等于 0.300000。原因是预期的值和实际的值都不是精确的 0.300000。它们都被稍微调整了一下，以便它们会显示这些看起来像圆整的值。

对于长双精度浮点数（long doubles）的测试几乎与双精度浮点数（doubles）的测试相同。只是类型发生了变化，如下所示：

```cpp
TEST("Test long double confirms")
{
    long double ld1 = 0.1;
    long double ld2 = 0.2;
    long double sum = ld1 + ld2;
    long double expected = 0.3;
    CONFIRM(expected, sum);
}
```

长双精度浮点数测试在我的机器上也因为与双精度浮点数测试相同的原因而失败。我们可以通过为这三种类型添加特殊重载来修复所有的浮点数确认。

这里是一个重载的`confirm`函数，它在比较浮点值时使用了一个小的误差范围：

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

我们需要的重载几乎与浮点数相同。以下是一个双精度浮点数的重载，它使用一个误差范围，这个误差范围是预期值的正负：

```cpp
inline void confirm (
    double expected,
    double actual,
    int line)
{
    if (actual < (expected - 0.000001) ||
        actual > (expected + 0.000001))
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

除了从浮点数到双精度浮点数的类型变化之外，这种方法使用了一个更小的误差范围，并且从字面值中省略了`f`后缀。

长双精度浮点数的重载函数与双精度浮点数的类似，如下所示：

```cpp
inline void confirm (
    long double expected,
    long double actual,
    int line)
{
    if (actual < (expected - 0.000001) ||
        actual > (expected + 0.000001))
    {
        throw ActualConfirmException(
            std::to_string(expected),
            std::to_string(actual),
            line);
    }
}
```

在为浮点数、双精度浮点数和长双精度浮点数添加了这些重载（overloads）之后，所有的测试都再次通过。我们将在*第十三章*，*如何测试浮点数和自定义值*中再次回顾比较浮点值的问题。我们目前拥有的比较解决方案很简单，并且现在可以工作。

我们已经涵盖了我们将要支持的所有确认类型。记住 TDD 规则，只做必要的事情。我们总是可以在以后增强确认的设计，这正是我们将在*第十二章*，*创建更好的测试确认*中要做的事情。

在结束这一章之前，我有一些关于编写确认的建议。这并不是我们还没有做的事情，但它确实值得提一下，以便你知道这个模式。

# 如何编写确认（confirms）

通常，你有很多不同的方式可以编写你的代码和测试。我在这里分享的是基于多年的经验，虽然这并不是编写测试的唯一方式，但我希望你能从中学习，并遵循类似的风格。具体来说，我想分享关于如何编写确认（confirms）的指导。

最重要的是要记住，将你的确认放在测试的正常流程之外，但仍然靠近它们所需的位置。当测试运行时，它会执行各种活动，你需要确保它们按预期工作。你可以在过程中添加确认以确保测试按预期进行。或者，你可能有一个简单的测试，只做一件事，并在最后需要一到多个确认以确保一切正常。所有这些都是好的。

考虑以下三个测试用例的例子。它们都做同样的事情，但我希望你关注它们的编写方式。以下是第一个例子：

```cpp
TEST("Test int confirms")
{
    int result = multiplyBy2(0);
    CONFIRM(0, result);
    result = multiplyBy2(1);
    CONFIRM(2, result);
    result = multiplyBy2(-1);
    CONFIRM(-2, result);
}
```

这个测试是之前用来确保我们可以确认整数值的测试。注意它如何执行一个动作并将结果分配给一个局部变量。然后，检查该变量以确保其值与预期相符。如果相符，测试将继续执行另一个动作并将结果分配给局部变量。这种模式持续进行，如果所有确认都符合预期值，则测试通过。

下面是相同测试用例的另一种写法：

```cpp
TEST("Test int confirms")
{
    CONFIRM(0, multiplyBy2(0));
    CONFIRM(2, multiplyBy2(1));
    CONFIRM(-2, multiplyBy2(-1));
}
```

这次，没有局部变量来存储每个动作的结果。有些人可能会认为这是一个改进。它确实更短。但我感觉这隐藏了正在测试的内容。我发现将确认视为可以从测试中移除而不改变测试行为的东西更好。当然，如果你移除了确认，那么测试可能会错过确认本应捕获的问题。我是在谈论心理上忽略确认，以了解测试做了什么，然后思考在过程中哪些内容需要验证。这些验证点变成了确认。

这是另一个例子：

```cpp
TEST("Test int confirms")
{
    int result1 = multiplyBy2(0);
    int result2 = multiplyBy2(1);
    int result3 = multiplyBy2(-1);
    CONFIRM(0, result1);
    CONFIRM(2, result2);
    CONFIRM(-2, result3);
}
```

这个例子避免了在确认中放置测试步骤。然而，我感觉它过分地将测试步骤与确认分离。在测试步骤中穿插确认并没有什么问题。这样做可以让你立即发现问题。这个例子将所有确认都放在了最后，这意味着它也必须等到最后才能发现问题。

然后还有这样一个问题，需要多个结果变量以便稍后逐一检查。这段代码在我看来显得有些生硬——就像一个程序员选择了漫长的路径去达到目标，而实际上有一条简单的路径可用。

第一个例子展示了这本书中迄今为止编写的测试风格，现在你可以看到为什么它们要以这种方式编写。它们在需要的地方使用确认，并且尽可能接近验证点。并且它们避免在确认中放置实际的测试步骤。

# 摘要

本章带我们超越了确认真伪值的基本能力。你现在可以验证任何你需要确保其与预期相符的内容。

我们通过将代码放入重载函数中，并使用模板版本来处理其他类型，简化了确认宏。你看到了如何确认简单数据类型，并通过先解引用来与指针一起工作。

需要重构的代码，你看到了当需要对代码进行设计更改时，TDD 是如何帮助你的。我本可以在本书中编写代码，让它看起来像是从一开始就写得完美无缺。但那样做对你没有帮助，因为没有人从一开始就能写出完美的代码。随着我们对知识的理解不断增长，我们有时需要更改代码。而 TDD 则让你有信心在问题一出现就立即进行更改，而不是等待——因为你推迟解决的问题往往会有扩大的趋势，而不是消失。

你应该正在了解如何编写你的测试，以及将确认信息融入测试的最佳方式。

到目前为止，我们一直在使用 C++ 17 中找到的 C++特性和功能。C++ 20 中有一个重要的新特性，可以帮助我们从编译器中获取行号。下一章将添加这个 C++ 20 特性，并探讨一些替代设计。即使我们保持现在的整体设计不变，下一章也会帮助你理解其他测试库可能如何进行不同的操作。
