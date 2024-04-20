# 第四章：深入了解 Lambda

恭喜！你刚刚掌握了纯函数的力量！现在是时候进入下一个级别——纯函数的超级版本，或者传说中的 lambda。它们存在的时间比对象更长，它们有一个围绕它们的数学理论（如果你喜欢这种东西的话），并且它们非常强大，正如我们将在本章和下一章中发现的那样。

本章将涵盖以下主题：

+   理解 lambda 的概念和历史

+   如何在 C++中编写 lambda

+   纯函数与 lambda 的比较

+   如何在类中使用 lambda

# 技术要求

您将需要一个支持 C++ 17 的 C++编译器。代码可以在 GitHub 存储库（[`github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp`](https://github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)）的`Chapter03`文件夹中找到。提供了一个`makefile`文件，以便您更轻松地编译和运行代码。

# 什么是 lambda？

那年是 1936 年。33 岁的数学家阿隆佐·邱奇发表了他关于数学基础的研究。在这样做的过程中，他创造了所谓的**lambda 演算**，这是最近创建的计算领域的模型。在与艾伦·图灵合作后，他随后证明了 lambda 演算等价于图灵机。这一发现的相关性对编程至关重要——这意味着我们可以通过使用 lambda 和利用 lambda 演算来为现代计算机编写任何程序。这就解释了为什么它被称为**lambda**——数学家们长期以来更喜欢用单个希腊字母来表示每个符号。但它到底是什么？

如果你忽略所有的数学符号，lambda 只是一个可以应用于变量或值的**纯函数**。让我们看一个例子。我们将学习如何在 C++中编写 lambda，但是现在我将使用 Groovy 语法，因为这是我知道的最简单的语法：

```cpp
def add = {first, second -> first + second}
add(1,2) //returns 3
```

`add`是一个 lambda。正如你所看到的，它是一个具有两个参数并返回它们的和的函数。由于 Groovy 具有可选类型，我不必指定参数的类型。此外，我不需要使用`return`语句来返回总和；它将自动返回最后一个语句的值。在 C++中，我们不能跳过类型或`return`语句，我们将在下一节中发现。

现在，让我们看一下 lambda 的另一个属性，即从上下文中捕获值的能力：

```cpp
def first = 5
def addToFirst = {second -> first + second}
addToFirst(10) // returns 5 + 10 = 15
```

在这个例子中，`first`不是函数的参数，而是在上下文中定义的变量。lambda *捕获*变量的值并在其主体内使用它。我们可以利用 lambda 的这个属性来简化代码或逐渐重构向不可变性。

我们将在未来的章节中探讨如何使用 lambda；现在，让我们演示如何在 C++中编写它们，如何确保它们是不可变的，以及如何从上下文中捕获值。

# C++中的 lambda

我们探讨了如何在 Groovy 中编写 lambda。那么，我们可以在 C++中使用它们的功能吗？自 C++ 11 以来，引入了特定的语法。让我们看看我们的`add` lambda 在 C++中会是什么样子：

```cpp
int main(){
    auto add = [](int first, int second){ return first + second;};
    cout << add(1,2) << endl; // writes 3
}
```

让我们按照以下方式解释语法：

+   我们的 lambda 以`[]`开始。这个块指定了我们从上下文中捕获的变量，我们将看到如何在一会儿使用它。由于我们没有捕获任何东西，这个块是空的。

+   接下来，我们有参数列表，`(int first, int second)`，就像任何其他 C++函数一样。

+   最后，我们编写 lambda 的主体，使用 return 语句：`{ return first + second; }`。

语法比 Groovy 有点更加正式，但感觉像 C++，这是一件好事；统一性有助于我们记住事情。

或者，我们可以使用箭头语法，如下面的代码所示：

```cpp
    auto add = [](int first, int second) -> int { return first +   
        second;};
```

箭头语法是 lambda 的标志，自从 Alonzo Church 在他的 lambda 演算中使用这种符号以来。除此之外，C++要求在 lambda 主体之前指定返回类型，这可能在涉及类型转换的情况下提供了清晰度。

由于历史原因，箭头语法以某种方式存在于所有函数式编程语言中。在 C++中很少有用，但是如果你想要习惯函数式编程，了解它是很有用的。

现在是时候探索如何从上下文中捕获变量了。正如我们之前提到的，这都在`[]`块中。

# 捕获变量

那么，如果我们想要捕获变量呢？在 Groovy 中，我们只需在 lambda 范围内使用变量。这在 C++中行不通，因为我们需要指定我们要捕获的变量以及捕获它们的方式。因此，如果我们只在`add` lambda 中使用`first`变量，我们将得到以下编译错误：

```cpp
int main(){
    int first = 5;
    auto addToFirst = [](int second){ return first + second;}; 
    // error: variable 'first' cannot be implicitly captured 
    cout << add(10) << endl;
}
```

为了在 C++中捕获变量，我们需要在`[]`块内使用捕获说明符。有多种方法可以做到这一点，具体取决于你的需求。最直观的方法是直接写出我们要捕获的变量的名称。在我们的情况下，由于我们要捕获第一个变量，我们只需要在 lambda 参数前添加`[first]`：

```cpp
int main(){
    int first = 5;
    auto addToFirst = first{ return first + second;};
    cout << addToFirst(10) << endl; // writes 15
}
```

正如我们将看到的，这意味着`first`变量是按值捕获的。由于 C++给程序员提供了很多控制权，我们期望它提供特定的语法来按引用捕获变量。现在，让我们更详细地探讨捕获语法。

# 按值和按引用捕获变量

我们知道按值捕获变量的说明符只是写变量的名称，即`[first]`。这意味着变量被复制，因此我们浪费了一些内存。解决方案是通过引用捕获变量。捕获说明符的语法非常直观——我们可以将变量名作为`[&first]`引用：

```cpp
int main(){
    int first = 5;
    auto addToFirstByReference = &first{ return first + 
        second;};
    cout << addToFirstByReference(10) << endl; // writes 15
}
```

我知道你在想什么：lambda 现在可以修改`first`变量的值吗，因为它是按引用传递的？剧透警告——是的，它可以。我们将在下一节重新讨论不可变性、纯函数和 lambda。现在，还有更多的语法要学习。例如，如果我们想要从上下文中捕获多个变量，我们是否必须在捕获说明符中写出它们所有？事实证明，有一些快捷方式可以帮助你避免这种情况。

# 捕获多个值

那么，如果我们想要捕获多个值呢？让我们探索一下如果我们添加了五个捕获的值，我们的 lambda 会是什么样子：

```cpp
    int second = 6;
    int third = 7;
    int fourth = 8;
    int fifth = 9;

    auto addTheFive = [&first, &second, &third, &fourth, &fifth]()   
    {return first + second + third + fourth + fifth;};
    cout << addTheFive() << endl; // writes 35
```

我们当前的语法有点多余，不是吗？我们可以使用默认捕获说明符。幸运的是，语言设计者也是这么想的；注意 lambda 参数前的`[&]`语法：

```cpp
    auto addTheFiveWithDefaultReferenceCapture = [&](){return first + second + third + fourth + fifth;};
    cout << addTheFiveWithDefaultReferenceCapture() << endl; // writes 35
```

`[&]`语法告诉编译器从上下文中引用所有指定的变量。这是*默认按引用捕获*说明符。

如果我们想要复制它们的值，我们需要使用*默认按值捕获*说明符，你需要记住这是唯一使用这种方式的地方。注意 lambda 参数前的`[=]`语法：

```cpp
auto addTheFiveWithDefaultValueCapture = [=](){return first + 
second + third + fourth + fifth;};
cout << addTheFiveWithDefaultValueCapture() << endl; // writes 35
```

`[=]`语法告诉编译器所有变量都将通过复制它们的值来捕获。至少，默认情况下是这样。如果出于某种原因，你想要除了`first`之外的所有变量都通过值传递，那么你只需将默认与变量说明符结合起来：

```cpp
auto addTheFiveWithDefaultValueCaptureForAllButFirst = [=, &first](){return first + second + third + fourth + fifth;};
cout << addTheFiveWithDefaultValueCaptureForAllButFirst() << endl; // writes 35
```

我们现在知道了如何按值和按引用捕获变量，以及如何使用默认说明符。这使我们留下了一个重要类型的变量——指针。

# 捕获指针值

指针只是简单的值。如果我们想要按值捕获指针变量，我们可以像下面的代码中那样写它的名称：

```cpp
    int* pFirst = new int(5);
    auto addToThePointerValue = pFirst{return *pFirst + 
        second;};
    cout << addToThePointerValue(10) << endl; // writes 15
    delete pFirst;
```

如果我们想要按引用捕获指针变量，捕获语法与捕获任何其他类型的变量相同：

```cpp
auto addToThePointerValue = &pFirst{return *pFirst + 
    second;};
```

默认的限定符的工作方式正如你所期望的那样；也就是说，`[=]`通过值来捕获指针变量：

```cpp
 auto addToThePointerValue = ={return *pFirst + second;};
```

相比之下，`[&]`通过引用来捕获指针变量，如下面的代码所示：

```cpp
    auto addToThePointerValue = &{return *pFirst + 
    second;};
```

我们将探讨通过引用捕获变量对不可变性可能产生的影响。但首先，由于有多种捕获 lambda 变量的方式，我们需要检查我们更喜欢哪一种，以及何时使用它们。

# 我们应该使用什么捕获？

我们已经看到了一些捕获值的选项，如下所示：

+   命名变量以通过值来捕获它；例如，`[aVariable]`

+   命名变量并在前面加上引用限定符以通过引用来捕获它；例如，`[&aVariable]`

+   使用默认值限定符通过值来捕获所有使用的变量；语法是`[=]`

+   使用默认引用限定符通过引用来捕获所有使用的变量；语法是`[&]`

实际上，我发现使用默认值限定符是大多数情况下最好的版本。这可能受到我偏好不改变捕获值的非常小的 lambda 的影响。我相信简单性非常重要；当你有多个选项时，很容易使语法比必要的更复杂。仔细考虑每个上下文，并使用最简单的语法；我的建议是从`[=]`开始，只有在需要时才进行更改。

我们已经探讨了如何在 C++中编写 lambda。我们还没有提到它们是如何实现的。当前的标准将 lambda 实现为一个在堆栈上创建的具有未知类型的 C++对象。就像任何 C++对象一样，它背后有一个类，有一个构造函数，一个析构函数，以及捕获的变量作为数据成员存储。我们可以将 lambda 传递给`function<>`对象，这样`function<>`对象将存储 lambda 的副本。此外，*lambda 使用延迟评估*，不同于`function<>`对象。

Lambda 似乎是编写纯函数的一种更简单的方法；那么，lambda 和纯函数之间的关系是什么？

# Lambda 和纯函数

我们在第二章中学到，纯函数具有三个特征：

+   它们总是对相同的参数值返回相同的值

+   它们没有副作用

+   它们不改变其参数的值

我们还发现在编写纯函数时需要注意不可变性。只要我们记得在哪里放置`const`关键字，这很容易。

那么，lambda 如何处理不可变性？我们需要做一些特殊的事情吗，还是它们只是工作？

# Lambda 的不可变性和通过值传递的参数

让我们从一个非常简单的 lambda 开始，如下所示：

```cpp
auto increment = [](int value) { 
    return ++value;
};
```

在这里，我们通过值传递参数，所以我们在调用 lambda 后不希望值发生任何改变：

```cpp
    int valueToIncrement = 41;
    cout << increment(valueToIncrement) << endl;// prints 42
    cout << valueToIncrement << endl;// prints 41
```

由于我们复制了值，我们可能使用了一些额外的内存字节和额外的赋值。我们可以添加一个`const`关键字来使事情更清晰：

```cpp
auto incrementImmutable = [](const int value) { 
    return value + 1;
};
```

由于`const`限定符，如果 lambda 尝试改变`value`，编译器将会报错。

但我们仍然通过值传递参数；那么通过引用传递呢？

# Lambda 的不可变性和通过引用传递的参数

让我们探讨当我们调用这个 lambda 时对输入参数的影响：

```cpp
auto increment = [](int& value) { 
    return ++value;
};
```

事实证明，这与你所期望的相当接近：

```cpp
int valueToIncrement = 41;
cout << increment(valueToIncrement) << endl;// prints 42
cout << valueToIncrement << endl;// prints 42
```

在这里，lambda 改变了参数的值。这还不够好，所以让我们使其不可变，如下面的代码所示：

```cpp
auto incrementImmutable = [](const int& value){
    return value + 1;
};
```

编译器会再次通过错误消息帮助我们，如果 lambda 尝试改变`value`。

好了，这样更好了；但指针呢？

# Lambda 的不可变性和指针参数

就像我们在第二章中看到的那样，关于指针参数有两个问题，如下所示：

+   lambda 能改变指针地址吗？

+   lambda 能改变指向的值吗？

再次，如果我们按值传递指针，地址不会改变：

```cpp
auto incrementAddress = [](int* value) { 
    return ++value;
};

int main(){
    int* pValue = new int(41);
    cout << "Address before:" << pValue << endl;
    cout << "Address returned by increment address:" <<   
    incrementAddress(pValue) << endl;
    cout << "Address after increment address:" << pValue << endl;
}

Output:
Address before:0x55835628ae70
Address returned by increment address:0x55835628ae74
Address after increment address:0x55835628ae70
```

通过引用传递指针会改变这一点：

```cpp
auto incrementAddressByReference = [](int*& value) { 
    return ++value;
};

void printResultsForIncrementAddressByReference(){
    int* pValue = new int(41);
    int* initialPointer = pValue;
    cout << "Address before:" << pValue << endl;
    cout << "Address returned by increment address:" <<    
    incrementAddressByReference(pValue) << endl;
    cout << "Address after increment address:" << pValue << endl;
    delete initialPointer;
}

Output:
Address before:0x55d0930a2e70
Address returned by increment address:0x55d0930a2e74
Address after increment address:0x55d0930a2e74
```

因此，我们需要再次使用适当的`const`关键字来保护自己免受这种变化的影响：

```cpp
auto incrementAddressByReferenceImmutable = [](int* const& value) { 
    return value + 1;
};

Output:
Address before:0x557160931e80
Address returned by increment address:0x557160931e84
Address after increment address:0x557160931e80
```

让我们也使值不可变。如预期的那样，我们需要另一个`const`关键字：

```cpp
auto incrementPointedValueImmutable = [](const int* const& value) { 
    return *value + 1;
};
```

虽然这样可以工作，但我建议您更倾向于使用更简单的方式传递`[](const int& value)`值，也就是说，只需对指针进行解引用并将实际值传递给 lambda 表达式，这将使参数语法更容易理解和更可重用。

所以，毫不意外！我们可以使用与纯函数相同的语法来确保不可变性。

但是 lambda 表达式能调用可变函数吗，比如 I/O 呢？

# Lambda 表达式和 I/O

测试 lambda 表达式和 I/O 的更好方法是`Hello, world`程序：

```cpp
auto hello = [](){cout << "Hello, world!" << endl;};

int main(){
    hello();
}
```

显然，lambda 表达式无法防止调用可变函数。这并不奇怪，因为我们对纯函数也学到了同样的事情。这意味着，类似于纯函数，程序员需要特别注意将 I/O 与其余可能是不可变的代码分开。

由于我们试图让编译器帮助我们强制实施不可变性，我们能为捕获的值做到这一点吗？

# Lambda 表达式的不可变性和捕获值

我们已经发现 lambda 表达式可以从上下文中捕获变量，无论是按值还是按引用。那么，这是否意味着我们可以改变它们的值呢？让我们来看看：

```cpp
int value = 1;
auto increment = [=](){return ++value;};
```

这段代码立即给出了一个编译错误——*无法对按值捕获的变量赋值*。这比按值传递参数要好，也就是说，不需要使用`const`关键字——它可以按预期工作。

# 按引用捕获的值的不可变性

那么，通过引用捕获的值呢？好吧，我们可以使用默认的引用说明符`[&]`，并在调用我们的`increment` lambda 之前和之后检查变量的值：

```cpp
void captureByReference(){
    int value = 1;
    auto increment = [&](){return ++value;};

    cout << "Value before: " << value << endl;
    cout << "Result of increment:" << increment() << endl;
    cout << "Value after: " << value << endl;
}

Output:
Value before: 1
Result of increment:2
Value after: 2
```

如预期的那样，`value`发生了变化。那么，我们如何防止这种变化呢？

不幸的是，没有简单的方法可以做到这一点。C++假设如果您通过引用捕获变量，您想要修改它们。虽然这是可能的，但它需要更多的语法糖。具体来说，我们需要捕获其转换为`const`类型的内容，而不是变量本身：

```cpp
#include <utility>
using namespace std;
...

    int value = 1;
    auto increment = [&immutableValue = as_const(value)](){return  
        immutableValue + 1;};

Output:
Value before: 1
Result of increment:2
Value after: 1
```

如果可以选择，我更喜欢使用更简单的语法。因此，除非我真的需要优化性能，我宁愿使用按值捕获的语法。

我们已经探讨了如何在捕获值类型时使 lambda 表达式不可变。但是在捕获指针类型时，我们能确保不可变性吗？

# 按值捕获的指针的不可变性

当我们使用指针时，事情变得有趣起来。如果我们按值捕获它们，就无法修改地址：

```cpp
    int* pValue = new int(1);
    auto incrementAddress = [=](){return ++pValue;}; // compilation 
    error
```

然而，我们仍然可以修改指向的值，就像下面的代码所示：

```cpp
    int* pValue = new int(1);
    auto increment= [=](){return ++(*pValue);};

Output:
Value before: 1
Result of increment:2
Value after: 2
```

限制不可变性需要一个`const int*`类型的变量：

```cpp
    const int* pValue = new int(1);
    auto increment= [=](){return ++(*pValue);}; // compilation error
```

然而，有一个更简单的解决方案，那就是只捕获指针的值：

```cpp
 int* pValue = new int(1);
 int value = *pValue;
 auto increment = [=](){return ++value;}; // compilation error
```

# 按引用捕获的指针的不可变性

通过引用捕获指针允许您改变内存地址：

```cpp
 auto increment = [&](){return ++pValue;};
```

我们可以使用与之前相同的技巧来强制内存地址的常量性：

```cpp
 auto increment = [&pImmutable = as_const(pValue)](){return pImmutable 
    + 1;};
```

然而，这变得相当复杂。这样做的唯一原因是由于以下原因：

+   我们希望避免最多复制 64 位

+   编译器不会为我们进行优化

最好还是坚持使用按值传递的值，除非您想在 lambda 表达式中进行指针运算。

现在您知道了 lambda 表达式在不可变性方面的工作原理。但是，在我们的 C++代码中，我们习惯于类。那么，lambda 表达式和类之间有什么关系呢？我们能将它们结合使用吗？

# Lambda 表达式和类

到目前为止，我们已经学习了如何在 C++中编写 lambda 表达式。所有的例子都是在类外部使用 lambda 表达式，要么作为变量，要么作为`main()`函数的一部分。然而，我们的大部分 C++代码都存在于类中。这就引出了一个问题——我们如何在类中使用 lambda 表达式呢？

为了探讨这个问题，我们需要一个简单类的例子。让我们使用一个表示基本虚数的类：

```cpp
class ImaginaryNumber{
    private:
        int real;
        int imaginary;

    public:
        ImaginaryNumber() : real(0), imaginary(0){};
        ImaginaryNumber(int real, int imaginary) : real(real), 
        imaginary(imaginary){};
};
```

我们想要利用我们新发现的 lambda 超能力来编写一个简单的`toString`函数，如下面的代码所示：

```cpp
string toString(){
    return to_string(real) + " + " + to_string(imaginary) + "i";
}
```

那么，我们有哪些选择呢？

嗯，lambda 是简单的变量，所以它们可以成为数据成员。或者，它们可以是`static`变量。也许我们甚至可以将类函数转换为 lambda。让我们接下来探讨这些想法。

# Lambda 作为数据成员

让我们首先尝试将其写为成员变量，如下所示：

```cpp
class ImaginaryNumber{
...
    public:
        auto toStringLambda = [](){
            return to_string(real) + " + " + to_string(imaginary) +  
             "i";
        };
...
}
```

不幸的是，这导致编译错误。如果我们想将其作为非静态数据成员，我们需要指定 lambda 变量的类型。为了使其工作，让我们将我们的 lambda 包装成`function`类型，如下所示：

```cpp
include <functional>
...
    public:
        function<string()> toStringLambda = [](){
            return to_string(real) + " + " + to_string(imaginary) +    
            "i";
        };
```

函数类型有一个特殊的语法，允许我们定义 lambda 类型。`function<string()>`表示函数返回一个`string`值并且不接收任何参数。

然而，这仍然不起作用。我们收到另一个错误，因为我们没有捕获正在使用的变量。我们可以使用到目前为止学到的任何捕获。或者，我们可以捕获`this`：

```cpp
 function<string()> toStringLambda = [this](){
     return to_string(real) + " + " + to_string(imaginary) + 
     "i";
 };
```

因此，这就是我们可以将 lambda 作为类的一部分编写，同时捕获类的数据成员。在重构现有代码时，捕获`this`是一个有用的快捷方式。但是，在更持久的情况下，我会避免使用它。最好直接捕获所需的变量，而不是整个指针。

# Lambda 作为静态变量

我们还可以将我们的 lambda 定义为`static`变量。我们不能再捕获值了，所以我们需要传入一个参数，但我们仍然可以访问`real`和`imaginary`私有数据成员：

```cpp
    static function<string(const ImaginaryNumber&)>   
         toStringLambdaStatic;
...
// after class declaration ends
function<string(const ImaginaryNumber&)> ImaginaryNumber::toStringLambdaStatic = [](const ImaginaryNumber& number){
    return to_string(number.real) + " + " + to_string(number.imaginary)  
        + "i";
};

// Call it
cout << ImaginaryNumber::toStringLambdaStatic(Imaginary(1,1)) << endl;
// prints 1+1i
```

# 将静态函数转换为 lambda

有时，我们需要将`static`函数转换为 lambda 变量。在 C++中，这非常容易，如下面的代码所示：

```cpp
static string toStringStatic(const ImaginaryNumber& number){
    return to_string(number.real) + " + " + to_string(number.imaginary)  
    + "i";
 }
string toStringUsingLambda(){
    auto toStringLambdaLocal = ImaginaryNumber::toStringStatic;
    return toStringLambdaLocal(*this);
}
```

我们可以简单地将一个来自类的函数分配给一个变量，就像在前面的代码中所示的那样：

```cpp
  auto toStringLambdaLocal = ImaginaryNumber::toStringStatic;
```

然后我们可以像使用函数一样使用变量。正如我们将要发现的那样，这是一个非常强大的概念，因为它允许我们在类内部定义函数时组合函数。

# Lambda 和耦合

在 lambda 和类之间的交互方面，我们有很多选择。它们既可以变得令人不知所措，也可以使设计决策变得更加困难。

虽然了解选项是好的，因为它们有助于进行困难的重构，但通过实践，我发现在使用 lambda 时最好遵循一个简单的原则；也就是说，选择减少 lambda 与代码其余部分之间耦合区域的选项是最好的。

例如，我们已经看到我们可以将我们的 lambda 写成类中的`static`变量：

```cpp
function<string(const ImaginaryNumber&)> ImaginaryNumber::toStringLambdaStatic = [](const ImaginaryNumber& number){
    return to_string(number.real) + " + " + to_string(number.imaginary)  
        + "i";
};
```

这个 lambda 的耦合区域与`ImaginaryNumber`类一样大。但它只需要两个值：实部和虚部。我们可以很容易地将它重写为一个纯函数，如下所示：

```cpp
auto toImaginaryString = [](auto real, auto imaginary){
    return to_string(real) + " + " + to_string(imaginary) + "i";
};
```

如果由于某种原因，您决定通过添加成员或方法、删除成员或方法、将其拆分为多个类或更改数据成员类型来更改虚数的表示，这个 lambda 将不需要更改。当然，它需要两个参数而不是一个，但参数类型不再重要，只要`to_string`对它们有效。换句话说，这是一个多态函数，它让您对表示数据结构的选项保持开放。

但我们将在接下来的章节中更多地讨论如何在设计中使用 lambda。

# 总结

你刚刚获得了 lambda 超能力！你不仅可以在 C++中编写简单的 lambda，还知道以下内容：

+   如何从上下文中捕获变量

+   如何指定默认捕获类型——按引用或按值

+   如何在捕获值时编写不可变的 lambda

+   如何在类中使用 lambda

我们还提到了低耦合设计原则以及 lambda 如何帮助实现这一点。在接下来的章节中，我们将继续提到这一原则。

如果我告诉你，lambda 甚至比我们目前所见到的更强大，你会相信吗？好吧，我们将发现通过函数组合，我们可以从简单的 lambda 发展到复杂的 lambda。

# 问题

1.  你能写出最简单的 lambda 吗？

1.  如何编写一个将作为参数传递的两个字符串值连接起来的 lambda？

1.  如果其中一个值是被值捕获的变量会发生什么？

1.  如果其中一个值是被引用捕获的变量会发生什么？

1.  如果其中一个值是被值捕获的指针会发生什么？

1.  如果其中一个值是被引用捕获的指针会发生什么？

1.  如果两个值都使用默认捕获说明符被值捕获会发生什么？

1.  如果两个值都使用默认捕获说明符被引用捕获会发生什么？

1.  如何在一个类的数据成员中写入与两个字符串值作为数据成员相同的 lambda？

1.  如何在同一个类中将相同的 lambda 写为静态变量？
