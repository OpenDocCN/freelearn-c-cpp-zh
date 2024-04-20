# 理解纯函数

纯函数是函数式编程的核心构建模块。它们是不可变的函数，这使它们简单和可预测。在 C++中编写纯函数很容易，但是有一些事情你需要注意。由于 C++中的函数默认是可变的，我们需要学习告诉编译器如何防止变异的语法。我们还将探讨如何将可变代码与不可变代码分开。

本章将涵盖以下主题：

+   理解纯函数是什么

+   在 C++中编写纯函数和使用元组返回多个参数的函数

+   确保 C++纯函数的不可变性

+   理解为什么 I/O 是可变的，需要与纯函数分开

# 技术要求

你需要一个支持 C++ 17 的 C++编译器。我使用的是 GCC 版本 7.3.0。代码示例在 GitHub（[`github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp`](https://github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)）的`Chapter02`文件夹中，并且有一个`makefile`文件供您使用。

# 什么是纯函数？

让我们花点时间思考一个简单的日常体验。当你打开灯开关时，会发生两种情况之一：

+   如果灯是开着的，它就会关掉

+   如果灯是关着的，它就会打开

灯开关的行为是非常可预测的。它是如此可预测，以至于当灯不亮时，你立刻认为有什么地方出了问题——可能是灯泡、保险丝或开关本身。

以下是你打开或关闭开关时不希望发生的一些事情：

+   你的冰箱不会关掉

+   你邻居的灯不会亮起

+   你的浴室水槽不会打开

+   你的手机不会重置

当你打开灯开关时为什么会发生所有这些事情？那将是非常混乱的；我们不希望生活中出现混乱，对吧？

然而，程序员经常在代码中遇到这种行为。调用函数通常会导致程序状态的改变；当这种情况发生时，我们说函数具有**副作用**。

函数式编程试图通过广泛使用纯函数来减少状态变化引起的混乱。纯函数是具有两个约束的函数：

+   它们总是对相同的参数值返回相同的输出值。

+   它们没有副作用。

让我们探讨如何编写灯开关的代码。我们假设灯泡是一个我们可以调用的外部实体；把它看作我们程序的**输入/输出**（**I/O**）的输出。结构化/面向对象程序员的自然代码看起来可能是这样的：

```cpp
void switchLight(LightBulb bulb){
    if(switchIsOn) bulb.turnOff();
    else bulb.turnOn();
}
```

这个函数有两个问题。首先，它使用了不属于参数列表的输入，即`switchIsOn`。其次，它直接对灯泡产生了副作用。

那么，纯函数是什么样子的呢？首先，它的所有参数都是可见的：

```cpp
void switchLight(boolean switchIsOn, LightBulb bulb){    if(switchIsOn) 
    bulb.turnOff();
    else bulb.turnOn();
}
```

其次，我们需要消除副作用。我们该如何做呢？让我们将下一个状态的计算与打开或关闭灯泡的动作分开：

```cpp
LightBulbSignal signalForBulb(boolean switchIsOn){
    if(switchIsOn) return LightBulbSignal.TurnOff;
    else return LightBulbSignal.TurnOn;
}
// use the output like this: sendSignalToLightBulb(signalForBulb(switchIsOn))
```

该函数现在是纯的，我们稍后会更详细地讨论这一点；但是，现在让我们简化如下：

```cpp
LightBulbSignal signalForBulb(boolean switchIsOn){
    return switchIsOn ? LightBulbSignal.TurnOff :    
    LightBulbSignal.TurnOn;
}
// use the output like this: sendSignalToLightBulb(signalForBulb(switchIsOn))
```

让我们更清晰一些（我会假设该函数是一个类的一部分）：

```cpp
static LightBulbSignal signalForBulb(const boolean switchIsOn){
    return switchIsOn ? LightBulbSignal.TurnOff :  
    LightBulbSignal.TurnOn;
}
// use the output like this: sendSignalToLightBulb(signalForBulb(switchIsOn))
```

这个函数非常无聊：它非常可预测，易于阅读，而且没有副作用。这听起来就像一个设计良好的灯开关。而且，这正是我们在维护数十年的大量代码时所希望的。

我们现在了解了纯函数是什么以及它为什么有用。我们还演示了如何将纯函数与副作用（通常是 I/O）分离的例子。这是一个有趣的概念，但它能带我们到哪里？我们真的可以使用这样简单的构造来构建复杂的程序吗？我们将在接下来的章节中讨论如何组合纯函数。现在，让我们专注于理解如何在 C++中编写纯函数。

# C++中的纯函数

在前面的例子中，您已经看到了我们在 C++中需要使用的纯函数的基本语法。您只需要记住以下四个想法：

+   纯函数没有副作用；如果它们是类的一部分，它们可以是`static`或`const`。

+   纯函数不改变它们的参数，因此每个参数都必须是`const`、`const&`或`const* const`类型。

+   纯函数总是返回值。从技术上讲，我们可以通过输出参数返回一个值，但通常更简单的是直接返回一个值。这意味着纯函数通常没有 void 返回类型。

+   前面的观点都不能保证没有副作用或不可变性，但它们让我们接近了。例如，数据成员可以标记为可变，`const`方法可以改变它们。

在接下来的章节中，我们将探讨如何编写自由函数和类方法作为纯函数。当我们浏览示例时，请记住我们现在正在探索语法，重点是如何使用编译器尽可能接近纯函数。

# 没有参数的纯函数

让我们从简单的开始。我们可以在没有参数的情况下使用纯函数吗？当然可以。一个例子是当我们需要一个默认值时。让我们考虑以下例子：

```cpp
int zero(){return 0;}
```

这是一个独立的函数。让我们了解如何在类中编写纯函数：

```cpp
class Number{
    public:
        static int zero(){ return 0; }
}
```

现在，`static`告诉我们该函数不会改变任何非静态数据成员。但是，这并不能阻止代码改变`static`数据成员的值：

```cpp
class Number{
    private:
        static int accessCount;
    public:
        static int zero(){++accessCount; return 0;}
        static int getCount() { return accessCount; }
};
int Number::accessCount = 0;
int main(){
Number::zero();
cout << Number::getCount() << endl; // will print 1
}
```

幸运的是，我们会发现我们可以通过恰当使用`const`关键字来解决大多数可变状态问题。以下情况也不例外：

```cpp
static const int accessCount;
```

现在我们已经对如何编写没有参数的纯函数有了一些了解，是时候添加更多参数了。

# 带有一个或多个参数的纯函数

让我们从一个带有一个参数的纯类方法开始，如下面的代码所示：

```cpp
class Number{
    public:
        static int zero(){ return 0; }
        static int increment(const int value){ return value + 1; }
}
```

两个参数呢？当然，让我们考虑以下代码：

```cpp
class Number{
    public:
        static int zero(){ return 0; }
        static int increment(const int value){ return value + 1; }
        static int add(const int first, const int second){ return first  
        + second; }
};
```

我们可以用引用类型做同样的事情，如下所示：

```cpp
class Number{
    public:
        static int zero(){ return 0; }
        static int increment(const int& value){ return value + 1; }
        static int add(const int& first, const int& second){ return 
        first + second; }
};
```

此外，我们可以用指针类型做同样的事情，尽管有点更多的语法糖：

```cpp
class Number{
    public:
        static int incrementValueFromPointer(const int* const value )   
        {return *value + 1;}
};
```

恭喜——您现在知道如何在 C++中编写纯函数了！

嗯，有点；不幸的是，不可变性在 C++中实现起来比我们迄今所见到的要复杂一些。我们需要更深入地研究各种情况。

# 纯函数和不可变性

1995 年的电影《阿波罗 13 号》是我最喜欢的惊悚片之一。它涉及太空、一个真实的故事和多个工程问题。在许多令人难忘的场景中，有一个特别能教给我们很多关于编程的场景。当宇航员团队正在准备一个复杂的程序时，由汤姆·汉克斯扮演的指挥官注意到，他的同事在一个指令开关上贴了一张标签，上面写着“不要按动”。指挥官问他的同事为什么这样做，他的回答大致是“我的头脑不清醒，我害怕我会按动这个开关把你送上太空。所以，我写下这个来提醒自己不要犯这个错误。”

如果这种技术对宇航员有效，那么对程序员也应该有效。幸运的是，我们有编译器告诉我们何时做错了。但是，我们需要告诉编译器我们希望它检查什么。

毕竟，我们可以编写纯函数，而不需要任何`const`或`static`。函数纯度不是语法问题，而是一个概念。正确地放置标签可以防止我们犯错。然而，我们会看到，编译器只能做到这一点。

让我们看看另一种实现我们之前讨论过的递增函数的方法：

```cpp
class Number{
    public:
        int increment(int value){ return ++value; }
};
int main(){
    Number number;
    int output = number.increment(Number::zero());
    cout << output << endl;
 }
```

这不是一个纯函数。你能看出为什么吗？答案就在下一行：

```cpp
 int increment(int value){ return ++value; }
```

`++value`不仅会递增`value`，还会改变输入参数。虽然在这种情况下并不是问题（`value`参数是按值传递的，所以只有它的副本被修改），但这仍然是一个副作用。这显示了在 C++中编写副作用有多容易，或者在任何不默认强制不可变性的语言中。幸运的是，只要我们告诉编译器我们确切地想要什么，编译器就可以帮助我们。

回想一下之前的实现如下：

```cpp
 static int increment(const int value){ return value + 1; }
```

如果你尝试在这个函数的主体中写`++value`或`value++`，编译器会立即告诉你，你试图改变一个`const`输入参数。这真是太好了，不是吗？

那么通过引用传递的参数呢？

# 不可变性和通过引用传递

问题本来可能更糟。想象一下以下函数：

```cpp
 static int increment(int& value){ return ++value; }
```

我们避免了按值传递，这涉及更多的内存字节。但是值会发生什么变化呢？让我们看看以下代码：

```cpp
  int value = Number::zero(); //value is 0
      cout << Number::increment(value) << endl;
      cout << value << endl; // value is now 1
```

`value`参数开始为`0`，但当我们调用函数时，它被递增，所以现在它的`value`是`1`。这就像每次你打开灯时，冰箱门都会打开。幸运的是，如果我们只添加一个小小的`const`关键字，我们会看到以下结果：

```cpp
static int increment(const int& value) {return value + 1; }
```

然后，编译器再次友好地告诉我们，在函数体中不能使用`++value`或`value++`。

这很酷，但指针参数呢？

# 不可变性和指针

在使用指针作为输入参数时，防止不需要的更改变得更加复杂。让我们看看当我们尝试调用这个函数时会发生什么：

```cpp
  static int increment(int* pValue)
```

以下事情可能会改变：

+   `pValue`指向的值可能会改变。

+   指针可能会改变其地址。

`pValue`指向的值在类似条件下可能会改变，就像我们之前发现的那样。例如，考虑以下代码：

```cpp
 static int increment(int* pValue){ return ++*pValue; }
```

这将改变指向的值并返回它。要使其不可更改，我们需要使用一个恰到好处的`const`关键字：

```cpp
 static int increment(int* const pValue){ return *pValue + 1; }
```

指针地址的更改比你期望的要棘手。让我们看一个会以意想不到的方式行为的例子：

```cpp
class Number {
    static int* increment(int* pValue){ return ++pValue; }
}

int main(){
    int* pValue = new int(10);
    cout << "Address: " << pValue << endl;
    cout << "Increment pointer address:" <<   
    Number::incrementPointerAddressImpure(pValue) << endl;
    cout << "Address after increment: " << pValue << endl;
    delete pValue;
}
```

在我的笔记本上运行这个程序会得到以下结果：

```cpp
Address: 0x55cd35098e80
Increment pointer address:0x55cd35098e80
Address after increment: 0x55cd35098e80
Increment pointer value:10
```

地址不会改变，即使我们在函数中使用`++pValue`进行递增。`pValue++`也是如此，但为什么会这样呢？

嗯，指针地址是一个值，它是按值传递的，所以函数体内的任何更改只适用于函数范围。要使地址更改，您需要按引用传递地址，如下所示：

```cpp
 static int* increment(int*& pValue){ return ++pValue; }
```

这告诉我们，幸运的是，编写更改指针地址的函数并不容易。我仍然觉得告诉编译器强制执行这个规则更安全：

```cpp
 static int* increment(int* const& pValue){ return ++pValue; }
```

当然，这并不妨碍你改变指向的值：

```cpp
  static int* incrementPointerAddressAndValue(int* const& pValue){
      (*pValue)++;
      return pValue + 1;
  }
```

为了强制不可变性，无论是值还是地址，你需要使用更多的`const`关键字，如下面的代码所示：

```cpp
  static const int* incrementPointerAddressAndValuePure(const int* 
      const& pValue){
          (*pValue)++;//Compilation error
          return pValue + 1;
  }
```

这涵盖了所有类型的类函数。但是，C++允许我们在类外编写函数。那么在这种情况下，`static`还有效吗？（剧透警告：并不完全如你所期望）。

# 不可变性和非类函数

到目前为止的所有示例都假设函数是类的一部分。C++允许我们编写不属于任何类的函数。例如，我们可以编写以下代码：

```cpp
int zero(){ return 0; }
int increment(int& value){ return ++value; }
const int* incrementPointerAddressAndValuePure(const int* const& pValue){
    return pValue + 1;
}
```

您可能已经注意到我们不再使用`static`了。您可以使用`static`，但需要注意它对类中的函数具有完全不同的含义。应用于独立函数的`static`意味着*您无法从不同的翻译单元中使用它*；因此，如果您在 CPP 文件中编写函数，它将只在该文件中可用，并且链接器会忽略它。

我们已经涵盖了所有类型的类和非类函数。但是对于具有输出参数的函数呢？事实证明，它们需要一些工作。

# 不可变性和输出参数

有时，我们希望函数改变我们传入的数据。在**标准模板库**（**STL**）中有许多例子，其中最简单的一个例子是`sort`：

```cpp
vector<int> values = {324, 454, 12, 45, 54564, 32};
     sort(values.begin(), values.end());
```

然而，这并不符合纯函数的概念；`sort`的纯函数等价物如下：

```cpp
vector<int> sortedValues = pureSort(values);
```

我能听到你在想，“但 STL 实现是为了优化而在原地工作，那么纯函数是否 less optimized 呢？”事实证明，纯函数式编程语言，比如 Haskell 或 Lisp，也会优化这样的操作；`pureSort`的实现只会移动指针，并且只有在指向的值之一发生变化时才会分配更多的内存。然而，这是两种不同的上下文；C++必须支持多种编程范式，而 Haskell 或 Lisp 则优化了不可变性和函数式风格。我们将在第十章中进一步讨论优化，即*性能优化*。现在，让我们来看看如何使这些类型的函数成为纯函数。

我们已经发现了如何处理一个输出参数。但是我们如何编写纯函数，使其具有多个输出参数呢？让我们考虑以下例子：

```cpp
void incrementAll(int& first, int& second){
    ++first;
    ++second;
}
```

解决这个问题的一个简单方法是用`vector<int>`替换这两个参数。但是如果参数具有不同的类型会怎么样？那么，我们可以使用一个结构体。但如果这是我们唯一需要它的时候呢？幸运的是，STL 提供了解决这个问题的方法，即通过元组：

```cpp
const tuple<int, int> incrementAllPure(const int& first, const int&  
    second){
        return make_tuple(first + 1, second + 1);
 }
 int main(){
     auto results = incrementAllPure(1, 2);
     // Can also use a simplified version
     // auto [first, second] = incrementAllPure(1, 2);
     cout << "Incremented pure: " << get<0>(results) << endl;
     cout << "Incremented pure: " << get<1>(results) << endl;
 }
```

元组有许多优点，如下所示：

+   它们可以用于多个值。

+   这些值可以具有不同的数据类型。

+   它们易于构建——只需一个函数调用。

+   它们不需要额外的数据类型。

根据我的经验，当您尝试将具有多个输出参数的函数渲染为纯函数，或者返回值和输出参数时，元组是一个很好的解决方案。但是，我经常在设计完成后尝试将它们重构为命名的*struct*或数据类。尽管如此，使用元组是一个非常有用的技术；只是要适度使用。

到目前为止，我们已经使用了很多`static`函数。但它们不是不好的实践吗？嗯，这取决于很多因素；我们将在接下来更详细地讨论这个问题。

# `static`函数不是不好的实践吗？

到目前为止，您可能会想知道纯函数是否好，因为它们与**面向对象编程**（**OOP**）或干净的代码规则相矛盾，即避免使用`static`。然而，直到现在，我们只编写了`static`函数。那么，它们是好的还是坏的呢？

使用`static`函数有两个反对意见。

对`static`函数的第一个反对意见是它们隐藏了全局状态。由于`static`函数只能访问`static`值，这些值就成为了全局状态。全局状态是不好的，因为很难理解是谁改变了它，当其值出乎意料时也很难调试。

但要记住纯函数的规则——纯函数应该对相同的输入值返回相同的输出值。因此，只有当函数不依赖于全局状态时，函数才是纯的。即使程序有状态，所有必要的值也作为输入参数发送给纯函数。不幸的是，我们无法轻易地通过编译器来强制执行这一点；避免使用任何类型的全局变量并将其转换为参数，这必须成为程序员的实践。

对于这种情况，特别是在使用全局常量时有一个特例。虽然常量是不可变状态，但考虑它们的演变也很重要。例如，考虑以下代码：

```cpp
static const string CURRENCY="EUR";
```

在这里，你应该知道，总会有一个时刻，常量会变成变量，然后你将不得不改变大量的代码来实现新的要求。我的建议是，通常最好也将常量作为参数传递进去。

对`static`函数的第二个反对意见是它们不应该是类的一部分。我们将在接下来的章节中更详细地讨论这一观点；暂且可以说，类应该将具有内聚性的函数分组在一起，有时纯函数应该在类中整齐地组合在一起。将具有内聚性的纯函数分组在一个类中还有另一种选择——只需使用一个命名空间。

幸运的是，我们不一定要在类中使用`static`函数。

# 静态函数的替代方案

我们在前一节中发现了如何通过使用`static`函数在`Number`类中编写纯函数：

```cpp
class Number{
    public:
        static int zero(){ return 0; }
        static int increment(const int& value){ return value + 1; }
        static int add(const int& first, const int& second){ return  
        first + second; }
};
```

然而，还有另一种选择；C++允许我们避免`static`，但保持函数不可变：

```cpp
class Number{
    public:
        int zero() const{ return 0; }
        int increment(const int& value) const{ return value + 1; }
        int add(const int& first, const int& second) const{ return 
        first + second; }
};
```

每个函数签名后面的`const`关键字只告诉我们该函数可以访问`Number`类的数据成员，但永远不能改变它们。

如果我们稍微改变这段代码，我们可以在类的上下文中提出一个有趣的不可变性问题。如果我们用一个值初始化数字，然后总是加上初始值，我们就得到了以下代码：

```cpp
class Number{
    private:
        int initialValue;

    public:
        Number(int initialValue) : initialValue(initialValue){}
        int initial() const{ return initialValue; }
        int addToInitial(const int& first) const{ return first + 
        initialValue; }
};

int main(){
    Number number(10);
    cout << number.addToInitial(20) << endl;
}
```

这里有一个有趣的问题：`addToInitial`函数是纯的吗？让我们按照以下标准来检查：

+   它有副作用吗？不，它没有。

+   它对相同的输入值返回相同的输出值吗？这是一个棘手的问题，因为函数有一个隐藏的参数，即`Number`类或其初始值。然而，没有人可以从`Number`类的外部改变`initialValue`。换句话说，`Number`类是不可变的。因此，该函数将对相同的`Number`实例和相同的参数返回相同的输出值。

+   它改变了参数的值吗？嗯，它只接收一个参数，并且不改变它。

结果是函数实际上是纯的。我们将在下一章中发现它也是*部分应用函数*。

我们之前提到程序中的一切都可以是纯的，除了 I/O。那么，我们对执行 I/O 的代码怎么办？

# 纯函数和 I/O

看一下以下内容，并考虑该函数是否是纯的：

```cpp
void printResults(){
    int* pValue = new int(10);
    cout << "Address: " << pValue << endl;
    cout << "Increment pointer address and value pure:" <<    
    incrementPointerAddressAndValuePure(pValue) << endl;
    cout << "Address after increment: " << pValue << endl;
    cout << "Value after increment: " << *pValue << endl;
    delete pValue;
}
```

好吧，让我们看看——它没有参数，所以值没有改变。但与我们之前的例子相比，有些不对劲，也就是它没有返回值。相反，它调用了一些函数，其中至少有一个是纯的。

那么，它有副作用吗？嗯，几乎每行代码都有一个：

```cpp
cout << ....
```

这行代码在控制台上写了一行字符串，这是一个副作用！`cout`基于可变状态，因此它不是一个纯函数。此外，由于它的外部依赖性，`cout`可能会失败，导致异常。

尽管我们的程序中需要 I/O，但我们可以做什么呢？嗯，很简单——只需将可变部分与不可变部分分开。将副作用与非副作用分开，并尽量减少不纯的函数。

那么，我们如何在这里实现呢？嗯，有一个纯函数等待从这个不纯函数中脱颖而出。关键是从问题开始；所以，让我们将`cout`分离如下：

```cpp
string formatResults(){
    stringstream output;
    int* pValue = new int(500);
    output << "Address: " << pValue << endl;
    output << "Increment pointer address and value pure:" << 
    incrementPointerAddressAndValuePure(pValue) << endl;
    output << "Address after increment: " << pValue << endl;
    output << "Value after increment: " << *pValue << endl;
    delete pValue;
    return output.str();
}

void printSomething(const string& text){
    cout << text;
}

printSomething(formatResults());
```

我们将由`cout`引起的副作用移到另一个函数中，并使初始函数的意图更清晰——即格式化而不是打印。看起来我们很干净地将纯函数与不纯函数分开了。

但是我们真的吗？让我们再次检查`formatResults`。它没有副作用，就像以前一样。我们正在使用`stringstream`，这可能不是纯函数，并且正在分配内存，但所有这些都是函数内部的局部变量。

内存分配是副作用吗？分配内存的函数可以是纯函数吗？毕竟，内存分配可能会失败。但是，在函数中几乎不可能避免某种形式的内存分配。因此，我们将接受一个纯函数可能会在某种内存失败的情况下失败。

那么，它的输出呢？它会改变吗？嗯，它没有输入参数，但它的输出可以根据`new`运算符分配的内存地址而改变。所以，它还不是一个纯函数。我们如何使它成为纯函数呢？这很容易——让我们传入一个参数，`pValue`：

```cpp
string formatResultsPure(const int* pValue){
    stringstream output;
    output << "Address: " << pValue << endl;
    output << "Increment pointer address and value pure:" << 
    incrementPointerAddressAndValuePure(pValue) << endl;
    output << "Address after increment: " << pValue << endl;
    output << "Value after increment: " << *pValue << endl;
    return output.str();
}

int main(){
    int* pValue = new int(500);
    printSomething(formatResultsPure(pValue));
    delete pValue;
}
```

在这里，我们使自己与副作用和可变状态隔离。代码不再依赖 I/O 或`new`运算符。我们的函数是纯的，这带来了额外的好处——它只做一件事，更容易理解它的作用，可预测，并且我们可以很容易地测试它。

关于具有副作用的函数，考虑以下代码：

```cpp
void printSomething(const string& text){
    cout << text;
}
```

我认为我们都可以同意，很容易理解它的作用，只要我们的其他函数都是纯函数，我们可以安全地忽略它。

总之，为了获得更可预测的代码，我们应该尽可能地将纯函数与不纯函数分开，并尽可能将不纯函数推到系统的边界。在某些情况下，这种改变可能很昂贵，拥有不纯函数在代码中也是完全可以的。只要确保你知道哪个是哪个。

# 总结

在本章中，我们探讨了如何在 C++中编写纯函数。由于有一些需要记住的技巧，这里是推荐的语法列表：

+   通过值传递的类函数：

+   `static int increment(const int value)`

+   `int increment(const int value) const`

+   通过引用传递的类函数：

+   `static int increment(const int& value)`

+   `int increment(const int&value) const`

+   通过值传递指针的类函数：

+   `static const int* increment(const int* const value)`

+   `const int* increment(const int* const value) const`

+   通过引用传递的类函数：

+   `static const int* increment(const int* const& value)`

+   `const int* increment(const int* const& value) const`

+   通过值传递的独立函数：`int increment(const int value)`

+   通过引用传递的独立函数：`int increment(const int& value)`

+   通过值传递指针的独立函数：`const int* increment(const int* value)`

+   通过引用传递的独立函数：`const int* increment(const int* const& value)`

我们还发现，虽然编译器有助于减少副作用，但并不总是告诉我们函数是纯函数还是不纯函数。我们始终需要记住编写纯函数时要使用的标准，如下所示：

+   它总是对相同的输入值返回相同的输出值。

+   它没有副作用。

+   它不会改变输入参数的值。

最后，我们看到了如何将通常与 I/O 相关的副作用与我们的纯函数分离。这很容易，通常需要传入值并提取函数。

现在是时候向前迈进了。当我们将函数视为设计的一等公民时，我们可以做更多事情。为此，我们需要学习 lambda 是什么以及它们如何有用。我们将在下一章中学习这个。

# 问题

1.  什么是纯函数？

1.  不可变性与纯函数有什么关系？

1.  你如何告诉编译器防止对按值传递的变量进行更改？

1.  你如何告诉编译器防止对按引用传递的变量进行更改？

1.  你如何告诉编译器防止对按引用传递的指针地址进行更改？

1.  你如何告诉编译器防止对指针指向的值进行更改？
