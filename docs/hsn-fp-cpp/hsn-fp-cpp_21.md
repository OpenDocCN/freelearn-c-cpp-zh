# 第十七章：评估

# 第一章

1.  什么是不可变函数？

不可变函数是一个不改变其参数值或程序状态的函数。

1.  如何编写一个不可变函数？

如果你希望编译器帮助你，将参数设为`const`。

1.  不可变函数如何支持代码简洁性？

因为它们不改变它们的参数，所以它们从代码中消除了任何潜在的复杂性，从而使程序员更好地理解它。

1.  不可变函数如何支持简单设计？

不可变函数很无聊，因为它们只做计算。因此，它们有助于长时间的维护。

1.  什么是高级函数？

高级函数是一个接收另一个函数作为参数的函数。

1.  STL 中可以给出哪些高级函数的例子？

STL 中有许多高级函数的例子，特别是在算法中。`sort`是我们在本章中使用的例子；然而，如果你查看`<algorithm>`头文件，你会发现许多其他例子，包括`find`、`find_if`、`count`、`search`等等。

1.  函数式循环相对于结构化循环的优势是什么？它们的潜在缺点是什么？

函数式循环避免了一次循环错误，并更清晰地表达了代码的意图。它们也是可组合的，因此可以通过链接多个循环来进行复杂的操作。然而，当组合时，它们需要多次通过集合，而这可以通过使用简单循环来避免。

1.  Alan Kay 的角度看 OOP 是什么？它如何与函数式编程相关？

Alan Kay 将 OOP 视为按细胞有机体原则构建代码的一种方式。细胞是通过化学信号进行通信的独立实体。因此，小对象之间的通信是 OOP 最重要的部分。

这意味着我们可以在表示为对象的数据结构上使用函数算法而不会产生任何冲突。

# 第二章

1.  什么是纯函数？

纯函数有两个约束条件，如下所示：

1.  +   它总是对相同的参数值返回相同的输出值。

+   它没有副作用。

1.  不可变性与纯函数有什么关系？

纯函数是不可变的，因为它们不会改变程序状态中的任何内容。

1.  如何告诉编译器防止传递的变量发生变化？

只需将参数定义为`const`，如下所示：

```cpp
int square(const int value)
```

1.  如何告诉编译器防止通过引用传递的变量发生变化？

只需将参数定义为`const&`，如下所示：

```cpp
int square(const int& value)
```

1.  如何告诉编译器防止通过引用传递的指针地址发生变化？

如果通过值传递指针，不需要任何操作，因为所有的更改都将局限于函数内部：

```cpp
int square(int* value)
```

如果通过引用传递指针，我们需要告诉编译器地址不能改变：

```cpp
int square(int*& const value)
```

1.  如何告诉编译器防止指针指向的值发生变化？

如果通过值传递指针，我们将应用与通过值传递的简单值相同的规则：

```cpp
int square(const int* value)
```

为了防止通过引用传递指针时对值和地址的更改，需要更多地使用`const`关键字：

```cpp
int square(const int&* const value)
```

# 第三章

1.  你可以写一个最简单的 lambda 吗？

最简单的 lambda 不接收参数并返回一个常量；可以是以下内容：

```cpp
auto zero = [](){return 0;};
```

1.  如何编写一个连接作为参数传递的两个字符串值的 lambda？

根据您喜欢的字符串连接方式，这个答案有几种变化。使用 STL 的最简单方法如下：

```cpp
auto concatenate = [](string first, string second){return first + second;};
```

1.  如果其中一个值是按值捕获的变量怎么办？

答案类似于前面的解决方案，但使用上下文中的值：

```cpp
auto concatenate = first{return first + second;};
```

当然，我们也可以使用默认的按值捕获符号，如下所示：

```cpp
auto concatenate = ={return first + second;};
```

1.  如果其中一个值是通过引用捕获的变量怎么办？

与前一个解决方案相比，除非您想要防止值的更改，否则几乎没有变化，如下所示：

```cpp
auto concatenate = &first{return first + second;};
```

如果要防止值的更改，我们需要转换为`const`：

```cpp
auto concatenate = &firstValue = as_const(first){return firstValue + second;};
```

1.  如果其中一个值是以值方式捕获的指针会怎样？

我们可以忽略不可变性，如下所示：

```cpp
auto concatenate = ={return *pFirst + second;};
```

或者，我们可以使用指向`const`类型的指针：

```cpp
const string* pFirst = new string("Alex");
auto concatenate = ={return *pFirst + second;};
```

或者，我们可以直接使用该值，如下所示：

```cpp
string* pFirst = new string("Alex");
first = *pFirst;
auto concatenate = ={return first + second;}
```

1.  如果其中一个值是以引用方式捕获的指针会怎样？

这使我们可以在 lambda 内部更改指向的值和指针地址。

最简单的方法是忽略不可变性，如下所示：

```cpp
auto concatenate = &{return *pFirst + second;};
```

如果我们想要限制不可变性，我们可以使用转换为`const`：

```cpp
auto concatenate = &first = as_const(pFirst){return *first + second;};
```

然而，通常最好的方法是直接使用该值，如下所示：

```cpp
string first = *pFirst;
auto concatenate = ={return first + second;};
```

1.  如果两个值都使用默认捕获说明符以值方式捕获，会怎么样？

这个解决方案不需要参数，只需要从上下文中捕获两个值：

```cpp
auto concatenate = [=](){return first + second;};
```

1.  如果两个值都使用默认捕获说明符以引用方式捕获，会怎么样？

如果我们不关心值的变化，我们可以这样做：

```cpp
auto concatenate = [&](){return first + second;};
```

为了保持不可变性，我们需要将其转换为`const`：

```cpp
auto concatenate = [&firstValue = as_const(first), &secondValue = as_const(second)](){return firstValue + secondValue;}
```

只使用默认的引用捕获说明符无法确保不可变性。请改用值方式捕获。

1.  如何在具有两个字符串值作为数据成员的类中将相同的 lambda 写为数据成员？

在类中，我们需要指定 lambda 变量的类型以及是否捕获两个数据成员或 this。

以下代码显示了如何使用`[=]`语法以复制方式捕获值：

```cpp
function<string()> concatenate = [=](){return first + second;};
```

以下代码显示了如何捕获`this`：

```cpp
function<string()> concatenate = [this](){return first + second;};
```

1.  如何在同一类中将相同的 lambda 写为静态变量？

我们需要将数据成员作为参数接收，如下所示：

```cpp
static function<string()> concatenate;
...
function<string()> AClass::concatenate = [](string first, string second){return first + second;};
```

我们已经看到，这比传递整个`AClass`实例作为参数更好，因为它减少了函数和类之间的耦合区域。

# 第四章

1.  什么是函数组合？

函数组合是函数的操作。它接受两个函数*f*和*g*，并创建第三个函数*C*，对于任何参数*x*，*C(x) = f(g(x))*。

1.  函数组合具有通常与数学操作相关联的属性。它是什么？

函数组合不是可交换的。例如，对一个数字的增量进行平方不同于对一个数字的平方进行增量。

1.  如何将带有两个参数的加法函数转换为带有一个参数的两个函数？

考虑以下函数：

```cpp
auto add = [](const int first, const int second){ return first + second; };
```

我们可以将前面的函数转换为以下形式：

```cpp
auto add = [](const int first){ 
    return first{
        return first + second;
    };
};
```

1.  如何编写一个包含两个单参数函数的 C++函数？

在本章中，我们看到借助模板和`auto`类型的魔力，这是非常容易做到的：

```cpp
template <class F, class G>
auto compose(F f, G g){
  return ={return f(g(value));};
}
```

1.  函数组合的优势是什么？

函数组合允许我们通过组合非常简单的函数来创建复杂的行为。此外，它允许我们消除某些类型的重复。它还通过允许以无限方式重新组合小函数来提高重用的可能性。

1.  实现函数操作的潜在缺点是什么？

函数的操作可以有非常复杂的实现，并且可能变得非常难以理解。抽象是有代价的，程序员必须始终平衡可组合性和小代码的好处与使用抽象操作的成本。

# 第五章

1.  什么是部分函数应用？

部分函数应用是从一个接受*N*个参数的函数中获取一个新函数的操作，该函数通过将其中一个参数绑定到一个值来接受*N-1*个参数。

1.  什么是柯里化？

柯里化是将接受*N*个参数的函数拆分为*N*个函数的操作，每个函数接受一个参数。

1.  柯里化如何帮助实现部分应用？

给定柯里化函数*f(x)(y)*，对*x = value*的*f*的部分应用可以通过简单地像这样调用*f*来获得：*g = f(value)*。

1.  **我们如何在 C++中实现部分应用？**

部分应用可以在 C++中手动实现，但使用`functional`头文件中的`bind`函数来实现会更容易。
