# Lambda 表达式

本章中我们将涵盖以下内容：

+   使用 lambda 表达式在运行时定义函数

+   通过将 lambda 包装到`std::function`中添加多态性

+   通过连接组合函数

+   使用逻辑连接创建复杂的谓词

+   使用相同的输入调用多个函数

+   使用`std::accumulate`和 lambda 实现`transform_if`

+   在编译时生成任意输入的笛卡尔积对

# 介绍

C++11 的一个重要新特性是**lambda 表达式**。在 C++14 和 C++17 中，lambda 表达式得到了一些新的添加，使它们变得更加强大。但首先，*什么是* lambda 表达式？

Lambda 表达式或 lambda 函数构造闭包。闭包是一个非常通用的术语，用来描述可以像函数一样*调用*的*无名对象*。为了在 C++ 中提供这样的能力，这样的对象必须实现`()`函数调用运算符，可以带参数也可以不带参数。在 C++11 之前，构造这样的对象而不使用 lambda 表达式可能看起来像下面这样：

```cpp
#include <iostream>
#include <string>

int main() {
    struct name_greeter {
        std::string name;

        void operator()() {
            std::cout << "Hello, " << name << 'n'; 
        }
    };

    name_greeter greet_john_doe {"John Doe"};
    greet_john_doe();
}
```

`name_greeter` 结构的实例显然携带一个字符串。请注意，这种结构类型和实例都不是无名的，但是 lambda 表达式可以是无名的，我们将会看到。就闭包而言，我们会说它们*捕获*了一个字符串。当像没有参数的函数一样调用示例实例时，它会打印出`"Hello, John Doe"`，因为我们用这个名字构造了它。

自从 C++11 以来，创建这样的闭包变得更加容易：

```cpp
#include <iostream>

int main() {
    auto greet_john_doe ([] {
        std::cout << "Hello, John Doen"; 
    });

    greet_john_doe();
}
```

就是这样。整个`name_greeter`结构都被一个小小的`[] { /* do something */ }`构造替代了，这一开始可能看起来有点像魔术，但本章的第一部分将会详细解释它的所有可能变体。

Lambda 表达式对于使代码*通用*和*整洁*非常有帮助。它们可以作为参数用于非常通用的算法，以便在处理特定用户定义类型时专门化它们的操作。它们还可以用于将工作包装在一起，包括数据，以便在线程中运行，或者只是保存工作并推迟实际执行。自从 C++11 推出以来，越来越多的库使用 lambda 表达式，因为它们在 C++ 中变得非常自然。另一个用例是元编程，因为 lambda 表达式也可以在编译时进行评估。然而，我们不会深入*那个*方向，因为这会很快超出本书的范围。

本章在很大程度上依赖一些*函数式编程*模式，这可能对新手或已经有经验但不熟悉这些模式的程序员看起来很奇怪。如果在接下来的示例中看到返回 lambda 表达式的 lambda 表达式，再返回 lambda 表达式，请不要感到沮丧或迷惑得太快。我们正在推动边界，以便为现代 C++ 做准备，在那里函数式编程模式越来越频繁地出现。如果在接下来的示例中看到一些代码看起来有点太复杂，请花点时间去理解它。一旦你通过了这一点，在野外的真实项目中复杂的 lambda 表达式将不再让你困惑。

# 使用 lambda 表达式在运行时定义函数

使用 lambda 表达式，我们可以封装代码以便以后调用，而且也可能在其他地方调用，因为我们可以复制它们。我们也可以封装代码以便以稍微不同的参数多次调用它，而不必为此实现一个全新的函数类。

Lambda 表达式的语法在 C++11 中是全新的，它在接下来的两个标准版本中略有变化，直到 C++17。在本节中，我们将看到 lambda 表达式的样子和含义。

# 如何做...

我们将编写一个小程序，其中我们将使用 lambda 表达式来熟悉它们：

1.  Lambda 表达式不需要任何库支持，但我们将向终端写入消息并使用字符串，因此我们需要这些头文件：

```cpp
      #include <iostream>
      #include <string>
```

1.  这次所有的事情都发生在主函数中。我们定义了两个不带参数并返回整数常量值`1`和`2`的函数对象。请注意，返回语句被大括号`{}`包围，就像在普通函数中一样，`()`括号表示无参数函数，是*可选的*，我们在第二个 lambda 表达式中没有提供它们。但`[]`括号必须在那里：

```cpp
      int main()
      {
          auto just_one ( [](){ return 1; } );
          auto just_two ( []  { return 2; } );
```

1.  现在，我们可以通过只写它们保存的变量的名称并附加括号来调用这两个函数对象。在这一行中，它们对于读者来说与*普通函数*是无法区分的：

```cpp
          std::cout << just_one() << ", " << just_two() << 'n';
```

1.  现在让我们忘记这些，定义另一个函数对象，称为`plus`，因为它接受两个参数并返回它们的和：

```cpp
          auto plus ( [](auto l, auto r) { return l + r; } );
```

1.  这也很容易使用，就像任何其他二进制函数一样。由于我们将其参数定义为`auto`类型，它将与定义了加法运算符`+`的任何东西一起工作，就像字符串一样：

```cpp
          std::cout << plus(1, 2) << 'n';
          std::cout << plus(std::string{"a"}, "b") << 'n';
```

1.  我们不需要将 lambda 表达式存储在变量中才能使用它。我们也可以*就地*定义它，然后在其后面的括号中写入参数`(1, 2)`：

```cpp
          std::cout 
            << [](auto l, auto r){ return l + r; }(1, 2) 
            << 'n';
```

1.  接下来，我们将定义一个闭包，它携带一个整数计数器值。每当我们调用它时，它会增加其计数器值并返回新值。为了告诉它它有一个内部计数器变量，我们在括号内写入`count = 0`，告诉它有一个初始化为整数值`0`的变量`count`。为了允许它修改自己捕获的变量，我们使用`mutable`关键字，因为否则编译器不会允许它：

```cpp
          auto counter (
              [count = 0] () mutable { return ++count; }
          );
```

1.  现在，让我们调用函数对象五次并打印它返回的值，这样我们以后可以看到递增的数字值：

```cpp
          for (size_t i {0}; i < 5; ++i) {
              std::cout << counter() << ", ";
          }
          std::cout << 'n';
```

1.  我们还可以获取现有变量并通过*引用*捕获它们，而不是给闭包自己的值副本。这样，捕获的变量可以被闭包递增，但在外部仍然可以访问。为了这样做，我们在括号之间写入`&a`，其中`&`表示我们只存储对变量的*引用*，而不是*副本*：

```cpp
          int a {0};
          auto incrementer ( [&a] { ++a; } );
```

1.  如果这样做有效，那么我们应该能够多次调用这个函数对象，然后观察它是否真的改变了变量`a`的值：

```cpp
          incrementer();
          incrementer();
          incrementer();

          std::cout 
            << "Value of 'a' after 3 incrementer() calls: " 
            << a << 'n';
```

1.  最后一个例子是*柯里化*。柯里化意味着我们接受一些参数的函数并将其存储在另一个函数对象中，该函数对象接受*更少*的参数。在这种情况下，我们存储`plus`函数并只接受*一个*参数，然后将其转发给`plus`函数。另一个参数是值`10`，我们将其保存在函数对象中。这样，我们得到一个函数，我们称之为`plus_ten`，因为它可以将该值添加到它接受的单个参数中：

```cpp
          auto plus_ten ( [=] (int x) { return plus(10, x); } );
          std::cout << plus_ten(5) << 'n';
      }
```

1.  在编译和运行程序之前，再次检查代码并尝试预测它将打印到终端的内容。然后运行它并检查实际输出：

```cpp
      1, 2
      3
      ab
      3
      1, 2, 3, 4, 5, 
      Value of a after 3 incrementer() calls: 3
      15
```

# 它是如何工作的...

我们刚刚做的并不是过于复杂--我们添加了数字，并递增和打印它们。我们甚至用一个函数对象连接了字符串，该函数对象被实现为将数字相加。但是对于那些尚不了解 lambda 表达式语法的人来说，这可能看起来很困惑。

所以，让我们首先看一下所有 lambda 表达式的特点：

![](img/8d0ec8da-5bcf-4a59-945e-35aeb40addfe.png)

通常情况下，我们可以省略大部分内容，这样可以节省一些输入，平均情况下，最短的 lambda 表达式可能是`[]{}`。它不接受任何参数，不捕获任何内容，本质上*什么也不做*。

那么剩下的是什么意思？

# 捕获列表

指定我们是否以及捕获了什么。有几种形式可以这样做。还有两种懒惰的变体：

+   如果我们写`[=] () {...}`，我们通过值捕获闭包从外部引用的每个变量，这意味着值会*被复制*

+   写`[&] () {...}`意味着闭包引用外部的一切都只通过*引用*捕获，不会导致复制。

当然，我们可以为每个变量单独设置捕获设置。写`[a, &b] () {...}`意味着我们通过*值*捕获变量`a`，通过*引用*捕获`b`。这是更多的打字工作，但通常更安全，因为我们不能意外地从外部捕获我们不想捕获的东西。

在这个示例中，我们将 lambda 表达式定义为`[count=0] () {...}`。在这种特殊情况下，我们没有从外部捕获任何变量，而是定义了一个名为`count`的新变量。它的类型是从我们初始化它的值中推断出来的，即`0`，所以它是一个`int`。

也可以通过值和引用来捕获一些变量，如：

+   `[a, &b] () {...}`：这通过复制捕获`a`，通过引用捕获`b`。

+   `[&, a] () {...}`：这通过复制捕获`a`，并通过引用捕获任何其他使用的变量。

+   `[=, &b, i{22}, this] () {...}`：这通过引用捕获`b`，通过复制捕获`this`，用值`22`初始化一个新变量`i`，并通过复制捕获任何其他使用的变量。

如果尝试捕获对象的成员变量，不能直接使用`[member_a] () {...}`。相反，必须捕获`this`或`*this`。

# mutable（可选）

如果函数对象应该能够*修改*它通过*复制*（`[=]`）捕获的变量，必须定义为`mutable`。这包括调用捕获对象的非 const 方法。

# constexpr（可选）

如果将 lambda 表达式明确标记为`constexpr`，如果不满足`constexpr`函数的条件，编译器将*报错*。`constexpr`函数和 lambda 表达式的优势在于，如果它们使用编译时常量参数调用，编译器可以在编译时评估它们的结果。这会导致后期二进制代码量减少。

如果我们没有明确声明 lambda 表达式为`constexpr`，但它符合要求，它将隐式地成为`constexpr`。如果我们*想要*一个 lambda 表达式是`constexpr`，最好是明确声明，因为编译器会在我们*错误*时帮助我们报错。

# 异常属性（可选）

这是指定函数对象在调用时是否能抛出异常并遇到错误情况的地方。

# 返回类型（可选）

如果我们想要对返回类型有终极控制，可能不希望编译器自动推断它。在这种情况下，我们可以写`[] () -> Foo {}`，告诉编译器我们确实总是返回`Foo`类型。

# 通过将 lambda 包装到 std::function 中添加多态性

假设我们想为某种可能会偶尔改变的值编写一个观察者函数，然后通知其他对象；比如气压指示器，或者股票价格，或者类似的东西。每当值发生变化时，应该调用一个观察者对象列表，然后它们做出反应。

为了实现这一点，我们可以在向量中存储一系列观察者函数对象，它们都接受一个`int`变量作为参数，表示观察到的值。我们不知道这些函数对象在调用新值时具体做什么，但我们也不在乎。

那个函数对象的向量将是什么类型？如果我们捕获具有`void f(int);`这样签名的*函数*的指针，那么`std::vector<void (*)(int)>`类型将是正确的。这实际上也适用于不捕获任何变量的任何 lambda 表达式，例如`[](int x) {...}`。但是，捕获某些东西的 lambda 表达式实际上是*完全不同的类型*，因为它不仅仅是一个函数指针。它是一个*对象*，它将一定数量的数据与一个函数耦合在一起！想想 C++11 之前的时代，当时没有 lambda。类和结构是将数据与函数耦合在一起的自然方式，如果更改类的数据成员类型，您将得到完全不同的类类型。一个向量不能使用相同的类型名称存储完全不同的类型，这是*自然*的。

告诉用户只能保存不捕获任何东西的观察者函数对象是不好的，因为它非常限制了使用情况。我们如何允许用户存储任何类型的函数对象，只限制调用接口，该接口接受表示将被观察的值的特定参数集？

这一部分展示了如何使用`std::function`解决这个问题，它可以作为任何 lambda 表达式的多态包装，无论它捕获了什么。

# 如何做...

在这一部分，我们将创建几个完全不同的 lambda 表达式，它们在捕获的变量类型方面完全不同，但在共同的函数调用签名方面相同。这些将被保存在一个使用`std::function`的向量中：

1.  让我们首先做一些必要的包含：

```cpp
      #include <iostream>
      #include <deque>
      #include <list>
      #include <vector>
      #include <functional>
```

1.  我们实现了一个小函数，它返回一个 lambda 表达式。它接受一个容器并返回一个捕获该容器的函数对象。函数对象本身接受一个整数参数。每当该函数对象被提供一个整数时，它将*追加*该整数到它捕获的容器中：

```cpp
      static auto consumer (auto &container){
          return [&] (auto value) {
              container.push_back(value);
          };
      }
```

1.  另一个小的辅助函数将打印我们提供的任何容器实例：

```cpp
      static void print (const auto &c)
      {
          for (auto i : c) {
              std::cout << i << ", ";
          }
          std::cout << 'n';
      }
```

1.  在主函数中，我们首先实例化了一个`deque`，一个`list`和一个`vector`，它们都存储整数：

```cpp
      int main()
      {
          std::deque<int>  d;
          std::list<int>   l;
          std::vector<int> v;
```

1.  现在我们使用`consumer`函数与我们的容器实例`d`，`l`和`v`：我们为这些产生消费者函数对象，并将它们全部存储在一个`vector`实例中。然后我们有一个存储三个函数对象的向量。这些函数对象每个都捕获一个对容器对象的引用。这些容器对象是完全不同的类型，所以函数对象也是完全不同的类型。尽管如此，向量持有`std::function<void(int)>`的实例。所有函数对象都被隐式地包装成这样的`std::function`对象，然后存储在向量中：

```cpp
          const std::vector<std::function<void(int)>> consumers 
              {consumer(d), consumer(l), consumer(v)};
```

1.  现在，我们通过循环遍历值并循环遍历消费者函数对象，将 10 个整数值输入所有数据结构，然后调用这些值：

```cpp
          for (size_t i {0}; i < 10; ++i) {
              for (auto &&consume : consumers) {
                  consume(i);
              }
          }
```

1.  现在所有三个容器应该包含相同的 10 个数字值。让我们打印它们的内容：

```cpp
          print(d);
          print(l);
          print(v);
      }
```

1.  编译和运行程序产生了以下输出，这正是我们所期望的：

```cpp
      $ ./std_function
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
```

# 它是如何工作的...

这个食谱的复杂部分是以下行：

```cpp
const std::vector<std::function<void(int)>> consumers 
        {consumer(d), consumer(l), consumer(v)};
```

对象`d`，`l`和`v`都被包装到一个`consumer(...)`调用中。这个调用返回函数对象，然后每个函数对象都捕获了`d`，`l`和`v`中的一个引用。尽管这些函数对象都接受`int`值作为参数，但它们捕获完全*不同*的变量的事实也使它们完全不同的*类型*。这就像试图将类型为`A`，`B`和`C`的变量塞进一个向量中，尽管这些类型*没有*任何共同之处。

为了修复这个问题，我们需要找到一个可以存储非常*不同*的函数对象的*通用*类型，也就是`std::function`。一个`std::function<void(int)>`对象可以存储任何接受整数参数并返回空的函数对象或传统函数。它使用多态性将其类型与底层函数对象类型分离。考虑我们写这样的东西：

```cpp
std::function<void(int)> f (
    &vector { vector.push_back(x); });
```

这里，从 lambda 表达式构造的函数对象被包装到了一个`std::function`对象中，每当我们调用`f(123)`时，这将导致一个*虚函数调用*，它被*重定向*到其中的实际函数对象。

在存储函数对象时，`std::function`实例应用了一些智能。如果我们在 lambda 表达式中捕获了越来越多的变量，它必须变得更大。如果它的大小不是太大，`std::function`可以将其存储在自身内部。如果存储的函数对象的大小太大，`std::function`将在堆上分配一块内存，然后将大的函数对象存储在那里。这不会影响我们代码的功能，但我们应该知道这一点，因为这可能会影响我们代码的*性能*。

很多新手程序员认为或希望`std::function<...>`实际上表达了 lambda 表达式的*类型*。不，它不是。它是一个多态库助手，用于包装 lambda 表达式并擦除它们的类型差异。

# 通过连接组合函数

很多任务实际上并不值得完全自定义代码来实现。例如，让我们看看程序员如何使用 Haskell 编程语言解决查找文本包含多少个唯一单词的任务。第一行定义了一个名为`unique_words`的函数，第二行演示了它在一个示例字符串中的使用：

![](img/d12ee785-b9b9-4b8e-9ba5-8c42f81de022.png)

哇，这太简短了！不多解释 Haskell 语法，让我们看看代码做了什么。它定义了一个名为`unique_words`的函数，它将一系列函数应用于其输入。它首先使用`map toLower`将输入的所有字符映射为小写。这样，像`FOO`和`foo`这样的单词可以被视为*相同*的单词。然后，`words`函数将一个句子拆分为单独的单词，例如从`"foo bar baz"`到`["foo", "bar", "baz"]`。下一步是对新的单词列表进行排序。这样，一个单词序列，比如`["a", "b", "a"]`就变成了`["a", "a", "b"]`。现在，`group`函数接管了。它将连续相同的单词分组成分组列表，所以`["a", "a", "b"]`变成了`[ ["a", "a"], ["b"] ]`。工作现在几乎完成了，因为我们现在只需要计算有多少*组*相同的单词，这正是`length`函数所做的。

这是一种*奇妙*的编程风格，因为我们可以从右到左读取*发生*的事情，因为我们只是在描述一个转换管道。我们不需要关心个别部分是如何实现的（除非它们是慢的或有 bug）。

然而，我们在这里不是为了赞扬 Haskell，而是为了提高我们的 C++技能。在 C++中也可以像这样工作。我们可能无法完全达到 Haskell 示例的优雅，但我们仍然拥有最快的编程语言。这个示例解释了如何使用 lambda 表达式在 C++中模拟*函数连接*。

# 如何做到...

在这一部分，我们定义了一些简单的玩具函数对象并*连接*它们，这样我们就得到了一个单一的函数，它将简单的玩具函数依次应用于我们给它的输入。为了做到这一点，我们编写了自己的连接辅助函数：

1.  首先，我们需要一些包含：

```cpp
      #include <iostream>
      #include <functional>
```

1.  然后，我们实现了辅助函数`concat`，它任意地接受许多参数。这些参数将是函数，比如`f`、`g`和`h`，结果将是另一个函数对象，它对任何输入应用`f(g(h(...)))`：

```cpp
      template <typename T, typename ...Ts>
      auto concat(T t, Ts ...ts)
      {
```

1.  现在，它变得有点复杂。当用户提供函数`f`，`g`和`h`时，我们将将其评估为`f( concat(g, h) )`，这再次扩展为`f( g( concat(h) ) )`，递归中止，因此我们得到`f( g( h(...) ) )`。这些用户函数的连接链被 lambda 表达式捕获，稍后可以接受一些参数`p`，然后将它们转发到`f(g(h(p)))`。这个 lambda 表达式就是我们返回的内容。`if constexpr`构造检查我们是否处于递归步骤中，剩下的要连接的函数多于一个：

```cpp
          if constexpr (sizeof...(ts) > 0) {
              return = { 
                  return t(concat(ts...)(parameters...)); 
              };
          }
```

1.  `if constexpr`构造的另一个分支是在递归的*末尾*时由编译器选择的。在这种情况下，我们只返回函数`t`，因为它是唯一剩下的参数：

```cpp
          else {
              return t;
          }
      }
```

1.  现在，让我们使用我们很酷的新函数连接助手与一些我们想要看到连接的函数。让我们从`main`函数开始，我们在其中定义两个简单的函数对象：

```cpp
      int main()
      {
          auto twice  ([] (int i) { return i * 2; });
          auto thrice ([] (int i) { return i * 3; });
```

1.  现在让我们进行连接。我们将我们的两个乘法函数对象与 STL 函数`std::plus<int>`进行连接，该函数接受两个参数并简单地返回它们的和。这样，我们得到一个执行`twice(thrice(plus(a, b)))`的函数。

```cpp
          auto combined (
              concat(twice, thrice, std::plus<int>{})
          );
```

1.  现在让我们使用它。`combined`函数现在看起来像一个普通的单一函数，编译器也能够连接这些函数，而没有任何不必要的开销：

```cpp
          std::cout << combined(2, 3) << 'n';
      }
```

1.  编译和运行我们的程序产生了以下输出，这也是我们预期的，因为`2 * 3 * (2 + 3)`是`30`：

```cpp
      $ ./concatenation
      30
```

# 工作原理...

这一部分的复杂之处在于`concat`函数。它看起来非常复杂，因为它将参数包`ts`解包到另一个 lambda 表达式中，该 lambda 表达式递归调用`concat`，并且参数更少：

```cpp
template <typename T, typename ...Ts>
auto concat(T t, Ts ...ts)
{
    if constexpr (sizeof...(ts) > 0) { 
        return = { 
            return t(concat(ts...)(parameters...)); 
        }; 
    } else {
        return = { 
            return t(parameters...); 
        };
    }
}
```

让我们编写一个更简单的版本，它精确地连接*三个*函数：

```cpp
template <typename F, typename G, typename H>
auto concat(F f, G g, H h)
{
    return = {
        return f( g( h( params... ) ) ); 
    };
}
```

这看起来已经很相似，但不那么复杂。我们返回一个 lambda 表达式，它捕获了`f`，`g`和`h`。这个 lambda 表达式任意接受许多参数，并将它们转发到`f`，`g`和`h`的调用链。当我们写`auto combined (concat(f, g, h))`，然后稍后用两个参数调用该函数对象，比如`combined(2, 3)`，那么`2, 3`将由前面的`concat`函数的`params`包表示。

再次看看更复杂的通用`concat`函数；我们真正不同的唯一一件事是`f ( g( h( params... ) ) )`的连接。相反，我们写`f( concat(g, h) )(params...)`，这在下一次递归调用中会评估为`f( g( concat(h) ) )(params...)`，然后最终结果为`f( g( h( params... ) ) )`。

# 使用逻辑连接创建复杂的谓词

在使用通用代码过滤数据时，我们最终会定义**谓词**，告诉我们想要什么数据，以及不想要什么数据。有时，谓词是不同谓词的*组合*。

例如，在过滤字符串时，我们可以实现一个谓词，如果其输入字符串以`"foo"`开头，则返回`true`。另一个谓词可以在其输入字符串以`"bar"`结尾时返回`true`。

我们可以通过组合来*重用*谓词，而不是一直编写自定义谓词。如果我们想要过滤以`"foo"`开头并以`"bar"`结尾的字符串，我们可以选择我们*现有*的谓词，并用逻辑*与*将它们*组合*起来。在本节中，我们将使用 lambda 表达式来寻找一种舒适的方法来做到这一点。

# 如何做...

我们将实现非常简单的字符串过滤谓词，然后我们将用一个小助手函数将它们以通用方式组合起来。

1.  像往常一样，我们首先包含一些头文件：

```cpp
      #include <iostream>
      #include <functional>
      #include <string>
      #include <iterator>
      #include <algorithm>
```

1.  因为我们以后会需要它们，我们实现了两个简单的谓词函数。第一个告诉我们一个字符串是否以字符`'a'`开头，第二个告诉我们一个字符串是否以字符`'b'`结尾：

```cpp
      static bool begins_with_a (const std::string &s)
      {
          return s.find("a") == 0;
      }

      static bool ends_with_b (const std::string &s)
      {
          return s.rfind("b") == s.length() - 1;
      }
```

1.  现在，让我们实现一个辅助函数，我们称之为`combine`。它以二进制函数作为第一个参数，这个函数可以是逻辑`AND`函数或逻辑`OR`函数，然后，它接受另外两个参数，这两个参数将被组合：

```cpp
      template <typename A, typename B, typename F>
      auto combine(F binary_func, A a, B b)
      {
```

1.  我们只需返回一个捕获新谓词*combination*的 lambda 表达式。它将一个参数转发到两个谓词，然后将两者的结果放入二进制函数中，并返回其结果：

```cpp
          return = {
              return binary_func(a(param), b(param));
          };
      }
```

1.  让我们声明我们在`main`函数中使用`std`命名空间来节省一些输入：

```cpp
      using namespace std;
```

1.  现在，让我们将两个谓词函数组合成另一个谓词函数，告诉我们给定的字符串是否以`a`开头*并且*以`b`结尾，就像`"ab"`或`"axxxb"`一样。作为二进制函数，我们选择`std::logical_and`。它是一个需要实例化的模板类，因此我们使用大括号来实例化它。请注意，我们没有提供模板参数，因为对于这个类，默认为`void`。这个类的特化自动推断所有参数类型：

```cpp
      int main()
      {
          auto a_xxx_b (combine(
              logical_and<>{}, 
              begins_with_a, ends_with_b));
```

1.  我们遍历标准输入，并将所有满足我们谓词的单词打印回终端：

```cpp
          copy_if(istream_iterator<string>{cin}, {},
                  ostream_iterator<string>{cout, ", "},
                  a_xxx_b);
          cout << 'n';
      }
```

1.  编译和运行程序产生以下输出。我们用四个单词输入程序，但只有两个满足谓词条件：

```cpp
      $ echo "ac cb ab axxxb" | ./combine
      ab, axxxb, 
```

# 还有更多...

STL 已经提供了一堆有用的函数对象，比如`std::logical_and`，`std::logical_or`，以及许多其他函数，因此我们不需要在每个项目中重新实现它们。查看 C++参考并探索已有的内容是一个好主意：

[`en.cppreference.com/w/cpp/utility/functional`](http://en.cppreference.com/w/cpp/utility/functional)

# 使用相同的输入调用多个函数

有很多任务会导致重复的代码。使用 lambda 表达式和一个包装这种重复任务的 lambda 表达式辅助函数可以很容易地消除大量重复的代码。

在本节中，我们将使用 lambda 表达式来转发一个带有所有参数的单个调用到多个接收者。这将在没有任何数据结构的情况下发生，因此编译器可以简单地生成一个没有开销的二进制文件。

# 如何做...

我们将编写一个 lambda 表达式辅助函数，将单个调用转发给多个对象，以及另一个 lambda 表达式辅助函数，将单个调用转发给其他函数的多个调用。在我们的示例中，我们将使用这个来使用不同的打印函数打印单个消息：

1.  首先让我们包含我们需要打印的 STL 头文件：

```cpp
      #include <iostream>
```

1.  首先，我们实现`multicall`函数，这是本教程的核心。它接受任意数量的函数作为参数，并返回一个接受一个参数的 lambda 表达式。它将此参数转发到之前提供的所有函数。这样，我们可以定义`auto call_all (multicall(f, g, h))`，然后`call_all(123)`导致一系列调用，`f(123); g(123); h(123);`。这个函数看起来非常复杂，因为我们需要一个语法技巧来展开参数包`functions`，通过使用`std::initializer_list`构造函数来进行一系列调用：

```cpp
      static auto multicall (auto ...functions)
      {
          return = {
              (void)std::initializer_list<int>{
                  ((void)functions(x), 0)...
              };
          };
      }
```

1.  下一个辅助函数接受一个函数`f`和一组参数`xs`。它的作用是对每个参数调用`f`。这样，`for_each(f, 1, 2, 3)`调用导致一系列调用：`f(1); f(2); f(3);`。这个函数本质上使用了与之前的其他函数相同的语法技巧，将参数包`xs`展开为一系列函数调用：

```cpp
      static auto for_each (auto f, auto ...xs) {
          (void)std::initializer_list<int>{
              ((void)f(xs), 0)...
          };
      }
```

1.  `brace_print`函数接受两个字符并返回一个新的函数对象，它接受一个参数`x`。它会*打印*它，用我们刚刚捕获的两个字符包围起来：

```cpp
      static auto brace_print (char a, char b) {
          return [=] (auto x) {
              std::cout << a << x << b << ", ";
          };
      }
```

1.  现在，我们终于可以在`main`函数中把所有东西都用起来了。首先，我们定义了`f`、`g`和`h`函数。它们代表接受值并将其打印在不同的大括号/括号中的打印函数。`nl`函数接受任何参数，只是打印一个换行字符：

```cpp
      int main()
      {
          auto f  (brace_print('(', ')'));
          auto g  (brace_print('[', ']'));
          auto h  (brace_print('{', '}'));
          auto nl ([](auto) { std::cout << 'n'; });
```

1.  让我们使用我们的`multicall`助手将它们全部组合起来：

```cpp
          auto call_fgh (multicall(f, g, h, nl));
```

1.  对于我们提供的每个数字，我们希望看到它们被不同的大括号/括号包围打印三次。这样，我们可以进行一次函数调用，最终得到五次对我们的多功能函数的调用，它又会调用`f`、`g`、`h`和`nl`四次。

```cpp
          for_each(call_fgh, 1, 2, 3, 4, 5);
      }
```

1.  在编译和运行之前，想一想期望的输出：

```cpp
      $ ./multicaller
      (1), [1], {1}, 
      (2), [2], {2}, 
      (3), [3], {3}, 
      (4), [4], {4}, 
      (5), [5], {5}, 
```

# 它是如何工作的...

我们刚刚实现的辅助函数看起来非常复杂。这是因为我们使用`std::initializer_list`来展开参数包。我们为什么要使用这种数据结构呢？让我们再看看`for_each`：

```cpp
auto for_each ([](auto f, auto ...xs) {
    (void)std::initializer_list<int>{
        ((void)f(xs), 0)...
    };
});
```

这个函数的核心是`f(xs)`表达式。`xs`是一个参数包，我们需要*展开*它，以便将其中的各个值取出并传递给各个`f`调用。不幸的是，我们不能只使用`...`符号写`f(xs)...`，这一点我们已经知道了。

我们可以使用`std::initializer_list`构造一个值列表，它具有可变参数的构造函数。诸如`return std::initializer_list<int>{f(xs)...};`这样的表达式可以胜任，但它有*缺点*。让我们看看`for_each`的一个实现，它只是这样做，所以它看起来比我们现在有的更简单：

```cpp
auto for_each ([](auto f, auto ...xs) {
    return std::initializer_list<int>{f(xs)...};
});
```

这更容易理解，但它的缺点是：

1.  它构造了一个实际的初始化器列表，其中包含所有`f`调用的返回值。在这一点上，我们不关心返回值。

1.  它*返回*了初始化列表，尽管我们想要一个*“发射并忘记”*的函数，它不返回*任何东西*。

1.  可能`f`是一个函数，甚至不返回任何东西，如果是这样，那么这甚至不会编译。

更复杂的`for_each`函数解决了所有这些问题。它做了以下几件事来实现这一点：

1.  它不是*返回*初始化列表，而是*将*整个表达式转换为`void`，使用`(void)std::initializer_list<int>{...}`。

1.  在初始化表达式中，它将`f(xs)...`包装成`(f(xs), 0)...`表达式。这导致返回值被*丢弃*，而`0`被放入初始化列表中。

1.  `(f(xs), 0)...`表达式中的`f(xs)`再次被转换为`void`，因此如果有的话，返回值确实没有被处理到任何地方。

将所有这些组合在一起不幸地导致了一个*丑陋*的结构，但它确实能正常工作，并且能够编译各种函数对象，无论它们是否返回任何东西或者返回什么。

这种技术的一个好处是，函数调用的顺序是有严格顺序保证的。

使用旧的 C 风格表示法`(void)expression`来转换任何东西是不建议的，因为 C++有自己的转换操作符。我们应该使用`reinterpret_cast<void>(expression)`，但这会进一步降低代码的*可读性*。

# 使用 std::accumulate 和 lambda 实现 transform_if

大多数使用`std::copy_if`和`std::transform`的开发人员可能已经问过自己，为什么没有`std::transform_if`。`std::copy_if`函数从源范围复制项目到目标范围，但会*跳过*用户定义的*谓词*函数未选择的项目。`std::transform`无条件地从源范围复制所有项目到目标范围，但在中间进行转换。转换由用户定义的函数提供，可能做简单的事情，比如乘以数字或将项目转换为完全不同的类型。

这样的函数现在已经存在很长时间了，但仍然没有`std::transform_if`函数。在本节中，我们将实现这个函数。通过实现一个函数，它在复制被谓词函数选择的所有项目的同时迭代范围，进行中间转换，这样做很容易。然而，我们将利用这个机会更深入地研究 lambda 表达式。

# 如何做...

我们将构建自己的`transform_if`函数，通过提供`std::accumulate`正确的函数对象来工作：

1.  我们需要像往常一样包含一些头文件：

```cpp
      #include <iostream>
      #include <iterator>
      #include <numeric>
```

1.  首先，我们将实现一个名为`map`的函数。它接受一个输入转换函数作为参数，并返回一个函数对象，它与`std::accumulate`很好地配合使用：

```cpp
      template <typename T>
      auto map(T fn)
      {
```

1.  我们返回的是一个接受*reduce*函数的函数对象。当这个对象被调用时，它会返回另一个函数对象，它接受一个*accumulator*和一个输入参数。它调用 reduce 函数对这个累加器和`fn`转换后的输入变量进行操作。如果这看起来很复杂，不要担心，我们稍后会把它整合在一起，看看它是如何真正工作的：

```cpp
          return [=] (auto reduce_fn) {
              return [=] (auto accum, auto input) {
                  return reduce_fn(accum, fn(input));
              };
          };
      }
```

1.  现在我们实现一个名为`filter`的函数。它的工作方式与`map`函数完全相同，但它保持输入*不变*，而`map`函数使用转换函数*转换*它。相反，我们接受一个谓词函数，并在不减少它们的情况下*跳过*输入变量，如果它们不被谓词函数接受：

```cpp
      template <typename T>
      auto filter(T predicate)
      {
```

1.  这两个 lambda 表达式与`map`函数中的表达式具有完全相同的函数签名。唯一的区别是`input`参数保持不变。谓词函数用于区分我们是在输入上调用`reduce_fn`函数，还是只是在不做任何更改的情况下将累加器向前推进：

```cpp
          return [=] (auto reduce_fn) {
              return [=] (auto accum, auto input) {
                  if (predicate(input)) {
                      return reduce_fn(accum, input);
                  } else {
                      return accum;
                  }
              };
          };
      }
```

1.  现在让我们最终使用这些辅助函数。我们实例化迭代器，让我们从标准输入中读取整数值：

```cpp
      int main()
      {
          std::istream_iterator<int> it {std::cin};
          std::istream_iterator<int> end_it;
```

1.  然后我们定义一个谓词函数`even`，如果我们有一个*偶数*，它就返回`true`。变换函数`twice`将它的整数参数乘以因子`2`：

```cpp
          auto even  ([](int i) { return i % 2 == 0; });
          auto twice ([](int i) { return i * 2; });
```

1.  `std::accumulate`函数接受一系列值并*累加*它们。累加意味着在默认情况下使用`+`运算符对值进行求和。我们想要提供我们自己的累加函数。这样，我们就不需要维护值的*总和*。我们做的是将范围的每个值赋给解引用的迭代器`it`，然后在*推进*它之后返回这个迭代器：

```cpp
          auto copy_and_advance ([](auto it, auto input) {
              *it = input;
              return ++it;
          });
```

1.  现在我们终于把这些部分整合在一起了。我们遍历标准输入并提供一个输出，`ostream_iterator`，它打印到终端。`copy_and_advance`函数对象通过将用户输入的整数赋值给它来处理输出迭代器。将值赋给输出迭代器有效地*打印*了被赋值的项目。但我们只想要用户输入中的*偶数*，并且我们想要*乘以*它们。为了实现这一点，我们将`copy_and_advance`函数包装到一个`even` *filter*中，然后再包装到一个`twice` *mapper*中：

```cpp
          std::accumulate(it, end_it,
              std::ostream_iterator<int>{std::cout, ", "},
              filter(even)(
                  map(twice)(
                      copy_and_advance
                  )
              ));
          std::cout << 'n';
      }
```

1.  编译和运行程序会产生以下输出。值`1`、`3`和`5`被丢弃，因为它们不是偶数，而值`2`、`4`和`6`在被加倍后被打印出来：

```cpp
      $ echo "1 2 3 4 5 6" | ./transform_if
      4, 8, 12, 
```

# 工作原理...

这个食谱看起来非常复杂，因为我们嵌套了很多 lambda 表达式。为了理解这是如何工作的，让我们首先来看一下`std::accumulate`的内部工作。这是在典型的 STL 实现中的样子：

```cpp
template <typename T, typename F>
T accumulate(InputIterator first, InputIterator last, T init, F f)
{
    for (; first != last; ++first) {
        init = f(init, *first);
    }
    return init;
}
```

这里，函数参数`f`承担了主要工作，而循环在用户提供的`init`变量中收集其结果。在通常的例子中，迭代器范围可能代表一个数字向量，比如`0, 1, 2, 3, 4`，而`init`值为`0`。`f`函数只是一个二元函数，可能使用`+`运算符计算两个项目的*总和*。

在这个例子中，循环只是将所有项目相加到`init`变量中，比如`init = (((0 + 1) + 2) + 3) + 4`。像这样写下来很明显，`std::accumulate`只是一个通用的*折叠*函数。折叠一个范围意味着对累加器变量应用二元操作，并逐步应用范围中包含的每个项目（每次操作的结果就是下一个累加器值）。由于这个函数是如此通用，我们可以做各种各样的事情，就像实现`std::transform_if`一样！`f`函数也被称为*reduce*函数。

`transform_if`的一个非常直接的实现如下所示：

```cpp
template <typename InputIterator, typename OutputIterator, 
          typename P, typename Transform>
OutputIterator transform_if(InputIterator first, InputIterator last,
                            OutputIterator out,
                            P predicate, Transform trans)
{
    for (; first != last; ++first) {
        if (predicate(*first)) {
            *out = trans(*first);
            ++out;
        }
    }
    return out;
}
```

这看起来与`std::accumulate`非常*相似*，如果我们将参数`out`视为`init`变量，并且*以某种方式*让函数`f`替代 if 结构及其主体！

我们实际上做到了。我们使用我们提供的二元函数对象构造了 if 结构及其主体，并将其作为参数提供给`std::accumulate`：

```cpp
auto copy_and_advance ([](auto it, auto input) {
    *it = input;
    return ++it;
});
```

`std::accumulate`函数将`init`变量放入二元函数的`it`参数中。第二个参数是源范围中每次循环迭代步骤的当前值。我们提供了一个*输出迭代器*作为`std::accumulate`的`init`参数。这样，`std::accumulate`不计算总和，而是将其迭代的项目转发到另一个范围。这意味着我们只是重新实现了`std::copy`，没有任何谓词和转换。

我们通过将`copy_and_advance`函数对象包装成*另一个*函数对象来添加使用谓词进行过滤：

```cpp
template <typename T>
auto filter(T predicate)
{
    return [=] (auto reduce_fn) {
        return [=] (auto accum, auto input) {
            if (predicate(input)) {
                return reduce_fn(accum, input);
            } else {
                return accum;
            }
        };
    };
}
```

这个构造一开始看起来并不简单，但是看看`if`结构。如果`predicate`函数返回`true`，它将参数转发给`reduce_fn`函数，这在我们的情况下是`copy_and_advance`。如果谓词返回`false`，则`accum`变量，即`std::accumulate`的`init`变量，将不经改变地返回。这实现了过滤操作的*跳过*部分。`if`结构位于内部 lambda 表达式中，其具有与`copy_and_advance`函数相同的二元函数签名，这使其成为一个合适的替代品。

现在我们能够*过滤*，但仍然没有*转换*。这是由`map`函数助手完成的：

```cpp
template <typename T>
auto map(T fn)
{
    return [=] (auto reduce_fn) {
        return [=] (auto accum, auto input) {
            return reduce_fn(accum, fn(input));
        };
    };
}
```

这段代码看起来简单得多。它再次包含了一个内部 lambda 表达式，其签名与`copy_and_advance`相同，因此可以替代它。实现只是转发输入值，但是*转换*了二元函数调用的*右*参数，使用`fn`函数。

稍后，当我们使用这些辅助函数时，我们写下了以下表达式：

```cpp
filter(even)(
    map(twice)(
        copy_and_advance
    )
)
```

`filter(even)`调用捕获了`even`谓词，并给了我们一个函数，它接受一个二元函数，以便将其包装成*另一个*二元函数，进行额外的*过滤*。`map(twice)`函数对`twice`转换函数做了同样的事情，但是将二元函数`copy_and_advance`包装成另一个二元函数，它总是*转换*右参数。

没有任何优化，我们将得到一个非常复杂的嵌套函数构造，调用函数并在其中间做很少的工作。然而，对于编译器来说，优化所有代码是一项非常简单的任务。生成的二进制代码就像是从`transform_if`的更直接的实现中得到的一样简单。这种方式在性能方面没有任何损失。但我们得到的是函数的非常好的可组合性，因为我们能够将`even`谓词与`twice`转换函数简单地组合在一起，几乎就像它们是*乐高积木*一样简单。

# 在编译时生成任何输入的笛卡尔积对

Lambda 表达式与参数包结合可以用于复杂的任务。在本节中，我们将实现一个函数对象，它接受任意数量的输入参数，并生成这组参数与*自身*的**笛卡尔积**。

笛卡尔积是一个数学运算。它表示为`A x B`，意思是集合`A`和集合`B`的笛卡尔积。结果是另一个*单一集合*，其中包含集合`A`和`B`的*所有*项目组合的对。该操作基本上意味着，*将 A 中的每个项目与 B 中的每个项目组合*。下图说明了该操作：

![](img/f83f3245-6b4c-4919-b137-17c2d6a11e7e.png)

在前面的图中，如果`A = (x, y, z)`，`B = (1, 2, 3)`，那么笛卡尔积是`(x, 1)`，`(x, 2)`，`(x, 3)`，`(y, 1)`，`(y, 2)`，等等。

如果我们决定`A`和`B`是*相同*的集合，比如`(1, 2)`，那么它的笛卡尔积是`(1, 1)`，`(1, 2)`，`(2, 1)`和`(2, 2)`。在某些情况下，这可能被声明为*冗余*，因为与*自身*的项目组合（如`(1, 1)`）或`(1, 2)`和`(2, 1)`的冗余组合可能是不需要的。在这种情况下，可以使用简单的规则过滤笛卡尔积。

在本节中，我们将实现笛卡尔积，但不使用任何循环，而是使用 lambda 表达式和参数包展开。

# 如何做...

我们实现了一个接受函数`f`和一组参数的函数对象。函数对象将*创建*参数集的笛卡尔积，*过滤*掉冗余部分，并*调用*`f`函数的每一个：

1.  我们只需要包括用于打印的 STL 头文件：

```cpp
      #include <iostream>
```

1.  然后，我们定义一个简单的辅助函数，用于打印一对值，并开始实现`main`函数：

```cpp
      static void print(int x, int y)
      {
          std::cout << "(" << x << ", " << y << ")n";
      }

      int main()
      {
```

1.  现在开始困难的部分。我们首先实现了`cartesian`函数的辅助函数，我们将在下一步中实现它。这个函数接受一个参数`f`，当我们以后使用它时，它将是`print`函数。其他参数是`x`和参数包`rest`。这些包含我们想要得到笛卡尔积的实际项目。看一下`f(x, rest)`表达式：对于`x=1`和`rest=2, 3, 4`，这将导致诸如`f(1, 2); f(1, 3); f(1, 4);`的调用。`(x < rest)`测试是为了消除生成的对中的冗余。我们稍后将更详细地看一下这一点：

```cpp
          constexpr auto call_cart (
              = constexpr {
                  (void)std::initializer_list<int>{
                      (((x < rest)
                          ? (void)f(x, rest)
                          : (void)0)
                      ,0)...
                  };
              });
```

1.  `cartesian`函数是整个配方中最复杂的代码。它接受参数包`xs`并返回一个捕获它的函数对象。返回的函数对象接受一个函数对象`f`。

对于参数包，`xs=1, 2, 3`，内部 lambda 表达式将生成以下调用：`call_cart(f, **1**, 1, 2, 3); call_cart(f, **2**, 1, 2, 3); call_cart(f, **3**, 1, 2, 3);`。从这一系列调用中，我们可以生成所有需要的笛卡尔积对。

请注意，我们使用`...`符号来*两次*展开`xs`参数包，一开始看起来很奇怪。第一次出现的`...`将整个`xs`参数包展开为`call_cart`调用。第二次出现会导致多个`call_cart`调用，第二个参数不同：

```cpp
          constexpr auto cartesian (= constexpr {
              return [=] (auto f) constexpr {
                  (void)std::initializer_list<int>{
                      ((void)call_cart(f, xs, xs...), 0)...
                  };
              };
          });
```

1.  现在，让我们生成数字集合`1, 2, 3`的笛卡尔积并打印这些配对。去除冗余配对后，这应该得到数字配对`(1, 2)`，`(2, 3)`和`(1, 3)`。如果我们忽略顺序并且不希望在一个配对中有相同的数字，那么就不可能有更多的组合。这意味着我们*不*希望`(1, 1)`，并且认为`(1, 2)`和`(2, 1)`是*相同*的配对。

首先，我们让`cartesian`生成一个函数对象，该对象已经包含了所有可能的配对，并接受我们的打印函数。然后，我们使用它来让我们的`print`函数被所有这些配对调用。

我们声明`print_cart`变量为`constexpr`，这样我们可以保证它所持有的函数对象（以及它生成的所有配对）在编译时创建：

```cpp
          constexpr auto print_cart (cartesian(1, 2, 3));

          print_cart(print);
      }
```

1.  编译和运行产生了以下输出，正如预期的那样。通过删除`call_cart`函数中的`(x < xs)`条件，可以尝试在代码中进行调整，看看我们是否会得到包含冗余配对和相同数字配对的完整笛卡尔积。

```cpp
      $ ./cartesian_product
      (1, 2)
      (1, 3)
      (2, 3)
```

# 它是如何工作的...

这是另一个看起来非常复杂的 lambda 表达式构造。但一旦我们彻底理解了这一点，我们就不会被任何 lambda 表达式所困惑！

因此，让我们仔细看一下。我们应该对需要发生的事情有一个清晰的认识：

![](img/957b7794-331d-4b2d-958e-ac82ee95071d.png)

这些是三个步骤：

1.  我们取我们的集合`1, 2, 3`，并从中组合*三个新*集合。每个集合的第一部分依次是集合中的一个单独项，第二部分是整个集合本身。

1.  我们将第一个项与集合中的每个项组合，得到尽可能多的*配对*。

1.  从这些得到的配对中，我们只挑选那些*不冗余*的（例如`(1, 2)`和`(2, 1)`是冗余的）和不相同编号的（例如`(1, 1)`）。

现在，回到实现：

```cpp
 constexpr auto cartesian (= constexpr {
     return = constexpr {
         (void)std::initializer_list<int>{
             ((void)call_cart(f, xs, xs...), 0)...
         };
     };
 });
```

内部表达式`call_cart(xs, xs...)`恰好表示将`(1, 2, 3)`分成这些新集合，比如`1, [1, 2, 3]`。完整表达式`((void)call_cart(f, xs, xs...), 0)...`与其他`...`在外面，对集合的每个值进行了这种分割，所以我们也得到了`2, [1, 2, 3]`和`3, [1, 2, 3]`。

第 2 步和第 3 步是由`call_cart`完成的：

```cpp
auto call_cart ([](auto f, auto x, auto ...rest) constexpr {
    (void)std::initializer_list<int>{
        (((x < rest)
            ? (void)f(x, rest)
            : (void)0)
        ,0)...
    };
});
```

参数`x`始终包含从集合中选取的单个值，`rest`包含整个集合。首先忽略`(x < rest)`条件。在这里，表达式`f(x, rest)`与`...`参数包展开一起生成函数调用`f(1, 1)`，`f(1, 2)`等等，这导致配对被打印。这是第 2 步。

第 3 步是通过筛选出只有`(x < rest)`适用的配对来实现的。

我们将所有的 lambda 表达式和持有它们的变量都设为`constexpr`。通过这样做，我们现在可以保证编译器将在编译时评估它们的代码，并编译出一个已经包含所有数字配对的二进制文件，而不是在运行时计算它们。请注意，*只有*当我们提供给 constexpr 函数的所有函数参数*在编译时已知*时才会发生这种情况。
