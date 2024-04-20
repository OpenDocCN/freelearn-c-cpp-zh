# 第二十三章：迭代器

本章中涵盖以下内容：

+   构建自己的可迭代范围

+   使自己的迭代器与 STL 迭代器类别兼容

+   使用迭代器包装器填充通用数据结构

+   按迭代器实现算法

+   使用反向迭代器适配器进行反向迭代

+   使用迭代器哨兵终止范围上的迭代

+   使用检查迭代器自动检查迭代器代码

+   构建自己的 zip 迭代器适配器

# 介绍

迭代器在 C++ 中是一个非常重要的概念。STL 的目标是尽可能灵活和通用，而迭代器在这方面非常有帮助。不幸的是，它们有时候使用起来有点乏味，这就是为什么许多新手会避免使用它们并退回到 *C-Style C++* 的原因。一个避免使用迭代器的程序员基本上放弃了 STL 的一半潜力。本章涉及迭代器并快速地介绍了它们的工作原理。这种非常快速的介绍可能不够，但 *配方* 真的是为了让人对迭代器内部有一个良好的感觉。

大多数容器类，但也包括老式的 C 风格数组，以某种方式都包含一组数据项的 *范围*。许多日常任务处理大量数据项时并不关心如何获取这些数据。然而，如果我们考虑，例如，一个整数数组和一个整数 *链表* 并想要计算这两种结构中所有项目的 *总和*，我们最终会得到两种不同的算法，可能看起来像下面这样：

+   一个处理数组并检查其大小并将其求和的算法如下：

```cpp
      int sum {0};
      for (size_t i {0}; i < array_size; ++i) { sum += array[i]; }
```

+   另一个算法，通过迭代链表直到达到其末尾：

```cpp
      int sum {0};
      while (list_node != nullptr) { 
          sum += list_node->value; list_node = list_node->next; 
      }
```

它们都是关于 *整数求和*，但我们输入的字符中有多少是直接与 *实际* 求和任务相关的？其中一个是否适用于第三种数据结构，比如 `std::map`，还是我们必须实现另一个版本？没有迭代器，这将使我们陷入荒谬的方向。

只有通过迭代器的帮助，才能以通用形式实现这一点：

```cpp
int sum {0};
for (int i : array_or_vector_or_map_or_list) { sum += i; }
```

这种漂亮而简短的所谓基于范围的 `for` 循环自 C++11 以来就存在了。它只是一种语法糖，类似于以下代码：

```cpp
{ 
    auto && __range = array_or_vector_or_map_or_list ; 
    auto __begin = std::begin(__range);
    auto __end   = std::end(__range);
    for ( ; __begin != __end; ++__begin) { 
        int i = *__begin; 
        sum += i;
    } 
}
```

对于那些已经使用过迭代器的人来说，这已经是老生常谈了，但对于那些没有使用过的人来说，这看起来完全像是魔术。想象一下我们的整数向量看起来像下面这样：

![](img/5fd26991-6353-4490-b01b-959c754fe5b5.png)

`std::begin(vector)` 命令与 `vector.begin()` 相同，返回一个指向第一个项目（**1**）的迭代器。`std::end(vector)` 与 `vector.end()` 相同，返回一个指向最后一个项目之后一个项目的迭代器（**5**之后）。

在每次迭代中，循环都会检查起始迭代器是否不等于结束迭代器。如果是这样，它将 *解引用* 起始迭代器，从而访问它指向的整数值。然后，它 *递增* 迭代器，重复与结束迭代器的比较，依此类推。在这一刻，帮助阅读循环代码时想象迭代器就是普通的 *C* 风格指针。事实上，普通的 C 风格指针也是一种有效的迭代器。

# 迭代器类别

迭代器有多个类别，它们有不同的限制。它们并不难记忆，只需记住一个类别所需的功能是从下一个更强大的类别继承的。迭代器类别的整个重点在于，如果算法知道它正在处理哪种类型的迭代器，它可以以一种优化的方式实现。这样，程序员可以放松并表达自己的意图，而编译器可以选择给定任务的 *最佳实现*。

让我们按正确的顺序来看一下：

![](img/d9d1c3e9-25b2-45d2-9630-3759bef7cb1d.png)

# 输入迭代器

输入迭代器只能被解引用来*读取*它们指向的值。一旦它们被增加，它们指向的最后一个值在增加过程中被*失效*。这意味着不可能多次迭代这样的范围。`std::istream_iterator`就是这一类的例子。

# 前向迭代器

前向迭代器与输入迭代器相同，但它们的区别在于它们表示的范围可以被多次迭代。`std::forward_list`的迭代器就是一个例子。这样的列表只能*向前*迭代，不能向后，但可以随意多次迭代。

# 双向迭代器

双向迭代器，顾名思义，可以被增加和减少，以便向前或向后迭代。例如，`std::list`、`std::set`和`std::map`的迭代器支持这一点。

# 随机访问迭代器

随机访问迭代器允许一次跳过多个值，而不是逐个步进。这适用于`std::vector`和`std::deque`的迭代器。

# 连续迭代器

这个类别指定了前面提到的所有要求，还要求被迭代的数据位于连续的内存中，就像在数组或`std::vector`中一样。

# 输出迭代器

输出迭代器与其他类别无关。这是因为迭代器可以是纯输出迭代器，只能被增加并用于*写入*它指向的数据。如果它们被读取，值将是未定义的。

# 可变迭代器

如果一个迭代器既是输出迭代器又是其他类别之一，它就是可变迭代器。它可以被读取和写入。如果我们从一个非 const 容器实例中获取迭代器，它通常会是这种类型。

# 构建自己的可迭代范围

我们已经意识到迭代器在各种容器上进行迭代时，有点像*标准接口*。我们只需要实现前缀增量运算符`++`、解引用运算符`*`和对象比较运算符`==`，然后我们就已经有了一个原始迭代器，可以适应时髦的 C++11 基于范围的`for`循环。

为了更好地适应这一点，这个示例展示了如何实现一个迭代器，当通过迭代时只发出一系列数字。它不依赖于任何容器结构或类似的东西。这些数字是在迭代时临时生成的。

# 如何做...

在这个示例中，我们将实现自己的迭代器类，然后通过它进行迭代：

1.  首先，我们包含头文件，这样我们就可以打印到终端：

```cpp
      #include <iostream>
```

1.  我们的迭代器类将被称为`num_iterator`：

```cpp
      class num_iterator {
```

1.  它唯一的数据成员是一个整数。该整数用于计数。构造函数用于初始化它。通常最好将构造函数设为*显式*，这样可以避免*意外*的隐式转换。请注意，我们还为`position`提供了默认值。这使得`num_iterator`类的实例可以默认构造。尽管在整个示例中我们不会使用默认构造函数，但这真的很重要，因为一些 STL 算法依赖于迭代器是默认可构造的：

```cpp
          int i;
      public:

          explicit num_iterator(int position = 0) : i{position} {}
```

1.  当解引用我们的迭代器(`*it`)时，它将发出一个整数：

```cpp
          int operator*() const { return i; }
```

1.  增加迭代器(`++it`)只会增加它的内部计数器`i`：

```cpp
          num_iterator& operator++() {
              ++i;
              return *this;
          }
```

1.  `for`循环将迭代器与结束迭代器进行比较。如果它们*不相等*，它将继续迭代：

```cpp
          bool operator!=(const num_iterator &other) const {
              return i != other.i;
          }
      };
```

1.  这就是迭代器类。我们仍然需要一个中间对象来编写`for (int i : intermediate(a, b)) {...}`，然后包含开始和结束迭代器，它被预设为从`a`到`b`进行迭代。我们称之为`num_range`：

```cpp
      class num_range {
```

1.  它包含两个整数成员，表示迭代应该从哪个数字开始，以及第一个数字过去的数字是多少。这意味着如果我们想要从`0`到`9`进行迭代，`a`设置为`0`，`b`设置为`10`：

```cpp
          int a;
          int b;

      public:
          num_range(int from, int to)
              : a{from}, b{to}
          {}
```

1.  我们只需要实现两个成员函数：`begin`和`end`函数。两者都返回指向数字范围开始和结束的迭代器：

```cpp
          num_iterator begin() const { return num_iterator{a}; }
          num_iterator end()   const { return num_iterator{b}; }
      };
```

1.  就是这样。我们可以使用它。让我们编写一个主函数，它只是迭代从`100`到`109`的范围，并打印出所有的值：

```cpp
      int main()
      {
          for (int i : num_range{100, 110}) {
              std::cout << i << ", ";
          }
          std::cout << 'n';
      }
```

1.  编译和运行程序产生以下终端输出：

```cpp
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
```

# 工作原理... 

考虑我们编写以下代码：

```cpp
for (auto x : range) { code_block; }
```

编译器将对其求值为以下内容：

```cpp
{ 
    auto __begin = std::begin(range);
    auto __end   = std::end(range);
    for ( ; __begin != __end; ++__begin) { 
        auto x = *__begin; 
        code_block
    } 
}
```

在查看这段代码时，很明显迭代器的唯一要求是以下三个运算符：

+   `operator!=`：不相等比较

+   `operator++`：前缀递增

+   `operator*`：解引用

范围的要求是它有一个`begin`和一个`end`方法，返回两个迭代器，表示范围的开始和结束。

在本书中，我们大多数时候使用`std::begin(x)`而不是`x.begin()`。这通常是一个很好的风格，因为`std::begin(x)`会自动调用`x.begin()`，如果该成员方法可用。如果`x`是一个没有`begin()`方法的数组，`std::begin(x)`会自动找出如何处理它。对`std::end(x)`也是一样。不提供`begin()`/`end()`成员的用户定义类型无法使用`std::begin`/`std::end`。

在这个食谱中，我们所做的只是将一个简单的数字计数算法适应到前向迭代器接口中。实现迭代器和范围总是涉及到这最少量的样板代码，这在一方面可能有点烦人。另一方面，查看使用`num_range`的循环是非常有益的，因为它看起来如此*完美简单*！

回头仔细看看迭代器和范围类的方法中哪些是`const`。忘记使这些函数`const`可能会使编译器在许多情况下*拒绝*您的代码，因为迭代`const`对象是一件很常见的事情。

# 使自己的迭代器与 STL 迭代器类别兼容

无论我们想出什么自己的容器数据结构，为了有效地*混合*它与所有 STL 的优点，我们需要使它们提供迭代器接口。在上一节中，我们学会了如何做到这一点，但我们很快意识到*一些*STL 算法*无法*与我们的自定义迭代器很好地编译。为什么？

问题在于很多 STL 算法试图找出更多关于它们被我们要求处理的迭代器的信息。不同的迭代器*类别*具有不同的功能，因此，可能有不同的可能性来实现*相同*的算法。例如，如果我们从一个`std::vector`复制*普通数字*到另一个，这可能是通过快速的`memcpy`调用来实现的。如果我们从`std::list`复制数据，这就不再可能了，项目必须一个接一个地逐个复制。STL 算法的实现者对这种自动优化进行了大量思考。为了帮助他们，我们可以为我们的迭代器提供一些关于它们的*信息*。本节展示了如何实现相同的功能。

# 如何做...

在本节中，我们将实现一个原始迭代器，计数数字并将其与最初无法与之一起编译的 STL 算法一起使用。然后我们做必要的工作使其与 STL 兼容。

1.  首先，我们需要像往常一样包含一些头文件：

```cpp
      #include <iostream>
      #include <algorithm>
```

1.  然后我们实现一个原始的数字计数迭代器，就像前一节一样。在对其进行迭代时，它将发出普通递增的整数。`num_range`充当一个方便的*begin*和*end*迭代器捐赠者：

```cpp
      class num_iterator 
      {
          int i;
      public:

          explicit num_iterator(int position = 0) : i{position} {}

          int operator*() const { return i; }

          num_iterator& operator++() {
              ++i;
              return *this;
          }

          bool operator!=(const num_iterator &other) const {
              return i != other.i;
          }

          bool operator==(const num_iterator &other) const {
              return !(*this != other); 
          }
      };

      class num_range {
          int a;
          int b;

      public:
          num_range(int from, int to)
              : a{from}, b{to}
          {}

          num_iterator begin() const { return num_iterator{a}; }
          num_iterator end()   const { return num_iterator{b}; }
      };
```

1.  为了使`std::`命名空间前缀保持在外部并保持代码可读性，我们声明使用`std`命名空间：

```cpp
      using namespace std;
```

1.  现在让我们实例化一个范围，从`100`到`109`。请注意，值`110`是结束迭代器的位置。这意味着`110`是范围之外的*第一个*数字（这就是为什么它从`100`到`109`）：

```cpp
      int main()
      {
          num_range r {100, 110};
```

1.  现在，我们使用`std::minmax_element`。这个算法返回一个`std::pair`，其中包含两个成员：指向范围中最低值的迭代器和指向最高值的迭代器。当然，这些值是`100`和`109`，因为这是我们构造范围的方式：

```cpp
          auto [min_it, max_it] (minmax_element(begin(r), end(r)));
          cout << *min_it << " - " << *max_it << 'n';
      }
```

1.  编译代码会导致以下错误消息。这是与`std::iterator_traits`相关的错误。稍后会详细介绍。*可能*会发生其他编译器和/或 STL 库实现的错误，或者*根本没有*错误。这个错误消息出现在 clang 版本 5.0.0（trunk 299766）中：

![](img/52d1b385-6f1a-4731-97a4-4389d0e9047b.png)

1.  为了解决这个问题，我们需要为我们的迭代器类激活迭代器特性功能。在`num_iterator`的定义之后，我们编写了`std::iterator_traits`类型的以下模板结构专门化。它告诉 STL 我们的`num_iterator`是前向迭代器类别，并且它迭代`int`值：

```cpp
      namespace std {
        struct iterator_traits<num_iterator> {

          using iterator_category = std::forward_iterator_tag;

          using value_type = int;

          using difference_type = void;

          using pointer = int*;

          using reference = int&;

        };
      }
```

1.  让我们再次编译它；我们可以看到它工作了！min/max 函数的输出如下，这正是我们期望的：

```cpp
      100 - 109
```

# 它是如何工作的...

一些 STL 算法需要了解它们所使用的迭代器类型的特性。其他一些需要知道迭代器迭代的项目类型。这有不同的实现原因。

然而，所有 STL 算法将通过`std::iterator_traits<my_iterator>`访问此类型信息，假设迭代器类型为`my_iterator`。这个特性类包含多达五种不同的类型成员定义：

+   `difference_type`：写`it1 - it2`的结果是什么类型？

+   `value_type`：我们使用`*it`访问的项目是什么类型（请注意，对于纯输出迭代器，这是`void`）？

+   `pointer`：为了指向一个项目，指针必须是什么类型？

+   `reference`：为了引用一个项目，引用必须是什么类型？

+   `iterator_category`：迭代器属于哪个类别？

`pointer`、`reference`和`difference_type`类型定义对于我们的`num_iterator`来说是没有意义的，因为它不迭代真正的*内存*值（我们只是*返回*`int`值，但它们不像数组中那样持久可用）。因此最好不定义它们，因为如果算法依赖于这些项目在内存中可引用，当与我们的迭代器结合时可能会出现*错误*。

# 还有更多...

直到 C++17，鼓励让迭代器类型直接继承自`std::iterator<...>`，这会自动填充我们的类所有类型定义。这仍然有效，但自 C++17 以来已不再鼓励。

# 使用迭代器适配器填充通用数据结构

在许多情况下，我们希望用大量数据填充任何容器，但数据源和容器没有*共同的接口*。在这种情况下，我们需要编写自己的手工制作的算法，只是处理如何将数据从源推送到接收端的问题。通常，这会让我们分心，无法专注于解决特定*问题*的实际工作。

我们可以用一行代码实现在概念上不同的数据结构之间传输数据的任务，这要归功于 STL 提供的另一个抽象：**迭代器适配器**。本节演示了如何使用其中一些迭代器适配器，以便让人感受到它们有多么有用。

# 如何做...

在本节中，我们使用一些迭代器包装器，只是为了展示它们的存在以及它们如何帮助我们在日常编程任务中。

1.  我们需要首先包含一些头文件：

```cpp
      #include <iostream>
      #include <string>
      #include <iterator>
      #include <sstream>
      #include <deque>
```

1.  声明我们使用命名空间`std`可以减少我们以后的输入：

```cpp
      using namespace std;
```

1.  我们从`std::istream_iterator`开始。我们将其专门化为`int`。这样，它将尝试将标准输入解析为整数。例如，如果我们对其进行迭代，它看起来就像是`std::vector<int>`。结束迭代器也是用相同类型实例化的，但没有任何构造参数：

```cpp
      int main()
      {
          istream_iterator<int> it_cin {cin};
          istream_iterator<int> end_cin;
```

1.  接下来，我们实例化`std::deque<int>`，并将所有整数从标准输入复制到 deque 中。deque 本身不是一个迭代器，所以我们使用`std::back_inserter`辅助函数将其包装成`std::back_insert_iterator`。这个特殊的迭代器包装器将对我们从标准输入获取的每个项目执行`v.push_back(item)`。这样，deque 会自动增长！

```cpp
          deque<int> v;

          copy(it_cin, end_cin, back_inserter(v));
```

1.  在下一个练习中，我们使用`std::istringstream`将项目复制到 deque 的*中间*。因此，让我们首先定义一些示例数字，以字符串的形式实例化流对象：

```cpp
          istringstream sstr {"123 456 789"};
```

1.  然后，我们需要一个提示，告诉我们在 deque 中插入的位置。这将是中间，所以我们使用 deque 的 begin 指针并将其传递给`std::next`函数。这个函数的第二个参数表示它将返回一个迭代器，向前移动了`v.size() / 2`步，也就是 deque 的*一半*。（我们将`v.size()`强制转换为`int`，因为`std::next`的第二个参数是作为第一个参数使用的迭代器的`difference_type`。在这种情况下，这是一个有符号整数类型。根据编译器标志，如果我们没有显式转换，编译器可能会在这一点*警告*。）

```cpp
          auto deque_middle (next(begin(v), 
                                  static_cast<int>(v.size()) / 2));
```

1.  现在，我们可以逐步将解析的整数从输入字符串流复制到 deque 中。再次强调，流迭代器包装器的结束迭代器只是一个没有构造参数的空的`std::istream_iterator<int>`（即代码行中的空`{}`括号）。deque 被包装成插入器包装器，它是一个`std::insert_iterator`，使用`deque_middle`迭代器指向 deque 的中间：

```cpp
          copy(istream_iterator<int>{sstr}, {}, inserter(v, deque_middle));
```

1.  现在，让我们使用`std::front_insert_iterator`在 deque 的前面插入一些项目：

```cpp
          initializer_list<int> il2 {-1, -2, -3};
          copy(begin(il2), end(il2), front_inserter(v));
```

1.  在最后一步，我们将整个 deque 的内容打印到用户 shell 上。`std::ostream_iterator`的工作原理类似于输出迭代器，在我们的例子中，它只是将所有从中复制的整数转发到`std::cout`，然后在每个项目后附加`", "`：

```cpp
          copy(begin(v), end(v), ostream_iterator<int>{cout, ", "});
          cout << 'n';
      }
```

1.  编译并运行程序会产生以下输出。你能辨别出哪个数字是由哪行代码插入的吗？

```cpp
      $ echo "1 2 3 4 5" | ./main
      -3, -2, -1, 1, 2, 123, 456, 789, 3, 4, 5,
```

# 它的工作原理...

在本节中，我们使用了许多不同的迭代器适配器。它们都有一个共同点，就是它们将一个对象包装成一个不是迭代器本身的迭代器。

# std::back_insert_iterator

`back_insert_iterator`可以包装`std::vector`、`std::deque`、`std::list`等。它将调用容器的`push_back`方法，将新项目*插入*到现有项目之后。如果容器实例不够大，它将自动增长。

# std::front_insert_iterator

`front_insert_iterator`和`back_insert_iterator`完全做相同的事情，但它调用容器的`push_front`方法，这会在所有现有项目*之前*插入新项目。请注意，对于像`std::vector`这样的容器，这意味着所有现有项目都需要向前移动一个位置，以便为前面的新项目留出空间。

# std::insert_iterator

这个迭代器适配器类似于其他插入器，但能够在现有项目*之间*插入新项目。构造这样一个包装器的`std::inserter`辅助函数需要两个参数。第一个参数是容器，第二个参数是指向新项目应该插入的位置的迭代器。

# std::istream_iterator

`istream_iterator`是另一个非常方便的适配器。它可以与任何`std::istream`对象一起使用（例如标准输入或文件），并将尝试根据实例化时的模板参数从该流对象中解析输入。在本节中，我们使用了`std::istream_iterator<int>(std::cin)`，它从标准输入中提取整数。

流的特殊之处在于我们通常无法预先知道流的长度。这就引出了一个问题，如果我们不知道流的结束在哪里，*结束*迭代器将指向哪里？它的工作方式是，迭代器*知道*当它到达流的末尾时。当它与结束迭代器进行比较时，它实际上*不会真正*与结束迭代器进行比较，而是返回流是否还有标记*剩余*。这就是为什么结束迭代器的构造函数不接受任何参数。

# std::ostream_iterator

`ostream_iterator`与`istream_iterator`相同，但工作方式相反：它不从*输入*流中获取标记，而是将标记推送到*输出*流中。与`istream_iterator`的另一个不同之处在于，它的构造函数接受第二个参数，该参数是一个字符串，应在每个项目后推送到输出流中。这很有用，因为这样我们可以在每个项目后打印一个分隔符`", "`或一个新行。

# 以迭代器实现算法

迭代器通常通过*移动*它们的*位置*从容器的一个项目迭代到另一个项目。但它们不一定需要在数据结构上进行迭代。迭代器也可以用于实现算法，在这种情况下，它们在递增（`++it`）时计算下一个值，并在解引用（`*it`）时返回该值。

在本节中，我们通过实现迭代器形式的斐波那契函数来演示这一点。斐波那契函数的递归定义如下：`F(n) = F(n - 1) + F(n - 2)`。它从`F(0) = 0`和`F(1) = 1`的初始值开始。这导致以下数字序列：

+   `F(0) = 0`

+   `F(1) = 1`

+   `F(2) = F(1) + F(0) = 1`

+   `F(3) = F(2) + F(1) = 2`

+   `F(4) = F(3) + F(2) = 3`

+   `F(5) = F(4) + F(3) = 5`

+   `F(6) = F(5) + F(4) = 8`

+   ...等等

如果我们以可调用函数的形式实现这一点，该函数将返回任何数字*n*的斐波那契值，我们最终将得到一个递归自调用函数，或者一个循环实现。这没问题，但是如果我们编写一些程序，需要按某种模式消耗斐波那契数，一个接一个地，我们将有两种可能性——要么我们为每个新的斐波那契数重新计算所有递归调用，这是一种浪费计算时间的做法，要么我们保存最后两个斐波那契数作为临时变量，并使用它们来计算下一个。在后一种情况下，我们重新实现了斐波那契算法的循环实现。看起来我们最终会*混合*斐波那契代码和解决不同问题的实际代码：

```cpp
size_t a {0};
size_t b {1};

for (size_t i {0}; i < N; ++i) {
    const size_t old_b {b};
    b += a;
    a  = old_b;

    // do something with b, which is the current fibonacci number
}
```

迭代器是解决这个问题的一个有趣方法。我们可以将基于循环的迭代式斐波那契实现中的步骤包装在斐波那契值*迭代器*的前缀递增`++`运算符实现中。正如本节所示，这是非常容易的。

# 如何做...

在本节中，我们专注于实现一个在迭代过程中生成斐波那契数列数字的迭代器。

1.  为了能够将斐波那契数打印到终端，我们首先需要包含一个头文件：

```cpp
      #include <iostream>
```

1.  我们称斐波那契迭代器为`fibit`。它将携带一个成员`i`，用于保存斐波那契序列中的索引位置，`a`和`b`将是保存最后两个斐波那契值的变量。如果使用默认构造函数实例化，斐波那契迭代器将初始化为值`F(0)`：

```cpp
      class fibit
      {
          size_t i {0};
          size_t a {0};
          size_t b {1};
```

1.  接下来，我们定义标准构造函数和另一个构造函数，它允许我们在任何斐波那契数步骤上初始化迭代器：

```cpp
      public:
          fibit() = default;

          explicit fibit(size_t i_)
              : i{i_}
          {}
```

1.  当解引用我们的迭代器（`*it`）时，它将只输出当前步骤的斐波那契数：

```cpp
          size_t operator*() const { return b; }
```

1.  在增加迭代器（++it）时，它将将其状态移动到下一个斐波那契数。这个函数包含与基于循环的斐波那契实现相同的代码：

```cpp
          fibit& operator++() {
              const size_t old_b {b};
              b += a;
              a = old_b;
              ++i;
              return *this;
          }
```

1.  在循环中使用时，增加的迭代器与结束迭代器进行比较，因此我们需要定义`!=`运算符。我们只比较斐波那契迭代器当前所在的*步骤*，这样可以更容易地为步骤`1000000`定义结束迭代器，例如，因为我们不需要提前昂贵地计算这么高的斐波那契数：

```cpp
          bool operator!=(const fibit &o) const { return i != o.i; }
      };
```

1.  为了能够在基于范围的`for`循环中使用斐波那契迭代器，我们必须事先实现一个范围类。我们称之为`fib_range`，它的构造函数将接受一个参数，告诉我们要在斐波那契范围内迭代多远：

```cpp
      class fib_range
      {
          size_t end_n;

      public:
          fib_range(size_t end_n_)
              : end_n{end_n_}
          {}
```

1.  它的`begin`和`end`函数返回指向位置`F(0)`和`F(end_n)`的迭代器：

```cpp
          fibit begin() const { return fibit{}; }
          fibit end()   const { return fibit{end_n}; }
      };
```

1.  好了，现在让我们忘记所有与迭代器相关的样板代码。因为我们现在有了一个辅助类，可以很好地隐藏所有的实现细节！让我们打印前 10 个斐波那契数：

```cpp
      int main()
      {
          for (size_t i : fib_range(10)) {
              std::cout << i << ", ";
          }
          std::cout << 'n';
      }
```

1.  编译和运行程序会产生以下 shell 输出：

```cpp
      1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
```

# 还有更多...

为了能够在 STL 中使用这个迭代器，它必须支持`std::iterator_traits`类。要了解如何做到这一点，请看*其他*的食谱，它处理了这个问题：*使您自己的迭代器与 STL 迭代器类别兼容*。

试着以迭代器的方式思考。这在许多情况下会导致非常优雅的代码。不要担心性能：编译器发现优化掉与迭代器相关的样板代码是*微不足道*的！

为了保持示例简单，我们没有做任何处理，但如果我们将斐波那契迭代器发布为库，就会发现它存在一个可用性缺陷--使用构造函数参数创建的`fibit`实例只能用作结束迭代器，因为它不包含有效的斐波那契值。我们的小型库不强制这种用法。有不同的可能性来解决这个问题：

+   将`fibit(size_t i_)`构造函数设置为私有，并将`fib_range`类声明为`fibit`类的友元。这样，用户只能以正确的方式使用它。

+   使用迭代器哨兵来防止用户解引用结束迭代器。看看我们介绍的那个食谱：*使用迭代器哨兵终止范围上的迭代*。

# 使用反向迭代器适配器进行反向迭代

有时，逆向迭代一个范围是有价值的，不是向前，而是*向后*。基于范围的`for`循环，以及所有 STL 算法通常通过*递增*迭代器来迭代给定的范围，尽管向后迭代需要*递减*它们。当然，可以将迭代器*包装*成一个层，将*递增*调用有效地转换为*递减*调用。这听起来像是为我们想要支持的每种类型编写大量样板代码。

STL 提供了一个有用的*反向迭代器适配器*，可以帮助我们设置这样的迭代器。

# 如何做...

在这一部分，我们将以不同的方式使用反向迭代器，只是为了展示它们的用法：

1.  我们首先需要包含一些头文件，就像往常一样：

```cpp
      #include <iostream>
      #include <list>
      #include <iterator>
```

1.  接下来，我们声明我们使用`std`命名空间，以节省一些输入。

```cpp
      using namespace std;
```

1.  为了有一些可以迭代的东西，让我们实例化一个整数列表：

```cpp
      int main()
      {
          list<int> l {1, 2, 3, 4, 5};
```

1.  现在让我们以相反的形式打印这些整数。为了做到这一点，我们使用`std::list`的`rbegin`和`rend`函数来遍历列表，并通过标准输出使用方便的`ostream_iterator`适配器将这些值输出：

```cpp
          copy(l.rbegin(), l.rend(), ostream_iterator<int>{cout, ", "});
          cout << 'n';
```

1.  如果一个容器没有提供方便的`rbegin`和`rend`函数，但至少提供了双向迭代器，`std::make_reverse_iterator`函数会提供帮助。它接受*普通*迭代器并将它们转换为*反向*迭代器：

```cpp
          copy(make_reverse_iterator(end(l)),
               make_reverse_iterator(begin(l)),
               ostream_iterator<int>{cout, ", "});
          cout << 'n';
      }
```

1.  编译和运行我们的程序会产生以下输出：

```cpp
      5, 4, 3, 2, 1, 
      5, 4, 3, 2, 1,
```

# 它是如何工作的...

为了能够将普通迭代器转换为反向迭代器，它必须至少支持双向迭代。这个要求由*双向*类别或更高级别的任何迭代器都可以满足。

反向迭代器有点像包含一个普通迭代器并完全模拟其接口，但它将增量操作重定向为减量操作。

下一个细节是关于开始和结束迭代器位置。让我们看一下下面的图表，它显示了一个可迭代范围中保留的标准数字序列。如果序列从`1`到`5`，那么开始迭代器必须指向元素`1`，结束迭代器必须指向`5`之后的一个元素：

![](img/5d43dc84-63ca-4492-ad21-1e3278268727.png)

在定义反向迭代器时，`rbegin`迭代器必须指向`5`，`rend`迭代器必须指向`1`*之前*的元素。把书倒过来看，就会完全合理。

如果我们希望我们自己的自定义容器类支持反向迭代，我们不需要自己实现所有这些细节；我们可以使用`std::make_reverse_iterator`辅助函数将普通迭代器包装成反向迭代器，它会为我们执行所有的操作重定向和偏移校正。

# 使用迭代器标记终止范围的迭代

STL 算法和基于范围的`for`循环都假设迭代的开始和结束位置是*预先*已知的。然而，在某些情况下，很难在迭代*到达*之前知道结束位置。

一个非常简单的例子是迭代普通的 C 风格字符串，其长度在*运行时*之前是未知的。通常迭代这样的字符串的代码看起来像这样：

```cpp
for (const char *c_ponter = some_c_string; *c_pointer != ''; ++c_pointer) {
    const char c = *c_pointer;
    // do something with c
}
```

将其放入基于范围的`for`循环的唯一方法似乎是将其包装成一个`std::string`，它有`begin()`和`end()`函数：

```cpp
for (char c : std::string(some_c_string)) { /* do something with c */ }
```

然而，`std::string`的构造函数将在我们的`for`循环可以迭代它之前迭代整个字符串。自 C++17 以来，我们也有`std::string_view`，但它的构造函数也会遍历字符串一次。对于*短*字符串来说，这不值得真正的麻烦，但这也只是一个在*其他情况*中可能值得麻烦的问题*类*的例子。当`std::istream_iterator`从`std::cin`中捕获输入时，它也必须处理这个问题，因为它的结束迭代器在用户*仍在输入*键时实际上不能指向用户输入的结尾。

C++17 带来了一个伟大的消息，即它不限制开始和结束迭代器必须是相同类型。本节演示了如何将这个*小规则变更*发挥到*极大作用*。

# 如何做...

在本节中，我们将一起构建一个迭代器和一个范围类，它使我们能够在不提前找到*结束*位置的情况下迭代一个未知长度的字符串。

1.  首先，像往常一样，我们需要包含头文件：

```cpp
      #include <iostream>
```

1.  迭代器标记是本节的一个非常核心的元素。令人惊讶的是，它的类定义可以完全为空：

```cpp
      class cstring_iterator_sentinel {};
```

1.  现在我们实现迭代器。它将包含一个字符串指针，这是我们要迭代的*容器*：

```cpp
      class cstring_iterator {
          const char *s {nullptr};
```

1.  构造函数只是将内部字符串指针初始化为用户提供的任何字符串。让我们将构造函数声明为显式的，以防止从字符串到字符串迭代器的意外隐式转换：

```cpp
      public:
          explicit cstring_iterator(const char *str)
              : s{str}
          {}
```

1.  在某个位置对迭代器进行解引用时，它只会返回该位置的字符值：

```cpp
          char operator*() const { return *s; }
```

1.  递增迭代器只是递增字符串中的位置：

```cpp
          cstring_iterator& operator++() {
              ++s;
              return *this;
          }
```

1.  这是有趣的部分。我们为比较实现了`!=`运算符，因为它被 STL 算法和基于范围的`for`循环使用。然而，这一次，我们不是为迭代器与其他*迭代器*的比较实现它，而是为迭代器与*哨兵*的比较实现它。当我们将一个迭代器与另一个迭代器进行比较时，我们只能检查它们的内部字符串指针是否都指向相同的地址，这有些限制。通过与空的哨兵对象进行比较，我们可以执行完全不同的语义——我们检查迭代器指向的字符是否是终止`''`字符，因为这代表了字符串的*结束*！

```cpp
          bool operator!=(const cstring_iterator_sentinel) const {
              return s != nullptr && *s != '';
          }
      };
```

1.  为了在基于范围的`for`循环中使用它，我们需要一个围绕它的范围类，它会发出开始和结束的迭代器：

```cpp
      class cstring_range {
          const char *s {nullptr};
```

1.  用户在实例化期间唯一需要提供的是要迭代的字符串：

```cpp
      public:
          cstring_range(const char *str)
              : s{str}
          {}
```

1.  我们从`begin()`函数返回一个普通的`cstring_iterator`，它指向字符串的开头。从`end()`函数返回的只是*哨兵类型*。请注意，如果没有哨兵类型，我们也会返回一个迭代器，但是我们怎么知道字符串的末尾呢，如果我们没有提前搜索呢？

```cpp
          cstring_iterator begin() const { 
              return cstring_iterator{s}; 
          }
          cstring_iterator_sentinel end() const { 
              return {}; 
          }
      };
```

1.  就是这样。我们可以立即使用它。来自用户的字符串是我们无法提前知道长度的输入的一个例子。为了强制用户在启动程序时至少提供一个参数，如果用户没有在 shell 中启动程序时提供至少一个参数，我们将中止程序：

```cpp
      int main(int argc, char *argv[])
      {
          if (argc < 2) {
              std::cout << "Please provide one parameter.n";
              return 1;
          }
```

1.  如果程序到这一点仍在执行，那么我们知道`argv[1]`包含一些用户字符串：

```cpp
          for (char c : cstring_range(argv[1])) {
              std::cout << c;
          }
          std::cout << 'n';
      }
```

1.  编译和运行程序会产生以下输出：

```cpp
      $ ./main "abcdef"
      abcdef
```

循环打印我们刚刚输入的内容并不奇怪，因为这只是一个关于基于哨兵的迭代器范围实现的微型示例。这种迭代终止方法将帮助您在遇到*与结束位置比较*方法无法帮助的情况下实现自己的迭代器时。

# 使用检查迭代器自动检查迭代器代码

无论迭代器有多有用，以及它们代表的通用接口是什么，迭代器都很容易被*误用*，就像指针一样。在处理指针时，代码必须以一种方式编写，即当指向无效内存位置时*永远*不会取消引用它们。迭代器也是一样，但是有*很多规则*规定了迭代器何时有效以及何时失效。通过稍微研究 STL 文档，这些规则可以很容易地学习到，但仍然有可能编写错误的代码。

在最好的情况下，这种错误的代码会在*测试*时在*开发人员*面前爆炸，而不是在客户的机器上。然而，在许多情况下，代码似乎只是默默地工作，尽管它会取消引用悬空指针、迭代器等。在这种情况下，如果我们生成显示未定义行为的代码，我们希望能够*及早警告*。

幸运的是，有帮助！GNU STL 实现有一个*调试模式*，GNU C++编译器以及 LLVM clang C++编译器都支持用于为我们生成*额外敏感*和*冗长*的二进制文件的*额外库*，这些二进制文件可以立即在各种各样的错误上爆炸。这是*易于使用*和*非常有用*的，我们将在本节中进行演示。Microsoft Visual C++标准库也提供了激活额外检查的可能性。

# 如何做...

在本节中，我们将编写一个故意访问无效迭代器的程序：

1.  首先，我们包括头文件。

```cpp
      #include <iostream>
      #include <vector>
```

1.  现在，让我们实例化一个整数向量，并获得指向第一个项目的迭代器，值为`1`。我们对向量应用`shrink_to_fit()`，以确保其容量*确实*为`3`，因为它的实现*可能*分配了比必要更多的内存作为未来项目插入的小保留：

```cpp
      int main()
      {
          std::vector<int> v {1, 2, 3};
          v.shrink_to_fit();

          const auto it (std::begin(v));
```

1.  然后我们打印取消引用的迭代器，这是完全正常的：

```cpp
          std::cout << *it << 'n';
```

1.  接下来，让我们向向量追加一个新的数字。由于向量的大小不足以容纳另一个数字，它将自动增加其大小。它通过分配一个新的更大的内存块，将所有现有的项目移动到新的内存块，然后删除*旧*内存来实现这一点。

```cpp
          v.push_back(123);
```

1.  现在，让我们再次通过这个迭代器从向量中打印`1`。这很糟糕。为什么？嗯，当向量将所有的值移动到新的内存块并丢弃旧的内存块时，它没有告诉迭代器这个变化。这意味着迭代器仍然指向旧的位置，我们无法知道自那时起它到底发生了什么：

```cpp
          std::cout << *it << 'n'; // bad bad bad!
      }
```

1.  编译和运行这个程序会导致无缺陷的执行。应用程序不会崩溃，但当取消引用无效的指针时打印的内容几乎是随机的。把它留在这种状态下是非常危险的，但在这一点上，如果我们自己没有看到这个 bug，就没有人告诉我们！[](img/a81a4f4e-651a-463e-9a47-9f537f7ef4a6.png)

1.  调试标志来拯救！*GNU* STL 实现支持一个名为`_GLIBCXX_DEBUG`的预处理宏，它在 STL 中激活了大量的健全性检查代码。这会使程序变慢，但它*找到了 bug*。我们可以通过在编译器命令行中添加`-D_GLIBCXX_DEBUG`标志，或者在`include`行之前的代码文件头部定义它来激活它。正如你所看到的，它会在激活不同的 sanitizers 时杀死应用程序。让我们用 clang 有用（用于 Microsoft Visual C++编译器的已检查迭代器的激活标志是`/D_ITERATOR_DEBUG_LEVEL=1`）编译代码！[](img/abbbeeff-f181-453d-b88f-a5d9477b5b2c.png)

1.  LLVM/clang 实现的 STL 也有调试标志，但它们用于调试*STL*本身，而不是用户代码。对于用户代码，你可以激活不同的 sanitizers。让我们使用 clang 编译代码，使用`-fsanitize=address -fsanitize=undefined`标志，看看会发生什么！[](img/907bccd5-a08c-449e-b970-98f763f65587.png)

哇，这是一个非常精确的描述出了什么问题。如果没有被截断，这个屏幕截图可能会跨越这本书的*多个页面*。请注意，这不是 clang 的专属功能，它也适用于 GCC。

如果由于某个库丢失而出现运行时错误，那么你的编译器没有自动提供**libasan**和**libubsan**。尝试通过软件包管理器或类似的方式安装它们。

# 它是如何工作的...

正如我们所看到的，我们不需要改变代码就能获得这种对于有 bug 的代码的*触发器*功能。它基本上是*免费*的，只需在编译程序时在命令行中添加一些编译器标志即可。

这个功能是由*sanitizers*实现的。通常，sanitizer 由一个额外的编译器模块和一个运行时库组成。当 sanitizers 被激活时，编译器会向二进制文件中添加*额外的* *信息*和*代码*，这些信息是来自我们的程序。在运行时，链接到程序二进制文件中的 sanitizer 库可以，例如，替换`malloc`和`free`函数，以*分析*程序如何处理它获取的内存。

Sanitizers 可以检测不同类型的 bug。举几个有价值的例子：

+   **越界**：每当我们访问数组、向量或类似的东西超出其合法内存范围时，就会触发这个功能。

+   **释放后使用**：如果我们在释放堆内存后引用它，sanitizers 就会触发这个功能（我们在本节中就是这样做的）。

+   **整数溢出**：如果整数变量通过计算与不适合该变量的值而溢出，就会触发这个功能。对于有符号整数，算术环绕是未定义行为。

+   **指针对齐**：一些架构如果在内存中有奇怪的对齐方式就无法访问内存。

有许多这样的 bug 可以被 sanitizers 检测到。

*不可行*总是激活所有可用的消毒剂，因为它们会使程序变得*更慢*。然而，在你的*单元测试*和*集成测试*中总是激活消毒剂是一个很好的风格。

# 还有更多...

有很多不同的消毒剂用于不同的 bug 类别，它们仍在不断发展。我们可以和应该在互联网上了解如何改进我们的测试二进制文件。GCC 和 LLVM 项目主页在它们的在线文档页面中列出了它们的消毒能力：

+   [`gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html`](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html)

+   [`clang.llvm.org/docs/index.html`](http://clang.llvm.org/docs/index.html)（在目录中查找*sanitizers*）

彻底测试消毒剂是*每个*程序员都应该意识到并且*总是*应该做的事情。不幸的是，在许多公司中这并不是这样，尽管有 bug 的代码是所有*恶意软件*和*计算机病毒*的最重要入口点。

当你作为软件开发人员得到一份新工作时，检查你的团队是否真的使用了所有可能的消毒方法。如果没有，你有机会在工作的第一天修复重要且隐蔽的错误！

# 构建你自己的拉链迭代器适配器

不同的编程语言导致不同的编程风格。这是因为表达事物的方式不同，它们在每种用例的优雅程度上也不同。这并不奇怪，因为每种语言都是根据特定的目标设计的。

一种非常特殊的编程风格是*纯* *函数式编程*。它与 C 或 C++程序员所习惯的*命令式*编程有着神奇的不同。虽然这种风格非常不同，但它在许多情况下能够产生极其优雅的代码。

这种优雅的实现之一是公式的实现，比如数学点积。给定两个数学向量，对它们应用点积意味着对向量中相同位置的数字进行成对乘法，然后将所有这些乘积值相加。两个向量`(a, b, c) * (d, e, f)`的点积是`(a * e + b * e + c * f)`。当然，我们也可以用 C 和 C++来做。它可能看起来像下面这样：

```cpp
std::vector<double> a {1.0, 2.0, 3.0};
std::vector<double> b {4.0, 5.0, 6.0};

double sum {0};
for (size_t i {0}; i < a.size(); ++i) {
    sum += a[i] * b[i];
}
// sum = 32.0
```

那些被认为*更加优雅*的语言是什么样子的？

*Haskell* 是一种纯函数式语言，这是你可以用一个神奇的一行代码计算两个向量的点积的方法：

![](img/8fe01ef4-1b08-4026-b098-566e94367867.png)

*Python* 不是一种纯函数式语言，但它在某种程度上支持类似的模式，就像在下一个例子中所看到的那样：

![](img/ae8dfd8b-886e-4c0f-9226-80d8f1619286.png)

STL 提供了一个特定的算法叫做`std::inner_product`，它也可以用一行代码解决这个特定的问题。但关键是，在许多其他语言中，这样的代码可以*即时*用一行代码写出，*不需要*支持这个确切目的的特定库函数。

不用深入解释这种外来语法，两个例子中的一个重要共同点是神奇的`zip`函数。它是做什么的？它接受两个向量`a`和`b`，并将它们转换为一个*混合*向量。例如：`[a1, a2, a3]`和`[b1, b2, b3]`在被合并在一起时会得到`[ (a1, b1), (a2, b2), (a3, b3) ]`。仔细看一下；它真的很像拉链拉链一样工作！

相关的一点是现在可以在*一个*组合范围上进行迭代，可以进行成对的乘法，然后将它们相加到一个累加变量中。在 Haskell 和 Python 的例子中，没有添加任何循环或索引变量的噪音。

不可能使 C++代码与 Haskell 或 Python 一样优雅和通用，但本节解释了如何使用迭代器实现类似魔术，通过实现*zip 迭代器*。通过特定库更优雅地解决两个向量的点积的示例问题，这超出了本书的范围。但是，本节试图展示迭代器库可以通过提供极其通用的构建块来帮助编写表达力强的代码。

# 如何做...

在本节中，我们将重新创建来自 Haskell 或 Python 的*zip*函数。它将被硬编码为`double`变量的向量，以免分散迭代器机制的注意力。

1.  首先，我们需要包含一些头文件：

```cpp
      #include <iostream>
      #include <vector>
      #include <numeric>
```

1.  接下来，我们定义`zip_iterator`类。在遍历`zip_iterator`范围时，我们将在每次迭代步骤中从两个容器中获得一对值。这意味着我们同时遍历两个容器：

```cpp
      class zip_iterator {
```

1.  zip 迭代器需要保存两个迭代器，每个容器一个：

```cpp
          using it_type = std::vector<double>::iterator;

          it_type it1;
          it_type it2;
```

1.  构造函数只是保存我们想要迭代的两个容器的迭代器：

```cpp
      public:
          zip_iterator(it_type iterator1, it_type iterator2)
              : it1{iterator1}, it2{iterator2}
          {}
```

1.  增加 zip 迭代器意味着增加成员迭代器：

```cpp
          zip_iterator& operator++() {
              ++it1;
              ++it2;
              return *this;
          }
```

1.  如果两个 zip 迭代器的成员迭代器都与另一个 zip 迭代器中的对应迭代器不相等，则两个 zip 迭代器是不相等的。通常，人们会使用逻辑或(`||`)而不是和(`&&`)，但想象一下，范围的长度不相等。在这种情况下，将不可能同时匹配*两个*结束迭代器。这样，当我们到达*任一*范围的*第一个*结束迭代器时，我们可以中止循环：

```cpp
          bool operator!=(const zip_iterator& o) const {
              return it1 != o.it1 && it2 != o.it2;
          }
```

1.  相等比较运算符只是使用其他运算符实现，但否定结果：

```cpp
          bool operator==(const zip_iterator& o) const {
              return !operator!=(o);
          }
```

1.  解引用 zip 迭代器可以访问两个容器在相同位置的元素：

```cpp
          std::pair<double, double> operator*() const {
              return {*it1, *it2};
          }
      };
```

1.  这就是迭代器代码。我们需要使迭代器与 STL 算法兼容，因此我们为此定义了所需的类型特征样板代码。它基本上表示这个迭代器只是一个前向迭代器，在解引用时返回一对双值。虽然在这个示例中我们没有使用`difference_type`，但 STL 的不同实现可能需要它才能编译：

```cpp
      namespace std {

      template <>
      struct iterator_traits<zip_iterator> {
         using iterator_category = std::forward_iterator_tag;
         using value_type = std::pair<double, double>;
         using difference_type = long int;
      };

      }
```

1.  下一步是定义一个范围类，从其`begin`和`end`函数返回 zip 迭代器：

```cpp
      class zipper {
          using vec_type = std::vector<double>;
          vec_type &vec1;
          vec_type &vec2;
```

1.  它需要引用两个现有容器，以便从中形成 zip 迭代器：

```cpp
      public:
          zipper(vec_type &va, vec_type &vb)
              : vec1{va}, vec2{vb}
          {}
```

1.  `begin`和`end`函数只是提供开始和结束指针对，以便从中构造 zip 迭代器实例：

```cpp
          zip_iterator begin() const { 
              return {std::begin(vec1), std::begin(vec2)}; 
          }
          zip_iterator end() const { 
              return {std::end(vec1), std::end(vec2)}; 
          }
      };
```

1.  就像 Haskell 和 Python 示例中一样，我们定义了两个`double`值的向量。我们还在主函数中默认使用`std`命名空间：

```cpp
      int main()
      {
          using namespace std;
          vector<double> a {1.0, 2.0, 3.0};
          vector<double> b {4.0, 5.0, 6.0};
```

1.  zipper 对象将它们组合成一个类似向量的范围，我们可以看到`a`和`b`值的对：

```cpp
          zipper zipped {a, b};
```

1.  我们将使用`std::accumulate`来将范围中的所有项相加。我们不能直接这样做，因为这意味着我们要对`std::pair<double, double>`的实例求和，而这种情况下求和的概念是不被定义的。因此，我们将定义一个辅助 lambda，它接受一对值，将其成员相乘，并将其添加到累加器中。`std::accumulate`可以很好地处理具有这种签名的 lambda：

```cpp
          const auto add_product ([](double sum, const auto &p) {
             return sum + p.first * p.second;
          });
```

1.  现在，我们将它传递给`std::accumulate`，以及 zipped 范围的开始和结束迭代器对，以及累加器变量的起始值`0.0`，最终包含产品的总和：

```cpp
          const auto dot_product (accumulate(
                  begin(zipped), end(zipped), 0.0, add_product));
```

1.  让我们打印点积结果：

```cpp
          cout << dot_product << 'n';
      }
```

1.  编译并运行程序会产生正确的结果：

```cpp
      32
```

# 还有更多...

好吧，这需要*很多*工作来获得一点点语法糖，而且仍然不像 Haskell 代码那样优雅，而且不费吹灰之力。一个很大的缺陷是我们的小 zip 迭代器的硬编码特性--它只适用于`std::vector`范围内的双变量。通过一些模板代码和一些类型特征，可以使拉链器更通用。这样，它可以组合列表和向量，或者双端队列和映射，即使它们是专门针对完全不同的容器项类型的。

为了真正正确地使这样的类通用化，需要付出大量的工作和思考。幸运的是，这样的库已经存在。一个流行的非 STL 库是*Boost* `zip_iterator`。它非常通用且易于使用。

顺便说一句，如果你来这里是为了看到在 C++中执行*点积*最优雅的方法，并且并不真的关心 zip-iterators 的概念，你应该看看`std::valarray`。自己看看：

```cpp
#include <iostream>
#include <valarray>

int main()
{
    std::valarray<double> a {1.0, 2.0, 3.0};
    std::valarray<double> b {4.0, 5.0, 6.0};

    std::cout << (a * b).sum() << 'n';
}
```

# Ranges 库

有一个非常非常有趣的 C++库，支持拉链器和所有其他类型的魔术迭代器适配器、过滤器等等：*ranges*库。它受到 Boost ranges 库的启发，有一段时间看起来它会进入 C++17，但不幸的是，我们将不得不等待*下一个*标准。这样做的不幸之处在于，它将*大大*改进用 C++编写*富有表现力*和*快速*代码的可能性，通过从*通用*和*简单*的代码块组合*复杂*功能。

它的文档中有一些非常简单的例子：

1.  计算从`1`到`10`的所有数字的平方和：

```cpp
      const int sum = accumulate(view::ints(1)
                               | view::transform([](int i){return i*i;})
                               | view::take(10), 0);
```

1.  从数字向量中过滤出所有奇数，并将其余部分转换为字符串：

```cpp
      std::vector<int> v {1,2,3,4,5,6,7,8,9,10};

      auto rng = v | view::remove_if([](int i){return i % 2 == 1;})
                   | view::transform([](int i){return std::to_string(i);});

      // rng == {"2"s,"4"s,"6"s,"8"s,"10"s};
```

如果你感兴趣并且等不及下一个 C++标准，可以查看[`ericniebler.github.io/range-v3/`](https://ericniebler.github.io/range-v3/)上的 ranges 文档。
