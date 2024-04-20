# 字符串、流类和正则表达式

我们将在本章中涵盖以下内容：

+   创建、连接和转换字符串

+   从字符串的开头和结尾修剪空白

+   在不构造`std::string`对象的情况下获得`std::string`的舒适性

+   从用户输入中读取值

+   计算文件中的所有单词

+   使用 I/O 流操纵器格式化输出

+   从文件输入初始化复杂对象

+   从`std::istream`迭代器填充容器

+   使用`std::ostream`迭代器进行通用打印

+   将输出重定向到特定代码段的文件

+   通过继承`std::char_traits`创建自定义字符串类

+   使用正则表达式库对输入进行标记化

+   在不同上下文中舒适地漂亮地打印数字

+   从`std::iostream`错误中捕获可读的异常

# 介绍

本章专门讨论任意数据的字符串处理、解析和打印。对于这样的工作，STL 提供了其*I/O 流库*。该库基本上由以下类组成，每个类都用灰色框表示：

![](img/11e3bdf6-16ba-4b13-a4f1-07d22b52f7b8.png)

箭头显示了类的继承结构。这一开始可能看起来很压抑，但在本章中我们将使用大多数这些类，并逐个熟悉它们。当查看 C++ STL 文档中的这些类时，我们将无法直接找到它们的*确切*名称。这是因为图表中的名称是我们作为应用程序员看到的，但它们实际上大多只是带有`basic_`类名前缀的类的 typedef（例如，我们将更容易地在 STL 文档中搜索`basic_istream`而不是`istream`）。`basic_*` I/O 流类是可以为不同字符类型进行特化的模板。图表中的类是针对`char`值进行特化的。我们将在整本书中使用这些特化。如果我们在这些类名前加上`w`字符，我们会得到`wistream`，`wostream`等等--这些是`wchar_t`的特化 typedef，而不是`char`，例如。

在图表的顶部，我们看到`std::ios_base`。我们基本上永远不会直接使用它，但它被列出是为了完整性，因为所有其他类都继承自它。下一个特化是`std::ios`，它体现了维护数据流的对象的概念，可以处于*良好*状态、运行*空*数据状态（EOF）或某种*失败*状态。

我们将实际使用的第一个特化是`std::istream`和`std::ostream`。`"i"`和`"o"`前缀代表输入和输出。我们在 C++编程的最早期就已经见过它们，以最简单的形式出现在`std::cout`和`std::cin`（但也有`std::cerr`）的对象中。这些是这些类的实例，它们始终全局可用。我们通过`ostream`进行数据输出，通过`istream`进行输入。

同时继承自`istream`和`ostream`的类是`iostream`。它结合了输入和输出功能。当我们了解到来自`istream`，`ostream`和`iostream`三者组成的所有类可以如何使用时，我们基本上已经准备好立即使用所有接下来的类了：

`ifstream`，`ofstream`和`fstream`分别继承自`istream`，`ostream`和`iostream`，但它们提升了它们的能力，以重定向 I/O 从计算机的*文件系统*到文件。

`istringstream`，`ostringstream`和`iostringstream`的工作方式非常类似。它们帮助在内存中构建字符串，并/或从中消耗数据。

# 创建、连接和转换字符串

即使是非常古老的 C++程序员也会知道`std::string`。在 C 中，特别是在解析、连接、复制字符串等方面，字符串处理是繁琐且痛苦的，而`std::string`在简单性和安全性方面确实是一大进步。

由于 C++11，当我们想要将所有权转移到其他函数或数据结构时，我们甚至不需要再复制字符串，因为我们可以*移动*它们。这样，在大多数情况下，几乎没有太多的开销。

`std::string`在过去几个标准增量中有一些新功能。C++17 中完全新的是`std::string_view`。我们将稍微玩弄一下两者（但还有另一个配方，更集中于`std::string_view`的特性），以便对它们有所了解，并了解它们在 C++17 时代的工作方式。

# 如何做到...

在本节中，我们将创建字符串和字符串视图，并对它们进行基本的连接和转换：

1.  像往常一样，我们首先包括头文件并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <string>
      #include <string_view>
      #include <sstream>
      #include <algorithm>      

      using namespace std;
```

1.  首先让我们创建字符串对象。最明显的方法是实例化一个`string`类的对象`a`。我们通过给构造函数传递一个 C 风格的字符串来控制它的内容（在编译后作为包含字符的静态数组嵌入到二进制文件中）。构造函数将复制它并将其作为字符串对象`a`的内容。或者，我们可以使用字符串字面量操作符`""s`来初始化它，而不是从 C 风格字符串初始化它。它可以即时创建一个字符串对象。使用它来构造对象`b`，我们甚至可以使用自动类型推断：

```cpp
      int main()
      {
          string a { "a"  };
          auto   b ( "b"s );
```

1.  我们刚刚创建的字符串是将它们的输入从构造函数参数复制到它们自己的缓冲区中。为了不复制，而是*引用*底层字符串，我们可以使用`string_view`实例。这个类也有一个字面操作符，称为`""sv`：

```cpp
          string_view c { "c"   };
          auto        d ( "d"sv );
```

1.  好的，现在让我们玩一下我们的字符串和字符串视图。对于这两种类型，`std::ostream`类都有`operator<<`的重载，因此它们可以轻松地打印出来：

```cpp
          cout << a << ", " << b << 'n';
          cout << c << ", " << d << 'n';
```

1.  字符串类重载了`operator+`，所以我们可以*添加*两个字符串并得到它们的连接作为结果。这样，`"a" + "b"`的结果是`"ab"`。以这种方式连接`a`和`b`很容易。对于`a`和`c`，情况就不那么容易了，因为 c 不是一个`string`，而是一个`string_view`。我们首先必须从`c`中获取字符串，然后将其添加到`a`中。此时，有人可能会问，“等等，为什么你要将`c`复制到一个中间字符串对象中，然后再将其添加到`a`中？你不能通过使用`c.data()`来避免那个复制吗？”这是一个好主意，但它有一个缺陷--`string_view`实例不一定要携带零终止的字符串。这是一个可能导致缓冲区溢出的问题：

```cpp
          cout << a + b << 'n';
          cout << a + string{c} << 'n';
```

1.  让我们创建一个新的字符串，其中包含我们刚刚创建的所有字符串和字符串视图。通过使用`std::ostringstream`，我们可以将任何变量*打印*到一个行为完全像`std::cout`的流对象中，但它不会打印到 shell。相反，它会打印到*字符串缓冲区*中。在我们使用`operator<<`将所有变量流到一起并在它们之间使用一些分隔空间后，我们可以从中构造并打印一个新的字符串对象`o.str()`：

```cpp
          ostringstream o;

          o << a << " " << b << " " << c << " " << d;
          auto concatenated (o.str());
          cout << concatenated << 'n';
```

1.  现在我们还可以通过将所有字母转换为大写来转换这个新字符串，例如。C 库函数`toupper`，它将小写字符映射为大写字符并保持其他字符不变，已经可用，并且可以与`std::transform`结合使用，因为字符串基本上也是一个具有`char`项的可迭代容器对象：

```cpp
          transform(begin(concatenated), end(concatenated), 
                    begin(concatenated), ::toupper);
          cout << concatenated << 'n';
      }
```

1.  编译和运行程序会产生以下输出，这正是我们所期望的：

```cpp
      $ ./creating_strings 
      a, b
      c, d
      ab
      ac
      a b c d
      A B C D
```

# 它是如何工作的...

显然，字符串可以像数字一样使用`+`运算符进行相加，但这与数学无关，而是产生*连接*的字符串。为了将其与`string_view`混合使用，我们需要首先转换为`std::string`。

然而，非常重要的一点是，当在代码中混合字符串和字符串视图时，我们绝不能假设`string_view`背后的基础字符串是*零终止*的！这就是为什么我们宁愿写`"abc"s + string{some_string_view}`而不是`"abc"s + some_string_view.data()`。除此之外，`std::string`提供了一个成员函数`append`，可以处理`string_view`实例，但它会改变字符串，而不是返回一个新的带有字符串视图内容的字符串。

`std::string_view`很有用，但在与字符串和字符串函数混合使用时要小心。我们不能假设它们是以零结尾的，这在标准字符串环境中会很快出问题。幸运的是，通常有适当的函数重载，可以正确处理它们。

然而，如果我们想要进行复杂的字符串连接和格式化等操作，我们不应该逐个在字符串实例上执行。`std::stringstream`、`std::ostringstream`和`std::istringstream`类更适合这样做，因为它们在附加时增强了内存管理，并提供了我们从一般流中了解的所有格式化功能。在本节中，我们选择了`std::ostringstream`类，因为我们要创建一个字符串而不是解析它。`std::istringstream`实例可以从现有字符串实例中实例化，然后我们可以轻松地将其解析为其他类型的变量。如果我们想要结合两者，`std::stringstream`是完美的全能选手。

# 修剪字符串开头和结尾的空格。

特别是在从用户输入中获取字符串时，它们经常被不需要的空格污染。在另一个示例中，我们去除了单词之间出现的多余空格。

现在让我们看看被空格包围的字符串并去除它。`std::string`有一些很好的辅助函数可以完成这项工作。

阅读了这个使用普通字符串对象执行此操作的示例后，确保还阅读以下示例。在那里，我们将看到如何避免不必要的副本或数据修改，使用新的`std::string_view`类。

# 如何做...

在本节中，我们将编写一个辅助函数，用于识别字符串中的周围空格并返回一个不包含它的副本，然后我们将对其进行简要测试。

1.  和往常一样，首先是头文件包含和使用指令：

```cpp
      #include <iostream>
      #include <string>
      #include <algorithm>
      #include <cctype>

      using namespace std;
```

1.  我们的修剪字符串周围空格的函数接受一个现有字符串的常量引用。它将返回一个没有任何周围空格的新字符串：

```cpp
      string trim_whitespace_surrounding(const string &s)
      {
```

1.  `std::string`提供了两个很有用的函数，这些函数对我们非常有帮助。第一个是`string::find_first_not_of`，它接受一个包含我们要跳过的所有字符的字符串。这当然是空格，意味着空格字符 `' '`, 制表符 `'t'` 和换行符 `'n'`。它会返回第一个非空格字符的位置。如果字符串中只有空格，它会返回`string::npos`。这意味着如果我们从中修剪空格，只剩下一个空字符串。因此，在这种情况下，让我们返回一个空字符串：

```cpp
          const char whitespace[] {" tn"};
          const size_t first (s.find_first_not_of(whitespace));
          if (string::npos == first) { return {}; }
```

1.  我们现在知道新字符串应该从哪里开始，但我们还不知道它应该在哪里结束。因此，我们使用另一个方便的字符串函数`string::find_last_not_of`。它将返回字符串中最后一个非空白字符的位置：

```cpp
          const size_t last (s.find_last_not_of(whitespace));
```

1.  使用`string::substr`，我们现在可以返回由空格包围但不包含空格的字符串部分。这个函数接受两个参数--一个*位置*，表示从字符串的哪个位置开始，以及在这个位置之后的*字符数*：

```cpp
          return s.substr(first, (last - first + 1));
      }
```

1.  就是这样。让我们编写一个主函数，在其中创建一个字符串，用各种空格包围文本句子，以便对其进行修剪：

```cpp
      int main()
      {
          string s {" tn string surrounded by ugly"
                    " whitespace tn "};
```

1.  我们打印字符串的未修剪和修剪版本。通过用括号括起字符串，更容易看出修剪前它包含的空格：

```cpp
          cout << "{" << s << "}n";
          cout << "{" 
               << trim_whitespace_surrounding(s) 
               << "}n";
      }
```

1.  编译和运行程序会产生我们预期的输出：

```cpp
      $ ./trim_whitespace 
      {  
        string surrounded by ugly whitespace    
         }
      {string surrounded by ugly whitespace}
```

# 它是如何工作的...

在这一部分，我们使用了`string::find_first_not_of`和`string::find_last_not_of`。这两个函数都接受一个 C 风格的字符串，它作为一个应该在搜索不同字符时跳过的字符列表。如果我们有一个携带字符串`"foo bar"`的字符串实例，并且在它上调用`find_first_not_of("bfo ")`，它将返回值`5`，因为`'a'`字符是第一个不在`"bfo "`字符串中的字符。参数字符串中字符的顺序并不重要。

相同的函数也存在相反的逻辑，尽管我们在这个示例中没有使用它们：`string::find_first_of`和`string::find_last_of`。

与基于迭代器的函数类似，我们需要检查这些函数是否返回字符串中的实际位置，还是表示它们*没有*找到满足约束条件的字符位置的值。如果它们没有找到，它们会返回`string::npos`。

从我们的辅助函数中检索到的字符位置，我们建立了一个不包含周围空白的子字符串，使用`string::substring`。这个函数接受一个相对偏移和一个字符串长度，然后返回一个新的字符串实例，其中包含了那个子字符串。例如，`string{"abcdef"}.substr(2, 2)`将返回一个新的字符串`"cd"`。

# 获得 std::string 的便利性，而不需要构造 std::string 对象的成本

`std::string`类是一个非常有用的类，因为它极大地简化了处理字符串的过程。一个缺点是，如果我们想传递它的子字符串，我们需要传递一个指针和一个长度变量，两个迭代器，或者子字符串的副本。我们在上一个示例中做到了这一点，我们通过获取不包含周围空白的子字符串范围的副本来实现了这一点。

如果我们想要将字符串或子字符串传递给甚至不支持`std::string`的库，我们只能提供一个原始字符串指针，这有点令人失望，因为它让我们回到了旧的 C 语言时代。就像子字符串问题一样，原始指针并不携带有关字符串长度的信息。这样，一个人将不得不实现一个指针和字符串长度的捆绑。

以简化的方式来说，这正是`std::string_view`。它自 C++17 起可用，并提供了一种将指向某个字符串的指针与该字符串的大小配对的方法。它体现了为数据数组提供引用类型的想法。

如果我们设计的函数以前接受`std::string`实例作为参数，但没有改变它们以需要字符串实例重新分配保存实际字符串负载的内存的方式，我们现在可以使用`std::string_view`，并且更兼容于 STL-agnostic 的库。我们可以让其他库提供对其复杂字符串实现背后的负载字符串的`string_view`视图，然后在我们的 STL 代码中使用它。这样，`string_view`类就充当了一个最小且有用的接口，可以在不同的库之间共享。

另一个很酷的事情是，`string_view`可以被用作对更大的字符串对象的子字符串的非复制引用。有很多可以利用它的可能性。在这一部分，我们将使用`string_view`来玩耍，以便对其优势和劣势有所了解。我们还将看到如何通过调整字符串视图而不是修改或复制实际字符串来隐藏字符串的周围空白。这种方法避免了不必要的复制或数据修改。

# 如何做...

我们将实现一个依赖于一些`string_view`特性的函数，然后，我们将看到我们可以将多少不同类型的数据输入到其中：

1.  首先是头文件包含和使用指令：

```cpp
      #include <iostream>
      #include <string_view>

      using namespace std;
```

1.  我们实现了一个函数，它只接受一个`string_view`作为参数：

```cpp
      void print(string_view v)
      {
```

1.  在对输入字符串进行任何操作之前，我们去除任何前导和尾随空白。我们不会改变字符串，但是通过将其缩小到实际的非空白部分，*视图*会改变。`find_first_not_of`函数将找到字符串中第一个不是空格（`' '`）、制表符（`'t'`）和换行符（`'n'`）的字符。通过`remove_prefix`，我们将内部的`string_view`指针移动到第一个非空白字符。如果字符串只包含空白，`find_first_not_of`函数将返回值`npos`，即`size_type(-1)`。由于`size_type`是无符号变量，这将变成一个非常大的数字。因此，我们取两者中较小的一个：`words_begin`或字符串视图的大小：

```cpp
          const auto words_begin (v.find_first_not_of(" tn"));
          v.remove_prefix(min(words_begin, v.size()));
```

1.  我们对尾随空白做同样的处理。`remove_suffix`会缩小视图的大小变量：

```cpp
          const auto words_end (v.find_last_not_of(" tn"));
          if (words_end != string_view::npos) {
              v.remove_suffix(v.size() - words_end - 1);
          }
```

1.  现在我们可以打印字符串视图及其长度：

```cpp
          cout << "length: " << v.length()
               << " [" << v << "]n";
      }
```

1.  在我们的主函数中，我们通过使用完全不同的参数类型来玩弄新的`print`函数。首先，我们给它一个运行时的`char*`字符串，来自`argv`指针。在运行时，它包含了我们可执行文件的文件名。然后，我们给它一个空的`string_view`实例。然后，我们用 C 风格的静态字符字符串和`""sv`字面量来给它提供参数，这会在我们的程序中构造一个`string_view`。最后，我们给它一个`std::string`。好处是，为了调用`print`函数，这些参数都没有被修改或复制。没有堆分配发生。对于许多和/或大字符串，这是非常高效的。

```cpp
      int main(int argc, char *argv[])
      {
          print(argv[0]);
          print({});
          print("a const char * array");
          print("an std::string_view literal"sv);
          print("an std::string instance"s);
```

1.  我们没有测试去除空白的功能。所以，让我们给它一个有很多前导和尾随空白的字符串：

```cpp
          print(" tn foobar n t ");
```

1.  另一个很酷的功能是，`string_view`给我们访问的字符串不必是*零终止*的。如果我们构造一个字符串，比如`"abc"`，没有尾随零，`print`函数仍然可以安全地处理它，因为`string_view`也携带了它指向的字符串的大小：

```cpp
          char cstr[] {'a', 'b', 'c'};
          print(string_view(cstr, sizeof(cstr)));
      }
```

1.  编译和运行程序会产生以下输出。所有字符串都被正确处理。我们填充了大量前导和尾随空白的字符串被正确过滤，没有零终止的`abc`字符串也被正确打印，没有任何缓冲区溢出：

```cpp
      $ ./string_view 
      length: 17 [./string_view]
      length: 0 []
      length: 20 [a const char * array]
      length: 27 [an std::string_view literal]
      length: 23 [an std::string instance]
      length: 6 [foobar]
      length: 3 [abc]
```

# 它是如何工作的...

我们刚刚看到，我们可以调用接受`string_view`参数的函数，基本上可以使用任何类似字符串的东西，它以连续方式存储字符。在我们的`print`调用中，没有对基础字符串进行任何*复制*。 

有趣的是，在我们的`print(argv[0])`调用中，字符串视图自动确定了字符串长度，因为这是一个约定的零结尾字符串。反过来，不能假设可以通过计算直到达到零终止符为止的项目数来确定`string_view`实例的数据长度。因此，我们必须始终小心地处理`string_view::data()`指向的字符串视图数据的指针。通常的字符串函数大多假定零终止，因此，使用指向字符串视图有效载荷的原始指针可能会非常严重地缓冲区溢出。最好使用已经期望字符串视图的接口。

除此之外，我们已经从`std::string`中知道了很多豪华的接口。

使用`std::string_view`来传递字符串或子字符串，可以避免复制或堆分配，同时不失去字符串类的便利性。但要注意`std::string_view`放弃了字符串以零结尾的假设。

# 从用户输入读取值

这本书中的许多食谱都从输入源（如标准输入或文件）中读取数值，并对其进行处理。这次我们只关注读取，并学习更多关于错误处理的知识，如果从流中读取的内容出现问题，我们需要处理它，而不是终止整个程序。

在这个食谱中，我们只会从用户输入中读取，但一旦我们知道如何做到这一点，我们也知道如何从任何其他流中读取。用户输入是通过`std::cin`读取的，这本质上是一个输入流对象，就像`ifstream`和`istringstream`的实例一样。

# 如何做...

在本节中，我们将读取用户输入到不同的变量中，并看看如何处理错误，以及如何将输入复杂化为有用的块：

1.  这次我们只需要`iostream`。因此，让我们包含这个单一的头文件，并声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>

      using namespace std;
```

1.  首先提示用户输入两个数字。我们将把它们解析成一个`int`和一个`double`变量。用户可以用空格分隔它们。例如，`1 2.3`是一个有效的输入：

```cpp
      int main()
      {
          cout << "Please Enter two numbers:n> ";
          int x;
          double y;
```

1.  解析和错误检查同时在`if`分支的条件部分完成。只有在两个数字都能解析出来时，它们对我们才有意义，我们才会打印它们：

```cpp
          if (cin >> x >> y) {
              cout << "You entered: " << x 
                   << " and " << y << 'n';
```

1.  如果由于任何原因解析失败，我们会告诉用户解析没有成功。`cin`流对象现在处于*失败状态*，直到我们再次清除失败状态之前，它不会给我们其他输入。为了能够解析新的输入，我们调用`cin.clear()`并丢弃到目前为止收到的所有输入。丢弃是用`cin.ignore`完成的，我们指定丢弃最大数量的字符，直到最终看到换行符，然后将其丢弃。之后的所有内容又变得有趣起来：

```cpp
          } else {
              cout << "Oh no, that did not go well!n";
              cin.clear();
              cin.ignore(
                  std::numeric_limits<std::streamsize>::max(),
                  'n');
          }
```

1.  现在让我们要求输入其他内容。我们让用户输入名字。由于名字可以由多个用空格分隔的单词组成，空格字符不再是一个好的分隔符。因此，我们使用`std::getline`，它接受一个流对象，比如`cin`，一个字符串引用，它将把输入复制到其中，以及一个分隔字符。让我们选择逗号（`,`）作为分隔字符。通过不仅仅使用`cin`，而是使用`cin >> ws`作为`getline`的流参数，我们可以使`cin`在任何名字之前丢弃任何前导空格。在每个循环步骤中，我们打印当前的名字，但如果一个名字是空的，我们就退出循环：

```cpp
          cout << "now please enter some "
                  "comma-separated names:n> ";

          for (string s; getline(cin >> ws, s, ',');) {
              if (s.empty()) { break; }
              cout << "name: "" << s << ""n";
          }
      }
```

1.  编译和运行程序会产生以下输出，假设我们只输入了有效的输入。数字是`"1 2"`，被正确解析，然后我们输入一些名字，它们也被正确列出。以两个连续逗号的形式输入空名字会退出循环：

```cpp
      $ ./strings_from_user_input 
      Please Enter two numbers:
      > 1 2
      You entered: 1 and 2
      now please enter some comma-separated names:
      > john doe,  ellen ripley,       alice,    chuck norris,,
      name: "john doe"
      name: "ellen ripley"
      name: "alice"
      name: "chuck norris"
```

1.  当再次运行程序时，在开始输入错误的数字时，我们看到程序正确地选择了另一个分支，丢弃了错误的输入，并正确地继续了名字的输入。尝试使用`cin.clear()`和`cin.ignore(...)`行，看看它们如何影响名字读取代码：

```cpp
      $ ./strings_from_user_input
      Please Enter two numbers:
      > a b
      Oh no, that did not go well!
      now please enter some comma-separated names:
      > bud spencer, terence hill,,
      name: "bud spencer"
      name: "terence hill"
```

# 工作原理...

在本节中，我们进行了一些复杂的输入检索。首先要注意的是，我们总是同时进行检索和错误检查。

表达式`cin >> x`的结果再次是对`cin`的引用。这样，我们可以写`cin >> x >> y >> z >> ...`。同时，它也可以在布尔上下文中转换为布尔值，比如`if`条件中。布尔值告诉我们最后一次读取是否成功。这就是为什么我们能够写`if (cin >> x >> y) {...}`。

例如，如果我们尝试读取一个整数，但输入包含`"foobar"`作为下一个标记，那么将其解析为整数是不可能的，流对象进入*失败状态*。这只对解析尝试很重要，但对整个程序并不重要。重置它然后尝试其他任何事情都是可以的。在我们的配方程序中，我们尝试在尝试读取两个数字失败后读取一系列名称。在尝试读取这些数字失败的情况下，我们使用`cin.clear()`将`cin`恢复到工作状态。但是，它的内部光标仍然停留在我们键入的内容而不是数字上。为了丢弃这个旧输入并清除名称输入的管道，我们使用了非常长的表达式`cin.ignore(std::numeric_limits<std::streamsize>::max(), 'n');`。这是必要的，因为我们想要从一个真正新鲜的缓冲区开始，当我们要求用户提供一系列名称时。

下面的循环一开始可能看起来很奇怪：

```cpp
for (string s; getline(cin >> ws, s, ',');) { ... }
```

在`for`循环的条件部分中，我们使用`getline`。`getline`函数接受一个输入流对象，一个字符串引用作为输出参数，以及一个分隔符字符。默认情况下，分隔符字符是换行符。在这里，我们将其定义为逗号（`,`）字符，因此列表中的所有名称，例如`"john, carl, frank"`，都将被单独读取。

到目前为止，一切都很好。但是将`cin >> ws`函数作为流对象提供是什么意思呢？这使得`cin`首先刷新所有空白字符，这些空白字符位于下一个非空白字符之前和最后一个逗号之后。回顾一下`"john, carl, frank"`的例子，我们将得到子字符串`"john"`，`" carl"`和`" frank"`，而不使用`ws`。注意`carl`和`frank`的不必要的前导空格字符？由于我们对输入流的`ws`预处理，这些实际上消失了。

# 在文件中计算所有单词

假设我们读取一个文本文件，并且想要计算文本中的单词数。我们定义一个单词是两个空格字符之间的字符范围。我们该如何做呢？

我们可以计算空格的数量，例如，因为单词之间必须有空格。在句子`"John has a funny little dog."`中，我们有五个空格字符，所以我们可以说有六个单词。

如果我们有一个带有空格噪音的句子，例如`" John has t anfunny little dog ."`？这个字符串中有太多不必要的空格，甚至不仅仅是空格。从本书的其他配方中，我们已经学会了如何去除这种多余的空格。因此，我们可以首先将字符串预处理为正常的句子形式，然后应用计算空格字符的策略。是的，这是可行的，但有一个*更*简单的方法。为什么我们不使用 STL 已经提供给我们的东西呢？

除了为这个问题找到一个优雅的解决方案之外，我们还将让用户选择是否从标准输入或文本文件中计算单词。

# 如何做...

在本节中，我们将编写一个一行函数，用于计算输入缓冲区中的单词，并让用户选择输入缓冲区的读取位置：

1.  首先让我们包括所有必要的头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <fstream>
      #include <string>
      #include <algorithm>
      #include <iterator>      

      using namespace std;
```

1.  我们的`wordcount`函数接受一个输入流，例如`cin`。它创建一个`std::input_iterator`迭代器，该迭代器从流中标记字符串，然后将它们传递给`std::distance`。`distance`参数接受两个迭代器作为参数，并尝试确定从一个迭代器位置到另一个迭代器位置需要多少递增步骤。对于*随机访问*迭代器来说，这很简单，因为它们实现了数学差异操作（`operator-`）。这样的迭代器可以像指针一样相互减去。然而，`istream_iterator`是*前向*迭代器，必须一直前进直到等于结束迭代器。最终，所需的步骤数就是单词数：

```cpp
      template <typename T>
      size_t wordcount(T &is)
      {
          return distance(istream_iterator<string>{is}, {});
      }
```

1.  在我们的主函数中，我们让用户选择输入流是`std::cin`还是输入文件：

```cpp
      int main(int argc, char **argv)
      {
          size_t wc;
```

1.  如果用户在 shell 中与文件名一起启动程序（例如`$ ./count_all_words some_textfile.txt`），那么我们将从`argv`命令行参数数组中获取该文件名，并打开它，以便将新的输入文件流输入到`wordcount`中：

```cpp
          if (argc == 2) {
              ifstream ifs {argv[1]};
              wc = wordcount(ifs);
```

1.  如果用户在没有任何参数的情况下启动程序，我们假设输入来自标准输入：

```cpp
          } else {
              wc = wordcount(cin);
          }
```

1.  就是这样，所以我们只需打印我们保存在变量`wc`中的单词数：

```cpp
          cout << "There are " << wc << " wordsn";
      };
```

1.  让我们编译并运行程序。首先，我们从标准输入中输入程序，没有任何文件参数。我们可以通过管道将 echo 调用与一些单词一起输入，或者启动程序并从键盘输入一些单词。在后一种情况下，我们可以通过按*Ctrl*+*D*来停止输入。这是将一些单词回显到程序中的方式：

```cpp
      $ echo "foo bar baz" | ./count_all_words 
      There are 3 words
```

1.  当以源代码文件作为输入启动程序时，它将计算它由多少个单词组成：

```cpp
      $ ./count_all_words count_all_words.cpp
      There are 61 words
```

# 它是如何工作的...

没有太多要说的了；大部分内容在实现时已经解释过了，因为这个程序非常简短。我们可以详细介绍一点的是，我们完全可以以相互替换的方式使用`std::cin`和`std::ifstream`实例。`cin`是`std::istream`类型，而`std::ifstream`继承自`std::istream`。看一下本章开头的类继承图表。这样，它们在运行时是完全可以互换的。

通过使用流抽象来保持代码模块化。这有助于解耦源代码部分，并使您的代码易于测试，因为您可以注入任何其他匹配类型的流。

# 使用 I/O 流操纵器格式化输出

在许多情况下，仅仅打印字符串和数字是不够的。有时，数字需要以十进制数打印，有时以十六进制数打印，有时甚至以八进制数打印。有时我们希望在十六进制数前面看到`"0x"`前缀，有时不希望。

在打印浮点数时，我们可能也有很多事情想要影响。小数值是否总是以相同的精度打印？它们是否应该被打印？或者，也许我们想要科学计数法？

除了科学表示法和十六进制、八进制等，我们还希望以整洁的形式呈现用户输出。有些输出可以以表格的形式排列，以使其尽可能可读。

当然，所有这些都可以通过输出流实现。当从输入流中*解析*值时，其中一些设置也很重要。在本教程中，我们将通过玩弄这些所谓的**I/O 操纵器**来感受一下。有时，它们看起来很棘手，所以我们也会深入一些细节。

# 如何做...

在本节中，我们将使用各种格式设置打印数字，以便熟悉 I/O 操纵器：

1.  首先，我们包括所有必要的标头，并声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <iomanip>
      #include <locale>      

      using namespace std;
```

1.  接下来，我们定义一个辅助函数，它以不同的样式打印单个整数值。它接受填充宽度和填充字符，默认为空格`' '`：

```cpp
      void print_aligned_demo(int val, 
                              size_t width, 
                              char fill_char = ' ')
      {
```

1.  使用`setw`，我们可以设置打印数字时的最小字符数。例如，如果我们以宽度为`6`打印`123`，我们会得到`" 123"`或`"123 "`。我们可以使用`std::left`、`std::right`和`std::internal`控制填充发生在哪一侧。在以十进制形式打印数字时，`internal`看起来与`right`相同。但是，例如，如果我们以宽度为`6`和`internal`打印值`0x1`，我们会得到`"0x 6"`。`setfill`操纵器定义了用于填充的字符。我们将尝试不同的样式：

```cpp
          cout << "================n";
          cout << setfill(fill_char);
          cout << left << setw(width) << val << 'n';
          cout << right << setw(width) << val << 'n';
          cout << internal << setw(width) << val << 'n';
      }
```

1.  在主函数中，我们开始使用刚刚实现的函数。首先，我们打印值`12345`，宽度为`15`。我们这样做两次，但第二次，我们使用`'_'`字符进行填充：

```cpp
      int main()
      {
          print_aligned_demo(123456, 15);
          print_aligned_demo(123456, 15, '_');
```

1.  之后，我们以与之前相同的宽度打印值`0x123abc`。但在这之前，我们应用了`std::hex`和`std::showbase`，告诉输出流对象`cout`应该以十六进制格式打印数字，并且应该在它们前面添加`"0x"`，以便明确表示它们应该被解释为十六进制：

```cpp
          cout << hex << showbase;
          print_aligned_demo(0x123abc, 15);
```

1.  我们也可以使用`oct`做同样的事情，告诉`cout`使用八进制系统来打印数字。`showbase`仍然有效，因此`0`将被添加到每个打印的数字前面：

```cpp
          cout << oct;
          print_aligned_demo(0123456, 15);
```

1.  使用`hex`和`uppercase`，我们得到了`"0x"`中的`'x'`大写打印。`'0x123abc'`中的`'abc'`也是大写的：

```cpp
          cout << "A hex number with upper case letters: "
               << hex << uppercase << 0x123abc << 'n';
```

1.  如果我们想再次以十进制格式打印`100`，我们必须记住之前已经将流切换为`hex`。通过使用`dec`，我们可以将其恢复为正常状态：

```cpp
          cout << "A number: " << 100 << 'n';
          cout << dec;

          cout << "Oops. now in decimal again: " << 100 << 'n';
```

1.  我们还可以配置布尔值的打印方式。默认情况下，`true`打印为`1`，`false`打印为`0`。使用`boolalpha`，我们可以将其设置为文本表示：

```cpp
          cout << "true/false values: " 
               << true << ", " << false << 'n';
          cout << boolalpha
               << "true/false values: "
               << true << ", " << false << 'n';
```

1.  让我们来看看`float`和`double`类型的浮点变量。如果我们打印一个数字，比如`12.3`，它当然会打印为`12.3`。如果我们有一个数字，比如`12.0`，输出流将会去掉小数点，我们可以使用`showpoint`来改变这一点。使用这个，小数点总是会显示：

```cpp
          cout << "doubles: "
               << 12.3 << ", "
               << 12.0 << ", "
               << showpoint << 12.0 << 'n';
```

1.  浮点数的表示可以是科学或固定的。`scientific`表示数字被*标准化*为这样一种形式，即小数点前只有一个数字，然后打印出指数，这是将数字乘回其实际大小所需的。例如，值`300.0`将被打印为`"3.0E2"`，因为`300`等于`3.0 * 10²`。`fixed`则恢复为正常的十进制表示法：

```cpp
          cout << "scientific double: " << scientific 
               << 123000000000.123 << 'n';
          cout << "fixed      double: " << fixed 
               << 123000000000.123 << 'n';
```

1.  除了表示法，我们还可以决定浮点数打印的精度。让我们创建一个非常小的值，并以小数点后 10 位的精度打印它，然后再以小数点后只有一位的精度打印它：

```cpp
          cout << "Very precise double: " 
               << setprecision(10) << 0.0000000001 << 'n';
          cout << "Less precise double: " 
               << setprecision(1)  << 0.0000000001 << 'n';
      }
```

1.  编译并运行程序会产生以下冗长的输出。前四个输出块是打印助手函数的输出，该函数对`setw`和`left`/`right`/`internal`修饰符进行了调整。之后，我们对基本表示、布尔表示和浮点数格式进行了调整。熟悉每种格式是个好主意：

```cpp
      $ ./formatting 
      ================
      123456         
               123456
               123456
      ================
      123456_________
      _________123456
      _________123456
      ================
      0x123abc       
             0x123abc
      0x       123abc
      ================
      0123456        
              0123456
              0123456
      A hex number with upper case letters: 0X123ABC
      A number: 0X64
      Oops. now in decimal again: 100
      true/false values: 1, 0
      true/false values: true, false
      doubles: 12.3, 12, 12.0000
      scientific double: 1.230000E+11
      fixed      double: 123000000000.123001
      Very precise double: 0.0000000001
      Less precise double: 0.0
```

# 它是如何工作的...

所有这些有时相当长的`<< foo << bar`流表达式如果读者不清楚每个表达式的含义，会让人感到困惑。因此，让我们来看一下现有格式修饰符的表格。它们都应该放在`input_stream >> modifier`或`output_stream << modifier`表达式中，然后影响接下来的输入或输出：

| **符号** | **含义** |
| --- | --- |
| `setprecision(int n)` | 设置打印或解析浮点值时的精度参数。 |
| `showpoint` / `noshowpoint` | 启用或禁用打印浮点数的小数点，即使它们没有任何小数位 |
| `fixed` / `scientific` / `hexfloat` / `defaultfloat` | 数字可以以固定样式（这是最直观的样式）或科学样式打印。`fixed`和`scientific`代表这些模式。`hexfloat`激活这两种模式，它以十六进制浮点表示法格式化浮点数。`defaultfloat`取消这两种模式。 |
| `showpos` / `noshowpos` | 启用或禁用打印正浮点值的`'+'`前缀 |
| `setw(int n)` | 读取或写入确切的`n`个字符。在读取时，这会截断输入。在打印时，如果输出长度小于`n`个字符，则会应用填充。 |
| `setfill(char c)` | 在应用填充（参见`setw`）时，用字符值`c`填充输出。默认值是空格（`' '`）。 |
| `internal` / `left` / `right` | `left`和`right`控制固定宽度打印（参见`setw`）的填充位置。`internal`将填充字符放在整数及其负号、十六进制前缀和十六进制打印值，或货币单位和值之间的中间位置。 |
| `dec` / `hex` / `oct` | 可以在十进制、十六进制和八进制基数系统中打印和解析整数值 |
| `setbase(int n)` | 这是`dec`/`hex`/`oct`的数值同义函数，如果与值`10`/`16`/`8`一起使用，则它们是等效的。其他值会将基础选择重置为`0`，这将再次导致十进制打印，或者根据输入的前缀进行解析。 |
| `quoted(string)` | 以引号打印字符串或从带引号的输入中解析字符串，然后删除引号。`string`可以是 String 类实例或 C 风格的字符数组。 |
| `boolalpha` / `noboolalpha` | 以字母表示形式而不是`1`/`0`字符串打印或解析布尔值 |
| `showbase` / `noshowbase` | 在打印或解析数字时启用或禁用基数前缀。对于`hex`，这是`0x`；对于`octal`，这是`0`。 |
| `uppercase` / `nouppercase` | 在打印浮点和十六进制值时启用或禁用大写或字母字符 |

熟悉它们的最佳方法是稍微研究它们的多样性并与它们玩耍。

然而，在与它们玩耍时，我们可能已经注意到这些修改器中的大多数似乎是*粘性的*，而其中一些则不是。粘性意味着一旦应用，它们似乎会永久地影响输入/输出，直到它们再次被重置。此表中唯一不粘性的是`setw`和`quoted`。它们只影响输入/输出中的下一项。这是很重要的，因为如果我们以某种格式打印一些输出，我们应该在之后整理我们的流对象格式设置，因为来自不相关代码的下一个输出可能看起来很疯狂。同样适用于输入解析，其中错误的 I/O 操作器选项可能会导致问题。

我们并没有真正使用它们中的任何一个，因为它们与格式无关，但出于完整性的原因，我们也应该看一下其他一些流状态操作器：

| **符号** | **含义** |
| --- | --- |
| `skipws` / `noskipws` | 启用或禁用输入流跳过空白的功能 |
| `unitbuf` / `nounitbuf` | 启用或禁用任何输出操作后立即刷新输出缓冲区 |
| `ws` | 可以在输入流上使用，以跳过流头部的任何空白 |
| `ends` | 在流中写入一个字符串终止`''`字符 |
| `flush` | 立即刷新输出缓冲区中的内容 |
| `endl` | 在输出流中插入一个 `'n'` 字符并刷新输出 |

从中，只有`skipws`/`noskipws`和`unitbuf`/`nounitbuf`是粘性的。

# 从文件输入初始化复杂对象

读取单独的整数、浮点数和单词字符串非常容易，因为输入流对象的 `>>` 操作符已经为所有这些类型重载了，并且输入流方便地为我们删除了所有中间的空白。

但是，如果我们有一个更复杂的结构，我们想要从输入流中读取，如果我们需要读取包含多个单词的字符串（因为它们通常会被分成单个单词，因为空白会被跳过），那该怎么办呢？

对于任何类型，都可以提供另一个输入流 `operator>>` 重载，我们将看到如何做到这一点。

# 如何做...

在本节中，我们将定义一个自定义数据结构，并提供从标准输入流中读取这些项目的功能：

1.  首先，我们需要包含一些头文件，并且为了方便起见，我们声明默认使用 `std` 命名空间：

```cpp
      #include <iostream>
      #include <iomanip>
      #include <string>
      #include <algorithm>
      #include <iterator>
      #include <vector>      

      using namespace std;
```

1.  作为一个复杂对象的例子，我们定义了一个 `city` 结构。一个城市应该有一个名称、一个人口数量和地理坐标：

```cpp
      struct city {
          string name;
          size_t population;
          double latitude;
          double longitude;
      };
```

1.  为了能够从串行输入流中读取这样一个城市，我们需要重载流函数 `operator>>`。在这个操作符中，我们首先使用 `ws` 跳过所有前导空白，因为我们不希望空白污染城市名称。然后，我们读取一整行文本输入。这意味着在输入文件中，只有一整行文本只携带城市对象的名称。然后，在换行符之后，跟着一个以空格分隔的数字列表，表示人口数量、地理纬度和经度：

```cpp
      istream& operator>>(istream &is, city &c)
      {
          is >> ws;
          getline(is, c.name);
          is >> c.population 
             >> c.latitude 
             >> c.longitude;
          return is;
      }
```

1.  在我们的主函数中，我们创建了一个可以容纳一系列城市项目的向量。我们使用 `std::copy` 来填充它。复制调用的输入是一个 `istream_iterator` 范围。通过将 `city` 结构类型作为模板参数传递给它，它将使用我们刚刚实现的 `operator>>` 函数重载：

```cpp
      int main()
      {
          vector<city> l;

          copy(istream_iterator<city>{cin}, {}, 
               back_inserter(l));
```

1.  为了查看我们的城市解析是否正确，我们打印了列表中的内容。I/O 格式化，`left << setw(15) <<`，导致城市名称被填充了空白，所以我们得到了一个很好的可读形式的输出：

```cpp
          for (const auto &[name, pop, lat, lon] : l) {
              cout << left << setw(15) << name
                   << " population=" << pop
                   << " lat=" << lat
                   << " lon=" << lon << 'n';
          }
      }
```

1.  我们将喂给我们的程序的文本文件看起来像这样。有四个示例城市及其人口数量和地理坐标：

```cpp
      Braunschweig
      250000 52.268874 10.526770
      Berlin
      4000000 52.520007 13.404954
      New York City
      8406000 40.712784 -74.005941
      Mexico City
      8851000 19.432608 -99.133208
```

1.  编译和运行程序产生了以下输出，这正是我们所期望的。尝试通过在城市名称之前添加一些不必要的空白来篡改输入文件，以查看它是如何被过滤掉的：

```cpp
      $ cat cities.txt  | ./initialize_complex_objects
      Braunschweig    population=250000 lat=52.2689 lon=10.5268
      Berlin          population=4000000 lat=52.52 lon=13.405
      New York City   population=8406000 lat=40.7128 lon=-74.0059
      Mexico City     population=8851000 lat=19.4326 lon=-99.1332
```

# 它是如何工作的...

这又是一个简短的示例。我们所做的唯一的事情就是创建一个新的结构 `city`，然后为这种类型重载 `std::istream` 迭代器的 `operator>>`，就是这样。这已经使我们能够从标准输入中反序列化城市项目使用 `istream_iterator<city>`。

关于错误检查可能还有一个未解决的问题。让我们再次看看 `operator>>` 的实现：

```cpp
      istream& operator>>(istream &is, city &c)
      {
          is >> ws;
          getline(is, c.name);
          is >> c.population >> c.latitude >> c.longitude;
          return is;
      }
```

我们正在读取很多不同的东西。如果其中一个失败了，下一个又怎么样？这是否意味着我们可能会用错误的“偏移量”读取所有后续的项目？不，这是不可能的。一旦这些项目中的一个无法从输入流中解析出来，输入流对象就会进入错误状态，并拒绝进一步解析任何内容。这意味着，例如 `c.population` 或 `c.latitude` 无法解析，剩余的 `>>` 操作数就会“跳过”，我们将以一个半反序列化的城市对象离开这个操作符函数范围。

在调用方面，当我们写 `if (input_stream >> city_object)` 时，我们会得到通知。当作为条件表达式使用时，这样的流表达式会被隐式转换为一个布尔值。如果输入流对象处于错误状态，则返回 `false`。知道这一点后，我们可以重置流并执行适当的操作。

在这个示例中，我们没有自己编写这样的`if`条件，因为我们让`std::istream_iterator<city>`进行反序列化。这个迭代器类的`operator++`实现在解析时也会检查错误。如果发生任何错误，它将拒绝进一步迭代。在这种状态下，当它与结束迭代器进行比较时，它将返回`true`，这使得`copy`算法终止。这样，我们就安全了。

# 从 std::istream 迭代器填充容器

在上一个示例中，我们学会了如何从输入流中组装复合数据结构，然后用它们填充列表或向量。

这一次，我们通过标准输入填充一个`std::map`，使问题变得有点困难。这里的问题是，我们不能只是用值填充单个结构，然后将其推回线性容器，比如列表或向量，因为`map`将其有效负载分为键和值部分。然而，它并不完全不同，正如我们将看到的那样。

学习了这个示例之后，我们将会对从字符流中序列化和反序列化复杂的数据结构感到满意。

# 如何做...

我们将定义另一个类似上一个示例的结构，但这次我们将把它填充到一个地图中，这使得它变得更加复杂，因为这个容器从键到值的映射，而不仅仅是在列表中保存所有值：

1.  首先，我们包括所有需要的头文件，并声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <iomanip>
      #include <map>
      #include <iterator>
      #include <algorithm>
      #include <numeric>      

      using namespace std;
```

1.  我们想要维护一个小的互联网迷因数据库。假设一个迷因有一个名称、一个描述以及它诞生或发明的年份。我们将把它们保存在一个`std::map`中，其中名称是键，而其他信息则作为与键关联的值打包在一个结构中：

```cpp
      struct meme {
          string description;
          size_t year;
      };
```

1.  让我们首先忽略键，只为`struct meme`实现一个流`operator>>`函数重载。我们假设描述被引号包围，后面跟着年份。这在文本文件中看起来像`"一些描述" 2017`。由于描述被引号包围，它可以包含空格，因为我们知道引号之间的所有内容都属于它。通过使用`is >> quoted(m.description)`读取，引号会自动用作分隔符，并在之后被丢弃。这非常方便。就在那之后，我们读取年份数字：

```cpp
      istream& operator>>(istream &is, meme &m) {
          return is >> quoted(m.description) >> m.year;
      }
```

1.  好的，现在我们考虑将迷因的名称作为地图的键。为了将迷因插入地图，我们需要一个`std::pair<key_type, value_type>`实例。`key_type`当然是`string`，而`value_type`是`meme`。名称也允许包含空格，所以我们使用与描述相同的`quoted`包装。`p.first`是名称，`p.second`是与之关联的整个`meme`结构。它将被馈送到我们刚刚实现的另一个`operator>>`实现中：

```cpp
      istream& operator >>(istream &is, 
                           pair<string, meme> &p) {
          return is >> quoted(p.first) >> p.second;
      }
```

1.  好的，就是这样。让我们编写一个主函数，实例化一个地图，并填充该地图。因为我们重载了流函数`operator>>`，`istream_iterator`可以直接处理这种类型。我们让它从标准输入反序列化我们的迷因项目，并使用`inserter`迭代器将它们泵入地图中：

```cpp
      int main()
      {
          map<string, meme> m;

          copy(istream_iterator<pair<string, meme>>{cin},
               {},
               inserter(m, end(m)));
```

1.  在打印我们拥有的内容之前，让我们首先找出地图中*最长*的迷因名称是什么。我们使用`std::accumulate`来实现这一点。它得到一个初始值`0u`（`u`表示无符号），并将按元素访问地图，以便将它们*合并*在一起。在`accumulate`中，合并通常意味着*添加*。在我们的情况下，我们不想得到任何数值的*总和*，而是最大的字符串长度。为了实现这一点，我们提供了一个辅助函数`max_func`给`accumulate`，它接受当前最大尺寸变量（必须是`unsigned`，因为字符串长度是无符号的）并将其与当前项目的迷因名称字符串长度进行比较，以便取两个值中的最大值。这将对每个元素发生。`accumulate`函数的最终返回值是最大的迷因名称长度：

```cpp
          auto max_func ([](size_t old_max, 
                            const auto &b) {
              return max(old_max, b.first.length());
          });
          size_t width {accumulate(begin(m), end(m), 
                                   0u, max_func)};
```

1.  现在，让我们快速地循环遍历 map 并打印每个项。我们使用`<< left << setw(width)`来获得一个漂亮的类似表格的打印：

```cpp
          for (const auto &[meme_name, meme_desc] : m) {
              const auto &[desc, year] = meme_desc;

              cout << left << setw(width) << meme_name
                   << " : " << desc
                   << ", " << year << 'n';
          }
      }
```

1.  就是这样。我们需要一个小的互联网迷因数据库文件，所以让我们用一些示例填充一个文本文件：

```cpp
      "Doge" "Very Shiba Inu. so dog. much funny. wow." 2013
      "Pepe" "Anthropomorphic frog" 2016
      "Gabe" "Musical dog on maximum borkdrive" 2016
      "Honey Badger" "Crazy nastyass honey badger" 2011
      "Dramatic Chipmunk" "Chipmunk with a very dramatic look" 2007
```

1.  使用示例 meme 数据库编译和运行程序产生以下输出：

```cpp
      $ cat memes.txt | ./filling_containers 
      Doge              : Very Shiba Inu. so dog. much funny. wow., 2013
      Dramatic Chipmunk : Chipmunk with a very dramatic look, 2007
      Gabe              : Musical dog on maximum borkdrive, 2016
      Honey Badger      : Crazy nastyass honey badger, 2011
      Pepe              : Anthropomorphic frog, 2016
```

# 它是如何工作的...

在这个示例中有三个特殊之处。一个是我们没有从串行字符流中填充普通向量或列表，而是从`std::map`这样的更复杂的容器中填充。另一个是我们使用了那些神奇的`quoted`流操作器。最后一个是`accumulate`调用，它找出了最大的键字符串大小。

让我们从`map`部分开始。我们的`struct meme`只包含一个`description`字段和`year`。互联网迷因的名称不是这个结构的一部分，因为它被用作 map 的键。当我们向 map 中插入东西时，我们可以提供一个具有键类型和值类型的`std::pair`。这就是我们所做的。我们首先为`struct meme`实现了流`operator>>`，然后我们为`pair<string, meme>`做了同样的事情。然后我们使用`istream_iterator<**pair<string, meme>**>{cin}`从标准输入中获取这些项，并使用`inserter(m, end(m))`将它们插入 map 中。

当我们从流中反序列化 meme 项时，我们允许名称和描述包含空格。这是很容易实现的，尽管我们每个 meme 只使用一行，因为我们对这些字段进行了引用。一行格式的示例如下：`"Name with spaces" "Description with spaces" 123`

处理输入和输出中的带引号字符串时，`std::quoted`是一个很好的帮助。如果我们有一个字符串`s`，使用`cout << quoted(s)`来打印它会加上引号。如果我们通过流反序列化一个字符串，例如，通过`cin >> quoted(s)`，它将读取下一个引号，用后面的内容填充字符串，并继续直到看到下一个引号，无论涉及多少空格。

在我们的累积调用中，最后一个看起来奇怪的是`max_func`：

```cpp
auto max_func ([](size_t old_max, const auto &b) {
    return max(old_max, b.first.length());
});

size_t width {accumulate(begin(m), end(m), 0u, max_func)};
```

显然，`max_func`接受一个`size_t`参数和另一个`auto-`类型的参数，结果是来自 map 的`pair`项。这一开始看起来很奇怪，因为大多数二进制缩减函数接受相同类型的参数，然后使用某种操作将它们合并在一起，就像`std::plus`一样。在这种情况下，情况确实很不同，因为我们不是合并实际的`pair`项。我们只从每对中选择键字符串长度，*丢弃*其余部分，然后使用`max`函数减少结果的`size_t`值。

在累积调用中，`max_func`的第一个调用得到我们最初提供的`0u`值作为左参数，并得到右侧的第一个 pair 项的引用。这导致`max(0u, string_length)`的返回值，这是*下一个*调用的左参数，下一个 pair 项作为右参数，依此类推。

# 使用 std::ostream 迭代器进行通用打印

使用输出流打印任何东西都很容易，因为 STL 已经为大多数基本类型提供了许多有用的`operator<<`重载。这样，包含这些类型项的数据结构可以很容易地使用`std::ostream_iterator`类进行打印，这在本书中我们已经经常做过。

在这个示例中，我们将集中讨论如何使用自定义类型以及在调用方面不需要太多代码的情况下，我们可以通过模板类型选择来操纵打印。

# 如何做...

我们将通过启用与新自定义类的组合来玩`std::ostream_iterator`，并查看其隐式转换能力，这可以帮助我们进行打印：

1.  首先是包含文件，然后我们声明默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <vector>
      #include <iterator>
      #include <unordered_map>
      #include <algorithm>      

      using namespace std;
```

1.  让我们实现一个转换函数，将数字映射到字符串。它应该为值`1`返回`"one"`，为值`2`返回`"two"`，依此类推：

```cpp
      string word_num(int i) {
```

1.  我们用我们需要的映射填充哈希映射，以便以后访问它们：

```cpp
          unordered_map<int, string> m {
              {1, "one"}, {2, "two"}, {3, "three"},
              {4, "four"}, {5, "five"}, //...
          };
```

1.  现在，我们可以使用哈希映射的 `find` 函数来查找参数 `i`，并返回它找到的内容。如果找不到任何内容，因为给定数字没有翻译，我们将返回字符串 `"unknown"`：

```cpp
          const auto match (m.find(i));
          if (match == end(m)) { return "unknown"; }
          return match->second;
      };
```

1.  我们稍后将使用的另一件事是 `struct bork`。它只包含一个整数，并且也可以从整数*隐式*构造出来。它有一个 `print` 函数，接受一个输出流引用，并根据其成员整数 `borks` 的值重复打印 `"bork"` 字符串：

```cpp
      struct bork {
          int borks;

          bork(int i) : borks{i} {}

          void print(ostream& os) const {
              fill_n(ostream_iterator<string>{os, " "}, 
                     borks, "bork!"s);
          }
      };
```

1.  为了方便使用 `bork::print`，我们为流对象重载了 `operator<<`，因此每当 `bork` 对象被流到输出流中时，它们会自动调用 `bork::print`。

```cpp
      ostream& operator<<(ostream &os, const bork &b) {
          b.print(os);
          return os;
      }
```

1.  现在我们终于可以开始实现实际的主函数了。我们最初只是创建了一个带有一些示例值的向量：

```cpp
      int main()
      {
          const vector<int> v {1, 2, 3, 4, 5};
```

1.  `ostream_iterator` 类型的对象需要一个模板参数，该参数表示它们可以打印哪种类型的变量。如果我们写 `ostream_iterator<**T**>`，它将在打印时使用 `ostream& operator(ostream&, const **T**&)`。这正是我们之前为 `bork` 类型实现的。这一次，我们只是打印整数，所以是 `ostream_iterator<**int**>`。它将使用 `cout` 进行打印，因此我们将其作为构造函数参数提供。我们在循环中遍历我们的向量，并将每个项目 `i` 分配给解引用的输出迭代器。这也是 STL 算法使用流迭代器的方式：

```cpp
          ostream_iterator<int> oit {cout};

          for (int i : v) { *oit = i; }
          cout << 'n';
```

1.  我们刚刚生成的迭代器的输出是正常的，但它打印数字时没有任何分隔符。如果我们希望在所有打印的项目之间有一些分隔空格，我们可以将自定义的间隔字符串作为输出流迭代器构造函数的第二个参数提供。这样，它将打印 `"1, 2, 3, 4, 5, "` 而不是 `"12345"`。不幸的是，我们无法轻松地告诉它在最后一个数字之后删除逗号空格字符串，因为迭代器在到达最后一个数字之前不知道它的结束：

```cpp
          ostream_iterator<int> oit_comma {cout, ", "};

          for (int i : v) { *oit_comma = i; }
          cout << 'n';
```

1.  将项目分配给输出流迭代器以便打印它们并不是使用它的错误方式，但这不是它们被发明的目的。想法是将它们与算法结合使用。最简单的算法是 `std::copy`。我们可以将向量的开始和结束迭代器作为输入范围，将输出流迭代器作为输出迭代器。它将打印向量的所有数字。让我们用输出迭代器和之前编写的循环来比较一下：

```cpp
          copy(begin(v), end(v), oit);
          cout << 'n';

          copy(begin(v), end(v), oit_comma);
          cout << 'n';
```

1.  还记得函数 `word_num` 吗，它将数字映射到字符串，比如 `1` 对应 `"one"`，`2` 对应 `"two"`，依此类推？是的，我们也可以用它们来打印。我们只需要使用一个输出流操作符，它是针对 `string` 进行模板专门化的，因为我们不再打印整数。而且我们使用 `std::transform` 而不是 `std::copy`，因为它允许我们在将每个项目复制到输出范围之前对输入范围中的每个项目应用转换函数：

```cpp
          transform(begin(v), end(v), 
                    ostream_iterator<string>{cout, " "}, 
                    word_num);
          cout << 'n';
```

1.  程序中的最后一行最终使用了 `struct bork`。我们可以为 `std::transform` 提供一个转换函数，但我们没有这样做。相反，我们可以在 `std::copy` 调用中创建一个专门针对 `bork` 类型的输出流迭代器。这将导致从输入范围整数*隐式*创建 `bork` 实例。这将给我们一些有趣的输出：

```cpp
          copy(begin(v), end(v), 
               ostream_iterator<bork>{cout, "n"});
      }
```

1.  编译和运行程序会产生以下输出。前两行与接下来的两行完全相同，这是我们预料到的。然后，我们得到了漂亮的、写出来的数字字符串，然后是大量的 `bork!` 字符串。这些出现在多行中，因为我们使用了 `"n"` 分隔字符串而不是空格：

```cpp
      $ ./ostream_printing 
      12345
      1, 2, 3, 4, 5, 
      12345
      1, 2, 3, 4, 5, 
      one two three four five 
      bork! 
      bork! bork! 
      bork! bork! bork! 
      bork! bork! bork! bork! 
      bork! bork! bork! bork! bork! 
```

# 它是如何工作的...

我们已经看到`std::ostream_iterator`实际上只是一个*语法技巧*，它将打印的行为压缩成迭代器的形式和语法。递增这样的迭代器*没有任何作用*。对其进行解引用只会返回一个代理对象，其赋值运算符将其参数转发到输出流。

对于类型`T`（如`ostream_iterator<T>`）进行特化的输出流迭代器可以使用提供了`ostream& operator<<(ostream&, const T&)`实现的所有类型。

`ostream_iterator`总是尝试调用其模板参数指定的类型的`operator<<`，它将尝试隐式转换类型（如果允许）。当我们迭代`A`类型的项目范围，但将这些项目复制到`output_iterator<B>`实例时，如果`A`可以隐式转换为`B`，这将起作用。我们对`struct bork`也是完全相同的操作：`bork`实例可以从整数值隐式转换。这就是为什么很容易将大量`"bork!"`字符串抛到用户 shell 上。

如果隐式转换不可能，我们可以自己做，使用`std::transform`，这就是我们与`word_num`函数结合使用的方法。

请注意，通常*允许自定义类型进行隐式转换*是*不好的风格*，因为这是一个常见的*bug 来源*，后期很难找到。在我们的示例用例中，隐式构造函数比危险更有用，因为该类除了打印之外没有其他用途。

# 将输出重定向到特定代码段的文件

`std::cout`提供了一个非常好的方法，可以在任何时候打印我们想要的内容，因为它简单易用，易于扩展，并且全局可访问。即使我们想要打印特殊消息，比如错误消息，我们想要将其与普通消息隔离开来，我们可以使用`std::cerr`，它与`cout`相同，但是将内容打印到标准错误通道而不是标准输出通道。

有时我们可能对日志记录有更复杂的需求。例如，我们想要将函数的输出*重定向*到文件，或者我们想要*静音*函数的输出，而不改变函数本身。也许它是一个我们无法访问源代码的库函数。也许它从未被设计为写入文件，但我们希望将其输出到文件中。

确实可以重定向流对象的输出。在本教程中，我们将看到如何以非常简单和优雅的方式做到这一点。

# 如何做到...

我们将实现一个辅助类，解决重定向流和再次恢复重定向的问题，使用构造函数/析构函数的魔法。然后我们看看如何使用它：

1.  这次我们只需要输入、输出和文件流的头文件。并将`std`命名空间声明为查找的默认命名空间：

```cpp
      #include <iostream>
      #include <fstream>     

      using namespace std;
```

1.  我们实现了一个类，它包含一个文件流对象和一个指向流缓冲区的指针。作为流对象的`cout`有一个内部流缓冲区，我们可以简单地交换。在交换的同时，我们可以保存之前的内容，以便稍后可以*撤消*任何更改。我们可以在 C++参考中查找其类型，但我们也可以使用`decltype`来找出`cout.rdbuf()`返回的类型。这通常不是所有情况下的良好做法，但在这种情况下，它只是一个指针类型：

```cpp
      class redirect_cout_region
      {
          using buftype = decltype(cout.rdbuf());

          ofstream ofs;
          buftype  buf_backup;
```

1.  我们的类的构造函数接受一个文件名字符串作为其唯一参数。文件名用于初始化文件流成员`ofs`。初始化后，我们可以将其输入`cout`作为新的流缓冲区。接受新缓冲区的相同函数也返回旧缓冲区的指针，因此我们可以保存它以便稍后恢复它：

```cpp
      public:
          explicit 
          redirect_cout_region (const string &filename)
              : ofs{filename}, 
                buf_backup{cout.rdbuf(ofs.rdbuf())}
          {}
```

1.  默认构造函数与其他构造函数的作用相同。不同之处在于它不会打开任何文件。将默认构造的文件流缓冲区输入到`cout`流缓冲区会导致`cout`被*停用*。它只会*丢弃*我们给它的输入进行打印。在某些情况下，这也是有用的：

```cpp
          redirect_cout_region()
              : ofs{}, 
                buf_backup{cout.rdbuf(ofs.rdbuf())}
          {}
```

1.  析构函数只是恢复了我们的更改。当这个类的对象超出范围时，`cout`的流缓冲区再次变为旧的：

```cpp
          ~redirect_cout_region() { 
              cout.rdbuf(buf_backup); 
          }
      };
```

1.  让我们模拟一个*输出密集*的函数，这样我们以后可以玩耍：

```cpp
      void my_output_heavy_function()
      {
          cout << "some outputn";
          cout << "this function does really heavy workn";
          cout << "... and lots of it...n";
          // ...
      }
```

1.  在主函数中，我们首先产生一些完全正常的输出：

```cpp
      int main()
      {
          cout << "Readable from normal stdoutn";
```

1.  现在我们正在打开另一个作用域，这个作用域中的第一件事就是用文本文件参数实例化我们的新类。文件流默认以读写模式打开文件，因此它为我们创建了这个文件。任何后续的输出现在都将重定向到这个文件，尽管我们使用`cout`进行打印：

```cpp
          {
              redirect_cout_region _ {"output.txt"};
              cout << "Only visible in output.txtn";
              my_output_heavy_function();
          }
```

1.  离开作用域后，文件被关闭，输出重新重定向到正常的标准输出。现在让我们在另一个作用域中实例化相同的类，但是通过它的默认构造函数。这样，下面打印的文本行将不会在任何地方可见。它只会被丢弃：

```cpp
          {
              redirect_cout_region _;
              cout << "This output will "
                      "completely vanishn";
          }
```

1.  离开那个作用域后，我们的标准输出被恢复，最后一行文本输出将再次在 shell 中可读：

```cpp
          cout << "Readable from normal stdout againn";
      }
```

1.  编译和运行程序产生了我们预期的输出。在 shell 中只有第一行和最后一行输出可见：

```cpp
      $ ./log_regions 
      Readable from normal stdout
      Readable from normal stdout again
```

1.  我们可以看到，创建了一个名为`output.txt`的新文件，并包含了第一个作用域的输出。第二个作用域的输出完全消失了。

```cpp
      $ cat output.txt 
      Only visible in output.txt
      some output
      this function does really heavy work
      ... and lots of it...
```

# 工作原理...

每个流对象都有一个内部缓冲区，它充当前端。这些缓冲区是可交换的。如果我们有一个流对象`s`，想要将其缓冲区保存到变量`a`中，并安装一个新的缓冲区`b`，则如下所示：`a = s.rdbuf(b)`。恢复它可以简单地使用`s.rdbuf(a)`来完成。

这正是我们在这个示例中所做的。另一个很酷的事情是我们可以*堆叠*这些`redirect_cout_region`助手：

```cpp
{
    cout << "print to standard outputn";

    redirect_cout_region la {"a.txt"};
    cout << "print to a.txtn";

    redirect_cout_region lb {"b.txt"};
    cout << "print to b.txtn";
}
cout << "print to standard output againn";
```

这是因为对象的销毁顺序与它们的构造顺序相反。使用对象的构造和销毁之间的紧密耦合的模式的概念被称为**资源获取即初始化**（**RAII**）。

有一件非常重要的事情应该提到--`redirect_cout_region`类的成员变量的*初始化顺序*：

```cpp
class redirect_cout_region {
    using buftype = decltype(cout.rdbuf());

    ofstream ofs;
    buftype  buf_backup;

public:
    explicit 
    redirect_cout_region(const string &filename)
        : ofs{filename}, 
          buf_backup{cout.rdbuf(ofs.rdbuf())}
    {}

...
```

正如我们所看到的，成员`buf_backup`是从取决于`ofs`的表达式构造的。这显然意味着`ofs`需要在`buf_backup`之前初始化。有趣的是，这些成员初始化的顺序*并不*取决于初始化列表项的顺序。初始化顺序只取决于*成员声明*的顺序！

如果一个类成员变量需要在另一个成员变量之后初始化，它们在类成员声明中也*必须*按照这个顺序出现。它们在构造函数的初始化列表中出现的顺序并不重要。

# 通过继承自 std::char_traits 创建自定义字符串类

`std::string`非常有用。然而，一旦人们需要一个具有略有不同语义的字符串处理的字符串类，一些人就倾向于编写自己的字符串类。

编写自己的字符串类很少是一个好主意，因为安全的字符串处理很困难。幸运的是，`std::string`只是模板类`std::basic_string`的专门类型定义。这个类包含了所有复杂的内存处理内容，但它不会对字符串的复制、比较等施加任何策略。这是通过接受一个包含特性类的模板参数导入到`basic_string`中的。

在本教程中，我们将看到如何构建我们自己的特性类，以此方式创建自定义字符串而无需重新实现任何内容。

# 如何做...

我们将实现两种不同的自定义字符串类：`lc_string`和`ci_string`。第一个类从任何字符串输入构造小写字符串。另一个类不转换任何字符串，但可以进行不区分大小写的字符串比较：

1.  让我们首先包含一些必要的头文件，然后声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <algorithm>
      #include <string>      

      using namespace std;
```

1.  然后我们重新实现了`std::tolower`函数，它已经在`<cctype>`中定义。已经存在的函数很好，但它不是`constexpr`。自 C++17 以来，一些`string`函数是`constexpr`，我们希望能够利用我们自己的自定义字符串特性类。该函数将大写字符映射到小写字符，并保持其他字符不变：

```cpp
      static constexpr char tolow(char c) {
          switch (c) {
          case 'A'...'Z': return c - 'A' + 'a';
          default:        return c;
          }
      }
```

1.  `std::basic_string`类接受三个模板参数：基础字符类型、字符特性类和分配器类型。在本节中，我们只更改字符特性类，因为它定义了字符串的行为。为了仅重新实现与普通字符串不同的部分，我们公开继承标准特性类：

```cpp
      class lc_traits : public char_traits<char> {
      public:
```

1.  我们的类接受输入字符串但将它们转换为小写。有一个函数，它逐个字符地执行此操作，因此我们可以在这里放置我们自己的`tolow`函数。这个函数是`constexpr`的，这就是为什么我们重新实现了一个`constexpr`的`tolow`函数：

```cpp
          static constexpr 
          void assign(char_type& r, const char_type& a ) {
              r = tolow(a);
          }
```

1.  另一个函数处理整个字符串复制到自己的内存中。我们使用`std::transform`调用将所有字符从源字符串复制到内部目标字符串，并同时将每个字符映射到其小写版本：

```cpp
          static char_type* copy(char_type* dest, 
                                 const char_type* src, 
                                 size_t count) {
              transform(src, src + count, dest, tolow);
              return dest;
          }
      };
```

1.  另一个特性有助于构建一个有效地将字符串转换为小写的字符串类。我们将编写另一个特性，它保持实际的字符串有效负载不变，但在比较字符串时不区分大小写。我们再次从现有的标准字符特性类继承，并且这次，我们重新定义了一些其他成员函数：

```cpp
      class ci_traits : public char_traits<char> {
      public:
```

1.  `eq`函数告诉我们两个字符是否相等。我们也这样做，但是我们比较它们的小写版本。这样`'A'`等于`'a'`：

```cpp
          static constexpr bool eq(char_type a, char_type b) {
              return tolow(a) == tolow(b);
          }
```

1.  `lt`函数告诉我们`a`的值是否小于`b`的值。我们在将两个字符再次转换为小写后，应用正确的逻辑运算符：

```cpp
          static constexpr bool lt(char_type a, char_type b) {
              return tolow(a) < tolow(b);
          }
```

1.  最后两个函数处理逐个字符的输入，接下来的两个函数处理逐个字符串的输入。`compare`函数类似于老式的`strncmp`函数。如果两个字符串在`count`定义的长度内相等，则返回`0`。如果它们不同，则返回一个负数或正数，告诉哪个输入字符串在词典顺序上更小。当然，必须在它们的小写版本上计算每个位置的字符之间的差异。好处是自 C++14 以来，这整个循环代码一直是`constexpr`函数的一部分：

```cpp
          static constexpr int compare(const char_type* s1,
                                       const char_type* s2,
                                       size_t count) {
              for (; count; ++s1, ++s2, --count) {
                  const char_type diff (tolow(*s1) - tolow(*s2));
                  if      (diff < 0) { return -1; }
                  else if (diff > 0) { return +1; }
              }
              return 0;
          }
```

1.  我们需要为我们的不区分大小写的字符串类实现的最后一个函数是`find`。对于给定的输入字符串`p`和长度`count`，它找到字符`ch`的位置。然后，它返回指向该字符的第一个出现的指针，如果没有，则返回`nullptr`。该函数中的比较必须使用`tolow`“眼镜”来进行，以使搜索不区分大小写。不幸的是，我们不能使用`std::find_if`，因为它不是`constexpr`，必须自己编写一个循环：

```cpp
          static constexpr 
          const char_type* find(const char_type* p,
                                size_t count,
                                const char_type& ch) {
              const char_type find_c {tolow(ch)};

              for (; count != 0; --count, ++p) {
                  if (find_c == tolow(*p)) { return p; }
              }

              return nullptr;
          }
      };
```

1.  好的，特性就是这些。既然我们现在已经有了它们，我们可以定义两种新的字符串类类型。`lc_string`表示*小写字符串*。`ci_string`表示*不区分大小写的字符串*。这两个类与`std::string`唯一的区别在于它们的字符特性类：

```cpp
      using lc_string = basic_string<char, lc_traits>;
      using ci_string = basic_string<char, ci_traits>;
```

1.  为了使输出流接受这些新的类进行打印，我们需要快速重载流`operator<<`：

```cpp
      ostream& operator<<(ostream& os, const lc_string& str) {
          return os.write(str.data(), str.size());
      }

      ostream& operator<<(ostream& os, const ci_string& str) {
          return os.write(str.data(), str.size());
      }
```

1.  现在我们终于可以开始实现实际的程序了。让我们实例化一个普通字符串、一个小写字符串和一个不区分大小写的字符串，并立即打印它们。它们在终端上应该都看起来正常，但小写字符串应该都是小写的：

```cpp
      int main()
      {
          cout << "   string: " 
               << string{"Foo Bar Baz"} << 'n'
               << "lc_string: " 
               << lc_string{"Foo Bar Baz"} << 'n'
               << "ci_string: "
               << ci_string{"Foo Bar Baz"} << 'n';
```

1.  为了测试不区分大小写的字符串，我们可以实例化两个基本相等但在某些字符的大小写上有所不同的字符串。当进行真正的不区分大小写比较时，它们应该看起来是相等的：

```cpp
          ci_string user_input {"MaGiC PaSsWoRd!"};
          ci_string password   {"magic password!"};
```

1.  因此，让我们比较它们，并打印出它们是否匹配：

```cpp
          if (user_input == password) {
              cout << "Passwords match: "" << user_input
                   << "" == "" << password << ""n";
          }
      }
```

1.  编译和运行程序会产生我们预期的结果。当我们首先以不同类型三次打印相同的字符串时，我们得到了不变的结果，但`lc_string`实例全部是小写的。只有在字符大小写不同的两个字符串的比较确实成功，并产生了正确的输出：

```cpp
      $ ./custom_string 
         string: Foo Bar Baz
      lc_string: foo bar baz
      ci_string: Foo Bar Baz
      Passwords match: "MaGiC PaSsWoRd!" == "magic password!"
```

# 它是如何工作的...

我们所做的所有子类化和函数重新实现对于初学者来说肯定看起来有点疯狂。我们从哪里得到所有这些函数签名，我们*神奇地*知道我们需要重新实现？

让我们首先看看`std::string`真正来自哪里：

```cpp
template <
    class CharT, 
    class Traits    = std::char_traits<CharT>, 
    class Allocator = std::allocator<CharT>
    > 
class basic_string;
```

`std::string`实际上是一个`std::basic_string<char>`，它扩展为`std::basic_string<char, std::char_traits<char>, std::allocator<char>>`。好吧，这是一个很长的类型描述，但是它是什么意思呢？所有这一切的重点是，可以基于单字节`char`项以及其他更大的类型来构建字符串。这使得可以处理更多的字符集，而不仅仅是典型的美国 ASCII 字符集。这不是我们现在要研究的东西。

然而，`char_traits<char>`类包含了`basic_string`在其操作中需要的算法。它知道如何比较、查找和复制字符和字符串。

`allocator<char>`类也是一个特性类，但它的特殊工作是处理字符串的分配和释放。这对我们来说现在并不重要，因为默认行为满足我们的需求。

如果我们希望字符串类的行为有所不同，我们可以尝试尽可能多地重用`basic_string`和`char_traits`已经提供的内容。这就是我们所做的。我们实现了两个`char_traits`子类，分别称为`case_insentitive`和`lower_caser`，并通过将它们用作标准`char_traits`类型的替代品，配置了两种全新的字符串类型。

为了探索其他可能性，以适应`basic_string`到您自己的需求，查阅 C++ STL 文档中的`std::char_traits`，看看它还有哪些其他函数可以重新实现。

# 使用正则表达式库对输入进行标记化

在复杂的字符串解析或转换以及将其分成块时，*正则表达式*是一个很好的帮助。在许多编程语言中，它们已经内置，因为它们非常有用和方便。

如果您还不了解正则表达式，请查看关于它们的*维基百科*文章，例如。当解析任何类型的文本时，它们肯定会扩展您的视野，因为很容易看到它们的有用性。例如，正则表达式可以测试电子邮件地址字符串或 IP 地址字符串是否有效，找到并提取符合复杂模式的大字符串中的子字符串等等。

在这个示例中，我们将从 HTML 文件中提取所有链接并列出给用户。代码将非常简短，因为自 C++11 以来，我们在 C++ STL 中内置了正则表达式支持。

# 如何做...

我们将定义一个检测链接的正则表达式，并将其应用于 HTML 文件，以便漂亮地打印出该文件中出现的所有链接：

1.  让我们首先包括所有必要的头文件，并声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <iterator>
      #include <regex>
      #include <algorithm>
      #include <iomanip>      

      using namespace std;
```

1.  稍后我们将生成一个可迭代范围，其中包含字符串。这些字符串总是成对出现，一个是链接，一个是链接描述。因此，让我们编写一个小帮助函数，漂亮地打印这些：

```cpp
      template <typename InputIt>
      void print(InputIt it, InputIt end_it)
      {
          while (it != end_it) {
```

1.  在每个循环步骤中，我们将迭代器递增两次，并复制链接和链接描述。在两个迭代器解引用之间，我们添加了另一个保护`if`分支，检查我们是否过早地到达了可迭代范围的末尾，只是为了安全起见：

```cpp
              const string link {*it++};
              if (it == end_it) { break; }
              const string desc {*it++};
```

1.  现在，让我们以一个漂亮的格式打印链接及其描述，就这样：

```cpp
              cout << left << setw(28) << desc 
                   << " : " << link << 'n';
          }
      }
```

1.  在主函数中，我们正在读取来自标准输入的所有内容。为了做到这一点，我们通过输入流迭代器从整个标准输入构造一个字符串。为了防止标记化，因为我们希望整个用户输入保持原样，我们使用`noskipws`。这个修饰符取消了空格跳过和标记化：

```cpp
      int main()
      {
          cin >> noskipws;
          const std::string in {istream_iterator<char>{cin}, {}};
```

1.  现在我们需要定义一个正则表达式，描述我们如何假设 HTML 链接的外观。正则表达式中的括号`()`定义了组。这些是我们想要访问的链接的部分--它链接到的 URL 及其描述：

```cpp
          const regex link_re {
              "<a href="([^"]*)"[^<]*>([^<]*)</a>"};
```

1.  `sregex_token_iterator`类与`istream_iterator`具有相同的外观和感觉。我们将整个字符串作为可迭代输入范围，并使用刚刚定义的正则表达式。还有第三个参数`{1, 2}`，它是一个整数值的初始化列表。它定义了我们要迭代表达式捕获的组 1 和 2：

```cpp
          sregex_token_iterator it {
              begin(in), end(in), link_re, {1, 2}};
```

1.  现在我们有一个迭代器，如果找到任何内容，它将发出链接和链接描述。我们将它与相同类型的默认构造的迭代器一起提供给我们之前实现的`print`函数：

```cpp
          print(it, {});
      }
```

1.  编译和运行程序后，我们得到以下输出。我在 ISO C++主页上运行了`curl`程序，它只是从互联网上下载了一个 HTML 页面。当然，也可以写`cat some_html_file.html | ./link_extraction`。我们使用的正则表达式基本上是固定的，假设了 HTML 文档中链接的外观。你可以尝试使它更通用：

```cpp
      $ curl -s "https://isocpp.org/blog" | ./link_extraction 
      Sign In / Suggest an Article : https://isocpp.org/member/login
      Register                     : https://isocpp.org/member/register
      Get Started!                 : https://isocpp.org/get-started
      Tour                         : https://isocpp.org/tour
      C++ Super-FAQ                : https://isocpp.org/faq
      Blog                         : https://isocpp.org/blog
      Forums                       : https://isocpp.org/forums
      Standardization              : https://isocpp.org/std
      About                        : https://isocpp.org/about
      Current ISO C++ status       : https://isocpp.org/std/status
      (...and many more...)
```

# 它的工作原理...

正则表达式（或简称*regex*）非常有用。它们可能看起来很神秘，但值得学习它们的工作原理。如果我们手动进行匹配，一个简短的正则表达式就可以节省我们编写许多行代码。

在这个示例中，我们首先实例化了一个 regex 类型的对象。我们将其构造函数与描述正则表达式的字符串一起使用。一个非常简单的正则表达式是`"."`，它匹配*每个*字符，因为点是正则表达式通配符。如果我们写`"a"`，那么这只匹配`'a'`字符。如果我们写`"ab*"`，那么这意味着"一个`a`，以及零个或任意多个`b`字符"。等等。正则表达式是另一个大的主题，在维基百科和其他网站或文献上有很好的解释。

让我们再看看我们假设是 HTML 链接的正则表达式。一个简单的 HTML 链接可能看起来像`<a href="some_url.com/foo">A great link</a>`。我们想要`some_url.com/foo`部分，以及`A great link`。因此，我们想出了以下正则表达式，其中包含用于匹配子字符串的*组*：

![](img/f6d75901-b27a-455d-bbb6-118f376bef15.png)

整个匹配本身始终是**Group 0**。在这种情况下，这是完整的`<a href ..... </a>`字符串。包含链接到的 URL 的引用`href`部分是**Group 1**。正则表达式中的`( )`括号定义了这样一个。另一个是在`<a ...>`和`</a>`之间的部分，其中包含链接描述。

有各种 STL 函数接受正则表达式对象，但我们直接使用了正则表达式令牌迭代器适配器，这是一个高级抽象，它在底层使用`std::regex_search`来自动化重复匹配工作。我们像这样实例化它：

```cpp
sregex_token_iterator it {begin(in), end(in), link_re, {1, 2}};
```

开始和结束部分表示我们的输入字符串，正则表达式令牌迭代器将在其上迭代并匹配所有链接。当然，这是我们实现的复杂正则表达式，用于匹配链接。`{1, 2}`部分是下一个看起来复杂的东西。它指示令牌迭代器在每次完全匹配时停止，并首先产生第 1 组，然后在递增迭代器后产生第 2 组，再次递增后，最终在字符串中搜索下一个匹配项。这种智能行为确实为我们节省了一些代码行。

让我们看另一个例子，确保我们理解了这个概念。假设正则表达式是`"a(b*)(c*)"`。它将匹配包含`a`字符的字符串，然后是零个或任意多个`b`字符，然后是零个或任意多个`c`字符：

```cpp
const string s {" abc abbccc "};
const regex re {"a(b*)(c*)"};

sregex_token_iterator it {begin(s), end(s), re, {1, 2}};

print( *it ); // prints b
++it;
print( *it ); // prints c
++it;
print( *it ); // prints bb
++it;
print( *it ); // prints ccc
```

还有`std::regex_iterator`类，它发出*在*正则表达式匹配之间的子字符串。

# 舒适地根据上下文动态地以不同方式打印数字

在上一个示例中，我们学会了如何使用输出流格式化输出。在做同样的事情时，我们意识到了两个事实：

+   大多数 I/O 操纵器是*粘性*的，因此我们必须在使用后恢复它们的效果，以免干扰其他不相关的代码，也会打印

+   如果我们不得不设置长链的 I/O 操纵器才能以特定格式打印出少量变量，这将非常乏味，看起来也不太可读。

很多人不喜欢 I/O 流，甚至在 C++中，他们仍然使用`printf`来格式化他们的字符串。

在这个示例中，我们将看到如何在代码中减少 I/O 操纵器的噪音，动态地格式化类型。

# 如何做...

我们将实现一个名为`format_guard`的类，它可以自动恢复任何格式设置。此外，我们添加了一个包装类型，它可以包含任何值，但在打印时，它以特殊格式显示，而不会给我们带来 I/O 操纵器的噪音：

1.  首先，我们包含一些头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <iomanip>      

      using namespace std;
```

1.  帮助类`format_guard`会为我们整理流格式设置。它的构造函数保存了`std::cout`在那一刻设置的格式标志。它的析构函数将它们恢复到构造函数调用时的状态。这实际上撤销了在之间应用的任何格式设置：

```cpp
      class format_guard {
          decltype(cout.flags()) f {cout.flags()};

      public:
          ~format_guard() { cout.flags(f); }
      };
```

1.  另一个小帮助类是`scientific_type`。因为它是一个类模板，它可以将任何有效载荷类型包装为成员变量。它基本上什么也不做：

```cpp
      template <typename T>
      struct scientific_type {
          T value;

          explicit scientific_type(T val) : value{val} {}
      };
```

1.  我们可以为任何类型定义完全自定义的格式设置，这些类型在之前被包装成`scientific_type`，因为如果我们为其重载流`operator>>`，那么当打印这些类型时，流库会执行完全不同的代码。这样，我们可以以科学浮点表示法打印科学值，使用大写格式和显式的`+`前缀（如果它们具有正值）。我们还使用我们的`format_guard`类来在离开此函数时整理所有设置：

```cpp
      template <typename T>
      ostream& operator<<(ostream &os, const scientific_type<T> &w) {
          format_guard _;
          os << scientific << uppercase << showpos;
          return os << w.value;
      }
```

1.  在主函数中，我们首先尝试使用`format_guard`类。我们打开一个新的作用域，首先获得该类的一个实例，然后我们对`std::cout`应用一些疯狂的格式标志：

```cpp
      int main()
      {
          {
              format_guard _;
              cout << hex << scientific << showbase << uppercase;

              cout << "Numbers with special formatting:n";
              cout << 0x123abc << 'n';
              cout << 0.123456789 << 'n';
          }
```

1.  在启用了许多格式标志的情况下打印了一些数字后，我们再次离开了作用域。在此期间，`format_guard`的析构函数整理了格式。为了测试这一点，我们*再次*打印完全相同的数字。它们应该看起来不同：

```cpp
          cout << "Same numbers, but normal formatting again:n";
          cout << 0x123abc << 'n';
          cout << 0.123456789 << 'n';
```

1.  现在我们要使用`scientific_type`。让我们依次打印三个浮点数。我们将第二个数字包装在`scientific_type`中。这样，它将以我们特殊的科学样式打印，但它之前和之后的数字将采用默认格式。同时，我们避免了丑陋的格式化行*噪音*：

```cpp
          cout << "Mixed formatting: "
               << 123.0 << " "
               << scientific_type{123.0} << " "
               << 123.456 << 'n';
      }
```

1.  编译和运行程序会产生以下结果。前两个数字以特定格式打印。接下来的两个数字以默认格式显示，这表明我们的`format_guard`工作得很好。最后一行的三个数字看起来也正如预期的那样。只有中间的数字具有`scientific_type`的格式，其余的都是默认格式：

```cpp
      $ ./pretty_print_on_the_fly 
      Numbers with special formatting:
      0X123ABC
      1.234568E-01
      Same numbers, but normal formatting again:
      1194684
      0.123457
      Mixed formatting: 123 +1.230000E+02 123.456
```

# 捕获 std::iostream 错误的可读异常

在本章的*任何*食谱中，我们都没有使用*异常*来捕获错误。虽然这是可能的，但在没有异常的情况下使用流对象已经非常方便。如果我们尝试解析 10 个值，但在中间某个地方失败了，整个流对象就会将自己设置为失败状态并停止进一步解析。这样，我们就不会遇到从流中错误的偏移解析变量的危险。我们可以在条件语句中进行解析，比如`if (cin >> foo >> bar >> ...)`。如果失败了，我们就处理它。在`try { ... } catch ...`块中进行解析似乎并不是很有利。

事实上，在 C++中引入异常之前，C++ I/O 流库已经存在。异常支持是后来添加的，这可能解释了为什么它们不是流库中的一流支持特性。

为了在流库中使用异常，我们必须单独配置每个流对象，以便在将自身设置为失败状态时抛出异常。不幸的是，异常对象中的错误解释并没有得到彻底的标准化。正如我们将在本节中看到的那样，这导致了不太有用的错误消息。如果我们真的想要在流对象中使用异常，我们可以*另外*轮询 C 库以获取文件系统错误状态以获得一些额外信息。

在本节中，我们将编写一个可以以不同方式失败的程序，使用异常处理这些失败，并看看如何在之后从中挤取更多信息。

# 如何做...

我们将实现一个程序，打开一个文件（可能失败），然后我们将从中读取一个整数（这也可能失败）。我们在激活异常的情况下进行这些操作，然后看看我们如何处理这些异常：

1.  首先，我们包含一些头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <fstream>
      #include <system_error>
      #include <cstring>      

      using namespace std;
```

1.  如果我们想要在流对象中使用异常，我们必须首先启用它们。为了使文件流对象在访问的文件不存在或存在解析错误时抛出异常，我们需要在异常掩码中设置一些失败位。如果我们之后做了一些失败的事情，它将触发异常。通过激活`failbit`和`badbit`，我们为文件系统错误和解析错误启用了异常：

```cpp
      int main()
      {
          ifstream f;
          f.exceptions(f.failbit | f.badbit);
```

1.  现在我们可以打开一个`try`块并访问一个文件。如果打开文件成功，我们尝试从中读取一个整数。只有在两个步骤都成功的情况下，我们才打印整数：

```cpp
          try {
              f.open("non_existant.txt");

              int i;
              f >> i;

              cout << "integer has value: " << i << 'n';
          }
```

1.  在两种预期的错误可能性中，都会抛出`std::ios_base::failure`的实例。这个对象有一个`what()`成员函数，应该解释触发异常的原因。不幸的是，这条消息的标准化被省略了，它并没有提供太多信息。然而，我们至少可以区分是否存在*文件系统*问题（例如文件不存在）或格式*解析*问题。全局变量`errno`甚至在 C++发明之前就存在了，并且被设置为一个错误值，我们现在可以检查。`strerror`函数将错误号转换为可读的字符串。如果`errno`为`0`，那么至少没有文件系统错误：

```cpp
          catch (ios_base::failure& e) {
              cerr << "Caught error: ";
              if (errno) {
                  cerr << strerror(errno) << 'n';
              } else {
                  cerr << e.what() << 'n';
              }
          }
      }
```

1.  编译程序并在两种不同的情况下运行它会产生以下输出。如果要打开的文件存在，但无法从中解析出整数，则会得到一个`iostream_category`错误消息：

```cpp
      $ ./readable_error_msg 
      Caught error: ios_base::clear: unspecified iostream_category error
```

1.  如果文件*不存在*，我们将收到一个不同于`strerror(errno)`的消息通知我们：

```cpp
      $ ./readable_error_msg
      Caught error: No such file or directory
```

# 它是如何工作的...

我们已经看到，我们可以通过`s.exceptions(s.failbit | s.badbit)`为流对象`s`启用异常。这意味着，例如，如果我们想在打开文件时无法打开文件时得到异常，就无法使用`std::ifstream`实例的构造函数：

```cpp
ifstream f {"non_existant.txt"};
f.exceptions(...); // too late for an exception

```

这是一个遗憾，因为异常实际上承诺它们使错误处理变得不那么笨拙，与老式的 C 风格代码相比，后者充斥着大量的`if`分支，处理每一步之后的错误。

如果我们尝试引发流失败的各种原因，我们会意识到没有抛出不同的异常。这样，我们只能找出*何时*发生错误，而不是*什么*具体的错误（当然，这对于*一般*的异常处理来说是*不*正确的，但对于 STL 流库来说是正确的）。这就是为什么我们另外查看了`errno`的值。这个全局变量是一个古老的构造，在旧日当没有 C++或一般的异常时就已经被使用。

如果任何与系统相关的函数遇到错误条件，它可以将`errno`变量设置为非`0`的值（`0`表示没有错误），然后调用者可以读取该错误号并查找其值的含义。唯一的问题是，当我们有一个多线程应用程序，并且所有线程都使用可以设置此错误变量的函数时，*它是谁*的错误值？如果我们即使没有错误也读取它，它可能携带一个错误值，因为在*不同线程*中运行的*其他*系统函数可能已经遇到了错误。幸运的是，自 C++11 以来，这个缺陷已经消失，进程中的每个线程都可以看到自己的`errno`变量。

不详细阐述古老的错误指示方法的利弊，当异常在基于系统的事物上触发时，它可以给我们提供有用的额外信息。异常告诉我们*何时*发生了，而`errno`可以告诉我们*发生了什么*，如果它是在系统级别发生的。
