# STL 算法的高级用法

本章将涵盖以下食谱：

+   使用 STL 算法实现 trie 类

+   使用 trie 实现搜索输入建议生成器

+   使用 STL 数值算法实现傅里叶变换公式

+   计算两个向量的误差和

+   使用 STL 算法实现 ASCII Mandelbrot 渲染器

+   构建我们自己的算法 - split

+   从标准算法中组合有用的算法 - gather

+   删除单词之间的连续空格

+   压缩和解压字符串

# 介绍

在上一章中，我们访问了基本的 STL 算法，并使用它们执行了简单的任务，以便对典型的 STL 接口有所了解：大多数 STL 算法接受形式为迭代器对的一个或多个范围作为输入/输出参数。它们通常还接受谓词函数、自定义比较函数或转换函数。最后，它们大多再次返回迭代器，因为这些迭代器通常可以随后被输入到其他一些算法中。

虽然 STL 算法旨在尽可能简化，但它们的接口也尽可能通用。这样可以最大程度地实现代码重用，但并不总是看起来太美观。一个有经验的 C++程序员，如果尽可能使用 STL 算法来表达尽可能多的想法，阅读其他人的代码会更轻松。这导致了程序员和读者之间理解的最大化共同基础。程序员的大脑可以更快地解析一个众所周知的算法名称，而不是理解一个复杂的循环，它主要做着类似的事情，但在某些细节上略有不同。

到目前为止，我们使用 STL 数据结构如此直观，以至于我们可以很好地避免指针、原始数组和其他粗糙的遗留结构。下一步是将我们对 STL 算法的理解提升到可以避免使用手工编写的循环控制结构复杂性的水平，而是用众所周知的 STL 算法来表达它们。通常，这是一个真正的改进，因为代码变得更简洁、更易读，同时更通用和数据结构无关。实际上，几乎总是可以避免编写手工循环，并将算法从`std`命名空间中取出，但有时，这确实会导致*笨拙的代码*。我们不打算区分什么是笨拙的，什么不是；我们只会探索可能性。

在本章中，我们将以创造性的方式使用 STL 算法，以寻找新的视角，并看看如何用现代 C++实现事物。在这个过程中，我们将实现我们自己的类似 STL 的算法，这些算法可以轻松地与现有数据结构和以相同方式设计的其他算法结合使用。我们还将*组合*现有的 STL 算法，以获得*新*的算法，这些算法以前不存在。这样组合的算法允许在现有算法的基础上实现更复杂的算法，同时它们本身以这种方式极其简短和易读。在这个小旅程中，我们还将看到 STL 算法在可重用性或美观性方面的具体问题。只有当我们充分了解所有的方式时，我们才能最好地决定哪种方式是正确的。

# 使用 STL 算法实现 trie 类

所谓的**trie**数据结构提出了一种将数据以易于搜索的方式存储的有趣方法。在将文本句子分割成单词列表时，通常可以将一些句子共有的前几个单词合并起来。

让我们看一下下面的图表，在这里，句子`"hi how are you"`和`"hi how do you do"`被保存在类似树状的数据结构中。它们共有的第一个单词是`"hi how"`，然后它们不同并分裂成树状：

![](img/b42b8317-e512-4bac-bc05-cb4ce62421b7.png)

因为 trie 数据结构结合了常见的前缀，所以它也被称为*前缀树*。使用 STL 已经给我们的东西实现这样的数据结构非常容易。本节集中在实现我们自己的 trie 类。

# 如何做...

在本节中，我们将实现我们自己的前缀树，只使用 STL 数据结构和算法。

1.  我们将包括我们使用的 STL 部分的所有头文件，并声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <optional>
      #include <algorithm>
      #include <functional>
      #include <iterator>
      #include <map>
      #include <vector>
      #include <string>

      using namespace std;
```

1.  整个程序围绕着一个 trie，我们首先必须实现一个类。在我们的实现中，trie 基本上是一个递归的映射。每个 trie 节点包含一个映射，它将有效载荷类型`T`的实例映射到下一个 trie 节点：

```cpp
      template <typename T>
      class trie
      {
          map<T, trie> tries;
```

1.  插入新项目序列的代码很简单。用户提供一个 begin/end 迭代器对，我们通过递归循环遍历它。如果用户输入序列是`{1, 2, 3}`，那么我们在子 trie 中查找`1`，然后在下一个子 trie 中查找`2`，以获取`3`的子 trie。如果这些子 trie 中的任何一个以前不存在，它们将被`std::map`的`[]`运算符隐式添加：

```cpp
      public:
          template <typename It>
          void insert(It it, It end_it) {
              if (it == end_it) { return; }
              tries[*it].insert(next(it), end_it);
          }
```

1.  我们还定义了方便的函数，使用户只需提供一个项目容器，然后自动查询迭代器：

```cpp
          template <typename C>
          void insert(const C &container) {
              insert(begin(container), end(container));
          }
```

1.  为了允许用户编写`my_trie.insert({"a", "b", "c"});`，我们必须帮助编译器正确推断出那一行的所有类型，所以我们添加了一个函数，它重载了带有`initializer_list`参数的插入接口：

```cpp
          void insert(const initializer_list<T> &il) {
              insert(begin(il), end(il));
          }
```

1.  我们还想看看 trie 中有什么，所以我们需要一个`print`函数。为了打印，我们可以通过 trie 进行深度优先搜索。从根节点到第一个叶子的路上，我们记录我们已经看到的所有有效载荷项目。这样，一旦到达叶子，我们就有了一个完整的序列，这是微不足道的可打印的。当`tries.empty()`为`true`时，我们看到我们到达了一个叶子。递归`print`调用之后，我们再次弹出最后添加的有效载荷项目：

```cpp
          void print(vector<T> &v) const {
              if (tries.empty()) {
                  copy(begin(v), end(v), 
                       ostream_iterator<T>{cout, " "});
                  cout << 'n';
              }
              for (const auto &p : tries) {
                  v.push_back(p.first);
                  p.second.print(v);
                  v.pop_back();
              }
          }
```

1.  递归的`print`函数传递一个可打印的有效载荷项目列表的引用，但用户应该在没有任何参数的情况下调用它。因此，我们定义了一个无参数的`print`函数，它构造了辅助列表对象：

```cpp
          void print() const {
              vector<T> v;
              print(v);
          }
```

1.  现在我们可以构建和打印 tries，我们可能想要搜索子 tries。这个想法是，如果 trie 包含序列，比如`{a, b, c}`和`{a, b, d, e}`，并且我们给它一个序列`{a, b}`进行搜索，它会返回包含`{c}`和`{d, e}`部分的子 trie。如果我们找到了子 trie，我们会返回一个`const`引用。有可能 trie 中没有这样的子 trie，如果 trie 不包含我们要搜索的序列。在这种情况下，我们仍然需要返回*something*。`std::optional`是一个很好的帮助类，因为如果没有匹配，我们可以返回一个*empty* optional 对象：

```cpp
          template <typename It>
          optional<reference_wrapper<const trie>> 
          subtrie(It it, It end_it) const {
              if (it == end_it) { return ref(*this); }
              auto found (tries.find(*it));
              if (found == end(tries)) { return {}; }

              return found->second.subtrie(next(it), end_it);
          }
```

1.  与`insert`方法类似，我们提供了`subtrie`方法的单参数版本，它会自动从输入容器中获取迭代器：

```cpp
          template <typename C>
          auto subtrie(const C &c) { 
              return subtrie(begin(c), end(c));
          }
      };
```

1.  就是这样。让我们在主函数中使用新的 trie 类，通过实例化一个专门用于`std::string`对象的 trie，并填充一些示例内容：

```cpp
      int main()
      {
          trie<string> t;

          t.insert({"hi", "how", "are", "you"});
          t.insert({"hi", "i", "am", "great", "thanks"});
          t.insert({"what", "are", "you", "doing"});
          t.insert({"i", "am", "watching", "a", "movie"});
```

1.  让我们先打印整个 trie：

```cpp
          cout << "recorded sentences:n";
          t.print();
```

1.  然后我们获得了所有以`"hi"`开头的输入句子的子 trie，并打印它：

```cpp
          cout << "npossible suggestions after "hi":n";

          if (auto st (t.subtrie(initializer_list<string>{"hi"})); 
              st) {
              st->get().print();
          }
      }
```

1.  编译和运行程序显示，当我们查询 trie 时，它确实只返回以`"hi"`开头的两个句子：

```cpp
      $ ./trie 
      recorded sentences:
      hi how are you 
      hi i am great thanks 
      i am watching a movie 
      what are you doing 

      possible suggestions after "hi":
      how are you 
      i am great thanks 
```

# 它是如何工作的...

有趣的是，单词序列*插入*的代码比在子 trie 中查找给定单词序列的代码更短更简单。所以，让我们先看一下插入代码：

```cpp
template <typename It>
void trie::insert(It it, It end_it) {
    if (it == end_it) { return; }
    tries[*it].insert(next(it), end_it);
}
```

迭代器对`it`和`end_it`表示要插入的单词序列。`tries[*it]`元素在子 trie 中查找序列中的第一个单词，然后，`.insert(next(it), end_it)`重新启动相同的函数，该函数在较低的子 trie 上，迭代器向前移动一个单词。`if (it == end_it) { return; }`行只是中止递归。空的`return`语句什么都不做，这一点起初有点奇怪。所有的插入都发生在`tries[*it]`语句中。`std::map`的方括号操作符`[]`要么返回给定键的现有项，要么使用该键*创建*一个项。关联值（映射类型是本食谱中的 trie）是从其默认构造函数构造的。这样，每当我们查找未知单词时，我们都会*隐式创建*一个新的 trie 分支。

在子 trie 中查找看起来更复杂，因为我们无法在隐式代码中*隐藏*太多内容：

```cpp
template <typename It>
optional<reference_wrapper<const trie>> 
subtrie(It it, It end_it) const {
    if (it == end_it) { return ref(*this); }
    auto found (tries.find(*it));
    if (found == end(tries)) { return {}; }

    return found->second.subtrie(next(it), end_it);
}
```

这段代码基本上围绕着`auto found (tries.find(*it));`语句。我们使用`find`来查找下一个更深的 trie 节点，而不是使用方括号操作符(`[]`)。如果我们在查找时使用`[]`操作符，trie 将为我们*创建*缺失的项，这*不*是我们在查找项是否存在时想要的！（顺便说一句，试着这样做。类方法是`const`，所以这甚至不可能。这可以挽救生命，帮助我们防止错误。）

另一个看起来可怕的细节是返回类型，`optional<reference_wrapper<const trie>>`。我们选择`std::optional`作为包装器，因为可能没有这样的子 trie 与我们正在寻找的输入序列相匹配。如果我们只插入了`"hello my friend"`，那么就不会有`"goodbye my friend"`序列可以查找。在这种情况下，我们只返回`{}`，这会给调用者一个空的 optional 对象。这仍然不能解释为什么我们使用`reference_wrapper`而不只是写`optional<const trie &>`。这里的重点是，具有`trie&`类型成员变量的可选实例是不可重新分配的，因此不会编译。使用`reference_wrapper`实现引用会导致可重新分配的对象。

# 使用 trie 实现搜索输入建议生成器

在互联网搜索引擎中输入内容时，界面通常会尝试猜测完整的搜索查询是什么样子。这种猜测通常基于过去的热门搜索查询。有时，这样的搜索引擎猜测很有趣，因为人们似乎在搜索引擎中输入了奇怪的查询。

![](img/ca1d8fff-6f82-468a-8a56-2b41b86e20f9.png)

在本节中，我们将使用前面食谱中实现的 trie 类，并构建一个小型的搜索查询建议引擎。

# 如何做到...

在本节中，我们将实现一个终端应用程序，它接受一些输入，然后尝试根据一个简单的文本文件数据库猜测用户可能想要查找的内容：

1.  和往常一样，首先是包含部分，我们定义使用`std`命名空间：

```cpp
      #include <iostream>
      #include <optional>
      #include <algorithm>
      #include <functional>
      #include <iterator>
      #include <map>
      #include <list>
      #include <string>
      #include <sstream>
      #include <fstream>  

      using namespace std;
```

1.  我们使用了 trie 食谱中的 trie 实现：

```cpp
      template <typename T>
      class trie
      {
          map<T, trie> tries;

      public:
          template <typename It>
          void insert(It it, It end_it) {
              if (it == end_it) { return; }
              tries[*it].insert(next(it), end_it);
          }

          template <typename C>
          void insert(const C &container) {
              insert(begin(container), end(container));
          }

          void insert(const initializer_list<T> &il) {
              insert(begin(il), end(il));
          }

          void print(list<T> &l) const {
              if (tries.empty()) {
                  copy(begin(l), end(l), 
                       ostream_iterator<T>{cout, " "});
                  cout << 'n';
              }
              for (const auto &p : tries) {
                  l.push_back(p.first);
                  p.second.print(l);
                  l.pop_back();
              }
          }

          void print() const {
              list<T> l;
              print(l);
          }

          template <typename It>
          optional<reference_wrapper<const trie>>
          subtrie(It it, It end_it) const {
              if (it == end_it) { return ref(*this); }
              auto found (tries.find(*it));
              if (found == end(tries)) { return {}; }

      return found->second.subtrie(next(it), end_it);
          }

          template <typename C>
          auto subtrie(const C &c) const { 
              return subtrie(begin(c), end(c));
          }
      };
```

1.  让我们添加一个小的辅助函数，打印一行提示用户输入一些文本：

```cpp
      static void prompt()
      {
          cout << "Next input please:n > ";
      }
```

1.  在主函数中，我们打开一个文本文件，它充当我们的句子数据库。我们逐行读取该文本文件，并将这些行输入到 trie 中：

```cpp
      int main()
      {
          trie<string> t;

          fstream infile {"db.txt"};
          for (string line; getline(infile, line);) {
              istringstream iss {line};
              t.insert(istream_iterator<string>{iss}, {});
          }
```

1.  现在我们已经从文本文件中的内容构建了 trie，我们需要为用户实现一个查询接口。我们提示用户输入一些文本，并等待输入一整行：

```cpp
          prompt();
          for (string line; getline(cin, line);) {
              istringstream iss {line};
```

1.  有了这个文本输入，我们查询 trie 以从中获取一个子 trie。如果我们在文本文件中已经有这样的输入序列，那么我们可以打印出输入如何继续，就像搜索引擎建议功能一样。如果我们找不到匹配的子 trie，我们只是告诉用户：

```cpp
              if (auto st (t.subtrie(istream_iterator<string>{iss}, {})); 
                  st) {
                  cout << "Suggestions:n";
                  st->get().print();
              } else {
                  cout << "No suggestions found.n";
              }
```

1.  之后，我们再次打印提示文本，并等待用户输入下一行。就是这样。

```cpp
              cout << "----------------n";
              prompt();
          }
      }
```

1.  在考虑启动程序之前，我们需要将一些内容填入`db.txt`。输入可以是任何内容，甚至不需要排序。每行文本将是一个 trie 序列：

```cpp
      do ghosts exist
      do goldfish sleep
      do guinea pigs bite
      how wrong can you be
      how could trump become president
      how could this happen to me
      how did bruce lee die
      how did you learn c++
      what would aliens look like
      what would macgiver do
      what would bjarne stroustrup do
      ...
```

1.  在运行程序之前，我们需要创建`db.txt`。它的内容可能是这样的：

```cpp
      hi how are you 
      hi i am great thanks 
      do ghosts exist
      do goldfish sleep
      do guinea pigs bite
      how wrong can you be
      how could trump become president
      how could this happen to me
      how did bruce lee die
      how did you learn c++
      what would aliens look like
      what would macgiver do
      what would bjarne stroustrup do
      what would chuck norris do
      why do cats like boxes
      why does it rain
      why is the sky blue
      why do cats hate water
      why do cats hate dogs
      why is c++ so hard
```

1.  编译和运行程序并输入一些内容看起来像这样：

```cpp
      $ ./word_suggestion 
      Next input please:
       > what would
      Suggestions:
      aliens look like 
      bjarne stroustrup do 
      chuck norris do 
      macgiver do 
      ----------------
      Next input please:
       > why do
      Suggestions:
      cats hate dogs 
      cats hate water 
      cats like boxes 
      ----------------
      Next input please:
       > 
```

# 它是如何工作的...

trie 的工作原理在上一个示例中已经解释过了，但是我们如何填充它以及如何查询它在这里看起来有点奇怪。让我们仔细看一下填充空 trie 的代码片段：

```cpp
fstream infile {"db.txt"};
for (string line; getline(infile, line);) {
    istringstream iss {line};
    t.insert(istream_iterator<string>{iss}, {});
}
```

循环将字符串`line`逐行填充文本文件的内容。然后，我们将字符串复制到一个`istringstream`对象中。从这样的输入流对象，我们可以创建一个`istream_iterator`，这很有用，因为我们的 trie 不仅接受一个容器实例来查找子 trie，而且主要是迭代器。这样，我们就不需要构造一个单词的向量或列表，可以直接使用字符串。通过*移动*`line`的内容到`iss`中，可以避免最后一块不必要的内存分配。不幸的是，`std::istringstream`没有提供接受`std::string`值来*移动*的构造函数。尽管如此，它仍然会*复制*它的输入字符串。

当读取用户的输入以在 trie 中查找时，我们使用完全相同的策略，但我们不使用输入*文件*流。相反，我们使用`std::cin`。对于我们的用例，这完全相同，因为`trie::subtrie`与`trie::insert`一样使用迭代器。

# 还有更多...

可以在 trie 的每个节点上添加*计数变量*。这样，就可以计算某个输入中前缀出现的频率。从而，我们可以根据它们的出现频率对我们的建议进行排序，这实际上就是搜索引擎所做的。智能手机触摸屏文本输入的单词建议也可以用这种方式实现。

这个修改留给读者作为一个练习。

# 使用 STL 数值算法实现傅立叶变换公式

**傅立叶变换**是信号处理中非常重要和著名的公式。它是近 200 年前发明的，但随着计算机的出现，它的用例数量真的飙升了。它被用于音频/图像/视频压缩、音频滤波器、医学成像设备、手机应用程序在听音乐时实时识别音轨等等。

由于一般数值应用场景的广泛性（当然不仅仅是傅立叶变换），STL 也试图在数值计算的上下文中提供帮助。傅立叶变换只是其中的一个例子，但也是一个棘手的例子。公式本身看起来像这样：

![](img/ecd82744-2cc6-4b76-9f93-5f03df78a598.jpg)

它描述的转换基本上是一个*总和*。总和的每个元素都是输入信号向量的数据点与表达式*exp(-2 * i * ...)*的乘积。这背后的数学对于不了解复数（或者只是不喜欢数学）的每个人来说都有点可怕，但是要*实现*它也不是完全必要完全理解数学。仔细观察公式，它说总和符号循环遍历信号的每个数据点（长度为`N`）使用循环变量`j`。变量`k`是另一个循环变量，因为傅立叶变换不是用来计算单个值的，而是用来计算一系列值的。在这个向量中，每个数据点代表了某个重复波频率的强度和相位，这个频率是原始信号的一部分或者不是。当使用手动循环来实现这个时，我们最终会得到类似以下的代码：

```cpp
csignal fourier_transform(const csignal &s) { 
    csignal t(s.size()); 
    const double pol {-2.0 * M_PI / s.size()};

    for (size_t k {0}; k < s.size(); ++k) { 
        for (size_t j {0}; j < s.size(); ++j) { 
            t[k] += s[j] * polar(1.0, pol * k * j); 
        }
    } 
    return t; 
}
```

`csignal` 类型可以是复数的 `std::vector` 向量。对于复数，有一个 `std::complex` STL 类，可以帮助表示它们。`std::polar` 函数基本上执行 *exp(-i * 2 * ...)* 部分。

这已经很好了，但我们将使用 STL 工具来实现它。

# 如何做...

在本节中，我们将实现傅立叶变换及其反向变换，然后玩弄一下，以转换一些信号：

1.  首先，我们包括所有的头文件，并声明我们使用 `std` 命名空间：

```cpp
      #include <iostream>
      #include <complex>
      #include <vector>
      #include <algorithm>
      #include <iterator>
      #include <numeric>
      #include <valarray>
      #include <cmath>      

      using namespace std;
```

1.  信号的数据点是一个复数，应该由 `std::complex` 表示，专门针对 `double` 类型。这样，类型别名 `cmplx` 代表两个耦合的 `double` 值，它们代表复数的 *实部* 和 *虚部*。整个信号是这些项目的向量，我们将其别名为 `csignal` 类型：

```cpp
      using cmplx   = complex<double>;
      using csignal = vector<cmplx>;
```

1.  为了迭代一个递增的数字序列，我们从数字迭代器配方中获取 *numeric iterator*。公式中的变量 `k` 和 `j` 将迭代这样的序列：

```cpp
      class num_iterator {
          size_t i;
      public:
          explicit num_iterator(size_t position) : i{position} {}

          size_t operator*() const { return i; }

          num_iterator& operator++() {
              ++i;
              return *this;
          }

          bool operator!=(const num_iterator &other) const {
              return i != other.i;
          }
      };
```

1.  傅立叶变换函数应该只接受一个信号并返回一个新的信号。返回的信号表示输入信号的傅立叶变换。由于从傅立叶变换信号到原始信号的反变换非常相似，我们提供了一个可选的 `bool` 参数，用于选择变换方向。请注意，`bool` 参数通常是不好的做法，特别是如果我们在函数签名中使用多个 `bool` 参数。这里我们只是为了简洁起见使用了一个。

我们要做的第一件事是分配一个具有初始信号大小的新信号向量：

```cpp
      csignal fourier_transform(const csignal &s, bool back = false)
      {
          csignal t (s.size());
```

1.  公式中有两个因素，它们总是看起来一样。让我们把它们打包到它们自己的变量中：

```cpp
          const double pol {2.0 * M_PI * (back ? -1.0 : 1.0)};
          const double div {back ? 1.0 : double(s.size())};
```

1.  `std::accumulate` 算法是执行求和公式的合适选择。我们将在一个递增的数字值范围上使用 `accumulate`。从这些值中，我们可以形成每一步的单独的加数。`std::accumulate` 算法在每一步调用一个二元函数。这个函数的第一个参数是在前面的步骤中已经计算出的 `sum` 的一部分的当前值，它的第二个参数是范围中的下一个值。我们查找信号 `s` 在当前位置的值，并将其与复数因子 `pol` 相乘。然后，我们返回新的部分和。二元函数被包装成 *另一个* lambda 表达式，因为我们将为每个 `accumulate` 调用使用不同的 `j` 值。因为这是一个二维循环算法，内部 lambda 是内部循环，外部 lambda 是外部循环：

```cpp
          auto sum_up ([=, &s] (size_t j) {
              return [=, &s] (cmplx c, size_t k) {
                  return c + s[k] * 
                      polar(1.0, pol * k * j / double(s.size()));
              };
          });
```

1.  傅立叶变换的内部循环部分现在由 `std::accumulate` 执行。对于算法的每个 `j` 位置，我们计算 *i = 0...N* 的所有加数的和。这个想法被包装成一个 lambda 表达式，我们将为结果傅立叶变换向量中的每个数据点执行它：

```cpp
          auto to_ft (=, &s{
              return accumulate(num_iterator{0}, 
                                num_iterator{s.size()}, 
                                cmplx{},
                                sum_up(j))
                  / div;
          });
```

1.  到目前为止，傅立叶代码还没有执行。我们只是准备了很多功能代码，现在我们将把它们付诸实践。`std::transform` 调用将生成值 *j = 0...N*，这是我们的外部循环。转换后的值都进入向量 `t`，然后我们将其返回给调用者：

```cpp
          transform(num_iterator{0}, num_iterator{s.size()}, 
                    begin(t), to_ft);

          return t;
      }
```

1.  我们将实现一些函数，帮助我们设置用于信号生成的函数对象。第一个是余弦信号生成器。它返回一个可以生成给定周期长度的余弦信号的 lambda 表达式。信号本身可以是任意长度，但周期长度是固定的。周期长度为 `N` 意味着信号在 `N` 步之后会重复。lambda 表达式不接受任何参数。我们可以重复调用它，每次调用都会返回下一个时间点的信号数据点：

```cpp
      static auto gen_cosine (size_t period_len){
          return [period_len, n{0}] () mutable { 
              return cos(double(n++) * 2.0 * M_PI / period_len); 
          };
      }
```

1.  我们要生成的另一个信号是方波。它在值`-1`和`+1`之间振荡，没有其他值。这个公式看起来很复杂，但它只是将线性递增的值`n`转换为`+1`和`-1`，振荡周期长度为`period_len`。

请注意，这次我们将`n`初始化为与`0`不同的值。这样，我们的方波从其输出值开始于`+1`的相位开始：

```cpp
      static auto gen_square_wave (size_t period_len)
      {
          return [period_len, n{period_len*7/4}] () mutable {
              return ((n++ * 2 / period_len) % 2) * 2 - 1.0;
          };
      }
```

1.  从这样的生成器生成实际信号可以通过分配一个新向量并用从重复信号生成器函数调用生成的值填充它来实现。`std::generate`完成了这项工作。它接受一个开始/结束迭代器对和一个生成器函数。对于每个有效的迭代器位置，它执行`*it = gen()`。通过将这段代码封装到一个函数中，我们可以轻松地生成信号向量：

```cpp
      template <typename F>
      static csignal signal_from_generator(size_t len, F gen)
      {
          csignal r (len);
          generate(begin(r), end(r), gen);
          return r;
      }
```

1.  最后，我们需要打印生成的信号。我们可以通过将其值复制到输出流迭代器中来简单地打印信号，但我们需要先转换数据，因为我们的信号数据点是复数值对。在这一点上，我们只对每个数据点的实值部分感兴趣；因此，我们通过`std::transform`调用将其抛出，仅提取这部分：

```cpp
      static void print_signal (const csignal &s)
      {
          auto real_val ([](cmplx c) { return c.real(); });
          transform(begin(s), end(s), 
                    ostream_iterator<double>{cout, " "}, real_val);
          cout << 'n';
      }
```

1.  傅立叶公式现在已经实现了，但我们还没有要转换的信号。这就是我们在主函数中要做的事情。让我们首先定义一个所有信号都符合的标准信号长度。

```cpp
      int main()
      {
          const size_t sig_len {100};
```

1.  现在让我们生成信号，对它们进行变换并打印它们，这发生在接下来的三个步骤中。第一步是生成余弦信号和方波信号。两者都具有相同的总信号长度和周期长度：

```cpp
          auto cosine      (signal_from_generator(sig_len, 
                 gen_cosine(     sig_len / 2)));
          auto square_wave (signal_from_generator(sig_len,
                 gen_square_wave(sig_len / 2)));
```

1.  现在我们有了余弦函数和方波信号。为了在它们之间生成第三个信号，我们取方波信号并计算其傅立叶变换（保存在`trans_sqw`向量中）。方波的傅立叶变换具有特定的形式，我们将对其进行一些操作。从索引`10`到`(signal_length - 10)`的所有项都设置为`0.0`。其余部分保持不变。将这个改变后的傅立叶变换转换回信号时间表示将给我们一个不同的信号。最后我们将看到它是什么样子的：

```cpp
          auto trans_sqw (fourier_transform(square_wave));

          fill (next(begin(trans_sqw), 10), prev(end(trans_sqw), 10), 0);
          auto mid (fourier_transform(trans_sqw, true));
```

1.  现在我们有了三个信号：`cosine`、`mid`和`square_wave`。对于每个信号，我们打印信号本身及其傅立叶变换。整个程序的输出将包括六行非常长的打印的双值列表：

```cpp
          print_signal(cosine);
          print_signal(fourier_transform(cosine));

          print_signal(mid);
          print_signal(trans_sqw);

          print_signal(square_wave);
          print_signal(fourier_transform(square_wave));
      }
```

1.  编译和运行程序会导致终端被大量的数字值填满。如果我们绘制输出，我们会得到以下图像：

![](img/e2b5e05b-3c87-44b4-aebd-ada36d752a21.png)

# 它是如何工作的...

这个程序包含两个复杂的部分。一个是傅立叶变换本身，另一个是使用可变 lambda 表达式生成信号。

让我们先专注于傅立叶变换。原始循环实现的核心（我们没有在我们的实现中使用，但在介绍中看过）如下所示：

```cpp
for (size_t k {0}; k < s.size(); ++k) {
    for (size_t j {0}; j < s.size(); ++j) {
        t[k] += s[j] * polar(1.0, pol * k * j / double(s.size()));
    }
}
```

通过 STL 算法`std::transform`和`std::accumulate`，我们编写了代码，可以总结为以下伪代码：

```cpp
transform(num_iterator{0}, num_iterator{s.size()}, ...
    accumulate((num_iterator0}, num_iterator{s.size()}, ...
        c + s[k] * polar(1.0, pol * k * j / double(s.size()));
```

结果与循环变体完全相同。这可以说是一个例子，严格使用 STL 算法并不会导致更好的代码。尽管如此，这种算法实现对数据结构的选择是不可知的。它也可以在列表上工作（尽管在我们的情况下这没有太多意义）。另一个好处是 C++17 的 STL 算法很容易*并行化*（我们将在本书的另一章中进行讨论），而原始循环必须重构以支持多处理（除非我们使用外部库，例如*OpenMP*，但这些实际上为我们重构了循环）。

另一个复杂的部分是信号生成。让我们再看一下`gen_cosine`：

```cpp
static auto gen_cosine (size_t period_len)
{
    return [period_len, n{0}] () mutable {
        return cos(double(n++) * 2.0 * M_PI / period_len);
    };
}
```

每个 lambda 表达式的实例都代表一个函数对象，它在每次调用时修改自己的状态。它的状态包括变量`period_len`和`n`。`n`变量是在每次调用时修改的变量。信号在每个时间点上都有不同的值，`n++`代表不断增加的时间点。为了从中获得实际的信号向量，我们创建了辅助`signal_from_generator`：

```cpp
template <typename F>
static auto signal_from_generator(size_t len, F gen)
{
    csignal r (len);
    generate(begin(r), end(r), gen);
    return r;
}
```

这个辅助函数分配一个指定长度的信号向量，并调用`std::generate`来填充它的数据点。对于向量`r`的每个项目，它调用函数对象`gen`一次，这正是我们可以使用`gen_cosine`创建的自修改函数对象的类型。

不幸的是，STL 的方式并没有使这段代码更加优雅。一旦 ranges 库加入 STL 俱乐部（希望在 C++20 中实现），这种情况很可能会改变。

# 计算两个向量的误差和

有不同的可能性来计算目标值和实际值之间的数值*误差*。测量由许多数据点组成的信号之间的差异通常涉及循环和相应数据点的减法等。

计算信号`a`和信号`b`之间的误差的一个简单公式如下：

![](img/a1db597b-9f14-48c2-bd5f-0a24b3019643.jpg)

对于每个*i*，它计算*a[i] - b[i]*，对这个差值进行平方（这样，负数和正数的差异可以进行比较），最后将这些值相加。这又是一个可以使用循环的情况，但出于有趣的原因，我们将使用 STL 算法来完成。好处是以这种方式获得了数据结构的独立性。我们的算法将适用于向量和类似列表的数据结构，其中不可能进行直接索引。

# 如何做...

在这一部分，我们将创建两个信号并计算它们的误差和：

1.  像往常一样，首先是包含语句。然后，我们声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <cmath>
      #include <algorithm>
      #include <numeric>
      #include <vector>
      #include <iterator>      

      using namespace std;
```

1.  我们将计算两个信号的误差和。两个信号将是正弦波和它的副本，但值类型不同--原始正弦波保存在`double`变量的向量中，其副本保存在`int`变量的向量中。因为将值从`double`变量复制到`int`变量会在小数点后截断其小数部分，我们会有一些*损失*。让我们将`double`值的向量命名为`as`，代表*模拟信号*，将`int`值的向量命名为`ds`，代表*数字信号*。然后，误差和将告诉我们实际损失有多大：

```cpp
      int main()
      {
          const size_t sig_len {100};
          vector<double> as (sig_len); // a for analog
          vector<int>    ds (sig_len); // d for digital
```

1.  为了生成正弦波信号，我们使用一个带有*mutable*计数器值`n`的小 lambda 表达式。我们可以随时调用它，每次调用它都会返回正弦波的下一个时间点的值。`std::generate`调用用生成的信号填充信号向量，`std::copy`调用随后将所有`double`变量的值复制到`int`变量的向量中：

```cpp
          auto sin_gen ([n{0}] () mutable { 
              return 5.0 * sin(n++ * 2.0 * M_PI / 100); 
          });

          generate(begin(as), end(as), sin_gen);
          copy(begin(as), end(as), begin(ds));
```

1.  首先打印信号，这样它们以后可以绘制：

```cpp
          copy(begin(as), end(as), 
               ostream_iterator<double>{cout, " "});
          cout << 'n';
          copy(begin(ds), end(ds), 
               ostream_iterator<double>{cout, " "});
          cout << 'n';
```

1.  现在来看实际的误差和，我们使用`std::inner_product`，因为它可以很容易地适应计算信号向量的每两个对应元素之间的差异。它将遍历两个范围，选择在范围中相应位置的项目，计算它们之间的差异，对其进行平方，并累积结果：

```cpp
          cout << inner_product(begin(as), end(as), begin(ds), 
                                0.0, std::plus<double>{},
                                [](double a, double b) { 
                                    return pow(a - b, 2); 
                                }) 
               << 'n';
      }
```

1.  编译和运行程序会给我们两行信号输出和第三行，其中包含一个单一的输出值，即两个信号之间的误差。误差是`40.889`。如果我们连续计算误差，首先是第一对项目，然后是前两对项目，然后是前三对项目，依此类推，那么我们得到的是累积误差曲线，它在绘制的图表上可见，如下所示：

![](img/33d6fd3e-c17d-4f36-a4fc-69a3a83d4804.png)

# 它是如何工作的...

在这个示例中，我们将循环遍历两个向量的任务，获取它们对应值的差异，对它们进行平方，最后将它们加总到一个`std::inner_product`调用中。在这个过程中，我们自己编写的唯一代码是 lambda 表达式`[](double a, double b) { return pow(a - b, 2); }`，它获取其参数的差值并对其进行平方。

对`std::inner_product`可能的实现的一瞥告诉我们为什么以及如何工作：

```cpp
template<class InIt1, class InIt2, class T, class F, class G>
T inner_product(InIt1 it1, InIt1 end1, InIt2 it2, T val,
                F bin_op1, G bin_op2)
{
    while (it1 != end1) {
        val = bin_op1(val, bin_op2(*it1, *it2));
        ++it1;
        ++it2;
    }
    return value;
}
```

该算法接受第一个范围的一对 begin/end 迭代器，以及第二个范围的另一个 begin 迭代器。在我们的情况下，它们是我们想要计算误差和的向量。下一个字符是初始值`val`。我们将其初始化为`0.0`。然后，该算法接受两个二元函数，即`bin_op1`和`bin_op2`。

此时，我们可能意识到这个算法与`std::accumulate`非常相似。唯一的区别是`std::accumulate`只适用于*一个*范围。如果我们用`*it`语句替换`bin_op2(*it1, *it2)`，那么我们基本上恢复了`accumulate`算法。因此，我们可以将`std::inner_product`视为`std::accumulate`的一个版本，它*zip*了一对输入范围。

在我们的情况下，*zipper*函数是`pow(a - b, 2)`，就是这样。对于另一个函数`bin_op1`，我们选择了`std::plus<double>`，因为我们希望所有的平方都被加在一起。

# 实现 ASCII Mandelbrot 渲染器

1975 年，数学家 Benoît Mandelbrot 创造了术语**分形**。分形是一个数学图形或集合，它具有某些有趣的数学特性，但最终看起来就像一件艺术品。当放大时，分形也看起来*无限* *重复*。最流行的分形之一是*Mandelbrot 集*，可以在以下海报上看到：

![](img/f18ecb5f-e3af-4184-b68c-1c99bd5cd4c5.jpg)

Mandelbrot 集的图像可以通过迭代特定的公式生成：

![](img/628bf694-e1cc-48c2-a1dd-d110a7d8fa6d.jpg)

变量*z*和*c*是*复数*。Mandelbrot 集由*c*的所有这样的值组成，如果应用足够多次公式，它就会*收敛*。这是海报的彩色部分。一些值会更早地收敛，一些会更晚地收敛，因此它们可以用不同的颜色来可视化。有些根本不会收敛--这些被涂成黑色。

STL 附带有有用的`std::complex`类，我们将尝试实现这个公式，而不使用显式循环，只是为了更好地了解 STL。

# 如何做...

在本节中，我们将在终端中打印与墙海报相同的图像的一小部分 ASCII 艺术：

1.  首先，我们包括所有的头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <algorithm>
      #include <iterator>
      #include <complex>
      #include <numeric>
      #include <vector>      

      using namespace std;
```

1.  Mandelbrot 集和公式操作复数。因此，我们定义了一个类型别名`cmplx`，它是`std::complex`类，专门针对双精度值。

```cpp
      using cmplx = complex<double>;
```

1.  可以将所有 ASCII Mandelbrot 图像的代码拼凑在大约 20 行左右，但我们将分别实现每个逻辑步骤，然后在最后组装所有步骤。第一步是实现一个从整数坐标到浮点坐标的缩放函数。我们一开始在终端上有字符位置的列和行。我们想要的是 Mandelbrot 集合坐标系中的复数类型坐标。为此，我们实现一个接受描述用户终端坐标系几何形状和我们要转换到的坐标系的参数的函数。这些值用于构建一个 lambda 表达式，然后返回。lambda 表达式接受一个`int`坐标并返回一个`double`坐标：

```cpp
      static auto scaler(int min_from, int max_from, 
                         double min_to, double max_to)
      {
          const int    w_from   {max_from - min_from};
          const double w_to     {max_to - min_to};
          const int    mid_from {(max_from - min_from) / 2 + min_from};
          const double mid_to   {(max_to - min_to) / 2.0 + min_to};

          return [=] (int from) {
              return double(from - mid_from) / w_from * w_to + mid_to;
          };
      }
```

1.  现在我们可以在一维坐标上进行变换，但 Mandelbrot 集合存在于二维坐标系中。为了从一个`(x, y)`坐标系转换到另一个，我们结合了一个 x 缩放器和一个 y 缩放器，并从它们的输出构造了一个`cmplx`实例：

```cpp
      template <typename A, typename B>
      static auto scaled_cmplx(A scaler_x, B scaler_y)
      {
          return = {
              return cmplx{scaler_x(x), scaler_y(y)};
          };
      }
```

1.  在能够将坐标转换为正确尺寸之后，我们现在可以实现 Mandelbrot 公式。我们正在实现的函数绝对不知道终端窗口或线性平面变换的概念，因此我们可以专注于 Mandelbrot 数学。我们对`z`进行平方并在循环中添加`c`，直到其`abs`值小于`2`。对于某些坐标，这永远不会发生，因此如果迭代次数超过`max_iterations`，我们也会跳出循环。最后，我们返回我们必须进行的迭代次数，直到`abs`值收敛：

```cpp
      static auto mandelbrot_iterations(cmplx c)
      {
          cmplx z {};
          size_t iterations {0};
          const size_t max_iterations {1000};
          while (abs(z) < 2 && iterations < max_iterations) {
              ++iterations;
              z = pow(z, 2) + c;
          }
          return iterations;
      }
```

1.  现在我们可以开始主函数，其中我们定义终端尺寸并实例化一个函数对象`scale`，它为两个轴的坐标值进行缩放：

```cpp
      int main()
      {
          const size_t w {100};
          const size_t h {40};

          auto scale (scaled_cmplx(
              scaler(0, w, -2.0, 1.0),
              scaler(0, h, -1.0, 1.0)
          ));
```

1.  为了对整个图像进行一维迭代，我们编写另一个转换函数，它接受一个一维`i`坐标。它根据我们假设的字符宽度计算出`(x, y)`坐标。在将`i`分解为行和列号后，它使用我们的`scale`函数进行转换并返回复杂坐标。

```cpp
          auto i_to_xy (= { return scale(i % w, i / w); });
```

1.  现在我们可以从一维坐标（`int`类型）通过二维坐标（`(int, int)`类型）转换到 Mandelbrot 集合坐标（`cmplx`类型），然后从那里计算迭代次数（再次是`int`类型）。让我们将所有这些组合在一个函数中，为我们设置这个调用链：

```cpp
          auto to_iteration_count (= { 
              return mandelbrot_iterations(i_to_xy(i));
          });
```

1.  现在我们可以设置所有数据。我们假设我们的 ASCII 图像的宽度为`w`个字符，高度为`h`个字符。这可以保存在一个具有`w * h`个元素的一维向量中。我们使用`std::iota`填充这个向量，其值范围为*0 ... (w*h - 1)*。这些数字可以作为我们构造的转换函数范围的输入源，我们刚刚封装在`to_iteration_count`中：

```cpp
          vector<int> v (w * h);
          iota(begin(v), end(v), 0);
          transform(begin(v), end(v), begin(v), to_iteration_count);
```

1.  基本上就是这样。我们现在有了`v`向量，我们用一维坐标初始化了它，但后来被 Mandelbrot 迭代计数器覆盖了。从这里，我们现在可以打印一个漂亮的图像。我们可以将终端窗口设置为`w`个字符宽，然后我们就不需要在中间打印换行符了。但我们也可以*创造性地滥用*`std::accumulate`来为我们做换行。`std::accumulate`使用二进制函数来减少一个范围。我们提供一个二进制函数，它接受一个输出迭代器（我们将在下一步中链接到终端），以及范围中的单个值。如果迭代次数大于 50，我们将打印这个值作为`*`字符。否则，我们只打印一个空格字符。如果我们在*行末*（因为计数变量`n`可以被`w`整除），我们打印一个换行符号：

```cpp
          auto binfunc ([w, n{0}] (auto output_it, int x) mutable {
              *++output_it = (x > 50 ? '*' : ' ');
              if (++n % w == 0) { ++output_it = 'n'; }
              return output_it;
          });
```

1.  通过在输入范围上调用`std:accumulate`，结合我们的二进制打印函数和`ostream_iterator`，我们可以将计算出的 Mandelbrot 集刷新到终端窗口：

```cpp
          accumulate(begin(v), end(v), ostream_iterator<char>{cout}, 
                     binfunc);
      }
```

1.  编译和运行程序会产生以下输出，看起来像最初详细的 Mandelbrot 图像，但是以简化的形式：

![](img/eb4a8203-0f69-4c38-b98e-213ea3541018.png)

# 它是如何工作的...

整个计算是在一维数组上的`std::transform`调用中进行的：

```cpp
vector<int> v (w * h);
iota(begin(v), end(v), 0);
transform(begin(v), end(v), begin(v), to_iteration_count);
```

那么，到底发生了什么，为什么会这样工作呢？`to_iteration_count`函数基本上是从`i_to_xy`到`scale`再到`mandelbrot_iterations`的调用链。以下图表说明了转换步骤：

![](img/cd9a2b82-10aa-4236-b3bd-a087eb34f71f.png)

这样，我们可以使用一维数组的索引作为输入，并得到该数组点表示的二维平面上 Mandelbrot 公式迭代的次数。好处是这三个转换完全不知道彼此。具有这种关注点分离的代码可以非常好地进行测试，因为每个组件都可以单独测试，而不受其他组件的影响。这样，很容易找到和修复错误，或者只是推理其正确性。

# 构建我们自己的算法 - split

在某些情况下，现有的 STL 算法是不够的。但是没有什么能阻止我们实现自己的算法。在解决特定问题之前，我们应该坚决思考一下，以认识到许多问题可以以通用方式解决。如果我们在解决自己的问题时经常堆积一些新的库代码，那么当我们的同行遇到类似的问题时，我们也在帮助他们。关键是要知道何时足够通用，何时不要追求比所需更通用--否则我们最终会得到一个新的通用语言。

在这个示例中，我们正在实现一个算法，我们将其称为`split`。它可以在特定值的每次出现时拆分任何项目范围，并将由此产生的块复制到输出范围中。

# 如何做...

在本节中，我们将实现我们自己的类似 STL 的算法，称为`split`，然后我们通过拆分示例字符串来检查它：

1.  首先，我们包含一些 STL 库部分，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <string>
      #include <algorithm>
      #include <iterator>
      #include <list>      

      using namespace std;
```

1.  这一节围绕的整个算法是`split`。它接受一对输入迭代器的开始/结束，并一个输出迭代器，这使它与`std::copy`或`std::transform`类似。其他参数是`split_val`和`bin_func`。`split_val`参数是我们在输入范围中搜索的值，它表示我们切割输入区间的分割点。`bin_func`参数是一个函数，它从一对标记分割块子范围的迭代器开始和结束进行转换。我们使用`std::find`遍历输入范围，因此我们从`split_val`值的出现跳到另一个出现。当将长字符串分割成其各个单词时，我们会从空格字符跳到空格字符。在每个分割值上，我们停下来形成一个块并将其馈送到输出范围中：

```cpp
      template <typename InIt, typename OutIt, typename T, typename F>
      InIt split(InIt it, InIt end_it, OutIt out_it, T split_val, 
                 F bin_func)
      {
          while (it != end_it) {
              auto slice_end (find(it, end_it, split_val));
              *out_it++ = bin_func(it, slice_end);

              if (slice_end == end_it) { return end_it; }
              it = next(slice_end);
          }
          return it;
      }
```

1.  让我们使用新的算法。我们构造一个我们想要拆分的字符串。标记最后一个块的结束和下一个块的开始的项目将是破折号字符`'-'`：

```cpp
      int main()
      {
          const string s {"a-b-c-d-e-f-g"};
```

1.  每当算法在一对迭代器上调用其`bin_func`时，我们希望从中构造一个新的字符串：

```cpp
          auto binfunc ([](auto it_a, auto it_b) {
              return string(it_a, it_b);
          });
```

1.  输出范围将是字符串的`std::list`。现在我们可以调用`split`算法，它与所有其他 STL 算法相比具有类似的设计：

```cpp
          list<string> l;
          split(begin(s), end(s), back_inserter(l), '-', binfunc);
```

1.  为了看到我们得到了什么，让我们打印新的分块字符串列表：

```cpp
          copy(begin(l), end(l), ostream_iterator<string>{cout, "n"});
      }
```

1.  编译和运行程序产生以下输出。它不再包含破折号，并显示它已经隔离了单词（在我们的示例字符串中当然只有单个字符）：

```cpp
      $ ./split 
      a
      b
      c
      d
      e
      f
      g
```

# 它是如何工作的...

`split`算法的工作方式类似于`std::transform`，因为它接受一个输入范围的起始/结束迭代器对和一个输出迭代器。它对输入范围进行某些操作，最终将结果赋值给输出迭代器。除此之外，它还接受一个名为`split_val`的项目值和一个二元函数。让我们重新审视整个实现以充分理解它：

```cpp
template <typename InIt, typename OutIt, typename T, typename F>
InIt split(InIt it, InIt end_it, OutIt out_it, T split_val, F bin_func)
{
    while (it != end_it) {
        auto slice_end (find(it, end_it, split_val));
        *out_it++ = bin_func(it, slice_end);

        if (slice_end == end_it) { return end_it; }
        it = next(slice_end);
    }
    return it;
}
```

循环要求迭代直到输入范围的末尾。在每次迭代期间，使用`std::find`调用来查找输入范围中下一个等于`split_val`的元素。在我们的情况下，该元素是破折号字符（`'-'`），因为我们想要在所有破折号位置分割我们的输入字符串。下一个破折号位置现在保存在`slice_end`中。循环迭代后，`it`迭代器被放在该分割位置的下一个项目上。这样，循环直接从破折号跳到破折号，而不是每个单独的项目上。

在这种情况下，迭代器`it`指向最后一个切片的开头，而`slice_end`指向最后一个切片的结尾。这两个迭代器结合起来标记了表示两个破折号符号之间的子范围的开始和结束。在字符串`"foo-bar-baz"`中，这意味着我们有三次循环迭代，每次我们得到一对迭代器，围绕着一个单词。但实际上我们不想要迭代器，而是`子字符串`。二元函数`bin_func`正是为我们做这件事。当我们调用`split`时，我们给它了以下二元函数：

```cpp
[](auto it_a, auto it_b) {
    return string(it_a, it_b);
}
```

`split`函数通过`bin_func`将每对迭代器传递，然后将其输入输出迭代器。实际上，我们从`bin_func`中得到了字符串实例，结果是`"foo"`、`"bar"`和`"baz"`：

# 还有更多...

实现字符串分割的一个有趣的替代方法是实现一个*迭代器*来完成相同的功能。我们暂时不会实现这样的迭代器，但让我们简要地看一下这种情况。

迭代器需要在每次增量时在分隔符之间跳跃。每当它被解引用时，它需要从它当前指向的迭代器位置创建一个字符串对象，它可以使用之前使用的`binfunc`这样的二元函数来完成。

如果我们有一个名为`split_iterator`的迭代器类，而不是一个名为`split`的算法，用户代码将如下所示：

```cpp
string s {"a-b-c-d-e-f-g"};
list<string> l;

auto binfunc ([](auto it_a, auto it_b) {
    return string(it_a, it_b);
});

copy(split_iterator{begin(s), end(s), ‘-‘, binfunc},{}, back_inserter(l));
```

这种方法的缺点是，实现迭代器通常比单个函数更加*复杂*。此外，迭代器代码中有许多微妙的边缘情况可能导致错误，因此迭代器解决方案需要更多的繁琐测试。另一方面，将这样的迭代器与其他 STL 算法结合起来非常简单。

# 从标准算法组合有用的算法 - gather

STL 算法的可组合性的一个很好的例子是`gather`。当时在 Adobe Systems 担任首席科学家的 Sean Parent 因为这个算法既有用又简短而使其广为流传。它的实现方式使它成为 STL 算法组合理念的理想典范。

`gather`算法操作任意项目类型的范围。它以特定的方式修改项目的顺序，使特定项目围绕着由调用者选择的特定位置聚集起来。

# 如何做...

在本节中，我们将实现`gather`算法以及它的一个额外变体。之后，我们将看看如何使用它：

1.  首先，我们添加所有的 STL 包含语句。然后，我们声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <algorithm>
      #include <string>
      #include <functional>      

      using namespace std;
```

1.  `gather`算法是标准算法组合的一个很好的例子。`gather`接受一个起始/结束迭代器对，以及另一个迭代器`gather_pos`，它指向中间某个位置。最后一个参数是一个谓词函数。使用这个谓词函数，算法将所有满足谓词的项目推到靠近`gather_pos`迭代器的位置。项目移动的实现由`std::stable_partition`完成。`gather`算法的返回值是一对迭代器。这些迭代器是从`stable_partition`调用返回的，这样，它们标记了现在聚集范围的开始和结束：

```cpp
      template <typename It, typename F>
      pair<It, It> gather(It first, It last, It gather_pos, F predicate)
      {
          return {stable_partition(first, gather_pos, not_fn(predicate)),
                  stable_partition(gather_pos, last, predicate)};
      }
```

1.  `gather`的另一个变体是`gather_sort`。它基本上与`gather`的工作方式相同，但它不接受一元谓词函数；它接受一个二元比较函数。这样，就可以聚集出现在`gather_pos`附近的*最小*或*最大*的值：

```cpp
      template <typename It>

      void gather_sort(It first, It last, It gather_pos)

      {

        using T = typename std::iterator_traits<It>::value_type;

        stable_sort(first, gather_pos, greater<T>{});

        stable_sort(gather_pos, last, less<T>{});

      }
```

1.  让我们把这些算法投入使用。我们首先使用一个谓词，告诉我们给定的字符参数是否是`'a'`字符。我们构造一个字符串，其中包含交错的`'a'`和`'_'`字符：

```cpp
      int main()
      {
          auto is_a ([](char c) { return c == 'a'; });
          string a {"a_a_a_a_a_a_a_a_a_a_a"};
```

1.  我们构造一个迭代器，它指向新字符串的中间。让我们在其上调用`gather`，看看会发生什么。之后，`'a'`字符应该被聚集在中间周围：

```cpp
          auto middle (begin(a) + a.size() / 2);

          gather(begin(a), end(a), middle, is_a);
          cout << a << 'n';
```

1.  让我们再次调用`gather`，但这次`gather_pos`迭代器不在中间而是在开头：

```cpp
          gather(begin(a), end(a), begin(a), is_a);
          cout << a << 'n';
```

1.  在第三次调用中，我们聚集了结束迭代器周围的项目：

```cpp
          gather(begin(a), end(a), end(a), is_a);
          cout << a << 'n';
```

1.  最后一次调用`gather`，我们尝试再次聚集所有`'a'`字符周围的中间位置。这将不会按预期工作，稍后我们将看到原因：

```cpp
          // This will NOT work as naively expected
          gather(begin(a), end(a), middle, is_a);
          cout << a << 'n';
```

1.  我们用下划线字符和一些数字值构造另一个字符串。在该输入序列上，我们应用`gather_sort`。`gather_pos`迭代器是字符串的中间，二元比较函数是`std::less<char>`：

```cpp
          string b {"_9_2_4_7_3_8_1_6_5_0_"};
          gather_sort(begin(b), end(b), begin(b) + b.size() / 2, 
                      less<char>{});
          cout << b << 'n';
      }
```

1.  编译和运行程序产生以下有趣的输出。前三行看起来像预期的样子，但第四行看起来像`gather`对字符串*没有*做任何操作。

在最后一行，我们可以看到`gather_short`函数的结果。数字朝着任一方向排序：

```cpp
      $ ./gather 
      _____aaaaaaaaaaa_____
      aaaaaaaaaaa__________
      __________aaaaaaaaaaa
      __________aaaaaaaaaaa
      _____9743201568______
```

# 工作原理...

最初，`gather`算法很难理解，因为它非常简短，但任务似乎很复杂。因此，让我们逐步进行：

![](img/fcb79a46-e8cb-4732-aee3-ba7818d713d7.png)

1.  初始状态是一系列项目，我们为其提供一个谓词函数。在图表中，我们的谓词函数返回`true`的所有项目都以*灰色*绘制。迭代器`a`和`c`标记整个范围，迭代器`b`指向*枢轴*元素。枢轴元素是我们想要*聚集*所有灰色项目的元素周围。

1.  `gather`算法在范围`a，b)`上调用`std::stable_partition`，在执行此操作时，它使用谓词的*否定*版本。它否定谓词，因为`std::stable_partition`将谓词返回`true`的所有项目移动到*前面*。我们希望发生*相反*的情况。

1.  另一个`std::stable_partition`调用被执行，但这次是在范围`[b，c)`上，并且*不*否定谓词。灰色项目被移动到输入范围的前面，这意味着它们都朝着被`b`指向的枢轴元素移动。

1.  现在项目围绕`b`聚集，算法返回迭代器，指向现在连续的灰色项目的开始和结束。

我们在相同的范围上多次调用`gather`。起初，我们聚集了范围中间的所有项目。然后我们聚集了范围的`begin()`周围，然后是范围的`end()`周围。这些情况很有趣，因为它们总是导致*一个*`std::stable_partition`调用在*空*范围上操作，这导致*没有动作*。

我们再次使用范围的参数`(begin, end, middle)`对`gather`进行最后一次调用，但没有起作用。为什么？乍一看，这似乎是一个错误，但实际上并不是。

想象一下字符范围`"aabb"`，以及一个谓词函数`is_character_a`，它只对`'a'`项为真--如果我们用一个指向字符范围中间的第三个迭代器调用它，我们会观察到相同的*错误*。原因是第一个`stable_partition`调用将在子范围`"aa"`上操作，而另一个`stable_partition`调用将在范围`"bb"`上操作。这一系列的调用不能导致最初我们天真地希望的`"baab"`。

为了在最后一种情况下得到我们想要的结果，我们可以使用`std::rotate(begin, begin + 1, end);`

`gather_sort`修改基本上与`gather`相同。唯一的区别是它不接受一个一元的*谓词*函数，而是一个二元的*比较*函数，就像`std::sort`一样。而且它不是调用`std::stable_partition`两次，而是调用`std::stable_sort`两次。

比较函数的否定不能使用`not_fn`来完成，就像我们在`gather`算法中所做的那样，因为`not_fn`不能用于二元函数。

# 删除单词之间的连续空格

因为字符串通常是从用户输入中读取的，它们可能包含狂野的格式，通常需要被清理。其中一个例子是包含太多空格的字符串。

在本节中，我们将实现一个巧妙的空格过滤算法，它从字符串中删除多余的空格，但保留单个空格字符。我们称这个算法为`remove_multi_whitespace`，它的接口看起来非常类似 STL。

# 如何做...

在本节中，我们将实现`remove_multi_whitespace`算法并查看它是如何工作的：

1.  像往常一样，我们首先进行一些包含，然后声明我们默认使用`std`命名空间：

```cpp
      #include <iostream>
      #include <string>
      #include <algorithm>      

      using namespace std;
```

1.  我们实现了一个新的 STL 风格的算法，称为`remove_multi_whitespace`。这个算法删除了空格的聚集出现，但不删除单个空格。这意味着字符串`"a b"`保持不变，但像`"a b"`这样的字符串

`"a b"`被缩减为`"a b"`。为了实现这一点，我们使用了一个自定义的二元谓词函数来调用`std::unique`。`std::unique`遍历一个可迭代范围，并总是查看连续的负载项对。然后它询问谓词函数两个项是否相等。如果是，那么`std::unique`会删除其中一个。之后，范围不再包含相邻的相等项的子范围。通常在这种情况下应用的谓词函数告诉两个项是否相等。我们所做的是给`std::unique`一个谓词，它告诉是否有两个连续的*空格*，以便将它们删除。就像`std::unique`一样，我们接受一对 begin/end 迭代器，然后返回一个指向范围新结尾的迭代器：

```cpp
      template <typename It>
      It remove_multi_whitespace(It it, It end_it)
      {
          return unique(it, end_it, [ {
              return isspace(a) && isspace(b);
          });
      }
```

1.  就是这样了。让我们构造一个包含一些不必要空格的字符串：

```cpp
      int main()
      {
          string s {"fooo     bar    t   baz"};

          cout << s << 'n';
```

1.  现在，我们使用*erase-remove idiom*来处理字符串，以摆脱多余的空格字符：

```cpp
          s.erase(remove_multi_whitespace(begin(s), end(s)), end(s));

          cout << s << 'n';
      }
```

1.  编译和运行程序产生以下输出：

```cpp
      $ ./remove_consecutive_whitespace 
      fooo     bar        baz
      fooo bar baz
```

# 它是如何工作的...

我们解决了问题的整个复杂性，没有任何循环或手动比较项目。我们只提供了一个谓词函数，告诉我们给定的两个字符是否是*空格*字符。然后我们将该谓词输入到`std::unique`中，*噗*，所有多余的空格都消失了。虽然本章还包含一些我们必须更努力地使用 STL 算法来表达我们的程序的示例，但这个算法是一个*真正*好的、简短的例子。

这个有趣的组合是如何详细工作的呢？让我们首先看一下`std::unique`的可能实现：

```cpp
template<typename It, typename P>
It unique(It it, It end, P p)
{
    if (it == end) { return end; }

    It result {it};
    while (++it != end) {
        if (!p(*result, *it) && ++result != it) {
            *result = std::move(*it);
        }
    }
    return ++result;
}
```

循环遍历范围项，直到它们不满足谓词条件。在满足谓词的位置，它将这样的项移动到上一次触发谓词的旧位置之后的一个项。不接受额外谓词函数的`std::unique`版本检查两个相邻项是否相等。这样，它可以，例如，将`"abbbbbbc"`转换为`"abc"`，从而消除*重复*字符。

我们想要的不是清除*所有*重复的字符，而是重复的*空格*。因此，我们的谓词不是说“两个参数字符相等”，而是“两个参数字符都是空格字符”。

最后要注意的一点是，`std::unique`和`remove_multi_whitespace`都不会真正从基础字符串中删除字符项。它们只是根据语义在字符串中移动字符，并告诉它的新结尾在哪里。必须仍然删除从新结尾到旧结尾的所有现在过时的字符。这就是为什么我们写了以下内容：

```cpp
s.erase(remove_multi_whitespace(begin(s), end(s)), end(s));
```

这遵循*擦除-移除*惯用法，我们已经从向量和列表中了解到。

# 压缩和解压缩字符串

这一部分涉及编码面试中相对流行的任务。基本思想是一个函数，它接受一个字符串，比如`"aaaaabbbbbbbccc"`，并将其转换为一个更短的字符串`"a5b7c3"`。它是`"a5"`，因为有五个`'a'`字符。然后是`"b7"`，因为有七个`'b'`字符。这是一个非常简单的*压缩*算法。对于普通文本来说，它的效用减少了，因为正常语言通常不会重复到使其文本表示使用这种压缩方案变得更短。然而，即使我们不得不在白板上手动实现，它相对容易实现。棘手的部分是，如果程序一开始的结构不是很好，很容易编写有 bug 的代码。处理字符串通常不是一件难事，但是如果使用了传统的 C 风格格式化函数，那么在这里实现缓冲区溢出错误的机会就会*很多*。

让我们尝试使用这种简单方案来实现字符串压缩和解压缩的 STL 方法。

# 如何做...

在这一部分，我们将为字符串实现简单的`compress`和`decompress`函数：

1.  首先，我们包括一些 STL 库，然后声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <string>
      #include <algorithm>
      #include <sstream>
      #include <tuple>      

      using namespace std;
```

1.  对于我们的廉价压缩算法，我们试图找到包含相同字符范围的文本块，并单独压缩它们。每当我们从一个字符串位置开始时，我们希望找到包含不同字符的第一个位置。我们使用`std::find`来找到范围中第一个与当前位置的字符不同的字符。之后，我们返回一个元组，其中包含指向第一个不同项的迭代器，填充当前范围的字符变量`c`，以及此子范围包含的出现次数：

```cpp
      template <typename It>
      tuple<It, char, size_t> occurrences(It it, It end_it)
      {
          if (it == end_it) { return {it, '?', 0}; }

          const char c {*it};
          const auto diff (find_if(it, end_it, 
                           c { return c != x; }));

          return {diff, c, distance(it, diff)};
      }
```

1.  `compress`算法不断调用`occurrences`函数。这样，我们从一个相同的字符组跳到另一个字符组。`r << c << n`行将字符推入输出流，然后是它在输入字符串的这一部分中的出现次数。输出是一个自动随着输出增长的字符串流。最后，我们从中返回一个字符串对象，其中包含压缩后的字符串：

```cpp
      string compress(const string &s)
      {
          const auto end_it (end(s));
          stringstream r;

          for (auto it (begin(s)); it != end_it;) {
              const auto [next_diff, c, n] (occurrences(it, end_it));
              r << c << n;
              it = next_diff;
          }

          return r.str();
      }
```

1.  `decompress`方法的工作方式类似，但更简单。它不断尝试从输入流中获取字符值，然后获取接下来的数字。根据这两个值，它可以构造一个包含字符的字符串，次数由数字表示。最后，我们再次从输出流返回一个字符串。顺便说一下，这个`decompress`函数是*不安全*的。它很容易被利用。你能猜到怎么做吗？我们稍后会看一下这个问题：

```cpp
      string decompress(const string &s)
      {
          stringstream ss{s};
          stringstream r;

          char c;
          size_t n;
          while (ss >> c >> n) { r << string(n, c); }

          return r.str();
      }
```

1.  在我们的主函数中，我们构造了一个有很多重复的简单字符串，算法在这个字符串上运行得非常好。让我们打印压缩版本，然后是压缩和再次解压缩版本。最后，我们应该得到与最初构造的相同的字符串：

```cpp
      int main()
      { 
          string s {"aaaaaaaaabbbbbbbbbccccccccccc"};
          cout << compress(s) << 'n';
          cout << decompress(compress(s)) << 'n';
      }
```

1.  编译和运行程序产生以下输出：

```cpp
      $ ./compress
      a9b9c11
      aaaaaaaaabbbbbbbbbccccccccccc
```

# 它的工作原理...

这个程序基本上围绕着两个函数：`compress`和`decompress`。

解压函数非常简单，因为它只包括变量声明、一行实际执行操作的代码和接下来的返回语句。实际执行操作的代码行是以下代码：

```cpp
while (ss >> c >> n) { r << string(n, c); }
```

它不断地从字符串流`ss`中读取字符`c`和计数器变量`n`。此时，`stringstream`类为我们隐藏了很多字符串解析的魔法。当这成功时，它将构造一个解压后的字符串块到字符串流中，从中可以将最终结果字符串返回给`decompress`的调用者。如果`c = 'a'`且`n = 5`，表达式`string(n, c)`将得到一个内容为"aaaaa"的字符串。

压缩函数更复杂。我们还为它编写了一个小的辅助函数。我们称这个辅助函数为`occurences`。所以，让我们先看一下`occurrences`。以下图表显示了它的工作原理：

![](img/888717aa-d626-4a9c-a675-414518c45197.png)

`occurences`函数接受两个参数：指向范围内字符序列开头的迭代器和该范围的结束迭代器。使用`find_if`，它找到第一个与最初指向的字符不同的字符。在图表中，这是迭代器`diff`。新位置和旧迭代器位置之间的差异就是相等项的数量（在图表中，`diff - it`等于**6**）。计算出这些信息后，`diff`迭代器可以被重用以执行下一次搜索。因此，我们将`diff`、子范围的字符和子范围的长度打包到一个元组中并返回它。

将信息排列如此，我们可以从子范围跳到子范围，并将中间结果推入压缩目标字符串中：

```cpp
for (auto it (begin(s)); it != end_it;) { 
    const auto [next_diff, c, n] (occurrences(it, end_it)); 
    r << c << n; 
    it = next_diff; 
}
```

# 还有更多...

在第 4 步中，我们提到`decompress`函数是不安全的。的确，它很容易被*利用*。

想象一下以下输入字符串："a00000"。压缩它将导致子字符串"a1"，因为只有一个字符'a'。接下来是五次'0'，这将导致"05"。这样，压缩后的字符串就是"a105"。不幸的是，这个压缩后的字符串表示"字符'a'出现 105 次"。这与我们最初的输入字符串无关。更糟糕的是，如果我们解压它，从一个六个字符的字符串变成一个 105 个字符的字符串。想象一下数字更大的情况——用户很容易就能够使我们的堆使用量*爆炸*，因为我们的算法没有准备好处理这样的输入。

为了防止这种情况，`compress`函数可以，例如，拒绝带有数字的输入，或者可以以特殊方式掩盖它们。`decompress`算法可以采取另一个条件，对结果字符串大小设置一个上限。我把这留给你作为练习。
