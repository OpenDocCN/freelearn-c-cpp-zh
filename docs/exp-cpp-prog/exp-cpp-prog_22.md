# STL 容器

在本章中，我们将介绍以下配方：

+   在`std::vector`上使用擦除-删除习惯用法

+   在*O(1)*时间内从未排序的`std::vector`中删除项目

+   以快速或安全的方式访问`std::vector`实例

+   保持`std::vector`实例排序

+   有效地和有条件地将项目插入`std::map`

+   了解`std::map::insert`的新插入提示语义

+   有效地修改`std::map`项的键

+   使用`std::unordered_map`与自定义类型

+   使用`std::set`从用户输入中过滤重复项并按字母顺序打印它们

+   使用`std::stack`实现简单的逆波兰计算器

+   使用`std::map`实现词频计数器

+   使用`std::set`实现用于在文本中查找非常长的句子的写作风格辅助工具

+   使用`std::priority_queue`实现个人待办事项列表

# 在`std::vector`上使用擦除-删除习惯用法

许多初学者 C++程序员了解`std::vector`，它基本上就像一个*自动增长的数组*，然后就停在那里。后来，他们只查找它的文档，以了解如何做非常具体的事情，例如*删除*项目。像这样使用 STL 容器只会触及它们帮助编写*清晰*、*可维护*和*快速*代码的表面。

本节重点是从向量实例中间删除项目。当一个项目从向量中消失，并且坐在其他项目的中间*之间*时，那么右边的所有项目都必须向*左*移动一个插槽（这使得这个任务的运行成本在*O(n)*内）。许多初学者程序员会使用*循环*来做到这一点，因为这也不是一件很难做的事情。不幸的是，他们在这样做的过程中可能会忽略很多优化潜力。最后，手工制作的循环既不如 STL 方式*快*，也不如*美观*，我们将在下面看到。

# 如何做...

在本节中，我们正在用一些示例整数填充`std::vector`实例，然后从中删除一些特定的项目。我们正在做的方式被认为是从向量中删除多个项目的*正确*方式。

1.  当然，在我们做任何事情之前，我们需要包括一些头文件。

```cpp
      #include <iostream>
      #include <vector>
      #include <algorithm>
```

1.  然后我们声明我们正在使用`std`命名空间，以节省一些输入。

```cpp
      using namespace std;
```

1.  现在我们创建一个整数向量，并用一些示例项目填充它。

```cpp
      int main()
      {
          vector<int> v {1, 2, 3, 2, 5, 2, 6, 2, 4, 8};
```

1.  下一步是删除项目。我们要删除什么？有多个`2`值。让我们把它们删除。

```cpp
          const auto new_end (remove(begin(v), end(v), 2));
```

1.  有趣的是，这只是两步中的一步。向量仍然具有相同的大小。下一行使它实际上更短。

```cpp
          v.erase(new_end, end(v));
```

1.  让我们在这里停下来，以便将向量的内容打印到终端，然后继续。

```cpp
          for (auto i : v) {
              cout << i << ", ";
          }
          cout << 'n';
```

1.  现在，让我们删除整个*类*的项目，而不是特定的*值*。为了做到这一点，我们首先定义一个谓词函数，它接受一个数字作为参数，并在它是*奇数*时返回`true`。

```cpp
          const auto odd ([](int i) { return i % 2 != 0; });
```

1.  现在我们使用`remove_if`函数，并将其与谓词函数一起使用。与之前的两步删除不同，我们现在只需一步。

```cpp
          v.erase(remove_if(begin(v), end(v), odd), end(v));
```

1.  现在所有奇数项都消失了，但向量的*容量*仍然是旧的 10 个元素。在最后一步中，我们还将其减少到向量的实际*当前*大小。请注意，这可能导致向量代码分配一个适合的新内存块，并将所有项目从旧内存块移动到新内存块。

```cpp
          v.shrink_to_fit();
```

1.  现在，让我们在第二次删除项目后打印内容，就这样。

```cpp
          for (auto i : v) {
              cout << i << ", ";
          }
          cout << 'n';
      }
```

1.  编译和运行程序产生了两种删除项目方法的以下两行输出。

```cpp
      $ ./main 
      1, 3, 5, 6, 4, 8, 
      6, 4, 8,
```

# 它是如何工作的...

在配方中显而易见的是，当从向量中间删除项目时，它们首先需要被*删除*，然后*擦除*。至少我们使用的函数有这样的名称。这显然令人困惑，但让我们仔细看看它，以理解这些步骤。

从向量中删除所有值为`2`的代码如下：

```cpp
const auto new_end (remove(begin(v), end(v), 2));
v.erase(new_end, end(v));
```

`std::begin`和`std::end`函数都接受一个向量实例作为参数，并返回指向*第一个*项目和*最后一个*项目之后的迭代器，就像即将出现的图表中所示的那样。

在将这些值和值`2`传递给`std::remove`函数后，它将将非`2`值向前移动，就像我们可以使用手动编程的循环来做的那样。该算法将严格保留所有非`2`值的顺序。快速查看插图可能有点令人困惑。在第 2 步中，仍然有一个值为`2`，而且向量应该变得更短，因为有四个值为`2`，它们都应该被移除。相反，初始数组中的`4`和`8`被复制了。这是怎么回事？

![](img/31fad71f-4671-4aae-8626-ff3f3785f7d1.png)

让我们只看看所有在范围内的项目，从插图上的`begin`迭代器到`new_end`迭代器。`new_end`迭代器指向的项目是范围之外的*第一个项目，因此不包括在内。只集中在这个区域（这些只是从`1`到包括`8`的项目），我们意识到*这*是从中删除所有`2`值的*正确*范围。

这就是`erase`调用发挥作用的地方：我们必须告诉向量，它不再应该认为从`new_end`到`end`的所有项目是向量的项目。这个顺序对于向量来说很容易遵循，因为它只需将其`end`迭代器指向`new_end`的位置，就完成了。请注意，`new_end`是`std::remove`调用的返回值，所以我们可以直接使用它。

请注意，向量所做的不仅仅是移动内部指针。如果该向量是更复杂对象的向量，它将调用所有要删除的项目的析构函数。

之后，向量看起来像图表中的第 3 步：它现在被认为是*更小*的。现在超出范围的旧项目仍然在内存中。

为了使向量只占用所需的内存，我们在最后进行`shrink_to_fit`调用。在该调用期间，它将分配所需的内存，移动所有项目并删除我们不再需要的较大块。

在第 8 步中，我们定义了一个*谓词*函数，并在一步中使用它与`std::remove_if`一起使用。这是有效的，因为无论删除函数返回什么迭代器，都可以安全地在向量的 erase 函数中使用。即使*没有找到奇数项*，`std::remove_if`函数也将*什么也不做*，并返回`end`迭代器。然后，像`v.erase(end, end);`这样的调用也不会做任何事情，因此它是无害的。

# 还有更多...

`std::remove`函数也适用于其他容器。当与`std::array`一起使用时，请注意它不支持调用`erase`的第二步，因为它们没有自动大小处理。仅仅因为`std::remove`有效地只是移动项目而不执行它们的实际删除，它也可以用于不支持调整大小的数据结构，例如数组。在数组的情况下，可以使用类似于字符串的哨兵值（例如`''`）覆盖新的结束迭代器之后的值。

# 在 O(1)时间内从未排序的 std::vector 中删除项目

从`std::vector`中间某处删除项目需要*O(n)*时间。这是因为删除项目后产生的间隙必须由将在间隙后面的所有项目向左移动一个插槽来填充。

在像这样移动项目的过程中，如果它们是复杂的和/或非常大的，并包括许多项目，这可能是昂贵的，我们保留它们的顺序。如果保留顺序不重要，我们可以优化这一点，正如本节所示。

# 如何做...

在本节中，我们将使用一些示例数字填充一个`std::vector`实例，并实现一个快速删除函数，它可以在*O(1)*时间内从向量中删除任何项目。

1.  首先，我们需要包含所需的头文件。

```cpp
      #include <iostream>
      #include <vector>
      #include <algorithm>
```

1.  然后，我们定义一个主函数，在其中实例化一个包含示例数字的向量。

```cpp
      int main()
      {
          std::vector<int> v {123, 456, 789, 100, 200};
```

1.  下一步是删除索引为`2`的值（当然是从零开始计数，所以是第三个数字`789`）。我们将使用的函数还没有实现。我们稍后再做。之后，我们打印向量的内容。

```cpp
          quick_remove_at(v, 2);

          for (int i : v) {
              std::cout << i << ", ";
          }                                           
          std::cout << 'n';
```

1.  现在，我们将删除另一个项目。它将是值为`123`，假设我们不知道它的索引。因此，我们将使用`std::find`函数，它接受一个范围（向量）和一个值，然后搜索该值的位置。然后，它会返回一个指向`123`值的*迭代器*。我们将使用相同的`quick_remove_at`函数，但这是*先前*接受*迭代器*的*重载*版本。它也还没有实现。

```cpp
          quick_remove_at(v, std::find(std::begin(v), std::end(v), 123));

          for (int i : v) {
              std::cout << i << ", ";
          }
          std::cout << 'n';
      }
```

1.  除了两个`quick_remove_at`函数，我们已经完成了。所以让我们来实现这些。（请注意，它们应该至少在主函数之前被声明。所以让我们在那里定义它们。）

这两个函数都接受一个*something*（在我们的例子中是`int`值）的向量的引用，所以我们不确定用户会使用什么样的向量。对我们来说，它是一个`T`值的向量。我们使用的第一个`quick_remove_at`函数接受*索引*值，这些值是*数字*，所以接口看起来像下面这样：

```cpp
      template <typename T>
      void quick_remove_at(std::vector<T> &v, std::size_t idx)
      {
```

1.  现在来到食谱的核心部分——我们如何快速删除项目而不移动太多其他项目？首先，我们简单地取出向量中最后一个项目的值，并用它来覆盖将要删除的项目。其次，我们切断向量的最后一个项目。这就是两个步骤。我们在这段代码周围加上了一些健全性检查。如果索引值显然超出了向量范围，我们就什么也不做。否则，例如在空向量上，代码会崩溃。

```cpp
          if (idx < v.size()) {
              v[idx] = std::move(v.back());
              v.pop_back();
          }
      }
```

1.  `quick_remove_at`的另一个实现方式类似。它不是接受一个数字索引，而是接受`std::vector<T>`的迭代器。以通用方式获取其类型并不复杂，因为 STL 容器已经定义了这样的类型。

```cpp
      template <typename T>
      void quick_remove_at(std::vector<T> &v, 
                           typename std::vector<T>::iterator it)
      {

```

1.  现在，我们将访问迭代器指向的值。就像在另一个函数中一样，我们将用向量中的最后一个元素来覆盖它。因为这次我们处理的不是数字索引，而是迭代器，所以我们需要以稍有不同的方式检查迭代器的位置是否合理。如果它指向人为结束的位置，我们就不能对其进行解引用。

```cpp
          if (it != std::end(v)) {
```

1.  在那个 if 块中，我们做的事情和之前一样——我们用最后一个位置的项目的值来覆盖要删除的项目，然后我们从向量中切断最后一个元素：

```cpp
              *it = std::move(v.back());
              v.pop_back();
          }
      }
```

1.  就是这样。编译和运行程序会产生以下输出：

```cpp
      $ ./main 
      123, 456, 200, 100,                           
      100, 456, 200,
```

# 它是如何工作的...

`quick_remove_at`函数可以快速删除项目，而不会触及太多其他项目。它以相对创造性的方式做到这一点：它在某种程度上*交换*了*实际项目*，即将被删除的项目和向量中的*最后一个*项目。尽管最后一个项目与实际选择的项目*没有关联*，但它处于*特殊位置*：删除最后一个项目是*便宜的*！向量的大小只需要减少一个位置，就完成了。在这一步中没有移动任何项目。看一下下面的图表，它有助于想象这是如何发生的：

![](img/91627e22-fdaf-41d9-a683-6c96f788f8b8.png)

食谱代码中的两个步骤看起来像这样：

```cpp
v.at(idx) = std::move(v.back());
v.pop_back();
```

这是迭代器版本，看起来几乎一样：

```cpp
*it = std::move(v.back());
v.pop_back();
```

逻辑上，我们*交换*所选项目和最后一个项目。但代码并不交换项目，而是将最后一个项目移动到第一个项目上。为什么？如果我们交换项目，那么我们将不得不将所选项目存储在一个*临时*变量中，将最后一个项目移动到所选项目上，然后再将临时值存储在最后一个位置上。这似乎是*无用*的，因为我们正要*删除*最后一个项目。

好的，交换是没有用的，一次性覆盖是更好的选择。看到这一点，我们可以说这一步也可以用简单的`*it = v.back();`来完成，对吗？是的，这完全是*正确*的，但是想象一下，我们在每个槽中存储了一些非常大的字符串，甚至是另一个向量或映射--在这种情况下，这个小赋值将导致非常昂贵的复制。中间的`std::move`调用只是一个*优化:*在*字符串*的示例情况下，字符串项内部指向*堆*中的一个大字符串。我们不需要复制它。相反，当*移动*一个字符串时，移动的目标指向另一个字符串的数据。移动源项目保持不变，但处于无用状态，这没关系，因为我们无论如何都要删除它。

# 以快速或安全的方式访问 std::vector 实例

`std::vector` 可能是 STL 中使用最广泛的容器，因为它像数组一样保存数据，并在该表示周围添加了很多便利。然而，对向量的错误访问仍然可能是危险的。如果一个向量包含 100 个元素，并且我们的代码意外地尝试访问索引 123 处的元素，这显然是不好的。这样的程序可能会崩溃，这可能是最好的情况，因为这种行为会非常明显地表明存在错误！如果它没有崩溃，我们可能会观察到程序偶尔表现得*奇怪*，这可能会比崩溃的程序带来更多的头痛。有经验的程序员可能会在任何直接索引的向量访问之前添加一些检查。这些检查不会增加代码的可读性，而且很多人不知道`std::vector`已经内置了边界检查！

# 如何做...

在本节中，我们将使用两种不同的方式来访问`std::vector`，然后看看如何利用它们来编写更安全的程序而不降低可读性。

1.  让我们包括所有需要的头文件，并用`123`的值填充一个示例向量`1000`次，这样我们就有了可以访问的东西：

```cpp
      #include <iostream>
      #include <vector>

      using namespace std;

      int main()
      {
          const size_t container_size {1000};
          vector<int> v (container_size, 123);
```

1.  现在，我们使用`[]`运算符越界访问向量：

```cpp
         cout << "Out of range element value: " 
              << v[container_size + 10] << 'n';
```

1.  接下来，我们使用`at`函数越界访问它：

```cpp
          cout << "Out of range element value: " 
               << v.at(container_size + 10) << 'n';
      }
```

1.  让我们运行程序看看会发生什么。错误消息是特定于 GCC 的。其他编译器会发出不同但类似的错误消息。第一次读取以一种奇怪的方式成功了。它没有导致程序崩溃，但它是一个完全不同的*值*，而不是`123`。我们看不到其他访问的输出行，因为它故意崩溃了整个程序。如果那个越界访问是一个意外，我们会更早地捕捉到它！

```cpp
      Out of range element value: -726629391
      terminate called after throwing an instance of 'std::out_of_range'
        what():  array::at: __n (which is 1010) >= _Nm (which is 1000)
      Aborted (core dumped)
```

# 它是如何工作的...

`std::vector`提供了`[]`运算符和`at`函数，它们基本上做的工作是一样的。然而，`at`函数执行额外的边界检查，并且如果超出向量边界，则抛出*异常*。这在我们这种情况下非常有用，但也会使程序变得稍微*慢*一些。

特别是在进行需要非常快速的索引成员的数值计算时，最好坚持使用`[]`索引访问。在任何其他情况下，`at`函数有助于发现通常可以忽略的性能损失的错误。

默认情况下使用`at`函数是一个好习惯。如果生成的代码太慢但已经被证明没有错误，那么在性能敏感的部分可以使用`[]`运算符。

# 还有更多...

当然，我们可以*处理*越界访问，而不是让整个应用程序*崩溃*。为了处理它，我们*捕获*异常，以防它被`at`函数抛出。捕获这样的异常很简单。我们只需用`try`块包围`at`调用，并在`catch`块中定义错误处理。

```cpp
try {
    std::cout << "Out of range element value: " 
              << v.at(container_size + 10) << 'n';
} catch (const std::out_of_range &e) {
     std::cout << "Ooops, out of range access detected: " 
               << e.what() << 'n';
}
```

顺便说一下，`std::array`也提供了`at`函数。

# 保持 std::vector 实例排序

数组和向量本身不会对它们的有效负载对象进行排序。但是如果我们需要这样做，并不意味着我们总是必须切换到自动执行排序的数据结构。如果`std::vector`非常适合我们的用例，那么以*排序方式*向其中添加项目仍然非常简单和实用。

# 如何做到...

在本节中，我们将用随机单词填充一个`std::vector`，对其进行排序，然后在保持向量排序单词顺序不变的同时插入更多单词。

1.  让我们首先包含我们将需要的所有头文件。

```cpp
      #include <iostream>
      #include <vector>
      #include <string>
      #include <algorithm>
      #include <iterator> 
      #include <cassert>
```

1.  我们还声明我们正在使用`std`命名空间，以避免一些`std::`前缀：

```cpp
      using namespace std;
```

1.  然后我们编写一个小的主函数，用一些随机字符串填充一个向量。

```cpp
      int main()
      {
          vector<string> v {"some", "random", "words", 
                            "without", "order", "aaa", 
                            "yyy"};
```

1.  接下来我们要做的是对该向量进行*排序*。在此之前，让我们使用 STL 的`is_sorted`函数和一些断言来检查向量在之前确实*没有*排序，但之后*已经*排序。

```cpp
          assert(false == is_sorted(begin(v), end(v)));
          sort(begin(v), end(v));
          assert(true == is_sorted(begin(v), end(v)));
```

1.  现在，我们最终使用一个新的`insert_sorted`函数将一些随机单词添加到排序后的向量中，之后我们仍然需要实现这个函数。这些单词应该放在正确的位置，以便向量在之后仍然是排序的：

```cpp
          insert_sorted(v, "foobar");
          insert_sorted(v, "zzz");
```

1.  因此，让我们现在在源文件中稍早实现`insert_sorted`。

```cpp
      void insert_sorted(vector<string> &v, const string &word)
      {
          const auto insert_pos (lower_bound(begin(v), end(v), word));
          v.insert(insert_pos, word);
      }
```

1.  现在回到我们停下的主函数中，我们现在可以继续打印向量，并看到插入过程的工作情况：

```cpp
          for (const auto &w : v) { 
              cout << w << " ";
          }
          cout << 'n';
      }
```

1.  编译和运行程序会产生以下很好排序的输出：

```cpp
      aaa foobar order random some without words yyy zzz
```

# 工作原理...

整个程序围绕`insert_sorted`函数构建，该函数执行本节所述的操作：对于任何新字符串，它定位排序向量中的位置，必须将其插入以*保持*向量中字符串的顺序。但是，我们假设向量在之前已经排序。否则，这将无法工作。

定位步骤由 STL 函数`lower_bound`完成，该函数接受三个参数。前两个参数表示底层范围的*开始*和*结束*。在这种情况下，范围是我们的单词向量。第三个参数是要插入的单词。然后函数找到范围中第一个*大于或等于*第三个参数的项目，并返回指向它的迭代器。

有了正确的位置，我们将其提供给`std::vector`成员方法`insert`，该方法只接受两个参数。第一个参数是一个迭代器，指向向量中应插入第二个参数的位置。我们可以使用刚刚从`lower_bound`函数中获得的相同迭代器，这似乎非常方便。第二个参数当然是要插入的项目。

# 还有更多...

`insert_sorted`函数非常通用。如果我们泛化其参数的类型，它也将适用于其他容器有效负载类型，甚至适用于其他容器，例如`std::set`、`std::deque`、`std::list`等等！（请注意，set 有自己的`lower_bound`成员函数，执行与`std::lower_bound`相同的操作，但效率更高，因为它专门为集合进行了优化。）

```cpp
template <typename C, typename T>
void insert_sorted(C &v, const T &item)
{
    const auto insert_pos (lower_bound(begin(v), end(v), item));
    v.insert(insert_pos, item);
}
```

当尝试从`std::vector`切换到其他类型的容器时，请注意并非所有容器都支持`std::sort`。该算法需要随机访问容器，例如`std::list`就不满足这个要求。

# 高效地和有条件地向 std::map 中插入项目

有时我们想要用键值对填充一个映射，并且在填充映射的过程中，可能会遇到两种不同的情况：

1.  关键尚不存在。创建一个*全新*的键值对。

1.  关键已经存在。获取*现有*项目并*修改*它。

我们可以简单地使用`map`的`insert`或`emplace`方法，并查看它们是否成功。如果不成功，我们就会遇到第二种情况，并修改现有的项目。在这两种情况下，insert 和 emplace 都会创建我们尝试插入的项目，而在第二种情况下，新创建的项目会被丢弃。在这两种情况下，我们都会得到一个无用的构造函数调用。

自 C++17 以来，有`try_emplace`函数，它使我们能够仅在插入时有条件地创建项目。让我们实现一个程序，该程序获取亿万富翁名单并构造一个告诉我们每个国家的亿万富翁数量的映射。除此之外，它还存储每个国家最富有的人。我们的示例不包含昂贵的创建项目，但是每当我们在现实项目中遇到这种情况时，我们都知道如何使用`try_emplace`来掌握它。

# 如何做...

在本节中，我们将实现一个应用程序，该应用程序从亿万富翁名单中创建一个映射。该映射将每个国家映射到该国最富有的人的引用以及告诉该国有多少亿万富翁的计数器。

1.  和往常一样，我们首先需要包含一些头文件，并声明我们默认使用`std`命名空间。

```cpp
      #include <iostream>
      #include <functional>
      #include <list>
      #include <map>

      using namespace std;
```

1.  让我们定义一个代表我们名单上亿万富翁物品的结构。

```cpp
      struct billionaire {
          string name;
          double dollars;
          string country;
      };
```

1.  在主函数中，我们首先定义亿万富翁名单。世界上有*很多*亿万富翁，所以让我们构建一个有限的名单，其中只包含一些国家中最富有的人。这个名单已经排序。排名实际上来自《福布斯》2017 年《世界亿万富翁》名单[`www.forbes.com/billionaires/list/:`](https://www.forbes.com/billionaires/list/)

```cpp
      int main()
      {
          list<billionaire> billionaires {
              {"Bill Gates", 86.0, "USA"},
              {"Warren Buffet", 75.6, "USA"},
              {"Jeff Bezos", 72.8, "USA"},
              {"Amancio Ortega", 71.3, "Spain"},
              {"Mark Zuckerberg", 56.0, "USA"},
              {"Carlos Slim", 54.5, "Mexico"},
              // ...
              {"Bernard Arnault", 41.5, "France"},
              // ...
              {"Liliane Bettencourt", 39.5, "France"},
              // ...
              {"Wang Jianlin", 31.3, "China"},
              {"Li Ka-shing", 31.2, "Hong Kong"}
              // ...
          };
```

1.  现在，让我们定义映射。它将国家字符串映射到一对。该对包含我们名单中每个国家的第一个亿万富翁的（`const`）副本。这自动是每个国家最富有的亿万富翁。对中的另一个变量是一个计数器，我们将为该国家的每个后续亿万富翁递增。

```cpp
          map<string, pair<const billionaire, size_t>> m;
```

1.  现在，让我们遍历列表，并尝试为每个国家插入一个新的有效负载对。该对包含我们当前正在查看的亿万富翁的引用和计数器值`1`。

```cpp
          for (const auto &b : billionaires) {
              auto [iterator, success] = m.try_emplace(b.country, b, 1);
```

1.  如果该步骤成功，那么我们就不需要做其他任何事情了。我们提供了构造函数参数`b, 1`的对已经被构造并插入到映射中。如果插入*不*成功，因为国家键已经存在，那么这对就不会被构造。如果我们的亿万富翁结构非常庞大，这将为我们节省复制它的运行时成本。

然而，在不成功的情况下，我们仍然需要递增该国家的计数器。

```cpp
              if (!success) {
                  iterator->second.second += 1;
              }
          }
```

1.  好的，就是这样。我们现在可以打印每个国家有多少亿万富翁，以及每个国家最富有的人是谁。

```cpp
          for (const auto & [key, value] : m) {
              const auto &[b, count] = value;

              cout << b.country << " : " << count 
                   << " billionaires. Richest is "
                   << b.name << " with " << b.dollars 
                   << " B$n";
          }
      }
```

1.  编译和运行程序产生以下输出。（当然，输出是有限的，因为我们限制了输入映射。）

```cpp
      $ ./efficient_insert_or_modify
      China : 1 billionaires. Richest is Wang Jianlin with 31.3 B$
      France : 2 billionaires. Richest is Bernard Arnault with 41.5 B$
      Hong Kong : 1 billionaires. Richest is Li Ka-shing with 31.2 B$
      Mexico : 1 billionaires. Richest is Carlos Slim with 54.5 B$
      Spain : 1 billionaires. Richest is Amancio Ortega with 71.3 B$
      USA : 4 billionaires. Richest is Bill Gates with 86 B$
```

# 它是如何工作的...

整个配方围绕着`std::map`的`try_emplace`函数展开，这是 C++17 的新功能。它具有以下签名：

```cpp
std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args);
```

因此，被插入的键是参数`k`，关联的值是从参数包`args`构造的。如果我们成功插入该项，那么函数将返回一个*迭代器*，该迭代器指向映射中的新节点，并与设置为`true`的布尔值*配对*。如果插入*不*成功，则返回对中的布尔值设置为`false`，并且迭代器指向新项将与之冲突的项。

这种特征在我们的情况下非常有用--当我们第一次看到来自特定国家的亿万富翁时，那么这个国家在映射中还不是一个键。在这种情况下，我们必须*插入*它，并附带将新计数器设置为`1`。如果我们已经看到来自特定国家的亿万富翁，我们必须获取对其现有计数器的引用，以便对其进行递增。这正是第 6 步发生的事情：

```cpp
if (!success) {
    iterator->second.second += 1;
}
```

请注意，`std::map`的`insert`和`emplace`函数的工作方式完全相同。一个关键的区别是，如果键已经存在，`try_emplace`将*不*构造与键关联的对象。这在类型的对象昂贵创建时提高了性能。

# 还有更多...

如果我们将地图的类型从`std::map`切换到`std::unordered_map`，整个程序仍然可以工作。这样，我们可以从一种实现简单地切换到另一种实现，它们具有不同的性能特征。在这个示例中，唯一可观察到的区别是，亿万富翁地图不再按字母顺序打印，因为哈希映射不像搜索树那样对对象进行排序。

# 了解 std::map::insert 的新插入提示语义。

在`std::map`中查找项目需要*O(log(n))*时间。对于插入新项目也是一样，因为必须查找插入它们的位置。因此，天真地插入*M*个新项目将需要*O(M * log(n))*的时间。

为了使这更有效，`std::map`插入函数接受一个可选的*插入提示*参数。插入提示基本上是一个迭代器，它指向即将插入的项目的未来位置附近。如果提示是正确的，那么我们就会得到*摊销*的*O(1)*插入时间。

# 如何做...

在本节中，我们将向`std::map`中插入多个项目，并为此使用插入提示，以减少查找次数。

1.  我们将字符串映射到数字，因此需要包含`std::map`和`std::string`的头文件。

```cpp
      #include <iostream>
      #include <map>
      #include <string>
```

1.  下一步是实例化一个地图，其中已经包含了一些示例字符。

```cpp
      int main()
      {
          std::map<std::string, size_t> m {{"b", 1}, {"c", 2}, {"d", 3}};
```

1.  现在我们将插入多个项目，并且对于每个项目，我们将使用插入提示。由于一开始我们没有提示可以使用，我们将首先插入指向地图的`end`迭代器。

```cpp
          auto insert_it (std::end(m));
```

1.  现在，我们将按字母表的顺序向地图中插入项目，始终使用我们拥有的迭代器提示，然后将其重新初始化为`insert`函数的返回值。下一个项目将被插入到提示的*前面*。

```cpp
          for (const auto &s : {"z", "y", "x", "w"}) {
              insert_it = m.insert(insert_it, {s, 1});
          }
```

1.  为了展示*不*应该这样做，我们插入一个字符串，它将被放在地图中最左边的位置，但给它一个完全*错误*的提示，它指向地图中最右边的位置——`end`。

```cpp
          m.insert(std::end(m), {"a", 1});
```

1.  最后，我们只是打印我们拥有的东西。

```cpp
          for (const auto & [key, value] : m) {
              std::cout << """ << key << "": " << value << ", ";
          }
          std::cout << 'n';
      }
```

1.  当我们编译和运行程序时，这是我们得到的输出。显然，错误的插入提示并没有造成太大的伤害，因为地图的顺序仍然是正确的。

```cpp
      "a": 1, "b": 1, "c": 2, "d": 3, "w": 1, "x": 1, "y": 1, "z": 1,
```

# 它是如何工作的...

在这个示例中，与普通地图插入的唯一区别是额外的提示迭代器。我们谈到了*正确*和*错误*的提示。

*正确*的提示将指向一个现有元素，该元素*大于*要插入的元素，以便新插入的键将刚好在提示*之前*。如果这不适用于用户在插入时提供的提示，插入函数将退回到非优化的插入，再次产生*O(log(n))*的性能。

对于第一次插入，我们得到了地图的`end`迭代器，因为我们没有更好的提示可以使用。在树中安装了一个“z”之后，我们知道安装“y”将在“z”的前面插入一个新项目，这使它成为一个正确的提示。如果在插入“y”之后将“x”放入树中，也是如此。这就是为什么可以使用由*上次*插入返回的迭代器进行*下次*插入。

重要的是要知道，在 C++11 之前，插入提示被认为是正确的，当它们指向新插入的项目的位置*之前*时。

# 还有更多...

有趣的是，错误的提示甚至不会破坏或干扰地图中项目的顺序，那么这是如何工作的，这意味着什么，插入时间是摊销*O(1)*吗？

`std::map`通常使用二叉搜索树实现。将新键插入搜索树时，将其与其他节点的键进行比较，从顶部开始。如果键比一个节点的键小或大，那么搜索算法将向左或向右分支，以进入下一个更深的节点。在这样做的同时，搜索算法将在达到当前树的最大深度的地方停止，在那里将新节点与其键放置。这一步可能破坏了树的平衡，因此之后也会使用重新平衡算法来纠正这一点，作为一项日常任务。

当我们将具有直接相邻键值的项目插入树中时（就像整数`1`是整数`2`的邻居一样，因为它们之间没有其他整数），它们通常也可以被插入到树中的相邻位置。可以轻松检查某个键和相应提示是否适用这种情况。如果适用，搜索算法步骤可以省略，这可以节省一些关键的运行时间。之后，重新平衡算法可能仍然需要运行。

当这样的优化通常可以完成，但并非总是如此时，这仍然可能导致平均性能提升。可以展示出在多次插入后稳定下来的*结果*运行时复杂度，然后称之为**摊销复杂度**。

![](img/ba7cd62b-4541-4793-9475-24b490c8929b.png)

如果插入提示错误，插入函数将简单地放弃提示，并重新使用搜索算法开始。这样做是正确的，但显然会更慢。

# 高效地修改 std::map 项的键

由于`std::map`数据结构以一种使键始终唯一且排序的方式映射到值，因此用户无法修改已插入的地图节点的键是至关重要的。为了防止用户修改完全排序的地图节点的键项，将`const`限定符添加到键类型中。

这种限制是完全合理的，因为它使用户更难以错误使用`std::map`。但是，如果我们真的需要更改一些映射项的键，我们该怎么办呢？

在 C++17 之前，我们必须从树中删除需要更改键值的项目，然后重新插入它们。这种方法的缺点是这总是不必要地重新分配一些内存，这在性能方面听起来很糟糕。

自 C++17 以来，我们可以删除和重新插入地图节点而不进行任何内存重新分配。我们将在本教程中看到它是如何工作的。

# 如何做...

我们实现了一个小应用程序，它以`std::map`结构对虚构比赛中的驾驶员的位置进行排序。在比赛中，当驾驶员相互超越时，我们需要更改他们的位置键，这是我们以新的 C++17 方式做的。

1.  让我们首先包括必要的头文件，并声明我们使用`std`命名空间。

```cpp
      #include <iostream>
      #include <map>      

      using namespace std;
```

1.  我们将在操纵地图结构之前和之后打印比赛名次，因此让我们为此实现一个小助手函数。

```cpp
      template <typename M>
      void print(const M &m)
      {
          cout << "Race placement:n";
          for (const auto &[placement, driver] : m) {
              cout << placement << ": " << driver << 'n';
          }
      }
```

1.  在主函数中，我们实例化和初始化一个映射，将整数值映射到包含驾驶员姓名的字符串。我们还打印地图，因为我们将在接下来的步骤中对其进行修改。

```cpp
      int main()
      {
          map<int, string> race_placement {
              {1, "Mario"}, {2, "Luigi"}, {3, "Bowser"},
              {4, "Peach"}, {5, "Yoshi"}, {6, "Koopa"},
              {7, "Toad"}, {8, "Donkey Kong Jr."}
          };

          print(race_placement);
```

1.  假设在一圈比赛中，鲍泽发生了一点小事故，掉到了最后一名，唐克·孔·朱尼尔趁机从最后一名跳到第三名。在这种情况下，我们首先需要从地图中提取它们的地图节点，因为这是操纵它们的键的唯一方法。`extract`函数是 C++17 的新功能。它可以从地图中删除项目而不产生任何与分配相关的副作用。让我们为这个任务打开一个新的范围。

```cpp
          {
              auto a (race_placement.extract(3));
              auto b (race_placement.extract(8));
```

1.  现在我们可以交换 Bowser 和 Donkey Kong Jr.的键。虽然地图节点的键通常是不可变的，因为它们被声明为`const`，但我们可以修改使用`extract`方法提取的项目的键。

```cpp
              swap(a.key(), b.key());
```

1.  在 C++17 中，`std::map`的`insert`方法得到了一个新的重载，可以接受提取节点的句柄，以便在不触及分配器的情况下插入它们。

```cpp
              race_placement.insert(move(a));
              race_placement.insert(move(b));
          }
```

1.  离开作用域后，我们完成了。我们打印新的比赛排名，然后让应用程序终止。

```cpp
          print(race_placement);
      }
```

1.  编译和运行程序产生以下输出。我们首先在新的地图实例中看到了比赛排名，然后在交换 Bowser 和 Donkey Kong Jr.的位置后再次看到它。

```cpp
      $ ./mapnode_key_modification 
      Race placement:
      1: Mario
      2: Luigi
      3: Bowser
      4: Peach
      5: Yoshi
      6: Koopa
      7: Toad
      8: Donkey Kong Jr.
      Race placement:
      1: Mario
      2: Luigi
      3: Donkey Kong Jr.
      4: Peach
      5: Yoshi
      6: Koopa
      7: Toad
      8: Bowser
```

# 工作原理...

在 C++17 中，`std::map`获得了一个新的成员函数 extract。它有两种形式：

```cpp
node_type extract(const_iterator position);
node_type extract(const key_type& x);
```

在本示例中，我们使用了第二种方法，它接受一个键，然后查找并提取与键参数匹配的地图节点。第一个方法接受一个迭代器，这意味着它*更快*，因为它不需要搜索项目。

如果我们尝试使用第二种方法（使用键进行搜索）提取不存在的项目，则会返回一个*空*的`node_type`实例。`empty()`成员方法返回一个布尔值，告诉我们`node_type`实例是否为空。访问空实例上的任何其他方法会导致未定义的行为。

在提取节点之后，我们能够使用`key()`方法修改它们的键，这为我们提供了对键的非 const 访问，尽管键通常是 const 的。

请注意，为了重新将节点插入地图中，我们必须将它们*移动*到`insert`函数中。这是有道理的，因为`extract`的目的是避免不必要的复制和分配。请注意，虽然我们移动了一个`node_type`实例，但这并不会导致任何容器值的实际移动。

# 还有更多...

使用提取方法提取的地图节点实际上非常灵活。我们可以从`map`实例中提取节点并将其插入到任何其他`map`甚至`multimap`实例中。它也可以在`unordered_map`和`unordered_multimap`实例之间，以及`set`/`multiset`和相应的`unordered_set`/`unordered_multiset`之间工作。

为了在不同的地图/集合结构之间移动项目，键、值和分配器的类型需要相同。请注意，即使是这种情况，我们也不能从`map`移动节点到`unordered_map`，或者从`set`移动节点到`unordered_set`。

# 使用自定义类型的 std::unordered_map

如果我们使用`std::unordered_map`而不是`std::map`，我们可以对要使用的键类型进行不同程度的自由选择。`std::map`要求所有键项之间存在自然顺序。这样，项目可以排序。但是，如果我们想要，例如，将数学向量作为键类型呢？对于这种类型，没有*较小*`<`关系是没有*意义*的，因为向量`(0, 1)`不比`(1, 0)`*小*或*大*。它们只是指向不同的方向。这对于`std::unordered_map`来说完全没问题，因为它不会通过它们的较小/较大的顺序关系来区分项目，而是通过*哈希值*。我们唯一需要做的就是为我们自己的类型实现一个*哈希函数*，以及一个*相等*的`==`运算符实现，告诉我们两个对象是否相同。本节将通过一个示例来演示这一点。

# 如何做...

在本节中，我们将定义一个简单的`coord`结构，它没有*默认*哈希函数，因此我们需要自己定义它。然后我们通过将`coord`值映射到数字来使用它。

1.  我们首先包含了打印和使用`std::unordered_map`所需的内容。

```cpp
      #include <iostream>
      #include <unordered_map>
```

1.  然后我们定义了我们自己的自定义结构，它不是通过*现有*哈希函数轻松哈希的：

```cpp
      struct coord {
          int x;
          int y;
      };
```

1.  我们不仅需要一个哈希函数才能将结构用作哈希映射的键，它还需要一个比较运算符的实现：

```cpp
      bool operator==(const coord &l, const coord &r)
      {
          return l.x == r.x && l.y == r.y;
      }
```

1.  为了扩展 STL 自己的哈希能力，我们将打开`std`命名空间，并创建我们自己的`std::hash`模板结构专门化。它包含与其他哈希专门化相同的`using`类型别名子句。

```cpp
      namespace std
      {

      template <>
      struct hash<coord>
      {
          using argument_type = coord;
          using result_type   = size_t;
```

1.  这个`struct`的核心是`operator()`的定义。我们只是添加了`struct coord`的数值成员值，这是一种较差的哈希技术，但为了展示如何实现它，这已经足够了。一个好的哈希函数试图尽可能均匀地分布值在整个值范围内，以减少*哈希冲突*的数量。

```cpp
          result_type operator()(const argument_type &c) const
          {
              return static_cast<result_type>(c.x) 
                   + static_cast<result_type>(c.y);
          }
      };

      }
```

1.  现在我们可以实例化一个新的`std::unordered_map`实例，它接受`struct coord`实例作为键，并将其映射到任意值。由于这个方法是关于使我们自己的类型适用于`std::unordered_map`，这已经足够了。让我们用我们自己的类型实例化一个基于哈希的映射，填充它一些项目，并打印它的：

```cpp
      int main()
      {

          std::unordered_map<coord, int> m {{{0, 0}, 1}, {{0, 1}, 2}, 
                                            {{2, 1}, 3}};

          for (const auto & [key, value] : m) {
              std::cout << "{(" << key.x << ", " << key.y 
                        << "): " << value << "} ";
          }
          std::cout << 'n';
      }
```

1.  编译和运行程序产生了以下输出：

```cpp
      $ ./custom_type_unordered_map
      {(2, 1): 3} {(0, 1): 2} {(0, 0): 1}
```

# 它是如何工作的...

通常，当我们实例化一个基于哈希的映射实现，比如`std::unordered_map`时，我们会写：

```cpp
std::unordered_map<key_type, value_type> my_unordered_map;
```

当编译器创建我们的`std::unordered_map`专门化时，背后发生了很多魔法，这并不太明显。因此，让我们来看一下它的完整模板类型定义：

```cpp
template<
    class Key,
    class T,
    class Hash      = std::hash<Key>,
    class KeyEqual  = std::equal_to<Key>,
    class Allocator = std::allocator< std::pair<const Key, T> >
> class unordered_map;
```

前两个模板类型是我们用`coord`和`int`填充的，这是简单和明显的部分。另外三个模板类型是可选的，因为它们会自动填充现有的标准模板类，这些类本身采用模板类型。这些类以我们对前两个参数的选择作为默认值。

关于这个方法，`class Hash`模板参数是有趣的：当我们没有明确定义其他任何东西时，它将专门化为`std::hash<key_type>`。STL 已经包含了许多类型的`std::hash`专门化，比如`std::hash<std::string>`，`std::hash<int>`，`std::hash<unique_ptr>`等等。这些类知道如何处理这些特定类型，以计算出最佳的哈希值。

然而，STL 并不知道如何从我们的`struct coord`计算哈希值。因此，我们所做的是定义*另一个*专门化，它知道如何处理它。编译器现在可以遍历它所知道的所有`std::hash`专门化列表，并找到我们的实现来匹配我们提供的键类型。

如果我们没有添加一个新的`std::hash<coord>`专门化，并将其命名为`my_hash_type`，我们仍然可以使用以下实例化行：

```cpp
std::unordered_map<coord, value_type, my_hash_type> my_unordered_map;
```

这显然需要输入更多的内容，而且不像编译器自己找到正确的哈希实现那样容易阅读。

# 从用户输入中过滤重复项并按字母顺序打印它们与 std::set

`std::set`是一个奇怪的容器：它的工作方式有点像`std::map`，但它只包含键作为值，没有键值对。因此，它几乎不能用作将一种类型的值映射到另一种类型的值。看起来，只是因为它的用例不太明显，很多开发人员甚至不知道它的存在。然后他们开始自己实现东西，尽管`std::set`在其中的一些情况下会非常有帮助。

这一部分展示了如何在一个示例中使用`std::set`，在这个示例中，我们收集了许多不同的项目，以*过滤*它们并输出*唯一*的选择。

# 如何做...

在这一部分，我们将从标准输入中读取一系列单词。所有*唯一*的单词都被放入一个`std::set`实例中。这样我们就可以列举出流中的所有唯一单词。

1.  我们将使用多种不同的 STL 类型，因此需要包含多个头文件。

```cpp
      #include <iostream>
      #include <set>
      #include <string>
      #include <iterator>
```

1.  为了节省一些输入，我们将声明我们正在使用`std`命名空间：

```cpp
      using namespace std;
```

1.  现在我们已经开始编写实际的程序，它以`main`函数实例化一个存储字符串的`std::set`开始。

```cpp
      int main()
      {
          set<string> s;
```

1.  接下来要做的事情是获取用户输入。我们只需从标准输入读取，并使用方便的`istream_iterator`。

```cpp
          istream_iterator<string> it {cin};
          istream_iterator<string> end;
```

1.  拥有一对`begin`和`end`迭代器，代表用户输入，我们可以使用`std::inserter`从中填充集合。

```cpp
          copy(it, end, inserter(s, s.end()));
```

1.  就是这样。为了看到我们从标准输入得到的*独特*单词，我们只需打印我们集合的内容。

```cpp
          for (const auto word : s) {
              cout << word << ", ";
          }
          cout << 'n';
      }
```

1.  让我们用以下输入编译和运行我们的程序。对于前面的输入，我们得到以下输出，其中所有重复项都被剔除，而独特的单词按字母顺序排序。

```cpp
      $ echo "a a a b c foo bar foobar foo bar bar" | ./program
      a, b, bar, c, foo, foobar,
```

# 它是如何工作的...

这个程序由两个有趣的部分组成。第一部分是使用`std::istream_iterator`来访问用户输入，第二部分是将其与我们的`std::set`实例结合起来，使用`std::copy`算法，然后将其包装成`std::inserter`实例！也许令人惊讶的是，只有一行代码就可以完成*标记化*输入、将其放入按字母顺序*排序*的集合中，并*删除*所有重复项的所有工作。

# std::istream_iterator

这个类在我们想要从流中处理大量*相同*类型的数据时非常有趣，这正是这个示例的情况：我们逐个单词解析整个输入，并将其以`std::string`实例的形式放入集合中。

`std::istream_iterator`接受一个模板参数。那就是我们想要的输入类型。我们选择了`std::string`，因为我们假设是文本单词，但也可以是`float`数字，例如。基本上可以是任何可以写成`cin >> var;`的类型。构造函数接受一个`istream`实例。标准输入由全局输入流对象`std::cin`表示，在这种情况下是一个可接受的`istream`参数。

```cpp
istream_iterator<string> it {cin};
```

我们实例化的输入流迭代器`it`能够做两件事：当它被解引用(`*it`)时，它会产生当前的输入符号。由于我们通过模板参数将迭代器类型化为`std::string`，所以该符号将是一个包含一个单词的字符串。当它被增加(`++it`)时，它将跳到下一个单词，我们可以通过再次解引用来访问它。

但是等等，在我们再次解引用之前，我们需要在每次增量之后小心。如果标准输入为空，迭代器就不应该再次被解引用。相反，我们应该终止我们解引用迭代器以获取每个单词的循环。让我们知道迭代器变得无效的中止条件是与`end`迭代器的比较。如果`it == end`成立，我们就超出了输入的末尾。

我们通过使用其无参数标准构造函数创建`std::istream_iterator`实例来创建结束迭代器。它的目的是作为每次迭代中的中止条件的对应物：

```cpp
istream_iterator<string> end;
```

一旦`std::cin`为空，我们的`it`迭代器实例将*注意到*并与`end`进行比较，返回`true`。

# std::inserter

我们在`std::copy`调用中使用`it`和`end`对作为*输入*迭代器。第三个参数必须是一个*输出*迭代器。对于这一点，我们不能只取`s.begin()`或`s.end()`。在一个空集合中，两者是相同的，所以我们甚至不能*解引用*它，无论是用于从中读取还是分配给它。

这就是`std::inserter`发挥作用的地方。它是一个返回`std::insert_iterator`的函数，它的行为类似于迭代器，但做的事情与通常的迭代器不同。当我们增加它时，它什么也不做。当我们解引用它并将某物赋给它时，它将取得它所附属的容器，并将该值作为*新*项插入其中！

通过`std::inserter`实例化`std::insert_iterator`需要两个参数：

```cpp
auto insert_it = inserter(s, s.end());
```

`s`是我们的集合，`s.end()`是一个迭代器，指向新项应该插入的位置。对于我们开始的空集合，这和`s.begin()`一样有意义。当用于其他数据结构如向量或列表时，第二个参数对于定义插入迭代器应该插入新项的位置至关重要。

# 将它放在一起

最后，*所有*的操作都发生在`std::copy`调用期间：

```cpp
copy(input_iterator_begin, input_iterator_end, insert_iterator);
```

这个调用从`std::cin`中通过输入迭代器取出下一个单词标记，并将其推入我们的`std::set`中。然后，它递增两个迭代器，并检查输入迭代器是否等于输入结束迭代器的对应项。如果不相等，那么标准输入中仍然有单词，所以它将*重复*。

重复的单词会自动被丢弃。如果集合已经包含特定单词，再次添加它将*没有效果*。这在`std::multiset`中是不同的，因为它会接受重复项。

# 使用 std::stack 实现一个简单的逆波兰表示法计算器

`std::stack`是一个适配器类，它允许用户像在真正的对象堆栈上一样将对象推入它，然后再从中弹出对象。在这一部分，我们围绕这个数据结构构建了一个逆波兰表示法（RPN）计算器，以展示如何使用它。

逆波兰表示法是一种可以用来以非常简单的方式解析数学表达式的表示法。在逆波兰表示法中，`1 + 2`表示为`1 2 +`。首先是操作数，然后是操作符。另一个例子：`(1 + 2) * 3`在逆波兰表示法中是`1 2 + 3 *`，这已经显示了为什么它更容易解析，因为我们不需要使用括号来定义子表达式。

![](img/c5365787-5e7f-4fab-afe2-ad3ae977ddb5.jpg)

# 如何做...

在这一部分，我们将从标准输入中读取一个逆波兰表示法的数学表达式，然后将其传递给一个评估函数。最后，我们将数值结果打印回给用户。

1.  我们将使用 STL 中的许多辅助函数，所以有一些包含：

```cpp
      #include <iostream>
      #include <stack>
      #include <iterator>
      #include <map>
      #include <sstream>
      #include <cassert>
      #include <vector>
      #include <stdexcept>
      #include <cmath>
```

1.  我们还声明我们使用`std`命名空间，以节省一些输入。

```cpp
      using namespace std;
```

1.  然后，我们立即开始实现我们的逆波兰表示法解析器。它将接受一个迭代器对，表示以字符串形式的数学表达式的开始和结束，这将逐个标记消耗掉。

```cpp
      template <typename IT>
      double evaluate_rpn(IT it, IT end)
      {
```

1.  当我们遍历标记时，我们需要记住一路上的所有*操作数*，直到看到一个*操作符*。这就是我们需要一个堆栈的地方。所有的数字都将被解析并保存为双精度浮点数，所以它将是一个`double`值的堆栈。

```cpp
          stack<double> val_stack;
```

1.  为了方便地访问堆栈上的元素，我们实现了一个辅助函数。它通过从堆栈中取出最高项来修改堆栈，然后返回该项。这样我们可以在以后的一个步骤中执行这个任务。

```cpp
          auto pop_stack ([&](){ 
              auto r (val_stack.top()); 
              val_stack.pop(); 
              return r; 
          });
```

1.  另一个准备工作是定义所有支持的数学运算。我们将它们保存在一个映射中，将每个操作标记与实际操作关联起来。这些操作由可调用的 lambda 表示，它们接受两个操作数，例如相加或相乘，然后返回结果。

```cpp
          map<string, double (*)(double, double)> ops {
              {"+", [](double a, double b) { return a + b; }},
              {"-", [](double a, double b) { return a - b; }},
              {"*", [](double a, double b) { return a * b; }},
              {"/", [](double a, double b) { return a / b; }},
              {"^", [](double a, double b) { return pow(a, b); }},
              {"%", [](double a, double b) { return fmod(a, b); }},
          };
```

1.  现在我们终于可以遍历输入了。假设输入迭代器给我们的是字符串，我们为每个标记提供一个新的`std::stringstream`，因为它可以解析数字。

```cpp
          for (; it != end; ++it) {
              stringstream ss {*it};
```

1.  现在对于每个标记，我们尝试从中获取一个`double`值。如果成功，我们就有了*操作数*，我们将其推入堆栈。

```cpp
              if (double val; ss >> val) {
                  val_stack.push(val);
              }
```

1.  如果它*不*成功，那么它必须是其他东西而不是操作符；在这种情况下，它只能是*操作数*。知道我们支持的所有操作都是*二元*的，我们需要从堆栈中弹出最后的*两个*操作数。

```cpp
              else {
                  const auto r {pop_stack()};
                  const auto l {pop_stack()};
```

1.  现在我们从解引用迭代器`it`中获取操作数，它已经发出了字符串。通过查询`ops`映射，我们得到一个接受两个操作数`l`和`r`作为参数的 lambda 对象。

```cpp
                  try {
                      const auto & op     (ops.at(*it));
                      const double result {op(l, r)};
                      val_stack.push(result);
                  }
```

1.  我们用`try`子句包围了数学部分的应用，这样我们就可以捕获可能发生的异常。映射的`at`调用将在用户提供我们不知道的数学操作时抛出`out_of_range`异常。在这种情况下，我们将重新抛出一个不同的异常，该异常说`invalid argument`并携带了我们不知道的操作字符串。

```cpp
                  catch (const out_of_range &) {
                      throw invalid_argument(*it);
                  }
```

1.  这就是全部。一旦循环终止，我们就在堆栈上得到了最终结果。所以我们就返回那个。 （在这一点上，我们可以断言堆栈大小是否为 1。如果不是，那么就会缺少操作。）

```cpp
              }
          }

          return val_stack.top();
      }
```

1.  现在我们可以使用我们的小 RPN 解析器。为了做到这一点，我们将标准输入包装成一个`std::istream_iterator`对，并将其传递给 RPN 解析器函数。最后，我们打印结果：

```cpp
      int main()
      {
          try {
              cout << evaluate_rpn(istream_iterator<string>{cin}, {}) 
                   << 'n';
          }
```

1.  我们再次将该行包装到`try`子句中，因为仍然有可能用户输入包含我们没有实现的操作。在这种情况下，我们必须捕获我们在这种情况下抛出的异常，并打印错误消息：

```cpp
          catch (const invalid_argument &e) {
              cout << "Invalid operator: " << e.what() << 'n';
          }
      }
```

1.  编译程序后，我们可以尝试一下。输入`"3 1 2 + * 2 /"`代表表达式`( 3 * (1 + 2) ) / 2`，并产生了正确的结果：

```cpp
      $ echo "3 1 2 + * 2 /" | ./rpn_calculator
      4.5
```

# 它是如何工作的...

整个算法围绕着将操作数推送到堆栈上直到我们在输入中看到一个操作。在这种情况下，我们再次从堆栈中弹出最后两个操作数，对它们应用操作，然后再次将结果推送到堆栈上。为了理解这个算法中的所有代码，重要的是要理解我们如何从输入中区分*操作数*和*操作*，如何处理我们的堆栈，以及如何选择和应用正确的数学操作。

# 堆栈处理

我们将项目推送到堆栈上，只需使用`std::stack`的`push`函数：

```cpp
val_stack.push(val);
```

从中弹出值看起来有点复杂，因为我们为此实现了一个 lambda，它捕获了对`val_stack`对象的引用。让我们看看相同的代码，增加一些注释：

```cpp
auto pop_stack ([&](){
    auto r (val_stack.top()); // Get top value copy
    val_stack.pop();          // Throw away top value
    return r;                 // Return copy
});
```

这个 lambda 是必要的，以便一步获取堆栈的顶部值并从中*删除*它。`std::stack`的接口设计并不允许在*单个*调用中执行此操作。但是，定义一个 lambda 很快很容易，所以我们现在可以这样获取值：

```cpp
double top_value {pop_stack()};
```

# 从用户输入中区分操作数和操作

在`evaluate_rpn`的主循环中，我们从迭代器中获取当前的字符串标记，然后查看它是否是操作数。如果字符串可以解析为`double`变量，那么它就是一个数字，因此也是一个操作数。我们认为所有不能轻松解析为数字的东西（例如`"+"`）都是*操作*。

用于这个任务的裸代码框架如下：

```cpp
stringstream ss {*it};
if (double val; ss >> val) {
    // It's a number!
} else {
    // It's something else than a number - an operation!
}
```

流操作符`>>`告诉我们它是否是一个数字。首先，我们将字符串包装到`std::stringstream`中。然后我们使用`stringstream`对象的能力从`std::string`流到`double`变量，这涉及解析。如果解析*失败*，我们知道它是因为我们要求它将某些东西解析为一个数字，而这不是一个数字。

# 选择和应用正确的数学操作

在我们意识到当前用户输入标记不是一个数字之后，我们只是假设它是一个操作，比如`+`或`*`。然后我们查询我们称为`ops`的映射，查找该操作并返回一个函数，该函数接受两个操作数，并返回总和，或乘积，或适当的其他内容。

映射本身的类型看起来相对复杂：

```cpp
map<string, double (*)(double, double)> ops { ... };
```

它从`string`映射到`double (*)(double, double)`。后者是什么意思？这种类型描述应该读作“*指向一个接受两个 double 并返回一个 double 的函数的指针*”。想象一下，`(*)`部分就是函数的名称，比如`double sum(double, double)`，这样就更容易阅读。这里的技巧是，我们的 lambda `[](double, double) { return /* some double */ }` 可以转换为实际匹配该指针描述的函数指针。通常不捕获任何内容的 lambda 都可以转换为函数指针。

这样，我们可以方便地向映射询问正确的操作：

```cpp
const auto & op     (ops.at(*it));
const double result {op(l, r)};
```

映射隐式地为我们做了另一项工作：如果我们说`ops.at("foo")`，那么`"foo"`是一个有效的键值，但我们没有存储任何名为这样的操作。在这种情况下，映射将抛出一个异常，我们在配方中捕获它。每当我们捕获它时，我们重新抛出一个不同的异常，以便提供对这种错误情况的描述性含义。用户将更清楚地知道`无效参数`异常意味着什么，而不是`超出范围`异常。请注意，`evaluate_rpn`函数的用户可能没有阅读其实现，因此可能不知道我们根本在内部使用映射。

# 还有更多...

由于`evaluate_rpn`函数接受迭代器，因此很容易用不同于标准输入流的输入来提供输入。这使得测试或适应不同的用户输入来源非常容易。

例如，通过从字符串流或字符串向量中使用迭代器进行输入，看起来像以下代码，`evaluate_rpn`根本不需要更改：

```cpp
int main()
{
    stringstream s {"3 2 1 + * 2 /"};
    cout << evaluate_rpn(istream_iterator<string>{s}, {}) << 'n';

    vector<string> v {"3", "2", "1", "+", "*", "2", "/"};
    cout << evaluate_rpn(begin(v), end(v)) << 'n';
}
```

在合适的地方使用迭代器。这样可以使您的代码非常可组合和可重用。

# 使用`std::map`实现单词频率计数器

`std::map`在对数据进行统计时非常有用。通过将可修改的有效负载对象附加到表示对象类别的每个键上，可以很容易地实现例如单词频率的直方图。这就是我们将在本节中做的事情。

# 如何做到...

在这一部分，我们将从标准输入中读取所有用户输入，例如可能是包含文章的文本文件。我们将输入标记化为单词，以便统计每个单词出现的次数。

1.  和往常一样，我们需要包括我们将要使用的数据结构的所有头文件。

```cpp
      #include <iostream>
      #include <map> 
      #include <vector> 
      #include <algorithm> 
      #include <iomanip>
```

1.  为了节省一些输入，我们声明使用`std`命名空间。

```cpp
      using namespace std;
```

1.  我们将使用一个辅助函数来裁剪可能附加的逗号、句号或冒号。

```cpp
      string filter_punctuation(const string &s)
      {
          const char *forbidden {".,:; "};
          const auto  idx_start (s.find_first_not_of(forbidden));
          const auto  idx_end   (s.find_last_not_of(forbidden));

          return s.substr(idx_start, idx_end - idx_start + 1);
      }
```

1.  现在我们开始实际的程序。我们将收集一个映射，将我们看到的每个单词与该单词频率的计数器关联起来。此外，我们还维护一个记录迄今为止我们见过的最长单词的大小的变量，这样当我们在程序结束时打印单词频率表时，我们可以很好地缩进它。

```cpp
      int main()
      {
          map<string, size_t> words;
          int max_word_len {0};
```

1.  当我们从`std::cin`流入一个`std::string`变量时，输入流会在途中去除空格。这样我们就可以逐个单词获取输入。

```cpp
          string s;
          while (cin >> s) {
```

1.  现在我们所拥有的单词，可能包含逗号、句号或冒号，因为它可能出现在句子的结尾或类似位置。我们使用之前定义的辅助函数来过滤掉这些。

```cpp
              auto filtered (filter_punctuation(s));
```

1.  如果这个单词是迄今为止最长的单词，我们需要更新`max_word_len`变量。

```cpp
              max_word_len = max<int>(max_word_len, filtered.length());
```

1.  现在我们将增加`words`映射中该单词的计数值。如果它是第一次出现，我们会在增加之前隐式地创建它。

```cpp
              ++words[filtered];
          }
```

1.  循环结束后，我们知道我们已经在`words`映射中保存了输入流中的所有唯一单词，并与表示每个单词频率的计数器配对。映射使用单词作为键，并按它们的*字母*顺序排序。我们想要的是按*频率*排序打印所有单词，因此频率最高的单词应该首先出现。为了实现这一点，我们首先实例化一个向量，将所有这些单词频率对放入其中，并将它们从映射移动到向量中。

```cpp
          vector<pair<string, size_t>> word_counts;
          word_counts.reserve(words.size());
          move(begin(words), end(words), back_inserter(word_counts));
```

1.  现在向量仍然以与`words`映射维护它们相同的顺序包含所有单词频率对。现在我们再次对其进行排序，以便将最频繁出现的单词放在开头，将最不频繁的单词放在末尾。

```cpp
          sort(begin(word_counts), end(word_counts),
              [](const auto &a, const auto &b) { 
                  return a.second > b.second; 
              });
```

1.  现在所有数据都已经排序好了，所以我们将其推送到用户终端。使用`std::setw`流操作符，我们以漂亮的缩进格式格式化数据，使其看起来有点像表格。

```cpp
          cout << "# " << setw(max_word_len) << "<WORD>" << " #<COUNT>n";
          for (const auto & [word, count] : word_counts) {
              cout << setw(max_word_len + 2) << word << " #" 
                   << count << 'n';
          }
      }
```

1.  编译程序后，我们可以将任何文本文件输入到其中以获得频率表。

```cpp
      $ cat lorem_ipsum.txt | ./word_frequency_counter
      #       <WORD> #<COUNT>
                  et #574
               dolor #302
                 sed #273
                diam #273
                 sit #259
               ipsum #259
      ...
```

# 它是如何工作的...

这个方法集中在收集所有单词到`std::map`中，然后将所有项目从映射中推出并放入`std::vector`中，然后以不同的方式进行排序，以便打印数据。为什么？

让我们来看一个例子。当我们统计字符串`"a a b c b b b d c c"`中的单词频率时，我们会得到以下的映射内容：

```cpp
a -> 2
b -> 4
c -> 3
d -> 1
```

然而，这不是我们想要向用户展示的顺序。程序应该首先打印`b`，因为它的频率最高。然后是`c`，然后是`a`，最后是`d`。不幸的是，我们无法要求映射给我们“*具有最高关联值的键*”，然后是“*具有第二高关联值的键*”，依此类推。

在这里，向量就派上用场了。我们将向量定义为包含字符串和计数器值对的对。这样它可以以与映射中的形式完全相同的形式保存项目。

```cpp
vector<pair<string, size_t>> word_counts;
```

然后我们使用`std::move`算法填充向量，使用单词频率对。这样做的好处是，保存在堆上的字符串部分不会被复制，而是从映射移动到向量中。这样我们就可以避免大量的复制。

```cpp
move(begin(words), end(words), back_inserter(word_counts));
```

一些 STL 实现使用了短字符串优化--如果字符串不太长，它将*不会*被分配到堆上，而是直接存储在字符串对象中。在这种情况下，移动并不更快。但移动也永远不会更慢！

下一个有趣的步骤是排序操作，它使用 lambda 作为自定义比较运算符：

```cpp
sort(begin(word_counts), end(word_counts),
        [](const auto &a, const auto &b) { return a.second > b.second; });
```

排序算法将成对地取出项目，并进行比较，这就是排序算法的工作原理。通过提供 lambda 函数，比较不仅仅是比较`a`是否小于`b`（这是默认实现），还比较`a.second`是否大于`b.second`。请注意，所有对象都是*字符串*和它们的计数器值的对，通过写`a.second`我们可以访问单词的计数器值。这样我们就将所有高频单词移动到向量的开头，将低频单词移动到向量的末尾。

# 实现一个写作风格辅助工具，用于在文本中查找非常长的句子，使用 std::multimap

每当需要以排序方式存储大量项目，并且它们按照键进行排序的时候，`std::multimap`是一个不错的选择。

让我们找一个例子使用情况：在德语写作中，使用非常长的句子是可以的。但在英语写作中，是*不可以*的。我们将实现一个工具，帮助德语作者分析他们的英语文本文件，重点关注所有句子的长度。为了帮助作者改进文本风格，它将根据句子的长度对句子进行分组。这样作者就可以选择最长的句子并将其拆分。

# 如何做...

在本节中，我们将从标准输入中读取所有用户输入，我们将通过整个句子而不是单词对其进行标记化。然后我们将所有句子收集到一个`std::multimap`中，并与其长度一起输出给用户。然后，我们将所有句子按其长度排序后返回给用户。

1.  像往常一样，我们需要包括所有需要的头文件。`std::multimap`来自与`std::map`相同的头文件。

```cpp
      #include <iostream>
      #include <iterator>
      #include <map>
      #include <algorithm>
```

1.  我们使用了很多来自`std`命名空间的函数，因此我们自动声明其使用。

```cpp
      using namespace std;
```

1.  当我们通过提取文本中句号之间的内容来对字符串进行标记化时，我们将得到由空格（如空格、换行符等）包围的文本句子。这些会以错误的方式增加它们的大小，因此我们使用一个辅助函数来过滤它们，现在我们定义它。

```cpp
      string filter_ws(const string &s)
      {
          const char *ws {" rnt"};
          const auto a (s.find_first_not_of(ws));
          const auto b (s.find_last_not_of(ws));
          if (a == string::npos) {
              return {};
          }
          return s.substr(a, b);
      }
```

1.  实际的句子长度计数函数应该接受一个包含所有文本的巨大字符串，然后返回一个`std::multimap`，将排序后的句子长度映射到句子。

```cpp
      multimap<size_t, string> get_sentence_stats(const string &content)
      {
```

1.  我们首先声明`multimap`结构，这是预期的返回值，以及一些迭代器。由于我们将有一个循环，我们需要一个`end`迭代器。然后我们使用两个迭代器来指向文本中连续的句号。两者之间的所有内容都是一个文本句子。

```cpp
          multimap<size_t, string> ret;

          const auto end_it (end(content));
          auto it1 (begin(content));
          auto it2 (find(it1, end_it, '.'));
```

1.  `it2`始终比`it1`多一个句号。只要`it1`没有到达文本的末尾，我们就没问题。第二个条件检查`it2`是否真的至少有一些字符。如果不是这样，它们之间就没有字符可读了。

```cpp
          while (it1 != end_it && distance(it1, it2) > 0) {
```

1.  我们从迭代器之间的所有字符创建一个字符串，并过滤掉其开头和结尾的所有空格，以便计算纯句子的长度。

```cpp
              string s {filter_ws({it1, it2})};
```

1.  可能句子中除了空格以外什么都没有。在这种情况下，我们只是丢弃它。否则，我们通过确定有多少个单词来计算其长度。这很容易，因为所有单词之间都有单个空格。然后我们将单词计数与句子一起保存在`multimap`中。

```cpp
              if (s.length() > 0) {
                  const auto words (count(begin(s), end(s), ' ') + 1);
                  ret.emplace(make_pair(words, move(s)));
              }
```

1.  对于下一个循环迭代，我们将主迭代器`it1`放在下一个句子的句号字符上。接下来的迭代器`it2`放在主迭代器的*旧*位置之后一个字符。

```cpp
              it1 = next(it2, 1);
              it2 = find(it1, end_it, '.');
          }
```

1.  循环结束后，`multimap`包含所有句子及其单词计数，并且可以返回。

```cpp
          return ret;
      }
```

1.  现在我们开始使用该函数。首先，我们告诉`std::cin`不要跳过空格，因为我们希望句子中的空格保持完整。为了读取整个文件，我们从输入流迭代器初始化一个`std::string`，它封装了`std::cin`。

```cpp
      int main()
      {
          cin.unsetf(ios::skipws);
          string content {istream_iterator<char>{cin}, {}};
```

1.  由于我们只需要`multimap`的结果进行打印，我们直接在循环中调用`get_sentence_stats`并将其与我们的字符串一起使用。在循环体中，我们逐行打印项目。

```cpp
          for (const auto & [word_count, sentence] 
                   : get_sentence_stats(content)) {
              cout << word_count << " words: " << sentence << ".n";
          }
      }
```

1.  编译代码后，我们可以从任何文本文件中输入文本到应用程序中。例如 Lorem Ipsum 文本产生以下输出。由于长文本有很多句子，输出非常长，因此它首先打印最短的句子，最后打印最长的句子。这样我们就可以先看到最长的句子，因为终端通常会自动滚动到输出的末尾。

```cpp
      $ cat lorem_ipsum.txt | ./sentence_length
      ...
      10 words: Nam quam nunc, blandit vel, luctus pulvinar, 
      hendrerit id, lorem.
      10 words: Sed consequat, leo eget bibendum sodales, 
      augue velit cursus nunc,.
      12 words: Cum sociis natoque penatibus et magnis dis 
      parturient montes, nascetur ridiculus mus.
      17 words: Maecenas tempus, tellus eget condimentum rhoncus, 
      sem quam semper libero, sit amet adipiscing sem neque sed ipsum.
```

# 它是如何工作的...

整个过程集中在将一个大字符串分解为文本句子，对其长度进行评估，然后在`multimap`中排序。因为`std::multimap`本身非常容易使用，程序的复杂部分是循环，它遍历句子：

```cpp
const auto end_it (end(content));
auto it1 (begin(content));         // (1) Beginning of string
auto it2 (find(it1, end_it, '.')); // (1) First '.' dot

while (it1 != end_it && std::distance(it1, it2) > 0) {
    string sentence {it1, it2};

    // Do something with the sentence string...

    it1 = std::next(it2, 1);      // One character past current '.' dot
    it2 = find(it1, end_it, '.'); // Next dot, or end of string
}
```

让我们在看下面的代码时，考虑以下图表，其中包含三个句子：

![](img/18c1c74b-9f55-4b94-b150-f5f08f678583.png)

`it1`和`it2`始终一起向前移动。这样它们总是指向*一个*句子的开头和结尾。`std::find`算法在这方面帮助了我们很多，因为它的工作方式是“*从当前位置开始，然后返回到下一个句号字符的迭代器。如果没有，返回结束迭代器*。”

在提取句子字符串后，我们确定它包含多少个单词，以便将其插入`multimap`中。我们使用*单词数*作为映射节点的*键*，并将字符串本身作为与之关联的有效负载对象。很容易有多个长度相同的句子。这将使我们无法将它们全部插入一个`std::map`中。但由于我们使用`std::multimap`，这不是问题，因为它可以轻松处理相同值的多个键。它将保持它们全部*有序*，这正是我们需要通过它们的长度枚举所有句子并将它们输出给用户。

# 还有更多...

在将整个文件读入一个大字符串后，我们遍历字符串并再次创建每个句子的副本。这是不必要的，因为我们也可以使用`std::string_view`，这将在本书的后面介绍。

另一种迭代获取两个连续点之间的字符串的方法是`std::regex_iterator`，这也将在本书的后面章节中介绍。

# 使用 std::priority_queue 实现个人待办事项列表

`std::priority_queue`是另一个容器适配器类，例如`std::stack`。它是另一个数据结构（默认情况下为`std::vector`）的包装器，并为其提供了类似队列的接口。这意味着可以逐步将项目推入其中，然后逐步将其弹出。被推入其中的东西*先*被弹出。这通常也被缩写为**先进先出**（**FIFO**）队列。这与堆栈相反，堆栈中*最后*推入的项目会*先*弹出。

虽然我们刚刚描述了`std::queue`的行为，但本节展示了`std::priority_queue`的工作原理。该适配器很特别，因为它不仅考虑 FIFO 特性，还将其与优先级混合在一起。这意味着 FIFO 原则被分解为具有优先级的子 FIFO 队列。

# 如何做...

在本节中，我们将建立一个便宜的*待办事项列表组织*结构。我们不解析用户输入，以便使程序简短并集中在`std::priority_queue`上。因此，我们只是将待办事项的无序列表与优先级和描述一起填充到优先级队列中，然后像从 FIFO 队列数据结构中读取一样，但是根据各个项目的优先级进行分组。

1.  我们首先需要包含一些头文件。`std::priority_queue`在头文件`<queue>`中。

```cpp
      #include <iostream>
      #include <queue>
      #include <tuple>
      #include <string>
```

1.  我们如何将待办事项存储在优先级队列中？问题是，我们不能添加项目并额外附加优先级。优先级队列将尝试使用队列中所有项目的*自然顺序*。我们现在可以实现自己的`struct todo_item`，并给它一个优先级数字和一个待办描述字符串，然后实现比较运算符`<`以使它们可排序。或者，我们可以使用`std::pair`，它使我们能够将两个东西聚合在一个类型中，并为我们自动实现比较。

```cpp
      int main()
      {
          using item_type = std::pair<int, std::string>;
```

1.  我们现在有了一个新类型`item_type`，它由整数优先级和字符串描述组成。因此，让我们实例化一个优先级队列，其中包含这样的项目。

```cpp
          std::priority_queue<item_type> q;
```

1.  我们现在将用不同优先级的不同项目填充优先级队列。目标是提供一个*无结构*的列表，然后优先级队列告诉我们以*哪种顺序*做*什么*。如果有漫画要读，还有作业要做，当然，作业必须先做。不幸的是，`std::priority_queue`没有接受初始化列表的构造函数，我们可以用它来从一开始就填充队列。（使用向量或普通列表，它会按照这种方式工作。）所以我们首先定义列表，然后在下一步中插入它。

```cpp
          std::initializer_list<item_type> il {
              {1, "dishes"},
              {0, "watch tv"},
              {2, "do homework"},
              {0, "read comics"},
          };
```

1.  我们现在可以舒适地遍历待办事项的无序列表，并使用`push`函数逐步插入它们。

```cpp
          for (const auto &p : il) {
              q.push(p);
          }
```

1.  所有项目都被隐式排序，因此我们有一个队列，它给我们最高优先级的项目。

```cpp
          while(!q.empty()) {
              std::cout << q.top().first << ": " << q.top().second << 'n';
              q.pop();
          }
          std::cout << 'n';
      }
```

1.  让我们编译并运行我们的程序。确实，它告诉我们，首先做家庭作业，洗完碗后，我们最终可以看电视和看漫画。

```cpp
      $ ./main
      2: do homework
      1: dishes
      0: watch tv
      0: read comics
```

# 它是如何工作的...

`std::priority`列表非常容易使用。我们只使用了三个函数：

1.  `q.push(item)`将项目推入队列。

1.  `q.top()`返回队列中首先出队的项目的引用。

1.  `q.pop()`移除队列中最前面的项目。

但是项目的排序是如何工作的？我们将优先级整数和待办事项描述字符串分组到一个`std::pair`中，并获得自动排序。如果我们有一个`std::pair<int, std::string>`实例`p`，我们可以写`p.first`来访问*整数*部分，`p.second`来访问*字符串*部分。我们在循环中做到了这一点，打印出所有的待办事项。

但是，优先队列是如何推断出`{2, "做家庭作业"}`比`{0, "看电视"}`更重要的，而不是我们告诉它比较数字部分？

比较运算符`<`处理不同的情况。假设我们比较`left < right`，`left`和`right`是一对。

1.  `left.first != right.first`，然后返回`left.first < right.first`。

1.  `left.first == right.first`，然后返回`left.second < right.second`。

这样，我们可以按需订购物品。唯一重要的是，优先级是对的成员，描述是对的*第二*成员。否则，`std::priority_queue`会以一种看起来字母顺序比优先级更重要的方式对项目进行排序。（在这种情况下，*看电视*会被建议作为*第一*件事情做，*做家庭作业*稍后一些时间。这对于我们这些懒惰的人来说至少是很好的！）
