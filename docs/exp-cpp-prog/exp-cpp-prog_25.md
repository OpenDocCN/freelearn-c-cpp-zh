# 第二十五章：STL 算法基础

本章中我们将涵盖以下内容：

+   将项目从一个容器复制到另一个容器

+   对容器进行排序

+   从容器中删除特定项目

+   转换容器的内容

+   在有序和无序向量中查找项目

+   使用`std::clamp`将向量的值限制在特定的数值范围内

+   使用`std::search`在字符串中查找模式并选择最佳实现

+   对大型向量进行抽样

+   生成输入序列的排列

+   实现字典合并工具

# 介绍

STL 不仅包含数据结构，还包括*算法*。数据结构帮助以不同的方式和不同的动机和目标*存储*和*维护*数据，而算法则对这些数据进行特定的*转换*。

让我们来看一个标准任务，比如对向量中的项目求和。这可以通过循环遍历向量并将所有项目累加到一个名为`sum`的累加器变量中轻松完成：

```cpp
 vector<int> v {100, 400, 200 /*, ... */ };

 int sum {0};
 for (int i : v) { sum += i; }

 cout << sum << 'n';
```

但是因为这是一个相当标准的任务，所以也有一个 STL 算法可以完成这个任务：

```cpp
cout << accumulate(begin(v), end(v), 0) << 'n';
```

在这种情况下，手工制作的循环变体并不比一行代码长多少，而且也不比一个一行代码难以阅读：`accumulate`。然而，在很多情况下，阅读一个 10 行代码的循环是很尴尬的，"我刚刚是否不得不研究整个循环才能理解它执行了一个标准任务 X？"，而不是看到一行代码，它使用了一个清楚说明它的名字的标准算法，比如`accumulate`、`copy`、`move`、`transform`或`shuffle`。

基本思想是提供丰富多样的算法，供程序员在日常工作中使用，以减少重复实现它们的需要。这样，程序员可以直接使用现成的算法实现，并集中精力解决*新*问题，而不是浪费时间在 STL 已经解决的问题上。另一个角度是正确性--如果程序员一遍又一遍地实现相同的东西，那么有可能在一次或另一次尝试中引入一点*错误*。这是完全不必要的，而且如果在代码审查期间被同事指出，这也是非常*尴尬*的，而与此同时，可以使用标准算法。

STL 算法的另一个重要点是*效率*。许多 STL 算法提供了相同算法的多个*专门*实现，这些实现根据它们所使用的*迭代器类型*的不同而执行不同的操作。例如，如果一个整数向量中的所有元素都应该被置零，可以使用 STL 算法`std::fill`来完成。因为向量的迭代器已经可以告诉编译器它是在*连续*内存上迭代，它可以选择使用使用 C 过程`memset`的`std::fill`实现。如果程序员将容器类型从`vector`更改为`list`，那么 STL 算法就不能再使用`memset`，而必须逐个迭代列表以将项目置零。如果程序员自己使用`memset`，那么实现将不必要地硬编码为使用向量或数组，因为大多数其他数据结构不会将它们的数据保存在连续的内存块中。在大多数情况下，试图变得聪明几乎没有意义，因为 STL 的实现者可能已经实现了相同的想法，这些想法可以免费使用。

让我们总结前面的观点。使用 STL 算法有以下好处：

+   **可维护性**：算法的名称已经清楚地说明了它们的功能。显式循环很少有比标准算法更易读且与数据结构无关的情况。

+   **正确性**：STL 已经由专家编写和审查，并且被如此多的人使用和测试，以至于在重新实现其复杂部分时，你很难达到相同的正确性程度。

+   **效率**：STL 算法默认至少与大多数手工编写的循环一样有效。

大多数算法都在*迭代器*上工作。关于迭代器如何工作的概念已经在第二十章中解释过了，*迭代器*。在本章中，我们将集中讨论使用 STL 算法解决不同问题，以便对它们如何有利地利用有所感触。展示*所有*STL 算法会使这本书变成一个非常无聊的 C++参考资料，尽管已经有一个 C++参考资料公开可用。

成为 STL 忍者的最佳方法是始终随身携带 C++参考资料，或者至少将其保存在浏览器书签中。在解决问题时，每个程序员都应该在脑海中回想一下这个问题，“我的问题是否有 STL 算法？”，然后再自己编写代码。

一个非常好而完整的 C++参考资料可以在线查看：

[`cppreference.com`](http://cppreference.com)

它也可以下载以供离线查看。

在工作面试中，熟练掌握 STL 算法通常被视为对 C++知识的强大指标。

# 从容器复制项目到其他容器

最重要的 STL 数据结构都有迭代器支持。这意味着至少可以通过`begin()`和`end()`函数获取迭代器，这些迭代器指向数据结构的基础有效负载数据，并允许对该数据进行迭代。迭代总是看起来一样，无论迭代的是什么类型的数据结构。

我们可以从向量、列表、双端队列、地图等获取迭代器。使用迭代器适配器，我们甚至可以将迭代器作为文件、标准输入和标准输出的接口。此外，正如我们在上一章中看到的，我们甚至可以将迭代器接口包装在算法周围。现在，在我们可以使用迭代器访问所有内容的地方，我们可以将它们与接受迭代器作为参数的 STL 算法结合使用。

展示迭代器如何将不同数据结构的本质抽象化的一个非常好的方法是`std::copy`算法，它只是将项目从一组迭代器复制到输出迭代器。在使用这样的算法时，底层数据结构的本质不再真正相关。为了证明这一点，我们将稍微使用一下`std::copy`。

# 如何做...

在本节中，我们将使用`std::copy`的不同变体：

1.  让我们首先包括我们使用的数据结构所需的所有头文件。此外，我们声明我们使用`std`命名空间：

```cpp
       #include <iostream>
       #include <vector>
       #include <map>
       #include <string>
       #include <tuple>
       #include <iterator>
       #include <algorithm>

       using namespace std;
```

1.  接下来我们将使用整数和字符串值的对。为了漂亮地打印它们，我们应该首先为它们重载`<<`流操作符：

```cpp
       namespace std {
       ostream& operator<<(ostream &os, const pair<int, string> &p)
       {
           return os << "(" << p.first << ", " << p.second << ")";
       }
       }
```

1.  在`main`函数中，我们用一些默认值填充了一个整数-字符串对的`vector`。然后我们声明了一个`map`变量，它将整数值与字符串值关联起来：

```cpp
       int main()
       {
           vector<pair<int, string>> v {
               {1, "one"}, {2, "two"}, {3, "three"}, 
               {4, "four"}, {5, "five"}};

           map<int, string> m;
```

1.  现在，我们使用`std::copy_n`从向量的前面精确地复制三个整数-字符串对到地图中。因为向量和地图是完全不同的数据结构，我们需要使用`insert_iterator`适配器来转换向量中的项目。`std::inserter`函数为我们生成这样的适配器。请始终注意，使用`std::copy_n`等算法与插入迭代器结合使用是将项目复制/插入到其他数据结构的最*通用*方法，但不是*最快*的方法。使用数据结构特定的成员函数来插入项目通常是最有效的方法：

```cpp
           copy_n(begin(v), 3, inserter(m, begin(m)));
```

1.  让我们打印一下映射之后的内容。在整本书中，我们经常使用`std::copy`函数打印容器的内容。`std::ostream_iterator`在这方面非常有帮助，因为它允许我们将用户 shell 的标准输出视为*另一个容器*，我们可以将数据复制到其中：

```cpp
           auto shell_it (ostream_iterator<pair<int, string>>{cout, 
                                                              ", "});

           copy(begin(m), end(m), shell_it);
           cout << 'n';
```

1.  让我们再次清空映射以进行下一个实验。这一次，我们将项目*从*向量*移动*到映射中，而且这一次，是*所有*项目：

```cpp
           m.clear();

           move(begin(v), end(v), inserter(m, begin(m)));
```

1.  我们再次打印映射的新内容。此外，由于`std::move`是一个也会改变数据*源*的算法，我们也将打印源向量。这样，我们就可以看到它在充当移动源时发生了什么：

```cpp
           copy(begin(m), end(m), shell_it);
           cout << 'n';

           copy(begin(v), end(v), shell_it);
           cout << 'n';
       }
```

1.  让我们编译并运行程序，看看它说了什么。前两行很简单。它们反映了应用`copy_n`和`move`算法后映射包含的内容。第三行很有趣，因为它显示了我们用作移动源的向量中的字符串现在为空。这是因为字符串的内容没有被复制，而是被有效地*移动*（这意味着映射使用了先前由向量中的字符串对象引用的堆内存中的字符串数据*）。我们通常不应该访问在重新分配之前作为移动源的项目，但为了这个实验，让我们忽略这一点：

```cpp
      $ ./copying_items
      (1, one), (2, two), (3, three), 
      (1, one), (2, two), (3, three), (4, four), (5, five), 
      (1, ), (2, ), (3, ), (4, ), (5, ),
```

# 它是如何工作的...

由于`std::copy`是 STL 算法中最简单的之一，因此其实现非常简短。让我们看看它是如何实现的：

```cpp
template <typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator it, InputIterator end_it, 
                    OutputIterator out_it)
{
    for (; it != end_it; ++it, ++out_it) {
        *out_it = *it;
    }
    return out_it;
}
```

这看起来确切地像一个人会天真地手动实现从一个可迭代范围到另一个可迭代范围的项目复制。在这一点上，人们也可以问，“那么为什么不手动实现它，循环足够简单，我甚至不需要返回值？”，这当然是一个很好的问题。

虽然`std::copy`不是使代码显著缩短的最佳示例，但许多其他具有更复杂实现的算法是。不明显的是这些 STL 算法的隐藏自动优化。如果我们碰巧使用存储其项目在连续内存中的数据结构（如`std::vector`和`std::array`）*和*项目本身是*平凡复制可分配的*，那么编译器将选择完全不同的实现（假设迭代器类型为指针）：

```cpp
template <typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator it, InputIterator end_it, 
                    OutputIterator out_it)
{
    const size_t num_items (distance(it, end_it));
    memmove(out_it, it, num_items * sizeof(*it));
    return it + num_items;
}
```

这是`std::copy`算法的`memmove`变体在典型的 STL 实现中的简化版本。它比标准循环版本*更快*，而且*这一次*，它也不那么容易阅读。但是，如果参数类型符合此优化的要求，`std::copy`用户会自动从中受益。编译器为所选择的算法选择可能的最快实现，而用户代码则很好地表达了算法的*做什么*，而没有用太多的*如何*细节来污染代码。

STL 算法通常提供了*可读性*和*最佳实现*之间的最佳权衡。

如果类型只包含一个或多个（由类/结构体包装）标量类型或类，通常可以将其视为平凡的可复制可分配类型，这些类型可以安全地使用`memcopy`/`memmove`进行移动，而无需调用用户定义的复制分配运算符。

我们还使用了`std::move`。它的工作原理与`std::copy`完全相同，但它在循环中将`std::move(*it)`应用于源迭代器，以将*lvalues*转换为*rvalues*。这使得编译器选择目标对象的移动赋值运算符，而不是复制赋值运算符。对于许多复杂对象，这样做*性能*更好，但*破坏*了源对象。

# 排序容器

对值进行排序是一个相当标准的任务，可以用多种方式完成。每个被迫学习大多数现有排序算法（以及它们的性能和稳定性权衡）的计算机科学学生都知道这一点。

因为这是一个解决的问题，程序员不应该浪费时间再次解决它，除非是为了学习目的。

# 如何做...

在本节中，我们将使用`std::sort`和`std::partial_sort`：

1.  首先，我们包括所有必要的内容，并声明我们使用`std`命名空间：

```cpp
       #include <iostream>
       #include <algorithm>
       #include <vector>
       #include <iterator>
       #include <random>       

       using namespace std;
```

1.  我们将多次打印整数向量的状态，因此让我们通过编写一个小程序来简化这个任务：

```cpp
       static void print(const vector<int> &v)
       {
           copy(begin(v), end(v), ostream_iterator<int>{cout, ", "});
           cout << 'n';
       }
```

1.  我们从一个包含一些示例数字的向量开始：

```cpp
       int main()
       {
           vector<int> v {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
```

1.  因为我们将多次对向量进行洗牌，以便使用不同的排序函数，所以我们需要一个随机数生成器：

```cpp
           random_device rd;
           mt19937 g {rd()};
```

1.  `std::is_sorted`函数告诉我们容器的内容是否已排序。这行应该打印`1`：

```cpp
           cout << is_sorted(begin(v), end(v)) << 'n';
```

1.  使用`std::shuffle`，我们摇动向量的内容，以便稍后再次对其进行排序。前两个参数表示将被洗牌的范围，第三个参数是随机数生成器：

```cpp
           shuffle(begin(v), end(v), g);
```

1.  `is_sorted`函数现在应该返回`false`，以便打印`0`，向量中的值应该相同，但顺序不同。我们将在将它们再次打印到 shell 后看到：

```cpp
           cout << is_sorted(begin(v), end(v)) << 'n';
           print(v);
```

1.  现在，我们使用`std::sort`重新建立原始项目排序。现在，终端上的相同打印应该再次给我们从一开始的排序顺序：

```cpp
           sort(begin(v), end(v));

           cout << is_sorted(begin(v), end(v)) << 'n';
           print(v);
```

1.  另一个有趣的函数是`std::partition`。也许，我们不想完全对列表进行排序，因为只需将小于某个值的项目放在前面就足够了。因此，让我们*分区*向量，以便将所有小于`5`的项目移到前面并打印它：

```cpp
           shuffle(begin(v), end(v), g);

           partition(begin(v), end(v), [] (int i) { return i < 5; });

           print(v);
```

1.  下一个与排序相关的函数是`std::partial_sort`。我们可以使用它来对容器的内容进行排序，但只能在某种程度上。它将所有向量元素中的`N`个最小元素放在向量的前半部分，并按排序顺序排列。其余的将驻留在第二半部分，不会排序：

```cpp
           shuffle(begin(v), end(v), g);
           auto middle (next(begin(v), int(v.size()) / 2));
           partial_sort(begin(v), middle, end(v));

           print(v);
```

1.  如果我们想对没有比较运算符的数据结构进行排序怎么办？让我们定义一个并创建这样项目的向量：

```cpp
           struct mystruct {
               int a;
               int b;
           };

           vector<mystruct> mv {{5, 100}, {1, 50}, {-123, 1000}, 
                                {3, 70}, {-10, 20}};
```

1.  `std::sort`函数可选地接受一个比较函数作为其第三个参数。让我们使用它，并提供一个这样的函数。只是为了显示这是可能的，我们通过它们的*第二*字段`b`进行比较。这样，它们将按照`mystruct::b`的顺序而不是`mystruct::a`的顺序出现：

```cpp
           sort(begin(mv), end(mv),
                [] (const mystruct &lhs, const mystruct &rhs) {
                    return lhs.b < rhs.b;
                });
```

1.  最后一步是打印排序后的`mystruct`项目向量：

```cpp
           for (const auto &[a, b] : mv) {
               cout << "{" << a << ", " << b << "} ";
           }
           cout << 'n';
       }
```

1.  让我们编译并运行我们的程序。

第一个`1`是在初始化排序向量后对`std::is_sorted`的调用的结果。然后，我们洗牌了向量，并从第二个`is_sorted`调用中得到了`0`。第三行显示了洗牌后的所有向量项目。下一个`1`是使用`std::sort`再次对其进行排序后的`is_sorted`调用的结果。

然后，我们再次洗牌整个向量，并使用`std::partition`进行*分区*。我们可以看到所有小于`5`的项目也在向量中的`5`的左边。所有大于`5`的项目在其右边。除此之外，它们似乎被洗牌了。

倒数第二行显示了`std::partial_sort`的结果。直到中间的所有项目都严格排序，但其余的没有。

在最后一行，我们可以看到我们的`mystruct`实例向量。它们严格按照它们的*第二*成员值排序：

```cpp
      $ ./sorting_containers 
      1
      0
      7, 1, 4, 6, 8, 9, 5, 2, 3, 10, 
      1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
      1, 2, 4, 3, 5, 7, 8, 10, 9, 6, 
      1, 2, 3, 4, 5, 9, 8, 10, 7, 6,
      {-10, 20} {1, 50} {3, 70} {5, 100} {-123, 1000}
```

# 它是如何工作的...

我们使用了与排序有关的不同算法：

| **算法** | **目的** |
| --- | --- |
| `std::sort` | 接受一个范围作为参数并简单地对其进行排序。 |
| `std::is_sorted` | 接受一个范围作为参数，并告诉*是否*该范围已排序。 |
| `std::shuffle` | 这在某种程度上是与排序相反的操作；它接受一个范围作为参数并*洗牌*其项目。 |
| `std::partial_sort` | 接受一个范围作为参数和另一个迭代器，告诉输入范围应排序到哪里。在该迭代器后面，其余项目将未排序。 |
| `std::partition` | 接受一个范围和一个*谓词函数*。谓词函数返回`true`的所有项目都移动到范围的前面。其余的移动到后面。 |

对于没有比较运算符`<`实现的对象，可以提供自定义比较函数。这些函数应该始终具有`bool function_name(const T &lhs, const T &rhs)`这样的签名，并且在执行过程中不应该有任何副作用。

还有其他算法，比如`std::stable_sort`，它也可以排序，但保留具有相同排序键的项目的顺序，以及`std::stable_partition`。

`std::sort`有不同的实现用于排序。根据迭代器参数的性质，它被实现为选择排序，插入排序，归并排序，或者完全针对较少数量的项目进行优化。在用户端，我们通常甚至不需要关心。

# 从容器中删除特定项目

复制，转换和过滤可能是数据范围上最常见的操作。在本节中，我们集中在过滤项目上。

从数据结构中过滤项目，或者简单地删除特定项目，对于不同的数据结构来说完全不同。例如，在链表中（如`std::list`），可以通过使其前驱指向其后继来删除节点。以这种方式从链接链中删除节点后，可以将其返回给分配器。在连续存储数据结构（`std::vector`，`std::array`，以及在某种程度上`std::deque`）中，只能通过用其他项目覆盖它们来删除项目。如果标记要删除的项目槽，那么在它后面的所有项目必须向前移动一个槽，以填补空白。这听起来很麻烦，但是如果我们想要从字符串中简单地删除空格，这应该可以在不多的代码的情况下实现。

当手头有任何一种数据结构时，我们实际上并不想关心如何删除一个项目。它应该只是发生。这就是`std::remove`和`std::remove_if`可以为我们做的事情。

# 如何做...

我们将通过不同的方式删除向量的内容：

1.  让我们导入所有需要的头文件，并声明我们使用`std`命名空间：

```cpp
       #include <iostream>
       #include <vector>
       #include <algorithm>
       #include <iterator>      

       using namespace std;
```

1.  一个简短的打印辅助函数将打印我们的向量：

```cpp
       void print(const vector<int> &v)
       {
           copy(begin(v), end(v), ostream_iterator<int>{cout, ", "});
           cout << 'n';
       }
```

1.  我们将从一个包含一些简单整数值的示例向量开始。我们也会打印它，这样我们就可以看到它在稍后应用于它的函数中如何改变：

```cpp
       int main()
       {
           vector<int> v {1, 2, 3, 4, 5, 6};
           print(v);
```

1.  现在让我们从向量中删除值为`2`的所有项目。`std::remove`以一种使实际上在向量中消失的值`2`的方式移动其他项目。因为在删除项目后向量的实际内容变短了，`std::remove`会返回一个指向*新结尾*的迭代器。新结尾迭代器和旧结尾迭代器之间的项目应被视为垃圾，因此我们告诉向量*擦除*它们。我们将两行删除代码放在一个新的作用域中，因为`new_end`迭代器在之后无论如何都会失效，所以它可以立即超出作用域：

```cpp
           {
               const auto new_end (remove(begin(v), end(v), 2));
               v.erase(new_end, end(v));
           }
           print(v);
```

1.  现在让我们删除所有*奇数*。为了做到这一点，我们实现一个谓词，告诉我们一个数字是否是奇数，并将其输入到`std::remove_if`函数中，该函数接受这样的谓词：

```cpp
           {
               auto odd_number ([](int i) { return i % 2 != 0; });
               const auto new_end (
                   remove_if(begin(v), end(v), odd_number));
               v.erase(new_end, end(v));
           }
           print(v);
```

1.  我们尝试的下一个算法是`std::replace`。我们使用它来用值`123`覆盖所有值为`4`的值。`std::replace`函数也存在为`std::replace_if`，它也接受谓词函数：

```cpp
           replace(begin(v), end(v), 4, 123);
           print(v);
```

1.  让我们完全将新值注入向量，并创建两个新的空向量，以便对它们进行另一个实验：

```cpp
           v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

           vector<int> v2;
           vector<int> v3;
```

1.  然后，我们再次实现一个奇数的谓词和另一个谓词函数，告诉我们一个数字是否是偶数：

```cpp
           auto odd_number  ([](int i) { return i % 2 != 0; });
           auto even_number ([](int i) { return i % 2 == 0; });
```

1.  接下来的两行做了完全相同的事情。它们将*偶数*值复制到向量`v2`和`v3`。第一行使用`std::remove_copy_if`算法，它将来自源容器的所有内容复制到另一个容器，该容器不满足谓词约束。另一行使用`std::copy_if`，它复制满足谓词约束的所有内容：

```cpp
           remove_copy_if(begin(v), end(v), 
                          back_inserter(v2), odd_number);
           copy_if(begin(v), end(v), 
                   back_inserter(v3), even_number);
```

1.  现在打印这两个向量应该得到相同的输出：

```cpp
           print(v2);
           print(v3);
       }
```

1.  让我们编译并运行程序。第一行输出显示了向量在初始化后的状态。第二行显示了删除所有值为`2`后的状态。下一行显示了删除所有奇数后的结果。在第四行之前，我们用`123`替换了所有值为`4`的值。

最后两行显示了向量`v2`和`v3`：

```cpp
      $ ./removing_items_from_containers 
      1, 2, 3, 4, 5, 6, 
      1, 3, 4, 5, 6, 
      4, 6, 
      123, 6, 
      2, 4, 6, 8, 10, 
      2, 4, 6, 8, 10, 
```

# 它是如何工作的...

我们已经使用了不同的算法，这些算法与过滤有关：

| **算法** | **目的** |
| --- | --- |
| `std::remove` | 接受范围和值作为参数，并删除该值的任何出现。返回修改后范围的新结束迭代器。 |
| `std::replace` | 接受范围和两个值作为参数，并用第二个值替换所有第一个值的出现。 |
| `std::remove_copy` | 接受范围、输出迭代器和值作为参数，并将不等于给定值的所有值从范围复制到输出迭代器。 |
| `std::replace_copy`工作原理类似于`std::replace`，但类似于`std::remove_copy`。源范围不会被改变。 |
| `std::copy_if` | 类似于`std::copy`，但还接受谓词函数作为参数，以便仅复制谓词接受的值，这使它成为一个*过滤*函数。 |

对于列出的每个算法，还存在一个`*_if`版本，它接受谓词函数而不是值，然后决定要删除或替换哪些值。

# 转换容器的内容

如果`std::copy`是应用于范围的最简单的 STL 算法，那么`std::transform`就是第二简单的 STL 算法。就像`copy`一样，它将项目从一个范围复制到另一个范围，但还接受一个转换函数。这个转换函数可以在分配给目标范围中的项目之前改变输入类型的值。此外，它甚至可以构造一个完全不同的类型，这在源范围和目标范围的有效负载项目类型不同的情况下非常有用。它很简单但仍然非常有用，这使得它成为可移植日常程序中使用的普通标准组件。

# 如何做到...

在本节中，我们将使用`std::transform`来修改向量的项目并将它们复制：

1.  与往常一样，我们首先需要包含所有必要的头文件，并为了节省一些输入，声明我们使用`std`命名空间：

```cpp
       #include <iostream>
       #include <vector>
       #include <string>
       #include <sstream>
       #include <algorithm>
       #include <iterator>       

       using namespace std;
```

1.  一个包含一些简单整数的向量将作为示例源数据结构：

```cpp
       int main()
       {
           vector<int> v {1, 2, 3, 4, 5};
```

1.  现在，我们将所有项目复制到`ostream_iterator`适配器中以进行打印。`transform`函数接受一个函数对象，该函数对象在每次复制操作期间接受容器有效负载类型的项目并对其进行转换。在这种情况下，我们计算每个数字项目的*平方*，因此代码将打印向量中项目的平方，而无需将它们存储在任何地方：

```cpp
           transform(begin(v), end(v), 
               ostream_iterator<int>{cout, ", "},
               [] (int i) { return i * i; });
           cout << 'n';
```

1.  让我们进行另一个转换。例如，从数字`3`，我们可以生成一个易于阅读的字符串，如`3² = 9`。以下的`int_to_string`函数对象就是使用`std::stringstream`对象实现了这一点：

```cpp
           auto int_to_string ([](int i) {
               stringstream ss;
               ss << i << "² = " << i * i;
               return ss.str();
           });
```

1.  我们刚刚实现的函数从整数值返回字符串值。我们还可以说它从整数到字符串的*映射*。使用`transform`函数，我们可以将所有这样的映射从整数向量复制到字符串向量中：

```cpp
           vector<string> vs;

           transform(begin(v), end(v), back_inserter(vs),
                     int_to_string);
```

1.  打印完这些之后，我们就完成了：

```cpp
           copy(begin(vs), end(vs), 
                ostream_iterator<string>{cout, "n"});
      }
```

1.  让我们编译并运行程序：

```cpp
      $ ./transforming_items_in_containers 
      1, 4, 9, 16, 25, 
      1² = 1
      2² = 4
      3² = 9
      4² = 16
      5² = 25
```

# 它是如何工作的...

`std::transform`函数的工作方式与`std::copy`完全相同，但在将源迭代器的值复制分配到目标迭代器时，它会在将结果分配给目标迭代器之前应用用户提供的转换函数到该值。

# 在有序和无序向量中查找项目

通常，我们需要告诉*是否*某种类型的项目存在于某个范围内。如果存在，我们通常还需要修改它或访问与之关联的其他数据。

有不同的策略来查找项目。如果项目按排序顺序出现，那么我们可以进行二进制搜索，这比逐个遍历项目要快。如果没有排序，我们又被困在线性遍历中。

典型的 STL 搜索算法都可以为我们做这两件事，因此了解它们及其特性是很好的。本节介绍了简单的线性搜索算法`std::find`，二进制搜索版本`std::equal_range`及其变体。

# 如何做...

在本节中，我们将在一个小例子数据集上使用线性和二进制搜索算法：

1.  我们首先包括所有必要的头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <vector>
      #include <list>
      #include <algorithm>
      #include <string>

      using namespace std;
```

1.  我们的数据集将由`city`结构组成，它只保存城市的名称和人口数量：

```cpp
      struct city {
          string name;
          unsigned population;
      };
```

1.  搜索算法需要能够将一个项目与另一个项目进行比较，因此我们为`city`结构实例重载了`==`运算符：

```cpp
      bool operator==(const city &a, const city &b) {
          return a.name == b.name && a.population == b.population;
      }
```

1.  我们还想打印`city`实例，因此我们重载了流运算符`<<`：

```cpp
      ostream& operator<<(ostream &os, const city &city) {
          return os << "{" << city.name << ", " 
                    << city.population << "}";
      }
```

1.  搜索函数通常返回迭代器。这些迭代器指向找到的项目，否则指向底层容器的结束迭代器。在最后一种情况下，我们不允许访问这样的迭代器。因为我们将打印我们的搜索结果，我们实现了一个函数，它返回另一个函数对象，该函数对象封装了数据结构的结束迭代器。在用于打印时，它将比较其迭代器参数与结束迭代器，然后打印项目，否则只是`<end>`：

```cpp
      template <typename C>
      static auto opt_print (const C &container)
      {
          return [end_it (end(container))] (const auto &item) {
              if (item != end_it) {
                  cout << *item << 'n';
              } else {
                  cout << "<end>n";
              }
          };
      }
```

1.  我们从一些德国城市的示例向量开始：

```cpp
      int main()
      {
          const vector<city> c {
              {"Aachen",        246000},
              {"Berlin",       3502000},
              {"Braunschweig",  251000},
              {"Cologne",      1060000}
          };
```

1.  使用这个帮助程序，我们构建了一个城市打印函数，它捕获了我们城市向量`c`的结束迭代器：

```cpp
          auto print_city (opt_print(c));
```

1.  我们使用`std::find`在向量中查找项目，该项目保存了科隆的城市项目。起初，这个搜索看起来毫无意义，因为我们确切地得到了我们搜索的项目。但是在此之前，我们不知道它在向量中的位置，`find`函数只返回了这一点。然而，我们可以，例如，使我们重载的`city`结构的`==`运算符只比较城市名称，然后我们可以只使用城市名称进行搜索，甚至不知道它的人口。但这不是一个好的设计。在下一步中，我们将以不同的方式进行：

```cpp
          {
              auto found_cologne (find(begin(c), end(c), 
                  city{"Cologne", 1060000}));
              print_city(found_cologne);
          }
```

1.  在不知道城市的人口数量，也不干扰其`==`运算符的情况下，我们只能通过比较其名称与向量的内容来搜索。`std::find_if`函数接受一个谓词函数对象，而不是特定的值。这样，我们可以在只知道其名称的情况下搜索科隆市的项目：

```cpp
          {
              auto found_cologne (find_if(begin(c), end(c), 
                  [] (const auto &item) {
                      return item.name == "Cologne";
                  }));
              print_city(found_cologne);
          }
```

1.  为了使搜索更加美观和表达力强，我们可以实现谓词构建器。`population_higher_than`函数对象接受一个人口规模，并返回一个告诉我们`city`实例是否比捕获的值具有更大人口的函数。让我们使用它来搜索我们小例子集中拥有两百万以上居民的德国城市。在给定的向量中，那个城市只有柏林：

```cpp
          {
              auto population_more_than ([](unsigned i) {
                  return [=] (const city &item) { 
                      return item.population > i; 
                  };
              });
              auto found_large (find_if(begin(c), end(c), 
                  population_more_than(2000000)));
              print_city(found_large);
          }
```

1.  我们刚刚使用的搜索函数遍历了我们的容器。因此它们的运行时复杂度为*O(n)*。STL 还有二进制搜索函数，它们在*O(log(n))*内工作。让我们生成一个新的例子数据集，它只包含一些整数值，并为此构建另一个`print`函数：

```cpp
          const vector<int> v {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

          auto print_int (opt_print(v));
```

1.  `std::binary_search`函数返回布尔值，只告诉我们*是否*找到了一个项目，但它*不*返回项目本身。重要的是，我们正在搜索的容器是*排序*的，否则，二进制搜索就无法正确工作：

```cpp
          bool contains_7 {binary_search(begin(v), end(v), 7)};
          cout << contains_7 << 'n';
```

1.  为了得到我们正在搜索的项目，我们需要其他 STL 函数。其中之一是`std::equal_range`。它不返回我们找到的项目的迭代器，而是一对迭代器。第一个迭代器指向第一个不小于我们正在寻找的值的项目。第二个迭代器指向第一个大于它的项目。在我们的范围内，从`1`到`10`，第一个迭代器指向实际的`7`，因为它是第一个不小于`7`的项目。第二个迭代器指向`8`，因为它是第一个大于`7`的项目。如果我们有多个值为`7`，那么这两个迭代器实际上代表*项目的子范围*：

```cpp
          auto [lower_it, upper_it] (
              equal_range(begin(v), end(v), 7));
          print_int(lower_it);
          print_int(upper_it);
```

1.  如果我们只需要一个迭代器；我们可以使用`std::lower_bound`或`std::upper_bound`。`lower_bound`函数只返回一个迭代器，指向第一个不小于我们搜索的项目。`upper_bound`函数返回一个迭代器，指向第一个大于我们搜索的项目：

```cpp
          print_int(lower_bound(begin(v), end(v), 7));
          print_int(upper_bound(begin(v), end(v), 7));
      }
```

1.  让我们编译并运行程序，看看输出是否符合我们的假设：

```cpp
      $ ./finding_items 
      {Cologne, 1060000}
      {Cologne, 1060000}
      {Berlin, 3502000}
      1
      7
      8
      7
      8
```

# 它是如何工作的...

这些是我们在这个配方中使用的搜索算法：

| **算法** | **目的** |
| --- | --- |
| `std::find` | 接受搜索范围和比较值作为参数。返回一个指向与比较值相等的第一个项目的迭代器。进行线性搜索。 |
| `std::find_if` | 类似于`std::find`，但使用谓词函数而不是比较值。 |
| `std::binary_search` | 接受搜索范围和比较值作为参数。执行二进制搜索，如果范围包含该值，则返回`true`。 |
| `std::lower_bound` | 接受搜索范围和比较值，然后对第一个*不小于*比较值的项目执行二进制搜索。返回指向该项目的迭代器。 |
| `std::upper_bound` | 类似于`std::lower_bound`，但返回一个指向第一个*大于*比较值的项目的迭代器。 |
| `std::equal_range` | 接受搜索范围和比较值，然后返回一对迭代器。第一个迭代器是`std::lower_bound`的结果，第二个迭代器是`std::upper_bound`的结果。 |

所有这些函数都接受自定义比较函数作为可选的附加参数。这样，搜索可以被定制，就像我们在配方中所做的那样。

让我们更仔细地看看`std::equal_range`是如何工作的。假设我们有一个向量，`v = {0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8}`，并调用`equal_range(begin(v), end(v), 7);`来对值`7`执行二进制搜索。由于`equal_range`给我们返回了一个下界和一个上界迭代器的一对，因此这些之后应该表示范围`{7, 7, 7}`，因为在排序向量中有这么多值为`7`。查看以下图表以获得更清晰的解释：

![](img/8e00c5a5-38e8-4902-95ad-566642ea317b.png)

首先，`equal_range`使用典型的二进制搜索方法，直到它遇到*不小于*搜索值的值范围。然后，它分成一个`lower_bound`调用和一个`upper_bound`调用，以将它们的返回值捆绑成一对作为返回值。

为了得到一个二进制搜索函数，它只返回符合要求的第一个项目，我们可以实现以下内容：

```cpp
template <typename Iterator, typename T>
Iterator standard_binary_search(Iterator it, Iterator end_it, T value)
{
    const auto potential_match (lower_bound(it, end_it, value));
    if (potential_match != end_it && value == *potential_match) {
        return potential_match;
    }
    return end_it;
}
```

该函数使用`std::lower_bound`来找到第一个不小于`value`的项目。然后，得到的`potential_match`可以指向三种不同的情况：

+   没有项目比`value`小。在这种情况下，它与`end_it`相同。

+   第一个不小于`value`的项目也*大于*`value`。因此，我们必须通过返回`end_it`来表示我们*没有*找到它。

+   `potential_match`指向的项目等于`value`。因此，它不仅是一个*potential*匹配，而且是一个*actual*匹配。因此我们可以返回它。

如果我们的类型`T`不支持`==`运算符，那么它至少必须支持二分搜索的`<`运算符。然后，我们可以将比较重写为`!(value < *potential_match) && !(*potential_match < value)`。如果既不小也不大，那么它必须相等。

STL 没有提供这样一个函数的一个潜在原因是缺乏关于可能存在多个命中的可能性的知识，就像在我们有多个值为`7`的图表中一样。

请注意，诸如`std::map`、`std::set`等的数据结构都有它们*自己的*`find`函数。当然，这些函数比更通用的算法更快，因为它们与数据结构的实现和数据表示紧密耦合。

# 使用 std::clamp 将向量的值限制在特定的数值范围内

在许多应用程序中，我们从某处获得数值数据。在我们可以绘制或以其他方式处理它之前，可能需要对其进行归一化，因为这些值之间的差异可能是随机的。

通常，这意味着对保存所有这些值的数据结构进行一次小的`std::transform`调用，结合一个简单的*scaling*函数。但是，如果我们*不知道*值有多大或多小，我们需要先通过数据找到合适的*dimensions*来进行缩放函数。

STL 包含了用于此目的的有用函数：`std::minmax_element`和`std::clamp`。使用这些函数，并将它们与一些 lambda 表达式粘合在一起，我们可以轻松地执行这样的任务。

# 如何做...

在本节中，我们将以两种不同的方式将向量的值从示例数值范围归一化为归一化范围，其中一种使用`std::minmax_element`，另一种使用`std::clamp`：

1.  与往常一样，我们首先需要包括以下头文件并声明我们使用`std`命名空间：

```cpp
       #include <iostream>
       #include <vector>
       #include <algorithm>
       #include <iterator>       

       using namespace std;
```

1.  我们实现了一个供以后使用的函数，它接受范围的最小值和最大值，以及一个新的最大值，以便它可以将旧范围的值投影到我们想要的较小范围。函数对象接受这样的值，并返回另一个函数对象，该函数对象正是进行这种转换。为了简单起见，新的最小值是`0`，因此无论旧数据有什么偏移，其归一化值始终相对于`0`。为了可读性，我们忽略了`max`和`min`可能具有相同值的可能性，这将导致除以零：

```cpp
       static auto norm (int min, int max, int new_max)
       {
           const double diff (max - min);
           return [=] (int val) {
               return int((val - min) / diff * new_max);
           };
       }
```

1.  另一个函数对象构建器称为`clampval`返回一个函数对象，该函数对象捕获`min`和`max`值，并在具有这些值的值上调用`std::clamp`，以限制它们的值在此范围内：

```cpp
       static auto clampval (int min, int max)
       {
           return [=] (int val) -> int {
               return clamp(val, min, max);
           };
       }
```

1.  我们要归一化的数据是一个包含不同值的向量。例如，这可能是某种热量数据，地形高度或随时间变化的股票价格：

```cpp
       int main()
       {
           vector<int> v {0, 1000, 5, 250, 300, 800, 900, 321};
```

1.  为了能够归一化数据，我们需要*最高*和*最低*值。`std::minmax_element`函数在这里非常有帮助。它为我们返回了一个指向这两个值的迭代器对：

```cpp
           const auto [min_it, max_it] (
               minmax_element(begin(v), end(v)));
```

1.  我们将所有值从第一个向量复制到第二个向量。让我们实例化第二个向量，并准备接受与第一个向量中的新项目一样多的新项目：

```cpp
           vector<int> v_norm;
           v_norm.reserve(v.size());
```

1.  使用`std::transform`，我们将值从第一个向量复制到第二个向量。在复制项目的同时，它们将使用我们的归一化辅助函数进行转换。旧向量的最小值和最大值分别为`0`和`1000`。归一化后的最小值和最大值分别为`0`和`255`：

```cpp
           transform(begin(v), end(v), back_inserter(v_norm),
                     norm(*min_it, *max_it, 255));
```

1.  在我们实现另一种归一化策略之前，我们先打印一下我们现在有的东西：

```cpp
           copy(begin(v_norm), end(v_norm), 
                ostream_iterator<int>{cout, ", "});
           cout << 'n';
```

1.  我们使用另一个名为`clampval`的辅助函数重用相同的归一化向量，它*将*旧范围限制为最小值为`0`和最大值为`255`的范围：

```cpp
           transform(begin(v), end(v), begin(v_norm), 
                     clampval(0, 255));
```

1.  打印这些值后，我们就完成了：

```cpp
           copy(begin(v_norm), end(v_norm),
                ostream_iterator<int>{cout, ", "});
           cout << 'n';
       }
```

1.  让我们编译并运行程序。将值减少到`0`到`255`的值，我们可以将它们用作 RGB 颜色代码的亮度值，例如：

```cpp
      $ ./reducing_range_in_vector 
      0, 255, 1, 63, 76, 204, 229, 81, 
      0, 255, 5, 250, 255, 255, 255, 255,
```

1.  当我们绘制数据时，我们得到以下图表。正如我们所看到的，*将*值除以最小值和最大值之间的差异的方法是原始数据的线性转换。*夹紧*图表丢失了一些信息。不同的情况下，这两种变化都可能有用：

![](img/f35fa700-2874-4d52-a587-32e53adebcf0.png)

# 它是如何工作的...

除了`std::transform`，我们使用了两种算法：

`std::minmax_element`只接受输入范围的开始和结束迭代器。它遍历范围并记录最大和最小的元素，然后返回这些值的一对，我们用于我们的缩放函数。

与之相反，`std::clamp`函数不适用于可迭代范围。它接受三个值：输入值、最小值和最大值。这个函数的输出是输入值被截断，以便它位于允许的最小值和最大值之间。我们也可以写`max(min_val, min(max_val, x))`而不是`std::clamp(x, min_val, max_val)`。

# 使用 std::search 在字符串中定位模式并选择最佳实现

在字符串中搜索字符串与在范围中查找*一个*对象是一个略有不同的问题。一方面，字符串当然也是一个可迭代范围（字符）；另一方面，在字符串中查找字符串意味着在*另一个*范围中查找一个范围。这伴随着每个潜在匹配位置的多次比较，因此我们需要一些其他的算法。

`std::string`已经包含一个`find`函数，它可以做我们正在谈论的事情；尽管如此，我们将在本节集中讨论`std::search`。尽管`std::search`可能主要用于字符串，但它适用于所有类型的容器。`std::search`更有趣的特性是，自 C++17 以来，它具有稍微不同的附加接口，并且允许简单地交换搜索算法本身。这些算法是经过优化的，可以根据使用情况自由选择。此外，如果我们能想出比已提供的更好的东西，我们还可以实现自己的搜索算法并将它们插入`std::search`。

# 如何做...

我们将使用新的`std::search`函数与字符串，并尝试其不同的变体与搜索器对象：

1.  首先，我们将包括所有必要的标头，并声明我们使用`std`命名空间：

```cpp
       #include <iostream>
       #include <string>
       #include <algorithm>
       #include <iterator>
       #include <functional>       

       using namespace std;
```

1.  我们将打印搜索算法返回给我们的位置的子字符串，因此让我们为此实现一个小助手：

```cpp
       template <typename Itr>
       static void print(Itr it, size_t chars)
       {
           copy_n(it, chars, ostream_iterator<char>{cout});
           cout << 'n';
       }
```

1.  一个*lorem-ipsum 风格*的字符串将作为我们的示例字符串，我们将在其中搜索一个子字符串。在这种情况下，这是`"elitr"`：

```cpp
       int main()
       {
           const string long_string {
               "Lorem ipsum dolor sit amet, consetetur"
               " sadipscing elitr, sed diam nonumy eirmod"};
           const string needle {"elitr"};
```

1.  旧的`std::search`接口接受我们正在搜索特定子字符串的字符串的开始/结束迭代器以及子字符串的开始/结束迭代器。然后返回一个指向它能够找到的子字符串的迭代器。如果没有找到字符串，返回的迭代器将是结束迭代器：

```cpp
           {
               auto match (search(begin(long_string), end(long_string),
                                  begin(needle), end(needle)));
               print(match, 5);
           }
```

1.  C++17 版本的`std::search`不接受两对迭代器，而是接受一对开始/结束迭代器和一个*searcher*对象。`std::default_searcher`接受我们在较大字符串中搜索的子字符串的开始/结束迭代器对：

```cpp
           {
               auto match (search(begin(long_string), end(long_string),
                   default_searcher(begin(needle), end(needle))));
               print(match, 5);
           }
```

1.  这种改变的重点是这样很容易切换搜索算法。`std::boyer_moore_searcher`使用*Boyer-Moore 搜索算法*进行更快的搜索：

```cpp
           {
               auto match (search(begin(long_string), end(long_string),
                   boyer_moore_searcher(begin(needle), 
                                        end(needle))));
               print(match, 5);
           }
```

1.  C++17 STL 带有三种不同的搜索器对象实现。第三个是 B*oyer-Moore-Horspool 搜索算法*实现：

```cpp
           {
               auto match (search(begin(long_string), end(long_string),
                   boyer_moore_horspool_searcher(begin(needle), 
                                                 end(needle))));
               print(match, 5);
           }
       }
```

1.  让我们编译并运行我们的程序。如果运行正确，我们应该在任何地方看到相同的字符串：

```cpp
      $ ./pattern_search_string 
      elitr
      elitr
      elitr
      elitr
```

# 它是如何工作的...

我们使用了四种不同的方法来使用`std::search`，以获得完全相同的结果。在什么情况下应该使用哪种？

假设我们在其中搜索模式的大字符串称为`s`，模式称为`p`。然后，`std::search(begin(s), end(s), begin(p), end(p));`和`std::search(begin(s), end(s), default_searcher(begin(p), end(p));`做的事情完全一样。

其他搜索函数对象是使用更复杂的搜索算法实现的。

+   `std::default_searcher`：这将重定向到传统的`std::search`实现

+   `std::boyer_moore_searcher`：这使用*Boyer-Moore*搜索算法

+   `std::boyer_moore_horspool_searcher`：这类似地使用*Boyer-Moore-Horspool*算法

其他算法有什么特别之处？Boyer-Moore 算法是根据一个特定的想法开发的——搜索模式与字符串进行比较，从右到左从模式的*末尾*开始。如果搜索字符串中的字符与覆盖位置处模式中的字符*不同*，并且在模式中*甚至不存在*，那么很明显，模式可以通过其*完整长度*移动到搜索字符串上。看一下下面的图表，在步骤 1 中发生了这种情况。如果当前比较的字符与该位置处模式的字符不同，但*包含*在模式中，则算法知道模式需要向右移动多少个字符才能正确对齐至少该字符，然后，它重新开始右到左的比较。在图表中，这在步骤 2 中发生。这样，与朴素的搜索实现相比，Boyer-Moore 算法可以省略很多*不必要*的比较：

![](img/2cefec94-bb6d-48ce-af86-cb86df918853.png)

当然，如果它没有带来自己的*权衡*，这将成为新的默认搜索算法。它比默认算法更快，但它需要快速查找数据结构，以确定哪些字符包含在搜索模式中，以及它们位于哪个偏移量。编译器将根据模式由哪些基础类型组成（在复杂类型之间变化为哈希映射，对于`char`等类型的基本查找表）选择不同复杂的实现。最终，这意味着如果搜索字符串不太大，则默认搜索实现将更快。如果搜索本身需要一些显着的时间，那么 Boyer-Moore 算法可以在*常数因子*的维度上带来性能增益。

**Boyer-Moore-Horspool**算法是 Boyer-Moore 算法的简化。它放弃了*坏字符*规则，这导致整个模式宽度的移位，如果找到一个搜索字符串字符，它在模式字符串中不存在。这个决定的权衡是它比未修改的 Boyer-Moore 稍慢，但它也需要*更少的数据结构*来进行操作。

不要试图*推断*在特定情况下哪种算法*应该*更快。始终使用对您的用户典型的数据样本*测量*代码的性能，并根据*结果*做出决定。

# 抽样大向量

当需要处理非常大量的数字数据时，在某些情况下，可能无法在可行的时间内处理所有数据。在这种情况下，可以对数据进行*抽样*，以减少进一步处理的总数据量，从而*加快*整个程序。在其他情况下，这可能不是为了减少处理工作量，而是为了*保存*或*传输*数据。

抽样的一个天真的想法可能是只选择每第*N*个数据点。在许多情况下这可能是可以的，但在信号处理中，例如，它*可能*会导致一种称为**混叠**的数学现象。如果每个样本之间的距离受到小的随机偏移的影响，混叠可以被减少。看一下下面的图表，它展示了一个*极端情况*，只是为了说明这一点--原始信号由正弦波组成，图表上的三角形点是在每个*100*个数据点处进行抽样的抽样点。不幸的是，这些点的信号在这些点上具有*相同的 y 值*！然而，方形点显示了当我们抽样每`100 + random(-15, +15)`个点时我们得到的结果。在这里，信号看起来仍然与原始信号非常不同，但至少不像固定步长抽样情况下完全*消失*。

`std::sample`函数不会对固定偏移的样本点进行随机更改，而是选择完全随机的点；因此，它的工作方式与这个例子有些不同：

![](img/9f2aae52-6c69-479c-9d10-af4a807faf4e.png)

# 如何做...

我们将对一个非常大的随机数据向量进行抽样。这些随机数据显示正态分布。在对其进行抽样后，结果点应该仍然显示出正态分布，我们将进行检查：

1.  首先，我们需要包括我们使用的所有内容，并声明我们使用`std`命名空间，以节省一些输入：

```cpp
       #include <iostream>
       #include <vector>
       #include <random>
       #include <algorithm>
       #include <iterator>
       #include <map>
       #include <iomanip>       

       using namespace std;
```

1.  如果我们在它们自己的常量变量中配置我们算法的特定特征，那么就更容易玩弄代码。这些是大型随机向量的大小和我们将从中获取的样本数量：

```cpp
       int main()
       {
           const size_t data_points   {100000};
           const size_t sample_points {100};
```

1.  大型的、随机填充的向量应该从随机数生成器中获得数字，该生成器从正态分布中输出数字。任何正态分布都可以由平均值和与平均值的标准偏差来描述：

```cpp
           const int    mean {10};
           const size_t dev  {3};
```

1.  现在，我们设置随机生成器。首先，我们实例化一个随机设备，并调用它一次以获得用于随机生成器构造函数的种子。然后，我们实例化一个应用正态分布于随机输出的分布对象：

```cpp
           random_device rd;
           mt19937 gen {rd()};
           normal_distribution<> d {mean, dev};
```

1.  现在，我们实例化一个整数向量，并用大量随机数填充它。这是通过使用`std::generate_n`算法实现的，该算法将调用一个生成器函数对象，将其返回值馈送到我们的向量中，使用`back_inserter`迭代器。生成器函数对象只是包装在`d(gen)`表达式周围，该表达式从随机设备获取随机数，并将其馈送到分布对象中：

```cpp
           vector<int> v;
           v.reserve(data_points);

           generate_n(back_inserter(v), data_points, 
               [&] { return d(gen); });
```

1.  现在，我们实例化另一个向量，它将包含较小的样本集：

```cpp
           vector<int> samples;
           v.reserve(sample_points);
```

1.  `std::sample`算法类似于`std::copy`，但它需要两个额外的参数：*样本数量*，它应该从输入范围中获取的样本数量，以及一个*随机数生成器*对象，它将用于获取随机抽样位置：

```cpp
           sample(begin(v), end(v), back_inserter(samples), 
                  sample_points, mt19937{random_device{}()});
```

1.  我们已经完成了抽样。其余的代码是为了显示目的。输入数据具有正态分布，如果抽样算法运行良好，那么抽样向量应该也显示正态分布。为了查看剩下多少正态分布，我们将打印值的*直方图*：

```cpp
           map<int, size_t> hist;

           for (int i : samples) { ++hist[i]; }
```

1.  最后，我们循环遍历所有项目以打印我们的直方图：

```cpp
           for (const auto &[value, count] : hist) {
               cout << setw(2) << value << " "
                    << string(count, '*') << 'n';
           }    
       }
```

1.  编译并运行程序后，我们看到抽样向量仍然大致显示出正态分布的特征：

![](img/429739c2-32fd-4b71-8557-d1af399d9b7d.png)

# 它的工作原理是...

`std::sample`算法是一个新算法，它随 C++17 一起推出。它的签名如下：

```cpp
template<class InIterator, class OutIterator,
         class Distance, class UniformRandomBitGenerator>
OutIterator sample(InIterator first, InIterator last,
                   SampleIterator out, Distance n, 
                   UniformRandomBitGenerator&& g);

```

输入范围由`first`和`last`迭代器表示，而`out`是输出操作符。这些迭代器的功能与`std::copy`中的功能完全相同；项从一个范围复制到另一个范围。`std::sample`算法在这方面是特殊的，因为它只会复制输入范围的一部分，因为它只对`n`个项进行抽样。它在内部使用均匀分布，因此源范围中的每个数据点都以相同的概率被选择。

# 生成输入序列的排列

在测试必须处理输入序列的代码时，如果参数的顺序不重要，测试它是否对该输入的*所有*可能的排列产生相同的输出是有益的。例如，这样的测试可以检查自己实现的*排序*算法是否正确排序。

无论出于什么原因，我们需要某个值范围的所有排列，`std::next_permutation`可以方便地为我们做到这一点。我们可以在可修改的范围上调用它，它会改变其项的*顺序*到下一个*字典序排列*。

# 如何做...

在本节中，我们将编写一个程序，从标准输入中读取多个单词字符串，然后我们将使用`std::next_permutation`来生成并打印这些字符串的所有排列：

1.  首先还是先来一些基础工作；我们包含所有必要的头文件，并声明我们使用`std`命名空间：

```cpp
      #include <iostream>
      #include <vector>
      #include <string>
      #include <iterator>
      #include <algorithm>      

      using namespace std;
```

1.  我们从一个字符串向量开始，我们用整个标准输入来填充它。下一步是*排序*向量：

```cpp
      int main()
      {
          vector<string> v {istream_iterator<string>{cin}, {}};
          sort(begin(v), end(v));
```

1.  现在，我们在用户终端上打印向量的内容。之后，我们调用`std::next_permutation`。它会系统地洗牌向量以生成其项的排列，然后我们再次打印。当达到*最后*一个排列时，`next_permutation`会返回`false`：

```cpp
          do {
              copy(begin(v), end(v), 
                   ostream_iterator<string>{cout, ", "});
              cout << 'n';
          } while (next_permutation(begin(v), end(v)));
      }
```

1.  让我们用一些示例输入编译并运行该函数：

```cpp
      $ echo "a b c" | ./input_permutations 
      a, b, c, 
      a, c, b, 
      b, a, c, 
      b, c, a, 
      c, a, b, 
      c, b, a,
```

# 它是如何工作的...

`std::next_permutation`算法使用起来有点奇怪。这是因为它只接受一个迭代器的开始/结束对，然后如果能找到下一个排列就返回`true`。否则，返回`false`。但是*下一个排列*到底是什么意思呢？

`std::next_permutation`用于找到项的下一个字典序排列的算法工作如下：

1.  找到最大的索引`i`，使得`v[i - 1] < v[i]`。如果没有，则返回`false`。

1.  现在，找到最大的索引`j`，使得`j >= i`且`v[j] > v[i - 1]`。

1.  在位置`j`和位置`i - 1`交换项。

1.  反转从位置`i`到范围末尾的项的顺序。

1.  返回`true`。

我们从中得到的各自排列的顺序总是相同的。为了看到所有可能的排列，我们首先对数组进行排序，因为如果我们输入了`"c b a"`，例如，算法会立即终止，因为这已经是元素的最后字典序排列。

# 实现字典合并工具

想象我们有一个排好序的东西列表，然后另外有人提出了*另一个*排好序的东西列表，我们想要彼此分享这些列表。最好的主意是将这两个列表合并。这两个列表的组合也应该是排好序的，这样，查找特定项就很容易了。

这样的操作也被称为**合并**。为了合并两个排好序的项范围，我们直观地会创建一个新范围，并从两个列表中的项中获取它。对于每个项的转移，我们必须比较输入范围的最前面的项，以便始终选择剩下的输入中的*最小*项。否则，输出范围将不再是排好序的。下面的图示更好地说明了这一点：

![](img/a6528449-cc62-4b3f-b4bd-1900646d9175.png)

`std::merge`算法可以为我们做到这一点，所以我们不需要太多地摆弄。在本节中，我们将看到如何使用这个算法。

# 如何做...

我们将建立一个廉价的字典，从英语单词到它们的德语翻译的一对一映射，并将它们存储在`std::deque`结构中。程序将从文件和标准输入中读取这样的字典，并再次在标准输出上打印一个大的合并字典。

1.  这次需要包含很多头文件，并且我们声明使用`std`命名空间：

```cpp
      #include <iostream>
      #include <algorithm>
      #include <iterator>
      #include <deque>
      #include <tuple>
      #include <string>
      #include <fstream>     

      using namespace std;
```

1.  一个字典条目应该包括从一种语言的字符串到另一种语言的字符串的对称映射：

```cpp
      using dict_entry = pair<string, string>;
```

1.  我们将同时将这样的对打印到终端并从用户输入中读取，因此我们需要重载`<<`和`>>`运算符：

```cpp
      namespace std {
      ostream& operator<<(ostream &os, const dict_entry p)
      {
          return os << p.first << " " << p.second;
      }

      istream& operator>>(istream &is, dict_entry &p)
      {
          return is >> p.first >> p.second;
      }

      }
```

1.  一个接受任何输入流对象的辅助函数将帮助我们构建一个字典。它构造了一个字典条目对的`std::deque`，并且它们都从输入流中读取，直到输入流为空。在返回之前，我们对它进行排序：

```cpp
      template <typename IS>
      deque<dict_entry> from_instream(IS &&is)
      {
          deque<dict_entry> d {istream_iterator<dict_entry>{is}, {}};
          sort(begin(d), end(d));
          return d;
      }
```

1.  我们从不同的输入流中创建了两个单独的字典数据结构。一个输入流是从`dict.txt`文件中打开的，我们假设它存在。它包含逐行的单词对。另一个流是标准输入：

```cpp
      int main()
      {
          const auto dict1 (from_instream(ifstream{"dict.txt"}));
          const auto dict2 (from_instream(cin));
```

1.  由于辅助函数`from_instream`已经为我们对这两个字典进行了排序，我们可以直接将它们输入`std::merge`算法。它通过它的开始/结束迭代器对接受两个输入范围，并且一个输出。输出将是用户的 shell：

```cpp
          merge(begin(dict1), end(dict1),
                begin(dict2), end(dict2),
                ostream_iterator<dict_entry>{cout, "n"});
      }
```

1.  现在我们可以编译程序了，但在运行之前，我们应该创建一个`dict.txt`文件，并填充一些示例内容。让我们用一些英语单词和它们的德语翻译来填充它：

```cpp
      car       auto
      cellphone handy
      house     haus
```

1.  现在，我们可以启动程序，同时将一些英语-德语翻译传递给它的标准输入。输出是一个合并且仍然排序的字典，其中包含了两个输入的翻译。我们可以从中创建一个新的字典文件：

```cpp
      $ echo "table tisch fish fisch dog hund" | ./dictionary_merge
      car auto
      cellphone handy
      dog hund
      fish fisch
      house haus
      table tisch
```

# 它是如何工作的...

`std::merge`算法接受两对开始/结束迭代器，表示输入范围。这些范围必须是*排序*的。第五个参数是一个输出迭代器，接受合并过程中传入的项目。

还有一种叫做`std::inplace_merge`的变体。这个算法和其他算法一样，但它不需要输出迭代器，因为它是*原地*工作的，正如它的名字所暗示的那样。它接受三个参数：一个*开始*迭代器，一个*中间*迭代器和一个*结束*迭代器。这些迭代器必须都引用相同数据结构中的数据。中间迭代器同时也是第一个范围的结束迭代器，以及第二个范围的开始迭代器。这意味着这个算法处理一个单一范围，实际上包括两个连续的范围，比如，例如`{A, C, B, D}`。第一个子范围是`{A, C}`，第二个子范围是`{B, D}`。然后`std::inplace_merge`算法可以在同一个数据结构中合并两者，结果是`{A, B, C, D}`。
