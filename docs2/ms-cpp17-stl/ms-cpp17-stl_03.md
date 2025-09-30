# 第三章：迭代器对算法

现在你已经了解了迭代器类型——既包括标准提供的也包括用户定义的——现在是时候看看你可以用迭代器做什么了。

本章你将学习：

+   “半开范围”的概念，这确定了两个迭代器如何定义一个*范围*

+   如何将每个标准算法分类为“只读”、“只写”、“转换”或“排列”；以及作为“单范围”、“双范围”或“一又一半范围”

+   一些标准算法，如`merge`和`make_heap`，仅仅是构建更高层次实体（如`stable_sort`和`priority_queue`）所必需的构建块。

+   如何根据除`operator<`之外的比较器对范围进行排序

+   如何使用*erase-remove 习语*操作排序数组

# 关于头文件的说明

本章讨论的大多数函数模板都定义在标准头文件`<algorithm>`中。另一方面，特殊的迭代器类型通常定义在`<iterator>`中。如果你想知道如何找到特定的实体，我强烈建议你咨询在线参考资料，如[cppreference.com](https://cppreference.com)，以获得权威答案；不要只是猜测！

# 只读范围算法

在前面的章节中，我们构建了一个我们称之为`distance`的算法，另一个称为`count_if`。这两个算法都出现在标准库中。

`std::count_if(a,b,p)`返回满足谓词函数`p`的元素数量，即在`a`和`b`之间，使得`p(e)`为`true`的元素数量`e`。

注意，每当说到“在`a`和`b`之间”，我们都是在谈论包括`*a`但不包括`*b`的范围——数学家称之为“半开范围”，并用不对称的符号`[a,b)`表示。为什么我们不能包括`*b`呢？首先，如果`b`是某个向量的`end()`，那么它根本不指向该向量的任何元素！所以一般来说，解引用范围的*终点*是一件危险的事情。其次，使用半开范围方便地允许我们表示*空*范围；例如，“从`x`到`x`”的范围是一个包含零数据元素的空范围。

在 C++中，半开范围与在 C 中一样自然。几十年来，我们一直在编写从下界（包含）到上界（不包含）的范围的 for 循环；这个习语如此常见，以至于偏离这个习语通常表明存在错误：

```cpp
    constexpr int N = 10;
    int a[N];

    // A correct for-loop.
    for (int i=0; i < N; ++i) {
      // ...
    }

    // One variety of "smelly" for-loop.
    for (int i=0; i <= N; ++i) {
      // ... 
    }

    // A correct invocation of a standard algorithm.
    std::count_if(std::begin(a), std::end(a), [](int){ return true; });

    // A "smelly" invocation.
    std::count_if(std::begin(a), std::end(a) - 1, [](int){ return true; });

    // A "trivial" invocation: counting a range of length zero.
    std::count_if(std::begin(a), std::begin(a), [](int){ return true; });
```

`std::distance(a,b)`返回`a`和`b`之间的元素数量——也就是说，你需要将`++`应用于`a`多少次才能到达`b`。你可以将这个函数视为在效果上等同于`std::count_if(a,b,[](auto&&){return true;})`。

正如我们在 第二章，*迭代器和范围* 中所看到的，如果相关的迭代器是随机访问迭代器，这个数字可以快速计算为 `(b - a)`，因此标准 `std::distance` 会这样做。请注意，`(b - a)` 可能是一个负数，如果你以“错误”的顺序给出了参数！

```cpp
    int a[] {1, 2, 3, 4, 5};
    std::list<int> lst {1, 2, 3, 4, 5};
    std::forward_list<int> flst {1, 2, 3, 4, 5};

    assert(std::distance(std::begin(a), std::end(a)) == 5);
    assert(std::distance(std::begin(lst), std::end(lst)) == 5);
    assert(std::distance(std::begin(lst), std::end(lst)) == 5);

    assert(std::distance(std::end(a), std::begin(a)) == -5);
```

当迭代器是随机访问迭代器时，`std::distance` 实际上只是进行减法操作；因此，传递“错误顺序”的参数是明确支持并由 C++ 标准认可的。然而，如果相关的迭代器仅仅是双向迭代器（例如 `std::list<int>::iterator`——见 第四章，*容器动物园*），则不支持“错误顺序”的迭代器。你可能期望对于所有迭代器类型，`std::distance(b,a) == -std::distance(a,b)` 应该成立；但考虑一下，`std::distance` 算法本身如何知道你给出的迭代器是否“错误顺序”呢？它唯一能做的事情（在没有 `operator-` 的情况下）是不断递增 `a`——可能超过容器的末尾，进入空间——在徒劳的希望中，它最终会到达 `b`：

```cpp
    // The following line gives an "incorrect" answer!
    // assert(std::distance(std::end(lst), std::begin(lst)) == 1);
    // And this one just segfaults!
    // std::distance(std::end(flst), std::begin(flst));
```

请参考 第四章 中 `std::list` 和 `std::forward_list` 的图示，*容器动物园*，以理解这个代码示例的奇怪行为。

`std::count(a,b,v)` 返回 `a` 和 `b` 之间等于 `v` 的元素数量——也就是说，对于 `e == v` 为真的元素 `e` 的数量。你可以将这个函数视为在效果上等同于 `std::count_if(a,b,&v{return e == v;})`，实际上两种版本应该给出相同的汇编代码。如果 C++ 在 1998 年就有 lambda 表达式，他们可能就不会将 `std::count` 算法放入标准库中。

注意到 `std::count(a,b,v)` 必然会遍历 `a` 和 `b` 之间的 **所有** 元素。它无法利用你可能对范围内数据排列的任何特殊信息。例如，假设我想计算 `std::set<int>` 中 `42` 的实例？我可以以下两种方式之一编写代码：

```cpp
    std::set<int> s { 1, 2, 3, 10, 42, 99 };
    bool present;

    // O(n): compare each element with 42
    present = std::count(s.begin(), s.end(), 42);

    // O(log n): ask the container to look up 42 itself
    present = s.count(42);
```

原始算法 `std::count` 在性能上不如第二种方法，后者只是简单地向 `set` 本身请求答案。这把整个集合的 O(*n*) 遍历转换成了 O(log *n*) 的树查找。同样，`std::unordered_set` 提供了一个大致为 O(1) 的 `count` 方法。

关于这些容器，更多内容请参阅第四章 《容器动物园》；目前这里的关键点是，有时你的数据中存在重要的结构，可以通过选择合适的工具来利用。尽管我在指出标准算法似乎“神奇地”做了正确的事情（例如 `std::distance` 委派给 `(b - a)`），但你不应想象这种“魔法”比它所做的那样更远。标准算法只知道它们被告知的内容，也就是说，只关于你传递给它们的 *迭代器类型* 的属性。它们永远不会根据 *底层数据元素* 之间的关系改变其行为。安排你的代码以利用底层数据中的关系（例如，“这些数据是有序的”，“这个范围跨越整个容器”）是作为程序员的你工作的一部分。

这里有一些类似于 `std::count` 和 `std::count_if` 的算法。

`std::find(a,b,v)` 和 `std::find_if(a,b,p)` 的功能与 `std::count(a,b,v)` 和 `std::count_if(a,b,p)` 分别相似，区别在于，`find` 变体不是遍历整个范围并返回匹配元素的 *计数*，而是只循环到找到第一个匹配项，然后返回指向匹配数据元素的迭代器。还有一个变体 `find_if_not`，它与 `find_if` 类似，但谓词的感测被否定；如果我们在 C++ 的早期历史中得到了 lambdas，这个变体可能就不需要存在了：

```cpp
    template<class InputIterator, class UnaryPredicate>
    InputIterator find_if(InputIterator first, InputIterator last,
      UnaryPredicate p) 
    {
      for (; first != last; ++first) {
        if (p(*first)) {
          return first;
        }
      }
      return last;
    }

    template<class It, class U>
    It find_if_not(It first, It last, U p) {
      return std::find_if(first, last, &{ return !p(e); }); 
    }

    template<class It, class T>
    It find(It first, It last, T value) {
      return std::find_if(first, last, &
        { return e == value; }); 
    }
```

注意，因为 `find` 在找到第一个匹配项时立即返回，所以它平均来说比 `count` 算法（无论什么情况都会扫描整个范围）要快。这种“立即返回”的行为通常被称为“短路”。

`std::all_of(a,b,p)`、`std::any_of(a,b,p)` 和 `std::none_of(a,b,p)` 根据提供的谓词函数 `p` 在范围中的元素中为真的频率返回 `true` 或 `false`。它们都可以建立在 `find` 算法之上，从而免费获得短路行为：

```cpp
    template<class It, class UnaryPredicate>
    bool all_of(It first, It last, UnaryPredicate p)
    {
      return std::find_if_not(first, last, p) == last;
    }

    template <class It, class U>
    bool any_of(It first, It last, U p)
    {
      return std::find_if(first, last, p) != last;
    }

    template <class It, class U>
    bool none_of(It first, It last, U p)
    {
      return std::find_if(first, last, p) == last;
    }
```

我还应该提一下一个与 `find` 相关的算法：`find_first_of`。它实现了在序列中查找固定集合中目标元素首次出现的操作——也就是说，就像 C 标准库中的 `strcspn`，但适用于任何类型，而不仅仅是 `char`。抽象地说，`find_first_of` 接受两个概念参数：要搜索的范围和目标元素集合。由于这是 STL，它们都作为范围传递，也就是说，迭代器对。因此，对这个算法的调用看起来像 `find_first_of(haystack, haystack, needle, needle)`：并排的两个迭代器对。这可能会让人困惑——当算法接受多个类似参数时要小心！

```cpp
    template <class It, class FwdIt>
    It find_first_of(It first, It last, FwdIt targetfirst,
      FwdIt targetlast)
    {
      return std::find_if(first, last, & {
        return std::any_of(targetfirst, targetlast, & {
          return e == t;
        });
      });
    }

    template <class It, class FwdIt, class BinaryPredicate>
    It find_first_of(It first, It last, FwdIt targetfirst,
      FwdIt targetlast, BinaryPredicate p)
    {
      return std::find_if(first, last, & {
        return std::any_of(targetfirst, targetlast, & {
          return p(e, t);
        });
      });
    }
```

注意，“稻草堆”迭代器预期是任何旧的`InputIterator`类型，但“针”迭代器必须至少是`ForwardIterator`。回想一下第二章，“迭代器和范围”，`ForwardIterator`类型的一个重要特点是它们可以被有意义地**复制**，使得相同的范围可以被多次遍历。这正是`find_first_of`所需要的！它对“稻草堆”范围中的每个字符进行一次遍历；因此，“针”必须是可重遍历的——顺便说一下，还必须是有限大小的！相反，没有特别要求“稻草堆”必须是有限的；它可能从可能无界的输入流中提取其元素：

```cpp
    std::istream_iterator<char> ii(std::cin);
    std::istream_iterator<char> iend{};
    std::string s = "hello";

    // Chomp characters from std::cin until finding an 'h', 'e', 'l', or 'o'.
    std::find_first_of(ii, iend, s.begin(), s.end());
```

谈到多个相似参数，让我们通过这两个来结束对简单只读算法的探讨：`std::equal`和`std::mismatch`。

`std::equal(a,b,c,d)`接受两个迭代器对：范围`a,b)`和范围`[c,d)`。如果两个范围元素逐个相等，则返回`true`，否则返回`false`。

`std::mismatch(a,b,c,d)`有点像`find`：它会告诉你确切哪一对元素破坏了匹配：

```cpp
    template<class T> constexpr bool is_random_access_iterator_v =
      std::is_base_of_v<std::random_access_iterator_tag, typename 
      std::iterator_traits<T>::iterator_category>;

    template<class It1, class It2, class B>
    auto mismatch(It1 first1, It1 last1, It2 first2, It2 last2, B p)
    {
      while (first1 != last1 && first2 != last2 && p(*first1, *first2)) {
        ++first1;
        ++first2;
      }
      return std::make_pair(first1, first2);
    }

    template<class It1, class It2>
    auto mismatch(It1 first1, It1 last1, It2 first2, It2 last2)
    {
      return std::mismatch(first1, last1, first2, last2, std::equal_to<>{});
    }

    template<class It1, class It2, class B>
    bool equal(It1 first1, It1 last1, It2 first2, It2 last2, B p)
    {
      if constexpr (is_random_access_iterator_v<It1> &&
        is_random_access_iterator_v<It2>) {
        // Ranges of different lengths can never be equal.
        if ((last2 - first2) != (last1 - first1)) {
          return false;
        }
      }
      return std::mismatch(first1, last1, first2, last2, p) ==
        std::make_pair(last1, last2);
    }

    template<class It1, class It2>
    bool equal(It1 first1, It1 last1, It2 first2, It2 last2)
    {
      return std::equal(first1, last1, first2, last2, std::equal_to<>{});
    }
```

注意到使用了`std::equal_to<>{}`作为谓词对象；在这本书中，我们不会深入探讨内置谓词，所以请假设`std::equal_to<>{}`是一个行为类似于`[{ return a == b; }`的对象，但涉及更多的**完美转发**。

最后，再次注意！C++17 标准库中的许多双范围算法也有被称为半范围算法的变体形式。例如，除了`std::mismatch(a,b,c,d)`之外，你还会发现`std::mismatch(a,b,c)`——第二个范围的“结束”点简单地假设为`c + std::distance(a, b)`。如果`c`实际上指向一个容器，其中`c + std::distance(a, b)`将是“超出范围”，那么，运气不佳！

因为“运气不佳”永远不是对技术问题的真正**伟大**回答，C++17 标准为许多在 C++14 中存在的半范围算法添加了安全的双范围变体。

# 使用 std::copy 移动数据

我们刚刚看到了几个双范围算法。`<algorithm>`头文件充满了双范围算法及其兄弟半范围算法。这种算法可能有多简单？

一个合理的回答可能是：“将每个数据元素从第一个范围复制到第二个范围。”实际上，STL 提供了这个算法，名为`std::copy`：

```cpp
    template<class InIt, class OutIt>
    OutIt copy(InIt first1, InIt last1, OutIt destination)
    {
      while (first1 != last1) {
        *destination = *first1;
        ++first1;
        ++destination;
      }
      return destination;
    }
```

注意，这是一个半范围算法。标准库实际上没有提供`std::copy`的双范围版本；假设如果你实际上正在尝试写入缓冲区，那么你一定已经检查了它的大小，所以在循环中检查“我们是否到达了缓冲区的末尾”将是既冗余又低效的。

现在，我可以几乎听到你在惊叹：“天哪！这正是导致我们有了 `strcpy`、`sprintf` 和 `gets` 的那种粗糙逻辑！这是对缓冲区溢出的邀请！”好吧，*如果你这样惊叹，那么你对 `gets` 的不良行为判断是正确的——实际上，`gets` 函数已经被正式从 C++17 标准库中移除。你对 `sprintf` 的看法也是正确的——任何需要该功能的人最好使用经过范围检查的版本 `snprintf`，在这个上下文中，它类似于一个“双范围算法”。但关于 `strcpy`，我不同意。对于 `gets`，确定输出缓冲区的正确大小是*不可能的*；对于 `sprintf`，是*困难的*；但对于 `strcpy`，是*微不足道的*：你只需测量输入缓冲区的 `strlen`，这就是你的答案。同样，对于 `std::copy`，"输入元素消耗" 和 "输出元素产生" 之间的关系是一对一，因此输出缓冲区的大小并不构成技术挑战。

注意，我们称之为 `destination` 的参数是一个*输出迭代器*。这意味着我们可以使用 `std::copy`，不仅可以在内存中移动数据，甚至可以将数据提供给任意的“接收”函数。例如：

```cpp
    class putc_iterator : public boost::iterator_facade<
      putc_iterator, // T
      const putc_iterator, // value_type
      std::output_iterator_tag
      >
    {
      friend class boost::iterator_core_access;

       auto& dereference() const { return *this; }
       void increment() {}
       bool equal(const putc_iterator&) const { return false; }
       public:
       // This iterator is its own proxy object!
       void operator= (char ch) const { putc(ch, stdout); }
    };

    void test()
    {
      std::string s = "hello";
      std::copy(s.begin(), s.end(), putc_iterator{});
    }
```

你可能会发现将这个版本的 `putc_iterator` 与 第二章 中提到的版本进行比较是有益的；这个版本使用了在 第二章 的末尾介绍的 `boost::iterator_facade`，并且还使用了一个常见的技巧来返回 `*this` 而不是一个新的代理对象。

现在，我们可以利用 `destination` 的灵活性来解决我们对缓冲区溢出的担忧！假设我们不是写入一个固定大小的数组，而是写入一个可调整大小的 `std::vector`（参见 第四章 的“容器动物园”）。那么，“写入一个元素”对应于“在向量上推入一个元素”。因此，我们可以编写一个非常类似于 `putc_iterator` 的输出迭代器，它将使用 `push_back` 而不是 `putc`，然后我们就有了一种防止溢出的填充向量的方法。实际上，标准库在 `<iterator>` 头文件中就提供了这样的输出迭代器：

```cpp
    namespace std {
      template<class Container>
      class back_insert_iterator {
        using CtrValueType = typename Container::value_type;
        Container *c;
      public:
        using iterator_category = output_iterator_tag;
        using difference_type = void;
        using value_type = void;
        using pointer = void;
        using reference = void;

        explicit back_insert_iterator(Container& ctr) : c(&ctr) {}

        auto& operator*() { return *this; }
        auto& operator++() { return *this; }
        auto& operator++(int) { return *this; }

        auto& operator= (const CtrValueType& v) {
            c->push_back(v);
            return *this;
        }
        auto& operator= (CtrValueType&& v) {
            c->push_back(std::move(v));
            return *this;
        }
      };

      template<class Container>
      auto back_inserter(Container& c)
      {
         return back_insert_iterator<Container>(c);
      }
    }

    void test()
    {
      std::string s = "hello";
      std::vector<char> dest;
      std::copy(s.begin(), s.end(), std::back_inserter(dest));
      assert(dest.size() == 5);
    }
```

函数调用 `std::back_inserter(dest)` 简单地返回一个 `back_insert_iterator` 对象。在 C++17 中，我们可以依赖模板类型推导来构造函数，并将该函数体的内容简单地写为 `return std::back_insert_iterator(dest)`；或者完全省略该函数，直接在我们的代码中写 `std::back_insert_iterator(dest)`--在 C++14 代码中则必须使用 `std::back_inserter(dest)` 来“应付”。然而，为什么我们要输入那么多额外的代码？名称 `back_inserter` 被故意选择为易于记忆，因为它是我们预期最常使用的。尽管 C++17 允许我们用 `std::pair` 替代 `std::make_pair`，用 `std::tuple` 替代 `std::make_tuple`，但在 C++17 中用繁琐的 `std::back_insert_iterator` 替代 `std::back_inserter` 是愚蠢的。即使在 C++17 中，你也应该首选 `std::back_inserter(dest)`。

# 主题变奏 - std::move 和 std::move_iterator

如你所猜，或者你可能已经在前面的实现中注意到，`std::copy` 算法通过从输入范围复制元素到输出工作。截至 C++11，你可能会想：如果我们不是 *复制* 元素，而是使用移动语义将它们从输入 *移动* 到输出会怎样？

STL 为此问题提供了两种不同的方法。第一种方法是最直接的：有一个 `std::move` 算法（定义在 `<algorithm>` 头文件中），其定义如下：

```cpp
    template<class InIt, class OutIt>
    OutIt move(InIt first1, InIt last1, OutIt destination)
    {
      while (first1 != last1) {
        *destination = std::move(*first1);
        ++first1;
        ++destination;
      }
      return destination;
    }
```

它与 `std::copy` 算法完全相同，只是在输入元素上添加了一个 `std::move` 操作（小心--这个内部 `std::move`，带有一个 *参数*，定义在 `<utility>` 头文件中，与定义在 `<algorithm>` 中的外部三个参数的 `std::move` 完全不同！它们共享一个名称是不幸的。讽刺的是，其他少数 STL 函数也遭受了类似的情况，比如 `std::remove`；参见 *从排序数组中删除* 部分，以及 第十二章，*文件系统*)。

另一种方法是我们之前看到的 `back_inserter` 的变体。而不是更换核心 *算法*，我们可以继续使用 `std::copy` 但以不同的方式参数化。假设我们传递了一个新的迭代器类型，它（就像 `back_inserter` 一样）围绕我们的原始对象并改变其行为？特别是，我们需要一个输入迭代器，其 `operator*` 返回一个右值。我们可以做到这一点！

```cpp
    template<class It>
    class move_iterator {
      using OriginalRefType = typename std::iterator_traits<It>::reference;
      It iter;
      public:
       using iterator_category = typename
         std::iterator_traits<It>::iterator_category;
       using difference_type = typename
         std::iterator_traits<It>::difference_type;
       using value_type = typename std::iterator_traits<It>::value_type;
       using pointer = It;
       using reference = std::conditional_t<
         std::is_reference_v<OriginalRefType>,
         std::remove_reference_t<OriginalRefType>&&,
         OriginalRefType
         >;

       move_iterator() = default;
       explicit move_iterator(It it) : iter(std::move(it)) {}

       // Allow constructing or assigning from any kind of move-iterator.
       // These templates also serve as our own type's copy constructor
       // and assignment operator, respectively.
       template<class U>
       move_iterator(const move_iterator<U>& m) : iter(m.base()) {}
       template<class U>
       auto& operator=(const move_iterator<U>& m)
         { iter = m.base(); return *this; }

       It base() const { return iter; }

       reference operator*() { return static_cast<reference>(*iter); }
       It operator->() { return iter; }
       decltype(auto) operator[](difference_type n) const 
         { return *std::move(iter[n]); }

      auto& operator++() { ++iter; return *this; }
      auto& operator++(int)
        { auto result = *this; ++*this; return result; }
      auto& operator--() { --iter; return *this; }
      auto& operator--(int)
        { auto result = *this; --*this; return result; } 

      auto& operator+=(difference_type n) const
        { iter += n; return *this; }
      auto& operator-=(difference_type n) const
        { iter -= n; return *this; }
    };

    // I've omitted the definitions of non-member operators
    // == != < <= > >= + - ; can you fill them in?

    template<class InputIterator>
    auto make_move_iterator(InputIterator& c) 
    {
      return move_iterator(c);
    }
```

对于这段代码的密集性，我表示歉意；请相信你可以安全地跳过细节。对于那些喜欢这类东西的人来说，你可能注意到我们提供了一个从 `move_iterator<U>` 到模板构造函数，它恰好也充当了我们的复制构造函数（当 `U` 与 `It` 类型相同时）；我们还提供了许多成员函数（例如 `operator[]` 和 `operator--`），它们的主体对于许多可能的 `It` 类型（例如，当 `It` 是一个前向迭代器时--见第二章，*迭代器和范围*）将产生错误，但这是可以的，因为它们的主体只有在用户实际在编译时尝试调用这些函数时才会实例化（如果用户实际上尝试对 `move_iterator<list_of_ints::iterator>` 进行 `--` 操作，那么当然会产生编译时错误）。

就像 `back_inserter` 一样，请注意，STL 为那些没有构造函数模板类型推导的预 C++17 编译器提供了一个辅助函数 `make_move_iterator`。在这种情况下，就像 `make_pair` 和 `make_tuple` 一样，"辅助" 名称比实际类名更丑陋，所以我建议你在代码中使用 C++17 的特性；如果你不需要，为什么要多打五个字符并实例化一个额外的函数模板呢？

现在我们有两种不同的方式将数据从一个容器或范围移动到另一个：`std::move` 算法和 `std::move_iterator` 适配器类。以下是这两种习惯用法的示例：

```cpp
    std::vector<std::string> input = {"hello", "world"};
    std::vector<std::string> output(2);

    // First approach: use the std::move algorithm
    std::move(input.begin(), input.end(), output.begin());

    // Second approach: use move_iterator
    std::copy(
      std::move_iterator(input.begin()),
      std::move_iterator(input.end()),
      output.begin()
    );
```

第一种方法，使用 `std::move`，如果你只是移动数据，显然要干净得多。那么，为什么标准库要提供这种“更混乱”的方法 `move_iterator` 呢？为了回答这个问题，我们不得不探索另一个与 `std::copy` 基本相关的算法。

# 使用 std::transform 进行复杂复制

你可能已经注意到了，当我们之前展示了 `std::copy` 的实现时，两个迭代器类型参数的 `value_type` 并没有限制必须相同。这是一个特性，而不是错误！这意味着我们可以编写依赖于隐式转换的代码，并且它将正确地执行：

```cpp
    std::vector<const char *> input = {"hello", "world"};
    std::vector<std::string> output(2);

    std::copy(input.begin(), input.end(), output.begin());

    assert(output[0] == "hello");
    assert(output[1] == "world");
```

看起来很简单，对吧？仔细看看！在我们对 `std::copy` 的实例化中，有一个调用隐式构造函数，它将 `const char *`（`*input.begin()` 的类型）转换为 `std::string`（`*output.begin()` 的类型）。所以，我们又一次看到了一个示例，即通用代码通过简单地提供某些迭代器类型，就能执行令人惊讶的复杂操作。

但有时你希望在复制操作期间应用一个复杂的转换函数--比隐式转换更复杂的函数。标准库已经为你准备好了！

```cpp
    template<class InIt, class OutIt, class Unary>
    OutIt transform(InIt first1, InIt last1, OutIt destination, Unary op)
    {
      while (first1 != last1) {
        *destination = op(*first1);
        ++first1;
        ++destination;
      }
      return destination;
    }

    void test() 
    {
      std::vector<std::string> input = {"hello", "world"};
      std::vector<std::string> output(2);

      std::transform(
        input.begin(),
        input.end(),
        output.begin(),
        [](std::string s) {
          // It works for transforming in-place, too!
          std::transform(s.begin(), s.end(), s.begin(), ::toupper);
          return s;
        }
      );

      assert(input[0] == "hello");
      assert(output[0] == "HELLO");
    }
```

有时候，你需要使用一个接受 *两个* 参数的函数来进行转换。库已经为你准备好了：

```cpp
    template<class InIt1, class InIt2, class OutIt, class Binary>
    OutIt transform(InIt1 first1, InIt1 last1, InIt2 first2, OutIt destination,
      Binary op)
    {
      while (first1 != last1) {
        *destination = op(*first1, *first2);
        ++first1;
        ++first2;
        ++destination;
      }
      return destination;
    }
```

这个版本的 `std::transform` 可以幽默地描述为一种一又二分之一的范围算法！

（关于三个参数的函数？四个参数的函数？不幸的是，`std::transform` 没有完全可变参数版本；可变模板直到 C++11 才被引入到 C++ 中。你可以尝试实现一个可变参数版本，看看会遇到什么问题——它们是可克服的，但绝对不是微不足道的。）

`std::transform` 的存在为我们提供了将数据元素从一个地方移动到另一个地方的第三种方法：

```cpp
    std::vector<std::string> input = {"hello", "world"};
    std::vector<std::string> output(2);

    // Third approach: use std::transform
    std::transform(
      input.begin(),
      input.end(),
      output.begin(),
      std::move<std::string&>
    );
```

我当然不推荐这种方法！它的最大和最明显的红旗是它包含了 `std::move` 模板的显式特化。每当你在模板名称后看到显式特化——那些模板名称后的尖括号——这几乎可以肯定是非常微妙和脆弱的代码。高级读者可能会喜欢弄清楚编译器如何推断出我指的是两个 `std::move` 中的哪一个；记住，一个在 `<utility>` 中，一个在 `<algorithm>` 中。

# 只写范围算法

我们在本章开始时查看了一些算法，例如 `std::find`，这些算法遍历一个范围，按顺序读取其元素而不进行修改。你可能会惊讶地发现逆操作也是有意义的：存在一组标准算法，它们遍历一个范围 *修改* 每个元素而不读取它！

`std::fill(a,b,v)` 做的正如其名所暗示的那样：将给定范围 `[a,b)` 的每个元素填充为提供的值 `v` 的副本。

`std::iota(a,b,v)` 稍微更有趣：它将给定范围的元素填充为 `++v` 的副本。也就是说，`std::iota(a,b,42)` 将将 `a[0]` 设置为 42，`a[1]` 设置为 43，`a[2]` 设置为 44，以此类推，直到 `b`。这个算法有趣的名字来源于 APL 编程语言，其中名为 `ι`（希腊字母 *iota*）的函数执行了这个操作。这个算法的另一个有趣之处在于，出于某种原因，它的定义可以在标准 `<numeric>` 头文件中找到，而不是在 `<algorithm>` 中。它就是这样一种怪异的算法。

`std::generate(a,b,g)` 更有趣：它将给定范围的元素填充为 `g()` 的连续结果，无论它是什么：

```cpp
    template<class FwdIt, class T>
    void fill(FwdIt first, FwdIt last, T value) {
      while (first != last) {
        *first = value;
         ++first;
      }
    }

    template<class FwdIt, class T>
    void iota(FwdIt first, FwdIt last, T value) {
      while (first != last) {
        *first = value;
        ++value;
        ++first;
      }
    }

    template<class FwdIt, class G>
    void generate(FwdIt first, FwdIt last, G generator) {
      while (first != last) {
        *first = generator();
        ++first;
      }
    }
```

这里是使用这些标准算法填充具有不同内容的字符串向量的示例。测试你的理解：你是否理解为什么每个调用会产生这样的输出？我选择的 `std::iota` 的例子特别有趣（但在现实世界的代码中不太可能有用）：

```cpp
    std::vector<std::string> v(4);

    std::fill(v.begin(), v.end(), "hello");
    assert(v[0] == "hello");
    assert(v[1] == "hello");
    assert(v[2] == "hello");
    assert(v[3] == "hello");

    std::iota(v.begin(), v.end(), "hello");
    assert(v[0] == "hello");
    assert(v[1] == "ello");
    assert(v[2] == "llo");
    assert(v[3] == "lo");

    std::generate(v.begin(), v.end(), [i=0]() mutable {
      return ++i % 2 ? "hello" : "world";
    });
    assert(v[0] == "hello");
    assert(v[1] == "world");
    assert(v[2] == "hello");
    assert(v[3] == "world");
```

# 影响对象生命周期的算法

`<memory>` 头文件提供了一组名为 `std::uninitialized_copy`、`std::uninitialized_default_construct` 和 `std::destroy`（完整列表，请参考在线参考资料，如 [cppreference.com](http://cppreference.com)）的晦涩算法。考虑以下使用显式析构函数调用销毁范围元素的算法：

```cpp
    template<class T>
    void destroy_at(T *p)
    {
      p->~T();
    }

    template<class FwdIt>
    void destroy(FwdIt first, FwdIt last)
    {
      for ( ; first != last; ++first) {
        std::destroy_at(std::addressof(*first));
      }
    }
```

注意，`std::addressof(x)`是一个方便的小辅助函数，它返回其参数的地址；它与`&x`完全相同，只是在`x`是某些类类型并且残酷地重载了自己的`operator&`的罕见情况下除外。

考虑这个使用显式 placement-new 语法“复制构造”到范围元素中的算法（注意，如果在复制过程中抛出异常，它会干净利落地清理）。这个算法显然不应该用于任何已经存在元素的任何范围；所以以下例子看起来非常牵强：

```cpp
    template<class It, class FwdIt>
    FwdIt uninitialized_copy(It first, It last, FwdIt out)
    {
      using T = typename std::iterator_traits<FwdIt>::value_type;
      FwdIt old_out = out;
      try {
        while (first != last) {
          ::new (static_cast<void*>(std::addressof(*out))) T(*first);
          ++first;
          ++out;
        }
        return out;
      } catch (...) {
        std::destroy(old_out, out);
        throw;
      }
    }

    void test()
    { 
      alignas(std::string) char b[5 * sizeof (std::string)];  
      std::string *sb = reinterpret_cast<std::string *>(b);

      std::vector<const char *> vec = {"quick", "brown", "fox"};

      // Construct three std::strings.
      auto end = std::uninitialized_copy(vec.begin(), vec.end(), sb);

      assert(end == sb + 3);

      // Destroy three std::strings.
      std::destroy(sb, end);
    }
```

我们将在第四章中了解更多关于这些算法应该如何使用的信息，*容器动物园*，当我们讨论`std::vector`时。

# 我们的第一个排列算法：std::sort

到目前为止，我们讨论的所有算法都是简单地按顺序遍历它们给定的范围，线性地从第一个元素到下一个元素。我们下一系列的算法不会这样表现。相反，它将给定范围中元素的值打乱，使得相同的值仍然出现，但顺序不同。这种操作的数学名称是排列。

最简单的排列算法要描述的是`std::sort(a,b)`。它做的是名字暗示的事情：对给定的范围进行排序，使得最小的元素出现在前面，最大的元素出现在后面。为了确定哪些元素是“最小的”，`std::sort(a,b)`使用`operator<`。

如果你想有不同的顺序，你可以尝试重载`operator<`以在不同的条件下返回`true`--但可能你应该使用算法的三参数版本，`std::sort(a,b,cmp)`。第三个参数应该是一个比较器；也就是说，一个函数、仿函数或 lambda，当其第一个参数“小于”第二个参数时返回`true`。例如：

```cpp
    std::vector<int> v = {3, 1, 4, 1, 5, 9};
    std::sort(v.begin(), v.end(), [](auto&& a, auto&& b) {
      return a % 7 < b % 7;
    });
    assert((v == std::vector{1, 1, 9, 3, 4, 5}));
```

注意，我在这个例子中仔细选择了我的 lambda，以便以确定的方式对数组进行排序。如果我用函数`(a % 6 < b % 6)`代替，那么可能会有两种可能的输出：要么是`{1, 1, 3, 9, 4, 5}`，要么是`{1, 1, 9, 3, 4, 5}`。标准的`sort`算法对于在给定比较函数下恰好相等的元素的相对位置没有任何保证！

为了解决这个问题（如果它确实是一个问题），你应该将你的`std::sort`使用替换为`std::stable_sort`。后者可能稍微慢一点，但它将保证在相等元素的情况下保留原始顺序--也就是说，在这种情况下，我们将得到`{1, 1, 3, 9, 4, 5}`，因为在原始（未排序）向量中，元素`3`在元素`9`之前。

使用 `sort` 和 `stable_sort` 还可能发生更糟糕的事情——如果我选择了比较函数 `(a % 6 < b)` 会怎样？那么我就会有一些元素对 `x, y`，其中 `x < y` 同时 `y < x`！（原始向量中的一个这样的元素对是 `5` 和 `9`。）在这种情况下，没有什么可以拯救我们；我们传递了一个“比较函数”，而这个函数根本就不是比较函数！这与传递空指针给 `std::sort` 的先决条件相违背。在排序数组时，确保你是基于一个有意义的比较函数进行排序！

# 交换、反转和划分

STL 除了 `std::sort` 之外还包含大量排列算法。许多这些算法可以被视为“构建块”，它们仅实现了整体排序算法的一小部分。

`std::swap(a,b)` 是最基本的构建块；它只是接受它的两个参数并将它们“交换”——也就是说，它交换它们的值。这是通过给定类型的移动构造函数和移动赋值运算符实现的。`swap` 在标准算法中实际上有点特殊，因为它是一个如此原始的操作，而且几乎总是有比执行 `temp = a; a = b; b = temp;` 等效操作更快的方式来交换两个任意对象。对于标准库类型（如 `std::vector`）的常用惯例是类型本身实现一个 `swap` 成员函数（如 `a.swap(b)`），然后在类型的同一命名空间中添加 `swap` 函数的重载——也就是说，如果我们正在实现 `my::obj`，我们会在命名空间 `my` 中添加重载，这样对于该特定类型的 `swap(a,b)`，将调用 `a.swap(b)` 而不是执行三个移动操作。以下是一个例子：

```cpp
    namespace my {
      class obj {
        int v;
      public:
        obj(int value) : v(value) {}

        void swap(obj& other) {
          using std::swap;
          swap(this->v, other.v);
        }
      };

      void swap(obj& a, obj& b) {
        a.swap(b);
      }
    } // namespace my

    void test()
    {
      int i1 = 1, i2 = 2;
      std::vector<int> v1 = {1}, v2 = {2};
      my::obj m1 = 1, m2 = 2;
      using std::swap;
      swap(i1, i2); // calls std::swap<int>(int&, int&)
      swap(v1, v2); // calls std::swap(vector&, vector&)
      swap(m1, m2); // calls my::swap(obj&, obj&)
    }
```

现在我们有了 `swap` 和双向迭代器，我们可以构建 `std::reverse(a,b)`，这是一个排列算法，它通过交换第一个元素与最后一个元素、第二个元素与倒数第二个元素，依此类推，简单地反转元素范围的顺序。`std::reverse` 的一个常见应用是反转字符串中较大的块顺序——例如，反转句子中单词的顺序：

```cpp
    void reverse_words_in_place(std::string& s)
    {
      // First, reverse the whole string.
      std::reverse(s.begin(), s.end());

      // Next, un-reverse each individual word.
      for (auto it = s.begin(); true; ++it) {
        auto next = std::find(it, s.end(), ' ');
        // Reverse the order of letters in this word.
        std::reverse(it, next);
        if (next == s.end()) {
          break;
        }
        it = next;
      }
    }

    void test()
    {
      std::string s = "the quick brown fox jumps over the lazy dog";
      reverse_words_in_place(s);
      assert(s == "dog lazy the over jumps fox brown quick the");
    }
```

对 `std::reverse` 的实现进行一点小的调整，我们得到了排序的另一个构建块，即 `std::partition`。与 `std::reverse` 从两端遍历范围无条件地交换每一对元素不同，`std::partition` 只有在元素相对于某个谓词函数“顺序错误”时才交换它们。在以下示例中，我们将所有 *偶数* 元素划分到范围的起始位置，所有 *奇数* 元素划分到范围的末尾。如果我们使用 `std::partition` 来构建 Quicksort 排序程序，我们将把小于枢轴元素的元素划分到范围的起始位置，把大于枢轴元素的元素划分到范围的末尾：

```cpp
    template<class BidirIt>
    void reverse(BidirIt first, BidirIt last)
    {
      while (first != last) {
        --last;
        if (first == last) break;
        using std::swap;
        swap(*first, *last);
        ++first;
      }
    }

    template<class BidirIt, class Unary>
    auto partition(BidirIt first, BidirIt last, Unary p)
    {
      while (first != last && p(*first)) {
        ++first;
      }

      while (first != last) {
        do {
          --last;
        } while (last != first && !p(*last));
        if (first == last) break;
        using std::swap;
        swap(*first, *last);
        do {
          ++first;
        } while (first != last && p(*first));
      }
      return first;
    }

    void test()  
    {
      std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6, 5};
      auto it = std::partition(v.begin(), v.end(), [](int x) {
        return x % 2 == 0;
      });
      assert(it == v.begin() + 3);
      assert((v == std::vector{6, 2, 4, 1, 5, 9, 1, 3, 5}));
    }
```

你可能会注意到前面代码的一个有趣之处：`reverse`和`partition`的代码几乎完全相同！唯一的区别是`partition`包含一个令人不快的 do-while 循环，而`reverse`只有简单的递增或递减。

你可能也注意到`partition`中的第一个 do-while 循环与我们之前看到的标准算法等价；即`std::find_if_not`。第二个 do-while 循环类似于`std::find_if`... 但它需要向后运行，而不是向前！不幸的是，我们没有`std::rfind_if`这样的算法。但是——正如你可能已经猜到的——标准库不会让我们陷入困境。

我们需要一个在`std::find_if`的目的上表现得像迭代器，但迭代“反向”的东西。标准库以`std::reverse_iterator<FwdIt>`适配器的形式提供了这个确切的东西。我们不会展示它的代码；如果你需要复习如何实现它，请回顾第二章，*迭代器和范围*。简而言之，`std::reverse_iterator<FwdIt>`对象就像一个`FwdIt`对象一样包装和表现，除了当你递增包装器时，它会递减被包装的对象，反之亦然。因此，我们可以用`reverse_iterator`来写`partition`，如下所示：

```cpp
    // Shorthands for "reversing" and "unreversing".
    template<class It>
    auto rev(It it) {
      return std::reverse_iterator(it);
    };

    template<class InnerIt>
    auto unrev(std::reverse_iterator<InnerIt> it) {
      return it.base();
    }

    template<class BidirIt, class Unary>
    auto partition(BidirIt first, BidirIt last, Unary p)
    {
      first = std::find_if_not(first, last, p);

      while (first != last) {
        last = unrev(std::find_if(rev(last), rev(first), p));
        if (first == last) break;
        using std::swap;
        swap(*first++, *--last);
        first = std::find_if_not(first, last, p);
      }
      return first;
    }
```

当然，有时在保持每个分区中元素相对顺序不变的情况下对范围进行分区是有用的。在这些情况下，可以使用`std::stable_partition(a,b,p)`（但请参阅关于`stable_partition`的警告部分：它可能会使用`operator new`分配内存）。

有一些非排列算法也处理分区：

`std::is_partitioned(a,b,p)`如果给定的范围已经通过谓词`p`分区（即满足`p`的所有元素都在前面，而不满足`p`的所有元素都在后面），则返回`true`。

`std::partition_point(a,b,p)`使用二分查找来找到已经分区范围内不满足`p`的第一个元素。

`std::partition_copy(a,b,ot,of,p)`将范围`[a,b)`中的每个元素复制到输出迭代器之一：对于满足`p(e)`的元素，`*ot++ = e`；对于不满足`p(e)`的元素，`*of++ = e`。

顺便说一下，如果你只想得到一个输出序列或另一个，那么你可以分别使用`std::copy_if(a,b,ot,p)`或`std::remove_copy_if(a,b,of,p)`。

# 旋转和排列

记得我们来自 *交换、反转和分区* 的代码，用来反转句子中单词的顺序吗？当“句子”只包含两个单词时，还有另一种看待反转的方法：你可以将其视为底层范围中元素的 *循环旋转*。`std::rotate(a,mid,b)` 将范围 `[a,b)` 的元素旋转，使得原本由 `mid` 指向的元素现在位于 `a`（并返回一个指向原本位于 `a` 的元素的迭代器）：

```cpp
    template<class FwdIt>
    FwdIt rotate(FwdIt a, FwdIt mid, FwdIt b)
    {
      auto result = a + (b - mid);

      // First, reverse the whole range.
      std::reverse(a, b);

      // Next, un-reverse each individual segment.
      std::reverse(a, result);
      std::reverse(result, b);

      return result;
    }

    void test()
    {
      std::vector<int> v = {1, 2, 3, 4, 5, 6};
      auto five = std::find(v.begin(), v.end(), 5);
      auto one = std::rotate(v.begin(), five, v.end());
      assert((v == std::vector{5, 6, 1, 2, 3, 4}));
      assert(*one == 1);
    }
```

另一个杂项但有时有用的排列算法是 `std::next_permutation(a,b)`。在循环中调用此函数将遍历所有 *n* 个元素的排列，这可能在你尝试暴力解决旅行商问题（小规模实例）时很有用：

```cpp
    std::vector<int> p = {10, 20, 30};
    std::vector<std::vector<int>> results;

    // Collect the permutations of these three elements.
    for (int i=0; i < 6; ++i) {
      results.push_back(p);
      std::next_permutation(p.begin(), p.end());
    }

    assert((results == std::vector<std::vector<int>>{
      {10, 20, 30},
      {10, 30, 20},
      {20, 10, 30},
      {20, 30, 10},
      {30, 10, 20},
      {30, 20, 10},
    }));
```

注意，`next_permutation` 使用“小于”关系来确定一个排列在字典序上“小于”另一个排列；例如，`{20, 10, 30}` 在字典序上“小于” `{20, 30, 10}`，因为 10 小于 30。因此，`next_permutation` 也有一个基于比较器的版本：`std::next_permutation(a,b,cmp)`。还有 `std::prev_permutation(a,b)` 和 `std::prev_permutation(a,b,cmp)`，它们在字典序上“向下”计数而不是“向上”。

顺便说一句，要按这种方式在字典序上比较两个序列，你可以使用来自 *只读范围算法* 部分的 `std::mismatch`，或者你可以直接使用标准提供的 `std::lexicographical_compare(a,b,c,d)`。

# 堆和堆排序

`std::make_heap(a,b)`（或其基于比较器的版本，`std::make_heap(a,b,cmp)`）接受一个未排序的元素范围，并将它们重新排列成一个满足最大堆属性的顺序：具有最大堆属性的数组中，索引 *i* 的每个元素将至少与索引 2*i*+1 和 2*i*+2 的元素之一相等。这意味着所有元素中的最大值将位于索引 0。这表明，最大元素将位于索引 0。

`std::push_heap(a,b)`（或其基于比较器的版本）假设范围 `[a,b-1)` 已经是一个最大堆。它将当前位于 `b[-1]` 的元素“冒泡”起来，通过与堆中的父元素交换，直到整个范围 `[a,b)` 的最大堆属性得到恢复。请注意，`make_heap` 可以通过简单地循环调用 `std::push_heap(a,++b)` 来实现。

`std::pop_heap(a,b)`（或其基于比较器的版本）假设范围 `[a,b)` 已经是一个最大堆。它将 `a[0]` 与 `b[-1]` 交换，使得最大元素现在位于范围的 *尾部* 而不是 *前端*；然后它与堆中的一个子元素交换，依此类推，“冒泡”下来直到最大堆属性得到恢复。在调用 `pop_heap(a,b)` 之后，最大元素将位于 `b[-1]`，范围 `[a, b-1)` 将具有最大堆属性。

`std::sort_heap(a,b)`（或其基于比较器的版本）接受一个具有最大堆属性的范围，并通过重复调用`std::pop_heap(a, b--)`将其排列成排序顺序。

使用这些构建块，我们可以实现经典的“堆排序”算法。标准库中的`std::sort`函数可能合理地实现如下（但在实践中通常实现为混合算法，例如“introsort”）：

```cpp
    template<class RandomIt>
    void push_heap(RandomIt a, RandomIt b)
    {
      auto child = ((b-1) - a);
      while (child != 0) {
        auto parent = (child - 1) / 2;
        if (a[child] < a[parent]) {
          return; // max-heap property has been restored
        }
        std::iter_swap(a+child, a+parent);
        child = parent;
      }
    }

    template<class RandomIt>
    void pop_heap(RandomIt a, RandomIt b)
    {
      using DistanceT = decltype(b - a);

      std::iter_swap(a, b-1);

      DistanceT parent = 0;
      DistanceT new_heap_size = ((b-1) - a);

      while (true) {
        auto leftchild = 2 * parent + 1;
        auto rightchild = 2 * parent + 2;
        if (leftchild >= new_heap_size) {
          return;
        }
        auto biggerchild = leftchild;
        if (rightchild < new_heap_size && a[leftchild] < a[rightchild]) {
          biggerchild = rightchild;
        }
        if (a[biggerchild] < a[parent]) {
          return; // max-heap property has been restored
        }
        std::iter_swap(a+parent, a+biggerchild);
        parent = biggerchild;
      }
    }

    template<class RandomIt>
    void make_heap(RandomIt a, RandomIt b)
    {
      for (auto it = a; it != b; ) {
        push_heap(a, ++it);
      }
    }

    template<class RandomIt>
    void sort_heap(RandomIt a, RandomIt b)
    {
      for (auto it = b; it != a; --it) {
        pop_heap(a, it);
      }
    }

    template<class RandomIt>
    void sort(RandomIt a, RandomIt b)
    {
      make_heap(a, b);
      sort_heap(a, b);
    }
```

我们将在第四章“容器动物园”中看到`push_heap`和`pop_heap`的另一个应用，当我们讨论`std::priority_queue`时。

# 合并和归并排序

既然我们谈论到了排序算法，让我们以不同的方式编写`sort`！

`std::inplace_merge(a,mid,b)`接受一个已经通过`std::sort(a,mid)`和`std::sort(mid,b)`排序的范围`a,b)`，并将两个子范围合并成一个排序的范围。我们可以使用这个构建块来实现经典的归并排序算法：

```cpp
    template<class RandomIt>
    void sort(RandomIt a, RandomIt b)
    {
      auto n = std::distance(a, b);
      if (n >= 2) {
        auto mid = a + n/2;
        std::sort(a, mid);
        std::sort(mid, b);
        std::inplace_merge(a, mid, b);
      }
    }
```

然而，请注意！名称`inplace_merge`似乎暗示合并是在“原地”发生的，无需任何额外的缓冲空间；但实际上并非如此。实际上，`inplace_merge`函数会为其自身分配一个缓冲区，通常是通过调用`operator new`。如果你在一个堆分配有问题的环境中编程，那么你应该*避免*使用`inplace_merge`。

可能会在堆上分配临时缓冲区的其他标准算法是`std::stable_sort`和`std::stable_partition`。

`std::merge(a,b,c,d,o)`是非分配合并算法；它接受两个迭代器对，代表范围`[a,b)`和`[c,d)`，并将它们合并到由`o`定义的输出范围中。

# 使用`std::lower_bound`在有序数组中进行搜索和插入

一旦数据范围被排序，就可以使用二分搜索在该数据内进行搜索，而不是使用较慢的线性搜索。实现二分搜索的标准算法称为`std::lower_bound(a,b,v)`：

```cpp
    template<class FwdIt, class T, class C>
    FwdIt lower_bound(FwdIt first, FwdIt last, const T& value, C lessthan)
    {
      using DiffT = typename std::iterator_traits<FwdIt>::difference_type;
      FwdIt it;
      DiffT count = std::distance(first, last);

      while (count > 0) {
        DiffT step = count / 2;
        it = first;
        std::advance(it, step);
        if (lessthan(*it, value)) {
          ++it;
          first = it;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      return first;
    }

    template<class FwdIt, class T>
    FwdIt lower_bound(FwdIt first, FwdIt last, const T& value) 
    {
      return std::lower_bound(first, last, value, std::less<>{});
    }
```

此函数返回一个指向范围中第一个不小于给定值`v`的元素的迭代器。如果范围中已经存在该值的实例，则返回的迭代器将指向它（实际上，它将指向范围中的第一个这样的值）。如果没有该值的实例，则返回的迭代器将指向`v`应该放置的位置。

我们可以使用`lower_bound`的返回值作为`vector::insert`的输入，以便在保持其排序顺序的同时将`v`插入到排序向量的正确位置：

```cpp
    std::vector<int> vec = {3, 7};
    for (int value : {1, 5, 9}) {
      // Find the appropriate insertion point...
      auto it = std::lower_bound(vec.begin(), vec.end(), value);
      // ...and insert our value there.
      vec.insert(it, value);
    }
    // The vector has remained sorted.
    assert((vec == std::vector{1, 3, 5, 7, 9}));
```

类似的函数 `std::upper_bound(a,b,v)` 返回一个指向范围中第一个大于给定值 `v` 的元素的迭代器。如果 `v` 不在给定的范围内，那么 `std::lower_bound` 和 `std::upper_bound` 将返回相同的值。但如果 `v` 存在于范围内，那么 `lower_bound` 将返回一个指向范围中 `v` 的第一个实例的迭代器，而 `upper_bound` 将返回一个指向范围中 `v` 的最后一个实例之后一个位置的迭代器。换句话说，使用这两个函数一起将给出一个包含仅 `v` 值实例的半开范围 `[lower, upper)`：

```cpp
    std::vector<int> vec = {2, 3, 3, 3, 4};
    auto lower = std::lower_bound(vec.begin(), vec.end(), 3);

    // First approach:
    // upper_bound's interface is identical to lower_bound's.
    auto upper = std::upper_bound(vec.begin(), vec.end(), 3);

    // Second approach:
    // We don't need to binary-search the whole array the second time.
    auto upper2 = std::upper_bound(lower, vec.end(), 3);
    assert(upper2 == upper);

    // Third approach:
    // Linear scan from the lower bound might well be faster
    // than binary search if our total range is really big.
    auto upper3 = std::find_if(lower, vec.end(), [ {
      return v != 3;
    });
    assert(upper3 == upper);

    // No matter which approach we take, this is what we end up with.
    assert(*lower >= 3);
    assert(*upper > 3);
    assert(std::all_of(lower, upper, [](int v) { return v == 3; }));
```

这处理了在有序数组中搜索和插入值的问题。但删除怎么办？

# 使用 std::remove_if 从有序数组中删除

在我们到目前为止关于标准泛型算法的所有讨论中，我们还没有涵盖如何从范围中删除元素的问题。这是因为“范围”的概念本质上是只读的：我们可能改变给定范围中元素的 *值*，但我们永远不能使用标准算法来缩短或延长 *范围本身*。在 *使用 std::copy 推送数据* 这一部分中，当我们使用 `std::copy` 向名为 `dest` 的向量“插入”时，并不是 `std::copy` 算法在进行插入；而是 `std::back_insert_iterator` 对象本身持有对底层容器的引用，并且能够将元素插入到容器中。`std::copy` 并没有将 `dest.begin()` 和 `dest.end()` 作为参数；相反，它使用了特殊的对象 `std::back_inserter(dest)`。

那么，我们如何从范围中删除项目呢？嗯，我们不能。我们所能做的就是从 *容器* 中删除项目；而 STL 的算法并不处理容器。因此，我们应该寻找一种重新排列范围值的方法，使得“删除”的项目最终会出现在可预测的位置，这样我们就可以快速地从底层容器中删除它们（使用除 STL 算法之外的其他方法）。

我们已经看到了一种可能的方法：

```cpp
    std::vector<int> vec = {1, 3, 3, 4, 6, 8};

    // Partition our vector so that all the non-3s are at the front
    // and all the 3s are at the end.
    auto first_3 = std::stable_partition(
      vec.begin(), vec.end(), [](int v){ return v != 3; }
    );

    assert((vec == std::vector{1, 4, 6, 8, 3, 3}));

    // Now erase the "tail" of our vector.
    vec.erase(first_3, vec.end());

    assert((vec == std::vector{1, 4, 6, 8}));
```

但这比实际需要的要浪费得多（注意，`stable_partition` 是那些在堆上分配临时缓冲区的不多算法之一！）。我们想要的算法实际上要简单得多：

```cpp
    template<class FwdIt, class T>
    FwdIt remove(FwdIt first, FwdIt last, const T& value) 
    {
      auto out = std::find(first, last, value);
      if (out != last) {
        auto in = out;
        while (++in != last) {
          if (*in == value) {
             // don't bother with this item
          } else {
             *out++ = std::move(*in);
          }
        }
      }
      return out;
    }

    void test()
    {
      std::vector<int> vec = {1, 3, 3, 4, 6, 8};

      // Partition our vector so that all the non-3s are at the front.
      auto new_end = std::remove(
        vec.begin(), vec.end(), 3
      );

      // std::remove_if doesn't preserve the "removed" elements.
      assert((vec == std::vector{1, 4, 6, 8, 6, 8}));

      // Now erase the "tail" of our vector.
      vec.erase(new_end, vec.end());

      assert((vec == std::vector{1, 4, 6, 8}));

      // Or, do both steps together in a single line.
      // This is the "erase-remove idiom":
      vec.erase(
        std::remove(vec.begin(), vec.end(), 3),
        vec.end()
      );

      // But if the array is very long, and we know it's sorted,
      // then perhaps it would be better to binary-search for
      // the elements to erase.
      // Here the "shifting-down" is still happening, but it's
      // happening inside vector::erase instead of inside std::remove.
      auto first = std::lower_bound(vec.begin(), vec.end(), 3);
      auto last = std::upper_bound(first, vec.end(), 3);
      vec.erase(first, last);
    }
```

`std::remove(a,b,v)` 从范围 `[a,b)` 中删除所有等于 `v` 的值。请注意，范围不必是有序的--但 `remove` 将通过“向下移动”非删除元素来填补范围中的空隙，从而保留原有的顺序。如果 `remove` 从范围中删除了 *k* 个元素，那么当 `remove` 函数返回时，范围末尾将有 *k* 个元素的值处于已移动状态，`remove` 的返回值将是一个指向第一个这种已移动元素的迭代器。

`std::remove_if(a,b,p)` 会移除所有满足给定谓词的元素；也就是说，它会移除所有使得 `p(e)` 为真的元素 `e`。就像 `remove` 一样，`remove_if` 会将元素向下移动以填充范围，并返回一个指向第一个“已移动”元素的迭代器。

从序列容器中删除项的常用惯用方法是所谓的 *erase-remove 惯用方法*，因为它涉及到将返回值直接传递到容器自己的 `.erase()` 成员函数。

另一个与 erase-remove 惯用方法一起工作的标准库算法是 `std::unique(a,b)`，它接受一个范围，并对每一组连续的等效项，移除除了第一个之外的所有项。像 `std::remove` 一样，输入范围不需要排序；算法将保留最初存在的任何排序：

```cpp
    std::vector<int> vec = {1, 2, 2, 3, 3, 3, 1, 3, 3};

    vec.erase(
      std::unique(vec.begin(), vec.end()),
      vec.end()
    );

    assert((vec == std::vector{1, 2, 3, 1, 3}));
```

最后，请注意，我们通常可以比 `std::remove` 做得更好，要么通过使用我们底层容器的 `erase` 成员函数（例如，我们将在下一章中看到 `std::list::erase` 可以比在 `std::list` 上的 erase-remove 惯用方法快得多）--即使我们从不需要排序顺序的向量中删除，我们通常也会更倾向于以下这样的泛型算法 `unstable_remove`，该算法已被提议用于未来的标准化，但在撰写本文时尚未被纳入 STL：

```cpp
    namespace my {
      template<class BidirIt, class T>
      BidirIt unstable_remove(BidirIt first, BidirIt last, const T& value)
      {
        while (true) {
          // Find the first instance of "value"...
          first = std::find(first, last, value);
          // ...and the last instance of "not value"...
          do {
            if (first == last) {
              return last;
            }
            --last;
          } while (*last == value);
          // ...and move the latter over top of the former.
          *first = std::move(*last);
          // Rinse and repeat.
          ++first;
        }
      }
    } // namespace my

    void test()
    {
      std::vector<int> vec = {4, 1, 3, 6, 3, 8};

      vec.erase(
        my::unstable_remove(vec.begin(), vec.end(), 3),
        vec.end()
      );

      assert((vec == std::vector{4, 1, 8, 6}));
    }
```

在下一章中，我们将探讨 *容器*--STL 对“所有这些元素到底存储在哪里？”这一问题的回答。

# 摘要

标准模板库为几乎每个需求都提供了一个泛型算法。如果你在进行算法操作，首先检查 STL！

STL 算法处理由一对迭代器定义的半开区间。在处理任何一元半区间算法时都要小心。

处理比较和排序的 STL 算法默认使用 `operator<`，但你始终可以传递一个两个参数的“比较器”。如果你想在整个数据范围上执行非平凡操作，请记住 STL 可能直接支持它（`std::move`，`std::transform`）或通过特殊迭代器类型间接支持（`std::back_inserter`，`std::istream_iterator`）。

你应该知道“排列”是什么，以及标准排列算法（`swap`，`reverse`，`rotate`，`partition`，`sort`）是如何相互实现的。只有三个 STL 算法（`stable_sort`，`stable_partition`，`inplace_merge`）可能会默默地从堆中分配内存；如果你负担不起堆分配，请像躲避瘟疫一样避开这三个算法。

使用 erase-remove 惯用方法来维护序列容器的排序顺序，即使你在删除项时也是如此。如果你不关心排序顺序，可以使用类似 `my::unstable_remove` 的方法。对于支持 `.erase()` 的容器，请使用 `.erase()`。
