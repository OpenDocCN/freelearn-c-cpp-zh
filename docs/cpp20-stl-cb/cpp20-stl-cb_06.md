# 第六章：*第六章*：STL 算法

STL 的许多功能都体现在容器接口的标准化上。如果一个容器具有特定的功能，那么该功能的接口很可能在所有容器类型中都是标准化的。这种标准化使得一个库成为可能，该库中的 *算法* 可以无缝地在具有公共接口的容器和序列上运行。

例如，如果我们想计算 `vector` 中所有 `int` 元素的总和，我们可以使用循环：

```cpp
vector<int> x { 1, 2, 3, 4, 5 };
long sum{};
for( int i : x ) sum += i;                     // sum is 15
```

或者我们可以使用一个算法：

```cpp
vector<int> x { 1, 2, 3, 4, 5 };
auto sum = accumulate(x.begin(), x.end(), 0);  // sum is 15
```

此语法同样适用于其他容器：

```cpp
deque<int> x { 1, 2, 3, 4, 5 };
auto sum = accumulate(x.begin(), x.end(), 0);  // sum is 15
```

算法版本不一定更短，但它更容易阅读和维护。而且算法通常比等效循环更高效。

从 C++20 开始，`ranges` 库提供了一套操作于 *ranges* 和 *views* 的替代算法。本书将适当地演示这些替代方案。有关 ranges 和 views 的更多信息，请参阅本书第一章 *Chaper 1* 中的配方 *使用 ranges 创建容器视图*，*New C++20 Features*。

大多数算法都在 `algorithm` 头文件中。一些数值算法，特别是 `accumulate()`，在 `numeric` 头文件中，一些与内存相关的算法在 `memory` 头文件中。

我们将在以下配方中介绍 STL 算法：

+   从一个迭代器复制到另一个迭代器

+   将容器元素连接成一个字符串

+   使用 `std::sort` 对容器进行排序

+   使用 `std::transform` 修改容器

+   在容器中查找项目

+   使用 `std::clamp` 限制容器的值在一个范围内

+   使用 `std::sample` 的示例数据集

+   生成数据序列的排列

+   合并排序后的容器

# 技术要求

您可以在 GitHub 上找到本章的代码文件：[`github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap06`](https://github.com/PacktPublishing/CPP-20-STL-Cookbook/tree/main/chap06)。

# 从一个迭代器复制到另一个迭代器

*复制算法* 通常用于在容器之间复制数据，但实际上，它们与迭代器一起工作，这要灵活得多。

## 如何做到这一点...

在本配方中，我们将通过实验 `std::copy` 和 `std::copy_n` 来深入了解它们的工作原理：

+   让我们从打印容器的函数开始：

    ```cpp
    void printc(auto& c, string_view s = "") {
        if(s.size()) cout << format("{}: ", s);
        for(auto e : c) cout << format("[{}] ", e);
        cout << '\n';
    }
    ```

+   在 `main()` 中，我们定义一个 `vector` 并使用 `printc()` 打印它：

    ```cpp
    int main() {
        vector<string> v1
            { "alpha", "beta", "gamma", "delta", 
              "epsilon" };
        printc(v1);
    }
    ```

我们得到以下输出：

```cpp
v1: [alpha] [beta] [gamma] [delta] [epsilon]
```

+   现在，让我们创建第二个 `vector`，它有足够的空间来复制第一个 `vector`：

    ```cpp
    vector<string> v2(v1.size());
    ```

+   我们可以使用 `std::copy()` 算法将 `v1` 复制到 `v2`：

    ```cpp
    std::copy(v1.begin(), v1.end(), v2.begin());
    printc(v2);
    ```

`std::copy()` 算法接受两个迭代器作为复制源的范围，以及一个迭代器作为目标。在这种情况下，我们给它 `v1` 的 `begin()` 和 `end()` 迭代器来复制整个 `vector`。`v2` 的 `begin()` 迭代器作为复制的目标。

我们现在的输出是：

```cpp
v1: [alpha] [beta] [gamma] [delta] [epsilon]
v2: [alpha] [beta] [gamma] [delta] [epsilon]
```

+   `copy()` 算法不会为目的地分配空间。因此，`v2` 必须已经为复制分配了空间。或者，您可以使用 `back_inserter()` 迭代器适配器在 `vector` 的末尾插入元素：

    ```cpp
    vector<string> v2{};
    std::copy(v1.begin(), v1.end(), back_inserter(v2))
    ```

+   我们还可以使用 `ranges::copy()` 算法来复制整个**范围**。容器对象作为范围，因此我们可以使用 `v1` 作为源。我们仍然使用迭代器作为目标：

    ```cpp
    vector<string> v2(v1.size());
    ranges::copy(v1, v2.begin());
    ```

这也可以与 `back_inserter()` 一起使用：

```cpp
vector<string> v2{};
ranges::copy(v1, back_inserter(v2));
```

输出：

```cpp
v2: [alpha] [beta] [gamma] [delta] [epsilon]
```

+   您可以使用 `copy_n()` 复制一定数量的元素：

    ```cpp
    vector<string> v3{};
    std::copy_n(v1.begin(), 3, back_inserter(v3));
    printc(v3, "v3");
    ```

在第二个参数中，`copy_n()` 算法是复制元素数量的**计数**。输出如下：

```cpp
v3: [alpha] [beta] [gamma]
```

+   此外，还有一个使用布尔**谓词函数**来决定哪些元素需要复制的 `copy_if()` 算法：

    ```cpp
    vector<string> v4{};
    std::copy_if(v1.begin(), v1.end(), back_inserter(v4), 
        [](string& s){ return s.size() > 4; });
    printc(v4, "v4");
    ```

还有一个 `ranges` 版本的 `copy_if()`：

```cpp
vector<string> v4{};
ranges::copy_if(v1, back_inserter(v4), 
    [](string& s){ return s.size() > 4; });
printc(v4, "v4");
```

输出仅包括长度超过 `4` 个字符的字符串：

```cpp
v4: [alpha] [gamma] [delta] [epsilon]
```

注意，值 `beta` 被排除了。

+   您可以使用这些算法中的任何一个将数据复制到或从任何序列中，包括流迭代器：

    ```cpp
    ostream_iterator<string> out_it(cout, " ");
    ranges::copy(v1, out_it)
    cout << '\n';
    ```

输出：

```cpp
alpha beta gamma delta epsilon
```

## 它是如何工作的…

`std::copy()` 算法非常简单。等效函数可能看起来像这样：

```cpp
template<typename Input_it, typename Output_it>
Output_it bw_copy(Input_it begin_it, Input_it end_it, 
                  Output_it dest_it) {
    while (begin_it != end_it) {
        *dest_it++ = *begin_it++;
    }
    return dest_it;
}
```

`copy()` 函数使用目标迭代器的赋值运算符从输入迭代器复制到输出迭代器，直到达到输入范围的末尾。

此外，还有一个名为 `std::move()` 的算法版本，它移动元素而不是复制它们：

```cpp
std::move(v1.begin(), v1.end(), v2.begin());
printc(v1, "after move: v1");
printc(v2, "after move: v2");
```

这执行的是移动赋值而不是复制赋值。移动操作后，`v1` 中的元素将为空，而原本在 `v1` 中的元素现在在 `v2` 中。输出如下所示：

```cpp
after move1: v1: [] [] [] [] []
after move1: v2: [alpha] [beta] [gamma] [delta] [epsilon]
```

还有一个 `ranges` 版本的 `move()` 算法执行相同的操作：

```cpp
ranges::move(v1, v2.begin());
```

这些算法的强大之处在于它们的简单性。通过让迭代器管理数据，这些简单、优雅的函数允许您无缝地在支持所需迭代器的任何 STL 容器之间复制或移动。

# 将容器元素连接成一个字符串

有时，库中没有算法来完成手头的任务。我们可以使用迭代器，使用与 `algorithms` 库相同的技巧，轻松编写一个。

例如，我们经常需要将容器中的元素，带分隔符，连接成一个字符串。一个常见的解决方案是使用简单的 `for()` 循环：

```cpp
for(auto v : c) cout << v << ', ';
```

这个解决方案的缺点是它留下了一个尾随分隔符：

```cpp
vector<string> greek{ "alpha", "beta", "gamma", 
                      "delta", "epsilon" };
for(auto v : greek) cout << v << ", ";
cout << '\n';
```

输出：

```cpp
alpha, beta, gamma, delta, epsilon,
```

这在测试环境中可能没问题，但在任何生产系统中，尾随逗号都是不可接受的。

`ranges::views` 库有一个 `join()` 函数，但它不提供分隔符：

```cpp
auto greek_view = views::join(greek);
```

`views::join()` 函数返回一个 `ranges::view` 对象。这需要额外的步骤来显示或将其转换为字符串。我们可以使用 `for()` 循环遍历视图：

```cpp
for(const char c : greek_view) cout << c;
cout << '\n';
```

输出如下所示：

```cpp
alphabetagammadeltaepsilon
```

所有的内容都在那里，但我们需要一个合适的分隔符来使其对我们的目的有用。

由于 `algorithms` 库中没有适合我们需求的函数，我们将编写一个。

## 如何做到这一点…

对于这个菜谱，我们将取容器中的元素，并用分隔符将它们连接成一个字符串：

+   在我们的 `main()` 函数中，我们声明了一个字符串向量：

    ```cpp
    int main() {
        vector<string> greek{ "alpha", "beta", "gamma",
            "delta", "epsilon" };
        ...
    }
    ```

+   现在，让我们编写一个简单的 `join()` 函数，该函数使用 `ostream` 对象将元素与分隔符连接起来：

    ```cpp
    namespace bw {
        template<typename I>
        ostream& join(I it, I end_it, ostream& o, 
                      string_view sep = "") {
            if(it != end_it) o << *it++;
            while(it != end_it) o << sep << *it++;
            return o;
        }
    }
    ```

我已经将这个函数放在了自己的 `bw` 命名空间中，以避免名称冲突。

我们可以用 `cout` 来调用它：

```cpp
bw::join(greek.begin(), greek.end(), cout, ", ") << '\n';
```

因为它返回 `ostream` 对象，所以我们可以跟在它后面使用 `<<` 向流中添加一个 *换行符*。

输出：

```cpp
alpha, beta, gamma, delta, epsilon
```

+   我们经常想要一个 `string`，而不是直接写入 `cout`。我们可以为此函数重载一个返回 `string` 对象的版本：

    ```cpp
    template<typename I>
    string join(I it, I end_it, string_view sep = "") {
        ostringstream ostr;
        join(it, end_it, ostr, sep);
        return ostr.str();
    }
    ```

这也放在了 `bw` 命名空间中。这个函数创建了一个 `ostringstream` 对象，并将其传递给 `bw::join()` 的 `ostream` 版本。它从 `ostringstream` 对象的 `str()` 方法返回一个 `string` 对象。

我们可以像这样使用它：

```cpp
string s = bw::join(greek.begin(), greek.end(), ", ");
cout << s << '\n';
```

输出：

```cpp
alpha, beta, gamma, delta, epsilon
```

+   让我们添加一个最终的重载，使其更容易使用：

    ```cpp
    string join(const auto& c, string_view sep = "") {
        return join(begin(c), end(c), sep);
    }
    ```

这个版本只接受一个容器和一个分隔符，这应该可以很好地满足大多数用例：

```cpp
string s = bw::join(greek, ", ");
cout << s << '\n';
```

输出：

```cpp
alpha, beta, gamma, delta, epsilon
```

## 它是如何工作的…

这个菜谱中的大部分工作都是由迭代器和 `ostream` 对象完成的：

```cpp
namespace bw {
    template<typename I>
    ostream& join(I it, I end_it, ostream& o, 
                  string_view sep = "") {
        if(it != end_it) o << *it++;
        while(it != end_it) o << sep << *it++;
        return o;
    }
}
```

分隔符放在第一个元素之后，在连续元素之间，并在最终元素之前停止。这意味着我们可以在每个元素 *之前* 添加一个分隔符，跳过第一个，或者在每个元素 *之后* 添加，跳过最后一个。如果我们在 `while()` 循环之前测试并跳过第一个元素，逻辑会更简单。我们就在 `while()` 循环之前这样做：

```cpp
if(it != end_it) o << *it++;
```

一旦我们处理掉了第一个元素，我们就可以在剩余的每个元素前简单地添加一个分隔符：

```cpp
while(it != end_it) o << sep << *it++;
```

我们返回 `ostream` 对象作为便利。这使用户能够轻松地向流中添加换行符或其他对象：

```cpp
bw::join(greek.begin(), greek.end(), cout, ", ") << '\n';
```

输出：

```cpp
alpha, beta, gamma, delta, epsilon
```

## 还有更多…

与库中的任何算法一样，`join()` 函数可以与任何支持 *forward iterators* 的容器一起工作。例如，这里有一个来自 `numbers` 库的 `double` 常量 `list`：

```cpp
namespace num = std::numbers;
list<double> constants { num::pi, num::e, num::sqrt2 };
cout << bw::join(constants, ", ") << '\n';
```

输出：

```cpp
3.14159, 2.71828, 1.41421
```

它甚至可以与 `ranges::view` 对象一起工作，就像在这个菜谱中之前定义的 `greek_view`：

```cpp
cout << bw::join(greek_view, ":") << '\n';
```

输出：

```cpp
a:l:p:h:a:b:e:t:a:g:a:m:m:a:d:e:l:t:a:e:p:s:i:l:o:n
```

# 使用 std::sort 对容器进行排序

如何有效地对可比较元素进行排序的问题本质上已经解决。对于大多数应用来说，没有必要重新发明轮子。STL 通过 `std::sort()` 算法提供了一个出色的排序解决方案。虽然标准没有指定排序算法，但它确实指定了当应用于 *n* 个元素的范围内时，最坏情况下的复杂度为 *O*(*n* log *n*)。

仅在几十年前，*快速排序* 算法被认为是对大多数用途的良好折衷方案，并且通常比其他类似算法更快。今天，我们有 *混合* 算法，这些算法会根据情况选择不同的方法，通常在运行时切换算法。大多数当前的 C++ 库使用一种混合方法，结合了 *introsort* 和 *插入排序*。`std::sort()` 在大多数常见情况下提供了卓越的性能。

## 如何做到这一点...

在这个菜谱中，我们将检查 `std::sort()` 算法。`sort()` 算法与任何具有随机访问迭代器的容器一起工作。这里，我们将使用 `int` 的 `vector`：

+   我们将从测试容器是否排序的函数开始：

    ```cpp
    void check_sorted(auto &c) {
        if(!is_sorted(c.begin(), c.end())) cout << "un";
        cout << "sorted: ";
    }
    ```

这使用了 `std::is_sorted()` 算法，并根据结果打印 `"sorted:"` 或 `"unsorted:"`。

+   我们需要一个函数来打印我们的 `vector`：

    ```cpp
    void printc(const auto &c) {
        check_sorted(c);
        for(auto& e : c) cout << e << ' ';
        cout << '\n';
    }
    ```

这个函数调用 `check_sorted()` 来显示在值之前容器的状态。

+   现在我们可以定义并打印 `main()` 函数中的 `int` 的 `vector`：

    ```cpp
    int main() {
        vector<int> v{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        printc(v);
        …
    }
    ```

输出看起来像这样：

```cpp
sorted: 1 2 3 4 5 6 7 8 9 10
```

+   为了测试 `std::sort()` 算法，我们需要一个未排序的向量。这是一个简单的函数来随机化我们的容器：

    ```cpp
    void randomize(auto& c) {
        static std::random_device rd;
        static std::default_random_engine rng(rd());
        std::shuffle(c.begin(), c.end(), rng);
    }
    ```

`std::random_device` 类使用你的系统硬件 *熵* 源。大多数现代系统都有一个，否则库将模拟它。`std::default_random_engine()` 函数从熵源生成随机数。这被 `std::shuffle()` 用于随机化容器。

我们现在可以用我们的容器调用 `randomize()` 并打印结果：

```cpp
randomize(v);
printc(v);
```

输出：

```cpp
unsorted: 6 3 4 8 10 1 2 5 9 7
```

当然，你的输出会不同，因为它被随机化了。事实上，每次我运行它时，我都会得到不同的结果：

```cpp
for(int i{3}; i; --i) {
    randomize(v);
    printc(v);
}
```

输出：

```cpp
unsorted: 3 1 8 5 10 2 7 9 6 4
unsorted: 7 6 5 1 3 9 10 2 4 8
unsorted: 4 2 3 10 1 9 5 6 8 7
```

+   要对向量进行排序，我们只需调用 `std::sort()`：

    ```cpp
    std::sort(v.begin(), v.end());
    printc(v);
    ```

输出：

```cpp
sorted: 1 2 3 4 5 6 7 8 9 10
```

默认情况下，`sort()` 算法使用 `<` 操作符对指定迭代器范围的元素进行排序。

+   `partial_sort()` 算法将排序容器的一部分：

    ```cpp
    cout << "partial_sort:\n";
    randomize(v);
    auto middle{ v.begin() + (v.size() / 2) };
    std::partial_sort(v.begin(), middle, v.end());
    printc(v);
    ```

`partial_sort()` 接受三个迭代器：开始、中间和结束。它将容器排序，使得中间之前的元素是有序的。中间之后的元素不保证保持原始顺序。以下是输出：

```cpp
unsorted: 1 2 3 4 5 10 7 6 8 9
```

注意到前五个元素是有序的，其余的不是。

+   `partition()` 算法 *不会* 对任何东西进行排序。它重新排列容器，使某些元素出现在容器的前面：

    ```cpp
    coutrandomize(v);
    printc(v);
    partition(v.begin(), v.end(), [](int i)
        { return i > 5; });
    printc(v);
    ```

第三个参数是一个 *谓词* lambda，它确定哪些元素将被移动到前面。

输出：

```cpp
unsorted: 4 6 8 1 9 5 2 7 3 10
unsorted: 10 6 8 7 9 5 2 1 3 4
```

注意值 `>5` 被移动到容器的前面。

+   `sort()` 算法支持一个可选的比较函数，可用于非标准比较。例如，给定一个名为 `things` 的类：

    ```cpp
    struct things {
        string s_;
        int i_;
        string str() const {
            return format("({}, {})", s_, i_);
        }
    };
    ```

我们可以创建一个 `vector` 的 `things`：

```cpp
vector<things> vthings{ {"button", 40},
    {"hamburger", 20}, {"blog", 1000},
    {"page", 100}, {"science", 60} };
```

我们需要一个函数来打印它们：

```cpp
void print_things(const auto& c) {
    for (auto& v : c) cout << v.str() << ' ';
    cout << '\n';
}
```

+   现在我们可以排序并打印 `things` 的 `vector`：

    ```cpp
    std::sort(vthings.begin(), vthings.end(), 
            [](const things &lhs, const things &rhs) {
        return lhs.i_ < rhs.i_;
    });
    print_things(vthings);
    ```

输出：

```cpp
(hamburger, 20) (button, 40) (science, 60) (page, 100) (blog, 1000)
```

注意比较函数按 `i_` 成员排序，所以结果是按 `i_` 排序的。我们也可以按 `s_` 成员排序：

```cpp
std::sort(vthings.begin(), vthings.end(), 
        [](const things &lhs, const things &rhs) {
    return lhs.s_ < rhs.s_;
});
print_things(vthings);
```

现在我们得到这个输出：

```cpp
(blog, 1000) (button, 40) (hamburger, 20) (page, 100) (science, 60)
```

## 它是如何工作的...

`sort()` 函数通过将排序算法应用于由两个迭代器指示的元素范围（范围的开始和结束）来工作。

默认情况下，这些算法使用 `<` 操作符来比较元素。可选地，它们可能使用 *比较函数*，通常作为 lambda 提供的：

```cpp
std::sort(vthings.begin(), vthings.end(), 
        [](const things& lhs, const things& rhs) {
    return lhs.i_ < rhs.i_;
});
```

比较函数接受两个参数并返回一个 `bool`。它的签名等同于以下：

```cpp
bool cmp(const Type1& a, const Type2& b);
```

`sort()` 函数使用 `std::swap()` 来移动元素。这在计算周期和内存使用上都很高效，因为它避免了为读取和写入正在排序的对象分配空间的需求。这也是为什么 `partial_sort()` 和 `partition()` 函数不能保证未排序元素的顺序。

# 使用 std::transform 修改容器

`std::transform()` 函数非常强大且灵活。它是库中更常用的一些算法之一，它将一个 *函数* 或 *lambda* 应用到容器中的每个元素上，同时将结果存储在另一个容器中，而原始容器保持不变。

由于其强大的功能，使用起来出奇地简单。

## 如何实现它...

在这个菜谱中，我们将探讨 `std::transform()` 函数的一些应用：

+   我们从一个简单的打印容器内容的函数开始：

    ```cpp
    void printc(auto& c, string_view s = "") {
        if(s.size()) cout << format("{}: ", s);
        for(auto e : c) cout << format("{} ", e);
        cout << '\n';
    }
    ```

我们将使用它来查看转换的结果。

+   在 `main()` 函数中，让我们声明几个向量：

    ```cpp
    int main() {
        vector<int> v1{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        vector<int> v2;
        printc(v1, "v1");
        ...
    }
    ```

这会打印出 `v1` 的内容：

```cpp
v1: 1 2 3 4 5 6 7 8 9 10
```

+   现在，我们可以使用 `transform()` 函数将每个值的平方插入到 `v2` 中：

    ```cpp
    cout << "squares:\n";
    transform(v1.begin(), v1.end(), back_inserter(v2),
        [](int x){ return x * x; });
    printc(v2, "v2");
    ```

`transform()` 函数接受四个参数。前两个参数是源范围的 `begin()` 和 `end()` 迭代器。第三个参数是目标范围的 `begin()` 迭代器。在这种情况下，我们使用 `back_inserter()` 算法将结果插入到 `v2` 中。第四个参数是转换函数。在这种情况下，我们使用一个简单的 lambda 来平方值。

输出：

```cpp
squares:
v2: 1 4 9 16 25 36 49 64 81 100
```

+   当然，我们可以用 `transform()` 函数处理任何类型。以下是一个将 `string` 对象的 `vector` 转换为小写的示例。首先，我们需要一个函数来返回字符串的小写值：

    ```cpp
    string str_lower(const string& s) {
        string outstr{};
        for(const char& c : s) {
            outstr += tolower(c);
        }
        return outstr;
    }
    ```

现在，我们可以使用 `str_lower()` 函数与 `transform` 一起使用：

```cpp
vector<string> vstr1{ "Mercury", "Venus", "Earth",
    "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Pluto" };
vector<string> vstr2;
printc(vstr1, "vstr1");
cout << "str_lower:\n";
transform(vstr1.begin(), vstr1.end(),
    back_inserter(vstr2), 
    [](string& x){ return str_lower(x); });
printc(vstr2, "vstr2");
```

这会对 `vstr1` 中的每个元素调用 `str_lower()` 并将结果插入到 `vstr2` 中。结果是：

```cpp
vstr: Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto
str_lower:
vstr: mercury venus earth mars jupiter saturn uranus neptune pluto
```

（是的，对我来说，冥王星始终是行星。）

+   还有一个 `ranges` 版本的 `transform`：

    ```cpp
    cout << "ranges squares:\n";
    auto view1 = views::transform(v1, [](int x){ 
        return x * x; });
    printc(view1, "view1");
    ```

`ranges` 版本具有更简洁的语法，并返回一个 `view` 对象，而不是填充另一个容器。

## 它是如何工作的...

`std::transform()` 函数的工作方式非常类似于 `std::copy()`，它增加了用户提供的函数。输入范围内的每个元素都会传递给该函数，函数的返回值会被复制分配给目标迭代器。这使得 `transform()` 成为一个独特且强大的算法。

值得注意的是，`transform()` 函数并不能保证元素会按顺序处理。如果你需要确保转换的顺序，你应该使用 `for` 循环代替：

```cpp
v2.clear();    // reset vector v2 to empty state
for(auto e : v1) v2.push_back(e * e);
printc(v2, "v2");
```

输出：

```cpp
v2: 1 4 9 16 25 36 49 64 81 100
```

# 在容器中查找项目

`algorithm` 库包含了一组用于在容器中查找元素的函数。`std::find()` 函数及其衍生函数会顺序遍历容器，并返回一个指向第一个匹配元素的迭代器，如果没有匹配则返回 `end()` 元素。

## 如何实现...

`find()` 算法与满足 *Forward* 或 *Input* 迭代器资格的任何容器一起工作。对于这个配方，我们将使用 `vector` 容器。`find()` 算法在容器中顺序搜索第一个匹配元素。在这个配方中，我们将通过几个示例来讲解：

+   我们首先在 `main()` 函数中声明一个 `int` 类型的 `vector`：

    ```cpp
    int main() {
        const vector<int> v{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        ...
    }
    ```

+   现在，让我们搜索值为 `7` 的元素：

    ```cpp
    auto it1 = find(v.begin(), v.end(), 7);
    if(it1 != v.end()) cout << format("found: {}\n", *it1);
    else cout << "not found\n";
    ```

`find()` 算法接受三个参数：`begin()` 和 `end()` 迭代器，以及要搜索的值。它返回指向它找到的第一个元素的迭代器，或者在搜索未成功找到匹配项时返回 `end()` 迭代器。

输出：

```cpp
found: 7
```

+   我们也可以搜索比标量更复杂的东西。该对象需要支持相等比较运算符 `==`。这里有一个简单的结构体，它重载了 `operator==()`：

    ```cpp
    struct City {
        string name{};
        unsigned pop{};
        bool operator==(const City& o) const {
            return name == o.name;
        }
        string str() const {
            return format("[{}, {}]", name, pop);
        }
    };
    ```

注意，`operator=()` 重载只比较 `name` 成员。

我还包含了一个 `str()` 函数，它返回 `City` 元素的字符串表示形式。

+   现在，我们可以声明一个 `City` 元素的 `vector`：

    ```cpp
    const vector<City> c{
        { "London", 9425622 },
        { "Berlin", 3566791 },
        { "Tokyo",  37435191 },
        { "Cairo",  20485965 }
    };
    ```

+   我们可以像搜索 `int` 类型的 `vector` 一样搜索 `City` 类型的 `vector`：

    ```cpp
    auto it2 = find(c.begin(), c.end(), City{"Berlin"});
    if(it2 != c.end()) cout << format("found: {}\n", 
        it2->str());
    else cout << "not found\n";
    ```

输出：

```cpp
found: [Berlin, 3566791]
```

+   如果我们想搜索 `pop` 成员而不是 `name`，我们可以使用带有谓词的 `find_if()` 函数：

    ```cpp
    auto it3 = find_if(begin(c), end(c),
        [](const City& item)
            { return item.pop > 20000000; });
    if(it3 != c.end()) cout << format("found: {}\n",
        it3->str());
    else cout << "not found\n";
    ```

谓词测试 `pop` 成员，所以我们得到这个输出：

```cpp
found: [Tokyo, 37435191]
```

+   注意，`find_if()` 的结果只返回满足谓词的第一个元素，即使 `vector` 中有两个元素的 `pop` 值大于 20,000,000。

`find()` 和 `find_if()` 函数只返回一个迭代器。`ranges` 库提供了 `ranges::views::filter()`，这是一个 *视图适配器*，它将给我们所有匹配的迭代器，而不会干扰我们的 `vector`：

```cpp
auto vw1 = ranges::views::filter(c, 
    [](const City& c){ return c.pop > 20000000; });
for(const City& e : vw1) cout << format("{}\n", e.str());
```

这给我们带来了输出中的所有匹配元素：

```cpp
[Tokyo, 37435191]
[Cairo, 20485965]
```

## 它是如何工作的……

`find()` 和 `find_if()` 函数顺序遍历容器，检查每个元素，直到找到匹配项。如果找到匹配项，它返回指向该匹配项的迭代器。如果在没有找到匹配项的情况下达到 `end()` 迭代器，它返回 `end()` 迭代器以指示没有找到匹配项。

`find()` 函数接受三个参数，即 `begin()` 和 `end()` 迭代器，以及搜索值。其签名如下：

```cpp
template<class InputIt, class T>
constexpr InputIt find(InputIt, InputIt, const T&)
```

`find_if()` 函数使用谓词而不是值：

```cpp
template<class InputIt, class UnaryPredicate>
constexpr InputIt find_if(InputIt, InputIt, UnaryPredicate)
```

## 还有更多……

两个 `find()` 函数都是顺序搜索，并在找到第一个匹配项时返回。如果你想找到更多匹配的元素，你可以使用 `ranges` 库中的 `filter()` 函数：

```cpp
template<ranges::viewable_range R, class Pred>
constexpr ranges::view auto ranges::views::filter(R&&, Pred&&);
```

`filter()` 函数返回一个 *视图*，这是一个非破坏性的容器窗口，只包含过滤后的元素。然后我们可以像使用任何其他容器一样使用这个视图：

```cpp
auto vw1 = std::ranges::views::filter(c,
    [](const City& c){ return c.pop > 20000000; });
for(const City& e : vw1) cout << format("{}\n", e.str());
```

输出：

```cpp
[Tokyo, 37435191]
[Cairo, 20485965]
```

# 使用 std::clamp 限制容器中的值范围

随着 C++17 的引入，`std::clamp()` 函数可以用来限制数值标量的范围，使其在最小值和最大值之间。该函数尽可能优化使用 *移动语义*，以实现最大速度和效率。

## 如何做到这一点...

我们可以通过在循环中使用 `clamp()` 或使用 `transform()` 算法来通过它约束容器中的值。让我们看看一些例子。

+   我们将从一个简单的函数开始，用于打印容器中的值：

    ```cpp
    void printc(auto& c, string_view s = "") {
        if(s.size()) cout << format("{}: ", s);
        for(auto e : c) cout << format("{:>5} ", e);
        cout << '\n';
    }
    ```

注意到 *格式字符串* `"{:>5} "`。这会将每个值右对齐到 `5` 个空格，以实现表格视图。

+   在 `main()` 函数中，我们将定义一个 *初始化列表* 以用于我们的容器。这允许我们多次使用相同的值：

    ```cpp
    int main() {
        auto il = { 0, -12, 2001, 4, 5, -14, 100, 200, 
          30000 };
        ...
    }
    ```

这是一个很好的值范围，可以与 `clamp()` 一起使用。

+   让我们也定义一些常数作为我们的限制：

    ```cpp
    constexpr int ilow{0};
    constexpr int ihigh{500};
    ```

我们将在对 `clamp()` 的调用中使用这些值。

+   现在，我们可以在 `main()` 函数中定义一个容器。我们将使用 `int` 的 `vector`：

    ```cpp
    vector<int> voi{ il };
    cout << "vector voi before:\n";
    printc(voi);
    ```

使用我们的初始化列表中的值，输出如下：

```cpp
vector voi before:
    0   -12  2001     4     5   -14   100   200 30000
```

+   现在，我们可以使用带有 `clamp()` 的 `for` 循环来限制值在 0 和 500 之间：

    ```cpp
    cout << "vector voi after:\n";
    for(auto& e : voi) e = clamp(e, ilow, ihigh);
    printc(voi);
    ```

这个函数将 `clamp()` 应用到容器中的每个值，分别使用 0 和 500 作为低和高限制。现在，输出如下：

```cpp
vector voi before:
    0   -12  2001     4     5   -14   100   200 30000
vector voi after:
    0     0   500     4     5     0   100   200   500
```

在 `clamp()` 操作之后，负值变为 0，大于 500 的值变为 500。

+   我们可以使用 `transform()` 算法，在 lambda 中使用 `clamp()` 来做同样的事情。这次我们将使用一个 `list` 容器：

    ```cpp
    cout << "list loi before:\n";
    list<int> loi{ il };
    printc(loi);
    transform(loi.begin(), loi.end(), loi.begin(), 
        ={ return clamp(e, ilow, ihigh); });
    cout << "list loi after:\n";
    printc(loi);
    ```

输出与带有 `for` 循环的版本相同：

```cpp
list loi before:
    0   -12  2001     4     5   -14   100   200 30000
list loi after:
    0     0   500     4     5     0   100   200   500
```

## 它是如何工作的...

`clamp()` 算法是一个简单的函数，看起来像这样：

```cpp
template<class T>
constexpr const T& clamp( const T& v, const T& lo,
        const T& hi ) {
    return less(v, lo) ? lo : less(hi, v) ? hi : v;
}
```

如果 `v` 的值小于 `lo`，则返回 `lo`。如果 `hi` 小于 `v`，则返回 `hi`。该函数快速且高效。

在我们的例子中，我们使用 `for` 循环将 `clamp()` 应用到容器中：

```cpp
for(auto& v : voi) v = clamp(v, ilow, ihigh);
```

我们还使用 lambda 在 `transform()` 算法中与 `clamp()` 一起使用：

```cpp
transform(loi.begin(), loi.end(), loi.begin(),
    ={ return clamp(v, ilow, ihigh); });
```

在我的实验中，两个版本都给出了相同的结果，并且都生成了类似 GCC 编译器的代码。编译大小（带有 `for` 循环的版本更小，正如预期的那样）和性能差异可以忽略不计。

通常，我更喜欢 `for` 循环，但 `transform()` 版本在其他应用中可能更灵活。

# 使用 std::sample 的样本数据集

`std::sample()` 算法从一系列值中随机抽取 *样本*，并将样本填充到目标容器中。这对于分析较大的数据集很有用，其中随机样本被认为是整个集合的代表。

样本集允许我们近似大量数据的特征，而无需分析整个集合。这以准确性为代价提供了效率，在很多情况下是一个公平的交易。

## 如何做到这一点...

在这个菜谱中，我们将使用一个包含 200,000 个随机整数的数组，具有 *标准正态分布*。我们将采样几百个值来创建每个值的频率直方图。

+   我们将从返回一个从`double`舍入的`int`的简单函数开始。标准库缺少这样的函数，我们稍后需要它：

    ```cpp
    int iround(const double& d) {
        return static_cast<int>(std::round(d));
    }
    ```

标准库提供了几个版本的`std::round()`，包括一个返回`long int`的版本。但我们需要一个`int`，这是一个简单的解决方案，它避免了编译器关于缩窄转换的警告，同时隐藏了难看的`static_cast`。

+   在`main()`函数中，我们将开始一些有用的常量：

    ```cpp
    int main() {
        constexpr size_t n_data{ 200000 };
        constexpr size_t n_samples{ 500 };
        constexpr int mean{ 0 };
        constexpr size_t dev{ 3 };
        ...
    }
    ```

我们有`n_data`和`n_samples`的值，分别用于数据容器和样本容器的尺寸。我们还有`mean`和`dev`的值，这是随机值正态分布的*均值*和*标准差*参数。

+   我们现在设置我们的*随机数生成器*和*分布*对象。这些用于初始化源数据集：

    ```cpp
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<> dist{ mean, dev };
    ```

`random_device`对象提供了对硬件随机数生成器的访问。`mt19937`类是*Mersenne Twister*随机数算法的一个实现，这是一个在大多数系统上表现良好的高质量算法，适用于我们使用的数据集大小。`normal_distribution`类提供了一个围绕*均值*的随机数分布，并提供了*标准差*。

+   现在我们用一个`n_data`数量的随机`int`值填充一个数组：

    ```cpp
    array<int, n_data> v{};
    for(auto& e : v) e = iround(dist(rng));
    ```

`array`容器的大小是固定的，所以模板参数包括一个`size_t`值，表示要分配的元素数量。我们使用`for()`循环来填充数组。

`rng`对象是硬件随机数生成器。这个对象被传递给我们的`normal_distribution`对象`dist()`，然后传递给我们的整数舍入函数`iround()`。

+   到目前为止，我们有一个包含 200,000 个数据点的数组。这有很多要分析，所以我们将使用`sample()`算法来抽取 500 个值的样本：

    ```cpp
    array<int, n_samples> samples{};
    sample(data.begin(), data.end(), samples.begin(), 
        n_samples, rng);
    ```

我们定义另一个`array`对象来存储样本。这个数组的大小是`n_samples`。然后我们使用`sample()`算法用`n_samples`个随机数据点填充数组。

+   我们创建一个直方图来分析样本。`map`结构非常适合这个用途，因为我们可以轻松地将每个值的频率映射出来：

    ```cpp
    std::map<int, size_t> hist{};
    for (const int i : samples) ++hist[i];
    ```

`for()`循环从`samples`容器中取每个值，并将其用作`map`中的键。增量表达式`++hist[i]`计算样本集中每个值的出现次数。

+   我们使用 C++20 的`format()`函数打印出直方图：

    ```cpp
    constexpr size_t scale{ 3 };
    cout << format("{:>3} {:>5} {:<}/{}\n", 
        "n", "count", "graph", scale);
    for (const auto& [value, count] : hist) {
        cout << format("{:>3} ({:>3}) {}\n", 
            value, count, string(count / scale, '*'));
    }
    ```

类似于`{:>3}`的`format()`指定符为一定数量的字符留出空间。尖括号指定了对齐方式，是右对齐还是左对齐。

`string(count, char)`构造函数创建一个`string`，其中包含重复指定次数的字符，在这种情况下，*n*个星号字符`*`，其中*n*是`count/scale`，即直方图中一个值的频率除以`scale`常量。

输出看起来像这样：

```cpp
$ ./sample
  n count graph/3
 -9 (  2)
 -7 (  5) *
 -6 (  9) ***
 -5 ( 22) *******
 -4 ( 24) ********
 -3 ( 46) ***************
 -2 ( 54) ******************
 -1 ( 59) *******************
  0 ( 73) ************************
  1 ( 66) **********************
  2 ( 44) **************
  3 ( 34) ***********
  4 ( 26) ********
  5 ( 18) ******
  6 (  9) ***
  7 (  5) *
  8 (  3) *
  9 (  1)
```

这是一个很好的直方图图形表示。第一个数字是值，第二个数字是该值的频率，星号是频率的视觉表示，其中每个星号代表样本集中`scale`（3）次出现。

每次运行代码时，你的输出都会不同。

## 它是如何工作的…

`std::sample()`函数从源容器中的随机位置选择特定数量的元素，并将它们复制到目标容器中。

`sample()`的签名如下所示：

```cpp
OutIter sample(SourceIter, SourceIter, OutIter, 
    SampleSize, RandNumGen&&);
```

前两个参数是容器中完整数据集的`begin()`和`end()`迭代器。第三个参数是样本目的地的迭代器。第四个参数是样本大小，最后一个参数是随机数生成函数。

`sample()`算法使用*均匀分布*，因此每个数据点被抽样的机会相同。

# 生成数据序列的排列组合

排列组合有许多用途，包括测试、统计学、研究等。`next_permutation()`算法通过重新排列容器以生成下一个*字典序*排列组合。

## 如何做到这一点…

对于这个食谱，我们将打印出一组三个字符串的排列组合：

+   我们首先创建一个用于打印容器内容的简短函数：

    ```cpp
    void printc(const auto& c, string_view s = "") {
        if(s.size()) cout << format("{}: ", s);
        for(auto e : c) cout << format("{} ", e);
        cout << '\n';
    }
    ```

我们将使用这个简单的函数来打印我们的数据集和排列组合。

+   在`main()`函数中，我们声明一个`string`对象的`vector`并使用`sort()`算法对其进行排序。

    ```cpp
    int main() {
        vector<string> vs{ "dog", "cat", "velociraptor" };
        sort(vs.begin(), vs.end());
        ...
    }
    ```

`next_permutation()`函数需要一个有序容器。

+   现在，我们可以使用`next_permutation()`在`do`循环中列出排列组合：

    ```cpp
    do {
        printc(vs);
    } while (next_permutation(vs.begin(), vs.end()));
    ```

`next_permutation()`函数会修改容器，如果还有另一个排列组合则返回`true`，如果没有则返回`false`。

输出列出了我们三只宠物的六种排列组合：

```cpp
cat dog velociraptor
cat velociraptor dog
dog cat velociraptor
dog velociraptor cat
velociraptor cat dog
velociraptor dog cat
```

## 它是如何工作的…

`std::next_permutation()`算法生成一组值的*字典序*排列组合，即基于字典顺序的排列组合。输入必须是有序的，因为算法按字典序遍历排列组合。所以，如果你从一个像 3, 2, 1 这样的集合开始，它将立即终止，因为这是这三个元素的最后一种字典序排列。

例如：

```cpp
vector<string> vs{ "velociraptor", "dog", "cat" };
do {
    printc(vs);
} while (next_permutation(vs.begin(), vs.end()));
```

这给出了以下输出：

```cpp
velociraptor dog cat
```

虽然术语*字典序*暗示了字母顺序，但实现使用标准比较运算符，因此它适用于任何可排序的值。

同样，如果集合中的值重复，它们只按*字典序*计数。这里有一个包含两个重复序列的五个值的`vector`：

```cpp
vector<int> vi{ 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
sort(vi.begin(), vi.end());
printc(vi, "vi sorted");
long count{};
do {
    ++count;
} while (next_permutation(vi.begin(), vi.end()));
cout << format("number of permutations: {}\n", count);
```

输出：

```cpp
Vi sorted: 1 1 2 2 3 3 4 4 5 5
number of permutations: 113400
```

这些值的排列组合共有 113,400 种。请注意，这不是*10!*（3,628,800），因为有些值是重复的。由于*3,3*和*3,3*排序相同，它们不是不同的*字典序*排列组合。

换句话说，如果我列出这个短集合的排列组合：

```cpp
vector<int> vi2{ 1, 3, 1 };
sort(vi2.begin(), vi2.end());
do {
    printc(vi2);
} while (next_permutation(vi2.begin(), vi2.end()));
```

由于有重复的值，我们只得到三个排列，而不是 *3!*（9）：

```cpp
1 1 3
1 3 1
3 1 1
```

# 合并排序容器

`std::merge()` 算法接受两个排序序列，并创建一个第三合并和排序的序列。这种技术通常作为 *归并排序* 的一部分，允许将大量数据分解成块，分别排序，然后合并到一个排序的目标中。

## 如何做到这一点...

对于这个菜谱，我们将取两个排序的 `vector` 容器，并使用 `std::merge()` 将它们合并到第三个 `vector` 中。

+   我们将从打印容器内容的一个简单函数开始：

    ```cpp
    void printc(const auto& c, string_view s = "") {
        if(s.size()) cout << format("{}: ", s);
        for(auto e : c) cout << format("{} ", e);
        cout << '\n';
    }
    ```

我们将使用这个结果来打印源和目标序列。

+   在 `main()` 函数中，我们将声明我们的源向量，以及目标向量，并将它们打印出来：

    ```cpp
    int main() {
        vector<string> vs1{ "dog", "cat", 
          "velociraptor" };
        vector<string> vs2{ "kirk", "sulu", "spock" };
        vector<string> dest{};
        printc(vs1, "vs1");
        printc(vs2, "vs2");
        ...
    }
    ```

输出结果如下：

```cpp
vs1: dog cat velociraptor
vs2: kirk sulu spock
```

+   现在我们可以对向量进行排序并再次打印它们：

    ```cpp
    sort(vs1.begin(), vs1.end());
    sort(vs2.begin(), vs2.end());
    printc(vs1, "vs1 sorted");
    printc(vs2, "vs2 sorted");
    ```

输出结果：

```cpp
vs1 sorted: cat dog velociraptor
vs2 sorted: kirk spock sulu
```

+   现在我们已经对源容器进行了排序，我们可以将它们合并以得到最终的合并结果：

    ```cpp
    merge(vs1.begin(), vs1.end(), vs2.begin(), vs2.end(), 
        back_inserter(dest));
    printc(dest, "dest");
    ```

输出结果：

```cpp
dest: cat dog kirk spock sulu velociraptor
```

这个输出表示将两个源合并到一个排序向量中。

## 它是如何工作的...

`merge()` 算法从两个源中获取 `begin()` 和 `end()` 迭代器，并为目标提供一个输出迭代器：

```cpp
OutputIt merge(InputIt1, InputIt1, InputIt2, InputIt2, OutputIt)
```

它接受两个输入范围，执行其合并/排序操作，并将结果序列发送到输出迭代器。
