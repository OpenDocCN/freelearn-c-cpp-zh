# 第九章. 容器

在本章中，我们将介绍：

+   以超快的方式比较字符串

+   使用无序集合和映射

+   创建一个映射，其中值也是键

+   使用多索引容器

+   获得单链表和内存池的好处

+   使用扁平关联容器

# 简介

本章专门介绍 Boost 容器及其直接相关的内容。本章提供了有关可以在日常编程中使用、可以使代码运行得更快、使新应用程序的开发更简单的 Boost 类的信息。

容器不仅通过功能不同，而且通过其成员的一些效率（复杂度）也不同。了解复杂度对于编写快速应用程序至关重要。本章不仅向您介绍了一些新的容器，还提供了何时以及何时不要使用特定类型的容器或其方法的建议。

那么，让我们开始吧！

# 以超快的方式比较字符串

操作字符串是一个常见任务。在这里，我们将看到如何使用一些简单的技巧快速执行字符串比较操作。这个配方是下一个配方的跳板，其中这里描述的技术将被用来实现常数时间复杂度的搜索。

所以，我们需要创建一个能够快速比较字符串相等的类。我们将创建一个模板函数来测量比较的速度：

```cpp
#include <string>

template <class T>
std::size_t test_default() {
    // Constants
    const std::size_t ii_max = 20000000;
    const std::string s(
        "Long long long string that "
        "will be used in tests to compare "
        "speed of equality comparisons."
    );

    // Making some data, that will be 
    // used in comparisons
    const T data[] = {
        T(s),
        T(s + s),
        T(s + ". Whooohooo"),
        T(std::string(""))
    };

    const std::size_t data_dimensions = sizeof(data) / sizeof(data[0]);
    std::size_t matches = 0u;
    for (std::size_t ii = 0; ii < ii_max; ++ii) {
        for (std::size_t i = 0; i < data_dimensions; ++i) {
            for (std::size_t j = 0; j < data_dimensions; ++j) {
                if (data[i] == data[j]) {
                    ++ matches;
                }
            }
        }
    }

    return matches;
}
```

## 准备就绪

这个配方只需要基本的 STL 和 C++知识。

## 如何做到这一点...

我们将使`std::string`成为我们自己的类的公共字段，并将所有比较代码添加到我们的类中，而不需要编写与存储的`std::string`一起工作的辅助方法，如下面的步骤所示：

1.  为了做到这一点，我们需要以下头文件：

    ```cpp
    #include <boost/functional/hash.hpp>
    ```

1.  现在，我们可以创建我们的快速比较类：

    ```cpp
    struct string_hash_fast {
        typedef std::size_t comp_type;

        const comp_type     comparison_;
        const std::string   str_;

        explicit string_hash_fast(const std::string& s)
            : comparison_(
                boost::hash<std::string>()(s)
            )
            , str_(s)
        {}
    };
    ```

1.  不要忘记定义相等比较运算符：

    ```cpp
    inline bool operator == (const string_hash_fast& s1, 
        const string_hash_fast& s2) 
    {
        return s1.comparison_ == s2.comparison_ 
            && s1.str_ == s2.str_;
    }

    inline bool operator != (const string_hash_fast& s1, 
        const string_hash_fast& s2) 
    {
        return !(s1 == s2);
    }
    ```

1.  然后，这就完成了！现在我们可以运行我们的测试，并使用以下代码查看结果：

    ```cpp
    #include <iostream>
    int main(int argc, char* argv[]) {
        if (argc < 2) {
            assert(
                test_default<string_hash_fast>()
                ==
                test_default<std::string>()
            );
            return 0;
        }

        switch (argv[1][0]) {
        case 'h':
            std::cout << "HASH matched: "
                      << test_default<string_hash_fast>();
            break;

        case 's':
            std::cout << "STD matched: "
                      << test_default<std::string>();
            break;

        default:
            assert(false);
            return -2;
        }
    }
    ```

## 它是如何工作的...

字符串比较之所以慢，是因为如果我们要求比较字符串的所有字符，如果字符串长度相等，我们就必须逐个比较这些字符。而不是这样做，我们用整数比较来替换字符串比较。这是通过哈希函数完成的——该函数使字符串具有某种短固定长度的表示。让我们谈谈苹果上的哈希值。想象一下，你有两个带有标签的苹果，如图所示，你希望检查这些苹果是否属于同一品种。比较这些苹果的最简单方法是通过标签来比较它们。否则，你将花费大量时间根据颜色、大小、形状和其他参数来比较苹果。哈希就像一个标签，反映了对象的价值。

![如何工作...](img/4880OS_09_01.jpg)

那么，让我们一步一步来。

在步骤 1 中，我们包含了包含哈希函数定义的头文件。在步骤 2 中，我们声明了我们的新字符串类，它包含 `str_`，这是字符串的原始值，以及 `comparison_`，这是计算出的哈希值。注意构造：

```cpp
boost::hash<std::string>()(s)
```

在这里，`boost::hash<std::string>` 是一个结构，一个功能对象，就像 `std::negate<>`。这就是为什么我们需要第一个括号——我们构建这个功能对象。第二个括号内包含 `s` 的括号是对 `std::size_t operator()(const std::string& s)` 的调用，它将计算哈希值。

现在看看步骤 3，我们定义了 `operator==`。看看以下代码：

```cpp
return s1.comparison_ == s2.comparison_ && s1.str_ == s2.str_;
```

此外，还要注意表达式的第二部分。哈希操作会丢失信息，这意味着可能存在多个字符串产生完全相同的哈希值。这意味着如果哈希值不匹配，则可以保证字符串不会匹配，否则我们要求使用传统方法比较字符串。

好吧，现在是时候比较数字了。如果我们使用默认的比较方法来测量执行时间，它将给出 819 毫秒；然而，我们的哈希比较几乎快两倍，只需 475 毫秒就能完成。

## 还有更多...

C++11 提供了哈希功能对象，你可以在 `std::` 命名空间中的 `<functional>` 头文件中找到它。你会知道默认的 Boost 哈希实现不会分配额外的内存，也没有虚拟函数。Boost 和 STL 中的哈希既快又可靠。

你还可以为你的自定义类型特化哈希。在 Boost 中，这是通过在自定义类型的命名空间中特化 `hash_value` 函数来完成的：

```cpp
// Must be in namespace of string_hash_fast class
inline std::size_t hash_value(const string_hash_fast& v) {
    return v.comparison_;
}
```

这与 STL 的 `std::hash` 特化不同，在 `std::` 命名空间中，你需要对 `hash<>` 结构进行模板特化。

在 Boost 中，哈希被定义为所有基本类型数组（例如 `int`、`float`、`double` 和 `char`），以及所有 STL 容器，包括 `std::array`、`std::tuple` 和 `std::type_index`。一些库也提供了哈希特化，例如，`Boost.Variant` 可以对任何 `boost::variant` 类进行哈希。

## 参见

+   阅读有关使用无序集和映射的菜谱，以了解更多关于哈希函数使用的信息。

+   `Boost.Functional/Hash` 的官方文档会告诉你如何组合多个哈希并提供更多示例。请阅读[`www.boost.org/doc/libs/1_53_0/doc/html/hash.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/ha)。

# 使用无序集和映射

在之前的菜谱中，我们看到了如何通过哈希来优化字符串比较。阅读之后，可能会产生以下疑问：“我们能否创建一个容器来缓存哈希值，以便更快地进行比较？”

答案是肯定的，我们可以做更多。我们可以实现几乎恒定的时间复杂度，用于搜索、插入和删除元素。

## 准备工作

需要具备 C++ 和 STL 容器的基本知识。阅读之前的食谱也会有所帮助。

## 如何做到这一点...

这将是所有食谱中最简单的一个：

1.  您只需要包含 `<boost/unordered_map.hpp>` 头文件，如果我们想使用映射，或者包含 `<boost/unordered_set.hpp>` 头文件，如果我们想使用集合。

1.  现在，您可以自由地使用 `boost::unordered_map` 而不是 `std::map`，以及使用 `boost::unordered_set` 而不是 `std::set`：

    ```cpp
    #include <boost/unordered_set.hpp>
    void example() {
        boost::unordered_set<std::string> strings;

        strings.insert("This");
        strings.insert("is");
        strings.insert("an");
        strings.insert("example");

        assert(strings.find("is") != strings.cend());
    }
    ```

## 它是如何工作的...

无序容器存储值并记住每个值的哈希。现在，如果您想在这些容器中查找一个值，它们将计算该值的哈希并搜索容器中的该哈希。找到哈希后，容器将检查找到的值与搜索值之间的相等性。然后，返回值的迭代器或容器的末尾迭代器。

因为容器可以搜索一个常宽整数的哈希值，它可能使用一些仅适用于整数的优化和算法。这些算法保证了常数搜索复杂度 O(1)，而传统的 `std::set` 和 `std::map` 提供的复杂度更差，为 O(log(N))，其中 N 是容器中元素的数量。这导致了一个情况，即传统 `std::set` 或 `std::map` 中的元素越多，其工作速度越慢。然而，无序容器的性能并不依赖于元素数量。

这样的高性能并不是免费的。在无序容器中，值是无序的（您不会感到惊讶，对吧？）。这意味着如果我们将从 `begin()` 到 `end()` 输出容器的元素，如下所示：

```cpp
template <class T>
void output_example() {
    T strings;

    strings.insert("CZ"); strings.insert("CD");
    strings.insert("A"); strings.insert("B");
    std::copy(
        strings.begin(),
        strings.end(),
        std::ostream_iterator<std::string>(std::cout, "  ")
    );
}
```

对于 `std::set` 和 `boost::unordered_set`，我们将得到以下输出：

```cpp
boost::unordered_set<std::string> : B  A  CD  CZ
std::set<std::string> : A  B  CD  CZ
```

那么，性能差异有多大？看看以下输出：

```cpp
$ TIME="%E" time ./unordered s

STD matched: 20000000

0:31.39

$ TIME="%E" time ./unordered h

HASH matched: 20000000

0:26.93

```

性能是通过以下代码测量的：

```cpp
template <class T>
std::size_t test_default() {
    // Constants
    const std::size_t ii_max = 20000000;
    const std::string s("Test string");

    T map;

    for (std::size_t ii = 0; ii < ii_max; ++ii) {
        map[s + boost::lexical_cast<std::string>(ii)] = ii;
    }

    // Inserting once more
    for (std::size_t ii = 0; ii < ii_max; ++ii) {
        map[s + boost::lexical_cast<std::string>(ii)] = ii;
    }

    return map.size();
}
```

注意，代码中包含大量的字符串构造，因此使用此测试来衡量加速并不完全正确。它在这里是为了表明无序容器通常比有序容器更快。

有时可能会出现需要在使用无序容器中定义用户自定义类型的情况：

```cpp
struct my_type {
    int         val1_;
    std::string val2_;
};
```

为了做到这一点，我们需要为该类型编写一个比较运算符：

```cpp
inline bool operator == (const my_type& v1, const my_type& v2) {
    return v1.val1_ == v2.val1_ && v1.val2_ == v2.val2_;}
```

现在，为该类型特别指定哈希函数。如果类型由多个字段组成，我们通常只需要组合所有参与相等比较的字段的哈希值：

```cpp
std::size_t hash_value(const my_type& v) {
    std::size_t ret = 0u;

    boost::hash_combine(ret, v.val1_);
    boost::hash_combine(ret, v.val2_);
    return ret;
}
```

### 注意

强烈推荐使用 `boost::hash_combine` 函数组合哈希值。

## 还有更多...

容器的多版本也是可用的：`boost::unordered_multiset` 在 `<boost/unordered_set.hpp>` 头文件中定义，而 `boost::unordered_multimap` 在 `<boost/unordered_map.hpp>` 头文件中定义。就像在 STL 的情况下，容器多版本能够存储多个相等的键值。

所有的无序容器都允许你指定自己的哈希函数，而不是默认的 `boost::hash`。它们还允许你特化自己的相等比较函数，而不是默认的 `std::equal_to`。

C++11 包含了所有来自 Boost 的无序容器。你可以在头文件中找到它们：`<unordered_set>` 和 `<unordered_map>`，在 `std::` 命名空间中，而不是 `boost::`。Boost 和 STL 版本具有相同的性能，并且必须以相同的方式工作。然而，Boost 的无序容器甚至在 C++03 编译器上也是可用的，并利用了 `Boost.Move` 的右值引用仿真，因此你可以使用这些容器来处理 C++03 中的移动只类。

C++11 没有提供 `hash_combine` 函数，因此你需要自己编写：

```cpp
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
```

或者直接使用 `boost::hash_combine`。

## 参考信息

+   关于 `Boost.Move` 的右值引用仿真的详细信息，请参考第一章中的*使用 C++11 移动仿真*配方。

+   关于无序容器的更多信息可以在官方网站上找到[`www.boost.org/doc/libs/1_53_0/doc/html/unordered.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/unordered.html)

+   关于组合哈希和计算范围哈希的更多信息，请访问[`www.boost.org/doc/libs/1_53_0/do`](http://www.boost.org/doc/libs/1_53_0/do)[c/html/hash.html](http://c/html/hash.html)

# 创建一个值也是键的映射

每年有几次，我们需要一种可以存储和索引一对值的东西。此外，我们需要使用第二个值来获取对的第一个部分，使用第一个值来获取第二个部分。困惑吗？让我给你举一个例子。我们正在创建一个词汇类，当用户将其值放入其中时，该类必须返回标识符；当用户将其标识符放入其中时，该类必须返回值。

为了更实用，用户将输入登录名到我们的词汇中，并希望获取一个人的唯一标识符。他们还希望使用标识符获取所有人员的姓名。

让我们看看如何使用 Boost 来实现它。

## 准备工作

为了完成这个配方，需要基本的 STL 和模板知识。

## 如何做到这一点...

这个配方是关于 `Boost.Bimap` 库的能力。让我们看看如何使用它来实现这个任务：

1.  我们需要以下包含：

    ```cpp
    #include <boost/bimap.hpp>
    #include <boost/bimap/multiset_of.hpp>
    ```

1.  现在我们已经准备好创建我们的词汇结构：

    ```cpp
        typedef boost::bimap<
            std::string,
            boost::bimaps::multiset_of<std::size_t>
        > name_id_type;

        name_id_type name_id;
    ```

1.  可以使用以下语法来填充：

    ```cpp
        // Inserting keys <-> values
        name_id.insert(name_id_type::value_type(
            "John Snow", 1
        ));

        name_id.insert(name_id_type::value_type(
            "Vasya Pupkin", 2
        ));

        name_id.insert(name_id_type::value_type(
            "Antony Polukhin", 3
        ));

        // Same person as "Antony Polukhin"
        name_id.insert(name_id_type::value_type(
            "Anton Polukhin", 3
        ));
    ```

1.  我们可以像处理映射的左侧一样处理双向映射的左侧：

    ```cpp
        std::cout << "Left:\n";
        typedef name_id_type::left_const_iterator  left_const_iterator;
        for (left_const_iterator it = name_id.left.begin(),
             iend = name_id.left.end();
             it!= iend;
             ++it)
        {
            std::cout << it->first << " <=> " << it->second 
                << '\n';
        }
    ```

1.  双向映射的右侧几乎与左侧相同：

    ```cpp
        std::cout << "\nRight:\n";
        typedef name_id_type::right_const_iterator right_const_iterator;
        for (right_const_iterator it = name_id.right.begin(),
             iend = name_id.right.end();
             it!= iend;
             ++it)
        {
            std::cout << it->first << " <=> " << it->second 
                << '\n';
        }
    ```

1.  我们还需要确保这个人在词汇中存在：

    ```cpp
        assert(
            name_id.find(name_id_type::value_type(
                "Anton Polukhin", 3
            )) != name_id.end()
        );
    ```

1.  那就是全部了。现在，如果我们把所有的代码（除了包含）放在 `int main()` 中，我们会得到以下输出：

    ```cpp
    Left:
    Anton Polukhin <=> 3
    Antony Polukhin <=> 3
    John Snow <=> 1
    Vasya Pupkin <=> 2

    Right:
    1 <=> John Snow
    2 <=> Vasya Pupkin
    3 <=> Antony Polukhin
    3 <=> Anton Polukhin
    ```

## 它是如何工作的...

在步骤 2 中，我们定义了 `bimap` 类型：

```cpp
    typedef boost::bimap<
        std::string,
        boost::bimaps::multiset_of<std::size_t>
    > name_id_type;
```

第一个模板参数表示第一个键必须是 `std::string` 类型，并且应该像 `std::set` 一样工作。第二个模板参数表示第二个键必须是 `std::size_t` 类型。多个第一个键可以有一个单一的第二个键值，就像在 `std::multimap` 中一样。

我们可以使用 `boost::bimaps::` 命名空间中的类来指定 `bimap` 的底层行为。我们可以将哈希映射作为第一个键的底层类型：

```cpp
#include <boost/bimap/unordered_set_of.hpp>
#include <boost/bimap/unordered_multiset_of.hpp>

typedef boost::bimap<
    boost::bimaps::unordered_set_of<std::string>, 
    boost::bimaps::unordered_multiset_of<std::size_t> 
> hash_name_id_type;
```

当我们没有指定键的行为，只是指定其类型时，`Boost.Bimap` 使用 `boost::bimaps::set_of` 作为默认行为。就像在我们的例子中，我们可以尝试使用 STL 表达以下代码：

```cpp
#include <boost/bimap/set_of.hpp>

typedef boost::bimap<
    boost::bimaps::set_of<std::string>, 
    boost::bimaps::multiset_of<std::size_t> 
> name_id_type;
```

使用 STL，它看起来像以下两个变量的组合：

```cpp
// name_id.left
std::map<std::string, std::size_t> key1; 

// name_id.right
std::multimap<std::size_t, std::string> key2; 
```

如前述注释所示，在步骤 4 中调用 `name_id.left` 将返回一个类似于 `std::map<std::string, std::size_t>` 接口的引用。在步骤 5 中从 `name_id.right` 调用将返回一个类似于 `std::multimap<std::size_t, std::string>` 接口的对象。

在步骤 6 中，我们处理整个 `bimap`，搜索键对，并确保它们在容器中。

## 还有更多...

不幸的是，C++11 没有与 `Boost.Bimap` 类似的东西。这里还有一些坏消息：`Boost.Bimap` 不支持右值引用，并且在某些编译器上，将显示大量警告。请参考您的编译器文档以获取有关抑制特定警告的信息。

好消息是，`Boost.Bimap` 通常比两个 STL 容器使用更少的内存，并且搜索速度与 STL 容器一样快。它内部没有虚函数调用，但确实使用了动态分配。

## 参见

+   下一个菜谱，*使用多索引容器*，将为您提供更多关于多索引以及可以替代 `Boost.Bimap` 的 Boost 库的信息。

+   有关 `bimap` 的更多示例和信息，请阅读官方文档，[`www.boost.org/doc/libs/1_53_0/libs/bimap/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/bimap/doc/html/index.html)

# 使用多索引容器

在之前的菜谱中，我们创建了一些词汇，当我们需要处理成对的内容时很有用。但是，如果我们需要更高级的索引呢？让我们编写一个索引人员的程序：

```cpp
struct person {
    std::size_t     id_;
    std::string     name_;
    unsigned int    height_;
    unsigned int    weight_;
    person(std::size_t id, const std::string& name, unsigned int height, unsigned int weight)
        : id_(id)
        , name_(name)
        , height_(height)
        , weight_(weight)
    {}
};

inline bool operator < (const person& p1, const person& p2) {
    return p1.name_ < p2.name_;
}
```

我们将需要很多索引；例如，按名称、ID、身高和体重。

## 准备工作

需要基本了解 STL 容器和无序映射。

## 如何做到这一点...

所有索引都可以由单个 `Boost.Multiindex` 容器构建和管理。

1.  为了做到这一点，我们需要很多包含：

    ```cpp
    #include <boost/multi_index_container.hpp>
    #include <boost/multi_index/ordered_index.hpp>
    #include <boost/multi_index/hashed_index.hpp>
    #include <boost/multi_index/identity.hpp>
    #include <boost/multi_index/member.hpp>
    ```

1.  最困难的部分是构建多索引类型：

    ```cpp
        typedef boost::multi_index::multi_index_container<
            person,
            boost::multi_index::indexed_by<
                // names are unique
                boost::multi_index::ordered_unique<
                    boost::multi_index::identity<person>
                >,
                // IDs are not unique, but we do not need then //ordered
                boost::multi_index::hashed_non_unique<
                    boost::multi_index::member<
                        person, std::size_t, &person::id_
                    >
                >,
                // Height may not be unique, but must be sorted
                boost::multi_index::ordered_non_unique<
                    boost::multi_index::member<
                        person, unsigned int, &person::height_
                    >
                >,
                // Weight may not be unique, but must be sorted
                boost::multi_index::ordered_non_unique<
                    boost::multi_index::member<
                        person, unsigned int, &person::weight_
                    >
                >
            > // closing for `boost::multi_index::indexed_by<
        > indexes_t;
    ```

1.  现在我们可以将值插入到我们的多索引中：

    ```cpp
        indexes_t persons;

        // Inserting values
        persons.insert(person(1, "John Snow", 185, 80));
        persons.insert(person(2, "Vasya Pupkin", 165, 60));
        persons.insert(person(3, "Antony Polukhin", 183, 70));
        // Same person as "Antony Polukhin"
        persons.insert(person(3, "Anton Polukhin", 182, 70));
    ```

1.  让我们构建一个用于打印索引内容的函数：

    ```cpp
    template <std::size_t IndexNo, class Indexes>
    void print(const Indexes& persons) {
        std::cout << IndexNo << ":\n";

        typedef typename Indexes::template nth_index<
                IndexNo
        >::type::const_iterator const_iterator_t;

        for (const_iterator_t it = persons.template get<IndexNo>().begin(),
             iend = persons.template get<IndexNo>().end();
             it != iend;
             ++it)
        {
            const person& v = *it;
            std::cout 
                << v.name_ << ", " 
                << v.id_ << ", " 
                << v.height_ << ", " 
                << v.weight_ << '\n'
            ;
        }

        std::cout << '\n';
    }
    ```

1.  按如下方式打印所有索引：

    ```cpp
        print<0>(persons);
        print<1>(persons);
        print<2>(persons);
        print<3>(persons);
    ```

1.  之前菜谱中的某些代码也可以使用：

    ```cpp
        assert(persons.get<1>().find(2)->name_ == "Vasya Pupkin");
        assert(
            persons.find(person(
                77, "Anton Polukhin", 0, 0
            )) != persons.end()
        );

        // Won' compile
        //assert(persons.get<0>().find("John Snow")->id_ == 1);
    ```

1.  现在如果我们运行我们的示例，它将输出索引的内容：

    ```cpp
    0:
    Anton Polukhin, 3, 182, 70
    Antony Polukhin, 3, 183, 70
    John Snow, 1, 185, 80
    Vasya Pupkin, 2, 165, 60

    1:
    John Snow, 1, 185, 80
    Vasya Pupkin, 2, 165, 60
    Anton Polukhin, 3, 182, 70
    Antony Polukhin, 3, 183, 70

    2:
    Vasya Pupkin, 2, 165, 60
    Anton Polukhin, 3, 182, 70
    Antony Polukhin, 3, 183, 70
    John Snow, 1, 185, 80

    3:
    Vasya Pupkin, 2, 165, 60
    Antony Polukhin, 3, 183, 70
    Anton Polukhin, 3, 182, 70
    John Snow, 1, 185, 80
    ```

## 它是如何工作的...

这里最困难的部分是使用 `boost::multi_index::multi_index_container` 构造一个多索引类型。第一个模板参数是我们将要索引的类。在我们的例子中，它是 `person`。第二个参数是类型 `boost::multi_index::indexed_by`，所有索引都必须描述为该类的模板参数。

现在，让我们看看第一个索引描述：

```cpp
  boost::multi_index::ordered_unique<
    boost::multi_index::identity<person>
  >
```

`boost::multi_index::ordered_unique` 类的使用意味着索引必须像 `std::set` 一样工作，并且具有所有其成员。`boost::multi_index::identity<person>` 类意味着索引将使用 `person` 类的 `operator <` 进行排序。

下一个表格显示了 `Boost.MultiIndex` 类型与 STL 容器之间的关系：

| The Boost.MultiIndex types | STL containers |
| --- | --- |
| `boost::multi_index::ordered_unique` | `std::set` |
| `boost::multi_index::ordered_non_unique` | `std::multiset` |
| `boost::multi_index::hashed_unique` | `std::unordered_set` |
| `boost::multi_index::hashed_non_unique` | `std::unordered_multiset` |
| `boost::multi_index::sequenced` | `std::list` |

让我们看看第二个索引：

```cpp
    boost::multi_index::hashed_non_unique<
      boost::multi_index::member<
         person, std::size_t, &person::id_
      >
    >
```

`boost::multi_index::hashed_non_unique` 类型意味着索引将像 `std::set` 一样工作，而 `boost::multi_index::member<person, std::size_t, &person::id_>` 意味着索引将仅对人的结构体中的单个成员字段应用哈希函数，即 `person::id_`。

现在剩余的索引不会造成麻烦，因此让我们看看在打印函数中使用索引的方式。获取特定索引的迭代器类型是通过以下代码完成的：

```cpp
    typedef typename Indexes::template nth_index<
            IndexNo
    >::type::const_iterator const_iterator_t;
```

这看起来稍微有些复杂，因为 `Indexes` 是一个模板参数。如果我们可以在这个 `indexes_t` 的作用域中编写此代码，示例将更简单：

```cpp
    typedef indexes_t::nth_index<0>::type::const_iterator const_iterator_t;
```

`nth_index` 成员元函数接受一个基于零的索引号来使用。在我们的例子中，索引 1 是 ID 的索引，索引 2 是高度的索引，以此类推。

现在，让我们看看如何使用 `const_iterator_t`：

```cpp
    for (const_iterator_t it = persons.template get<IndexNo>().begin(),
         iend = persons.template get<IndexNo>().end();
         it != iend;
         ++it)
    {
        const person& v = *it;
        // ...
```

这也可以通过在作用域中简化 `indexes_t`：

```cpp
    for (const_iterator_t it = persons.get<0>().begin(),
         iend = persons.get<0>().end();
         it != iend;
         ++it)
    {
        const person& v = *it;
        // ...
```

函数 `get<indexNo>()` 返回索引。我们可以几乎像使用 STL 容器一样使用那个索引。

## 还有更多...

C++11 没有多个索引库。`Boost.MultiIndex` 库是一个快速库，不使用虚拟函数。`Boost.MultiIndex` 的官方文档包含了性能和内存使用度量，显示在大多数情况下，这个库使用的内存比基于 STL 的手写代码少。不幸的是，`boost::multi_index::multi_index_container` 不支持 C++11 特性，也没有使用 `Boost.Move` 的右值引用模拟。

## 参考信息

+   `Boost.MultiIndex` 的官方文档包含教程、性能度量、示例以及其他 `Boost.Multiindex` 库的有用功能描述。请参阅[`www.boost.org/doc/libs/1_53_0/libs/multi_index/doc/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/multi_index/doc/index.html)。

# 获取单链表和内存池的好处

现在，当我们需要非关联和非有序容器时，我们通常使用 `std::vector`。这在 *C++ Coding Standards* 一书中由 *Andrei Alexandrescu* 和 *Herb Sutter* 推荐，甚至那些没有读过这本书的用户通常也会使用 `std::vector`。为什么？因为 `std::list` 比较慢，并且使用的资源比 `std::vector` 多得多。`std::deque` 容器非常接近 `std::vector`，但它存储的值不是连续的。

一切都很好，直到我们不需要容器；然而，如果我们需要容器，删除和插入元素不会使迭代器失效。然后我们被迫选择较慢的 `std::list`。

但是等等，Boost 对于这种情况有一个很好的解决方案！

## 准备工作

为了理解介绍部分，需要具备对 STL 容器的良好了解。之后，只需要对 C++ 和 STL 容器有基本了解即可。

## 如何做到这一点...

在这个菜谱中，我们将同时使用两个 Boost 库：`Boost.Pool` 和来自 `Boost.Container` 的单链表。

1.  我们需要以下头文件：

    ```cpp
    #include <boost/pool/pool_alloc.hpp>
    #include <boost/container/slist.hpp>
    ```

1.  现在我们需要描述我们的列表类型。这可以通过以下代码实现：

    ```cpp
    typedef boost::fast_pool_allocator<int> allocator_t;
    typedef boost::container::slist<int, allocator_t> slist_t;
    ```

1.  我们可以像使用 `std::list` 一样使用我们的单链表。看看用于测量两种列表类型速度的函数：

    ```cpp
    template <class ListT>
    void test_lists() {
        typedef ListT list_t;

        // Inserting 1000000 zeros
        list_t  list(1000000, 0);
        for (int i = 0; i < 1000; ++i) {
            list.insert(list.begin(), i);
        }

        // Searching for some value
        typedef typename list_t::iterator iterator;
        iterator it = std::find(list.begin(), list.end(), 777);
        assert(it != list.end());

        // Erasing some values
        for (int i = 0; i < 100; ++i) {
            list.pop_front();
        }

        // Iterator still valid and points to same value
        assert(it != list.end());
        assert(*it == 777);

        // Inserting more values
        for (int i = -100; i < 10; ++i) {
            list.insert(list.begin(), i);
        }

        // Iterator still valid and points to same value
        assert(it != list.end());
        assert(*it == 777);

        list_specific(list, it);
    }
    ```

1.  每种列表类型特有的功能被移动到 `list_specific` 函数中：

    ```cpp
    void list_specific(slist_t& list, slist_t::iterator it) {
        typedef slist_t::iterator iterator;

        // Erasing element 776
        assert( *(++iterator(it)) == 776);
        assert(*it == 777);
        list.erase_after(it);
        assert(*it == 777);
        assert( *(++iterator(it)) == 775);

        // Freeing memory
        boost::singleton_pool<
            boost::pool_allocator_tag,
            sizeof(int)
        >::release_memory();
    }

    #include <list>
    typedef std::list<int> stdlist_t;

    void list_specific(stdlist_t& list, stdlist_t::iterator it) {
        typedef stdlist_t::iterator iterator;

        // Erasing element 776
        ++it;
        assert( *it == 776);
        it = list.erase(it);
        assert(*it == 775);
    }
    ```

## 它是如何工作的...

当我们使用 `std::list` 时，我们可能会注意到速度变慢，因为列表的每个节点都需要单独的分配。这意味着通常当我们向 `std::list` 插入 10 个元素时，容器会调用 new 10 次。

正是因为这个原因，我们使用了来自 `Boost.Pool` 的 `boost::fast_pool_allocator<int>`。这个分配器试图分配更大的内存块，这样在稍后的阶段，多个节点可以构建而无需调用分配新的内存。

`Boost.Pool` 库有一个缺点——它使用内存来满足内部需求。通常，每个元素都会使用额外的 `sizeof` 指针。为了解决这个问题，我们使用了来自 `Boost.Containers` 的单链表。

`boost::container::slist` 类更紧凑，但它的迭代器只能向前迭代。对于熟悉 STL 容器的读者来说，步骤 3 将是微不足道的，所以我们转到步骤 4 来查看一些 `boost::container::slist` 特定的功能。由于单链表迭代器只能向前迭代，传统的插入和删除算法将需要线性时间 O(N)。那是因为当我们删除或插入时，前一个元素必须被修改以指向列表的新元素。为了解决这个问题，单链表有 `erase_after` 和 `insert_after` 方法，它们可以在常数时间 O(1) 内工作。这些方法在迭代器的当前位置之后插入或删除元素。

### 注意

然而，在单链表的开始处删除和插入值并没有太大的区别。

仔细看看以下代码：

```cpp
    boost::singleton_pool<
        boost::pool_allocator_tag,
        sizeof(int)
    >::release_memory();
```

这是因为 `boost::fast_pool_allocator` 不会释放内存，所以我们必须手动完成。来自 第三章 的 *在作用域退出时做某事* 配方将有助于释放 `Boost.Pool`。

让我们看看执行结果以查看差异：

```cpp
$TIME="Runtime=%E RAM=%MKB" time ./slist_and_pool l

std::list: Runtime=0:00.05 RAM=32440KB

$ TIME="Runtime=%E RAM=%MKB" time ./slist_and_pool s

slist_t:   Runtime=0:00.02 RAM=17564KB

```

如我们所见，`slist_t` 使用了内存的一半，并且比 `std::list` 类快两倍。

## 还有更多...

C++11 有 `std::forward_list`，它与 `boost::containers::slist` 非常相似。它也有 `*_after` 方法，但没有 `size()` 方法。它们的性能相同，并且它们都没有虚拟函数，所以这些容器既快又可靠。然而，Boost 版本也可以在 C++03 编译器上使用，并且甚至通过 `Boost.Move` 提供对右值引用仿真的支持。

池不是 C++11 的一部分。请使用 Boost 的版本；它既快又不使用虚拟函数。

### 注意

为什么 `boost::fast_pool_allocator` 不能自己释放内存？那是因为 C++03 没有状态分配器，所以容器不会复制和存储分配器。这使得无法实现一个可以自己释放内存的 `boost::fast_pool_allocator` 函数。

## 参见

+   `Boost.Pool` 的官方文档包含了更多关于内存池的示例和类。请在此处阅读：[`www.boost.org/doc/libs/1_53_0/libs/pool/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/pool/doc/html/index.html)。

+   *使用平面关联容器* 的配方将向您介绍 `Boost.Container` 中的一些更多类。您也可以阅读 `Boost.Container` 的官方文档，自己学习该库，或在其类中获取完整的参考文档：[`www.boost.org/doc/libs/1_53_0/doc/html/container.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/container.html)。

+   有关为什么可能需要状态分配器的信息，请参阅 [`www.boost.org/doc/libs/1_53_0/doc/html/interprocess/allocators_containers.html#interprocess.allocators_containers.allocator_introduction`](http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess/allocators_containers.html#interprocess.allocators_containers.allocator_introduction)。

+   *向量与列表*，以及其他来自 C++ 编程语言发明者 *Bjarne Stroustrup* 的有趣话题，可以在 [`channel9.msdn.com/Events/GoingNative/GoingNative-2012/Keynote-Bjarne-Stroustrup`](http://channel9.msdn.com/Events/GoingNative/GoingNative-2012/Keynote-Bjarne-Stroustrup)[-Cpp11-Style](http://-Cpp11-Style) 找到。

# 使用平面关联容器

在阅读了之前的菜谱后，一些读者可能会开始到处使用快速池分配器；特别是，对于 `std::set` 和 `std::map`。好吧，我不会阻止你这样做，但至少让我们看看一个替代方案：平面关联容器。这些容器是在传统的向量容器之上实现的，并按顺序存储值。

## 准备就绪

需要掌握 STL 关联容器的基本知识。

## 如何做到这一点...

平面容器是 `Boost.Container` 库的一部分。我们已经在之前的菜谱中看到了如何使用其中的一些容器。在这个菜谱中，我们将使用一个 `flat_set` 关联容器：

1.  我们只需要包含一个头文件：

    ```cpp
    #include <boost/container/flat_set.hpp>
    ```

1.  之后，我们可以自由地构建平面容器：

    ```cpp
        boost::container::flat_set<int> set;
    ```

1.  为元素预留空间：

    ```cpp
        set.reserve(4096);
    ```

1.  填充容器：

    ```cpp
        for (int i = 0; i < 4000; ++i) {
            set.insert(i);
        }
    ```

1.  现在，我们可以像使用 `std::set` 一样使用它：

    ```cpp
        // 5.1
        assert(set.lower_bound(500) - set.lower_bound(100) == 400);

        // 5.2
        set.erase(0);

        // 5.3
        set.erase(5000);

        // 5.4
        assert(std::lower_bound(set.cbegin(), set.cend(), 900000) == set.cend());

        // 5.5
        assert(
            set.lower_bound(100) + 400 
            == 
            set.find(500)
        );
    ```

## 它是如何工作的...

步骤 1 和 2 很简单，但步骤 3 需要特别注意。这是在使用平面关联容器和 `std::vector` 时最重要的步骤之一。

`boost::container::flat_set` 类将它的值有序地存储在向量中，这意味着任何元素的插入或删除都花费线性时间 O(N)，就像在 `std::vector` 的情况下。这是一个必要的恶。但为此，我们几乎每个元素节省了三倍多的内存使用，更友好的处理器缓存存储，以及随机访问迭代器。看看第 5 步，`5.1`，在那里我们获取由 `lower_bound` 成员函数调用返回的两个迭代器之间的距离。使用平面集合获取距离需要常数时间 O(1)，而同样操作 `std::set` 的迭代器需要线性时间 O(N)。在 `5.1` 的情况下，使用 `std::set` 获取距离会比获取平面集合容器的距离慢 400 倍。

回到第 3 步。如果不预留内存，插入元素有时会变慢且内存效率较低。`std::vector`类分配所需的内存块，并在该块上就地构造元素。当我们不预留内存而插入一些元素时，可能会出现预分配的内存块上没有剩余空间的情况，因此`std::vector`将分配两倍于之前分配的内存块。之后，`std::vector`将复制或移动第一个块中的元素到第二个块，删除第一个块中的元素，并释放第一个块的内存。只有在那时，插入才会发生。这种复制和释放内存可能会在插入过程中多次发生，从而大大降低速度。

### 备注

如果你知道`std::vector`或任何扁平容器必须存储的元素数量，在插入之前为这些元素预留空间。这个规则没有例外！

第 4 步很简单，我们在这里插入元素。请注意，我们正在插入有序元素。这不是必需的，但推荐这样做以加快插入速度。在`std::vector`的末尾插入元素比在中间或开头插入要便宜得多。

在第 5 步中，`5.2`和`5.3`没有太大区别，除了它们的执行速度。删除元素的规定与插入元素的规定几乎相同，所以请参阅前面的段落以获取解释。

### 备注

也许我在告诉你关于容器的一些简单事情，但我看到一些非常流行的产品使用了 C++11 的特性，有大量的优化，以及 STL 容器的糟糕使用，特别是`std::vector`。

在第 5 步中，`5.4`展示了`std::lower_bound`函数使用`boost::container::flat_set`比使用`std::set`要快，因为具有随机访问迭代器。

在第 5 步中，`5.5`也展示了随机访问迭代器的优势。请注意，我们在这里没有使用`std::find`函数。这是因为该函数需要线性时间 O(N)，而成员`find`函数需要对数时间 O(log(N))。

## 还有更多...

我们应该在什么时候使用扁平容器，什么时候使用常规容器？嗯，这取决于你，但这里有一个从`Boost.Container`官方文档中摘录的差异列表，这将帮助你做出决定：

+   比标准关联容器查找更快

+   比标准关联容器迭代得更快

+   对于小对象（如果使用`shrink_to_fit`则对于大对象）内存消耗更少

+   改善缓存性能（数据存储在连续内存中）

+   非稳定迭代器（在插入和删除元素时迭代器会被无效化）

+   无法存储不可复制和不可移动的值类型

+   比标准关联容器具有较弱的异常安全性（复制/移动构造函数在删除和插入时移动值可能会抛出异常）

+   比标准关联容器插入和删除更慢（特别是对于不可移动的类型）

不幸的是，C++11 没有扁平容器。Boost 中的扁平容器速度快，有很多优化，并且不使用虚函数。`Boost.Containers`中的类通过`Boost.Move`支持 rvalue 引用的模拟，因此您可以在 C++03 编译器上自由使用它们。

## 参见

+   有关`Boost.Container`的更多信息，请参考*获取单链表和内存池的好处*配方。

+   在第一章中，*使用 C++11 移动模拟*的配方，将向您介绍如何在 C++03 兼容的编译器上实现仿值引用的基础知识。

+   `Boost.Container`的官方文档包含了关于`Boost.Container`的大量有用信息以及每个类的完整参考。请访问[`www.boost.org/doc/libs/1_53_0/doc/html/container.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/container.html)了解更多信息。
