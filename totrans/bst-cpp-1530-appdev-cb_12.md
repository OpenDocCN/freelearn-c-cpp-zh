# 第十二章。冰山一角

在本章中，我们将涵盖：

+   处理图

+   可视化图

+   使用真随机数生成器

+   使用可移植的数学函数

+   编写测试用例

+   在一个测试模块中组合多个测试用例

+   操作图像

# 简介

Boost 是一个庞大的库集合。其中一些库很小，适合日常使用，而其他一些则需要单独的书籍来描述它们的所有功能。本章致力于介绍这些大型库，并为你提供一些基础知识以开始使用。

前两个菜谱将解释 `Boost.Graph` 的用法。这是一个拥有大量算法的大库。我们将看到一些基础知识，以及它最重要的部分——图的可视化。

我们还将看到一个非常有用的生成真随机数的菜谱。这对于编写安全的加密系统来说是一个非常重要的要求。

一些 C++ 标准库缺少数学函数。我们将看到如何使用 Boost 来解决这个问题。但本书的格式没有足够的空间来描述所有这些函数。

编写测试用例在 *编写测试用例* 和 *在一个测试模块中组合多个测试用例* 菜谱中描述。这对于任何生产级系统来说都很重要。

最后一个菜谱是关于一个在我大学期间帮助我在许多课程中取得成功的库。可以使用它创建和修改图像。我本人用它来可视化不同的算法、在图像中隐藏数据、签名图像和生成纹理。

不幸的是，即使是这一章也无法告诉你关于所有 Boost 库的信息。也许有一天我会写另一本书...然后是几本更多。

# 处理图

一些任务需要数据的图形表示。`Boost.Graph` 是一个库，旨在提供一种灵活的方式来在内存中构建和表示图。它还包含许多用于处理图的算法，例如拓扑排序、广度优先搜索、深度优先搜索和 Dijkstra 最短路径。

好吧，让我们用 `Boost.Graph` 执行一些基本任务！

## 准备工作

对于这个菜谱，只需要具备基本的 C++ 和模板知识。

## 如何做...

在这个菜谱中，我们将描述一个图类型，创建该类型的图，向图中添加一些顶点和边，并搜索特定的顶点。这应该足以开始使用 `Boost.Graph`。

1.  我们首先描述图类型：

    ```cpp
    #include <boost/graph/adjacency_list.hpp>
    #include <string>

    typedef std::string vertex_t;
    typedef boost::adjacency_list<
        boost::vecS
        , boost::vecS
        , boost::bidirectionalS
        , vertex_t
    > graph_type;
    ```

1.  现在我们来构建它：

    ```cpp
        graph_type graph;
    ```

1.  让我们使用一个非可移植的技巧来加速图构建：

    ```cpp
        static const std::size_t vertex_count = 5;
        graph.m_vertices.reserve(vertex_count);
    ```

1.  现在我们已经准备好向图中添加顶点了：

    ```cpp
    typedef boost::graph_traits<graph_type>
                ::vertex_descriptor descriptor_t;

        descriptor_t cpp 
            = boost::add_vertex(vertex_t("C++"), graph);
        descriptor_t stl 
            = boost::add_vertex(vertex_t("STL"), graph);
        descriptor_t boost 
            = boost::add_vertex(vertex_t("Boost"), graph);
        descriptor_t guru 
            = boost::add_vertex(vertex_t("C++ guru"), graph);
        descriptor_t ansic 
            = boost::add_vertex(vertex_t("C"), graph);
    ```

1.  是时候用边连接顶点了：

    ```cpp
        boost::add_edge(cpp, stl, graph);
        boost::add_edge(stl, boost, graph);
        boost::add_edge(boost, guru, graph);
        boost::add_edge(ansic, guru, graph);
    ```

1.  我们编写一个搜索顶点的函数：

    ```cpp
    template <class GraphT>
    void find_and_print(const GraphT& g, boost::string_ref name) {
    ```

1.  现在我们将编写代码来获取所有顶点的迭代器：

    ```cpp
        typedef typename boost::graph_traits<graph_type>
                ::vertex_iterator vert_it_t;

        vert_it_t it, end;
        boost::tie(it, end) = boost::vertices(g);
    ```

1.  是时候运行搜索以查找所需的顶点了：

    ```cpp
        typedef boost::graph_traits<graph_type>::vertex_descriptor desc_t;
        for (; it != end; ++ it) {
            desc_t desc = *it;
            if (boost::get(boost::vertex_bundle, g)[desc] 
                     == name.data()) {
                break;
            }
        }
        assert(it != end);
        std::cout << name << '\n';
    } /* find_and_print */
    ```

## 它是如何工作的...

在第 1 步，我们描述了我们的图必须看起来像什么以及它必须基于什么类型。`boost::adjacency_list`是一个表示图作为二维结构的类，其中第一个维度包含顶点，第二个维度包含该顶点的边。`boost::adjacency_list`必须是表示图的默认选择；它适用于大多数情况。

第一个模板参数`boost::adjacency_list`描述了用于表示每个顶点的边列表的结构；第二个描述了存储顶点的结构。我们可以使用特定的选择器为这些结构选择不同的 STL 容器，如下表所示：

| 选择器 | STL 容器 |
| --- | --- |
| `boost::vecS` | `std::vector` |
| `boost::listS` | `std::list` |
| `boost::slistS` | `std::slist` |
| `boost::setS` | `std::set` |
| `boost::multisetS` | `std::multiset` |
| `boost::hash_setS` | `std::hash_set` |

第三个模板参数用于创建无向、有向或双向图。分别使用`boost::undirectedS`、`boost::directedS`和`boost::bidirectionalS`选择器。

第五个模板参数描述了将用作顶点的数据类型。在我们的例子中，我们选择了`std::string`。我们也可以支持边的数据类型，并将其作为模板参数提供。

第 2 步和第 3 步是微不足道的，但在第 4 步，您将看到一种非可移植的方式来加快图构建。在我们的例子中，我们使用`std::vector`作为存储顶点的容器，因此我们可以强制它为所需数量的顶点保留内存。这导致在将顶点插入图时，内存分配/释放和复制操作更少。这一步是非可移植的，因为它高度依赖于`boost::adjacency_list`的当前实现以及存储顶点的所选容器类型。

在第 4 步，我们看到如何将顶点添加到图中。注意`boost::graph_traits<graph_type>`的使用。`boost::graph_traits`类用于获取特定于图类型的类型。我们将在本章后面看到其用法和一些特定于图类型的描述。第 5 步展示了我们需要做什么来通过边连接顶点。

### 注意

如果我们提供了边的数据类型，添加边的样子如下：

`boost::add_edge(ansic, guru, edge_t(initialization_parameters), graph)`

注意，在第 6 步中，图类型是一个`template`参数。这建议为了实现更好的代码重用并使此函数能够与其它图类型一起工作。

在第 7 步，我们看到如何遍历图中的所有顶点。顶点迭代器的类型来自`boost::graph_traits`。函数`boost::tie`是`Boost.Tuple`的一部分，用于从元组中获取值到变量中。因此，调用`boost::tie(it, end) = boost::vertices(g)`将`begin`迭代器放入`it`变量中，将`end`迭代器放入`end`变量中。

这可能让你感到惊讶，但顶点迭代器的解引用并不返回顶点数据。相反，它返回顶点描述符 `desc`，可以在 `boost::get(boost::vertex_bundle, g)[desc]` 中使用以获取顶点数据，就像我们在第 8 步中所做的那样。顶点描述符类型在许多 `Boost.Graph` 函数中使用；我们在第 5 步的边构造函数中看到了它的使用。

### 注意

如前所述，`Boost.Graph` 库包含了众多算法的实现。你将发现许多搜索策略已经实现，但在这本书中我们不会讨论它们。我们将仅限于介绍图库的基础知识。

## 还有更多...

`Boost.Graph` 库不是 C++11 的一部分，也不会成为 C++1y 的一部分。当前的实现不支持 C++11 功能。如果我们使用的是难以复制的顶点，我们可以使用以下技巧来提高速度：

```cpp
vertex_descriptor desc = boost::add_vertex(graph);boost::get(boost::vertex_bundle, g_)[desc] = std::move(vertex_data);
```

它避免了 `boost::add_vertex(vertex_data, graph)` 的复制构造，而是使用带有移动赋值的默认构造。

`Boost.Graph` 的效率取决于多个因素，例如底层容器类型、图表示、边和顶点数据类型。

## 相关内容

+   阅读关于 *可视化图* 的食谱可以帮助你更轻松地处理图。你也可以考虑阅读以下链接中的官方文档：

    [`www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_contents.html`](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_contents.html)

# 可视化图

由于可视化问题，制作操作图的程序从未容易过。当我们使用 STL 容器，如 `std::map` 和 `std::vector` 时，我们总能打印容器的内容并查看内部发生的情况。但是，当我们处理复杂的图时，很难以清晰的方式可视化内容：顶点太多，边太多。

在这个食谱中，我们将探讨使用 **Graphviz** 工具对 `Boost.Graph` 的可视化。

## 准备工作

要可视化图，你需要一个 Graphviz 可视化工具。还需要了解前面的食谱。

## 如何操作...

可视化分为两个阶段。在第一阶段，我们让程序以文本格式输出图的描述；在第二阶段，我们将第一步的输出导入到某个可视化工具中。本食谱中的编号步骤都是关于第一阶段的内容。

1.  让我们像前一个食谱中那样为 `graph_type` 编写 `std::ostream` 操作符：

    ```cpp
    #include <boost/graph/graphviz.hpp>
    std::ostream& operator<<(std::ostream& out, const graph_type& g) {
        detail::vertex_writer<graph_type> vw(g);
        boost::write_graphviz(out, g, vw);
        return out;
    }
    ```

1.  在前面的步骤中使用到的 `detail::vertex_writer` 结构必须定义为以下内容：

    ```cpp
    namespace detail {

        template <class GraphT>
        class vertex_writer {
            const GraphT& g_;

        public:
            explicit vertex_writer(const GraphT& g)
                : g_(g)
            {}

            template <class VertexDescriptorT>
            void operator()(std::ostream& out, 
               const VertexDescriptorT& d) const 
            {
               out << " [label=\""
                   << boost::get(boost::vertex_bundle, g_)[d] 
                   << "\"]"; 
            }
        }; // vertex_writer

    } // namespace detail
    ```

就这些了。现在，如果我们使用 `std::cout << graph;` 命令可视化前一个食谱中的图，输出可以被用来使用 `dot` 命令行工具创建图形图片：

```cpp
$ dot -Tpng -o dot.png

digraph G {

0 [label="C++"];

1 [label="STL"];

2 [label="Boost"];

3 [label="C++ guru"];

4 [label="C"];

0->1 ;

1->2 ;

2->3 ;

4->3 ;

}

```

前一个命令的输出如图所示：

![如何操作...](img/4880OS_12_02.jpg)

如果命令行让你感到害怕，我们也可以使用 **Gvedit** 或 **XDot** 程序进行可视化。

## 它是如何工作的...

`Boost.Graph` 库包含将图输出为 Graphviz (DOT) 格式的函数。如果我们按步骤 1 使用两个参数写入 `boost::write_graphviz(out, g)`，该函数将输出一个顶点从 `0` 开始编号的图图片。这并不很有用，因此我们提供了一个 `vertex_writer` 类的实例，该实例输出顶点名称。

正如我们在第二步中看到的，输出格式必须是 DOT 格式，这是 Graphviz 工具可以理解的。你可能需要阅读 Graphviz 文档以获取有关 DOT 格式的更多信息。

如果你想在可视化过程中向边添加一些数据，我们需要将边可视化实例作为第四个参数提供给 `boost::write_graphviz`。

## 还有更多...

C++11 不包含 `Boost.Graph` 或图形可视化的工具。但你不必担心——有很多其他的图形格式和可视化工具，`Boost.Graph` 可以与它们中的很多一起工作。

## 参见

+   *与图一起工作* 的配方包含有关 `Boost.Graphs` 构造的信息。

+   你可以在 [`www.graphviz.org/`](http://www.graphviz.org/) 找到关于 DOT 格式和 Graphviz 的很多信息。

+   Boost 的官方文档 `Boost.Graph` 库包含多个示例和有用的信息，可以在 [`www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_`](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_)[contents.html](http://contents.html) 找到。

# 使用真正的随机数生成器

我知道很多商业产品使用错误的方法来获取随机数。遗憾的是，一些公司仍然在密码学和银行软件中使用 `rand()`。

让我们看看如何使用 `Boost.Random` 获取一个完全随机的均匀分布，这对于银行软件来说是合适的。

## 准备工作

对于这个配方，需要具备基本的 C++ 知识。了解不同类型的分布也将有所帮助。这个配方中的代码需要链接到 `boost_random` 库。

## 如何做到这一点...

要创建真正的随机数，我们需要从操作系统或处理器那里得到一些帮助。这是使用 Boost 可以做到的：

1.  我们需要包含以下头文件：

    ```cpp
    #include <boost/config.hpp>
    #include <boost/random/random_device.hpp>
    #include <boost/random/uniform_int_distribution.hpp>
    ```

1.  高级随机数提供者在不同的平台上有不同的名称：

    ```cpp
        static const std::string provider =
    #ifdef BOOST_WINDOWS
            "Microsoft Strong Cryptographic Provider"
    #else
            "/dev/urandom"
    #endif
        ;
    ```

1.  现在我们已经准备好使用 `Boost.Random` 初始化生成器：

    ```cpp
        boost::random_device device(provider);
    ```

1.  让我们获取一个在 1000 到 65535 之间返回值的均匀分布：

    ```cpp
        boost::random::uniform_int_distribution<unsigned short> random(1000);
    ```

就这样。现在我们可以使用 `random(device)` 调用来获取真正的随机数。

## 它是如何工作的...

为什么 `rand()` 函数不适合银行？因为它生成伪随机数，这意味着黑客可以预测下一个生成的数字。这是所有伪随机数算法的问题。一些算法更容易预测，而一些则更难预测，但仍然有可能。

这就是为什么我们在本例中使用 `boost::random_device`（参见第 3 步）。该设备从整个操作系统中收集关于随机事件的信息，以构建一个不可预测的硬件生成的数字。此类事件的例子包括按键之间的延迟、某些硬件中断之间的延迟以及内部 CPU 随机数生成器。

操作系统可能拥有多个此类随机数生成器。在我们的 POSIX 系统示例中，我们使用了 `/dev/urandom` 而不是更安全的 `/dev/random`，因为后者在捕获足够的随机事件之前会保持阻塞状态。等待熵值可能需要几秒钟，这对于应用程序通常是不合适的。使用 `/dev/random` 来创建长期有效的 `GPG/SSL/SSH` 密钥。

现在我们已经完成了生成器的设置，是时候进入第 4 步，讨论分布类。如果生成器只是生成数字（通常是均匀分布），分布类将一个分布映射到另一个。在第 4 步中，我们创建了一个均匀分布，它返回一个无符号短整型的随机数。参数 `1000` 表示该分布必须返回大于或等于 `1000` 的数字。我们还可以提供一个最大数字作为第二个参数，默认情况下等于返回类型可以存储的最大值。

## 还有更多...

`Boost.Random` 为不同的需求提供了大量的真/伪随机生成器和分布。避免复制分布和生成器；这可能会变成一个昂贵的操作。

C++11 支持不同的分布类和生成器。您将在 `std::` 命名空间中的 `<random>` 头文件中找到本示例中的所有类。`Boost.Random` 库不使用 C++11 功能，并且对于该库来说也不是必需的。您应该使用 Boost 实现，还是 STL？Boost 提供了跨系统的更好可移植性；然而，某些 STL 实现可能有汇编优化的实现，并可能提供一些有用的扩展。

## 参见

+   官方文档包含了一个完整的生成器和分布列表及其描述；它可在以下链接中找到：

    [`www.boost.org/doc/libs/1_53_0/doc/html`](http://www.boost.org/doc/libs/1_53_0/doc/html) [/boost_random.html](http:///boost_random.html)

# 使用可移植的数学函数

一些项目需要特定的三角函数、用于数值求解常微分方程的库以及与分布和常量一起工作。所有这些 `Boost.Math` 的部分都很难放入甚至是一本书中。一个单独的配方肯定是不够的。所以让我们专注于处理浮点类型的基本日常使用函数。

我们将编写一个可移植的函数，用于检查输入值是否为无穷大和不是数字（NaN）值，并在值为负时更改其符号。

## 准备工作

对于此食谱，需要具备 C++ 的基本知识。那些了解 C99 标准的人会发现本食谱中有许多共同之处。

## 如何做...

执行以下步骤以检查输入值是否为无穷大和 NaN 值，并在值为负时更改符号：

1.  我们需要以下头文件：

    ```cpp
    #include <boost/math/special_functions.hpp>
    #include <cassert>
    ```

1.  断言无穷大和 NaN 可以这样做：

    ```cpp
    template <class T>
    void check_float_inputs(T value) {
        assert(!boost::math::isinf(value));
        assert(!boost::math::isnan(value));
    ```

1.  使用以下代码来更改符号：

    ```cpp
        if (boost::math::signbit(value)) {
            value = boost::math::changesign(value);
        }

        // ...
    } // check_float_inputs
    ```

就这些！现在我们可以检查 `check_float_inputs(std::sqrt(-1.0))` 和 `check_float_inputs(std::numeric_limits<double>::max() * 2.0)` 将导致断言。

## 它是如何工作的...

实数类型有特定的值，无法使用相等运算符进行检查。例如，如果变量 `v` 包含 NaN，`assert(v!=v)` 可能通过或不通过，这取决于编译器。

对于此类情况，`Boost.Math` 提供了可以可靠地检查无穷大和 NaN 值的函数。

第 3 步包含 `boost::math::signbit` 函数，需要澄清。此函数返回一个有符号位，当数字为负时为 1，当数字为正时为 0。换句话说，如果值为负，则返回 `true`。

看到第 3 步，一些读者可能会问：“为什么我们不能直接乘以 `-1` 而不是调用 `boost::math::changesign`？”。我们可以。但是乘法可能比 `boost::math::changesign` 慢，并且对于特殊值不起作用。例如，如果你的代码可以处理 `nan`，第 3 步中的代码将能够改变 `-nan` 的符号并将 `nan` 写入变量。

### 注意

`Boost.Math` 库维护者建议将此示例中的数学函数用圆括号括起来，以避免与 C 宏冲突。最好写成 `(boost::math::isinf)(value)` 而不是 `boost::math::isinf(value)`。

## 还有更多...

C99 包含了本食谱中描述的所有函数。为什么在 Boost 中需要它们呢？嗯，一些编译器供应商认为程序员不需要它们，所以你不会在一个非常流行的编译器中找到它们。另一个原因是 `Boost.Math` 函数可以用于像数字一样行为的类。

`Boost.Math` 是一个非常快速、便携、可靠的库。

## 参见

+   Boost 的官方文档包含许多有趣的示例和教程，这些可以帮助你熟悉 `Boost.Math`；浏览到 [`www.boost.org/doc/libs/1_53_0/libs/math/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/math/doc/html/index.html)

# 编写测试用例

本食谱和下一个食谱致力于自动测试 `Boost.Test` 库，该库被许多 Boost 库使用。让我们动手实践，为我们的类编写一些测试。

```cpp
#include <stdexcept>
struct foo {
    int val_;

    operator int() const;
    bool is_not_null() const;
    void throws() const; // throws(std::logic_error)
};
```

## 准备工作

对于此食谱，需要具备 C++ 的基本知识。本食谱的代码需要链接到 `boost_unit_test_framework` 库的静态版本。

## 如何做...

说实话，Boost 中有不止一个测试库。我们将查看功能最强大的一款。

1.  要使用它，我们需要定义宏并包含以下头文件：

    ```cpp
    #define BOOST_TEST_MODULE test_module_name
    #include <boost/test/unit_test.hpp>
    ```

1.  每组测试都必须在测试用例中编写：

    ```cpp
    BOOST_AUTO_TEST_CASE(test_no_1) {
    ```

1.  检查某个函数是否返回`true`的结果如下：

    ```cpp
        foo f1 = {1}, f2 = {2};
        BOOST_CHECK(f1.is_not_null());
    ```

1.  检查不等性的实现方式如下：

    ```cpp
        BOOST_CHECK_NE(f1, f2);
    ```

1.  检查抛出异常的代码如下：

    ```cpp
        BOOST_CHECK_THROW(f1.throws(), std::logic_error);
    } // BOOST_AUTO_TEST_CASE(test_no_1)
    ```

就这样！编译和链接后，我们将得到一个可执行文件，该文件将自动测试`foo`并以人类可读的格式输出测试结果。

## 它是如何工作的...

编写单元测试很容易；你知道函数是如何工作的，以及在特定情况下它应该产生什么结果。所以你只需检查预期的结果是否与函数的实际输出相同。这就是我们在步骤 3 中所做的。我们知道`f1.is_not_null()`将返回`true`，并进行了检查。在步骤 4 中，我们知道`f1`不等于`f2`，因此也进行了检查。调用`f1.throws()`将产生`std::logic_error`异常，并检查是否抛出了预期类型的异常。

在步骤 2 中，我们正在创建一个测试用例——一组检查以验证`foo`结构的正确行为。在单个源文件中我们可以有多个测试用例。例如，如果我们添加以下代码：

```cpp
BOOST_AUTO_TEST_CASE(test_no_2) {
    foo f1 = {1}, f2 = {2};
    BOOST_REQUIRE_NE(f1, f2);
    // ...
} // BOOST_AUTO_TEST_CASE(test_no_2)
```

此代码将与`test_no_1`测试用例一起运行。传递给`BOOST_AUTO_TEST_CASE`宏的参数只是测试用例的唯一名称，在出错时会显示。

```cpp
Running 2 test cases...
main.cpp(15): error in "test_no_1": check f1.is_not_null() failed
main.cpp(17): error in "test_no_1": check f1 != f2 failed [0 == 0]
main.cpp(19): error in "test_no_1": exception std::logic_error is expected
main.cpp(24): fatal error in "test_no_2": critical check f1 != f2 failed [0 == 0]

*** 4 failures detected in test suite "test_module_name"
```

`BOOST_REQUIRE_*`和`BOOST_CHECK_*`宏之间有一个小的区别。如果`BOOST_REQUIRE_*`宏检查失败，当前测试用例的执行将停止，`Boost.Test`将运行下一个测试用例。然而，失败的`BOOST_CHECK_*`不会停止当前测试用例的执行。

步骤 1 需要额外的注意。注意`BOOST_TEST_MODULE`宏定义。这个宏必须在包含`Boost.Test`头文件之前定义，否则程序链接将失败。更多信息可以在本食谱的“也见”部分找到。

## 还有更多...

一些读者可能会想，“为什么我们在步骤 4 中写`BOOST_CHECK_NE(f1, f2)`而不是`BOOST_CHECK(f1 != f2)`？”答案很简单：步骤 4 中的宏提供了更易读和更详细的输出。

C++11 缺乏对单元测试的支持。然而，可以使用`Boost.Test`库来测试 C++11 代码。记住，你拥有的测试越多，你得到的代码就越可靠！

## 也见

+   “在一个测试模块中组合多个测试用例”食谱中包含有关测试和`BOOST_TEST_MODULE`宏的更多信息

+   请参阅 Boost 的官方文档以获取完整的测试宏列表和`Boost.Test`高级特性的信息；它可在以下链接中找到：

    [`www.boost.org/doc/libs/1_53_0/libs/test/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/index.html)

# 在一个测试模块中组合多个测试用例

编写自动测试对你的项目很有好处。但是，当项目很大并且许多开发者都在工作时，管理测试用例就变得很困难。在这个菜谱中，我们将探讨如何运行单个测试以及如何在单个模块中组合多个测试用例。

让我们假设有两个开发者正在测试`foo.hpp`头文件中声明的`foo`结构，我们希望给他们分别提供源文件来编写测试。这样，开发者就不会相互打扰，可以并行工作。然而，默认的测试运行必须执行两个开发者的测试。

## 准备工作

此菜谱需要具备基本的 C++知识。此菜谱部分重用了前一个菜谱中的代码，并且还需要链接到`boost_unit_test_framework`库的静态版本。

## 如何做到这一点...

此菜谱使用前一个菜谱中的代码。这是一个非常有用的菜谱，用于测试大型项目；不要低估它。

1.  在前一个菜谱的`main.cpp`中的所有头文件中，只留下这两行：

    ```cpp
    #define BOOST_TEST_MODULE test_module_name
    #include <boost/test/unit_test.hpp>
    ```

1.  让我们将前一个示例中的测试用例移动到两个不同的源文件中：

    ```cpp
    // developer1.cpp
    #include <boost/test/unit_test.hpp>
    #include "foo.hpp"
    BOOST_AUTO_TEST_CASE(test_no_1) {
        // ...
    }

    ///////////////////////////////////////////////////////////

    // developer2.cpp
    #include <boost/test/unit_test.hpp>
    #include "foo.hpp"
    BOOST_AUTO_TEST_CASE(test_no_2) {
        // ...
    }
    ```

就这样！因此，编译和链接所有源文件和两个测试用例将在程序执行时工作。

## 如何工作...

所有魔法都是由`BOOST_TEST_MODULE`宏完成的。如果它在`<boost/test/unit_test.hpp>`之前定义，`Boost.Test`就会认为这个源文件是主要的，并且所有辅助测试基础设施都必须放在里面。否则，只有测试宏将从`<boost/test/unit_test.hpp>`中包含。

如果你将它们与包含`BOOST_TEST_MODULE`宏的源文件链接，则会运行所有的`BOOST_AUTO_TEST_CASE`测试。当在一个大项目上工作时，每个开发者可能只启用自己的源文件的编译和链接。这给了开发者独立性，并提高了开发速度——在调试时不需要编译外部源代码和运行外部测试。

## 还有更多...

`Boost.Test`库之所以好，是因为它能够选择性地运行测试。我们可以选择要运行的测试，并将它们作为命令行参数传递。例如，以下命令将只运行`test_no_1`测试用例：

```cpp
./testing_advanced –run=test_no_1

```

以下命令将运行两个测试用例：

```cpp
./testing_advanced –run=test_no_1,test_no_2

```

不幸的是，C++11 标准没有内置的测试支持，而且看起来 C++1y 也不会采用`Boost.Test`的类和方法。

## 相关内容

+   *编写测试用例* 菜单包含有关`Boost.Test`库的更多信息。有关`Boost.Test`的更多信息，请阅读 Boost 的官方文档，网址为[`www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf.html`](http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf.html)。

+   勇敢的读者可以查看 Boost 库中的一些测试用例。这些测试用例位于`boost`文件夹中的`libs`子文件夹中。例如，`Boost.LexicalCast`测试用例位于`boost_1_53_0\libs\conversion\test`。

# 操作图像

我给你留了一些真正美味的东西作为甜点——Boost 的**通用图像库**（**GIL**），它允许你操作图像而无需过多关注图像格式。

让我们用它做一些简单而有趣的事情；让我们写一个程序，将任何图片取反。

## 准备工作

这个配方需要基本的 C++、模板和`Boost.Variant`知识。示例需要链接 PNG 库。

## 如何做...

为了简单起见，我们将只处理 PNG 图像。

1.  让我们从包含头文件开始：

    ```cpp
    #include <boost/gil/gil_all.hpp>
    #include <boost/gil/extension/io/png_dynamic_io.hpp>
    #include <string>
    ```

1.  现在我们需要定义我们希望与之工作的图像类型：

    ```cpp
        typedef boost::mpl::vector<
                boost::gil::gray8_image_t,
                boost::gil::gray16_image_t,
                boost::gil::rgb8_image_t,
                boost::gil::rgb16_image_t
        > img_types;
    ```

1.  以这种方式实现打开现有 PNG 图像：

    ```cpp
        std::string file_name(argv[1]);
        boost::gil::any_image<img_types> source;
        boost::gil::png_read_image(file_name, source);
    ```

1.  我们需要将操作应用于图片，如下所示：

    ```cpp
        boost::gil::apply_operation(
            view(source),
            negate()
        );
    ```

1.  以下代码行将帮助你写入图像：

    ```cpp
        boost::gil::png_write_view("negate_" + file_name, 
          const_view(source));
    ```

1.  让我们看看修改操作：

    ```cpp
    struct negate {
        typedef void result_type; // required

        template <class View>
        void operator()(const View& source) const {
            // ...
        }
    }; // negate
    ```

1.  `operator()`的主体包括获取通道类型：

    ```cpp
    typedef typename View::value_type value_type;
    typedef typename boost::gil::channel_type<value_type>::type 
        channel_t;
    ```

1.  它也遍历像素：

    ```cpp
    const std::size_t channels 
        = boost::gil::num_channels<View>::value;
    const channel_t max_val = (std::numeric_limits<channel_t>::max)();

    for (unsigned int y = 0; y < source.height(); ++y) {
        for (unsigned int x = 0; x < source.width(); ++x) {
            for (unsigned int c = 0; c < channels; ++c) {
                source(x, y)[c] = max_val - source(x, y)[c];
            }
        }
    }
    ```

现在我们来看看我们程序的结果：

![如何做...](img/4880OS_12_01.jpg)

上一张图片是下一张图片的负片：

![如何做...](img/4880OS_12_03.jpg)

## 它是如何工作的...

在第 2 步中，我们正在描述我们希望与之工作的图像类型。这些图像是每像素 8 位和 16 位的灰度图像以及每像素 8 位和 16 位的 RGB 图片。

`boost::gil::any_image<img_types>`类是一种`Boost.Variant`，可以持有`img_types`变量之一的图像。正如你可能已经猜到的，`boost::gil::png_read_image`将图像读取到图像变量中。

第 4 步中的`boost::gil::apply_operation`函数几乎等于`Boost.Variant`库中的`boost::apply_visitor`。注意`view(source)`的使用。`boost::gil::view`函数在图像周围构建一个轻量级包装器，将其解释为二维像素数组。

你还记得我们为`Boost.Variant`从`boost::static_visitor`派生访问者吗？当我们使用 GIL 的变体版本时，我们需要在`visitor`内部创建一个`result_type`类型定义。你可以在第 6 步中看到它。

一点理论：图像由称为像素的点组成。单个图像具有相同类型的像素。然而，不同图像的像素可能在通道数和单通道颜色位上有所不同。通道表示一种主颜色。在 RGB 图像的情况下，我们将有一个由三个通道组成的像素——红色、绿色和蓝色。在灰度图像的情况下，我们将有一个表示灰度的单个通道。

回到我们的图像。在第 2 步中，我们描述了我们希望与之工作的图像类型。在第 3 步中，其中一种图像类型从文件中读取并存储在源变量中。在第 4 步中，为所有图像类型实例化了`negate`访问者的`operator()`方法。

在第 7 步中，我们可以看到如何从图像视图中获取通道类型。

在第 8 步中，我们遍历像素和通道并将它们取反。取反是通过`max_val - source(x, y)[c]`完成的，并将结果写回图像视图。

我们在步骤 5 中写回一个图像。

## 还有更多...

C++11 没有内置处理图像的方法。

`Boost.GIL`库运行速度快且效率高。编译器对其代码进行了很好的优化，我们甚至可以使用一些`Boost.GIL`方法来帮助优化器展开循环。但本章只讨论了库的一些基础知识，所以现在是时候停止了。

## 参见

+   关于`Boost.GIL`的更多信息可以在 Boost 的官方文档中找到；请访问[`www.boost.org/doc/libs/1_53_0/libs/gil/doc/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/gil/doc/index.html)

+   参见第一章中的*在变量/容器中存储多个选定的类型*配方，以获取有关`Boost.Variant`库的更多信息
