# 第4章. 编译时技巧

在本章中，我们将涵盖：

+   在编译时检查大小

+   启用模板函数对整型类型的用法

+   禁用模板函数对真实类型的用法

+   从数字创建一个类型

+   实现一个类型特性

+   选择模板参数的最佳运算符

+   在C++03中获取表达式的类型

# 简介

在本章中，我们将看到一些基本示例，说明如何使用Boost库在编译时检查、调整算法以及在其他元编程任务中。

一些读者可能会问，“我们为什么要关心编译时的事情？”这是因为程序的发布版本只编译一次，但运行多次。我们在编译时做得越多，运行时的工作就越少，从而产生更快、更可靠的程序。只有在代码中包含检查的部分被执行时，才会执行运行时检查。编译时检查不会让你编译出一个有错误的程序。

这章可能是最重要的章节之一。没有它，理解Boost源和其他类似Boost的库是不可能的。

# 在编译时检查大小

让我们想象我们正在编写一些序列化函数，该函数将值存储在指定大小的缓冲区中：

[PRE0]

这段代码有以下问题：

+   缓冲区的大小没有被检查，所以它可能会溢出

+   这个函数可以与非平凡旧数据（POD）类型一起使用，这可能导致不正确的行为

我们可以通过添加一些断言来部分修复它，例如：

[PRE1]

但是，这是一个不好的解决方案。`BufSizeV`和`sizeof(value)`的值在编译时是已知的，因此我们可以潜在地使代码在缓冲区太小的情况下无法编译，而不是有运行时断言（如果在调试期间没有调用该函数，它可能不会触发，甚至在发布模式下可能被优化掉，所以会发生非常糟糕的事情）。

## 准备工作

这个配方需要一些关于C++模板和`Boost.Array`库的知识。

## 如何做...

让我们使用`Boost.StaticAssert`和`Boost.TypeTraits`库来纠正解决方案，输出将如下所示：

[PRE2]

## 它是如何工作的...

`BOOST_STATIC_ASSERT`宏只能在断言表达式可以在编译时评估并且隐式转换为`bool`的情况下使用。这意味着你只能在其中使用`sizeof()`、静态常量和其他常量表达式。如果断言表达式评估为`false`，`BOOST_STATIC_ASSERT`将停止我们的程序编译。在`serialization()`函数的情况下，如果第一个静态断言失败，这意味着有人为非常小的缓冲区使用了该函数，并且该代码必须由程序员修复。C++11标准有一个与Boost版本等效的`static_assert`关键字。

这里有一些更多的例子：

[PRE3]

### 注意

如果`BOOST_STATIC_ASSERT`宏的断言表达式中有逗号符号，我们必须将整个表达式用额外的括号括起来。

最后一个例子非常接近我们在`serialize()`函数的第二行看到的。所以现在是我们更多地了解`Boost.TypeTraits`库的时候了。这个库提供大量编译时元函数，允许我们获取类型信息并修改类型。元函数的使用看起来像`boost::function_name<parameters>::value`或`boost::function_name<parameters>::type`。元函数`boost::is_pod<T>::value`只有在`T`是POD类型时才会返回`true`。

让我们看看更多的例子：

[PRE4]

### 注意

一些编译器甚至可能在没有`typename`关键字的情况下编译此代码，但这种行为违反了C++标准，因此强烈建议使用`typename`。

## 更多内容...

`BOOST_STATIC_ASSSERT`宏有一个更详细的变体，称为`BOOST_STATIC_ASSSERT_MSG`，如果断言失败，它将在编译器日志（或IDE窗口）中输出错误消息。看看下面的代码：

[PRE5]

在C++11模式下，使用g++编译器编译前面的代码将给出以下结果：

[PRE6]

`BOOST_STATIC_ASSSERT`、`BOOST_STATIC_ASSSERT_MSG`以及类型特性库中的任何函数都不会产生运行时惩罚。所有这些函数都是在编译时执行的，不会在二进制文件中添加任何汇编指令。

`Boost.TypeTraits`库部分被纳入C++11标准；因此，你可能会在`std::`命名空间中的`<type_traits>`头文件中找到特性。C++11 `<type_traits>`有一些函数在`Boost.TypeTraits`中不存在，但一些元函数只在Boost中存在。当Boost和STL中有类似函数时，STL版本（在罕见情况下）可能因为编译器特定的内建函数使用而稍微好一些。

如我们之前提到的，`BOOST_STATIC_ASSERT_MSG`宏也被纳入C++11（甚至C11）作为`static_assert(expression, message)`关键字。

如果你需要跨编译器的可移植性或STL `<type_traits>`中不存在的元函数，请使用这些库的Boost版本。

## 参见

+   本章接下来的食谱将给出更多关于如何使用静态断言和类型特性的例子和想法。

+   请阅读`Boost.StaticAssert`的官方文档，以获取更多示例，链接为[http://www.boost.org/doc/libs/1_53_0/doc/html/boost_sta](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_sta)[ticassert.html](http://ticassert.html)

# 启用对整型模板函数的使用

这是一种常见的情况，当我们有一个实现了某些功能的模板类。看看下面的代码片段：

[PRE7]

在执行前面的代码之后，我们有了该类的两个额外的优化版本，一个用于整型，另一个用于实型：

[PRE8]

现在的问题是，如何让编译器自动为指定的类型选择正确的类。

## 准备工作

此配方需要了解C++模板的知识。

## 如何做到这一点...

我们将使用 `Boost.Utility` 和 `Boost.TypeTraits` 来解决这个问题：

1.  让我们从包含头文件开始：

    [PRE9]

1.  让我们在通用实现中添加一个具有默认值的额外模板参数：

    [PRE10]

1.  按照以下方式修改优化版本，这样编译器现在会将它们视为模板部分特化：

    [PRE11]

1.  就这样！现在编译器将自动选择正确的类：

    [PRE12]

## 它是如何工作的...

`boost::enable_if_c` 模板是一个有点棘手的模板。它利用了 **SFINAE（Substitution Failure Is Not An Error**） 原则，该原则在模板实例化过程中被使用。以下是该原则的工作方式：如果在函数或类模板的实例化过程中形成了无效的参数或返回类型，则实例化将从重载解析集中移除，并且不会导致编译错误。现在让我们回到解决方案，看看它是如何与传递给 `data_processor` 类的 `T` 参数的不同类型一起工作的。

如果我们将 `int` 作为 `T` 类型传递，编译器首先会尝试实例化模板部分特化，然后再使用我们的非特化（通用）版本。当它尝试实例化一个 `float` 版本时，`boost::is_float<T>::value` 元函数将返回 `false`。`boost::enable_if_c<false>::type` 元函数无法正确实例化（因为 `boost::enable_if_c<false>` 没有提供 `::type`），这就是 SFINAE 发挥作用的地方。由于类模板无法实例化，这必须被解释为不是错误，编译器将跳过这个模板特化。接下来，部分特化是针对整型类型进行优化的。`boost::is_integral<T>::value` 元函数将返回 `true`，`boost::enable_if_c<true>::type` 可以实例化，这使得可以实例化整个 `data_processor` 特化。编译器找到了匹配的部分特化，因此它不需要尝试实例化非特化方法。

现在，让我们尝试传递一些非算术类型（例如，`const char *`），看看编译器会做什么。首先编译器会尝试实例化模板部分特化。具有 `is_float<T>::value` 和 `is_integral<T>::value` 的特化将无法实例化，因此编译器将尝试实例化我们的通用版本，并且会成功。

没有使用 `boost::enable_if_c<>`，对于任何类型，所有部分特化的版本都可能同时实例化，这会导致歧义和编译失败。

### 注意

如果你使用模板并且编译器报告无法在两个模板类的方法之间进行选择，你可能需要 `boost::enable_if_c<>`。

## 还有更多...

这种方法的另一种版本被称为 `boost::enable_if`（末尾没有 `_c`）。它们之间的区别在于 `enable_if_c` 接受常量作为模板参数；然而，简短版本接受一个具有 `value` 静态成员的对象。例如，`boost::enable_if_c<boost::is_integral<T>::value >::type` 等于 `boost::enable_if<boost::is_integral<T> >::type>`。

C++11 在 `<type_traits>` 头文件中定义了 `std::enable_if`，其行为与 `boost::enable_if_c` 完全相同。它们之间没有区别，除了 Boost 的版本可以在非 C++11 编译器上工作，提供更好的可移植性。

所有启用函数仅在编译时执行，不会在运行时增加性能开销。然而，添加一个额外的模板参数可能会在 `typeid(T).name()` 中产生更大的类名，并且在某些平台上比较两个 `typeid()` 结果时可能会增加极小的性能开销。

## 参见

+   下一个示例将给出更多关于 `enable_if` 用法的示例。

+   你还可以查阅 `Boost.Utility` 的官方文档。它包含许多示例和许多有用的类（这些类在这本书中得到了广泛的应用）。请参阅[http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm)。

+   你也可以阅读一些关于模板部分特殊化的文章，请参阅[http://msdn.microsoft.com/en-us/library/3967w96f%28v=vs.110%29.aspx](http://msdn.microsoft.com/en-us/library/3967w96f%28v=vs.110%29.aspx)。

# 禁用模板函数对真实类型的用法

我们继续使用 Boost 元编程库。在前一个示例中，我们看到了如何使用 `enable_if_c` 与类一起使用，现在该看看它在模板函数中的用法了。考虑以下示例。

最初，我们有一个适用于所有可用类型的模板函数：

[PRE13]

现在我们使用 `process_data` 函数编写代码时，我们为具有 `operator +=` 函数的类型使用优化的 `process_data` 版本：

[PRE14]

但是，我们不想改变已经编写的代码；相反，只要可能，我们希望强制编译器自动使用优化函数来替代默认函数。

## 准备工作

阅读前一个示例以了解 `boost::enable_if_c` 的作用，并理解 SFINAE 的概念。然而，仍然需要了解模板知识。

## 如何做到这一点...

使用 Boost 库可以完成模板魔法。让我们看看如何做：

1.  我们将需要 `boost::has_plus_assign<T>` 元函数和 `<boost/enable_if.hpp>` 头文件：

    [PRE15]

1.  现在我们将禁用具有加法赋值运算符的类型的默认实现：

    [PRE16]

1.  为具有加法赋值运算符的类型启用优化版本：

    [PRE17]

1.  现在，用户不会感觉到差异，但优化版本将在可能的情况下被使用：

    [PRE18]

## 它是如何工作的...

`boost::disable_if_c<bool_value>::type` 元函数在 `bool_value` 等于 `true` 时禁用方法（与 `boost::enable_if_c<!bool_value>::type` 的工作方式相同）。

如果我们将一个类作为 `boost::enable_if_c` 或 `boost::disable_if_c` 的第二个参数传递，在成功评估的情况下，它将通过 `::type` 返回。

让我们逐步了解模板实例化的过程。如果我们传递 `int` 作为 `T` 类型，首先编译器将搜索具有所需签名的函数重载。因为没有这样的函数，下一步将是实例化这个函数的模板版本。例如，编译器从我们的第二个（优化）版本开始；在这种情况下，它将成功评估 `typename boost::enable_if_c<boost::has_plus_assign<T>::value, T>::type` 表达式，并将得到 `T` 返回类型。但是，编译器不会停止；它将继续实例化尝试。它将尝试实例化我们的第一个函数版本，但在评估 `typename boost::disable_if_c<boost::has_plus_assign<T>::value>` 时将失败。这个失败不会被当作错误处理（参考 SFINAE）。正如你所看到的，没有 `enable_if_c` 和 `disable_if_c`，将会有歧义。

## 还有更多...

与 `enable_if_c` 和 `enable_if` 一样，禁用函数也有 `disable_if` 版本：

[PRE19]

C++11 既没有 `disable_if_c`，也没有 `disable_if`（你可以使用 `std::enable_if<!bool_value>::type` 代替）。

如前一个食谱中提到的，所有启用和禁用函数都仅在编译时执行，不会在运行时增加性能开销。

## 参见

+   从头开始阅读这一章，以获取更多编译时技巧的示例。

+   考虑阅读 `Boost.TypeTraits` 的官方文档，以获取更多示例和元函数的完整列表。[http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html)。

+   `Boost.Utility` 库可能为你提供了更多 `boost::enable_if` 的使用示例。更多信息请参阅[http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm)。

# 从数字创建类型

我们已经看到了如何在没有使用 `boost::enable_if_c` 的情况下选择函数的例子。让我们考虑以下例子，其中我们有一个用于处理 POD 数据类型的泛型方法：

[PRE20]

此外，我们还有一个针对 1、4 和 8 字节大小的相同函数的优化版本。我们如何重写 process 函数，以便它可以调用优化版本？

## 准备工作

高度推荐至少阅读这一章的第一个食谱，这样你就不会因为这里发生的事情而感到困惑。模板和元编程不应该让你感到害怕（或者准备好看到很多它们）。

## 如何做到这一点...

我们将看到如何将模板类型的尺寸转换为某种类型的变量，以及如何使用该变量来推断正确的函数重载。

1.  让我们定义 `process_impl` 函数的通用和优化版本：

    [PRE21]

1.  现在，我们已经准备好编写过程函数：

    [PRE22]

## 它是如何工作的...

这里最有趣的部分是 `boost::mpl::int_<sizeof(T)>(). sizeof(T)` 在编译时执行，因此其输出可以用作模板参数。`boost::mpl::int_<>` 类只是一个空的类，它持有整型类型的编译时值（在 `Boost.MPL` 库中，这样的类被称为积分常量）。它可以像以下代码所示实现：

[PRE23]

我们需要一个此类实例，这就是为什么我们在 `boost::mpl::int_<sizeof(T)>()` 的末尾有一个圆括号。

现在，让我们更详细地看看编译器将如何决定使用哪个 `process_impl` 函数。首先，编译器将尝试匹配具有第二个参数而不是模板的函数。如果 `sizeof(T)` 是 4，编译器将尝试搜索具有类似 `process_impl(T, boost::mpl::int_<8>)` 签名的函数，并找到来自 `detail` 命名空间的 4 字节优化版本。如果 `sizeof(T)` 是 34，编译器将找不到具有类似 `process_impl(T, boost::mpl::int_<34>)` 签名的函数，并将使用模板变体 `process_impl(const T& val, Tag /*ignore*/)`。

## 还有更多...

`Boost.MPL` 库有几个元编程的数据结构。在这个菜谱中，我们只是触及了冰山一角。您可能会发现以下来自 MPL 的积分常量类很有用：

+   `bool_`

+   `int_`

+   `long_`

+   `size_t`

+   `char_`

所有的 `Boost.MPL` 函数（除了 `for_each` 运行时函数）都是在编译时执行的，不会增加运行时开销。`Boost.MPL` 库不是 C++11 的一部分，但许多 STL 库为了满足自己的需求，实现了它的一些函数。

## 参见

+   第 8 章 [Metaprogramming](ch08.html "第 8 章。元编程") 中的菜谱将为您提供更多 `Boost.MPL` 库使用的示例。如果您有信心，您也可以尝试阅读其文档，[http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html)。

+   在 [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/fill.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/fill.html) 和 [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/copy.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/copy.html) 上阅读更多关于标签使用的示例。

# 实现类型特性

我们需要实现一个类型特性，当它作为模板参数传递 `std::vector` 类型时，返回 true。

## 准备工作

需要一些关于 `Boost.TypeTrait` 或 STL 类型特性的基本知识。

## 如何实现...

让我们看看如何实现一个类型特性：

[PRE24]

## 它是如何工作的...

几乎所有的工作都是由 `boost::true_type` 和 `boost::false_type` 类完成的。`boost::true_type` 类中有一个布尔 `::value` 静态常量，其值等于 `true`，而 `boost::false_type` 类中有一个布尔 `::value` 静态常量，其值等于 `false`。它们还有一些 typedef，通常是从 `boost::mpl::integral_c` 派生出来的，这使得使用从 `true_type/false_type` 派生的类型与 `Boost.MPL` 一起使用变得容易。

我们第一个 `is_stdvector` 结构是一个通用的结构，当找不到此类结构的模板特化版本时，总是会被使用。我们的第二个 `is_stdvector` 结构是 `std::vector` 类型的模板特化（注意，它是从 `true_type` 派生出来的！）所以，当我们向 `is_stdvector` 结构传递向量类型时，将使用模板特化版本，否则将使用通用版本，它是从 `false_type` 派生出来的。

### 注意

3 行 在我们的特性中，`boost::false_type` 和 `boost::true_type` 前面没有公共关键字，因为我们使用了 `struct` 关键字，并且默认使用公共继承。

## 更多...

那些使用与 C++11 兼容的编译器的读者可以使用 `std::` 命名空间中声明的 `<type_traits>` 头文件中的 `true_type` 和 `false_type` 类型来创建他们自己的类型特性。

如同往常，Boost 版本更具有可移植性，因为它可以在 C++03 编译器上使用。

## 参见

+   本章中的几乎所有配方都使用了类型特性。有关更多示例和信息，请参阅 [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/i](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/i)[ndex.html](http://ndex.html)。

# 选择模板参数的最佳运算符

想象一下，我们正在使用来自不同供应商的类，这些类实现了不同数量的算术运算，并且具有从整数构造函数。我们确实想编写一个函数，当传递任何类给它时，它会增加一个。我们还希望这个函数是有效的！看看下面的代码：

[PRE25]

## 准备工作

需要一些关于 C++ 模板和 `Boost.TypeTrait` 或 STL 类型特性的基本知识。

## 如何做...

所有选择都可以在编译时完成。这可以通过使用 `Boost.TypeTraits` 库来实现，如下面的步骤所示：

1.  让我们从制作正确的函数对象开始：

    [PRE26]

1.  之后我们将需要一系列类型特性：

    [PRE27]

1.  然后，我们就准备好推导出正确的函数对象并使用它：

    [PRE28]

## 它是如何工作的...

所有魔法都是通过 `conditional<bool Condition, class T1, class T2>` 元函数完成的。当这个元函数接受 `true` 作为第一个参数时，它通过 `::type` typedef 返回 `T1`。当 `boost::conditional` 元函数接受 `false` 作为第一个参数时，它通过 `::type` typedef 返回 `T2`。它就像某种编译时 `if` 语句。

因此，`step0_t`包含一个`detail::plus_functor`元函数，而`step1_t`将包含`step0_t`或`detail::plus_assignable_functor`。`step2_t`类型将包含`step1_t`或`detail::post_inc_functor`。`step3_t`类型将包含`step2_t`或`detail::pre_inc_functor`。每个`step*_t`类型定义包含的内容是通过类型特性推导得出的。

## 还有更多...

这个函数有一个C++11版本，可以在`std::`命名空间中的`<type_traits>`头文件中找到。Boost在不同的库中有这个函数的多个版本，例如，`Boost.MPL`有`boost::mpl::if_c`函数，它的工作方式与`boost::conditional`完全相同。它还有一个版本`boost::mpl::if_`（没有`c`结尾），它将为第一个模板参数调用`::type`；如果它派生自`boost::true_type`（或是一个`boost::true_type`类型），在`::type`调用期间将返回其第二个参数，否则将返回最后一个模板参数。我们可以将我们的`inc()`函数重写为使用`Boost.MPL`，如下面的代码所示：

[PRE29]

## 参考阅读

+   食谱*启用模板函数对整型类型的用法*

+   食谱*禁用模板函数对真实类型的用法*

+   `Boost.TypeTraits`文档有一个完整的可用元函数列表。请阅读它，网址为[http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html)。

+   [第8章](ch08.html "第8章。元编程")中的*元编程*食谱将为你提供更多`Boost.MPL`库使用的示例。如果你感到自信，你也可以尝试阅读其文档，网址为[http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html)。

+   有一个提议要为C++添加类型切换，你可能对此感兴趣。请阅读它，网址为[http://www.stroustrup.com/OOPSLA-ty](http://www.stroustrup.com/OOPSLA-ty)[peswitch-draft.pdf](http://peswitch-draft.pdf)。

# 在C++03中获取表达式的类型

在之前的食谱中，我们看到了一些关于`boost::bind`使用的示例。这是一个好用的工具，但有一个小缺点；在C++03中很难将`boost::bind`元函数的仿函数作为变量存储。

[PRE30]

在C++11中，我们可以使用`auto`关键字代替`???`，并且这会起作用。在C++03中有没有办法做到这一点？

## 准备工作

C++11的`auto`和`decltype`关键字的知识可能有助于你理解这个食谱。

## 如何做到这一点...

我们需要一个`Boost.Typeof`库来获取表达式的返回类型：

[PRE31]

## 它是如何工作的...

它只是创建了一个名为`var`的变量，并将表达式的值作为第二个参数传递。`var`的类型由表达式的类型检测得出。

## 还有更多...

经验丰富的C++11读者会注意到，新标准中有更多关键字用于检测表达式类型。也许`Boost.Typeof`也有针对它们的宏。让我们看看以下C++11代码：

[PRE32]

使用 `Boost.Typeof`，前面的代码可以写成以下形式：

[PRE33]

C++11 版本的 `decltype(expr)` 会推导并返回 `expr` 的类型。

[PRE34]

使用 `Boost.Typeof`，前面的代码可以写成以下形式：

[PRE35]

### 注意

C++11 在函数声明末尾有特殊的语法来指定返回类型。不幸的是，这在 C++03 中无法模拟，所以我们不能在宏中使用 `t1` 和 `t2` 变量。

你可以在模板和任何其他编译时表达式中自由使用 `BOOST_TYPEOF()` 函数的结果：

[PRE36]

但不幸的是，这种魔法并不总是不需要帮助就能工作。例如，用户定义的类并不总是会被检测到，因此以下代码在某些编译器上可能会失败：

[PRE37]

在这种情况下，你可以给 `Boost.Typeof` 提供帮助，并注册一个模板：

[PRE38]

然而，三个最受欢迎的编译器即使在没有 `BOOST_TYPEOF_REGISTER_TEMPLATE` 和没有 C++11 的情况下也能正确检测类型。

## 参见

+   `Boost.Typeof` 的官方文档中有更多示例。有关信息请参阅[http://www.boost.org/doc/libs/1_53_0/doc/html/typeof.html](http://www.boost.org/doc/libs/1_53_0/doc/html/typeof.html)。

+   *Bjarne Stroustrup* 可能会向你介绍一些 C++11 的特性。有关信息请参阅[http://www.stroustrup.com/C++11FAQ.html](http://www.stroustrup.com/C++11FAQ.html)。
