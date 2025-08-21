# 第二章：Boost 实用程序的初次接触

在本书的过程中，我们将专注于处理不同子系统的多个 Boost 库，例如文件系统、线程、网络 I/O 和各种容器等。在每一章中，我们将深入探讨一些这样的库的细节。这一章不同之处在于，我们将挑选一些有用和多样化的技巧，这些技巧几乎可以帮助您解决所有编程情况。为此，我们为我们列出了以下主题：

+   简单数据结构

+   处理异构值

+   处理命令行参数

+   其他实用程序和编译时检查

这是一个厨房水槽章节，您可以不断回来查找一个在手头问题上似乎适用的有趣技术。

# 简单数据结构

在本节中，我们将介绍两个不同的库，它们将帮助您创建立即有用的简单数据结构：Boost.Optional 和 Boost.Tuple。Boost.Optional 可用于表示可选值；可能存在也可能不存在的对象。Boost.Tuple 用于创建异构值的有序集合。

## Boost.Optional

让我们假设您需要在数据存储中维护有关音乐家的信息。除其他事项外，您可以查找艺术家发布的最新专辑。您已经用 C++编写了一个简单的 API 来实现这一点：

```cpp
std::string find_latest_album_of(const std::string& artisteName);
```

为简单起见，我们将忽略两个或更多艺术家可能共享相同名称的可能性。以下是这个函数的一个简单实现：

```cpp
 1 #include <string>
 2 #include <map>
 3
 4 typedef std::map<std::string, std::string> artiste_album_map;
 5
 6 extern artiste_album_map latest_albums;
 7
 8 std::string find_latest_album_of(
 9                     const std::string& artiste_name) {
10   auto iter = latest_albums.find(artiste_name);
11
12   if (iter != latest_albums.end()) {
13     return iter->second;
14   } else {
15     return "";
16   }
17 }
```

我们在一个名为`latest_albums`的映射中存储了艺术家的名字和他们的最新专辑。`find_latest_album_of`函数接受一个艺术家的名字，并使用`std::map`的`find`成员函数来查找最新专辑。如果找不到条目，它会返回一个空字符串。现在，有些艺术家可能还没有发布专辑。对于这种情况返回一个空字符串似乎是合理的，直到你意识到音乐家有他们独特的怪癖，有时会发布没有名字的专辑。那么，你如何区分音乐家尚未发布专辑的情况和音乐家最新专辑没有标题的情况？在一种情况下，没有值可返回，而在另一种情况下，它是一个空字符串。

`boost::optional<T>`模板可用于表示可选值；可能存在也可能不存在的值。在这种情况下，它是为我们的问题量身定制的。要表示可能存在也可能不存在的`std::string`值，您可以使用`boost::optional<std::string>`。我们可以使用`boost::optional`重写`find_latest_album_of`函数，如下面的代码列表所示：

**列表 2.1：使用 Boost.Optional**

```cpp
 1 #include <string>
 2 #include <map>
 3 #include <boost/optional.hpp>
 4
 5 typedef std::map<std::string, std::string> artiste_album_map;
 6
 7 extern artiste_album_map latest_albums;
 8 
 9 boost::optional<std::string> find_latest_album_of(
10                             const std::string& artiste_name) {
11   auto iter = latest_albums.find(artiste_name);
12
13   if (iter != latest_albums.end()) {
14     return iter->second;
15   } else {
16     return boost::none;
17   }
18 }
```

我们简单地返回找到的值（第 14 行），它会自动包装在`boost::optional`容器中。如果没有值可返回，我们返回一个特殊对象`boost::none`（第 16 行）。这会导致返回一个空的`boost::optional`对象。使用`boost::optional`的代码正是我们需要的；它检查容器中是否存在一个键，并返回值，或指示它不存在，而没有任何歧义（即空与无标题）。

### 提示

`boost::optional`的默认初始化实例始终为空。如果存储在`boost::optional`中的值是可移动的（参见附录，*C++11 语言特性模拟*），包装器`optional`对象也是可移动的。如果存储的值是可复制的，包装器`optional`对象也是可复制的。

我们可以将列表 2.1 中的查找函数泛化到任何具有类似映射或字典接口的容器中，如下所示：

**列表 2.2：使用可选项进行通用查找**

```cpp
 1 #include <boost/optional.hpp>
 2
 3 template <typename C>
 4 boost::optional<typename C::mapped_type>
 5 lookup(const C& dict, const typename C::key_type& key)
 6 {
 7   typename C::const_iterator it = dict.find(key);
 8   if (it != dict.end()) {
 9     return it->second;
10   } else {
11     return boost::none;
12   }
13 }
```

在前面的代码中，我们已将`lookup`转换为函数模板，可以在任何`map`、`multimap`、它们的无序变体或任何其他非标准容器上调用，暴露类似的接口。它是基于容器类型`C`进行参数化的。容器类型`C`必须具有嵌套类型定义：`key_type`和`mapped_type`，对应于地图存储的键和值的类型；这是标准库中`std:map`和其他关联容器满足的约束。

`typename`关键字的使用（第 4、5、7 行）可能需要一些解释。如果我们从这些行中省略`typename`关键字，编译器将无法识别`C::mapped_type`、`C::key_type`和`C::const_iterator`作为类型的名称。因为`mapped_type`、`key_type`和`const_iterator`是依赖于类型模板参数`C`的名称，所以需要告诉编译器它们标识类型。我们使用`typename`关键字来做到这一点。

### 访问存储在 boost::optional 中的值

您可以检查`optional`对象是否包含值或为空，并提取非空`optional`对象中存储的值：

```cpp
 1 std::string artiste("Korn");
 2 boost::optional<std::string> album = 
 3                             find_latest_album_of(artiste);
 4 if (album) {
 5   std::cout << "The last album from " << artiste;
 6
 7   if (album->empty()) {
 8     std::cout << " is untitled\n";
 9   } else {
10     std::cout << " is named " << *album << '\n';
11   }
12 } else {
13   std::cout << "No information on albums from " 
14             << artiste << '\n';
15 }
```

在调用`find_latest_album_of`的代码中，为了测试返回的值是否为空，我们在布尔上下文中调用对象（第 4 行）。如果评估为`true`，这意味着`album`不为空。如果它有一个值，我们可以使用重载的`operator*`（第 10 行）获得对包含值的引用。我们可以使用重载的`operator->`访问底层对象的成员；在这种情况下，我们调用`std::string`的空成员函数（第 7 行）。我们还可以使用非空`boost::optional`对象的`get`成员函数来访问存储的值，而不是使用重载的`operator*`。通过调用`operator*`、`get`或`operator->`对空的可选值进行解引用会导致运行时错误，这就是为什么我们首先检查`optional`对象是否为空，然后再尝试对其进行解引用。

### get_value_or

使用`optional`，我们指示专辑可能有也可能没有值。但有时我们需要使用应该接受可选值但没有的 API。在这种情况下，我们可能希望返回带有一些默认值的空值。想象一下，问巴黎居民他们最喜欢的城市，对于那些没有回答的人，巴黎将被用作默认最爱：

```cpp
 1 void printFavoriteCity(const std::string& name,
 2                        const std::string& city)
 3 {
 4   std::cout << name "'s favorite city is " << city << '\n';
 5 }
 6
 7 boost::optional<std::string> getFavoriteCity(
 8                           const std::string& resident_id);
 9 ...
10 std::string resident = "Serge";
11 boost::optional<std::string> fav_city = 
12                                     getFavoriteCity(resident);
13
14 printFavoriteCity(fav_city.get_value_or("Paris"));
```

如果想象中的`getFavoriteCity`函数返回一个空值，我们希望将`Paris`传递给`printFavoriteCity`函数。我们使用`get_value_or`成员函数来实现这一点（第 14 行）。

### Boost.Optional 与指针

如果我们没有使用`optional`，那么`find_last_album_of`或`lookup`函数会返回什么来指示没有找到值？它们要么需要返回指向动态分配对象的指针，要么在没有找到值时返回`nullptr`。除了使用动态内存，这还要求调用函数管理返回的动态分配对象的生命周期。这种情况可以通过智能指针（第三章，“内存管理和异常安全性”）来缓解，但它并不能消除昂贵的自由存储分配。`boost::optional`类消除了自由存储分配，并将封装的对象存储在其布局中。此外，它存储一个布尔标志来跟踪它是否被初始化。

## Boost.Tuple

Boost Tuples 是一种将不同类型的数据组合成有序元组并传递它们的很酷的方法。结构也可以做同样的事情，但元组有一些特殊之处：

+   您可以编写通用代码来操作各种元组，例如打印它们的所有成员并比较两个元组的结构和类型是否相似。

+   每个新的结构或类在您的软件中定义了一个新的类型。类型应该表示接口和行为。用类型表示数据的每个临时聚集会导致类型的泛滥，这些类型在问题空间或其抽象中没有意义。

Boost Tuple 是一个非常有用的库，它可以帮助您方便地创建用于一起移动相关数据的模式，例如在函数之间交换数据。Boost Tuples 是`std::pair`的泛化，用于创建 2 元组。

### 提示

如果您正在使用支持良好的 C++11 的 C++编译器，应该使用标准库中的`std::tuple`工具，这是 C++11 标准中包含的 Boost 库之一。需要包含的头文件是`<tuple>`。我们在这里讨论的大部分内容都适用于`std::tuple`。

### 创建元组

让我们看一个例子。给定不同时间点的股票价格系列，我们想要找出买入和卖出股票以最大化利润的最佳两个时间点。我们可以假设没有卖空的选项，也就是说，必须先买入再卖出。为简单起见，可以假定输入是一个双精度浮点数的向量。在这个向量中，我们对表示最佳买入和卖出股票的索引对感兴趣，以最大化利润：

**清单 2.3：使用元组**

```cpp
 1 #include <boost/tuple/tuple.hpp>
 2 #include <vector>
 3
 4 boost::tuple<size_t, size_t, double>
 5      getBestTransactDays(std::vector<double> prices)
 6 {
 7   double min = std::numeric_limits<double>::max();
 8   double gain = 0.0, max_gain = 0.0;
 9   size_t min_day, max_day;
10   size_t buy_day;
11   for (size_t i = 0, days = prices.size(); i < days; ++i) {
12     if (prices[i] < min) {
13       min = prices[i];
14       min_day = i;
15     } else if ((gain = prices[i] - min) > max_gain) {
16       max_gain = gain;
17       buy_day = min_day;
18       max_day = i;
19     }
20   }
21
22   return boost::make_tuple(buy_day, max_day, max_gain);
23 }
```

函数`getBestTransactDays`返回一个包含两个无符号整数（`size_t`）和一个双精度浮点数（第 4 行）的元组，表示买入和卖出股票的最大利润的两个索引，以及可能的最大利润。函数的返回类型是`boost::tuple<size_t, size_t, double>`。头文件`boost/tuple/tuple.hpp`提供了处理元组所需的函数和类型（第 1 行）。

函数`getBestTransactDays`实现了一个简单的线性算法，通过遍历向量，跟踪到目前为止看到的最低股价。如果当前元素的值小于目前为止的最低股价，则将其设置为新的最低价，并记录其索引（第 12-14 行）。该函数还跟踪最大收益，即到目前为止记录的价格差的最大值。如果我们遇到一个与最低价的差值高于最大收益的元素，则将此差值记录为新的最大收益（第 15 行），并记录实现此收益所需的交易日（第 16-18 行）。

我们使用`boost::make_tuple`（第 22 行）创建元组，这是一个方便的函数，用于从其元素创建元组，而无需显式模板实例化。您也可以在第 22 行的位置创建并返回一个元组，如下所示：

```cpp
22 boost::tuple<size_t, size_t, double> best_buy(buy_day, max_day, 
23                                         max_gain);
24 return best_buy;
```

正如您所看到的，`boost::make_tuple`更加紧凑，并且作为一个函数模板，它会自动解析其参数的类型，以创建正确类型的元组。这是一个经常见到的模式，您可以使用工厂函数模板来实例化类模板，从而自动检测类型。

### 访问元组元素

有几种方法可以访问元组中的元素。看一下调用`getBestTransactDays`函数的以下示例：

```cpp
 1 std::vector<double> stockPrices;
 2 ...
 3 boost::tuple<size_t, size_t, double> best_buy = 
 4                              getBestTransactDays(stockPrices);
 5 
 6 size_t buyDay = boost::get<0>(best_buy);  // Access 0th element
 7 size_t sellDay = boost::get<1>(best_buy); // Access 1st element
 8 double profit = boost::get<2>(best_buy); // Access 2nd element
```

我们还可以使用`boost::tie`将元组中的元素解包到单独的变量中：

```cpp
 1 size_t buyDay, sellDay;
 2 double profit;
 3 boost::tie(buyDay, sellDay, profit) =  
 4                 getBestTransactDays(stockPrices);
```

上一行代码将把元组的第一个元素分配给`buyDay`，第二个元素分配给`sellDay`，第三个元素分配给`profit`。如果我们只对元组中的部分元素感兴趣，可以使用`boost::tuples::ignore`忽略其他元素。以下是相同的示例，但这次我们使用`boost::tuples::ignore`忽略了`sellDay`：

```cpp
 1 size_t buyDay, sellDay;
 2 boost::tie(buyDay, sellDay, boost::tuples::ignore) =
 3                              getBestTransactDays(stockPrices);
```

### 比较元组

相同长度的元组可以使用关系运算符进行比较，例如`==`，`<`，`>`，`<=`和`>=`。在任何这样的比较中，将比较每个位置上的对应元素。对应位置上的元素的类型不需要完全相同；它们只需要能够使用相关的关系运算符进行比较即可：

```cpp
 1 boost::tuple<int, int, std::string> t1 = 
 2                          boost::make_tuple(1, 2, "Hello");
 3 boost::tuple<double, double, const char*> t2 = 
 4                         boost::make_tuple(1, 2, "Hi");
 5 assert(t1 < t2);   // because Hello < Hi
```

请注意，元组`t1`和`t2`中的实际类型不同，但两者长度相同，并且相应位置的元素可以相互比较。通常，比较会在决定比较结果的第一对元素处停止。在这个例子中，所有三个元素都被比较，因为前两个元素相等。

```cpp
 1 boost::tuple<int, int, std::string> t1 = 
 2                          boost::make_tuple(1, 20, "Hello");
 3 boost::tuple<double, double, const char*> t2 = 
 4                        boost::make_tuple(1, 2, "Hi");
 5 assert(t1 > t2);    // because 20 > 2
```

以下代码用于定义具有非常少代码的结构的关系运算符：

```cpp
 1 struct my_type {
 2   int a;
 3   double b;
 4   char c;
 5 };
 6
 7 bool operator<(const my_type& left, const my_type& right) {
 8   return boost::make_tuple(left.a, left.b, left.c) <
 9                 boost::make_tuple(right.a, right.b, right.c);
10 }
```

### 使用元组编写通用代码

现在我们将编写一个通用函数来查找元组中元素的数量：

```cpp
 1 template <typename T>
 2 size_t tuple_length(const T&) {
 3   return boost::tuples::length<T>::value;
 4 }
```

这个函数简单地使用`boost::tuples::length<T>`元函数来计算元组中元素的数量。这个计算是在编译时进行的。**元函数**只是一个类模板，它具有从其模板参数在编译时计算出的可访问的静态成员或嵌套类型（参见第七章，“高阶和编译时编程”，有一个更严格的定义）。在这种情况下，`boost::tuples::length<T>`元函数有一个名为`value`的公共静态成员，它被计算为元组`T`中的元素数量。如果您使用标准库中的元组，应该使用`std::tuple_size<T>`而不是`boost::tuples::length<T>`。这只是一个使用元函数和类型计算的通用编程的小例子。

# 使用异构值

在程序的生命周期中需要一个可以在不同时间点容纳不同类型数据的值并不是什么新鲜事。C++支持 C 的`union`构造，它本质上允许您拥有一个单一类型，可以在不同时间点假定不同底层 POD 类型的值。**POD**或**Plain Old Data**类型，粗略地说，是不需要任何特殊初始化、销毁和复制步骤的类型，其语义等效物可以通过逐字复制其内存布局来创建。

这些限制意味着大多数 C++类，包括大多数标准库中的类，永远不能成为联合的一部分。从 C++11 开始，对联合的这些限制有所放宽，现在可以在联合中存储具有非平凡构造、销毁和复制语义（即非 POD 类型）的对象。但是，存储在联合中的这些对象的生命周期管理不是自动的，可能会很麻烦，因此最好避免。

来自 Boost 的两个库，Variant 和 Any，提供了有用的变体类型，提供了与联合相同的功能，但没有许多限制。使用 Variants 和 Any，在标准库容器中存储异构数据变得非常容易和无误。这些库代表了可辨别的联合类型。各种类型的值可以存储在可辨别的联合中，并且类型信息与值一起存储。

除了存储异构类型的数据，我们经常需要在不同表示之间进行转换，例如，文本到数字的转换以及反之。Boost Conversion 提供了一种无缝转换类型的方法，其中包括使用统一的语法进行类型转换。我们将在以下部分中查看 Any、Variant 和 Conversion 库。

## Boost.Variant

Boost Variant 避免了 C++联合的所有问题，并提供了一个类似联合的构造，定义在一组任意类型上，而不仅仅是 POD 类型。我们可以使用 Boost Variant 头文件库定义一个变体数据类型，通过使用`boost::variant`模板实例化一个类型列表。类型列表标识了变体对象在不同时间点可以假定的不同类型的值。列表中的不同类型可以是多样的和不相关的，只需满足一个绑定条件——即每个类型都是可复制的或至少可移动的。甚至可以创建包含其他变体的变体。

在我们的第一个示例中，我们创建了一个整数、一个`std::string`和两个用户定义类型`Foo`和`Bar`的变体。通过这个例子，我们说明了创建变体类型的约束以及可以对这种变体值执行的操作：

第 2.4 节：创建和使用变体

```cpp
 1 #include <boost/variant.hpp>
 2 #include <string>
 3
 4 struct Foo {
 5   Foo(int n = 0) : id_(n) {} // int convertible to Foo
 6 private:
 7   int id_;
 8 };
 9 
10 struct Bar {
11   Bar(int n = 0) : id_(n) {} // int convertible to Bar
12 private:
13   int id_;
14 };  
15 
16 int main()
17 {
18   boost::variant<Foo, int, std::string> value; // error if Foo 
19                                 // not be default constructible
20   boost::variant<std::string, Foo, Bar> value2;
21 
22   value = 1;                 // sets int, not Foo
23   int *pi = boost::get<int>(&value);
24   assert(pi != 0);
25   value = "foo";             // sets std::string
26   value = Foo(42);           // sets Foo
27
28   // value2 = 1;             // ERROR: ambiguous - Foo or Bar?
29   // std::cout << value << ' ' << value2 << '\n'; // ERROR:
30                   // Foo, Bar cannot be streamed to ostream
31 }
```

我们创建了两个基本类型：`Foo`（第 4 行）和`Bar`（第 10 行）；我们可以从`int`隐式初始化两者。我们定义了一个名为`value`的变体（第 18 行），它包含三种类型，`Foo`、`int`和`std::string`。第二个变体，`value2`（第 20 行），定义为`std::string`、`Foo`和`Bar`。

默认情况下，每个变体实例都被值初始化为其第一个类型的对象。因此，`value`被默认构造为`Foo`实例——在变体的类型参数列表中的第一个类型。同样，`value2`被默认构造为`std::string`——在其类型参数列表中的第一个类型。如果第一个类型是 POD 类型，它将被零初始化。因此，第一个类型必须是默认可构造的，变体才能是默认可构造的。

我们将一个整数赋给`value`（第 22 行）。这将使它成为`int`而不是`Foo`，因为整数可以隐式转换为`Foo`。我们使用`boost::get<T>`函数模板在`value`的地址上使用`T=int`（第 23 行）进行确认，并确认它不是空指针（第 24 行）。

我们将`const char*`赋给`value`（第 25 行），它隐式转换为`std::string`，并存储在`value`中，覆盖了先前存储的整数值。接下来，我们分配了一个`Foo`对象（第 26 行），它覆盖了先前的`std::string`值。

如果我们尝试将整数分配给`value2`（第 28 行，已注释），它将导致编译错误。变量`value2`被定义为`std::string`、`Foo`和`Bar`的变体，整数可以隐式转换为`Foo`或`Bar`，但都不是更好的选择，因此会导致歧义，编译器会抛出错误。通常情况下，变体的初始化和赋值不应该导致对变体中要实例化的类型产生歧义。

如果我们尝试将`value`的内容流式传输到`std::cout`（第 29 行，已注释），那么同样，我们将遇到编译错误。这是因为变体支持的类型之一（`Foo`）不可*流式传输*，这意味着它不能使用插入运算符（`<<`）写入到`ostreams`中。

### 访问变体中的值

我们使用`boost::get<T>`函数模板来访问变体中类型为`T`的值，其中`T`是我们想要的具体类型的值。当在变体引用上调用此函数时，如果存储的值不是指定类型，则返回对存储值的引用，或抛出`boost::bad_get`异常。当在指向变体的指针上调用时，如果存储的值不是指定类型，则返回存储值的地址，如果存储的值不是指定类型，则返回空指针。后者的行为可以用来测试变体是否存储了特定类型的值，就像在列表 2.4（第 23 行）中使用的那样。`get<>`的这种行为与`dynamic_cast`的行为非常相似：

第 2.5 节：访问变体中的值

```cpp
 1 #include <boost/variant.hpp>
 2 #include <string>
 3 #include <cassert>
 4 
 5 int main() {
 6   boost::variant<std::string, int> v1;
 7   v1 = "19937";                    // sets string
 8   int i1;
 9 
10   try {    
11     i1 = boost::get<int>(v1);      // will fail, throw
12   } catch (std::exception& e) {
13     std::cerr << e.what() << '\n';
14   }
15 
16   int *pi = boost::get<int>(&v1);  // will return null
17   assert(pi == 0);
18 
19   size_t index = v1.which();        // returns 0
20 }
```

在前面的代码中，我们创建了一个变体`v1`，可以存储`std::string`或`int`值（第 6 行）。我们将`v1`设置为字符串`"19937"`（第 7 行）。我们使用`boost::get<int>`函数尝试从`v1`中获取整数（第 11 行），但由于此时`v1`存储的是一个字符串，所以会抛出异常。接下来，我们使用`boost::get<int>`的指针重载，该重载获取变体`v1`的地址。如果其类型与通过`get`函数的模板参数请求的类型匹配，则返回存储值的指针。如果不匹配，就像在这种情况下一样，将返回空指针（第 16 和 17 行）。最后，我们可以通过调用`which`成员函数获取当前存储在变体中的值的类型的从零开始的索引。由于`v1`包含`std::string`，并且`v1`的声明类型是`boost::variant<std::string, int>`，因此`v1.which()`应该返回变体声明中`std::string`的索引——在这种情况下是 0（第 19 行）。

#### 编译时访问

变体中存储的值如何被消耗通常取决于值的类型。使用 if-else 梯子检查变体的每种可能类型可能会迅速加剧代码的可读性和可维护性。当然，我们可以使用变体的`which`成员方法找出当前值的类型的从零开始的索引，但这对我们目前没有什么用。相反，我们将看一下 Boost Variant 库提供的非常优雅和多功能的编译时访问机制，没有这个机制，处理变体将会相当麻烦。

这个想法是创建一个访问者类，其中包含一个重载的函数调用运算符（`operator()`），用于处理可能存储在变体中的每种类型。使用函数`boost::apply_visitor`，我们可以根据它包含的值的类型在变体对象上调用访问者类中的适当重载。

访问者类应该公开继承`boost::static_visitor<T>`模板，其中`T`是重载的函数调用运算符的返回类型。默认情况下，`T`是`void`。让我们看一个例子：

**清单 2.6：变体的编译时访问**

```cpp
 1 #include <boost/variant.hpp>
 2 
 3 struct SimpleVariantVisitor :public boost::static_visitor<void>
 4 {
 5   void operator() (const std::string& s) const
 6   { std::cout << "String: " << s << '\n'; }
 7 
 8   void operator() (long n) const
 9   { std::cout << "long: " << n << '\n'; }
10 };
11 
12 int main()
13 {
14   boost::variant<std::string, long, double> v1;
15   v1 = 993.3773;
16 
17   boost::apply_visitor(SimpleVariantVisitor(), v1);
18 }
```

我们创建了一个类型为`std::string`、`long`和`double`的变体称为`v1`（第 14 行）。我们将其设置为类型为`double`的值（第 15 行）。最后，我们在`v1`上调用类型为`SimpleVariantVistor`的访问者（第 17 行）。`SimpleVariantVisitor`继承自`boost::apply_visitor<void>`（第 3 行），并包含`std::string`（第 5 行）和`long`（第 8 行）的重载，但没有`double`的重载。每个重载都将其参数打印到标准输出。

重载的解析发生在编译时而不是运行时。因此，每种值类型的重载必须可用。如果其参数类型是最适合变体中存储的值类型的类型，则会调用特定的重载。此外，如果所有类型都可以转换为重载的参数类型，则单个重载可以处理多种类型。

有趣的是，在前面的例子中，没有`double`的重载可用。然而，允许缩小转换，并且使用`long`的重载进行潜在的缩小。在这种情况下，`long`的重载处理`long`和`double`类型。另一方面，如果我们有`double`和`long`的单独重载，但没有`std::string`的重载，我们将会遇到编译错误。这是因为从`std::string`到`long`或`double`甚至没有缩小转换可用，重载解析将失败。作为编译时机制，这与变体对象中实际存储的值的类型无关。

#### 通用访问者

您可以创建一个处理一系列类型的成员函数模板。在处理不同类型的代码没有显着差异的情况下，可能有意义使用这样的成员模板。以下是一个打印变体内容的访问者的示例：

**清单 2.7：通用的编译时访问**

```cpp
 1 #include <boost/variant.hpp>
 2
 3 struct PrintVisitor : boost::static_visitor<>
 4 {
 5    template <typename T>
 6    void operator() (const T& t) const {
 7      std::cout << t << '\n';
 8    }
 9 };
10
11 boost::variant<std::string, double, long, Foo> v1;
12 boost::apply_visitor(PrintVisitor(), v1);
```

在上述代码中，我们定义了一个类型为`std::string`、`double`、`long`和`Foo`的变体。访问者类`PrintVisitor`包含一个通用的`operator()`。只要变体中的所有类型都是*可流化的*，这段代码就会编译并将变体的值打印到标准输出。

#### 将访问者应用于容器中的变体

通常，我们有一个包含变体对象的 STL 容器，并且我们希望使用我们的访问者访问每个对象。我们可以利用`std::for_each` STL 算法和`boost::apply_visitor`的单参数重载来实现这一目的。`boost::apply_visitor`的单参数重载接受一个访问者实例，并返回一个将访问者应用于传递的元素的函数对象。以下示例最好说明了用法：

```cpp
 1 #include <boost/variant.hpp>
 2
 3 std::vector<boost::variant<std::string, double, long> > vvec;
 4 …
 5 std::for_each(vvec.begin(), vvec.end(),
 6                  boost::apply_visitor(SimpleVariantVisitor()));
```

### 定义递归变体

过去几年中，有一个特定的数据交换格式—JavaScript 对象表示法或 JSON—的流行度呈现了惊人的增长。它是一种简单的基于文本的格式，通常比 XML 更简洁。最初用作 JavaScript 中的对象文字，该格式比 XML 更易读。它也是一种相对简单的格式，易于理解和解析。在本节中，我们将使用`boost::variants`来表示格式良好的 JSON 内容，并看看变体如何处理递归定义。

#### JSON 格式

首先，我们将看一个人员记录的 JSON 表示的例子：

```cpp
    {
        "Name": "Lucas",
        "Age": 38,
        "PhoneNumbers" : ["1123654798", "3121548967"],
        "Address" : { "Street": "27 Riverdale", "City": "Newtown", 
                             "PostCode": "902739"}
    }
```

上述代码是一个 JSON 对象的示例——它包含标识未命名对象属性的键值对。属性名称是带引号的字符串，例如`"Name"`、`"Age"`、`"PhoneNumbers"`（可以有多个）和`"Address"`。它们的值可以是简单字符串（`"Name"`）或数值（`"Age"`），或这些值的数组（`"PhoneNumbers"`）或其他对象（`"Address"`）。一个冒号（`:`）将键与值分开。键值对之间用逗号分隔。对象中的键值对列表用大括号括起来。这种格式允许任意级别的嵌套，如`"Address"`属性的值本身就是一个对象。您可以创建更多嵌套对象，这些对象是其他嵌套对象属性的值。

您可以将许多这样的记录组合在一个数组中，这些记录被方括号括起来，并用逗号分隔：

```cpp
[
    {
        "Name": "Lucas",
        "Age": 38,
        "PhoneNumbers" : ["1123654798", "3121548967"],
        "Address" : { "Street": "27 Riverdale", "City": "Newtown", 
                             "PostCode": "902739"}
    },
    {
        "Name": "Damien",
        "Age": 52,
        "PhoneNumbers" : ["6427851391", "3927151648"],
        "Address": {"Street": "11 North Ave.", "City" : "Rockport", 
                        "PostCode": "389203"}
    },
    … 
]
```

一个格式良好的 JSON 文本包含一个对象或零个或多个对象、数值、字符串、布尔值或空值的数组。对象本身包含零个或多个由唯一字符串表示的唯一属性。每个属性的值可以是字符串、数值、布尔值、空值、另一个对象或这些值的数组。因此，JSON 内容中的基本令牌是数值、字符串、布尔值和空值。聚合是对象和数组。

#### 使用递归变体表示 JSON 内容

如果我们要声明一个变体来表示 JSON 中的基本令牌，它会是这样的：

```cpp
 1 struct JSONNullType {};
 2 boost::variant<std::string, double, bool, JSONNullType> jsonToken;
```

类型`JSONNullType`是一个空类型，可以用来表示 JSON 中的空元素。

为了扩展这个变体以表示更复杂的 JSON 内容，我们将尝试表示一个 JSON 对象——一个键值对作为一种类型。键始终是字符串，但值可以是上面列出的任何类型或另一个嵌套对象。因此，JSON 对象的定义本质上是递归的，这就是为什么我们需要递归变体定义来对其进行建模。

为了在前述变体类型中包含 JSON 对象的定义，我们使用一个名为`boost::make_recursive_variant`的元函数。它接受一个类型列表，并将生成的递归变体类型定义为一个名为`type`的嵌套类型。因此，这是我们如何编写变体的递归定义的方式：

```cpp
 1 #define BOOST_VARIANT_NO_FULL_RECURSIVE_VARIANT_SUPPORT
 2 #include <boost/variant.hpp>
 3
 4 struct JSONNullType {};
 5
 6 typedef boost::make_recursive_variant<
 7                      std::string,
 8                      double,
 9                      bool,
10                      JSONNullType,
11                      std::map<std::string,
12                               boost::recursive_variant_>
13                     >::type JSONValue;
```

第 1 行的`#define`语句可能对许多编译器是必要的，特别是对于支持递归变体的使用`make_recursive_variant`的限制。

我们使用`boost::make_recursive_variant`元函数（第 6 行）定义递归变体。在类型列表中，我们添加了一个新类型`std::map`，其键的类型为`std::string`（第 11 行），值的类型为`boost::recursive_variant_`（第 12 行）。特殊类型`boost::recursive_variant_`用于指示外部变体类型可以作为映射中的值出现。因此，我们在变体定义中捕获了 JSON 对象的递归特性。

这个定义还不完整。一个格式良好的 JSON 内容可能包含所有这些不同类型的元素的数组。这些数组也可以是对象属性的值，或者嵌套在其他数组中。如果我们选择用向量来表示一个数组，那么对前述定义的扩展就很容易了：

**清单 2.8a：JSON 的递归变体**

```cpp
 1 #define BOOST_VARIANT_NO_FULL_RECURSIVE_VARIANT_SUPPORT
 2 #include <boost/variant.hpp>
 3
 4 struct JSONNullType {};
 5
 6 typedef boost::make_recursive_variant<
 7                      std::string,
 8                      double,
 9                      bool,
10                      JSONNullType,
11                      std::map<std::string,
12                               boost::recursive_variant_>,
13                      std::vector<boost::recursive_variant_>
14                     >::type JSONValue;
15
16 typedef std::vector<JSONValue> JSONArray;
17 typedef std::map<std::string, JSONValue> JSONObject;
```

我们添加了一个类型——`std::vector<boost::recursive_variant_>`（第 13 行），它表示了`JSONValue`对象的数组。凭借这一额外的行，我们现在支持了更多的可能性：

+   顶层数组由 JSON 对象、其他 JSON 数组和基本类型的标记组成

+   对象的数组值属性

+   另一个 JSON 数组中的数组值元素

这是`JSONValue`的完整定义。此外，我们为递归聚合类型——JSON 数组和 JSON 对象创建了 typedefs（第 16 行和第 17 行）。

#### 访问递归变体

我们现在将编写一个访问者，以标准表示法打印存储在变体中的 JSON 数据。访问递归变体与访问非递归变体没有区别。我们仍然需要定义能够处理变体可能存储的所有类型值的重载。此外，在递归聚合类型（在本例中为`JSONArray`和`JSONObject`）的重载中，我们可能需要递归访问其每个元素：

**清单 2.8b：访问递归变体**

```cpp
 1 void printArrElem(const JSONValue& val);
 2 void printObjAttr(const JSONObject::value_type& val); 
 3
 4 struct JSONPrintVisitor : public boost::static_visitor<void>
 5 {
 6   void operator() (const std::string& str) const
 7   {
 8     std::cout << '"' << escapeStr(str) << '"';
 9   }
10
11   void operator() (const JSONNullType&) const
12   {
13     std::cout << "null";
14   }
15
16   template <typename T>
17   void operator()(const T& value) const
18   {
19     std::cout << std::boolalpha << value;
20   }
21
22   void operator()(const JSONArray& arr) const
23   {
24     std::cout << '[';
25
26     if (!arr.empty()) {
27       boost::apply_visitor(*this, arr[0]);
28       std::for_each(arr.begin() + 1, arr.end(), printArrElem);
29     }
30 
31     std::cout << "\n";
32   }
33
34   void operator()(const JSONObject& object) const
35   {
36     std::cout << '{';
37 
38     if (!object.empty()) {
39       const auto& kv_pair = *(object.begin());
40       std::cout << '"' << escapeStr(kv_pair.first) << '"';
41       std::cout << ':';
42       boost::apply_visitor(*this, kv_pair.second);
43
44       auto it = object.begin();
45       std::for_each(++it, object.end(), printObjAttr);
46     }
47     std::cout << '}';
48   }
49
50 };
51
52 void printArrElem(const JSONValue& val) {
53   std::cout << ',';
54   boost::apply_visitor(JSONPrintVisitor(), val);
55 }
56
57 void printObjAttr(const JSONObject::value_type& val) {
58   std::cout << ',';
59   std::cout << '"' << escapeStr(val.first) << '"';
60   std::cout << ':';
61   boost::apply_visitor(JSONPrintVisitor(), val.second);
62 }
```

访问者`JSONPrintVisitor`公开继承自`boost::static_visitor<void>`，并为 JSON 值的不同可能类型提供了`operator()`的重载。它有一个`std::string`的重载（第 6 行），它在转义任何嵌入引号和其他需要转义的字符后，用双引号打印字符串（第 8 行）。为此，我们假设有一个名为`escapeStr`的函数可用。我们还有一个`JSONNullType`（第 11 行）的重载，它只是打印不带引号的字符串`null`。其他类型的值，如`double`或`bool`，由成员模板处理（第 17 行）。对于`bool`值，它使用`std::boolalpha` `ostream`操作器（第 19 行）打印不带引号的字符串`true`和`false`。

主要工作由`JSONArray`（第 22 行）和`JSONObject`（第 34 行）的两个重载完成。`JSONArray`重载打印了用方括号括起来并用逗号分隔的数组元素。它打印了`JSONValues`向量的第一个元素（第 27 行），然后对这个向量应用`std::for_each`通用算法，从第二个元素开始打印后续元素并用逗号分隔（第 28 行）。为此，它将`printArrElem`函数的指针作为第三个参数传递给`std::for_each`。`printArrElem`（第 52 行）函数通过应用`JSONPrintVisitor`（第 54 行）打印每个元素。

`JSONObject`重载将映射的元素打印为以逗号分隔的键值对列表。第一对被打印为带引号的转义键（第 40 行），然后是一个冒号（第 41 行），接着调用`boost::apply_visitor`（第 42 行）。后续的对通过使用`std::for_each`和`printObjAttr`函数指针（第 45 行）迭代映射的剩余元素，以逗号分隔前面的对来打印。这个逻辑类似于`JSONArray`的重载。`printObjAttr`函数（第 57 行）打印传递给它的每个键值对，前缀是一个逗号（第 58 行），打印转义的带引号的键（第 59 行），打印一个冒号（第 60 行），并在变体值上调用访问者（第 61 行）。

## Boost.Any

Boost Any 库采用了与 Boost Variant 不同的方法来存储异构数据。与 Variant 不同，Any 允许您存储几乎任何类型的数据，而不限于固定集合，并且保留存储数据的运行时类型信息。因此，它根本不使用模板，并且要求在使用 Boost Any 编译代码时启用**运行时类型识别**（**RTTI**）（大多数现代编译器默认情况下保持启用）。

### 提示

为了使 Boost Any 库正常工作，您不能禁用程序的 RTTI 生成。

在下面的示例中，我们创建了`boost::any`的实例来存储数字数据、字符数组和非 POD 类型对象：

**清单 2.9：使用 Boost Any**

```cpp
 1 #include <boost/any.hpp>
 2 #include <vector>
 3 #include <iostream>
 4 #include <string>
 5 #include <cassert>
 6 using boost::any_cast;
 7
 8 struct MyValue {
 9   MyValue(int n) : value(n) {}
10
11   int get() const { return value; }
12
13   int value;
14 };
15
16 int main() {
17   boost::any v1, v2, v3, v4;
18
19   assert(v1.empty());
20   const char *hello = "Hello";
21   v1 = hello;
22   v2 = 42;
23   v3 = std::string("Hola");
24   MyValue m1(10);
25   v4 = m1;
26
27   try {
28     std::cout << any_cast<const char*>(v1) << '\n';
29     std::cout << any_cast<int>(v2) << '\n';
30     std::cout << any_cast<std::string>(v3) << '\n';
31     auto x = any_cast<MyValue>(v4);
32     std::cout << x.get() << '\n';
33   } catch (std::exception& e) {
34     std::cout << e.what() << '\n';
35   }
36 }
```

您还可以使用`any_cast`的非抛出版本，而不是传递引用的方式，而是传递`any`对象的地址。如果存储的类型与要转换的类型不匹配，这将返回一个空指针，而不是抛出异常。以下代码片段说明了这一点：

```cpp
 1 boost::any v1 = 42;2 boost::any v2 = std::string("Hello");
 3 std::string *str = boost::any_cast<std::string>(&v1);
 4 assert(str == nullptr);
 5 int *num = boost::any_cast<int>(&v2);
 6 assert(num == nullptr);
 7
 8 num = boost::any_cast<int>(&v1);
 9 str = boost::any_cast<std::string>(&v2);
10 assert(num != nullptr);
11 assert(str != nullptr);
```

我们将`any`对象的地址传递给`any_cast`（第 3、5、8 和 9 行），除非`any_cast`的类型参数与`any`对象中存储的值的类型匹配，否则它将返回空值。使用`any_cast`的指针重载，我们可以编写一个通用的谓词来检查`any`变量是否存储了给定类型的值：

```cpp
template <typename T>
bool is_type(boost::any& any) {
  return ( !any.empty() && boost::any_cast<T>(&any) );
}
```

这就是您将如何使用它：

```cpp
boost::any v1 = std::string("Hello");
assert( is_type<std::string>(v1) );
```

`boost::any_cast`的这种行为模拟了`dynamic_cast`的工作原理。

在清单 2.9 中，我们使用不同的`boost::any`类型的实例来存储不同类型的值。但是，同一个`boost::any`实例可以在不同的时间存储不同类型的值。以下代码片段使用`any`的`swap`成员函数说明了这一点：

```cpp
 1 boost::any v1 = 19937;
 2 boost::any v2 = std::string("Hello");
 3
 4 assert(boost::any_cast<int>(&v1) != nullptr);
 5 assert(boost::any_cast<std::string>(&v2) != nullptr);
 6
 7 v1 = 22.36;
 8 v1.swap(v2);
 9 assert(boost::any_cast<std::string>(&v1) != nullptr);
10 assert(boost::any_cast<double>(&v2) != nullptr);
```

我们首先将`double`类型的值赋给`v1`（第 7 行），而它原来是`int`类型的值（第 1 行）。接下来，我们交换`v1`的内容与`v2`（第 8 行），而`v2`原来是`std::string`类型的值（第 2 行）。现在我们可以期望`v1`包含一个`std::string`值（第 9 行），而`v2`包含一个`double`值（第 10 行）。

除了使用`any_cast`的指针重载，我们还可以使用`any`的`type`成员函数来访问存储值的类型：

**清单 2.10：在 Any 中访问类型信息**

```cpp
boost::any value;
value = 20;
if (value.type().hash_code() == typeid(int).hash_code()) {
  std::cout << boost::any_cast<int>(value) << '\n';
}
```

`any`的`type`成员函数返回一个`std::type_info`对象（在标准库头文件`<typeinfo>`中定义）。为了检查这个类型是否与给定的类型相同，我们将其与通过对给定类型应用`typeid`运算符获得的`type_info`对象进行比较（在本例中是`int`）。我们不直接比较这两个`type_info`对象，而是比较它们使用`type_info`的`hash_code`成员函数获得的哈希码。

## Boost.Conversion

如果您曾尝试解析文本输入（来自文件、标准输入、网络等）并尝试对其中的数据进行语义转换，您可能会感到需要一种将文本转换为数值的简便方法。相反的问题是根据数值和文本程序变量的值编写文本输出。`basic_istream`和`basic_ostream`类提供了读取和写入特定类型值的功能。然而，这些用法的编程模型并不直观或健壮。C++标准库及其扩展提供了各种转换函数，具有不同程度的控制、灵活性和普遍缺乏可用性。例如，存在一整套函数，用于在数值和字符格式之间进行转换，或者反过来（例如，`atoi`、`strtol`、`strtod`、`itoa`、`ecvt`、`fcvt`等）。如果我们尝试编写用于类型转换的通用代码，我们甚至无法使用这些函数中的任何一个，因为它们只适用于特定类型之间的转换。我们如何定义一个通用的转换语法，可以扩展到任意类型？

Boost `Conversion`库引入了一对函数模板，提供了非常直观和统一的转换语法，也可以通过用户定义的特化进行扩展。我们将逐一查看转换模板。

### lexical_cast

`lexical_cast`函数模板可用于将源类型转换为目标类型。其语法类似于各种 C++转换的语法：

```cpp
#include <boost/lexical_cast.hpp>
namespace boost {
template <typename T, typename S>
T lexical_cast (const S& source);
}
```

以下示例显示了我们如何使用`lexical_cast`将字符串转换为整数：

**清单 2.11：使用 lexical_cast**

```cpp
 1 std::string str = "1234";
 2
 3 try {
 4   int n = boost::lexical_cast<int>(str);
 5   assert(n == 1234);
 6 } catch (std::exception& e) {
 7   std::cout << e.what() << '\n';
 8 }
```

我们应用`lexical_cast`（第 4 行）将`std::string`类型的值转换为`int`类型的值。这种方法的美妙之处在于它可以为所有转换提供统一的语法，并且可以扩展到新类型。如果字符串不包含有效的数字字符串，则`lexical_cast`调用将抛出`bad_lexical_cast`类型的异常。

提供了`lexical_cast`函数模板的重载，允许转换字符数组的一部分：

```cpp
#include <boost/lexical_cast.hpp>
namespace boost {
template <typename T >
T lexical_cast (const char* str, size_t size);
}
```

我们可以以以下方式使用前述函数：

```cpp
 1 std::string str = "abc1234";
 2
 3 try {
 4   int n = boost::lexical_cast<int>(str.c_str() + 3, 4);
 5   assert(n == 1234);
 6 } catch (std::exception& e) {
 7   std::cout << e.what() << '\n';
 8 }
```

在转换可流式传输的类型的对象时，`lexical_cast`将对象流式传输到`ostream`对象，例如`stringstream`的实例，并将其作为目标类型读取回来。

### 提示

可流式传输的对象可以转换为字符流，并插入到`ostream`对象中，例如`stringstream`的实例。换句话说，如果定义了类型`T`，使得`ostream& operator<<(ostream&, const T&)`，则称其为可流式传输。

为每个此类操作设置和拆卸流对象会产生一些开销。因此，在某些情况下，`lexical_cast`的默认版本可能无法提供最佳性能。在这种情况下，您可以为涉及的类型集合专门化`lexical_cast`模板，并使用快速库函数或提供自己的快速实现。`Conversion`库已经优化了所有常见类型对的`lexical_cast`。

除了`lexical_cast`模板之外，还有其他模板可用于不同数值类型之间的转换（`boost::numeric_cast`）、类层次结构中的向下转换和交叉转换（`polymorphic_downcast`、`polymorphic_cast`）。您可以参考在线文档以获取有关这些功能的更多信息。

# 处理命令行参数

命令行参数，就像 API 参数一样，是帮助您调整命令行行为的遥控按钮。一组精心设计的命令行选项在很大程度上支持命令的功能。在本节中，我们将看到 Boost.Program_Options 库如何帮助您为自己的程序添加对丰富和标准化的命令行选项的支持。

## 设计命令行选项

C 为程序的命令行提供了最原始的抽象。使用传递给主函数的两个参数-参数的数量（`argc`）和参数的列表（`argv`）-您可以了解到传递给程序的每个参数及其相对顺序。以下程序打印出`argv[0]`，这是程序本身的路径，用它调用程序。当使用一组命令行参数运行时，程序还会将每个参数分别打印在一行上。

大多数程序需要添加更多的逻辑和验证来验证和解释命令行参数，因此需要一个更复杂的框架来处理命令行参数：

```cpp
1 int main(int argc, char *argv[])
2 {
3   std::cout << "Program name: " << argv[0] << '\n';
4
5   for (int i = 1; i < argc; ++i) {
6     std::cout << "argv[" << i << "]: " << argv[i] << '\n';
7   }
8 }
```

### diff 命令-一个案例研究

程序通常会记录一组修改其行为的命令行选项和开关。让我们来看看 Unix 中`diff`命令的例子。`diff`命令是这样运行的：

```cpp
$ diff file1 file2

```

它打印出两个文件内容之间的差异。有几种方式可以选择打印出差异。对于每个不同的块，您可以选择打印出几行额外的上下文，以更好地理解出现差异的上下文。这些周围的行或"上下文"在两个文件之间是相同的。为此，您可以使用以下的其中一种替代方案：

```cpp
$ diff -U 5 file1 file2
$ diff --unified=5 file1 file2

```

在这里，您选择打印五行额外的上下文。您还可以通过指定默认值为三来选择默认值：

```cpp
$ diff --unified file1 file2

```

在前面的例子中，`-U`或`--unified`是命令行选项的例子。前者是一个由单个前导破折号和单个字母（`-U`）组成的短选项。后者是一个由两个前导破折号和多字符选项名称（`--unified`）组成的长选项。

数字`5`是一个选项值；是前面的选项（`-U`或`--unified`）的参数。选项值与前面的短选项之间用空格分隔，但与前面的长选项之间用等号(`=`)分隔。

如果您正在"diffing"两个 C 或 C++源文件，您可以使用命令行开关或标志`-p`来获取更有用的信息。开关是一个不带选项值的选项。使用此开关，您可以打印出在检测到特定差异的上下文中 C 或 C++函数的名称。没有与之对应的长选项。

`diff`命令是一个非常强大的工具，可以在完整目录中查找文件内容的差异。当对比两个目录时，如果一个文件存在于一个目录中而另一个目录中不存在，`diff`默认会忽略此文件。但是，您可能希望查看新文件的内容。为此，您将使用`-N`或`--new-file`开关。如果我们现在想要在两个 C++源代码目录上运行我们的`diff`命令来识别更改，我们可以这样做：

```cpp
$ diff -pN –unified=5 old_source_dir new_source_dir

```

您不必眼尖才能注意到我们使用了一个名为`-pN`的选项。这实际上不是一个单一的选项，而是两个开关（`-p`）和（`-N`）合并在一起。

从这个案例研究中应该能够看出某些模式或约定：

+   用单破折号开始短选项

+   用双破折号开始长选项

+   用空格分隔短选项和选项值

+   用等号分隔长选项和选项值

+   合并短开关

这些是高度符合 POSIX 的系统（如 Linux）上*事实上*标准化的约定。然而，并不是唯一遵循的约定。Windows 命令行经常使用前斜杠(`/`)代替连字符。它们通常不区分短选项和长选项，并有时使用冒号(`:`)代替等号来分隔选项和其选项值。Java 命令以及几个旧的 Unix 系统中的命令使用单个前导连字符来表示短选项和长选项。其中一些使用空格来分隔选项和选项值，无论是短选项还是长选项。在解析命令行时，如何处理从平台到平台变化的这么多复杂规则？这就是 Boost 程序选项库产生重大影响的地方。

## 使用 Boost.Program_Options

Boost 程序选项库为您提供了一种声明性的解析命令行的方式。您可以指定程序支持的选项和开关集合以及每个选项支持的选项值类型。您还可以指定要为命令行支持的约定集合。然后，您可以将所有这些信息提供给库函数，该函数解析和验证命令行，并将所有命令行数据提取到类似字典的结构中，从中可以访问单个数据位。现在，我们将编写一些代码来模拟`diff`命令的先前提到的选项：

**清单 2.12a：使用 Boost 程序选项**

```cpp
 1 #include <boost/program_options.hpp>
 2
 3 namespace po = boost::program_options;
 4 namespace postyle = boost::program_options::command_line_style;
 5 
 6 int main(int argc, char *argv[])
 7 {
 8   po::options_description desc("Options");
 9   desc.add_options()
10      ("unified,U", po::value<unsigned int>()->default_value(3),
11             "Print in unified form with specified number of "
12             "lines from the surrounding context")
13      (",p", "Print names of C functions "
14             " containing the difference")
15      (",N", "When comparing two directories, if a file exists in"
16             " only one directory, assume it to be present but "
17             " blank in the other directory")
18      ("help,h", "Print this help message");
```

在前面的代码片段中，我们使用`options_description`对象声明了命令行的结构。连续的选项使用`add_options`返回的对象中的重载函数调用`operator()`来声明。您可以像在`std::cout`上级联调用插入运算符(`<<`)一样级联调用此运算符。这使得选项的规范非常易读。

我们声明了`--unified`或`-U`选项，指定长选项和短选项在单个字符串中，用逗号分隔（第 10 行）。第二个参数表示我们期望一个数字参数，如果在命令行上未指定参数，则默认值将为`3`。第三个字段是选项的描述，将用于生成文档字符串。

我们声明了短选项`-p`和`-N`（第 13 和 15 行），但由于它们没有相应的长选项，它们是以逗号开头，后跟短选项(`",p"`和`",N"`)。它们也不需要选项值，所以我们只提供它们的描述。

到目前为止一切顺利。现在我们将通过解析命令行并获取值来完成代码示例。首先，我们将指定在 Windows 和 Unix 中要遵循的风格：

**清单 2.12b：使用 Boost 程序选项**

```cpp
19   int unix_style    = postyle::unix_style
20                      |postyle::short_allow_next;
21
22   int windows_style = postyle::allow_long
23                      |postyle::allow_short
24                      |postyle::allow_slash_for_short
25                      |postyle::allow_slash_for_long
26                      |postyle::case_insensitive
27                      |postyle::short_allow_next
28                      |postyle::long_allow_next;
```

前面的代码突出了 Windows 和 Unix 约定之间的一些重要区别：

+   一个更或多或少标准化的 Unix 风格可预先准备好并称为`unix_style`。然而，我们必须自己构建 Windows 风格。

+   `short_allow_next`标志允许您用空格分隔短选项和其选项值；这在 Windows 和 Unix 上都可以使用。

+   `allows_slash_for_short`和`allow_slash_for_long`标志允许选项以斜杠开头；这是 Windows 上的常见做法。

+   `case_insensitive`标志适用于 Windows，通常习惯于不区分大小写的命令和选项。

+   在 Windows 上，`long_allow_next`标志允许长选项和选项值用空格而不是等号分隔。

现在，让我们看看如何使用所有这些信息解析符合规范的命令行。为此，我们将声明一个`variables_map`类型的对象来读取所有数据，然后解析命令行：

**清单 2.12c：使用 Boost 程序选项**

```cpp
29   po::variables_map vm;
30   try {
31     po::store(
32       po::command_line_parser(argc, argv)
33          .options(desc)
34          .style(unix_style)  // or windows_style
35          .run(), vm);
36
37     po::notify(vm); 
38
39     if (argc == 1 || vm.count("help")) {
40       std::cout << "USAGE: " << argv[0] << '\n'
41                 << desc << '\n';
42       return 0;
43     }
44   } catch (po::error& poe) {
45     std::cerr << poe.what() << '\n'
46               << "USAGE: " << argv[0] << '\n' << desc << '\n';
47     return EXIT_FAILURE;
48   }
```

我们使用`command_line_parser`函数创建一个命令行解析器（第 32 行）。我们在返回的解析器上调用`options`成员函数来指定在`desc`中编码的解析规则（第 33 行）。我们链式调用更多的成员函数，将其传递给解析器的`style`成员函数以指定预期的样式（第 34 行），并调用`run`成员函数来执行实际的解析。调用`run`返回一个包含从命令行解析的数据的数据结构。调用`boost::program_options::store`将从这个数据结构中解析的数据存储在`variables_map`对象`vm`中（第 31-35 行）。最后，我们检查程序是否在没有参数或使用`help`选项的情况下调用，并打印帮助字符串（第 39 行）。将`option_description`实例`desc`流式传输到`ostream`会打印一个帮助字符串，该字符串是根据`desc`中编码的命令行规则自动生成的（第 41 行）。所有这些都封装在一个 try-catch 块中，以捕获由对`run`的调用抛出的任何命令行解析错误（第 35 行）。在出现这样的错误时，将打印错误详细信息（第 45 行），并打印使用详细信息（第 46 行）。

如果你注意到，你会发现我们在第 37 行调用了一个名为`notify(…)`的函数。在更高级的用法中，我们可以选择使用从命令行读取的值来设置变量或对象成员，或执行其他后处理操作。这些操作可以在声明选项描述时为每个选项指定，但这些操作只能通过调用`notify`来启动。为了保持一致，不要删除对`notify`的调用。

现在我们可以提取通过命令行传递的值了：

**清单 2.12d：使用 Boost 程序选项**

```cpp
49   unsigned int context = 0;
50   if (vm.count("unified")) {
51     context = vm["unified"].as<unsigned int>();
52   }
53
54   bool print_cfunc = (vm.count("p") > 0);
```

### 解析位置参数

如果你注意到了，你会注意到我们没有做任何事情来读取两个文件名；`diff`命令的两个主要操作数。我们之所以这样做是为了简单起见，现在我们将修复这个问题。我们这样运行`diff`命令：

```cpp
$ diff -pN --unified=5 old_source_dir new_source_dir

```

`old_source_dir`和`new_source_dir`参数被称为位置参数。它们既不是选项也不是开关，也不是任何选项的参数。为了处理它们，我们将不得不使用一些新技巧。首先，我们必须告诉解析器我们期望的这些参数的数量和类型。其次，我们必须告诉解析器这些是位置参数。以下是代码片段：

```cpp
 1 std::string file1, file2;
 2 po::options_description posparams("Positional params");
 3 posparams.add_options()
 4         ("file1", po::value<std::string>(&file1)->required(), "")
 5         ("file2", po::value<std::string>(&file2)->required(), "");
 6 desc.add(posparams);
 7
 8
 9 po::positional_options_description posOpts;
10 posOpts.add("file1", 1);  // second param == 1 indicates that
11 posOpts.add("file2", 1);  //  we expect only one arg each
12
13 po::store(po::command_line_parser(argc, argv)14                 .options(desc)
15                 .positional(posOpts)
16                 .style(windows_style)
17                 .run(), vm);
```

在前面的代码中，我们设置了一个名为`posparams`的第二个选项描述对象，用于识别位置参数。我们使用`value`参数的`required()`成员函数（第 4 和 5 行）添加了名称为`"file1"`和`"file2"`的选项，并指示这些参数是必需的。我们还指定了两个字符串变量`file1`和`file2`来存储位置参数。所有这些都添加到主选项描述对象`desc`（第 6 行）。为了使解析器不寻找名为`"--file1"`和`"--file2"`的实际选项，我们必须告诉解析器这些是位置参数。这是通过定义一个`positional_options_description`对象（第 9 行）并添加应该被视为位置选项的选项（第 10 和 11 行）来完成的。在`add(…)`调用中的第二个参数指定了应该考虑该选项的位置参数的数量。由于我们想要一个文件名，分别用于选项`file1`和`file2`，所以我们在两次调用中都指定为`1`。命令行上的位置参数根据它们添加到位置选项描述的顺序进行解释。因此，在这种情况下，第一个位置参数将被视为`file1`，第二个参数将被视为`file2`。

### 多个选项值

在某些情况下，单个选项可能需要多个选项值。例如，在编译期间，你将多次使用`-I`选项来指定多个目录。为了解析这样的选项及其选项值，你可以将目标类型指定为矢量，如下面的代码片段所示：

```cpp
 1 po::options_description desc("Options");
 2 desc.add_option()
 3      ("include,I", po::value<std::vector<std::string> >(),
 4       "Include files.")
 5      (…);
```

这将在这样的调用上起作用：

```cpp
$ c++ source.cpp –o target -I path1 -I path2 -I path3

```

然而，在某些情况下，你可能想要指定多个选项值，但只指定一次选项本身。假设你正在运行一个命令来发现连接到一组服务器的每个资产（本地存储、NIC、HBA 等）的命令。你可以有这样一个命令：

```cpp
$ discover_assets --servers svr1 svr2 svr3 --uid user

```

在这种情况下，为了模拟`--server`选项，你需要像这样使用`multitoken()`指令：

```cpp
 1 po::options_description desc("Options");
 2 desc.add_option()
 3      ("servers,S", 
 4       po::value<std::vector<std::string> >()->multitoken(),
 5       "List of hosts or IPs.")
 6      ("uid,U", po::value<std::string>, "User name");
```

你可以通过变量映射这样检索矢量值参数：

```cpp
1 std::vector<std::string> servers = vm["servers"];
```

或者，你可以在选项定义时使用变量挂钩，就像这样：

```cpp
1 std::vector<std::string> servers;
2 desc.add_option()
3      ("servers,S",
4       po::value<std::vector<std::string> >(&servers
5          ->multitoken(),
6       "List of hosts or IPs.")…;
```

确保在解析命令行后不要忘记调用`notify`。

### 提示

尝试支持在同一命令中同时使用多个令牌的位置参数和选项可能会使解析器混淆，通常应该避免。

程序选项库使用 Boost Any 进行实现。为了使程序选项库正常工作，你不能禁用程序的 RTTI 生成。

# 其他实用程序和编译时检查

Boost 包括许多微型库，提供小而有用的功能。它们中的大多数都不够复杂，无法成为单独的库。相反，它们被分组在`Boost.Utility`和`Boost.Core`下。我们将在这里看两个这样的库。

我们还将看一些有用的方法，尽早在编译时检测错误，并使用 Boost 的不同设施从程序的编译环境和工具链中获取信息。

## BOOST_CURRENT_FUNCTION

在编写调试日志时，能够写入函数名称以及有关调用日志的函数的一些限定信息非常有用。这些信息（显然）在编译源代码时对编译器是可用的。然而，打印它的方式对不同的编译器是不同的。即使对于同一个编译器，可能有多种方法来做到这一点。如果你想编写可移植的代码，这是一个你必须注意隐藏的瑕疵。这方面最好的工具是宏`BOOST_CURRENT_FUNCTION`，正式是`Boost.Utility`的一部分，在下面的示例中展示了它的作用：

**清单 2.13：漂亮打印当前函数名**

```cpp
 1 #include <boost/current_function.hpp>
 2 #include <iostream>
 3
 4 namespace FoFum {
 5 class Foo
 6 {
 7 public:
 8   void bar() {
 9     std::cout << BOOST_CURRENT_FUNCTION << '\n';
10     bar_private(5);
11   }
12
13   static void bar_static() {
14     std::cout << BOOST_CURRENT_FUNCTION << '\n';
15   }
16
17 private:
18   float bar_private(int x) const {
19     std::cout << BOOST_CURRENT_FUNCTION << '\n';
20   return 0.0;
21   }
22 };
23 } // end namespace FoFum
24
25 namespace {
26 template <typename T>
27 void baz(const T& x)
28 {
29   std::cout << BOOST_CURRENT_FUNCTION << '\n';
30 }
32 } // end unnamed namespace
33
34 int main()
35 {
36   std::cout << BOOST_CURRENT_FUNCTION << '\n';
37   FoFum::Foo f;
38   f.bar();
39   FoFum::Foo::bar_static();
40   baz(f);
41 }
```

根据你的编译器，你看到的输出格式会有所不同。GNU 编译器倾向于有更可读的输出，而在 Microsoft Visual Studio 上，你会看到一些非常复杂的输出，包括调用约定等细节。特别是，在 Visual Studio 上，模板实例化的输出要复杂得多。这是我在我的系统上看到的一个示例输出。

使用 GNU g++：

```cpp
int main()
void FoFum::Foo::bar()
float FoFum::Foo::bar1(int) const
static void FoFum::Foo::bar_static()
void {anonymous}::baz(const T&) [with T = FoFum::Foo]
```

使用 Visual Studio：

```cpp
int __cdecl main(void)
void __thiscall FoFum::Foo::bar(void)
float __thiscall FoFum::Foo::bar1(int) const
void __cdecl FoFum::Foo::bar_static(void)
void __cdecl 'anonymous-namespace'::baz<class FoFum::Foo>(const class FoFum::Foo &)
```

你可以立即看到一些不同之处。GNU 编译器从非静态方法中调用静态方法。在 Visual Studio 中，你必须根据调用约定进行区分（`__cdecl`用于静态成员方法以及全局方法，`__thiscall`用于实例方法）。你可能想看一下`current_function.hpp`头文件，以找出在幕后使用了哪些宏。例如，在 GNU 编译器中，是`__PRETTY_FUNCTION__`，而在 Visual Studio 中是`__FUNCSIG__`。

## Boost.Swap

Boost Swap 库是另一个有用的微型库，是 Boost Core 的一部分：

```cpp
#include <boost/core/swap.hpp>
namespace boost {
  template<typename T1, typename T2>
  void swap(T1& left, T2& right);}
```

它围绕交换对象的一个众所周知的习语。让我们首先看看问题本身，以了解发生了什么。

在`std`命名空间中有一个全局的`swap`函数。在许多情况下，对于在特定命名空间中定义的类型，可能会在相同的命名空间中提供一个专门的`swap`重载。在编写通用代码时，这可能会带来一些挑战。想象一个通用函数调用其参数的`swap`：

```cpp
 1 template <typename T>
 2 void process_values(T& arg1, T& arg2, …)
 3 {
 4   …
 5   std::swap(arg1, arg2);
```

在上面的代码片段中，我们在第 5 行调用`std::swap`来执行交换。虽然这是良好形式的，但在某些情况下可能不会产生期望的结果。考虑命名空间`X`中的以下类型和函数：

```cpp
 1 namespace X {
 2   struct Foo {};
 3
 4   void swap(Foo& left, Foo& right) { 
 5     std::cout << BOOST_CURRENT_FUNCTION << '\n';
 6   }
 7 }
```

当然，`X::Foo`是一个平凡的类型，`X::swap`是一个无操作，但它们可以被一个有意义的实现替换，我们在这里所做的观点仍然成立。

那么，如果你在两个类型为`X::Foo`的参数上调用函数`process_values`会发生什么？

```cpp
 1 X::Foo f1, f2;
 2 process_values(f1, f2, …); // calls process_values<X::Foo>
```

对`process_values`的调用（第 2 行）将在传递给`X::Foo`的实例上调用`std::swap`，即`f1`和`f2`。然而，我们可能希望在`f1`和`f2`上调用`X::swap`，因为这是一个更合适的重载。有一种方法可以做到这一点；你可以调用`boost::swap`。下面是`process_values`模板片段的重写：

```cpp
 1 #include <boost/core/swap.hpp>
 2
 3 template <typename T>
 4 void process_values(T& arg1, T& arg2, …)
 5 {
 6   …
 7   boost::swap(arg1, arg2);
```

如果你现在运行这段代码，你会看到`X::swap`重载将其名称打印到控制台。要理解`boost::swap`是如何调用适当的重载的，我们需要了解如何在没有`boost::swap`的情况下解决这个问题：

```cpp
 1 template <typename T>
 2 void process_values(T& arg1, T& arg2, …)
 3 {
 4   …
 5   using std::swap;
 6   swap(arg1, arg2);
```

如果我们没有`using`声明（第 5 行），对`swap`的调用（第 6 行）仍然会成功，对于一个在命名空间中定义的类型`T`，该命名空间中定义了`T`的`swap`重载——这要归功于**参数相关查找**（**ADL**）——`X::Foo`和`X::swap`就是这样的类型。然而，对于在全局命名空间中定义的类型，它会失败（假设你没有在全局命名空间中定义一个通用的`swap`）。有了`using`声明（第 5 行），我们为对`swap`的未限定调用创建了回退。当 ADL 成功找到命名空间级别的`swap`重载时，对`swap`的调用就会解析为这个重载。当 ADL 找不到这样的重载时，就会使用`std::swap`，如`using`声明所规定的那样。问题在于这是一个不明显的技巧，你必须知道才能使用它。你团队中的每个工程师都不一定都了解 C++中的所有名称查找规则。与此同时，他总是可以使用`boost::swap`，它本质上是将这段代码包装在一个函数中。现在你可以只使用一个版本的`swap`，并期望每次调用时调用最合适的重载。

## 编译时断言

编译时断言要求在代码的某个点上某些条件必须为真。任何条件的违反都会导致编译失败。这是一种在编译时发现错误的有效方法，否则这些错误可能会在运行时造成严重的困扰。它还可以帮助减少由于模板实例化失败而产生的编译器错误消息的数量和冗长程度。

运行时断言旨在证实代码中某些必须为真的条件的不变性。这样的条件可能是逻辑或算法的结果，也可能基于某些已记录的约定。例如，如果你正在编写一个将一个数字提高到某个幂的函数，那么你如何处理数和幂都为零的数学上未定义的情况？你可以使用断言来明确表达这一点，如下面的代码片段所示（第 6 行）：

```cpp
 1 #include <cassert>
 2
 3 double power(double base, double exponent)
 4 {
 5   // no negative powers of zero
 6   assert(base != 0 || exponent > 0);
 7   …
 8 }
```

这样的不变性违反表明存在错误或缺陷，需要修复，并导致程序在调试构建中发生灾难性故障。Boost 提供了一个名为`BOOST_STATIC_ASSERT`的宏，它接受一个可以在编译时求值的表达式，并在这个表达式求值为假时触发编译失败。

例如，您可能已经设计了一个内存分配器类模板，该模板仅用于“小”对象。当然，小是任意的，但您可以设计您的分配器以优化大小为 16 字节或更小的对象。如果您想强制正确使用您的类，您应该简单地阻止其对大于 16 字节的类的实例化。这是我们的第一个例子`BOOST_STATIC_ASSERT`，它可以帮助您强制执行分配器的小对象语义：

**清单 2.16a：使用编译时断言**

```cpp
 1 #include <boost/static_assert.hpp>
 2
 3 template <typename T>
 4 class SmallObjectAllocator
 5 {
 6   BOOST_STATIC_ASSERT(sizeof(T) <= 16);
 7
 8 public:
 9   SmallObjectAllocator() {}
10 };
```

我们定义了一个名为`SmallObjectAllocator`的虚拟分配器模板（第 3 和第 4 行），并在类范围内调用`BOOST_STATIC_ASSERT`宏（第 6 行）。我们将一个必须在编译时可能求值的表达式传递给宏。现在，`sizeof`表达式总是由编译器求值的，而 16 是一个整数字面量，因此表达式`sizeof(T) <= 16`可以完全在编译时求值，并且可以传递给`BOOST_STATIC_ASSERT`。如果我们现在用类型`Foo`实例化`SmallObjectAllocator`，其大小为 32 字节，我们将由于第 6 行的静态断言而得到编译器错误。这是可以触发断言的代码：

**清单 2.16b：使用编译时断言**

```cpp
11 struct Foo
12 {
13   char data[32];
14 };
15
16 int main()
17 {
18   SmallObjectAllocator<int> intAlloc;
19   SmallObjectAllocator<Foo> fooAlloc; // ERROR: sizeof(Foo) > 16
20 }
```

我们定义了一个类型`Foo`，其大小为 32 字节，大于`SmallObjectAllocator`支持的最大大小（第 13 行）。我们使用类型`int`（第 18 行）和`Foo`（第 19 行）实例化`SmallObjectAllocator`模板。`SmallObjectAllocator<Foo>`的编译失败，我们得到一个错误消息。

### 提示

C++11 支持使用新的`static_assert`关键字进行编译时断言。如果您使用的是 C++11 编译器，`BOOST_STATIC_ASSERT`在内部使用`static_assert`。

实际的错误消息自然会因编译器而异，特别是在 C++03 编译器上。在 C++11 编译器上，因为这在内部使用`static_assert`关键字，错误消息往往更加统一和有意义。然而，在 C++11 之前的编译器上，您也可以得到一个相当准确的错误行。在我的系统上，使用 GNU g++编译器在 C++03 模式下，我得到了以下错误：

```cpp
StaticAssertTest.cpp: In instantiation of 'class SmallObjectAllocator<Foo>':
StaticAssertTest.cpp:19:29:   required from here
StaticAssertTest.cpp:6:3: error: invalid application of 'sizeof' to incomplete type 'boost::STATIC_ASSERTION_FAILURE<false>'
```

编译器错误的最后一行引用了一个不完整的类型`boost::STATIC_ASSERTION_FAILURE<false>`，它来自`BOOST_STATIC_ASSERT`宏的内部。很明显，第 6 行出现了错误，静态断言失败。如果我切换到 C++11 模式，错误消息会更加合理：

```cpp
StaticAssertTest.cpp: In instantiation of 'class SmallObjectAllocator<Foo>':
StaticAssertTest.cpp:19:29:   required from here
StaticAssertTest.cpp:6:3: error: static assertion failed: sizeof(T) <= 16
```

还有另一种静态断言宏的变体称为`BOOST_STATIC_ASSERT`，它将消息字符串作为第二个参数。对于 C++11 编译器，它只是打印这个消息作为错误消息。在 C++11 之前的编译器下，这个消息可能会或可能不会出现在编译器错误内容中。您可以这样使用它：

```cpp
 1 BOOST_STATIC_ASSERT_MSG(sizeof(T) <= 16, "Objects of size more" 
 2                         " than 16 bytes not supported.");
```

并非所有表达式都可以在编译时求值。大多数情况下，涉及常量整数、类型大小和一般类型计算的表达式可以在编译时求值。Boost TypeTraits 库和 Boost **Metaprogramming Library** (**MPL**)提供了几个元函数，使用这些元函数可以在编译时对类型进行许多复杂的条件检查。我们用一个小例子来说明这种用法。我们将在后面的章节中看到更多这种用法的例子。

我们不仅可以在类范围内使用静态断言，还可以在函数和命名空间范围内使用。这是一个函数模板库的示例，允许对不同的 POD 类型进行位操作。在实例化这些函数时，我们在编译时断言传递的类型是 POD 类型：

**清单 2.17：使用编译时断言**

```cpp
 1 #include <boost/static_assert.hpp>
 2 #include <boost/type_traits.hpp>
 3
 4 template <typename T, typename U>
 5 T bitwise_or (const T& left, const U& right)
 6 {
 7   BOOST_STATIC_ASSERT(boost::is_pod<T>::value && 
 8                       boost::is_pod<U>::value);
 9   BOOST_STATIC_ASSERT(sizeof(T) >= sizeof(U));
10
11   T result = left;
12   unsigned char *right_array =
13           reinterpret_cast<unsigned char*>(&right);
14   unsigned char *left_array =
15           reinterpret_cast<unsigned char*>(&result);
16   for (size_t i = 0; i < sizeof(U); ++i) {
17     left_array[i] |= right_array[i];
18   }
19
20   return result;
21 }
```

在这里，我们定义了一个函数`bitwise_or`（第 4 和 5 行），它接受两个对象，可能是不同类型和大小的，并返回它们内容的按位或。在这个函数内部，我们使用了元函数`boost::is_pod<T>`来断言传递的两个对象都是 POD 类型（第 7 行）。此外，因为函数的返回类型是`T`，即左参数的类型，我们断言函数必须始终首先调用较大的参数（第 9 行），以便没有数据丢失。

## 使用预处理宏进行诊断

在我作为软件工程师的职业生涯中，有很多次我曾经在建立在五种不同 Unix 和 Windows 上的单一代码库的产品上工作，通常是并行进行的。通常这些构建服务器会是大型服务器，附带数百吉字节的存储空间，用于多个产品进行构建。会有无数的环境、工具链和配置共存于同一服务器上。将这些系统稳定到一切都能完美构建的程度肯定花费了很长时间。有一天，地狱就在一夜之间降临了，尽管没有进行任何重大的提交，我们的软件开始表现得很奇怪。我们花了将近一天的时间才发现有人动了环境变量，结果我们使用了不同版本的编译器进行链接，并且使用了与我们的第三方库构建时不同的运行时进行链接。我不需要告诉你，即使在那个时候，这对于构建系统来说也不是理想的情况。不幸的是，你可能仍然会发现这样混乱的环境，需要很长时间来设置，然后被一个轻率的改变破坏。在半天的徒劳努力之后拯救我们的是明智地使用预处理宏在程序启动时倾倒有关构建系统的信息，包括编译器名称、版本、架构等。我们很快就能从程序倾倒的数据中获得足够的信息，在它不可避免地崩溃之前，我们就发现了编译器不匹配的问题。

这样的信息对于可能能够通过利用特定接口在每个编译器或平台上提供库的最优实现的库编写者来说是双重有用的，并且可以根据预处理宏定义进行条件编译。然而，使用这些宏的弊端在于不同编译器、平台和环境之间的绝对差异，包括它们的命名和功能是什么。Boost 通过其`Config`和`Predef`库提供了一个更加统一的一组用于获取有关软件构建环境信息的预处理宏。我们将看一下这些库中一些有用的宏。

`Predef`库是一个仅包含头文件的库，提供了各种宏，用于在编译时获取有关构建环境的有用信息。可用的信息可以分为不同的类别。我们将看一下以下代码，以说明如何访问和使用这些信息，而不是提供一个选项的长列表并解释它们的作用——在线文档已经充分做到了这一点。

清单 2.18a：使用 Predef 中的诊断宏

```cpp
 1 #include <boost/predef.h>
 2 #include <iostream>
 3
 4 void checkOs()
 5 {
 6   // identify OS
 7 #if defined(BOOST_OS_WINDOWS)
 8   std::cout << "Windows" << '\n';
 9 #elif defined(BOOST_OS_LINUX)
10   std::cout << "Linux" << '\n';
11 #elif defined(BOOST_OS_MACOS)
12   std::cout << "MacOS" << '\n';
13 #elif defined(BOOST_OS_UNIX)
14   std::cout << Another UNIX" << '\n'; // *_AIX, *_HPUX, etc. 
15 #endif
16 }
```

前面的函数使用了`Predef`库中的`BOOST_OS_*`宏来识别代码所构建的操作系统。我们只展示了三种不同操作系统的宏。在线文档提供了用于识别不同操作系统的完整列表的宏。

清单 2.18b：使用 Predef 中的诊断宏

```cpp
 1 #include <boost/predef.h>
 2 #include <iostream>
 34 void checkArch()
 5 {
 6   // identify architecture
 7 #if defined(BOOST_ARCH_X86)
 8  #if defined(BOOST_ARCH_X86_64)
 9   std::cout << "x86-64 bit" << '\n';
10  #else
11   std::cout << "x86-32 bit" << '\n';
12  #endif
13 #elif defined(BOOST_ARCH_ARM)
14   std::cout << "ARM" << '\n';
15 #else
16   std::cout << "Other architecture" << '\n';
17 #endif
18 }
```

前面的函数使用了`Predef`库中的`BOOST_ARCH_*`宏来识别代码所构建的平台的架构。我们只展示了 x86 和 ARM 架构的宏；在线文档提供了用于识别不同架构的完整列表的宏。

清单 2.18c：使用 Predef 中的诊断宏

```cpp
 1 #include <boost/predef.h>
 2 #include <iostream>
 3
 4 void checkCompiler()
 5 {
 6   // identify compiler
 7 #if defined(BOOST_COMP_GNUC)
 8   std::cout << "GCC, Version: " << BOOST_COMP_GNUC << '\n';
 9 #elif defined(BOOST_COMP_MSVC)
10   std::cout << "MSVC, Version: " << BOOST_COMP_MSVC << '\n';
11 #else
12   std::cout << "Other compiler" << '\n';
13 #endif
14 }
```

前面的函数使用了`Predef`库中的`BOOST_COMP_*`宏来识别用于构建代码的编译器。我们只展示了 GNU 和 Microsoft Visual C++编译器的宏。在线文档提供了用于识别不同编译器的完整宏列表。当定义了特定编译器的`BOOST_COMP_*`宏时，它会评估为其数值版本。例如，在 Visual Studio 2010 上，`BOOST_COMP_MSVC`评估为`100030319`。这可以被翻译为版本`10.0.30319`：

2.18d 清单：使用 Predef 中的诊断宏

```cpp
 1 #include <boost/predef.h>
 2 #include <iostream>
 3
 4 void checkCpp11()
 5 {
 6   // Do version checks
 7 #if defined(BOOST_COMP_GNUC)
 8  #if BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(4, 8, 1)
 9    std::cout << "Incomplete C++ 11 support" << '\n';
10  #else
11    std::cout << "Most C++ 11 features supported" << '\n';
12  #endif
13 #elif defined(BOOST_COMP_MSVC)
14  #if BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(12, 0, 0)
15    std::cout << "Incomplete C++ 11 support" << '\n';
16  #else
17    std::cout << "Most C++ 11 features supported" << '\n';
18  #endif
19 #endif
20 }
```

在上面的代码中，我们使用`BOOST_VERSION_NUMBER`宏来构建与当前版本的 GNU 或 Microsoft Visual C++编译器进行比较的版本。如果 GNU 编译器版本小于 4.8.1 或 Microsoft Visual Studio C++编译器版本小于 12.0，我们会打印出对 C++11 的支持可能不完整。

在本节的最后一个示例中，我们使用`boost/config.hpp`中的宏来打印编译器、平台和运行时库的名称（第 6、7 和 8 行）。我们还使用`boost/version.hpp`中定义的两个宏来打印所使用的 Boost 版本，一个作为字符串（第 10 行），一个作为数值（第 11 行）：

2.19 清单：使用配置信息宏

```cpp
 1 #include <boost/config.hpp>
 2 #include <boost/version.hpp>
 3 #include <iostream>
 4 
 5 void buildEnvInfo() {
 6   std::cout << "Compiler: " << BOOST_COMPILER << '\n'
 7             << "Platform: " << BOOST_PLATFORM << '\n'
 8             << "Library: " << BOOST_STDLIB << '\n';
 9
10   std::cout << "Boost version: " << BOOST_LIB_VERSION << '['
11                             << BOOST_VERSION << ']' << '\n';
12 }
```

# 自测问题

对于多项选择题，选择所有适用的选项：

1.  使用`boost::swap`而不是`std::swap`的优点是什么？

a. 没有真正的优势

b. `boost::swap`会调用传递类型提供的交换重载

c. `boost::swap`比`std::swap`更快

d. `boost::swap`不会抛出异常

1.  您能在单个调用中将访问者应用于多个变体参数吗？（提示：您可能需要查阅在线文档）

a. 是的。访问者只能应用于一个或两个变体参数

b. 是的。访问者可以应用于一个或多个参数

c. 不。成员操作符只接受一个变体参数

d. 以上都不是

1.  以下是否是有效的编译时断言？

`BOOST_STATIC_ASSERT(x == 0); // x is some variable`

a. 是的，只要`x`是整数类型

b. 是的，只要`x`声明为`const static`数值变量

c. 不，`x`是一个变量，其值在编译时无法知道

d. 只有涉及`sizeof`的表达式在`BOOST_STATIC_ASSERT`中是有效的

1.  当我们说类型`X`是 POD 类型时，我们是什么意思？

a. `X`没有用户定义的构造函数或析构函数

b. 通过按位复制其内存布局可以复制`X`

c. `X`没有用户定义的复制构造函数或复制赋值运算符

d. 以上所有

1.  在类型为`boost::variant<std::string, double>`的默认构造对象中存储的类型和值是什么？

a. 类型是`const char*`，值是`NULL`

b. 类型是`double`，值是`0.0`

c. 类型是`std::string`，值是默认构造的`std::string`

d. 类型是`boost::optional<double>`，值为空

1.  查看 Boost 库在线文档中 Boost.Optional 的参考资料。如果在一个空的`optional`对象上调用`get`和`get_ptr`方法会发生什么？

a. 两者都会抛出`boost::empty_optional`异常

b. `get`抛出异常，而`get_ptr`返回空指针

c. `get`会断言，而`get_ptr`会返回空指针

d. `get`和`get_ptr`都会断言

# 总结

本章是对几个 Boost 库的快速介绍，这些库帮助您完成重要的编程任务，如解析命令行、创建类型安全的变体类型、处理空值和执行编译时检查。

希望您已经欣赏了 Boost 库中的多样性以及它们为您的代码提供的表达能力。在这个过程中，您也会更加熟悉使用 Boost 库编译代码并根据需要链接到适当的库。

在下一章中，我们将看看如何使用各种 Boost 智能指针的变种以确定性地管理堆内存和其他资源，以及在异常安全的方式下。

# 参考资料

奇怪的递归模板模式：[`en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Curiously_Recurring_Template_Pattern`](https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Curiously_Recurring_Template_Pattern)
