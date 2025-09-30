

# 第九章：命名参数、方法链和构建者模式

在本章中，我们将探讨一个解决非常常见的 C++ 问题的方案：参数过多。不，我们不是在谈论 C++ 程序员之间的争论，比如是否在行尾或下一行的开头放置花括号（我们对此问题没有解决方案）。这是关于 C++ 函数参数过多的问题。如果你长时间维护过大型 C++ 系统，你一定见过这种情况——函数开始时声明简单，随着时间的推移，为了支持新功能，常常会添加额外的参数，这些参数通常是默认值。

本章节将涵盖以下主题：

+   长函数声明有什么问题？

+   那么，替代方案是什么呢？

+   使用命名参数习语的缺点是什么？

+   如何将命名参数习语进行泛化？

# 技术要求

这里是示例代码的链接：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter09`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter09)。

这里是 Google Benchmark 库的链接：[`github.com/google/benchmark`](https://github.com/google/benchmark)（请参阅*第四章*，*从简单到微妙*，获取安装说明）。

参数的问题

每个在某个时候参与过足够大的 C++ 系统开发的人都必须向函数添加参数。为了避免破坏现有代码，新参数通常会被赋予一个默认值，这通常保留了旧的功能。第一次这样做效果很好，第二次还可以，然后就必须在每次函数调用时开始计算参数。长函数声明也存在其他问题，如果我们想要更好的解决方案，花时间理解这些问题是值得的。我们在分析问题之后，再继续寻找解决方案。

## 许多参数有什么问题？

不论是代码一开始就设计了很多参数，还是随着时间的推移“有机地”增长，它都是脆弱的，容易受到程序员错误的侵害。主要问题是通常有很多相同类型的参数，它们可能会被错误地计数。考虑设计一个文明建设游戏——当玩家创建一个新城市时，会构建一个相应的对象。玩家可以选择在城市建设哪些设施，游戏会设置可用的资源选项：

```cpp
class City {
  public:
  enum center_t { KEEP, PALACE, CITADEL };
  City(size_t number_of_buildings,
       size_t number_of_towers,
       size_t guard_strength,
       center_t center,
       bool with_forge,
       bool with_granary,
       bool has_fresh_water,
       bool is_coastal,
       bool has_forest);
  ...
};
```

看起来我们已经处理了一切。为了开始游戏，让我们给每个玩家一个带有城堡、瞭望塔、两座建筑和一个卫兵公司的城市：

```cpp
City Capital(2, 1, 1, City::KEEP,
             false, false, false, false);
```

你能看出错误吗？幸运的是，编译器可以——参数不足。由于编译器不会让我们在这里犯错误，所以这不算什么大问题，我们只需要为`has_forest`添加参数即可。此外，假设游戏将城市放置在河边，所以现在它有水了：

```cpp
City Capital(2, 1, 1, City::KEEP,
             false, true, false, false, false);
```

这很简单……哎呀！现在我们有了河边的城市，但没有淡水（河里到底有什么？）。至少镇民们不会饿肚子，多亏了他们意外获得的免费粮仓。那个错误——将“真实”值传递给了错误的参数——将在调试过程中被发现。此外，这段代码相当冗长，我们可能会发现自己一遍又一遍地输入相同的值。也许游戏默认尝试在河流和森林附近放置城市？那么好吧：

```cpp
class City {
  public:
  enum center_t { KEEP, PALACE, CITADEL };
  City(size_t number_of_buildings,
       size_t number_of_towers,
       size_t guard_strength,
       enter_t center,
       bool with_forge,
       bool with_granary,
       bool has_fresh_water = true,
       bool is_coastal = false,
       bool has_forest = true);
  ...
};
```

现在，让我们回到我们第一次尝试创建城市的尝试——现在它编译了，少了一个参数，但我们并没有意识到我们误算了参数。游戏取得了巨大成功，在下一个更新中，我们得到了一个令人兴奋的新建筑——寺庙！当然，我们需要为构造函数添加一个新参数。在`with_granary`之后，与其他所有建筑一起，在地形特征之前添加它是有意义的。但然后我们必须编辑对`City`构造函数的每一个调用。更糟糕的是，由于没有寺庙的`false`对于程序员和编译器来说看起来与没有淡水的`false`完全一样，所以很容易出错。新参数必须插入正确的位置，在一长串看起来非常相似的价值中。

当然，现有的游戏代码没有寺庙也能工作，所以它们只在新更新的代码中需要。在不必要的情况下不干扰现有代码是有价值的。如果我们把新参数加在最后，并给它一个默认值，那么任何未更改的构造函数调用仍然会创建与之前完全相同的城市：

```cpp
class City {
  public:
  enum center_t { KEEP, PALACE, CITADEL };
  City(size_t number_of_buildings,
       size_t number_of_towers,
       size_t guard_strength,
       center_t center,
       bool with_forge,
       bool with_granary,
       bool has_fresh_water = true,
       bool is_coastal = false,
       bool has_forest = true,
       bool with_temple = false);
  ...
};
```

但现在，我们让短期便利主导我们的长期界面设计。参数不再有任何逻辑分组，从长远来看，错误的可能性更大。此外，我们没有完全解决不需要更改的代码更新问题——下一个版本添加了新的地形，沙漠，随之而来的是另一个参数：

```cpp
class City {
  public:
  enum center_t { KEEP, PALACE, CITADEL };
  City(size_t number_of_buildings,
       size_t number_of_towers,
       size_t guard_strength,
       center_t center,
       bool with_forge,
       bool with_granary,
       bool is_coastal = false,
       bool has_forest = true,
       bool with_temple = false,
       bool is_desert = false);
  ...
};
```

一旦开始，我们必须为所有在末尾添加的新参数提供默认值。此外，为了在沙漠中创建一个城市，我们还需要指定它是否有寺庙。没有逻辑上的理由说明为什么它必须是这样，但我们受制于界面演变的过程。当你考虑到我们使用的许多类型可以相互转换时，情况变得更糟：

```cpp
City Capital(2, 1, false, City::KEEP,
             false, true, false, false, false);
```

这创建了一个没有守卫公司的城市，而不是程序员在将第三个参数设置为`false`时预期的任何其他东西。甚至`enum`类型也不能提供完全的保护。你可能已经注意到，所有新的城市通常都是从城堡开始的，所以将其作为默认值也是有意义的：

```cpp
// Example 01
class City {
  public:
  enum center_t { KEEP, PALACE, CITADEL };
  City(size_t number_of_buildings,
       size_t number_of_towers,
       size_t guard_strength,
       center_t center = KEEP,
       bool with_forge = false,
       bool with_granary = false,
       bool has_fresh_water = true,
       bool is_coastal = false,
       bool has_forest = true,
       bool with_temple = false,
       bool is_desert = false);
  ...
};
```

现在，我们不必输入那么多参数，甚至可能避免一些错误（如果你没有写参数，你就不能写错顺序）。但是，我们也可以创建新的参数：

```cpp
City Capital(2, 1, City::CITADEL);
```

我们刚刚雇佣的两个守卫公司（因为`CITADEL`的数值是`2`）将发现自己在这个低级的堡垒（我们本想改变但未改变）的空间非常紧张。C++11 的`enum class`提供了更好的保护，因为每个都是不同类型，无需转换为整数，但总体问题仍然存在。正如我们所见，将大量值作为单独的参数传递给 C++函数有两个问题。首先，它创建了非常长的声明和函数调用，容易出错。其次，如果我们需要添加一个值或更改参数的类型，需要编辑大量的代码。这两个问题的解决方案在 C++创建之前就已经存在；它来自 C——使用聚合，即结构体——将许多值组合成一个参数。

## 聚合参数

使用聚合参数，我们创建一个包含所有值的结构体或类，而不是为每个值添加一个参数。我们不必局限于一个聚合；例如，我们的城市可能需要几个结构体，一个用于游戏设置的所有地形相关特性，另一个用于玩家直接控制的所有特性：

```cpp
struct city_features_t {
  size_t number_of_buildings = 1;
  size_t number_of_towers = 0;
  size_t guard_strength = 0;
  enum center_t { KEEP, PALACE, CITADEL };
  center_t center = KEEP;
  bool with_forge = false;
  bool with_granary = false;
  bool with_temple = false;
};
struct terrain_features_t {
  bool has_fresh_water = true;
  bool is_coastal = false;
  bool has_forest = true;
  bool is_desert = false;
};
class City {
  public:
  City(city_features_t city_features,
       terrain_features_t terrain_features);
  ...
};
```

这个解决方案有很多优点。首先，可以通过名称显式地分配值，非常明显（并且非常冗长）：

```cpp
city_features_t city_features;
city_features.number_of_buildings = 2;
city_features.center = city_features::KEEP;
...
terrain_features_t terrain_features;
terrain_features.has_fresh_water = true;
...
City Capital(city_features, terrain_features);
```

看到每个参数的值要容易得多，错误的可能性也小得多（另一种方法，结构体的聚合初始化，只是将问题从一种初始化移到另一种初始化）。如果我们需要添加一个新特性，大多数情况下我们只需向聚合类型之一添加一个新的数据成员。只有实际处理新参数的代码需要更新；所有只是传递参数并转发它们的函数和类都不需要做任何改变。我们甚至可以为聚合类型提供默认值，为所有参数提供默认值，就像我们在上一个例子中所做的那样。

这总的来说是解决具有许多参数的函数问题的优秀解决方案。然而，它有一个缺点：聚合必须显式创建和初始化，一行一行地来。这在许多情况下都很好，特别是当这些类和结构体代表我们将长期保留的状态变量时。但是，当纯粹用作参数容器时，它们会创建不必要的冗长代码，从聚合变量必须有一个名字的事实开始。我们实际上并不需要这个名字，因为我们只用它一次来调用函数，但我们必须想出一个。使用临时变量可能会很有诱惑力：

```cpp
struct city_features_t {
  size_t number_of_buildings = 1;
  size_t number_of_towers = 0;
  size_t guard_strength = 0;
  enum center_t { KEEP, PALACE, CITADEL };
  center_t center = KEEP;
  bool with_forge = false;
  bool with_granary = false;
  bool with_temple = false;
};
struct terrain_features_t {
  bool has_fresh_water = true;
  bool is_coastal = false;
  bool has_forest = true;
  bool is_desert = false;
};
City Capital({2, 1, 0, KEEP, true, false, false},
             {true, false, false, true});
```

这可以工作，但它又把我们带回到了起点；一个具有长列表的易于混淆的布尔参数的函数。我们遇到的基本问题是 C++函数具有位置参数，而我们试图找出一种方法来让我们能够通过名称指定参数。聚合对象主要作为副作用解决了这个问题，并且如果整体设计从将一组值收集到一个类中受益，你当然应该这样做。然而，作为一个专门针对命名参数问题的解决方案，没有其他更持久的理由将值组合在一起，它们就不够用了。我们现在将看到如何解决这个问题。

# C++中的命名参数

我们已经看到如何将逻辑上相关的值收集到聚合对象中给我们带来一个副作用；我们可以将这些值传递给函数，并通过名称而不是通过长列表中的顺序来访问它们。关键是*逻辑上相关*；除了它们碰巧在单个函数调用中一起使用之外，没有其他原因将值聚合在一起会创建不必要的对象，我们宁愿不发明这些名称。我们需要一种方法来创建临时聚合，最好是不需要显式的名称或声明。我们有一个解决方案来解决这个问题，C++中已经存在很长时间了；它只需要从不同角度的新视角，这正是我们现在要做的。

## 方法链式调用

方法链式调用是一种从 C++借用的技术；它起源于**Smalltalk**。其主要目的是消除不必要的局部变量。你可能已经使用了方法链式调用，尽管你可能没有意识到。考虑以下你可能多次编写过的代码：

```cpp
// Example 02
int i, j;
std::cout << i << j;
```

最后一行调用了插入操作符`<<`两次。第一次是在操作符左侧的对象上调用，即`std::cout`。第二次调用是在什么对象上？一般来说，操作符语法只是调用一个名为`operator<<()`的函数的一种方式。通常，这个特定的操作符是一个非成员函数，但`std::ostream`类也有几个成员函数重载，其中之一是用于`int`值的。所以，最后一行实际上是这样的：

```cpp
// Example 02-
std::cout.operator<<(i).operator<<(j);
```

第二次调用`operator<<()`是在第一次调用的结果上进行的。等效的 C++代码如下：

```cpp
// Example 02
auto& out1 = std::cout.operator(i);
out1.operator<<(j);
```

这就是方法链式调用——对一个方法函数的调用返回下一个方法应该调用的对象。在`std::cout`的情况下，成员`operator<<()`返回对对象本身的引用。顺便说一下，非成员`operator<<()`做的是同样的事情，只是它没有隐含的参数`this`，而是将流对象作为显式的第一个参数。现在，我们可以使用方法链式调用消除显式命名的参数对象。

## 方法链式调用和命名参数

正如我们之前看到的，当聚合参数对象不是主要用于持有参数时，它们工作得很好；如果我们需要一个对象来持有系统的状态，并且我们随着时间的推移构建它并长时间保留它，我们也可以将这个对象作为单个参数传递给任何需要这个状态的函数。我们遇到的问题是只为单个函数调用创建聚合对象。另一方面，我们也不喜欢编写带有许多参数的函数。这尤其适用于通常大多数参数都设置为默认值，只有少数参数发生变化的函数。回到我们的游戏，假设每天的游戏时间都通过一个函数调用进行处理。

该函数在每个游戏日被调用一次，以推进城市的一天，并处理游戏可以生成的各种随机事件的后果：

```cpp
class City {
  ...
  void day(bool flood = false, bool fire = false,
    bool revolt = false, bool exotic_caravan = false,
    bool holy_vision = false, bool festival = false, ... );
  ...
};
```

在一段时间内可能会发生很多不同的事件，但很少有一天会同时发生多于一个事件。我们默认将所有参数设置为`false`，但这并没有真正帮助；这些事件没有特定的顺序，如果节日发生，即使它们仍然等于它们的默认值，也必须指定所有之前的参数。

一个聚合对象非常有帮助，但我们需要创建并命名它：

```cpp
class City {
  ...
  struct DayEvents {
    bool flood = false;
    bool fire = false;
    ...
  };
  void day(DayEvents events);
  ...
};
City capital(...);
City::DayEvents events;
events.fire = true;
capital.day(events);
```

我们希望只为`City::day()`的调用创建一个临时的`DayEvents`对象，但我们需要一种方法来设置其数据成员。这正是方法链发挥作用的地方：

```cpp
// Example 03
class City {
  ...
  class DayEvents {
    friend City;
    bool flood = false;
    bool fire = false;
    public:
    DayEvents() = default;
    DayEvents& SetFlood() { flood = true; return *this; }
    DayEvents& SetFire() { fire = true; return *this; }
    ...
  };
  void day(DayEvents events);
  ...
};
City capital(...);
capital.day(City::DayEvents().SetFire());
```

默认构造函数构建一个未命名的临时对象。在这个对象上，我们调用`SetFire()`方法。它修改对象并返回对该对象的引用。我们将创建并修改后的临时对象传递给`day()`函数，该函数处理当天的事件，显示城市被火焰更新的图形，播放火灾的声音，并更新城市的状态以反映一些建筑被火灾损坏。

由于每个`Set()`方法都返回对同一对象的引用，我们可以在方法链中调用多个方法来指定多个事件。当然，`Set()`方法也可以接受参数；例如，我们可以有一个方法可以设置事件标志，无论是从默认的`false`变为`true`，还是从`true`变为`false`：

```cpp
DayEvents& SetFire(bool value = true) {
  fire = value;
  return *this;
}
```

今天是我们城市的集市日，恰好与一个大型节日重合，所以国王除了已经驻扎在城里的两个卫兵公司外，还雇佣了一个额外的卫兵公司：

```cpp
City capital(...);
capital.day(City::DayEvents().
            SetMarket().
            SetFestival().
            SetGuard(3));
```

注意，对于所有没有发生的事件，我们无需指定任何内容。我们现在有了真正的命名参数；当我们调用一个函数时，我们可以按任何顺序、按名称传递参数，并且我们不需要明确提及任何希望保留为默认值的参数。这是 C++的命名参数习语。使用命名参数的调用当然比使用位置参数的调用更冗长；每个参数都必须明确写出名称。这正是练习的目的。另一方面，如果我们有一长串不需要更改的默认参数，我们就会处于优势地位。可能有人会问的一个问题是性能——我们有很多额外的函数调用，构造函数，以及每个命名参数的`Set()`调用，这肯定要花费一些代价。让我们找出它到底花费了多少代价。

## 命名参数习语的性能

显然，在命名参数调用中发生的事情更多，因为调用了更多的函数。另一方面，函数调用非常简单，如果它们在头文件中定义，并且整个实现对编译器可见，那么编译器没有理由不内联所有的`Set()`调用并消除不必要的临时变量。通过良好的优化，我们可能期望命名参数习语和显式命名的聚合对象有相似的性能。

测量单个函数调用性能的适当工具是微基准测试。我们使用 Google 微基准测试库来完成这个目的。虽然基准测试通常写在单个文件中，但如果我们希望我们调用的函数是外部的，而不是内联的，我们需要另一个源文件。另一方面，`Set()`方法应该肯定被内联，因此它们应该在头文件中定义。第二个源文件应包含我们使用命名或位置参数调用的函数的定义。这两个文件在链接时合并：

```cpp
$CXX named_args.C named_args_extra.C -g -O4 -I. \
  -Wall -Wextra -Werror -pedantic --std=c++14 \
  -I$GBENCH_DIR/include $GBENCH_DIR/lib/libbenchmark.a \
  -lpthread -lrt -lm -o named_args
```

我们可以将位置参数、命名参数和聚合参数进行比较。结果将取决于参数的类型和数量。例如，对于一个有四个布尔参数的函数，我们可以比较以下调用：

```cpp
// Example 04
// Positional arguments:
Positional p(true, false, true, false);
// Named arguments idiom:
Named n(Named::Options().SetA(true).SetC(true));
// Aggregate object:
Aggregate::Options options;
options.a = true;
options.c = true;
Aggregate a(options));
```

基准测试测量的性能将极大地取决于编译器和控制优化的选项。例如，这些数字是在 GCC12 上使用-O3 收集的：

```cpp
Benchmark                 Time  UserCounters...
BM_positional_const   0.233 ns  items_per_second=138.898G/s
BM_named_const        0.238 ns  items_per_second=134.969G/s
BM_aggregate_const    0.239 ns  items_per_second=135.323G/s
```

对于编译器能够内联和优化的显式命名的聚合对象，没有明显的性能损失。命名参数和位置参数的表现相似。请注意，函数调用的性能很大程度上取决于程序同时进行的其他操作，因为参数是在寄存器中传递的，而寄存器的可用性受上下文的影响。

在我们的基准测试中，我们使用了编译时常量作为参数值。这并不罕见，特别是对于指定某些选项的参数——在每次调用点，许多选项将是静态的、不变的（在其他代码中调用相同函数的地方，这些值是不同的，但在这行代码中，许多值在编译时就已经固定）。例如，如果我们有一个特殊的代码分支来处理游戏中的自然灾害，普通分支将始终调用我们的日模拟，将洪水、火灾和其他灾害标志设置为`false`。但是，同样经常的是，参数是在运行时计算的。这如何影响性能？让我们创建另一个基准测试，其中参数值是从向量中检索的，例如：

```cpp
// Example 04
std::vector<int> v; // Fill v with random values
size_t i = 0;
// ... Benchmark loop ...
const bool a = v[i++];
const bool b = v[i++];
const bool c = v[i++];
const bool d = v[i++];
if (i == v.size()) i = 0; // Assume v.size() % 4 == 0
Positional p(a, b, c, d); // Positional arguments
Named n(Named::Options().
  SetA(a).SetC(b).SetC(c).SetD(d)); // Named arguments
Aggregate::Options options;
options.a = a;
options.b = b;
options.c = c;
options.d = d;
Aggregate a(options)); // Aggregate object
```

顺便说一下，以这种方式缩短前面的代码是不明智的：

```cpp
Positional p(v[i++], v[i++], v[i++], v[i++]);
```

原因是参数评估的顺序是未定义的，所以哪个`i++`调用首先执行是任意的。如果`i`从`0`开始，这个调用最终可能调用`Positional(v[0], v[1], v[2], v[3])`或`Positional(v[3], v[2], v[1], v[0])`或任何其他排列。

在相同的编译器和硬件上，我们现在得到不同的数字：

```cpp
Benchmark                 Time  UserCounters...
BM_positional_vars     50.8 ns  items_per_second=630.389M/s
BM_named_vars          49.4 ns  items_per_second=647.577M/s
BM_aggregate_vars      45.8 ns  items_per_second=647.349M/s
```

从结果中我们可以看到，编译器完全消除了未命名临时对象（或命名聚合）的开销，并为将参数传递到函数的这三种方式生成了性能相似的代码。一般来说，编译器优化的结果难以预测。例如，CLANG 产生的结果显著不同（当大多数参数是编译时常量时，命名参数调用更快，但当它们是运行时值时则较慢）。

基准测试并不倾向于任何特定的参数传递机制。我们可以说，命名参数习语的表现不会比显式命名的聚合对象或等效的位置参数差，至少，如果编译器能够消除未命名的临时对象的话。在某些编译器上，如果函数有很多参数，命名参数可能会更快。如果优化没有发生，调用可能会稍微慢一些。另一方面，在许多情况下，函数调用的性能本身并不关键；例如，我们的城市只有在玩家建造时才会构建，游戏中的几次构建。每日事件在游戏日中只处理一次，这可能会占用几秒钟的真实时间，至少这样玩家可以享受与游戏的互动。另一方面，在性能关键代码中反复调用的函数应该尽可能内联，我们也可以期待在这种情况下参数传递的优化会更好。总的来说，我们可以得出结论，除非特定函数调用的性能对程序性能至关重要，否则不应该担心命名参数的开销。对于性能关键的调用，应该根据具体情况测量性能，并且命名参数可能比位置参数更快。

# 一般方法链

C++ 中方法链的应用不仅限于参数传递（我们已经在流式 I/O 的形式中看到了另一个应用，尽管它隐藏得很好）。在其他上下文中使用时，考虑一些更通用的方法链形式是有帮助的。

## 方法链与方法级联

“方法级联”这个术语在 C++ 的上下文中并不常见，而且有很好的理由——C++ 并不支持它。方法级联指的是在同一个对象上调用一系列方法。例如，在支持方法级联的 *Dart* 中，我们可以写出以下代码：

```cpp
var opt = Options();
opt.SetA()..SetB();
```

这段代码首先在 opt 对象上调用 `SetA()`，然后在该同一对象上调用 `SetB()`。等效的代码是这样的：

```cpp
var opt = Options();
opt.SetA()
opt.SetB();
```

但是等等，我们不是刚刚用 C++ 和我们的选项对象做了同样的事情吗？我们确实做了，但我们忽略了一个重要的区别。在方法链中，下一个方法应用于前一个方法的结果。这是 C++ 中的链式调用：

```cpp
Options opt;
opt.SetA().SetB();
```

这个链式调用等同于以下代码：

```cpp
Options opt;
Options& opt1 = opt.SetA();
Options& opt2 = opt1.SetB();
```

C++ 没有级联语法，但与级联等效的代码会是这样的：

```cpp
Options opt;
opt.SetA();
opt.SetB();
```

但这正是我们之前所做的事情，简短的形式也是一样的：

```cpp
Options opt;
opt.SetA().SetB();
```

在这种情况下，C++ 的级联之所以成为可能，是因为这些方法返回的是同一对象的引用。我们仍然可以说，等效的代码是这样的：

```cpp
Options opt;
Options& opt1 = opt.SetA();
Options& opt2 = opt1.SetB();
```

并且，这在技术上是真的。但是，由于方法的编写方式，我们还有额外的保证，即`opt`、`opt1`和`opt2`都指向同一个对象。方法级联始终可以通过方法链实现，但它限制了接口，因为所有调用都必须返回对`this`的引用。这种实现技术有时被称为 C++中有些笨拙的名称`*this`。更通用的链式调用能做什么呢？让我们看看。

## 通用方法链

如果链式方法不返回对对象的引用，它应该返回一个新的对象。通常，这个对象是同一类型的，或者至少是来自同一类层次结构的类型，如果方法是多态的。例如，让我们考虑一个实现数据集合的类。它有一个使用谓词（一个可调用对象，一个具有`operator()`返回`true`或`false`的对象）过滤数据的方法。它还有一个对集合进行排序的方法。这些方法中的每一个都会创建一个新的集合对象，并保持原始对象不变。现在，如果我们想过滤集合中的所有有效数据，并且假设我们有一个`is_valid`谓词对象，我们可以创建一个有效数据的排序集合：

```cpp
Collection c;
... store data in the collection ...
Collection valid_c = c.filter(is_valid);
Collection sorted_valid_c = valid_c.sort();
```

可以使用方法链消除中间对象：

```cpp
Collection c;
...
Collection sorted_valid_c = c.filter(is_valid).sort();
```

在阅读最后一节之后，应该清楚这是一个方法链的例子，而且比我们之前看到的更通用——每个方法都返回同一类型的对象，但不是同一个对象。在这个例子中，链式调用和级联调用的区别非常明显——级联调用会对原始集合进行过滤和排序（假设我们决定支持这样的操作）。

## 类层次结构中的方法链

当应用于类层次结构时，方法链会遇到一个特定的问题；假设我们的`sort()`方法返回一个排序后的数据集合，它是一个不同类型的对象，`SortedCollection`，这个对象是从`Collection`类派生出来的。它之所以是派生类，是因为排序后我们可以支持高效的搜索，因此`SortedCollection`类有一个基类没有的`search()`方法。我们仍然可以使用方法链，甚至可以在派生类上调用基类的方法，但这样做会中断链：

```cpp
// Example 05
class SortedCollection;
class Collection {
  public:
  Collection filter();
  // sort() converts Collection to SortedCollection.
  SortedCollection sort();
};
class SortedCollection : public Collection {
  public:
  SortedCollection search();
  SortedCollection median();
};
SortedCollection Collection::sort() {
  SortedCollection sc;
  ... sort the collection ...
  return sc;
}
Collection c;
auto c1 = c.sort().search().filter.median();
```

在这个例子中，链式调用工作了一段时间：我们能够对一个 `Collection` 进行排序，搜索结果，并过滤搜索结果。对 `sort()` 的调用作用于 `Collection` 并返回一个 `SortedCollection`。对 `search()` 的调用需要一个 `SortedCollection`，所以它按预期工作。对 `filter()` 的调用需要一个 `Collection`；该方法可以在派生类（如 `SortedCollection`）上调用，但返回的结果仍然是一个 `Collection`。然后链式调用中断：对 `median()` 的调用需要一个 `SortedCollection`，我们确实有，但 `filter()` 有效地将其转换回 `Collection`。没有办法告诉 `median()` 该对象实际上是一个 `SortedCollection`（除了强制类型转换）。

多态或虚函数在这里没有帮助；首先，我们需要在基类中为 `search()` 和 `median()` 定义虚函数，尽管我们并不打算在那里支持这些功能，因为只有派生类支持它们。我们不能声明它们为纯虚函数，因为我们使用 `Collection` 作为具体类，任何具有纯虚函数的类都是抽象类，因此不能实例化此类对象。我们可以使这些函数在运行时终止，但至少我们已经将编程错误的检测——在未排序的集合中进行搜索——从编译时移动到运行时。更糟糕的是，它甚至没有帮助：

```cpp
class SortedCollection;
class Collection {
  public:
  Collection filter();
  // Converts Collection to SortedCollection
  SortedCollection sort();
  virtual SortedCollection median();
};
class SortedCollection : public Collection {
  public:
  SortedCollection search();
  SortedCollection median() override;
};
SortedCollection Collection::sort() {
  SortedCollection sc;
  ... sort the collection ...
  return sc;
}
SortedCollection Collection::median() {
  cout << "Collection::median called!!!" << endl;
  abort();
  return {};     // Still need to return something
}
Collection c;
auto c1 = c.sort().search().filter().median();
```

这行不通，因为 `Collection::filter` 返回的是对象的一个副本，而不是对其的引用。它返回的对象是基类，`Collection`。如果在一个 `SortedCollection` 对象上调用它，它会从派生对象中剥离基类部分并返回。如果你认为将 `filter()` 也设为虚函数，并在派生类中重写它，可以以重写基类中每个函数为代价解决这个问题，那么你还有另一个惊喜——虚函数必须具有相同的返回类型，除了协变返回类型。对基类和派生类的引用是协变返回类型。而类本身，通过值返回的，则不是。

注意，如果我们返回对象引用，这个问题就不会发生。然而，我们只能返回调用对象的引用；如果我们在一个方法函数体中创建一个新对象并返回对其的引用，那么它将是一个指向临时对象的悬垂引用，该临时对象在函数返回时被删除。结果是未定义的行为（程序很可能会崩溃）。另一方面，如果我们总是返回原始对象的引用，我们最初就不能将其类型从基类更改为派生类。

C++ 解决这个问题的方法涉及使用模板和一个奇特的设计模式。事实上，这个词 *curious* 甚至出现在它的名字中——奇特重复的模板模式。这本书中有一个关于 CRTP 模式的完整章节。该模式在我们的案例中的应用相对直接——基类需要从其函数返回正确的类型，但无法做到，因为它不知道类型是什么。解决方案——将正确的类型作为模板参数传递给基类。当然，基类必须是一个基类模板才能使这起作用：

```cpp
template <typename T> class Collection {
  public:
  Collection() {}
  T filter(); // "*this" is really a T, not a Collection
  T sort() {
    T sc; // Create new sorted collection
    ...
    return sc;
  }
};
class SortedCollection :
  public Collection<SortedCollection> {
  public:
  SortedCollection search();
  SortedCollection median();
};
Collection<SortedCollection> c;
auto c1 = c.sort().search().filter().median();
```

这里的链与我们的初始示例类似：在 `Collection` 上调用 `sort()` 返回一个 `SortedCollection`，然后 `search()` 应用到 `SortedCollection` 上并返回另一个 `SortedCollection`，接着调用 `filter()`。这一次，基类 `Collection` 知道对象的真正类型，因为 `Collection` 本身是在派生对象类型上实例化的模板。因此，`filter()` 在任何集合上工作，但返回与初始集合相同类型的对象——在我们的例子中，两者都是 `SortedCollection` 对象。最后，`median()` 需要一个 `SortedCollection` 并获取它。

这是一个复杂的解决方案。虽然它有效，但其复杂性表明，当对象类型需要在链的中间改变时，应谨慎使用方法链。这有一个很好的理由——改变对象类型与调用一系列方法在本质上不同。这是一个更重大的事件，可能应该明确表示，并且新对象应该有自己的名称。

既然我们已经知道了方法链是什么，让我们看看它还能在哪些地方有用。

# 构造者模式

让我们几乎回到本章的开头，再次看看我们是如何向 C++ 函数传递命名参数的。我们不是使用带有许多参数的构造函数，而是选择了一个选项对象，其中每个参数都被明确命名：

```cpp
City GreensDale(City::Options()
  .SetCenter(City::KEEP)
  .SetBuildings(3)
  .SetGuard(1)
  .SetForge()
);
```

现在，让我们专注于 `Options` 对象本身，特别是我们构建它的方式。构造函数不会创建一个完成的对象（这只会把问题从 `City` 构造函数转移到 `Options` 构造函数）。相反，我们逐步构建对象。这是一个非常通用设计模式——构造者模式的一个特例。

## 构造者模式的基本

当我们决定一个对象不能仅通过构造函数独立构建成我们认为的完整状态时，就会使用构造者设计模式。相反，我们编写一个辅助类或构造者类来构建这些对象，并将它们交给程序。

你可能会问的第一个问题是“为什么？”——构造函数不是应该做这个工作吗？可能有几个原因。一个非常常见的原因是我们使用一个更通用的对象来表示一些更具体的数据集。例如，我们想要一个对象来存储斐波那契数字或素数，我们决定使用`std::vector`来存储它们。现在我们遇到了一个问题：向量具有 STL 提供的任何构造函数，但我们需要确保我们的向量中有正确的数字，而且我们不能编写一个新的构造函数。我们可以创建一个只包含素数的特殊类，但最终我们会得到很多类，它们在构造方式不同后，被以非常相似的方式使用。当我们使用向量处理所有这些数字时，这将是完全足够的。或者，我们可以在任何地方使用向量，并在程序需要时将正确的值写入其中。这也不是一个好主意：我们暴露并重复了大量我们希望封装和重用的底层代码（这就是我们为什么想要为每种数字编写一个构造函数的原因）。

解决方案是建造者模式：计算和存储数字的代码封装在一个建造者类中，但建造者创建的对象是一个通用的向量。例如，这里是一个斐波那契数字（以 1, 1 开始，后续每个数字是前两个数字之和的序列）的建造者：

```cpp
// Example 08
class FibonacciBuilder {
  using V = std::vector<unsigned long>;
  V cache_ { 1, 1, 2, 3, 5 };
  public:
  V operator()(size_t n) {
    while (cache_.size() < n) {     // Cache new numbers
      cache_.push_back(cache_[cache_.size() - 1] +
                       cache_[cache_.size() - 2]);
    }
    return V{cache_.begin(), cache_.begin() + n};
  }
};
```

假设我们的程序需要为某些算法（运行时值`n`可能变化的算法）获取前`n`个斐波那契数字的序列。我们可能需要这些数字多次，有时比之前更大的`n`值，有时比之前更小的值。我们只需要询问建造者：

```cpp
FibonacciBuilder b;
auto fib10 = b(10);
```

我们可以在程序中某个地方保留已知的值，但这会使程序复杂化，需要额外的跟踪工作。将这项工作移到一个仅用于构建斐波那契数字的类中会更好——一个建造者。缓存斐波那契数字值得吗？可能并不真的值得，但请记住，这是一个简洁的例子：如果我们需要，比如说，素数而不是斐波那契数字，重用已知值将非常有价值（但代码会更长）。

使用构建器的另一个常见原因是构建对象的代码可能过于复杂，不适合构造函数。通常，这会表现为如果我们尝试编写一个构造函数，我们必须传递给构造函数的大量参数。我们在本节开头构建`City`的`Options`参数的方式是这种复杂性的一个简单例子，其中`Options`对象充当其自己的构建器。构建器最有用的特定情况包括构建过程是条件性的，并且构建一个对象所需的数据在数量和类型上根据某些运行时变量而变化。再次强调，我们的`City`是这种情况的一个简单例子：没有单个`City`需要每个构造函数参数，但没有`Options`及其（简单）构建器，我们就必须为它们中的每一个提供一个参数。

我们为 Fibonacci 向量构建器看到的方法是 C++中 Builder 模式的常见变体；它并不非常令人兴奋，但它是有效的。在本章中，我们将看到一些实现 Builder 的替代方法。第一个方法是对我们构建`Options`的方式进行了泛化。

## 流畅构建器

我们构建`Options`对象的方式是通过方法链式调用。每个方法都朝着构建最终对象迈出小小的一步。这种方法的通用名称是流畅接口。虽然它不仅限于设计构建器，但流畅接口在 C++中主要作为构建复杂对象的一种方式而流行。

流畅构建器依赖于方法链式调用：构建器类的每个成员函数都贡献于正在构建的对象的构建，并返回构建器本身，以便工作可以继续。例如，这里有一个`Employee`类（可能用于某个工作场所数据库）：

```cpp
// Example 09
class Employee {
  std::string prefix_;
  std::string first_name_;
  std::string middle_name_;
  std::string last_name_;
  std::string suffix_;
  friend class EmployeeBuilder;
  Employee() = default;
  public:
  friend std::ostream& operator<<(std::ostream& out,
                                  const Employee& e);
};
```

我们稍后还会向这个类添加更多数据，但目前已经有了足够的数据成员，使得单个构造函数难以使用（太多相同类型的参数）。我们可以使用一个带有`Options`对象的构造函数，但，向前看，我们预计在对象构建过程中需要进行一些计算：我们可能需要验证某些数据，员工记录的其他部分可能是条件性的：两个不能同时设置的字段，一个字段默认值依赖于其他字段等。因此，让我们开始为这个类设计一个构建器。

`EmployeeBuilder`需要首先构建一个`Employee`对象，然后提供几个链式方法来设置对象的不同字段，最后将构建好的对象传递出去。可能涉及一些错误检查，或者影响多个字段的更复杂操作，但一个基本的构建器看起来像这样：

```cpp
// Example 09
class EmployeeBuilder {
  Employee e_;
  public:
  EmployeeBuilder& SetPrefix(std::string_view s) {
    e_.prefix_ = s; return *this;
  }
  EmployeeBuilder& SetFirstName(std::string_view s) {
    e_.first_name_ = s ; return *this;
  }
  EmployeeBuilder& SetMiddleName(std::string_view s) {
    e_.middle_name_ = s; return *this;
  }
  EmployeeBuilder& SetLastName(std::string_view s) {
    e_.last_name_ = s; return *this;
  }
  EmployeeBuilder& SetSuffix(std::string_view s) {
    e_.suffix_ = s; return *this;
  }
  operator Employee() {
    assert(!e_.first_name_.empty() &&
           !e_.last_name_.empty());
    return std::move(e_);
  }
};
```

在这个过程中，我们必须做出几个设计决策。首先，我们决定将构造函数 `Employee::Employee()` 设置为私有，这样只有像 `EmployeeBuilder` 这样的友元才能创建这些对象。这确保了部分初始化或无效的 `Employee` 对象不会出现在程序中：获取这些对象的唯一方式是通过构建器。这通常是更安全的选择，但有时我们需要能够默认构造对象（例如，在容器中使用或用于许多序列化/反序列化实现）。其次，构建器持有正在构建的对象，直到它可以被移动到调用者那里。这是一个常见的方法，但我们必须小心只使用每个构建器对象一次。我们还可以提供一种重新初始化构建器的方法；这通常在构建器需要执行大量计算，其结果被用于构建多个对象时进行。最后，为了构建一个 `Employee` 对象，我们首先需要构建一个构建器：

```cpp
Employee Homer = EmployeeBuilder()
  .SetFirstName("Homer")
  .SetMiddleName("J")
  .SetLastName("Simpson")
;
```

另一种常见的方法是提供一个静态函数 `Employee::create()` 来构建一个构建器；在这种情况下，构建器的构造函数被设置为私有，并允许友元访问。

如我们在关于 *类层次结构中的方法链* 的章节中提到的，链式方法不必都返回同一类的引用。如果我们的 `Employee` 对象具有内部结构，例如家庭地址、工作地点等单独的子记录，我们也可以对构建器采用更结构化的方法。

这里的目标是设计一个接口，使得客户端代码可以看起来像这样：

```cpp
// Example 09
Employee Homer = EmployeeBuilder()
  .SetFirstName("Homer")
  .SetMiddleName("J")
  .SetLastName("Simpson")
  .Job()
    .SetTitle("Safety Inspector")
    .SetOffice("Sector 7G")
  .Address()
    .SetHouse("742")
    .SetStreet("Evergreen Terrace")
    .SetCity("Springfield")
  .Awards()
    .Add("Remorseless Eating Machine")
;
```

要做到这一点，我们需要实现一个具有公共基类的构建器层次结构：

```cpp
// Example 09
class JobBuilder;
class AwardBuilder;
class AbstractBuilder {
  protected:
  Employee& e_;
  public:
  explicit AbstractBuilder(Employee& e) : e_(e) {}
  operator Employee() {
    assert(!e_.first_name_.empty() &&
           !e_.last_name_.empty());
      return std::move(e_);
  }
  JobBuilder Job();
  AddressBuilder Address();
  AwardBuilder Awards();
};
```

我们仍然从 `EmployeeBuilder` 开始，它构建 `Employee` 对象；其余的构建器持有对其的引用，并且通过在 `AbstractBuilder` 上调用相应的成员函数，我们可以为同一个 `Employee` 对象切换到不同类型的构建器。注意，虽然 `AbstractBuilder` 作为所有其他构建器的基类，但没有纯虚函数（或任何其他虚函数）：如我们之前所见，运行时多态在方法链中并不特别有用：

```cpp
class EmployeeBuilder : public AbstractBuilder {
  Employee employee_;
  public:
  EmployeeBuilder() : AbstractBuilder(employee_) {}
  EmployeeBuilder& SetPrefix(std::string_view s){
    e_.prefix_ = s; return *this;
  }
  …
};
```

要添加工作信息，我们切换到 `JobBuilder`：

```cpp
// Example 09
class JobBuilder : public AbstractBuilder {
  public:
  explicit JobBuilder(Employee& e) : AbstractBuilder(e) {}
  JobBuilder& SetTitle(std::string_view s) {
    e_.title_ = s; return *this;
  }
  …
  JobBuilder& SetManager(std::string_view s) {
    e_.manager_ = s; return *this;
  }
  JobBuilder& SetManager(const Employee& manager) {
    e_.manager_ = manager.first_name_ + " " +
                  manager.last_name_;
     return *this;
  }
  JobBuilder& CopyFrom(const Employee& other) {
    e_.manager_ = other.manager_;
    …
    return *this;
  }
};
JobBuilder AbstractBuilder::Job() {
  return JobBuilder(e_);
}
```

一旦我们有了 `JobBuilder`，它所有的链式方法都返回对自身的引用；当然，`JobBuilder` 也是一个 `AbstractBuilder`，因此我们可以在任何时候切换到另一个构建器类型，例如 `AddressBuilder`。注意，我们可以仅通过 `JobBuilder` 的前向声明来声明 `AbstractBuilder::Job()` 方法，但实现必须推迟到类型本身定义之后。

在这个例子中，我们也看到了 Builder 模式的灵活性，这仅使用构造函数很难实现。例如，有两种方式可以定义一个员工的经理：我们可以提供名字，或者使用另一个员工记录。此外，我们可以从另一个员工的记录中复制工作场所信息，并且仍然可以使用`Set`方法修改不同的字段。

其他如`AddressBuilder`之类的 Builder 可能类似。但也可能有非常不同的 Builder。例如，一个员工可以有任意数量的奖项：

```cpp
// Example 09
class Employee {
  … name, job, address, etc …
  std::vector<std::string> awards_;
};
```

相应的 Builder 需要反映它添加到对象中的信息的性质：

```cpp
// Example 09
class AwardBuilder : public AbstractBuilder {
  public:
  explicit AwardBuilder(Employee& e) : AbstractBuilder(e)
  {}
  AwardBuilder& Add(std::string_view award) {
    e_.awards_.emplace_back(award); return *this;
  }
};
AwardBuilder AbstractBuilder::Awards() {
  return AwardBuilder(e_);
}
```

我们可以多次调用`AwardBuilder::Add()`来构建特定的`Employee`对象。

这里是我们的 Builder 在起作用。注意，对于不同的员工，我们可以使用不同的方式来提供所需的信息：

```cpp
Employee Barry = EmployeeBuilder()
  .SetFirstName("Barnabas")
  .SetLastName("Mackleberry")
;
```

我们可以使用一个员工记录来将经理的名字添加到另一个员工中：

```cpp
Employee Homer = EmployeeBuilder()
  .SetFirstName("Homer")
  .SetMiddleName("J")
  .SetLastName("Simpson")
  .Job()
    .SetTitle("Safety Inspector")
    .SetOffice("Sector 7G")
    .SetManager(Barry) // Writes "Barnabas Mackleberry"
  .Address()
    .SetHouse("742")
    .SetStreet("Evergreen Terrace")
    .SetCity("Springfield")
  .Awards()
    .Add("Remorseless Eating Machine")
;
```

我们可以在员工之间复制就业记录：

```cpp
Employee Lenny = EmployeeBuilder()
  .SetFirstName("Lenford")
  .SetLastName("Leonard")
  .Job()
    .CopyFrom(Homer)
;
```

一些字段，如姓名和姓氏，是可选的，Builder 在可以访问之前会检查完成后的记录是否完整（参见上面的`AbstractBuilder::operator Employee()`）。其他字段，如名字后缀，也是可选的：

```cpp
Employee Smithers = EmployeeBuilder()
  .SetFirstName("Waylon")
  .SetLastName("Smithers")
  .SetSuffix("Jr") // Only when needed!
;
```

流畅的 Builder 模式是 C++中构建具有许多组件的复杂对象的有力模式，尤其是在对象的某些部分是可选的情况下。然而，对于包含大量高度结构化数据的对象，它可能会变得相当冗长。当然，也有其他选择。

## 隐式 Builder

我们已经看到过一个例子，其中使用了 Builder 模式而没有专门的 Builder 对象：所有用于命名参数传递的`Options`对象都充当它们自己的 Builder。我们将看到另一个版本，这个版本特别有趣，因为这里没有显式的 Builder 对象。这种设计特别适合构建嵌套层次数据，例如 XML 文件。我们将演示如何使用它来构建一个（非常简化的）HTML 编写器。

在这个设计中，HTML 记录将被相应的类表示：一个用于`<P>`标签的类，另一个用于`<UL>`标签，等等。所有这些类都详细说明了共同的基类`HTMLElement`：

```cpp
class HTMLElement {
  public:
  const std::string name_;
  const std::string text_;
  const std::vector<HTMLElement> children_;
  HTMLElement(std::string_view name, std::string_view text)
    : name_(name), text_(text) {}
  HTMLElement(std::string_view name, std::string_view text,
              std::vector<HTMLElement>&& children)
    : name_(name), text_(text),
      children_(std::move(children)) {}
  friend std::ostream& operator<<(std::ostream& out,
    const HTMLElement& element);
};
```

当然，HTML 元素还有很多其他内容，但我们必须保持简单。此外，我们的基元素允许无限嵌套：任何元素都可以有一个子元素向量，每个子元素也可以有子元素，依此类推。因此，元素的打印是递归的：

```cpp
std::ostream& operator<<(std::ostream& out,
                         const HTMLElement& element) {
  out << "<" << element.name_ << ">\n";
  if (!element.text_.empty())
    out << "  " << element.text_ << "\n";
  for (const auto& e : element.children_) out << e;
  out << "</" << element.name_ << ">" << std::endl;
  return out;
}
```

注意，为了添加子元素，我们必须以`std::vector`的形式提供它们，然后这些向量会被移动到`HTMLElement`对象中。右值引用意味着向量参数将是一个临时值或`std::move`的结果。然而，我们不会自己将子元素添加到向量中：这是特定元素（如`<P>`、`<UL>`等）的派生类的工作。特定的元素类可以在语法不允许时阻止添加子元素，以及强制对 HTML 元素字段的其他限制。

这些特定类看起来是什么样子？简单的类将看起来像这样：

```cpp
class HTML : public HTMLElement {
  public:
  HTML() : HTMLElement("html", "") {}
  HTML(std::initializer_list<HTMLElement> children) :
    HTMLElement("html", "", children) {};
};
```

这个`HTML`类代表`<html>`标签。像`Body`、`Head`、`UL`、`OL`等这样的类也是以完全相同的方式编写的。代表`<P>`标签的`P`类类似，但它不允许嵌套对象，因此它只有一个接受文本参数的构造函数。非常重要的一点是，这些类不添加任何数据成员；它们必须初始化基类`HTMLElement`对象，而不再初始化其他任何内容。如果你再次查看基类，原因应该很明显：我们存储了一个`HTMLElement`子对象向量。然而，它们被构造了——无论是作为`HTML`还是作为`UL`或其他任何东西——现在它们只是`HTMLElement`对象。任何额外的数据都将丢失。

你可能还会注意到，`HTMLElement`构造函数的向量参数是从`std::initializer_list`参数初始化的。这种转换是由编译器隐式地从构造函数参数列表中完成的：

```cpp
// Example 10
auto doc = HTML{
  Head{
    Title{"Mary Had a Little Lamb"}
  },
  Body{
    P{"Mary Had a Little Lamb"},
    OL{
      LI{"Its fleece was white as snow"},
      LI{"And everywhere that Mary went"},
      LI{"The lamb was sure to go"}
    }
  }
};
```

这个语句以调用使用两个参数构造`HTML`对象的方式开始。它们是`Head`和`Body`对象，但它们被编译器转换为`HTMLElement`并放入`std::initializer_list<HTMLElement>`中。然后这个列表被用来初始化一个向量，这个向量被移动到`HTML`对象的`children_`向量中。`Head`和`Body`对象本身也有子对象，其中一个（`OL`）有自己的子对象。

注意，如果你想要额外的构造函数参数，这会变得有点棘手，因为你不能将常规参数与初始化列表元素混合。这个问题在`LI`类中就出现了。根据我们到目前为止所学的内容，实现这个类的直接方法如下：

```cpp
//Example 10
class LI : public HTMLElement {
  public:
  explicit LI(std::string_view text) :
    HTMLElement("li", text) {}
  LI(std::string_view text,
     std::initializer_list<HTMLElement> children) :
    HTMLElement("li", text, children) {}
};
```

不幸的是，你不能用类似这样的方式调用这个构造函数：

```cpp
//Example 10
LI{"A",
  UL{
    LI{"B"},
    LI{"C"}
  }
}
```

显然，程序员想要的第一个参数是`"A"`，第二个参数（以及如果有，任何更多的参数）应该放入初始化列表中。但这行不通：通常，为了形成一个初始化列表，我们必须将元素序列放在大括号`{…}`中。只有当整个参数列表与初始化列表匹配时，这些大括号才能省略。对于不是初始化列表一部分的某些参数，我们必须明确指出：

```cpp
//Example 10
LI{"A",
  {UL{        // Notice { before UL!
    LI{"B"},
    LI{"C"}
  }}            // And closing } here
}
```

如果你不想写额外的花括号，你必须稍微改变构造函数：

```cpp
//Example 11
class LI : public HTMLElement {
  public:
  explicit LI(std::string_view text) :
    HTMLElement("li", text) {}
  template <typename ... Children>
  LI(std::string_view text, const Children&... children) :
    HTMLElement("li", text,
           std::initializer_list<HTMLElement>{children...})
  {}
};
```

我们不是使用初始化列表参数，而是使用参数包并将其显式转换为初始化列表（然后将其转换为向量）。当然，参数包将接受任意类型和任意数量的参数，而不仅仅是`HTMLElement`及其派生类，但转换到初始化列表将失败。如果你想遵循任何未失败实例化的模板都不应在其主体中产生编译错误的实践，你必须将参数包中的类型限制为从`HTMLElement`派生的类。这可以通过使用 C++20 概念轻松完成：

```cpp
// Example 12
class LI : public HTMLElement {
  public:
  explicit LI(std::string_view text) :
    HTMLElement("li", text) {}
  LI(std::string_view text,
     const std::derived_from<HTMLElement>
     auto& ... children) :
    HTMLElement("li", text,
           std::initializer_list<HTMLElement>{children...})
  {}
};
```

如果你没有使用 C++20 但仍想限制参数类型，你应该阅读本书的*第七章*，*SFINAE、概念和重载解析管理*。

使用参数包作为构建初始化列表的中间件，我们可以避免额外的花括号，并像这样编写我们的 HTML 文档：

```cpp
// Examples 11, 12
auto doc = HTML{
  Head{
    Title{"Mary Had a Little Lamb"}
  },
  Body{
    P{"Mary Had a Little Lamb"},
    OL{
      LI{"Its fleece was white as snow"},
      LI{"And everywhere that Mary went"},
      LI{"The lamb was sure to go"}
    },
    UL{
      LI{"It followed her to school one day"},
      LI{"Which was against the rules",
        UL{
          LI{"It made the children laugh and play"},
          LI{"To see a lamb at school"}
        }
      },
      LI{"And so the teacher turned it out"}
    }
  }
};
```

这确实看起来是构建器模式的应用：尽管最终的构建是通过调用`HTMLElement`构造函数的单个调用完成的，但该构造函数只是将已经构建的子元素向量移动到其最终位置。实际的构建是按照所有构建器都会做的步骤进行的。但是构建器对象在哪里？没有，不是明确存在的。构建器功能的一部分是由所有派生对象提供的，例如`HTML`、`UL`等。它们可能看起来像代表相应的 HTML 结构，但事实并非如此：在构建整个文档之后，我们只有`HTMLElement`对象。派生对象仅用于构建文档。其余的构建器代码是由编译器在执行参数包、初始化列表和向量之间的所有隐式转换时生成的。顺便说一句，任何半不错的优化编译器都会去掉所有中间副本，并将输入字符串直接复制到最终向量中，这些字符串将存储在那里。

当然，这是一个非常简化的例子，但在任何实际应用中，我们都需要使用 HTML 元素存储更多的数据，并为程序员提供一种初始化这些数据的方法。我们可以将隐式构建方法与流畅接口相结合，为所有`HTMLElement`对象提供一个简单的方法来添加可选值，例如样式、类型等。

到目前为止，你已经看到了三种不同的构建器设计：有一个“传统”的构建器，它有一个单一的构建器对象；有一个使用方法链的流畅构建器；还有一个使用许多小的构建器辅助对象并大量依赖编译器生成代码的隐式构建器。还有其他的设计，但它们大多是你已经学过的方法的变体和组合。我们对构建器模式的研究已经接近尾声。

# 摘要

再次，我们看到了 C++从现有语言中基本创建新语言的能力；C++没有命名函数参数，只有位置参数。这是核心语言的一部分。然而，我们能够以合理的方式扩展语言并添加对命名参数的支持，使用方法链技术。我们还探讨了方法链在命名参数习语之外的其它应用。

这些应用之一，流畅构建器，再次是创建新语言的练习：流畅接口的普遍力量在于它可以用来创建特定领域的领域特定语言，以执行某些数据上的指令序列。因此，流畅构建器可以用来允许程序员以特定领域熟悉的步骤序列来描述对象的构建。当然，还有隐式构建器，它（通过适当的缩进）甚至让 C++代码看起来有点像正在构建的 HTML 文档。

下一章介绍了本书中唯一的纯粹面向性能的习语。我们在几个章节中讨论了内存分配的性能成本及其对几个模式实现的影响。下一个习语，局部缓冲区优化，直接攻击问题，通过完全避免内存分配来解决问题。

# 问题

1.  为什么具有许多相同或相关类型参数的函数会导致脆弱的代码？

1.  如何通过聚合参数对象提高代码的可维护性和健壮性？

1.  命名参数习语是什么，它与聚合参数有何不同？

1.  方法链和级联有什么区别？

1.  建造者模式是什么？

1.  什么是流畅式接口，它在什么情况下被使用？
