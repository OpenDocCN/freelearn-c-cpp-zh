# 第九章。模板和常用容器

在 第七章，*动态内存分配*中，我们讨论了如果你想要创建一个在编译时大小未知的数组时，如何使用动态内存分配。动态内存分配的形式为 `int * array = new int[ number_of_elements ]`。

你还看到，使用 `new[]` 关键字进行动态分配需要你在稍后对数组调用 `delete[]`，否则你会有一个内存泄漏。必须以这种方式管理内存是件辛苦的工作。

有没有一种方法可以创建一个动态大小的数组，并且由 C++ 自动为你管理内存？答案是肯定的。C++ 有一些对象类型（通常称为容器），可以自动处理动态内存的分配和释放。UE4 提供了几种容器类型，用于在动态可调整大小的集合中存储你的数据。

存在两种不同的模板容器组。有 UE4 容器家族（以 `T*` 开头）和 C++ **标准模板库**（**STL**）容器家族。UE4 容器和 C++ STL 容器之间有一些差异，但这些差异并不大。UE4 容器集合的设计考虑了游戏性能。C++ STL 容器也表现良好，它们的接口稍微一致一些（API 的一致性是你更愿意看到的）。你使用哪个容器集合取决于你。然而，建议你使用 UE4 容器集合，因为它保证当你尝试编译代码时，不会出现跨平台问题。

# UE4 中的调试输出

本章（以及后续章节）中的所有代码都需要你在 UE4 项目中工作。为了测试 `TArray`，我创建了一个基本的代码项目，名为 `TArrays`。在 `ATArraysGameMode::ATArraysGameMode` 构造函数中，我正在使用调试输出功能将文本打印到控制台。

以下是代码的样式：

```cpp
ATArraysGameMode::ATArraysGameMode(const class FPostConstructInitializeProperties& PCIP) : Super(PCIP)
{
  if( GEngine )
  {
    GEngine->AddOnScreenDebugMessage( 0, 30.f, FColor::Red,  "Hello!" );
  }
}
```

如果你编译并运行此项目，当你开始游戏时，你将在游戏窗口的左上角看到调试文本。你可以使用调试输出在任何时候查看程序的内部结构。只需确保在调试输出时 `GEngine` 对象存在。前面代码的输出如下所示：

![UE4 中的调试输出](img/00133.jpeg)

# UE4 的 `TArray<T>`

`TArrays` 是 UE4 的动态数组版本。要了解 `TArray<T>` 变量的含义，你首先必须知道尖括号 `<T>` 选项代表什么。《T>` 选项意味着数组中存储的数据类型是变量。你想要一个 `int` 类型的数组吗？那么创建一个 `TArray<int>` 变量。一个 `double` 类型的 `TArray` 变量？创建一个 `TArray<double>` 变量。

所以，一般来说，无论哪里出现`<T>`，你都可以插入你选择的 C++类型。让我们继续，并通过示例来展示这一点。

## 使用 TArray<T>的示例

`TArray<int>`变量只是一个`int`类型的数组。`TArray<Player*>`变量将是一个`Player*`指针的数组。数组是动态可调整大小的，并且可以在创建后向数组的末尾添加元素。

要创建`TArray<int>`变量，你只需要使用正常的变量分配语法：

```cpp
TArray<int> array;
```

使用成员函数对`TArray`变量进行更改。有一些成员函数可以在`TArray`变量上使用。你需要了解的第一个成员函数是向数组添加值的方式，如下面的代码所示：

```cpp
array.Add( 1 );
array.Add( 10 );
array.Add( 5 );
array.Add( 20 );
```

这四行代码将在内存中产生数组值，如下面的图所示：

![使用 TArray<T>的示例](img/00134.jpeg)

当你调用`array.Add(number)`时，新数字会添加到数组的末尾。由于我们按照顺序将数字**1**、**10**、**5**和**20**添加到数组中，这就是它们将进入数组中的顺序。

如果你想在数组的开头或中间插入一个数字，这也是可能的。你只需要使用`array.Insert(value, index)`函数，如下面的代码行所示：

```cpp
array.Insert( 9, 0 );
```

此函数将数字**9**推入数组的**0**位置（即前面）。这意味着数组中的其余元素将向右偏移，如下面的图所示：

![使用 TArray<T>的示例](img/00135.jpeg)

我们可以使用以下代码行将另一个元素插入到数组的**2**位置：

```cpp
array.Insert( 30, 2 );
```

此函数将数组重新排列，如下面的图所示：

![使用 TArray<T>的示例](img/00136.jpeg)

### 小贴士

如果你将一个数字插入到数组中超出边界的位置，UE4 将会崩溃。所以请小心，不要这样做。

## 遍历 TArray

你可以使用两种方式遍历（遍历）`TArray`变量的元素：要么使用基于整数的索引，要么使用迭代器。我将在下面展示这两种方法。

### 普通 for 循环和方括号表示法

使用整数来索引数组元素有时被称为“普通”的`for`循环。可以使用`array[index]`来访问数组元素，其中`index`是元素在数组中的数值位置：

```cpp
for( int index = 0; index < array.Num(); index++ )
{
  // print the array element to the screen using debug message
  GEngine->AddOnScreenDebugMessage( index, 30.f, FColor::Red,  FString::FromInt( array[ index ] ) );
}
```

### 迭代器

你也可以使用迭代器逐个遍历数组的元素，如下面的代码所示：

```cpp
int count = 0;	// keep track of numerical index in array
for( TArray<int>::TIterator it = array.CreateIterator(); it; ++it  )
{
  GEngine->AddOnScreenDebugMessage( count++, 30.f, FColor::Red,  FString::FromInt( *it ) );
}
```

迭代器是数组的指针。迭代器可以用来检查或更改数组内的值。以下图示了一个迭代器的示例：

![迭代器](img/00137.jpeg)

迭代器的概念：它是一个外部对象，可以查看并检查数组的值。执行++操作将迭代器移动到检查下一个元素。

迭代器必须适合它正在遍历的集合。要遍历 `TArray<int>` 变量，你需要一个 `TArray<int>::TIterator` 类型的迭代器。

我们使用 `*` 来查看迭代器后面的值。在上面的代码中，我们使用 `(*it)` 从迭代器中获取整数值。这被称为解引用。解引用迭代器意味着查看其值。

在 `for` 循环的每次迭代结束时发生的 `++it` 操作会增加迭代器，将其移动到列表中的下一个元素。

将代码插入程序并尝试运行。以下是到目前为止我们创建的示例程序，使用 `TArray`（所有内容都在 `ATArraysGameMode::ATArraysGameMode()` 构造函数中）：

```cpp
ATArraysGameMode::ATArraysGameMode(const class FPostConstructInitializeProperties& PCIP) : Super(PCIP)
{
  TArray<int> array;
  array.Add( 1 );
  array.Add( 10 );
  array.Add( 5 );
  array.Add( 20 );
  array.Insert( 9, 0 );// put a 9 in the front
  array.Insert( 30, 2 );// put a 30 at index 2
  if( GEngine )
  {
    for( int index = 0; index < array.Num(); index++ )
    {
      GEngine->AddOnScreenDebugMessage( index, 30.f, FColor::Red,  FString::FromInt( array[ index ] ) );
    }
  }
}
```

上述代码的输出如下所示：

![迭代器](img/00138.jpeg)

## 在 TArray 中查找元素是否存在

在 UE4 中搜索容器非常简单。通常使用 `Find` 成员函数来完成。使用我们之前创建的数组，我们可以通过以下代码行找到值 `10` 的索引：

```cpp
int index = array.Find( 10 ); // would be index 3 in image above
```

# TSet<T>

`TSet<int>` 变量存储一组整数。`TSet<FString>` 变量存储一组字符串。`TSet` 和 `TArray` 之间的主要区别在于 `TSet` 不允许重复——`TSet` 内部的所有元素都保证是唯一的。`TArray` 变量不介意相同元素的重复。

要向 `TSet` 添加数字，只需调用 `Add`。以下是一个示例声明：

```cpp
TSet<int> set;
set.Add( 1 );
set.Add( 2 );
set.Add( 3 );
set.Add( 1 );// duplicate! won't be added
set.Add( 1 );// duplicate! won't be added
```

这就是 `TSet` 在以下图中的样子：

![TSet<T>](img/00139.jpeg)

`TSet` 中不允许有相同值的重复条目。注意 `TSet` 中的条目没有编号，就像在 `TArray` 中那样：你无法使用方括号来访问 `TSet` 数组中的条目。

## 遍历 TSet

为了查看 `TSet` 数组，你必须使用迭代器。你不能使用方括号表示法来访问 `TSet` 的元素：

```cpp
int count = 0;	// keep track of numerical index in set
for( TSet<int>::TIterator it = set.CreateIterator(); it; ++it )
{
  GEngine->AddOnScreenDebugMessage( count++, 30.f, FColor::Red,  FString::FromInt( *it ) );
}
```

## 交集 TSet

`TSet` 数组有两个特殊函数，而 `TArray` 变量没有。两个 `TSet` 数组的交集基本上是它们共有的元素。如果我们有两个 `TSet` 数组，如 `X` 和 `Y`，并且我们将它们相交，结果将是一个新的第三个 `TSet` 数组，它只包含它们共有的元素。查看以下示例：

```cpp
TSet<int> X;
X.Add( 1 );
X.Add( 2 );
X.Add( 3 );
TSet<int> Y;
Y.Add( 2 );
Y.Add( 4 );
Y.Add( 8 );
TSet<int> common = X.Intersect(Y); // 2
```

`X` 和 `Y` 之间的公共元素将是元素 `2`。

## 并集 TSet

从数学上讲，两个集合的并集就是将所有元素插入到同一个集合中。由于我们在这里讨论的是集合，所以不会有任何重复。

如果我们从上一个示例中的 `X` 和 `Y` 集合创建一个并集，我们将得到一个新的集合，如下所示：

```cpp
TSet<int> uni = X.Union(Y); // 1, 2, 3, 4, 8
```

## 查找 TSet

你可以通过在集合上使用 `Find()` 成员函数来确定一个元素是否在 `TSet` 中。如果元素存在于 `TSet` 中，`TSet` 将返回匹配查询的 `TSet` 中的条目指针；如果请求的元素不存在于 `TSet` 中，它将返回 `NULL`。

# TMap<T, S>

`TMap<T, S>` 创建了一种在 RAM 中的表格。`TMap` 表示将左侧的键映射到右侧的值。你可以将 `TMap` 视为一个两列的表格，其中键位于左侧列，值位于右侧列。

## 玩家物品清单

例如，假设我们想要创建一个 C++ 数据结构来存储玩家的物品清单。在表格的左侧（键），我们会有一个 `FString` 用于物品的名称。在右侧（值），我们会有一个 `int` 用于该物品的数量。

| 项目（键） | 数量（值） |
| --- | --- |
| 苹果 | 4 |
| 饼干 | 12 |
| 剑 | 1 |
| 防护盾 | 2 |

要在代码中实现这一点，我们只需使用以下代码：

```cpp
TMap<FString, int> items;
items.Add( "apples", 4 );
items.Add( "donuts", 12 );
items.Add( "swords", 1 );
items.Add( "shields", 2 );
```

一旦你创建了你的 `TMap`，你可以通过使用方括号并传递一个键到括号之间来访问 `TMap` 内的值。例如，在上面的代码中的 `items` 映射中，`items[ "apples" ]` 的值是 4。

### 提示

如果你使用方括号访问映射中尚不存在的键，UE4 将会崩溃，所以请小心！C++ STL 在这样做时不会崩溃。

## 迭代 TMap

为了迭代 `TMap`，你也需要使用迭代器：

```cpp
for( TMap<FString, int>::TIterator it = items.CreateIterator(); it; ++it )
{
  GEngine->AddOnScreenDebugMessage( count++, 30.f, FColor::Red,
  it->Key + FString(": ") + FString::FromInt( it->Value ) );
}
```

`TMap` 迭代器与 `TArray` 或 `TSet` 迭代器略有不同。`TMap` 迭代器包含一个 `Key` 和一个 `Value`。我们可以使用 `it->Key` 访问内部的键，以及使用 `it->Value` 访问 `TMap` 内部的值。

![迭代 TMap](img/00140.jpeg)

# 常用容器的 C++ STL 版本

我想介绍几个容器的 C++ STL 版本。STL 是标准模板库，大多数 C++ 编译器都附带它。我想介绍这些 STL 版本的原因是它们的行为与 UE4 中相同容器的行为略有不同。在某些方面，它们的行为非常好，但游戏程序员经常抱怨 STL 存在性能问题。特别是，我想介绍 STL 的 `set` 和 `map` 容器。

### 注意

如果你喜欢 STL 的接口但想要更好的性能，有一个由电子艺界（Electronic Arts）实现的 STL 库的知名重实现，称为 EASTL，你可以使用它。它提供了与 STL 相同的功能，但实现了更好的性能（基本上是通过消除边界检查等方式）。它可在 GitHub 上找到：[`github.com/paulhodge/EASTL`](https://github.com/paulhodge/EASTL)。

## C++ STL 集合

C++ set 是一些独特且排序的项。关于 STL `set` 的优点是它保持了集合元素的排序。快速排序一串值的一种简单方法是只是将它们放入同一个 `set` 中。`set` 会为您处理排序。

我们可以回到一个简单的 C++ 控制台应用程序来使用集合。要使用 C++ STL set，您需要包含 `<set>`，如下所示：

```cpp
#include <iostream>
#include <set>
using namespace std;

int main()
{
  set<int> intSet;
  intSet.insert( 7 );
  intSet.insert( 7 );
  intSet.insert( 8 );
  intSet.insert( 1 );

  for( set<int>::iterator it = intSet.begin(); it != intSet.end();  ++it )
  {
    cout << *it << endl;
  }
}
```

下面的代码是前一个代码的输出：

```cpp
1
7
8
```

重复的 `7` 被过滤掉，元素在 `set` 中按递增顺序保持。我们遍历 STL 容器元素的方式与 UE4 的 `TSet` 数组类似。`intSet.begin()` 函数返回一个指向 `intSet` 头部的迭代器。

停止迭代的条件是当 `it` 成为 `intSet.end()`。`intSet.end()` 实际上是 `set` 结束之后的一个位置，如下面的图所示：

![C++ STL set](img/00141.jpeg)

### 在 `<set>` 中查找元素

要在 STL `set` 中查找一个元素，我们可以使用 `find()` 成员函数。如果我们正在寻找的项目在 `set` 中，我们会得到一个指向我们正在搜索的元素的迭代器。如果我们正在寻找的项目不在 `set` 中，我们会得到 `set.end()`，如下所示：

```cpp
set<int>::iterator it = intSet.find( 7 );
if( it != intSet.end() )
{
  //  7  was inside intSet, and *it has its value
  cout << "Found " << *it << endl;
}
```

### 练习

询问用户一组三个独特的名字。逐个输入每个名字，然后按顺序打印它们。如果用户重复一个名字，则要求他们输入另一个名字，直到您得到三个。

### 解决方案

前一个练习的解决方案可以使用以下代码找到：

```cpp
#include <iostream>
#include <string>
#include <set>
using namespace std;
int main()
{
  set<string> names;
  // so long as we don't have 3 names yet, keep looping,
  while( names.size() < 3 )
  {
    cout << names.size() << " names so far. Enter a name" << endl;
    string name;
    cin >> name;
    names.insert( name ); // won't insert if already there,
  }
  // now print the names. the set will have kept order
  for( set<string>::iterator it = names.begin(); it !=  names.end(); ++it )
  {
    cout << *it << endl;
  }
}
```

## C++ STL map

C++ STL `map` 对象与 UE4 的 `TMap` 对象非常相似。它所做的唯一一件事是 `TMap` 不做的是在映射内部保持排序顺序。排序引入了额外的成本，但如果您希望您的映射是排序的，选择 STL 版本可能是一个不错的选择。

要使用 C++ STL `map` 对象，我们需要包含 `<map>`。在以下示例程序中，我们使用一些键值对填充项目映射：

```cpp
#include <iostream>
#include <string>
#include <map>
using namespace std;
int main()
{
  map<string, int> items;
  items.insert( make_pair( "apple", 12 ) );
  items.insert( make_pair( "orange", 1 ) );
  items.insert( make_pair( "banana", 3 ) );
  // can also use square brackets to insert into an STL map
  items[ "kiwis" ] = 44;

  for( map<string, int>::iterator it = items.begin(); it !=  items.end(); ++it )
  {
    cout << "items[ " << it->first << " ] = " << it->second <<  endl;
  }
}
```

这是前一个程序的输出：

```cpp
items[ apple ] = 12
items[ banana ] = 3
items[ kiwis ] = 44
items[ orange ] = 1
```

注意到 STL map 的迭代器语法与 `TMap` 的语法略有不同：我们使用 `it->first` 访问键，使用 `it->second` 访问值。

注意到 C++ STL 还在 `TMap` 上提供了一些语法糖；您可以使用方括号来插入 C++ STL `map`。您不能使用方括号来插入 `TMap`。

### 在 `<map>` 中查找元素

您可以使用 STL map 的 `find` 成员函数在映射中搜索一个 `<key, value>` 对。

### 练习

询问用户将五个项目和它们的数量输入到一个空的 `map` 中。按顺序打印结果。

### 解决方案

前一个练习的解决方案使用了以下代码：

```cpp
#include <iostream>
#include <string>
#include <map>
using namespace std;
int main()
{
  map<string, int> items;
  cout << "Enter 5 items, and their quantities" << endl;
  while( items.size() < 5 )
  {
    cout << "Enter item" << endl;
    string item;
    cin >> item;
    cout << "Enter quantity" << endl;
    int qty;
    cin >> qty;
    items[ item ] = qty; // save in map, square brackets
    // notation
  }

  for( map<string, int>::iterator it = items.begin(); it !=  items.end(); ++it )
  {
    cout << "items[ " << it->first << " ] = " << it->second <<  endl;
  }
}
```

在这个解决方案代码中，我们首先创建 `map<string, int> items` 来存储我们打算取的所有项目。询问用户一个项目和数量；然后我们使用方括号符号将 `item` 保存到 `items` 映射中。

# 概述

UE4 的容器和 C++ STL 家族的容器都非常适合存储游戏数据。通常，通过选择合适的数据容器，编程问题可以简化很多次。

在下一章中，我们将通过跟踪玩家携带的物品并将这些信息存储在`TMap`对象中，实际上开始编写我们游戏的开头部分。
