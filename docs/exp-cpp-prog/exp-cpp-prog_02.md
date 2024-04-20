# 第二章：介绍 C++17 标准模板库

正如您所知，C++语言是 Bjarne Stroustrup 于 1979 年开发的产物。C++编程语言由国际标准化组织（ISO）标准化。最初的标准化于 1998 年发布，通常称为 C++98，下一个标准化 C++03 于 2003 年发布，主要是一个修复错误的版本，只有一个语言特性用于值初始化。2011 年 8 月，C++11 标准发布，对核心语言进行了多项增加，包括对标准模板库（STL）的一些重大有趣的更改；C++11 基本上取代了 C++03 标准。C++14 于 2014 年 12 月发布，带有一些新功能，后来，C++17 标准于 2017 年 7 月 31 日发布。在撰写本书时，C++17 是 C++编程语言的最新修订版。

本章需要支持 C++17 特性的编译器：gcc 版本 7 或更高。由于 gcc 版本 7 是撰写本书时的最新版本，本章将使用 gcc 版本 7.1.0。

本章将涵盖以下主题：

+   STL 概述

+   STL 架构

+   容器

+   迭代器

+   算法

+   函数对象

+   STL 容器

+   序列

+   关联

+   无序

+   适配器

让我们在接下来的章节逐个了解 STL 的主题。

# 标准模板库架构

C++标准模板库（STL）提供了现成的通用容器、可应用于容器的算法以及用于导航容器的迭代器。STL 是用 C++模板实现的，模板允许在 C++中进行通用编程。

STL 鼓励 C++开发人员专注于手头的任务，摆脱了编写低级数据结构和算法的束缚。STL 是一个经过时间考验的库，可以实现快速应用程序开发。

STL 是一项有趣的工作和架构。它的秘密公式是编译时多态性。为了获得更好的性能，STL 避免了动态多态性，告别了虚函数。总的来说，STL 有以下四个组件：

+   算法

+   函数对象

+   迭代器

+   容器

STL 架构将所有上述四个组件连接在一起。它具有许多常用的算法，并提供性能保证。有趣的是，STL 算法可以在不了解包含数据的容器的情况下无缝工作。这是由于迭代器提供了高级遍历 API，完全抽象了容器内部使用的底层数据结构。STL 广泛使用运算符重载。让我们逐个了解 STL 的主要组件，以便对 STL 的概念有一个良好的理解。

# 算法

STL 算法由 C++模板驱动；因此，相同的算法可以处理任何数据类型，独立于容器中数据的组织方式。有趣的是，STL 算法足够通用，可以使用模板支持内置和用户定义的数据类型。事实上，算法通过迭代器与容器交互。因此，算法关心的是容器支持的迭代器。然而，算法的性能取决于容器内部使用的数据结构。因此，某些算法仅适用于特定的容器，因为 STL 支持的每个算法都期望一种特定类型的迭代器。

# 迭代器

迭代器是一种设计模式，但有趣的是，STL 的工作开始得早于此

*四人帮*将他们与设计模式相关的工作发布给了软件社区。迭代器本身是允许遍历容器以访问、修改和操作容器中存储的数据的对象。迭代器以如此神奇的方式进行操作，以至于我们并不意识到或需要知道数据存储和检索的位置和方式。

以下图像直观地表示了一个迭代器：

![](img/540e553e-34b4-4deb-94d0-2b1671b4c429.png)

从前面的图像中，您可以理解每个迭代器都支持`begin()` API，它返回第一个元素的位置，`end()` API 返回容器中最后一个元素的下一个位置。

STL 广泛支持以下五种类型的迭代器：

+   输入迭代器

+   输出迭代器

+   前向迭代器

+   双向迭代器

+   随机访问迭代器

容器实现了迭代器，让我们可以轻松地检索和操作数据，而不需要深入了解容器的技术细节。

以下表格解释了这五种迭代器中的每一种：

| 迭代器的类型 | 描述 |
| --- | --- |
| 输入迭代器 |

+   它用于从指定的元素读取数据

+   它只能用于单次导航，一旦到达容器的末尾，迭代器将失效

+   它支持前置和后置递增运算符

+   它不支持递减运算符

+   它支持解引用

+   它支持`==`和`!=`运算符来与其他迭代器进行比较

+   `istream_iterator`迭代器是输入迭代器

+   所有的容器都支持这种迭代器

|

| 输出迭代器 |
| --- |

+   它用于修改指定的元素

+   它只能用于单次导航，一旦到达容器的末尾，迭代器将失效

+   它支持前置和后置递增运算符

+   它不支持递减运算符

+   它支持解引用

+   它不支持`==`和`!=`运算符

+   `ostream_iterator`、`back_inserter`、`front_inserter`迭代器是输出迭代器的例子

+   所有的容器都支持这种迭代器

|

| 前向迭代器 |
| --- |

+   它支持输入迭代器和输出迭代器的功能

+   它允许多次导航

+   它支持前置和后置递增运算符

+   它支持解引用

+   `forward_list`容器支持前向迭代器

|

| 双向迭代器 |
| --- |

+   它是一个支持双向导航的前向迭代器

+   它允许多次导航

+   它支持前置和后置递增运算符

+   它支持前置和后置递减运算符

+   它支持解引用

+   它支持`[]`运算符

+   `list`、`set`、`map`、`multiset`和`multimap`容器支持双向迭代器

|

| 随机访问迭代器 |
| --- |

+   可以使用任意偏移位置访问元素

+   它支持前置和后置递增运算符

+   它支持前置和后置递减运算符

+   它支持解引用

+   它是最功能完备的迭代器，因为它支持前面列出的其他类型迭代器的所有功能

+   `array`、`vector`和`deque`容器支持随机访问迭代器

+   支持随机访问的容器自然也支持双向和其他类型的迭代器

|

# 容器

STL 容器通常是动态增长和收缩的对象。容器使用复杂的数据结构来存储数据，并提供高级函数来访问数据，而不需要我们深入了解数据结构的复杂内部实现细节。STL 容器非常高效且经过时间考验。

每个容器使用不同类型的数据结构以高效地存储、组织和操作数据。尽管许多容器可能看起来相似，但它们在内部的行为却有所不同。因此，选择错误的容器会导致应用程序性能问题和不必要的复杂性。

容器有以下几种类型：

+   顺序

+   关联

+   容器适配器

容器中存储的对象是复制或移动的，而不是引用。我们将在接下来的章节中用简单而有趣的示例探索每种类型的容器。

# 函数对象

函数对象是行为类似于常规函数的对象。美妙之处在于函数对象可以替代函数指针。函数对象是方便的对象，可以让您扩展或补充 STL 函数的行为，而不会违反面向对象编程原则。

函数对象易于实现；您只需重载函数运算符。函数对象也被称为函数对象。

以下代码将演示如何实现一个简单的函数对象：

```cpp
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

template <typename T>
class Printer {
public:
  void operator() ( const T& element ) {
    cout << element << "t";
  }
};

int main () {
  vector<int> v = { 10, 20, 30, 40, 50 };

  cout << "nPrint the vector entries using Functor" << endl;

  for_each ( v.begin(), v.end(), Printer<int>() );

  cout << endl;

  return 0;
}
```

让我们快速使用以下命令编译程序：

```cpp
g++ main.cpp -std=c++17
./a.out
```

让我们检查程序的输出：

```cpp
Print the vector entries using Functor
10  20  30  40  50
```

希望您意识到函数对象是多么简单和酷。

# 序列容器

STL 支持一系列有趣的序列容器。序列容器以线性方式存储同类数据类型，可以按顺序访问。STL 支持以下序列容器：

+   数组

+   向量

+   列表

+   `forward_list`

+   双端队列

由于存储在 STL 容器中的对象只是值的副本，STL 期望用户定义的数据类型满足一定的基本要求，以便将这些对象存储在容器中。存储在 STL 容器中的每个对象都必须提供以下最低要求：

+   默认构造函数

+   一个复制构造函数

+   赋值运算符

让我们在以下小节中逐一探索序列容器。

# 数组

STL 数组容器是一个固定大小的序列容器，就像 C/C++内置数组一样，只是 STL 数组具有大小感知，并且比内置的 C/C++数组更智能。让我们通过一个示例了解 STL 数组：

```cpp
#include <iostream>
#include <array>
using namespace std;
int main () {
  array<int,5> a = { 1, 5, 2, 4, 3 };

  cout << "nSize of array is " << a.size() << endl;

  auto pos = a.begin();

  cout << endl;
  while ( pos != a.end() ) 
    cout << *pos++ << "t";
  cout << endl;

  return 0;
}
```

前面的代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++ main.cpp -std=c++17
./a.out 
```

程序的输出如下：

```cpp
Size of array is 5
1     5     2     4     3
```

# 代码演示

以下行声明了一个固定大小（`5`）的数组，并用五个元素初始化数组：

```cpp
array<int,5> a = { 1, 5, 2, 4, 3 };
```

一旦声明，大小就无法更改，就像 C/C++内置数组一样。`array::size()`方法返回数组的大小，不管初始化列表中初始化了多少个整数。`auto pos = a.begin()`方法声明了一个`array<int,5>`的迭代器，并将数组的起始位置赋给它。`array::end()`方法指向数组中最后一个元素之后的一个位置。迭代器的行为类似于或模仿 C++指针，对迭代器进行解引用会返回迭代器指向的值。迭代器的位置可以向前和向后移动，分别使用`++pos`和`--pos`。

# 数组中常用的 API

以下表格显示了一些常用的数组 API：

| **API** | **描述** |
| --- | --- |
| `at( int index )` | 这返回索引指向的位置存储的值。索引是从零开始的。如果索引超出数组的索引范围，此 API 将抛出`std::out_of_range`异常。 |
| `operator [ int index ]` | 这是一个不安全的方法，如果索引超出数组的有效范围，它不会抛出任何异常。这比`at`略快，因为此 API 不执行边界检查。 |
| `front()` | 这返回数组中的第一个元素。 |
| `back()` | 这返回数组中的最后一个元素。 |
| `begin()` | 这返回数组中第一个元素的位置 |
| `end()` | 这返回数组中最后一个元素的位置之后的一个位置 |
| `rbegin()` | 这返回反向开始位置，即返回数组中最后一个元素的位置 |
| `rend()` | 这返回反向结束位置，即返回数组中第一个元素之前的一个位置 |
| `size()` | 这返回数组的大小 |

数组容器支持随机访问；因此，给定一个索引，数组容器可以以*O(1)*或常量时间的运行复杂度获取一个值。

数组容器元素可以使用反向迭代器以反向方式访问：

```cpp
#include <iostream>
#include <array>
using namespace std;

int main () {

    array<int, 6> a;
    int size = a.size();
    for (int index=0; index < size; ++index)
         a[index] = (index+1) * 100;   

    cout << "nPrint values in original order ..." << endl;

    auto pos = a.begin();
    while ( pos != a.end() )
        cout << *pos++ << "t";
    cout << endl;

    cout << "nPrint values in reverse order ..." << endl;

    auto rpos = a.rbegin();
    while ( rpos != a.rend() )
    cout << *rpos++ << "t";
    cout << endl;

    return 0;
}
```

我们将使用以下命令来获取输出：

```cpp
./a.out
```

输出如下：

```cpp
Print values in original order ...
100   200   300   400   500   600

Print values in reverse order ...
600   500   400   300   200   100
```

# Vector

向量是一个非常有用的序列容器，它的工作方式与数组完全相同，只是向量可以在运行时增长和缩小，而数组的大小是固定的。然而，在数组和向量底层使用的数据结构是一个简单的内置 C/C++风格数组。

让我们看下面的例子更好地理解向量：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main () {
  vector<int> v = { 1, 5, 2, 4, 3 };

  cout << "nSize of vector is " << v.size() << endl;

  auto pos = v.begin();

  cout << "nPrint vector elements before sorting" << endl;
  while ( pos != v.end() )
    cout << *pos++ << "t";
  cout << endl;

  sort( v.begin(), v.end() );

  pos = v.begin();

  cout << "nPrint vector elements after sorting" << endl;

  while ( pos != v.end() )
    cout << *pos++ << "t";
  cout << endl;

  return 0;
}
```

可以编译上述代码，并使用以下命令查看输出：

```cpp
g++ main.cpp -std=c++17
./a.out
```

程序的输出如下：

```cpp
Size of vector is 5

Print vector elements before sorting
1     5     2     4     3

Print vector elements after sorting
1     2     3     4     5
```

# 代码演示

以下行声明了一个向量，并用五个元素初始化了向量：

```cpp
vector<int> v = { 1, 5, 2, 4, 3 };
```

然而，向量还允许使用`vector::push_back<data_type>( value )` API 将值附加到向量的末尾。`sort()`算法接受两个表示必须排序的数据范围的随机访问迭代器。由于向量在内部使用内置的 C/C++数组，就像 STL 数组容器一样，向量也支持随机访问迭代器；因此，`sort()`函数是一个运行时复杂度为对数的高效算法，即*O(N log2 (N))*。

# 常用的向量 API

以下表格显示了一些常用的向量 API：

| **API** | **描述** |
| --- | --- |
| `at ( int index )` | 返回存储在索引位置的值。如果索引无效，则会抛出`std::out_of_range`异常。 |
| `operator [ int index ]` | 返回存储在索引位置的值。这个函数比`at( int index )`更快，因为它不执行边界检查。 |
| `front()` | 返回向量中存储的第一个值。 |
| `back()` | 返回向量中存储的最后一个值。 |
| `empty()` | 如果向量为空，则返回 true，否则返回 false。 |
| `size()` | 返回向量中存储的值的数量。 |
| `reserve( int size )` | 这会保留向量的初始大小。当向量大小达到其容量时，插入新值需要向量调整大小。这使得插入消耗*O(N)*的运行复杂度。`reserve()`方法是对描述的问题的一种解决方法。 |
| `capacity()` | 返回向量的总容量，而大小是向量中实际存储的值。 |
| `clear()` | 这会清除所有的值。 |
| `push_back<data_type>( value )` | 这会在向量的末尾添加一个新值。 |

使用`istream_iterator`和`ostream_iterator`从向量中读取和打印会非常有趣和方便。以下代码演示了向量的使用：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
using namespace std;

int main () {
    vector<int> v;

    cout << "nType empty string to end the input once you are done feeding the vector" << endl;
    cout << "nEnter some numbers to feed the vector ..." << endl;

    istream_iterator<int> start_input(cin);
    istream_iterator<int> end_input;

    copy ( start_input, end_input, back_inserter( v ) );

    cout << "nPrint the vector ..." << endl;
    copy ( v.begin(), v.end(), ostream_iterator<int>(cout, "t") );
    cout << endl;

    return 0;
}
```

请注意，程序的输出被跳过，因为输出取决于您输入的输入。请随意在命令行上尝试这些指令。

# 代码演示

基本上，复制算法接受一系列迭代器，其中前两个参数表示源，第三个参数表示目标，这恰好是向量：

```cpp
istream_iterator<int> start_input(cin);
istream_iterator<int> end_input;

copy ( start_input, end_input, back_inserter( v ) );
```

`start_input`迭代器实例定义了一个从`istream`和`cin`接收输入的`istream_iterator`迭代器，而`end_input`迭代器实例定义了一个文件结束分隔符，默认情况下是一个空字符串(`""`)。因此，输入可以通过在命令行输入终端中键入`""`来终止。

同样，让我们了解下面的代码片段：

```cpp
cout << "nPrint the vector ..." << endl;
copy ( v.begin(), v.end(), ostream_iterator<int>(cout, "t") );
cout << endl;
```

复制算法用于将向量中的值逐个复制到`ostream`中，并用制表符(`t`)分隔输出。

# 向量的缺陷

每个 STL 容器都有自己的优点和缺点。没有一个 STL 容器在所有情况下都表现更好。向量在内部使用数组数据结构，而在 C/C++中数组的大小是固定的。因此，当您尝试在向量中添加新值时，如果向量的大小已经达到了最大容量，那么向量将分配新的连续位置，可以容纳旧值和新值，并且在连续位置开始复制旧值。一旦所有数据元素都被复制，向量将使旧位置无效。

每当这种情况发生时，向量插入将需要*O(N)*的运行时复杂度。随着向量大小随时间增长，*O(N)*的运行时复杂度将导致性能相当糟糕。如果您知道所需的最大大小，可以预留足够的初始大小来克服这个问题。然而，并不是在所有情况下都需要使用向量。当然，向量支持动态大小和随机访问，在某些情况下具有性能优势，但您正在处理的功能可能并不真正需要随机访问，这种情况下列表、双端队列或其他某些容器可能更适合您。

# 列表

列表 STL 容器在内部使用双向链表数据结构。因此，列表仅支持顺序访问，在最坏的情况下在列表中搜索随机值可能需要*O(N)*的运行时复杂度。然而，如果您确定只需要顺序访问，列表确实提供了自己的好处。列表 STL 容器允许您以常数时间复杂度在末尾、前面或中间插入数据元素，即在最佳、平均和最坏的情况下都是*O(1)*的运行时复杂度。

以下图片展示了列表 STL 使用的内部数据结构：

![](img/90865fd4-3858-4fbc-bdd2-60c23f16c550.png)

让我们编写一个简单的程序来亲身体验使用列表 STL：

```cpp
#include <iostream>
#include <list>
#include <iterator>
#include <algorithm>
using namespace std;

int main () {

  list<int> l;

  for (int count=0; count<5; ++count)
    l.push_back( (count+1) * 100 );

  auto pos = l.begin();

  cout << "nPrint the list ..." << endl;
  while ( pos != l.end() )
    cout << *pos++ << "-->";
  cout << " X" << endl;

  return 0;
}
```

我相信到现在为止，您已经品尝到了 C++ STL 的优雅和强大之处。观察到语法在所有 STL 容器中保持不变，是不是很酷？您可能已经注意到，无论您使用数组、向量还是列表，语法都保持不变。相信我，当您探索其他 STL 容器时，也会有同样的感觉。

话虽如此，前面的代码是不言自明的，因为我们在其他容器中做了几乎相同的事情。

让我们尝试对列表进行排序，如下面的代码所示：

```cpp
#include <iostream>
#include <list>
#include <iterator>
#include <algorithm>
using namespace std;

int main () {

    list<int> l = { 100, 20, 80, 50, 60, 5 };

    auto pos = l.begin();

    cout << "nPrint the list before sorting ..." << endl;
    copy ( l.begin(), l.end(), ostream_iterator<int>( cout, "-->" ));
    cout << "X" << endl;

    l.sort();

    cout << "nPrint the list after sorting ..." << endl;
    copy ( l.begin(), l.end(), ostream_iterator<int>( cout, "-->" ));
    cout << "X" << endl; 

    return 0;
}
```

您注意到了`sort()`方法吗？是的，列表容器有自己的排序算法。列表容器支持自己版本的排序算法的原因是，通用的`sort()`算法需要一个随机访问迭代器，而列表容器不支持随机访问。在这种情况下，相应的容器将提供自己的高效算法来克服这个缺点。

有趣的是，列表支持的`sort`算法的运行时复杂度为*O(N log2 N)*。

# 列表中常用的 API

以下表格显示了 STL 列表中最常用的 API：

| **API** | **描述** |
| --- | --- |
| `front()` | 这返回列表中存储的第一个值 |
| `back() ` | 这返回列表中存储的最后一个值 |
| `size()` | 这返回列表中存储的值的数量 |
| `empty()` | 当列表为空时返回`true`，否则返回`false` |
| `clear()` | 这会清除列表中存储的所有值 |
| `push_back<data_type>( value )` | 这在列表的末尾添加一个值 |
| `push_front<data_type>( value )` | 这在列表的前面添加一个值 |
| `merge( list )` | 这将两个相同类型值的排序列表合并 |
| `reverse()` | 这会反转列表 |
| `unique()` | 这从列表中删除重复的值 |
| `sort()` | 这会对列表中存储的值进行排序 |

# Forward list

STL 的`forward_list`容器是建立在单向链表数据结构之上的；因此，它只支持向前导航。由于`forward_list`在内存和运行时方面每个节点消耗一个较少的指针，因此与列表容器相比，它被认为更有效。然而，作为性能优势的额外代价，`forward_list`必须放弃一些功能。

以下图表显示了`forward_list`中使用的内部数据结构：

![](img/149c8d91-7bb4-4194-9635-f9bf4a0ead5e.png)

让我们来探索以下示例代码：

```cpp
#include <iostream>
#include <forward_list>
#include <iterator>
#include <algorithm>
using namespace std;

int main ( ) {

  forward_list<int> l = { 10, 10, 20, 30, 45, 45, 50 };

  cout << "nlist with all values ..." << endl;
  copy ( l.begin(), l.end(), ostream_iterator<int>(cout, "t") );

  cout << "nSize of list with duplicates is " << distance( l.begin(), l.end() ) << endl;

  l.unique();

  cout << "nSize of list without duplicates is " << distance( l.begin(), l.end() ) << endl;

  l.resize( distance( l.begin(), l.end() ) );

  cout << "nlist after removing duplicates ..." << endl;
  copy ( l.begin(), l.end(), ostream_iterator<int>(cout, "t") );
  cout << endl;

  return 0;

}
```

输出可以通过以下命令查看：

```cpp
./a.out
```

输出将如下所示：

```cpp
list with all values ...
10    10    20    30    45    45    50
Size of list with duplicates is 7

Size of list without duplicates is 5

list after removing duplicates ...
10    20   30   45   50
```

# 代码演示

以下代码声明并初始化了`forward_list`容器，其中包含一些唯一的值和一些重复的值：

```cpp
forward_list<int> l = { 10, 10, 20, 30, 45, 45, 50 };
```

由于`forward_list`容器不支持`size()`函数，我们使用`distance()`函数来找到列表的大小：

```cpp
cout << "nSize of list with duplicates is " << distance( l.begin(), l.end() ) << endl;
```

以下`forward_list<int>::unique()`函数会移除重复的整数，只保留唯一的值：

```cpp
l.unique();
```

# `forward_list`容器中常用的 API

下表显示了常用的`forward_list` API：

| **API** | **描述** |
| --- | --- |
| `front()` | 这返回`forward_list`容器中存储的第一个值 |
| `empty()` | 当`forward_list`容器为空时返回 true，否则返回 false。 |
| `clear()` | 这会清除`forward_list`中存储的所有值。 |
| `push_front<data_type>( value )` | 这会将一个值添加到`forward_list`的前面。 |
| `merge( list )` | 这会合并两个排序的`forward_list`容器，其值类型相同 |
| `reverse()` | 这会颠倒`forward_list`容器 |
| `unique()` | 这会从`forward_list`容器中移除重复的值。 |
| `sort()` | 这会对`forward_list`中存储的值进行排序 |

让我们再来看一个例子，以更好地理解`forward_list`容器：

```cpp
#include <iostream>
#include <forward_list>
#include <iterator>
#include <algorithm>
using namespace std;

int main () {

    forward_list<int> list1 = { 10, 20, 10, 45, 45, 50, 25 };
    forward_list<int> list2 = { 20, 35, 27, 15, 100, 85, 12, 15 };

    cout << "nFirst list before sorting ..." << endl;
    copy ( list1.begin(), list1.end(), ostream_iterator<int>(cout, "t") );
    cout << endl; 

    cout << "nSecond list before sorting ..." << endl;
    copy ( list2.begin(), list2.end(), ostream_iterator<int>(cout, "t") );
    cout << endl;

    list1.sort();
    list2.sort();

    cout << "nFirst list after sorting ..." << endl;
    copy ( list1.begin(), list1.end(), ostream_iterator<int>(cout, "t") );
    cout << endl; 

    cout << "nSecond list after sorting ..." << endl;
    copy ( list2.begin(), list2.end(), ostream_iterator<int>(cout, "t") );
    cout << endl;    

    list1.merge ( list2 );

    cout << "nMerged list ..." << endl;
    copy ( list1.begin(), list1.end(), ostream_iterator<int>(cout, "t") );

    cout << "nMerged list after removing duplicates ..." << endl;
    list1.unique(); 
    copy ( list1.begin(), list1.end(), ostream_iterator<int>(cout, "t") );

    return 0;
}
```

上面的代码片段是一个有趣的例子，演示了`sort()`、`merge()`和`unique()` STL 算法的实际用途。

输出可以通过以下命令查看：

```cpp
./a.out
```

程序的输出如下：

```cpp
First list before sorting ...
10   20   10   45   45   50   25
Second list before sorting ...
20   35   27   15   100  85   12   15

First list after sorting ...
10   10   20   25   45   45   50
Second list after sorting ...
12   15   15   20   27   35   85   100

Merged list ...
10   10   12   15   15   20   20   25   27   35   45   45  50   85  100
Merged list after removing duplicates ...
10   12   15   20   25   27   35   45   50   85  100
```

输出和程序都很容易理解。

# Deque

deque 容器是一个双端队列，其使用的数据结构可以是动态数组或向量。在 deque 中，可以在前面和后面插入元素，时间复杂度为*O(1)*，而在向量中，插入元素在后面的时间复杂度为*O(1)*，而在前面的时间复杂度为*O(N)*。deque 不会遭受向量遭受的重新分配问题。然而，deque 具有向量的所有优点，只是在性能方面略优于向量，因为每一行都有几行动态数组或向量。

以下图表显示了 deque 容器中使用的内部数据结构：

![](img/69ccaa1c-1be2-4f91-89b2-fa780a1d54b0.png)

让我们编写一个简单的程序来尝试 deque 容器：

```cpp
#include <iostream>
#include <deque>
#include <algorithm>
#include <iterator>
using namespace std;

int main () {
  deque<int> d = { 10, 20, 30, 40, 50 };

  cout << "nInitial size of deque is " << d.size() << endl;

  d.push_back( 60 );
  d.push_front( 5 );

  cout << "nSize of deque after push back and front is " << d.size() << endl;

  copy ( d.begin(), d.end(), ostream_iterator<int>( cout, "t" ) );
  d.clear();

  cout << "nSize of deque after clearing all values is " << d.size() <<
endl;

  cout << "nIs the deque empty after clearing values ? " << ( d.empty()
? "true" : "false" ) << endl;

return 0;
}
```

输出可以通过以下命令查看：

```cpp
./a.out
```

程序的输出如下：

```cpp
Intitial size of deque is 5

Size of deque after push back and front is 7

Print the deque ...
5  10  20  30  40  50  60
Size of deque after clearing all values is 0

Is the deque empty after clearing values ? true
```

# deque 中常用的 API

下表显示了常用的 deque API：

| **API** | **描述** |
| --- | --- |
| `at ( int index )` | 这返回存储在索引位置的值。如果索引无效，则会抛出`std::out_of_range`异常。 |
| `operator [ int index ]` | 这返回存储在索引位置的值。与`at( int index )`相比，此函数不执行边界检查，因此速度更快。 |
| `front()` | 这返回 deque 中存储的第一个值。 |
| `back() ` | 这返回 deque 中存储的最后一个值。 |
| `empty()` | 如果 deque 为空则返回`true`，否则返回`false`。 |
| `size() ` | 这返回 deque 中存储的值的数量。 |
| `capacity()` | 这会返回 deque 的总容量，而`size()`返回 deque 中实际存储的值的数量。 |
| `clear()` | 这会清除所有值。 |
| `push_back<data_type>( value )` | 这会在 deque 的末尾添加一个新值。 |

# 关联容器

关联容器以排序的方式存储数据，与序列容器不同。因此，关联容器不会保留插入数据的顺序。关联容器在搜索值时非常高效，具有*O(log n)*的运行时复杂度。每次向容器添加新值时，如果需要，容器将重新排序内部存储的值。

STL 支持以下类型的关联容器：

+   集合

+   映射

+   多重集

+   多重映射

+   无序集合

+   无序多重集

+   无序映射

+   无序多重映射

关联容器将数据组织为键-值对。数据将根据键进行排序，以实现随机和更快的访问。关联容器有两种类型：

+   有序

+   无序

以下关联容器属于有序容器，因为它们以特定的方式排序。有序关联容器通常使用某种形式的**二叉搜索树**（**BST**）；通常使用红黑树来存储数据：

+   集合

+   映射

+   多重集

+   多重映射

以下关联容器属于无序容器，因为它们没有以任何特定方式排序，并且它们使用哈希表：

+   无序集合

+   无序映射

+   无序多重集

+   无序多重映射

让我们在以下小节中通过示例了解先前提到的容器。

# 集合

集合容器以排序的方式仅存储唯一的值。集合使用值作为键来组织值。集合容器是不可变的，也就是说，存储在集合中的值不能被修改；但是，值可以被删除。集合通常使用红黑树数据结构，这是一种平衡二叉搜索树。集合操作的时间复杂度保证为*O(log N)*。

让我们使用一个集合编写一个简单的程序：

```cpp
#include <iostream>
#include <set>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

int main( ) {
    set<int> s1 = { 1, 3, 5, 7, 9 };
    set<int> s2 = { 2, 3, 7, 8, 10 };

    vector<int> v( s1.size() + s2.size() );

    cout << "nFirst set values are ..." << endl;
    copy ( s1.begin(), s1.end(), ostream_iterator<int> ( cout, "t" ) );
    cout << endl;

    cout << "nSecond set values are ..." << endl;
    copy ( s2.begin(), s2.end(), ostream_iterator<int> ( cout, "t" ) );
    cout << endl;

    auto pos = set_difference ( s1.begin(), s1.end(), s2.begin(), s2.end(), v.begin() ); 
    v.resize ( pos - v.begin() );

    cout << "nValues present in set one but not in set two are ..." << endl;
    copy ( v.begin(), v.end(), ostream_iterator<int> ( cout, "t" ) );
    cout << endl; 

    v.clear();

    v.resize ( s1.size() + s2.size() );

    pos = set_union ( s1.begin(), s1.end(), s2.begin(), s2.end(), v.begin() );

    v.resize ( pos - v.begin() );

    cout << "nMerged set values in vector are ..." << endl;
    copy ( v.begin(), v.end(), ostream_iterator<int> ( cout, "t" ) );
    cout << endl; 

    return 0;
}
```

可以使用以下命令查看输出：

```cpp
./a.out
```

程序的输出如下：

```cpp
First set values are ...
1   3   5   7   9

Second set values are ...
2   3   7   8   10

Values present in set one but not in set two are ...
1   5   9

Merged values of first and second set are ...
1   2   3   5   7   8   9  10
```

# 代码演示

以下代码声明并初始化了两个集合`s1`和`s2`：

```cpp
set<int> s1 = { 1, 3, 5, 7, 9 };
set<int> s2 = { 2, 3, 7, 8, 10 };
```

以下行将确保向量有足够的空间来存储结果向量中的值：

```cpp
vector<int> v( s1.size() + s2.size() );
```

以下代码将打印`s1`和`s2`中的值：

```cpp
cout << "nFirst set values are ..." << endl;
copy ( s1.begin(), s1.end(), ostream_iterator<int> ( cout, "t" ) );
cout << endl;

cout << "nSecond set values are ..." << endl;
copy ( s2.begin(), s2.end(), ostream_iterator<int> ( cout, "t" ) );
cout << endl;
```

`set_difference()`算法将使用集合`s1`中仅存在而不在`s2`中的值填充向量`v`。迭代器`pos`将指向向量中的最后一个元素；因此，向量`resize`将确保向量中的额外空间被移除：

```cpp
auto pos = set_difference ( s1.begin(), s1.end(), s2.begin(), s2.end(), v.begin() ); 
v.resize ( pos - v.begin() );
```

以下代码将打印向量`v`中填充的值：

```cpp
cout << "nValues present in set one but not in set two are ..." << endl;
copy ( v.begin(), v.end(), ostream_iterator<int> ( cout, "t" ) );
cout << endl;
```

`set_union()`算法将合并集合`s1`和`s2`的内容到向量中，然后调整向量的大小以适应合并后的值：

```cpp
pos = set_union ( s1.begin(), s1.end(), s2.begin(), s2.end(), v.begin() );
v.resize ( pos - v.begin() );
```

以下代码将打印向量`v`中填充的合并值：

```cpp
cout << "nMerged values of first and second set are ..." << endl;
copy ( v.begin(), v.end(), ostream_iterator<int> ( cout, "t" ) );
cout << endl;
```

# 集合中常用的 API

以下表格描述了常用的集合 API：

| **API** | **描述** |
| --- | --- |
| `insert( value )` | 这会将一个值插入到集合中 |
| `clear()` | 这会清除集合中的所有值 |
| `size()` | 这会返回集合中存在的条目总数 |
| `empty()` | 如果集合为空，则会打印`true`，否则返回`false` |
| `find()` | 这会查找具有指定键的元素并返回迭代器位置 |
| `equal_range()` | 这会返回与特定键匹配的元素范围 |
| `lower_bound()` | 这会返回指向第一个不小于给定键的元素的迭代器 |
| `upper_bound()` | 这会返回指向第一个大于给定键的元素的迭代器 |

# 映射

映射按键组织值。与集合不同，映射每个值都有一个专用键。映射通常使用红黑树作为内部数据结构，这是一种平衡的 BST，可以保证在映射中搜索或定位值的*O(log N)*运行时效率。映射中存储的值根据键使用红黑树进行排序。映射中使用的键必须是唯一的。映射不会保留输入的顺序，因为它根据键重新组织值，也就是说，红黑树将被旋转以平衡红黑树高度。

让我们写一个简单的程序来理解映射的用法：

```cpp
#include <iostream>
#include <map>
#include <iterator>
#include <algorithm>
using namespace std;
int main ( ) {

  map<string, long> contacts;

  contacts["Jegan"] = 123456789;
  contacts["Meena"] = 523456289;
  contacts["Nitesh"] = 623856729;
  contacts["Sriram"] = 993456789;

  auto pos = contacts.find( "Sriram" );

  if ( pos != contacts.end() )
    cout << pos->second << endl;

  return 0;
}
```

让我们编译并检查程序的输出：

```cpp
g++ main.cpp -std=c++17
./a.out
```

输出如下：

```cpp
Mobile number of Sriram is 8901122334
```

# 代码漫游

以下行声明了一个映射，其中`string`名称作为键，`long`手机号作为存储在映射中的值：

```cpp
map< string, long > contacts;
```

以下代码片段添加了四个按名称组织的联系人：

```cpp
 contacts[ "Jegan" ] = 1234567890;
 contacts[ "Meena" ] = 5784433221;
 contacts[ "Nitesh" ] = 4567891234;
 contacts[ "Sriram" ] = 8901122334;
```

以下行将尝试在联系人映射中查找名为`Sriram`的联系人；如果找到`Sriram`，则`find()`函数将返回指向键值对位置的迭代器；否则返回`contacts.end()`位置：

```cpp
 auto pos = contacts.find( "Sriram" );
```

以下代码验证迭代器`pos`是否已达到`contacts.end()`并打印联系人号码。由于映射是一个关联容器，它存储`key=>value`对；因此，`pos->first`表示键，`pos->second`表示值：

```cpp
 if ( pos != contacts.end() )
 cout << "nMobile number of " << pos->first << " is " << pos->second << endl;
 else
 cout << "nContact not found." << endl;
```

# 映射中常用的 API

以下表格显示了常用的映射 API：

| **API** | **描述** |
| --- | --- |
| `at ( key )` | 如果找到键，则返回相应键的值；否则抛出`std::out_of_range`异常 |
| `operator[ key ]` | 如果找到键，则更新相应键的现有值；否则，将添加一个具有相应`key=>value`的新条目 |
| `empty()` | 如果映射为空，则返回`true`，否则返回`false` |
| `size()` | 返回映射中存储的`key=>value`对的数量 |
| `clear()` | 清除映射中存储的条目 |
| `count()` | 返回与给定键匹配的元素数量 |
| `find()` | 查找具有指定键的元素 |

# 多重集合

多重集合容器的工作方式与集合容器类似，只是集合只允许存储唯一的值，而多重集合允许存储重复的值。如你所知，在集合和多重集合容器的情况下，值本身被用作键来组织数据。多重集合容器就像一个集合；它不允许修改存储在多重集合中的值。

让我们写一个使用多重集合的简单程序：

```cpp
#include <iostream>
#include <set>
#include <iterator>
#include <algorithm>
using namespace std;

int main() {
  multiset<int> s = { 10, 30, 10, 50, 70, 90 };

  cout << "nMultiset values are ..." << endl;

  copy ( s.begin(), s.end(), ostream_iterator<int> ( cout, "t" ) );
  cout << endl;

  return 0;
}
```

可以使用以下命令查看输出：

```cpp
./a.out
```

程序的输出如下：

```cpp
Multiset values are ...
10 30 10 50 70 90
```

有趣的是，在前面的输出中，你可以看到多重集合包含重复的值。

# 多重映射

多重映射与映射完全相同，只是多重映射容器允许使用相同的键存储多个值。

让我们用一个简单的例子来探索多重映射容器：

```cpp
#include <iostream>
#include <map>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

int main() {
  multimap< string, long > contacts = {
    { "Jegan", 2232342343 },
    { "Meena", 3243435343 },
    { "Nitesh", 6234324343 },
    { "Sriram", 8932443241 },
    { "Nitesh", 5534327346 }
  };

  auto pos = contacts.find ( "Nitesh" );
  int count = contacts.count( "Nitesh" );
  int index = 0;

  while ( pos != contacts.end() ) { 
  cout << "\nMobile number of " << pos->first << " is " << 
  pos->second << endl; 
  ++index; 
  ++pos;
  if ( index == count )
     break; 
}
  return 0;
}
```

该程序可以编译，并且可以使用以下命令查看输出：

```cpp
g++ main.cpp -std=c++17

./a.out
```

程序的输出如下：

```cpp
Mobile number of Nitesh is 6234324343
Mobile number of Nitesh is 5534327346
```

# 无序集合

无序集合的工作方式与集合类似，只是这些容器的内部行为不同。集合使用红黑树，而无序集合使用哈希表。集合操作的时间复杂度为*O(log N)*，而无序集合操作的时间复杂度为*O(1)*；因此，无序集合比集合更快。

无序集合中存储的值没有特定的顺序，不像集合那样以排序的方式存储值。如果性能是标准，那么无序集合是一个不错的选择；然而，如果需要以排序的方式迭代值，那么集合是一个不错的选择。

# 无序映射

无序映射的工作方式类似于映射，只是这些容器的内部行为不同。映射使用红黑树，而无序映射使用哈希表。映射操作的时间复杂度为*O(log N)*，而无序映射操作的时间复杂度为*O(1)*；因此，无序映射比映射更快。

无序映射中存储的值没有任何特定的顺序，不像映射中的值按键排序。

# 无序多重集

无序多重集的工作方式类似于多重集，只是这些容器的内部行为不同。多重集使用红黑树，而无序多重集使用哈希表。多重集操作的时间复杂度为*O(log N)*，而无序多重集操作的时间复杂度为*O(1)*。因此，无序多重集比多重集更快。

无序多重集中存储的值没有任何特定的顺序，不像多重集中的值以排序的方式存储。如果性能是标准，无序多重集是一个不错的选择；然而，如果需要以排序的方式迭代值，则多重集是一个不错的选择。

# 无序多重映射

无序多重映射的工作方式类似于多重映射，只是这些容器的内部行为不同。多重映射使用红黑树，而无序多重映射使用哈希表。多重映射操作的时间复杂度为*O(log N)*，而无序多重映射操作的时间复杂度为*O(1)*；因此，无序多重映射比多重映射更快。

无序多重映射中存储的值没有任何特定的顺序，不像多重映射中的值按键排序。如果性能是标准，那么无序多重映射是一个不错的选择；然而，如果需要以排序的方式迭代值，则多重映射是一个不错的选择。

# 容器适配器

容器适配器通过组合而不是继承来适配现有容器以提供新的容器。

STL 容器不能通过继承来扩展，因为它们的构造函数不是虚拟的。在整个 STL 中，您可以观察到，虽然在运算符重载和模板方面都使用了静态多态性，但出于性能原因，动态多态性是有意避免的。因此，通过对现有容器进行子类化来扩展 STL 并不是一个好主意，因为容器类并没有设计成像基类一样行为，这会导致内存泄漏。

STL 支持以下容器适配器：

+   栈

+   队列

+   优先队列

让我们在以下小节中探索容器适配器。

# 栈

栈不是一个新的容器；它是一个模板适配器类。适配器容器包装现有容器并提供高级功能。栈适配器容器提供栈操作，同时隐藏对栈不相关的不必要功能。STL 栈默认使用双端队列容器；然而，在栈实例化期间，我们可以指示栈使用任何满足栈要求的现有容器。

双端队列、列表和向量满足栈适配器的要求。

栈遵循**后进先出**（**LIFO**）的原则。

# 栈中常用的 API

以下表格显示了常用的栈 API：

| **API** | **描述** |
| --- | --- |
| `top()` | 这将返回栈中的顶部值，即最后添加的值 |
| `push<data_type>( value )` | 这将提供的值推送到栈的顶部 |
| `pop()` | 这将从栈中移除顶部的值 |
| `size()` | 这将返回栈中存在的值的数量 |
| `empty()` | 如果栈为空，则返回`true`；否则返回`false` |

是时候动手了；让我们编写一个简单的程序来使用栈：

```cpp
#include <iostream>
#include <stack>
#include <iterator>
#include <algorithm>
using namespace std;

int main ( ) {

  stack<string> spoken_languages;

  spoken_languages.push ( "French" );
  spoken_languages.push ( "German" );
  spoken_languages.push ( "English" );
  spoken_languages.push ( "Hindi" );
  spoken_languages.push ( "Sanskrit" );
  spoken_languages.push ( "Tamil" );

  cout << "nValues in Stack are ..." << endl;
  while ( ! spoken_languages.empty() ) {
              cout << spoken_languages.top() << endl;
        spoken_languages.pop();
  }
  cout << endl;

  return 0;
}
```

程序可以编译，并且可以使用以下命令查看输出：

```cpp
g++ main.cpp -std=c++17

./a.out
```

程序的输出如下：

```cpp
Values in Stack are ...
Tamil
Kannada
Telugu
Sanskrit
Hindi
English
German
French
```

从前面的输出中，我们可以看到栈的 LIFO 行为。

# 队列

队列基于**先进先出**（FIFO）原则工作。队列不是一个新的容器；它是一个模板化的适配器类，它包装了一个现有的容器，并提供了队列操作所需的高级功能，同时隐藏了对队列无关的不必要功能。STL 队列默认使用双端队列容器；然而，我们可以在队列实例化期间指示队列使用满足队列要求的任何现有容器。

在队列中，新值可以添加到后面并从前面删除。双端队列、列表和向量满足队列适配器的要求。

# 队列中常用的 API

以下表格显示了常用的队列 API：

| **API** | **描述** |
| --- | --- |
| `push()` | 这在队列的后面追加一个新值 |
| `pop()` | 这删除队列前面的值 |
| `front()` | 这返回队列前面的值 |
| `back()` | 这返回队列的后面的值 |
| `empty()` | 当队列为空时返回`true`；否则返回`false` |
| `size()` | 这返回存储在队列中的值的数量 |

让我们在以下程序中使用队列：

```cpp
#include <iostream>
#include <queue>
#include <iterator>
#include <algorithm>
using namespace std;

int main () {
  queue<int> q;

  q.push ( 100 );
  q.push ( 200 );
  q.push ( 300 );

  cout << "nValues in Queue are ..." << endl;
  while ( ! q.empty() ) {
    cout << q.front() << endl;
    q.pop();
  }

  return 0;
}
```

程序可以编译，并且可以使用以下命令查看输出：

```cpp
g++ main.cpp -std=c++17

./a.out
```

程序的输出如下：

```cpp
Values in Queue are ...
100
200
300
```

从前面的输出中，您可以观察到值以它们被推入的相同顺序弹出，即 FIFO。

# 优先队列

优先队列不是一个新的容器；它是一个模板化的适配器类，它包装了一个现有的容器，并提供了优先队列操作所需的高级功能，同时隐藏了对优先队列无关的不必要功能。优先队列默认使用向量容器；然而，双端队列容器也满足优先队列的要求。因此，在优先队列实例化期间，您可以指示优先队列也使用双端队列。

优先队列以这样的方式组织数据，使得最高优先级的值首先出现；换句话说，值按降序排序。

双端队列和向量满足优先队列适配器的要求。

# 优先队列中常用的 API

以下表格显示了常用的优先队列 API：

| **API** | **描述** |
| --- | --- |
| `push()` | 这在优先队列的后面追加一个新值 |
| `pop()` | 这删除优先队列前面的值 |
| `empty()` | 当优先队列为空时返回`true`；否则返回`false` |
| `size()` | 这返回存储在优先队列中的值的数量 |
| `top()` | 这返回优先队列前面的值 |

让我们编写一个简单的程序来理解`priority_queue`：

```cpp
#include <iostream>
#include <queue>
#include <iterator>
#include <algorithm>
using namespace std;

int main () {
  priority_queue<int> q;

  q.push( 100 );
  q.push( 50 );
  q.push( 1000 );
  q.push( 800 );
  q.push( 300 );

  cout << "nSequence in which value are inserted are ..." << endl;
  cout << "100t50t1000t800t300" << endl;
  cout << "Priority queue values are ..." << endl;

  while ( ! q.empty() ) {
    cout << q.top() << "t";
    q.pop();
  }
  cout << endl;

  return 0;
}
```

程序可以编译，并且可以使用以下命令查看输出：

```cpp
g++ main.cpp -std=c++17

./a.out
```

程序的输出如下：

```cpp
Sequence in which value are inserted are ...
100   50   1000  800   300

Priority queue values are ...
1000  800   300   100   50
```

从前面的输出中，您可以观察到`priority_queue`是一种特殊类型的队列，它重新排列输入，使得最高值首先出现。

# 总结

在本章中，您学习了现成的通用容器、函数对象、迭代器和算法。您还学习了集合、映射、多重集和多重映射关联容器，它们的内部数据结构以及可以应用于它们的常见算法。此外，您还学习了如何使用各种容器与实际的代码示例。

在下一章中，您将学习模板编程，这将帮助您掌握模板的基本知识。
