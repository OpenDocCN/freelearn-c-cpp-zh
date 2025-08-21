# 第六章：Bimap 和多索引容器

标准库有有序和无序的关联容器，用于存储对象并使用某个**键**高效地查找它们。键可以是文本类型、数字类型或一级对象。对于有序容器，如`std::set`和`std::map`，键必须具有明确定义的排序关系，允许对任何一组键进行排序。对于无序容器，必须能够为每个键计算一个整数哈希值，并且另外确定任何两个键是否相等，以某种等价定义。键表示查找的索引或标准，并且所有标准库关联容器都支持仅使用单个标准进行查找。换句话说，您不能使用多个独立标准高效地查找对象。

假设您有一个称为`PersonEntry`的类型来描述一个人。`PersonEntry`类型具有名称、年龄、电话号码等属性。您将在容器中存储多个`PersonEntry`类型的对象，并且在不同的时间，您可能需要使用不同的属性（如名称、年龄、电话号码等）查找`PersonEntry`对象。虽然标准库容器在涉及集合的许多常见任务时表现出色，但当您需要一个基于多个标准高效存储数据并搜索数据的数据结构时，它们表现得很糟糕。Boost 提供了一小部分通用容器，专门用于这种需求，其中我们在本章中研究了其中的两个。本章分为以下几个部分：

+   多标准查找的容器

+   Boost Multi-index 容器

+   Boost Bimap

# 多标准查找的容器

考虑一个`PersonEntry`类型的对象集合，如下面的代码所定义：

```cpp
 1 struct PersonEntry
 2 {
 3   std::string name;
 4   std::string phoneNumber;
 5   std::string city;
 6 };
```

这种类型的对象可能代表电话簿中的一个条目。您将如何设计一个数据结构，使您能够按名称查找一个人？我们可以使用`std::set`存储`PersonEntry`对象，为`PersonEntry`定义适当的排序关系。由于我们想按名称搜索，因此应该按名称定义排序关系：

```cpp
 1 bool operator<(const PersonEntry& left, 
 2                const PersonEntry& right) {
 3   return left.name< right.name;
 4 }
```

现在`std::set`仅存储唯一元素，任何两个具有相同名称的`PersonEntry`对象都将被视为重复。由于现实生活中常见同名，我们应该选择一个允许重复的容器，即`std::multiset`。然后我们可以使用以下代码插入元素并按名称查找它们：

**清单 6.1：使用多重映射进行查找**

```cpp
 1 #include <set>
 2 #include <iostream>
 3 #include <string>
 4
 5 struct PersonEntry {
 6   std::string name;
 7   std::string phoneNumber;
 8   std::string city;
 9 };
10
11 int main() {
12   std::multiset<PersonEntry> directory;
13   PersonEntry p1{"Arindam Mukherjee", "550 888 9999", "Pune"};
14   PersonEntry p2{"Arindam Mukherjee", "990 770 2458", 
15                  "Calcutta"};
16   directory.insert(p1);
17   directory.insert(p2);
18   auto it1 = directory.lower_bound(
19                 PersonEntry{ "Arindam Mukherjee", "", "" });
20   auto it2 = directory.upper_bound(
21                 PersonEntry{ "Arindam Mukherjee", "", "" });
22
23   while (it1 != it2) {
24     std::cout << "Found: [" <<it1->name << ", "
25               <<it1->phoneNumber << ", " <<it1->city << "]\n";
26     ++it1;
27   }
28 }
```

我们创建了两个具有相同名称的人的`PersonEntry`对象（第 13-15 行），并将它们插入`multiset`（第 16-17 行）。使用了 C++11 的新颖统一初始化语法来初始化对象。然后我们查找名称为`"Arindam Mukherjee"`的对象。在`multiset`中正确的方法是确定匹配元素的范围。`lower_bound`成员函数返回指向第一个匹配元素的迭代器（第 18-19 行）。`upper_bound`成员函数返回指向紧随最后一个匹配元素的第一个元素的迭代器（第 20-21 行）。如果没有匹配的元素，两者都返回指向第一个元素的迭代器，如果有匹配的元素，则返回指向匹配元素后面的第一个元素的迭代器。然后我们遍历 low, high)定义的范围，并打印所有匹配的元素。如果您注意到，我们构造了临时的`PersonEntry`对象来执行查找。现在，如果想要进行反向查找，即根据电话号码查找并找出它属于谁，这是完全合理的。在前面的安排中，我们该如何做呢？我们可以始终通过容器执行线性搜索，或者我们可以使用一个按电话号码排序的对象的字典中的`PersonEntry`对象的引用的单独容器；这两种方法都不是特别优雅或高效。这就是 Boost Multi-index 库的用武之地。

# Boost Multi-index 容器

Boost Multi-index 库实际上提供了一个称为`multi_index_container`的单个通用容器，用于存储对象和指定一个或多个索引的选项，通过这些索引可以查找对象。每个索引将在对象的潜在不同字段上使用不同的标准。索引被定义并指定为容器的模板参数，这确实使容器的声明有些令人生畏。但是，这最终使容器的实现更加紧凑，具有大量的编译时优化。事实上，使用这些容器最困难的部分实际上是确保它们的声明正确；因此让我们解构一下`PersonEntry`对象的这种容器的声明：

**列表 6.2：定义多索引容器**

```cpp
 1 #include <boost/multi_index_container.hpp>
 2 #include <boost/multi_index/indexed_by.hpp>
 3 #include <boost/multi_index/ordered_index.hpp>
 4 #include <boost/multi_index/identity.hpp>
 5
 6 using namespace boost::multi_index;
 7
 8 typedef ordered_non_unique<identity<PersonEntry>> by_person;
 9 typedef multi_index_container<PersonEntry,
10                       indexed_by<by_person>> directory_t;

```

在前面的片段中，我们创建了`PersonEntry`对象的`multi_index_container`的 typedef（第 9-10 行）。我们使用了之前定义的名为`person_index`的单个索引（第 8 行）。`person_index`是用于在容器中查找对象的索引类型。它被定义为`ordered_non_unique<identity<PersonEntry>>`。这意味着索引通过它们定义的排序关系保持`PersonEntry`对象的顺序，并允许重复（非唯一）。这个索引提供了与`std::multiset<PersonEntry>`相同的语义。现在，如果我们想要按电话号码查找`PersonEntry`对象，我们需要定义额外的索引：

**列表 6.3：定义多索引容器**

```cpp
 1 #include <boost/multi_index_container.hpp>
 2 #include <boost/multi_index/indexed_by.hpp>
 3 #include <boost/multi_index/ordered_index.hpp>
 4 #include <boost/multi_index/identity.hpp>
 5 #include <boost/multi_index/member.hpp>
 6 #include "PersonEntry.h"  // contains PersonEntry definition
 7 using namespace boost::multi_index;
 8
 9 typedef ordered_non_unique<member<PersonEntry, std::string,
10                           &PersonEntry::name>> by_name;
11 typedef ordered_unique<member<PersonEntry, std::string,
12                        &PersonEntry::phoneNumber>>by_phone;
13
14 typedef multi_index_container<PersonEntry,
15                             indexed_by<by_name,
16                                        by_phone>> directory_t;

```

在这里，我们定义了两种索引类型：一个名为`by_name`的索引类型，用于按名称字段查找对象，以及一个名为`phone_index`的第二索引类型，用于按电话号码查找（第 9-12 行）。我们使用`member`模板来指示我们希望基于`PersonEntry`的数据成员`name`或`phoneNumber`（类型为`std::string`）创建索引。

我们将`indexed_by`模板的一个特化作为参数传递给`multi_index_container`模板。我们想要启用的所有索引都作为这个特化的参数列出（第 15-16 行）。现在让我们看看这些类型是如何工作的。我们假设列表 6.3 中的所有头文件都已包含，并且列表 6.3 中定义的所有类型都在以下列表中可用：

**列表 6.4：使用 Boost Multi-index 容器**

```cpp
 1 int main()
 2 {
 3   directory_t phonedir;
 4   PersonEntry p1{"Arindam Mukherjee", "550 888 9999", "Pune"};
 5   PersonEntry p2{"Arindam Mukherjee", "990 770 2458", 
 6                  "Calcutta"};
 7   PersonEntry p3{"Ace Ventura", "457 330 1288", "Tampa"};
 8
 9   phonedir.insert(p1);
10   phonedir.insert(p2);
11   phonedir.insert(p3);
12 
13   auto iter = phonedir.find("Ace Ventura");
14   assert(iter != phonedir.end() && iter->city == "Tampa");
15
16   auto& ph_indx = phonedir.get<1>();
17   auto iter2 = ph_indx.find("990 770 2458");
18   assert(iter2 != ph_indx.end());
19   assert(iter2->city == "Calcutta");
20
21   for (auto& elem: ph_indx) {
22     std::cout << elem.name <<" lives in " << elem.city
23         << " and can be reached at "<< elem.phoneNumber
24         << '\n';
25   }
26 }
```

在这个例子中，我们创建了一个`PersonEntry`对象的多索引容器，按照列表 6.3 中定义的`name`和`phoneNumber`字段进行索引。我们插入了三个`PersonEntry`对象（第 8-10 行）。然后我们在容器中按名称进行查找（第 12-13 行）。容器的行为默认为第一个索引`by_name`（列表 6.3，第 9-10 行）。因此，调用`find`方法使用第一个索引（`by_name`）进行查找。要按电话号码查找，我们需要获取对第二个索引的引用。为此，我们使用`multi_index_container`的`get`成员模板，传递`1`，这是`by_phone`索引的从零开始的位置（第 15 行）。然后我们可以像在`std::set`上一样在返回的索引引用上调用方法（第 16-18 行）。我们甚至可以使用基于范围的 for 循环结构（第 21 行）或使用实际迭代器来遍历索引。

在前面的例子中，两个索引都是有序的，这要求它们基于的元素（`name`或`phoneNumber`字段）应该定义一个排序关系。在这种情况下，这两个字段都是`std::string`类型，因此排序关系是明确定义的。但是如果没有可用的排序关系，我们需要自己定义一个重载的`operator<`来进行排序。或者，我们可以定义一个函数对象来执行类型的两个元素之间的排序比较，并将其类型作为`member`模板的尾随参数传递。Boost Multi-index 的在线文档有更多详细信息。

如果为索引类型指定数字位置似乎不太理想，可以改用标签。这会稍微改变`by_phone`索引的声明，但可以使代码更易读。以下是如何为`phone_index`做到这一点：

```cpp
 1 struct phone_tag {};
 2 typedef ordered_unique< <tag<phone_tag>, member<PersonEntry, 
 3          std::string, &PersonEntry::phoneNumber>> by_phone;
 4
 5 auto& ph_indx = phonedir.get<phone_tag>(); 

```

在上面的片段中，我们定义了一个名为`phone_tag`的空结构，只是作为特定索引的标签（第 1 行）。然后我们定义了索引类型`by_phone`，作为`ordered_unique`模板的特化。`ordered_unique`模板的第一个参数指定用于检索此索引的标签（`phone_tag`）。`ordered_unique`的第二个模板参数是`member<PersonEntry, std::string, &PersonEntry::phoneNumber>`；它指定每个`PersonEntry`对象的`phoneNumber`成员将用作此索引的键，并且它的类型是`std::string`（第 2-3 行）。最后，我们通过调用`phonedir`的`get`成员模板来访问索引，但是传递的是标签`phone_tag`而不是数字索引（第 5 行）。

## 索引类型

`ordered_unique`和`ordered_non_unique`索引分别对应于`std::set`和`std::multiset`的语义。使用这些索引，不仅可以获得对数级别的查找和插入，还可以执行容器元素的有序遍历。如果您不关心有序遍历，还可以使用`hashed_unique`和`hashed_non_unique`索引，它们提供了出色的插入和查找性能（常数预期时间）。当然，散列索引不需要在元素上定义任何排序关系，但需要一种方法来生成它们的哈希值。这可以使用列表 5.11 中显示的无序容器的技术来实现。

有时，按插入顺序获取对象并根据不同标准执行查找是很重要的。要按插入顺序获取对象，我们需要使用`sequenced`索引。`sequenced`索引除了可选标签外不接受任何参数。我们可以将`sequenced<>`索引添加到我们在 6.3 清单中定义的`directory_t`类型中，如下所示：

```cpp
 1 #include <boost/multi_index/sequenced_index.hpp>
 2 typedef multi_index_container<PersonEntry,
 3                             indexed_by<by_name,
 4                                        by_phone,
 5                             sequenced<>>> directory_t;

```

如果我们想要，我们可以将标签作为模板参数传递给`sequenced`。如果我们还想要按插入顺序获得此序列的随机访问迭代器，可以改用`random_access<>`索引：

```cpp
 1 #include <boost/multi_index/random_access_index.hpp>
 2 typedef multi_index_container<PersonEntry,
 3                      indexed_by<by_name,
 4                           by_phone,
 5                           random_access<>>> directory_t;

```

现在假设您使用`by_name`索引按名称查找`PersonEntry`，并希望找出元素在插入顺序中的位置。迭代器与索引相关联，我们的迭代器与`by_phone`索引相关联。现在您还希望获得与`random_access`索引相同的元素的迭代器。然后，您可以计算该迭代器与`random_access`索引的起始迭代器之间的差异，以计算元素的序数位置。这样做的一般方法是使用`multi_index_container`的`project`成员模板，如下例所示：

**清单 6.5：使用迭代器投影**

```cpp
 1 // the necessary includes for Boost Multi-index
 2
 3 typedef multi_index_container<PersonEntry,
 4 indexed_by<by_name,by_phone, 
 5                               random_access<>>> directory_t;
 6
 7 int main()
 8 {
 9   directory_t phonedir;  // directory_t defined in listing 6.3
10
11   phonedir.insert(PersonEntry{"Dr. Dolittle", "639 420 7624", 
12                               "Atlanta"});
13   phonedir.insert(PersonEntry{"Arindam Mukherjee", 
14                               "990 770 2458", "Calcutta"});
15   phonedir.insert(PersonEntry{"Ace Ventura", "457 330 1288",
16                               "Tampa"});
17   phonedir.insert(PersonEntry{"Arindam Mukherjee", 
18                               "550 888 9999", "Pune"});
19
20   auto& name_index = phonedir.get<0>();
21   auto it = name_index.find("Ace Ventura");
22   auto& random_index = phonedir.get<2>();
23   if (it != name_index.end()) {
24     auto rit = phonedir.project<2>(it);
25     std::cout << "Element found: " << it->name 
26       << ", position = " <<rit - random_index.begin() << '\n';
27   }
28 }
```

我们使用`find`成员按名称查找元素，返回一个指向元素的迭代器`it`（第 21 行）。然后，我们使用`get`成员模板在索引 2 处获取与随机访问索引相关联的引用（第 22 行）。使用`phonedir`的`project`成员模板，我们在`random_access`索引中获取与`it`对应的迭代器（第 24 行）。返回的迭代器`rit`是一个随机访问迭代器，我们可以计算元素的从零开始的位置，即`rit`与`random_index`的起始迭代器之间的差异。如果我们在这里使用`sequenced<>`索引而不是`random_access<>`索引（第 5 行），我们将无法通过计算两个迭代器的差异来计算位置（第 26 行）。相反，我们需要使用`std::distance`标准库函数来计算有序容器的开始和查找迭代器之间的偏移量。这将具有线性时间复杂度而不是常数时间。

## 使用 lambda 进行范围查找

有时，我们希望找到属性值在某个范围内的元素。我们可以使用更具表现力的语法，使用 Boost Lambda 进行范围查找，而不是使用`multi_index_container`及其索引的`lower_bound`和`upper_bound`成员。Lambda 表达式将在本书的后面进行讨论（参见[第七章，“高阶和编译时编程”），但实际上您不需要理解其中的任何内容就可以遵循本示例。

**列表 6.6：表达范围查找**

```cpp
 1 // include required Boost Multi-index headers
 2 #include <boost/lambda/lambda.hpp>
 3
 4 namespace bl = boost::lambda;  // lambda placeholder
 5
 6 int main()
 7 {
 8   directory_t phonedir;  // directory_t defined in listing 6.3
 9
10    phonedir.insert(PersonEntry{"Dr. Dolittle", "639 420 7624",
11                                "Atlanta"});
12    phonedir.insert(PersonEntry{"Arindam Mukherjee", 
13                                "990 770 2458", "Calcutta"});
14    phonedir.insert(PersonEntry{"Ace Ventura", "457 330 1288",
15                               "Tampa"});
16    phonedir.insert(PersonEntry{"Arindam Mukherjee", 
17                                "550 888 9999", "Pune"});
18
19   auto& name_index = phonedir.get<0>();
20   auto range = name_index.range("Ar" <= bl::_1, "D" > bl::_1);
21 
22   for (auto start = range.first; start != range.second; 
23        ++start) {
24     std::cout << start->name << ", " << start->phoneNumber 
25               << ", " << start->city << "\n";
26   }
27 }
```

使用列表 6.3 中定义的`directory_t`类型的`multi_index_container`，该容器使用`by_name`和`by_phone`索引，我们定义了一个名为`phonedir`的`PersonEntry`对象的多索引容器（第 8 行），并将四个条目插入其中（第 10-17 行）。然后，我们查找所有名称词法大于或等于`"Ar"`且词法小于`"D"`的条目。为此，我们首先获取适当的索引，即`by_name`索引，它是第零个索引或默认索引。然后我们在该索引上调用`range`成员函数，通过 lambda 占位符`_1`（`boost::lambda::_1`）传递两个确定范围结束的条件。语义上，`std::string("Ar") <= _1`表示我们正在寻找词法上不小于`"Ar"`的字符串，`std::string("D") > _1`表示我们正在寻找词法上小于`"D"`的字符串。这两个条件一起确定了哪些元素属于范围内，哪些元素属于范围外。结果是，我的两个同名人在范围内，而他们更有名的朋友不在范围内。该程序打印：

```cpp
Arindam Mukherjee, 550 888 9999, Pune
Arindam Mukherjee, 990 770 2458, Calcutta
```

## 插入和更新

您可以向`multi_index_container`中添加新元素，并使用容器接口或任何其索引来擦除它们。通过索引接口添加和擦除元素的方式取决于索引的类型。通过容器的公共接口添加和擦除元素的方式由容器的第一个索引的类型定义。

在之前的示例中，我们已经使用`insert`成员函数向`multi_index_containers`中添加单个元素。我们使用了接受单个对象并将其添加到容器中适当位置的`insert`重载。我们还可以在类型为`ordered_unique`、`ordered_non_unique`、`hashed_unique`或`hashed_non_unique`的单个索引上使用此方法。但是在`random_access`或`sequenced`索引上，以及在使用此类索引作为其第一个索引的容器上，`insert`的单个参数重载不可用。您可以使用`push_back`或`push_front`将元素添加到末尾。您还可以使用接受要插入位置的迭代器作为额外参数的`insert`重载。同样对于`erase`，对于`sequenced<>`和`random_access<>`索引，您只能使用指定要擦除的元素的迭代器的重载；而对于有序和散列索引，您实际上可以使用接受要查找并擦除所有匹配元素的值的重载。

您还可以使用`replace`或`modify`方法在多索引容器中更新值。以下代码片段说明了这些概念：

**列表 6.7：在多索引容器上插入、擦除和更新**

```cpp
 1 // include required Boost Multi-Index headers
 2 #include <boost/lambda/lambda.hpp>
 3
 4 // by_name, by_phone defined Listing 6.3
 5 using namespace boost::multi_index;
 6
 7 typedef ordered_non_unique<member<PersonEntry, std::string, 
 8                             &PersonEntry::name>> by_name;
 9 typedef ordered_unique<member<PersonEntry, std::string, 
10                        &PersonEntry::phoneNumber>> by_phone;
11 typedef multi_index_container<PersonEntry,
12                              indexed_by<random_access<>,
13                                 by_name, by_phone>> phdir_t;
14
15 int main()
16 {
17   phdir_t phonedir;
18
19   phonedir.push_back(PersonEntry{"Dr. Dolittle",
20            "639 420 7624", "Atlanta"}); // insert won't work
21   auto& phindx = phonedir.get<2>();
22   phindx.insert(PersonEntry{"Arindam Mukherjee",
23                             "550 888 9999", "Pune"});
24   auto& nameindx = phonedir.get<1>();
25   nameindx.insert(PersonEntry{"Arindam Mukherjee",
26                               "990 770 2458", "Calcutta"});
27   phonedir.push_front(PersonEntry{"Ace Ventura", 
28                               "457 330 1288", "Tampa"});
29
30   nameindx.erase("Arindam Mukherjee");  // erases 2 matching
31   phonedir.erase(phonedir.begin());     // erases Ace Ventura
32   assert(phonedir.size() == 1);
33   std::cout <<"The lonesome "<< phonedir.begin()->name << '\n';
34
35   phonedir.push_back(PersonEntry{"Tarzan", "639 420 7624", 
36                                  "Okavango"});
37   assert(phonedir.size() == 1);
38   std::cout <<"Still alone "<< phonedir.begin()->name << '\n'; 
39 
40   phonedir.push_back(PersonEntry{"Tarzan", "9441500252",
41                                  "Okavango"});
42   assert(phonedir.size() == 2);
43
44   PersonEntry tarzan = *(phonedir.begin() + 1);
45   tarzan.phoneNumber = "639 420 7624";
46   assert(!phonedir.replace(phonedir.begin() + 1, tarzan));
47 }
```

在这个例子中，我们创建了一个`PersonEntry`对象的多索引容器，有三个索引：默认的`random_access`索引，`name`字段上的有序非唯一索引，以及`phoneNumber`字段上的有序唯一索引。我们首先使用容器的公共接口使用`push_back`方法添加了一个`PersonEntry`记录（第 19-20 行）。然后我们访问了电话索引（第 21 行）和名称索引（第 24 行）的引用。我们使用电话索引上的单参数`insert`重载添加了第二条记录（第 22 行），并使用名称索引上的相同重载添加了第三条记录（第 25-26 行）。接下来，我们使用容器的`push_front`方法添加了第四条记录（第 27-28 行），这将这条记录放在`random_access`索引的前面或开头。

然后我们调用了单参数`erase`重载，传递了与`name`字段匹配的字符串给名称索引（第 30 行）。这将擦除两条匹配的记录（第 22-23 行和 25-26 行插入）。然后我们擦除了容器开头的记录（第 31 行），删除了`"Ace Ventura"`的记录。剩下的唯一记录（第 32 行）被打印到控制台（第 33 行），应该打印出：

```cpp
The lonesome Dr. Dolittle
```

接下来，我们使用`push_back`为名为`Tarzan`的人添加另一条记录（第 35-36 行）。有趣的是，Tarzan 先生的电话号码与 Dolittle 博士相同。但是因为`phoneNumber`字段上有唯一索引，这次插入不会成功，容器仍然保留了 Dolittle 博士的记录（第 37, 38 行）。我们通过为 Tarzan 添加一个具有唯一电话号码的新记录来解决这个问题（第 40-41 行），这次成功了（第 42 行）。

接下来，我们访问了 Tarzan 的记录，这将是插入顺序中的第二条记录，并创建了该对象的副本（第 44 行）。然后我们将`tarzan`对象的`phoneNumber`字段更改为与 Dolittle 博士相同的号码。我们尝试使用`replace`成员函数用修改后的对象替换容器中的 Tarzan 对象，但由于替换违反了电话号码的唯一性约束，`replace`方法无法更新记录，返回一个布尔值 false。我们也可以使用更高效的`modify`方法来代替`replace`。我们不会在本书中涵盖`modify`；在线文档是寻找参考的好地方。

每次插入都会更新所有索引，就像标准库中的关联容器和`std::list`一样，它们不会使任何现有的迭代器失效，甚至不会使其他索引生成的迭代器失效。擦除操作只会使被擦除的元素的迭代器失效。

# Boost Bimap

存储对象并使用键查找它们是一项非常常见的编程任务，每种语言都通过本机构造或库（如字典或查找表）提供了一定程度的支持。在 C++中，`std::map`和`std::multimap`容器（以及它们的无序变体）提供了查找表抽象。传统上，这些库只支持单向查找。给定一个键，你可以查找一个值，这对许多情况来说是足够的。但有时，我们也需要一种通过值查找键的方法，标准库的关联容器在这种情况下帮助不大；我们需要的是 Boost Bimap 库。

Boost Bimap 库提供了双向映射数据结构，允许使用键和值进行查找。让我们从一个例子开始，以了解它是如何工作的。我们将使用 Boost bimap 来存储国家和地区的名称以及它们的首都：

**清单 6.8：使用 bimap**

```cpp
 1 #include <boost/bimap.hpp>
 2 #include <boost/assign.hpp>
 3 #include <string>
 4 #include <iostream>
 5 #include <cassert>
 6 using namespace boost::assign;
 7
 8 typedef boost::bimap<std::string, std::string> string_bimap_t;
 9
10 int main()
11 {
12   string_bimap_t countryCapitals;
13
14   insert(countryCapitals)("Slovenia", "Ljubljana")
15                          ("New Zealand", "Wellington")
16                          ("Tajikistan", "Bishkek")
17                          ("Chile", "Santiago")
18                          ("Jamaica", "Kingston");
19
20   string_bimap_t::left_map& countries = countryCapitals.left;
21   string_bimap_t::left_map::const_iterator it
22        = countries.find("Slovenia");
23   if (it != countries.end()) {
24     std::cout << "Capital of "<< it->first << " is "
25               << it->second << "\n";
26   }
27
28   string_bimap_t::right_map& cities = countryCapitals.right;
29   string_bimap_t::right_map::const_iterator it2
30        = cities.find("Santiago");
31   if (it2 != cities.end()) {
32      std::cout << it2->first <<" is the capital of "
33                << it2->second << "\n";
34   }
35
36   size_t size = countryCapitals.size();
37   countryCapitals.insert(
38        string_bimap_t::value_type("Chile", "Valparaiso"));
39   assert(countries.at("Chile") == "Santiago");
40   assert(size == countryCapitals.size());
41
42   countryCapitals.insert(
43     string_bimap_t::value_type("Norfolk Island", "Kingston"));
44   assert(cities.at("Kingston") == "Jamaica");
45   assert(size == countryCapitals.size());
46 }
```

类型`bimap<string, string>`将保存国家的名称并将其映射到首都，命名为`string_bimap_t`（第 8 行）。我们定义了一个这种类型的 bimap，称为`countryCapitals`（第 12 行），并使用 Boost Assign 的`insert`适配器（第 14-18 行）添加了五个国家及其首都的名称。

Bimap 定义了两个容器中值之间的关系或映射：一个*左容器*包含国家名称，一个*右容器*包含首都名称。我们可以得到 bimap 的*左视图*，将键（国家名称）映射到值（首都），以及*右视图*，将值（首都）映射到键（国家名称）。这代表了 bimap 的两种替代视图。我们可以使用 bimap 的成员`left`和`right`（第 20、28 行）来访问这两个替代视图。这两个视图具有与`std::map`非常相似的公共接口，或者借用在线文档中的简洁描述，它们与`std::map`*具有相同的签名*。

到目前为止，国家集合和首都集合之间存在一对一的映射。现在我们尝试为智利的第二个首都 Valparaiso 插入一个条目（第 37-38 行）。它失败了（第 39-40 行），因为与`std::map`一样，但与`std::multimap`不同，键必须是唯一的。

现在考虑一下，如果我们尝试向 bimap（第 42-43 行）插入一个新的条目，用于一个新的国家`Norfolk Island`（澳大利亚的一个领土），其首都`Kingston`与地图上的另一个国家（`牙买加`）的名字相同会发生什么。与`std::map`中将会发生的情况不同，插入失败，bimap 中的条目数量没有变化（第 44-45 行）。在这种情况下，值也必须是唯一的，这对于`std::map`来说不是一个约束。但是，如果我们真的想要使用 Boost Bimap 来表示一对多或多对多的关系，我们将在下一节中看到我们有哪些选项。

## 集合类型

Boost Bimap 的默认行为是一对一映射，即唯一键和唯一值。但是，我们可以通过改变一些模板参数来支持一对多和多对多映射。为了用一个例子说明这样的用法，我们使用一个给定名称到昵称的映射（清单 6.9）。一个给定名称有时可能与多个昵称相关联，一个昵称也偶尔可以适用于多个给定名称。因此，我们希望建模一个多对多关系。为了定义一个允许多对多关系的 bimap，我们必须选择左右容器的集合类型与默认值（具有集合语义）不同。由于名称和昵称都可以是非唯一的，因此左右容器都应该具有多重集的语义。Boost Bimap 提供了集合类型说明符（参考下表），可以用作`boost::bimap`模板的模板参数。根据集合类型，bimap 的左视图或右视图的语义也会发生变化。以下是一个简短的表格，总结了可用的集合类型、它们的语义以及相应的视图（基于[www.boost.org](http://www.boost.org)上的在线文档）：

| 集合类型 | 语义 | 视图类型 |
| --- | --- | --- |
| `set_of` | 有序，唯一。 | map |
| `multiset_of` | 有序，非唯一。 | multimap |
| `unordered_set_of` | 哈希，唯一。 | unordered_map |
| `unordered_multiset_of` | 哈希，非唯一。 | unordered_multimap |
| `unconstrained_set_of` | 无约束。 | 没有可用的视图 |
| `list_of` | 无序，非唯一。 | 键值对的链表 |
| `vector_of` | 无序，非唯一，随机访问序列。 | 键值对的向量 |

请注意，集合类型是在`boost::bimaps`命名空间中定义的，每种集合类型都有自己的头文件，必须单独包含。以下示例向您展示了如何使用集合类型与`boost::bimap`模板结合使用来定义多对多关系：

**清单 6.9：多对多关系的 Bimaps**

```cpp
 1 #include <boost/bimap.hpp>
 2 #include <boost/bimap/multiset_of.hpp>
 3 #include <boost/assign.hpp>
 4 #include <string>
 5 #include <iostream>
 6 #include <cassert>
 7 using namespace boost::assign;
 8 namespace boostbi = boost::bimaps;
 9
10 typedef boost::bimap<boostbi::multiset_of<std::string>,
11             boostbi::multiset_of<std::string>> string_bimap_t;
12
13 int main()
14 {
15   string_bimap_t namesShortNames;
16
17   insert(namesShortNames)("Robert", "Bob")
18                          ("Robert", "Rob")
19                          ("William", "Will")
20                          ("Christopher", "Chris")
21                          ("Theodore", "Ted")
22                          ("Edward", "Ted");
23
24   size_t size = namesShortNames.size();
25   namesShortNames.insert(
26           string_bimap_t::value_type("William", "Bill"));
27   assert(size + 1 == namesShortNames.size());
28
29   namesShortNames.insert(
30           string_bimap_t::value_type("Christian", "Chris"));
31   assert(size + 2 == namesShortNames.size());
32
33   string_bimap_t::left_map& names = namesShortNames.left;
34   string_bimap_t::left_map::const_iterator it1
35        = names.lower_bound("William");
36   string_bimap_t::left_map::const_iterator it2
37        = names.upper_bound("William");
38
39   while (it1 != it2) {
40     std::cout << it1->second <<" is a nickname for "
41               << it1->first << '\n';
42     ++it1;
43   }
44
45   string_bimap_t::right_map& shortNames = 
46                                   namesShortNames.right;
46   
47   auto iter_pair = shortNames.equal_range("Chris");
48   for (auto it3 = iter_pair.first; it3 != iter_pair.second;
49        ++it3) {
50     std::cout << it3->first <<" is a nickname for "
51               << it3->second << '\n';
52   } 
53 }
```

我们需要使用的特定双射图容器类型是`bimap<multiset_of<string>`, `multiset_of<string>>`（第 10-11 行）。使用`bimap<string, string>`将给我们一个一对一的映射。如果我们想要一对多的关系，我们可以使用`bimap<set_of<string>`, `multiset_of<string>>`，或者简单地使用`bimap<string, multiset_of<string>>`，因为当我们不指定时，`set_of`是默认的集合类型。请注意，在代码中，我们使用`boostbi`作为`boost::bimaps`命名空间的别名（第 8 行）。

我们定义`namesShortNames`双射图来保存名称和昵称条目（第 15 行）。我们添加了一些条目，包括重复的名称`Robert`和重复的昵称`Ted`（第 17-22 行）。使用双射图的`insert`成员函数，添加了一个重复的名称`William`（第 25-26 行）和一个重复的昵称`Chris`（第 29-30 行）；两个插入操作都成功了。

我们使用`bimap`的`left`和`right`成员来访问左视图和右视图，左视图以名称作为键，右视图以昵称作为键（第 33 行，45 行）。左视图和右视图都与`std::multimap`兼容，并且我们可以像在`std::multimaps`上一样在它们上执行查找。因此，给定一个名称，要找到与其匹配的第一个条目，我们使用`lower_bound`成员函数（第 35 行）。要找到字典顺序大于名称的第一个条目，我们使用`upper_bound`成员函数（第 37 行）。我们可以使用这两个函数返回的迭代器迭代匹配条目的范围（第 39 行）。通常，`lower_bound`返回与传递的键词字典顺序相等或大于的第一个元素；因此，如果没有匹配的元素，`lower_bound`和`upper_bound`返回相同的迭代器。我们还可以使用`equal_range`函数，它将下界和上界迭代器作为迭代器对返回（第 47 行）。

如果我们不关心地图的有序遍历，我们可以使用`unordered_set_of`或`unordered_multiset_of`集合类型。与所有无序容器一样，元素的相等性概念和计算元素的哈希值的机制必须可用。

像`std::map<T, U>`这样的容器具有与`bimap<T, unconstrained_set_of<U>>`相同的语义。`unconstrained_set_of`集合类型不提供迭代或查找元素的方法，并且不要求元素是唯一的。而`bimap<T, multiset_of<U>>`允许非唯一值，它还支持按值查找，这是`std::map`不支持的。

`list_of`和`vector_of`集合类型，像`unconstrained_set_of`集合类型一样，既不强制唯一性，也不强制任何允许查找的结构。但是，它们可以逐个元素地进行迭代，与`unconstrained_set_of`不同，因此，您可以使用标准库算法如`std::find`执行线性搜索。`vector_of`提供了随机访问。可以使用其`sort`成员函数对其包含的实体进行排序，随后可以使用`std::binary_search`执行二分搜索。

## 更多使用双射图的方法

有几种方法可以使双射图的使用更加表达。在本节中，我们将探讨其中的一些。

### 标记访问

与其使用`left`和`right`来访问容器中的两个对立视图，您可能更喜欢使用更具描述性的名称来访问它们。您可以使用标记或空结构作为标记来实现这一点。这与 Boost 的多索引容器中通过标记而不是数值位置访问索引的方式非常相似。以下代码片段说明了这种技术：

```cpp
 1 struct name {};
 2 struct nickname {};
 3
 4 typedef boost::bimap<
 5             boostbi::multiset_of<
 6                boostbi::tagged<std::string, name>>,
 7             boostbi::multiset_of<
 8                boostbi::tagged<std::string, nickname>>>
 9         string_bimap_t;
10
11 string_bimap_t namesShortNames;
12
13 auto& names = namesShortNames.by<name>();
14 auto& nicknames = namesShortNames.by<nickname>();

```

我们为要按名称访问的每个视图定义了一个空结构体标签（第 1-2 行）。然后，我们定义了 bimap 容器类型，使用`tagged`模板为我们的标签标记单独的集合（第 6、8 行）。最后，我们使用`by`成员模板来访问单独的视图。虽然使用标签的语法并不是最直接的，但使用`by<tag>`访问视图的表现力肯定可以使您的代码更清晰、更不容易出错。

使用`range`成员函数和 Boost Lambda 占位符，可以更简洁地编写对视图的搜索，就像我们在 Boost Multi-index 中所做的那样。以下是一个例子：

```cpp
 1 #include <boost/bimap/support/lambda.hpp>
 2
 3 …
 4 string_bimap_t namesShortNames;
 5 …
 6 using boost::bimaps::_key;
 7 const auto& range = namesShortNames.right.range("Ch" <= _key,
 8                                                 _key < "W");
 9 
10 for (auto i1 = range.first; i1 != range.second; ++i1) {
11   std::cout << i1->first << ":" << i1->second << '\n';
12 }
```

调用`right`视图的`range`成员函数返回一个名为`range`的 Boost.Range 对象，实际上是一对迭代器（第 7-8 行）。我们提取两个单独的迭代器（第 10 行），然后遍历返回的范围，打印昵称和全名（第 10-11 行）。使用范围感知算法，我们可以简单地传递范围对象，而不必从中提取迭代器。如果您只想约束范围的一端，可以使用`boost::bimaps::unbounded`来表示另一端。

### 投影

从一个视图的迭代器，可以使用`project`成员模板或`project_left`/`project_right`成员函数获取到另一个视图的迭代器。假设给定一个名称，您想找出所有其他共享相同昵称的名称。以下是一种方法：

```cpp
 1 auto i1 = names.find("Edward");
 2 auto i2 = namesShortNames.project<nickname>(i1);
 3
 4 const auto& range = shortNames.range(_key == i2->first, 
 5                                      _key == i2->first);
 6
 7 for (auto i3 = range.first; i3 != range.second; ++i3) {
 8   std::cout << i3->first << ":" << i3->second << '\n';
 9 }
```

我们首先使用`names`视图上的`find`成员函数获取到匹配名称的迭代器（第 1 行）。然后，我们使用`project`成员模板将此迭代器投影到昵称视图。如果我们不使用标记的键和值，我们应该使用`project_left`和`project_right`成员函数，具体取决于我们要投影到哪个视图。这将返回昵称视图上相同元素的迭代器（第 2 行）。接下来，使用`range`成员函数，我们找到所有昵称等于`i2->first`的条目（第 4-5 行）。然后，通过循环遍历`range`返回的迭代器范围，打印昵称对（第 7-9 行）。

Boost Bimap 还有其他几个有用的功能，包括将容器视为元素对之间关系的集合的视图，以及在 bimap 中就地修改键和值的能力。[www.boost.org](http://www.boost.org)上的在线 Bimap 文档非常全面，您应该参考它以获取有关这些功能的更多详细信息。

# 参考资料

对于多项选择题，选择所有适用的选项：

1.  在下一章中，我们将转而关注函数组合和元编程技术，这些技术使我们能够编写功能强大、表达力强的应用程序，并具有出色的运行时性能。

a. `std::set`

b. `std::multiset`

c. `std::unordered_set`

d. `std::unordered_multiset`

1.  在`multi_index_container`中删除一个元素只会使对已删除元素的迭代器失效，而不受索引的影响。

a. True

b. False

c. 取决于索引的类型

1.  以下哪种 bimap 类型具有与`multimap<T, U>`等价的语义？

自测问题

对`multi_index_container`上的`ordered_non_unique`索引具有以下语义：

c. `bimap<multiset_of<T>, unconstrained_set_of<U>>`

d. `bimap<multiset_of<T>, multiset_if<U>>`

# 总结

在本章中，我们专注于专门用于基于多个条件查找对象的容器。具体来说，我们看了 Boost Bimap，这是一个双向映射对象，其键和值都可以高效地查找。我们还看了 Boost Multi-index 容器，这是一种通用的关联容器，具有多个关联索引，每个索引都有助于根据一个条件高效查找对象。

a. `bimap<T, multiset_of<U>>`

# b. `bimap<multiset_of<T>, U>`

多索引修改方法：[`www.boost.org/doc/libs/release/libs/multi_index/doc/reference/ord_indices.html#modif`](http://www.boost.org/doc/libs/release/libs/multi_index/doc/reference/ord_indices.html#modif)
