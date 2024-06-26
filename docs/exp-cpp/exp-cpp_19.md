# 第十六章：实现基于对话框的搜索引擎

在这本书中，我们已经走了这么远！我们已经学习了 C++应用程序开发的基础知识，并讨论了构建和设计面向全球的应用程序。我们还深入研究了数据结构和算法，这是高效编程的核心。现在是时候利用所有这些技能来设计复杂的软件，比如搜索引擎了。

随着互联网的普及，搜索引擎已成为最受欢迎的产品。大多数用户从搜索引擎开始他们的网络之旅。各种网络搜索服务，如 Google、Baidu、Yandex 等，每天接收大量的流量，处理数万亿的请求。搜索引擎在不到一秒的时间内处理每个请求。尽管它们维护了成千上万的服务器来处理负载，但它们高效处理的核心是数据结构和算法、数据架构策略和缓存。

设计高效搜索系统的问题不仅出现在网络搜索引擎中。本地数据库、**客户关系管理**（**CRM**）系统、会计软件等都需要强大的搜索功能。在本章中，我们将了解搜索引擎的基础知识，并讨论用于构建快速搜索引擎的算法和数据结构。您将了解网络搜索引擎的一般工作原理，并了解需要高处理能力的项目中使用的新数据结构。您还将建立信心，去构建自己的搜索引擎，与现有的搜索引擎竞争。

在本章中，我们将涵盖以下主题：

+   理解搜索引擎的结构

+   理解和设计用于在搜索引擎中将关键词映射到文档的倒排索引

+   为搜索平台的用户设计和构建推荐引擎

+   使用知识图谱设计基于对话框的搜索引擎

# 技术要求

本章中使用`g++`编译器和`-std=c++2a`选项来编译示例。您可以在[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)找到本章中使用的源文件。

# 理解搜索引擎的结构

想象一下世界上数十亿的网页。在搜索引擎界面中输入一个单词或短语，不到一秒钟就会返回一个长长的结果列表。搜索引擎如此快速地处理如此多的网页，这是奇迹般的。它是如何如此快速地找到正确的文档的呢？为了回答这个问题，我们将做程序员可以做的最明智的事情，设计我们自己的引擎。

以下图表显示了搜索引擎背后的基本思想：

![](img/e863777e-e6a1-428a-9543-34793e6ebfc4.png)

**用户**使用搜索引擎的**用户界面**输入单词。**搜索引擎**扫描所有文档，对其进行过滤，按相关性对其进行排序，并尽快向用户做出响应。我们主要关注的是网络搜索引擎的实现。寻找某物需要在数十亿的文档中进行搜索。

让我们试着想出一种方法来从数十亿的文档中找到短语“Hello, world!”（为了简洁起见，我们将网页称为文档）。扫描每个文档以查找该短语将需要大量的时间。如果我们认为每个文档至少有 500 个单词，搜索特定单词或单词组合将需要很长时间。更实际的方法是事先扫描所有文档。这个扫描过程包括在文档中建立每个单词出现的索引，并将信息存储在数据库中，这也被称为**文档索引**。当用户输入一个短语时，搜索引擎将在其数据库中查找这些单词，并返回满足查询的文档链接。

在搜索文档之前，引擎验证用户输入并不会有害。用户在短语中出现拼写错误并不罕见。除了拼写错误，如果引擎自动完成单词和短语，用户体验会更好。例如，当用户输入“hello”时，引擎可能建议搜索短语“Hello, world!”。一些搜索引擎跟踪用户，存储有关其最近搜索、请求设备的详细信息等信息。例如，如果用户搜索“如何重新启动计算机”，如果搜索引擎知道用户的操作系统，结果会更好。如果是 Linux 发行版，搜索引擎将对搜索结果进行排序，使描述如何重新启动基于 Linux 的计算机的文档首先出现。

我们还应该注意定期出现在网络上的新文档。后台作业可能会持续分析网络以查找新内容。我们称这个作业为**爬虫**，因为它爬行网络并索引文档。爬虫下载文档以解析其内容并构建索引。已经索引的文档可能会得到更新，或者更糟的是被删除。因此，另一个后台作业应定期更新现有文档。您可能会遇到爬行网络以解析文档的任务术语**蜘蛛**。

下面更新的图表更详细地说明了搜索引擎的结构：

![](img/80b2e7a9-acbb-4672-a83c-d3bce75bde7c.png)

搜索具有广泛的应用。想象一下最简单的搜索形式——在数组中查找一个单词：

```cpp
using words = std::vector<std::string>;
words list = get_list_of_words(); // suppose the function is implemented

auto find_in_words(const std::string& term)
{
  return std::find(list.begin(), list.end(), term);
}
```

尽管前面的例子适用于最简单的搜索引擎，但真正的问题是设计一个可扩展的搜索引擎。您不希望通过搜索字符串数组来处理用户请求。相反，您应该努力实现一个能够搜索数百万个文档的可扩展搜索引擎。这需要大量的思考和设计，因为一切都很重要，从正确选择的数据结构到高效的数据处理算法。现在让我们更详细地讨论搜索引擎的组件。我们将整合从之前章节学到的所有技能来设计一个好的搜索引擎。

# 提供方便的用户界面

在构建提供令人惊叹的用户体验的细粒度用户界面上投入时间和资源至关重要。关键在于简单。界面越简单，使用起来就越好。我们将以市场主导地位的 Google 为例。它在页面中央有一个简单的输入字段。用户在字段中输入请求，引擎会建议一些短语：

![](img/51e853fb-bc2e-4002-b368-fbd3e49d7057.png)

我们不认为用户是懒惰的人，但提供建议列表是有帮助的，因为有时用户不知道他们正在寻找的确切术语。让我们集中精力在建议列表的结构和实施上。毕竟，我们对解决问题感兴趣，而不是设计漂亮的用户界面。我们不会在本章讨论用户界面设计；更好的是集中在搜索引擎的后端。然而，在继续之前，有一件事情我们应该考虑。我们正在实现的搜索引擎是基于对话的。用户查询引擎并可以从几个答案中选择以缩小结果列表。例如，假设用户查询“一台电脑”，搜索引擎会问“台式机还是笔记本？”。这会大大减少搜索结果并为用户提供更好的结果。我们将使用决策树来实现这一点。但在此之前，让我们了解搜索引擎的复杂性。

首先，存在**输入标记化**的问题。这涉及文档解析和搜索短语分析。您可能构建了一个很好的查询解析器，但由于用户在查询中犯了一个错误，它就会出现问题。让我们来看看处理模糊查询的一些方法。

# 处理查询中的拼写错误

用户在输入时犯错并非罕见。虽然这似乎是一件简单的事情，但对于搜索引擎设计者来说可能会是一个真正的问题。如果用户输入了 helo worl 而不是 hello world，那么在数百万份文档中进行搜索可能会产生意外的错误结果。你可能熟悉搜索引擎提供的自动建议。例如，当我们输入错误时，谷歌搜索界面是这样的：

![](img/3885d527-3de0-4b25-946d-c20714d9eb89.png)

注意屏幕截图底部的两行。其中一行显示了 hello world 的搜索结果，这表明搜索引擎假定用户输入的查询存在拼写错误，并主动显示了正确查询的结果。然而，仍然有可能用户确实想要搜索他们输入的确切单词。因此，用户体验提供了下一行，即搜索 helo worl 的结果。

因此，在构建搜索引擎时，我们需要解决几个问题，首先是用户请求。首先，我们需要为用户提供一个方便的界面来输入他们的文本。界面还应该与用户进行交互，以提供更好的结果。这包括根据部分输入的单词提供建议，就像之前讨论的那样。使搜索引擎与用户进行交互是用户界面的另一个改进，我们将在本章中讨论。

接下来是检查拼写错误或不完整单词，这并不是一件容易的事。保留字典中所有单词的列表并比较用户输入的单词可能需要一段时间。为了解决这个问题，必须使用特定的数据结构和算法。例如，在检查用户查询中的拼写错误时，找到单词之间的**Levenshtein 距离**可能会有所帮助。Levenshtein 距离是一个单词需要添加、删除或替换的字符数，使其等于另一个单词。例如，*world*和*worl*之间的 Levenshtein 距离是 1，因为从*world*中删除字母*d*或在*worl*中添加*d*可以使这些单词相等。*coding*和*sitting*之间的距离是 4，因为以下四次编辑将一个单词变成另一个单词：

1.  coding -> cod**t**ing（在中间插入**t**）

1.  co**d**ting -> co**t**ting（将**t**替换为**d**）

1.  c**o**tting -> c**i**tting（将**o**替换为**i**）

1.  **c**itting -> **s**itting（将**c**替换为**s**）

现在，想象一下，如果我们要将每个用户输入与成千上万个单词进行比较以找到最接近的单词，处理将需要多长时间。另一种方法是使用一个大的**trie**（数据结构）来预先发现可能的拼写错误。Trie 是一个有序搜索树，其中键是字符串。看一下下面表示 trie 的图表：

![](img/296999a5-2bd6-4b40-bc87-9fdf30532cc2.png)

每条路径代表一个有效的单词。例如，a 节点指向 n 和 r 节点。注意 n 后面的#。它告诉我们，直到这个节点的路径代表一个单词，an。然而，它继续指向 d，然后是另一个#，意味着直到这个节点的路径代表另一个单词，and。对于 trie 的其余部分也适用相同的逻辑。例如，想象一下*world*的 trie 部分：

![](img/d174ec96-3f73-441a-8f25-6ebd22d5038a.png)

当引擎遇到*worl*时，它会通过前面的 trie。w 没问题，o 也没问题，直到单词的倒数第二个字符 l 之前的所有字符都没问题。在前面的图表中，l 后面没有终端节点，只有 d。这意味着我们可以确定没有*worl*这样的单词；所以它可能是*world*。为了提供良好的建议和检查拼写错误，我们应该有用户语言的完整词典。当你计划支持多种语言时，情况会变得更加困难。然而，尽管收集和存储词典可以说是一项简单的任务，更困难的任务是收集所有网页文档并相应地存储以进行快速搜索。搜索引擎收集和解析网站以构建搜索引擎数据库的工具、程序或模块（如前所述）称为爬虫。在更深入地研究我们将如何存储这些网页之前，让我们快速看一下爬虫的功能。

# 爬取网站

每次用户输入查询时搜索数百万个文档是不现实的。想象一下，当用户在系统的 UI 上点击搜索按钮后，搜索引擎解析网站以搜索用户查询。这将永远无法完成。搜索引擎从网站发送的每个请求都需要一些时间。即使时间少于一毫秒（0.001 秒），在用户等待查询完成的同时分析和解析所有网站将需要很长时间。假设访问和搜索一个网站大约需要 0.5 毫秒（即使如此，这也是不合理的快）。这意味着搜索 100 万个网站将需要大约 8 分钟。现在想象一下你打开谷歌搜索并进行查询，你会等待 8 分钟吗？

正确的方法是将所有信息高效地存储在数据库中，以便搜索引擎快速访问。爬虫下载网页并将它们存储为临时文档，直到解析和索引完成。复杂的爬虫可能还会解析文档，以便更方便地存储。重要的一点是，下载网页不是一次性的行为。网页的内容可能会更新。此外，在此期间可能会出现新页面。因此，搜索引擎必须保持其数据库的最新状态。为了实现这一点，它安排爬虫定期下载页面。智能的爬虫可能会在将内容传递给索引器之前比较内容的差异。

通常，爬虫作为多线程应用程序运行。开发人员应该尽可能快地进行爬取，因为保持数十亿个文档的最新状态并不是一件容易的事。正如我们已经提到的，搜索引擎不直接搜索文档。它在所谓的索引文件中进行搜索。虽然爬取是一个有趣的编码任务，但在本章中我们将主要集中在索引上。下一节介绍搜索引擎中的索引功能。

# 索引文档

搜索引擎的关键功能是索引。以下图表显示了爬虫下载的文档如何被处理以构建索引文件：

![](img/17ba8dc9-5ed9-428a-8b09-0db88bbcb4d4.png)

在前面的图表中，索引显示为**倒排索引**。正如你所看到的，用户查询被引导到倒排索引。虽然在本章中我们在**索引**和**倒排索引**这两个术语之间交替使用，但**倒排索引**是更准确的名称。首先，让我们看看搜索引擎的索引是什么。索引文档的整个目的是提供快速的搜索功能。其思想很简单：每次爬虫下载文档时，搜索引擎会处理其内容，将其分成指向该文档的单词。这个过程称为**标记化**。假设我们从维基百科下载了一个包含以下文本的文档（为了简洁起见，我们只列出了段落的一部分作为示例）：

```cpp
In 1979, Bjarne Stroustrup, a Danish computer scientist, began work on "C with Classes", the predecessor to C++. The motivation for creating a new language originated from Stroustrup's experience in programming for his PhD thesis. Stroustrup found that Simula had features that were very helpful for large software development...
```

搜索引擎将前面的文档分成单独的单词，如下所示（出于简洁起见，这里只显示了前几个单词）：

```cpp
In
1979
Bjarne
Stroustrup
a
Danish
computer
scientist
began
work
...
```

将文档分成单词后，引擎为文档中的每个单词分配一个**标识符**（**ID**）。假设前面文档的 ID 是 1，下表显示了单词指向（出现在）ID 为 1 的文档： 

| In | 1 |
| --- | --- |
| 1979 | 1 |
| Bjarne | 1 |
| Stroustrup | 1 |
| a | 1 |
| Danish | 1 |
| computer | 1 |
| scientist | 1 |
| ... |  |

可能有几个文档包含相同的单词，因此前表实际上可能看起来更像以下表：

| In | 1, 4, 14, 22 |
| --- | --- |
| 1979 | 1, 99, 455 |
| Bjarne | 1, 202, 1314 |
| Stroustrup | 1, 1314 |
| a | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... |
| Danish | 1, 99, 102, 103 |
| 计算机 | 1, 4, 5, 6, 24, 38, ... |
| scientist | 1, 38, 101, 3958, ... |

下表表示了倒排索引。它将单词与爬虫下载的文档的 ID 进行了映射。现在，当用户通过键入*computer*查询引擎时，结果是基于从索引中检索到的 ID 生成的，即在前面的示例中是 1, 4, 5, 6, 24, 38, ...。索引还有助于找到更复杂查询的结果。例如，*计算机科学家*匹配以下文档：

| computer | **1**, 4, 5, 6, 24, **38**, ... |
| --- | --- |
| scientist | **1**, **38**, 101, 3958, ... |

为了回应用户并提供包含两个术语的文档，我们应该找到引用文档的交集（参见前表中的粗体数字），例如，1 和 38。

请注意，用户查询在与索引匹配之前也会被标记化。标记化通常涉及单词规范化。如果没有规范化，*计算机科学家*查询将不会返回任何结果（请注意查询中的大写字母）。让我们更多地了解一下这个。

# 标记化文档

你可能还记得第一章中的标记化概念，*构建 C++应用程序*，我们讨论了编译器如何通过将源文件标记化为更小的、不可分割的单元（称为标记）来解析源文件。搜索引擎以类似的方式解析和标记化文档。

我们不会详细讨论这个，但你应该考虑文档是以一种方式处理的，这意味着标记（在搜索引擎上下文中具有意义的不可分割的术语）是规范化的。例如，我们正在查看的所有单词都是小写的。因此，索引表应该如下所示：

| in | 1, 4, 14, 22 |
| --- | --- |
| 1979 | 1, 99, 455 |
| bjarne | 1, 202, 1314 |
| stroustrup | 1, 1314 |
| a | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... |
| danish | 1, 99, 102, 103 |
| computer | 1, 4, 5, 6, 24, 38, ... |
| scientist | 1, 38, 101, 3958, ... |

作为 C++程序员，看到 bjarne 或 stroustrup 变成小写可能会让您感到不舒服。然而，由于我们正在将用户输入与倒排索引键进行匹配，我们应该考虑用户输入可能不具有我们期望的形式。因此，我们需要对用户输入应用相同的规则，以使其与倒排索引的形式匹配。

接下来，注意 a。毫不夸张地说，这是每个文档中都出现的一个词。其他类似的例子是*the*，*an*，*in*等词。我们称它们为**停用词**；它们在实际处理之前被过滤掉。通常，搜索引擎会忽略它们，因此倒排索引更新为以下形式：

| 1979 | 1, 99, 455 |
| --- | --- |
| bjarne | 1, 202, 1314 |
| stroustrup | 1, 1314 |
| danish | 1, 99, 102, 103 |
| computer | 1, 4, 5, 6, 24, 38, ... |
| scientist | 1, 38, 101, 3958, ... |

您应该注意，规范化不仅仅是将单词变成小写。它还涉及将单词转换为它们的正常形式。

将单词规范化为其根形式（或其词干）也称为**词干提取**。

看一下我们在本节开头使用的文档中的以下句子：

```cpp
The motivation for creating a new language originated from Stroustrup's experience in programming for his PhD thesis.
```

creating，originated 和 Stroustrup's 已经被规范化，因此倒排索引将具有以下形式：

| motivation | 1 |
| --- | --- |
| **create** | 1 |
| new | 1 |
| language | 1 |
| **originate** | 1 |
| **stroustrup** | 1 |
| experience | 1 |
| programming | 1 |
| phd | 1 |
| thesis | 1 |

还要注意，我们已经忽略了停用词，并且在前面的表中没有包括*the*。

标记化是索引创建的第一步。除此之外，我们可以以任何使搜索更好的方式处理输入，如下一节所示。

# 对结果进行排序

相关性是搜索引擎最重要的特性之一。仅仅返回与用户输入匹配的文档是不够的。我们应该以一种方式对它们进行排名，以便最相关的文档首先出现。

一种策略是记录文档中每个单词的出现次数。例如，描述计算机的文档可能包含单词*computer*的多次出现，如果用户搜索*a computer*，结果将显示包含最多*computer*出现次数的文档。以下是一个示例索引表：

| computer | 1{18}, 4{13}, 899{3} |
| --- | --- |
| map | 4{9}, 1342{4}, 1343{2} |
| world | 12{1} |

花括号中的值定义了文档中每个单词的出现次数。

当向用户呈现搜索结果时，我们可以考虑许多因素。一些搜索引擎会存储与用户相关的信息，以便返回个性化的结果。甚至用户用于访问搜索引擎的程序（通常是网络浏览器）也可能改变搜索平台的结果。例如，Linux 操作系统上搜索*重新安装操作系统*的用户会得到包含*重新安装 Ubuntu*的结果，因为浏览器提供了操作系统类型和版本信息。然而，考虑到隐私问题，有些搜索引擎完全消除了个性化用户数据的使用。

文档的另一个属性是更新日期。新鲜内容始终具有更高的优先级。因此，当向用户返回文档列表时，我们可能还会按其内容更新的顺序重新排列它们。对文档的相关排名的担忧将我们带到下一节，我们将在那里讨论推荐引擎。

# 构建推荐引擎

我们在上一章介绍了**人工智能**（**AI**）和**机器学习**（**ML**）。推荐引擎可以被视为一个 AI 驱动的解决方案，或者一个简单的条件语句集合。构建一个接收用户数据并返回最满足该输入的选项的系统是一个复杂的任务。将 ML 纳入这样的任务中可能听起来相当合理。

然而，你应该考虑到推荐引擎可能包括一系列规则，这些规则在输出给最终用户之前对数据进行处理。推荐引擎可以在预期和意想不到的地方运行。例如，在亚马逊浏览产品时，推荐引擎会根据我们当前查看的产品向我们推荐产品。电影数据库会根据我们之前观看或评分的电影向我们推荐新电影。对许多人来说，这可能看起来出乎意料，但推荐引擎也在搜索引擎背后运行。

你可能熟悉一些电子商务平台推荐产品的方式。大多数情况下，建议窗格的标题类似于“购买此产品的顾客还购买了...”。回想一下我们在上一章介绍的聚类分析。现在，如果我们试图了解这些建议是如何工作的，我们可能会发现一些聚类算法。

让我们简单地看一下并设想一些推荐机制。比如，一个书店网站。约翰买了一本名为“掌握 Qt5”的书，那么我们可以把这个信息放在表格中：

| | 掌握 Qt5 |
| --- | --- |
| 约翰 | 是 |

接下来，约翰决定购买一本 C++书籍，*掌握 C++编程*。莱娅购买了一本名为*设计模式*的书。卡尔购买了三本书，名为*学习 Python*、*掌握机器学习*和*Python 机器学习*。表格被更新，现在看起来是这样的：

| | 掌握 Qt5 | 掌握 C++编程 | 设计模式 | 学习 Python | 掌握机器学习 | Python 机器学习 |
| --- | --- | --- | --- | --- | --- | --- |
| 约翰 | 是 | 是 | 否 | 否 | 否 | 否 |
| 莱娅 | 否 | 否 | 是 | 否 | 否 | 否 |
| 卡尔 | 否 | 否 | 否 | 是 | 是 | 是 |

现在，让我们想象哈鲁特访问网站并购买了之前列出的两本书，*学习 Python*和*Python 机器学习*。向他推荐书籍*掌握 Qt5*是否合理？我们认为不合理。但我们知道他购买了哪些书，我们也知道另一个用户卡尔购买了三本书，其中两本与哈鲁特购买的书相同。因此，向哈鲁特推荐*掌握机器学习*可能是合理的，告诉他购买这两本书的其他顾客也购买了这本书。这是推荐引擎从高层次的工作原理的一个简单例子。 

# 使用知识图谱

现在，让我们回到我们的搜索引擎。用户正在搜索一位著名的计算机科学家——比如，唐纳德·克努斯。他们在搜索框中输入这个名字，然后从整个网络中得到排序后的最佳结果。再次看看谷歌搜索。为了充分利用用户界面，谷歌向我们展示了一些关于搜索主题的简要信息。在这种情况下，它在网页右侧显示了这位伟大科学家的几张图片和一些关于他的信息。这个部分看起来是这样的：

![](img/703e4005-d07b-4044-8fdd-c249be7de3e0.png)

这种方式，搜索引擎试图满足用户的基本需求，让他们能够更快地找到信息，甚至无需访问任何网站。在这种情况下，我们最感兴趣的是放置在前面信息框下面的建议框。它的标题是“人们还搜索”，看起来是这样的：

![](img/97e5ef3e-1898-47e4-8368-92a7fcba26c5.png)

这些是基于搜索 Donald Knuth 后搜索 Alan Turing 的用户活动的推荐。这促使推荐引擎提出建议，即如果有人新搜索 Donald Knuth，他们可能也对 Alan Turing 感兴趣。

我们可以通过谷歌称之为**知识图谱**的东西来组织类似的建议机制。这是一个由节点组成的图，每个节点代表一些可搜索的主题、人物、电影或其他任何东西。图数据结构是一组节点和连接这些节点的边，就像以下图表中的那样：

![](img/45cb1241-4a3d-44db-9fde-8bb9a30d3c84.png)

在知识图谱中，每个节点代表一个单一实体。所谓实体，我们指的是城市、人、宠物、书籍，或者几乎你能想象到的任何其他东西。现在，图中的边代表实体之间的连接。每个节点可以通过多个节点连接到另一个节点。例如，看看这两个节点：

![](img/ca796132-cf5f-4ed1-b8f0-f703724ac5f5.png)

这两个节点只包含文本。我们可能猜测 Donald Knuth 是一个名字，而《计算机程序设计艺术》是某种艺术。建立知识图谱的本质是我们可以将每个节点与代表其类型的另一个节点相关联。以下图表扩展了之前的图表：

![](img/5e6932f4-bec8-455f-8dd5-f15a92aa7946.png)

看看我们添加的两个新节点。其中一个代表一个**人**，而另一个代表一本**书**。更令人兴奋的是，我们将 Donald Knuth 节点与**人**节点连接，并标记为 is a 关系。同样，我们将**《计算机程序设计艺术》**节点连接到书籍节点，因此我们可以说《计算机程序设计艺术》是一本书。现在让我们将 Donald Knuth 与他写的书连接起来：

![](img/e68dfd54-f84d-4ee7-89ef-5c187ee45498.png)

所以，现在我们有了一个完整的关系，因为我们知道 Donald Knuth 是一位作者《计算机程序设计艺术》的人，而这本书又代表一本书。

让我们再添加几个代表人的节点。以下图表显示了我们如何添加了 Alan Turing 和 Peter Weyland 节点：

![](img/d433010e-be15-41f9-9669-d36aefa0590e.png)

所以，Alan Turing 和 Peter Weyland 都是人。现在，如果这是搜索引擎知识库的一部分，那么它给了我们对用户搜索意图的很好洞察。当我们点击 Donald Knuth 的结果时，我们知道这是关于一个人的。如果需要，我们可以建议用户查看我们在知识图谱中积累的其他人。是否合理建议搜索 Donald Knuth 的用户也查看 Alan Turing 和 Peter Weyland 的页面？这里就有棘手的部分：尽管两者都是人，它们之间并没有强烈的联系。因此，我们需要一些额外的东西来定义两个不同人之间连接的相关性。看看图表的以下添加：

![](img/9d8c7081-485a-487e-9692-9f9e43c47eb0.png)

现在清楚了，Donald Knuth 和 Alan Turing 共享相同的活动，被表示为“计算机科学”节点，代表了一门研究领域，而 Peter Weyland 原来是一个虚构的角色。所以，Peter Weyland 和 Donald Knuth 相关的唯一一件事就是他们都是人。看一下我们放在从人节点到计算机科学节点的边上的数字。假设我们将关系评分从 0 到 100，后者表示关系最强。所以，我们为 Alan Turing 和 Donald Knuth 都放了 99。我们本应该省略从 Peter Weyland 到计算机科学的边，而不是放 0，但我们故意这样做来显示对比。这些数字是权重。我们给边添加权重以强调连接因素；也就是说，Alan Turing 和 Donald Knuth 共享相同的事物，并且彼此之间关系密切。如果我们将 Steve Jobs 作为知识图中的一个新人物，图将会是这样的：

![](img/39e8ff27-3843-4762-b607-d5af7f171c25.png)

看一下边的权重。Steve Jobs 与计算机科学有一定关系，但他更多地与“商人”和“影响者”节点相关。同样，我们现在可以看到 Peter Weyland 与 Steve Jobs 的关系比与 Donald Knuth 的关系更密切。现在，对于推荐引擎来说，建议搜索 Donald Knuth 的用户也应该看看 Alan Turing 更具信息量，因为他们都是人，并且与计算机科学的关系权重相等或接近相等。这是一个很好的例子，展示了如何在搜索引擎中整合这样的图。我们接下来要做的是向您介绍使用类似知识图来构建一个更智能的框架，以提供相关的搜索结果。我们称之为基于对话的搜索。

# 实现基于对话的搜索引擎

最后，让我们来设计搜索引擎的一部分，这部分将为我们提供精细的用户界面。正如我们在本章开头提到的，基于对话的搜索引擎涉及构建一个用户界面，询问用户与其查询相关的问题。这种方法在我们有模糊的结果的情况下最为适用。例如，搜索 Donald 的用户可能心里想的是以下之一：

+   *唐纳德·克努斯*，伟大的计算机科学家

+   *唐纳德·达克*，卡通人物

+   *唐纳德·邓恩*，杰瑞德·邓恩的真名，虚构的角色

+   *唐纳德·特朗普*，商人和第 45 任美国总统

前面的列表只是对 Donald 搜索词的潜在结果的一个小例子。那么，缺乏基于对话的方法的搜索引擎会怎么做呢？它们会为用户输入的最佳匹配提供相关结果列表。例如，在撰写本书时，搜索 Donald 会得到一个与 Donald Trump 相关的网站列表，尽管我当时心里想的是 Donald Knuth。在这里，我们可以看到最佳匹配和用户最佳匹配之间的微妙差别。

搜索引擎收集大量数据用于个性化搜索结果。如果用户从事网站开发领域的工作，他们的大部分搜索请求都会与该特定领域有关。这对于提供用户更好的搜索结果非常有帮助。例如，一个搜索历史记录中大部分请求都与网站开发相关的用户，在搜索 zepelin 时将会得到更好、更专注的结果。理想的搜索引擎将提供链接到 Zeplin 应用程序用于构建 Web UI 的网站，而对于其他用户，引擎将提供有关摇滚乐队 Led Zeppelin 的信息的结果。

设计基于对话框的搜索引擎是提供用户更好界面的下一步。如果我们已经有了强大的知识库，构建起来就足够简单了。我们将使用前一节中描述的知识图概念。假设当用户输入搜索词时，我们从知识图中获取所有匹配的主题，并为用户提供潜在命中列表，如下图所示：

![](img/2fcc30b2-7cd7-4d82-bd2b-7d3d5b733845.png)

因此，用户现在更容易选择一个主题，并节省回忆完整名称的时间。来自知识图的信息可以（对于一些搜索引擎而言）在用户输入查询时合并到自动建议中。此外，我们将着手处理搜索引擎的主要组件。显然，本章无法涵盖实现的每个方面，但我们将讨论的基本组件足以让您开始设计和实现自己的搜索引擎。

我们不会去烦恼搜索引擎的用户界面部分。我们最关心的是后端。当谈论应用程序的后端时，通常指的是用户看不到的部分。更具体地说，让我们看一下下面的图表：

![](img/1ef212ea-f319-4913-a62e-db2027a3059c.png)

正如您所看到的，大部分引擎位于后端。虽然用户界面可能感觉简单，但它是整个搜索系统的重要部分。这是用户开始他们旅程的地方，界面设计得越好，用户在搜索时的不适感就越少。我们将集中在后端；以下是我们将讨论的几个主要模块：

+   **查询解析器**：分析用户查询，规范化单词，并收集查询中每个术语的信息，以便稍后传递给查询处理器。

+   **查询处理器**：使用索引和辅助数据库检索与查询相关的数据，并构建响应。

+   **对话生成器**：为用户在搜索时提供更多选择。对话生成器是一个辅助模块。发出请求的用户可以省略对话，也可以使用它来进一步缩小搜索结果。

我们跳过了一些在搜索引擎中常见的组件（如爬虫），而是集中在与基于对话框的搜索引擎密切相关的组件上。现在让我们从查询解析器开始。

# 实现查询解析器

查询解析器做的就是其名字所暗示的：*解析*查询。作为查询解析器的基本任务，我们应该通过空格来分隔单词。例如，用户查询*zeplin best album*被分成以下术语：`zeplin`，`best`和`album`。以下类表示基本的查询解析器：

```cpp
// The Query and Token will be defined in the next snippet
class QueryParser
{
public:
  static Query parse(const std::string& query_string) {
 auto tokens = QueryParser::tokenize(query_string);
    // construct the Query object and return
    // see next snippet for details
 }

private:
  static std::vector<Token> tokenize(const std::string& raw_query) {
    // return tokenized query string
  }
};
```

看一下前面的`parse()`函数。这是类中唯一的公共函数。我们将添加更多的私有函数，这些函数从`parse()`函数中调用，以完全解析查询并将结果作为`Query`对象返回。`Query`表示一个简单的结构，包含有关查询的信息，如下所示：

```cpp
struct Query
{
  std::string raw_query;
  std::string normalized_query;
  std::vector<Token> tokens;
  std::string dialog_id; // we will use this in Dialog Generator
};
```

`raw_query`是用户输入的查询的文本表示，而`normalized_query`是规范化后的相同查询。例如，如果用户输入*good books, a programmer should read*，`raw_query`就是这个确切的文本，而`normalized_query`是*good books programmer should read*。在下面的片段中，我们不使用`normalized_query`，但在完成实现时您将需要它。我们还将标记存储在`Token`向量中，其中`Token`是一个结构，如下所示：

```cpp
struct Token
{
  using Word = std::string;
  using Weight = int;
  Word value;
  std::unordered_map<Word, Weight> related;
};
```

`related`属性表示与标记**语义相关**的单词列表。如果两个单词在概念上表达相似的含义，我们称它们为**语义相关**。例如，单词*best*和*good*，或者*album*和*collection*可以被认为是语义相关的。您可能已经猜到了哈希表值中权重的目的。我们使用它来存储相似性的`Weight`。

**权重**的范围是在利用搜索引擎的过程中应该进行配置的内容。假设我们选择的范围是从 0 到 99。单词*best*和*good*的相似性权重可以表示为接近 90 的数字，而单词*album*和*collection*的相似性权重可能在 40 到 70 之间偏离。选择这些数字是棘手的，它们应该在引擎的开发和利用过程中进行调整。

最后，`Query`结构的`dialog_id`表示如果用户选择了生成器建议的路径，则生成的对话的 ID。我们很快就会谈到这一点。现在让我们继续完成`parse()`函数。

看一下`QueryParser`类的以下补充内容：

```cpp
class QueryParser
{
public:
  static Query parse(const std::string& query_string, 
                     const std::string& dialog_id = "")
  {
    Query qr;
    qr.raw_query = query_string;
    qr.dialog_id = dialog_id;
    qr.tokens = QueryParser::tokenize(query_string);
    QueryParser::retrieve_word_relations(qr.tokens);
    return qr;
  }

private:
  static std::vector<Token> tokenize(const std::string& raw_string) {
    // 1\. split raw_string by space
    // 2\. construct for each word a Token
    // 3\. return the list of tokens 
  }

  static void retrieve_word_relations(std::vector<Token>& tokens) {
    // for each token, request the Knowledge Base
    // to retrieve relations and update tokens list
  }
};
```

尽管前面的代码片段中的两个私有函数（`tokenize`和`retrieve_word_relations`）没有实现，但基本思想是对搜索查询进行规范化和收集信息。在继续实现查询处理器之前，请查看前面的代码。

# 实现查询处理器

查询处理器执行搜索引擎的主要工作，即从搜索索引中检索结果，并根据搜索查询响应相关的文档列表。在本节中，我们还将涵盖对话生成。

正如您在前一节中看到的，查询解析器构造了一个包含标记和`dialog_id`的`Query`对象。我们将在查询处理器中使用这两者。

由于可扩展性问题，建议为对话生成器单独设计一个组件。出于教育目的，我们将保持实现简洁，但您可以重新设计基于对话的搜索引擎，并完成与爬虫和其他辅助模块的实现。

`Query`对象中的标记用于向搜索索引发出请求，以检索与每个单词相关联的文档集。以下是相应的`QueryProcessor`类的外观：

```cpp
struct Document {
  // consider this
};

class QueryProcessor
{
public:
  using Documents = std::vector<Document>;
  static Documents process_query(const Query& query) {
 if (!query.dialog_id.empty()) {
 // request the knowledge graph for new terms
 }
 // retrieve documents from the index
 // sort and return documents
 }
};
```

将前面的代码片段视为实现的介绍。我们希望表达`QueryProcessor`类的基本思想。它具有`process_query()`函数，根据查询参数中的标记从索引中检索文档。这里的关键作用由搜索索引发挥。我们定义其构造方式和存储文档的方式对于进行快速查询至关重要。同时，作为附加参数提供的对话 ID 允许`process_query()`函数请求知识库（或知识图）以检索与查询相关的更多相关标记。

还要考虑到`QueryProcessor`还负责生成对话（即定义一组路径，为用户提供查询的可能场景）。生成的对话将发送给用户，当用户进行另一个查询时，使用的对话将通过我们已经看到的对话 ID 与该查询相关联。

尽管前面的实现大多是介绍性的（因为实际代码的规模太大，无法放入本章），但它是您进一步设计和实现引擎的良好基础。

# 总结

从头开始构建搜索引擎是一项需要经验丰富的程序员来完成的任务。本书涉及了许多主题，并在本章中通过设计搜索引擎将大部分主题结合起来。

我们已经了解到，网络搜索引擎是由爬虫、索引器和用户界面等多个组件组成的复杂系统。爬虫负责定期检查网络，下载网页供搜索引擎索引。索引会产生一个名为倒排索引的大型数据结构。倒排索引，或者简称索引，是一种将单词与它们出现的文档进行映射的数据结构。

接下来，我们定义了推荐引擎是什么，并尝试为我们的搜索引擎设计一个简单的推荐引擎。推荐引擎与本章讨论的基于对话的搜索引擎功能相连。基于对话的搜索引擎旨在向用户提供有针对性的问题，以更好地了解用户实际想要搜索的内容。

通过从 C++的角度讨论计算机科学的各种主题，我们完成了本书的阅读。我们从 C++程序的细节开始，然后简要介绍了使用数据结构和算法进行高效问题解决。了解一种编程语言并不足以在编程中取得成功。您需要解决需要数据结构、算法、多线程等技能的编码问题。此外，解决不同的编程范式可能会极大地增强您对计算机科学的认识，并使您以全新的方式看待问题解决。在本书中，我们涉及了几种编程范式，比如函数式编程。

最后，正如您现在所知，软件开发不仅仅局限于编码。架构和设计项目是成功应用开发的关键步骤之一。第十章，*设计面向全球的应用程序*，到第十六章，*实现基于对话的搜索*，大部分与设计现实世界应用程序的方法和策略有关。让本书成为您从 C++开发者的角度进入编程世界的入门指南。通过开发更复杂的应用程序来发展您的技能，并与同事和刚刚开始职业生涯的人分享您的知识。学习新知识的最佳方式之一就是教授它。

# 问题

1.  爬虫在搜索引擎中的作用是什么？

1.  为什么我们称搜索索引为倒排索引？

1.  令牌化单词在索引之前的主要规则是什么？

1.  推荐引擎的作用是什么？

1.  知识图是什么？

# 进一步阅读

有关更多信息，请参考以下书籍：

*信息检索导论*，*Christopher Manning 等*，[`www.amazon.com/Introduction-Information-Retrieval-Christopher-Manning/dp/0521865719/`](https://www.amazon.com/Introduction-Information-Retrieval-Christopher-Manning/dp/0521865719/)
