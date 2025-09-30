# 14

# 理解 STL 基础

本章将继续我们扩展 C++ 编程知识库的追求，通过深入研究一个核心 C++ 库，该库已经彻底融入了语言的常见用法。我们将通过检查该库的子集来探索 C++ 中的 **标准模板库** (**STL**)，这些子集代表了一些常见的工具，它们既可以简化我们的编程，也可以使熟悉 STL 的他人更容易理解我们的代码。

在本章中，我们将涵盖以下主要内容：

+   概述 C++ 中 STL 的内容和目的

+   理解如何使用基本 STL 容器 – `list`、`iterator`、`vector`、`deque`、`stack`、`queue`、`priority_queue`、`map` 和 `map` 通过一个函数对象使用

+   自定义 STL 容器

到本章结束时，你将能够利用核心 STL 类来提高你的编程技能。因为你已经理解了构建库所必需的基本 C++ 语言和面向对象编程特性，你会发现你现在有能力导航和理解几乎任何 C++ 类库，包括 STL。通过熟悉 STL，你将能够显著扩展你的编程知识库，并成为一个更加精明和有价值的程序员。

让我们通过检查一个高度使用的类库——STL，来增加我们的 C++ 工具箱。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub 网址找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter14`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter14)。每个完整程序示例都可以在 GitHub 仓库中找到，位于相应章节标题（子目录）下的文件中，该文件以章节编号开头，后面跟着一个连字符，然后是本章中的示例编号。例如，本章的第一个完整程序可以在上述 GitHub 目录下的 `Chapter14` 子目录中找到，文件名为 `Chp14-Ex1.cpp`。

本章的 CiA 视频可以在以下网址观看：[`bit.ly/3PCL5IJ`](https://bit.ly/3PCL5IJ)。

# 概述 STL 的内容和目的

C++ 中的 **标准模板库** 是一个标准类和工具的库，它扩展了 C++ 语言。STL 的使用非常普遍，以至于它似乎成了语言本身的一部分；它是 C++ 的一个基本且不可或缺的部分。C++ 中的 STL 由四个关键组件组成，构成了库：**容器**、**迭代器**、**函数**和**算法**。

STL 还影响了 C++标准库，提供了编程标准的一套；这两个库实际上共享一些共同的特征和组件，最显著的是容器和迭代器。我们已经使用了标准库的组件，例如`<iostream>`用于 IOStreams，`<exception>`用于异常处理，以及`<new>`用于`new()`和`delete()`运算符。在本章中，我们将探讨 STL 和 C++标准库之间的许多重叠组件。

STL 拥有一系列完整的**容器**类。这些类封装了传统的数据结构，以便将相似的项目收集在一起并统一处理。容器类分为几个类别——顺序、关联和无序。让我们总结这些类别，并给出每个类别的几个示例：

+   `list`、`queue`或`stack`。值得注意的是，`queue`和`stack`可以被视为对更基本容器（如`list`）的定制或自适应接口。尽管如此，`queue`和`stack`仍然提供对其元素的顺序访问。

+   `set`或`map`。

+   `unordered_set`或`unordered_map`。

为了使这些容器类能够用于任何数据类型（并保持强类型检查），我们使用了模板来抽象和泛化收集项的数据类型。实际上，我们在*第十三章*“使用模板”中使用了模板构建了自己的容器类，包括`LinkList`和`Array`，因此我们已经对模板化的容器类有了基本了解！

此外，STL 还提供了一套完整的**迭代器**，使我们能够“遍历”或遍历容器。迭代器跟踪我们的当前位置，而不会破坏相应对象集合的内容或顺序。我们将看到迭代器如何使我们能够在 STL 中更安全地处理容器类。

STL 还包含大量有用的**算法**。例如，排序、计算可能满足条件的集合中元素的数量、在元素内搜索特定元素或子序列，或以各种方式复制元素。其他算法示例包括修改对象序列（替换、交换和删除值）、将集合划分为范围，或将集合合并在一起。此外，STL 还包含许多其他有用的算法和实用工具。

最后，STL 包括函数。实际上，更准确的说法是 STL 包括`operator()`（函数调用运算符），通过这样做，我们可以通过函数指针实现参数化的灵活性。尽管这不是 STL 的初级特性，我们将在本章中看到一个与 STL 容器类结合的小型、简单的示例，在即将到来的部分*使用函数对象检查 STL map*中。

在本章中，我们将关注 STL 的容器类部分。尽管我们不会检查 STL 中的每个容器类，但我们将回顾这些类中的大量内容。我们会注意到，其中一些容器类与我们在这本书的前几章中一起构建的类相似。顺便提一下，在本书的增量章节进展中，我们也构建了 C++ 语言和 OOP 技能，这些技能是解码像 STL 这样的 C++ 类库所必需的。

让我们继续前进，看看选择性的 STL 类，并在解释每个类的同时检验我们的 C++ 知识。

# 理解如何使用基本的 STL 容器

在本节中，我们将通过解码各种 STL 容器类来检验我们的 C++ 技能。我们将看到，从核心 C++ 语法到 OOP 技能，我们已经掌握的语言特性使我们能够轻松地解释我们现在将要检查的 STL 的各种组件。最值得注意的是，我们将运用我们的模板知识！例如，我们的封装和继承知识将指导我们理解如何在 STL 类中使用各种方法。然而，我们会注意到，在 STL 中虚拟函数和抽象类非常罕见。掌握 STL 中新类的能力的最佳方式是拥抱详细说明每个类的文档。有了 C++ 的知识，我们可以轻松地导航到给定的类，解码如何成功使用它。

C++ STL 中的容器类实现了各种 `list`、`iterator`、`vector`、`deque`、`stack`、`queue`、`priority_queue` 和 `map`。

让我们从检查如何利用一个非常基本的 STL 容器 `list` 开始。

## 使用 STL list

STL 的 `list` 类封装了实现链表所需的数据结构。我们可以这样说，`list` 实现了链表的抽象数据类型。回想一下，我们在 *第六章*，*使用继承实现层次结构* 中通过创建 `LinkedListElement` 和 `LinkedList` 类来构建了自己的链表。STL `list` 允许轻松地插入、删除和排序元素。不支持对单个元素的直接访问（称为 *随机访问*）。相反，你必须迭代地遍历链表中的前一个项目，直到达到所需的项目。STL `list` 是顺序容器的一个很好的例子。

STL 的 `list` 实际上支持对其元素的 bidirectional sequential access（它使用双链表实现）。STL 还提供了 `forward_list`，允许以比 `list` 更小的内存占用对元素进行 unidirectional sequential access；`forward_list` 使用单链表实现（就像我们的 `LinkedList` 类一样）。

STL 的 `list` 类有许多成员函数；我们将从这个例子中开始，看看一些流行的函数，以熟悉基本 STL 容器类的使用。

现在，让我们看看我们如何利用 STL `list` 类。这个例子可以在我们的 GitHub 上找到，作为一个完整的、带有必要类定义的工作程序，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex1.cpp)

```cpp
#include <list>
using std::list;
int main()
{   
    list<Student> studentBody;   // create a list
    Student s1("Jul", "Li", 'M', "Ms.", 3.8, "C++",
               "117PSU");
    // Note: simple heap instance below, later you can opt
    // for a smart pointer to ease allocation/deallocation
    Student *s2 = new Student("Deb", "King", 'H', "Dr.", 
                              3.8, "C++", "544UD");
    // Add Students to the studentBody list. 
    studentBody.push_back(s1);
    studentBody.push_back(*s2);
    // The next 3 instances are anonymous objects in main()
    studentBody.push_back(Student("Hana", "Sato", 'U', 
                          "Dr.", 3.8, "C++", "178PSU"));
    studentBody.push_back(Student("Sara", "Kato", 'B',
                          "Dr.", 3.9, "C++", "272PSU"));
    studentBody.push_back(Student("Giselle", "LeBrun", 'R',
                          "Ms.", 3.4, "C++", "299TU"));
    while (!studentBody.empty())
    {
       studentBody.front().Print();
       studentBody.pop_front();
    }
    delete s2;  // delete any heap instances
    return 0;
}
```

让我们检查上述程序段，其中我们创建并使用了一个 STL `list`。首先，我们 `#include <list>` 来包含适当的 STL 头文件。我们还添加 `using std::list;` 以从标准命名空间包含 `list`。现在，在 `main()` 中，我们可以使用 `list<Student> studentBody;` 实例化一个列表。我们的列表将包含 `Student` 实例。然后，我们使用 `new()` 分配在栈上创建 `Student s1` 和在堆上创建 `Student *s2`。

接下来，我们使用 `list::push_back()` 将 `s1` 和 `*s2` 都添加到列表中。注意，我们正在将对象传递给 `push_back()`。当我们将 `Student` 实例添加到 `studentBody` 列表中时，列表将内部复制这些对象，并在它们不再是列表的成员时适当地清理这些对象。我们需要记住，如果我们的任何实例已经在堆上分配，例如 `*s2`，那么在 `main()` 结束时，我们必须删除该实例的副本。展望 `main()` 的结尾，我们可以看到我们适当地 `delete s2;`。

接下来，我们将另外三个学生添加到列表中。这些 `Student` 实例没有局部标识符。这些学生是在 `push_back()` 调用中实例化的，例如，`studentBody.push_back(Student("Hana", "Sato", 'U', "Dr.", 3.8, "C++", "178PSU"));`。在这里，我们实例化了一个 *匿名（栈）对象*，它将在 `push_back()` 调用结束时从栈中正确弹出并销毁。请注意，`push_back()` 还将为这些实例创建它们在 `list` 中的生命周期内的本地副本。

现在，在一个 while 循环中，我们反复检查列表是否 `empty()`，如果不是，我们检查 `front()` 项并调用我们的 `Student::Print()` 方法。然后我们使用 `pop_front()` 从列表中移除该项。

让我们看看这个程序的输出：

```cpp
Ms. Jul M. Li with id: 117PSU GPA:  3.8 Course: C++
Dr. Deb H. King with id: 544UD GPA:  3.8 Course: C++
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
Ms. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
```

现在我们已经解析了一个简单的 STL `list` 类，让我们继续了解 `iterator` 的概念，以补充像 `list` 这样的容器。

## 使用 STL 迭代器

很频繁地，我们需要一种非破坏性的方法来遍历一组对象。例如，在给定的容器中维护第一个、最后一个和当前位置很重要，特别是如果该集合可能被多个方法、类或线程访问。使用 **迭代器**，STL 提供了一种遍历任何容器类的通用方法。

迭代器的使用具有明显的优势。一个类可以创建一个指向集合中第一个成员的 `iterator`。然后迭代器可以被移动到集合的后续下一个成员。迭代器可以提供对由 `iterator` 指向的元素访问。

总体来说，一个容器的状态信息可以通过 `iterator` 维护。迭代器提供了一种安全的方法，通过将状态信息从容器抽象出来，而不是放入迭代器类中，来实现交错访问。

我们可以将迭代器想象成一本书中的书签，两个人或更多人正在参考。第一个人按顺序阅读书籍，将书签整洁地放在他们期望继续阅读的地方。当第一个人离开时，另一个人在书中查找一个重要项目，并将书签移动到书中的另一个位置以保存他们的位置。当第一个人回来时，他们会发现自己失去了当前的位置，并不在他们期望的地方。每个用户都应该有自己的书签或迭代器。这个类比是，迭代器（理想情况下）允许安全地交错访问可能由应用程序中的多个组件处理的一个资源。如果没有迭代器，你可能会无意中修改一个容器，而其他用户并不知道。STL 迭代器大多数情况下，但并非总是，能够达到这个理想目标。

让我们看看如何利用 STL `iterator`。这个例子可以在我们的 GitHub 上作为一个完整的程序找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex2.cpp)

```cpp
#include <list>
#include <iterator>
using std::list;
using std::iterator;
bool operator<(const Student &s1, const Student &s2)
{   // overloaded operator< -- required to use list::sort()
    return s1.GetGpa() < s2.GetGpa();
}
int main()
{
    list<Student> studentBody;  
    Student s1("Jul", "Li", 'M', "Ms.", 3.8, "C++",
               "117PSU");
    // Add Students to the studentBody list.
    studentBody.push_back(s1);
    // The next Student instances are anonymous objects
    studentBody.push_back(Student("Hana", "Sato", 'U',
                          "Dr.", 3.8, "C++", "178PSU"));
    studentBody.push_back(Student("Sara", "Kato", 'B',
                          "Dr.", 3.9, "C++", "272PSU"));
    studentBody.push_back(Student("Giselle", "LeBrun", 'R',
                          "Ms.", 3.4, "C++", "299TU"));
    studentBody.sort();  // sort() will rely on operator< 
    // Though we'll generally prefer range-for loops, let's
    // understand and demo using an iterator for looping.
    // Create a list iterator; set to first item in list.
    // We'll next simplify iterator notation with 'auto'.
    list <Student>::iterator listIter =studentBody.begin();
    while (listIter != studentBody.end())
    {
        Student &temp = *listIter;
        temp.EarnPhD();
        ++listIter;    // prefer pre-inc (less expensive)
    } 
    // Simplify iterator declaration using 'auto'
    auto autoIter = studentBody.begin();
    while (autoIter != studentBody.end())
    {
        (*autoIter).Print();  
        ++autoIter;
    }
    return 0;
}
```

让我们来看看之前定义的代码段。在这里，我们包含了 STL 中的 `<list>` 和 `<iterator>` 头文件。我们还添加了 `using std::list;` 和 `using std::iterator;` 来包含标准命名空间中的 `list` 和 `iterator`。与之前的 `main()` 函数一样，我们使用 `list<Student> studentbody;` 实例化了一个可以包含 `Student` 实例的 `list`。然后我们实例化几个 `Student` 实例，并使用 `push_back()` 将它们添加到列表中。再次注意，几个 `Student` 实例是 *匿名对象*，在 `main()` 中没有局部标识符。这些实例将在 `push_back()` 完成时从栈中弹出。这没有问题，因为 `push_back()` 将为列表创建局部副本。

现在，我们可以使用 `studentBody.sort();` 对列表进行排序。需要注意的是，这个 `list` 方法要求我们重载 `operator<` 以提供两个 `Student` 实例之间比较的方法。幸运的是，我们已经做到了！我们选择通过比较 `gpa` 来实现 `operator<`，但它也可以使用 `studentId` 进行比较。

现在我们有了`list`，我们可以创建一个`iterator`并将其设置为指向`list`的第一个项目。我们通过声明`list <Student>::iterator listIter = studentBody.begin();`来这样做。一旦建立了迭代器，我们就可以使用它安全地从开始（因为它被初始化）到`end()`循环遍历`list`。我们通过`Student &temp = *listIter;`将局部引用变量`temp`赋值给列表循环迭代的当前第一个元素。然后我们通过`temp.EarnPhD();`在这个实例上调用一个方法，然后通过`++listIter;`将迭代器增加一个元素。

在随后的循环中，我们使用 `auto` 简化了迭代器的声明。`auto` 关键字允许迭代器的类型由其初始使用确定。在这个循环中，我们还消除了对 `temp` 的使用——我们只需在括号内取消迭代器的引用，然后使用 `(*autoIter).Print()` 调用 `Print()`。使用 `++autoIter` 简单地前进到列表中的下一个项目以进行处理。

让我们看看这个程序的排序输出（按 `gpa` 排序）：

```cpp
Dr. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
Dr. Jul M. Li with id: 117PSU GPA:  3.8 Course: C++
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
```

现在我们已经看到了`iterator`类的实际应用，让我们来调查一些额外的 STL 容器类，从`vector`开始。

## 使用 STL `vector`

STL 的 `vector` 类实现了动态数组的抽象数据类型。回想一下，我们在 *第十三章*，*使用模板* 中通过创建一个 `Array` 类来创建了自己的动态数组。然而，STL 版本将更加广泛。

`vector`（动态或可调整大小的数组）将根据需要扩展以容纳超出其初始大小的额外元素。`vector`类通过重载`operator[]`允许直接（即*随机访问*）访问元素。访问特定索引的元素不需要遍历所有先前元素。

然而，在`vector`的中间添加元素是耗时的。也就是说，除了在`vector`的末尾添加之外，还需要将插入点之后的所有元素在内部重新排列；这还可能需要`vector`的内部调整大小。

显然，与`list`和`vector`相比，它们有不同的优点和缺点。每个都是针对数据集合的不同需求而设计的。我们可以选择最适合我们需求的一个。

让我们看看一些常见的`vector`成员函数。这远非一个完整的列表：

![图片](img/Figure_14.1_B19087.jpg)

STL 的 `vector` 类还包括重载的 `operator=`（赋值用源 `vector` 替换目标 `vector`），`operator==`（逐元素比较向量），和 `operator[]`（返回对请求位置的引用，即可写内存）。

让我们看看我们如何利用 STL `vector` 类及其基本操作。这个例子可以作为完整的程序在 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex3.cpp)

```cpp
#include <vector>
using std::vector;
int main()
{   // instantiate two vectors
    vector<Student> studentBody1, studentBody2; 
    // add 3 Students, which are anonymous objects 
    studentBody1.push_back(Student("Hana", "Sato", 'U',
"Dr.", 3.8, "C++", "178PSU"));
    studentBody1.push_back(Student("Sara", "Kato", 'B',
                           "Dr.", 3.9, "C++", "272PSU"));
    studentBody1.push_back(Student("Giselle", "LeBrun",
                         'R', "Ms.", 3.4, "C++", "299TU"));
    // Compare this loop to next loop using an iterator and
    // also to the preferred range-for loop further beyond
    for (int i = 0; i < studentBody1.size(); i++)   
        studentBody1[i].Print();   // print first vector
    studentBody2 = studentBody1;   // assign one to another
    if (studentBody1 == studentBody2)
        cout << "Vectors are the same" << endl;
    // Notice: auto keyword simplifies iterator declaration
    for (auto iter = studentBody2.begin();
              iter != studentBody2.end(); iter++)
        (*iter).EarnPhD();
   // Preferred range-for loop (and auto to simplify type)
    for (const auto &student : studentBody2)
        student.Print();
    if (!studentBody1.empty())   // clear first vector 
        studentBody1.clear();
    return 0;
}
```

在之前列出的代码段中，我们 `#include <vector>` 以包含适当的 STL 头文件。我们还添加 `using std::vector;` 以从标准命名空间包含 `vector`。现在，在 `main()` 中，我们可以使用 `vector<Student> studentBody1, studentBody2;` 实例化两个向量。然后，我们可以使用 `vector::push_back()` 方法连续将几个 `Student` 实例添加到我们的第一个向量中。再次注意，`Student` 实例在 `main()` 中是 *匿名对象*。也就是说，没有局部标识符引用它们——它们仅被创建以放入我们的向量中，在插入时为每个实例创建局部副本。一旦我们在向量中有元素，我们就遍历第一个向量，使用 `studentBody1[i].Print();` 打印每个 `Student`。

接下来，我们通过使用 `studentBody1 = studentBody2;` 将一个向量赋值给另一个向量来演示 `vector` 的重载赋值运算符。在这里，我们在赋值过程中从右向左进行深度复制。然后，我们可以使用条件语句中的重载比较运算符来测试两个向量是否相等，即 `if (studentBody1 == studentBody2);`。

然后，我们使用 `auto iter = studentBody2.begin();` 指定的迭代器在 `for` 循环中对第二个向量的内容应用 `EarnPhD()`。`auto` 关键字允许迭代器的类型由其初始使用确定。然后，我们使用首选的范围-for 循环（以及使用 `auto` 简化范围-for 循环中的变量类型）打印出第二个向量的内容。最后，我们检查第一个 `vector` 是否为 `empty()`，然后使用 `studentBody1.clear();` 逐个清除元素。我们现在已经看到了 `vector` 方法和它们的功能的样本。

让我们看看这个程序的输出：

```cpp
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
Ms. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
Vectors are the same
Everyone to earn a PhD
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
Dr. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
```

接下来，让我们调查 STL `deque` 类，以进一步了解 STL 容器。

## 使用 STL deque

STL `deque` 类（发音为 *deck*）实现了双端队列的抽象数据类型。这种 ADT 扩展了队列是先进先出的概念。相反，`deque` 类提供了更大的灵活性。在 `deque` 的两端添加元素是快速的。在 `deque` 的中间添加元素是耗时的。`deque` 是一个顺序容器，尽管比列表更灵活。

你可能会想象`deque`是`queue`的一个特殊化；它不是。相反，灵活的`deque`类将作为实现其他容器类的基础，我们很快就会看到。在这些情况下，私有继承将允许我们隐藏`deque`作为底层实现（具有广泛的功能）以供更限制性、特殊化的类使用。

让我们看看一些常见的`deque`成员函数。这远非一个完整的列表：

![图片](img/Figure_14.2_B19087.jpg)

STL 的`deque`类还包括重载的`operator=`（源到目标`deque`的赋值）和`operator[]`（返回请求位置的引用——可写内存）。

让我们看看我们如何利用 STL 的`deque`类。这个例子作为一个完整的程序可以在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex4.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex4.cpp)

```cpp
#include <deque> 
using std::deque;
int main()
{
    deque<Student> studentBody;   // create a deque
    Student s1("Tim", "Lim", 'O', "Mr.", 3.2, "C++",
               "111UD");
    // the remainder of the Students are anonymous objects
    studentBody.push_back(Student("Hana", "Sato", 'U',
                          "Dr.",3.8, "C++", "178PSU"));
    studentBody.push_back(Student("Sara", "Kato", 'B',
                          "Dr.", 3.9, "C++", "272PSU"));
    studentBody.push_front(Student("Giselle", "LeBrun",
                          'R',"Ms.", 3.4, "C++", "299TU"));
    // insert one past the beginning 
    studentBody.insert(std::next(studentBody.begin()), 
    Student("Anne", "Brennan", 'B', "Ms.", 3.9, "C++",
            "299CU"));
    studentBody[0] = s1;  // replace 0th element; 
                          // no bounds checking!
    while (!studentBody.empty())
    {
        studentBody.front().Print();
        studentBody.pop_front();
    }
    return 0;
}
```

在之前列出的代码段中，我们`#include <deque>`来包含适当的 STL 头文件。我们还添加了`using std::deque;`来从标准命名空间包含`deque`。现在，在`main()`中，我们可以实例化一个`deque`来包含`Student`实例，使用`deque<Student> studentBody;`。然后我们调用`deque::push_back()`或`deque::push_front()`来向我们的`deque`添加几个`Student`实例（一些匿名对象）。我们正在掌握这个！现在，我们使用`studentBody.insert(std::next(studentBody.begin()), Student("Anne", "Brennan", 'B', "Ms.", 3.9, "C++", "299CU"));`在`deque`的前端之后插入一个`Student`。

接下来，我们利用重载的`operator[]`将一个`Student`插入到我们的`deque`中，使用`studentBody[0] = s1;`。请务必注意，`operator[]`对我们的`deque`不进行任何边界检查！在这个语句中，我们将`Student` `s1`插入到`deque`的第 0 个位置，而不是曾经占据那个位置的`Student`。一个更安全的做法是使用`deque::at()`方法，它将包含边界检查。关于上述赋值，我们还想确保`operator=`已经为`Person`和`Student`两个类重载，因为每个类都有动态分配的数据成员。

现在，我们循环直到我们的`deque`为`empty()`，使用`studentBody.front().Print();`提取并打印`deque`的前端元素。在每次迭代中，我们也会使用`studentBody.pop_front();`从`deque`中弹出前端的项目。

让我们看看这个程序的输出：

```cpp
Mr. Tim O. Lim with id: 111UD GPA:  3.2 Course: C++
Ms. Anne B. Brennan with id: 299CU GPA:  3.9 Course: C++
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
```

既然我们对`deque`有了感觉，接下来让我们研究一下 STL 的`stack`类。

## 使用 STL 栈

STL 的 `stack` 类实现了栈的抽象数据类型。栈 ADT 包含一个公共接口，该接口不公开其底层实现。毕竟，栈可能会更改其实现；ADT 的使用不应以任何方式依赖于其底层实现。STL 的 `stack` 被视为基本顺序容器的自适应接口。

回想一下，我们在 *第六章*，*使用继承实现层次结构* 中创建了自己的 `Stack` 类，使用 `LinkedList` 作为私有基类。STL 版本将更加丰富；有趣的是，它使用 `deque` 作为其底层的私有基类。由于 `deque` 是 STL `stack` 的私有基类，`deque` 的更多通用底层能力被隐藏；只使用适用的方法来实现栈的公共接口。此外，由于实现方式被隐藏，`stack` 可以在以后使用另一个容器类实现，而不会影响其使用。

让我们看看一些常见的 `stack` 成员函数。这远非一个完整的列表。重要的是要注意，`stack` 的公共接口远小于其私有基类 `deque` 的接口：

![](img/Figure_14.3_B19087.jpg)

STL 的 `stack` 类还包括重载的 `operator=`（源栈到目标栈的赋值）、`operator==` 和 `operator!=`（两个栈的相等/不等）、以及 `operator<`、`operator>`、`operator<=` 和 `operator>=`（栈的比较）。

让我们看看如何利用 STL 的 `stack` 类。这个例子可以作为完整的工作程序在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex5.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex5.cpp)

```cpp
#include <stack>   // template class preamble
using std::stack;
int main()
{
    stack<Student> studentBody;   // create a stack
    // add Students to the stack (anonymous objects)
    studentBody.push(Student("Hana", "Sato", 'U', "Dr.",
                             3.8, "C++", "178PSU"));
    studentBody.push(Student("Sara", "Kato", 'B', "Dr.",
                             3.9, "C++", "272PSU"));
    studentBody.push(Student("Giselle", "LeBrun", 'R',
                             "Ms.", 3.4, "C++", "299TU"));
    while (!studentBody.empty())
    {
        studentBody.top().Print();
        studentBody.pop();
    }
    return 0;
}
```

在上述代码段中，我们 `#include <stack>` 包含适当的 STL 头文件。我们还添加 `using std::stack;` 来包含标准命名空间中的 `stack`。现在，在 `main()` 中，我们可以使用 `stack<Student> studentBody;` 实例化一个 `stack` 来包含 `Student` 实例。然后，我们调用 `stack::push()` 向我们的 `stack` 添加几个 `Student` 实例。注意，我们正在使用传统的 `push()` 方法，这有助于栈的 ADT。

然后，我们在 `stack` 不为 `empty()` 时循环遍历。我们的目标是使用 `studentBody.top().Print();` 访问并打印栈顶元素。然后，我们使用 `studentBody.pop();` 整洁地弹出栈顶元素。

让我们看看这个程序的输出：

```cpp
Ms. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
```

接下来，让我们研究 STL 的 `queue` 类，以进一步丰富我们的 STL 容器系列。

## 使用 STL 队列

STL 的`queue`类实现了队列 ADT。作为典型的队列类，STL 的`queue`类支持成员插入和删除的**先进先出**（**FIFO**）顺序。

回想一下，我们在*第六章*中自己实现了`Queue`类，*使用继承实现层次结构*；我们使用私有继承从`LinkedList`类派生我们的`Queue`。STL 版本将更加丰富；STL 的`queue`类使用`deque`作为其底层实现（也使用私有继承）。记住，由于实现方式通过私有继承被隐藏，`queue`可以在以后使用其他数据类型实现，而不会影响其公共接口。STL 的`queue`类是基本顺序容器的一个自适应接口的另一个例子。

让我们看看一些常见的`queue`成员函数。这远非一个完整的列表。重要的是要注意，`queue`的公共接口远小于其私有基类`deque`的接口：

![图片](img/Figure_14.4_B19087.jpg)

STL 的`queue`类还包括重载的`operator=`（源队列到目标队列的赋值），`operator==`和`operator!=`（两个队列的相等/不等），以及`operator<`，`operator>`，`operator<=`和`operator>=`（队列的比较）。

让我们看看如何利用 STL 的`queue`类。这个例子可以作为完整的工作程序在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex6.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex6.cpp)

```cpp
#include <queue>  
using std::queue;
int main()
{
    queue<Student> studentBody;  // create a queue
    // add Students to the queue (anonymous objects)
    studentBody.push(Student("Hana", "Sato", 'U', "Dr.",
                             3.8, "C++", "178PSU"));
    studentBody.push(Student("Sara", "Kato", 'B' "Dr.",
3.9, "C++", "272PSU"));
    studentBody.push(Student("Giselle", "LeBrun", 'R',
                             "Ms.", 3.4, "C++", "299TU"));
    while (!studentBody.empty())
    {
        studentBody.front().Print();
        studentBody.pop();
    }
    return 0;
}
```

在之前的代码段中，我们首先`#include <queue>`来包含适当的 STL 头文件。我们还添加了`using std::queue;`以从标准命名空间中包含`queue`。现在，在`main()`中，我们可以使用`queue<Student> studentBody;`来实例化一个`queue`以包含`Student`实例。然后我们调用`queue::push()`来将几个`Student`实例添加到我们的`queue`中。回想一下，在队列 ADT 中，`push()`意味着我们在队列的末尾添加一个元素。一些程序员更喜欢使用术语*enqueue*来描述这个操作；然而，STL 选择了将此操作命名为`push()`。在队列 ADT 中，`pop()`将从队列的前端移除一个项目；一个更好的术语是*dequeue*，但 STL 并没有选择这个术语。我们可以适应。

然后，我们在`queue`不为`empty()`时循环遍历。我们的目标是使用`studentBody.front().Print();`访问并打印前端的元素。然后我们使用`studentBody.pop();`整洁地从`queue`中移除前端元素。我们的工作就完成了。

让我们看看这个程序的输出：

```cpp
Dr. Hana U. Sato with id: 178PSU GPA:  3.8 Course: C++
Dr. Sara B. Kato with id: 272PSU GPA:  3.9 Course: C++
Ms. Giselle R. LeBrun with id: 299TU GPA:  3.4 Course: C++
```

现在我们已经尝试了`queue`，让我们来调查 STL 的`priority_queue`类。

## 使用 STL 优先队列

STL 的`priority_queue`类实现了优先队列的抽象数据类型。优先队列 ADT 支持修改后的 FIFO 插入和删除成员的顺序；元素是*加权*的。最前面的元素是最大值（由重载的`operator<`确定）并且其余元素按照从大到小的顺序依次排列。STL 的`priority_queue`类被认为是顺序容器的自适应接口。

回想一下，我们在*第六章*，*通过继承实现层次结构*中实现了自己的`PriorityQueue`类。我们使用公有继承来允许我们的`PriorityQueue`专门化我们的`Queue`类，添加额外的支持优先级（加权）入队方案的方法。`Queue`的底层实现（带有私有基类`LinkedList`）是隐藏的。通过使用公有继承，我们允许我们的`PriorityQueue`能够通过向上转型（在我们学习了*第七章*，*通过多态利用动态绑定）后作为`Queue`使用。我们做出了一个可接受的设计选择：*PriorityQueue Is-A*（专门化）*Queue*，有时可以以更通用的形式处理。我们还回忆起，`Queue`和`PriorityQueue`都不能向上转型到其底层实现`LinkedList`，因为`Queue`是私有地从`LinkedList`派生的；我们不能越过非公有继承边界进行向上转型。

相比之下，STL 版本的`priority_queue`使用 STL 的`vector`作为其底层实现。回想一下，由于实现方式是隐藏的，`priority_queue`可能在以后使用其他数据类型实现，而不会影响其公共接口。

STL 的`priority_queue`允许检查，但不允许修改，最顶端的元素。STL 的`priority_queue`不允许通过其元素进行插入。也就是说，只能添加元素以产生从大到小的顺序。因此，可以检查最顶端的元素，并且可以移除最顶端的元素。

让我们看看一些常见的`priority_queue`成员函数。这不是一个完整的列表。重要的是要注意，`priority_queue`的公共接口远小于其私有基类`vector`：

![图片](img/Figure_14.5_B19087.jpg)

与之前检查的容器类不同，STL 的`priority_queue`没有重载运算符，包括`operator=`, `operator==`, 和 `operator<`。

`priority_queue` 最有趣的方法是 `void emplace(args);`。这是允许优先级入队机制向此 ADT 添加项的成员函数。我们还注意到必须使用 `top()` 来返回顶部元素（与 `queue` 使用的 `front()` 相比）。但是，STL 的 `priority_queue` 并不是使用 `queue` 实现的。要利用 `priority_queue`，我们需要 `#include <queue>`，就像我们为 `queue` 做的那样。

由于 `priority_queue` 的使用与 `queue` 非常相似，因此我们将在本章末尾的问题集中进一步探讨其编程方面的应用。

现在我们已经看到了许多 STL 中的顺序容器类型示例（包括自适应接口），接下来让我们研究 STL 的 `map` 类，这是一个关联容器。

## 检查 STL `map`

STL 的 `map` 类实现了哈希表的抽象数据类型。`map` 类允许在哈希表或映射中快速存储和检索元素。如果需要将多个数据项与单个键关联起来，可以使用 `multimap`。

哈希表（映射）在存储和查找数据方面非常快。性能保证为 *O(log(n))*。STL 的 `map` 被视为关联容器，因为它将键与值关联起来，以便快速检索值。

让我们看看一些常见的 `map` 成员函数。这不是一个完整的列表：

![图片](img/Figure_14.6_B19087.jpg)

STL 类 `map` 还包括重载的运算符 `operator==`（逐元素比较映射）作为全局函数实现。STL `map` 还包括重载的 `operator[]`（返回与用作索引的键关联的映射元素引用；这是可写内存）。

让我们看看如何利用 STL 的 `map` 类。这个示例可以作为完整的可工作程序在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex7.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex7.cpp)

```cpp
#include <map>
using std::map;
using std::pair;
bool operator<(const Student &s1, const Student &s2)
{   // We need to overload operator< to compare Students
    return s1.GetGpa() < s2.GetGpa();
}
int main()
{
    Student s1("Hana", "Lo", 'U', "Dr.", 3.8, "C++",
               "178UD");
    Student s2("Ali", "Li", 'B', "Dr.", 3.9, "C++",
               "272UD");
    Student s3("Rui", "Qi", 'R', "Ms.", 3.4, "C++",
               "299TU");
    Student s4("Jiang", "Wu", 'C', "Ms.", 3.8, "C++",
               "887TU");
    // create three pairings of ids to Students
    pair<string, Student> studentPair1
                                (s1.GetStudentId(), s1);
    pair<string, Student> studentPair2
                                (s2.GetStudentId(), s2);
    pair<string, Student> studentPair3
                                (s3.GetStudentId(), s3);
    // Create map of Students w string keys
    map<string, Student> studentBody;
    studentBody.insert(studentPair1);  // insert 3 pairs
    studentBody.insert(studentPair2);
    studentBody.insert(studentPair3);
    // insert using virtual indices per map
    studentBody[s4.GetStudentId()] = s4; 
    // Iterate through set with map iterator – let's 
    // compare to range-for and auto usage just below
    map<string, Student>::iterator mapIter;
    mapIter = studentBody.begin();
    while (mapIter != studentBody.end())
    {   
        // set temp to current item in map iterator
        pair<string, Student> temp = *mapIter;
        Student &tempS = temp.second;  // get 2nd element
        // access using mapIter
        cout << temp.first << " ";
        cout << temp.second.GetFirstName();  
        // or access using temporary Student, tempS  
        cout << " " << tempS.GetLastName() << endl;
        ++mapIter;
    }
    // Now, let's iterate through our map using a range-for
    // loop and using 'auto' to simplify the declaration
    // (this decomposes the pair to 'id' and 'student')
    for (auto &[id, student] : studentBody)
        cout << id << " " << student.GetFirstName() << " " 
             << student.GetLastName() << endl;
    return 0;
}
```

让我们检查前面的代码段。同样，我们包含适用的头文件 `#include <map>`。我们还添加了 `using std::map;` 和 `using std::pair;` 以包含标准命名空间中的 `map` 和 `pair`。接下来，我们创建了四个 `Student` 实例。然后，我们创建了三个 `pair` 实例，用于将每个学生与其键（即他们的相应 `studentId`）关联起来，使用声明 `pair<string, Student> studentPair1 (s1.GetStudentId(), s1);`。这可能会让人感到困惑，但让我们把这个声明分解成其组成部分。在这里，实例的数据类型是 `pair<string, Student>`，变量名是 `studentPair1`，`(s1.GetStudentId(), s1)` 是传递给特定 `pair` 实例构造函数的参数。

我们将创建一个以键（即他们的 `studentId`）为索引的 `Student` 实例的哈希表（`map`）。接下来，我们声明一个 `map` 来存储 `Student` 实例的集合，使用 `map<string, Student> studentBody;`。在这里，我们表明键和元素之间的关联将是一个 `string` 和一个 `Student`。然后，我们使用相同的数据类型声明一个 map 迭代器 `map<string, Student>::iterator mapIter;`。

现在，我们只需将三个 `pair` 实例插入到 `map` 中。例如，`studentBody.insert(studentPair1);` 就是一个这样的插入操作。然后，我们使用 `map` 的重载 `operator[]` 将第四个 `Student`，`s4`，插入到 `map` 中，如下所示：`studentBody[s4.GetStudentId()] = s4;`。请注意，`studentId` 被用作 `operator[]` 中的索引值；这个值将成为哈希表中 `Student` 的键值。

接下来，我们声明并设置 map 迭代器到 `map` 的开始处，然后处理 `map`，直到它到达 `end()`。在循环中，我们将一个变量 `temp` 设置为地图前端的 `pair`，由地图迭代器指示。我们还设置 `tempS` 作为 `map` 中一个 `Student` 的临时引用，由 `temp.second`（当前由地图迭代器管理的 `pair` 中的第二个值）指示。现在我们可以使用 `temp.first`（当前 `pair` 中的第一个项目）打印出每个 `Student` 实例的 `studentId`（键，它是一个 `string`）。在同一个语句中，我们可以使用 `temp.second.GetFirstName()` 打印出每个 `Student` 实例的 `firstName`（因为与键对应的 `Student` 是当前 `pair` 中的第二个项目）。同样，我们也可以使用 `tempS.GetLastName()` 打印一个学生的 `lastName`，因为 `tempS` 在每次循环迭代的开始时被初始化为当前 `pair` 中的第二个元素。

最后，作为之前演示的更繁琐的通过 `map` 迭代的方法的替代方案，让我们检查程序中的最后一个循环。在这里，我们使用范围-for 循环来处理 `map`。使用 `auto` 与 `&[id, student]` 将指定我们将迭代的类型数据。括号（`[]`）将分解 `pair`，将迭代元素分别绑定到 `id` 和 `student` 作为标识符。注意我们现在迭代 `studentBody` map 的简便性。

让我们看看这个程序的输出：

```cpp
178UD Hana Lo
272UD Ali Li
299TU Rui Qi
887TU Jiang Wu
178UD Hana Lo
272UD Ali Li
299TU Rui Qi
887TU Jiang Wu
```

接下来，让我们看看 STL `map` 的一个替代方案，这将介绍我们到 STL `functor` 概念。

## 使用函数对象检查 STL map

STL 的 `map` 类具有很大的灵活性，就像许多 STL 类一样。在我们之前的 `map` 示例中，我们假设 `Student` 类中存在一种比较方法。毕竟，我们已经为两个 `Student` 实例重载了 `operator<`。然而，如果我们不能修改一个没有提供这个重载操作符的类，并且我们也不选择将 `operator<` 作为外部函数重载，会发生什么呢？

幸运的是，在实例化 `map` 或 map 迭代器时，我们可以指定一个第三种数据类型用于模板类型扩展。这种额外的数据类型将是一种特定的类，称为函数对象。一个 `operator()`。正是在重载的 `operator()` 中，我们将提供对相关对象的比较方法。函数对象本质上通过重载 `operator()` 来模拟封装函数指针。

让我们看看我们如何修改我们的 `map` 示例以利用一个简单的函数对象。这个例子可以作为完整的程序在 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex8.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter14/Chp14-Ex8.cpp)

```cpp
#include <map>
using std::map;
using std::pair;
struct comparison   // This struct represents a 'functor'
{                   // that is, a 'function object'
    bool operator() (const string &key1, 
                     const string &key2) const
    {   
        int ans = key1.compare(key2);
        if (ans >= 0) return true;  // return a boolean
        else return false;  
    }
    // default constructor and destructor are adequate
};
int main()
{
    Student s1("Hana", "Sato", 'U', "Dr.", 3.8, "C++", 
               "178PSU");
    Student s2("Sara", "Kato", 'B', "Dr.", 3.9, "C++",
               "272PSU");
    Student s3("Jill", "Long", 'R', "Dr.", 3.7, "C++",
               "234PSU");
    // Now, map is maintained in sorted (decreasing) order
    // per ‹comparison› functor using operator()
    map<string, Student, comparison> studentBody;
    map<string, Student, comparison>::iterator mapIter;
    // The remainder of the program is similar to prior
}   // map program. See online code for complete example.
```

在之前提到的代码片段中，我们首先引入了一个用户定义的 `comparison` 类型。这可以是一个 `class` 或 `struct`。在这个结构定义中，我们重载了函数调用操作符 (`operator()`)，并为 `Student` 实例的两个 `string` 键提供了一个比较方法。这种比较将允许 `Student` 实例按照比较函数对象的顺序插入。

现在，当我们实例化我们的 `map` 和 map 迭代器时，我们将模板类型扩展的第三个参数指定为我们的 `comparison` 类型（函数对象）。并且，在这个类型中巧妙地嵌入了一个重载的函数调用操作符 `operator()`，它将提供所需的比较。剩余的代码将与我们的原始 `map` 程序类似。

当然，函数对象可以以比我们在这里看到的容器类 `map` 更多的方式使用，更高级的方式。不过，你现在已经对函数对象如何应用于 STL 有了一定的了解。

现在我们已经看到了如何利用各种 STL 容器类，让我们考虑为什么我们可能想要自定义一个 STL 类，以及如何做到这一点。

# 自定义 STL 容器

大多数 C++类都可以以某种方式自定义，包括 STL 中的类。然而，我们必须意识到 STL 内部做出的设计决策，这些决策将限制我们如何自定义这些组件。因为 STL 容器类故意不包含虚析构函数或其他虚函数，所以我们不应该通过公有继承来对这些类进行特殊化。请注意，C++不会阻止我们这样做，但我们从*第七章*，*通过多态使用动态绑定*中了解到，我们永远不应该重写非虚函数。STL 选择不包含虚析构函数和其他虚函数，以允许对这些类进行进一步的特殊化，这是很久以前在 STL 容器被设计时做出的一个稳健的设计选择。

然而，我们可以使用私有或保护继承，或者使用包含或关联的概念，将 STL 容器类作为构建块来使用。也就是说，为了隐藏新类的底层实现，其中 STL 为新类提供了一个稳健且隐藏的实现。我们只需为新类提供一个自己的公共接口，并在幕后将工作委托给我们的底层实现（无论是私有或保护基类，还是包含或关联的对象）。

在扩展任何模板类时，包括使用私有或保护基类的 STL 中的模板类，必须非常小心和谨慎。这种谨慎也适用于包含或关联到其他模板类。模板类通常只有在创建了具有特定类型的模板类实例之后才会编译（或进行语法检查）。这意味着创建的任何派生或包装类只能在创建了特定类型的实例后才能完全测试。

对于新类，需要放置适当的重载运算符，以便这些运算符可以自动与自定义类型一起工作。请记住，某些运算符函数，如`operator=`，并不是从基类显式继承到派生类的，并且需要为每个新类编写。这是合适的，因为派生类可能需要完成比`operator=`的通用版本更多的工作。记住，如果你不能修改需要选择重载运算符的类的定义，你必须将该运算符函数实现为一个外部函数。

除了自定义容器外，我们还可以选择增强 STL 中现有算法的算法。在这种情况下，我们会使用许多 STL 函数之一作为新算法底层实现的一部分。

在编程中，从现有库中定制类是常见的。例如，考虑我们如何在*第十一章*“处理异常”中扩展标准库 `exception` 类以创建自定义异常（尽管该场景使用了公有继承，这不会应用于自定义 STL 类）。记住，STL 提供了一套非常完整的容器类。你很少需要增强 STL 类——可能只有针对特定领域需求的类。然而，你现在知道定制 STL 类所涉及的风险。记住，在增强类时必须小心谨慎。我们现在可以看到，为任何我们创建的类进行适当的面向对象组件测试是必要的。

我们已经考虑了如何在我们的程序中自定义 STL 容器类和算法。我们也看到了许多 STL 容器类的实际应用示例。现在，在进入下一章之前，让我们简要回顾这些概念。

# 摘要

在本章中，我们将 C++ 知识扩展到面向对象语言特性之外，以熟悉 C++ 标准模板库。由于这个库在 C++ 中使用非常普遍，因此了解它包含的类的范围和广度至关重要。我们现在准备在我们的代码中使用这些有用且经过良好测试的类。

我们已经研究了相当多的 STL 示例；通过检查选定的 STL 类，我们应该能够独立理解 STL 的其余部分（或任何 C++ 库）。

我们已经看到了如何使用常见的和基本的 STL 类，如 `list`、`iterator`、`vector`、`deque`、`stack`、`queue`、`priority_queue` 和 `map`。我们还看到了如何结合容器类使用函数对象。我们被提醒，我们现在有工具可以定制任何类，甚至可以通过私有或保护继承，或者通过包含或关联来定制来自类库（如 STL）的类。

通过检查选定的 STL 类，我们还看到我们有能力理解 STL 的剩余深度和广度，以及解码许多对我们可用的其他类库。在我们导航每个成员函数的原型时，我们注意到关键语言概念，例如使用 `const`，或者一个方法返回一个指向表示可写内存的对象的引用。每个原型都揭示了新类使用的机制。在编程努力中走到这一步是非常令人兴奋的！

通过在 C++ 中浏览 STL，我们已经通过 C++ 增加了额外的、有用的功能到我们的 C++ 资料库。使用 STL（封装传统数据结构）将确保我们的代码可以轻松被其他无疑也在使用 STL 的程序员理解。依赖经过良好测试的 STL 来确保这些常见容器和实用工具，可以确保我们的代码更加无错误。

现在我们准备继续前进到*第十五章*，*测试类和组件*。我们希望用有用的 OO 组件测试技能来补充我们的 C++编程技能。测试技能将帮助我们了解我们是否以健壮的方式创建了、扩展了或增强了类。这些技能将使我们成为更好的程序员。让我们继续前进！

# 问题

1.  用从*第十三章*的练习中替换你的模板`Array`类，即*使用模板*。创建一个`Student`实例的`vector`。使用`vector`操作在向量中插入、检索、打印、比较和删除对象。或者，使用 STL 的`list`。利用这个机会使用 STL 文档来导航这些类可用的全部操作集。

a. 考虑是否需要重载运算符。考虑是否需要一个`iterator`来提供对集合的安全交错访问。

b. 创建第二个`Student`实例的`vector`。将一个赋值给另一个。打印两个向量。

1.  将本章中的`map`修改为根据`lastName`而不是`studentId`索引`Student`实例的哈希表（map）。

1.  将本章中的`queue`示例修改为使用`priority_queue`。确保使用优先入队机制`priority_queue::emplace()`将元素添加到`priority_queue`中。你还需要使用`top()`而不是`front()`。注意，`priority_queue`可以在`<queue>`头文件中找到。

1.  尝试使用`sort()`算法。确保包含`#include <algorithm>`。对一个整数数组进行排序。记住，许多容器都有内置的排序机制，但本地集合类型，如语言提供的数组，则没有（这就是为什么你应该使用基本整数数组）。
