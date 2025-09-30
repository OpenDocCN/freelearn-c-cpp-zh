# 经典多态与泛型编程

C++标准库有两个截然不同但同样重要的任务。其中之一是提供某些具体数据类型或函数的稳固实现，这些类型或函数在许多不同的程序中都有用，但并未内置于核心语言语法中。这就是为什么标准库包含了`std::string`、`std::regex`、`std::filesystem::exists`等等。标准库的另一个任务是提供广泛使用的**抽象算法**（如排序、搜索、反转、排序等）的稳固实现。在本章中，我们将明确说明当我们说某段代码是“抽象的”时，我们指的是什么，并描述标准库用来提供抽象的两种方法：**经典多态**和**泛型编程**。

在本章中，我们将探讨以下主题：

+   具体的（单态）函数，其行为不可参数化

+   通过基类、虚拟成员函数和继承实现经典多态

+   通过概念、要求和模型实现泛型编程

+   每种方法的实际优缺点

# 具体的单态函数

什么是区分抽象算法和具体函数的特征？这最好通过例子来说明。让我们编写一个函数，将数组中的每个元素乘以 2：

```cpp
    class array_of_ints {
      int data[10] = {};
      public:
        int size() const { return 10; }
        int& at(int i) { return data[i]; }
    };

    void double_each_element(array_of_ints& arr)
    {
      for (int i=0; i < arr.size(); ++i) {
        arr.at(i) *= 2;
      }
    }
```

我们的功能`double_each_element`**仅**与`array_of_int`类型的对象一起工作；传递不同类型的对象将不起作用（甚至无法编译）。我们将此类版本的`double_each_element`称为**具体**或**单态**函数。我们称它们为**具体**，因为它们对我们来说不够**抽象**。想象一下，如果 C++标准库提供了一个仅对一种特定数据类型工作的具体`sort`例程，那会多么痛苦！

# 经典的多态函数

我们可以通过经典**面向对象**（**OO**）编程的技术来提高我们算法的抽象级别，如 Java 和 C#等语言所示。面向对象的方法是决定我们希望哪些行为是可定制的，然后将它们声明为**抽象基类**的公共虚拟成员函数：

```cpp
    class container_of_ints {
      public:
      virtual int size() const = 0;
      virtual int& at(int) = 0;
    };

    class array_of_ints : public container_of_ints {
      int data[10] = {};
      public:
        int size() const override { return 10; }
        int& at(int i) override { return data[i]; }
    };

    class list_of_ints : public container_of_ints {
      struct node {
        int data;
        node *next;
      };
      node *head_ = nullptr;
      int size_ = 0;
      public:
       int size() const override { return size_; }
       int& at(int i) override {
        if (i >= size_) throw std::out_of_range("at");
        node *p = head_;
        for (int j=0; j < i; ++j) {
          p = p->next;
        }
        return p->data;
      }
      ~list_of_ints();
    };

    void double_each_element(container_of_ints& arr) 
    {
      for (int i=0; i < arr.size(); ++i) {
        arr.at(i) *= 2;
      } 
    }

    void test()
    {
      array_of_ints arr;
      double_each_element(arr);

      list_of_ints lst;
      double_each_element(lst);
    }
```

在`test`内部，对`double_each_element`的两次不同调用可以编译，因为在经典的 OO 术语中，一个`array_of_ints` **是** 一个`container_of_ints`（即它继承自`container_of_ints`并实现了相关的虚拟成员函数），一个`list_of_ints` **也是** 一个`container_of_ints`。然而，任何给定的`container_of_ints`对象的行为由其**动态类型**参数化；也就是说，由与该特定对象关联的函数指针表。

由于我们现在可以通过传递不同动态类型的对象来参数化`double_each_element`函数的行为，而不必直接编辑其源代码，我们可以说这个函数已经变得*多态*。

然而，这个多态函数只能处理那些是基类`container_of_ints`的子类的类型。例如，你不能将`std::vector<int>`传递给这个函数；如果你尝试这样做，你会得到编译错误。经典多态很有用，但它并没有把我们带到完全泛型的地步。

经典（面向对象）多态的一个优点是源代码仍然与编译器生成的机器代码保持一对一的对应关系。在机器代码级别，我们仍然只有一个`double_each_element`函数，具有一个签名和一个定义良好的入口点。例如，我们可以将`double_each_element`的地址作为函数指针。

# 模板泛型编程

在现代 C++中，编写完全泛型算法的典型方式是将算法实现为一个*模板*。我们仍然将以`.size()`和`.at()`公共成员函数为依据实现函数模板，但我们不再要求参数`arr`是任何特定类型。因为我们的新函数将是一个模板，我们将告诉编译器“我不关心`arr`的类型是什么。无论它是什么类型，只要生成一个新的函数（即模板实例化），使其参数类型为该类型。”

```cpp
    template<class ContainerModel>
    void double_each_element(ContainerModel& arr)
    {
      for (int i=0; i < arr.size(); ++i) {
        arr.at(i) *= 2;
      }
    }

    void test()
    {
      array_of_ints arr;
      double_each_element(arr);

      list_of_ints lst;
      double_each_element(lst);

      std::vector<int> vec = {1, 2, 3};
      double_each_element(vec);
    }
```

在大多数情况下，如果我们能够用文字精确地描述我们的模板类型参数`ContainerModel`必须支持的操作，这有助于我们设计更好的程序。这些操作的总和构成了 C++中所谓的*概念*；在这个例子中，我们可以说概念`Container`由“有一个名为`size`的成员函数，它返回容器的大小作为一个`int`（或与`int`相当的东西）；并且有一个名为`at`的成员函数，它接受一个`int`索引（或可以隐式转换为`int`的东西）并产生对容器中*索引*元素的非 const 引用。”每当某个类`array_of_ints`正确地提供概念`Container`所需的操作，使得`array_of_ints`可以与`double_each_element`一起使用时，我们就说具体类`array_of_ints`*是*`Container`概念的模型。这就是为什么我在前面的例子中将模板类型参数命名为`ContainerModel`。

使用`Container`作为模板类型参数本身的名称更为传统，从现在开始我将这样做；我只是不想一开始就混淆`Container`概念和特定函数模板的特定模板类型参数之间的区别，这个函数模板恰好希望将其参数设置为模型`Container`概念的具体类。

当我们使用模板实现一个抽象算法，使得算法的行为可以在编译时通过任何建模适当概念的类型进行参数化时，我们说我们在进行泛型编程。

注意，我们关于 `Container` 概念的描述并没有提到我们期望包含的元素类型是 `int`；并且不是巧合的是，我们发现现在我们甚至可以使用我们的通用 `double_each_element` 函数，即使容器不包含 `int`！

```cpp
    std::vector<double> vecd = {1.0, 2.0, 3.0};
    double_each_element(vecd);
```

这种额外的泛型级别是使用 C++ 模板进行泛型编程而不是经典多态的一个大优点。经典多态在稳定的 *接口签名*（例如，`.at(i)` 总是返回 `int&`）后面隐藏了不同类的不同行为，但一旦你开始与变化的签名打交道，经典多态就不再是这项工作的好工具。

泛型编程的另一个优点是，它通过增加内联的机会提供了闪电般的速度。经典多态的例子必须反复查询 `container_of_int` 对象的虚表以找到其特定虚拟 `at` 方法的地址，并且通常无法在编译时看到虚拟调度。模板函数 `double_each_element<array_of_int>` 可以直接调用 `array_of_int::at` 或甚至完全内联调用。

由于模板泛型编程可以如此轻松地处理复杂的需求，并且在处理类型方面非常灵活——甚至对于像 `int` 这样的原始类型，在经典多态中失败的情况下——标准库使用模板来处理所有算法及其操作的容器。因此，标准库中算法和容器部分通常被称为 **标准模板库** 或 **STL**。

对的——技术上，STL 只是 C++ 标准库的一小部分！然而，在这本书中，就像在现实生活中一样，我们有时可能会不小心使用 STL 这个词，而实际上我们指的是标准库，反之亦然。

在我们深入研究 STL 提供的标准泛型算法之前，让我们先看看几个手写的通用算法。这里有一个函数模板 `count`，它返回容器中元素的总数：

```cpp
    template<class Container>
    int count(const Container& container)
    {
      int sum = 0;
      for (auto&& elt : container) {
        sum += 1;
      }
      return sum;
    }
```

这里是 `count_if`，它返回满足用户提供的 *谓词* 函数的元素数量：

```cpp
    template<class Container, class Predicate>
    int count_if(const Container& container, Predicate pred) 
    { 
      int sum = 0;
      for (auto&& elt : container) {
        if (pred(elt)) {
            sum += 1;
        }
      }
      return sum;
    }
```

这些函数的使用方式如下：

```cpp
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    assert(count(v) == 8);

    int number_above =
      count_if(v, [](int e) { return e > 5; });
    int number_below =
      count_if(v, [](int e) { return e < 5; });

    assert(number_above == 2);
    assert(number_below == 5);
```

在那个小小的表达式`pred(elt)`中蕴含了如此多的力量！我鼓励你尝试用经典多态重新实现`count_if`函数，只是为了了解整个系统在哪里崩溃。在现代 C++的语法糖下隐藏着许多不同的签名。例如，在我们的`count_if`函数中的范围 for 循环语法被编译器转换（或降低）为基于`container.begin()`和`container.end()`的 for 循环，每个都需要返回一个迭代器，其类型取决于`container`本身的类型。另一个例子，在泛型编程版本中，我们从未指定——我们从未*需要*指定——`pred`是否通过值或引用接收其参数`elt`。尝试用`virtual bool operator()`做*那*件事！

谈到迭代器：你可能已经注意到，本章中的所有示例函数（无论它们是单态、多态还是泛型）都是用容器来表达的。当我们编写`count`时，我们计算整个容器中的元素数量。当我们编写`count_if`时，我们计算整个容器中匹配的元素数量。这证明是一种非常自然的方式来编写，特别是在现代 C++中；如此之多，以至于我们可以期待在 C++20 或 C++23 中看到基于容器的算法（或其近亲，基于范围的算法）的出现。然而，STL 可以追溯到 20 世纪 90 年代和现代 C++之前。因此，STL 的作者假设主要处理容器将会非常昂贵（由于所有那些昂贵的拷贝构造——记住移动语义和移动构造直到 C++11 才出现）；因此，他们设计了 STL 主要处理一个更轻量级的概念：*迭代器*。这将是下一章的主题。

# 摘要

经典多态和泛型编程都处理了参数化算法行为的本质问题：例如，编写一个与任何任意匹配操作一起工作的搜索函数。

经典的多态通过指定一个具有一组封闭的*抽象基类*和*虚成员函数*的类，以及编写接受从该基类继承的具体类实例的指针或引用的*多态函数*来解决该问题。

泛型编程通过指定一个具有一组封闭的*要求*的概念，并用具体类实例化*函数模板*来解决这个问题，这些具体类模拟了该概念。

经典多态在处理高级参数化（例如，操作任何签名的函数对象）和类型之间的关系（例如，操作任意容器的元素）方面存在困难。因此，标准模板库大量使用了基于模板的泛型编程，而几乎没有使用经典多态。

当你使用泛型编程时，如果你能记住你类型的概念性要求，或者甚至将它们明确地写下来，这将有所帮助；但截至 C++17，编译器无法直接帮助你检查这些要求。
