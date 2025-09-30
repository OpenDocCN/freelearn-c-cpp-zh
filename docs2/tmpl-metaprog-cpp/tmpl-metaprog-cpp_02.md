# 第一章：*第一章*：模板简介

作为一名 C++ 开发者，你应该至少熟悉，如果不是精通**模板元编程**，通常简称为**模板**。模板元编程是一种编程技术，它使用模板作为编译器生成代码的蓝图，并帮助开发者避免编写重复的代码。尽管通用库大量使用模板，但 C++ 语言中模板的语法和内部工作原理可能会让人望而却步。甚至由 C++ 语言的创造者 Bjarne Stroustrup 和 C++ 标准化委员会主席 Herb Sutter 编辑的 *C++ Core Guidelines* 也称模板为“相当糟糕”。

这本书旨在阐明 C++ 语言这一领域，并帮助你成为模板元编程的大师。

在本章中，我们将讨论以下主题：

+   理解模板的需求

+   编写你的第一个模板

+   理解模板术语

+   模板简史

+   模板的优势与劣势

学习如何使用模板的第一步是理解它们实际上解决了什么问题。让我们从这一点开始。

# 理解模板的需求

每个语言特性都是为了帮助开发者在使用该语言时解决遇到的问题或任务。模板的目的是帮助我们避免编写只略有不同的重复代码。

为了举例说明，让我们以经典的 `max` 函数为例。这样的函数接受两个数值参数，并返回两个中的较大值。我们可以轻松地实现如下：

```cpp
int max(int const a, int const b)
```

```cpp
{
```

```cpp
   return a > b ? a : b;
```

```cpp
}
```

这效果相当不错，但正如你所见，它只适用于 `int` 类型的值（或可转换为 `int` 的类型）。如果我们需要相同的功能，但参数类型为 `double` 呢？那么，我们可以为 `double` 类型重载这个函数（创建一个具有相同名称但参数数量或类型不同的函数）：

```cpp
double max(double const a, double const b)
```

```cpp
{
```

```cpp
   return a > b ? a : b;
```

```cpp
}
```

然而，`int` 和 `double` 并不是唯一的数值类型。还有 `char`、`short`、`long`、`long` 以及它们的无符号版本，`unsigned char`、`unsigned short`、`unsigned long` 和 `unsigned long`。还有 `float` 和 `long double` 这两种类型。还有其他类型，例如 `int8_t`、`int16_t`、`int32_t` 和 `int64_t`。还可能有其他可以比较的类型，例如 `bigint`、`Matrix`、`point2d` 以及任何重载了 `operator>` 的用户定义类型。一个通用库如何为所有这些类型提供一个通用的函数，比如 `max` 呢？它可以重载所有内置类型的函数，也许还可以重载其他库类型，但不能为任何用户定义类型重载。

使用不同参数重载函数的替代方法是使用`void*`来传递不同类型的参数。请记住，这是一个不好的做法，以下示例仅作为一个在没有模板的世界中可能的选择。然而，为了讨论的目的，我们可以设计一个排序函数，该函数将对任何可能类型的元素数组执行快速排序算法，这些类型提供了严格的弱排序。快速排序算法的详细信息可以在网上找到，例如在维基百科上[`en.wikipedia.org/wiki/Quicksort`](https://en.wikipedia.org/wiki/Quicksort)。

快速排序算法需要比较和交换任意两个元素。然而，由于我们不知道它们的类型，实现不能直接这样做。解决方案是依赖于**回调函数**，这些函数作为参数传递，并在必要时被调用。一个可能的实现如下：

```cpp
using swap_fn = void(*)(void*, int const, int const);
```

```cpp
using compare_fn = bool(*)(void*, int const, int const);
```

```cpp
int partition(void* arr, int const low, int const high, 
```

```cpp
              compare_fn fcomp, swap_fn fswap)
```

```cpp
{
```

```cpp
   int i = low - 1;
```

```cpp
   for (int j = low; j <= high - 1; j++)
```

```cpp
   {
```

```cpp
      if (fcomp(arr, j, high))
```

```cpp
      {
```

```cpp
         i++;
```

```cpp
         fswap(arr, i, j);
```

```cpp
      }
```

```cpp
   }
```

```cpp
   fswap(arr, i + 1, high);
```

```cpp
   return i + 1;
```

```cpp
}
```

```cpp
void quicksort(void* arr, int const low, int const high, 
```

```cpp
               compare_fn fcomp, swap_fn fswap)
```

```cpp
{
```

```cpp
   if (low < high)
```

```cpp
   {
```

```cpp
      int const pi = partition(arr, low, high, fcomp, 
```

```cpp
         fswap);
```

```cpp
      quicksort(arr, low, pi - 1, fcomp, fswap);
```

```cpp
      quicksort(arr, pi + 1, high, fcomp, fswap);
```

```cpp
   }
```

```cpp
}
```

为了调用`quicksort`函数，我们需要为每种类型提供这些比较和交换函数的实现，我们将这些类型作为数组传递给函数。以下是`int`类型的实现：

```cpp
void swap_int(void* arr, int const i, int const j)
```

```cpp
{
```

```cpp
   int* iarr = (int*)arr;
```

```cpp
   int t = iarr[i];
```

```cpp
   iarr[i] = iarr[j];
```

```cpp
   iarr[j] = t;
```

```cpp
}
```

```cpp
bool less_int(void* arr, int const i, int const j)
```

```cpp
{
```

```cpp
   int* iarr = (int*)arr;
```

```cpp
   return iarr[i] <= iarr[j];
```

```cpp
}
```

在所有这些定义完成后，我们可以编写如下代码来对整数数组进行排序：

```cpp
int main()
```

```cpp
{
```

```cpp
   int arr[] = { 13, 1, 8, 3, 5, 2, 1 };
```

```cpp
   int n = sizeof(arr) / sizeof(arr[0]);
```

```cpp
   quicksort(arr, 0, n - 1, less_int, swap_int);
```

```cpp
}
```

这些示例主要关注函数，但相同的问题也适用于类。假设你想编写一个类，该类模拟一个具有可变大小且在内存中连续存储元素的数值集合。你可以提供以下实现（这里只展示了声明部分）来存储整数：

```cpp
struct int_vector
```

```cpp
{
```

```cpp
   int_vector();
```

```cpp
   size_t size() const;
```

```cpp
   size_t capacity() const;
```

```cpp
   bool empty() const;
```

```cpp
   void clear();
```

```cpp
   void resize(size_t const size);
```

```cpp
   void push_back(int value);
```

```cpp
   void pop_back();
```

```cpp
   int at(size_t const index) const;
```

```cpp
   int operator[](size_t const index) const;
```

```cpp
private:
```

```cpp
   int* data_;
```

```cpp
   size_t size_;
```

```cpp
   size_t capacity_;
```

```cpp
};
```

看起来一切都很不错，但当你需要存储`double`类型、`std::string`类型或任何用户定义类型的值时，你必须编写相同的代码，每次只更改元素的类型。这并不是人们想要做的事情，因为它是一项重复性的工作，而且当需要更改某些内容（例如添加新功能或修复错误）时，你需要在多个地方应用相同的更改。

最后，当你需要定义变量时，可能会遇到类似的问题，尽管这种情况不太常见。让我们考虑一个持有换行符的变量的情况。你可以这样声明它：

```cpp
constexpr char NewLine = '\n';
```

如果你需要相同的常量，但用于不同的编码，例如宽字符串字面量、UTF-8 等，你可以有多个变量，具有不同的名称，如下例所示：

```cpp
constexpr wchar_t NewLineW = L'\n';
```

```cpp
constexpr char8_t NewLineU8 = u8'\n';
```

```cpp
constexpr char16_t NewLineU16 = u'\n';
```

```cpp
constexpr char32_t NewLineU32 = U'\n';
```

模板是一种技术，允许开发者编写蓝图，使编译器能够为我们生成所有这些重复的代码。在下一节中，我们将看到如何将前面的代码片段转换为 C++模板。

# 编写你的第一个模板

现在是时候看看在 C++语言中如何编写模板了。在本节中，我们将从三个简单的示例开始，每个示例对应前面展示的代码片段。

之前讨论的`max`函数的模板版本看起来如下：

```cpp
template <typename T>
```

```cpp
T max(T const a, T const b)
```

```cpp
{
```

```cpp
   return a > b ? a : b;
```

```cpp
}
```

你会注意到，类型名（如`int`或`double`）已被替换为`T`（代表*类型*）。`T`被称为`template<typename T>`或`typename<class T>`。请记住，`T`是一个参数，因此它可以有任何一个名字。我们将在下一章中了解更多关于模板参数的内容。

到目前为止，你放在源代码中的这个模板只是一个蓝图。编译器将根据其使用情况生成代码。更确切地说，它将为模板使用的每个类型实例化一个函数重载。以下是一个示例：

```cpp
struct foo{};
```

```cpp
int main()
```

```cpp
{   
```

```cpp
   foo f1, f2;
```

```cpp
   max(1, 2);     // OK, compares ints
```

```cpp
   max(1.0, 2.0); // OK, compares doubles
```

```cpp
   max(f1, f2);   // Error, operator> not overloaded for 
```

```cpp
                  // foo
```

```cpp
}
```

在这个代码片段中，我们首先用两个整数调用`max`函数，这是可以的，因为`operator>`对于`int`类型是可用的。这将生成一个重载`int max(int const a, int const b)`。其次，我们用两个双精度浮点数调用`max`函数，这同样是可以的，因为`operator>`对于双精度浮点数也是可用的。因此，编译器将生成另一个重载`double max(double const a, double const b)`。然而，对`max`的第三次调用将生成编译器错误，因为`foo`类型没有重载`operator>`。

在这里不深入太多细节的情况下，应该提到调用`max`函数的完整语法如下：

```cpp
max<int>(1, 2);
```

```cpp
max<double>(1.0, 2.0);
```

```cpp
max<foo>(f1, f2);
```

编译器能够推导出模板参数的类型，因此没有必要写出它。然而，在某些情况下，这是不可能的；在这些情况下，你需要显式指定类型，使用这种语法。

上一节中涉及函数的第二个例子是处理`void*`参数的`quicksort()`实现。这个实现可以很容易地转换成模板版本，只需做很少的修改。以下是一个示例：

```cpp
template <typename T>
```

```cpp
void swap(T* a, T* b)
```

```cpp
{
```

```cpp
   T t = *a;
```

```cpp
   *a = *b;
```

```cpp
   *b = t;
```

```cpp
}
```

```cpp
template <typename T>
```

```cpp
int partition(T arr[], int const low, int const high)
```

```cpp
{
```

```cpp
   T pivot = arr[high];
```

```cpp
   int i = (low - 1);
```

```cpp
   for (int j = low; j <= high - 1; j++)
```

```cpp
   {
```

```cpp
      if (arr[j] < pivot)
```

```cpp
      {
```

```cpp
         i++;
```

```cpp
         swap(&arr[i], &arr[j]);
```

```cpp
      }
```

```cpp
   }
```

```cpp
   swap(&arr[i + 1], &arr[high]);
```

```cpp
   return i + 1;
```

```cpp
}
```

```cpp
template <typename T>
```

```cpp
void quicksort(T arr[], int const low, int const high)
```

```cpp
{
```

```cpp
   if (low < high)
```

```cpp
   {
```

```cpp
      int const pi = partition(arr, low, high);
```

```cpp
      quicksort(arr, low, pi - 1);
```

```cpp
      quicksort(arr, pi + 1, high);
```

```cpp
   }
```

```cpp
}
```

`quicksort`函数模板的使用与之前所见非常相似，只是不需要传递回调函数的指针：

```cpp
int main()
```

```cpp
{
```

```cpp
   int arr[] = { 13, 1, 8, 3, 5, 2, 1 };
```

```cpp
   int n = sizeof(arr) / sizeof(arr[0]);
```

```cpp
   quicksort(arr, 0, n - 1);
```

```cpp
}
```

在上一节中，我们讨论的第三个例子是`vector`类。它的模板版本如下所示：

```cpp
template <typename T>
```

```cpp
struct vector
```

```cpp
{
```

```cpp
   vector();
```

```cpp
   size_t size() const;
```

```cpp
   size_t capacity() const;
```

```cpp
   bool empty() const;
```

```cpp
   void clear();
```

```cpp
   void resize(size_t const size);
```

```cpp
   void push_back(T value);
```

```cpp
   void pop_back();
```

```cpp
   T at(size_t const index) const;
```

```cpp
   T operator[](size_t const index) const;
```

```cpp
private:
```

```cpp
   T* data_;
```

```cpp
   size_t size_;
```

```cpp
   size_t capacity_;
```

```cpp
};
```

与`max`函数的情况一样，变化很小。在类的上方一行有模板声明，元素类型`int`已被类型模板参数`T`所取代。这个实现可以这样使用：

```cpp
int main()
```

```cpp
{   
```

```cpp
   vector<int> v;
```

```cpp
   v.push_back(1);
```

```cpp
   v.push_back(2);
```

```cpp
}
```

这里需要注意的一点是，在声明变量`v`时，我们必须指定其元素类型，在我们的代码片段中是`int`，因为否则编译器无法推断它们的类型。在 C++17 中，这种情况是可能的，这个主题被称为**类模板参数推导**，将在*第四章* *高级模板概念*中讨论。

第四个也是最后一个例子是关于当只有类型不同时声明多个变量。我们可以用模板替换所有这些变量，如下面的代码片段所示：

```cpp
template<typename T>
```

```cpp
constexpr T NewLine = T('\n');
```

此模板可以按以下方式使用：

```cpp
int main()
```

```cpp
{
```

```cpp
   std::wstring test = L"demo";
```

```cpp
   test += NewLine<wchar_t>;
```

```cpp
   std::wcout << test;
```

```cpp
}
```

本节中的示例表明，无论模板代表函数、类还是变量，其声明和使用语法都是相同的。这引导我们进入下一节，我们将讨论模板的类型和模板术语。

# 理解模板术语

到目前为止，在本章中，我们使用了通用术语模板。然而，有四个不同的术语描述了我们所编写的模板类型：

+   之前见过的 `max` 模板。

+   `class`、`struct` 或 `union` 关键字）。一个例子是我们之前章节中编写的 `vector` 类。

+   之前章节中的 `NewLine` 模板。

+   **别名模板**是用于模板化类型别名的术语。我们将在下一章中看到别名模板的示例。

模板可以用一个或多个参数进行参数化（在我们迄今为止的示例中，有一个单个参数）。这些被称为**模板参数**，可以分为三类：

+   `template<typename T>`，其中参数代表在模板使用时指定的类型。

+   `template<size_t N>` 或 `template<auto n>`，其中每个参数都必须有一个结构化类型，这包括整数类型、浮点类型（对于 C++20）、指针类型、枚举类型、左值引用类型以及其他类型。

+   `template<typename K, typename V, template<typename> typename C>`，其中参数的类型是另一个模板。

可以通过提供替代实现来专门化模板。这些实现可以依赖于模板参数的特性。专门化的目的是为了实现优化或减少代码膨胀。有两种专门化的形式：

+   **部分专门化**：这是只为部分模板参数提供的替代实现。

+   **（显式）完全专门化**：这是当所有模板参数都提供时模板的专门化。

编译器从模板生成代码的过程称为 `vector<int>`，编译器在 `T` 出现的每个地方都替换了 `int` 类型。

模板实例化可以有两种形式：

+   `vector<int>` 和 `vector<double>`，它将为 `int` 和 `double` 类型实例化 `vector` 类模板，而不会更多。

+   **显式实例化**：这是一种明确告诉编译器要创建哪些模板实例化的方法，即使这些实例化在您的代码中没有被明确使用。这在创建库文件时很有用，因为未实例化的模板不会被放入对象文件中。它们还有助于减少编译时间和对象大小，我们将在稍后看到。

本节中提到的所有术语和主题将在本书的其他章节中详细说明。本节旨在作为模板术语的简要参考指南。但请记住，还有许多与模板相关的其他术语将在适当的时候介绍。

# 模板的历史简述

模板元编程是 C++的泛型编程实现。这种范式最早在 20 世纪 70 年代被探索，而第一个支持它的主要语言是 20 世纪 80 年代上半叶的 Ada 和 Eiffel。David Musser 和 Alexander Stepanov 在 1989 年的一篇名为《泛型编程》的论文中定义了泛型编程，如下所述：

泛型编程围绕着从具体、高效的算法抽象出通用算法的想法，这些通用算法可以与不同的数据表示结合，以产生各种有用的软件。

这定义了一种编程范式，其中算法是根据稍后指定的类型定义的，并根据其使用进行实例化。

模板不是 Bjarne Stroustrup 开发的最初**C with Classes**语言的组成部分。Stroustrup 描述 C++中模板的第一篇论文出现在 1986 年，即他的书《C++编程语言，第一版》出版后一年。模板在 1990 年成为 C++语言的一部分，在 ANSI 和 ISO C++标准化委员会成立之前。

在 20 世纪 90 年代初，Alexander Stepanov、David Musser 和 Meng Lee 在 C++中对各种泛型概念进行了实验。这导致了**标准模板库**（**STL**）的第一个实现。当 ANSI/ISO 委员会在 1994 年了解到这个库时，它迅速将其添加到草案规范中。STL 与 C++语言一起在 1998 年标准化，这被称为 C++98。

C++标准的较新版本，统称为**现代 C++**，引入了各种对模板元编程的改进。以下表格简要列出它们：

![Table 1.1

![img/B18367_Table_1.1.jpg]

表 1.1

所有这些特性，以及模板元编程的其他方面，将是本书的唯一主题，将在以下章节中详细介绍。现在，让我们看看使用模板的优势和劣势是什么。

# 模板的优势和劣势

在开始使用模板之前，了解使用模板的好处以及它们可能带来的劣势是很重要的。

让我们先指出其优势：

+   模板帮助我们避免编写重复的代码。

+   模板促进了通用库的创建，这些库提供算法和类型，例如标准 C++库（有时错误地称为 STL），它可以在许多应用中使用，无论它们的类型如何。

+   模板的使用可以导致代码更少且更好。例如，使用标准库中的算法可以帮助编写更少的代码，这些代码可能更容易理解和维护，并且由于这些算法的开发和测试所付出的努力，可能更健壮。

当谈到劣势时，以下几点值得提及：

+   虽然语法被认为是复杂且繁琐的，但只要稍加练习，这实际上不应该在模板的开发和使用中构成真正的障碍。

+   与模板代码相关的编译器错误通常很长且难以理解，这使得确定其原因是极其困难的。较新的 C++编译器在这些类型的错误简化方面取得了进展，尽管它们通常仍然是一个重要问题。C++20 标准中包含的概念被看作是尝试之一，旨在为编译错误提供更好的诊断。

+   由于它们完全在头文件中实现，因此它们会增加编译时间。每当对模板进行更改时，包含该头文件的所有翻译单元都必须重新编译。

+   模板库以一组一个或多个头文件的形式提供，这些头文件必须与使用它们的代码一起编译。

+   从模板在头文件中的实现中产生的另一个缺点是缺乏信息隐藏。整个模板代码都可在头文件中供任何人阅读。库开发者经常求助于使用诸如`detail`或`details`之类的命名空间来包含库内部应使用且不应直接被库使用者调用的代码。

+   由于未使用的代码不会被编译器实例化，因此它们可能更难验证。因此，在编写单元测试时，必须确保良好的代码覆盖率。这对于库尤其如此。

尽管缺点列表可能看起来更长，但使用模板并不是一件坏事或需要避免的事情。相反，模板是 C++语言的一个强大功能。模板并不总是被正确理解，有时会被误用或过度使用。然而，模板的明智使用无疑具有优势。本书将尝试提供对模板及其使用的更好理解。

# 摘要

本章介绍了 C++编程语言中模板的概念。

我们首先学习了那些解决方案是使用模板的问题。然后，我们通过函数模板、类模板和变量模板的简单示例来了解模板的外观。我们介绍了模板的基本术语，这些内容将在接下来的章节中进一步讨论。在章节的末尾，我们简要回顾了 C++编程语言中模板的历史。我们以讨论使用模板的优缺点结束本章。所有这些主题都将帮助我们更好地理解下一章的内容。

在下一章中，我们将探讨 C++中模板的基础知识。

# 问题

1.  我们为什么需要模板？它们提供了哪些优势？

1.  如何调用模板函数？对于模板类又是如何？

1.  存在多少种模板参数类型，它们是什么？

1.  什么是部分专业化？什么是完全专业化？

1.  使用模板的主要缺点是什么？

# 进一步阅读

+   *泛型编程，David Musser，Alexander Stepanov*，[`stepanovpapers.com/genprog.pdf`](http://stepanovpapers.com/genprog.pdf)

+   *C++历史：1979−1991，Bjarne Stroustrup*，[`www.stroustrup.com/hopl2.pdf`](https://www.stroustrup.com/hopl2.pdf)

+   *C++历史*，[`en.cppreference.com/w/cpp/language/history`](https://en.cppreference.com/w/cpp/language/history)

+   *C++模板：利与弊，Sergey Chepurin*，[`www.codeproject.com/Articles/275063/Templates-in-Cplusplus-Pros-and-Cons`](https://www.codeproject.com/Articles/275063/Templates-in-Cplusplus-Pros-and-Cons)
