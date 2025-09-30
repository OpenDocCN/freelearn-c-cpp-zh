# 深入现代 C++

自 1979 年发明以来，C++ 编程语言发生了巨大的变化。在这个时代的一些人可能对使用 C++ 语言进行编码感到有些害怕，因为它不够用户友好。我们必须处理的内存管理有时使人们不愿意使用这种语言。幸运的是，自从 **C++11**（也称为 **现代 C++**），以及 **C++14** 和 **C++17** 发布以来，已经引入了许多特性，以简化 C++ 语言的代码。此外，最好的部分是，C++ 编程语言是任何项目的绝佳语言，从底层编程到 Web 编程，以及函数式编程。

本章是本书开始旅程的最佳起点，因为它针对 C++ 程序员，旨在更新他们的知识，并将讨论以下主题：

+   理解现代 C++ 中的几个新特性

+   在现代 C++ 中实现 C++ 标准库

+   Lambda 表达式的使用及其包含的所有特性

+   使用智能指针避免手动内存管理

+   使用元组处理多个返回值

# 接近现代 C++ 的几个新特性

那么，与旧版本相比，现代 C++ 有什么新变化？与现代版本相比，现代 C++ 有许多变化，如果我们讨论所有这些变化，书籍的页数将显著增加。然而，我们将讨论我们应该知道的现代 C++ 的新特性，以便我们在编码活动中更加高效。我们将讨论几个新关键字，如 `auto`、`decltype` 和 `nullptr`。我们还将讨论 `begin()` 和 `end()` 函数的增强，现在它已成为非成员类函数。我们还将讨论对 `for-each` 技术的增强支持，使用 `range-based for loop` 技术遍历集合。

本章接下来的几个小节也将讨论现代 C++ 的新特性，即 Lambda 表达式、智能指针和元组，这些特性是在 C++11 版本中刚刚添加的。

# 使用 auto 关键字自动定义数据类型

在现代 C++ 之前，C++ 语言有一个名为 `auto` 的关键字，用于显式指定变量应该具有 **自动持续时间**。遵循变量的自动持续时间将在定义点（以及如果相关则初始化）创建变量，并在它们定义的块退出时销毁变量。例如，局部变量将在函数开始定义时创建，并在程序退出包含该局部变量的函数时销毁。

自 C++11 以来，`auto`关键字被用来告诉编译器从其初始化器推导出正在声明的变量的实际类型。自 C++14 起，该关键字也可以应用于函数，以指定函数的返回类型，即尾随返回类型。现在，在现代 C++中，使用`auto`关键字指定自动持续时间已被废除，因为所有变量默认都设置为自动持续时间。

以下是一个`auto.cpp`代码示例，展示了在变量中使用`auto`关键字。我们将使用`auto`关键字定义四个变量，然后使用`typeid()`函数找出每个变量的数据类型。让我们看一下：

```cpp
    /* auto.cpp */

    #include <iostream>
    #include <typeinfo>

    int main()
    {
      std::cout << "[auto.cpp]" << std::endl;

      // Creating several auto-type variables
      auto a = 1;
      auto b = 1.0;
      auto c = a + b;
      auto d = {b, c};

      // Displaying the preceding variables' type
      std::cout << "type of a: " << typeid(a).name() << std::endl;
      std::cout << "type of b: " << typeid(b).name() << std::endl;
      std::cout << "type of c: " << typeid(c).name() << std::endl;
      std::cout << "type of d: " << typeid(d).name() << std::endl;
      return 0;
    }

```

如前述代码所示，我们有一个`a`变量将存储`integer`值，还有一个`b`变量将存储`double`值。我们计算`a`和`b`的和，并将结果存储在变量`c`中。在这里，我们期望`c`将存储`double`对象，因为我们添加了`integer`和`double`对象。最后是`d`变量，它将存储`initializer_list<double>`数据类型。当我们运行前述代码时，我们将在控制台上看到以下输出：

![](img/5d5e1164-6f13-4afb-9011-c6444f0c3589.png)

如前一个快照所示，我们只给出了数据类型的首个字符，例如对于`integer`类型是`i`，对于`double`类型是`d`，而对于`initializer_list<double>`类型是`St16initializer_listIdE`，即最后一个小写的`d`字符代表`double`类型。

我们可能需要在编译器选项中启用**运行时类型信息**（**RTTI**）功能来检索数据类型对象。然而，GCC 默认已经启用了该功能。此外，`typeid()`函数的使用输出取决于编译器。我们可能会得到原始类型名称或只是一个符号，就像我们在前述示例中所做的那样。

此外，对于变量，如我们之前讨论的，`auto`关键字也可以应用于函数以自动推导函数的返回类型。假设我们有一个名为`add()`的简单函数，用于计算两个参数的和：

```cpp
    int add(int i, int j)
    {
      return i + j;
    }

```

我们可以将前述方法重构为使用`auto`关键字，如以下代码行所示：

```cpp
    auto add(int i, int j)
    {
      return i + j;
    }

```

与自动类型变量类似，编译器可以根据函数返回的值来决定正确的返回类型。并且，如前述代码所示，该函数确实会返回整数值，因为我们只是添加了两个整数值。

现代 C++中使用`auto`关键字实现的另一个特性是尾随返回类型语法。通过使用此功能，我们可以指定返回类型、函数原型或函数签名中的其余部分。从前述代码中，我们可以重构它以使用该功能，如下所示：

```cpp
    auto add(int i, int j) -> int
    {
      return i + j;
    }

```

你可能会问我为什么在箭头符号（`->`）之后再次指定数据类型，即使我们已经使用了`auto`关键字。我们将在下一节介绍`decltype`关键字时找到答案。此外，通过使用这个特性，我们现在可以通过修改`main()`方法的语法来稍微重构前面的`auto.cpp`代码，而不是以下`main()`函数签名的语法：

```cpp
    int main()
    {
      // The body of the function
    }

```

我们可以将签名语法更改为以下代码行：

```cpp
    auto main -> int
    {
      // The body of the function
    }

```

现在，我们将看到本书中的所有代码都使用这个尾随返回类型特性来应用现代 C++语法。

# 使用`decltype`关键字查询表达式的类型

在前一小节中，我们讨论了`auto`关键字可以根据存储的值的类型自动推断变量的类型。该关键字还可以根据其返回值的类型推断函数的返回类型。现在，让我们将`auto`关键字和`decltype`关键字结合起来，以获得现代 C++的强大功能。

在我们将两个关键字结合起来之前，我们将找出`decltype`关键字的作用——它是用来询问对象或表达式的类型。让我们看一下以下几行简单的变量声明：

```cpp
    const int func1();
    const int& func2();
    int i;

    struct X { double d; };
    const X* x = new X();

```

现在，基于前面的代码，我们可以使用`decltype`关键字声明其他变量，如下所示：

```cpp
    // Declaring const int variable
    // using func1() type
    decltype(func1()) f1;

    // Declaring const int& variable
    // using func2() type
    decltype(func2()) f2;

    // Declaring int variable
    // using i type
    decltype(i) i1;

    // Declaring double variable
    // using struct X type
    decltype(x->d) d1; // type is double
    decltype((x->d)) d2; // type is const double&

```

如前述代码所示，我们可以根据另一个对象的类型指定对象的类型。现在，假设我们需要重构前面的`add()`方法成为一个模板。如果没有`auto`和`decltype`关键字，我们将有以下模板实现：

```cpp
    template<typename I, typename J, typename K>
    K add(I i, J j)
    {
      return i + j;
    }

```

幸运的是，由于`auto`关键字可以指定函数的返回类型，即尾随返回类型，而`decltype`关键字可以根据表达式推断类型，因此我们可以重构前面的模板如下：

```cpp
    template<typename I, typename J>
    auto add(I i, J j) -> decltype(i + j)
    {
      return i + j;
    }

```

为了证明这一点，让我们编译并运行以下`decltype.cpp`代码。我们将使用以下模板来计算两种不同值类型（`integer`和`double`）的和：

```cpp
    /* decltype.cpp */
    #include <iostream>

    // Creating template
    template<typename I, typename J>
    auto add(I i, J j) -> decltype(i + j)
    {
      return i + j;
    }

    auto main() -> int
    {
      std::cout << "[decltype.cpp]" << std::endl;

      // Consuming the template
      auto d = add<int, double>(2, 2.5);

      // Displaying the preceding variables' type
      std::cout << "result of 2 + 2.5: " << d << std::endl;

      return 0;
    }

```

编译过程应该平稳运行，没有错误。如果我们运行前面的代码，屏幕上将会看到以下输出：

![](img/9b573b77-6cc6-41a6-94cb-8fc04dd314c8.png)

如我们所见，我们已经成功地将`auto`和`decltype`关键字结合起来创建了一个比现代 C++宣布之前更简单的模板。

# 指向空指针

现代 C++中另一个新特性是名为`nullptr`的关键字，它取代了`NULL`宏来表示空指针。现在，使用`NULL`宏表示零数值或空指针时不再存在歧义。假设我们在声明中有以下两个方法的签名：

```cpp
    void funct(const char *);
    void funct(int)

```

前一个函数将传递一个指针作为参数，后一个函数将传递整数作为其参数。然后，我们调用`funct()`方法并传递`NULL`宏作为参数，如下所示：

```cpp
    funct(NULL);

```

我们打算调用的是前面的函数。然而，由于我们传递了`NULL`参数，这基本上定义为`0`，所以后面的函数将被调用。在现代 C++中，我们可以使用`nullptr`关键字来确保我们将空指针传递给参数。`funct()`方法的调用应该如下所示：

```cpp
    funct(nullptr);

```

现在编译器将调用前面的函数，因为它将一个空指针传递给参数，这正是我们所期望的。从此将不再存在歧义，并且可以避免未来不必要的麻烦。

# 使用非成员`begin()`和`end()`函数返回迭代器

在现代 C++之前，为了遍历一个序列，我们调用每个容器的`begin()`和`end()`成员方法。对于数组，我们可以通过遍历索引来遍历其元素。由于 C++11，语言有一个非成员函数--`begin()`和`end()`--来检索序列的迭代器。假设我们有一个以下元素的数组：

```cpp
    int arr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

```

当语言没有`begin()`和`end()`函数时，我们需要使用以下代码行中的索引来遍历数组的元素：

```cpp
    for (unsigned int i = 0; i < sizeof(arr)/sizeof(arr[0]); ++i)
    // Do something to the array

```

幸运的是，使用`begin()`和`end()`函数，我们可以重构前面的`for`循环，使其如下所示：

```cpp
    for (auto i = std::begin(arr); i != std::end(arr); ++i)
    // Do something to the array

```

如我们所见，使用`begin()`和`end()`函数创建了一个紧凑的代码，因为我们不需要担心数组的长度，因为`begin()`和`end()`的迭代器指针会为我们做这件事。为了比较，让我们看看以下`begin_end.cpp`代码：

```cpp
    /* begin_end.cpp */
    #include <iostream>

    auto main() -> int
    {
      std::cout << "[begin_end.cpp]" << std::endl;

      // Declaring an array
      int arr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

      // Displaying the array elements
      // using conventional for-loop
      std::cout << "Displaying array element using conventional for-
       loop";
      std::cout << std::endl;
      for (unsigned int i = 0; i < sizeof(arr)/sizeof(arr[0]); ++i)
      std::cout << arr[i] << " ";
      std::cout << std::endl;

      // Displaying the array elements
      // using non-member begin() and end()
      std::cout << "Displaying array element using non-member begin()
       and end()";
      std::cout << std::endl;
      for (auto i = std::begin(arr); i != std::end(arr); ++i)
       std::cout << *i << " ";
      std::cout << std::endl;

      return 0;
    }

```

为了证明前面的代码，我们可以编译代码，当我们运行它时，以下输出应该显示在控制台屏幕上：

![](img/3219ad51-9855-43ce-b596-536273d9c047.png)

如截图所示，当我们使用传统的`for-loop`或`begin()`和`end()`函数时，我们得到了完全相同的输出。

# 使用基于范围的 for 循环遍历集合

在现代 C++中，有一个新的特性被添加来支持遍历集合的`for-each`技术。如果你想在集合或数组中执行某些操作，而不关心元素的数量或索引，这个特性非常有用。该特性的语法也很简单。假设我们有一个名为`arr`的数组，我们想使用`基于范围的 for 循环`技术遍历每个元素；我们可以使用以下语法：

```cpp
    for (auto a : arr)
    // Do something with a

```

因此，我们可以重构我们之前的`begin_end.cpp`代码，使用`基于范围的 for 循环`，如下所示：

```cpp
    /* range_based_for_loop.cpp */
    #include <iostream>

    auto main() -> int
    {
      std::cout << "[range_based_for_loop.cpp]" << std::endl;

      // Declaring an array
      int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

      // Displaying the array elements
      // using non-member begin() and end()
      std::cout << "Displaying array element using range-based for
        loop";
      std::cout << std::endl;
      for (auto a : arr) std::cout << a << " ";
      std::cout << std::endl;

      return 0;
    }

```

我们在前面代码中看到的语法现在更简单了。如果我们编译前面的代码，我们应该找不到错误，如果我们运行代码，我们应该在控制台屏幕上看到以下输出：

![](img/3f3f2d37-a65c-491d-86e6-c0cc8523e391.png)

我们现在有一个新的技术来遍历集合，而不关心集合的索引。我们将在本书中继续使用它。

# 利用 C++语言与 C++标准库的结合

C++ 标准库是一组功能强大的类和函数，它们具有创建应用程序所需的所有功能。它们由 C++ ISO 标准委员会控制，并受到在 C++11 之前引入的 **标准模板库** （**STL**）的影响，这些是 C++11 之前的通用库。标准库中的所有功能都在 `std` 命名空间中声明，并且不再有以 `.h` 结尾的头文件（除了包含到 C++ 标准库中的 ISO C90 C 标准库的 18 个头文件）。

有几个头文件包含 C++ 标准库的声明。然而，在这些简短的章节中几乎不可能讨论所有头文件。因此，我们将讨论我们在日常编码活动中最常使用的某些功能。

# 将任何对象放入容器中

**容器** 是一个用于存储其他对象并管理其包含的对象使用的内存的对象。数组是 C++11 中添加的一个新特性，用于存储特定数据类型的集合。它是一个顺序容器，因为它存储相同数据类型的对象并将它们线性排列。让我们看一下以下代码片段：

```cpp
    /* array.cpp */
    #include <array>
    #include <iostream>

    auto main() -> int
    {
      std::cout << "[array.cpp]" << std::endl;

      // Initializing an array containing five integer elements
      std::array<int, 10> arr = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

      // Displaying the original elements of the array
      std::cout << "Original Data : ";
      for(auto a : arr) std::cout << a << " ";
      std::cout << std::endl;

      // Modifying the content of
      // the 1st and 3rd element of the array
      arr[1] = 9;
      arr[3] = 7;

      // Displaying the altered array elements
      std::cout << "Manipulated Data: ";
      for(auto a : arr) std::cout << a << " ";
      std::cout << std::endl;

      return 0;
     }

```

如前述代码所示，我们实例化了一个名为 `arr` 的新数组，将其长度设置为 `10`，并且只允许 `int` 元素。正如我们所猜测的，代码的输出是一行从 `0` 到 `9` 的数字，这显示在原始数据中，另一行将显示修改后的数据，如以下截图所示：

![](img/9bf41be3-c581-4cf9-87d9-be35bc06f403.png)

如果我们使用 `std::array` 声明一个数组，那么不会有性能问题；我们在 `array.cpp` 代码中使用它，并将其与我们在 `begin_end.cpp` 代码中使用的常规数组进行比较。然而，在现代 C++ 中，我们得到了一个新的数组声明，它具有友好的值语义，因此它可以按值传递给或从函数返回。此外，这个新数组声明的接口使得查找大小和使用它与 **标准模板库** （**STL**）-风格的基于迭代器的算法更加方便。

使用数组作为容器是好的，因为我们可以在其中存储数据并对其进行操作。如果我们想排序或查找特定元素，我们也可以这样做。然而，由于数组是一个编译时不可调整大小的对象，我们必须在最初就决定要使用的数组大小，因为我们不能在之后改变它的大小。换句话说，我们不能从现有数组中插入或删除元素。作为解决这个问题和容器使用最佳实践的一部分，我们现在可以使用 `vector` 来存储我们的集合。让我们看一下以下代码：

```cpp
    /* vector.cpp */
    #include <vector>
    #include <iostream>

    auto main() -> int
    {
      std::cout << "[vector.cpp]" << std::endl;

      // Initializing a vector containing three integer elements
      std::vector<int> vect = { 0, 1, 2 };

      // Displaying the original elements of the vector
      std::cout << "Original Data : ";
      for (auto v : vect) std::cout << v << " ";
      std::cout << std::endl;

      // Adding two new data
      vect.push_back(3);
      vect.push_back(4);

      // Displaying the elements of the new vector
      // and reverse the order
      std::cout << "New Data Added : ";
      for (auto v : vect) std::cout << v << " ";
      std::cout << std::endl;

      // Modifying the content of
      // the 2nd and 4th element of the vector
      vect.at(2) = 5;
      vect.at(4) = 6;

      // Displaying the altered array elements
      std::cout << "Manipulate Data: ";
      for (auto v : vect) std::cout << v << " ";
      std::cout << std::endl;

      return 0;
    }

```

现在，在我们的前一段代码中，我们有一个 `vector` 实例而不是 `array` 实例。正如我们所见，我们使用 `push_back()` 方法为 `vector` 实例提供了一个额外的值。我们可以在任何时候添加值。由于 `vector` 有一个 `at()` 方法，它返回特定索引元素的引用，因此对每个元素的操纵也更简单。以下截图是我们运行代码时将看到的输出：

![图片](img/d6c12da0-75ee-450a-b324-a86e4bf59dad.png)

当我们想要通过索引访问 `vector` 实例中的特定元素时，最好始终使用 `at()` 方法而不是 `[]` 操作符。这是因为，当我们意外访问超出范围的索引时，`at()` 方法将抛出 `out_of_range` 异常。否则，`[]` 操作符将给出未定义的行为。

# 使用算法

我们可以对 `array` 或 `vector` 中的集合元素进行排序，以及查找特定元素的内容。为此，我们必须使用 C++ 标准库提供的算法功能。让我们看一下以下代码，以演示算法功能中的排序元素能力：

```cpp
    /* sort.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    bool comparer(int a, int b)
    {
      return (a > b);
    }

    auto main() -> int
    {
      std::cout << "[sort.cpp]" << std::endl;

      // Initializing a vector containing several integer elements
      std::vector<int> vect = { 20, 43, 11, 78, 5, 96 };

      // Displaying the original elements of the vector
      std::cout << "Original Data : ";
      for (auto v : vect)
      std::cout << v << " ";
      std::cout << std::endl;

      // Sorting the vector element ascending
      std::sort(std::begin(vect), std::end(vect));

      // Displaying the ascending sorted elements
      // of the vector
      std::cout << "Ascending Sorted : ";
      for (auto v : vect)
      std::cout << v << " ";
      std::cout << std::endl;

      // Sorting the vector element descending
      // using comparer
      std::sort(std::begin(vect), std::end(vect), comparer);

      // Displaying the descending sorted elements
      // of the vector
      std::cout << "Descending Sorted: ";
      for (auto v : vect)
      std::cout << v << " ";
      std::cout << std::endl;

      return 0;
   }

```

如前所述的代码所示，我们调用了 `sort()` 方法两次。首先，我们只提供了我们想要排序的元素的范围。然后我们添加了比较函数 `comparer()`，以便提供给 `sort()` 方法以获得更多灵活性。从前面的代码中，我们将在控制台上看到的输出如下：

![图片](img/a39ab3f7-55b8-4dda-8f2f-4d253d92f0d9.png)

从前面的截图，我们可以看到开始时 `vector` 中有六个元素。然后我们使用简单的 `sort()` 方法对向量元素进行排序。然后我们再次调用 `sort()` 方法，但这次我们不是使用简单的 `sort()` 方法，而是向 `sort()` 方法提供了 `comparer()`。结果，由于 `comparer()` 函数从两个输入中寻找较大的值，向量元素将按降序排序。

现在，让我们转向算法功能具有的另一项能力，即查找特定元素。假设我们在代码中有一个 `Vehicle` 类。它有两个名为 `m_vehicleType` 和 `m_totalOfWheel` 的私有字段，我们可以分别通过名为 `GetType()` 和 `GetNumOfWheel()` 的获取器方法检索其值。它还有两个构造函数，一个是默认构造函数，另一个是用户定义的。类的声明应该如下所示：

```cpp
    /* vehicle.h */
    #ifndef __VEHICLE_H__
    #define __VEHICLE_H__

    #include <string>

    class Vehicle
    {
      private:
        std::string vehicleType;
        int totalOfWheel;

      public:
        Vehicle(
          const std::string &type,
          int _wheel);
        Vehicle();
        ~Vehicle();
        std::string GetType() const {return vehicleType;}
        int GetNumOfWheel() const {return totalOfWheel;}
    };

    #endif // End of __VEHICLE_H__

```

`Vehicle` 类的实现如下：

```cpp
    /* vehicle.cpp */
    #include "vehicle.h"

    using namespace std;

    // Constructor with default value for
    // m_vehicleType and m_totalOfWheel
    Vehicle::Vehicle() : m_totalOfWheel(0)
    {
    }

    // Constructor with user-defined value for
    // m_vehicleType and m_totalOfWheel
    Vehicle::Vehicle( const string &type, int wheel) :
     m_vehicleType(type),
     m_totalOfWheel(wheel)
    {
    }

    // Destructor
    Vehicle::~Vehicle()
    {
    }

```

我们将在 `vector` 容器中存储一个 `Vehicle` 集合，然后我们将根据其属性搜索一些元素。代码如下所示：

```cpp
    /* find.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>
    #include "../vehicle/vehicle.h"

    using namespace std;

    bool TwoWheeled(const Vehicle &vehicle)
    {
      return _vehicle.GetNumOfWheel() == 2 ? 
        true : false;
     }

    auto main() -> int
    {
      cout << "[find.cpp]" << endl;

      // Initializing several Vehicle instances
      Vehicle car("car", 4);
      Vehicle motorcycle("motorcycle", 2);
      Vehicle bicycle("bicycle", 2);
      Vehicle bus("bus", 6);

      // Assigning the preceding Vehicle instances to a vector
      vector<Vehicle> vehicles = { car, motorcycle, bicycle, bus };

      // Displaying the elements of the vector
      cout << "All vehicles:" << endl;;
      for (auto v : vehicles)
        std::cout << v.GetType() << endl;
      cout << endl;

      // Displaying the elements of the vector
      // which are the two-wheeled vehicles
      cout << "Two-wheeled vehicle(s):" << endl;;
      auto tw = find_if(
                      begin(vehicles),
                      end(vehicles),
                      TwoWheeled);
      while (tw != end(vehicles))
      {
        cout << tw->GetType() << endl ;
        tw = find_if(++tw, end(vehicles), TwoWheeled);
      }
      cout << endl;

      // Displaying the elements of the vector
      // which are not the two-wheeled vehicles
      cout << "Not the two-wheeled vehicle(s):" << endl;;
      auto ntw = find_if_not(begin(vehicles),
                           end(vehicles),
                           TwoWheeled);
      while (ntw != end(vehicles))
      {
        cout << ntw->GetType() << endl ;
        ntw = find_if_not(++ntw, end(vehicles), TwoWheeled);
      }

      return 0;
     }

```

正如我们所见，我们实例化了四个`Vehicle`对象，然后将它们存储在`vector`中。在那里，我们试图找到有两个轮子的车辆。我们使用`find_if()`函数来完成这个目的。我们还拥有`TwoWheeled()`方法来提供比较值。由于我们正在寻找两轮车辆，我们将通过调用`GetNumOfWheel()`方法检查`Vehicle`类中的`totalOfWheel`变量。相比之下，如果我们想找到不符合比较值的元素，我们可以使用在 C++11 中添加的`find_if_not()`函数。我们得到的输出应该看起来像这样：

![](img/7c5af549-86c7-4414-896e-e9fe96a9cb20.png)

正如我们在`vehicle.cpp`代码和`find.cpp`代码中看到的，我们现在在`*.cpp`文件中添加了`using namespace std;`行。我们这样做是为了使我们的编码活动更加高效，因为我们不必输入很多单词。相比之下，在`vehicle.h`中，我们仍然使用`std::`后跟方法或属性名称，而不是在开头使用 std 命名空间。在头文件中不声明`using namespace`是最佳实践，因为头文件是我们将交付的文件，如果我们为实例创建一些库。我们库的用户可能有一个与我们的库中函数同名的方法。这肯定会在这两个函数之间造成冲突。

我们将最常使用的一个算法特性是`for_each`循环。与使用`for`循环相比，在许多情况下使用`for_each`循环会使我们的代码更加简洁。它也比`for`循环更简单、更不容易出错，因为我们可以为`for_each`循环定义一个特定的函数。现在让我们重构之前的代码，使用`for_each`循环。代码如下所示：

```cpp
    /* for_each.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>
    #include "vehicle.h"

    using namespace std;

    void PrintOut(const Vehicle &vehicle)
    {
      cout << vehicle.GetType() << endl;
    }

    auto main() -> int
   {
      cout << "[for_each.cpp]" << endl;

      // Initializing several Vehicle instances
      Vehicle car("car", 4);
      Vehicle motorcycle("motorcycle", 2);
      Vehicle bicycle("bicycle", 2);
      Vehicle bus("bus", 6);

      // Assigning the preceding Vehicle instances to a vector
      vector<Vehicle> vehicles = { car, motorcycle, bicycle, bus };

      // Displaying the elements of the vector
      cout << "All vehicles:" << endl;
      for_each(begin(vehicles), end(vehicles), PrintOut);

      return 0;
    }

```

现在，使用`for_each`循环，我们的代码更加清晰。我们只需要提供第一个和最后一个迭代器，然后传递一个函数——在这种情况下是`PrintOut()`函数——它将在范围内的每个元素上被调用。

# 使用 Lambda 表达式简化函数表示法

Lambda 表达式是一种匿名表示法，表示执行操作或计算的东西。在函数式编程中，Lambda 表达式对于产生一等和纯函数非常有用，我们将在本书的单独章节中讨论。现在，让我们通过调查 Lambda 表达式的三个基本部分来熟悉 C++11 中引入的这一新特性：

+   capturing list: []

+   参数列表: ()

+   body: {}

这三个基本部分的顺序如下：

```cpp
    [](){} 

```

捕获列表部分也用作标记来识别 Lambda 表达式。它是一个用于表达式中要涉及值的占位符。Lambda 表达式捕获的默认值只有两个：一个是反引号符号（`&`），它将隐式地通过引用捕获自动变量，另一个是等号（`=`），它将隐式地通过复制捕获自动变量（我们将在下一节中进一步讨论）。参数列表与每个函数中的捕获列表类似，在这些函数中我们可以传递值给它。函数体是函数本身的实现。

# 使用 Lambda 表达式处理小函数

想象一下，我们有一个非常小的单行函数，我们只调用它一次。如果我们需要在需要的时候直接编写该函数的操作，那就更好了。实际上，在我们之前讨论 C++标准库的例子中，我们就已经有了这样一个函数。只需回到`for_each.cpp`文件，我们就能找到只被`for_each()`调用一次的`PrintOut()`函数。如果我们使用 Lambda 表达式，可以使这个`for_each`循环更易于阅读。让我们看一下以下代码片段，以了解我们是如何重构`for_each.cpp`文件的：

```cpp
    /* lambda_tiny_func.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>
    #include "../vehicle/vehicle.h"

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_tiny_func.cpp]" << endl;

      // Initializing several Vehicle instances
      Vehicle car("car", 4);
      Vehicle motorcycle("motorcycle", 2);
      Vehicle bicycle("bicycle", 2);
      Vehicle bus("bus", 6);

      // Assigning the preceding Vehicle instances to a vector
      vector<Vehicle> vehicles = { car, motorcycle, bicycle, bus };

      // Displaying the elements of the vector
      // using Lambda expression
      cout << "All vehicles:" << endl;
      for_each(
             begin(vehicles),
             end(vehicles),
             [](const Vehicle &vehicle){
                 cout << vehicle.GetType() << endl;
            });

      return 0;
    }

```

如我们所见，我们已经将`for_each.cpp`文件中使用的`PrintOut()`函数转换成了 Lambda 表达式，并将其传递给了`for_each`循环。它确实会给出与`for_each.cpp`文件相同的输出。然而，现在我们的代码变得更加简洁和易于阅读。

# 使用 Lambda 表达式处理多行函数

Lambda 表达式也可以用于多行函数，因此我们可以将函数体放在它上面。这将使我们的代码更易于阅读。让我们编写一个新的代码。在那个代码中，我们将有一个整数集合，并检查所选元素是否是素数。我们可以创建一个单独的函数，例如`PrintPrime()`，然后调用它。然而，由于素数检查操作只调用一次，如果将其转换为 Lambda 表达式，代码将更易于阅读。代码应该看起来像这样：

```cpp
    /* lambda_multiline_func.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_multiline_func.cpp]" << endl;

      // Initializing a vector containing integer element
      vector<int> vect;
      for (int i = 0; i < 10; ++i)
        vect.push_back(i);

      // Displaying whether or not the element is prime number
      for_each(
             begin(vect),
             end(vect),
             [](int n) {
                cout << n << " is";
                if(n < 2)
                {
                  if(n == 0)
                  cout << " not";
                }
                else
                {
                  for (int j = 2; j < n; ++j)
                    {
                       if (n % j == 0)
                       {
                         cout << " not";
                         break;
                       }
                   }
                 }

                cout << " prime number" << endl;
            });

        return 0;
     }

```

我们应该在屏幕上看到的输出如下：

![图片](img/04d24940-d788-4a9d-a8cd-006f15cfc228.png)

如前一个屏幕截图所示，我们已经成功使用 Lambda 表达式识别了素数。

# 从 Lambda 表达式返回值

我们之前关于 Lambda 表达式的两个示例只是为了在控制台上打印。这意味着该函数不需要返回任何值。然而，如果我们想在函数内部进行计算并返回计算结果，我们可以要求 Lambda 表达式为某个实例返回一个值。让我们看一下以下代码，以了解这个 Lambda 的使用：

```cpp
    /* lambda_returning_value.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_returning_value.cpp]" << endl;

      // Initializing a vector containing integer element
      vector<int> vect;
      for (int i = 0; i < 10; ++i)
        vect.push_back(i);

      // Displaying the elements of vect
      cout << "Original Data:" << endl;
      for_each(
             begin(vect),
             end(vect),
             [](int n){
                cout << n << " ";
            });
      cout << endl;

      // Creating another vect2 vector
      vector<int> vect2;
      // Resize the size of vect2 exactly same with vect
      vect2.resize(vect.size());
      // Doubling the elements of vect and store to vect2
      transform(
              begin(vect),
              end(vect),
              begin(vect2),
              [](int n) {
                return n * n;
            });

      // Displaying the elements of vect2
      cout << "Squared Data:" << endl;
      for_each(
             begin(vect2),
             end(vect2),
             [](int n) {
                cout << n << " ";
            });
      cout << endl;

      // Creating another vect3 vector
      vector<double> vect3;
      // Resize the size of vect3 exactly same with vect
      vect3.resize(vect.size());
      // Finding the average of the elements of vect
      // and store to vect2
      transform(
              begin(vect2),
              end(vect2),
              begin(vect3),
              [](int n) -> double {
                return n / 2.0;
            });

      // Displaying the elements of vect3
      cout << "Average Data:" << endl;
      for_each(
             begin(vect3),
             end(vect3),
             [](double d) {
                cout << d << " ";
            });
      cout << endl;

      return 0;
     }

```

当我们在前一个代码中使用 `transform()` 方法时，我们有一个返回 `n * n` 计算结果的 Lambda 表达式。然而，表达式中没有声明返回类型。这是因为我们可以省略返回类型的声明，因为编译器已经理解该表达式将返回一个 `integer` 值。因此，当我们有另一个与 `vect` 大小相同的向量 `vect2` 时，我们可以调用 `transform()` 方法以及 Lambda 表达式，并将 `vect` 的值加倍并存储在 `vect2` 中。

如果我们想，我们可以指定 Lambda 表达式的返回类型。正如前一个代码所示，我们根据 `vect` 向量的所有值转换了 `vect3` 向量，但现在我们使用箭头符号 (`->`) 指定返回类型为 `double`。前一个代码的结果应如下截图所示：

![](img/e395ef78-2ab2-418d-a186-6fe8b1884f70.png)

如前一个截图所示，我们已成功使用 Lambda 表达式找到了加倍和平均值。

# 将值捕获到 Lambda 表达式中

在我们之前的 Lambda 表达式示例中，我们保持捕获部分和方括号 (`[]`) 为空，因为 Lambda 没有捕获任何内容，并且编译器生成的匿名对象中没有额外的成员变量。我们也可以在方括号中指定我们想要捕获的对象。让我们看一下以下代码片段来讨论这个问题：

```cpp
    /* lambda_capturing_by_value.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_capturing_by_value.cpp]" << endl;

      // Initializing a vector containing integer element
      vector<int> vect;
      for (int i = 0; i < 10; ++i)
      vect.push_back(i);

      // Displaying the elements of vect
      cout << "Original Data:" << endl;
      for_each(
             begin(vect),
             end(vect),
             [](int n){
                cout << n << " ";
             });
      cout << endl;

      // Initializing two variables
      int a = 2;
      int b = 8;

      // Capturing value explicitly from the two variables
      cout << "Printing elements between " << a;
      cout << " and " << b << " explicitly [a,b]:" << endl;
      for_each(
             begin(vect),
             end(vect),
             a,b{
                if (n >= a && n <= b)
                cout << n << " ";
             });
      cout << endl;

      // Modifying variable a and b
      a = 3;
      b = 7;

      // Capturing value implicitly from the two variables
      cout << "printing elements between " << a;
      cout << " and " << b << " implicitly[=]:" << endl;
      for_each(
             begin(vect),
             end(vect),
             ={
                if (n >= a && n <= b)
                cout << n << " ";
            });
      cout << endl;

      return 0;
    }

```

在前一个代码中，我们将尝试显式和隐式地捕获 Lambda 表达式中的值。假设我们有两个变量，`a` 和 `b`，并且我们想显式地捕获这些值，我们可以在 Lambda 表达式中使用 `[a,b]` 语句来指定它们，然后使用函数体内的值。此外，如果我们想隐式地捕获值，只需在捕获部分使用 `[=]`，然后表达式将知道当我们指定它们在函数体内时，我们打算使用哪个变量。如果我们运行前一个代码，我们将在屏幕上得到以下输出：

![](img/f55d1c5a-b38f-4bed-a8e2-b2e2945878c8.png)

我们还可以在不修改 Lambda 表达式函数体外部的值的情况下，修改我们捕获的值的内部状态。为此，我们可以使用之前使用过的相同技术，并添加 `mutable` 关键字，如下面的代码块所示：

```cpp
    /* lambda_capturing_by_value_mutable.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_capturing_by_value_mutable.cpp]" << endl;

      // Initializing a vector containing integer element
      vector<int> vect;
      for (int i = 0; i < 10; ++i)
        vect.push_back(i);

      // Displaying the elements of vect
      cout << "Original Data:" << endl;
      for_each(
             begin(vect),
             end(vect),
             [](int n){
                 cout << n << " ";
            });
      cout << endl;

      // Initializing two variables
      int a = 1;
      int b = 1;

      // Capturing value from the two variables
      // without mutate them
      for_each(
             begin(vect),
             end(vect),
             = mutable {
                 const int old = x;
                 x *= 2;
                 a = b;
                 b = old;
             });

      // Displaying the elements of vect
      cout << "Squared Data:" << endl;
      for_each(
             begin(vect),
             end(vect),
             [](int n) {
                  cout << n << " ";
            });
      cout << endl << endl;

      // Displaying value of variable a and b
      cout << "a = " << a << endl;
      cout << "b = " << b << endl;

      return 0;
    }

```

前一个代码将加倍 `vect` 向量的元素。它使用 Lambda 表达式中的值捕获以及 `mutable` 关键字。正如我们所见，我们通过引用 `(int& x)` 传递了向量元素，并将其乘以二，然后改变了 `a` 和 `b` 的值。然而，由于我们使用了 `mutable` 关键字，`a` 和 `b` 的最终结果将保持不变，尽管我们已经通过引用传递了向量。控制台上的输出如下截图所示：

![图片](img/a1a58756-c71c-4554-afd8-05f6e71dff45.png)

如果我们要更改 `a` 和 `b` 变量的值，我们必须使用 Lambda 表达式通过引用捕获。我们可以通过在 Lambda 表达式中传递引用到尖括号中来实现这一点，例如，`[&a, &b]`。有关更多详细信息，请参阅以下代码片段：

```cpp
    /* lambda_capturing_by_reference.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_capturing_by_reference.cpp]" << endl;

      // Initializing a vector containing integer element
      vector<int> vect;
      for (int i = 0; i < 10; ++i)
        vect.push_back(i);

      // Displaying the elements of vect
      cout << "Original Data:" << endl;
      for_each(
             begin(vect),
             end(vect),
             [](int n){
                 cout << n << " ";
            });
      cout << endl;

      // Initializing two variables
      int a = 1;
      int b = 1;

      // Capturing value from the two variables
      // and mutate them
      for_each(
             begin(vect),
             end(vect),
             &a, &b{
                 const int old = x;
                 x *= 2;
                 a = b;
                 b = old;
            });

      // Displaying the elements of vect
      cout << "Squared Data:" << endl;
      for_each(
             begin(vect),
             end(vect),
             [](int n) {
                 cout << n << " ";
            });
      cout << endl << endl;

      // Displaying value of variable a and b
      cout << "a = " << a << endl;
      cout << "b = " << b << endl;

      return 0;
     }

```

上述代码与 `lambda_capturing_by_value_mutable.cpp` 文件具有相同的行为，该文件将 `vect` 向量的元素翻倍。然而，通过引用捕获，它现在在 `for_each` 循环中处理 `a` 和 `b` 时也会修改它们的值。`a` 和 `b` 的值将在代码结束时改变，如下面的截图所示：

![图片](img/7d62a43b-5efe-443e-899c-c80c88f26608.png)

# 使用初始化捕获准备值

C++14 即将推出的 Lambda 表达式的另一个伟大特性是其初始化捕获。表达式可以捕获变量的值并将其分配给表达式的变量。让我们看看以下实现初始化捕获的代码片段：

```cpp
    /* lambda_initialization_captures.cpp */
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_initialization_captures.cpp]" << endl;

      // Initializing a variable
      int a = 5;
      cout << "Initial a = " << a << endl;

      // Initializing value to lambda using the variable
      auto myLambda = [&x = a]() { x += 2; };

      // Executing the Lambda
      myLambda();

      // Displaying a new value of the variable
      cout << "New a = " << a << endl;

      return 0;
     }

```

如前述代码所示，我们有一个名为 `a` 的整型变量，其值为 `5`。Lambda 表达式 `myLambda` 然后捕获 `a` 的值并在代码中执行它。结果是，现在 `a` 的值将变为 `7`，因为它增加了 `2`。当我们运行上述代码时，以下输出截图应该出现在我们的控制台窗口中：

![图片](img/66131890-ed47-4d35-93cc-c79b68cbd840.png)

从前面的快照中，我们看到我们可以在 Lambda 表达式中准备要包含在计算中的值。

# 编写一个通用的 Lambda 表达式，用于多次与多种不同的数据类型一起使用

在 C++14 之前，我们必须明确指定参数列表的类型。幸运的是，现在在 C++14 中，Lambda 表达式接受 `auto` 作为有效的参数类型。因此，我们现在可以构建一个通用的 Lambda 表达式，如下所示。在下面的代码中，我们只有一个 Lambda 表达式，用于找出传递给表达式的两个数中哪个是最大的。我们将使用 `auto` 关键字在参数声明中，以便它可以通过任何数据类型传递。因此，`findMax()` 函数的参数可以通过 `int` 和 `float` 数据类型传递。代码应如下所示：

```cpp
    /* lambda_expression_generic.cpp */
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[lambda_expression_generic.cpp]" << endl;

      // Creating a generic lambda expression
      auto findMax = [](auto &x, auto &y){
        return x > y ? x : y; };

      // Initializing various variables
      int i1 = 5, i2 = 3;
      float f1 = 2.5f, f2 = 2.05f;

      // Consuming generic lambda expression
      // using integer data type
      cout << "i1 = 5, i2 = 3" << endl;
      cout << "Max: " << findMax(i1, i2) << endl << endl;

      // Consuming generic lambda expression
      // using double data type
      cout << "f1 = 2.5f, f2 = 2.05f" << endl;
      cout << "Max: " << findMax(f1, f2) << endl << endl;

      return 0;
     }

```

我们将在控制台上看到的输出应如下所示：

![图片](img/f2d0eb85-dba1-4aa5-bd8e-445522a4e8a2.png)

C++17 语言计划为 Lambda 表达式引入两个新特性--它们是捕获 `*this`，这允许表达式通过复制捕获封装的对象，以及 `constexpr` Lambda 表达式，这允许我们在编译时使用 Lambda 表达式的结果并生成 `constexpr` 对象。然而，由于 C++17 尚未发布，我们目前无法尝试它。

# 使用智能指针避免手动内存管理

智能指针非常有用，在高效使用 C++ 方面具有基本知识。C++11 在 `memory` 头文件中为智能指针添加了许多新功能。在 C++11 之前，我们长时间使用 `auto_ptr` 作为智能指针。然而，它相当不安全，因为它具有不兼容的复制语义。现在它也被弃用了，我们不应该再使用它。幸运的是，C++ 提出了 `unique_ptr`，它具有类似的功能，但增加了额外的功能，例如添加 `deleters` 和对数组的支持。我们可以用 `auto_ptr` 做的任何事情，我们都可以，并且应该用 `unique_ptr` 来做。我们将深入讨论 `unique_ptr` 以及 C++11 中其他新的智能指针——`shared_ptr` 和 `weak_ptr`。

# 使用 unique_ptr 替换原始指针

我们接下来将看到的是 `unique_ptr` 指针。它是快速、高效的，并且是原始或裸指针的近似直接替换。它提供了独占所有权的语义，它独占拥有它所指向的对象。由于其独占性，当其析构函数被调用时，如果它有一个非空指针，它可以销毁该对象。由于它的独占性，它也不能被复制，因为它没有复制构造函数和复制赋值运算符。尽管它不能被复制，但它可以移动，因为它提供了一个移动构造函数和一个移动赋值运算符。

这些是我们可以使用来构造 `unique_ptr` 的方法：

```cpp
    auto up1 = unique_ptr<int>{};
    auto up2 = unique_ptr<int>{ nullptr };
    auto up3 = unique_ptr<int>{ new int { 1234 } };

```

根据前面的代码，`up1` 和 `up2` 将构建两个指向空（null）的新 `unique_ptr`，而 `up3` 将指向包含 `1234` 值的地址。然而，C++14 添加了一个新的库函数来构造 `unique_ptr`，即 `make_unique`。因此，我们可以按照以下方式构造一个新的 `unique_ptr` 指针：

```cpp
    auto up4 = make_unique<int>(1234);

```

`up4` 变量也将指向包含 `1234` 值的地址。

现在，让我们看一下以下代码块：

```cpp
    /* unique_ptr_1.cpp */
    #include <memory>
    #include <iostream>

    using namespace std;

    struct BodyMass
    {
      int Id;
      float Weight;

      BodyMass(int id, float weight) :
        Id(id),
        Weight(weight)
        {
          cout << "BodyMass is constructed!" << endl;
          cout << "Id = " << Id << endl;
          cout << "Weight = " << Weight << endl;
        }

       ~BodyMass()
       {
         cout << "BodyMass is destructed!" << endl;
       }
     };

     auto main() -> int
     {
       cout << "[unique_ptr_1.cpp]" << endl;
       auto myWeight = make_unique<BodyMass>(1, 165.3f);
       cout << endl << "Doing something!!!" << endl << endl;
       return 0;
     }

```

我们尝试构造一个新的 `unique_ptr` 指针，使其指向包含 `BodyMass` 数据类型的地址。在 `BodyMass` 中，我们有一个构造函数以及一个析构函数。现在，让我们通过运行前面的代码来查看 `unique_ptr` 指针是如何工作的。屏幕上显示的输出应该类似于以下截图：

![](img/1f91c15d-0216-4193-9396-a4d0aba1463d.png)

如前述截图所示，当我们构造 `unique_ptr` 时将调用构造函数。此外，与传统的 C++ 语言不同，在传统 C++ 语言中，当我们使用指针时必须手动释放内存，在现代 C++ 中，当它超出作用域时，内存将自动释放。我们可以看到，当程序退出时，`BodyMass` 的析构函数被调用，这意味着 `myWeight` 已超出作用域。

现在，让我们通过分析以下代码片段来测试 `unique_ptr` 的唯一性：

```cpp
    /* unique_ptr_2.cpp */
    #include <memory>
    #include <iostream>

    using namespace std;

    struct BodyMass
    {
      int Id;
      float Weight;

      BodyMass(int id, float weight) :
        Id(id), 
        Weight(weight)
        {
          cout << "BodyMass is constructed!" << endl;
          cout << "Id = " << Id << endl;
          cout << "Weight = " << Weight << endl;
        }

 BodyMass(const BodyMass &other) :
 Id(other.Id),
 Weight(other.Weight)
 {
 cout << "BodyMass is copy constructed!" << endl;
 cout << "Id = " << Id << endl;
 cout << "Weight = " << Weight << endl;
 }

      ~BodyMass()
       {
          cout << "BodyMass is destructed!" << endl;
       }
    };

    auto main() -> int
    {
      cout << "[unique_ptr_2.cpp]" << endl;

      auto myWeight = make_unique<BodyMass>(1, 165.3f);

      // The compiler will forbid to create another pointer
      // that points to the same allocated memory/object
      // since it's unique pointer
      //auto myWeight2 = myWeight;

      // However, we can do the following expression
      // since it actually copies the object that has been allocated
      // (not the unique_pointer)
      auto copyWeight = *myWeight;

      return 0;
    }

```

如前述代码所示，我们可以看到我们不能将 `unique_ptr` 实例赋值给另一个指针，因为这会破坏 `unique_ptr` 的唯一性。如果我们尝试以下表达式，编译器将抛出错误：

```cpp
    auto myWeight2 = myWeight;

```

然而，由于 `unique_ptr` 已经分配了内存，我们可以将其值赋给另一个对象。为了证明这一点，我们添加了一个复制构造函数来记录以下表达式执行时的日志：

```cpp
    auto copyWeight = *myWeight;

```

如果我们运行前面的 `unique_ptr_2.cpp` 代码，我们将在屏幕上看到以下输出：

![图片](img/f33ba25b-3a90-47b9-a234-0d796c2c263e.png)

如前一个屏幕截图所示，当执行复制赋值时调用复制构造函数。这证明了我们可以复制 `unique_ptr` 对象的值，但不能复制对象本身。

如我们之前讨论的，`unique_ptr` 有移动构造函数，尽管它没有复制构造函数。这种构造的使用可以在以下代码片段中找到：

```cpp
    /* unique_ptr_3.cpp */
    #include <memory>
    #include <iostream>

    using namespace std;

    struct BodyMass
    {
      int Id;
      float Weight;

      BodyMass(int id, float weight) :
        Id(id), 
        Weight(weight)
        {
          cout << "BodyMass is constructed!" << endl;
          cout << "Id = " << Id << endl;
          cout << "Weight = " << Weight << endl;
        }

      ~BodyMass()
       {
         cout << "BodyMass is destructed!" << endl;
       }
    };

    unique_ptr<BodyMass> GetBodyMass()
    {
      return make_unique<BodyMass>(1, 165.3f);
    }

    unique_ptr<BodyMass> UpdateBodyMass(
      unique_ptr<BodyMass> bodyMass)
      {
        bodyMass->Weight += 1.0f;
        return bodyMass;
      }

     auto main() -> int
     {
       cout << "[unique_ptr_3.cpp]" << endl;

       auto myWeight = GetBodyMass();

       cout << "Current weight = " << myWeight->Weight << endl;

       myWeight = UpdateBodyMass(move(myWeight));

       cout << "Updated weight = " << myWeight->Weight << endl;

       return 0;
     }

```

在前面的代码中，我们有两个新的函数--`GetBodyMass()` 和 `UpdateBodyMass()`。我们从 `GetBodyMass()` 函数中构造了一个新的 `unique_ptr` 对象，然后使用 `UpdateBodyMass()` 函数更新其 *Weight* 的值。我们可以看到，当我们向 `UpdateBodyMass()` 函数传递参数时使用了 `move` 函数。这是因为 `unique_ptr` 没有复制构造函数，必须移动才能更新其属性的值。前面代码的屏幕输出如下：

![图片](img/5188478c-d944-4612-b5cf-2b73c89a204d.png)

# 使用 `shared_ptr` 共享对象

与 `unique_ptr` 相比，`shared_ptr` 实现了共享所有权的语义，因此它提供了复制构造函数和复制赋值的特性。尽管它们在实现上有所不同，但 `shared_ptr` 实际上是 `unique_ptr` 的计数版本。我们可以调用 `use_count()` 方法来找出 `shared_ptr` 引用的计数值。每个有效的 `shared_ptr` 实例都被计为一个。我们可以将 `shared_ptr` 实例复制到其他 `shared_ptr` 变量中，引用计数将增加。当 `shared_ptr` 对象被销毁时，析构函数会减少引用计数。只有当计数达到零时，对象才会被删除。现在让我们检查以下 `shared_ptr` 代码：

```cpp
    /* shared_ptr_1.cpp */
    #include <memory>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[shared_ptr_1.cpp]" << endl;

      auto sp1 = shared_ptr<int>{};

      if(sp1)
         cout << "sp1 is initialized" << endl;
      else
         cout << "sp1 is not initialized" << endl;
      cout << "sp1 pointing counter = " << sp1.use_count() << endl;
      if(sp1.unique())
         cout << "sp1 is unique" << endl;
      else
        cout << "sp1 is not unique" << endl;
      cout << endl;

      sp1 = make_shared<int>(1234);

      if(sp1)
        cout << "sp1 is initialized" << endl;
      else
        cout << "sp1 is not initialized" << endl;
      cout << "sp1 pointing counter = " << sp1.use_count() << endl;
      if(sp1.unique())
        cout << "sp1 is unique" << endl;
      else
        cout << "sp1 is not unique" << endl;
      cout << endl;

      auto sp2 = sp1;

      cout << "sp1 pointing counter = " << sp1.use_count() << endl;
      if(sp1.unique())
        cout << "sp1 is unique" << endl;
      else
        cout << "sp1 is not unique" << endl;
      cout << endl;

      cout << "sp2 pointing counter = " << sp2.use_count() << endl;
      if(sp2.unique())
        cout << "sp2 is unique" << endl;
      else
        cout << "sp2 is not unique" << endl;
      cout << endl;

      sp2.reset();

      cout << "sp1 pointing counter = " << sp1.use_count() << endl;
      if(sp1.unique())
        cout << "sp1 is unique" << endl;
      else
        cout << "sp1 is not unique" << endl;
      cout << endl;

      return 0;
    }

```

在我们检查前面代码的每一行之前，让我们看一下应该在控制台窗口中出现的以下输出：

![图片](img/88c9cd88-ace1-4cb8-a3bf-bd423b32376a.png)

首先，我们创建了一个名为 `sp1` 的 `shared_ptr` 对象，但没有对其进行实例化。从控制台可以看到，`sp1` 未初始化，计数器仍然是 `0`。它也不是唯一的，因为指针指向了空值。然后我们使用 `make_shared` 方法构造 `sp1`。现在，`sp1` 已初始化，计数器变为 `1`。它也变得唯一，因为它是唯一的 `shared_ptr` 对象之一（由计数器的值 `1` 证明）。接下来，我们创建另一个变量名为 `sp2`，并将 `sp1` 复制到它。结果，`sp1` 和 `sp2` 现在共享同一个对象，这由计数器和唯一性值证明。然后，在 `sp2` 中调用 `reset()` 方法将销毁 `sp2` 的对象。现在，`sp1` 的计数器变为 `1`，它再次变得唯一。

在 `shared_ptr_1.cpp` 代码中，我们使用 `shared_ptr<int>` 声明 `unique_ptr` 对象，然后调用 `make_shared<int>` 来实例化指针。这是因为我们只需要分析 `shared_ptr` 的行为。然而，我们应该为共享指针使用 `make_shared<>`，因为它必须在内存中某个地方保持引用计数，并且将计数器和内存对象一起分配，而不是分别分配。

# 使用 `weak_ptr` 指针跟踪对象

我们在前面章节中讨论了 `shared_ptr`。实际上，这个指针是一个有点胖的指针。它逻辑上指向两个对象，被管理的对象和通过 `use_count()` 方法使用的引用计数器。每个 `shared_ptr` 基本上都有一个强引用计数，它防止对象被删除，以及一个弱引用计数，如果 `shared_ptr` 对象的使用计数达到 `0`，则不会阻止对象被删除，尽管我们甚至没有使用弱引用计数。因此，我们可以只使用一个引用计数，这样我们就可以使用 `weak_ptr` 指针。`weak_ptr` 指针指向由 `shared_ptr` 管理的对象。`weak_ptr` 的优点是它可以用来引用一个对象，但只有当对象仍然存在时我们才能访问它，而且当强引用计数达到零时，它不会阻止其他引用持有者通过某种方式删除对象。这在处理数据结构时非常有用。让我们看一下以下代码块来分析 `weak_ptr` 的使用：

```cpp
    /* weak_ptr_1.cpp */
    #include <memory>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[weak_ptr_1.cpp]" << endl;

      auto sp = make_shared<int>(1234);

      auto wp = weak_ptr<int>{ sp };

      if(wp.expired())
       cout << "wp is expired" << endl;
      else
       cout << "wp is not expired" << endl;
      cout << "wp pointing counter = " << wp.use_count() << endl;
      if(auto locked = wp.lock())
       cout << "wp is locked. Value = " << *locked << endl;
      else
      {
        cout << "wp is unlocked" << endl;
        wp.reset();
      }
      cout << endl;

      sp = nullptr;

      if(wp.expired())
       cout << "wp is expired" << endl;
      else
       cout << "wp is not expired" << endl;
      cout << "wp pointing counter = " << wp.use_count() << endl;
      if(auto locked = wp.lock())
       cout << "wp is locked. Value = " << *locked << endl;
      else
      {
        cout << "wp is unlocked" << endl;
        wp.reset();
      }
      cout << endl;

      return 0;
     }

```

在分析前面的代码之前，让我们先看看如果我们运行代码，输出控制台中的以下截图：

![](img/ed5b0412-e22b-4ff3-86b2-2cb32d2480b7.png)

首先，我们实例化 `shared_ptr`，正如我们之前讨论的，`weak_ptr` 指向由 `shared_ptr` 管理的对象。然后，我们将 `wp` 赋值给 `shared_ptr` 变量 `sp`。在我们有一个 `weak_ptr` 指针之后，我们检查其行为。通过调用 `expired()` 方法，我们可以确定引用的对象是否已被删除。由于 `wp` 变量刚刚构造，它尚未过期。`weak_ptr` 指针还通过调用 `use_count()` 方法持有对象的计数值，正如我们在 `shared_ptr` 中使用的那样。然后我们调用 `locked()` 方法创建一个管理引用对象的 `shared_ptr` 并找到 `weak_ptr` 指向的值。我们现在有一个指向包含 `1234` 值的地址的 `shared_ptr` 变量。

我们之后将 `sp` 重置为 `nullptr`。尽管我们没有触摸 `weak_ptr` 指针，但它也被改变了。正如我们从控制台截图中所见，现在 `wp` 已过期，因为对象已被删除。计数器也变为 `0`，因为它指向了无物。此外，它被解锁了，因为 `shared_ptr` 对象已被删除。

# 使用元组存储许多不同的数据类型

我们将了解元组，这是一种能够容纳元素集合的对象，每个元素可以是不同类型。这是 C++11 中的一个新特性，为函数式编程提供了力量。当创建一个返回值的函数时，元组将非常有用。此外，由于在函数式编程中函数不会改变全局状态，我们可以返回包含所有需要更改的值的元组。现在，让我们检查以下代码片段：

```cpp
    /* tuples_1.cpp */
    #include <tuple>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[tuples_1.cpp]" << endl;

      // Initializing two Tuples
      tuple<int, string, bool> t1(1, "Robert", true);
      auto t2 = make_tuple(2, "Anna", false);

      // Displaying t1 Tuple elements
      cout << "t1 elements:" << endl;
      cout << get<0>(t1) << endl;
      cout << get<1>(t1) << endl;
      cout << (get<2>(t1) == true ? "Male" : "Female") << endl;
      cout << endl;

      // Displaying t2 Tuple elements
      cout << "t2 elements:" << endl;
      cout << get<0>(t2) << endl;
      cout << get<1>(t2) << endl;
      cout << (get<2>(t2) == true ? "Male" : "Female") << endl;
      cout << endl;

      return 0;
    }

```

在前面的代码中，我们使用 `tuple<int, string, bool>` 和 `make_tuple` 不同的构造技术创建了两个元组 `t1` 和 `t2`。然而，这两种不同的技术将产生相同的结果。显然，在代码中，我们使用 `get<x>(y)` 访问元组中的每个元素，其中 `x` 是索引，`y` 是元组对象。而且，我们有信心在控制台上得到以下结果：

![](img/d1853ef6-56aa-4020-949d-d1e527e4f8ff.png)

# 拆分元组值

在元组类中，另一个有用的成员函数是 `tie()`，它用于将元组拆分为单个对象或创建一个 `lvalue` 引用类型的元组。此外，在元组中，我们还有 `ignore` 辅助类，它是一个占位符，用于在拆分元组时跳过元素。让我们看看 `tie()` 和 `ignore` 在以下代码块中的用法：

```cpp
    /* tuples_2.cpp */
    #include <tuple>
    #include <iostream>

    using namespace std;

    auto main() -> int
   {
      cout << "[tuples_2.cpp]" << endl;

      // Initializing two Tuples
      tuple<int, string, bool> t1(1, "Robert", true);
      auto t2 = make_tuple(2, "Anna", false);

      int i;
      string s;
      bool b;

      // Unpacking t1 Tuples
      tie(i, s, b) = t1;
      cout << "tie(i, s, b) = t1" << endl;
      cout << "i = " << i << endl;
      cout << "s = " << s << endl;
      cout << "b = " << boolalpha << b << endl;
      cout << endl;

      // Unpacking t2 Tuples
      tie(ignore, s, ignore) = t2;
      cout << "tie(ignore, s, ignore) = t2" << endl;
      cout << "new i = " << i << endl;
      cout << "new s = " << s << endl;
      cout << "new b = " << boolalpha << b << endl;
      cout << endl;

      return 0;
    }

```

在前面的代码中，我们有与 `tuples_1.cpp` 相同的两个元组。我们想使用 `tie()` 方法将 `t1` 分别拆分为变量 `i`、`s` 和 `b`。然后，我们将 `t2` 拆分到 `s` 变量中，忽略 `t2` 中的 `int` 和 `bool` 数据。如果我们运行代码，输出应该是以下内容：

![](img/cb5b261a-d591-4942-8062-caadadba0aac.png)

# 返回元组值类型

如我们之前讨论的，当我们想要编写一个返回多个数据的函数时，我们可以在函数式编程中最大化地使用元组。让我们看一下以下代码块，以了解如何返回元组以及访问返回值：

```cpp
    /* tuples_3.cpp */
    #include <tuple>
    #include <iostream>

    using namespace std;

    tuple<int, string, bool> GetData(int DataId)
    {
      if (DataId == 1) 
        return std::make_tuple(0, "Chloe", false);
      else if (DataId == 2) 
        return std::make_tuple(1, "Bryan", true);
      else 
        return std::make_tuple(2, "Zoey", false);
     }

    auto main() -> int
    {
      cout << "[tuples_3.cpp]" << endl;

      auto name = GetData(1);
      cout << "Details of Id 1" << endl;
      cout << "ID = " << get<0>(name) << endl;
      cout << "Name = " << get<1>(name) << endl;
      cout << "Gender = " << (get<2>(name) == true ? 
        "Male" : "Female");
      cout << endl << endl;

      int i;
      string s;
      bool b;
      tie(i, s, b) = GetData(2);
      cout << "Details of Id 2" << endl;
      cout << "ID = " << i << endl;
      cout << "Name = " << s << endl;
      cout << "Gender = " << (b == true ? "Male" : "Female");
      cout << endl;

      return 0;
    }

```

如前述代码所示，我们有一个名为 `GetData()` 的新函数，返回一个 `Tuple` 类型的值。从该函数中，我们将消费其返回的数据。我们首先创建一个名为 `name` 的变量，并从 `GetData()` 函数中获取其值。我们还可以使用 `tie()` 方法来解包来自 `GetData()` 函数的元组，正如我们在代码中访问 ID = `2` 的数据时所见。当我们运行代码时，控制台上的输出应该类似于以下截图：

![图片](img/f46900eb-e1ba-4d02-9f88-af7b6a43b879.png)

# 摘要

通过完成本章，我们刷新了在 C++ 语言中的经验。现在我们知道 C++ 更为现代，它提供了许多辅助我们创建更优程序的功能。我们可以使用标准库来使我们的代码更高效，因为我们不需要编写太多的冗余函数。我们可以使用 Lambda 表达式来使我们的代码整洁、易于阅读和易于维护。我们还可以使用智能指针，这样我们就不必再担心内存管理了。此外，由于我们关注函数式编程中的不可变性，我们将在下一章中深入讨论这一点；元组的使用可以帮助我们确保我们的代码中不涉及任何全局状态。

在下一章中，我们将讨论一等函数和纯函数，这些函数用于净化我们的类，并确保当前函数不涉及任何外部状态。因此，它将避免在函数式代码中产生副作用。
