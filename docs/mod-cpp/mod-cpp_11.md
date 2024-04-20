# 标准库容器、算法和迭代器

本章中将涵盖以下教程：

+   将向量用作默认容器

+   使用位集处理固定大小的位序列

+   使用`vector<bool>`来处理可变大小的位序列

+   在范围内查找元素

+   对范围进行排序

+   初始化范围

+   在范围上使用集合操作

+   使用迭代器在容器中插入新元素

+   编写自己的随机访问迭代器

+   使用非成员函数访问容器

# 将向量用作默认容器

标准库提供了各种类型的容器，用于存储对象的集合；库包括序列容器（如`vector`、`array`或`list`）、有序和无序关联容器（如`set`和`map`），以及不存储数据但提供适应接口向序列容器提供适配的容器适配器（如`stack`和`queue`）。它们都是作为类模板实现的，这意味着它们可以与任何类型一起使用（只要满足容器要求）。虽然您应该始终使用最适合特定问题的容器（不仅在插入、删除、访问元素和内存使用速度方面提供良好性能，而且使代码易于阅读和维护），但默认选择应该是`vector`。在本教程中，我们将看到为什么`vector`应该是首选容器，并且`vector`的最常见操作是什么。

# 准备工作

读者应该熟悉类 C 数组，包括静态分配和动态分配。

类模板`vector`在`<vector>`头文件中的`std`命名空间中可用。

# 如何做...

要初始化`std::vector`类模板，可以使用以下任何一种方法，但您不仅限于这些：

+   从初始化列表初始化：

```cpp
        std::vector<int> v1 { 1, 2, 3, 4, 5 };
```

+   从类 C 数组初始化：

```cpp
        int arr[] = { 1, 2, 3, 4, 5 }; 
        std::vector<int> v2(arr, arr + 5); // { 1, 2, 3, 4, 5 }
```

+   从另一个容器初始化：

```cpp
        std::list<int> l{ 1, 2, 3, 4, 5 }; 
        std::vector<int> v3(l.begin(), l.end()); //{ 1, 2, 3, 4, 5 }
```

+   从计数和值初始化：

```cpp
        std::vector<int> v4(5, 1); // {1, 1, 1, 1, 1}
```

要修改`std::vector`的内容，请使用以下任何一种方法，但您不仅限于这些：

+   使用`push_back()`在向量末尾添加一个元素：

```cpp
        std::vector<int> v1{ 1, 2, 3, 4, 5 };
        v1.push_back(6); // v1 = { 1, 2, 3, 4, 5, 6 }
```

+   使用`pop_back()`从向量末尾删除一个元素：

```cpp
        v1.pop_back();
```

+   使用`insert()`在向量中的任何位置插入：

```cpp
        int arr[] = { 1, 2, 3, 4, 5 };
        std::vector<int> v2;
        v2.insert(v2.begin(), arr, arr + 5); // v2 = { 1, 2, 3, 4, 5 }
```

+   使用`emplace_back()`在向量末尾创建一个元素：

```cpp
        struct foo
        {
          int a;
          double b;
          std::string c;

          foo(int a, double b, std::string const & c) :
            a(a), b(b), c(c) {}
        };

        std::vector<foo> v3;
        v3.emplace_back(1, 1.0, "one"s); 
        // v3 = { foo{1, 1.0, "one"} }
```

+   通过`emplace()`在向量中的任何位置创建元素插入：

```cpp
        v3.emplace(v3.begin(), 2, 2.0, "two"s);
        // v3 = { foo{2, 2.0, "two"}, foo{1, 1.0, "one"} }
```

要修改向量的整个内容，请使用以下任何一种方法，但您不仅限于这些：

+   使用`operator=`从另一个向量分配；这将替换容器的内容：

```cpp
        std::vector<int> v1{ 1, 2, 3, 4, 5 };
        std::vector<int> v2{ 10, 20, 30 };
        v2 = v1; // v1 = { 1, 2, 3, 4, 5 }
```

+   使用`assign()`方法从由开始和结束迭代器定义的另一个序列分配；这将替换容器的内容：

```cpp
        int arr[] = { 1, 2, 3, 4, 5 };
        std::vector<int> v3;
        v3.assign(arr, arr + 5); // v3 = { 1, 2, 3, 4, 5 }
```

+   使用`swap()`方法交换两个向量的内容：

```cpp
        std::vector<int> v4{ 1, 2, 3, 4, 5 };
        std::vector<int> v5{ 10, 20, 30 };
        v4.swap(v5); // v4 = { 10, 20, 30 }, v5 = { 1, 2, 3, 4, 5 }
```

+   使用`clear()`方法删除所有元素：

```cpp
        std::vector<int> v6{ 1, 2, 3, 4, 5 };
        v6.clear(); // v6 = { }
```

+   使用`erase()`方法删除一个或多个元素（需要定义要删除的向量元素范围的迭代器或一对迭代器）：

```cpp
        std::vector<int> v7{ 1, 2, 3, 4, 5 };
        v7.erase(v7.begin() + 2, v7.begin() + 4); // v7 = { 1, 2, 5 }
```

要获取向量中第一个元素的地址，通常将向量的内容传递给类 C API，可以使用以下任何一种方法：

+   使用`data()`方法，返回指向第一个元素的指针，直接访问存储向量元素的底层连续内存序列；这仅在 C++11 之后才可用：

```cpp
        void process(int const * const arr, int const size) 
        { /* do something */ }

        std::vector<int> v{ 1, 2, 3, 4, 5 };
        process(v.data(), static_cast<int>(v.size()));
```

+   获取第一个元素的地址：

```cpp
        process(&v[0], static_cast<int>(v.size()));
```

+   获取由`front()`方法引用的元素的地址：

```cpp
        process(&v.front(), static_cast<int>(v.size()));
```

+   使用从`begin()`返回的迭代器指向的元素的地址：

```cpp
        process(&*v.begin(), static_cast<int>(v.size()));
```

# 它是如何工作的...

`std::vector`类被设计为 C++中最类似和可互操作的 C 类似数组的容器。向量是一个可变大小的元素序列，保证在内存中连续存储，这使得向量的内容可以轻松地传递给一个类似 C 的函数，该函数接受一个指向数组元素的指针，通常还有一个大小。使用向量而不是 C 类似的数组有许多好处，这些好处包括：

+   开发人员不需要进行直接的内存管理，因为容器在内部执行这些操作，分配内存，重新分配和释放。

请注意，向量用于存储对象实例。如果需要存储指针，请不要存储原始指针，而是智能指针。否则，您需要处理指向对象的生命周期管理。

+   +   修改向量大小的可能性。

+   简单的赋值或两个向量的连接。

+   直接比较两个向量。

`vector`类是一个非常高效的容器，所有实现都提供了许多优化，大多数开发人员无法使用 C 类似的数组进行。对其元素的随机访问以及在向量末尾的插入和删除是一个常数*O(1)*操作（前提是不需要重新分配内存），而在其他任何地方的插入和删除是一个线性*O(n)*操作。

与其他标准容器相比，向量具有各种好处：

+   它与类似 C 的数组和类似 C 的 API 兼容；其他容器的内容（除了`std::array`）需要在传递给期望数组的类似 C 的 API 之前复制到向量中。

+   它具有所有容器中元素的最快访问速度。

+   存储元素的每个元素内存开销为零，因为元素存储在连续的空间中，就像 C 数组一样（不像其他容器，如`list`需要额外的指针指向其他元素，或者需要哈希值的关联容器）。

`std::vector`在语义上与类似 C 的数组非常相似，但大小可变。向量的大小可以增加和减少。有两个属性定义了向量的大小：

+   *Capacity*是向量在不执行额外内存分配的情况下可以容纳的元素数量；这由`capacity()`方法表示。

+   *Size*是向量中实际元素的数量；这由`size()`方法表示。

大小始终小于或等于容量。当大小等于容量并且需要添加新元素时，需要修改容量，以便向量有更多元素的空间。在这种情况下，向量分配新的内存块，并将先前的内容移动到新位置，然后释放先前分配的内存。尽管这听起来很耗时（而且确实如此），但实现会按指数增加容量，每次需要更改时将其加倍。因此，平均而言，向量的每个元素只需要移动一次（这是因为在增加容量时向量的所有元素都会移动，但然后可以添加相等数量的元素而不需要进行更多的移动，因为插入是在向量的末尾进行的）。

如果事先知道要插入向量的元素数量，可以首先调用`reserve()`方法将容量增加到至少指定的数量（如果指定的大小小于当前容量，则此方法不执行任何操作），然后再插入元素。

另一方面，如果您需要释放额外保留的内存，可以使用`shrink_to_fit()`方法来请求，但是否释放任何内存是一个实现决定。自 C++11 以来，可用的另一种非绑定方法是与临时的空向量进行交换：

```cpp
    std::vector<int> v{ 1, 2, 3, 4, 5 };
    std::vector<int>().swap(v); // v.size = 0, v.capacity = 0
```

调用`clear()`方法只会从向量中删除所有元素，但不会释放任何内存。

应该注意，向量实现了特定于其他类型容器的操作：

+   `stack`：使用`push_back()`和`emplace_back()`在末尾添加，使用`pop_back()`从末尾移除。请记住，`pop_back()`不会返回已移除的最后一个元素。如果有必要，您需要显式访问它，例如，在移除元素之前使用`back()`方法。

+   `list`：使用`insert()`和`emplace()`在序列中间添加元素，使用`erase()`从序列中的任何位置移除元素。

# 还有更多...

C++容器的经验法则是：除非有充分的理由使用其他容器，否则使用`std::vector`作为默认容器。

# 另请参阅

+   *使用 bitset 表示固定大小的位序列*

+   *使用`vector<bool>`表示可变大小的位序列*

# 使用 bitset 表示固定大小的位序列

开发人员通常会使用位标志进行操作；这可能是因为他们使用操作系统 API（通常用 C 编写），这些 API 接受各种类型的参数（例如选项或样式）以位标志的形式，或者因为他们使用执行类似操作的库，或者仅仅因为某些类型的问题自然而然地使用位标志来解决。可以考虑使用与位和位操作相关的替代方案，例如定义具有每个选项/标志的一个元素的数组，或者定义一个具有成员和函数来模拟位标志的结构，但这些通常更加复杂，而且如果您需要将表示位标志的数值传递给函数，则仍然需要将数组或结构转换为位序列。因此，C++标准提供了一个称为`std::bitset`的固定大小位序列的容器。

# 准备工作

对于本示例，您必须熟悉位操作（与、或、异或、非和移位）。

`bitset`类位于`<bitset>`头文件中的`std`命名空间中。bitset 表示固定大小的位序列，其大小在编译时定义。为方便起见，在本示例中，所有示例都将使用 8 位的位集。

# 如何做到...

要构造一个`std::bitset`对象，请使用其中一个可用的构造函数：

+   所有位都设置为 0 的空位集：

```cpp
        std::bitset<8> b1; // [0,0,0,0,0,0,0,0]
```

+   从数值创建一个位集：

```cpp
        std::bitset<8> b2{ 10 }; // [0,0,0,0,1,0,1,0]
```

+   从包含`'0'`和`'1'`的字符串创建一个位集：

```cpp
        std::bitset<8> b3{ "1010"s }; // [0,0,0,0,1,0,1,0]
```

+   从包含表示`'0'`和`'1'`的任意两个字符的字符串创建一个位集；在这种情况下，我们必须指定哪个字符表示 0，哪个字符表示 1：

```cpp
        std::bitset<8> b4 
          { "ooooxoxo"s, 0, std::string::npos, 'o', 'x' }; 
          // [0,0,0,0,1,0,1,0]
```

测试集合中的单个位或整个集合的特定值，可以使用任何可用的方法：

+   `count()` 以获取设置为 1 的位数：

```cpp
        std::bitset<8> bs{ 10 };
        std::cout << "has " << bs.count() << " 1s" << std::endl;
```

+   `any()` 用于检查是否至少有一个位设置为 1：

```cpp
        if (bs.any()) std::cout << "has some 1s" << std::endl;
```

+   `all()` 以检查是否所有位都设置为 1：

```cpp
        if (bs.all()) std::cout << "has only 1s" << std::endl;
```

+   `none()` 以检查是否所有位都设置为 0：

```cpp
        if (bs.none()) std::cout << "has no 1s" << std::endl;
```

+   `test()` 用于检查单个位的值：

```cpp
        if (!bs.test(0)) std::cout << "even" << std::endl;
```

+   `operator[]` 用于访问和测试单个位：

```cpp
        if(!bs[0]) std::cout << "even" << std::endl;
```

要修改位集的内容，请使用以下任何方法：

+   成员运算符`|=`, `&=`, `^= `和`~` 以执行二进制或、与、异或和非操作，或非成员运算符`|`, `&`, 和`^`：

```cpp
        std::bitset<8> b1{ 42 }; // [0,0,1,0,1,0,1,0]
        std::bitset<8> b2{ 11 }; // [0,0,0,0,1,0,1,1]
        auto b3 = b1 | b2;       // [0,0,1,0,1,0,1,1]
        auto b4 = b1 & b2;       // [0,0,0,0,1,0,1,0]
        auto b5 = b1 ^ b2;       // [1,1,0,1,1,1,1,0]
        auto b6 = ~b1;           // [1,1,0,1,0,1,0,1]
```

+   成员运算符`<<=`, `<<`, `>>=`, `>>` 以执行移位操作：

```cpp
        auto b7 = b1 << 2;       // [1,0,1,0,1,0,0,0]
        auto b8 = b1 >> 2;       // [0,0,0,0,1,0,1,0]
```

+   `flip()` 以将整个集合或单个位从 0 切换为 1 或从 1 切换为 0：

```cpp
        b1.flip();               // [1,1,0,1,0,1,0,1]
        b1.flip(0);              // [1,1,0,1,0,1,0,0]
```

+   `set()` 以将整个集合或单个位更改为`true`或指定的值：

```cpp
        b1.set(0, true);         // [1,1,0,1,0,1,0,1]
        b1.set(0, false);        // [1,1,0,1,0,1,0,0]
```

+   `reset()` 以将整个集合或单个位更改为 false：

```cpp
        b1.reset(2);             // [1,1,0,1,0,0,0,0]
```

要将位集转换为数值或字符串值，请使用以下方法：

+   `to_ulong()` 和 `to_ullong()` 以转换为`unsigned long`或`unsigned long long`：

```cpp
        std::bitset<8> bs{ 42 };
        auto n1 = bs.to_ulong();  // n1 = 42UL
        auto n2 = bs.to_ullong(); // n2 = 42ULL
```

+   `to_string()` 以转换为`std::basic_string`；默认情况下，结果是一个包含`'0'`和`'1'`的字符串，但您可以为这两个值指定不同的字符：

```cpp
        auto s1 = bs.to_string();         // s1 = "00101010"
        auto s2 = bs.to_string('o', 'x'); // s2 = "ooxoxoxo"
```

# 工作原理...

如果您曾经使用过 C 或类似 C 的 API，那么您可能写过或至少看过操作位来定义样式、选项或其他类型值的代码。这通常涉及操作，例如：

+   定义位标志；这些可以是枚举、类中的静态常量，或者是 C 风格中使用`#define`引入的宏。通常，有一个表示无值的标志（样式、选项等）。由于这些被认为是位标志，它们的值是 2 的幂。

+   从集合（即数值）中添加和移除标志。使用位或运算符（`value |= FLAG`）添加位标志，使用位与运算符和取反的标志（`value &= ~FLAG`）来移除位标志。

+   测试标志是否已添加到集合中（`value & FLAG == FLAG`）。

+   调用带有标志作为参数的函数。

以下是一个简单的示例，显示了定义控件边框样式的标志，该控件可以在左侧、右侧、顶部或底部有边框，或者包括这些任意组合，甚至没有边框：

```cpp
    #define BORDER_NONE   0x00
    #define BORDER_LEFT   0x01
    #define BORDER_TOP    0x02
    #define BORDER_RIGHT  0x04
    #define BORDER_BOTTOM 0x08

    void apply_style(unsigned int const style)
    {
      if (style & BORDER_BOTTOM) { /* do something */ }
    }

    // initialize with no flags
    unsigned int style = BORDER_NONE;
    // set a flag
    style = BORDER_BOTTOM;
    // add more flags
    style |= BORDER_LEFT | BORDER_RIGHT | BORDER_TOP;
    // remove some flags
    style &= ~BORDER_LEFT;
    style &= ~BORDER_RIGHT;
    // test if a flag is set
    if ((style & BORDER_BOTTOM) == BORDER_BOTTOM) {}
    // pass the flags as argument to a function
    apply_style(style);
```

标准的`std::bitset`类旨在作为 C++中使用位集的 C 风格工作方式的替代方案。它使我们能够编写更健壮和更安全的代码，因为它通过成员函数抽象了位操作，尽管我们仍然需要确定集合中的每个位表示什么：

+   使用`set()`和`reset()`方法来添加和移除标志，这些方法将位的值设置为 1 或 0（或`true`和`false`）；或者，我们可以使用索引运算符来达到相同的目的。

+   使用`test()`方法来测试位是否被设置。

+   通过构造函数从整数或字符串进行转换，通过成员函数将值转换为整数或字符串，以便可以在期望整数的地方使用 bitset 的值（例如作为函数的参数）。

除了上述操作，`bitset`类还有其他用于执行位操作、移位、测试等的附加方法，这些方法在前一节中已经展示过。

从概念上讲，`std::bitset`是一个表示数值的类，它使您能够访问和修改单个位。然而，在内部，bitset 具有一个整数值数组，它执行位操作。bitset 的大小不限于数值类型的大小；它可以是任何大小，只要它是一个编译时常量。

前一节中的控制边框样式示例可以以以下方式使用`std::bitset`来编写：

```cpp
    struct border_flags
    {
      static const int left = 0;
      static const int top = 1;
      static const int right = 2;
      static const int bottom = 3;
    };

    // initialize with no flags
    std::bitset<4> style;
    // set a flag
    style.set(border_flags::bottom);
    // set more flags
    style
      .set(border_flags::left)
      .set(border_flags::top)
      .set(border_flags::right);
    // remove some flags
    style[border_flags::left] = 0;
    style.reset(border_flags::right);
    // test if a flag is set
    if (style.test(border_flags::bottom)) {}
    // pass the flags as argument to a function
    apply_style(style.to_ulong());
```

# 还有更多...

bitset 可以从整数创建，并可以使用`to_ulong()`或`to_ullong()`方法将其值转换为整数。但是，如果 bitset 的大小大于这些数值类型的大小，并且请求的数值类型大小之外的任何位被设置为`1`，那么这些方法会抛出`std::overflow_error`异常，因为该值无法表示为`unsigned long`或`unsigned long long`。为了提取所有位，我们需要执行以下操作，如下面的代码所示：

+   清除超出`unsigned long`或`unsigned long long`大小的位。

+   将值转换为`unsigned long`或`unsigned long long`。

+   将位集向左移动`unsigned long`或`unsigned long long`位数。

+   一直执行此操作，直到检索到所有位。

```cpp
    template <size_t N>
    std::vector<unsigned long> bitset_to_vectorulong(std::bitset<N> bs)
    {
      auto result = std::vector<unsigned long> {};
      auto const size = 8 * sizeof(unsigned long);
      auto const mask = std::bitset<N>{ static_cast<unsigned long>(-1)};

      auto totalbits = 0;
      while (totalbits < N)
      {
        auto value = (bs & mask).to_ulong();
        result.push_back(value);
        bs >>= size;
        totalbits += size;
      }

      return result;
    }

    std::bitset<128> bs =
           (std::bitset<128>(0xFEDC) << 96) |
           (std::bitset<128>(0xBA98) << 64) |
           (std::bitset<128>(0x7654) << 32) |
           std::bitset<128>(0x3210);

    std::cout << bs << std::endl;

    auto result = bitset_to_vectorulong(bs);
    for (auto const v : result) 
      std::cout << std::hex << v << std::endl;
```

对于无法在编译时知道 bitset 大小的情况，替代方案是`std::vector<bool>`，我们将在下一个示例中介绍。

# 另请参阅

+   *使用`vector<bool>`来表示可变大小的位序列*

# 使用`vector<bool>`来表示可变大小的位序列

在前面的示例中，我们看到了如何使用`std::bitset`来表示固定大小的位序列。然而，有时`std::bitset`不是一个好选择，因为在编译时你不知道位的数量，只是定义一个足够大的位集也不是一个好主意，因为你可能会遇到实际上不够大的情况。这种情况的标准替代方案是使用`std::vector<bool>`容器，它是`std::vector`的一个特化，具有空间和速度优化，因为实现实际上不存储布尔值，而是为每个元素存储单独的位。

然而，因此，`std::vector<bool>`不符合标准容器或顺序容器的要求，`std::vector<bool>::iterator`也不符合前向迭代器的要求。因此，这种特化不能在期望向量的通用代码中使用。另一方面，作为一个向量，它具有与`std::bitset`不同的接口，并且不能被视为数字的二进制表示。没有直接的方法可以从数字或字符串构造`std::vector<bool>`，也不能将其转换为数字或字符串。

# 准备就绪...

本示例假设您熟悉`std::vector`和`std::bitset`。如果您没有阅读之前的示例，*将向量用作默认容器*和*使用 bitset 来表示固定大小的位序列*，请在继续之前阅读。

`vector<bool>`类在`<vector>`头文件中的`std`命名空间中可用。

# 如何做...

要操作`std::vector<bool>`，可以使用与`std::vector<T>`相同的方法，如下例所示：

+   创建一个空向量：

```cpp
        std::vector<bool> bv; // []
```

+   向向量中添加位：

```cpp
        bv.push_back(true);  // [1]
        bv.push_back(true);  // [1, 1]
        bv.push_back(false); // [1, 1, 0]
        bv.push_back(false); // [1, 1, 0, 0]
        bv.push_back(true);  // [1, 1, 0, 0, 1]
```

+   设置单个位的值：

```cpp
        bv[3] = true;        // [1, 1, 0, 1, 1]
```

+   使用通用算法：

```cpp
        auto count_of_ones = std::count(bv.cbegin(), bv.cend(), true);
```

+   从向量中删除位：

```cpp
        bv.erase(bv.begin() + 2); // [1, 1, 1, 1]
```

# 它是如何工作的...

`std::vector<bool>`不是标准向量，因为它旨在通过存储每个元素的单个位而不是布尔值来提供空间优化。因此，它的元素不是以连续序列存储的，也不能替代布尔数组。由于这个原因：

+   索引运算符不能返回对特定元素的引用，因为元素不是单独存储的：

```cpp
        std::vector<bool> bv;
        bv.resize(10);
        auto& bit = bv[0];      // error
```

+   出于前面提到的同样原因，解引用迭代器不能产生对`bool`的引用：

```cpp
        auto& bit = *bv.begin(); // error
```

+   不能保证单个位可以在不同线程中同时独立操作。

+   向量不能与需要前向迭代器的算法一起使用，比如`std::search()`。

+   如果这样的代码需要在列表中提到的任何操作，`std::vector<T>`无法满足预期，那么向量就不能在一些通用代码中使用。

`std::vector<bool>`的替代方案是`std::dequeu<bool>`，它是一个标准容器（双端队列），满足所有容器和迭代器的要求，并且可以与所有标准算法一起使用。然而，这不会像`std::vector<bool>`提供空间优化。

# 还有更多...

`std::vector<bool>`接口与`std::bitset`非常不同。如果想以类似的方式编写代码，可以创建一个在`std::vector<bool>`上的包装器，看起来像`std::bitset`。以下实现提供了类似于`std::bitset`中可用的成员：

```cpp
    class bitvector
    {
      std::vector<bool> bv;
    public:
      bitvector(std::vector<bool> const & bv) : bv(bv) {}
      bool operator[](size_t const i) { return bv[i]; }

      inline bool any() const {
        for (auto b : bv) if (b) return true;
          return false;
      }

      inline bool all() const {
        for (auto b : bv) if (!b) return false;
          return true;
      }

      inline bool none() const { return !any(); }

      inline size_t count() const {
        return std::count(bv.cbegin(), bv.cend(), true);
      }

      inline size_t size() const { return bv.size(); }

      inline bitvector & add(bool const value) {
        bv.push_back(value);
        return *this;
      }

      inline bitvector & remove(size_t const index) {
        if (index >= bv.size())
          throw std::out_of_range("Index out of range");
        bv.erase(bv.begin() + index);
        return *this;
      }

      inline bitvector & set(bool const value = true) {
        for (size_t i = 0; i < bv.size(); ++i)
          bv[i] = value;
        return *this;
      }

      inline bitvector& set(size_t const index, bool const value = true) {
        if (index >= bv.size())
          throw std::out_of_range("Index out of range");
        bv[index] = value;
        return *this;
      }

      inline bitvector & reset() {
        for (size_t i = 0; i < bv.size(); ++i) bv[i] = false;
        return *this;
      }

      inline bitvector & reset(size_t const index) {
        if (index >= bv.size())
          throw std::out_of_range("Index out of range");
        bv[index] = false;
        return *this;
      }

      inline bitvector & flip() {
        bv.flip();
        return *this;
      }

      std::vector<bool>& data() { return bv; }
    };
```

这只是一个基本的实现，如果要使用这样的包装器，应该添加额外的方法，比如位逻辑操作、移位、也许从流中读取和写入等等。然而，通过上述代码，我们可以写出以下例子：

```cpp
    bitvector bv;
    bv.add(true).add(true).add(false); // [1, 1, 0]
    bv.add(false);                     // [1, 1, 0, 0]
    bv.add(true);                      // [1, 1, 0, 0, 1]

    if (bv.any()) std::cout << "has some 1s" << std::endl;
    if (bv.all()) std::cout << "has only 1s" << std::endl;
    if (bv.none()) std::cout << "has no 1s" << std::endl;
    std::cout << "has " << bv.count() << " 1s" << std::endl;

    bv.set(2, true);                   // [1, 1, 1, 0, 1]
    bv.set();                          // [1, 1, 1, 1, 1]

    bv.reset(0);                       // [0, 1, 1, 1, 1]
    bv.reset();                        // [0, 0, 0, 0, 0]

    bv.flip();                         // [1, 1, 1, 1, 1]
```

# 另请参阅

+   *将向量用作默认容器*

+   *使用 bitset 来表示固定大小的位序列*

# 在范围内查找元素

在任何应用程序中，我们经常做的最常见的操作之一就是搜索数据。因此，标准库提供了许多用于搜索标准容器或任何可以表示范围并由开始和结束迭代器定义的东西的通用算法，这并不奇怪。在这个示例中，我们将看到这些标准算法是什么，以及它们如何使用。

# 准备工作

在这个示例中的所有示例中，我们将使用`std::vector`，但所有算法都适用于由开始和结束迭代器定义的范围，无论是输入迭代器还是前向迭代器，具体取决于算法（有关各种类型迭代器的更多信息，请参阅示例*编写自己的随机访问迭代器*）。所有这些算法都在`<algorithm>`头文件中的`std`命名空间中可用。

# 如何做...

以下是可以用于在范围中查找元素的算法列表：

+   使用`std::find()`来在范围中查找值；这个算法返回一个迭代器，指向第一个等于该值的元素：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto it = std::find(v.cbegin(), v.cend(), 3);
        if (it != v.cend()) std::cout << *it << std::endl;
```

+   使用`std::find_if()`来查找范围中满足一元谓词条件的值；这个算法返回一个迭代器，指向谓词返回`true`的第一个元素：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto it = std::find_if(v.cbegin(), v.cend(), 
                               [](int const n) {return n > 10; });
        if (it != v.cend()) std::cout << *it << std::endl;
```

+   使用`std::find_if_not()`来查找范围中不满足一元谓词的条件的值；这个算法返回一个迭代器，指向谓词返回`false`的第一个元素：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto it = std::find_if_not(v.cbegin(), v.cend(), 
                            [](int const n) {return n % 2 == 1; });
        if (it != v.cend()) std::cout << *it << std::endl;
```

+   使用`std::find_first_of()`在另一个范围中搜索来自另一个范围的任何值的出现；这个算法返回一个迭代器，指向找到的第一个元素：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };
        std::vector<int> p{ 5, 7, 11 };

        auto it = std::find_first_of(v.cbegin(), v.cend(),
                                     p.cbegin(), p.cend());
        if (it != v.cend()) 
          std::cout << "found " << *it
                    << " at index " << std::distance(v.cbegin(), it)
                    << std::endl;
```

+   使用`std::find_end()`来查找范围中元素子范围的最后出现；这个算法返回一个迭代器，指向范围中最后一个子范围的第一个元素：

```cpp
        std::vector<int> v1{ 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1 };
        std::vector<int> v2{ 1, 0, 1 };

        auto it = std::find_end(v1.cbegin(), v1.cend(),
                                v2.cbegin(), v2.cend());
        if (it != v1.cend())
          std::cout << "found at index "
                    << std::distance(v1.cbegin(), it) << std::endl;
```

+   使用`std::search()`来查找范围中子范围的第一个出现；这个算法返回一个迭代器，指向范围中子范围的第一个元素：

```cpp
        auto text = "The quick brown fox jumps over the lazy dog"s;
        auto word = "over"s;

        auto it = std::search(text.cbegin(), text.cend(),
                              word.cbegin(), word.cend());

        if (it != text.cend())
          std::cout << "found " << word
                    << " at index " 
                    << std::distance(text.cbegin(), it) << std::endl;
```

+   使用带有*searcher*的`std::search()`，*searcher*是实现搜索算法并满足一些预定义标准的类。这个重载的`std::search()`是在 C++17 中引入的，可用的标准 searchers 实现了*Boyer-Moore*和*Boyer-Moore-Horspool*字符串搜索算法：

```cpp
        auto text = "The quick brown fox jumps over the lazy dog"s;
        auto word = "over"s;

        auto it = std::search(
          text.cbegin(), text.cend(),
          std::make_boyer_moore_searcher(word.cbegin(), word.cend()));

        if (it != text.cend())
          std::cout << "found " << word
                    << " at index " 
                    << std::distance(text.cbegin(), it) << std::endl;
```

+   使用`std::search_n()`来在范围中搜索值的*N*个连续出现；这个算法返回一个迭代器，指向范围中找到的序列的第一个元素：

```cpp
        std::vector<int> v{ 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1 };

        auto it = std::search_n(v.cbegin(), v.cend(), 2, 0);
        if (it != v.cend())
          std::cout << "found at index " 
                    << std::distance(v.cbegin(), it) << std::endl;
```

+   使用`std::adjacent_find()`来查找范围中相邻的两个元素，它们相等或满足二元谓词；这个算法返回一个迭代器，指向找到的第一个元素：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto it = std::adjacent_find(v.cbegin(), v.cend());
        if (it != v.cend())
          std::cout << "found at index " 
                    << std::distance(v.cbegin(), it) << std::endl;

       auto it = std::adjacent_find(
         v.cbegin(), v.cend(),
         [](int const a, int const b) {
           return IsPrime(a) && IsPrime(b); });

        if (it != v.cend())
          std::cout << "found at index " 
                    << std::distance(v.cbegin(), it) << std::endl;
```

+   使用`std::binary_search()`来查找排序范围中是否存在元素；这个算法返回一个布尔值，指示是否找到了该值：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto success = std::binary_search(v.cbegin(), v.cend(), 8);
        if (success) std::cout << "found" << std::endl;
```

+   使用`std::lower_bound()`来查找范围中第一个不小于指定值的元素；这个算法返回一个指向元素的迭代器：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto it = std::lower_bound(v.cbegin(), v.cend(), 1);
        if (it != v.cend())
          std::cout << "lower bound at "
                    << std::distance(v.cbegin(), it) << std::endl;
```

+   使用`std::upper_bound()`来查找范围中大于指定值的第一个元素；这个算法返回一个指向元素的迭代器：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto it = std::upper_bound(v.cbegin(), v.cend(), 1);
        if (it != v.cend())
          std::cout << "upper bound at "
                    << std::distance(v.cbegin(), it) << std::endl;
```

+   使用`std::equal_range()`来查找范围中值等于指定值的子范围。这个算法返回一对迭代器，定义了子范围的第一个和结束迭代器；这两个迭代器等同于`std::lower_bound()`和`std::upper_bound()`返回的迭代器：

```cpp
        std::vector<int> v{ 1, 1, 2, 3, 5, 8, 13 };

        auto bounds = std::equal_range(v.cbegin(), v.cend(), 1);
        std::cout << "range between indexes "
                  << std::distance(v.cbegin(), bounds.first)
                  << " and "
                  << std::distance(v.cbegin(), bounds.second)
                  << std::endl;
```

# 工作原理...

这些算法的工作方式非常相似：它们都以定义可搜索范围的迭代器和依赖于每个算法的其他参数作为参数。除了`std::search()`返回一个布尔值，`std::equal_range()`返回一对迭代器之外，它们都返回指向搜索元素或子范围的迭代器。这些迭代器必须与范围的结束迭代器（即最后一个元素之后的位置）进行比较，以检查搜索是否成功。如果搜索没有找到元素或子范围，则返回值是结束迭代器。

所有这些算法都有多个重载，但在*如何做...*部分，我们只看了一个特定的重载，以展示如何使用该算法。要获取所有重载的完整参考，请参阅其他来源。

在所有前面的示例中，我们使用了常量迭代器，但是所有这些算法都可以使用可变迭代器和反向迭代器。因为它们以迭代器作为输入参数，所以它们可以使用标准容器、类 C 数组或任何表示序列并具有迭代器的东西。

有必要特别注意`std::binary_search()`算法：定义要搜索的范围的迭代器参数至少应满足前向迭代器的要求。无论提供的迭代器的类型如何，比较的次数始终与范围的大小的对数成正比。但是，如果迭代器是随机访问的，则迭代器的增量数量是不同的，在这种情况下，增量的数量也是对数的，或者不是随机访问的，这种情况下，它是线性的，并且与范围的大小成正比。

除了`std::find_if_not()`之外，所有这些算法在 C++11 之前就已经存在。但是，它们的一些重载已经在更新的标准中引入。例如，`std::search()`在 C++17 中引入了几个重载。其中一个重载的形式如下：

```cpp
    template<class ForwardIterator, class Searcher>
    ForwardIterator search(ForwardIterator first, ForwardIterator last,
                           const Searcher& searcher );
```

此重载搜索由搜索器函数对象定义的模式的出现，标准提供了几种实现：

+   `default_searcher` 基本上将搜索委托给标准的`std::search()`算法。

+   `boyer_moore_searcher` 实现了 Boyer-Moore 算法用于字符串搜索。

+   `boyer_moore_horspool_algorithm` 实现了 Boyer-Moore-Horspool 算法用于字符串搜索。

# 还有更多...

许多标准容器都有一个成员函数`find()`，用于在容器中查找元素。当这样的方法可用且符合您的需求时，应优先使用这些成员函数，因为这些成员函数是根据每个容器的特点进行了优化。

# 另请参阅

+   *使用向量作为默认容器*

+   *初始化范围*

+   *在范围上使用集合操作*

+   *对范围进行排序*

# 对范围进行排序

在前面的食谱中，我们看了搜索范围的标准通用算法。我们经常需要做的另一个常见操作是对范围进行排序，因为许多例程，包括一些搜索算法，都需要排序的范围。标准库提供了几个用于对范围进行排序的通用算法，在本食谱中，我们将看到这些算法是什么，以及它们如何使用。

# 准备工作

排序通用算法使用由开始和结束迭代器定义的范围，并且可以对标准容器、类 C 数组或任何表示序列并具有随机迭代器的东西进行排序。但是，本食谱中的所有示例都将使用`std::vector`。

# 如何做...

以下是搜索范围的标准通用算法列表：

+   使用`std::sort()`对范围进行排序：

```cpp
        std::vector<int> v{3, 13, 5, 8, 1, 2, 1};

        std::sort(v.begin(), v.end());
        // v = {1, 1, 2, 3, 5, 8, 13}

        std::sort(v.begin(), v.end(), std::greater<>());
        // v = {13, 8, 5, 3, 2, 1 ,1}
```

+   使用`std::stable_sort()`对范围进行排序，但保持相等元素的顺序：

```cpp
        struct Task
        {
          int priority;
          std::string name;
        };

        bool operator<(Task const & lhs, Task const & rhs) {
          return lhs.priority < rhs.priority;
        }

        bool operator>(Task const & lhs, Task const & rhs) {
          return lhs.priority > rhs.priority;
        }

        std::vector<Task> v{ 
          { 10, "Task 1"s }, { 40, "Task 2"s }, { 25, "Task 3"s },
          { 10, "Task 4"s }, { 80, "Task 5"s }, { 10, "Task 6"s },
        };

        std::stable_sort(v.begin(), v.end());
        // {{ 10, "Task 1" },{ 10, "Task 4" },{ 10, "Task 6" },
        //  { 25, "Task 3" },{ 40, "Task 2" },{ 80, "Task 5" }}

        std::stable_sort(v.begin(), v.end(), std::greater<>());
        // {{ 80, "Task 5" },{ 40, "Task 2" },{ 25, "Task 3" },
        //  { 10, "Task 1" },{ 10, "Task 4" },{ 10, "Task 6" }}
```

+   使用`std::partial_sort()`对范围的一部分进行排序（并使其余部分处于未指定的顺序）：

```cpp
        std::vector<int> v{ 3, 13, 5, 8, 1, 2, 1 };

        std::partial_sort(v.begin(), v.begin() + 4, v.end());
        // v = {1, 1, 2, 3, ?, ?, ?}

        std::partial_sort(v.begin(), v.begin() + 4, v.end(),
                          std::greater<>());
        // v = {13, 8, 5, 3, ?, ?, ?}
```

+   使用`std::partial_sort_copy()`对范围的一部分进行排序，通过将已排序的元素复制到第二个范围并保持原始范围不变：

```cpp
        std::vector<int> v{ 3, 13, 5, 8, 1, 2, 1 };
        std::vector<int> vc(v.size());

        std::partial_sort_copy(v.begin(), v.end(), 
                               vc.begin(), vc.end());
        // v = {3, 13, 5, 8, 1, 2, 1}
        // vc = {1, 1, 2, 3, 5, 8, 13}

        std::partial_sort_copy(v.begin(), v.end(), 
                               vc.begin(), vc.end(), std::greater<>());
        // vc = {13, 8, 5, 3, 2, 1, 1}
```

+   使用`std::nth_element()`对范围进行排序，使得第*N*个元素是如果范围完全排序时将在该位置的元素，并且它之前的元素都更小，之后的元素都更大，没有任何保证它们也是有序的：

```cpp
        std::vector<int> v{ 3, 13, 5, 8, 1, 2, 1 };

        std::nth_element(v.begin(), v.begin() + 3, v.end());
        // v = {1, 1, 2, 3, 5, 8, 13}

        std::nth_element(v.begin(), v.begin() + 3, v.end(),
                         std::greater<>());
        // v = {13, 8, 5, 3, 2, 1, 1}
```

+   使用`std::is_sorted()`来检查一个范围是否已排序：

```cpp
        std::vector<int> v { 1, 1, 2, 3, 5, 8, 13 };

        auto sorted = std::is_sorted(v.cbegin(), v.cend());
        sorted = std::is_sorted(v.cbegin(), v.cend(), 
                                std::greater<>());
```

+   使用`std::is_sorted_until()`来从范围的开头找到一个已排序的子范围：

```cpp
        std::vector<int> v{ 3, 13, 5, 8, 1, 2, 1 };

        auto it = std::is_sorted_until(v.cbegin(), v.cend());
        auto length = std::distance(v.cbegin(), it);
```

# 它是如何工作的...

所有前面的一般算法都接受随机迭代器作为参数来定义要排序的范围，并且其中一些还额外接受一个输出范围。它们都有重载，一个需要比较函数来对元素进行排序，另一个不需要，并使用`operator<`来比较元素。

这些算法的工作方式如下：

+   +   `std::sort()`修改输入范围，使其元素根据默认或指定的比较函数进行排序；排序的实际算法是一个实现细节。

+   `std::stable_sort()`类似于`std::sort()`，但它保证保留相等元素的原始顺序。

+   `std::partial_sort()`接受三个迭代器参数，表示范围中的第一个、中间和最后一个元素，其中中间可以是任何元素，而不仅仅是自然中间位置的元素。结果是一个部分排序的范围，使得原始范围的前`middle - first`个最小元素，即`[first, last)`，在`[first, middle)`子范围中找到，其余元素以未指定的顺序在`[middle, last)`子范围中。

+   `std::partial_sort_copy()`不是`std::partial_copy()`的变体，正如名称可能暗示的那样，而是`std::sort()`的变体。它对范围进行排序，而不改变它，通过将其元素复制到输出范围。算法的参数是输入范围和输出范围的第一个和最后一个迭代器。如果输出范围的大小*M*大于或等于输入范围的大小*N*，则输入范围完全排序并复制到输出范围；输出范围的前*N*个元素被覆盖，最后*M-N*个元素保持不变。如果输出范围小于输入范围，则只有输入范围中的前*M*个排序元素被复制到输出范围（在这种情况下，输出范围完全被覆盖）。

+   `std::nth_element()`基本上是选择算法的实现，这是一种用于找到范围中第*N*个最小元素的算法。该算法接受三个迭代器参数，表示范围的第一个、第*N*个和最后一个元素，并部分排序范围，以便在排序后，第*N*个元素是如果范围已完全排序时将在该位置的元素。在修改后的范围中，第*n*个元素之前的所有*N-1*个元素都小于它，第*n*个元素之后的所有元素都大于它。但是，这些其他元素的顺序没有保证。

+   `std::is_sorted()`检查指定范围是否根据指定或默认的比较函数进行排序，并返回一个布尔值来指示。

+   `std::is_sorted_until()`找到指定范围的已排序子范围，从开头开始，使用提供的比较函数或默认的`operator<`。返回的值是表示已排序子范围的上界的迭代器，也是最后一个已排序元素的迭代器。

# 还有更多...

一些标准容器，如`std::list`和`std::forward_list`，提供了一个成员函数`sort()`，该函数针对这些容器进行了优化。应优先使用这些成员函数，而不是一般的标准算法`std::sort()`。

# 另请参阅

+   *使用 vector 作为默认容器*

+   *初始化一个范围*

+   *在范围上使用集合操作*

+   *在范围内查找元素*

# 初始化范围

在之前的示例中，我们探索了用于在范围内搜索和对范围进行排序的一般标准算法。算法库提供了许多其他一般算法，其中包括用于填充范围值的几个算法。在本示例中，您将了解这些算法是什么以及应该如何使用它们。

# 准备工作

本示例中的所有示例都使用`std::vector`。但是，像所有一般算法一样，我们将在本示例中看到的算法使用迭代器来定义范围的边界，因此可以与任何标准容器、类似 C 的数组或定义了前向迭代器的表示序列的自定义类型一起使用。

除了`std::iota()`，它在`<numeric>`头文件中可用，所有其他算法都在`<algorithm>`头文件中找到。

# 操作步骤...

要为范围分配值，请使用以下任何标准算法：

+   `std::fill()` 用于为范围内的所有元素分配一个值；范围由第一个和最后一个前向迭代器定义：

```cpp
        std::vector<int> v(5);
        std::fill(v.begin(), v.end(), 42);
        // v = {42, 42, 42, 42, 42}
```

+   `std::fill_n()` 用于为范围内的多个元素分配值；范围由第一个前向迭代器和一个计数器定义，该计数器指示应分配指定值的元素数量：

```cpp
        std::vector<int> v(10);
        std::fill_n(v.begin(), 5, 42);
        // v = {42, 42, 42, 42, 42, 0, 0, 0, 0, 0}
```

+   `std::generate()` 用于将函数返回的值分配给范围内的元素；范围由第一个和最后一个前向迭代器定义，并且该函数为范围内的每个元素调用一次：

```cpp
        std::random_device rd{};
        std::mt19937 mt{ rd() };
        std::uniform_int_distribution<> ud{1, 10};
        std::vector<int> v(5);
        std::generate(v.begin(), v.end(), 
                      [&ud, &mt] {return ud(mt); }); 
```

+   `std::generate_n()` 用于将函数返回的值分配给范围内的多个元素；范围由第一个前向迭代器和一个计数器定义，该计数器指示应为每个元素调用一次的函数分配值：

```cpp
        std::vector<int> v(5);
        auto i = 1;
        std::generate_n(v.begin(), v.size(), [&i] { return i*i++; });
        // v = {1, 4, 9, 16, 25}
```

+   `std::iota()` 用于为范围内的元素分配顺序递增的值；范围由第一个和最后一个前向迭代器定义，并且使用从指定初始值开始的前缀`operator++`递增值：

```cpp
        std::vector<int> v(5);
        std::iota(v.begin(), v.end(), 1);
        // v = {1, 2, 3, 4, 5}
```

# 工作原理...

`std::fill()` 和 `std::fill_n()` 的工作方式类似，但在指定范围的方式上有所不同：前者由第一个和最后一个迭代器指定，后者由第一个迭代器和计数指定。第二个算法返回一个迭代器，如果计数大于零，则表示代表最后一个分配的元素，否则表示范围的第一个元素的迭代器。

`std::generate()` 和 `std::generate_n()` 也类似，只是在指定范围的方式上有所不同。第一个使用两个迭代器定义范围的下限和上限，第二个使用第一个元素的迭代器和计数。与`std::fill_n()`一样，`std::generate_n()`也返回一个迭代器，如果计数大于零，则表示代表最后一个分配的元素，否则表示范围的第一个元素的迭代器。这些算法为范围内的每个元素调用指定的函数，并将返回的值分配给元素。生成函数不接受任何参数，因此不能将参数的值传递给函数，因为这是用于初始化范围元素的函数。如果需要使用元素的值来生成新值，则应使用`std::transform()`。

`std::iota()` 的名称取自 APL 编程语言中的 ι (iota) 函数，尽管它是最初的 STL 的一部分，但它仅在 C++11 中的标准库中包含。此函数接受范围的第一个和最后一个迭代器以及分配给范围的第一个元素的初始值，然后使用前缀`operator++`为范围中的其余元素生成顺序递增的值。

# 另请参阅

+   *使用向量作为默认容器*

+   *对范围进行排序*

+   *在范围上使用集合操作*

+   *在范围内查找元素*

+   *生成伪随机数* 第九章的示例，*使用数字和字符串*

+   *初始化伪随机数生成器的内部状态的所有位* 第九章的示例，*使用数字和字符串*

# 在范围上使用集合操作

标准库提供了几种用于集合操作的算法，使我们能够对排序范围进行并集、交集或差异操作。在本示例中，我们将看到这些算法是什么以及它们是如何工作的。

# 准备工作

集合操作的算法使用迭代器，这意味着它们可以用于标准容器、类似 C 的数组或任何表示具有输入迭代器的序列的自定义类型。本示例中的所有示例都将使用`std::vector`。

对于下一节中的所有示例，我们将使用以下范围：

```cpp
    std::vector<int> v1{ 1, 2, 3, 4, 4, 5 };
    std::vector<int> v2{ 2, 3, 3, 4, 6, 8 };
    std::vector<int> v3;
```

# 操作步骤...

使用以下通用算法进行集合操作：

+   `std::set_union()`计算两个范围的并集并将结果存储到第三个范围中：

```cpp
        std::set_union(v1.cbegin(), v1.cend(),
                       v2.cbegin(), v2.cend(),
                       std::back_inserter(v3));
        // v3 = {1, 2, 3, 3, 4, 4, 5, 6, 8}
```

+   `std::merge()`将两个范围的内容合并到第三个范围中；这类似于`std::set_union()`，不同之处在于它将输入范围的整个内容复制到输出范围中，而不仅仅是它们的并集：

```cpp
        std::merge(v1.cbegin(), v1.cend(),
                   v2.cbegin(), v2.cend(),
                   std::back_inserter(v3));
        // v3 = {1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 8}
```

+   `std::set_intersection()`计算两个范围的交集并将结果存储到第三个范围中：

```cpp
        std::set_intersection(v1.cbegin(), v1.cend(),
                              v2.cbegin(), v2.cend(),
                              std::back_inserter(v3));
        // v3 = {2, 3, 4}
```

+   `std::set_difference()`计算两个范围的差异并将结果存储到第三个范围中；输出范围将包含来自第一个范围的元素，这些元素在第二个范围中不存在：

```cpp
        std::set_difference(v1.cbegin(), v1.cend(),
                            v2.cbegin(), v2.cend(),
                            std::back_inserter(v3));
        // v3 = {1, 4, 5}
```

+   `std::set_symmetric_difference()`计算两个范围的对称差并将结果存储到第三个范围中；输出范围将包含存在于任一输入范围中但仅存在于一个输入范围中的元素：

```cpp
        std::set_symmetric_difference(v1.cbegin(), v1.cend(),
                                      v2.cbegin(), v2.cend(),
                                      std::back_inserter(v3));
        // v3 = {1, 3, 4, 5, 6, 8}
```

+   `std::includes()`用于检查一个范围是否是另一个范围的子集（即，它的所有元素也存在于另一个范围中）：

```cpp
        std::vector<int> v1{ 1, 2, 3, 4, 4, 5 };
        std::vector<int> v2{ 2, 3, 3, 4, 6, 8 };
        std::vector<int> v3{ 1, 2, 4 };
        std::vector<int> v4{ };

        auto i1 = std::includes(v1.cbegin(), v1.cend(), 
                                v2.cbegin(), v2.cend()); // i1 = false
        auto i2 = std::includes(v1.cbegin(), v1.cend(), 
                                v3.cbegin(), v3.cend()); // i2 = true
        auto i3 = std::includes(v1.cbegin(), v1.cend(), 
                                v4.cbegin(), v4.cend()); // i3 = true
```

# 工作原理...

所有从两个输入范围产生新范围的集合操作实际上具有相同的接口，并且以类似的方式工作：

+   它们接受两个输入范围，每个范围由第一个和最后一个输入迭代器定义。

+   它们接受一个输出迭代器，指向将插入元素的输出范围。

+   它们有一个重载，接受一个额外的参数，表示必须返回`true`的比较二进制函数对象，如果第一个参数小于第二个参数。当未指定比较函数对象时，将使用`operator<`。

+   它们返回一个指向构造的输出范围结尾的迭代器。

+   输入范围必须使用`operator<`或提供的比较函数进行排序，具体取决于所使用的重载。

+   输出范围不得与两个输入范围重叠。

我们将使用 POD 类型`Task`的向量进行额外示例，这与我们之前使用的类型相同：

```cpp
    struct Task
    {
      int priority;
      std::string name;
    };

    bool operator<(Task const & lhs, Task const & rhs) {
      return lhs.priority < rhs.priority;
    } 

    bool operator>(Task const & lhs, Task const & rhs) {
      return lhs.priority > rhs.priority;
    }

    std::vector<Task> v1{
      { 10, "Task 1.1"s },
      { 20, "Task 1.2"s },
      { 20, "Task 1.3"s },
      { 20, "Task 1.4"s },
      { 30, "Task 1.5"s },
      { 50, "Task 1.6"s },
    };

    std::vector<Task> v2{
      { 20, "Task 2.1"s },
      { 30, "Task 2.2"s },
      { 30, "Task 2.3"s },
      { 30, "Task 2.4"s },
      { 40, "Task 2.5"s },
      { 50, "Task 2.6"s },
    };
```

每个算法产生输出范围的特定方式在此处描述：

+   `std::set_union()`将输入范围中存在的所有元素复制到输出范围，生成一个新的排序范围。如果一个元素在第一个范围中出现*M*次，在第二个范围中出现*N*次，那么第一个范围中的所有*M*个元素将按其现有顺序复制到输出范围中，然后如果*N > M*，则从第二个范围中复制* N-M *个元素到输出范围中，否则为 0 个元素：

```cpp
        std::vector<Task> v3;
        std::set_union(v1.cbegin(), v1.cend(),
                       v2.cbegin(), v2.cend(),
                       std::back_inserter(v3));
        // v3 = {{10, "Task 1.1"},{20, "Task 1.2"},{20, "Task 1.3"},
        //       {20, "Task 1.4"},{30, "Task 1.5"},{30, "Task 2.3"},
        //       {30, "Task 2.4"},{40, "Task 2.5"},{50, "Task 1.6"}}
```

+   `std::merge()`将两个输入范围中的所有元素复制到输出范围中，生成一个新的排序范围，其排序方式与比较函数有关：

```cpp
        std::vector<Task> v4;
        std::merge(v1.cbegin(), v1.cend(),
                   v2.cbegin(), v2.cend(),
                   std::back_inserter(v4));
        // v4 = {{10, "Task 1.1"},{20, "Task 1.2"},{20, "Task 1.3"},
        //       {20, "Task 1.4"},{20, "Task 2.1"},{30, "Task 1.5"},
        //       {30, "Task 2.2"},{30, "Task 2.3"},{30, "Task 2.4"},
        //       {40, "Task 2.5"},{50, "Task 1.6"},{50, "Task 2.6"}}
```

+   `std::set_intersection()`将在两个输入范围中找到的所有元素复制到输出范围中，生成一个新的排序范围，其排序方式与比较函数有关：

```cpp
        std::vector<Task> v5;
        std::set_intersection(v1.cbegin(), v1.cend(),
                              v2.cbegin(), v2.cend(),
                              std::back_inserter(v5));
        // v5 = {{20, "Task 1.2"},{30, "Task 1.5"},{50, "Task 1.6"}}
```

+   `std::set_difference()`将第一个输入范围中所有未在第二个输入范围中找到的元素复制到输出范围。对于在两个范围中找到的等效元素，适用以下规则：如果一个元素在第一个范围中出现*M*次，在第二个范围中出现*N*次，如果*M > N*，则复制*M-N*次；否则不复制：

```cpp
        std::vector<Task> v6;
        std::set_difference(v1.cbegin(), v1.cend(),
                            v2.cbegin(), v2.cend(),
                            std::back_inserter(v6));
        // v6 = {{10, "Task 1.1"},{20, "Task 1.3"},{20, "Task 1.4"}}
```

+   `std::set_symmetric_difference()`将在两个输入范围中找到的元素中不在两者中都找到的元素复制到输出范围。如果一个元素在第一个范围中出现*M*次，在第二个范围中出现*N*次，则如果*M > N*，则将第一个范围中的最后*M-N*个元素复制到输出范围中，否则将第二个范围中的最后*N-M*个元素复制到输出范围中：

```cpp
        std::vector<Task> v7;
        std::set_symmetric_difference(v1.cbegin(), v1.cend(),
                                      v2.cbegin(), v2.cend(),
                                      std::back_inserter(v7));
        // v7 = {{10, "Task 1.1"},{20, "Task 1.3"},{20, "Task 1.4"}
        //       {30, "Task 2.3"},{30, "Task 2.4"},{40, "Task 2.5"}}
```

另一方面，`std::includes()`不会产生输出范围；它只检查第二个范围是否包含在第一个范围中。如果第二个范围为空或其所有元素都包含在第一个范围中，则返回`true`；否则返回`false`。它还有两个重载，其中一个指定比较二进制函数对象。

# 另请参阅

+   *将向量用作默认容器*

+   *对范围进行排序*

+   *初始化范围*

+   *使用迭代器在容器中插入新元素*

+   *在范围中查找元素*

# 使用迭代器在容器中插入新元素

在使用容器时，通常有必要在开头、结尾或中间某处插入新元素。有一些算法，比如我们在前面的食谱中看到的那些*在范围上使用集合操作*，需要一个范围的迭代器来插入，但如果你简单地传递一个迭代器，比如`begin()`返回的迭代器，它不会插入，而是覆盖容器的元素。此外，使用`end()`返回的迭代器无法在末尾插入。为了执行这样的操作，标准库提供了一组迭代器和迭代器适配器，使这些情况成为可能。

# 准备就绪

本食谱中讨论的迭代器和适配器在`<iterator>`头文件中的`std`命名空间中可用。如果包括诸如`<algorithm>`之类的头文件，则不必显式包括`<iterator>`。

# 如何做到...

使用以下迭代器适配器在容器中插入新元素：

+   `std::back_inserter()`用于在末尾插入元素，适用于具有`push_back()`方法的容器：

```cpp
        std::vector<int> v{ 1,2,3,4,5 };
        std::fill_n(std::back_inserter(v), 3, 0);
        // v={1,2,3,4,5,0,0,0}
```

+   `std::front_inserter()`用于在开头插入元素，适用于具有`push_front()`方法的容器：

```cpp
        std::list<int> l{ 1,2,3,4,5 };
        std::fill_n(std::front_inserter(l), 3, 0);
        // l={0,0,0,1,2,3,4,5}
```

+   `std::inserter()`用于在容器中的任何位置插入，适用于具有`insert()`方法的容器：

```cpp
        std::vector<int> v{ 1,2,3,4,5 };
        std::fill_n(std::inserter(v, v.begin()), 3, 0);
        // v={0,0,0,1,2,3,4,5}

        std::list<int> l{ 1,2,3,4,5 };
        auto it = l.begin();
        std::advance(it, 3);
        std::fill_n(std::inserter(l, it), 3, 0);
        // l={1,2,3,0,0,0,4,5}
```

# 工作原理...

`std::back_inserter()`、`std::front_inserter()`和`std::inserter()`都是创建类型为`std::back_insert_iterator`、`std::front_insert_iterator`和`std::insert_iterator`的迭代器适配器的辅助函数。这些都是输出迭代器，用于向它们构造的容器追加、前置或插入。增加和取消引用这些迭代器不会做任何事情。但是，在赋值时，这些迭代器调用容器的以下方法：

+   `std::back_insterter_iterator`调用`push_back()`

+   `std::front_inserter_iterator`调用`push_front()`

+   `std::insert_iterator`调用`insert()`

以下是`std::back_inserter_iterator`的过度简化实现：

```cpp
    template<class C>
    class back_insert_iterator {
    public:
      typedef back_insert_iterator<C> T;
      typedef typename C::value_type V;

      explicit back_insert_iterator( C& c ) :container( &c ) { }

      T& operator=( const V& val ) { 
        container->push_back( val );
        return *this;
      }

      T& operator*() { return *this; }

      T& operator++() { return *this; }

      T& operator++( int ) { return *this; }
      protected:
      C* container;
    };
```

由于赋值运算符的工作方式，这些迭代器只能与一些标准容器一起使用：

+   `std::back_insert_iterator`可以与`std::vector`、`std::list`、`std::deque`和`std::basic_string`一起使用。

+   `std::front_insert_iterator`可与`std::list`、`std::forward_list`和`std:deque`一起使用。

+   `std::insert_iterator`可以与所有标准容器一起使用。

以下示例在`std::vector`的开头插入了三个值为 0 的元素：

```cpp
    std::vector<int> v{ 1,2,3,4,5 };
    std::fill_n(std::inserter(v, v.begin()), 3, 0);
    // v={0,0,0,1,2,3,4,5}
```

`std::inserter()`适配器接受两个参数：容器和元素应该插入的迭代器。在容器上调用`insert()`时，`std::insert_iterator`会增加迭代器，因此在再次分配时，它可以在下一个位置插入一个新元素。以下是为这个迭代器适配器实现的赋值运算符：

```cpp
    T& operator=(const V& v)
    {  
      iter = container->insert(iter, v);
      ++iter;
      return (*this);
    }
```

# 还有更多...

这些迭代器适配器旨在与将多个元素插入范围的算法或函数一起使用。当然，它们也可以用于插入单个元素，但在这种情况下，只需调用`push_back()`、`push_front()`或`insert()`就更简单和直观了。应避免以下示例：

```cpp
    std::vector<int> v{ 1,2,3,4,5 };
    *std::back_inserter(v) = 6; // v = {1,2,3,4,5,6}

    std::back_insert_iterator<std::vector<int>> it(v);
    *it = 7;                    // v = {1,2,3,4,5,6,7}
```

# 另请参阅

+   *在范围上使用集合操作*

# 编写自己的随机访问迭代器

在第八章中，*学习现代核心语言特性*，我们看到了如何通过实现迭代器和自由的`begin()`和`end()`函数来启用自定义类型的范围-based for 循环，以返回自定义范围的第一个和最后一个元素的迭代器。您可能已经注意到，在该示例中提供的最小迭代器实现不符合标准迭代器的要求，因为它不能被复制构造或分配，也不能被递增。在这个示例中，我们将建立在这个示例的基础上，展示如何创建一个满足所有要求的随机访问迭代器。

# 准备工作

对于这个示例，您应该了解标准定义的迭代器类型及其不同之处。它们的要求的很好的概述可以在[`www.cplusplus.com/reference/iterator/`](http://www.cplusplus.com/reference/iterator/)上找到。

为了举例说明如何编写随机访问迭代器，我们将考虑在第八章的*为自定义类型启用基于范围的 for 循环*示例中使用的`dummy_array`类的变体，这是一个非常简单的数组概念，除了作为演示迭代器的代码库之外，没有实际价值：

```cpp
    template <typename Type, size_t const SIZE>
    class dummy_array
    {
      Type data[SIZE] = {};
    public:
      Type& operator[](size_t const index)
      {
        if (index < SIZE) return data[index];
        throw std::out_of_range("index out of range");
      }

     Type const & operator[](size_t const index) const
     {
       if (index < SIZE) return data[index];
       throw std::out_of_range("index out of range");
     }

      size_t size() const { return SIZE; }
    };
```

下一节中显示的所有代码，迭代器类、`typedef`和`begin()`和`end()`函数，都将成为这个类的一部分。

# 如何做...

为了为前面部分显示的`dummy_array`类提供可变和常量随机访问迭代器，将以下成员添加到类中：

+   迭代器类模板，它是用元素的类型和数组的大小参数化的。该类必须有以下公共的`typedef`，定义标准的同义词：

```cpp
        template <typename T, size_t const Size>
        class dummy_array_iterator
        {
        public:
          typedef dummy_array_iterator            self_type;
          typedef T                               value_type;
          typedef T&                              reference;
          typedef T*                              pointer;
          typedef std::random_access_iterator_tag iterator_category;
          typedef ptrdiff_t                       difference_type;
        };
```

+   迭代器类的私有成员：指向数组数据的指针和数组中的当前索引：

```cpp
        private:
           pointer ptr = nullptr;
           size_t index = 0;
```

+   迭代器类的私有方法，用于检查两个迭代器实例是否指向相同的数组数据：

```cpp
        private:
          bool compatible(self_type const & other) const
          {
            return ptr == other.ptr;
          }
```

+   迭代器类的显式构造函数：

```cpp
        public:
           explicit dummy_array_iterator(pointer ptr, 
                                         size_t const index) 
             : ptr(ptr), index(index) { }
```

+   迭代器类成员以满足所有迭代器的通用要求：可复制构造，可复制分配，可销毁，前缀和后缀可递增。在这个实现中，后递增运算符是根据前递增运算符实现的，以避免代码重复：

```cpp
        dummy_array_iterator(dummy_array_iterator const & o) 
           = default;
        dummy_array_iterator& operator=(dummy_array_iterator const & o) 
           = default;
        ~dummy_array_iterator() = default;

        self_type & operator++ ()
        {
           if (index >= Size) 
             throw std::out_of_range("Iterator cannot be incremented past 
                                      the end of range.");
          ++index;
          return *this;
        }

        self_type operator++ (int)
        {
          self_type tmp = *this;
          ++*this;
          return tmp;
        }
```

+   迭代器类成员以满足输入迭代器要求：测试相等/不相等，作为右值解引用：

```cpp
        bool operator== (self_type const & other) const
        {
          assert(compatible(other));
          return index == other.index;
        }

        bool operator!= (self_type const & other) const
        {
          return !(*this == other);
        }

        reference operator* () const
        {
          if (ptr == nullptr)
            throw std::bad_function_call();
          return *(ptr + index);
        }

        reference operator-> () const
        {
          if (ptr == nullptr)
            throw std::bad_function_call();
          return *(ptr + index);
        }
```

+   迭代器类成员以满足前向迭代器要求：默认可构造：

```cpp
        dummy_array_iterator() = default;
```

+   迭代器类成员以满足双向迭代器要求：可递减：

```cpp
        self_type & operator--()
        {
          if (index <= 0) 
            throw std::out_of_range("Iterator cannot be decremented 
                                     past the end of range.");
          --index;
          return *this;
        }

        self_type operator--(int)
        {
          self_type tmp = *this;
          --*this;
          return tmp;
        }
```

+   迭代器类成员以满足随机访问迭代器要求：算术加和减，与其他迭代器不相等的可比性，复合赋值，和偏移解引用：

```cpp
        self_type operator+(difference_type offset) const
        {
          self_type tmp = *this;
          return tmp += offset;
        }

        self_type operator-(difference_type offset) const
        {
          self_type tmp = *this;
          return tmp -= offset;
        }

        difference_type operator-(self_type const & other) const
        {
          assert(compatible(other));
          return (index - other.index);
        }

        bool operator<(self_type const & other) const
        {
          assert(compatible(other));
          return index < other.index;
        }

        bool operator>(self_type const & other) const
        {
          return other < *this;
        }

        bool operator<=(self_type const & other) const
        {
          return !(other < *this);
        }

        bool operator>=(self_type const & other) const
        {
          return !(*this < other);
        }

        self_type & operator+=(difference_type const offset)
        {
          if (index + offset < 0 || index + offset > Size)
            throw std::out_of_range("Iterator cannot be incremented 
                                     past the end of range.");
          index += offset;
          return *this;
        }

        self_type & operator-=(difference_type const offset)
        {
          return *this += -offset;
        }

        value_type & operator[](difference_type const offset)
        {
          return (*(*this + offset));
        }

        value_type const & operator[](difference_type const offset) const
        {
          return (*(*this + offset));
        }
```

+   为`dummy_array`类添加可变和常量迭代器的`typedef`：

```cpp
        public:
           typedef dummy_array_iterator<Type, SIZE> 
                   iterator;
           typedef dummy_array_iterator<Type const, SIZE> 
                   constant_iterator;
```

+   添加公共的`begin()`和`end()`函数到`dummy_array`类中，以返回数组中第一个和最后一个元素的迭代器：

```cpp
        iterator begin() 
        {
          return iterator(data, 0);
        }

        iterator end()
        {
          return iterator(data, SIZE);
        }

        constant_iterator begin() const
        {
          return constant_iterator(data, 0);
        }

        constant_iterator end() const
        {
          return constant_iterator(data, SIZE);
        }
```

# 它是如何工作的...

标准库定义了五种迭代器类别：

+   *输入迭代器*：这是最简单的类别，仅保证单遍历顺序算法的有效性。增加后，之前的副本可能会变得无效。

+   *输出迭代器*：这些基本上是可以用来写入指定元素的输入迭代器。

+   *前向迭代器*：这些可以读取（和写入）指定元素的数据。它们满足输入迭代器的要求，并且此外，必须支持默认构造，并且必须支持多遍历场景而不使之前的副本无效。

+   *双向迭代器*：这些是前向迭代器，此外，还支持递减，因此可以向两个方向移动。

+   *随机访问迭代器*：这些支持在常数时间内访问容器中的任何元素。它们实现了双向迭代器的所有要求，并且还支持算术运算`+`和`-`，复合赋值`+=`和`-=`，与其他迭代器的比较`<`，`<=`，`>`，`>=`，以及偏移解引用运算符。

还实现了输出迭代器要求的前向、双向和随机访问迭代器称为*可变迭代器*。

在前一节中，我们看到了如何实现随机访问迭代器，逐步介绍了每个迭代器类别的要求（因为每个迭代器类别包括前一个迭代器类别的要求并添加新的要求）。迭代器类模板对于常量和可变迭代器是通用的，我们定义了两个同义词，称为`iterator`和`constant_iterator`。

在实现内部迭代器类模板之后，我们还定义了`begin()`和`end()`成员函数，返回数组中第一个和最后一个元素的迭代器。这些方法有重载，根据`dummy_array`类实例是可变的还是常量的，返回可变或常量迭代器。

有了`dummy_array`类及其迭代器的这种实现，我们可以编写以下示例。有关更多示例，请查看本书附带的源代码：

```cpp
    dummy_array<int, 3> a;
    a[0] = 10;
    a[1] = 20;
    a[2] = 30;

    std::transform(a.begin(), a.end(), a.begin(), 
                   [](int const e) {return e * 2; });

    for (auto&& e : a) std::cout << e << std::endl;

    auto lp = [](dummy_array<int, 3> const & ca)
    {
      for (auto const & e : ca) 
        std::cout << e << std::endl;
    };

    lp(a);

    dummy_array<std::unique_ptr<Tag>, 3> ta;
    ta[0] = std::make_unique<Tag>(1, "Tag 1");
    ta[1] = std::make_unique<Tag>(2, "Tag 2");
    ta[2] = std::make_unique<Tag>(3, "Tag 3");

    for (auto it = ta.begin(); it != ta.end(); ++it)
      std::cout << it->id << " " << it->name << std::endl;
```

# 还有更多...

除了`begin()`和`end()`之外，容器可能还有其他方法，例如`cbegin()`/`cend()`（用于常量迭代器），`rbegin()`/`rend()`（用于可变反向迭代器），以及`crbegin()`/`crend()`（用于常量反向迭代器）。实现这一点留作练习给你。

另一方面，在现代 C++中，返回第一个和最后一个迭代器的这些函数不必是成员函数，而可以作为非成员函数提供。实际上，这是下一个配方的主题，*使用非成员函数访问容器*。

# 另请参阅

+   第八章的*学习现代核心语言特性*配方中的为自定义类型启用基于范围的 for 循环

+   第八章的*学习现代核心语言特性*配方中的创建类型别名和别名模板

# 使用非成员函数访问容器

标准容器提供了`begin()`和`end()`成员函数，用于检索容器的第一个和最后一个元素的迭代器。实际上有四组这样的函数。除了`begin()`/`end()`，容器还提供了`cbegin()`/`cend()`来返回常量迭代器，`rbegin()`/`rend()`来返回可变的反向迭代器，以及`crbegin()`/`crend()`来返回常量反向迭代器。在 C++11/C++14 中，所有这些都有非成员等价物，可以与标准容器、类 C 数组和任何专门化它们的自定义类型一起使用。在 C++17 中，甚至添加了更多的非成员函数；`std::data()`--返回指向包含容器元素的内存块的指针，`std::size()`--返回容器或数组的大小，`std::empty()`--返回给定容器是否为空。这些非成员函数用于通用代码，但可以在代码的任何地方使用。

# 准备工作

在这个配方中，我们将以我们在上一个配方中实现的`dummy_array`类及其迭代器为例。在继续本配方之前，您应该先阅读那个配方。

非成员`begin()`/`end()`函数和其他变体，以及非成员`data()`、`size()`和`empty()`在`std`命名空间中的`<iterator>`头文件中可用，该头文件隐式地包含在以下任何一个头文件中：`<array>`、`<deque>`、`<forward_list>`、`<list>`、`<map>`、`<regex>`、`<set>`、`<string>`、`<unordered_map>`、`<unordered_set>`和`<vector>`。

在这个配方中，我们将提到`std::begin()`/`std::end()`函数，但讨论的一切也适用于其他函数：`std::cbegin()`/`std::cend()`、`std::rbegin()`/`std::rend()`和`std::crbegin()`/`std::crend()`。

# 如何做...

使用非成员`std::begin()`/`std::end()`函数和其他变体，以及`std::data()`、`std::size()`和`std::empty()`与：

+   标准容器：

```cpp
        std::vector<int> v1{ 1, 2, 3, 4, 5 };
        auto sv1 = std::size(v1);  // sv1 = 5
        auto ev1 = std::empty(v1); // ev1 = false
        auto dv1 = std::data(v1);  // dv1 = v1.data()
        for (auto i = std::begin(v1); i != std::end(v1); ++i)
          std::cout << *i << std::endl;

        std::vector<int> v2;
        std::copy(std::cbegin(v1), std::cend(v1),
                  std::back_inserter(v2));
```

+   （类似 C 的）数组：

```cpp
        int a[5] = { 1, 2, 3, 4, 5 };
        auto pos = std::find_if(std::crbegin(a), std::crend(a), 
                                [](int const n) {return n % 2 == 0; });
        auto sa = std::size(a);  // sa = 5
        auto ea = std::empty(a); // ea = false
        auto da = std::data(a);  // da = a
```

+   提供相应成员函数`begin()`/`end()`、`data()`、`empty()`或`size()`的自定义类型：

```cpp
        dummy_array<std::string, 5> sa;
        dummy_array<int, 5> sb;
        sa[0] = "1"s;
        sa[1] = "2"s;
        sa[2] = "3"s;
        sa[3] = "4"s;
        sa[4] = "5"s;

        std::transform(
          std::begin(sa), std::end(sa), 
          std::begin(sb), 
          [](std::string const & s) {return std::stoi(s); });
        // sb = [1, 2, 3, 4, 5]

        auto sa_size = std::size(sa); // sa_size = 5
```

+   类型未知的通用代码：

```cpp
        template <typename F, typename C>
        void process(F&& f, C const & c)
        {
          std::for_each(std::begin(c), std::end(c), 
                        std::forward<F>(f));
        }

        auto l = [](auto const e) {std::cout << e << std::endl; };

        process(l, v1); // std::vector<int>
        process(l, a);  // int[5]
        process(l, sa); // dummy_array<std::string, 5>
```

# 工作原理...

这些非成员函数是在不同版本的标准中引入的，但它们在 C++17 中都被修改为返回`constexpr auto`：

+   C++11 中的`std::begin()`和`std::end()`

+   `std::cbegin()`/`std::cend()`，`std::rbegin()`/`std::rend()`和`std::crbegin()`/`std::crend()`在 C++14 中

+   C++17 中的`std::data()`、`std::size()`和`std::empty()`

`begin()`/`end()`函数族有容器类和数组的重载，它们所做的只是：

+   返回调用容器对应成员函数的结果。

+   返回数组的第一个或最后一个元素的指针。

`std::begin()`/`std::end()`的实际典型实现如下：

```cpp
    template<class C>
    constexpr auto inline begin(C& c) -> decltype(c.begin())
    {
      return c.begin();
    }
    template<class C>
    constexpr auto inline end(C& c) -> decltype(c.end())
    {
      return c.end();
    }

    template<class T, std::size_t N>
    constexpr T* inline begin(T (&array)[N])
    {
      return array;
    }

    template<class T, std::size_t N>
    constexpr T* inline begin(T (&array)[N])
    {
      return array+N;
    }
```

可以为没有相应的`begin()`/`end()`成员但仍可迭代的容器提供自定义专门化。标准库实际上为`std::initializer_list`和`std::valarray`提供了这样的专门化。

必须在定义原始类或函数模板的相同命名空间中定义专门化。因此，如果要专门化任何`std::begin()`/`std::end()`对，必须在`std`命名空间中执行。

C++17 中引入的用于容器访问的其他非成员函数也有几个重载：

+   `std::data()`有几个重载；对于类`C`，它返回`c.data()`，对于数组，它返回数组，对于`std::initializer_list<T>`，它返回`il.begin()`。

```cpp
        template <class C> 
        constexpr auto data(C& c) -> decltype(c.data())
        {
          return c.data();
        }

        template <class C> 
        constexpr auto data(const C& c) -> decltype(c.data())
        {
          return c.data();
        }

        template <class T, std::size_t N>
        constexpr T* data(T (&array)[N]) noexcept
        {
          return array;
        }

        template <class E> 
        constexpr const E* data(std::initializer_list<E> il) noexcept
        {
          return il.begin();
        }
```

+   `std::size()`有两个重载；对于类`C`，它返回`c.size()`，对于数组，它返回大小`N`。

```cpp
        template <class C> 
        constexpr auto size(const C& c) -> decltype(c.size())
        {
          return c.size();
        }

        template <class T, std::size_t N>
        constexpr std::size_t size(const T (&array)[N]) noexcept
        {
          return N;
        }
```

+   `std::empty()` 有几种重载形式；对于类 `C`，它返回 `c.empty()`，对于数组它返回 `false`，对于 `std::initializer_list<T>` 它返回 `il.size() == 0`。

```cpp
        template <class C> 
        constexpr auto empty(const C& c) -> decltype(c.empty())
        {
          return c.empty();
        }

        template <class T, std::size_t N> 
        constexpr bool empty(const T (&array)[N]) noexcept
        {
          return false;
        }

        template <class E> 
        constexpr bool empty(std::initializer_list<E> il) noexcept
        {
          return il.size() == 0;
        }
```

# 还有更多...

这些非成员函数主要用于模板代码，其中容器类型未知，可以是标准容器、类似 C 的数组或自定义类型。使用这些函数的非成员版本使我们能够编写更简单、更少的代码，可以处理所有这些类型的容器。

然而，使用这些函数并不应该局限于通用代码。虽然这更多是个人偏好的问题，但保持一致并在代码中的任何地方使用它们可能是一个好习惯。所有这些方法都有轻量级的实现，很可能会被编译器内联，这意味着与使用相应的成员函数相比，不会有任何额外开销。

# 另请参阅

+   *编写自己的随机访问迭代器*
