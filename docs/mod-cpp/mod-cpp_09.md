# 处理数字和字符串

本章包含的示例有：

+   在数字和字符串类型之间进行转换

+   数字类型的限制和其他属性

+   生成伪随机数

+   初始化伪随机数生成器的内部状态的所有位

+   使用原始字符串字面量来避免转义字符

+   创建熟练的用户定义字面量

+   创建原始用户定义字面量

+   创建字符串助手库

+   使用正则表达式验证字符串的格式

+   使用正则表达式解析字符串的内容

+   使用正则表达式替换字符串的内容

+   使用 string_view 而不是常量字符串引用

# 在数字和字符串类型之间进行转换

在数字和字符串类型之间进行转换是一种普遍的操作。在 C++11 之前，几乎没有支持将数字转换为字符串和反向转换的功能，开发人员大多需要使用不安全的类型函数，并通常编写自己的实用函数，以避免一遍又一遍地编写相同的代码。有了 C++11，标准库提供了用于在数字和字符串之间进行转换的实用函数。在本示例中，您将学习如何使用现代 C++标准函数在数字和字符串之间进行转换。

# 准备工作

本示例中提到的所有实用函数都位于`<string>`头文件中。

# 如何做到...

在需要在数字和字符串之间进行转换时，请使用以下标准转换函数：

+   要将整数或浮点类型转换为字符串类型，请使用`std::to_string()`或`std::to_wstring()`，如下面的代码片段所示：

```cpp
        auto si = std::to_string(42);      // si="42" 
        auto sl = std::to_string(42l);     // sl="42" 
        auto su = std::to_string(42u);     // su="42" 
        auto sd = std::to_wstring(42.0);   // sd=L"42.000000" 
        auto sld = std::to_wstring(42.0l); // sld=L"42.000000"
```

+   要将字符串类型转换为整数类型，请使用`std::stoi()`，`std::stol()`，`std::stoll()`，`std::stoul()`或`std::stoull()`；请参阅以下代码片段：

```cpp
        auto i1 = std::stoi("42");                 // i1 = 42 
        auto i2 = std::stoi("101010", nullptr, 2); // i2 = 42 
        auto i3 = std::stoi("052", nullptr, 8);    // i3 = 42 
        auto i4 = std::stoi("0x2A", nullptr, 16);  // i4 = 42
```

+   要将字符串类型转换为浮点类型，请使用`std::stof()`，`std::stod()`或`std::stold()`，如下面的代码片段所示：

```cpp
        // d1 = 123.45000000000000 
        auto d1 = std::stod("123.45"); 
        // d2 = 123.45000000000000 
        auto d2 = std::stod("1.2345e+2"); 
        // d3 = 123.44999980926514 
        auto d3 = std::stod("0xF.6E6666p3");
```

# 它的工作原理...

要在整数或浮点类型与字符串类型之间进行转换，可以使用`std::to_string()`或`std::to_wstring()`函数。这些函数位于`<string>`头文件中，并且对于有符号和无符号整数和实数类型都有重载。它们产生与调用适当格式说明符的`std::sprintf()`和`std::swprintf()`产生的相同结果。以下代码片段列出了这两个函数的所有重载。

```cpp
    std::string to_string(int value); 
    std::string to_string(long value); 
    std::string to_string(long long value); 
    std::string to_string(unsigned value); 
    std::string to_string(unsigned long value); 
    std::string to_string(unsigned long long value); 
    std::string to_string(float value); 
    std::string to_string(double value); 
    std::string to_string(long double value); 
    std::wstring to_wstring(int value); 
    std::wstring to_wstring(long value); 
    std::wstring to_wstring(long long value); 
    std::wstring to_wstring(unsigned value); 
    std::wstring to_wstring(unsigned long value); 
    std::wstring to_wstring(unsigned long long value); 
    std::wstring to_wstring(float value); 
    std::wstring to_wstring(double value); 
    std::wstring to_wstring(long double value);
```

在进行相反的转换时，有一整套函数，它们的名称格式为**ston**（**字符串到数字**），其中**n**代表**i**（`整数`），**l**（`长整型`），**ll**（`长长整型`），**ul**（`无符号长整型`）或**ull**（`无符号长长整型`）。以下清单显示了所有这些函数，每个函数都有两个重载，一个接受`std::string`，另一个接受`std::wstring`作为第一个参数：

```cpp
    int stoi(const std::string& str, std::size_t* pos = 0,  
             int base = 10); 
    int stoi(const std::wstring& str, std::size_t* pos = 0,  
             int base = 10); 
    long stol(const std::string& str, std::size_t* pos = 0,  
             int base = 10); 
    long stol(const std::wstring& str, std::size_t* pos = 0,  
             int base = 10); 
    long long stoll(const std::string& str, std::size_t* pos = 0,  
                    int base = 10); 
    long long stoll(const std::wstring& str, std::size_t* pos = 0,  
                    int base = 10); 
    unsigned long stoul(const std::string& str, std::size_t* pos = 0, 
                        int base = 10); 
    unsigned long stoul(const std::wstring& str, std::size_t* pos = 0,  
                        int base = 10); 
    unsigned long long stoull(const std::string& str,  
                              std::size_t* pos = 0, int base = 10); 
    unsigned long long stoull(const std::wstring& str,  
                              std::size_t* pos = 0, int base = 10); 
    float       stof(const std::string& str, std::size_t* pos = 0); 
    float       stof(const std::wstring& str, std::size_t* pos = 0); 
    double      stod(const std::string& str, std::size_t* pos = 0); 
    double      stod(const std::wstring& str, std::size_t* pos = 0); 
    long double stold(const std::string& str, std::size_t* pos = 0); 
    long double stold(const std::wstring& str, std::size_t* pos = 0);
```

字符串到整数类型函数的工作方式是在非空白字符之前丢弃所有空格，然后尽可能多地取字符以形成有符号或无符号数字（取决于情况），然后将其转换为请求的整数类型（`stoi()`将返回`整数`，`stoul()`将返回`无符号长整型`，依此类推）。在所有以下示例中，结果都是整数`42`，除了最后一个示例，结果是`-42`：

```cpp
    auto i1 = std::stoi("42");             // i1 = 42 
    auto i2 = std::stoi("   42");          // i2 = 42 
    auto i3 = std::stoi("   42fortytwo");  // i3 = 42 
    auto i4 = std::stoi("+42");            // i4 = 42 
    auto i5 = std::stoi("-42");            // i5 = -42
```

有效的整数可能由以下部分组成：

+   一个符号，加号（`+`）或减号（`-`）（可选）。

+   前缀`0`表示八进制基数（可选）。

+   前缀`0x`或`0X`表示十六进制基数（可选）。

+   一系列数字。

可选前缀`0`（表示八进制）仅在指定基数为`8`或`0`时应用。类似地，可选前缀`0x`或`0X`（表示十六进制）仅在指定基数为`16`或`0`时应用。

将字符串转换为整数的函数具有三个参数：

+   输入字符串。

+   一个指针，如果不为空，将接收处理的字符数，可以包括任何被丢弃的前导空格，符号和基数前缀，因此不应与整数值的数字数量混淆。

+   指示基数的数字；默认情况下为`10`。

输入字符串中的有效数字取决于基数。对于基数`2`，唯一有效的数字是`0`和`1`；对于基数`5`，它们是`01234`。对于基数`11`，有效数字是`0-9`和字符`A`和`a`。这一直持续到我们达到基数`36`，它具有有效字符`0-9`，`A-Z`和`a-z`。

以下是将各种基数的字符串转换为十进制整数的更多示例。同样，在所有情况下，结果要么是`42`，要么是`-42`：

```cpp
    auto i6 = std::stoi("052", nullptr, 8); 
    auto i7 = std::stoi("052", nullptr, 0); 
    auto i8 = std::stoi("0x2A", nullptr, 16); 
    auto i9 = std::stoi("0x2A", nullptr, 0); 
    auto i10 = std::stoi("101010", nullptr, 2); 
    auto i11 = std::stoi("22", nullptr, 20); 
    auto i12 = std::stoi("-22", nullptr, 20); 

    auto pos = size_t{ 0 }; 
    auto i13 = std::stoi("42", &pos);      // pos = 2 
    auto i14 = std::stoi("-42", &pos);     // pos = 3 
    auto i15 = std::stoi("  +42dec", &pos);// pos = 5
```

需要注意的一点是，如果转换失败，这些转换函数会抛出异常。可以抛出两种异常：

+   `std::invalid_argument`：如果无法执行转换：

```cpp
        try 
        { 
           auto i16 = std::stoi(""); 
        } 
        catch (std::exception const & e) 
        { 
           // prints "invalid stoi argument" 
           std::cout << e.what() << std::endl; 
        }
```

+   `std::out_of_range`：如果转换后的值超出了结果类型的范围（或者如果底层函数将`errno`设置为`ERANGE`）：

```cpp
        try 
        { 
           // OK
           auto i17 = std::stoll("12345678901234");  
           // throws std::out_of_range 
           auto i18 = std::stoi("12345678901234"); 
        } 
        catch (std::exception const & e) 
        { 
           // prints "stoi argument out of range"
           std::cout << e.what() << std::endl; 
        }
```

将字符串转换为浮点类型的另一组函数非常相似，只是它们没有用于数字基数的参数。有效的浮点值可以在输入字符串中有不同的表示：

+   十进制浮点表达式（可选符号，带有可选小数点的十进制数字序列，可选的`e`或`E`后跟带有可选符号的指数）。

+   二进制浮点表达式（可选符号，`0x`或`0X`前缀，带有可选小数点的十六进制数字序列，可选的`p`或`P`后跟带有可选符号的指数）。

+   无穷大表达式（可选符号后跟不区分大小写的`INF`或`INFINITY`）。

+   非数字表达式（可选符号后跟不区分大小写的`NAN`和可能的其他字母数字字符）。

以下是将字符串转换为双精度浮点数的各种示例：

```cpp
    auto d1 = std::stod("123.45");         // d1 =  123.45000000000000 
    auto d2 = std::stod("+123.45");        // d2 =  123.45000000000000 
    auto d3 = std::stod("-123.45");        // d3 = -123.45000000000000 
    auto d4 = std::stod("  123.45");       // d4 =  123.45000000000000 
    auto d5 = std::stod("  -123.45abc");   // d5 = -123.45000000000000 
    auto d6 = std::stod("1.2345e+2");      // d6 =  123.45000000000000 
    auto d7 = std::stod("0xF.6E6666p3");   // d7 =  123.44999980926514 

    auto d8 = std::stod("INF");            // d8 = inf 
    auto d9 = std::stod("-infinity");      // d9 = -inf 
    auto d10 = std::stod("NAN");           // d10 = nan 
    auto d11 = std::stod("-nanabc");       // d11 = -nan
```

之前看到的浮点基数 2 科学计数法，以`0xF.6E6666p3`的形式出现，不是本篇文章的主题。但是，为了清楚起见，提供了一个简短的描述；尽管建议您查看其他参考资料以获取详细信息。基数 2 科学计数法中的浮点常数由几个部分组成：

+   十六进制前缀`0x`。

+   一个整数部分，在这个例子中是`F`，在十进制中是 15。

+   一个小数部分，在这个例子中是`6E6666`，或者用二进制表示为`011011100110011001100110`。要将其转换为十进制，我们需要加上二的倒数幂：`1/4 + 1/8 + 1/32 + 1/64 + 1/128 + ...`。

+   一个后缀，表示 2 的幂；在这个例子中，`p3`表示 2 的 3 次幂。

十进制等价值的值由乘以有效数字（由整数和小数部分组成）和基数的幂决定。对于给定的十六进制基数 2 浮点文字，有效数字是`15.4312499...`（注意第七位后的数字没有显示），基数是 2，指数是 3。因此，结果是`15.4212499... * 8`，即`123.44999980926514`。

# 另请参阅

+   **数字类型的限制和其他属性**

# 数字类型的限制和其他属性

有时，有必要知道和使用数值类型表示的最小和最大值，比如`char`、`int`或`double`。许多开发人员在这方面使用标准 C 宏，如`CHAR_MIN`/`CHAR_MAX`、`INT_MIN`/`INT_MAX`或`DBL_MIN`/`DBL_MAX`。C++提供了一个名为`numeric_limits`的类模板，为每种数值类型提供了特化，使您能够查询类型的最小和最大值，但不仅限于此，并提供了用于查询类型属性的其他常量，例如类型是否有符号，它需要多少位来表示其值，对于浮点类型是否可以表示无穷大等。在 C++11 之前，`numeric_limits<T>`的使用是有限的，因为它不能在需要常量的地方使用（例如数组的大小和 switch case）。因此，开发人员更喜欢在他们的代码中使用 C 宏。在 C++11 中，情况已经不再是这样了，因为`numeric_limits<T>`的所有静态成员现在都是`constexpr`，这意味着它们可以在需要常量表达式的所有地方使用。

# 准备工作

`numeric_limits<T>`类模板在`<limits>`头文件中的`std`命名空间中可用。

# 如何做...

使用`std::numeric_limits<T>`来查询数值类型`T`的各种属性：

+   使用`min()`和`max()`静态方法来获取类型的最小和最大有限数：

```cpp
        template<typename T, typename I> 
        T minimum(I const start, I const end) 
        { 
          T minval = std::numeric_limits<T>::max(); 
          for (auto i = start; i < end; ++i) 
          { 
            if (*i < minval) 
              minval = *i; 
          } 
          return minval; 
        } 

        int range[std::numeric_limits<char>::max() + 1] = { 0 }; 

        switch(get_value()) 
        { 
          case std::numeric_limits<int>::min(): 
          break; 
        }
```

+   使用其他静态方法和静态常量来检索数值类型的其他属性：

```cpp
        auto n = 42; 
        std::bitset<std::numeric_limits<decltype(n)>::digits>  
          bits { static_cast<unsigned long long>(n) };
```

在 C++11 中，`std::numeric_limits<T>`的使用没有限制；因此，在现代 C++代码中最好使用它而不是 C 宏。

# 它是如何工作的...

`std::numeric_limits<T>`是一个类模板，使开发人员能够查询数值类型的属性。实际值可以通过特化获得，并且标准库为所有内置数值类型（`char`、`short`、`int`、`long`、`float`、`double`等）提供了特化。此外，第三方可能为其他类型提供额外的实现。例如，一个数值库可能实现了`bigint`整数类型和`decimal`类型，并为这些类型提供了`numeric_limits`的特化（如`numeric_limits<bigint>`和`numeric_limits<decimal>`）。

以下数值类型的特化在`<limits>`头文件中可用。请注意，`char16_t`和`char32_t`的特化是 C++11 中的新内容；其他的在此之前就已经可用了。除了列出的特化之外，该库还包括了这些数值类型的每个`cv-qualified`版本的特化，并且它们与未经修饰的特化相同。例如，考虑类型`int`；有四个实际的特化（它们是相同的）：`numeric_limits<int>`、`numeric_limits<const int>`、`numeric_limits<volatile int>`和`numeric_limits<const volatile int>`：

```cpp
    template<> class numeric_limits<bool>; 
    template<> class numeric_limits<char>; 
    template<> class numeric_limits<signed char>; 
    template<> class numeric_limits<unsigned char>; 
template<> class numeric_limits<wchar_t>; 
    template<> class numeric_limits<char16_t>; 
    template<> class numeric_limits<char32_t>; 
    template<> class numeric_limits<short>; 
    template<> class numeric_limits<unsigned short>; 
    template<> class numeric_limits<int>; 
    template<> class numeric_limits<unsigned int>; 
    template<> class numeric_limits<long>; 
    template<> class numeric_limits<unsigned long>; 
    template<> class numeric_limits<long long>; 
    template<> class numeric_limits<unsigned long long>; 
    template<> class numeric_limits<float>; 
    template<> class numeric_limits<double>; 
    template<> class numeric_limits<long double>;
```

如前所述，在 C++11 中，`numeric_limits`的所有静态成员都是`constexpr`，这意味着它们可以在需要常量表达式的所有地方使用。它们比 C++宏有几个主要优势：

+   它们更容易记住，因为你唯一需要知道的是你本来就应该知道的类型的名称，而不是无数的宏名称。

+   它们支持在 C 中不可用的类型，比如`char16_t`和`char32_t`。

+   它们是你不知道类型的模板的唯一可能的解决方案。

+   最小值和最大值只是它提供的各种类型属性中的两个；因此，它的实际用途超出了数值限制。顺便说一句，因此，这个类可能本应该被称为`numeric_properties`，而不是`numeric_limits`。

以下函数模板`print_type_properties()`打印类型的最小和最大有限值以及其他信息：

```cpp
    template <typename T> 
    void print_type_properties() 
    { 
      std::cout  
        << "min="  
        << std::numeric_limits<T>::min()        << std::endl 
        << "max=" 
        << std::numeric_limits<T>::max()        << std::endl 
        << "bits=" 
        << std::numeric_limits<T>::digits       << std::endl 
        << "decdigits=" 
        << std::numeric_limits<T>::digits10     << std::endl 
        << "integral=" 
        << std::numeric_limits<T>::is_integer   << std::endl 
        << "signed=" 
        << std::numeric_limits<T>::is_signed    << std::endl 
        << "exact=" 
        << std::numeric_limits<T>::is_exact     << std::endl 
        << "infinity=" 
        << std::numeric_limits<T>::has_infinity << std::endl; 
    }
```

如果我们为无符号`short`，`int`和`double`调用`print_type_properties()`函数，将得到以下输出：

| `unsigned short` | `int` | `double` |
| --- | --- | --- |
| min=0max=65535bits=16decdigits=4integral=1signed=0exact=1infinity=0 | min=-2147483648max=2147483647bits=31decdigits=9integral=1signed=1exact=1infinity=0 | min=2.22507e-308max=1.79769e+308bits=53decdigits=15integral=0signed=1exact=0infinity=1 |

需要注意的一点是`digits`和`digits10`常量之间的区别：

+   `digits`表示整数类型的位数（如果有符号位，则不包括符号位，如果有填充位，则包括填充位），浮点类型的尾数位数。

+   `digits10`是类型可以表示的十进制数字的数量，而不需要更改。为了更好地理解这一点，让我们考虑一下`unsigned short`的情况。这是一个 16 位整数类型。它可以表示 0 到 65536 之间的数字。它可以表示多达五位十进制数字，即从 10,000 到 65,536，但它不能表示所有五位十进制数字，因为从 65,537 到 99,999 的数字需要更多的位。因此，它可以表示的最大数字而不需要更多位的是四位十进制数字（从 1,000 到 9,999）。这是`digits10`指示的值。对于整数类型，它与常量`digits`有直接关系；对于整数类型`T`，`digits10`的值为`std::numeric_limits<T>::digits * std::log10(2)`。

# 生成伪随机数

生成随机数对于各种应用程序都是必要的，从游戏到密码学，从抽样到预测。然而，“随机数”这个术语实际上并不正确，因为通过数学公式生成数字是确定性的，不会产生真正的随机数，而是看起来随机的数字，称为“伪随机”。真正的随机性只能通过基于物理过程的硬件设备实现，即使这也可能受到质疑，因为人们甚至可能认为宇宙实际上是确定性的。现代 C++提供了通过包含数字生成器和分布的伪随机数库来生成伪随机数的支持。从理论上讲，它也可以产生真正的随机数，但在实践中，这些实际上可能只是伪随机数。

# 准备工作

在这个示例中，我们讨论了生成伪随机数的标准支持。理解随机和伪随机数之间的区别是关键。另一方面，熟悉各种统计分布也是一个优势。然而，你必须知道均匀分布是什么，因为库中的所有引擎都产生均匀分布的数字。

# 如何做...

要在应用程序中生成伪随机数，应执行以下步骤：

1.  包含头文件`<random>`：

```cpp
        #include <random>
```

1.  使用`std::random_device`生成器来为伪随机引擎提供种子：

```cpp
        std::random_device rd{};
```

1.  使用可用的引擎之一生成数字并用随机种子初始化它：

```cpp
        auto mtgen = std::mt19937{ rd() };
```

1.  使用可用的分布之一将引擎的输出转换为所需的统计分布之一：

```cpp
        auto ud = std::uniform_int_distribution<>{ 1, 6 };
```

1.  生成伪随机数：

```cpp
        for(auto i = 0; i < 20; ++i) 
          auto number = ud(mtgen);
```

# 它是如何工作的...

伪随机数库包含两种类型的组件：

+   *引擎*是随机数的生成器；这些可以产生具有均匀分布的伪随机数，或者如果可用，实际随机数。

+   *分布*将引擎的输出转换为统计分布。

所有引擎（除了`random_device`）都以均匀分布产生整数，所有引擎都实现以下方法：

+   `min()`: 这是一个静态方法，返回生成器可以产生的最小值。

+   `max()`: 这是一个静态方法，返回生成器可以产生的最大值。

+   `seed()`: 用起始值初始化算法（除了 `random_device`，它不能被种子化）。

+   `operator()`: 生成一个在 `min()` 和 `max()` 之间均匀分布的新数字。

+   `discard()`: 生成并丢弃给定数量的伪随机数。

以下引擎可用：

+   线性同余引擎：这是一个使用以下公式产生数字的线性同余生成器：

*x(i) = (A * x(i-1) + C) mod M*

+   mersenne_twister_engine：这是一个 Mersenne twister 生成器，保留了 *W * (N-1) * R* 位的值；每次需要生成一个数字时，它提取 *W* 位。当所有位都被使用时，它通过移位和混合位来扭转大值，以便它有一个新的位组来提取。

+   subtract_with_carry_engine：这是一个基于以下公式实现 *减去进位* 算法的生成器：

*x(i) = (x(i - R) - x(i - S) - cy(i - 1)) mod M*

在上述公式中，*cy* 定义为：

*cy(i) = x(i - S) - x(i - R) - cy(i - 1) < 0 ? 1 : 0*

此外，该库还提供了引擎适配器，它们也是包装另一个引擎并基于基础引擎的输出生成数字的引擎。引擎适配器实现了前面提到的基础引擎的相同方法。以下引擎适配器可用：

+   discard_block_engine：从基础引擎生成的每个 P 个数字块中仅保留 R 个数字，丢弃其余数字。

+   independent_bits_engine：生成具有与基础引擎不同位数的数字的生成器。

+   shuffle_order_engine：保持基础引擎生成的 K 个数字的洗牌表，并从该表返回数字，用基础引擎生成的数字替换它们。

所有这些引擎和引擎适配器都产生伪随机数。然而，该库还提供了另一个称为 `random_device` 的引擎，它应该产生非确定性数字，但这并不是一个实际的约束，因为可能没有随机熵的物理来源。因此，`random_device` 的实现实际上可能基于伪随机引擎。`random_device` 类不能像其他引擎一样进行种子化，并且具有一个额外的名为 `entropy()` 的方法，返回随机设备的熵，对于确定性生成器为 0，对于非确定性生成器为非零。然而，这并不是确定设备实际上是确定性还是非确定性的可靠方法。例如，GNU `libstdc++` 和 LLVM `libc++` 实现了一个非确定性设备，但对于熵返回 `0`。另一方面，`VC++` 和 `boost.random` 对于熵分别返回 `32` 和 `10`。

所有这些生成器产生均匀分布的整数。然而，这只是大多数应用程序中需要的许多可能统计分布中的一个。为了能够以其他分布（整数或实数）产生数字，该库提供了几个称为 *分布* 的类，它们根据它们实现的统计分布将引擎的输出转换为数字。以下分布可用：

| **类型** | **类名** | **数字** | **统计分布** |
| --- | --- | --- | --- |
| 均匀 | 均匀整数分布 | 整数 | 均匀 |
| 均匀实数分布 | 实数 | 均匀 |
| 伯努利 | 伯努利分布 | 布尔 | 伯努利 |
| 二项式 | 二项分布 | 整数 | 二项式 |
| 负二项式 | 负二项分布 | 整数 | 负二项式 |
| 几何分布 | 整数 | 几何 |
| 泊松 | 泊松分布 | 整数 | 泊松 |
| 指数 | 指数分布 | 实数 | 指数 |
| 伽玛 | 伽玛分布 | 实数 | 伽玛 |
| 威布尔 | 威布尔分布 | 实数 | 威布尔 |
| 极值分布 | 实数 | 极值 |
| 正态 | `normal_distribution` | real | 标准正态（高斯） |
|  | `lognormal_distribution` | real | 对数正态 |
|  | `chi_squared_distribution` | real | 卡方 |
|  | `cauchy_distribution` | real | 柯西 |
|  | `fisher_f_distribution` | real | 费舍尔 F 分布 |
|  | `student_t_distribution` | real | 学生 t 分布 |
| 采样 | `discrete_distribution` | 整数 | 离散 |
|  | `piecewise_constant_distribution` | real | 在常数子区间上分布的值 |
|  | `piecewise_linear_distribution` | real | 在定义的子区间上分布的值 |

库提供的每个引擎都有优缺点。线性同余引擎具有较小的内部状态，但速度不是很快。另一方面，减法进位引擎非常快，但需要更多内部状态的内存。Mersenne 扭曲器是它们中最慢的，也是内部状态最大的一个，但在适当初始化时可以产生最长的不重复数字序列。在以下示例中，我们将使用`std::mt19937`，一个 32 位 Mersenne 扭曲器，内部状态有 19,937 位。

生成随机数的最简单方法如下：

```cpp
    auto mtgen = std::mt19937 {}; 
    for (auto i = 0; i < 10; ++i) 
      std::cout << mtgen() << std::endl;
```

在这个例子中，`mtgen`是一个`std::mt19937` Mersenne 扭曲器。要生成数字，只需要使用调用运算符来推进内部状态并返回下一个伪随机数。然而，这段代码有缺陷，因为引擎没有被种子化。因此，它总是产生相同的数字序列，这在大多数情况下可能不是你想要的。

有不同的方法来初始化引擎。一种方法，与 C rand 库常见，是使用当前时间。在现代 C++中，应该是这样的：

```cpp
    auto seed = std::chrono::high_resolution_clock::now() 
                .time_since_epoch() 
                .count(); 
    auto mtgen = std::mt19937{ static_cast<unsigned int>(seed) };
```

在这个例子中，`seed`是一个表示自时钟时代以来的滴答数的数字，直到当前时刻。然后使用这个数字来种子化引擎。这种方法的问题是`seed`的值实际上是确定性的，在某些类别的应用中可能容易受到攻击。更可靠的方法是用真正的随机数来种子化生成器。`std::random_device`类是一个应该返回真正随机数的引擎，尽管实现实际上可能基于伪随机生成器：

```cpp
    std::random_device rd; 
    auto mtgen = std::mt19937 {rd()};
```

所有引擎产生的数字都遵循均匀分布。要将结果转换为另一个统计分布，我们必须使用分布类。为了展示生成的数字如何根据所选的分布进行分布，我们将使用以下函数。该函数生成指定数量的伪随机数，并计算它们在映射中的重复次数。然后使用映射中的值生成类似条形图的图表，显示每个数字发生的频率：

```cpp
    void generate_and_print( 
      std::function<int(void)> gen,  
      int const iterations = 10000) 
    { 
      // map to store the numbers and their repetition 
      auto data = std::map<int, int>{}; 

      // generate random numbers 
      for (auto n = 0; n < iterations; ++n) 
        ++data[gen()]; 

      // find the element with the most repetitions 
      auto max = std::max_element( 
                 std::begin(data), std::end(data),  
                 [](auto kvp1, auto kvp2) { 
        return kvp1.second < kvp2.second; }); 

      // print the bars 
      for (auto i = max->second / 200; i > 0; --i) 
      { 
        for (auto kvp : data) 
        { 
          std::cout 
            << std::fixed << std::setprecision(1) << std::setw(3) 
            << (kvp.second / 200 >= i ? (char)219 : ' '); 
        } 

        std::cout << std::endl; 
      } 

      // print the numbers 
      for (auto kvp : data) 
      { 
        std::cout 
          << std::fixed << std::setprecision(1) << std::setw(3) 
          << kvp.first; 
      } 

      std::cout << std::endl; 
    }
```

以下代码使用`std::mt19937`引擎生成在范围`[1, 6]`内均匀分布的随机数；这基本上就是掷骰子时得到的结果：

```cpp
    std::random_device rd{}; 
    auto mtgen = std::mt19937{ rd() }; 
    auto ud = std::uniform_int_distribution<>{ 1, 6 }; 
    generate_and_print([&mtgen, &ud]() {return ud(mtgen); });
```

程序的输出如下：

![](img/0aa92f90-d587-4c3d-880b-bd0a859d81b4.png)

在下一个和最后一个例子中，我们将分布更改为均值为`5`，标准差为`2`的正态分布。这个分布产生实数；因此，为了使用先前的`generate_and_print()`函数，数字必须四舍五入为整数：

```cpp
    std::random_device rd{}; 
    auto mtgen = std::mt19937{ rd() }; 
    auto nd = std::normal_distribution<>{ 5, 2 }; 

    generate_and_print( 
      [&mtgen, &nd]() { 
        return static_cast<int>(std::round(nd(mtgen))); });
```

以下是先前代码的输出：

![](img/2f94f238-cc47-44ce-a685-b2e9e437441e.png)

# 另请参阅

+   *初始化伪随机数生成器的所有内部状态位*

# 初始化伪随机数生成器的所有内部状态位

在上一个教程中，我们已经看过了伪随机数库及其组件以及如何用它来产生不同统计分布中的数字。 在那个教程中忽略的一个重要因素是伪随机数生成器的正确初始化。 在本教程中，您将学习如何初始化生成器以产生最佳序列的伪随机数。

# 准备工作

您应该阅读前一个教程，*生成伪随机数*，以了解伪随机数库提供了什么。

# 如何做...

为了正确初始化伪随机数生成器以产生最佳序列的伪随机数，请执行以下步骤：

1.  使用`std::random_device`生成随机数以用作种子值：

```cpp
        std::random_device rd;
```

1.  为引擎的所有内部位生成随机数据：

```cpp
        std::array<int, std::mt19937::state_size> seed_data {};
        std::generate(std::begin(seed_data), std::end(seed_data), 
                      std::ref(rd));
```

1.  从先前生成的伪随机数据创建一个`std::seed_seq`对象：

```cpp
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
```

1.  创建引擎对象并初始化表示引擎内部状态的所有位；例如，`mt19937`有 19,937 位的内部状态：

```cpp
        auto eng = std::mt19937{ seq };
```

1.  根据应用程序的要求使用适当的分布：

```cpp
        auto dist = std::uniform_real_distribution<>{ 0, 1 };
```

# 它是如何工作的...

在上一个教程中显示的所有示例中，我们使用了一个`std::mt19937`引擎来产生伪随机数。 尽管梅森旋转器比其他引擎慢，但它可以产生最长的非重复数字序列，并具有最佳的频谱特性。 但是，以前的教程中显示的引擎初始化不会产生这种效果。 通过仔细分析（超出了本教程或本书的目的），可以证明引擎倾向于重复产生一些值并省略其他值，从而生成不均匀分布的数字，而是二项式或泊松分布。 问题在于`mt19937`的内部状态有 624 个 32 位整数，在上一个教程的示例中，我们只初始化了其中一个。

在使用伪随机数库时，请记住以下经验法则（在信息框中显示）：

为了产生最佳结果，引擎在生成数字之前必须正确初始化其所有内部状态。

伪随机数库提供了一个特定目的的类，称为`std::seed_seq`。 这是一个可以用任意数量的 32 位整数进行种子化，并在 32 位空间中产生请求的整数数量的生成器。

在*如何做...*部分的上述代码中，我们定义了一个名为`seed_data`的数组，其中包含与`mt19937`生成器的内部状态相等的 32 位整数数量；即 624 个整数。 然后，我们使用`std::random_device`生成的随机数初始化了数组。 稍后，该数组用于种子`std::seed_seq`，而`std::seed_seq`又用于种子`mt19937`生成器。

# 创建熟悉的用户定义文字

文字是内置类型（数字，布尔，字符，字符串和指针）的常量，不能在程序中更改。 语言定义了一系列前缀和后缀来指定文字（前缀/后缀实际上是文字的一部分）。 C++11 允许通过定义称为*文字运算符*的函数来创建用户定义的文字，引入后缀以指定文字。 这些仅适用于数字字符和字符串类型。 这打开了在将来版本中定义标准文字并允许开发人员创建自己的文字的可能性。 在本教程中，我们将看到如何创建我们自己的熟悉文字。

# 准备工作

用户定义文字可以有两种形式：*原始*和*熟*。原始文字不会被编译器处理，而熟文字是编译器处理的值（示例可以包括处理字符字符串中的转义序列或从文字 0xBAD 中识别整数值 2898）。原始文字仅适用于整数和浮点类型，而熟文字也适用于字符和字符字符串文字。

# 如何做到...

要创建熟用户定义文字，应遵循以下步骤：

1.  将您的文字定义在单独的命名空间中，以避免名称冲突。

1.  始终使用下划线（`_`）作为用户定义后缀的前缀。

1.  为熟文字定义以下形式的文字运算符：

```cpp
        T operator "" _suffix(unsigned long long int); 
        T operator "" _suffix(long double); 
        T operator "" _suffix(char); 
        T operator "" _suffix(wchar_t); 
        T operator "" _suffix(char16_t); 
        T operator "" _suffix(char32_t); 
        T operator "" _suffix(char const *, std::size_t); 
        T operator "" _suffix(wchar_t const *, std::size_t); 
        T operator "" _suffix(char16_t const *, std::size_t); 
        T operator "" _suffix(char32_t const *, std::size_t);
```

以下示例创建了一个用于指定千字节的用户定义文字：

```cpp
    namespace compunits 
    { 
      constexpr size_t operator "" _KB(unsigned long long const size) 
      { 
        return static_cast<size_t>(size * 1024); 
      } 
    } 

    auto size{ 4_KB };         // size_t size = 4096; 

    using byte = unsigned char; 
    auto buffer = std::array<byte, 1_KB>{};
```

# 它是如何工作的...

当编译器遇到具有用户定义后缀`S`的用户定义文字时（对于第三方后缀，它总是具有前导下划线，因为没有前导下划线的后缀是为标准库保留的），它会进行无限定名称查找，以便识别具有名称`operator "" S`的函数。如果找到一个，那么根据文字的类型和文字运算符的类型调用它。否则，编译器将产生错误。

在*如何做到...*部分的示例中，文字运算符称为`operator "" _KB`，其参数类型为`unsigned long long int`。这是处理整数类型的文字运算符的唯一可能类型。类似地，对于浮点数用户定义文字，参数类型必须是`long double`，因为对于数值类型，文字运算符必须能够处理可能的最大值。此文字运算符返回`constexpr`值，以便在需要编译时值的地方使用，例如在上面示例中指定数组大小时。

当编译器识别用户定义文字并且必须调用适当的用户定义文字运算符时，它将根据以下规则从重载集中选择重载：

+   **对于整数文字**：按以下顺序调用：接受`unsigned long long`的运算符，接受`const char*`的原始文字运算符，或文字运算符模板。

+   **对于浮点文字**：按以下顺序调用：接受`long double`的运算符，接受`const char*`的原始文字运算符，或文字运算符模板。

+   **对于字符文字**：根据字符类型（`char`、`wchar_t`、`char16_t`和`char32_t`）调用适当的运算符。

+   **对于字符串文字**：根据接受指向字符字符串和大小的指针的字符串类型调用适当的运算符。

在以下示例中，我们定义了一个单位和数量的系统。我们希望使用千克、件、升和其他类型的单位进行操作。这在需要处理订单并且需要为每个商品指定数量和单位的系统中可能很有用。以下内容在命名空间`units`中定义：

+   用于单位可能类型（千克、米、升和件）的作用域枚举：

```cpp
        enum class unit { kilogram, liter, meter, piece, };
```

+   用于指定特定单位的数量的类模板（例如 3.5 千克或 42 件）：

```cpp
        template <unit U> 
        class quantity 
        {
          const double amount; 
          public: 
            constexpr explicit quantity(double const a) : 
              amount(a) {} 

          explicit operator double() const { return amount; } 
        };
```

+   `quantity`类模板的`operator+`和`operator-`函数，以便能够添加和减去数量：

```cpp
        template <unit U> 
        constexpr quantity<U> operator+(quantity<U> const &q1, 
                                        quantity<U> const &q2) 
        {
          return quantity<U>(static_cast<double>(q1) + 
                             static_cast<double>(q2)); 
        } 

        template <unit U> 
        constexpr quantity<U> operator-(quantity<U> const &q1, 
                                        quantity<U> const &q2)
        {
          return quantity<U>(static_cast<double>(q1) - 
                             static_cast<double>(q2));
        }
```

+   文字运算符用于创建`quantity`文字，定义在名为`unit_literals`的内部命名空间中。这样做的目的是避免与其他命名空间中的文字可能发生的名称冲突。如果确实发生这样的冲突，开发人员可以在需要定义文字的范围中使用适当的命名空间来选择他们应该使用的文字：

```cpp
        namespace unit_literals 
        { 
          constexpr quantity<unit::kilogram> operator "" _kg( 
              long double const amount) 
          { 
            return quantity<unit::kilogram>  
              { static_cast<double>(amount) }; 
          } 

          constexpr quantity<unit::kilogram> operator "" _kg( 
             unsigned long long const amount) 
          { 
            return quantity<unit::kilogram>  
              { static_cast<double>(amount) }; 
          } 

          constexpr quantity<unit::liter> operator "" _l( 
             long double const amount) 
          { 
             return quantity<unit::liter>  
               { static_cast<double>(amount) }; 
          } 

          constexpr quantity<unit::meter> operator "" _m( 
             long double const amount) 
          { 
            return quantity<unit::meter>  
              { static_cast<double>(amount) }; 
          } 

          constexpr quantity<unit::piece> operator "" _pcs( 
             unsigned long long const amount) 
          { 
            return quantity<unit::piece>  
              { static_cast<double>(amount) }; 
          } 
        }
```

仔细观察，可以注意到先前定义的文字运算符不同：

+   `_kg`既适用于整数文字，也适用于浮点文字；这使我们能够创建整数值和浮点值，比如`1_kg`和`1.0_kg`。

+   `_l`和`_m`仅适用于浮点文字；这意味着我们只能使用浮点数定义这些单位的数量文字，比如`4.5_l`和`10.0_m`。

+   `_pcs`仅适用于整数字面值；这意味着我们只能定义整数数量的片数，比如`42_pcs`。

有了这些文字操作符，我们可以操作各种数量。以下示例显示了有效和无效的操作：

```cpp
    using namespace units; 
    using namespace unit_literals; 

    auto q1{ 1_kg };    // OK
    auto q2{ 4.5_kg };  // OK
    auto q3{ q1 + q2 }; // OK
    auto q4{ q2 - q1 }; // OK

    // error, cannot add meters and pieces 
    auto q5{ 1.0_m + 1_pcs }; 
    // error, cannot have an integer number of liters 
    auto q6{ 1_l }; 
    // error, can only have an integer number of pieces 
    auto q7{ 2.0_pcs}
```

`q1`是 1 千克的数量；这是一个整数值。由于存在重载的`operator "" _kg(unsigned long long const)`，因此可以从整数 1 正确地创建文字。同样，`q2`是 4.5 千克的数量；这是一个实数值。由于存在`overload operator "" _kg(long double)`，因此可以从双精度浮点值 4.5 创建文字。

另一方面，`q6`是 1 升的数量。由于没有重载的`operator "" _l(unsigned long long)`，因此无法创建文字。这将需要一个接受`unsigned long long`的重载，但这样的重载不存在。同样，`q7`是 2.0 个零件的数量，但零件文字只能从整数值创建，因此这将生成另一个编译器错误。

# 还有更多...

尽管用户定义文字从 C++11 开始可用，但标准文字操作符仅从 C++14 开始可用。以下是这些标准文字操作符的列表：

+   `operator""s`用于定义`std::basic_string`文字：

```cpp
        using namespace std::string_literals; 

        auto s1{  "text"s }; // std::string 
        auto s2{ L"text"s }; // std::wstring 
        auto s3{ u"text"s }; // std::u16string 
        auto s4{ U"text"s }; // std::u32string
```

+   `operator""h`、`operator""min`、`operator""s`、`operator""ms`、`operator""us`和`operator""ns`用于创建`std::chrono::duration`值：

```cpp
        using namespace std::literals::chrono_literals; 

        // std::chrono::duration<long long> 
        auto timer {2h + 42min + 15s};
```

+   `operator""if`、`operator""i`和`operator""il`用于创建`std::complex`值：

```cpp
        using namespace std::literals::complex_literals; 

        auto c{ 12.0 + 4.5i }; // std::complex<double>
```

# 另请参阅

+   *使用原始字符串文字来避免转义字符*

+   *创建原始用户定义文字*

# 创建原始用户定义文字

在上一个教程中，我们已经看到了 C++11 允许库实现者和开发人员创建用户定义文字以及 C++14 标准中可用的用户定义文字的方式。然而，用户定义文字有两种形式，一种是熟练的形式，在这种形式中，文字值在提供给文字操作符之前由编译器处理，另一种是原始形式，在这种形式中，文字不会被编译器解析。后者仅适用于整数和浮点类型。在本教程中，我们将看看如何创建原始用户定义文字。

# 做好准备

在继续本教程之前，强烈建议您阅读上一个教程《创建熟悉的用户定义文字》，因为这里不会重复介绍有关用户定义文字的一般细节。

为了举例说明原始用户定义的文字如何创建，我们将定义二进制文字。这些二进制文字可以是 8 位、16 位和 32 位（无符号）类型。这些类型将被称为`byte8`、`byte16`和`byte32`，我们创建的文字将被称为`_b8`、`_b16`和`_b32`。

# 操作步骤

要创建原始用户定义文字，您应该按照以下步骤进行：

1.  将您的文字定义在一个单独的命名空间中，以避免名称冲突。

1.  始终使用下划线（`_`）前缀来定义使用的后缀。

1.  定义以下形式的文字操作符或文字操作符模板：

```cpp
        T operator "" _suffix(const char*); 

        template<char...> T operator "" _suffix();
```

以下示例显示了 8 位、16 位和 32 位二进制文字的可能实现：

```cpp
    namespace binary 
    { 
      using byte8  = unsigned char; 
      using byte16 = unsigned short; 
      using byte32 = unsigned int; 

      namespace binary_literals 
      { 
        namespace binary_literals_internals 
        { 
          template <typename CharT, char... bits> 
          struct binary_struct; 

          template <typename CharT, char... bits> 
          struct binary_struct<CharT, '0', bits...> 
          { 
            static constexpr CharT value{ 
              binary_struct<CharT, bits...>::value }; 
          }; 

          template <typename CharT, char... bits> 
          struct binary_struct<CharT, '1', bits...> 
          { 
            static constexpr CharT value{ 
              static_cast<CharT>(1 << sizeof...(bits)) | 
              binary_struct<CharT, bits...>::value }; 
          }; 

          template <typename CharT> 
          struct binary_struct<CharT> 
          { 
            static constexpr CharT value{ 0 }; 
          }; 
        } 

        template<char... bits> 
        constexpr byte8 operator""_b8() 
        { 
          static_assert( 
            sizeof...(bits) <= 8, 
            "binary literal b8 must be up to 8 digits long"); 

          return binary_literals_internals:: 
                    binary_struct<byte8, bits...>::value; 
        } 

        template<char... bits> 
        constexpr byte16 operator""_b16() 
        { 
          static_assert( 
            sizeof...(bits) <= 16, 
            "binary literal b16 must be up to 16 digits long"); 

          return binary_literals_internals:: 
                    binary_struct<byte16, bits...>::value; 
        } 

        template<char... bits> 
        constexpr byte32 operator""_b32() 
        { 
          static_assert( 
             sizeof...(bits) <= 32, 
             "binary literal b32 must be up to 32 digits long"); 

          return binary_literals_internals:: 
                    binary_struct<byte32, bits...>::value; 
        } 

      } 
    }
```

# 工作原理

上一节中的实现使我们能够定义二进制文字的形式 1010_b8（十进制值为 10 的`byte8`值）或 000010101100_b16（十进制值为 2130496 的`byte16`值）。但是，我们要确保不超过每种类型的数字位数。换句话说，像 111100001_b8 这样的值应该是非法的，编译器应该产生错误。

首先，我们在一个名为`binary`的命名空间中定义了所有内容，并开始引入几个类型别名（`byte8`、`byte16`和`byte32`）。

文字操作符模板定义在一个名为`binary_literal_internals`的嵌套命名空间中。这是一个很好的做法，以避免与其他命名空间中的其他文字操作符发生名称冲突。如果发生这样的情况，您可以选择在正确的范围内使用适当的命名空间（例如，在一个函数或块中使用一个命名空间，在另一个函数或块中使用另一个命名空间）。

这三个文字操作符模板非常相似。唯一不同的是它们的名称（`_b8`、`_16`和`_b32`）、返回类型（`byte8`、`byte16`和`byte32`）以及在静态断言中检查数字个数的条件。

我们将在以后的配方中探讨可变参数模板和模板递归的细节；然而，为了更好地理解，这就是这个特定实现的工作原理：`bits`是一个模板参数包，不是单个值，而是模板可以实例化的所有值。例如，如果我们考虑文字`1010_b8`，那么文字操作符模板将被实例化为`operator"" _b8<'1', '0', '1', '0'>()`。在继续计算二进制值之前，我们检查文字中的数字个数。对于`_b8`，这个数字不能超过八个（包括任何尾随的零）。类似地，对于`_b16`，它应该是最多 16 位数字，对于`_b32`，它应该是 32 位。为此，我们使用`sizeof...`操作符，它返回参数包中的元素数（在这种情况下是`bits`）。

如果数字个数正确，我们可以继续展开参数包并递归计算二进制文字表示的十进制值。这是通过另一个类模板及其专业化的帮助完成的。这些模板定义在另一个名为`binary_literals_internals`的嵌套命名空间中。这也是一个很好的做法，因为它将实现细节（除非使用了显式的 using namespace 指令将其提供给当前命名空间）隐藏起来，不让客户端看到。

尽管这看起来像是递归，但它并不是真正的运行时递归，因为在编译器展开并从模板生成代码之后，我们最终得到的基本上是对具有不同参数数量的重载函数的调用。这在后面的配方*使用可变数量参数的函数模板*中有进一步解释。

`binary_struct`类模板有一个模板类型`CharT`，用于函数的返回类型（我们需要这个，因为我们的文字操作符模板应该返回`byte8`、`byte16`或`byte32`），还有一个参数包：

```cpp
    template <typename CharT, char... bits> 
    struct binary_struct;
```

这个类模板的几个专业化版本都带有参数包分解。当包的第一个数字是'0'时，计算出的值保持不变，我们继续展开包的其余部分。如果包的第一个数字是'1'，那么新值就是 1 左移包剩余位数的数字，或者包的其余部分的值：

```cpp
    template <typename CharT, char... bits> 
    struct binary_struct<CharT, '0', bits...> 
    { 
      static constexpr CharT value{ 
        binary_struct<CharT, bits...>::value }; 
    }; 

    template <typename CharT, char... bits> 
    struct binary_struct<CharT, '1', bits...> 
    { 
      static constexpr CharT value{ 
        static_cast<CharT>(1 << sizeof...(bits)) | 
        binary_struct<CharT, bits...>::value }; 
    };
```

最后一个专业化涵盖了包为空的情况；在这种情况下，我们返回 0：

```cpp
    template <typename CharT> 
    struct binary_struct<CharT> 
    { 
      static constexpr CharT value{ 0 }; 
    };
```

在定义了这些辅助类之后，我们可以按预期实现`byte8`、`byte16`和`byte32`二进制文字。请注意，我们需要将`binary_literals`命名空间的内容引入当前命名空间，以便使用文字操作符模板：

```cpp
    using namespace binary; 
    using namespace binary_literals; 
    auto b1 = 1010_b8; 
    auto b2 = 101010101010_b16; 
    auto b3 = 101010101010101010101010_b32;
```

以下定义触发编译器错误，因为`static_assert`中的条件不满足：

```cpp
    // binary literal b8 must be up to 8 digits long 
    auto b4 = 0011111111_b8; 
    // binary literal b16 must be up to 16 digits long 
    auto b5 = 001111111111111111_b16; 
    // binary literal b32 must be up to 32 digits long 
auto b6 = 0011111111111111111111111111111111_b32;
```

# 另请参阅

+   *使用原始字符串文字来避免转义字符*

+   *创建熟悉的用户定义文字*

+   *使用可变数量参数的函数模板* 第十章 的配方，*探索函数*

+   *创建类型别名和别名模板*食谱第八章，*学习现代核心语言特性*

# 使用原始字符串文字来避免转义字符

字符串可能包含特殊字符，例如非打印字符（换行符、水平和垂直制表符等）、字符串和字符分隔符（双引号和单引号）或任意的八进制、十六进制或 Unicode 值。这些特殊字符以反斜杠开头的转义序列引入，后面跟着字符（例如`'`和`"`）、其指定的字母（例如`n`表示换行，`t`表示水平制表符）或其值（例如八进制 050、十六进制 XF7 或 Unicode U16F0）。因此，反斜杠字符本身必须用另一个反斜杠字符转义。这导致更复杂的文字字符串，很难阅读。

为了避免转义字符，C++11 引入了不处理转义序列的原始字符串文字。在这个示例中，您将学习如何使用各种形式的原始字符串文字。

# 准备就绪

在这个示例中，以及本书的其余部分，我将使用`s`后缀来定义`basic_string`文字。这已经在食谱*创建熟用户定义的文字*中介绍过。

# 如何做...

为了避免转义字符，使用以下定义字符串文字：

1.  `R"( literal )"`作为默认形式：

```cpp
        auto filename {R"(C:\Users\Marius\Documents\)"s};
        auto pattern {R"((\w+)=(\d+)$)"s}; 

        auto sqlselect { 
          R"(SELECT * 
          FROM Books 
          WHERE Publisher='Paktpub' 
          ORDER BY PubDate DESC)"s};
```

1.  `R"delimiter( literal )delimiter"`其中`delimiter`是实际字符串中不存在的任何字符序列，当序列`)"`实际上应该是字符串的一部分时。这里有一个以`!!`为分隔符的例子：

```cpp
        auto text{ R"!!(This text contains both "( and )".)!!"s }; 
        std::cout << text << std::endl;
```

# 工作原理...

当使用字符串文字时，转义不会被处理，字符串的实际内容将被写在分隔符之间（换句话说，你看到的就是你得到的）。下面的例子显示了看起来相同的原始文字字符串；然而，第二个字符串仍然包含转义字符。由于在字符串文字的情况下不处理这些字符，它们将按原样打印在输出中：

```cpp
    auto filename1 {R"(C:\Users\Marius\Documents\)"s}; 
    auto filename2 {R"(C:\\Users\\Marius\\Documents\\)"s}; 

    // prints C:\Users\Marius\Documents\  
    std::cout << filename1 << std::endl; 

    // prints C:\\Users\\Marius\\Documents\\  
    std::cout << filename2 << std::endl;
```

如果文本必须包含`)"`序列，则必须使用不同的分隔符，形式为`R"delimiter( literal )delimiter"`。根据标准，分隔符中可能包含以下字符：

基本源字符集的任何成员，除了：空格、左括号（右括号）、反斜杠\和表示水平制表符、垂直制表符、换页和换行的控制字符。

原始字符串文字可以由`L`、`u8`、`u`和`U`中的一个前缀，表示宽字符、UTF-8、UTF-16 或 UTF-32 字符串文字。以下是这种字符串文字的例子。请注意，字符串末尾的`operator ""s`存在使编译器推断类型为各种字符串类而不是字符数组：

```cpp
    auto t1{ LR"(text)"  };  // const wchar_t* 
    auto t2{ u8R"(text)" };  // const char* 
    auto t3{ uR"(text)"  };  // const char16_t* 
    auto t4{ UR"(text)"  };  // const char32_t* 

    auto t5{ LR"(text)"s  }; // wstring 
    auto t6{ u8R"(text)"s }; // string 
    auto t7{ uR"(text)"s  }; // u16string 
    auto t8{ UR"(text)"s  }; // u32string
```

# 另请参阅

+   *创建熟用户定义的文字*

# 创建一个字符串助手库

标准库中的字符串类型是一个通用实现，缺少许多有用的方法，例如更改大小写、修剪、拆分和其他可能满足不同开发人员需求的方法。存在提供丰富的字符串功能集的第三方库。然而，在这个示例中，我们将看到实现几种简单但有用的方法，这些方法在实践中经常需要。目的是看看如何使用字符串方法和标准通用算法来操作字符串，但也是为了有一个可重用的代码参考，可以在您的应用程序中使用。

在这个示例中，我们将实现一个小型字符串工具库，该库将提供以下功能的函数：

+   将字符串更改为小写或大写。

+   反转字符串。

+   从字符串的开头和/或结尾修剪空格。

+   从字符串的开头和/或结尾修剪特定的字符集。

+   在字符串中的任何位置删除字符的出现。

+   使用特定分隔符对字符串进行标记化。

# 准备工作

我们将要实现的字符串库应该适用于所有标准字符串类型，`std::string`、`std::wstring`、`std::u16string`和`std::u32string`。为了避免指定诸如`std::basic_string<CharT, std::char_traits<CharT>, std::allocator<CharT>>`这样的长名称，我们将使用以下字符串和字符串流的别名模板：

```cpp
    template <typename CharT> 
    using tstring =  
       std::basic_string<CharT, std::char_traits<CharT>,  
                         std::allocator<CharT>>; 

    template <typename CharT> 
    using tstringstream =  
       std::basic_stringstream<CharT, std::char_traits<CharT>,  
                               std::allocator<CharT>>;
```

要实现这些字符串辅助函数，我们需要包含`<string>`头文件用于字符串和`<algorithm>`用于我们将使用的一般标准算法。

在本教程中的所有示例中，我们将使用 C++14 的标准用户定义的字符串字面量操作符，因此我们需要显式使用`std::string_literals`命名空间。

# 如何做...

1.  要将字符串转换为小写或大写，使用通用目的算法`std::transform()`对字符串的字符应用`tolower()`或`toupper()`函数：

```cpp
        template<typename CharT> 
        inline tstring<CharT> to_upper(tstring<CharT> text) 
        { 
          std::transform(std::begin(text), std::end(text), 
                         std::begin(text), toupper); 
          return text; 
        } 

        template<typename CharT> 
        inline tstring<CharT> to_lower(tstring<CharT> text) 
        { 
          std::transform(std::begin(text), std::end(text),  
                         std::begin(text), tolower); 
          return text; 
        }
```

1.  要颠倒字符串，使用通用目的算法`std::reverse()`：

```cpp
        template<typename CharT> 
        inline tstring<CharT> reverse(tstring<CharT> text) 
        { 
          std::reverse(std::begin(text), std::end(text)); 
          return text; 
        }
```

1.  要修剪字符串，在开头、结尾或两者都使用`std::basic_string`的`find_first_not_of()`和`find_last_not_of()`方法：

```cpp
        template<typename CharT> 
        inline tstring<CharT> trim(tstring<CharT> const & text) 
        { 
          auto first{ text.find_first_not_of(' ') }; 
          auto last{ text.find_last_not_of(' ') }; 
          return text.substr(first, (last - first + 1)); 
        } 

        template<typename CharT> 
        inline tstring<CharT> trimleft(tstring<CharT> const & text) 
        { 
          auto first{ text.find_first_not_of(' ') }; 
          return text.substr(first, text.size() - first); 
        } 

        template<typename CharT> 
        inline tstring<CharT> trimright(tstring<CharT> const & text) 
        { 
          auto last{ text.find_last_not_of(' ') }; 
          return text.substr(0, last + 1); 
        }
```

1.  要从字符串中修剪给定集合中的字符，使用`std::basic_string`的`find_first_not_of()`和`find_last_not_of()`的重载方法，它们接受一个字符串参数，定义要查找的字符集：

```cpp
        template<typename CharT> 
        inline tstring<CharT> trim(tstring<CharT> const & text,  
                                   tstring<CharT> const & chars) 
        { 
          auto first{ text.find_first_not_of(chars) }; 
          auto last{ text.find_last_not_of(chars) }; 
          return text.substr(first, (last - first + 1)); 
        } 

        template<typename CharT> 
        inline tstring<CharT> trimleft(tstring<CharT> const & text,  
                                       tstring<CharT> const & chars) 
        { 
          auto first{ text.find_first_not_of(chars) }; 
          return text.substr(first, text.size() - first); 
        } 

        template<typename CharT> 
        inline tstring<CharT> trimright(tstring<CharT> const &text, 
                                        tstring<CharT> const &chars) 
        { 
          auto last{ text.find_last_not_of(chars) }; 
          return text.substr(0, last + 1); 
        }
```

1.  要从字符串中删除字符，使用`std::remove_if()`和`std::basic_string::erase()`：

```cpp
        template<typename CharT> 
        inline tstring<CharT> remove(tstring<CharT> text,  
                                     CharT const ch) 
        { 
          auto start = std::remove_if( 
                          std::begin(text), std::end(text),  
                          = {return c ==  ch; }); 
          text.erase(start, std::end(text)); 
          return text; 
        }
```

1.  根据指定的分隔符拆分字符串，使用`std::getline()`从初始化为字符串内容的`std::basic_stringstream`中读取。从流中提取的标记被推入字符串向量中：

```cpp
        template<typename CharT> 
        inline std::vector<tstring<CharT>> split 
           (tstring<CharT> text, CharT const delimiter) 
        {
          auto sstr = tstringstream<CharT>{ text }; 
          auto tokens = std::vector<tstring<CharT>>{}; 
          auto token = tstring<CharT>{}; 
          while (std::getline(sstr, token, delimiter))  
          { 
            if (!token.empty()) tokens.push_back(token); 
          } 
          return tokens; 
        }
```

# 工作原理...

为了实现库中的实用函数，我们有两个选择：

+   函数将修改通过引用传递的字符串。

+   函数不会改变原始字符串，而是返回一个新字符串。

第二个选项的优点是它保留了原始字符串，这在许多情况下可能有所帮助。否则，在这些情况下，您首先必须复制字符串并更改副本。此处提供的实现采用了第二种方法。

我们在*如何做...*部分中实现的第一个函数是`to_upper()`和`to_lower()`。这些函数将字符串的内容更改为大写或小写。实现这个最简单的方法是使用`std::transform()`标准算法。这是一个通用目的算法，它对范围的每个元素应用一个函数（由开始和结束迭代器定义），并将结果存储在另一个范围中，只需要指定开始迭代器。输出范围可以与输入范围相同，这正是我们用来转换字符串的方法。应用的函数是`toupper()`或`tolower()`：

```cpp
    auto ut{ string_library::to_upper("this is not UPPERCASE"s) };  
    // ut = "THIS IS NOT UPPERCASE" 

    auto lt{ string_library::to_lower("THIS IS NOT lowercase"s) };  
    // lt = "this is not lowercase"
```

我们考虑的下一个函数是`reverse()`，正如其名称所示，它颠倒了字符串的内容。为此，我们使用了`std::reverse()`标准算法。这个通用目的算法颠倒了由开始和结束迭代器定义的范围的元素：

```cpp
    auto rt{string_library::reverse("cookbook"s)}; // rt = "koobkooc"
```

在修剪方面，字符串可以在开头、结尾或两侧修剪。因此，我们实现了三个不同的函数：`trim()`用于两端修剪，`trimleft()`用于修剪字符串开头，`trimright()`用于修剪字符串结尾。函数的第一个版本仅修剪空格。为了找到要修剪的正确部分，我们使用`std::basic_string`的`find_first_not_of()`和`find_last_not_of()`方法。这些方法返回字符串中不是指定字符的第一个和最后一个字符。随后，调用`std::basic_string`的`substr()`方法返回一个新字符串。`substr()`方法接受字符串中的索引和要复制到新字符串的元素数：

```cpp
    auto text1{"   this is an example   "s}; 
    // t1 = "this is an example" 
    auto t1{ string_library::trim(text1) }; 
    // t2 = "this is an example   " 
    auto t2{ string_library::trimleft(text1) }; 
    // t3 = "   this is an example" 
    auto t3{ string_library::trimright(text1) };
```

有时从字符串中修剪其他字符和空格可能很有用。为了做到这一点，我们为修剪函数提供了重载，指定要删除的一组字符。该集合也被指定为一个字符串。实现非常类似于之前的实现，因为`find_first_not_of()`和`find_last_not_of()`都有重载，接受包含要从搜索中排除的字符的字符串：

```cpp
    auto chars1{" !%\n\r"s}; 
    auto text3{"!!  this % needs a lot\rof trimming  !\n"s}; 
    auto t7{ string_library::trim(text3, chars1) };        
    // t7 = "this % needs a lot\rof trimming" 
    auto t8{ string_library::trimleft(text3, chars1) };    
    // t8 = "this % needs a lot\rof trimming  !\n" 
    auto t9{ string_library::trimright(text3, chars1) };   
    // t9 = "!!  this % needs a lot\rof trimming"
```

如果需要从字符串的任何部分删除字符，则修剪方法将无效，因为它们只处理字符串开头和结尾的连续字符序列。为此，我们实现了一个简单的`remove()`方法。这使用了`std:remove_if()`标准算法。`std::remove()`和`std::remove_if()`都以一种可能一开始不太直观的方式工作。它们通过重新排列范围的内容（使用移动赋值）来删除满足条件的元素。需要删除的元素被放置在范围的末尾，并且该函数返回一个指向表示已删除元素的范围中的第一个元素的迭代器。这个迭代器基本上定义了修改后的范围的新结尾。如果没有删除任何元素，则返回的迭代器是原始范围的结束迭代器。然后使用返回的迭代器的值调用`std::basic_string::erase()`方法，该方法实际上擦除了由两个迭代器定义的字符串的内容。在我们的情况下，两个迭代器是`std::remove_if()`返回的迭代器和字符串的末尾：

```cpp
    auto text4{"must remove all * from text**"s}; 
    auto t10{ string_library::remove(text4, '*') };  
    // t10 = "must remove all  from text" 
    auto t11{ string_library::remove(text4, '!') };  
    // t11 = "must remove all * from text**"
```

我们实现的最后一个方法基于指定的分隔符拆分字符串的内容。有各种方法可以实现这一点。在这个实现中，我们使用了`std::getline()`。这个函数从输入流中读取字符，直到找到指定的分隔符，并将字符放入一个字符串中。在从输入缓冲区开始读取之前，它调用`erase()`方法清除输出字符串的内容。在循环中调用此方法会产生放置在向量中的标记。在我们的实现中，从结果集中跳过了空标记：

```cpp
    auto text5{"this text will be split   "s}; 
    auto tokens1{ string_library::split(text5, ' ') };  
    // tokens1 = {"this", "text", "will", "be", "split"} 
    auto tokens2{ string_library::split(""s, ' ') };    
    // tokens2 = {}
```

# 另请参阅

+   *创建熟制用户定义的字面量*

+   *创建类型别名和别名模板* 第八章 的配方，*学习现代核心语言特性*

# 使用正则表达式验证字符串的格式

正则表达式是一种用于在文本中执行模式匹配和替换的语言。C++11 通过标准库提供了对正则表达式的支持，通过`<regex>`头文件中提供的一组类、算法和迭代器。在本配方中，我们将看到如何使用正则表达式来验证字符串是否与模式匹配（示例可以包括验证电子邮件或 IP 地址格式）。

# 准备工作

在本配方中，我们将在必要时解释我们使用的正则表达式的细节。但是，为了使用 C++标准库进行正则表达式，您应该至少具有一些正则表达式的基本知识。正则表达式语法和标准的描述超出了本书的目的；如果您对正则表达式不熟悉，建议您在继续专注于正则表达式的配方之前先阅读更多相关内容。

# 操作步骤

为了验证字符串是否与正则表达式匹配，请执行以下步骤：

1.  包括头文件`<regex>`和`<string>`以及命名空间`std::string_literals`，用于 C++14 标准用户定义的字符串字面量：

```cpp
        #include <regex> 
        #include <string> 
        using namespace std::string_literals;
```

1.  使用原始字符串字面量指定正则表达式，以避免转义反斜杠（可能经常发生）。以下正则表达式验证大多数电子邮件格式：

```cpp
        auto pattern {R"(^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$)"s};
```

1.  创建一个`std::regex`/`std::wregex`对象（取决于所使用的字符集）来封装正则表达式：

```cpp
        auto rx = std::regex{pattern};
```

1.  - 要忽略大小写或指定其他解析选项，请使用具有额外参数的重载构造函数，用于正则表达式标志：

```cpp
        auto rx = std::regex{pattern, std::regex_constants::icase}; 
```

1.  - 使用`std::regex_match()`来将正则表达式与整个字符串匹配：

```cpp
        auto valid = std::regex_match("marius@domain.com"s, rx);
```

# - 工作原理...

- 考虑验证电子邮件地址格式的问题，尽管这看起来可能是一个微不足道的问题，但实际上很难找到一个简单的正则表达式，涵盖所有可能的有效电子邮件格式。在这个示例中，我们不会试图找到那个最终的正则表达式，而是应用一个对大多数情况来说足够好的正则表达式。我们将用于此目的的正则表达式是：

```cpp
    ^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$
```

- 以下表格解释了正则表达式的结构：

| - **部分** | **描述** |
| --- | --- |
| - `^` | 字符串的开头 |
| - `[A-Z0-9._%+-]+` | 至少一个字符在 A-Z，0-9 范围内，或者是-，%，+或-中的一个，表示电子邮件地址的本地部分 |
| - `@` | 字符@ |
| - `[A-Z0-9.-]+` | 至少一个字符在 A-Z，0-9 范围内，或者是-，%，+或-中的一个，表示域部分的主机名 |
| - `\.` | 分隔域名主机名和标签的点 |
| - `[A-Z]{2,}` | 可以包含 2 到 63 个字符的域的 DNS 标签 |
| - `$` | 字符串的结尾 |

- 请记住，实际上域名由主机名后跟一个以点分隔的 DNS 标签列表组成。例如`localhost`，`gmail.com`或`yahoo.co.uk`。我们使用的这个正则表达式不匹配没有 DNS 标签的域，比如 localhost（例如`root@localhost`是一个有效的电子邮件）。域名也可以是用括号指定的 IP 地址，例如`[192.168.100.11]`（如`john.doe@[192.168.100.11]`）。包含这些域的电子邮件地址将不匹配上面定义的正则表达式。尽管这些相对罕见的格式不会被匹配，但是正则表达式可以覆盖大多数电子邮件格式。

- 本章示例中的正则表达式仅用于教学目的，并不打算直接用于生产代码。正如前面所解释的，此示例并未涵盖所有可能的电子邮件格式。

- 我们首先包含了必要的头文件，`<regex>`用于正则表达式，`<string>`用于字符串。下面显示的`is_valid_email()`函数（基本上包含了*如何操作...*部分的示例）接受一个表示电子邮件地址的字符串，并返回一个布尔值，指示该电子邮件是否具有有效格式。我们首先构造一个`std::regex`对象，以封装用原始字符串文字指示的正则表达式。使用原始字符串文字是有帮助的，因为它避免了在正则表达式中用于转义字符的反斜杠。然后函数调用`std::regex_match()`，传递输入文本和正则表达式：

```cpp
    bool is_valid_email_format(std::string const & email) 
    { 
      auto pattern {R"(^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$)"s}; 

      auto rx = std::regex{pattern}; 

      return std::regex_match(email, rx); 
    }
```

- `std::regex_match()`方法尝试将正则表达式与整个字符串匹配。如果成功，则返回`true`，否则返回`false`：

```cpp
    auto ltest = [](std::string const & email)  
    { 
      std::cout << std::setw(30) << std::left  
                << email << " : "  
                << (is_valid_email_format(email) ?  
                   "valid format" : "invalid format") 
                << std::endl; 
    }; 

    ltest("JOHN.DOE@DOMAIN.COM"s);         // valid format 
    ltest("JOHNDOE@DOMAIL.CO.UK"s);        // valid format 
    ltest("JOHNDOE@DOMAIL.INFO"s);         // valid format 
    ltest("J.O.H.N_D.O.E@DOMAIN.INFO"s);   // valid format 
    ltest("ROOT@LOCALHOST"s);              // invalid format 
    ltest("john.doe@domain.com"s);         // invalid format
```

- 在这个简单的测试中，唯一不匹配正则表达式的电子邮件是`ROOT@LOCALHOST`和`john.doe@domain.com`。第一个包含一个没有点前缀 DNS 标签的域名，这种情况在正则表达式中没有涵盖。第二个只包含小写字母，在正则表达式中，本地部分和域名的有效字符集都是大写字母 A 到 Z。

不要用额外的有效字符（例如`[A-Za-z0-9._%+-]`）使正则表达式复杂化，我们可以指定匹配时忽略大小写。这可以通过`std::basic_regex`类的构造函数的额外参数来实现。用于此目的的可用常量在`regex_constants`命名空间中定义。对`is_valid_email_format()`的以下轻微更改将使其忽略大小写，并允许大小写字母的电子邮件正确匹配正则表达式：

```cpp
    bool is_valid_email_format(std::string const & email) 
    { 
      auto rx = std::regex{ 
        R"(^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$)"s, 
        std::regex_constants::icase}; 

      return std::regex_match(email, rx); 
    }
```

这个`is_valid_email_format()`函数非常简单，如果正则表达式与要匹配的文本一起作为参数提供，它可以用于匹配任何内容。但是，希望能够使用单个函数处理不仅是多字节字符串（`std::string`），还包括宽字符串（`std::wstring`）。这可以通过创建一个函数模板来实现，其中字符类型作为模板参数提供：

```cpp
    template <typename CharT> 
    using tstring = std::basic_string<CharT, std::char_traits<CharT>,  
                                      std::allocator<CharT>>; 

    template <typename CharT> 
    bool is_valid_format(tstring<CharT> const & pattern,  
                         tstring<CharT> const & text) 
    { 
      auto rx = std::basic_regex<CharT>{  
        pattern, std::regex_constants::icase }; 

      return std::regex_match(text, rx); 
    }
```

我们首先创建了`std::basic_string`的别名模板，以简化其使用。新的`is_valid_format()`函数是一个函数模板，与我们的`is_valid_email()`的实现非常相似。但是，现在我们使用`std::basic_regex<CharT>`而不是`std::regex`的`typedef`，它是`std::basic_regex<char>`，并且模式作为第一个参数提供。我们现在实现了一个名为`is_valid_email_format_w()`的新函数，用于宽字符串，它依赖于这个函数模板。但是，函数模板可以被重用来实现其他验证，例如车牌是否具有特定格式：

```cpp
    bool is_valid_email_format_w(std::wstring const & text) 
    { 
      return is_valid_format( 
        LR"(^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$)"s,  
        text); 
    } 

    auto ltest2 = [](auto const & email) 
    { 
      std::wcout << std::setw(30) << std::left 
         << email << L" : " 
         << (is_valid_email_format_w(email) ? L"valid" : L"invalid") 
         << std::endl; 
    }; 

    ltest2(L"JOHN.DOE@DOMAIN.COM"s);       // valid
    ltest2(L"JOHNDOE@DOMAIL.CO.UK"s);      // valid
    ltest2(L"JOHNDOE@DOMAIL.INFO"s);       // valid
    ltest2(L"J.O.H.N_D.O.E@DOMAIN.INFO"s); // valid
    ltest2(L"ROOT@LOCALHOST"s);            // invalid
    ltest2(L"john.doe@domain.com"s);       // valid
```

在上面显示的所有示例中，唯一不匹配的是`ROOT@LOCAHOST`，这是预期的。

`std::regex_match()`方法实际上有多个重载版本，其中一些版本有一个参数，是指向`std::match_results`对象的引用，用于存储匹配结果。如果没有匹配，则`std::match_results`为空，大小为 0。否则，如果有匹配，`std::match_results`对象不为空，大小为匹配的子表达式数加 1。

函数的以下版本使用了上述重载，并将匹配的子表达式返回到`std::smatch`对象中。请注意，正则表达式已更改，因为定义了三个标题组--一个用于域的本地部分，一个用于主机名部分，一个用于 DNS 标签。如果匹配成功，则`std::smatch`对象将包含四个子匹配对象：第一个匹配整个字符串，第二个匹配第一个捕获组（本地部分），第三个匹配第二个捕获组（主机名），第四个匹配第三个和最后一个捕获组（DNS 标签）。结果以元组的形式返回，其中第一个项目实际上表示成功或失败：

```cpp
    std::tuple<bool, std::string, std::string, std::string>
    is_valid_email_format_with_result(std::string const & email) 
    { 
      auto rx = std::regex{  
        R"(^([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,})$)"s,  
        std::regex_constants::icase }; 
      auto result = std::smatch{}; 
      auto success = std::regex_match(email, result, rx); 

      return std::make_tuple( 
        success,  
        success ? result[1].str() : ""s, 
        success ? result[2].str() : ""s,  
        success ? result[3].str() : ""s); 
    }
```

在上述代码之后，我们使用 C++17 结构化绑定将元组的内容解包到命名变量中：

```cpp
    auto ltest3 = [](std::string const & email) 
    { 
      auto [valid, localpart, hostname, dnslabel] =  
       is_valid_email_format_with_result(email); 

      std::cout << std::setw(30) << std::left 
         << email << " : " 
         << std::setw(10) << (valid ? "valid" : "invalid") 
         << "local=" << localpart  
         << ";domain=" << hostname  
         << ";dns=" << dnslabel 
         << std::endl; 
    }; 

    ltest3("JOHN.DOE@DOMAIN.COM"s); 
    ltest3("JOHNDOE@DOMAIL.CO.UK"s); 
    ltest3("JOHNDOE@DOMAIL.INFO"s); 
    ltest3("J.O.H.N_D.O.E@DOMAIN.INFO"s); 
    ltest3("ROOT@LOCALHOST"s); 
    ltest3("john.doe@domain.com"s);
```

程序的输出将如下所示：

```cpp
 JOHN.DOE@DOMAIN.COM            : valid 
 local=JOHN.DOE;domain=DOMAIN;dns=COM 
 JOHNDOE@DOMAIL.CO.UK           : valid 
 local=JOHNDOE;domain=DOMAIL.CO;dns=UK 
 JOHNDOE@DOMAIL.INFO            : valid 
 local=JOHNDOE;domain=DOMAIL;dns=INFO 
 J.O.H.N_D.O.E@DOMAIN.INFO      : valid 
 local=J.O.H.N_D.O.E;domain=DOMAIN;dns=INFO 
 ROOT@LOCALHOST                 : invalid 
 local=;domain=;dns= 
 john.doe@domain.com            : valid 
 local=john.doe;domain=domain;dns=com
```

# 还有更多...

正则表达式有多个版本，C++标准库支持其中的六个：ECMAScript，基本 POSIX，扩展 POSIX，awk，grep 和 egrep（带有选项`-E`的 grep）。默认使用的语法是 ECMAScript，为了使用其他语法，您必须在定义正则表达式时显式指定语法。除了指定语法，还可以指定解析选项，例如忽略大小写匹配。

标准库提供的类和算法比我们迄今所见的更多。库中提供的主要类如下（它们都是类模板，为方便起见，为不同的字符类型提供了`typedef`）：

+   类模板`std::basic_regex`定义了正则表达式对象：

```cpp
        typedef basic_regex<char>    regex; 
        typedef basic_regex<wchar_t> wregex;
```

+   类模板`std::sub_match`表示与捕获组匹配的字符序列；这个类实际上是从`std::pair`派生出来的，它的`first`和`second`成员表示匹配序列中第一个和最后一个字符的迭代器；如果没有匹配序列，则这两个迭代器是相等的：

```cpp
        typedef sub_match<const char *>            csub_match; 
        typedef sub_match<const wchar_t *>         wcsub_match; 
        typedef sub_match<string::const_iterator>  ssub_match; 
        typedef sub_match<wstring::const_iterator> wssub_match;
```

+   类模板`std::match_results`是匹配的集合；第一个元素始终是目标中的完全匹配，其他元素是子表达式的匹配：

```cpp
        typedef match_results<const char *>            cmatch; 
        typedef match_results<const wchar_t *>         wcmatch; 
        typedef match_results<string::const_iterator>  smatch; 
        typedef match_results<wstring::const_iterator> wsmatch;
```

正则表达式标准库中可用的算法如下：

+   `std::regex_match()`: 这尝试将正则表达式（由`std::basic_regex`实例表示）与整个字符串匹配。

+   `std::regex_search()`: 这尝试将正则表达式（由`std::basic_regex`实例表示）与字符串的一部分（包括整个字符串）匹配。

+   `std::regex_replace()`: 这根据指定的格式替换正则表达式的匹配项。

正则表达式标准库中可用的迭代器如下：

+   `std::regex_interator`：用于遍历字符串中模式出现的常量前向迭代器。它有一个指向`std::basic_regex`的指针，必须存活到迭代器被销毁。在创建和递增时，迭代器调用`std::regex_search()`并存储算法返回的`std::match_results`对象的副本。

+   `std::regex_token_iterator`：用于遍历字符串中正则表达式的每个匹配的子匹配的常量前向迭代器。在内部，它使用`std::regex_iterator`来遍历子匹配。由于它存储指向`std::basic_regex`实例的指针，因此正则表达式对象必须存活到迭代器被销毁。

# 另请参阅

+   *使用正则表达式解析字符串的内容*

+   *使用正则表达式替换字符串的内容*

+   *使用结构化绑定处理多返回值* 第八章的示例，*学习现代核心语言特性*

# 使用正则表达式解析字符串的内容

在前面的示例中，我们已经看到如何使用`std::regex_match()`来验证字符串的内容是否与特定格式匹配。库提供了另一个名为`std::regex_search()`的算法，它可以匹配字符串的任何部分，而不仅仅是整个字符串，如`regex_match()`所做的那样。然而，这个函数不允许在输入字符串中搜索所有正则表达式的出现。为此，我们需要使用库中可用的迭代器类之一。

在这个示例中，您将学习如何使用正则表达式解析字符串的内容。为此，我们将考虑解析包含名称-值对的文本文件的问题。每个这样的对在不同行上定义，格式为`name = value`，但以`#`开头的行表示注释，必须被忽略。以下是一个例子：

```cpp
    #remove # to uncomment the following lines 
    timeout=120 
    server = 127.0.0.1 

    #retrycount=3
```

# 准备工作

有关 C++11 中正则表达式支持的一般信息，请参阅*使用正则表达式验证字符串的格式*示例。需要基本的正则表达式知识才能继续进行这个示例。

在以下示例中，`text`是一个变量，定义如下：

```cpp
    auto text { 
      R"( 
        #remove # to uncomment the following lines 
        timeout=120 
        server = 127.0.0.1 

        #retrycount=3 
      )"s};
```

# 如何做...

为了搜索字符串中正则表达式的出现，您应该执行以下操作：

1.  包括头文件`<regex>`和`<string>`以及命名空间`std::string_literals`，用于 C++14 标准用户定义的字符串字面量：

```cpp
        #include <regex> 
        #include <string> 
        using namespace std::string_literals;
```

1.  使用原始字符串字面量指定正则表达式，以避免转义反斜杠（这可能经常发生）。以下正则表达式验证了先前提出的文件格式：

```cpp
        auto pattern {R"(^(?!#)(\w+)\s*=\s*([\w\d]+[\w\d._,\-:]*)$)"s};
```

1.  创建一个`std::regex`/`std::wregex`对象（取决于所使用的字符集）来封装正则表达式：

```cpp
        auto rx = std::regex{pattern};
```

1.  要在给定文本中搜索正则表达式的第一个匹配项，使用通用算法`std::regex_search()`（示例 1）：

```cpp
        auto match = std::smatch{}; 
        if (std::regex_search(text, match, rx)) 
        { 
          std::cout << match[1] << '=' << match[2] << std::endl; 
        }
```

1.  要在给定文本中查找正则表达式的所有出现，使用迭代器`std::regex_iterator`（示例 2）：

```cpp
        auto end = std::sregex_iterator{}; 
        for (auto it=std::sregex_iterator{ std::begin(text),  
                                           std::end(text), rx }; 
             it != end; ++it) 
        { 
          std::cout << ''' << (*it)[1] << "'='"  
                    << (*it)[2] << ''' << std::endl; 
        }
```

1.  要遍历匹配的所有子表达式，请使用迭代器`std::regex_token_iterator`（示例 3）：

```cpp
        auto end = std::sregex_token_iterator{}; 
        for (auto it = std::sregex_token_iterator{ 
                          std::begin(text),  std::end(text), rx }; 
             it != end; ++it) 
        { 
          std::cout << *it << std::endl; 
        }
```

# 工作原理...

一个简单的正则表达式，可以解析之前显示的输入文件，可能看起来像这样：

```cpp
    ^(?!#)(\w+)\s*=\s*([\w\d]+[\w\d._,\-:]*)$
```

这个正则表达式应该忽略所有以`#`开头的行；对于不以`#`开头的行，匹配一个名称，后面跟着一个等号，然后是由字母数字字符和几个其他字符（下划线、点、逗号等）组成的值。这个正则表达式的确切含义如下所述：

| **部分** | **描述** |
| --- | --- |
| `^` | 行首 |
| `(?!#)` | 负向先行断言，确保不可能匹配#字符。 |
| `(\w)+` | 代表至少一个单词字符的捕获组 |
| `\s*` | 任何空格 |
| `=` | 等号 |
| `\s*` | 任何空格 |
| `([\w\d]+[\w\d._,\-:]*)` | 代表以字母数字字符开头的值的捕获组，但也可以包含点、逗号、反斜杠、连字符、冒号或下划线。 |
| `$` | 行尾 |

我们可以使用`std::regex_search()`在输入文本中搜索匹配项。这个算法有几个重载，但一般来说它们的工作方式相同。您必须指定要处理的字符范围，一个输出`std::match_results`对象，它将包含匹配的结果，以及表示正则表达式和匹配标志的`std::basic_regex`对象（定义搜索方式）。如果找到了匹配项，函数返回`true`，否则返回`false`。

在前一节的第一个示例中（参见第 4 个列表项），`match`是`std::smatch`的一个实例，它是`std::match_results`的`typedef`，模板类型为`string::const_iterator`。如果找到了匹配项，这个对象将包含所有匹配子表达式的一系列值的匹配信息。索引为 0 的子匹配始终是整个匹配。索引为 1 的子匹配是第一个匹配的子表达式，索引为 2 的子匹配是第二个匹配的子表达式，依此类推。由于我们的正则表达式中有两个捕获组（即子表达式），所以在成功的情况下，`std::match_results`将有三个子匹配。表示名称的标识符在索引 1 处，等号后面的值在索引 2 处。因此，这段代码只打印以下内容：

```cpp
 timeout=120
```

`std::regex_search()`算法无法遍历文本中所有可能的匹配项。为了做到这一点，我们需要使用迭代器。`std::regex_iterator`用于此目的。它不仅允许遍历所有匹配项，还允许访问匹配项的所有子匹配项。迭代器实际上在构造时调用`std::regex_search()`，并在每次递增时记住调用的结果`std::match_results`。默认构造函数创建一个表示序列末尾的迭代器，可用于测试何时应该停止遍历匹配项的循环。

在前一节的第二个示例中（参见第 5 个列表项），我们首先创建一个序列结束迭代器，然后开始遍历所有可能的匹配项。在构造时，它将调用`std::regex_match()`，如果找到匹配项，我们可以通过当前迭代器访问其结果。这将一直持续，直到找不到匹配项（序列结束）。这段代码将打印以下输出：

```cpp
 'timeout'='120' 
 'server'='127.0.0.1'
```

`std::regex_iterator`的替代方法是`std::regex_token_iterator`。它的工作方式类似于`std::regex_iterator`的工作方式，并且实际上在内部包含这样一个迭代器，只是它使我们能够访问匹配的特定子表达式。这在*如何做...*部分的第三个示例中（第 6 个列表项）中显示。我们首先创建一个序列末尾的迭代器，然后循环遍历匹配，直到达到序列末尾。在我们使用的构造函数中，我们没有指定通过迭代器访问的子表达式的索引；因此，将使用默认值 0。这意味着此程序将打印整个匹配：

```cpp
 timeout=120 
 server = 127.0.0.1
```

如果我们只想访问第一个子表达式（在我们的情况下是名称），我们只需要在令牌迭代器的构造函数中指定子表达式的索引。这次，我们得到的输出只有名称：

```cpp
    auto end = std::sregex_token_iterator{}; 
    for (auto it = std::sregex_token_iterator{ std::begin(text),  
                   std::end(text), rx, 1 }; 
         it != end; ++it) 
    { 
      std::cout << *it << std::endl; 
    }
```

关于令牌迭代器的一个有趣之处是，如果子表达式的索引为`-1`，它可以返回字符串的未匹配部分，此时它返回一个与最后匹配和序列末尾之间的字符序列相对应的`std::match_results`对象：

```cpp
    auto end = std::sregex_token_iterator{}; 
    for (auto it = std::sregex_token_iterator{ std::begin(text),  
                   std::end(text), rx, -1 }; 
         it != end; ++it) 
    { 
      std::cout << *it << std::endl; 
    }
```

该程序将输出以下内容（请注意，空行实际上是输出的一部分）：

```cpp

 #remove # to uncomment the following lines 

 #retrycount=3
```

# 另请参阅

+   *使用正则表达式验证字符串格式*

+   *使用正则表达式替换字符串的内容*

# 使用正则表达式替换字符串的内容

在最后两个示例中，我们已经看到如何在字符串或字符串的一部分上匹配正则表达式，并遍历匹配和子匹配。正则表达式库还支持基于正则表达式的文本替换。在本示例中，我们将看到如何使用`std::regex_replace()`执行此类文本转换。

# 准备工作

关于 C++11 中正则表达式支持的一般信息，请参考*使用正则表达式验证字符串格式*的示例。

# 如何做...

为了使用正则表达式执行文本转换，您应该执行以下操作：

1.  包括`<regex>`和`<string>`，以及命名空间`std::string_literals`，用于 C++14 标准用户定义的字符串字面量： 

```cpp
        #include <regex> 
        #include <string> 
        using namespace std::string_literals;
```

1.  使用`std::regex_replace()`算法，并将替换字符串作为第三个参数。考虑以下示例：用三个连字符替换由`a`、`b`或`c`组成的恰好三个字符的所有单词：

```cpp
        auto text{"abc aa bca ca bbbb"s}; 
        auto rx = std::regex{ R"(\b[a|b|c]{3}\b)"s }; 
        auto newtext = std::regex_replace(text, rx, "---"s);
```

1.  使用`std::regex_replace()`算法，并在第三个参数中使用以`$`为前缀的匹配标识符。例如，将“姓，名”中的名替换为“名 姓”，如下所示：

```cpp
        auto text{ "bancila, marius"s }; 
        auto rx = std::regex{ R"((\w+),\s*(\w+))"s }; 
        auto newtext = std::regex_replace(text, rx, "$2 $1"s);
```

# 工作原理...

`std::regex_replace()`算法有几个重载，具有不同类型的参数，但参数的含义如下：

+   执行替换的输入字符串。

+   封装了用于标识要替换的字符串部分的正则表达式的`std::basic_regex`对象。

+   用于替换的字符串格式。

+   可选的匹配标志。

返回值取决于使用的重载，可以是字符串，也可以是作为参数提供的输出迭代器的副本。用于替换的字符串格式可以是简单字符串，也可以是以`$`前缀表示的匹配标识符：

+   `$&`表示整个匹配。

+   `$1`，`$2`，`$3`等表示第一个、第二个、第三个子匹配等。

+   `$``表示第一个匹配前的字符串部分。

+   `$'`表示最后匹配后的字符串部分。

在*如何做...*部分所示的第一个例子中，初始文本包含由恰好三个`a`、`b`或`c`字符组成的两个单词，`abc`和`bca`。正则表达式指示在单词边界之间恰好有三个字符的表达式。这意味着一个子文本，比如`bbbb`，将不会匹配该表达式。替换的结果是字符串文本将会是`--- aa --- ca bbbb`。

可以为`std::regex_replace()`算法指定匹配的附加标志。默认情况下，匹配标志是`std::regex_constants::match_default`，基本上指定了 ECMAScript 作为用于构造正则表达式的语法。例如，如果我们想要只替换第一次出现的匹配，那么我们可以指定`std::regex_constants::format_first_only`。在下一个例子中，结果是`--- aa bca ca bbbb`，因为替换在找到第一个匹配后停止。

```cpp
    auto text{ "abc aa bca ca bbbb"s }; 
    auto rx = std::regex{ R"(\b[a|b|c]{3}\b)"s }; 
    auto newtext = std::regex_replace(text, rx, "---"s, 
                     std::regex_constants::format_first_only);
```

然而，替换字符串可以包含特殊指示符，用于整个匹配、特定子匹配，或者未匹配的部分，如前面所解释的。在*如何做...*部分所示的第二个例子中，正则表达式识别至少一个字符的单词，后面跟着逗号和可能的空格，然后是另一个至少一个字符的单词。第一个单词应该是姓，第二个单词应该是名。替换字符串采用`$2 $1`格式。这是一个指令，用另一个字符串替换匹配的表达式（在这个例子中，整个原始字符串），由第二个子匹配后跟一个空格，然后是第一个子匹配。

在这种情况下，整个字符串都是一个匹配。在下一个例子中，字符串内将有多个匹配，并且它们都将被替换为指定的字符串。在这个例子中，我们替换了在元音字母开头的单词之前的不定冠词*a*为不定冠词*an*（当然，这并不包括以元音音素开头的单词）：

```cpp
    auto text{"this is a example with a error"s}; 
    auto rx = std::regex{R"(\ba ((a|e|i|u|o)\w+))"s}; 
    auto newtext = std::regex_replace(text, rx, "an $1");
```

正则表达式将字母*a*识别为一个单词（`\b`表示单词边界，所以`\ba`表示一个只有一个字母*a*的单词），后面跟着一个空格和至少以元音字母开头的至少两个字符的单词。当识别到这样的匹配时，它将被替换为一个由固定字符串*an*后跟一个空格和匹配的第一个子表达式（即单词本身）组成的字符串。在这个例子中，`newtext`字符串将是*this is an example with an error*。

除了子表达式的标识符（`$1`，`$2`等），还有其他标识符用于整个匹配（`$&`），第一个匹配之前的字符串部分（`$``），以及最后一个匹配之后的字符串部分（`$'`）。在最后一个例子中，我们改变了日期的格式从`dd.mm.yyyy`到`yyyy.mm.dd`，同时显示了匹配的部分。

```cpp
    auto text{"today is 1.06.2016!!"s}; 
    auto rx =  
       std::regex{R"((\d{1,2})(\.|-|/)(\d{1,2})(\.|-|/)(\d{4}))"s};       
    // today is 2016.06.1!! 
    auto newtext1 = std::regex_replace(text, rx, R"($5$4$3$2$1)"); 
    // today is [today is ][1.06.2016][!!]!! 
    auto newtext2 = std::regex_replace(text, rx, R"([$`][$&][$'])");
```

正则表达式匹配一个或两位数字，后面跟着一个点、连字符或斜杠；然后是另一个一位或两位数字；然后是一个点、连字符或斜杠；最后是四位数字。

对于`newtext1`，替换字符串是`$5$4$3$2$1`；这意味着年份，后面是第二个分隔符，然后是月份，第一个分隔符，最后是日期。因此，对于输入字符串*"today is 1.06.2016!"*，结果是*"today is 2016.06.1!!"*。

对于`newtext2`，替换字符串是`[$`][$&][$']`；这意味着第一个匹配之前的部分，后面跟着整个匹配，最后是最后一个匹配之后的部分都在方括号中。然而，结果并不是*"[!!][1.06.2016][today is ]"*，这可能是你第一眼期望的，而是*"today is [today is ][1.06.2016][!!]!!"*。原因是被替换的是匹配的表达式，在这种情况下，那只是日期（*"1.06.2016"*）。这个子字符串被另一个字符串替换，由初始字符串的所有部分组成。

# 另请参阅

+   *使用正则表达式验证字符串的格式*

+   *使用正则表达式解析字符串的内容*

# 使用`string_view`代替常量字符串引用

在处理字符串时，临时对象经常被创建，即使你可能并不真正意识到。许多时候，临时对象是无关紧要的，只是为了将数据从一个地方复制到另一个地方（例如，从函数到其调用者）。这代表了一个性能问题，因为它们需要内存分配和数据复制，这是希望避免的。为此，C++17 标准提供了一个名为`std::basic_string_view`的新字符串类模板，它表示对字符串（即字符序列）的非拥有常量引用。在这个示例中，你将学习何时以及如何使用这个类。

# 准备工作

`string_view`类在`string_view`头文件中的`std`命名空间中可用。

# 如何做...

应该使用`std::string_view`来传递参数给函数（或者从函数返回值），而不是`std::string const &`，除非你的代码需要调用其他需要`std::string`参数的函数（在这种情况下，需要进行转换）：

```cpp
    std::string_view get_filename(std::string_view str) 
    { 
      auto const pos1 {str.find_last_of('')}; 
      auto const pos2 {str.find_last_of('.')}; 
      return str.substr(pos1 + 1, pos2 - pos1 - 1); 
    } 

    char const file1[] {R"(c:\test\example1.doc)"}; 
    auto name1 = get_filename(file1); 

    std::string file2 {R"(c:\test\example2)"}; 
    auto name2 = get_filename(file2); 

    auto name3 = get_filename(std::string_view{file1, 16});
```

# 它是如何工作的...

在我们看新字符串类型如何工作之前，让我们考虑下面的一个函数的例子，该函数应该提取没有扩展名的文件名。这基本上是在 C++17 之前你会如何编写前一节中的函数。

请注意，在这个例子中，文件分隔符是`\`（反斜杠），就像在 Windows 中一样。对于基于 Linux 的系统，它必须更改为`/`（斜杠）。

```cpp
    std::string get_filename(std::string const & str) 
    { 
      auto const pos1 {str.find_last_of('')}; 
      auto const pos2 {str.find_last_of('.')}; 
      return str.substr(pos1 + 1, pos2 - pos1 - 1); 
    } 

    auto name1 = get_filename(R"(c:\test\example1.doc)"); // example1 
    auto name2 = get_filename(R"(c:\test\example2)");     // example2 
    if(get_filename(R"(c:\test\_sample_.tmp)").front() == '_') {}
```

这是一个相对简单的函数。它接受一个`std::string`的常量引用，并识别由最后一个文件分隔符和最后一个点界定的子字符串，基本上表示一个没有扩展名（也没有文件夹名称）的文件名。

然而，这段代码的问题在于，它创建了一个、两个，甚至可能更多的临时对象，这取决于编译器的优化。函数参数是一个常量`std::string`引用，但函数被调用时使用了一个字符串字面值，这意味着需要从字面值构造`std::string`。这些临时对象需要分配和复制数据，这既耗时又消耗资源。在最后一个例子中，我们只想检查文件名的第一个字符是否是下划线，但为此我们至少创建了两个临时字符串对象。

`std::basic_string_view`类模板旨在解决这个问题。这个类模板与`std::basic_string`非常相似，两者几乎具有相同的接口。原因是`std::basic_string_view`旨在用来代替对`std::basic_string`的常量引用，而不需要进一步的代码更改。

就像`std::basic_string`一样，对于所有类型的标准字符都有特殊化：

```cpp
    typedef basic_string_view<char>     string_view; 
    typedef basic_string_view<wchar_t>  wstring_view; 
    typedef basic_string_view<char16_t> u16string_view; 
    typedef basic_string_view<char32_t> u32string_view;
```

`std::basic_string_view`类模板定义了对字符的一个常量连续序列的引用。顾名思义，它表示一个视图，不能用于修改字符的引用序列。一个`std::basic_string_view`对象的大小相对较小，因为它所需的只是指向序列中第一个字符的指针和长度。它不仅可以从`std::basic_string`对象构造，还可以从指针和长度或者以空字符结尾的字符序列构造（在这种情况下，它将需要对字符串进行初始遍历以找到长度）。因此，`std::basic_string_view`类模板也可以用作多种类型字符串的通用接口（只要数据只需要被读取）。另一方面，从`std::basic_string_view`转换为`std::basic_string`很容易，因为前者既有`to_string()`又有一个转换的`operator std::basic_string`来创建一个新的`std::basic_string`对象。

将`std::basic_string_view`传递给函数并返回`std::basic_string_view`仍然会创建这种类型的临时对象，但这些对象在堆栈上是小型对象（对于 64 位平台，指针和大小可能为 16 字节）；因此，它们应该比分配堆空间和复制数据产生更少的性能成本。

请注意，所有主要的编译器都提供了 std::basic_string 的实现，其中包括小字符串优化。尽管实现细节不同，但它们通常依赖于具有静态分配的字符数（对于 VC++和 gcc 5 或更新版本为 16）的缓冲区，不涉及堆操作，只有在字符串的大小超过该字符数时才需要堆操作。

除了与`std::basic_string`中可用的相同的方法之外，`std::basic_string_view`还有两个：

+   `remove_prefix()`: 通过增加*N*个字符来缩小视图的起始位置，并通过减少*N*个字符来缩小长度。

+   `remove_suffix()`: 通过减少*N*个字符来缩小视图的长度。

以下示例中使用这两个成员函数来修剪`std::string_view`中的空格，无论是在开头还是结尾。函数的实现首先查找第一个不是空格的元素，然后查找最后一个不是空格的元素。然后，它从末尾删除最后一个非空格字符之后的所有内容，并从开头删除第一个非空格字符之前的所有内容。函数返回修剪后的新视图：

```cpp
    std::string_view trim_view(std::string_view str) 
    { 
      auto const pos1{ str.find_first_not_of(" ") }; 
      auto const pos2{ str.find_last_not_of(" ") }; 
      str.remove_suffix(str.length() - pos2 - 1); 
      str.remove_prefix(pos1); 

      return str; 
    } 

    auto sv1{ trim_view("sample") }; 
    auto sv2{ trim_view("  sample") }; 
    auto sv3{ trim_view("sample  ") }; 
    auto sv4{ trim_view("  sample  ") }; 

    auto s1{ sv1.to_string() }; 
    auto s2{ sv2.to_string() }; 
    auto s3{ sv3.to_string() }; 
    auto s4{ sv4.to_string() };
```

在使用`std::basic_string_view`时，您必须注意两件事：您不能更改视图引用的基础数据，必须管理数据的生命周期，因为视图是一个非拥有引用。

# 另见

+   *创建字符串助手库*
