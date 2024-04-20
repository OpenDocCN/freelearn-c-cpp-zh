# 语言特性

# 问题

这是本章的问题解决部分。

# 15. IPv4 数据类型

编写一个表示 IPv4 地址的类。实现所需的函数，以便能够从控制台读取和写入这些地址。用户应该能够以点分形式输入值，例如`127.0.0.1`或`168.192.0.100`。这也是 IPv4 地址应该格式化为输出流的形式。

# 16. 在范围内枚举 IPv4 地址

编写一个程序，允许用户输入表示范围的两个 IPv4 地址，并列出该范围内的所有地址。扩展为前一个问题定义的结构以实现所请求的功能。

# 17. 创建具有基本操作的 2D 数组

编写一个表示具有元素访问（`at()`和`data()`）、容量查询、迭代器、填充和交换方法的二维数组容器的类模板。应该可以移动此类型的对象。

# 18. 具有任意数量参数的最小函数

编写一个函数模板，可以接受任意数量的参数，并使用`operator <`进行比较返回它们所有的最小值。编写此函数模板的变体，可以使用二进制比较函数进行参数化，而不是使用`operator <`。

# 19. 将一系列值添加到容器中

编写一个通用函数，可以将任意数量的元素添加到具有`push_back(T&& value)`方法的容器的末尾。

# 20. 容器任何、全部、无

编写一组通用函数，使其能够检查给定容器中是否存在任何、全部或任何指定参数。这些函数应该使得能够编写以下代码成为可能：

```cpp
std::vector<int> v{ 1, 2, 3, 4, 5, 6 };
assert(contains_any(v, 0, 3, 30));

std::array<int, 6> a{ { 1, 2, 3, 4, 5, 6 } };
assert(contains_all(a, 1, 3, 5, 6));

std::list<int> l{ 1, 2, 3, 4, 5, 6 };
assert(!contains_none(l, 0, 6));
```

# 21. 系统句柄包装器

考虑一个操作系统句柄，例如文件句柄。编写一个包装器，处理句柄的获取和释放，以及其他操作，如验证句柄的有效性和从一个对象移动句柄所有权。

# 22. 各种温度标度的文字

编写一个小型库，使得能够以三种最常用的标度（摄氏度、华氏度和开尔文）表示温度，并在它们之间进行转换。该库必须使您能够以所有这些标度编写温度文字，例如`36.5_deg`表示摄氏度，`97.7_f`表示华氏度，`309.65_K`表示开尔文；对这些值执行操作；并在它们之间进行转换。

# 解决方案

以下是上述问题解决部分的解决方案。

# 15. IPv4 数据类型

该问题要求编写一个类来表示 IPv4 地址。这是一个 32 位值，通常以十进制点格式表示，例如`168.192.0.100`；它的每个部分都是一个 8 位值，范围从 0 到 255。为了方便表示和处理，我们可以使用四个`unsigned char`来存储地址值。这样的值可以从四个`unsigned char`或从一个`unsigned long`构造。为了能够直接从控制台（或任何其他输入流）读取值，并能够将值写入控制台（或任何其他输出流），我们必须重载`operator>>`和`operator<<`。以下清单显示了可以满足所请求功能的最小实现：

```cpp
class ipv4
{
   std::array<unsigned char, 4> data;
public:
   constexpr ipv4() : data{ {0} } {}
   constexpr ipv4(unsigned char const a, unsigned char const b, 
                  unsigned char const c, unsigned char const d):
      data{{a,b,c,d}} {}
   explicit constexpr ipv4(unsigned long a) :
      data{ { static_cast<unsigned char>((a >> 24) & 0xFF), 
              static_cast<unsigned char>((a >> 16) & 0xFF),
              static_cast<unsigned char>((a >> 8) & 0xFF),
              static_cast<unsigned char>(a & 0xFF) } } {}
   ipv4(ipv4 const & other) noexcept : data(other.data) {}
   ipv4& operator=(ipv4 const & other) noexcept 
   {
      data = other.data;
      return *this;
   }

   std::string to_string() const
   {
      std::stringstream sstr;
      sstr << *this;
      return sstr.str();
   }

   constexpr unsigned long to_ulong() const noexcept
   {
      return (static_cast<unsigned long>(data[0]) << 24) |
             (static_cast<unsigned long>(data[1]) << 16) |
             (static_cast<unsigned long>(data[2]) << 8) |
              static_cast<unsigned long>(data[3]);
   }

   friend std::ostream& operator<<(std::ostream& os, const ipv4& a)
   {
      os << static_cast<int>(a.data[0]) << '.' 
         << static_cast<int>(a.data[1]) << '.'
         << static_cast<int>(a.data[2]) << '.'
         << static_cast<int>(a.data[3]);
      return os;
   }

   friend std::istream& operator>>(std::istream& is, ipv4& a)
   {
      char d1, d2, d3;
      int b1, b2, b3, b4;
      is >> b1 >> d1 >> b2 >> d2 >> b3 >> d3 >> b4;
      if (d1 == '.' && d2 == '.' && d3 == '.')
         a = ipv4(b1, b2, b3, b4);
      else
         is.setstate(std::ios_base::failbit);
      return is;
   }
};
```

`ipv4`类可以如下使用：

```cpp
int main()
{
   ipv4 address(168, 192, 0, 1);
   std::cout << address << std::endl;

   ipv4 ip;
   std::cout << ip << std::endl;
   std::cin >> ip;
   if(!std::cin.fail())
      std::cout << ip << std::endl;
}
```

# 16. 在范围内枚举 IPv4 地址

为了能够在给定范围内枚举 IPv4 地址，首先应该能够比较 IPv4 值。因此，我们应该至少实现`operator<`，但以下清单包含所有比较运算符的实现：`==`、`!=`、`<`、`>`、`<=`和`>=`。此外，为了增加 IPv4 值，提供了前缀和后缀`operator++`的实现。以下代码是前一个问题中 IPv4 类的扩展：

```cpp
ipv4& operator++()
{
   *this = ipv4(1 + to_ulong());
   return *this;
}

ipv4& operator++(int)
{
   ipv4 result(*this);
   ++(*this);
   return *this;
}

friend bool operator==(ipv4 const & a1, ipv4 const & a2) noexcept
{
   return a1.data == a2.data;
}

friend bool operator!=(ipv4 const & a1, ipv4 const & a2) noexcept
{
   return !(a1 == a2);
}

friend bool operator<(ipv4 const & a1, ipv4 const & a2) noexcept
{
   return a1.to_ulong() < a2.to_ulong();
}

friend bool operator>(ipv4 const & a1, ipv4 const & a2) noexcept
{
   return a2 < a1;
}

friend bool operator<=(ipv4 const & a1, ipv4 const & a2) noexcept
{
   return !(a1 > a2);
}

friend bool operator>=(ipv4 const & a1, ipv4 const & a2) noexcept
{
   return !(a1 < a2);
}
```

通过对前一个问题中的`ipv4`类进行这些更改，我们可以编写以下程序：

```cpp
int main()
{
   std::cout << "input range: ";
   ipv4 a1, a2;
   std::cin >> a1 >> a2;
   if (a2 > a1)
   {
      for (ipv4 a = a1; a <= a2; a++)
      {
         std::cout << a << std::endl;
      }
   }
   else 
   {
      std::cerr << "invalid range!" << std::endl;
   }
}
```

# 17. 创建具有基本操作的 2D 数组

在看如何定义这样的结构之前，让我们考虑一下它的几个测试用例。以下片段显示了所有请求的功能：

```cpp
int main()
{
   // element access
   array2d<int, 2, 3> a {1, 2, 3, 4, 5, 6};
   for (size_t i = 0; i < a.size(1); ++i)
      for (size_t j = 0; j < a.size(2); ++j)
      a(i, j) *= 2;

   // iterating
   std::copy(std::begin(a), std::end(a), 
      std::ostream_iterator<int>(std::cout, " "));

   // filling 
   array2d<int, 2, 3> b;
   b.fill(1);

   // swapping
   a.swap(b);

   // moving
   array2d<int, 2, 3> c(std::move(b));
}
```

请注意，对于元素访问，我们使用`operator()`，比如`a(i,j)`，而不是`operator[]`，比如`a[i][j]`，因为只有前者可以接受多个参数（每个维度的索引）。后者只能有一个参数，并且为了使表达式`a[i][j]`有效，它必须返回一个中间类型（基本上表示一行），然后再重载`operator[]`以返回单个元素。

已经有存储固定或可变长度元素序列的标准容器。这个二维数组类应该只是这样一个容器的适配器。在选择`std::array`和`std::vector`之间，我们应该考虑两件事：

+   `array2d`类应该具有移动语义，以便能够移动对象

+   应该可以使用列表初始化此类型的对象

`std::array`容器只有在其持有的元素是可移动构造和可移动分配时才可移动。另一方面，它不能从`std::initializer_list`构造。因此，更可行的选择仍然是`std::vector`。

在内部，此适配器容器可以将其数据存储在向量的向量中（每行是一个具有`C`个元素的`vector<T>`，而 2D 数组中有`R`个这样的元素存储在`vector<vector<T>>`中）或者类型为`T`的`R![](img/2f9ae4c1-380b-4377-84dd-a28429c062c5.png)C`元素的单个向量中。在后一种情况下，第`i`行和第`j`列的元素位于索引`i * C + j`处。这种方法具有较小的内存占用，将所有数据存储在单个连续块中，并且实现起来也更简单。因此，这是首选解决方案的原因。

这里展示了具有所请求功能的二维数组类的可能实现：

```cpp
template <class T, size_t R, size_t C>
class array2d
{
   typedef T                 value_type;
   typedef value_type*       iterator;
   typedef value_type const* const_iterator;
   std::vector<T>            arr;
public:
   array2d() : arr(R*C) {}
   explicit array2d(std::initializer_list<T> l):arr(l) {}
   constexpr T* data() noexcept { return arr.data(); }
   constexpr T const * data() const noexcept { return arr.data(); }

   constexpr T& at(size_t const r, size_t const c) 
   {
      return arr.at(r*C + c);
   }

   constexpr T const & at(size_t const r, size_t const c) const
   {
      return arr.at(r*C + c);
   }

   constexpr T& operator() (size_t const r, size_t const c)
   {
      return arr[r*C + c];
   }

   constexpr T const & operator() (size_t const r, size_t const c) const
   {
      return arr[r*C + c];
   }

   constexpr bool empty() const noexcept { return R == 0 || C == 0; }

   constexpr size_t size(int const rank) const
   {
      if (rank == 1) return R;
      else if (rank == 2) return C;
      throw std::out_of_range("Rank is out of range!");
   }

   void fill(T const & value)
   {
      std::fill(std::begin(arr), std::end(arr), value);
   }

   void swap(array2d & other) noexcept { arr.swap(other.arr); }

   const_iterator begin() const { return arr.data(); }
   const_iterator end() const   { return arr.data() + arr.size(); }
   iterator       begin()       { return arr.data(); }
   iterator       end()         { return arr.data() + arr.size(); }
};
```

# 18\. 具有任意数量参数的最小函数

可以使用可变函数模板编写可以接受可变数量参数的函数模板。为此，我们需要实现编译时递归（实际上只是通过一组重载函数进行调用）。以下片段显示了如何实现所请求的函数：

```cpp
template <typename T>
T minimum(T const a, T const b) { return a < b ? a : b; }

template <typename T1, typename... T>
T1 minimum(T1 a, T... args)
{
   return minimum(a, minimum(args...));
}

int main()
{
   auto x = minimum(5, 4, 2, 3);
}
```

为了能够使用用户提供的二进制比较函数，我们需要编写另一个函数模板。比较函数必须是第一个参数，因为它不能跟随函数参数包。另一方面，这不能是前一个最小函数的重载，而是具有不同名称的函数。原因是编译器无法区分模板参数列表`<typename T1, typename... T>`和`<class Compare, typename T1, typename... T>`。更改很小，应该很容易在此片段中跟踪：

```cpp
template <class Compare, typename T>
T minimumc(Compare comp, T const a, T const b) 
{ return comp(a, b) ? a : b; }

template <class Compare, typename T1, typename... T>
T1 minimumc(Compare comp, T1 a, T... args)
{
   return minimumc(comp, a, minimumc(comp, args...));
}

int main()
{
   auto y = minimumc(std::less<>(), 3, 2, 1, 0);
}
```

# 19\. 向容器添加一系列值

使用可变函数模板可以编写具有任意数量参数的函数。该函数应该将容器作为第一个参数，然后是表示要添加到容器后面的值的可变数量的参数。但是，使用折叠表达式可以显着简化编写这样的函数模板。这里展示了这样的实现：

```cpp
template<typename C, typename... Args>
void push_back(C& c, Args&&... args)
{
   (c.push_back(args), ...);
}
```

可以在以下清单中看到使用此函数模板的各种容器类型的示例：

```cpp
int main()
{
   std::vector<int> v;
   push_back(v, 1, 2, 3, 4);
   std::copy(std::begin(v), std::end(v), 
             std::ostream_iterator<int>(std::cout, " "));

   std::list<int> l;
   push_back(l, 1, 2, 3, 4);
   std::copy(std::begin(l), std::end(l), 
             std::ostream_iterator<int>(std::cout, " "));
}
```

# 20\. 容器任何，全部，无

能够检查变量数量的存在或不存在的要求表明，我们应该编写可变函数模板。然而，这些函数需要一个辅助函数，一个通用的函数，用于检查元素是否在容器中找到，并返回一个`bool`来指示成功或失败。由于所有这些函数，我们可以称之为`contains_all`，`contains_any`和`contains_none`，都是对辅助函数返回的结果应用逻辑运算符，我们将使用折叠表达式来简化代码。在折叠表达式扩展后启用短路评估，这意味着我们只评估导致明确结果的元素。因此，如果我们正在寻找所有 1、2 和 3 的存在，并且 2 缺失，那么在查找容器中的值 2 时，函数将返回而不检查值 3：

```cpp
template<class C, class T>
bool contains(C const & c, T const & value)
{
   return std::end(c) != std::find(std::begin(c), std::end(c), value);
}

template<class C, class... T>
bool contains_any(C const & c, T &&... value)
{
   return (... || contains(c, value));
}

template<class C, class... T>
bool contains_all(C const & c, T &&... value)
{
   return (... && contains(c, value));
}

template<class C, class... T>
bool contains_none(C const & c, T &&... value)
{
   return !contains_any(c, std::forward<T>(value)...);
}
```

# 21. 系统句柄包装器

系统句柄是对系统资源的引用形式。因为所有操作系统最初至少是用 C 编写的，所以创建和释放句柄是通过专用系统函数完成的。这增加了因错误处理而导致资源泄漏的风险，例如在异常情况下。在下面的代码片段中，特定于 Windows，您可以看到一个函数，在该函数中打开文件，从中读取，并最终关闭。然而，这有一些问题：在一个情况下，开发人员忘记在离开函数之前关闭句柄；在另一种情况下，在句柄正确关闭之前调用了一个抛出异常的函数，而没有捕获异常。然而，由于函数抛出异常，清理代码永远不会执行：

```cpp
void bad_handle_example()
{
   bool condition1 = false;
   bool condition2 = true;
   HANDLE handle = CreateFile(L"sample.txt",
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              nullptr,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL,
                              nullptr);

   if (handle == INVALID_HANDLE_VALUE)
      return;

   if (condition1)
   {
      CloseHandle(handle);
      return;
   }

   std::vector<char> buffer(1024);
   unsigned long bytesRead = 0;
   ReadFile(handle, 
            buffer.data(), 
            buffer.size(), 
            &bytesRead, 
            nullptr);

   if (condition2)
   {
      // oops, forgot to close handle
      return;
   }

   // throws exception; the next line will not execute
   function_that_throws();

   CloseHandle(handle);
}
```

C++包装类可以确保在包装对象超出范围并被销毁时正确处理句柄（无论是通过正常执行路径还是作为异常的结果）。一个合适的实现应该考虑不同类型的句柄，以及一系列值来指示无效句柄（如 0/null 或-1）。下面显示的实现提供了：

+   在对象被销毁时显式获取和自动释放句柄

+   移动语义以实现句柄所有权的转移

+   比较运算符用于检查两个对象是否引用相同的句柄

+   其他操作，如交换和重置

这里展示的实现是 Kenny Kerr 实现的句柄类的修改版本，并发表在 2011 年 7 月的 MSDN 杂志文章*Windows with C++ - C++ and the Windows API*中，[`msdn.microsoft.com/en-us/magazine/hh288076.aspx`](https://msdn.microsoft.com/en-us/magazine/hh288076.aspx)。尽管这里显示的句柄特性是指 Windows 句柄，但编写适用于其他平台的特性应该是相当简单的。

```cpp
template <typename Traits>
class unique_handle
{
   using pointer = typename Traits::pointer;
   pointer m_value;
public:
   unique_handle(unique_handle const &) = delete;
   unique_handle& operator=(unique_handle const &) = delete;

   explicit unique_handle(pointer value = Traits::invalid()) noexcept
      :m_value{ value }
   {}

   unique_handle(unique_handle && other) noexcept
      : m_value{ other.release() }
   {}

   unique_handle& operator=(unique_handle && other) noexcept
   {
      if (this != &other)
         reset(other.release());
      return *this;
   }

   ~unique_handle() noexcept
   {
      Traits::close(m_value);
   }

   explicit operator bool() const noexcept
   {
      return m_value != Traits::invalid();
   }

   pointer get() const noexcept { return m_value; }

   pointer release() noexcept
   {
      auto value = m_value;
      m_value = Traits::invalid();
      return value;
   }

   bool reset(pointer value = Traits::invalid()) noexcept
   {
      if (m_value != value)
      {
         Traits::close(m_value);
         m_value = value;
      }
      return static_cast<bool>(*this);
   }

   void swap(unique_handle<Traits> & other) noexcept
   {
      std::swap(m_value, other.m_value);
   }
};

template <typename Traits>
void swap(unique_handle<Traits> & left, unique_handle<Traits> & right) noexcept
{
   left.swap(right);
}

template <typename Traits>
bool operator==(unique_handle<Traits> const & left,
                unique_handle<Traits> const & right) noexcept
{
   return left.get() == right.get();
}

template <typename Traits>
bool operator!=(unique_handle<Traits> const & left,
                unique_handle<Traits> const & right) noexcept
{
   return left.get() != right.get();
}

struct null_handle_traits
{
   using pointer = HANDLE;
   static pointer invalid() noexcept { return nullptr; }
   static void close(pointer value) noexcept
   {
      CloseHandle(value);
   }
};

struct invalid_handle_traits
{
   using pointer = HANDLE;
   static pointer invalid() noexcept { return INVALID_HANDLE_VALUE; }
   static void close(pointer value) noexcept
   {
      CloseHandle(value);
   }
};

using null_handle = unique_handle<null_handle_traits>;
using invalid_handle = unique_handle<invalid_handle_traits>;
```

有了这种句柄类型的定义，我们可以用更简单的术语重写先前的示例，避免所有那些因为异常而未正确关闭句柄的问题，这些异常发生时没有得到正确处理，或者仅仅是因为开发人员忘记在不再需要时释放资源。这段代码既更简单又更健壮：

```cpp
void good_handle_example()
{
   bool condition1 = false;
   bool condition2 = true;

   invalid_handle handle{
      CreateFile(L"sample.txt",
                 GENERIC_READ,
                 FILE_SHARE_READ,
                 nullptr,
                 OPEN_EXISTING,
                 FILE_ATTRIBUTE_NORMAL,
                 nullptr) };

   if (!handle) return;

   if (condition1) return;

   std::vector<char> buffer(1024);
   unsigned long bytesRead = 0;
   ReadFile(handle.get(),
            buffer.data(),
            buffer.size(),
            &bytesRead,
            nullptr);

   if (condition2) return;

   function_that_throws();
}
```

# 22. 各种温度标度的文字

为了满足这一要求，我们需要为多种类型、运算符和函数提供实现：

+   称为`scale`的支持温度标度的枚举。

+   一个类模板，用于表示温度值，参数化为`quantity`，称为`quantity`。

+   比较运算符`==`、`!=`、`<`、`>`、`<=`和`>=`，用于比较相同类型的两个数量。

+   算术运算符`+`和`-`用于添加和减去相同类型的值。此外，我们可以实现成员运算符`+=`和`-+`。

+   一个函数模板，用于将温度从一种标度转换为另一种，称为`temperature_cast`。这个函数本身不执行转换，而是使用类型特性来执行转换。

+   用于创建用户定义的温度字面量的文字操作符`""_deg`，`""_f`和`""_k`。

为了简洁起见，以下代码片段仅包含处理摄氏度和华氏度温度的代码。您应该将其视为进一步练习，以扩展代码以支持开尔文标度。附带书籍的代码包含了所有三个所需标度的完整实现。

`are_equal()`函数是一个用于比较浮点值的实用函数：

```cpp
bool are_equal(double const d1, double const d2, 
               double const epsilon = 0.001)
{
   return std::fabs(d1 - d2) < epsilon;
}
```

可能的温度标度的枚举和表示温度值的类定义如下：

```cpp
namespace temperature
{
   enum class scale { celsius, fahrenheit, kelvin };

   template <scale S>
   class quantity
   {
      const double amount;
   public:
      constexpr explicit quantity(double const a) : amount(a) {}
      explicit operator double() const { return amount; }
   };
}
```

`quantity<S>`类的比较操作符可以在这里看到：

```cpp
namespace temperature 
{
   template <scale S>
   inline bool operator==(quantity<S> const & lhs, quantity<S> const & rhs)
   {
      return are_equal(static_cast<double>(lhs), static_cast<double>(rhs));
   }

   template <scale S>
   inline bool operator!=(quantity<S> const & lhs, quantity<S> const & rhs)
   {
      return !(lhs == rhs);
   }

   template <scale S>
   inline bool operator< (quantity<S> const & lhs, quantity<S> const & rhs)
   {
      return static_cast<double>(lhs) < static_cast<double>(rhs);
   }

   template <scale S>
   inline bool operator> (quantity<S> const & lhs, quantity<S> const & rhs)
   {
      return rhs < lhs;
   }

   template <scale S>
   inline bool operator<=(quantity<S> const & lhs, quantity<S> const & rhs)
   {
      return !(lhs > rhs);
   }

   template <scale S>
   inline bool operator>=(quantity<S> const & lhs, quantity<S> const & rhs)
   {
      return !(lhs < rhs);
   }

   template <scale S>
   constexpr quantity<S> operator+(quantity<S> const &q1, 
                                   quantity<S> const &q2)
   {
      return quantity<S>(static_cast<double>(q1) + 
                         static_cast<double>(q2));
   }

   template <scale S>
   constexpr quantity<S> operator-(quantity<S> const &q1, 
                                   quantity<S> const &q2)
   {
      return quantity<S>(static_cast<double>(q1) - 
                         static_cast<double>(q2));
   }
}
```

为了在不同温度标度之间进行转换，我们将定义一个名为`temperature_cast()`的函数模板，该函数利用了几个类型特征来执行实际的转换。所有这些都在这里显示，尽管并非所有类型特征；其他类型特征可以在附带书籍的代码中找到：

```cpp
namespace temperature
{
   template <scale S, scale R>
   struct conversion_traits
   {
      static double convert(double const value) = delete;
   };

   template <>
   struct conversion_traits<scale::celsius, scale::fahrenheit>
   {
      static double convert(double const value)
      {
         return (value * 9) / 5 + 32;
      }
   };

   template <>
   struct conversion_traits<scale::fahrenheit, scale::celsius>
   {
      static double convert(double const value)
      {
         return (value - 32) * 5 / 9;
      }
   };

   template <scale R, scale S>
   constexpr quantity<R> temperature_cast(quantity<S> const q)
   {
      return quantity<R>(conversion_traits<S, R>::convert(
         static_cast<double>(q)));
   }
}
```

用于创建温度值的文字操作符显示在以下代码片段中。这些操作符定义在一个名为`temperature_scale_literals`的单独命名空间中，这是一种良好的做法，以减少与其他文字操作符的名称冲突的风险：

```cpp
namespace temperature
{
   namespace temperature_scale_literals
   {
      constexpr quantity<scale::celsius> operator "" _deg(
         long double const amount)
      {
         return quantity<scale::celsius> {static_cast<double>(amount)};
      }

      constexpr quantity<scale::fahrenheit> operator "" _f(
         long double const amount)
      {
         return quantity<scale::fahrenheit> {static_cast<double>(amount)};
      }
   }
}
```

以下示例显示了如何定义两个温度值，一个是摄氏度，一个是华氏度，并在两者之间进行转换：

```cpp
int main()
{
   using namespace temperature;
   using namespace temperature_scale_literals;

   auto t1{ 36.5_deg };
   auto t2{ 79.0_f };

   auto tf = temperature_cast<scale::fahrenheit>(t1);
   auto tc = temperature_cast<scale::celsius>(tf);
   assert(t1 == tc);
}
```
