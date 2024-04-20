# 第十七章：算法和数据结构

# 问题

以下是本章的问题解决部分。

# 45\. 优先队列

编写一个表示优先队列的数据结构，该队列提供最大元素的常数时间查找，但在添加和删除元素时具有对数时间复杂度。队列在末尾插入新元素，并从顶部删除元素。默认情况下，队列应该使用`operator<`来比较元素，但用户应该能够提供一个比较函数对象，如果第一个参数小于第二个参数，则返回`true`。实现必须提供至少以下操作：

+   `push()` 用于添加新元素

+   `pop()` 用于移除顶部元素

+   `top()` 提供对顶部元素的访问

+   `size()` 指示队列中元素的数量

+   `empty()` 指示队列是否为空

# 46\. 循环缓冲区

创建一个表示固定大小的循环缓冲区的数据结构。当缓冲区填满超出其固定大小时，循环缓冲区会覆盖现有元素。您必须编写的类应该：

+   禁止默认构造

+   支持创建指定大小的对象

+   允许检查缓冲区容量和状态（`empty()`、`full()`、`size()`、`capacity()`）

+   添加一个新元素，这个操作可能会覆盖缓冲区中最旧的元素

+   从缓冲区中删除最旧的元素

+   支持遍历其元素

# 47\. 双缓冲区

编写一个表示可以同时写入和读取的缓冲区的类，而不会发生两个操作的冲突。读取操作必须在进行写入操作时提供对旧数据的访问。新写入的数据必须在写入操作完成后可供读取。

# 48\. 范围内最频繁的元素

编写一个函数，给定一个范围，返回出现最频繁的元素以及它在范围内出现的次数。如果有多个元素出现相同的最大次数，则函数应返回所有这些元素。例如，对于范围`{1,1,3,5,8,13,3,5,8,8,5}`，它应该返回`{5, 3}`和`{8, 3}`。

# 49\. 文本直方图

编写一个程序，给定一个文本，确定并打印每个字母的频率直方图。频率是每个字母出现次数与字母总数的百分比。程序应该只计算字母的出现次数，忽略数字、符号和其他可能的字符。频率必须基于字母计数而不是文本大小来确定。

# 50\. 过滤电话号码列表

编写一个函数，给定一个电话号码列表，仅返回来自指定国家的号码。国家由其电话国家代码表示，例如 44 代表英国。电话号码可能以国家代码开头，后跟`+`和国家代码，或者没有国家代码。最后一类必须被忽略。

# 51\. 转换电话号码列表

编写一个函数，给定一个电话号码列表，将它们转换为都以指定电话国家代码开头，前面加上`+`号。还应该删除电话号码中的任何空格。以下是输入和输出示例列表：

```cpp
07555 123456    => +447555123456
07555123456     => +447555123456
+44 7555 123456 => +447555123456
44 7555 123456  => +447555123456
7555 123456     => +447555123456
```

# 52\. 生成字符串的所有排列

编写一个函数，在控制台上打印给定字符串的所有可能的排列。您应该提供这个函数的两个版本：一个使用递归，一个不使用递归。

# 53\. 电影的平均评分

编写一个计算并打印电影列表的平均评分的程序。每部电影都有一个从 1 到 10 的评分列表（其中 1 是最低分，10 是最高分）。为了计算评分，您必须在计算平均值之前删除最高和最低评分的 5%。结果必须以一个小数点显示。

# 54\. 两两算法

编写一个通用函数，给定一个范围，返回一个新的范围，其中包含输入范围的连续元素对。如果输入范围的元素数是奇数，则必须忽略最后一个元素。例如，如果输入范围是`{1, 1, 3, 5, 8, 13, 21}`，结果必须是`{{1, 1}, {3, 5}, {8, 13}}`。

# 55\. 压缩算法

编写一个函数，给定两个范围，返回一个新的范围，其中包含来自两个范围的元素对。如果两个范围的大小不同，结果必须包含输入范围中最小的元素数量。例如，如果输入范围是`{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }`和`{ 1, 1, 3, 5, 8, 13, 21 }`，结果应该是`{{1,1}, {2,1}, {3,3}, {4,5}, {5,8}, {6,13}, {7,21}}`。

# 56\. 选择算法

编写一个函数，给定一系列值和一个投影函数，将每个值转换为一个新值，并返回一个选择的值的新范围。例如，如果你有一个类型为 book 的值，它有`id`、`title`和`author`，并且有一系列这样的书值，函数应该能够选择书的标题。下面是函数应该如何使用的一个例子：

```cpp
struct book
{
   int         id;
   std::string title;
   std::string author;
};

std::vector<book> books{
   {101, "The C++ Programming Language", "Bjarne Stroustrup"},
   {203, "Effective Modern C++", "Scott Meyers"},
   {404, "The Modern C++ Programming Cookbook", "Marius Bancila"}};

auto titles = select(books, [](book const & b) {return b.title; });
```

# 57\. 排序算法

编写一个函数，给定一对随机访问迭代器来定义其下限和上限，使用快速排序算法对范围的元素进行排序。排序函数应该有两个重载：一个使用`operator<`来比较范围的元素并按升序放置它们，另一个使用用户定义的二进制比较函数来比较元素。

# 58\. 节点之间的最短路径

编写一个程序，给定节点网络和它们之间的距离，计算并显示从指定节点到其他所有节点的最短距离，以及起点和终点节点之间的路径。作为输入，考虑以下无向图：

![](img/2fe0878c-3372-40bf-a988-81c9d47ab199.png)

这个图的程序输出应该是以下内容：

```cpp
A -> A : 0     A
A -> B : 7     A -> B
A -> C : 9     A -> C
A -> D : 20    A -> C -> D
A -> E : 20    A -> C -> F -> E
A -> F : 11    A -> C -> F
```

# 59\. 鼬程序

编写一个程序，实现理查德·道金斯的鼬计算机模拟，道金斯在《盲眼的看守者》第三章中描述如下：

我们再次使用我们的计算机猴子，但是它的程序有一个关键的不同。它再次开始选择一个随机序列的 28 个字母，就像以前一样...它重复复制它，但有一定的随机错误的机会 - '突变' - 在复制中。计算机检查原始短语的突变无意义短语，选择其中最像目标短语“METHINKS IT IS LIKE A WEASEL”的那个，即使只是稍微地。

# 60\. 生命游戏

编写一个程序，实现约翰·霍顿·康威提出的生命游戏细胞自动机。这个游戏的宇宙是一个正方形单元格的网格，可以有两种状态之一：死或活。每个细胞与其相邻的邻居进行交互，每一步都会发生以下交易：

+   任何活细胞如果少于两个活邻居，则死亡，就像是由于人口不足引起的

+   任何有两个或三个活邻居的活细胞将继续到下一代

+   任何有超过三个活邻居的活细胞将死亡，就像是由于过度生育引起的

+   任何有三个活邻居的死细胞将成为活细胞，就像是通过繁殖一样

游戏在每次迭代中的状态应该显示在控制台上，为了方便起见，你应该选择一个合理的大小，比如 20 行 x50 列。

# 解决方案

以下是上述问题解决部分的解决方案。

# 45\. 优先队列

优先队列是一个抽象数据类型，其元素附有优先级。优先队列不像先进先出容器那样工作，而是按照它们的优先级顺序提供元素。这种数据结构在算法中被用于迪杰斯特拉最短路径、普林姆算法、堆排序、A*搜索算法、用于数据压缩的哈夫曼编码等。

实现优先队列的一个非常简单的方法是使用`std::vector`作为元素的基础容器，并始终保持其排序。这意味着最大和最小元素总是在两端。然而，这种方法并不提供最有效的操作。

可以用来实现优先队列的最合适的数据结构是堆。这是一种基于树的数据结构，满足以下属性：如果*P*是*C*的父节点，则*P*的键（值）要么大于或等于（在最大堆中），要么小于或等于（在最小堆中）*C*的键。

标准库提供了几个用于处理堆的操作：

+   `std::make_heap()`: 这为给定范围创建一个最大堆，使用`operator<`或用户提供的比较函数来排序元素

+   `std::push_heap()`: 这在最大堆的末尾插入一个新元素

+   `std::pop_heap()`: 这会移除堆的第一个元素（通过交换第一个和最后一个位置的值，并使子范围`[first, last-1)`成为最大堆）

使用`std::vector`保存数据和堆的标准函数的优先队列实现可以如下所示：

```cpp
template <class T,
   class Compare = std::less<typename std::vector<T>::value_type>>
class priority_queue
{
   typedef typename std::vector<T>::value_type value_type;
   typedef typename std::vector<T>::size_type size_type;
   typedef typename std::vector<T>::reference reference;
   typedef typename std::vector<T>::const_reference const_reference;
public:
   bool empty() const noexcept { return data.empty(); }
   size_type size() const noexcept { return data.size(); }

   void push(value_type const & value)
   {
      data.push_back(value);
      std::push_heap(std::begin(data), std::end(data), comparer);
   }

   void pop()
   {
      std::pop_heap(std::begin(data), std::end(data), comparer);
      data.pop_back();
   }

   const_reference top() const { return data.front(); }

   void swap(priority_queue& other) noexcept
   {
      swap(data, other.data);
      swap(comparer, other.comparer);
   }
private:
   std::vector<T> data;
   Compare comparer;
};

template<class T, class Compare>
void swap(priority_queue<T, Compare>& lhs,
          priority_queue<T, Compare>& rhs) 
noexcept(noexcept(lhs.swap(rhs)))
{
   lhs.swap(rhs);
}
```

可以如下使用这个类：

```cpp
int main()
{
   priority_queue<int> q;
   for (int i : {1, 5, 3, 1, 13, 21, 8})
   {
      q.push(i);
   }

   assert(!q.empty());
   assert(q.size() == 7);

   while (!q.empty())
   {
      std::cout << q.top() << ' ';
      q.pop();
   }
}
```

# 46\. 循环缓冲区

循环缓冲区是一个固定大小的容器，其行为就好像它的两端连接在一起形成一个虚拟的循环内存布局。它的主要好处是你不需要大量的内存来保留数据，因为旧条目会被新条目覆盖。循环缓冲区用于 I/O 缓冲，有界日志（当您只想保留最后的消息时），异步处理的缓冲区等。

我们可以区分两种情况：

1.  添加到缓冲区的元素数量尚未达到其容量（其用户定义的固定大小）。在这种情况下，它的行为类似于一个常规容器，如向量。

1.  添加到缓冲区的元素数量已经达到并超过了其容量。在这种情况下，缓冲区的内存被重用，并且旧元素被覆盖。

我们可以用以下方式表示这样的结构：

+   一个预先分配了一定数量元素的常规容器

+   一个头指针，用于指示最后插入元素的位置

+   一个大小计数器，用于指示容器中的元素数量，不能超过其容量（因为在这种情况下元素被覆盖）

循环缓冲区的两个主要操作是：

+   向缓冲区添加一个新元素。我们总是在头指针（或索引）的下一个位置插入。这是下面显示的`push()`方法。

+   从缓冲区中移除一个现有元素。我们总是移除最旧的元素。该元素位于`head - size`的位置（这必须考虑索引的循环特性）。这是下面显示的`pop()`方法。

这样的数据结构的实现如下所示：

```cpp
template <class T>
class circular_buffer
{
   typedef circular_buffer_iterator<T> const_iterator;

   circular_buffer() = delete;
public:
   explicit circular_buffer(size_t const size) :data_(size)
   {}

   bool clear() noexcept { head_ = -1; size_ = 0; }
   bool empty() const noexcept { return size_ == 0; }
   bool full() const noexcept { return size_ == data_.size(); }
   size_t capacity() const noexcept { return data_.size(); }
   size_t size() const noexcept { return size_; }

   void push(T const item)
   {
      head_ = next_pos();
      data_[head_] = item;
      if (size_ < data_.size()) size_++;
   }

   T pop()
   {
      if (empty()) throw std::runtime_error("empty buffer");
      auto pos = first_pos();
      size_--;
      return data_[pos];
   }

   const_iterator begin() const
   {
      return const_iterator(*this, first_pos(), empty());
   }

   const_iterator end() const
   {
      return const_iterator(*this, next_pos(), true);
   }

private:
   std::vector<T> data_;
   size_t head_ = -1;
   size_t size_ = 0;

   size_t next_pos() const noexcept 
   { return size_ == 0 ? 0 : (head_ + 1) % data_.size(); }
   size_t first_pos() const noexcept 
   { return size_ == 0 ? 0 : (head_ + data_.size() - size_ + 1) % 
                             data_.size(); }

   friend class circular_buffer_iterator<T>;
};
```

由于索引在连续内存布局上的循环特性，这个类的迭代器类型不能是指针类型。迭代器必须能够通过在索引上应用模运算来指向元素。以下是这样一个迭代器的可能实现：

```cpp
template <class T>
class circular_buffer_iterator
{
   typedef circular_buffer_iterator        self_type;
   typedef T                               value_type;
   typedef T&                              reference;
   typedef T const&                        const_reference;
   typedef T*                              pointer;
   typedef std::random_access_iterator_tag iterator_category;
   typedef ptrdiff_t                       difference_type;
public:
   circular_buffer_iterator(circular_buffer<T> const & buf, 
                            size_t const pos, bool const last) :
   buffer_(buf), index_(pos), last_(last)
   {}

   self_type & operator++ ()
   {
      if (last_)
         throw std::out_of_range("Iterator cannot be incremented past the end of range.");
      index_ = (index_ + 1) % buffer_.data_.size();
      last_ = index_ == buffer_.next_pos();
      return *this;
   }

   self_type operator++ (int)
   {
      self_type tmp = *this;
      ++*this;
      return tmp;
   }

   bool operator== (self_type const & other) const
   {
      assert(compatible(other));
      return index_ == other.index_ && last_ == other.last_;
   }

   bool operator!= (self_type const & other) const
   {
      return !(*this == other);
   }

   const_reference operator* () const
   {
      return buffer_.data_[index_];
   }

   const_reference operator-> () const
   {
      return buffer_.data_[index_];
   }
private:
   bool compatible(self_type const & other) const
   {
      return &buffer_ == &other.buffer_;
   }

   circular_buffer<T> const & buffer_;
   size_t index_;
   bool last_;
};
```

有了这些实现，我们可以编写如下的代码。请注意，在注释中，第一个范围显示内部向量的实际内容，第二个范围显示通过迭代器访问时的逻辑内容：

```cpp
int main()
{
   circular_buffer<int> cbuf(5); // {0, 0, 0, 0, 0} -> {}

   cbuf.push(1);                 // {1, 0, 0, 0, 0} -> {1}
   cbuf.push(2);                 // {1, 2, 0, 0, 0} -> {1, 2}
   cbuf.push(3);                 // {1, 2, 3, 0, 0} -> {1, 2, 3}

   auto item = cbuf.pop();       // {1, 2, 3, 0, 0} -> {2, 3}
   cbuf.push(4);                 // {1, 2, 3, 4, 0} -> {2, 3, 4}
   cbuf.push(5);                 // {1, 2, 3, 4, 5} -> {2, 3, 4, 5}
   cbuf.push(6);                 // {6, 2, 3, 4, 5} -> {2, 3, 4, 5, 6}

   cbuf.push(7);                 // {6, 7, 3, 4, 5} -> {3, 4, 5, 6, 7}
   cbuf.push(8);                 // {6, 7, 8, 4, 5} -> {4, 5, 6, 7, 8}

   item = cbuf.pop();            // {6, 7, 8, 4, 5} -> {5, 6, 7, 8}
   item = cbuf.pop();            // {6, 7, 8, 4, 5} -> {6, 7, 8}
   item = cbuf.pop();            // {6, 7, 8, 4, 5} -> {7, 8}

   item = cbuf.pop();            // {6, 7, 8, 4, 5} -> {8}
   item = cbuf.pop();            // {6, 7, 8, 4, 5} -> {}

   cbuf.push(9);                 // {6, 7, 8, 9, 5} -> {9}
}
```

# 47\. 双缓冲

这里描述的问题是典型的双缓冲情况。双缓冲是多重缓冲的最常见情况，这是一种允许读者看到数据的完整版本而不是写入者产生的部分更新版本的技术。这是一种常见的技术 - 尤其是在计算机图形学中 - 用于避免闪烁。

为了实现所请求的功能，我们应该编写的缓冲类必须有两个内部缓冲区：一个包含正在写入的临时数据，另一个包含已完成（或提交）的数据。在写操作完成时，临时缓冲区的内容将写入主缓冲区。对于内部缓冲区，下面的实现使用`std::vector`。当写操作完成时，我们不是从一个缓冲区复制数据到另一个缓冲区，而是交换两者的内容，这是一个更快的操作。通过`read()`函数提供对已完成数据的访问，该函数将读取缓冲区的内容复制到指定的输出，或者通过直接元素访问（重载的`operator[]`）。对读缓冲区的访问与`std::mutex`同步，以确保在一个线程从缓冲区读取时另一个线程正在向缓冲区写入是安全的：

```cpp
template <typename T>
class double_buffer
{
   typedef T           value_type;
   typedef T&          reference;
   typedef T const &   const_reference;
   typedef T*          pointer;
public:
   explicit double_buffer(size_t const size) :
      rdbuf(size), wrbuf(size)
   {}

   size_t size() const noexcept { return rdbuf.size(); }

   void write(T const * const ptr, size_t const size)
   {
      std::unique_lock<std::mutex> lock(mt);
      auto length = std::min(size, wrbuf.size());
      std::copy(ptr, ptr + length, std::begin(wrbuf));
      wrbuf.swap(rdbuf);
   }

   template <class Output>
   void read(Output it) const
   {
      std::unique_lock<std::mutex> lock(mt);
      std::copy(std::cbegin(rdbuf), std::cend(rdbuf), it);
   }

   pointer data() const
   {
       std::unique_lock<std::mutex> lock(mt);
       return rdbuf.data();
   }

   reference operator[](size_t const pos)
   {
      std::unique_lock<std::mutex> lock(mt);
      return rdbuf[pos];
   }

   const_reference operator[](size_t const pos) const
   {
      std::unique_lock<std::mutex> lock(mt);
      return rdbuf[pos];
   }

   void swap(double_buffer other)
   {
      std::swap(rdbuf, other.rdbuf);
      std::swap(wrbuf, other.wrbuf);
   }

private:
   std::vector<T>     rdbuf;
   std::vector<T>     wrbuf;
   mutable std::mutex mt;
};
```

以下是这个双缓冲类如何被两个不同实体用于写入和读取的示例：

```cpp
template <typename T>
void print_buffer(double_buffer<T> const & buf)
{
   buf.read(std::ostream_iterator<T>(std::cout, " "));
   std::cout << std::endl;
}

int main()
{
   double_buffer<int> buf(10);

   std::thread t([&buf]() {
      for (int i = 1; i < 1000; i += 10)
      {
         int data[] = { i, i + 1, i + 2, i + 3, i + 4, 
                        i + 5, i + 6,i + 7,i + 8,i + 9 };
         buf.write(data, 10);

         using namespace std::chrono_literals;
         std::this_thread::sleep_for(100ms);
       }
   });

   auto start = std::chrono::system_clock::now();
   do
   {
      print_buffer(buf);

      using namespace std::chrono_literals;
      std::this_thread::sleep_for(150ms);
   } while (std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now() - start).count() < 12);

   t.join();
}
```

# 48. 范围内最频繁的元素

为了确定并返回范围内最频繁的元素，你应该这样做：

+   在`std::map`中计算每个元素的出现次数。键是元素，值是它的出现次数。

+   使用`std::max_element()`确定映射的最大元素。结果是一个映射元素，即包含元素及其出现次数的一对。

+   复制所有映射元素，其值（出现次数）等于最大元素的值，并将其作为最终结果返回。

先前描述的步骤的实现如下所示：

```cpp
template <typename T>
std::vector<std::pair<T, size_t>> find_most_frequent(
   std::vector<T> const & range)
{
   std::map<T, size_t> counts;
   for (auto const & e : range) counts[e]++;

   auto maxelem = std::max_element(
      std::cbegin(counts), std::cend(counts),
      [](auto const & e1, auto const & e2) {
         return e1.second < e2.second;
   });

   std::vector<std::pair<T, size_t>> result;

   std::copy_if(
      std::begin(counts), std::end(counts),
      std::back_inserter(result),
      maxelem {
         return kvp.second == maxelem->second;
   });

   return result;
}
```

`find_most_frequent()`函数可以如下使用：

```cpp
int main()
{
   auto range = std::vector<int>{1,1,3,5,8,13,3,5,8,8,5};
   auto result = find_most_frequent(range);

   for (auto const & e : result)
   {
      std::cout << e.first << " : " << e.second << std::endl;
   }
}
```

# 49. 文本直方图

直方图是数值数据分布的表示。广为人知的直方图是摄影和图像处理中使用的颜色和图像直方图。如此描述的文本直方图是给定文本中字母频率的表示。这个问题在某种程度上与之前的问题类似，只是现在范围的元素是字符，我们必须确定它们的频率。要解决这个问题，你应该：

+   使用映射计算每个字母的出现次数。键是字母，值是它的出现次数。

+   在计数时，忽略所有不是字母的字符。大写和小写字符必须被视为相同，因为它们代表相同的字母。

+   使用`std::accumulate()`来计算给定文本中所有字母出现次数的总数。

+   使用`std::for_each()`或基于范围的`for`循环遍历映射的所有元素，并将出现次数转换为频率。

以下是该问题的一个可能实现：

```cpp
std::map<char, double> analyze_text(std::string_view text)
{
   std::map<char, double> frequencies;
   for (char ch = 'a'; ch <= 'z'; ch++)
      frequencies[ch] = 0;

   for (auto ch : text)
   {
      if (isalpha(ch))
         frequencies[tolower(ch)]++;
   }

   auto total = std::accumulate(
      std::cbegin(frequencies), std::cend(frequencies),
      0ull,
      [](auto sum, auto const & kvp) {
         return sum + static_cast<unsigned long long>(kvp.second);
   });

   std::for_each(
      std::begin(frequencies), std::end(frequencies),
      total {
         kvp.second = (100.0 * kvp.second) / total;
   });

   return frequencies;
}
```

以下程序在控制台上打印文本中字母的频率：

```cpp
int main()
{
   auto result = analyze_text(R"(Lorem ipsum dolor sit amet, consectetur 
      adipiscing elit, sed do eiusmod tempor incididunt ut labore et 
      dolore magna aliqua.)");

   for (auto const & kvp : result)
   {
      std::cout << kvp.first << " : "
                << std::fixed
                << std::setw(5) << std::setfill(' ')
                << std::setprecision(2) << kvp.second << std::endl;
   }
}
```

# 50. 过滤电话号码列表

解决这个问题相对简单：你必须遍历所有电话号码，并将以国家代码开头的电话号码复制到一个单独的容器（如`std::vector`）中。如果指定的国家代码是，例如，44，那么你必须同时检查 44 和+44。使用`std::copy_if()`函数可以以这种方式过滤输入范围。这个问题的解决方案如下所示：

```cpp
bool starts_with(std::string_view str, std::string_view prefix)
{
   return str.find(prefix) == 0;
}

template <typename InputIt>
std::vector<std::string> filter_numbers(InputIt begin, InputIt end,
                                        std::string const & countryCode)
{
   std::vector<std::string> result;
   std::copy_if(
      begin, end,
      std::back_inserter(result),
      countryCode {
         return starts_with(number, countryCode) ||
                starts_with(number, "+" + countryCode);
   });
   return result;
}

std::vector<std::string> filter_numbers(
   std::vector<std::string> const & numbers,
   std::string const & countryCode)
{
   return filter_numbers(std::cbegin(numbers), std::cend(numbers), 
                         countryCode);
}
```

这是如何使用这个函数的：

```cpp
int main()
{
   std::vector<std::string> numbers{
      "+40744909080",
      "44 7520 112233",
      "+44 7555 123456",
      "40 7200 123456",
      "7555 123456"
   };

   auto result = filter_numbers(numbers, "44");

   for (auto const & number : result)
   {
      std::cout << number << std::endl;
   }
}
```

# 51. 转换电话号码列表

这个问题在某些方面与之前的问题有些相似。但是，我们不是选择以指定国家代码开头的电话号码，而是要转换每个号码，使它们都以该国家代码前面加上`+`。有几种情况必须考虑：

+   电话号码以 0 开头。这表示没有国家代码的号码。要修改号码以包括国家代码，必须用实际国家代码替换 0，前面加上`+`。

+   电话号码以国家代码开头。在这种情况下，我们只需在开头添加`+`号。

+   电话号码以`+`开头，后面跟着国家代码。在这种情况下，号码已经是预期格式。

+   没有这些情况适用，因此结果是通过将以`+`为前缀的国家代码和电话号码连接在一起获得的。

为简单起见，我们将忽略号码实际上可能带有另一个国家代码前缀的可能性。您可以将其作为进一步的练习，修改实现以处理带有不同国家前缀的电话号码。这些号码应该从列表中删除。

在所有前述情况中，可能存在号码包含空格的情况。根据要求，这些必须被移除。`std::remove_if()`和`isspace()`函数用于此目的。

以下是所描述解决方案的实现：

```cpp
bool starts_with(std::string_view str, std::string_view prefix)
{
   return str.find(prefix) == 0;
}

void normalize_phone_numbers(std::vector<std::string>& numbers,
                             std::string const & countryCode)
{
   std::transform(
      std::cbegin(numbers), std::cend(numbers),
      std::begin(numbers),
      countryCode {
         std::string result;
         if (number.size() > 0)
         {
            if (number[0] == '0')
               result = "+" + countryCode + 
                        number.substr(1);
            else if (starts_with(number, countryCode))
               result = "+" + number;
            else if (starts_with(number, "+" + countryCode))
               result = number;
            else
               result = "+" + countryCode + number;
      }

      result.erase(
         std::remove_if(std::begin(result), std::end(result),
            [](const char ch) {return isspace(ch); }),
         std::end(result));

      return result;
   });
}
```

以下程序根据要求规范化给定的电话号码列表，并将它们打印在控制台上：

```cpp
int main()
{
   std::vector<std::string> numbers{
      "07555 123456",
      "07555123456",
      "+44 7555 123456",
      "44 7555 123456",
      "7555 123456"
   };

   normalize_phone_numbers(numbers, "44");

   for (auto const & number : numbers)
   {
      std::cout << number << std::endl;
   }
}
```

# 52. 生成字符串的所有排列

您可以通过利用标准库中的一些通用算法来解决这个问题。所需版本中最简单的是非递归版本，至少在使用`std::next_permutation()`时是这样。该函数将输入范围（需要排序）转换为从所有可能的排列中的下一个排列，按字典顺序排序，使用`operator<`或指定的比较函数对象。如果存在这样的排列，则返回`true`，否则，它将范围转换为第一个排列并返回`false`。因此，基于`std::next_permuation()`的非递归实现如下所示：

```cpp
void print_permutations(std::string str)
{
   std::sort(std::begin(str), std::end(str));

   do
   {
      std::cout << str << std::endl;
   } while (std::next_permutation(std::begin(str), std::end(str)));
}
```

递归的替代方法稍微复杂一些。实现它的一种方法是有一个输入和输出字符串；最初，输入字符串是我们想要生成排列的字符串，输出字符串为空。我们从输入字符串中一次取一个字符并将其放入输出字符串。当输入字符串变为空时，输出字符串表示下一个排列。执行此操作的递归算法如下：

+   如果输入字符串为空，则打印输出字符串并返回

+   否则遍历输入字符串中的所有字符，并对每个元素执行以下操作：

+   通过从输入字符串中删除第一个字符并将其连接到输出字符串的末尾来递归调用该方法

+   旋转输入字符串，使第一个字符成为最后一个字符，第二个字符成为第一个字符，依此类推

该算法在以下图表中得到了可视化解释：

![](img/bbcdafcd-a3e0-4a0c-9b79-9f465a5a4602.png)

对于旋转输入字符串，我们可以使用标准库函数`std::rotate()`，它对一系列元素执行左旋转。实现所描述的递归算法如下：

```cpp
void next_permutation(std::string str, std::string perm)
{
   if (str.empty()) std::cout << perm << std::endl;
   else
   {
      for (size_t i = 0; i < str.size(); ++i)
      {
         next_permutation(str.substr(1), perm + str[0]);

         std::rotate(std::begin(str), std::begin(str) + 1, std::end(str));
      }
   }
}

void print_permutations_recursive(std::string str)
{
   next_permutation(str, "");
}
```

这就是这两种实现的用法：

```cpp
int main()
{
   std::cout << "non-recursive version" << std::endl;
   print_permutations("main");

   std::cout << "recursive version" << std::endl;
   print_permutations_recursive("main");
}
```

# 53. 电影的平均评分

该问题需要使用截断均值来计算电影评分。这是一种统计测度，用于计算平均值，计算后丢弃概率分布或样本的高端和低端的部分。通常，这是通过在两端删除相等数量的点来完成的。对于这个问题，您需要删除最高和最低用户评分的 5%。

计算给定范围的截断均值的函数应该执行以下操作：

+   对范围进行排序，使元素按升序或降序排序

+   删除两端所需百分比的元素

+   计算所有剩余元素的总和

+   通过将总和除以剩余元素的数量来计算平均值

这里显示的`truncated_mean()`函数实现了所描述的算法：

```cpp
double truncated_mean(std::vector<int> values, double const percentage)
{
   std::sort(std::begin(values), std::end(values));
   auto remove_count = static_cast<size_t>(
                          values.size() * percentage + 0.5);

   values.erase(std::begin(values), std::begin(values) + remove_count);
   values.erase(std::end(values) - remove_count, std::end(values));

   auto total = std::accumulate(
      std::cbegin(values), std::cend(values),
      0ull,
      [](auto const sum, auto const e) {
         return sum + e; });
   return static_cast<double>(total) / values.size();
}
```

使用此函数来计算并打印电影平均评分的程序可能如下所示：

```cpp
struct movie
{
   int              id;
   std::string      title;
   std::vector<int> ratings;
};

void print_movie_ratings(std::vector<movie> const & movies)
{
   for (auto const & m : movies)
   {
      std::cout << m.title << " : " 
                << std::fixed << std::setprecision(1)
                << truncated_mean(m.ratings, 0.05) << std::endl;
   }
}

int main()
{
   std::vector<movie> movies
   {
      { 101, "The Matrix", {10, 9, 10, 9, 9, 8, 7, 10, 5, 9, 9, 8} },
      { 102, "Gladiator", {10, 5, 7, 8, 9, 8, 9, 10, 10, 5, 9, 8, 10} },
      { 103, "Interstellar", {10, 10, 10, 9, 3, 8, 8, 9, 6, 4, 7, 10} }
   };

   print_movie_ratings(movies);
}
```

# 54\. 两两配对算法

为了解决这个问题提出的两两函数必须将输入范围的相邻元素配对，并产生添加到输出范围的`std::pair`元素。以下代码清单提供了两种实现：

+   一个以迭代器作为参数的通用函数模板：一个起始和结束迭代器定义了输入范围，一个输出迭代器定义了结果要插入的输出范围的位置

+   一个重载，它以`std::vector<T>`作为输入参数，并以`std::vector<std::pair<T, T>>`作为结果返回；这个只是调用第一个重载：

```cpp
template <typename Input, typename Output>
void pairwise(Input begin, Input end, Output result)
{
   auto it = begin;
   while (it != end)
   {
      auto v1 = *it++; if (it == end) break;
      auto v2 = *it++;
      result++ = std::make_pair(v1, v2);
   }
}
template <typename T>
std::vector<std::pair<T, T>> pairwise(std::vector<T> const & range)
{
   std::vector<std::pair<T, T>> result;
   pairwise(std::begin(range), std::end(range),
            std::back_inserter(result));
   return result;
}
```

以下程序将整数向量的元素配对，并在控制台上打印出这些配对：

```cpp
int main()
{
   std::vector<int> v{ 1, 1, 3, 5, 8, 13, 21 };
   auto result = pairwise(v);

   for (auto const & p : result)
   {
      std::cout << '{' << p.first << ',' << p.second << '}' << std::endl;
   }
}
```

# 55\. 压缩算法

这个问题与之前的问题相对类似，尽管有两个输入范围而不只是一个。结果再次是一个`std::pair`范围。然而，两个输入范围可能包含不同类型的元素。同样，这里显示的实现包含两个重载：

+   一个以迭代器作为参数的通用函数。每个输入范围都有一个起始和结束迭代器定义其边界，一个输出迭代器定义了结果必须写入的输出范围的位置。

+   一个函数，它接受两个`std::vector`参数，一个包含类型`T`的元素，另一个包含类型`U`的元素，并返回一个`std::vector<std::pair<T, U>>`。这个重载只是调用前一个：

```cpp
template <typename Input1, typename Input2, typename Output>
void zip(Input1 begin1, Input1 end1, 
         Input2 begin2, Input1 end2, 
         Output result)
{
   auto it1 = begin1;
   auto it2 = begin2;
   while (it1 != end1 && it2 != end2)
   {
      result++ = std::make_pair(*it1++, *it2++);
   }
}

template <typename T, typename U>
std::vector<std::pair<T, U>> zip(
   std::vector<T> const & range1, 
   std::vector<U> const & range2)
{
   std::vector<std::pair<T, U>> result;

   zip(std::begin(range1), std::end(range1),
       std::begin(range2), std::end(range2),
       std::back_inserter(result));

   return result;
}
```

在下面的清单中，您可以看到两个整数向量被压缩在一起，并且结果打印在控制台上：

```cpp
int main()
{
   std::vector<int> v1{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   std::vector<int> v2{ 1, 1, 3, 5, 8, 13, 21 };

   auto result = zip(v1, v2);
   for (auto const & p : result)
   {
      std::cout << '{' << p.first << ',' << p.second << '}' << std::endl;
   }
}
```

# 56\. 选择算法

您必须实现的`select()`函数以`std::vector<T>`作为输入参数，并以`F`类型的函数返回`std::vector<R>`作为结果，其中`R`是将`F`应用于`T`的结果。我们可以使用`std::result_of()`在编译时推断调用表达式的返回类型。在内部，`select()`函数应该使用`std::transform()`来迭代输入向量的元素，对每个元素应用函数`f`，并将结果插入输出向量。

以下清单显示了该函数的实现：

```cpp
template <
   typename T, typename A, typename F,
   typename R = typename std::decay<typename std::result_of<
                typename std::decay<F>::type&(
                typename std::vector<T, A>::const_reference)>::type>::type>
std::vector<R> select(std::vector<T, A> const & c, F&& f)
{
   std::vector<R> v;
   std::transform(std::cbegin(c), std::cend(c),
                  std::back_inserter(v),
                  std::forward<F>(f));
   return v;
}
```

这个函数可以这样使用：

```cpp
int main()
{
   std::vector<book> books{
      {101, "The C++ Programming Language", "Bjarne Stroustrup"},
      {203, "Effective Modern C++", "Scott Meyers"},
      {404, "The Modern C++ Programming Cookbook", "Marius Bancila"}};

   auto titles = select(books, [](book const & b) {return b.title; });
   for (auto const & title : titles)
   {
      std::cout << title << std::endl;
   }
}
```

# 57\. 排序算法

**快速排序**是一个比较排序算法，用于定义了全序的数组元素。当实现良好时，它比*归并排序*或*堆排序*要快得多。

尽管在最坏情况下，该算法进行了![](img/66508cd7-1912-4285-bee6-c31db3d8d58c.png)次比较（当范围已经排序），但平均复杂度仅为![](img/0ce17681-465d-4ac5-9b80-114f13fa5f2c.png)。快速排序是一种分治算法；它将一个大范围分成较小的范围并递归地对它们进行排序。有几种分区方案。在这里显示的实现中，我们使用了*Tony Hoare*开发的原始方案。该方案的算法如下伪代码所示：

```cpp
algorithm quicksort(A, lo, hi) is
   if lo < hi then
      p := partition(A, lo, hi)
      quicksort(A, lo, p)
      quicksort(A, p + 1, hi)

algorithm partition(A, lo, hi) is
   pivot := A[lo]
   i := lo - 1
   j := hi + 1
   loop forever
      do
         i := i + 1
      while A[i] < pivot

      do
         j := j - 1
      while A[j] > pivot

      if i >= j then
         return j

      swap A[i] with A[j]
```

算法的通用实现应该使用迭代器而不是数组和索引。以下实现的要求是迭代器是随机访问的（因此可以在常数时间内移动到任何元素）：

```cpp
template <class RandomIt>
RandomIt partition(RandomIt first, RandomIt last)
{
   auto pivot = *first;
   auto i = first + 1;
   auto j = last - 1;
   while (i <= j)
   {
      while (i <= j && *i <= pivot) i++;
      while (i <= j && *j > pivot) j--;
      if (i < j) std::iter_swap(i, j);
   }

   std::iter_swap(i - 1, first);

   return i - 1;
}

template <class RandomIt>
void quicksort(RandomIt first, RandomIt last)
{
   if (first < last)
   {
      auto p = partition(first, last);
      quicksort(first, p);
      quicksort(p + 1, last);
   }
}
```

如下所示的`quicksort()`函数可用于对各种类型的容器进行排序：

```cpp
int main()
{
   std::vector<int> v{ 1,5,3,8,6,2,9,7,4 };
   quicksort(std::begin(v), std::end(v));

   std::array<int, 9> a{ 1,2,3,4,5,6,7,8,9 };
   quicksort(std::begin(a), std::end(a));

   int a[]{ 9,8,7,6,5,4,3,2,1 };
   quicksort(std::begin(a), std::end(a));
}
```

要求排序算法必须允许指定用户定义的比较函数。在这种情况下，唯一的变化是分区函数，其中我们使用用户定义的比较函数来比较当前元素与枢轴：

```cpp
template <class RandomIt, class Compare>
RandomIt partitionc(RandomIt first, RandomIt last, Compare comp)
{
   auto pivot = *first;
   auto i = first + 1;
   auto j = last - 1;
   while (i <= j)
   {
      while (i <= j && comp(*i, pivot)) i++;
      while (i <= j && !comp(*j, pivot)) j--;
      if (i < j) std::iter_swap(i, j);
   }

   std::iter_swap(i - 1, first);

   return i - 1;
}

template <class RandomIt, class Compare>
void quicksort(RandomIt first, RandomIt last, Compare comp)
{
   if (first < last)
   {
      auto p = partitionc(first, last, comp);
      quicksort(first, p, comp);
      quicksort(p + 1, last, comp);
   }
}
```

使用这个重载，我们可以按降序对范围进行排序，如下例所示：

```cpp
int main()
{
   std::vector<int> v{ 1,5,3,8,6,2,9,7,4 };
   quicksort(std::begin(v), std::end(v), std::greater<>());
}
```

也可以实现快速排序算法的迭代版本。迭代版本的性能在大多数情况下与递归版本相同（但在范围已经排序的最坏情况下会降级）。从递归版本的算法转换为迭代版本相对简单；通过使用堆栈来模拟递归调用并存储分区的边界来实现。以下是使用`operator<`比较元素的迭代实现版本：

```cpp
template <class RandomIt>
void quicksorti(RandomIt first, RandomIt last)
{
   std::stack<std::pair<RandomIt, RandomIt>> st;
   st.push(std::make_pair(first, last));
   while (!st.empty())
   {
      auto iters = st.top();
      st.pop();

      if (iters.second - iters.first < 2) continue;

      auto p = partition(iters.first, iters.second);

      st.push(std::make_pair(iters.first, p));
      st.push(std::make_pair(p+1, iters.second));
   }
}
```

这个迭代实现可以像它的递归版本一样使用：

```cpp
int main()
{
   std::vector<int> v{ 1,5,3,8,6,2,9,7,4 };
   quicksorti(std::begin(v), std::end(v));
}
```

# 58\. 节点之间的最短路径

要解决提出的问题，必须使用 Dijkstra 算法来找到图中的最短路径。尽管原始算法找到两个给定节点之间的最短路径，但这里的要求是找到指定节点与图中所有其他节点之间的最短路径，这是算法的另一个版本。

实现算法的有效方法是使用优先队列。算法的伪代码如下（参见[`en.wikipedia.org/wiki/Dijkstra%27s_algorithm`](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)）：

```cpp
function Dijkstra(Graph, source):
   dist[source] ← 0                 // Initialization

   create vertex set Q
   for each vertex v in Graph: 
      if v ≠ source
         dist[v] ← INFINITY         // Unknown distance from source to v
         prev[v] ← UNDEFINED        // Predecessor of v

      Q.add_with_priority(v, dist[v])

   while Q is not empty:            // The main loop
      u ← Q.extract_min()           // Remove and return best vertex
      for each neighbor v of u:     // only v that is still in Q
         alt ← dist[u] + length(u, v) 
         if alt < dist[v]
            dist[v] ← alt
            prev[v] ← u
            Q.decrease_priority(v, alt)

   return dist[], prev[]
```

为了表示图，我们可以使用以下数据结构，该数据结构可用于定向或单向图。该类支持添加新顶点和边，并可以返回顶点列表和指定顶点的邻居（即节点和到它们的距离）：

```cpp
template <typename Vertex = int, typename Weight = double>
class graph
{
public:
   typedef Vertex                     vertex_type;
   typedef Weight                     weight_type;
   typedef std::pair<Vertex, Weight>  neighbor_type;
   typedef std::vector<neighbor_type> neighbor_list_type;
public:
   void add_edge(Vertex const source, Vertex const target, 
                 Weight const weight, bool const bidirectional = true)
   {
      adjacency_list[source].push_back(std::make_pair(target, weight));
      adjacency_list[target].push_back(std::make_pair(source, weight));
   }

   size_t vertex_count() const { return adjacency_list.size(); }
   std::vector<Vertex> verteces() const
   {
      std::vector<Vertex> keys;
      for (auto const & kvp : adjacency_list)
         keys.push_back(kvp.first);
      return keys;
   }

   neighbor_list_type const & neighbors(Vertex const & v) const
   {
      auto pos = adjacency_list.find(v);
      if (pos == adjacency_list.end())
         throw std::runtime_error("vertex not found");
      return pos->second;
   }

   constexpr static Weight Infinity = 
             std::numeric_limits<Weight>::infinity();
private:
   std::map<vertex_type, neighbor_list_type> adjacency_list;
};
```

如前面伪代码中描述的最短路径算法的实现可能如下所示。使用`std::set`（即自平衡二叉搜索树）而不是优先队列。`std::set`对于添加和删除顶部元素具有与二叉堆（用于优先队列）相同的复杂度。另一方面，`std::set`还允许在`log(n)`时间内找到和删除任何其他元素，这有助于通过删除和再次插入来实现减小键步骤：

```cpp
template <typename Vertex, typename Weight>
void shortest_path(
   graph<Vertex, Weight> const & g,
   Vertex const source,
   std::map<Vertex, Weight>& min_distance,
   std::map<Vertex, Vertex>& previous)
{
   auto const n = g.vertex_count();
   auto const verteces = g.verteces();

   min_distance.clear();
   for (auto const & v : verteces)
      min_distance[v] = graph<Vertex, Weight>::Infinity;
   min_distance[source] = 0;

   previous.clear();

   std::set<std::pair<Weight, Vertex> > vertex_queue;
   vertex_queue.insert(std::make_pair(min_distance[source], source));

   while (!vertex_queue.empty())
   {
      auto dist = vertex_queue.begin()->first;
      auto u = vertex_queue.begin()->second;

      vertex_queue.erase(std::begin(vertex_queue));

      auto const & neighbors = g.neighbors(u);
      for (auto const & neighbor : neighbors)
      {
         auto v = neighbor.first;
         auto w = neighbor.second;
         auto dist_via_u = dist + w;
         if (dist_via_u < min_distance[v])
         {
            vertex_queue.erase(std::make_pair(min_distance[v], v));

            min_distance[v] = dist_via_u;
            previous[v] = u;
            vertex_queue.insert(std::make_pair(min_distance[v], v));
         }
      }
   }
}
```

以下辅助函数以指定的格式打印结果：

```cpp
template <typename Vertex>
void build_path(
   std::map<Vertex, Vertex> const & prev, Vertex const v,
   std::vector<Vertex> & result)
{
   result.push_back(v);

   auto pos = prev.find(v);
   if (pos == std::end(prev)) return;

   build_path(prev, pos->second, result);
}

template <typename Vertex>
std::vector<Vertex> build_path(std::map<Vertex, Vertex> const & prev, 
                               Vertex const v)
{
   std::vector<Vertex> result;
   build_path(prev, v, result);
   std::reverse(std::begin(result), std::end(result));
   return result;
}

template <typename Vertex>
void print_path(std::vector<Vertex> const & path)
{
   for (size_t i = 0; i < path.size(); ++i)
   {
      std::cout << path[i];
      if (i < path.size() - 1) std::cout << " -> ";
   }
}
```

以下程序解决了给定的任务：

```cpp
int main()
{
   graph<char, double> g;
   g.add_edge('A', 'B', 7);
   g.add_edge('A', 'C', 9);
   g.add_edge('A', 'F', 14);
   g.add_edge('B', 'C', 10);
   g.add_edge('B', 'D', 15);
   g.add_edge('C', 'D', 11);
   g.add_edge('C', 'F', 2);
   g.add_edge('D', 'E', 6);
   g.add_edge('E', 'F', 9);

   char source = 'A';
   std::map<char, double> min_distance;
   std::map<char, char> previous;
   shortest_path(g, source, min_distance, previous);

   for (auto const & kvp : min_distance)
   {
      std::cout << source << " -> " << kvp.first << " : "
                << kvp.second << '\t';

      print_path(build_path(previous, kvp.first));

      std::cout << std::endl;
   }
}
```

# 59\. 鼬程序

鼬程序是理查德·道金斯提出的一个思想实验，旨在演示积累的小改进（通过自然选择选择的带来好处的突变）产生快速结果，与主流误解相反，即进化是以大的飞跃发生的。鼬模拟的算法，如维基百科所述（参见[`en.wikipedia.org/wiki/Weasel_program`](https://en.wikipedia.org/wiki/Weasel_program)），如下所示：

1.  从一个随机字符串开始，长度为 28 个字符。

1.  制作此字符串的 100 个副本，每个字符有 5%的机会被替换为随机字符。

1.  将每个新字符串与目标 METHINKS IT IS LIKE A WEASEL 进行比较，并为每个字符串打分（字符串中正确位置的字母数）。

1.  如果任何新字符串得分完美（28），则停止。

1.  否则，取得分最高的字符串并转到步骤 2。

可能的实现如下。`make_random()`函数创建与目标相同长度的随机起始序列；`fitness()`函数计算每个突变字符串的得分（即与目标的相似度）；`mutate()`函数从父字符串产生一个新字符串，并为每个字符变异给出一定的机会： 

```cpp
class weasel
{
   std::string target;
   std::uniform_int_distribution<> chardist;
   std::uniform_real_distribution<> ratedist;
   std::mt19937 mt;
   std::string const allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
public:
   weasel(std::string_view t) :
      target(t), chardist(0, 26), ratedist(0, 100)
   {
      std::random_device rd;
      auto seed_data = std::array<int, std::mt19937::state_size> {};
      std::generate(std::begin(seed_data), std::end(seed_data), 
      std::ref(rd));
      std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
      mt.seed(seq);
   }
   void run(int const copies)
   {
      auto parent = make_random();
      int step = 1;
      std::cout << std::left << std::setw(5) << std::setfill(' ') 
                << step << parent << std::endl;

      do
      {
         std::vector<std::string> children;
         std::generate_n(std::back_inserter(children), copies, 
            [parent, this]() {return mutate(parent, 5); });

         parent = *std::max_element(
            std::begin(children), std::end(children),
            this {
               return fitness(c1) < fitness(c2); });

         std::cout << std::setw(5) << std::setfill(' ') << step 
                << parent << std::endl;

         step++;
      } while (parent != target);
   }
private:
   weasel() = delete;

   double fitness(std::string_view candidate) const
   {
      int score = 0;
      for (size_t i = 0; i < candidate.size(); ++i)
      {
         if (candidate[i] == target[i])
            score++;
      }
      return score;
   }

   std::string mutate(std::string_view parent, double const rate)
   {
      std::stringstream sstr;
      for (auto const c : parent)
      {
         auto nc = ratedist(mt) > rate ? c : allowed_chars[chardist(mt)];
         sstr << nc;
      }
      return sstr.str();
    }

   std::string make_random()
   {
      std::stringstream sstr;
      for (size_t i = 0; i < target.size(); ++i)
      {
         sstr << allowed_chars[chardist(mt)];
      }
      return sstr.str();
   }
};
```

这是如何使用该类的：

```cpp
int main()
{
   weasel w("METHINKS IT IS LIKE A WEASEL");
   w.run(100);
}
```

# 60\. 生命游戏

下面介绍的`universe`类实现了如上所述的游戏。有几个有趣的功能：

+   `initialize()` 生成一个起始布局；尽管书中的代码包含更多选项，但这里只列出了两个：`random`，生成一个随机布局，和 `ten_cell_row`，表示网格中间的 10 个细胞的一行。

+   `reset()` 将所有细胞设置为 `dead`。

+   `count_neighbors()` 返回活着的邻居数量。它使用一个辅助的可变函数模板 `count_alive()`。虽然这可以用折叠表达式实现，但在 Visual C++ 中尚不支持，因此我选择不在这里使用它。

+   `next_generation()` 根据过渡规则产生游戏的新状态。

+   `display()` 在控制台上显示游戏状态；这使用系统调用来擦除控制台，尽管您可以使用其他方法来做到这一点，比如特定的操作系统 API。

+   `run()` 初始化起始布局，然后以用户指定的间隔产生新的一代，进行用户指定次数的迭代，或者无限期地进行（如果迭代次数设置为 0）。

```cpp
class universe
{
private:
   universe() = delete;
public:
   enum class seed
   {
      random, ten_cell_row
   };
public:
   universe(size_t const width, size_t const height):
      rows(height), columns(width),grid(width * height), dist(0, 4)
   {
      std::random_device rd;
      auto seed_data = std::array<int, std::mt19937::state_size> {};
      std::generate(std::begin(seed_data), std::end(seed_data), 
      std::ref(rd));
      std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
      mt.seed(seq);
   }

   void run(seed const s, int const generations, 
            std::chrono::milliseconds const ms = 
               std::chrono::milliseconds(100))
   {
      reset();
      initialize(s);
      display();

      int i = 0;
      do 
      {
         next_generation();
         display();

         using namespace std::chrono_literals;
         std::this_thread::sleep_for(ms);
      } while (i++ < generations || generations == 0);
   }

private:
   void next_generation()
   {
      std::vector<unsigned char> newgrid(grid.size());

      for (size_t r = 0; r < rows; ++r)
      {
         for (size_t c = 0; c < columns; ++c)
         {
            auto count = count_neighbors(r, c);

            if (cell(c, r) == alive)
            {
               newgrid[r * columns + c] = 
                  (count == 2 || count == 3) ? alive : dead;
            }
            else 
            {
               newgrid[r * columns + c] = (count == 3) ? alive : dead;
            }
         }
      }

      grid.swap(newgrid);
   }

   void reset_display()
   {
#ifdef WIN32
      system("cls");
#endif
   }

   void display()
   {
      reset_display();

      for (size_t r = 0; r < rows; ++r)
      {
         for (size_t c = 0; c < columns; ++c)
         {
            std::cout << (cell(c, r) ? '*' : ' ');
         }
         std::cout << std::endl;
      }
   }

   void initialize(seed const s)
   {
      if (s == seed::ten_cell_row)
      {
         for (size_t c = columns / 2 - 5; c < columns / 2 + 5; c++)
            cell(c, rows / 2) = alive;
      }
      else
      {
         for (size_t r = 0; r < rows; ++r)
         {
            for (size_t c = 0; c < columns; ++c)
            {
               cell(c, r) = dist(mt) == 0 ? alive : dead;
            }
         }
      }
   }

   void reset()
   {
      for (size_t r = 0; r < rows; ++r)
      {
         for (size_t c = 0; c < columns; ++c)
         {
            cell(c, r) = dead;
         }
      }
   }

   int count_alive() { return 0; }

   template<typename T1, typename... T>
   auto count_alive(T1 s, T... ts) { return s + count_alive(ts...); }

   int count_neighbors(size_t const row, size_t const col)
   {
      if (row == 0 && col == 0) 
         return count_alive(cell(1, 0), cell(1,1), cell(0, 1));
      if (row == 0 && col == columns - 1)
         return count_alive(cell(columns - 2, 0), cell(columns - 2, 1), 
                            cell(columns - 1, 1));
      if (row == rows - 1 && col == 0)
         return count_alive(cell(0, rows - 2), cell(1, rows - 2), 
                            cell(1, rows - 1));
      if (row == rows - 1 && col == columns - 1)
         return count_alive(cell(columns - 1, rows - 2), 
                            cell(columns - 2, rows - 2), 
                            cell(columns - 2, rows - 1));
      if (row == 0 && col > 0 && col < columns - 1)
         return count_alive(cell(col - 1, 0), cell(col - 1, 1), 
                            cell(col, 1), cell(col + 1, 1), 
                            cell(col + 1, 0));
      if (row == rows - 1 && col > 0 && col < columns - 1)
         return count_alive(cell(col - 1, row), cell(col - 1, row - 1), 
                            cell(col, row - 1), cell(col + 1, row - 1), 
                            cell(col + 1, row));
      if (col == 0 && row > 0 && row < rows - 1)
         return count_alive(cell(0, row - 1), cell(1, row - 1), 
                            cell(1, row), cell(1, row + 1), 
                            cell(0, row + 1));
      if (col == columns - 1 && row > 0 && row < rows - 1)
         return count_alive(cell(col, row - 1), cell(col - 1, row - 1), 
                            cell(col - 1, row), cell(col - 1, row + 1), 
                            cell(col, row + 1));

      return count_alive(cell(col - 1, row - 1), cell(col, row - 1), 
                         cell(col + 1, row - 1), cell(col + 1, row), 
                         cell(col + 1, row + 1), cell(col, row + 1), 
                         cell(col - 1, row + 1), cell(col - 1, row));
   }

   unsigned char& cell(size_t const col, size_t const row)
   {
      return grid[row * columns + col];
   }

private:
   size_t rows;
   size_t columns;

   std::vector<unsigned char> grid;
   const unsigned char alive = 1;
   const unsigned char dead = 0;

   std::uniform_int_distribution<> dist;
   std::mt19937 mt;
};
```

这是如何从随机状态开始运行 100 次迭代的游戏：

```cpp
int main()
{
   using namespace std::chrono_literals;
   universe u(50, 20);
   u.run(universe::seed::random, 100, 100ms);
}
```

以下是程序输出的一个示例（屏幕截图表示生命游戏宇宙中的单次迭代）：

![](img/9f48c4ae-d266-4151-b814-33bf8473953b.png)
