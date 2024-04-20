# 第十二章：数学问题

# 问题

这是本章的问题解决部分。

# 1. 可被 3 和 5 整除的自然数之和

编写一个计算和打印所有自然数的程序，这些自然数可被 3 或 5 整除，直到用户输入的给定限制为止。

# 2. 最大公约数

编写一个程序，给定两个正整数，将计算并打印两者的最大公约数。

# 3. 最小公倍数

编写一个程序，给定两个或多个正整数，计算并打印它们的最小公倍数。

# 4. 给定数字以下的最大质数

编写一个程序，计算并打印小于用户提供的数字的最大质数，该数字必须是正整数。

# 5. 性感素数

编写一个程序，打印用户输入限制范围内的所有性感素数对。

# 6. 过剩数

编写一个程序，打印所有过剩数及其过剩值，直到用户输入的数字为止。

# 7. 亲和数

编写一个程序，打印小于 1,000,000 的所有亲和数对的列表。

# 8. 阿姆斯特朗数

编写一个程序，打印所有三位数的阿姆斯特朗数。

# 9. 数的质因数

编写一个程序，打印用户输入数字的质因数。

# 10. 格雷码

编写一个程序，显示所有 5 位数的普通二进制表示、格雷码表示和解码的格雷码值。

# 11. 将数值转换为罗马数字

编写一个程序，给定用户输入的数字，打印其罗马数字等价物。

# 12. 最大 Collatz 序列

编写一个程序，确定并打印出哪个数字最多产生最长的 Collatz 序列，以及它的长度是多少。

# 13. 计算 Pi 的值

编写一个计算 Pi 值的程序，精确到小数点后两位。

# 14. 验证 ISBN

编写一个程序，验证用户输入的 10 位值（作为字符串）是否表示有效的 ISBN-10 号码。

# 解决方案

以上是上述问题解决部分的解决方案。

# 1. 可被 3 和 5 整除的自然数之和

解决此问题的方法是迭代从 3（1 和 2 不能被 3 整除，因此没有测试它们的意义）到用户输入的限制的所有数字。使用模运算来检查一个数字除以 3 和 5 的余数是否为 0。然而，能够加到更大限制的技巧是使用`long long`而不是`int`或`long`进行求和，否则在加到 100,000 之前会发生溢出：

```cpp
int main()
{
   unsigned int limit = 0;
   std::cout << "Upper limit:";
   std::cin >> limit;

   unsigned long long sum = 0;
   for (unsigned int i = 3; i < limit; ++i)
   {
     if (i % 3 == 0 || i % 5 == 0)
        sum += i;
   }

   std::cout << "sum=" << sum << std::endl;
}
```

# 2. 最大公约数

两个或多个非零整数的最大公约数（*gcd*简称），也称为最大公因数（*gcf*）、最大公因数（*hcf*）、最大公度量（*gcm*）或最大公约数，是能够整除它们所有的最大正整数。可以计算 gcd 的几种方法；一种有效的方法是欧几里得算法。对于两个整数，该算法是：

```cpp
gcd(a,0) = a
gcd(a,b) = gcd(b, a mod b)
```

这可以在 C++中使用递归函数非常简单地实现：

```cpp
unsigned int gcd(unsigned int const a, unsigned int const b)
{
   return b == 0 ? a : gcd(b, a % b);
}
```

欧几里得算法的非递归实现应该如下所示：

```cpp
unsigned int gcd(unsigned int a, unsigned int b)
{
   while (b != 0) {
      unsigned int r = a % b;
      a = b;
      b = r;
   }
   return a;
}
```

在 C++17 中，头文件`<numeric>`中有一个名为`gcd()`的`constexpr`函数，用于计算两个数字的最大公约数。

# 3. 最小公倍数

两个或多个非零整数的**最小公倍数**（**lcm**），也称为最小公倍数，或最小公倍数，是可以被它们所有整除的最小正整数。计算最小公倍数的一种可能方法是将问题简化为计算最大公约数。在这种情况下使用以下公式：

```cpp
lcm(a, b) = abs(a, b) / gcd(a, b)
```

计算最小公倍数的函数可能如下所示：

```cpp
int lcm(int const a, int const b)
{
   int h = gcd(a, b);
   return h ? (a * (b / h)) : 0;
}
```

要计算多于两个整数的*lcm*，可以使用头文件`<numeric>`中的`std::accumulate`算法：

```cpp
template<class InputIt>
int lcmr(InputIt first, InputIt last)
{
   return std::accumulate(first, last, 1, lcm);
}
```

在 C++17 中，有一个名为`lcm()`的`constexpr`函数，位于头文件`<numeric>`中，用于计算两个数的最小公倍数。

# 4. 给定数字的最大质数

质数是只有两个因子 1 和本身的数。要找到小于给定数字的最大质数，你应该首先编写一个确定一个数是否为质数的函数，然后调用这个函数，从给定数字开始，向 1 递减直到遇到第一个质数。有各种算法可以确定一个数是否为质数。确定质数性的常见实现如下：

```cpp
bool is_prime(int const num) 
{
   if (num <= 3) { return num > 1; }
   else if (num % 2 == 0 || num % 3 == 0) 
   { 
      return false; 
   }
   else 
   {
      for (int i = 5; i * i <= num; i += 6) 
      {
         if (num % i == 0 || num % (i + 2) == 0) 
         {
            return false;
         }
      }
      return true;
   }
}
```

这个函数可以这样使用：

```cpp
int main()
{
   int limit = 0;
   std::cout << "Upper limit:";
   std::cin >> limit;

   for (int i = limit; i > 1; i--)
   {
      if (is_prime(i))
      {
         std::cout << "Largest prime:" << i << std::endl;
         return 0;
      }
   }
}
```

# 5. 性质质数对

性质质数是相差六的质数（例如 5 和 11，或 13 和 19）。还有*孪生质数*，相差两，和*表兄质数*，相差四。

在上一个挑战中，我们实现了一个确定整数是否为质数的函数。我们将重用该函数进行此练习。你需要做的是检查一个数字`n`是否为质数，数字`n+6`也是质数，并在这种情况下将这对数字打印到控制台上：

```cpp
int main()
{
   int limit = 0;
   std::cout << "Upper limit:";
   std::cin >> limit;

   for (int n = 2; n <= limit; n++)
   {
      if (is_prime(n) && is_prime(n+6))
      {
         std::cout << n << "," << n+6 << std::endl;
      }
   }
}
```

你可以将其作为进一步的练习来计算和显示性质质数的三元组、四元组和五元组。

# 6. 丰富数

丰富数，也被称为过剩数，是一个其真因子之和大于该数本身的数。一个数的真因子是除了该数本身以外的正的质因子。真因子之和超过该数本身的数量被称为过剩。例如，数字 12 有真因子 1、2、3、4 和 6。它们的和是 16，这使得 12 成为一个丰富数。它的过剩是 4（即 16-12）。

要确定真因子的和，我们尝试从 2 到该数的平方根的所有数字（所有质因子都小于或等于这个值）。如果当前数字，我们称之为`i`，能够整除该数，那么`i`和`num/i`都是因子。然而，如果它们相等（例如，如果`i=3`，而`n=9`，那么`i`能整除 9，但`n/i=3`），我们只添加`i`，因为真因子只能被添加一次。否则，我们添加`i`和`num/i`并继续：

```cpp
int sum_proper_divisors(int const number)
{
   int result = 1;
   for (int i = 2; i <= std::sqrt(number); i++)
   {
      if (number%i == 0)
      {
         result += (i == (number / i)) ? i : (i + number / i);
      }
   }
   return result;
}
```

打印丰富数就像迭代到指定的限制，计算真因子的和并将其与数字进行比较一样简单：

```cpp
void print_abundant(int const limit)
{
   for (int number = 10; number <= limit; ++number)
   {
      auto sum = sum_proper_divisors(number);
      if (sum > number)
      {
         std::cout << number << ", abundance=" 
                   << sum - number << std::endl;
      }
   }
}

int main()
{
   int limit = 0;
   std::cout << "Upper limit:";
   std::cin >> limit;

   print_abundant(limit);
}
```

# 7. 亲和数

如果一个数的真因子之和等于另一个数的真因子之和，那么这两个数被称为亲和数。一个数的真因子是除了该数本身以外的正的质因子。亲和数不应该与*友好数*混淆。例如，数字 220 的真因子是 1、2、4、5、10、11、20、22、44、55 和 110，它们的和是 284。284 的真因子是 1、2、4、71 和 142；它们的和是 220。因此，数字 220 和 284 被称为亲和数。

解决这个问题的方法是遍历所有小于给定限制的数字。对于每个数字，计算其真因子的和。我们称这个和为`sum1`。重复这个过程并计算`sum1`的真因子的和。如果结果等于原始数字，那么数字和`sum1`是亲和数：

```cpp
void print_amicables(int const limit)
{
   for (int number = 4; number < limit; ++number)
   {
      auto sum1 = sum_proper_divisors(number);
      if (sum1 < limit)
      {
         auto sum2 = sum_proper_divisors(sum1);
         if (sum2 == number && number != sum1)
         {
            std::cout << number << "," << sum1 << std::endl;
         }
      }
   }
}
```

在上面的示例中，`sum_proper_divisors()`是在丰富数问题的解决方案中看到的函数。

上述函数会两次打印数字对，比如 220,284 和 284,220。修改这个实现，只打印每对一次。

# 8. 阿姆斯特朗数

阿姆斯特朗数（以迈克尔·F·阿姆斯特朗命名），也称为自恋数，完美的数字不变量或完美的数字，是一个等于其自身的数字，当它们被提升到数字的幂时。例如，最小的阿姆斯特朗数是 153，它等于![](img/8a736b24-c3af-4da2-a9da-12789af4ee9e.png)。

要确定一个三位数是否是一个自恋数，您必须首先确定它的数字，以便对它们的幂求和。然而，这涉及到除法和取模运算，这些都是昂贵的。计算它的一个更快的方法是依赖于这样一个事实，即一个数字是数字的和，乘以 10 的零基位置的幂。换句话说，对于最多 1,000 的数字，我们有`a*10² + b*10² + c`。因为你只需要确定三位数，这意味着`a`将从 1 开始。这比其他方法更快，因为乘法比除法和取模运算更快。这样一个函数的实现看起来像这样：

```cpp
void print_narcissistics()
{
   for (int a = 1; a <= 9; a++)
   {
      for (int b = 0; b <= 9; b++)
      {
         for (int c = 0; c <= 9; c++)
         {
            auto abc = a * 100 + b * 10 + c;
            auto arm = a * a * a + b * b * b + c * c * c;
            if (abc == arm)
            {
               std::cout << arm << std::endl;
            }
         }
      }
   }
}
```

您可以将其作为进一步的练习，编写一个确定自恋数的函数，直到达到限制，而不管它们的位数如何。这样一个函数会更慢，因为你首先必须确定数字的数字序列，将它们存储在一个容器中，然后将数字加到适当的幂（数字的数量）。

# 9. 数字的质因数

正整数的质因数是能够完全整除该整数的质数。例如，8 的质因数是 2 x 2 x 2，42 的质因数是 2 x 3 x 7。要确定质因数，您应该使用以下算法：

1.  当`n`可以被 2 整除时，2 是一个质因数，必须添加到列表中，而`n`变为`n/2`的结果。完成此步骤后，`n`是一个奇数。

1.  从 3 迭代到`n`的平方根。当当前数字，我们称之为`i`，除以`n`时，`i`是一个质因数，必须添加到列表中，而`n`变为`n/i`的结果。当`i`不再除以`n`时，将`i`增加 2（以获得下一个奇数）。

1.  当`n`是大于 2 的质数时，上述步骤将不会导致`n`变为 1。因此，如果在第 2 步结束时`n`仍大于 2，则`n`是一个质因数。

```cpp
std::vector<unsigned long long> prime_factors(unsigned long long n)
{
   std::vector<unsigned long long> factors;
   while (n % 2 == 0) {
      factors.push_back(2);
      n = n / 2;
   }
   for (unsigned long long i = 3; i <= std::sqrt(n); i += 2)
   {
      while (n%i == 0) {
         factors.push_back(i);
         n = n / i;
      }
   }

   if (n > 2) 
      factors.push_back(n);
   return factors;
}

int main()
{
   unsigned long long number = 0;
   std::cout << "number:";
   std::cin >> number;

   auto factors = prime_factors(number);
   std::copy(std::begin(factors), std::end(factors),
        std::ostream_iterator<unsigned long long>(std::cout, " "));
}
```

作为进一步的练习，确定数字 600,851,475,143 的最大质因数。

# 10. 格雷码

格雷码，也称为反射二进制码或简单反射二进制，是一种二进制编码形式，其中两个连续的数字只相差一个位。要执行二进制反射格雷码编码，我们需要使用以下公式：

```cpp
if b[i-1] = 1 then g[i] = not b[i]
else g[i] = b[i]
```

这相当于以下内容：

```cpp
g = b xor (b logically right shifted 1 time)
```

要解码二进制反射格雷码，应使用以下公式：

```cpp
b[0] = g[0]
b[i] = g[i] xor b[i-1]
```

这些可以用 C++编写如下，对于 32 位无符号整数：

```cpp
unsigned int gray_encode(unsigned int const num)
{
   return num ^ (num >> 1);
}

unsigned int gray_decode(unsigned int gray)
{
   for (unsigned int bit = 1U << 31; bit > 1; bit >>= 1)
   {
      if (gray & bit) gray ^= bit >> 1;
   }
   return gray;
}
```

要打印所有 5 位整数，它们的二进制表示，编码的格雷码表示和解码的值，我们可以使用以下代码：

```cpp
std::string to_binary(unsigned int value, int const digits)
{
   return std::bitset<32>(value).to_string().substr(32-digits, digits);
}

int main()
{
   std::cout << "Number\tBinary\tGray\tDecoded\n";
   std::cout << "------\t------\t----\t-------\n";

   for (unsigned int n = 0; n < 32; ++n)
   {
      auto encg = gray_encode(n);
      auto decg = gray_decode(encg);

      std::cout 
         << n << "\t" << to_binary(n, 5) << "\t" 
         << to_binary(encg, 5) << "\t" << decg << "\n";
   }
}
```

# 11. 将数值转换为罗马数字

罗马数字，如今所知，使用七个符号：I = 1，V = 5，X = 10，L = 50，C = 100，D = 500，M = 1000。该系统使用加法和减法来组成数字符号。从 1 到 10 的符号是 I，II，III，IV，V，VI，VII，VIII，IX 和 X。罗马人没有零的符号，而是用*nulla*来表示。在这个系统中，最大的符号在左边，最不重要的在右边。例如，1994 年的罗马数字是 MCMXCIV。如果您不熟悉罗马数字的规则，您应该在网上阅读更多。

要确定一个数字的罗马数字，使用以下算法：

1.  从最高（M）到最低（I）检查每个罗马基本符号

1.  如果当前值大于符号的值，则将符号连接到罗马数字并从当前值中减去其值

1.  重复直到当前值达到零

例如，考虑 42：小于 42 的第一个罗马基本符号是 XL，它是 40。我们将它连接到罗马数字上，得到 XL，并从当前数字中减去，得到 2。小于 2 的第一个罗马基本符号是 I，它是 1。我们将它添加到罗马数字上，得到 XLI，并从数字中减去 1，得到 1。我们再添加一个 I 到罗马数字中，它变成了 XLII，并再次从数字中减去 1，达到 0，因此停止：

```cpp
std::string to_roman(unsigned int value)
{
   std::vector<std::pair<unsigned int, char const*>> roman {
      { 1000, "M" },{ 900, "CM" }, { 500, "D" },{ 400, "CD" }, 
      { 100, "C" },{ 90, "XC" }, { 50, "L" },{ 40, "XL" },
      { 10, "X" },{ 9, "IX" }, { 5, "V" },{ 4, "IV" }, { 1, "I" }};

   std::string result;
   for (auto const & kvp : roman) {
      while (value >= kvp.first) {
         result += kvp.second;
         value -= kvp.first;
      }
   }
   return result;
}
```

这个函数可以按照以下方式使用：

```cpp
int main()
{
   for(int i = 1; i <= 100; ++i) 
   {
      std::cout << i << "\t" << to_roman(i) << std::endl; 
   }

   int number = 0;
   std::cout << "number:";
   std::cin >> number;
   std::cout << to_roman(number) << std::endl;
}
```

# 12. 最大的 Collatz 序列

Collatz 猜想，也称为乌拉姆猜想、角谷谜题、斯韦茨猜想、哈斯算法或锡拉丘兹问题，是一个未经证实的猜想，它指出如下所述的序列总是达到 1。该系列定义如下：从任何正整数`n`开始，并从前一个整数获得每个新项：如果前一个项是偶数，则下一个项是前一个项的一半，否则是前一个项的 3 倍加 1。

你要解决的问题是生成所有小于一百万的正整数的 Collatz 序列，确定其中最长的序列，并打印其长度和产生它的起始数字。虽然我们可以应用蛮力法为每个数字生成序列并计算达到 1 之前的项数，但更快的解决方案是保存已经生成的所有序列的长度。当从值`n`开始的序列的当前项变小于`n`时，那么它是一个其序列已经被确定的数字，因此我们可以简单地获取其缓存长度并将其添加到当前长度以确定从`n`开始的序列的长度。然而，这种方法引入了对 Collatz 序列的计算的限制，因为在某个时候，缓存将超过系统可以分配的内存量：

```cpp
std::pair<unsigned long long, long> longest_collatz(
   unsigned long long const limit)
{
   long length = 0;
   unsigned long long number = 0;
   std::vector<int> cache(limit + 1, 0);

   for (unsigned long long i = 2; i <= limit; i++) 
   {
      auto n = i;
      long steps = 0;
      while (n != 1 && n >= i) 
      {
         if ((n % 2) == 0) n = n / 2;
         else n = n * 3 + 1;
         steps++;
      }
      cache[i] = steps + cache[n];

      if (cache[i] > length) 
      {
         length = cache[i];
         number = i;
```

```cpp
      }
   }

   return std::make_pair(number, length);
}
```

# 13. 计算 Pi 的值

用蒙特卡洛模拟大致确定 Pi 的值是一个合适的解决方案。这是一种使用输入的随机样本来探索复杂过程或系统行为的方法。该方法在许多应用和领域中使用，包括物理学、工程学、计算机、金融、商业等。

为了做到这一点，我们将依赖以下想法：直径为`d`的圆的面积是`PI * d² / 4`。边长等于`d`的正方形的面积是`d²`。如果我们将两者相除，我们得到`PI/4`。如果我们将圆放在正方形内并在正方形内生成均匀分布的随机数，那么圆内的数字计数应该与圆的面积成正比，正方形内的数字计数应该与正方形的面积成正比。这意味着将正方形和圆中的总命中数相除应该得到`PI/4`。生成的点越多，结果就越准确。

为了生成伪随机数，我们将使用 Mersenne twister 和均匀统计分布：

```cpp
template <typename E = std::mt19937, 
          typename D = std::uniform_real_distribution<>>
double compute_pi(E& engine, D& dist, int const samples = 1000000)
{
   auto hit = 0;
   for (auto i = 0; i < samples; i++)
   {
      auto x = dist(engine);
      auto y = dist(engine);
      if (y <= std::sqrt(1 - std::pow(x, 2))) hit += 1;
   }
   return 4.0 * hit / samples;
}

int main()
{
   std::random_device rd;
   auto seed_data = std::array<int, std::mt19937::state_size> {};
   std::generate(std::begin(seed_data), std::end(seed_data), 
                 std::ref(rd));
   std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
   auto eng = std::mt19937{ seq };
   auto dist = std::uniform_real_distribution<>{ 0, 1 };

   for (auto j = 0; j < 10; j++)
      std::cout << compute_pi(eng, dist) << std::endl;
}
```

# 14. 验证 ISBN

**国际标准书号**（**ISBN**）是书籍的唯一数字标识符。目前使用的是 13 位格式。然而，对于这个问题，你需要验证使用 10 位数字的旧格式。10 位数字中的最后一位是一个校验和。选择这一位数字是为了使所有十个数字的和，每个数字乘以它的（整数）权重，从 10 到 1 递减，是 11 的倍数。

`validate_isbn_10`函数如下所示，接受一个 ISBN 作为字符串，并在字符串长度为 10、所有十个元素都是数字，并且所有数字乘以它们的权重（或位置）的和是 11 的倍数时返回`true`：

```cpp
bool validate_isbn_10(std::string_view isbn)
{
   auto valid = false;
   if (isbn.size() == 10 &&
       std::count_if(std::begin(isbn), std::end(isbn), isdigit) == 10)
   {
      auto w = 10;
      auto sum = std::accumulate(
         std::begin(isbn), std::end(isbn), 0,
         &w {
            return total + w-- * (c - '0'); });

     valid = !(sum % 11);
   }
   return valid;
}
```

你可以把这看作是进一步练习，以改进这个函数，使其能够正确验证包括连字符的 ISBN-10 号码，比如`3-16-148410-0`。另外，你也可以编写一个验证 ISBN-13 号码的函数。
