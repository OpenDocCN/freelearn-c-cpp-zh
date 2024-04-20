# 字符串和正则表达式

# 问题

这是本章的问题解决部分。

# 23\. 二进制转字符串

编写一个函数，给定一个 8 位整数范围（例如数组或向量），返回一个包含输入数据十六进制表示的字符串。该函数应能够产生大写和小写内容。以下是一些输入和输出示例：

输入：`{ 0xBA, 0xAD, 0xF0, 0x0D }`，输出：`"BAADF00D"`或`"baadf00d"`

输入：`{ 1,2,3,4,5,6 }`，输出：`"010203040506"`

# 24\. 字符串转二进制

编写一个函数，给定一个包含十六进制数字的字符串作为输入参数，返回表示字符串内容的数值反序列化的 8 位整数向量。以下是示例：

输入：`"BAADF00D"`或`"baadF00D"`，输出：`{0xBA, 0xAD, 0xF0, 0x0D}`

输入`"010203040506"`，输出：`{1, 2, 3, 4, 5, 6}`

# 25\. 文章标题大写

编写一个函数，将输入文本转换为大写版本，其中每个单词以大写字母开头，其他所有字母都是小写。例如，文本`"the c++ challenger"`应转换为`"The C++ Challenger"`。

# 26\. 用分隔符连接字符串

编写一个函数，给定一个字符串列表和一个分隔符，通过连接所有输入字符串并用指定的分隔符分隔，创建一个新字符串。分隔符不得出现在最后一个字符串之后，当没有提供输入字符串时，函数必须返回一个空字符串。

示例：输入`{ "this","is","an","example" }`和分隔符`' '`（空格），输出：`"this is an example"`。

# 27\. 使用可能的分隔符将字符串拆分为标记

编写一个函数，给定一个字符串和可能的分隔符字符列表，将字符串分割成由任何分隔符分隔的标记，并将它们返回到一个`std::vector`中。

示例：输入：`"this,is.a sample!!"`，使用分隔符`",.! "`，输出：`{"this", "is", "a", "sample"}`。

# 28\. 最长回文子串

编写一个函数，给定输入字符串，找到并返回字符串中最长的回文序列。如果存在相同长度的多个回文序列，则应返回第一个。

# 29\. 车牌验证

考虑格式为`LLL-LL DDD`或`LLL-LL DDDD`（其中`L`是从*A*到*Z*的大写字母，`D`是数字）的车牌，编写：

+   一个验证车牌号是否为正确格式的函数

+   一个函数，给定输入文本，提取并返回文本中找到的所有车牌号

# 30\. 提取 URL 部分

编写一个函数，给定表示 URL 的字符串，解析并提取 URL 的各个部分（协议、域名、端口、路径、查询和片段）。

# 31\. 转换字符串中的日期

编写一个函数，给定一个包含格式为`dd.mm.yyyy`或`dd-mm-yyyy`的日期的文本，将文本转换为包含格式为`yyyy-mm-dd`的日期。

# 解决方案

这是上述问题解决部分的解决方案。

# 23\. 二进制转字符串

为了编写一个通用的函数，可以处理各种范围，如`std::array`、`std::vector`、类 C 数组或其他范围，我们应该编写一个函数模板。在下面，有两个重载；一个接受一个容器作为参数和一个标志，指示大小写风格，另一个接受一对迭代器（标记范围的第一个元素和最后一个元素的后一个元素）和指示大小写的标志。范围的内容被写入一个`std::ostringstream`对象，使用适当的 I/O 操纵器，如宽度、填充字符或大小写标志：

```cpp
template <typename Iter>
std::string bytes_to_hexstr(Iter begin, Iter end, 
                            bool const uppercase = false)
{
   std::ostringstream oss;
   if(uppercase) oss.setf(std::ios_base::uppercase);
   for (; begin != end; ++begin)
     oss << std::hex << std::setw(2) << std::setfill('0') 
         << static_cast<int>(*begin);
   return oss.str();
}

template <typename C>
std::string bytes_to_hexstr(C const & c, bool const uppercase = false)
{
   return bytes_to_hexstr(std::cbegin(c), std::cend(c), uppercase);
}
```

这些函数可以如下使用：

```cpp
int main()
{
   std::vector<unsigned char> v{ 0xBA, 0xAD, 0xF0, 0x0D };
   std::array<unsigned char, 6> a{ {1,2,3,4,5,6} };
   unsigned char buf[5] = {0x11, 0x22, 0x33, 0x44, 0x55};

   assert(bytes_to_hexstr(v, true) == "BAADF00D");
   assert(bytes_to_hexstr(a, true) == "010203040506");
   assert(bytes_to_hexstr(buf, true) == "1122334455");

   assert(bytes_to_hexstr(v) == "baadf00d");
   assert(bytes_to_hexstr(a) == "010203040506");
   assert(bytes_to_hexstr(buf) == "1122334455");
}
```

# 24\. 字符串转二进制

这里请求的操作与前一个问题中实现的相反。然而，这一次，我们可以编写一个函数而不是一个函数模板。输入是一个`std::string_view`，它是一个字符序列的轻量级包装器。输出是一个 8 位无符号整数的向量。下面的`hexstr_to_bytes`函数将每两个文本字符转换为一个`unsigned char`值（`"A0"`变成`0xA0`），将它们放入一个`std::vector`中，并返回该向量：

```cpp
unsigned char hexchar_to_int(char const ch)
{
   if (ch >= '0' && ch <= '9') return ch - '0';
   if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
   if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
      throw std::invalid_argument("Invalid hexadecimal character");
}

std::vector<unsigned char> hexstr_to_bytes(std::string_view str)
{
   std::vector<unsigned char> result;
   for (size_t i = 0; i < str.size(); i += 2) 
   {
      result.push_back(
         (hexchar_to_int(str[i]) << 4) | hexchar_to_int(str[i+1]));
   }
   return result;
}
```

这个函数假设输入字符串包含偶数个十六进制数字。在输入字符串包含奇数个十六进制数字的情况下，最后一个将被丢弃（所以`"BAD"`变成了`{0xBA}`）。作为进一步的练习，修改前面的函数，使得它不是丢弃最后一个奇数位，而是考虑一个前导零，这样`"BAD"`就变成了`{0x0B, 0xAD}`。另外，作为另一个练习，您可以编写一个函数的版本，它可以反序列化内容，其中十六进制数字由分隔符分隔，比如空格（例如`"BA AD F0 0D"`）。

下一个代码示例显示了如何使用这个函数：

```cpp
int main()
{
   std::vector<unsigned char> expected{ 0xBA, 0xAD, 0xF0, 0x0D, 0x42 };
   assert(hexstr_to_bytes("BAADF00D42") == expected);
   assert(hexstr_to_bytes("BaaDf00d42") == expected);
}
```

# 25\. 将文章标题大写

函数模板`capitalize()`，实现如下，可以处理任何类型字符的字符串。它不修改输入字符串，而是创建一个新的字符串。为此，它使用一个`std::stringstream`。它遍历输入字符串中的所有字符，并在遇到空格或标点符号时将指示新单词的标志设置为`true`。当它们表示一个单词中的第一个字符时，输入字符被转换为大写，否则转换为小写：

```cpp
template <class Elem>
using tstring = std::basic_string<Elem, std::char_traits<Elem>, 
                                  std::allocator<Elem>>;
template <class Elem>
using tstringstream = std::basic_stringstream<
   Elem, std::char_traits<Elem>, std::allocator<Elem>>;

template <class Elem>
tstring<Elem> capitalize(tstring<Elem> const & text)
{
   tstringstream<Elem> result;
   bool newWord = true;
   for (auto const ch : text)
   {
      newWord = newWord || std::ispunct(ch) || std::isspace(ch);
      if (std::isalpha(ch))
      {
         if (newWord)
         {
            result << static_cast<Elem>(std::toupper(ch));
            newWord = false;
         }
         else
            result << static_cast<Elem>(std::tolower(ch));
      }
      else result << ch;
   }
   return result.str();
}
```

在下面的程序中，您可以看到如何使用这个函数来大写文本：

```cpp
int main()
{
   using namespace std::string_literals;
   assert("The C++ Challenger"s ==
          capitalize("the c++ challenger"s));
   assert("This Is An Example, Should Work!"s == 
          capitalize("THIS IS an ExamplE, should wORk!"s));
}
```

# 26\. 用分隔符连接字符串

以下代码中列出了两个名为`join_strings()`的重载。一个接受一个字符串容器和一个表示分隔符的字符序列的指针，而另一个接受两个随机访问迭代器，表示范围的第一个和最后一个元素，以及一个分隔符。它们都返回一个通过连接所有输入字符串创建的新字符串，使用输出字符串流和`std::copy`函数。这个通用函数将指定范围中的所有元素复制到一个输出范围中，由输出迭代器表示。我们在这里使用了一个`std::ostream_iterator`，它使用`operator<<`每次迭代器被赋予一个值时将指定的值写入指定的输出流： 

```cpp
template <typename Iter>
std::string join_strings(Iter begin, Iter end, 
                         char const * const separator)
{
   std::ostringstream os;
   std::copy(begin, end-1, 
             std::ostream_iterator<std::string>(os, separator));
   os << *(end-1);
   return os.str();
}

template <typename C>
std::string join_strings(C const & c, char const * const separator)
{
   if (c.size() == 0) return std::string{};
   return join_strings(std::begin(c), std::end(c), separator);
}

int main()
{
   using namespace std::string_literals;
   std::vector<std::string> v1{ "this","is","an","example" };
   std::vector<std::string> v2{ "example" };
   std::vector<std::string> v3{ };

   assert(join_strings(v1, " ") == "this is an example"s);
   assert(join_strings(v2, " ") == "example"s);
   assert(join_strings(v3, " ") == ""s);
}
```

作为进一步的练习，您应该修改接受迭代器作为参数的重载，以便它可以与其他类型的迭代器一起工作，比如双向迭代器，从而使得可以使用这个函数与列表或其他容器一起使用。

# 27\. 使用可能的分隔符列表将字符串拆分为标记

两种不同版本的拆分函数如下所示：

+   第一个使用单个字符作为分隔符。为了拆分输入字符串，它使用一个字符串流，该字符串流初始化为输入字符串的内容，使用`std::getline()`从中读取块，直到遇到下一个分隔符或行尾字符。

+   第二个版本使用了一个可能的字符分隔符列表，指定在`std::string`中。它使用`std:string::find_first_of()`来定位从给定位置开始的任何分隔符字符的第一个位置。它在循环中这样做，直到整个输入字符串被处理。提取的子字符串被添加到结果向量中：

```cpp
template <class Elem>
using tstring = std::basic_string<Elem, std::char_traits<Elem>, 
                                  std::allocator<Elem>>;

template <class Elem>
using tstringstream = std::basic_stringstream<
   Elem, std::char_traits<Elem>, std::allocator<Elem>>;
template<typename Elem>
inline std::vector<tstring<Elem>> split(tstring<Elem> text, 
                                        Elem const delimiter)
{
   auto sstr = tstringstream<Elem>{ text };
   auto tokens = std::vector<tstring<Elem>>{};
   auto token = tstring<Elem>{};
   while (std::getline(sstr, token, delimiter))
   {
      if (!token.empty()) tokens.push_back(token);
   }
   return tokens;
}

template<typename Elem>
inline std::vector<tstring<Elem>> split(tstring<Elem> text, 
                                        tstring<Elem> const & delimiters)
{
   auto tokens = std::vector<tstring<Elem>>{};
   size_t pos, prev_pos = 0;
   while ((pos = text.find_first_of(delimiters, prev_pos)) != 
   std::string::npos)
   {
      if (pos > prev_pos)
      tokens.push_back(text.substr(prev_pos, pos - prev_pos));
      prev_pos = pos + 1;
   }
   if (prev_pos < text.length())
   tokens.push_back(text.substr(prev_pos, std::string::npos));
   return tokens;
}
```

下面的示例代码显示了如何使用一个分隔符字符或多个分隔符来拆分不同的字符串的两个示例：

```cpp
int main()
{
   using namespace std::string_literals;
   std::vector<std::string> expected{"this", "is", "a", "sample"};
   assert(expected == split("this is a sample"s, ' '));
   assert(expected == split("this,is a.sample!!"s, ",.! "s));
}
```

# 28\. 最长回文子字符串

解决这个问题的最简单方法是尝试蛮力方法，检查每个子字符串是否为回文。然而，这意味着我们需要检查*C(N, 2)*个子字符串（其中*N*是字符串中的字符数），时间复杂度将是*![](img/76505ab6-7d29-4aab-9955-744ed0bcd1b6.png)*。通过存储子问题的结果，复杂度可以降低到*![](img/2f7e78fe-014a-40b2-9524-bc0f479781a1.png)*。为此，我们需要一个大小为![](img/a4173824-4963-42ca-b9ab-fd97affe7750.png)的布尔值表，其中`[i, j]`处的元素指示位置`i`到`j`的子字符串是否为回文。我们首先通过将所有`[i,i]`处的元素初始化为`true`（单字符回文）和所有`[i,i+i]`处的元素初始化为`true`（所有连续两个相同字符的两字符回文）来开始。然后，我们继续检查大于两个字符的子字符串，如果`[i+i,j-1]`处的元素为`true`且字符串中位置`i`和`j`的字符也相等，则将`[i,j]`处的元素设置为`true`。在此过程中，我们保留最长回文子字符串的起始位置和长度，以便在完成计算表后提取它。

在代码中，这个解决方案如下所示：

```cpp
std::string longest_palindrome(std::string_view str)
{
   size_t const len = str.size();
   size_t longestBegin = 0;
   size_t maxLen = 1;

   std::vector<bool> table(len * len, false);
   for (size_t i = 0; i < len; i++)
      table[i*len + i] = true;

   for (size_t i = 0; i < len - 1; i++)
   {
      if (str[i] == str[i + 1]) 
      {
         table[i*len + i + 1] = true;
         if (maxLen < 2)
         {
            longestBegin = i;
            maxLen = 2;
         }
      }
   }

   for (size_t k = 3; k <= len; k++)
   {
      for (size_t i = 0; i < len - k + 1; i++)
      {
         size_t j = i + k - 1;
         if (str[i] == str[j] && table[(i + 1)*len + j - 1])
         {
            table[i*len +j] = true;
            if (maxLen < k)
            {
               longestBegin = i;
               maxLen = k;
            }
         }
      }
   }
   return std::string(str.substr(longestBegin, maxLen));
}
```

以下是`longest_palindrome()`函数的一些测试用例：

```cpp
int main()
{
   using namespace std::string_literals;
   assert(longest_palindrome("sahararahnide") == "hararah");
   assert(longest_palindrome("level") == "level");
   assert(longest_palindrome("s") == "s");
}
```

# 29\. 验证车牌

解决这个问题的最简单方法是使用正则表达式。符合描述格式的正则表达式是`"[A-Z]{3}-[A-Z]{2} \d{3,4}"`。

第一个函数只需验证输入字符串是否只包含与此正则表达式匹配的文本。为此，我们可以使用`std::regex_match()`，如下所示：

```cpp
bool validate_license_plate_format(std::string_view str)
{
   std::regex rx(R"([A-Z]{3}-[A-Z]{2} \d{3,4})");
   return std::regex_match(str.data(), rx);
}

int main()
{
   assert(validate_license_plate_format("ABC-DE 123"));
   assert(validate_license_plate_format("ABC-DE 1234"));
   assert(!validate_license_plate_format("ABC-DE 12345"));
   assert(!validate_license_plate_format("abc-de 1234"));
}
```

第二个函数略有不同。它不是匹配输入字符串，而是必须识别字符串中正则表达式的所有出现。因此，正则表达式将更改为`"([A-Z]{3}-[A-Z]{2} \d{3,4})*"`。要遍历所有匹配项，我们必须使用`std::sregex_iterator`，如下所示：

```cpp
std::vector<std::string> extract_license_plate_numbers(
                            std::string const & str)
{
   std::regex rx(R"(([A-Z]{3}-[A-Z]{2} \d{3,4})*)");
   std::smatch match;
   std::vector<std::string> results;

   for(auto i = std::sregex_iterator(std::cbegin(str), std::cend(str), rx); 
       i != std::sregex_iterator(); ++i) 
   {
      if((*i)[1].matched)
      results.push_back(i->str());
   }
   return results;
}

int main()
{
   std::vector<std::string> expected {
      "AAA-AA 123", "ABC-DE 1234", "XYZ-WW 0001"};
   std::string text("AAA-AA 123qwe-ty 1234 ABC-DE 123456..XYZ-WW 0001");
   assert(expected == extract_license_plate_numbers(text));
}
```

# 30\. 提取 URL 部分

这个问题也适合使用正则表达式来解决。然而，找到一个可以匹配任何 URL 的正则表达式是一个困难的任务。这个练习的目的是帮助您练习正则表达式库的技能，而不是找到特定目的的终极正则表达式。因此，这里使用的正则表达式仅供教学目的。

您可以使用在线测试器和调试器，如[`regex101.com/`](https://regex101.com/)，尝试正则表达式。这可以帮助您解决正则表达式并针对各种数据集尝试它们。

对于此任务，我们将认为 URL 具有以下部分：`protocol`和`domain`是必需的，而`port`、`path`、`query`和`fragment`都是可选的。以下结构用于从解析 URL 返回结果（或者，您可以返回一个元组，并使用结构化绑定将变量绑定到元组的各个子部分）：

```cpp
struct uri_parts
{
   std::string                protocol;
   std::string                domain;
   std::optional<int>         port;
   std::optional<std::string> path;
   std::optional<std::string> query;
   std::optional<std::string> fragment;
};
```

可以解析 URL 并提取并返回其部分的函数可能具有以下实现。请注意，返回类型是`std::optional<uri_parts>`，因为该函数可能无法将输入字符串与正则表达式匹配；在这种情况下，返回值为`std::nullopt`：

```cpp
std::optional<uri_parts> parse_uri(std::string uri)
{
   std::regex rx(R"(^(\w+):\/\/([\w.-]+)(:(\d+))?([\w\/\.]+)?(\?([\w=&]*)(#?(\w+))?)?$)");
   auto matches = std::smatch{};
   if (std::regex_match(uri, matches, rx))
   {
      if (matches[1].matched && matches[2].matched)
      {
         uri_parts parts;
         parts.protocol = matches[1].str();
         parts.domain = matches[2].str();
         if (matches[4].matched)
            parts.port = std::stoi(matches[4]);
         if (matches[5].matched)
            parts.path = matches[5];
         if (matches[7].matched)
            parts.query = matches[7];
         if (matches[9].matched)
            parts.fragment = matches[9];
         return parts;
      }
   }
   return {};
}
```

以下程序使用包含不同部分的两个 URL 测试`parse_uri()`函数：

```cpp
int main()
{
   auto p1 = parse_uri("https://packt.com");
   assert(p1.has_value());
   assert(p1->protocol == "https");
   assert(p1->domain == "packt.com");
   assert(!p1->port.has_value());
   assert(!p1->path.has_value());
   assert(!p1->query.has_value());
   assert(!p1->fragment.has_value());

   auto p2 = parse_uri("https://bbc.com:80/en/index.html?lite=true#ui");
   assert(p2.has_value());
   assert(p2->protocol == "https");
   assert(p2->domain == "bbc.com");
   assert(p2->port == 80);
   assert(p2->path.value() == "/en/index.html");
   assert(p2->query.value() == "lite=true");
   assert(p2->fragment.value() == "ui");
}
```

# 31\. 将字符串中的日期转换

可以使用`std::regex_replace()`和正则表达式执行文本转换。可以匹配指定格式日期的正则表达式是`(\d{1,2})(\.|-|/)(\d{1,2})(\.|-|/)(\d{4})`。这个正则表达式定义了五个捕获组；第一个是日期，第二个是分隔符（`.`或`-`），第三个是月份，第四个再次是分隔符（`.`或`-`），第五个是年份。

由于我们想要将日期从格式 `dd.mm.yyyy` 或 `dd-mm-yyyy` 转换为 `yyyy-mm-dd`，因此 `std::regex_replace()` 的正则表达式替换格式字符串应该是 `"($5-$3-$1)"`：

```cpp
std::string transform_date(std::string_view text)
{
   auto rx = std::regex{ R"((\d{1,2})(\.|-|/)(\d{1,2})(\.|-|/)(\d{4}))" };
   return std::regex_replace(text.data(), rx, R"($5-$3-$1)");
}

int main()
{
   using namespace std::string_literals;
   assert(transform_date("today is 01.12.2017!"s) == 
          "today is 2017-12-01!"s);
}
```
