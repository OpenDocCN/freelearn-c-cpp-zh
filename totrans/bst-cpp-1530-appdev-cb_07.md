# 第七章. 字符串操作

在本章中，我们将涵盖：

+   改变大小写和不区分大小写的比较

+   使用正则表达式匹配字符串

+   使用正则表达式搜索和替换字符串

+   使用安全的 printf-like 函数格式化字符串

+   替换和删除字符串

+   使用两个迭代器表示字符串

+   使用字符串类型的引用

# 简介

整章都致力于字符串更改、搜索和表示的不同方面。我们将看到如何使用 Boost 库轻松完成一些常见的字符串相关任务。本章内容足够简单；它解决了非常常见的字符串操作任务。那么，让我们开始吧！

# 改变大小写和不区分大小写的比较

这是一个相当常见的任务。我们有两个非 Unicode 或 ANSI 字符字符串：

```cpp
#include <string>
std::string str1 = "Thanks for reading me!";
std::string str2 = "Thanks for reading ME!";
```

我们需要以不区分大小写的方式比较它们。有很多方法可以做到这一点；让我们看看 Boost 的方法。

## 准备工作

在这里我们只需要`std::string`的基本知识。

## 如何做到这一点...

这里有一些不同的方法来进行不区分大小写的比较：

1.  最简单的一个是：

    ```cpp
    #include <boost/algorithm/string/predicate.hpp>

    boost::iequals(str1, str2)
    ```

1.  使用 Boost 谓词和 STL 方法：

    ```cpp
    #include <boost/algorithm/string/compare.hpp>
    #include <algorithm>

    str1.size() == str2.size() && std::equal(
      str1.begin(),
      str1.end(),
      str2.begin(),
      boost::is_iequal()
    )
    ```

1.  创建两个字符串的小写副本：

    ```cpp
    #include <boost/algorithm/string/case_conv.hpp>

    std::string str1_low = boost::to_lower_copy(str1);
    std::string str2_low = boost::to_lower_copy(str2);
    assert(str1_low == str2_low);
    ```

1.  创建原始字符串的大写副本：

    ```cpp
    #include <boost/algorithm/string/case_conv.hpp>

    std::string str1_up = boost::to_upper_copy(str1);
    std::string str2_up = boost::to_upper_copy(str2);
    assert(str1_up == str2_up);
    ```

1.  将原始字符串转换为小写：

    ```cpp
    #include <boost/algorithm/string/case_conv.hpp>

    boost::to_lower(str1);
    boost::to_lower(str2);
    assert(str1 == str2);
    ```

## 它是如何工作的...

第二种方法并不明显。在第二种方法中，我们比较字符串的长度；如果它们的长度相同，我们使用`boost::is_iequal`谓词逐字符比较字符串。`boost::is_iequal`谓词以不区分大小写的方式比较两个字符。

### 注意

`Boost.StringAlgorithm`库在方法或类的名称中使用`i`，如果这个方法是不区分大小写的。例如，`boost::is_iequal`、`boost::iequals`、`boost::is_iless`以及其他。

## 还有更多...

`Boost.StringAlgorithm`库中所有与大小写相关的函数和功能对象都接受`std::locale`。默认情况下（以及在我们的示例中），方法和类使用默认构造的`std::locale`。如果我们大量处理字符串，那么构造一个`std::locale`变量一次并传递给所有方法可能是一个很好的优化。另一个好的优化是使用'C'区域设置（如果您的应用程序逻辑允许的话）通过`std::locale::classic()`：

```cpp
  // On some platforms std::locale::classic() works
  // faster than std::locale()
  boost::iequals(str1, str2, std::locale::classic());
```

### 注意

没有什么禁止你使用这两种优化。

很不幸，C++11 没有`Boost.StringAlgorithm`的字符串函数。所有算法都是快速且可靠的，所以不要害怕在你的代码中使用它们。

## 参见

+   关于 Boost 字符串算法库的官方文档可以在[`www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html)找到

+   请参阅 Andrei Alexandrescu 和 Herb Sutter 所著的《*C++编码标准*》一书，了解如何用几行代码创建一个不区分大小写的字符串的示例。

# 使用正则表达式匹配字符串

让我们做一些有用的事情！通常，用户的输入必须使用某些正则表达式特定的模式进行检查，这提供了一种灵活的匹配方式。问题是正则表达式语法有很多；使用一种语法编写的表达式不能很好地由另一种语法处理。另一个问题是长正则表达式不容易编写。

因此，在这个菜谱中，我们将编写一个程序，该程序可能使用不同类型的正则表达式语法，并检查输入字符串是否与指定的正则表达式匹配。

## 准备中

这个菜谱需要基本的 STL 知识。了解正则表达式语法可能会有帮助，但并非必需。

需要将示例链接到 `libboost_regex` 库。

## 如何做到这一点...

这个正则表达式匹配器由 `main()` 函数中的几行代码组成；然而，我经常使用它。它总有一天会帮到你的。

1.  为了实现它，我们需要以下头文件：

    ```cpp
    #include <boost/regex.hpp>
    #include <iostream>
    ```

1.  在程序开始时，我们需要输出可用的正则表达式语法：

    ```cpp
    int main() {
      std::cout 
        << "Available regex syntaxes:\n"
        << "\t[0] Perl\n"
        << "\t[1] Perl case insensitive\n"
        << "\t[2] POSIX extended\n"
        << "\t[3] POSIX extended case insensitive\n"
        << "\t[4] POSIX basic\n"
        << "\t[5] POSIX basic case insensitive\n"
        << "Choose regex syntax: ";
    ```

1.  现在根据选择的语法正确设置标志：

    ```cpp
      boost::regex::flag_type flag;
      switch (std::cin.get()) {
        case '0': flag = boost::regex::perl;
          break;
        case '1': flag = boost::regex::perl|boost::regex::icase;
          break;

        case '2': flag = boost::regex::extended;
          break;
        case '3': flag = boost::regex::extended|boost::regex::icase;
          break;
        case '4': flag = boost::regex::basic;
          break;

        case '5': flag = boost::regex::basic|boost::regex::icase;
          break;
        default:
          std::cout << "Inccorect number of regex syntax."
                    <<"Exiting... \n";
          return -1;
      } 
      // Disabling exceptions
      flag |= boost::regex::no_except;
    ```

1.  现在，我们将循环请求正则表达式模式：

    ```cpp
      // Restoring std::cin
      std::cin.ignore();
      std::cin.clear();

      std::string regex, str;
      do {
        std::cout << "Input regex: ";
        if (!std::getline(std::cin, regex) || regex.empty()) {
          return 0;
        }

        // Without `boost::regex::no_except`flag this 
        // constructor may throw
        const boost::regex e(regex, flag);
        if (e.status()) {
          std::cout << "Incorrect regex pattern!\n";
          continue;
        }
    ```

1.  在循环中获取一个字符串进行匹配：

    ```cpp
        std::cout << "String to match: ";
        while (std::getline(std::cin, str) && !str.empty()) {
    ```

1.  将正则表达式应用于它并输出结果：

    ```cpp
          bool matched = boost::regex_match(str, e);
          std::cout << (matched ? "MATCH\n" : "DOES NOT MATCH\n");
          std::cout << "String to match: ";
        } // end of `while (std::getline(std::cin, str))`
    ```

1.  通过恢复 `std::cin` 并请求新的正则表达式模式来完成我们的示例：

    ```cpp
        // Restoring std::cin
        std::cin.ignore();
        std::cin.clear();
      } while (1);
    } // int main()
    ```

    现在如果我们运行前面的示例，我们会得到以下输出：

    ```cpp
    Available regex syntaxes:
            [0] Perl
            [1] Perl case insensitive
            [2] POSIX extended
            [3] POSIX extended case insensitive
            [4] POSIX basic
            [5] POSIX basic case insensitive
    Choose regex syntax: 0
    Input regex: (\d{3}[#-]){2}
    String to match: 123-123#
    MATCH
    String to match: 312-321-
    MATCH
    String to match: 21-123-
    DOES NOT MATCH
    String to match: ^Z
    Input regex: \l{3,5}
    String to match: qwe
    MATCH
    String to match: qwert
    MATCH
    String to match: qwerty
    DOES NOT MATCH
    String to match: QWE
    DOES NOT MATCH
    String to match: ^Z

    Input regex: ^Z
    Press any key to continue . . .
    ```

## 它是如何工作的...

所有这些都是由 `boost::regex` 类完成的。它构建了一个能够进行正则表达式解析和编译的对象。`flags` 变量添加了额外的配置选项。

如果正则表达式不正确，它会抛出异常；如果传递了 `boost::regex::no_except` 标志，它会在 `status()` 调用中返回非零值以报告错误（就像在我们的示例中一样）：

```cpp
if (e.status()) {
  std::cout << "Incorrect regex pattern!\n";
  continue;
}
```

这将导致：

```cpp
Input regex: (incorrect regex(
Incorrect regex pattern!
Input regex:

```

正则表达式匹配是通过调用 `boost::regex_match` 函数来完成的。如果匹配成功，则返回 `true`。可以传递额外的标志给 `regex_match`，但我们为了避免示例的简洁性而避免了它们的用法。

## 还有更多...

C++11 几乎包含了所有的 `Boost.Regex` 类和标志。它们可以在 `std::` 命名空间中的 `<regex>` 头文件中找到（而不是 `boost::`）。官方文档提供了有关 C++11 和 `Boost.Regex` 之间差异的信息。它还包含了一些性能指标，表明 `Boost.Regex` 很快。

## 参见

+   “使用正则表达式搜索和替换字符串”菜谱将为你提供更多有关 `Boost.Regex` 使用的详细信息

+   你也可以考虑官方文档来获取有关标志、性能指标、正则表达式语法和 C++11 兼容性的更多信息，请参阅 [`www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html)

# 使用正则表达式搜索和替换字符串

我的妻子非常喜欢 *使用正则表达式匹配字符串* 配方，并告诉我，除非我将它改进到能够根据正则表达式匹配替换输入字符串的部分，否则我不会得到食物。每个匹配的子表达式（正则表达式中的括号部分）必须从 1 开始有一个唯一的数字；这个数字将用于创建一个新的字符串。

这就是更新后的程序将如何工作的样子：

```cpp
Available regex syntaxes:
        [0] Perl
        [1] Perl case insensitive
        [2] POSIX extended
        [3] POSIX extended case insensitive
        [4] POSIX basic
        [5] POSIX basic case insensitive
Choose regex syntax: 0

Input regex: (\d)(\d)
String to match: 00
MATCH: 0, 0,
Replace pattern: \1#\2
RESULT: 0#0
String to match: 42
MATCH: 4, 2,
Replace pattern: ###\1-\1-\2-\1-\1###
RESULT: ###4-4-2-4-4###
…
```

## 准备工作

我们将使用来自 *使用正则表达式匹配字符串* 配方的代码。在使用此配方之前，您应该阅读它。

需要将示例链接到 `libboost_regex` 库。

## 如何做到这一点...

此配方基于上一个配方中的代码。让我们看看需要更改什么。

1.  不需要包含额外的头文件；然而，我们需要一个额外的字符串来存储替换模式：

    ```cpp
      std::string regex, str, replace_string;
    ```

1.  我们将用 `boost::regex_find` 替换 `boost::regex_match` 并输出匹配的结果：

    ```cpp
      std::cout << "String to match: ";
      while (std::getline(std::cin, str) && !str.empty()) {
        boost::smatch results;
        bool matched = regex_search(str, results, e);
        if (matched) {
          std::cout << "MATCH: ";
          std::copy(
            results.begin() + 1, 
            results.end(), 
            std::ostream_iterator<std::string>( std::cout, ", ")
          );
    ```

1.  之后，我们需要获取替换模式并应用它：

    ```cpp
          std::cout << "\nReplace pattern: ";
          if (std::getline(std::cin, replace_string) && !replace_string.empty()) {
            std::cout << "RESULT: " << boost::regex_replace(str, e, replace_string); 
          } else {
            // Restoring std::cin
            std::cin.ignore();
            std::cin.clear();
          }
        } else { // `if (matched) `
          std::cout << "DOES NOT MATCH";
        }
    ```

就这样！大家都满意，我也吃饱了。

## 它是如何工作的...

`boost::regex_search` 函数不仅返回一个真或假（如 `boost::regex_match` 函数所做的那样）的值，而且还存储匹配的部分。我们使用以下构造来输出匹配的部分：

```cpp
    std::copy(
      results.begin() + 1, 
      results.end(), 
      std::ostream_iterator<std::string>( std::cout, ", ")
    );
```

注意，我们通过跳过第一个结果（`results.begin() + 1`）来输出结果；这是因为 `results.begin()` 包含整个正则表达式匹配。

`boost::regex_replace` 函数执行所有替换并返回修改后的字符串。

## 还有更多...

`regex_*` 函数有不同的变体；其中一些接收双向迭代器而不是字符串，而另一些则向迭代器提供输出。

`boost::smatch` 是 `boost::match_results<std::string::const_iterator>` 的 `typedef`；因此，如果您使用的是 `std::string::const_iterator` 以外的其他双向迭代器，您需要将您的双向迭代器类型用作 `match_results` 的模板参数。

`match_results` 具有格式化功能，因此我们可以用它来调整示例。而不是：

```cpp
std::cout << "RESULT: " << boost::regex_replace(str, e, replace_string);
```

我们可以使用以下内容：

```cpp
std::cout << "RESULT: " << results.format(replace_string);
```

顺便说一句，`replace_string` 可能具有不同的格式：

```cpp
Input regex: (\d)(\d)
String to match: 12
MATCH: 1, 2,
Replace pattern: $1-$2---$&---$$
RESULT: 1-2---12---$
```

此配方中的所有类和函数都存在于 C++11 中，位于 `<regex>` 头文件的 `std::` 命名空间中。

## 参考以下内容

+   关于 `Boost.Regex` 的官方文档将为您提供更多示例以及有关性能、C++11 标准兼容性和正则表达式语法的更多信息，请参阅 [`www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html)。*使用正则表达式匹配字符串* 配方将向您介绍 `Boost.Regex` 的基础知识。

# 使用安全的 printf-like 函数格式化字符串

`printf`函数族对安全性构成了威胁。允许用户将他们自己的字符串作为类型并格式化说明符是非常糟糕的设计。那么当需要用户定义的格式时我们该怎么办？我们该如何实现以下类的`std::string to_string(const std::string& format_specifier) const;`成员函数？

```cpp
class i_hold_some_internals {
  int i;
  std::string s;
  char c;
  // ...
};
```

## 准备工作

对于这个示例，只需要基本的 STL 知识就足够了。

## 如何做到这一点...

我们希望允许用户为字符串指定自己的输出格式。

1.  为了安全地做到这一点，我们需要以下头文件：

    ```cpp
    #include <boost/format.hpp>
    ```

1.  现在我们将为用户添加一些注释：

    ```cpp
      // fmt parameter must contain the following:
      //  $1$ for outputting integer 'i'
      //  $2$ for outputting string 's'
      //  $3$ for outputting character 'c'
      std::string 
        to_string(const std::string& format_specifier) const {
    ```

1.  现在是时候让它们全部工作了：

    ```cpp
        boost::format f(format_specifier);
        unsigned char flags = boost::io::all_error_bits;
        flags ^= boost::io::too_many_args_bit;
        f.exceptions(flags);
        return (f % i % s % c).str();
      }
    ```

    就这些了。看看这段代码：

    ```cpp
      i_hold_some_internals class_instance;

      std::cout << class_instance.to_string(
        "Hello, dear %2%! "
        "Did you read the book for %1% %% %3%\n"
      );

      std::cout << class_instance.to_string(
        "%1% == %1% && %1%%% != %1%\n\n"
      );
    ```

    假设`class_instance`有一个成员`i`等于`100`，一个成员`s`等于`"Reader"`，一个成员`c`等于`'!'`。然后，程序将输出以下内容：

    ```cpp
    Hello, dear Reader! Did you read the book for 100 % !
    100 == 100 && 100% != 100
    ```

## 它是如何工作的...

`boost::format`类接受指定结果的字符串。参数通过`operator%`传递给`boost::format`。在格式指定字符串中指定的`%1%`、`%2%`、`%3%`、`%4%`等值将被传递给`boost::format`的参数替换。

当格式字符串包含的参数少于传递给`boost::format`的参数时，我们禁用异常：

```cpp
  boost::format f(format_specifier);
  unsigned char flags = boost::io::all_error_bits;
  flags ^= boost::io::too_many_args_bit;
```

这样做是为了允许一些格式，例如：

```cpp
  // Outputs 'Reader'
  std::cout << class_instance.to_string("%2%\n\n");
```

## 更多...

如果格式不正确会发生什么？

```cpp
  try {
    class_instance.to_string("%1% %2% %3% %4% %5%\n");
    assert(false);
  } catch (const std::exception& e) {
    // boost::io::too_few_args exception must be caught
    std::cout << e.what() << '\n';
  }
```

好吧，在这种情况下，不会触发断言，以下行将被输出到控制台：

```cpp
boost::too_few_args: format-string referred to more arguments than were passed
```

C++11 没有`std::format`。`Boost.Format`库不是一个非常快的库；尽量不在性能关键部分使用它。

## 参见

+   官方文档包含了关于`Boost.Format`库性能的更多信息。有关扩展 printf-like 格式的更多示例和文档，请访问[`www.boost.org/doc/libs/1_53_0/libs/format/`](http://www.boost.org/doc/libs/1_53_0/libs/format/)

# 字符串的替换和删除

需要在字符串中删除某些内容、替换字符串的一部分或删除子字符串的第一个或最后一个出现的情况非常常见。STL 允许我们完成大部分这些操作，但通常需要编写过多的代码。

我们在*改变大小写和大小写不敏感比较*的示例中看到了`Boost.StringAlgorithm`库的应用。让我们看看它如何简化我们在需要修改字符串时的生活：

```cpp
#include <string>
const std::string str = "Hello, hello, dear Reader.";
```

## 准备工作

对于这个示例，需要基本的 C++知识。

## 如何做到这一点...

这个示例展示了`Boost.StringAlgorithm`库中不同的字符串删除和替换方法是如何工作的。

删除操作需要包含`#include <boost/algorithm/string/erase.hpp>`头文件：

```cpp
namespace ba = boost::algorithm;
std::cout << "\n erase_all_copy   :" << ba::erase_all_copy(str, ",");
std::cout << "\n erase_first_copy :" << ba::erase_first_copy(str, ",");
std::cout << "\n erase_last_copy  :" << ba::erase_last_copy(str, ",");
std::cout << "\n ierase_all_copy  :" << ba::ierase_all_copy(str, "hello");
std::cout << "\n ierase_nth_copy  :" << ba::ierase_nth_copy(str, ",", 1);
```

这段代码将输出以下内容：

```cpp
erase_all_copy     :Hello hello dear Reader.
erase_first_copy   :Hello hello, dear Reader.
erase_last_copy    :Hello, hello dear Reader.
ierase_all_copy    :, , dear Reader.
ierase_nth_copy    :Hello, hello dear Reader.
```

替换操作需要包含`<boost/algorithm/string/replace.hpp>`头文件：

```cpp
namespace ba = boost::algorithm;

std::cout << "\n replace_all_copy  :" << ba::replace_all_copy(str, ",", "!");
std::cout << "\n replace_first_copy  :" << ba::replace_first_copy(str, ",", "!");
std::cout << "\n replace_head_copy  :" << ba::replace_head_copy(str, 6, "Whaaaaaaa!");
```

这段代码将输出以下内容：

```cpp
replace_all_copy    :Hello! hello! dear Reader.
replace_first_copy  :Hello! hello, dear Reader.
replace_head_copy   :Whaaaaaaa! hello, dear Reader.
```

## 它是如何工作的...

所有示例都是自文档化的。唯一不明显的是`replace_head_copy`函数。它接受要替换的字节数作为第二个参数，以及替换字符串作为第三个参数。所以，在上面的例子中，`Hello`被替换为`Whaaaaaaa!`。

## 还有更多...

还有修改字符串的内置方法。它们不仅以`_copy`结尾并返回`void`。所有不区分大小写的方法（以`i`开头的方法）接受`std::locale`作为最后一个参数，并使用默认构造的`locale`作为默认参数。

C++11 没有`Boost.StringAlgorithm`方法和类。

## 参见

+   官方文档包含大量示例和所有方法的完整参考，请访问[`www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html)

+   有关`Boost.StringAlgorithm`库的更多信息，请参阅本章的*改变大小写和大小写不敏感比较*配方。

# 用两个迭代器表示字符串

有时候我们需要将一些字符串分割成子字符串并对这些子字符串进行操作。例如，计算字符串中的空格数，当然，我们希望使用 Boost 并尽可能高效。

## 准备工作

您需要了解一些基本的 STL 算法知识才能使用此配方。

## 如何做...

我们不会计算空格数；相反，我们将字符串分割成句子。您将看到使用 Boost 做这件事非常简单。

1.  首先，包含正确的头文件：

    ```cpp
    #include <boost/algorithm/string/split.hpp>
    #include <boost/algorithm/string/classification.hpp>
    #include <algorithm>
    ```

1.  现在，让我们定义我们的测试字符串：

    ```cpp
    int main() {
      const char str[] 
        = "This is a long long character array."
          "Please split this character array to sentences!"
          "Do you know, that sentences are separated using period, "
          "exclamation mark and question mark? :-)"
      ; 
    ```

1.  现在我们为我们的分割迭代器创建一个`typedef`：

    ```cpp
    typedef boost::split_iterator<const char*> split_iter_t;
    ```

1.  构建那个迭代器：

    ```cpp
      split_iter_t sentences = boost::make_split_iterator(str, 
        boost::algorithm::token_finder(boost::is_any_of("?!."))
      );  
    ```

1.  现在我们可以遍历匹配项之间：

    ```cpp
      for (unsigned int i = 1; !sentences.eof(); ++sentences, ++i) {
        boost::iterator_range<const char*> range = *sentences;
        std::cout << "Sentence #" << i << " : \t" << range << '\n';
    ```

1.  计算字符数：

    ```cpp
        std::cout << "Sentence has " << range.size() << " characters.\n";
    ```

1.  然后计算空格数：

    ```cpp
        std::cout 
          << "Sentence has " 
          << std::count(range.begin(), range.end(), ' ') 
          << " whitespaces.\n\n";
      } // end of for(...) loop
    } // end of main()
    ```

    就这样。现在如果我们运行这个示例，它将输出：

    ```cpp
    Sentence #1 :   This is a long long character array
    Sentence has 35 characters.
    Sentence has 6 whitespaces.

    Sentence #2 :   Please split this character array to sentences
    Sentence has 46 characters.
    Sentence has 6 whitespaces.

    Sentence #3 :   Do you know, that sentences are separated using dot,
    exclamation mark and question mark
    Sentence has 87 characters.
    Sentence has 13 whitespaces.

    Sentence #4 :    :-)
    Sentence has 4 characters.
    Sentence has 1 whitespaces.
    ```

## 如何工作...

此配方的核心思想是我们不需要从子字符串中构造`std::string`。我们甚至不需要一次性对整个字符串进行分词。我们只需要找到第一个子字符串，并将其作为一对迭代器返回，一对迭代器分别指向子字符串的开始和结束。如果我们需要更多子字符串，找到下一个子字符串，并返回该子字符串的迭代器对。

![如何工作...](img/4880OS_07_02.jpg)

现在，让我们更仔细地看看`boost::split_iterator`。我们使用`boost::make_split_iterator`函数构建了一个迭代器，该函数以`range`作为第一个参数，以二进制查找谓词（或二进制谓词）作为第二个参数。当`split_iterator`被解引用时，它返回第一个子字符串作为`boost::iterator_range<const char*>`，它只包含一对迭代器，并有一些方法可以与之交互。当我们增加`split_iterator`时，它将尝试找到下一个子字符串，如果没有找到子字符串，`split_iterator::eof()`将返回`true`。

## 还有更多...

`boost::iterator_range` 类在所有 Boost 库中都有广泛的应用。你可能会发现在需要返回一对迭代器或函数应该接受/使用一对迭代器的情况下，它对你的代码和库很有用。

`boost::split_iterator<>` 和 `boost::iterator_range<>` 类接受一个前向迭代器类型作为模板参数。因为我们之前的工作是在字符数组上，所以我们提供了 `const char*` 作为迭代器。如果我们使用 `std::wstring`，我们需要使用 `boost::split_iterator<std::wstring::const_iterator>` 和 `boost::iterator_range<std::wstring::const_iterator>` 类型。

C++11 既没有 `iterator_range` 也没有 `split_iterator`。

由于 `boost::iterator_range` 类没有虚拟函数和动态内存分配，它既快又高效。然而，它的输出流操作符 `<<` 对字符数组没有特定的优化，因此流操作较慢。

`boost::split_iterator` 类中有一个 `boost::function` 类，因此构造它可能较慢；然而，迭代只会增加微小的开销，你甚至可能在性能关键部分都感觉不到。

## 参见

+   下一个配方将告诉你关于 `boost::iterator_range<const char*>` 的一个很好的替代方案。

+   `Boost.StringAlgorithm` 的官方文档将为你提供关于类和大量示例的更详细信息，请参阅 [`www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html)。

+   更多关于 `boost::iterator_range` 的信息可以在以下链接找到：[`www.boost.org/doc/libs/1_53_0/libs/range/doc/html/range/reference/utilities.html`](http://www.boost.org/doc/libs/1_53_0/libs/range/doc/html/range/reference/utilities.html)。它是 `Boost.Range` 库的一部分，本书没有描述，但你可能希望自学。

# 使用字符串类型的引用

这个配方是本章最重要的配方！让我们看看一个非常常见的案例，其中我们编写一个函数，该函数接受一个字符串，并返回 `starts` 和 `ends` 参数传递的字符值之间的字符串部分：

```cpp
#include <string>
#include <algorithm>

std::string between_str(const std::string& input, char starts,
   char ends)
{
  std::string::const_iterator pos_beg 
    = std::find(input.begin(), input.end(), starts);

  if (pos_beg == input.end()) {
    return std::string(); // Empty
  }

  ++ pos_beg;
  std::string::const_iterator pos_end 
    = std::find(input.begin(), input.end(), ends);

  return std::string(pos_beg, pos_end);
}
```

你喜欢这个实现吗？在我看来，它看起来很糟糕；考虑以下对其的调用：

```cpp
between_str("Getting expression (between brackets)", '(', ')');
```

在这个例子中，一个临时的 `std::string` 变量将从 `"Getting expression (between brackets)"` 构造出来。字符数组足够长，所以有很大可能在 `std::string` 构造函数内部调用动态内存分配，并将字符数组复制到其中。然后，在 `between_str` 函数的某个地方，将构造一个新的 `std::string`，这也可能导致另一个动态内存分配，并导致复制。

因此，这个简单的函数可能，并且在大多数情况下会：

+   调用动态内存分配（两次）

+   复制字符串（两次）

+   释放内存（两次）

我们能做得更好吗？

## 准备工作

此配方需要基本的 STL 和 C++ 知识。

## 如何实现...

在这里我们实际上并不需要一个 `std::string` 类，我们只需要指向字符数组的指针以及数组的大小。Boost 提供了 `std::string_ref` 类。

1.  要使用 `boost::string_ref` 类，需要包含以下头文件：

    ```cpp
    #include <boost/utility/string_ref.hpp>
    ```

1.  修改方法的签名：

    ```cpp
    boost::string_ref between(
      const boost::string_ref& input, 
      char starts, 
      char ends) 
    ```

1.  在函数体内将 `std::string` 改为 `boost::string_ref`：

    ```cpp
    {
      boost::string_ref::const_iterator pos_beg 
        = std::find(input.cbegin(), input.cend(), starts);
      if (pos_beg == input.cend()) {
        return boost::string_ref(); // Empty
      }
      ++ pos_beg;
      boost::string_ref::const_iterator pos_end 
        = std::find(input.cbegin(), input.cend(), ends);
      // ...
    ```

1.  `boost::string_ref` 构造函数接受大小作为第二个参数，因此我们需要稍微修改代码：

    ```cpp
      if (pos_end == input.cend()) {
        return boost::string_ref(pos_beg, input.end() - pos_beg);
      }
      return boost::string_ref(pos_beg, pos_end - pos_beg);
    }
    ```

    就这样！现在我们可以调用 `between("Getting expression (between brackets)", '(', ')')`，它将无需任何动态内存分配和字符复制即可工作。我们仍然可以使用它来处理 `std::string`：

    ```cpp
    between(std::string("(expression)"), '(', ')')
    ```

## 它是如何工作的...

如前所述，`boost::string_ref` 只包含指向字符数组的指针和数据的大小。它有很多构造函数，并且可以以不同的方式初始化：

```cpp
  boost::string_ref r0("^_^");

  std::string O_O("O__O");
  boost::string_ref r1 = O_O;

  std::vector<char> chars_vec(10, '#');
  boost::string_ref r2(&chars_vec.front(), chars_vec.size());
```

`boost::string_ref` 类拥有容器类所需的所有方法，因此它可以与 STL 算法和 Boost 算法一起使用：

```cpp
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <iterator>

void string_ref_algorithms_examples() {
  boost::string_ref r("O_O");
  // Finding symbol
  std::find(r.cbegin(), r.cend(), '_');

  // Will print 'o_o'
  boost::to_lower_copy(std::ostream_iterator<char>(std::cout), r);
  std::cout << '\n';

  // Will print 'O_O'
  std::cout << r << '\n';

  // Will print '^_^'
  boost::replace_all_copy(
    std::ostream_iterator<char>(std::cout), r, "O", "^"
  );
}
```

### 注意

`boost::string_ref` 类实际上并不拥有字符串，因此它所有的方法都返回常量迭代器。正因为如此，我们不能在修改数据的函数中使用它，例如 `boost::to_lower(r)`。

当使用 `boost::string_ref` 时，我们应该特别注意它所引用的数据；它必须在整个 `boost::string_ref` 的生命周期内存在且有效。

## 还有更多...

`boost::string_ref` 类不是 C++11 的组成部分，但它被提议包含在下一个标准中。

`string_ref` 类快速且高效；在可能的情况下使用它们。

`boost::string_ref` 类实际上是 `boost::` 命名空间中的一个 typedef：

```cpp
typedef basic_string_ref<char, std::char_traits<char> >
   string_ref;
```

你可能还会发现 `boost::` 命名空间中宽字符的以下 typedefs 有用：

```cpp
typedef basic_string_ref<wchar_t,  std::char_traits<wchar_t> > 
   wstring_ref;

typedef basic_string_ref<char16_t, std::char_traits<char16_t> > 
   u16string_ref;

typedef basic_string_ref<char32_t, std::char_traits<char32_t> > 
   u32string_ref;
```

## 参见

+   将 `string_ref` 包含到 C++ 标准中的官方提案可以在 [`www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3442.html`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3442.html) 找到

+   `string_ref` 的 Boost 文档可以在 [`www.boost.org/doc/libs/1_53_0/libs/utility/doc/html/string_ref.html`](http://www.boost.org/doc/libs/1_53_0/libs/utility/doc/html/string_ref.html) 找到
