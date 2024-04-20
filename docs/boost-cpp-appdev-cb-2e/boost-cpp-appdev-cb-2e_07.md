# 第七章：操作字符串

在本章中，我们将涵盖：

+   更改大小写和不区分大小写比较

+   使用正则表达式匹配字符串

+   使用正则表达式搜索和替换字符串

+   使用安全的 printf 样式函数格式化字符串

+   替换和删除字符串

+   用两个迭代器表示一个字符串

+   使用对字符串类型的引用

# 介绍

整个章节都致力于不同方面的更改、搜索和表示字符串。我们将看到如何使用 Boost 库轻松完成一些常见的与字符串相关的任务。这一章很容易；它涉及非常常见的字符串操作任务。所以，让我们开始吧！

# 更改大小写和不区分大小写比较

这是一个非常常见的任务。我们有两个非 Unicode 或 ANSI 字符字符串：

```cpp
#include <string> 
std::string str1 = "Thanks for reading me!"; 
std::string str2 = "Thanks for reading ME!"; 
```

我们需要以不区分大小写的方式进行比较。有很多方法可以做到这一点，让我们看看 Boost 的方法。

# 准备工作

这里我们只需要基本的`std::string`知识。

# 如何做...

以下是进行不区分大小写比较的不同方法：

1.  最简单的方法是：

```cpp
#include <boost/algorithm/string/predicate.hpp> 

const bool solution_1 = (
     boost::iequals(str1, str2)
);
```

1.  使用 Boost 谓词和标准库方法：

```cpp
#include <boost/algorithm/string/compare.hpp> 
#include <algorithm> 

const bool solution_2 = (
    str1.size() == str2.size() && std::equal(
        str1.begin(),
        str1.end(),
        str2.begin(),
        boost::is_iequal()
    )
);
```

1.  制作两个字符串的小写副本：

```cpp
#include <boost/algorithm/string/case_conv.hpp> 

void solution_3() {
    std::string str1_low = boost::to_lower_copy(str1);
    std::string str2_low = boost::to_lower_copy(str2);
    assert(str1_low == str2_low);
}
```

1.  制作原始字符串的大写副本：

```cpp
#include <boost/algorithm/string/case_conv.hpp> 

void solution_4() {
    std::string str1_up = boost::to_upper_copy(str1);
    std::string str2_up = boost::to_upper_copy(str2);
    assert(str1_up == str2_up);
}
```

1.  将原始字符串转换为小写：

```cpp
#include <boost/algorithm/string/case_conv.hpp> 

void solution_5() {
    boost::to_lower(str1);
    boost::to_lower(str2);
    assert(str1 == str2);
}
```

# 它是如何工作的...

第二种方法并不明显。在第二种方法中，我们比较字符串的长度。如果它们长度相同，我们使用`boost::is_iequal`谓词的实例逐个字符比较字符串，该谓词以不区分大小写的方式比较两个字符。

`Boost.StringAlgorithm`库在方法或类的名称中使用`i`，如果该方法是不区分大小写的。例如，`boost::is_iequal`，`boost::iequals`，`boost::is_iless`等。

# 还有更多...

`Boost.StringAlgorithm`库的每个函数和函数对象都接受`std::locale`。默认情况下（在我们的示例中），方法和类使用默认构造的`std::locale`。如果我们大量使用字符串，一次构造`std::locale`变量并将其传递给所有方法可能是一个很好的优化。另一个很好的优化是通过`std::locale::classic()`使用*C*语言环境（如果您的应用逻辑允许）：

```cpp
  // On some platforms std::locale::classic() works 
  // faster than std::locale().
  boost::iequals(str1, str2, std::locale::classic()); 
```

没有人禁止您同时使用这两种优化。

不幸的是，C++17 没有来自`Boost.StringAlgorithm`的字符串函数。所有的算法都快速可靠，所以不要害怕在代码中使用它们。

# 另请参阅

+   Boost String Algorithms 库的官方文档可以在[`boost.org/libs/algorithm/string`](http://boost.org/libs/algorithm/string)找到

+   请参阅 Andrei Alexandrescu 和 Herb Sutter 的*C++编程标准*一书，了解如何使用几行代码制作不区分大小写的字符串的示例

# 使用正则表达式匹配字符串

让我们做一些有用的事情！当用户的输入必须使用一些**正则表达式**进行检查时，这是一个常见情况。问题在于有很多正则表达式语法，使用一种语法编写的表达式在其他语法中处理得不好。另一个问题是，长的正则表达式不那么容易编写。

因此，在这个示例中，我们将编写一个支持不同正则表达式语法并检查输入字符串是否匹配指定正则表达式的程序。

# 入门

这个示例需要基本的标准库知识。了解正则表达式语法可能会有所帮助。

需要将示例链接到`boost_regex`库。

# 如何做...

这个正则表达式匹配器示例由`main()`函数中的几行代码组成：

1.  要实现它，我们需要以下标头：

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
        << "\t[5] POSIX basic case insensitive\n\n" 
        << "Choose regex syntax: "; 
```

1.  现在，根据所选择的语法正确设置标志：

```cpp
    boost::regex::flag_type flag;
    switch (std::cin.get()) 
    {
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
        std::cout << "Incorrect number of regex syntax. Exiting...\n";
        return 1;
    }

    // Disabling exceptions.
    flag |= boost::regex::no_except;
```

1.  我们现在在循环中请求正则表达式模式：

```cpp
    // Restoring std::cin.
    std::cin.ignore();
    std::cin.clear();

    std::string regex, str;
    do {
        std::cout << "Input regex: ";
        if (!std::getline(std::cin, regex) || regex.empty()) {
            return 0;
        }

        // Without `boost::regex::no_except`flag this
        // constructor may throw.
        const boost::regex e(regex, flag);
        if (e.status()) {
            std::cout << "Incorrect regex pattern!\n";
            continue;
        }
```

1.  在循环中获取`要匹配的字符串`：

```cpp
        std::cout << "String to match: ";
        while (std::getline(std::cin, str) && !str.empty()) {
```

1.  对其应用正则表达式并输出结果：

```cpp
            const bool matched = boost::regex_match(str, e);
            std::cout << (matched ? "MATCH\n" : "DOES NOT MATCH\n");
            std::cout << "String to match: ";
        } // end of `while (std::getline(std::cin, str))`
```

1.  我们将通过恢复`std::cin`并请求新的正则表达式模式来完成我们的示例：

```cpp
        // Restoring std::cin.
        std::cin.ignore();
        std::cin.clear();
    } while (1);
} // int main() 
```

现在，如果我们运行前面的示例，我们将得到以下输出：

```cpp
 Available regex syntaxes:
```

```cpp
 [0] Perl
 [1] Perl case insensitive
 [2] POSIX extended
 [3] POSIX extended case insensitive
 [4] POSIX basic
 [5] POSIX basic case insensitive
```

```cpp
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

# 工作原理...

所有的匹配都是由`boost::regex`类完成的。它构造了一个能够进行正则表达式解析和编译的对象。通过`flag`输入变量将额外的配置选项传递给类。

如果正则表达式不正确，`boost::regex`会抛出异常。如果传递了`boost::regex::no_except`标志，它会在`status()`调用中返回非零以报告错误（就像我们的示例中一样）：

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
```

通过调用`boost::regex_match`函数来进行正则表达式匹配。如果匹配成功，它将返回`true`。可以向`regex_match`传递其他标志，但为了简洁起见，我们避免了它们的使用。

# 还有更多...

C++11 几乎包含了所有`Boost.Regex`类和标志。它们可以在`std::`命名空间的`<regex>`头文件中找到（而不是`boost::`）。官方文档提供了关于 C++11 和`Boost.Regex`的差异的信息。它还包含一些性能测量，表明`Boost.Regex`很快。一些标准库存在性能问题，因此在 Boost 和标准库版本之间明智地进行选择。

# 另请参阅

+   *使用正则表达式搜索和替换字符串*示例将为您提供有关`Boost.Regex`用法的更多信息

+   您还可以考虑官方文档，以获取有关标志、性能测量、正则表达式语法和 C++11 兼容性的更多信息，网址为[`boost.org/libs/regex`](http://boost.org/libs/regex)

# 使用正则表达式搜索和替换字符串

我的妻子非常喜欢*通过正则表达式匹配字符串*示例。但是，她想要更多，并告诉我，除非我提升这个配方以便能够根据正则表达式匹配替换输入字符串的部分，否则我将得不到食物。

好的，它来了。每个匹配的子表达式（括号中的正则表达式部分）必须从 1 开始获得一个唯一的编号；这个编号将用于创建一个新的字符串。

这就是更新后的程序应该工作的方式：

```cpp
 Available regex syntaxes:
```

```cpp
 [0] Perl
 [1] Perl case insensitive
 [2] POSIX extended
 [3] POSIX extended case insensitive
 [4] POSIX basic
 [5] POSIX basic case insensitive
```

```cpp

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
```

# 准备工作

我们将重用*通过正则表达式匹配字符串*示例中的代码。建议在阅读本示例之前先阅读它。

需要链接一个示例到`boost_regex`库。

# 如何做到...

这个配方是基于前一个配方的代码。让我们看看必须改变什么：

1.  不需要包含额外的头文件。但是，我们需要一个额外的字符串来存储替换模式：

```cpp
    std::string regex, str, replace_string;
```

1.  我们用`boost::regex_match`替换为`boost::regex_find`并输出匹配的结果：

```cpp
        std::cout << "String to match: ";
        while (std::getline(std::cin, str) && !str.empty()) {
            boost::smatch results;
            const bool matched = regex_search(str, results, e);
            if (matched)  {
                std::cout << "MATCH: ";
                std::copy(
                    results.begin() + 1, 
                    results.end(), 
                    std::ostream_iterator<std::string>(std::cout, ", ")
                );
```

1.  之后，我们需要获取替换模式并应用它：

```cpp
                std::cout << "\nReplace pattern: ";
                if (
                        std::getline(std::cin, replace_string)
                        && !replace_string.empty())
                {
                    std::cout << "RESULT: " << 
                        boost::regex_replace(str, e, replace_string)
                    ; 
                } else {
                    // Restoring std::cin.
                    std::cin.ignore();
                    std::cin.clear();
                }
            } else { // `if (matched) `
                std::cout << "DOES NOT MATCH";
            }
```

就是这样！每个人都很开心，我也吃饱了。

# 工作原理...

`boost::regex_search`函数不仅返回`true`或`false`值（不像`boost::regex_match`函数那样），而且还存储匹配的部分。我们使用以下结构输出匹配的部分：

```cpp
    std::copy( 
        results.begin() + 1,  
        results.end(),  
        std::ostream_iterator<std::string>( std::cout, ", ") 
    ); 
```

请注意，我们通过跳过第一个结果（`results.begin() + 1`）输出了结果，这是因为`results.begin()`包含整个正则表达式匹配。

`boost::regex_replace`函数执行所有替换并返回修改后的字符串。

# 还有更多...

`regex_*`函数有不同的变体，其中一些接收双向迭代器而不是字符串，有些则向迭代器提供输出。

`boost::smatch`是`boost::match_results<std::string::const_iterator>`的`typedef`。如果您使用的是`std::string::const_iterator`之外的其他双向迭代器，您应该将您的双向迭代器的类型作为`boost::match_results`的模板参数。

`match_results`有一个格式函数，因此我们可以使用它来调整我们的示例，而不是：

```cpp
std::cout << "RESULT: " << boost::regex_replace(str, e, replace_string); 
```

我们可以使用以下内容：

```cpp
std::cout << "RESULT: " << results.format(replace_string); 
```

顺便说一下，`replace_string`支持多种格式：

```cpp
Input regex: (\d)(\d)
 String to match: 12
 MATCH: 1, 2,
 Replace pattern: $1-$2---$&---$$
 RESULT: 1-2---12---$
```

此处的所有类和函数都存在于 C++11 的`<regex>`头文件的`std::`命名空间中。

# 另请参阅

`Boost.Regex`的官方文档将为您提供更多关于性能、C++11 标准兼容性和正则表达式语法的示例和信息，网址为[`boost.org/libs/regex`](http://boost.org/libs/regex)。*通过正则表达式匹配字符串*示例将告诉您`Boost.Regex`的基础知识。

# 使用安全的 printf 样式函数格式化字符串

`printf`系列函数对安全性构成威胁。允许用户将自己的字符串作为类型并格式化说明符是非常糟糕的设计。那么当需要用户定义的格式时，我们该怎么办？我们应该如何实现以下类的成员函数`std::string to_string(const std::string& format_specifier) const;`？

```cpp
class i_hold_some_internals 
{
    int i;
    std::string s;
    char c;
    // ...
}; 
```

# 准备工作

对标准库的基本知识就足够了。

# 如何做到...

我们希望允许用户为字符串指定自己的输出格式：

1.  为了以安全的方式进行操作，我们需要以下头文件：

```cpp
#include <boost/format.hpp>
```

1.  现在，我们为用户添加一些注释：

```cpp
    // `fmt` parameter may contain the following:
    // $1$ for outputting integer 'i'.
    // $2$ for outputting string 's'.
    // $3$ for outputting character 'c'.
    std::string to_string(const std::string& fmt) const {
```

1.  是时候让所有部分都运行起来了：

```cpp
        boost::format f(fmt);
        unsigned char flags = boost::io::all_error_bits;
        flags ^= boost::io::too_many_args_bit;
        f.exceptions(flags);
        return (f % i % s % c).str();
    }
```

就是这样。看一下这段代码：

```cpp
int main() {
    i_hold_some_internals class_instance;

    std::cout << class_instance.to_string(
        "Hello, dear %2%! "
        "Did you read the book for %1% %% %3%\n"
    );

    std::cout << class_instance.to_string(
        "%1% == %1% && %1%%% != %1%\n\n"
    );
}
```

假设`class_instance`有一个成员`i`等于`100`，一个成员`s`等于`"Reader"`，一个成员`c`等于`'!'`。然后，程序将输出如下内容：

```cpp
 Hello, dear Reader! Did you read the book for 100 % !
 100 == 100 && 100% != 100
```

# 它是如何工作的...

`boost::format`类接受指定结果字符串格式的字符串。参数通过`operator%`传递给`boost::format`。在指定字符串格式中，`%1%`、`%2%`、`%3%`、`%4%`等值会被传递给`boost::format`的参数替换。

我们还禁用了异常，以防格式字符串包含的参数少于传递给`boost::format`的参数：

```cpp
    boost::format f(format_specifier);
    unsigned char flags = boost::io::all_error_bits;
    flags ^= boost::io::too_many_args_bit;
```

这样做是为了允许一些这样的格式：

```cpp
    // Outputs 'Reader'.
    std::cout << class_instance.to_string("%2%\n\n");
```

# 还有更多...

在格式不正确的情况下会发生什么？

没有什么可怕的，会抛出一个异常：

```cpp
    try {
        class_instance.to_string("%1% %2% %3% %4% %5%\n");
        assert(false);
    } catch (const std::exception& e) {
        // boost::io::too_few_args exception is catched.
        std::cout << e.what() << '\n';
    }
```

前一个代码片段通过控制台输出了以下行：

```cpp
 boost::too_few_args: format-string referred to more arguments than
    were passed
```

C++17 没有`std::format`。`Boost.Format`库不是一个非常快的库。尽量不要在性能关键的部分大量使用它。

# 另请参阅

官方文档包含了有关`Boost.Format`库性能的更多信息。在[`boost.org/libs/format`](http://boost.org/libs/format)上还有更多关于扩展 printf 格式的示例和文档。

[﻿](http://boost.org/libs/format)

# 替换和擦除字符串

我们需要在字符串中擦除某些内容，替换字符串的一部分，或者擦除某些子字符串的第一个或最后一个出现的情况非常常见。标准库允许我们做更多的部分，但通常需要编写太多的代码。

我们在*更改大小写和不区分大小写比较*示例中看到了`Boost.StringAlgorithm`库的实际应用。让我们看看当我们需要修改一些字符串时，它如何简化我们的生活：

```cpp
#include <string> 
const std::string str = "Hello, hello, dear Reader."; 
```

# 准备工作

这个示例需要对 C++有基本的了解。

# 如何做到...

这个示例展示了`Boost.StringAlgorithm`库中不同的字符串擦除和替换方法的工作原理：

1.  擦除需要`#include <boost/algorithm/string/erase.hpp>`头文件：

```cpp
#include <boost/algorithm/string/erase.hpp>

void erasing_examples() {
    namespace ba = boost::algorithm;
    using std::cout;

    cout << "\n erase_all_copy :" << ba::erase_all_copy(str, ",");
    cout << "\n erase_first_copy:" << ba::erase_first_copy(str, ",");
    cout << "\n erase_last_copy :" << ba::erase_last_copy(str, ",");
    cout << "\n ierase_all_copy :" << ba::ierase_all_copy(str, "hello");
    cout << "\n ierase_nth_copy :" << ba::ierase_nth_copy(str, ",", 1);
}
```

这段代码输出如下内容：

```cpp
 erase_all_copy   :Hello hello dear Reader.
 erase_first_copy :Hello hello, dear Reader.
 erase_last_copy  :Hello, hello dear Reader.
 ierase_all_copy   :, , dear Reader.
 ierase_nth_copy  :Hello, hello dear Reader.
```

1.  替换需要`<boost/algorithm/string/replace.hpp>`头文件：

```cpp
#include <boost/algorithm/string/replace.hpp>

void replacing_examples() {
    namespace ba = boost::algorithm;
    using std::cout;

    cout << "\n replace_all_copy :" 
        << ba::replace_all_copy(str, ",", "!");

    cout << "\n replace_first_copy :"
        << ba::replace_first_copy(str, ",", "!");

    cout << "\n replace_head_copy :"
        << ba::replace_head_copy(str, 6, "Whaaaaaaa!");
}
```

这段代码输出如下内容：

```cpp
 replace_all_copy :Hello! hello! dear Reader.
 replace_first_copy :Hello! hello, dear Reader.
 replace_head_copy :Whaaaaaaa! hello, dear Reader.
```

# 它是如何工作的...

所有示例都是自解释的。唯一不明显的是`replace_head_copy`函数。它接受要替换的字节数作为第二个参数，替换字符串作为第三个参数。因此，在前面的示例中，`Hello`被替换为`Whaaaaaaa!`。

# 还有更多...

还有一些可以就地修改字符串的方法。它们不以`_copy`结尾，返回`void`。所有不区分大小写的方法（以`i`开头的方法）都接受`std::locale`作为最后一个参数，并使用默认构造的 locale 作为默认参数。

您经常使用不区分大小写的方法并且需要更好的性能吗？只需创建一个持有`std::locale::classic()`的`std::locale`变量，并将其传递给所有算法。在小字符串上，大部分时间都被`std::locale`构造所消耗，而不是算法：

```cpp
#include <boost/algorithm/string/erase.hpp>

void erasing_examples_locale() {
    namespace ba = boost::algorithm;

    const std::locale loc = std::locale::classic();

    const std::string r1
        = ba::ierase_all_copy(str, "hello", loc);

    const std::string r2
        = ba::ierase_nth_copy(str, ",", 1, loc);

    // ...
}
```

C++17 没有`Boost.StringAlgorithm`方法和类。然而，它有一个`std::string_view`类，可以在没有内存分配的情况下使用子字符串。您可以在本章的下两个配方中找到更多关于类似`std::string_view`的类的信息。

# 另请参阅

+   官方文档包含大量示例和所有方法的完整参考[`boost.org/libs/algorithm/string`](http://boost.org/libs/algorithm/string)

+   有关`Boost.StringAlgorithm`库的更多信息，请参见本章的*更改大小写和不区分大小写比较*配方

# 用两个迭代器表示一个字符串

有时我们需要将一些字符串拆分成子字符串并对这些子字符串进行操作。在这个配方中，我们想将字符串拆分成句子，计算字符和空格，当然，我们想使用 Boost 并尽可能高效。

# 准备工作

对于这个配方，您需要一些标准库算法的基本知识。

# 如何做...

使用 Boost 非常容易：

1.  首先，包括正确的头文件：

```cpp
#include <iostream>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <algorithm>
```

1.  现在，让我们定义我们的测试字符串：

```cpp
int main() {
    const char str[] =
        "This is a long long character array."
        "Please split this character array to sentences!"
        "Do you know, that sentences are separated using period, "
        "exclamation mark and question mark? :-)"
    ;
```

1.  我们为我们的分割迭代器制作了一个`typedef`：

```cpp
    typedef boost::split_iterator<const char*> split_iter_t;
```

1.  构造该迭代器：

```cpp
    split_iter_t sentences = boost::make_split_iterator(str,
        boost::algorithm::token_finder(boost::is_any_of("?!."))
    );
```

1.  现在，我们可以在匹配之间进行迭代：

```cpp
    for (unsigned int i = 1; !sentences.eof(); ++sentences, ++i) {
        boost::iterator_range<const char*> range = *sentences;
        std::cout << "Sentence #" << i << " : \t" << range << '\n';
```

1.  计算字符的数量：

```cpp
        std::cout << range.size() << " characters.\n";
```

1.  并计算空格：

```cpp
        std::cout 
            << "Sentence has " 
            << std::count(range.begin(), range.end(), ' ') 
            << " whitespaces.\n\n";
    } // end of for(...) loop
} // end of main()
```

就是这样。现在，如果我们运行一个示例，它将输出：

```cpp
 Sentence #1 : This is a long long character array
 35 characters.
 Sentence has 6 whitespaces.

 Sentence #2 : Please split this character array to sentences
 46 characters.
 Sentence has 6 whitespaces.

 Sentence #3 : Do you know, that sentences are separated using dot,
 exclamation mark and question mark
 90 characters.
 Sentence has 13 whitespaces.

 Sentence #4 : :-)
 4 characters.
 Sentence has 1 whitespaces.
```

# 它是如何工作的...

这个配方的主要思想是我们不需要从子字符串构造`std::string`。我们甚至不需要一次性对整个字符串进行标记。我们所需要做的就是找到第一个子字符串，并将其作为一对迭代器返回到子字符串的开头和结尾。如果我们需要更多的子字符串，找到下一个子字符串并返回该子字符串的一对迭代器。

![](img/00015.jpeg)

现在，让我们更仔细地看看`boost::split_iterator`。我们使用`boost::make_split_iterator`函数构造了一个，它将`range`作为第一个参数，二进制查找谓词（或二进制谓词）作为第二个参数。当解引用`split_iterator`时，它将第一个子字符串作为`boost::iterator_range<const char*>`返回，它只是保存一对指针并有一些方法来处理它们。当我们递增`split_iterator`时，它会尝试找到下一个子字符串，如果没有找到子字符串，`split_iterator::eof()`将返回`true`。

默认构造的分割迭代器表示`eof()`。因此，我们可以将循环条件从`!sentences.eof()`重写为`sentences != split_iter_t()`。您还可以使用分割迭代器与算法，例如：`std::for_each(sentences, split_iter_t(), [](auto range){ /**/ });`。

# 还有更多...

`boost::iterator_range`类广泛用于所有 Boost 库。即使在您自己的代码中，当需要返回一对迭代器或者函数需要接受/处理一对迭代器时，您可能会发现它很有用。

`boost::split_iterator<>`和`boost::iterator_range<>`类接受前向迭代器类型作为模板参数。因为在前面的示例中我们使用字符数组，所以我们提供了`const char*`作为迭代器。如果我们使用`std::wstring`，我们需要使用`boost::split_iterator<std::wstring::const_iterator>`和`boost::iterator_range<std::wstring::const_iterator>`类型。

C++17 中既没有`iterator_range`也没有`split_iterator`。然而，正在讨论接受类似`iterator_range`的类，可能会有名为`std::span`的名称。

`boost::iterator_range`类没有虚函数和动态内存分配，它非常快速和高效。然而，它的输出流操作符`<<`对字符数组没有特定的优化，因此流操作可能会很慢。

`boost::split_iterator`类中有一个`boost::function`类，因此为大型函数构造它可能会很慢。迭代只会增加微小的开销，即使在性能关键的部分，你也不会感觉到。

# 另请参阅

+   下一个示例将告诉您`boost::iterator_range<const char*>`的一个很好的替代品

+   `Boost.StringAlgorithm`的官方文档可能会为您提供有关类的更详细信息以及大量示例的信息，网址为[`boost.org/libs/algorithm/string`](http://boost.org/libs/algorithm/string)

+   关于`boost::iterator_range`的更多信息可以在这里找到：[`boost.org/libs/range`](http://boost.org/libs/range)；它是`Boost.Range`库的一部分，本书中没有描述，但您可能希望自行研究它

# 使用对字符串类型的引用

这个示例是本章中最重要的示例！让我们看一个非常常见的情况，我们编写一些接受字符串并返回在`starts`和`ends`参数中传递的字符值之间的字符串部分的函数：

```cpp
#include <string>
#include <algorithm>

std::string between_str(const std::string& input, char starts, char ends) {
    std::string::const_iterator pos_beg 
        = std::find(input.begin(), input.end(), starts);
    if (pos_beg == input.end()) {
        return std::string();
    }
    ++ pos_beg;

    std::string::const_iterator pos_end 
        = std::find(pos_beg, input.end(), ends);

    return std::string(pos_beg, pos_end);
}
```

你喜欢这个实现吗？在我看来，这个实现很糟糕。考虑对它的以下调用：

```cpp
between_str("Getting expression (between brackets)", '(', ')'); 
```

在这个示例中，从`"Getting expression (between brackets)"`构造了一个临时的`std::string`变量。字符数组足够长，因此在`std::string`构造函数内可能会调用动态内存分配，并将字符数组复制到其中。然后，在`between_str`函数的某个地方，将构造新的`std::string`，这可能还会导致另一个动态内存分配和复制。

因此，这个简单的函数可能会，并且在大多数情况下会：

+   调用动态内存分配（两次）

+   复制字符串（两次）

+   释放内存（两次）

我们能做得更好吗？

# 准备工作

这个示例需要对标准库和 C++有基本的了解。

# 如何做...

在这里我们实际上并不需要`std::string`类，我们只需要一些轻量级的类，它不管理资源，只有一个指向字符数组和数组大小的指针。Boost 有`boost::string_view`类可以满足这个需求。

1.  要使用`boost::string_view`类，请包含以下头文件：

```cpp
#include <boost/utility/string_view.hpp>
```

1.  更改方法的签名：

```cpp
boost::string_view between(
    boost::string_view input,
    char starts,
    char ends)
```

1.  在函数体内的任何地方将`std::string`更改为`boost::string_view`：

```cpp
{
    boost::string_view::const_iterator pos_beg 
        = std::find(input.cbegin(), input.cend(), starts);
    if (pos_beg == input.cend()) {
        return boost::string_view();
    }
    ++ pos_beg;

    boost::string_view::const_iterator pos_end 
        = std::find(pos_beg, input.cend(), ends);
    // ...
```

1.  `boost::string_view`构造函数接受大小作为第二个参数，因此我们需要稍微更改代码：

```cpp
    if (pos_end == input.cend()) {
        return boost::string_view(pos_beg, input.end() - pos_beg);
    }

    return boost::string_view(pos_beg, pos_end - pos_beg);
}
```

就是这样！现在我们可以调用`between("Getting expression (between brackets)", '(', ')')`，而且它将在没有任何动态内存分配和字符复制的情况下工作。而且我们仍然可以将其用于`std::string`：

```cpp
   between(std::string("(expression)"), '(', ')')
```

# 工作原理...

如前所述，`boost::string_view`只包含一个指向字符数组的指针和数据大小。它有很多构造函数，可以以不同的方式初始化：

```cpp
    boost::string_view r0("^_^");

    std::string O_O("O__O");
    boost::string_view r1 = O_O;

    std::vector<char> chars_vec(10, '#');
    boost::string_view r2(&chars_vec.front(), chars_vec.size());
```

`boost::string_view`类具有`container`类所需的所有方法，因此可以与标准库算法和 Boost 算法一起使用：

```cpp
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <iterator>
#include <iostream>

void string_view_algorithms_examples() {
    boost::string_view r("O_O");
    // Finding single symbol.
    std::find(r.cbegin(), r.cend(), '_');

    // Will print 'o_o'.
    boost::to_lower_copy(std::ostream_iterator<char>(std::cout), r);
    std::cout << '\n';

    // Will print 'O_O'.
    std::cout << r << '\n';

    // Will print '^_^'.
    boost::replace_all_copy(
        std::ostream_iterator<char>(std::cout), r, "O", "^"
    );
    std::cout << '\n';

    r = "100";
    assert(boost::lexical_cast<int>(r) == 100);
}
```

`boost::string_view`类实际上并不拥有字符串，因此它的所有方法都返回常量迭代器。因此，我们不能在修改数据的方法中使用它，比如`boost::to_lower(r)`。

在使用`boost::string_view`时，我们必须额外注意它所引用的数据；它必须存在并且在整个`boost::string_view`变量的生命周期内都有效。

在 Boost 1.61 之前，没有`boost::string_view`类，而是使用`boost::string_ref`类。这些类非常接近。`boost::string_view`更接近 C++17 的设计，并且具有更好的 constexpr 支持。自 Boost 1.61 以来，`boost::string_ref`已被弃用。

`string_view`类是快速和高效的，因为它们从不分配内存，也没有虚函数！在任何可能的地方使用它们。它们被设计为`const std::string&`和`const char*`参数的即插即用替代品。这意味着你可以替换以下三个函数：

```cpp
void foo(const std::string& s);
void foo(const char* s);
void foo(const char* s, std::size_t s_size);
```

用一个单一的：

```cpp
void foo(boost::string_view s);
```

# 还有更多...

`boost::string_view`类是一个 C++17 类。如果您的编译器兼容 C++17，可以在`std::`命名空间的`<string_view>`头文件中找到它。

Boost 和标准库的版本支持对`string_view`的 constexpr 使用；然而，`std::string_view`目前具有更多的标记为 constexpr 的函数。

请注意，我们已经通过值接受了`string_view`变量，而不是常量引用。这是传递`boost::string_view`和`std::string_view`的推荐方式，因为：

+   `string_view`是一个具有平凡类型的小类。通过值传递它通常会导致更好的性能，因为减少了间接引用，并且允许编译器进行更多的优化。

+   在其他情况下，当没有性能差异时，编写`string_view val`比编写`const string_view& val`更短。

就像 C++17 的`std::string_view`一样，`boost::string_view`类实际上是一个`typedef`：

```cpp
typedef basic_string_view<char, std::char_traits<char> > string_view; 
```

您还可以在`boost::`和`std::`命名空间中找到宽字符的以下 typedef：

```cpp
typedef basic_string_view<wchar_t,  std::char_traits<wchar_t> > wstring_view; 

typedef basic_string_view<char16_t, std::char_traits<char16_t> > u16string_view; 

typedef basic_string_view<char32_t, std::char_traits<char32_t> > u32string_view; 
```

# 另请参阅

`string_ref`和`string_view`的 Boost 文档可以在[`boost.org/libs/utility`](http://boost.org/libs/utility)找到。
