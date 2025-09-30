# 正则表达式

在上一章中，我们学习了 C++ 中的格式化输入和输出。我们了解到，只要确保你处于 `C` 位置，格式化输出就有很好的解决方案，但尽管有众多输入解析方法，即使是解析字符串中的 `int` 这样的简单任务也可能相当困难。（回想一下，在两种最保险的方法中，`std::stoi(x)` 需要将 `x` 转换为堆分配的 `std::string`，而冗长的 `std::from_chars(x.begin(), x.end(), &value, 10)` 在 C++17 的供应商采用方面落后于其他部分。）解析数字中最棘手的部分是确定如何处理输入中 *不是* 数字的部分！

如果可以将解析任务分解为两个子任务，解析会变得更容易：首先，确定输入中对应于一个“输入项”的确切字节数（这被称为 *词法分析*）；其次，解析该项的值，如果该项的值超出范围或无意义，则进行一些错误恢复。如果我们将这种方法应用于整数输入，*词法分析* 对应于找到输入中最长的初始数字序列，而 *解析* 对应于计算该序列的十进制数值。

*正则表达式*（或 *regexes*）是许多编程语言提供的一种工具，用于解决词法分析问题，不仅适用于数字序列，还适用于任意复杂的输入格式。自 2011 年以来，正则表达式一直是 C++ 标准库的一部分，位于 `<regex>` 头文件中。在本章中，我们将向您展示如何使用正则表达式简化一些常见的解析任务。

请记住，正则表达式对于你日常工作中遇到的 *大多数* 解析任务来说可能是过度杀鸡用牛刀。它们可能很慢、体积庞大，并且不可避免地需要堆分配（即，正则表达式数据类型不是如 第八章 中描述的 *分配器感知*）。正则表达式真正发光的地方是对于即使手写的解析代码也会很慢的复杂任务；以及对于极其简单的任务，正则表达式的可读性和健壮性超过了它们的性能成本。简而言之，正则表达式支持使 C++ 向日常可使用脚本语言（如 Python 和 Perl）迈进了一步。

在本章中，我们将学习：

+   “修改后的 ECMAScript”，C++ 正则表达式使用的方言

+   如何使用正则表达式匹配、搜索甚至替换子串

+   悬挂迭代器的进一步危险

+   避免的正则表达式功能

# 正则表达式是什么？

正则表达式是一种记录识别字符串字节或字符是否属于（或不属于）某种“语言”规则的方法。在这个语境中，“语言”可以是“所有数字序列的集合”到“所有有效 C++标记序列的集合”的任何东西。本质上，“语言”只是将所有字符串的世界划分为两个集合——匹配语言规则的字符串集合，以及不匹配的字符串集合。

一些类型的语言遵循足够简单的规则，以至于可以通过一个“有限状态机”来识别，这是一个完全没有记忆的计算机程序——只是一个程序计数器和扫描输入的单个指针。数字序列的语言当然属于可以通过有限状态机识别的语言类别。我们称这些语言为“正则语言”。

还存在非正则语言。一个非常常见的非正则语言是“有效的算术表达式”，或者简化其本质，就是“正确匹配的括号”。任何能够区分正确匹配的字符串`(((())))`与不正确匹配的字符串`(((()))`和`(((()))))`的程序，本质上必须能够“计数”——区分四个括号的情况与三个或五个括号的情况。这种计数方式不能没有可修改的变量或下推栈；因此，括号匹配不是正则语言。

结果表明，对于任何正则语言，都存在一种简单直接的方法来编写识别它的有限状态机的表示，这当然也是语言规则的表示。我们称这种表示为“正则表达式”，或“regex”。正则表达式的标准符号是在 20 世纪 50 年代开发的，并在 20 世纪 70 年代末的 Unix 程序（如`grep`和`sed`）中得到确立——这些程序至今仍非常值得学习，但当然超出了本书的范围。

C++标准库提供了几种不同的正则表达式语法“风味”，但默认风味（以及你应该始终使用的风味）是从 ECMAScript 标准（更广为人知的 JavaScript 语言）全面借鉴的，只是在方括号结构附近进行了少量修改。我在本章末尾包含了一个关于 ECMAScript 正则表达式语法的入门介绍；但如果你曾经使用过`grep`，你将能够轻松地跟随本章的其余部分，而无需查阅该部分。

# 关于反斜杠转义说明

在本章中，我们将频繁地提到包含字面反斜杠的字符串和正则表达式。正如你所知，要在 C++ 中写入包含字面反斜杠的字符串，你必须用另一个反斜杠来 *转义* 反斜杠：因此 `"\n"` 表示一个换行符，但 `"\\n"` 表示由“反斜杠”和“n”组成的两个字符字符串。这类事情通常很容易跟踪，但在这个章节中，我们不得不特别小心。正则表达式完全作为库特性实现；所以当你写 `std::regex("\n")` 时，正则表达式库会看到一个只包含单个空白字符的“正则表达式”，如果你写 `std::regex("\\n")`，库会看到一个以反斜杠开头的两个字符字符串，库会 *解释* 它为一个表示“换行”的两个字符转义序列。如果你想将 *字面* 反斜杠-n 的概念传达给正则表达式库，你必须让正则表达式库看到三个字符字符串 `\\\\n`，这意味着在 C++ 源代码中写入五个字符字符串 `"\\\\n"`。

你可能在前一段落注意到了我将在本章中使用的解决方案。当我提到一个 *C++ 字符串字面量* 或字符串值时，我会用双引号将其括起来，就像这样："cat"，"a\.b"。当我提到一个 *正则表达式*，就像你在电子邮件或文本编辑器中输入的那样，或者将其传递给库进行评估时，我将不使用引号来表示：`cat`，`a\.b`。只需记住，当你看到未加引号的字符串时，那是一个字符序列的字面表示，如果你想要将其放入 C++ 字符串字面量中，你需要将所有的反斜杠都加倍，因此：`a\.b` 将在源代码中以 `std::regex("a\\.b")` 的形式出现。

我听到一些人在问：那么 *原始字符串字面量* 呢？原始字符串字面量是 C++11 中的一个特性，它允许你通过使用 `R` 和一些括号来“转义”整个字符串来写出 `a\.b` 这样的字符序列，就像这样--`R"(a\.b)"`--而不是转义字符串中的每个反斜杠。如果你的字符串本身包含括号，那么你可以在第一个括号之前和最后一个括号之后写任何任意字符串，就像这样：`R"fancy(a\.b)fancy"`。这样的原始字符串字面量可以包含任何字符--反斜杠、引号，甚至是换行符--只要它不包含连续的序列 `)fancy"`（如果你认为它可能包含这个序列，那么你只需选择一个新的任意字符串，例如 `)supercalifragilisticexpialidocious"`）。

C++原始字符串字面量的语法，其前缀为`R`，让人联想到 Python 中原始字符串字面量的语法（其前缀为`r`）。在 Python 中，`r"a\.b"`同样表示字面量字符串`a\.b`；在代码中，用如`r"abc"`这样的字符串表示正则表达式是既常见又符合习惯的，即使它们不包含任何特殊字符。但请注意`r"a\.b"`和`R"(a\.b)"`之间至关重要的区别——C++版本有一个额外的括号组！并且括号是正则表达式语法中的*重要特殊字符*。C++字符串字面量`"(cat)"`和`R"(cat)"`与白天和黑夜一样不同——前者表示五个字符的正则表达式`(cat)`，而后者表示三个字符的字符串`cat`。如果你不小心写了`R"(cat)"`而本意是`"(cat)"`（或者等价地，`R"((cat))"`），你的程序将会有一个非常微妙的错误。甚至更糟糕的是，`R"a*(b*)a*"`是一个具有惊人含义的有效正则表达式！因此，我建议你在使用原始字符串字面量表示正则表达式时要非常小心；通常，双重*所有*反斜杠比只担心双重*最外层*的括号更安全、更清晰。

原始字符串字面量*适用于*其他语言所说的“heredocs”：

```cpp
    void print_help() {
      puts(R"(The regex special characters are:
      \ - escaping
      | - separating alternatives
      . - match any character
      [] - character class or set
      () - capturing parentheses, or lookahead
      ?*+ - "zero or one", "zero or more", "one or more"
      {} - "exactly N" or "M to N" repetitions
      ^$ - beginning and end of a "line"
      \b - word boundary
      \d \s \w - digit, space, and word
      (?=foo) (?!foo) - lookahead; negative lookahead
    )");
```

也就是说，原始字符串字面量是 C++中唯一可以不进行任何转义就编码换行符的字符串字面量。这对于向用户打印长消息或可能用于 HTTP 头等用途非常有用；但是原始字符串与括号的行为使得它们在使用正则表达式时稍微有些危险——我不会在这本书中使用它们。

# 将正则表达式实体化为`std::regex`对象

要在 C++中使用正则表达式，你不能直接使用如`"c[a-z]*t"`这样的字符串。相反，你必须使用这个字符串来构建一个*正则表达式对象*，类型为`std::regex`，然后将`regex`对象作为参数之一传递给*匹配函数*，例如`std::regex_match`、`std::regex_search`或`std::regex_replace`。每个`std::regex`类型的对象都编码了给定表达式的完整有限状态机，构建这个有限状态机需要大量的计算和内存分配；因此，如果我们需要将大量的输入文本与相同的正则表达式进行匹配，那么库提供一种只需支付一次这种昂贵构建的方法是非常方便的。另一方面，这也意味着`std::regex`对象构建相对较慢，复制成本较高；在紧缩的内循环中构建正则表达式是降低程序性能的好方法：

```cpp
    std::regex rx("(left|right) ([0-9]+)");
    // Construct the regex object "rx" outside the loop.
    std::string line;
    while (std::getline(std::cin, line)) {
      // Inside the loop, use the same "rx" over and over.
      if (std::regex_match(line, rx)) {
        process_command(line);
      } else {
        puts("Unrecognized command.");
      }
    }
```

请记住，这个 `regex` 对象具有值语义；当我们“匹配”一个输入字符串与正则表达式时，我们并没有修改 `regex` 对象本身。正则表达式没有记忆它匹配过什么。因此，当我们想要从正则表达式匹配操作中提取信息——例如，“命令是否说要向左或向右移动？我们看到了什么数字？”——我们将不得不引入一个新的实体，我们可以对其进行修改。

`regex` 对象提供了以下方法：

`std::regex(str, flags)` 通过将给定的 `str` 转换（或“编译”）成有限状态机来构建一个新的 `std::regex` 对象。可以通过位掩码参数 `flags` 指定影响编译过程本身的选项：

+   `std::regex::icase`：将所有字母字符视为不区分大小写

+   `std::regex::nosubs`：将所有括号组视为非捕获组

+   `std::regex::multiline`：使非消耗性断言 `^`（和 `$`）在输入中的 `"\n"` 字符之后（和之前）立即匹配，而不是仅在输入的开始（和结束）处匹配

你可以将其他几个选项按位或到标志中；但其他选项要么将正则表达式语法“风味”从 ECMAScript 转向文档较少且测试较少的风味（`basic`、`extended`、`awk`、`grep`、`egrep`），引入区域设置依赖性（`collate`），或者根本不执行任何操作（`optimize`）。因此，你应该在生产代码中避免使用所有这些选项。

注意，尽管将字符串转换为 `regex` 对象的过程通常被称为“编译正则表达式”，但它仍然是一个动态过程，在调用 `regex` 构造函数时发生，而不是在编译你的 C++ 程序期间。如果你在正则表达式中犯了语法错误，它将在运行时被捕获，而不是在编译时——`regex` 构造函数将抛出一个类型为 `std::regex_error` 的异常，它是 `std::runtime_error` 的子类。健壮的代码还应该准备好 `regex` 构造函数抛出 `std::bad_alloc`；回想一下，`std::regex` 不是分配器感知的。

`rx.mark_count()` 返回正则表达式中的括号捕获组的数量。这个方法的名字来源于短语“标记子表达式”，这是“捕获组”的一个较老的别名。

`rx.flags()` 返回最初传递给构造函数的位掩码。

# 匹配和搜索

要询问给定的输入字符串 `haystack` 是否符合给定的正则表达式 `rneedle`，你可以使用 `std::regex_match(haystack, rneedle)`。正则表达式始终放在最后，这与 JavaScript 的语法 `haystack.match(rneedle)` 和 Perl 的 `haystack =~ rneedle` 相似，尽管它与 Python 的 `re.match(rneedle, haystack)` 相反。如果正则表达式匹配整个输入字符串，则 `regex_match` 函数返回 `true`，否则返回 `false`：

```cpp
    std::regex rx("(left|right) ([0-9]+)");
    std::string line;
    while (std::getline(std::cin, line)) {
      if (std::regex_match(line, rx)) {
        process_command(line);
      } else {
        printf("Unrecognized command '%s'.\n",
          line.c_str());
      }
    }
```

`regex_search` 函数在正则表达式与输入字符串的任何部分匹配时返回 `true`。本质上，它只是在提供的正则表达式两边加上 `.*`，然后运行 `regex_match` 算法；但实现通常可以比重新编译整个新的正则表达式更快地执行 `regex_search`。

要在字符缓冲区的一部分（例如，当你从网络连接或文件中批量拉取数据时）进行匹配，你可以将迭代器对传递给 `regex_match` 或 `regex_search`，这与我们在 第三章 中看到的非常相似，*The Iterator-Pair Algorithms*。在下面的例子中，范围 `[p, end)` 之外的字节永远不会被考虑，并且 "string" `p` 不需要以空字符终止：

```cpp
    void parse(const char *p, const char *end)
    {
      static std::regex rx("(left|right) ([0-9]+)");
      if (std::regex_match(p, end, rx)) {
        process_command(p, end);
      } else {
        printf("Unrecognized command '%.*s'.\n",
          int(end - p), p);
      }
    }
```

此接口与我们之前在 第九章 中看到的 `std::from_chars` 类似，*Iostreams*。

# 从匹配中提取子匹配

要使用正则表达式进行输入的 *lexing* 阶段，你需要一种方法来提取匹配每个捕获组的输入子字符串。在 C++ 中，你通过创建一个类型为 `std::smatch` 的 *match 对象* 来这样做。不，这不是一个打字错误！match 对象类型的名称确实是 `smatch`，代表 `std::string` match；还有一个 `cmatch` 用于 `const char *` 匹配。`smatch` 或 `cmatch` 之间的区别是它们内部存储的 *迭代器类型*：`smatch` 存储 `string::const_iterator`，而 `cmatch` 存储 `const char *`。

在构建了一个空的 `std::smatch` 对象后，你将通过引用将其作为 `regex_match` 或 `regex_search` 的中间参数传递。这些函数将 "填充" `smatch` 对象，包含有关匹配的子字符串的信息，*如果* 正则表达式匹配实际上成功了。如果匹配失败，那么 `smatch` 对象将变为（或保持）空。

这里是一个使用 `std::smatch` 从我们的 "robot command" 中提取匹配方向和整数距离的子字符串的例子：

```cpp
    std::pair<std::string, std::string>
    parse_command(const std::string& line)
    {
      static std::regex rx("(left|right) ([0-9]+)");
      std::smatch m;
      if (std::regex_match(line, m, rx)) {
        return { m[1], m[2] };
      } else {
        throw "Unrecognized command!";
      }
    }

    void test() {
      auto [dir, dist] = parse_command("right 4");
      assert(dir == "right" && dist == "4");
    }
```

注意，我们使用一个 `static` 正则表达式对象来避免每次函数进入时都构造（"编译"）一个新的正则表达式对象。以下代码使用 `const char *` 和 `std::cmatch` 仅用于比较：

```cpp
    std::pair<std::string, std::string>
    parse_command(const char *p, const char *end)
    {
      static std::regex rx("(left|right) ([0-9]+)");
      std::cmatch m;
      if (std::regex_match(p, end, m, rx)) {
        return { m[1], m[2] };
      } else {
        throw "Unrecognized command!";
      }
    }

    void test() {
      char buf[] = "left 20";
      auto [dir, dist] = parse_command(buf, buf + 7);
      assert(dir == "left" && dist == "20");
    }
```

在这两种情况下，在带有 `return` 的行上都会发生一些有趣的事情。在成功将输入字符串与我们的正则表达式匹配后，我们可以查询匹配对象 `m` 来找出输入字符串中哪些部分对应于正则表达式中的各个捕获组。在我们的例子中，第一个捕获组 (`(left|right)`) 对应于 `m[1]`，第二个组 (`([0-9]+)`) 对应于 `m[2]`，依此类推。如果你尝试引用正则表达式中不存在的组，例如我们的例子中的 `m[3]`，你将得到一个空字符串；访问匹配对象永远不会抛出异常。

组 `m[0]` 是一个特殊情况：它指的是整个匹配序列。如果匹配是由 `std::regex_match` 填充的，这将始终是整个输入字符串；如果匹配是由 `std::regex_search` 填充的，那么这将只是与正则表达式匹配的字符串部分。

此外，还有两个命名组：`m.prefix()` 和 `m.suffix()`。这些指的是不是匹配部分的序列——分别在匹配子串之前和之后。如果匹配成功，则 `m.prefix() + m[0] + m.suffix()` 表示整个输入字符串。

所有这些“组”对象都不是由 `std::string` 对象表示的——那会太昂贵了——而是由轻量级的 `std::sub_match<It>` 类型对象表示（其中 `It` 是 `std::string::const_iterator` 或 `const char *`，如前所述）。每个 `sub_match` 对象都可以隐式转换为 `std::string`，并且其行为在很大程度上类似于 `std::string_view`：你可以比较子匹配与字符串字面量，询问它们的长度，甚至可以使用 `operator<<` 将它们输出到 C++ 流中，而无需将它们转换为 `std::string`。这种轻量级效率的缺点是，每次我们处理指向可能不属于我们的容器的迭代器时都会遇到的同样缺点：我们面临 *悬垂迭代器* 的风险：

```cpp
    static std::regex rx("(left|right) ([0-9]+)");
    std::string line = "left 20";
    std::smatch m;
    std::regex_match(line, m, rx);
      // m[1] now holds iterators into line
    line = "hello world";
      // reallocate line's underlying buffer
    std::string oops = m[1];
      // this invokes undefined behavior because
      // of iterator invalidation
const char * to std::string) might cause iterator-invalidation bugs in harmless-looking code. Consider the following:
```

```cpp
    static std::regex rx("(left|right) ([0-9]+)");
    std::smatch m;
    std::regex_match("left 20", m, rx);
      // m[1] would hold iterators into a temporary
      // string, so they would ALREADY be invalid.
      // Fortunately this overload is deleted.
```

幸运的是，标准库预见到这种潜伏的恐怖，并通过提供特殊案例重载 `regex_match(std::string&&, std::smatch&, const std::regex&)` 来避免它，该重载是 *显式删除的*（使用与删除不想要的特殊成员函数相同的 `=delete` 语法）。这确保了前面的看似无辜的代码将无法编译，而不是成为迭代器无效化错误的来源。尽管如此，迭代器无效化错误仍然可能发生，就像前面的例子中那样；为了防止这些错误，你应该将 `smatch` 对象视为极其临时的，有点像捕获整个世界的 `[&]` lambda。一旦 `smatch` 对象被填充，在提取你关心的 `smatch` 部分之前，不要触摸环境中的任何其他内容！

总结来说，`smatch` 或 `cmatch` 对象提供了以下方法：

+   `m.ready()`: 如果 `m` 自构造以来已被填充，则为真。

+   `m.empty()`: 如果 `m` 代表一个失败的匹配（即，如果它是最近由失败的 `regex_match` 或 `regex_search` 填充的），则为真；如果 `m` 代表一个成功的匹配，则为假。

+   `m.prefix()`、`m[0]`、`m.suffix()`：代表输入字符串中未匹配的前缀、匹配和未匹配后缀部分的 `sub_match` 对象。（如果 `m` 代表一个失败的匹配，那么这些都没有意义。）

+   `m[k]`: 代表输入字符串中由第 *k* 个捕获组匹配的部分的 `sub_match` 对象。`m.str(k)` 是 `m[k].str()` 的便捷简写。

+   `m.size()`: 如果 `m` 表示一个失败的匹配，则为零；否则，比表示 `m` 的正则表达式中捕获组的数量多一个。请注意，`m.size()` 总是与 `operator[]` 一致；有意义的子匹配对象的范围始终是 `m[0]` 到 `m[m.size()-1]`。

+   `m.begin()`、`m.end()`：使能够对匹配对象进行范围 for 循环语法的迭代器。

一个 `sub_match` 对象提供了以下方法：

+   `sm.first`: 匹配输入子字符串开头的迭代器。

+   `sm.second`: 匹配输入子字符串末尾的迭代器。

+   `sm.matched`: 如果 `sm` 参与了成功的匹配，则为真；如果 `sm` 是一个可选分支的一部分，该分支被绕过，则为假。例如，如果正则表达式是 `(a)|(b)` 并且输入是 `"a"`，则会有 `m[1].matched && !m[2].matched`；而如果输入是 `"b"`，则会有 `m[2].matched && !m[1].matched`。

+   `sm.str()`: 匹配的输入子字符串，提取并转换为 `std::string`。

+   `sm.length()`: 匹配输入子字符串的长度（`second - first`）。相当于 `sm.str().length()`，但速度更快。

+   `sm == "foo"`: 与 `std::string`、`const char *` 或单个 `char` 进行比较。相当于 `sm.str() == "foo"`，但速度更快。不幸的是，C++17 标准库没有提供任何重载的 `operator==` 操作符，用于接受 `std::string_view`。

虽然你可能在实际代码中永远不会用到这个，但有可能创建一个存储到容器中迭代器（除了 `std::string` 或 `char` 缓冲区）的匹配或子匹配对象。例如，这里是我们相同的函数，但将正则表达式与 `std::list<char>` 匹配——愚蠢，但它有效！

```cpp
    template<class Iter>
    std::pair<std::string, std::string>
    parse_command(Iter begin, Iter end) 
    {
      static std::regex rx("(left|right) ([0-9]+)");
      std::match_results<Iter> m;
      if (std::regex_match(begin, end, m, rx)) {
        return { m.str(1), m.str(2) };
      } else {
        throw "Unrecognized command!";
      }
    }

    void test() {
      char buf[] = "left 20";
      std::list<char> lst(buf, buf + 7);
      auto [dir, dist] = parse_command(lst.begin(), lst.end());
      assert(dir == "left" && dist == "20");
    }
```

# 将子匹配转换为数据值

只为了完成解析的闭环，这里有一个例子，说明我们如何从子匹配中解析字符串和整数值，以实际移动我们的机器人：

```cpp
    int main()
    {
      std::regex rx("(left|right) ([0-9]+)");
      int pos = 0;
      std::string line;
      while (std::getline(std::cin, line)) {
        try {
          std::smatch m;
          if (!std::regex_match(line, m, rx)) {
              throw std::runtime_error("Failed to lex");
          }
          int how_far = std::stoi(m.str(2));
          int direction = (m[1] == "left") ? -1 : 1;
          pos += how_far * direction;
          printf("Robot is now at %d.\n", pos);
        } catch (const std::exception& e) {
          puts(e.what());
          printf("Robot is still at %d.\n", pos);
        }
      }
    }
```

任何未识别或无效的字符串输入将通过我们自定义的 `"Failed to lex"` 异常或由 `std::stoi()` 抛出的 `std::out_of_range` 异常来诊断。如果我们修改 `pos` 之前添加一个整数溢出的检查，我们将有一个坚不可摧的输入解析器。

如果我们想要处理负整数和大小写不敏感的方向，以下修改将有效：

```cpp
    int main()
    {
      std::regex rx("((left)|right) (-?[0-9]+)", std::regex::icase);
      int pos = 0;
      std::string line;
      while (std::getline(std::cin, line)) {
        try {
          std::smatch m;
          if (!std::regex_match(line, m, rx)) {
            throw std::runtime_error("Failed to lex");
          }
          int how_far = std::stoi(m.str(3));
          int direction = m[2].matched ? -1 : 1;
          pos += how_far * direction;
          printf("Robot is now at %d.\n", pos);
        } catch (const std::exception& e) {
          puts(e.what());
          printf("Robot is still at %d.\n", pos);
        }
      }
    }
```

# 遍历多个匹配项

考虑正则表达式 `(?!\d)\w+`，它匹配单个 C++ 标识符。我们已经知道如何使用 `std::regex_match` 来判断输入字符串是否是 C++ 标识符，以及如何使用 `std::regex_search` 来找到给定输入行中的第一个 C++ 标识符。但如果我们想要找到给定输入行中的所有 C++ 标识符呢？

这里的基本思想是在循环中调用 `std::regex_search`。然而，由于非消耗性的“向后看”锚点，如 `^` 和 `\b`，这会变得复杂。要从头开始正确实现 `std::regex_search` 的循环，我们必须保留这些锚点的状态。`std::regex_search`（以及 `std::regex_match`）通过提供自己的标志来支持这种用例——这些标志决定了这个特定匹配操作的有限状态机的 *起始状态*。对我们来说，唯一重要的标志是 `std::regex::match_prev_avail`，它告诉库迭代器 `begin`（表示输入的开始）实际上不在输入的“开始”处（即它可能不匹配 `^`），并且如果你想要知道输入的上一字符用于 `\b`，检查 `begin[-1]` 是安全的：

```cpp
    auto get_all_matches(
      const char *begin, const char *end,
      const std::regex& rx,
      bool be_correct)
    {
      auto flags = be_correct ?
      std::regex_constants::match_prev_avail :
      std::regex_constants::match_default;
      std::vector<std::string> result;
      std::cmatch m;
      std::regex_search(begin, end, m, rx);
      while (!m.empty()) {
        result.push_back(m[0]);
        begin = m[0].second;
        std::regex_search(begin, end, m, rx, flags);
      }
      return result;
    }

    void test() {
      char buf[] = "baby";
      std::regex rx("\\bb.");
        // get the first 2 letters of each word starting with "b"
      auto v = get_all_matches(buf, buf+4, rx, false);
      assert(v.size() == 2);
        // oops, "by" is considered to start on a word boundary! 

      v = get_all_matches(buf, buf+4, rx, true);
      assert(v.size() == 1);
        // "by" is correctly seen as part of the word "baby"
    }
```

在前面的示例中，当 `!be_correct` 时，每次 `regex_search` 调用都是独立处理的，所以从单词 `"by"` 的第一个字母搜索 `\bb.` 和从单词 `"baby"` 的第三个字母搜索 `\bb.` 之间没有区别。但是当我们把 `match_prev_avail` 传递给 `regex_search` 的后续调用时，它会实际退后一步——看看 `"by"` 前面的字母是否是一个“单词”字母。由于前面的 `"a"` 是一个单词字母，第二个 `regex_search` 正确地拒绝将 `"by"` 作为匹配项。

在循环中使用 `regex_search` 很简单... 除非给定的正则表达式可能会匹配一个空字符串！如果正则表达式返回一个成功的匹配 `m`，其中 `m[0].length() == 0`，那么我们就会有一个无限循环。所以我们的 `get_all_matches()` 的内部循环实际上应该看起来更像是这样：

```cpp
    while (!m.empty()) {
      result.push_back(m[0]);
      begin = m[0].second;
      if (begin == end) break;
      if (m[0].length() == 0) ++begin;
      if (begin == end) break;
      std::regex_search(begin, end, m, rx, flags);
    }
```

标准库提供了一个名为 `std::regex_iterator` 的“便利”类型，它将封装前面代码片段的逻辑；使用 `regex_iterator` 可能会节省你一些与零长度匹配相关的微妙错误。遗憾的是，它不会节省你的任何打字，而且它略微增加了悬挂迭代器陷阱的可能性。`regex_iterator` 与 `match_results` 一样，在底层迭代器类型上进行了模板化，所以如果你正在匹配 `std::string` 输入，你想要 `std::sregex_iterator`，如果你正在匹配 `const char *` 输入，你想要 `std::cregex_iterator`。以下是将前面的示例重新编码为 `sregex_iterator` 的代码：

```cpp
    auto get_all_matches(
      const char *begin, const char *end,
      const std::regex& rx)
    {
      std::vector<std::string> result;
      using It = std::cregex_iterator;
      for (It it(begin, end, rx); it != It{}; ++it) {
        auto m = *it;
        result.push_back(m[0]);
      }
      return result;
    }
```

考虑一下这个笨拙的 for 循环如何从辅助类中受益

来自 第九章末尾的示例 的 `streamer<T>`，*Iostreams*。

你也可以手动遍历每个匹配中的子匹配，或者使用一个“便利”库类型。手动的话，看起来可能像这样：

```cpp
    auto get_tokens(const char *begin, const char *end,
      const std::regex& rx)
    {
      std::vector<std::string> result;
      using It = std::cregex_iterator;
      std::optional<std::csub_match> opt_suffix;
      for (It it(begin, end, rx); it != It{}; ++it) {
        auto m = *it;
        std::csub_match nonmatching_part = m.prefix();
        result.push_back(nonmatching_part);
        std::csub_match matching_part = m[0];
        result.push_back(matching_part);
        opt_suffix = m.suffix();
      }
      if (opt_suffix.has_value()) {
        result.push_back(opt_suffix.value());
      }
      return result;
    }
```

回想一下，`regex_iterator` 只是 `regex_search` 的包装，所以在这种情况下，`m.prefix()` 保证包含整个非匹配部分，一直回溯到上一个匹配的末尾。通过交替推送非匹配前缀和匹配项，并以非匹配后缀的特殊情况结束，我们将输入字符串分割成一个交替出现 "单词" 和 "单词分隔符" 的向量。如果要保存的只是 "单词" 或 "分隔符"，或者甚至要保存 `m[1]` 而不是 `m[0]`，则很容易修改此代码；或者甚至保存 `m[1]` 而不是 `m[0]`，等等。

库类型 `std::sregex_token_iterator` 非常直接地封装了所有这些逻辑，尽管如果你不熟悉前面的手动代码，其构造函数接口相当复杂。`sregex_token_iterator` 的构造函数接受一个输入迭代器对、一个正则表达式，然后是一个 *子匹配索引的向量*，其中索引 `-1` 是一个特殊情况，表示 "前缀（以及后缀）。"

```cpp
    auto get_tokens(const char *begin, const char *end,
      const std::regex& rx)
    {
      std::vector<std::string> result;
      using TokIt = std::cregex_token_iterator;
      for (TokIt it(begin, end, rx, {-1, 0}); it != TokIt{}; ++it) {
        std::csub_match some_part = *it;
        result.push_back(some_part);
      }
      return result;
    }
```

如果我们将数组 `{-1, 0}` 改为仅 `{0}`，那么我们的结果向量将只包含

仅匹配 `rx` 的输入字符串的片段。如果我们将其更改为 `{1, 2, 3}`，我们的

循环将只看到每个 `rx` 匹配 `m` 中的那些子匹配（`m[1]`、`m[2]` 和 `m[3]`）。回想一下，由于 `|` 操作符，子匹配可以被跳过，使得 `m[k].matched` 为假。`regex_token_iterator` 不会跳过这些匹配。例如：

```cpp
    std::string input = "abc123...456...";
    std::vector<std::ssub_match> v;
    std::regex rx("([0-9]+)|([a-z]+)");
    using TokIt = std::sregex_token_iterator;
    std::copy(
      TokIt(input.begin(), input.end(), rx, {1, 2}),
      TokIt(),
      std::back_inserter(v)
    );
    assert(!v[0].matched); assert(v[1] == "abc");
    assert(v[2] == "123"); assert(!v[3].matched);
    assert(v[4] == "456"); assert(!v[5].matched);
```

`regex_token_iterator` 最吸引人的用途可能是将字符串在空白边界处分割成 "单词"。不幸的是，它并不比老式方法（如 `istream_iterator<string>` （见第九章 [part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d]，*Iostreams*）或 `strtok_r`）更容易使用——或者更容易调试。

# 使用正则表达式进行字符串替换

如果你来自 Perl，或者你经常使用命令行工具 `sed`，你可能会主要将正则表达式视为修改字符串的一种方式——例如，"删除所有匹配此正则表达式的子串"，或者"将所有此单词的实例替换为另一个单词"。C++ 标准库确实提供了一种名为 `std::regex_replace` 的正则表达式替换功能。它是基于 JavaScript 的 `String.prototype.replace` 方法，这意味着它自带了一种独特的格式化迷你语言。

`std::regex_replace(str, rx, "replacement")` 返回一个由 `std::string` 构造的字符串，该字符串通过在 `str` 中搜索每个匹配正则表达式 `rx` 的子串，并将每个这样的子串替换为字面字符串 `"replacement"`。例如：

```cpp
    std::string s = "apples and bananas";
    std::string t = std::regex_replace(s, std::regex("a"), "e");
    assert(t == "epples end benenes");
    std::string u = std::regex_replace(s, std::regex("[ae]"), "u");
    assert(u == "upplus und bununus");
```

然而，如果 `"replacement"` 包含任何 `'$'` 字符，会发生特殊的事情！

+   `"$&"` 被替换为整个匹配子串，`m[0]`。libstdc++ 和 libc++ 都支持 `"$0"` 作为 `"$&"` 的非标准同义词。

+   `"$1"`被替换为第一个子匹配`m[1]`；`"$2"`被替换为`m[2]`；以此类推，直到`"$99"`。无法引用第 100 个子匹配。`"$100"`表示"`m[10]`"后面跟着一个字面字符`'0'`。要表示"`m[1]`"后面跟着一个字面字符`'0'`，请写`"$010"`。

+   `"$`"`（这是一个反引号）被替换为`m.prefix()`。

+   `"$'"`（这是一个单引号）被替换为`m.suffix()`。

+   `"$$"`被替换为一个字面美元符号。

注意，`"$`"`和`"$'"`远非对称，因为`m.prefix()`始终指向最后一个匹配的末尾和当前匹配的开始之间的字符串部分，而`m.suffix()`始终指向当前匹配的末尾和字符串末尾之间的字符串部分！你永远不会在实际代码中使用`"$`"`或`"$'"`。

这里是一个使用`regex_replace`从代码片段中删除所有`std::`实例或将它们全部更改为`my::`的示例：

```cpp
    auto s = "std::sort(std::begin(v), std::end(v))";
    auto t = std::regex_replace(s, std::regex("\\bstd::(\\w+)"), "$1");
    assert(t == "sort(begin(v), end(v))");
    auto u = std::regex_replace(s, std::regex("\\bstd::(\\w+)"), "my::$1");
    assert(u == "my::sort(my::begin(v), my::end(v))");
```

JavaScript 的`String.prototype.replace`允许你传入一个任意函数而不是带美元符号的格式字符串。C++的`regex_replace`目前还不支持任意函数，但可以轻松编写自己的版本来实现这一点：

```cpp
    template<class F>
    std::string regex_replace(std::string_view haystack,
      const std::regex& rx, const F& f)
    {
      std::string result;
      const char *begin = haystack.data();
      const char *end = begin + haystack.size();
      std::cmatch m, lastm;
      if (!std::regex_search(begin, end, m, rx)) {
        return std::string(haystack);
      }
      do {
        lastm = m;
        result.append(m.prefix());
        result.append(f(m));
        begin = m[0].second;
        begin += (begin != end && m[0].length() == 0);
        if (begin == end) break;
      } while (std::regex_search(begin, end, m, rx,
        std::regex_constants::match_prev_avail));
      result.append(lastm.suffix());
      return result;
    }

    void test()
    {
      auto s = "std::sort(std::begin(v), std::end(v))";
      auto t = regex_replace(s, std::regex("\\bstd::(\\w+)"),
        [](auto&& m) {
          std::string result = m[1].str();
          std::transform(m[1].first, m[1].second,
          begin(result), ::toupper);
          return result;
        });
      assert(t == "SORT(BEGIN(v), END(v))");
    }
```

使用这个改进的`regex_replace`，你可以轻松执行复杂的操作，例如“将每个标识符从`snake_case`转换为`CamelCase`”。

这就结束了我们对 C++ `<regex>`头文件中提供的功能的快速浏览。本章的其余部分是对 ECMAScript 方言的正则表达式符号的详细介绍。我希望它对之前没有使用过正则表达式的读者有所帮助，并且对那些已经使用过正则表达式的人来说，它将作为一个复习和参考。

# ECMAScript 正则表达式语法的入门指南

在 ECMAScript 方言中读取和编写正则表达式的规则很简单。正则表达式只是一系列字符（例如`a[bc].d*e`），并且你应该从左到右读取它。大多数字符仅代表自身，因此`cat`是一个有效的正则表达式，仅匹配字面字符串`"cat"`。唯一不表示自身的字符——也是构建表示比`"cat"`更有趣的语言的正则表达式的唯一方式——是以下标点符号：

```cpp
    ^ $ \ . * + ? ( ) [ ] { } |
```

`\`——如果你使用正则表达式来描述涉及标点符号的字符串集合，你可以使用反斜杠来转义这些特殊字符。例如，`\$42\.00`是一个正则表达式，表示只包含字符串`"$42.00"`的单例语言。也许有些令人困惑的是，反斜杠还被用来将一些普通字符转换为特殊字符！`n`是一个表示字母`n`的正则表达式，但`\n`是一个表示换行符的正则表达式。`d`是一个表示字母`d`的正则表达式，但`\d`是一个等同于`[0-9]`的正则表达式。

C++的正则表达式语法所识别的反斜杠字符的完整列表是：

+   `\1`、`\2`、... `\10`、... 用于后向引用（应避免使用）

+   `\b`用于单词边界和`\B`用于`(?!\b)`

+   `\d` 用于 `[[:digit:]]` 和 `\D` 用于 `[^[:digit:]]`

+   `\s` 用于 `[[:space:]]` 和 `\S` 用于 `[^[:space:]]`

+   `\w` 用于 `[0-9A-Za-z_]` 和 `\W` 用于 `[⁰-9A-Za-z_]`

+   `\cX` 用于各种“控制字符”（应避免使用）

+   `\xXX` 用于十六进制，具有通常的含义

+   `\u00XX` 用于 Unicode，具有通常的含义

+   `\0`、`\f`、`\n`、`\r`、`\t`、`\v` 具有它们通常的含义

`.`——这个特殊字符表示“正好一个字符”，几乎没有其他要求。例如，`a.c` 是一个有效的正则表达式，并匹配如 `"aac"`、`"a!c"` 和 `"a\0c"` 这样的输入。然而，`.` 永远不会匹配换行符或回车符；并且由于 C++ 正则表达式在字节级别工作，而不是在 Unicode 级别，`.` 会匹配任何单个字节（除了 `'\\n'` 和 `'\\r'`），但即使它们偶然组成一个有效的 UTF-8 代码点，也不会匹配多个字节的序列。

`[]`——一个包含在方括号内的字符组表示“正好是这个集合中的一个”，因此 `c[aou]t` 是一个有效的正则表达式，并匹配字符串 `"cat"`、`"cot"` 和 `"cut"`。你可以使用方括号语法来“转义”大多数字符；例如，`[$][.][*][+][?][(][)][[][{][}][|]` 是一个单成员语言的正则表达式，其唯一成员是字符串 `"$.*+?()[{}|"`。然而，你不能使用方括号来转义 `]`、`\` 或 `^`。

`[^]`——一个以 `^` 开头并包含在方括号内的字符组表示“正好一个，不是这个集合中的”，因此 `c[^aou]t` 将匹配 `"cbt"` 或 `"c^t"` 但不会匹配 `"cat"`。ECMAScript 方言不特别处理 `[]` 或 `[^]` 的平凡情况；`[]` 表示“来自空集的正好一个字符”（也就是说，它永远不会匹配任何内容），而 `[^]` 表示“不是来自空集的正好一个字符”（也就是说，它匹配任何单个字符——就像 `.` 但更好，因为它会匹配换行符和回车符）。

`[]` 语法对一些字符有特殊处理：如果 `-` 出现在方括号内，除了作为第一个或最后一个字符外，它表示一个“范围”，其左右邻居为范围。因此，`ro[s-v]e` 是一个正则表达式，用于匹配语言成员为四个字符串：`"rose"`、`"rote"`、`"roue"`和`"rove"`。一些常用范围——与 `<ctype.h>` 头文件中暴露的范围相同——使用方括号内的 `[:foo:]` 语法内置：`[[:digit:]]` 与 `[0-9]` 相同，`[[:upper:][:lower:]]` 与 `[[:alpha:]]` 相同，即 `[A-Za-z]`，等等。

还有一些内置语法看起来像 `[[.x.]]` 和 `[[=x=]]`；它们处理与区域设置相关的比较，你永远不会需要使用它们。只需知道，如果你需要在方括号字符类中包含字符 `[`，最好使用反斜杠转义：`foo[=([;]` 和 `foo[(\[=;]` 匹配字符串 `"foo="`、`"foo("`、`"foo["` 和 `"foo;"`，但 `foo[([=;]` 是一个无效的正则表达式，在尝试从它构造 `std::regex` 对象时会在运行时抛出异常。

`+`--一个表达式或单个字符后面紧跟 `+` 可以匹配前面的表达式或字符任意正次数。例如，正则表达式 `ba+` 匹配字符串 `"ba"`、`"baa"`、`"baaa"` 等等。

`*`--一个表达式或单个字符后面紧跟 `*` 可以匹配前面的表达式或字符任意次数，包括零次。所以正则表达式 `ba*` 匹配字符串 `"ba"`、`"baa"` 和 `"baaa"`，甚至单独的 `"b"`。

`?`--一个表达式或单个字符后面紧跟 `?` 可以匹配前面的表达式或字符正好零次或一次。例如，正则表达式 `coo?t` 只匹配 `"cot"` 和 `"coot"`。

`{n}`--一个表达式或单个字符后面紧跟一个花括号中的整数，会精确匹配前面的表达式或字符指定次数。例如，`b(an){2}a` 是一个匹配 `"banana"` 的正则表达式；`b(an){3}a` 是一个匹配 `"bananana"` 的正则表达式。

`{m,n}`--当花括号结构由两个用逗号分隔的整数 *m* 和 *n* 组成时，该结构匹配前面的表达式或字符从 *m* 到 *n* 次数（包括）。所以 `b(an){2,3}a` 是一个只匹配字符串 `"banana"` 和 `"bananana"` 的正则表达式。

`{m,}`--留空 *n* 实际上使其无限；所以 `x{42,}` 表示“匹配 `x` 42 次或更多”，相当于 `x{42}x*`。ECMAScript 语法不允许留空 *m*。

`|`--两个正则表达式可以用 `|` 连接起来，表示“或”的概念。例如，`cat|dog` 是一个只匹配字符串 `"cat"` 和 `"dog"` 的正则表达式；而 `(tor|shark)nado` 匹配 `"tornado"` 或 `"sharknado"`。在正则表达式中，`|` 运算符的优先级非常低，就像它在 C++ 表达式中的优先级一样。

`()`--括号的作用就像在数学中一样，用于括住一个子表达式，将其紧密绑定并作为一个单元处理。例如，`ba*` 表示“字符 `b`，然后是零个或多个 `a` 的实例；但 `(ba)*` 表示“零个或多个 `ba` 的实例。”所以前者匹配 `"b"`、`"ba"`、`"baa"` 等等；但带括号的那个版本匹配 `""`、`"ba"`、`"baba"` 等等。

括号也有第二个用途--它们不仅用于 *分组*，还用于 *捕获* 匹配的部分以进行进一步处理。正则表达式中的每个开括号 `(` 都会在结果 `std::smatch` 对象中生成另一个子匹配。

如果你想要将某些子表达式紧密地组合在一起而不生成子匹配，你可以使用语法 `(?:foo)` 的非捕获组：

```cpp
    std::string s = "abcde";
    std::smatch m;
    std::regex_match(s, m, std::regex("(a|b)*(.*)e"));
    assert(m.size() == 3 && m[2] == "cd");
    std::regex_match(s, m, std::regex("(?:a|b)*(.*)e"));
    assert(m.size() == 2 && m[1] == "cd");
```

非捕获性可能在某些隐藏的上下文中很有用；但通常，如果你只是使用常规捕获 `()` 并忽略你不在乎的子匹配，而不是在你的代码库中散布 `(?:)` 以尝试压制所有未使用的子匹配，这将使读者更清楚。未使用的子匹配在性能上非常便宜。

# 非消耗性结构

`(?=foo)` 匹配输入中的模式 `foo`，然后“回滚”以使输入实际上没有消耗。这被称为“向前查看”。所以例如 `c(?=a)(?=a)(?=a)at` 匹配 `"cat"`；而 `(?=.*[A-Za-z])(?=.*[0-9]).*` 匹配包含至少一个字母字符和至少一个数字的任何字符串。

`(?!foo)` 是一个“负向前查看”；它向前查看以匹配输入中的 `foo`，但如果 `foo` 被接受，则*拒绝*匹配，如果 `foo` 被拒绝，则*接受*匹配。所以，例如，`(?!\d)\w+` 匹配任何 C++ 标识符或关键字--也就是说，任何不以数字开头的字母数字字符序列。请注意，第一个字符必须不匹配 `\d` 但不被 `(?!\d)` 结构消耗；它仍然必须被 `\w` 接受。类似外观的正则表达式 `[⁰-9]\w+` 会“错误地”接受像 `"#xyzzy"` 这样的字符串，这些字符串不是有效的标识符。

`(?=)` 和 `(?!)` 不仅是非消耗性的，而且是*非捕获性的*，就像 `(?:)` 一样。但是，写 `(?=(foo))` 来捕获“向前查看”的部分的全部或部分是完全可行的。

`^` 和 `$`--一个单独的、不在任何方括号内的撇号 `^` 仅匹配要匹配的字符串的开始；而 `$` 仅匹配字符串的末尾。这在 `std::regex_search` 的上下文中非常有用，可以“锚定”正则表达式到输入字符串的开始或结束。在 `std::regex::multiline` 正则表达式中，`^` 和 `$` 分别作为“向后查看”和“向前查看”断言：

```cpp
    std::string s = "ab\ncd";
    std::regex rx("^ab$[^]^cd$", std::regex::multiline);

    assert(std::regex_match(s, rx));
```

将所有这些放在一起，我们可能会写出正则表达式 `foo[a-z_]+(\d|$)` 来匹配“字母 `foo` 后跟一个或多个其他字母和/或下划线；然后跟一个数字或行尾。”

如果你需要深入了解正则表达式语法，请参阅 [cppreference.com](https://cppreference.com)。如果还不够--C++ 从 ECMAScript 风格的正则表达式复制来的最好之处在于，任何关于 JavaScript 正则表达式的教程也适用于 C++！你甚至可以在浏览器控制台中测试正则表达式。C++ 正则表达式和 JavaScript 正则表达式之间的唯一区别是，C++ 支持字符类如 `[[:digit:]]`、`[[.x.]]` 和 `[[=x=]]` 的双方括号语法，而 JavaScript 不支持。JavaScript 将这些正则表达式视为与 `[\[:digit:]]`、`[\[.x\]]` 和 `[\[=x\]]` 分别等价。

# 隐藏的 ECMAScript 功能和陷阱

在本章的早期，我提到了一些 `std::regex` 的特性，你最好避免使用，例如 `std::regex::collate`、`std::regex::optimize` 以及改变方言远离 ECMAScript 的标志。ECMAScript 正则表达式语法本身也包含一些晦涩且应避免的特性。

一个反斜杠后跟一个或多个数字（除了 `\0`）会创建一个**回溯引用**。回溯引用 `\1` 匹配“与我第一个捕获组匹配的相同字符序列”；例如，正则表达式 `(cat|dog)\1` 会匹配字符串 `"catcat"` 和 `"dogdog"`，但不会匹配 `"catdog"`，而 `(a*)(b*)c\2\1` 会匹配 `"aabbbcbbbaa"`，但不会匹配 `"aabbbcbbba"`。回溯引用可以具有微妙而奇怪的语义，特别是当与 `(?=foo)` 这样的非消耗性构造结合使用时，我建议在可能的情况下避免使用它们。

如果你遇到回溯引用的问题，首先检查的是你的反斜杠转义。记住，`std::regex("\1")` 是匹配 ASCII 控制字符编号 1 的正则表达式。你本想输入的是 `std::regex("\\1")`。

使用回溯引用将你带出了**正则语言**的世界，进入了更广泛的**上下文相关语言**的世界，这意味着库必须放弃其基于有限状态机的高效匹配算法，转而使用更强大但昂贵且缓慢的“回溯”算法。这似乎是避免回溯引用的另一个很好的理由，除非它们绝对必要。

然而，截至 2017 年，大多数供应商实际上并不会根据正则表达式中的**回溯引用的存在**来切换算法；他们会在 ECMAScript 正则表达式方言中基于回溯引用的**可能性**使用较慢的回溯算法。然后，因为没有任何供应商愿意为没有回溯引用的方言 `std::regex::awk` 和 `std::regex::extended` 实现整个第二个算法，他们最终甚至为这些方言使用回溯算法！同样，大多数供应商将 `regex_match(s, rx)` 实现为 `regex_match(s, m, rx)`，然后丢弃昂贵的计算结果 `m`，而不是使用可能更快的 `regex_match(s, rx)` 算法。这样的优化可能在未来的 10 年内出现在某个库中，但我不会为此而等待。

另一个鲜为人知的特性是，`*`、`+` 和 `?` 量词默认都是**贪婪的**，这意味着例如 `(a*)` 会尽可能多地匹配 `a` 字符。你可以通过在量词后附加一个额外的 `?` 来将贪婪量词转换为**非贪婪的**；例如 `(a*?)` 会匹配尽可能少的 `a` 字符。除非你使用捕获组，否则这不会产生任何区别。以下是一个例子：

```cpp
    std::string s = "abcde";
    std::smatch m;
    std::regex_match(s, m, std::regex(".*([bcd].*)e"));
    assert(m[1] == "d");
    std::regex_match(s, m, std::regex(".*?([bcd].*)e"));
    assert(m[1] == "bcd");
```

在第一种情况下，`.*` 贪婪地匹配 `abc`，只留下 `d` 由捕获组进行匹配。在第二种情况下，`.*?` 非贪婪地只匹配 `a`，留下 `bcd` 给捕获组。实际上，`.*?` 更愿意匹配空字符串；但是，如果没有整体匹配被拒绝，它就不能这样做。

注意，非贪婪性的语法并不遵循“正常”的运算符组合规则。根据我们对 C++ 运算符语法的了解，我们预计 `a+*` 应该意味着 `(a+)*`（它确实如此），而 `a+?` 应该意味着 `(a+)?`（但它并不这样）。因此，如果你在正则表达式中看到连续的标点符号字符，要小心——它可能意味着与你的直觉告诉你的不同！

# 摘要

正则表达式（regexes）是在解析之前从输入字符串中提取片段的好方法。C++ 中的默认正则表达式方言与 JavaScript 相同。利用这一点。

在可能的情况下，避免使用原始字符串字面量，因为额外的括号可能会造成混淆。在正则表达式中，尽可能限制转义反斜杠的数量，通过使用方括号来转义特殊字符。

`std::regex rx` 基本上是不可变的，代表一个有限状态机。`std::smatch m` 是可变的，并包含关于草堆字符串中特定匹配的信息。子匹配 `m[0]` 代表整个匹配的子字符串；`m[k]` 代表第 *k* 个捕获组。

`std::regex_match(s, m, rx)` 将针针对整个草堆字符串进行匹配；`std::regex_search(s, m, rx)` 在草堆中寻找针。记住，草堆在前，针在后，就像在 JavaScript 和 Perl 中一样。

`std::regex_iterator`, `std::regex_token_iterator`, 和 `std::regex_replace` 是在 `regex_search` 基础上构建的相对不便的“便利”函数。在使用这些包装器之前，先熟悉 `regex_search`。

警惕悬挂迭代器错误！永远不要修改或销毁一个仍被 `regex_iterator` 引用的 `regex`；永远不要修改或销毁一个仍被 `smatch` 引用的 `string`。
