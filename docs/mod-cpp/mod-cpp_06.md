# 第六章：使用字符串

在某个时候，您的应用程序将需要与人们交流，这意味着使用文本；例如输出文本，以文本形式接收数据，然后将该数据转换为适当的类型。C++标准库有丰富的类集合，用于操作字符串，将字符串和数字之间进行转换，并获取特定语言和文化环境的本地化字符串值。

# 将字符串类作为容器使用

C++字符串基于`basic_string`模板类。这个类是一个容器，所以它使用迭代器访问和方法来获取信息，并且具有包含有关其保存的字符类型的信息的模板参数。有不同的特定字符类型的`typedef`：

```cpp
    typedef basic_string<char,
       char_traits<char>, allocator<char> > string; 
    typedef basic_string<wchar_t,
       char_traits<wchar_t>, allocator<wchar_t> > wstring; 
    typedef basic_string<char16_t,
       char_traits<char16_t>, allocator<char16_t> > u16string; 
    typedef basic_string<char32_t,
       char_traits<char32_t>, allocator<char32_t> > u32string;
```

`string`类基于`char`，`wstring`基于`wchar_t`宽字符，`16string`和`u32string`类分别基于 16 位和 32 位字符。在本章的其余部分，我们将集中讨论`string`类，但它同样适用于其他类。

比较、复制和访问字符串中的字符将需要针对不同大小的字符编写不同的代码，而特性模板参数提供了实现。对于`string`，这是`char_traits`类。例如，当这个类复制字符时，它将把这个动作委托给`char_traits`类及其`copy`方法。特性类也被流类使用，因此它们还定义了适合文件流的文件结束值。

字符串本质上是一个零个或多个字符的数组，当需要时分配内存，并在销毁`string`对象时释放它。在某些方面，它与`vector<char>`对象非常相似。作为容器，`string`类通过`begin`和`end`方法提供迭代器访问：

```cpp
    string s = "hellon"; 
    copy(s.begin(), s.end(), ostream_iterator<char>(cout));
```

在这里，调用`begin`和`end`方法以从`string`中的项获取迭代器，然后将这些迭代器传递给`<algorithm>`中的`copy`函数，以通过`ostream_iterator`临时对象将每个字符复制到控制台。在这方面，`string`对象类似于`vector`，因此我们使用先前定义的`s`对象：

```cpp
vector<char> v(s.begin(), s.end()); 
copy(v.begin(), v.end(), ostream_iterator<char>(cout));
```

使用`begin`和`end`方法填充`vector`对象，这些方法在`string`对象上提供了一系列字符，然后使用`copy`函数将这些字符以与之前相同的方式打印到控制台。

# 关于字符串的信息

`max_size`方法将给出计算机架构上指定字符类型的字符串的最大大小，这可能会非常大。例如，在具有 2GB 内存的 64 位 Windows 计算机上，`string`对象的`max_size`将返回 40 亿个字符，而对于`wstring`对象，该方法将返回 20 亿个字符。这显然比机器上的内存多！其他大小方法返回更有意义的值。`length`方法返回与`size`方法相同的值，即字符串中有多少项（字符）。`capacity`方法指示已分配多少内存用于字符串的字符数。

您可以通过调用其`compare`方法将`string`与另一个字符串进行比较。这将返回一个`int`而不是`bool`（但请注意，`int`可以被静默转换为`bool`），其中返回值为`0`表示两个字符串相同。如果它们不相同，此方法将返回一个负值，如果参数字符串大于操作数字符串，则返回一个正值。在这方面，*大于*和*小于*将按字母顺序测试字符串的顺序。此外，还为`<`、`<=`、`==`、`>=`和`>`定义了全局运算符来比较字符串对象。

`string`对象可以通过`c_str`方法像 C 字符串一样使用。返回的指针是`const`的；您应该注意，如果更改了`string`对象，指针可能会失效，因此不应存储此指针。您不应该使用`&str[0]`来获取 C++字符串`str`的 C 字符串指针，因为字符串类使用的内部缓冲区不能保证为`NUL`终止。`c_str`方法用于返回一个指针，可以用作 C 字符串，因此是`NUL`终止的。

如果要从 C++字符串复制数据到 C 缓冲区，可以调用`copy`方法。您将目标指针和要复制的字符数作为参数传递（以及可选的偏移），该方法将尝试将最多指定数量的字符复制到目标缓冲区：*但不包括空终止字符*。该方法假定目标缓冲区足够大以容纳复制的字符（您应该采取措施来确保这一点）。如果要传递缓冲区的大小，以便该方法为您执行此检查，请调用`_Copy_s`方法。

# 修改字符串

字符串类具有标准的容器访问方法，因此您可以使用`at`方法和`[]`运算符通过引用（读写访问）访问单个字符。您可以使用`assign`方法替换整个字符串，或者使用`swap`方法交换两个字符串对象的内容。此外，您可以使用`insert`方法在指定位置插入字符，使用`erase`方法删除指定的字符，使用`clear`方法删除所有字符。该类还允许您使用`push_back`和`pop_back`方法将字符推送到字符串的末尾（并删除最后一个字符）。

```cpp
    string str = "hello"; 
    cout << str << "n"; // hello 
    str.push_back('!'); 
    cout << str << "n"; // hello! 
    str.erase(0, 1); 
    cout << str << "n"; // ello!
```

您可以使用`append`方法或`+=`运算符在字符串的末尾添加一个或多个字符。

```cpp
    string str = "hello"; 
    cout << str << "n";  // hello 
    str.append(4, '!'); 
    cout << str << "n";  // hello!!!! 
    str += " there"; 
    cout << str << "n";  // hello!!!! there
```

`<string>`库还定义了一个全局的`+`运算符，用于将两个字符串连接成第三个字符串。

如果要更改字符串中的字符，可以使用`[]`运算符通过索引访问字符，并使用引用来覆盖字符。您还可以使用`replace`方法在指定位置用来自 C 字符串或 C++字符串的字符或通过迭代器访问的其他容器替换一个或多个字符。

```cpp
    string str = "hello"; 
    cout << str << "n";    // hello 
    str.replace(1, 1, "a"); 
    cout << str << "n";    // hallo
```

最后，您可以将字符串的一部分提取为新字符串。`substr`方法接受偏移和可选计数。如果省略字符的计数，则子字符串将从指定位置到字符串的末尾。这意味着您可以通过传递偏移为 0 和计数小于字符串大小的方式复制字符串的左侧部分，或者通过仅传递第一个字符的索引来复制字符串的右侧部分。

```cpp
    string str = "one two three"; 
    string str1 = str.substr(0, 3);  
    cout << str1 << "n";          // one 
    string str2 = str.substr(8); 
    cout << str2 << "n";          // three
```

在此代码中，第一个示例将前三个字符复制到一个新字符串中。在第二个示例中，复制从第八个字符开始，一直到末尾。

# 搜索字符串

`find`方法可以使用字符、C 字符串或 C++字符串进行传递，并且您可以提供一个初始搜索位置来开始搜索。`find`方法返回搜索文本的位置（而不是迭代器），或者如果找不到文本，则返回`npos`值。偏移参数和`find`方法的成功返回值使您能够重复解析字符串以查找特定项。`find`方法在正向方向搜索指定的文本，还有一个`rfind`方法可以在反向方向执行搜索。

请注意，`rfind`并不是`find`方法的完全相反。`find`方法在字符串中向前移动搜索点，并在每个点上将搜索字符串与搜索点之后的字符进行比较（所以首先是搜索文本的第一个字符，然后是第二个字符，依此类推）。`rfind`方法向后移动搜索点，但比较仍然是*向前*进行的。所以，假设`rfind`方法没有给出偏移量，第一次比较将在字符串末尾与搜索文本大小的偏移量处进行。然后，通过将搜索文本中的第一个字符与搜索字符串中搜索点后的字符进行比较，如果成功，则将搜索文本中的第二个字符与搜索点后的字符进行比较。因此，比较是沿着搜索点移动的方向相反进行的。

这变得重要，因为如果你想使用`find`方法的返回值作为偏移量来解析一个字符串，每次搜索后你应该将搜索偏移量*向前*移动，而对于`rfind`，你应该将其*向后*移动。

例如，要在以下字符串中搜索`the`的所有位置，你可以调用：

```cpp
    string str = "012the678the234the890"; 
    string::size_type pos = 0; 
    while(true) 
    { 
        pos++; 
        pos = str.find("the",pos); 
        if (pos == string::npos) break; 
        cout << pos << " " << str.substr(pos) << "n"; 
    } 
    // 3 the678the234the890 
    // 9 the234the890 
    // 15 the890
```

这将在字符位置 3、9 和 15 找到搜索文本。要向后搜索字符串，可以调用：

```cpp
    string str = "012the678the234the890"; 
    string::size_type pos = string::npos; 
    while(true) 
    { 
        pos--; pos = str.rfind("the",pos); 
        if (pos == string::npos) break; 
        cout << pos << " " << str.substr(pos) << "n"; 
    } 
    // 15 the890 
    // 9 the234the890 
    // 3 the678the234the890
```

突出显示的代码显示了应该进行的更改，告诉你需要从末尾开始搜索并使用`rfind`方法。当你有一个成功的结果时，你需要在下一次搜索之前减少位置。与`find`方法一样，如果找不到搜索文本，`rfind`方法会返回`npos`。

有四种方法允许你搜索多个单个字符中的一个。例如：

```cpp
    string str = "012the678the234the890"; 
    string::size_type pos = str.find_first_of("eh"); 
    if (pos != string::npos) 
    { 
        cout << "found " << str[pos] << " at position "; 
        cout << pos << " " << str.substr(pos) << "n"; 
    } 
    // found h at position 4 he678the234the890
```

搜索字符串是`eh`，`find_first_of`会在字符串中找到`e`或`h`字符时返回。在这个例子中，字符`h`首先在位置 4 被找到。你可以提供一个偏移参数来开始搜索，所以你可以使用`find_first_of`的返回值来解析字符串。`find_last_of`方法类似，但它以相反的方向搜索搜索文本中的字符。

还有两种搜索方法，它们将查找搜索文本中*不是*提供的字符：`find_first_not_of`和`find_last_not_of`。例如：

```cpp
    string str = "012the678the234the890"; 
    string::size_type pos = str.find_first_not_of("0123456789"); 
    cout << "found " << str[pos] << " at position "; 
    cout << pos << " " << str.substr(pos) << "n"; 
    // found t at position 3 the678the234the890
```

这段代码查找的是非数字字符，所以它在位置 3（第四个字符）找到了`t`。

没有库函数可以从`string`中修剪空白字符，但你可以通过使用 find 函数找到非空白字符，然后将其作为`substr`方法的适当索引来修剪字符串的左侧和右侧空格。

```cpp
    string str = "  hello  "; 
    cout << "|" << str << "|n";  // |  hello  | 
    string str1 = str.substr(str.find_first_not_of(" trn")); 
    cout << "|" << str1 << "|n"; // |hello  | 
    string str2 = str.substr(0, str.find_last_not_of(" trn") + 1); 
    cout << "|" << str2 << "|n"; // |  hello|
```

在上面的代码中，创建了两个新的字符串：一个左侧修剪空格，另一个右侧修剪空格。第一个向前搜索第一个非空白字符，并将其用作子字符串的起始索引（因为没有提供计数，所以将复制所有剩余的字符串）。在第二种情况下，字符串是反向搜索非空白字符，但返回的位置将是`hello`的最后一个字符；因为我们需要从第一个字符开始的子字符串，所以我们增加这个索引以获得要复制的字符数。

# 国际化

`<locale>`头文件包含了本地化时间、日期和货币格式的类，还提供了本地化的字符串比较和排序规则。

C 运行时库还具有全局函数来执行本地化。但是，在以下讨论中，重要的是区分 C 函数和 C 区域设置。C 区域设置是 C 和 C++程序中使用的默认区域设置，包括本地化规则，可以用国家或文化的区域设置替换。C 运行时库提供了更改区域设置的函数，C++标准库也提供了这些函数。

由于 C++标准库提供了本地化类，这意味着可以创建多个表示区域设置的对象。区域设置对象可以在函数中创建，并且只能在那里使用，或者可以全局应用于线程，并且仅由在该线程上运行的代码使用。这与 C 本地化函数相反，其中更改区域设置是全局的，因此所有代码（以及所有执行线程）都会受到影响。

`locale` 类的实例可以通过类构造函数或类的静态成员创建。C++流类将使用区域设置（稍后解释），如果要更改区域设置，则调用流对象的 `imbue` 方法。在某些情况下，您可能需要直接访问其中一个规则，并且可以通过区域设置对象访问它们。

# 使用 facet

国际化规则称为**facet**。区域设置对象是 facet 的容器，可以使用 `has_facet` 函数测试区域设置是否具有特定 facet；如果有，可以通过调用 `use_facet` 函数获得 facet 的 `const` 引用。以下表格总结了七个类别的七种类别的六种 facet 类型。facet 类是 `locale::facet` 嵌套类的子类。

| **Facet 类型** | **描述** |
| --- | --- |
| `codecvt`，`ctype` | 在不同编码方案之间进行转换，并用于对字符进行分类并将其转换为大写或小写 |
| `collate` | 控制字符串中字符的排序和分组，包括比较和哈希字符串 |
| `messages` | 从目录中检索本地化消息 |
| `money` | 将表示货币的数字转换为字符串，反之亦然 |
| `num` | 将数字转换为字符串，反之亦然 |
| `时间` | 将数字形式的时间和日期转换为字符串，反之亦然 |

facet 类用于将数据转换为字符串，因此它们都具有用于字符类型的模板参数。`money`，`num` 和 `time` facet 由三个类表示。具有 `_get` 后缀的类处理解析字符串，而具有 `_put` 后缀的类处理格式化为字符串。对于 `money` 和 `num` facet，有一个包含标点规则和符号的 `punct` 后缀的类。

由于 `_get` facet 用于将字符序列转换为数值类型，因此类具有模板参数，您可以使用该参数指示 `get` 方法将用于表示字符范围的输入迭代器类型。同样，`_put` facet 类具有模板参数，您可以使用该参数提供 `put` 方法将转换后的字符串写入的输出迭代器类型。对于两种迭代器类型都提供了默认类型。

`messages` facet 用于与 POSIX 代码兼容。该类旨在允许您为应用程序提供本地化字符串。其想法是，用户界面中的字符串被索引，并且在运行时，您可以通过 `messages` facet 使用索引访问本地化字符串。但是，Windows 应用程序通常使用使用**消息编译器**编译的消息资源文件。也许正因为这个原因，标准库提供的 `messages` facet 并不执行任何操作，但是基础设施已经存在，您可以派生自己的 `messages` facet 类。

`has_facet`和`use_facet`函数是为你想要的特定类型的 facet 进行模板化的。所有 facet 类都是`locale::facet`类的子类，但通过这个模板参数，编译器将实例化一个返回你请求的特定类型的函数。所以，例如，如果你想要为法语区域设置格式化时间和日期字符串，你可以调用这段代码：

```cpp
    locale loc("french"); 
    const time_put<char>& fac = use_facet<time_put<char>>(loc);
```

在这里，`french`字符串标识了区域设置，这是 C 运行时库`setlocale`函数使用的语言字符串。第二行获取了用于将数字时间转换为字符串的 facet，因此函数模板参数是`time_put<char>`。这个类有一个叫做`put`的方法，你可以调用它来执行转换：

```cpp
    time_t t = time(nullptr); 
    tm *td = gmtime(&t); 
    ostreambuf_iterator<char> it(cout); 
    fac.put(it, cout, ' ', td, 'x', '#'); 
    cout << "n";
```

`time`函数（通过`<ctime>`）返回一个带有当前时间和日期的整数，然后使用`gmtime`函数将其转换为`tm`结构。`tm`结构包含年、月、日、小时、分钟和秒的各个成员。`gmtime`函数返回一个在函数中静态分配的结构的地址，因此你不必删除它占用的内存。

facet 将`tm`结构中的数据格式化为一个字符串，通过作为第一个参数传递的输出迭代器。在这种情况下，输出流迭代器是从`cout`对象构造的，因此 facet 将把格式化流写入控制台（第二个参数没有被使用，但因为它是一个引用，你必须传递一些东西，所以也在那里使用了`cout`对象）。第三个参数是分隔符字符（同样，这也没有被使用）。第五和（可选的）第六个参数指示你需要的格式化。这些是与 C 运行时库函数`strftime`中使用的相同的格式化字符，作为两个单个字符，而不是 C 函数使用的格式字符串。在这个例子中，`x`用于获取日期，`#`用作字符串的长版本的修饰符。

代码将给出以下输出：

```cpp
    samedi 28 janvier 2017
```

注意单词没有大写，也没有标点符号，还要注意顺序：星期几名称，日期，月份，然后年份。

如果`locale`对象构造函数参数被更改为`german`，那么输出将是：

```cpp
    Samstag, 28\. January 2017
```

项目的顺序与法语中相同，但单词是大写的，使用了标点符号。如果你使用`turkish`，那么结果是：

```cpp
    28 Ocak 2017 Cumartesi
```

在这种情况下，星期几在字符串的末尾。

两个国家因共同语言而分裂，将给出两个不同的字符串，以下是`american`和`english-uk`的结果：

```cpp
    Saturday, January 28, 2017
28 January 2017
```

这里以时间作为示例，因为没有流，所以对于`tm`结构使用插入运算符是一个不寻常的情况。对于其他类型，有插入运算符将它们放入流中，因此流可以使用区域设置来国际化它显示的类型。例如，你可以将一个`double`插入到`cout`对象中，该值将被打印到控制台上。默认区域设置，美国英语，使用句点将整数部分与小数部分分开，但在其他文化中使用逗号。

`imbue`函数将改变本地化，直到随后调用该方法为止：

```cpp
    cout.imbue(locale("american")); 
    cout << 1.1 << "n"; 
    cout.imbue(locale("french")); 
    cout << 1.1 << "n"; 
    cout.imbue(locale::classic());
```

在这里，流对象被本地化为美国英语，然后浮点数`1.1`被打印到控制台上。接下来，本地化被更改为法语，这时控制台将显示`1,1`。在法语中，小数点是逗号。最后一行通过传递从`static classic`方法返回的区域设置了流对象。这返回了所谓的**C 区域**，它是 C 和 C++中的默认区域，是美国英语。

`static`方法`global`可以用来设置每个流对象默认使用的区域设置。当从流类创建对象时，它调用`locale::global`方法获取默认区域设置。流会克隆这个对象，以便它有自己独立于通过调用`global`方法设置的任何本地设置的副本。请注意，`cin`和`cout`流对象在调用`main`函数之前创建，这些对象将使用默认的 C 区域设置，直到您使用其他区域设置。然而，重要的是要指出，一旦流被创建，`global`方法对流没有影响，`imbue`是改变流使用的区域设置的唯一方法。

`global`方法还将调用 C `setlocale`函数来改变 C 运行时库函数使用的区域设置。这很重要，因为一些 C++函数（例如`to_string`，`stod`，如下文所述）将使用 C 运行时库函数来转换值。然而，C 运行时库对 C++标准库一无所知，因此调用 C `setlocale`函数来更改默认区域设置不会影响随后创建的流对象。

值得指出的是，`basic_string`类使用模板参数指示的字符特征类比较字符串。`string`类使用`char_traits`类，其`compare`方法的版本直接比较两个字符串中对应的字符。这种比较不考虑比较字符的文化规则。如果您想进行使用文化规则的比较，可以通过`collate` facet 来实现：

```cpp
    int compare( 
       const string& lhs, const string& rhs, const locale& loc) 
    { 
        const collate<char>& fac = use_facet<collate<char>>(loc); 
        return fac.compare( 
            &lhs[0], &lhs[0] + lhs.size(), &rhs[0], &rhs[0] + rhs.size()); 
    }
```

# 字符串和数字

标准库包含了各种函数和类，用于在 C++字符串和数值之间进行转换。

# 将字符串转换为数字

C++标准库包含了名为`stod`和`stoi`的函数，它们将 C++ `string`对象转换为数值（`stod`转换为`double`，`stoi`转换为`integer`）。例如：

```cpp
    double d = stod("10.5"); 
    d *= 4; 
    cout << d << "n"; // 42
```

这将使用值`10.5`初始化浮点变量`d`，然后在计算中使用该值，并将结果打印到控制台。输入字符串可能包含无法转换的字符。如果是这种情况，那么字符串的解析将在那一点结束。您可以提供一个指向`size_t`变量的指针，该变量将被初始化为无法转换的第一个字符的位置：

```cpp
    string str = "49.5 red balloons"; 
    size_t idx = 0; 
    double d = stod(str, &idx); 
    d *= 2; 
    string rest = str.substr(idx); 
    cout << d << rest << "n"; // 99 red balloons
```

在前面的代码中，`idx`变量将被初始化为`4`的值，表示`5`和`r`之间的空格是第一个无法转换为`double`的字符。

# 将数字转换为字符串

`<string>`库提供了各种重载的`to_string`函数，用于将整数类型和浮点类型转换为`string`对象。这个函数不允许你提供任何格式化细节，所以对于整数，你不能指示字符串表示的基数（例如，十六进制），对于浮点数转换，你无法控制选项，比如有效数字的数量。`to_string`函数是一个简单的函数，功能有限。更好的选择是使用流类，如下一节所述。

# 使用流类

您可以使用`cout`对象（`ostream`类的实例）将浮点数和整数打印到控制台，也可以使用`ofstream`的实例将它们打印到文件中。这两个类都将使用成员方法和操作器将数字转换为字符串，并影响输出字符串的格式。同样，`cin`对象（`istream`类的实例）和`ifstream`类可以从格式化流中读取数据。

操纵器是接受流对象引用并返回该引用的函数。标准库有各种全局插入操作符，其参数是流对象的引用和函数指针。适当的插入操作符将调用带有流对象作为参数的函数指针。这意味着操纵器将可以访问并操纵它被插入的流。对于输入流，还有具有函数参数的提取操作符，该参数将调用带有流对象的函数。

C++流的架构意味着在你的代码中调用流接口和获取数据的底层基础设施之间有一个缓冲区。C++标准库提供了将字符串对象作为缓冲区的流类。对于输出流，你可以在项目插入到流中后访问字符串，这意味着字符串将包含根据这些插入操作符格式化的项目。同样，你可以提供一个包含格式化数据的字符串作为输入流的缓冲区，当你使用提取操作符从流中提取数据时，实际上是解析字符串并将字符串的部分转换为数字。

此外，流类有一个`locale`对象，流对象将调用此区域的转换部分，将一个编码的字符序列转换为另一个编码。

# 输出浮点数

`<ios>`库有操纵器可以改变流如何处理数字。默认情况下，输出流将以十进制格式打印浮点数，范围在`0.001`到“100000”之间，对于超出此范围的数字，它将使用带有尾数和指数的科学格式。这种混合格式是`defaultfloat`操纵器的默认行为。如果你总是想使用科学计数法，那么你应该在输出流中插入`scientific`操纵器。

如果你想仅使用十进制格式显示浮点数（即小数点左侧的整数部分和右侧的小数部分），那么可以通过使用`fixed`操纵器修改输出流。可以通过调用`precision`方法来改变小数位数：

```cpp
    double d = 123456789.987654321; 
    cout << d << "n"; 
    cout << fixed; 
    cout << d << "n"; 
    cout.precision(9); 
    cout << d << "n"; 
    cout << scientific; 
    cout << d << "n";
```

上述代码的输出是：

```cpp
 1.23457e+08
 123456789.987654
 123456789.987654328
 1.234567900e+08
```

第一行显示科学计数法用于大数。第二行显示了`fixed`的默认行为，即给出小数点后 6 位小数。通过调用`precision`方法将其更改为给出 9 位小数（可以通过在流中插入“iomanip”库中的`setprecision`操纵器来实现相同的效果）。最后，通过调用`precision`方法将格式切换为科学格式，小数点后有 9 位数字。默认情况下，指数由小写的`e`表示。如果你愿意，可以使用`uppercase`操纵器（和`nouppercase`）将其改为大写。请注意，分数部分存储的方式意味着在固定格式中，小数点后有 9 位数字，我们看到第九位数字是`8`，而不是预期的`1`。

你还可以指定正数是否显示`+`符号；`showpos`操纵器将显示该符号，但默认的`noshowpos`操纵器将不显示该符号。`showpoint`操纵器将确保即使浮点数是整数，也会显示小数点。默认值是`noshowpoint`，这意味着如果没有小数部分，就不会显示小数点。

`setw`操纵器（在“iomanip”头文件中定义）可用于整数和浮点数。实际上，这个操纵器定义了在控制台上打印时下一个（仅下一个）放入流中的项目所占用的最小宽度空间：

```cpp
    double d = 12.345678; 
    cout << fixed; 
    cout << setfill('#'); 
    cout << setw(15) << d << "n";
```

为了说明`setw`操纵器的效果，此代码调用`setfill`操纵器，该操纵器指示应打印井号（`#`）而不是空格。代码的其余部分表示应以固定格式（默认情况下为 6 位小数）在 15 个字符宽的空间中打印数字。结果是：

```cpp
    ######12.345678
```

如果数字为负数（或使用`showpos`），则默认情况下符号将与数字一起显示；如果使用`internal`操纵器（在`<ios>`中定义），则符号将左对齐在为数字设置的空间中：

```cpp
    double d = 12.345678; 
    cout << fixed; 
    cout << showpos << internal; 
    cout << setfill('#'); 
    cout << setw(15) << d << "n";
```

上述代码的结果如下：

```cpp
    +#####12.345678
```

请注意，空格右侧的`+`符号由井号表示。

`setw`操纵器通常用于允许您以格式化的列输出数据表：

```cpp
    vector<pair<string, double>> table 
    { { "one",0 },{ "two",0 },{ "three",0 },{ "four",0 } }; 

    double d = 0.1; 
    for (pair<string,double>& p : table) 
    { 
        p.second = d / 17.0; 
        d += 0.1; 
    } 

    cout << fixed << setprecision(6); 

    for (pair<string, double> p : table) 
    { 
        cout << setw(6)  << p.first << setw(10) << p.second << "n"; 
    }
```

这将使用字符串和数字填充`vector`对。`vector`用字符串值和零初始化，然后在`for`循环中更改浮点数（这里实际计算无关紧要；重点是创建一些具有多个小数位的数字）。数据以两列打印出来，数字以 6 位小数打印。这意味着，包括前导零和小数点，每个数字将占用 8 个空间。文本列被指定为 6 个字符宽，数字列被指定为 10 个字符宽。默认情况下，当您指定列宽时，输出将右对齐，这意味着每个数字前面有两个空格，文本根据字符串的长度进行填充。输出如下：

```cpp
 one  0.005882
 two  0.011765
 three  0.017647
 four  0.023529
```

如果要使列中的项目左对齐，则可以使用`left`操纵器。这将影响所有列，直到使用`right`操纵器将对齐方式更改为右对齐为止：

```cpp
    cout << fixed << setprecision(6) << left;
```

这将输出：

```cpp
 one   0.005882
 two   0.011765
 three 0.017647
 four  0.023529
```

如果要为两列设置不同的对齐方式，则需要在打印值之前设置对齐方式。例如，要左对齐文本并右对齐数字，请使用以下代码：

```cpp
    for (pair<string, double> p : table) 
    { 
        cout << setw(6) << left << p.first  
            << setw(10) << right << p.second << "n"; 
    }
```

上述代码的结果如下：

```cpp
 one     0.005882
 two     0.011765
 three   0.017647
 four    0.023529
```

# 输出整数

整数也可以使用`setw`和`setfill`方法以列的形式打印。您可以插入操纵器以使用八进制（`oct`），十进制（`dec`）和十六进制（`hex`）打印整数。（您还可以使用`setbase`操纵器并传递要使用的基数，但允许的唯一值是 8、10 和 16。）可以使用`showbase`和`noshowbase`操纵器打印带有指示基数的数字（八进制前缀为`0`或十六进制前缀为`0x`）或不带。如果使用`hex`，则大于`9`的数字是字母`a`到`f`，默认情况下这些是小写的。如果您希望这些为大写，则可以使用`uppercase`操纵器（并使用`nouppercase`操纵器转换为小写）。

# 输出时间和金钱

`<iomanip>`中的`put_time`函数传递了一个初始化为时间和日期的`tm`结构和一个格式字符串。该函数返回`_Timeobj`类的一个实例。顾名思义，您实际上不应该创建此类的变量；相反，应该使用该函数将具有特定格式的时间/日期插入流中。有一个插入运算符将打印`_Timeobj`对象。该函数的使用方式如下：

```cpp
    time_t t = time(nullptr); 
    tm *pt = localtime(&t); 
    cout << put_time(pt, "time = %X date = %x") << "n";
```

这将输出：

```cpp
    time = 20:08:04 date = 01/02/17
```

该函数将使用流中的区域设置，因此如果将区域设置为流中，然后调用`put_time`，则将使用区域设置的时间/日期格式化规则和格式字符串。格式字符串使用`strftime`的格式标记：

```cpp
    time_t t = time(nullptr); 
    tm *pt = localtime(&t); 
    cout << put_time(pt, "month = %B day = %A") << "n"; 
    cout.imbue(locale("french")); 
    cout << put_time(pt, "month = %B day = %A") << "n";
```

上述代码的输出如下：

```cpp
 month = March day = Thursday
 month = mars day = jeudi
```

类似地，`put_money`函数返回一个`_Monobj`对象。同样，这只是一个包含您传递给此函数的参数的容器，您不应该使用此类的实例。相反，您应该将此函数插入到输出流中。实际工作发生在插入运算符中，该运算符获取当前区域设置上的货币 facet，使用它来将数字格式化为适当数量的小数位，并确定小数点字符；如果使用了千位分隔符，则在适当位置插入它之前。

```cpp
    Cout << showbase; 
    cout.imbue(locale("German")); 
    cout << "German" << "n"; 
    cout << put_money(109900, false) << "n"; 
    cout << put_money("1099", true) << "n"; 
    cout.imbue(locale("American")); 
    cout << "American" << "n"; 
    cout << put_money(109900, false) << "n"; 
    cout << put_money("1099", true) << "n";
```

前面代码的输出是：

```cpp
 German
 1.099,00 euros
 EUR10,99
 American
 $1,099.00
 USD10.99
```

您可以使用`double`或字符串提供欧分或分的数字，并且`put_money`函数将使用适当的小数点（德国为`,`，美国为`.`）和适当的千位分隔符（德国为`.`，美国为`,`）格式化欧元或美元的数字。将`showbase`操作器插入到输出流中意味着`put_money`函数将显示货币符号，否则只会显示格式化的数字。`put_money`函数的第二个参数指定使用货币字符（`false`）还是国际符号（`true`）。

# 使用流将数字转换为字符串

流缓冲区类负责从适当的源（文件、控制台等）获取字符并写入字符，并且从`<streambuf>`中的抽象类`basic_streambuf`派生。此基类定义了两个虚拟方法，`overflow`和`underflow`，派生类重写这些方法以从与派生类关联的设备中写入和读取字符（分别）。流缓冲区类执行将项目放入流中的基本操作，由于缓冲区处理字符，因此该类使用字符类型和字符特征的参数进行模板化。

顾名思义，如果使用`basic_stringbuf`，则流缓冲区将是一个字符串，因此读取字符的源和写入字符的目的地是该字符串。如果使用此类为流对象提供缓冲区，这意味着您可以使用为流编写的插入或提取运算符，将格式化的数据写入或从字符串中读取。`basic_stringbuf`缓冲区是可扩展的，因此当您在流中插入项目时，缓冲区将适当地扩展。有`typedef`，其中缓冲区是`string`（`stringbuf`）或`wstring`（`wstringbuf`）。

例如，假设您有一个已定义的类，并且还定义了插入运算符，以便您可以使用`cout`对象将值打印到控制台：

```cpp
    struct point 
    { 
        double x = 0.0, y = 0.0; 
        point(){} 
        point(double _x, double _y) : x(_x), y(_y) {} 
    }; 
```

```cpp

    ostream& operator<<(ostream& out, const point& p) 
    { 
        out << "(" << p.x << "," << p.y << ")"; 
        return out; 
    }
```

使用`cout`对象很简单--考虑以下代码片段：

```cpp
    point p(10.0, -5.0); 
    cout << p << "n";         // (10,-5)
```

您可以使用`stringbuf`将格式化的输出定向到字符串而不是控制台：

```cpp
    stringbuf buffer;  
    ostream out(&buffer); 
    out << p; 
    string str = buffer.str(); // contains (10,-5)
```

由于流对象处理格式，这意味着您可以插入任何数据类型，只要有插入运算符，并且可以使用任何`ostream`格式化方法和任何操作器。所有这些方法和操作器的格式化输出将插入到缓冲区中的字符串对象中。

另一个选项是使用`<sstream>`中的`basic_ostringstream`类。该类是基于用作缓冲区的字符串的字符类型的模板（因此`string`版本是`ostringstream`）。它派生自`ostream`类，因此您可以在任何使用`ostream`对象的地方使用实例。格式化的结果可以通过`str`方法访问：

```cpp
    ostringstream os; 
    os << hex; 
    os << 42; 
    cout << "The value is: " << os.str() << "n";
```

此代码以十六进制（`2a`）获取`42`的值；这是通过在流中插入`hex`操作器，然后插入整数来实现的。通过调用`str`方法获取格式化的字符串。

# 使用流从字符串中读取数字

`cin`对象是`istream`类的一个实例（在`<istream>`库中），可以从控制台输入字符并将其转换为你指定的数字形式。`ifstream`类（在`<ifstream>`库中）也允许你从文件中输入字符并将其转换为数字形式。与输出流一样，你可以使用流类与字符串缓冲区，以便你可以从字符串对象转换为数字值。

`basic_istringstream`类（在`<sstream>`库中）是从`basic_istream`类派生的，所以你可以创建流对象，并从这些对象中提取项目（数字和字符串）。该类在字符串对象上提供了这个流接口（`typedef`关键字`istringstream`基于`string`，`wistringstream`基于`wstring`）。当你构造这个类的对象时，你用一个包含数字的`string`初始化对象，然后你使用`>>`操作符从基本内置类型中提取对象，就像你使用`cin`从控制台提取这些项目一样。

需要重申的是，提取操作符将空白字符视为流中项目之间的分隔符，因此它们将忽略所有前导空白字符，读取非空白字符直到下一个空白字符，并尝试将这个子字符串转换为适当的类型，如下所示：

```cpp
    istringstream ss("-1.0e-6"); 
    double d; 
    ss >> d;
```

这将用值`-1e-6`初始化变量`d`。与`cin`一样，你必须知道流中项目的格式；所以，如果在前面的例子中，你尝试从字符串中提取一个`double`而不是一个整数，当遇到小数点时，对象将停止提取字符。如果字符串的一部分没有被转换，你可以将剩下的部分提取到一个字符串对象中：

```cpp
    istringstream ss("-1.0e-6"); 
    int i; 
    ss >> i; 
    string str; 
    ss >> str; 
    cout << "extracted " << i << " remainder " << str << "n";
```

这将在控制台上打印以下内容：

```cpp
    extracted -1 remainder .0e-6
```

如果字符串中有多个数字，你可以通过多次调用`>>`操作符来提取这些数字。流还支持一些操作器。例如，如果字符串中的数字是以`hex`格式，你可以使用`hex`操作器通知流，如下所示：

```cpp
    istringstream ss("0xff"); 
    int i; 
    ss >> hex; 
    ss >> i;
```

这表示字符串中的数字是十六进制格式，变量`i`将被初始化为 255。如果字符串包含非数字值，那么流对象仍然会尝试将字符串转换为适当的格式。在下面的片段中，你可以通过调用`fail`函数测试这样的提取是否失败：

```cpp
    istringstream ss("Paul was born in 1942"); 
    int year; 
    ss >> year; 
    if (ss.fail()) cout << "failed to read number" << "n";
```

如果你知道字符串包含文本，你可以将它提取到字符串对象中，但请记住空白字符被视为分隔符：

```cpp
    istringstream ss("Paul was born in 1942"); 
    string str; 
    ss >> str >> str >> str >> str; 
    int year; 
    ss >> year;
```

在这里，数字之前有四个单词，所以代码会读取一个`string`四次。如果你不知道数字在字符串中的位置，但你知道字符串中有一个数字，你可以移动内部缓冲指针，直到它指向一个数字：

```cpp
    istringstream ss("Paul was born in 1942"); 
    string str;    
    while (ss.eof() && !(isdigit(ss.peek()))) ss.get(); 
    int year; 
    ss >> year; 
    if (!ss.fail()) cout << "the year was " << year << "n";
```

`peek`方法返回当前位置的字符，但不移动缓冲指针。这段代码检查这个字符是否是一个数字，如果不是，就通过调用`get`方法移动内部缓冲指针。（这段代码测试`eof`方法以确保在缓冲结束后没有尝试读取字符。）如果你知道数字从哪里开始，你可以调用`seekg`方法将内部缓冲指针移动到指定位置。

`<istream>`库有一个叫做`ws`的操作器，可以从流中移除空白字符。回想一下我们之前说过，没有函数可以从字符串中移除空白字符。这是因为`ws`操作器从*流*中移除空白字符，而不是从*字符串*中移除，但是由于你可以使用字符串作为流的缓冲，这意味着你可以间接地使用这个函数从字符串中移除空白字符：

```cpp
    string str = "  hello  "; 
    cout << "|" << str1 << "|n"; // |  hello  | 
    istringstream ss(str); 
    ss >> ws; 
    string str1; 
    ss >> str1; 
    ut << "|" << str1 << "|n";   // |hello|
```

`ws`函数本质上是遍历输入流中的项目，并在遇到非空白字符时返回。如果流是文件或控制台流，则`ws`函数将从这些流中读取字符；在这种情况下，缓冲区由已分配的字符串提供，因此它会跳过字符串开头的空格。请注意，流类将后续空格视为流中值之间的分隔符，因此在这个例子中，流将从缓冲区中读取字符，直到遇到空格，并且本质上会*左-**和右-修剪*字符串。但是，这不一定是您想要的。如果您有一个由空格填充的字符串，这段代码只会提供第一个单词。

`<iomanip>`库中的`get_money`和`get_time`操作器允许您使用货币和时间区域设置从字符串中提取货币和时间：

```cpp
    tm indpday = {}; 
    string str = "4/7/17"; 
    istringstream ss(str); 
    ss.imbue(locale("french")); 
    ss >> get_time(&indpday, "%x"); 
    if (!ss.fail())  
    { 
       cout.imbue(locale("american")); 
       cout << put_time(&indpday, "%x") << "n";  
    }
```

在上述代码中，流首先用法国格式（日/月/年）的日期初始化，然后使用区域设置的标准日期表示提取日期。日期被解析为`tm`结构，然后使用`put_time`在美国区域设置中以标准日期表示打印出来。结果是：

```cpp
    7/4/2017
```

# 使用正则表达式

正则表达式是文本模式，可以被正则表达式解析器用来搜索匹配模式的文本字符串，并在必要时用其他文本替换匹配的项目。

# 定义正则表达式

**正则表达式**（**regex**）由定义模式的字符组成。表达式包含对解析器有意义的特殊符号，如果您想在表达式中的搜索模式中使用这些符号，可以用反斜杠（`\`）对它们进行转义。您的代码通常会将表达式作为`string`对象传递给`regex`类的实例作为构造函数参数。然后将该对象传递给`<regex>`中的函数，这些函数将使用表达式来解析文本以匹配模式的序列。

下表总结了`regex`类可以匹配的*一些*模式。

| **模式** | **解释** | **示例** |
| --- | --- | --- |
| literals | 匹配确切的字符 | `li` 匹配 `flip` `lip` `plier` |
| [group] | 匹配组中的单个字符 | `[at]` 匹配 `cat`, `cat`, `top`, `pear` |
| [^group] | 匹配不在组中的单个字符 | `[^at]` 匹配 **c**at, t**o**p, to**p**, **p**ear, p**e**ar, pea**r** |
| [first-last] | 匹配范围`first`到`last`中的任何字符 | `[0-9]` 匹配数字 **1**02, 1**0**2, 10**2** |
| {n} | 元素精确匹配 n 次 | **91{2}** 匹配 **911** |
| {n,} | 元素匹配 n 次或更多次 | `wel{1,}` 匹配 `well` 和 **wel**come |
| {n,m} | 元素匹配 n 到 m 次 | `9{2,4}` 匹配 `99`, `999`, `9999`, `9999`9 但不匹配 9 |
| . | 通配符，除了`n`之外的任何字符 | `a.e` 匹配 `ate` 和 `are` |
| * | 元素匹配零次或多次 | `d*.d` 匹配 `.1`, `0.1`, `10.1` 但不匹配 10 |
| + | 元素匹配一次或多次 | `d*.d` 匹配 `0.1`, `10.1` 但不匹配 10 或 .1 |
| ? | 元素匹配零次或一次 | `tr?ap` 匹配 `trap` 和 `tap` |
| &#124; | 匹配由`&#124;`分隔的元素中的任何一个 | `th(e&#124;is&#124;at)` 匹配 `the`, `this`, `that` |
| [[:class:]] | 匹配字符类 | `[[:upper:]]` 匹配大写字符：`I` am `R`ichard |
| n | 匹配换行符 |  |
| s | 匹配任何单个空格 |  |
| d | 匹配任何单个数字 | `d` 是 `[0-9]` |
| w | 匹配单词中的字符（大写和小写字符） |  |
| b | 匹配字母数字字符和非字母数字字符之间的边界 | `d{2}b` 匹配 9`99` 和 99`99 bd{2}` 匹配 `99`9 和 `99`99 |
| $ | 行的结尾 | `s$`匹配一行末尾的单个空格 |
| ^ | 行的开头 | `^d`匹配如果一行以数字开头 |

你可以使用正则表达式来定义一个要匹配的模式--Visual C++编辑器允许你在搜索对话框中这样做（这是一个很好的测试平台来开发你的表达式）。

定义一个匹配模式要比定义一个*不*匹配的模式容易得多。例如，表达式`w+b<w+>`将匹配字符串`"vector<int>"`，因为它有一个或多个单词字符，后面跟着一个非单词字符（`<`），然后是一个或多个单词字符，最后是`>`。这个模式不会匹配字符串`"#include <regex>"`，因为`include`后面有一个空格，`b`表示字母数字字符和非字母数字字符之间有一个边界。

表格中的`th(e|is|at)`示例表明，当你想提供替代方案时，你可以使用括号来分组模式。然而，括号还有另一个用途--它们允许你捕获组。因此，如果你想执行替换操作，你可以搜索一个模式作为一个组，然后稍后引用该组作为一个命名的子组（例如，搜索`(Joe)`，这样你就可以用`Tom`替换`Joe`）。你还可以在表达式中引用由括号指定的子表达式（称为反向引用）：

```cpp
    ([A-Za-z]+) +1
```

这个表达式说：*搜索包含一个或多个字符在 a 到 z 和 A 到 Z 范围内的单词；这个单词叫 1，所以找到它出现两次并且中间有一个空格*。

# 标准库类

要进行匹配或替换，你必须创建一个正则表达式对象。这是一个`basic_regex`类的对象，它有字符类型和正则表达式特征类的模板参数。这个类有两个`typedef`：`regex`表示`char`，`wregex`表示宽字符，它们的特征由`regex_traits`和`wregex_traits`类描述。

特征类确定了正则表达式类如何解析表达式。例如，回想一下之前的文本，你可以用`w`表示一个单词，`d`表示一个数字，`s`表示一个空格。`[[::]]`语法允许你使用更具描述性的名称来表示字符类：`alnum`，`digit`，`lower`等等。由于这些是依赖于字符集的文本序列，特征类将有适当的代码来测试表达式是否使用了支持的字符类。

适当的正则表达式类将解析表达式，以便`<regex>`库中的函数使用表达式来识别文本中的模式：

```cpp
    regex rx("([A-Za-z]+) +1");
```

这搜索重复的单词使用了反向引用。请注意，正则表达式使用`1`表示反向引用，但在字符串中反斜杠必须转义（`\`）。如果你使用字符类如`s`和`d`，那么你将需要做很多转义。相反，你可以使用原始字符串（`R"()"`），但要记住引号内的第一组括号是原始字符串的语法的一部分，不是正则表达式组的一部分：

```cpp
    regex rx(R"(([A-Za-z]+) +1)");
```

哪种更易读完全取决于你；两者都在双引号内引入了额外的字符，这可能会让人快速浏览时对正则表达式匹配的内容感到困惑。

请记住，正则表达式本质上是一个程序，因此`regex`解析器将确定该表达式是否有效，如果无效，对象、构造函数将抛出`regex_error`类型的异常。异常处理将在下一章中解释，但重要的是要指出，如果异常没有被捕获，将导致应用在运行时中止。异常的`what`方法将返回错误的基本描述，`code`方法将返回`regex_constants`命名空间中`error_type`枚举中的常量之一。没有指示错误发生在表达式的哪个位置。您应该在外部工具（例如 Visual C++搜索）中彻底测试您的表达式。

构造函数可以使用字符串（C 或 C++）或一对迭代器来调用字符串（或其他容器）中一系列字符的范围，或者可以传递一个初始化列表，其中列表中的每个项都是一个字符。正则表达式有各种不同的语言风格；`basic_regex`类的默认风格是**ECMAScript**。如果您想要不同的语言（基本 POSIX、扩展 POSIX、awk、grep 或 egrep），可以传递`regex_constants`命名空间中`syntax_option_type`枚举中定义的常量之一（也可以作为`basic_regex`类中定义的常量的副本）作为构造函数参数。

您只能指定一种语言风格，但您可以将其与其他`syntax_option_type`常量结合使用：`icase`指定不区分大小写，`collate`使用匹配中的区域设置，`nosubs`表示您不想捕获组，`optimize`优化匹配。

该类使用`getloc`方法获取解析器使用的区域设置，并使用`imbue`重置区域设置。如果您`imbue`一个区域设置，那么在使用`assign`方法重置之前，您将无法使用`regex`对象进行任何匹配。这意味着有两种使用`regex`对象的方法。如果要使用当前区域设置，则将正则表达式传递给构造函数：如果要使用不同的区域设置，则使用默认构造函数创建一个空的`regex`对象，然后使用`imbue`调用区域设置，并使用`assign`方法传递正则表达式。一旦解析了正则表达式，就可以调用`mark_count`方法获取表达式中捕获组的数量（假设您没有使用`nosubs`）。

# 匹配表达式

一旦构造了一个`regex`对象，您可以将其传递给`<regex>`库中的方法，以在字符串中搜索模式。`regex_match`函数传入一个字符串（C 或 C++）或容器中一系列字符的迭代器以及一个构造的`regex`对象。在其最简单的形式中，该函数只有在有精确匹配时才会返回`true`，也就是说，表达式完全匹配搜索字符串：

```cpp
    regex rx("[at]"); // search for either a or t 
    cout << boolalpha; 
    cout << regex_match("a", rx) << "n";  // true 
    cout << regex_match("a", rx) << "n";  // true 
    cout << regex_match("at", rx) << "n"; // false
```

在前面的代码中，搜索表达式是给定范围内的单个字符（`a`或`t`），因此前两个`regex_match`调用返回`true`，因为搜索的字符串是一个字符。最后一个调用返回`false`，因为匹配与搜索的字符串不同。如果在正则表达式中删除`[]`，那么只有第三个调用返回`true`，因为您要查找确切的字符串`at`。如果正则表达式是`[at]+`，这样您要查找一个或多个字符`a`和`t`，那么所有三个调用都返回`true`。您可以通过传递`match_flag_type`枚举中的一个或多个常量来改变匹配的方式。

如果将`match_results`对象的引用传递给此函数，那么在搜索之后，该对象将包含有关匹配位置和字符串的信息。`match_results`对象是`sub_match`对象的容器。如果函数成功，这意味着整个搜索字符串与表达式匹配，在这种情况下，返回的第一个`sub_match`项将是整个搜索字符串。如果表达式有子组（用括号标识的模式），那么这些子组将是`match_results`对象中的其他`sub_match`对象。

```cpp
    string str("trumpet"); 
    regex rx("(trump)(.*)"); 
    match_results<string::const_iterator> sm; 
    if (regex_match(str, sm, rx)) 
    { 
        cout << "the matches were: "; 
        for (unsigned i = 0; i < sm.size(); ++i)  
        { 
            cout << "[" << sm[i] << "," << sm.position(i) << "] "; 
        } 
        cout << "n"; 
    } // the matches were: [trumpet,0] [trump,0] [et,5]
```

在这里，表达式是字面量`trump`后面跟着任意数量的字符。整个字符串与此表达式匹配，并且有两个子组：字面字符串`trump`和在`trump`被移除后剩下的任何内容。

`match_results`类和`sub_match`类都是基于用于指示匹配项的迭代器类型进行模板化的。有`typedef`调用`cmatch`和`wcmatch`，其中模板参数是`const char*`和`const wchar_t*`，`smatch`和`wsmatch`，其中参数是在`string`和`wstring`对象中使用的迭代器，分别（类似地，还有子匹配类：`csub_match`，`wcsub_match`，`ssub_match`和`wssub_match`）。

`regex_match`函数可能会非常严格，因为它寻找模式和搜索字符串之间的精确匹配。`regex_search`函数更加灵活，因为它返回`true`，如果搜索字符串中有与表达式匹配的子字符串。请注意，即使在搜索字符串中有多个匹配项，`regex_search`函数也只会找到第一个。如果要解析字符串，必须多次调用该函数，直到它指示没有更多的匹配项为止。这就是具有对搜索字符串的迭代器访问的重载变得有用的地方：

```cpp
    regex rx("bd{2}b"); 
    smatch mr; 
    string str = "1 4 10 42 100 999"; 
    string::const_iterator cit = str.begin(); 
    while (regex_search(cit, str.cend(), mr, rx)) 
    { 
        cout << mr[0] << "n"; 
        cit += mr.position() + mr.length(); 
    }
```

在这里，表达式将匹配由空格包围的 2 位数（`d{2}`），两个`b`模式表示匹配项之前和之后的边界。循环从指向字符串开头的迭代器开始，当找到匹配项时，该迭代器将增加到该位置，然后增加匹配项的长度。`regex_iterator`对象，稍后解释，包装了这种行为。

`match_results`类为包含的`sub_match`对象提供了迭代器访问，因此您可以使用范围`for`。最初，似乎容器的工作方式有些奇怪，因为它知道`sub_match`对象在搜索字符串中的位置（通过`position`方法，该方法接受子匹配对象的索引），但`sub_match`对象似乎只知道它所引用的字符串。然而，仔细检查`sub_match`类后，可以发现它是从`pair`派生而来的，其中两个参数都是字符串迭代器。这意味着`sub_match`对象具有指定原始字符串中子字符串范围的迭代器。`match_result`对象知道原始字符串的起始位置，并且可以使用`sub_match.first`迭代器来确定子字符串的起始字符位置。

`match_result`对象具有`[]`运算符（和`str`方法），返回指定组的子字符串；这将是使用原始字符串中字符范围的迭代器构造的字符串。`prefix`方法返回匹配项之前的字符串，`suffix`方法返回匹配项之后的字符串。因此，在前面的代码中，第一个匹配项将是`10`，前缀将是`1 4`，后缀将是`42 100 999`。相比之下，如果访问`sub_match`对象本身，它只知道它的长度和字符串，这是通过调用`str`方法获得的。

`match_result`对象还可以通过`format`方法返回结果。这需要一个格式字符串，其中通过`$`符号标识的编号占位符标识匹配的组（`$1`、`$2`等）。输出可以是流，也可以从方法中作为字符串返回：

```cpp
    string str("trumpet"); 
    regex rx("(trump)(.*)"); 
    match_results<string::const_iterator> sm; 
    if (regex_match(str, sm, rx)) 
    { 
        string fmt = "Results: [$1] [$2]"; 
        cout << sm.format(fmt) << "n"; 
    } // Results: [trump] [et]
```

使用`regex_match`或`regex_search`，您可以使用括号来标识子组。如果模式匹配，则可以使用适当的`match_results`对象通过引用获取这些子组。如前所示，`match_results`对象是`sub_match`对象的容器。子匹配可以使用`<`、`!=`、`==`、`<=`、`>`和`>=`运算符进行比较，这些运算符比较迭代器指向的项目（即子字符串）。此外，`sub_match`对象可以插入到流中。

# 使用迭代器

该库还为正则表达式提供了一个迭代器类，它提供了一种不同的解析字符串的方式。由于该类涉及字符串的比较，因此它使用元素类型和特性进行模板化。该类需要迭代字符串，因此第一个模板参数是字符串迭代器类型，元素和特性类型可以从中推导出来。`regex_iterator`类是一个前向迭代器，因此它具有`++`运算符，并且提供了一个`*`运算符，用于访问`match_result`对象。在先前的代码中，您看到`match_result`对象被传递给`regex_match`和`regex_search`函数，它们用于包含它们的结果。这引发了一个问题，即通过`regex_iterator`访问的`match_result`对象是由什么代码填充的。答案在于迭代器的`++`运算符：

```cpp
    string str = "the cat sat on the mat in the bathroom"; 
    regex rx("(b(.at)([^ ]*)"); 
    regex_iterator<string::iterator> next(str.begin(), str.end(), rx); 
    regex_iterator<string::iterator> end; 

    for (; next != end; ++next) 
    { 
        cout << next->position() << " " << next->str() << ", "; 
    } 
    cout << "n"; 
    // 4 cat, 8 sat, 19 mat, 30 bathroom
```

在这段代码中，搜索包含第二个和第三个字母为`at`的单词的字符串。`b`表示模式必须位于单词的开头（`.`表示单词可以以任何字母开头）。这三个字符周围有一个捕获组，另一个捕获组包含一个或多个非空格字符。

迭代器对象`next`是使用要搜索的字符串和`regex`对象的迭代器构造的。`++`运算符本质上调用`regex_search`函数，同时保持执行下一次搜索的位置。如果搜索未找到模式，则运算符将返回**序列结束**迭代器，这是由默认构造函数创建的迭代器（在此代码中为`end`对象）。此代码打印出完整的匹配，因为我们使用`str`方法的默认参数（`0`）。如果您想要实际匹配的子字符串，请使用`str(1)`，结果将是：

```cpp
    4 cat, 8 sat, 19 mat, 30 bat
```

由于`*`（和`->`）运算符可以访问`match_result`对象，因此您还可以访问`prefix`方法以获取匹配之前的字符串，`suffix`方法将返回匹配之后的字符串。

`regex_iterator`类允许您迭代匹配的子字符串，而`regex_token_iterator`进一步提供了对所有子匹配的访问。在使用中，这个类与`regex_iterator`相同，只是在构造时不同。`regex_token_iterator`构造函数有一个参数，用于指示您希望通过`*`运算符访问哪个子匹配。值为`-1`表示您想要前缀，值为`0`表示您想要整个匹配，值为`1`或更高表示您想要编号的子匹配。如果愿意，您可以传递一个带有您想要的子匹配类型的`int vector`或 C 数组：

```cpp
    using iter = regex_token_iterator<string::iterator>; 
    string str = "the cat sat on the mat in the bathroom"; 
    regex rx("b(.at)([^ ]*)");  
    iter next, end; 

    // get the text between the matches 
    next = iter(str.begin(), str.end(), rx, -1); 
    for (; next != end; ++next) cout << next->str() << ", "; 
    cout << "n"; 
    // the ,  ,  on the ,  in the , 

    // get the complete match 
    next = iter(str.begin(), str.end(), rx, 0); 
    for (; next != end; ++next) cout << next->str() << ", "; 
    cout << "n"; 
    // cat, sat, mat, bathroom, 

    // get the sub match 1 
    next = iter(str.begin(), str.end(), rx, 1); 
    for (; next != end; ++next) cout << next->str() << ", "; 
    cout << "n"; 
    // cat, sat, mat, bat, 

    // get the sub match 2 
    next = iter(str.begin(), str.end(), rx, 2); 
    for (; next != end; ++next) cout << next->str() << ", "; 
    cout << "n"; 
    // , , , hroom,
```

# 替换字符串

`regex_replace` 方法与其他方法类似，它接受一个字符串（C 字符串或 C++ `string` 对象，或者字符范围的迭代器）、一个 `regex` 对象和可选标志。此外，该函数有一个格式字符串，并返回一个 `string`。格式字符串基本上是传递给每个匹配结果的 `results_match` 对象的 `format` 方法的结果，用于正则表达式的匹配。然后，这个格式化的字符串被用作相应匹配的子字符串的替换。如果没有匹配，那么将返回搜索字符串的副本。

```cpp
    string str = "use the list<int> class in the example"; 
    regex rx("b(list)(<w*> )"); 
    string result = regex_replace(str, rx, "vector$2"); 
    cout << result << "n"; // use the vector<int> class in the example
```

在上述代码中，我们说整个匹配的字符串（应该是由一些文本后跟 `>` 和空格组成的 `list<`）应该被替换为 `vector,` 后跟第二个子匹配（`<` 后跟一些文本后跟 `>` 和空格）。结果是 `list<int>` 将被替换为 `vector<int>`。

# 使用字符串

该示例将作为文本文件读取和处理电子邮件。互联网消息格式的电子邮件将分为两部分：头部和消息主体。这是简单的处理，因此不会尝试处理 MIME 电子邮件主体格式（尽管此代码可以用作该处理的起点）。电子邮件主体将在第一个空行之后开始，互联网标准规定行不应超过 78 个字符。如果超过，它们不得超过 998 个字符。这意味着换行符（回车、换行对）用于保持此规则，并且段落的结束由空行表示。

头部更加复杂。在最简单的形式中，头部在单行上，并且采用 `name:value` 的形式。头部名称与头部值之间由冒号分隔。头部可以使用称为折叠空格的格式分成多行，其中将分割头部的换行符放置在空格（空格、制表符等）之前。这意味着以空格开头的行是前一行上头部的继续。头部通常包含由分号分隔的 `name=value` 对，因此能够分隔这些子项是有用的。有时这些子项没有值，也就是说，将有一个由分号终止的子项。

该示例将将电子邮件作为一系列字符串，并根据这些规则创建一个包含头部集合和包含主体的字符串的对象。

# 创建项目

为项目创建一个文件夹，并创建一个名为 `email_parser.cpp` 的 C++ 文件。由于此应用程序将读取文件并处理字符串，因此添加适当的库包含并添加代码以从命令行获取文件名：

```cpp
    #include <iostream> 
    #include <fstream> 
    #include <string> 

    using namespace std; 

    void usage() 
    { 
        cout << "usage: email_parser file" << "n"; 
        cout << "where file is the path to a file" << "n"; 
    } 

    int main(int argc, char *argv[]) 
    { 
        if (argc <= 1) 
        { 
            usage(); 
            return 1; 
        } 

        ifstream stm; 
        stm.open(argv[1], ios_base::in); 
        if (!stm.is_open()) 
        { 
            usage(); 
            cout << "cannot open " << argv[1] << "n"; 
            return 1; 
        } 

        return 0; 
    }
```

头部将有一个名称和一个主体。主体可以是单个字符串，也可以是一个或多个子项。创建一个表示头部主体的类，并暂时将其视为单行。在 `usage` 函数之前添加以下类：

```cpp
    class header_body 
    { 
        string body; 
    public: 
        header_body() = default; 
        header_body(const string& b) : body(b) {} 
        string get_body() const { return body; } 
    };
```

这只是将该类封装在一个 `string` 周围；稍后我们将添加代码来分离 `body` 数据成员中的子项。现在创建一个表示电子邮件的类。在 `header_body` 类之后添加以下代码：

```cpp
    class email 
    { 
        using iter = vector<pair<string, header_body>>::iterator; 
        vector<pair<string, header_body>> headers; 
        string body; 

    public: 
        email() : body("") {} 

        // accessors 
        string get_body() const { return body; } 
        string get_headers() const; 
        iter begin() { return headers.begin(); } 
        iter end() { return headers.end(); } 

        // two stage construction 
        void parse(istream& fin); 
    private: 
        void process_headers(const vector<string>& lines); 
    };
```

`headers` 数据成员保存头部作为名称/值对。项目存储在 `vector` 中而不是 `map` 中，因为当电子邮件从邮件服务器传递到邮件服务器时，每个服务器可能会添加已存在于电子邮件中的头部，因此头部会重复。我们可以使用 `multimap`，但是我们将失去头部的顺序，因为 `multimap` 将以有助于搜索项目的顺序存储项目。

`vector` 保持容器中插入的项目的顺序，因此我们将按顺序解析电子邮件，这意味着 `headers` 数据成员将按照电子邮件中的顺序包含头部项目。添加适当的包含以便您可以使用 `vector` 类。

正文和标题有一个单独的字符串访问器。此外，还有访问器从 `headers` 数据成员返回迭代器，以便外部代码可以遍历 `headers` 数据成员（此类的完整实现将具有允许您按名称搜索标题的访问器，但在此示例的目的上，只允许迭代）。

该类支持两阶段构造，其中大部分工作是通过将输入流传递给 `parse` 方法来完成的。`parse` 方法将电子邮件作为 `vector` 对象中的一系列行读入，并调用一个私有函数 `process_headers` 来将这些行解释为标题。

`get_headers` 方法很简单：它只是遍历标题，并以 `name: value` 的格式将一个标题放在每一行中。添加内联函数：

```cpp
    string get_headers() const 
    { 
        string all = ""; 
        for (auto a : headers) 
        { 
            all += a.first + ": " + a.second.get_body(); 
            all += "n"; 
        } 
        return all; 
    }
```

接下来，您需要从文件中读取电子邮件并提取正文和标题。 `main` 函数已经有打开文件的代码，所以创建一个 `email` 对象，并将文件的 `ifstream` 对象传递给 `parse` 方法。现在使用访问器打印出解析后的电子邮件。在 `main` 函数的末尾添加以下内容：

```cpp
 email eml; eml.parse(stm); cout << eml.get_headers(); cout << "n"; cout << eml.get_body() << "n"; 

        return 0; 
    }
```

在 `email` 类声明之后，添加 `parse` 函数的定义：

```cpp
    void email::parse(istream& fin) 
    { 
        string line; 
        vector<string> headerLines; 
        while (getline(fin, line)) 
        { 
            if (line.empty()) 
            { 
                // end of headers 
                break; 
            } 
            headerLines.push_back(line); 
        } 

        process_headers(headerLines); 

        while (getline(fin, line)) 
        { 
            if (line.empty()) body.append("n"); 
            else body.append(line); 
        } 
    }
```

这个方法很简单：它反复调用 `<string>` 库中的 `getline` 函数来读取一个 `string`，直到检测到换行符。在方法的前半部分，字符串存储在一个 `vector` 中，然后传递给 `process_headers` 方法。如果读取的字符串为空，意味着已经读取了空行--在这种情况下，所有标题都已经读取。在方法的后半部分，读取电子邮件的正文。

`getline` 函数将剥离用于将电子邮件格式化为 78 个字符行长度的换行符，因此循环只是将行附加为一个字符串。如果读取了空行，则表示段落结束，因此将换行符添加到正文字符串中。

在 `parse` 方法之后，添加 `process_headers` 方法：

```cpp
    void email::process_headers(const vector<string>& lines) 
    { 
        string header = ""; 
        string body = ""; 
        for (string line : lines) 
        { 
            if (isspace(line[0])) body.append(line); 
            else 
            { 
                if (!header.empty()) 
                { 
                    headers.push_back(make_pair(header, body)); 
                    header.clear(); 
                    body.clear(); 
                } 

                size_t pos = line.find(':'); 
                header = line.substr(0, pos); 
                pos++; 
                while (isspace(line[pos])) pos++; 
                body = line.substr(pos); 
            } 
        } 

        if (!header.empty()) 
        { 
            headers.push_back(make_pair(header, body)); 
        } 
    }
```

此代码遍历集合中的每一行，当它有一个完整的标题时，将字符串拆分为名称/正文对。在循环内，第一行测试第一个字符是否为空格；如果不是，则检查 `header` 变量是否有值；如果有，则将名称/正文对存储在类 `headers` 数据成员中，然后清除 `header` 和 `body` 变量。

以下代码对从集合中读取的行进行操作。此代码假定这是标题行的开头，因此在此处搜索字符串以找到冒号并在此处拆分。标题的名称在冒号之前，标题的正文（去除前导空格）在冒号之后。由于我们不知道标题正文是否会折叠到下一行，因此不存储名称/正文；相反，允许 `while` 循环重复一次，以便测试下一行的第一个字符是否为空格，如果是，则将其附加到正文。将名称/正文对保留到 `while` 循环的下一次迭代的操作意味着最后一行不会存储在循环中，因此在方法的末尾有一个测试，以查看 `header` 变量是否为空，如果不是，则存储名称/正文对。

现在可以编译代码（记得使用 `/EHsc` 开关）来测试是否有拼写错误。要测试代码，您应该将电子邮件从您的电子邮件客户端保存为文件，然后使用该文件的路径运行 `email_parser` 应用程序。以下是互联网消息格式 RFC 5322 中提供的示例电子邮件消息之一，您可以将其放入文本文件中以测试代码。

```cpp
    Received: from x.y.test
 by example.net
 via TCP
 with ESMTP
 id ABC12345
 for <mary@example.net>;  21 Nov 1997 10:05:43 -0600
Received: from node.example by x.y.test; 21 Nov 1997 10:01:22 -0600
From: John Doe <jdoe@node.example>
To: Mary Smith <mary@example.net>
Subject: Saying Hello
Date: Fri, 21 Nov 1997 09:55:06 -0600
Message-ID: <1234@local.node.example>

This is a message just to say hello.
So, "Hello".
```

您可以通过电子邮件消息测试应用程序，以显示解析已考虑到标题格式，包括折叠空格。

# 处理标题子项

下一步是将头部内容处理为子项。为此，请在`header_body`类的`public`部分添加以下突出显示的声明：

```cpp
    public: 
        header_body() = default; 
        header_body(const string& b) : body(b) {} 
        string get_body() const { return body; } 
        vector<pair<string, string>> subitems(); 
    };
```

每个子项将是一个名称/值对，由于子项的顺序可能很重要，因此子项存储在`vector`中。更改`main`函数，删除对`get_headers`的调用，而是逐个打印每个头部：

```cpp
    email eml; 
    eml.parse(stm); 
    for (auto header : eml) { cout << header.first << " : "; vector<pair<string, string>> subItems = header.second.subitems(); if (subItems.size() == 0) { cout << header.second.get_body() << "n"; } else { cout << "n"; for (auto sub : subItems) { cout << "   " << sub.first; if (!sub.second.empty()) 
                cout << " = " << sub.second;         
                cout << "n"; } } } 
    cout << "n"; 
    cout << eml.get_body() << endl;
```

由于`email`类实现了`begin`和`end`方法，这意味着范围`for`循环将调用这些方法来访问`email::headers`数据成员上的迭代器。每个迭代器将访问一个`pair<string,header_body>`对象，因此在此代码中，我们首先打印出头部名称，然后访问`header_body`对象上的子项。如果没有子项，头部仍将有一些文本，但不会被拆分为子项，因此我们调用`get_body`方法获取要打印的字符串。如果有子项，则打印出这些子项。有些项将有主体，有些将没有。如果该项有主体，则以`name = value`的形式打印子项。

最后一步是解析头部内容以将其拆分为子项。在`header_body`类下面，添加以下方法的定义：

```cpp
    vector<pair<string, string>> header_body::subitems() 
    { 
        vector<pair<string, string>> subitems; 
        if (body.find(';') == body.npos) return subitems; 

        return subitems; 
    }
```

由于子项使用分号分隔，因此可以简单测试`body`字符串中是否有分号。如果没有分号，则返回一个空的`vector`。

现在，代码必须重复解析字符串，提取子项。有几种情况需要解决。大多数子项将以“name=value;”的形式存在，因此必须提取此子项并在等号字符处拆分，并丢弃分号。

有些子项没有值，形式为`name;`，在这种情况下，分号被丢弃，并且使用空字符串存储子项的值。最后，头部中的最后一项可能没有以分号结尾，因此这必须考虑在内。

添加以下`while`循环：

```cpp
    vector<pair<string, string>> subitems; 
    if (body.find(';') == body.npos) return subitems; 
    size_t start = 0;
 size_t end = start; while (end != body.npos){}
```

正如其名称所示，`start`变量是子项的起始索引，`end`是子项的结束索引。第一步是忽略任何空格，因此在`while`循环中添加：

```cpp
    while (start != body.length() && isspace(body[start])) 
    { 
        start++; 
    } 
    if (start == body.length()) break;
```

这只是在引用空格字符的情况下递增`start`索引，只要它尚未达到字符串的末尾。如果达到字符串的末尾，这意味着没有更多的字符，因此循环结束。

接下来，添加以下内容以搜索`=`和`;`字符并处理搜索情况之一：

```cpp
    string name = ""; 
    string value = ""; 
    size_t eq = body.find('=', start); 
    end = body.find(';', start); 

    if (eq == body.npos) 
    { 
        if (end == body.npos) name = body.substr(start); 
        else name = body.substr(start, end - start); 
    } 
    else 
    {
    } 
    subitems.push_back(make_pair(name, value)); 
    start = end + 1;
```

如果搜索项找不到，则`find`方法将返回`npos`值。第一次调用查找`=`字符，第二次调用查找分号。如果找不到`=`，则该项没有值，只有一个名称。如果找不到分号，则意味着`name`是从`start`索引到字符串末尾的整个字符串。如果有分号，则`name`是从`start`索引到由`end`指示的索引（因此要复制的字符数为`end-start`）。如果在子项中找到`=`字符，则需要在此处拆分字符串，稍后将显示该代码。一旦`name`和`value`变量被赋值，它们将被插入到`subitems`数据成员中，并且`start`索引移动到`end`索引之后的字符。如果`end`索引是`npos`，则`start`索引的值将无效，但这并不重要，因为`while`循环将测试`end`索引的值，并且如果索引是`npos`，则会中断循环。

最后，您需要添加当子项中有`=`字符时的代码。添加以下突出显示的文本：

```cpp
    if (eq == body.npos) 
    { 
        if (end == body.npos) name = body.substr(start); 
        else name = body.substr(start, end - start); 
    } 
    else 
    { 
 if (end == body.npos) { name = body.substr(start, eq - start); value = body.substr(eq + 1); } else { if (eq < end) { name = body.substr(start, eq - start); value = body.substr(eq + 1, end - eq - 1); } else { name = body.substr(start, end - start); } } 
    }
```

第一行测试是否搜索分号失败。在这种情况下，名称是从`start`索引到等号字符之前的字符，值是等号后的文本直到字符串的末尾。

如果等号和分号字符有有效的索引，那么还有一种情况需要检查。可能等号字符的位置在分号之后，这种情况下意味着这个子项没有值，并且等号字符将用于后续子项。

在这一点上，您可以编译代码并使用包含电子邮件的文件进行测试。程序的输出应该是将电子邮件分割为标题和正文，每个标题分割为子项，这些子项可能是简单的字符串或`name=value`对。

# 总结

在本章中，您已经看到了支持字符串的各种 C++标准库类。您已经了解了如何从流中读取字符串，如何将字符串写入流，如何在数字和字符串之间进行转换，以及如何使用正则表达式来操作字符串。当您编写代码时，您将不可避免地花费时间来运行代码，以检查它是否符合您的规范。这将涉及提供检查算法结果的代码，将中间代码记录到调试设备的代码，当然还有在调试器下运行代码。下一章将全面讨论调试代码！
