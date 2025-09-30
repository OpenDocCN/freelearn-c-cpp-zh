# 第六章：Qt 核心基础

本章将帮助你掌握 Qt 基本数据处理和存储的方法。首先，你将学习如何处理文本数据以及如何将文本与正则表达式进行匹配。接下来，我们将概述 Qt 容器并描述与它们相关的常见陷阱。然后，你将了解如何从文件中存储和检索数据，以及如何使用不同的存储格式来存储文本和二进制数据。到本章结束时，你将能够高效地在你的游戏中实现非平凡逻辑和数据处理。你还将知道如何在游戏中加载外部数据，以及如何将你的数据保存到永久存储中以便将来使用。

本章涵盖的主要主题：

+   文本处理

+   Qt 容器

+   将数据序列化为 INI、JSON、XML 和二进制数据

+   保存应用程序的设置

# 文本处理

具有图形用户界面（游戏当然属于这一类别）的应用程序能够通过显示文本并期望用户输入文本与用户交互。我们已经在之前的章节中通过使用 `QString` 类触及了这一主题的表面。现在，我们将进一步深入探讨。

# 字符串编码

C++ 语言没有指定字符串的编码。因此，任何 `char*` 数组和任何 `std::string` 对象都可以使用任意编码。当使用这些类型与原生 API 和第三方库交互时，你必须参考它们的文档以了解它们使用的编码。操作系统原生 API 使用的编码通常取决于当前的区域设置。第三方库通常使用与原生 API 相同的编码，但某些库可能期望另一种编码，例如 UTF-8。

字符串字面量（即你用引号包裹的每个裸文本）将使用实现定义的编码。自 C++11 以来，你可以指定你的文本将具有的编码：

+   `u8"text"` 将生成一个 UTF-8 编码的 `const char[]` 数组

+   `u"text"` 将生成一个 UTF-16 编码的 `const char16_t[]` 数组

+   `U"text"` 将生成一个 UTF-32 编码的 `const char32_t[]` 数组

不幸的是，用于解释源文件的编码仍然是实现定义的，因此将非 ASCII 符号放入字符串字面量中是不安全的。你应该使用转义序列（例如 `\unnnn`）来编写这样的字面量。

在 Qt 中，文本使用 `QString` 类进行存储，该类内部使用 Unicode。Unicode 允许我们表示世界上几乎所有的语言中的字符，并且是大多数现代操作系统中文本原生编码的事实标准。存在多种基于 Unicode 的编码。`QString` 内容的内存表示类似于 UTF-16 编码。基本上，它由一个 16 位值的数组组成，其中每个 Unicode 字符由 1 或 2 个值表示。

当从 `char` 数组或 `std::string` 对象构造 `QString` 时，使用适当的转换方法非常重要，该转换方法取决于文本的初始编码。默认情况下，`QString` 假设输入文本的编码为 UTF-8。UTF-8 与 ASCII 兼容，因此将 UTF-8 或仅 ASCII 文本传递给 `QString(const char *str)` 是正确的。`QString` 提供了多个静态方法来从其他编码转换，例如 `QString::fromLatin1()` 或 `QString::fromUtf16()`。`QString::fromLocal8Bit()` 方法假定与系统区域设置对应的编码。

如果您必须在同一个程序中结合使用 `QString` 和 `std::string`，`QString` 提供了 `toStdString()` 和 `fromStdString()` 方法来执行转换。这些方法也假设 `std::string` 的编码为 UTF-8，因此如果您的字符串使用其他编码，则不能使用它们。

字面量的默认表示（例如，`"text"`）不是 UTF-16，因此每次将其转换为 `QString` 时，都会发生分配和转换。可以使用 `QStringLiteral` 宏避免这种开销：

```cpp
QString str = QStringLiteral("I'm writing my games using Qt"); 
```

`QStringLiteral` 执行两个操作：

+   它在您的字符串字面量前添加一个 `u` 前缀，以确保它在编译时以 UTF-16 编码

+   它以低廉的成本创建一个 `QString` 并指示它使用字面量，而不进行任何分配或编码转换

将所有字符串字面量（除了需要翻译的）包装在 `QStringLiteral` 中是一个好习惯，但这不是必需的，所以如果您忘记这样做，请不要担心。

# `QByteArray` 和 `QString`

`QString` 总是包含 UTF-16 编码的字符串，但如果有未知（尚未确定）编码的数据怎么办？或者，如果数据甚至不是文本怎么办？在这些情况下，Qt 使用 `QByteArray` 类。当您直接从文件读取数据或从网络套接字接收数据时，Qt 将数据作为 `QByteArray` 返回，表示这是一个没有关于编码信息的任意字节数组：

```cpp
QFile file("/path/to/file");
file.open(QFile::ReadOnly);
QByteArray array = file.readAll();
```

在标准库中，`QByteArray` 的最接近等价物是 `std::vector<char>`。正如其名称所暗示的，这只是一个带有一些有用方法的字节数组。在前面的示例中，如果您知道您读取的文件是 UTF-8 编码的，您可以按以下方式将数据转换为字符串：

```cpp
QString text = QString::fromUtf8(array);
```

如果您不知道文件使用什么编码，最好使用系统编码，因此 `QString::fromLocal8Bit` 会更好。同样，在写入文件时，您需要在将字符串传递给 `write()` 函数之前将其转换为字节数组：

```cpp
QString text = "new file content\n";
QFile file("/path/to/file");
file.open(QFile::WriteOnly);
QByteArray array = text.toUtf8();
file.write(array);
```

您可以使用 `file.close()` 来关闭文件。`QFile` 也会在删除时自动关闭文件，因此如果您的 `QFile` 对象在完成文件操作后立即超出作用域，则不需要显式调用 `close()`。

# 使用其他编码

正如我们已经提到的，`QString` 提供了方便的方法来解码和编码在最受欢迎的编码中，如 UTF-8、UTF-16 和 Latin1。然而，Qt 也能处理许多其他编码。你可以使用 `QTextCodec` 类来访问它们。例如，如果你有一个 Big-5 编码的文件，你可以通过其名称请求 Qt 的编解码器对象，并使用 `fromUnicode()` 和 `toUnicode()` 方法：

```cpp
QByteArray big5Encoded = big5EncodedFile.readAll();
QTextCodec *big5Codec = QTextCodec::codecForName("Big5");
QString text = big5Codec->toUnicode(big5Encoded);
QByteArray big5EncodedBack = big5Codec->fromUnicode(text); 
```

你可以使用 `QTextCodec::availableCodecs()` 静态方法列出你的安装上支持的编解码器。在大多数安装中，Qt 可以处理近 1,000 种不同的文本编解码器。

# 基本字符串操作

涉及文本字符串的最基本任务包括添加或删除字符串中的字符、连接字符串以及访问字符串内容。在这方面，`QString` 提供了一个与 `std::string` 兼容的接口，但它还超越了这一点，暴露了许多更多有用的方法。

使用 `prepend()` 和 `append()` 方法可以在字符串的开始或末尾添加数据。使用 `insert()` 方法可以在字符串的中间插入数据，该方法将其第一个参数作为我们需要开始插入字符的位置，第二个参数是实际文本。所有这些方法都有一些重载，可以接受不同可以包含文本数据的对象，包括经典的 `const char*` 数组。

从字符串中删除字符的方式类似。基本方法是使用接受我们需要删除字符的位置和要删除的字符数的 `remove()` 方法，如下所示：

```cpp
QString str = QStringLiteral("abcdefghij");
str.remove(2, 4); // str = "abghij" 
```

还有一个接受另一个字符串的 `remove()` 重载。当调用时，它会从原始字符串中删除所有其出现。这个重载有一个可选参数，它指定比较应该是在默认的大小写敏感（`Qt::CaseSensitive`）还是大小写不敏感（`Qt::CaseInsensitive`）的方式下进行：

```cpp
QString str = QStringLiteral("Abracadabra");
str.remove(QStringLiteral("ab"), Qt::CaseInsensitive);
// str = "racadra"
```

要连接字符串，你可以简单地将两个字符串相加，或者将一个字符串追加到另一个字符串的末尾：

```cpp
QString str1 = QStringLiteral("abc");
QString str2 = QStringLiteral("def");
QString str1_2 = str1 + str2;
QString str2_1 = str2;
str2_1.append(str1); 
```

访问字符串可以分为两种用例。第一种是你希望提取字符串的一部分。为此，你可以使用这些三种方法之一——`left()`、`right()` 和 `mid()`——它们从字符串的开始或结束返回指定数量的字符，或者从字符串的指定位置提取指定长度的子字符串：

```cpp
QString original = QStringLiteral("abcdefghij");
QString l = original.left(3); // "abc"
QString r = original.right(2); // "ij"
QString m = original.mid(2, 5); // "cdefg" 
```

第二种用例是你希望访问字符串的单个字符。索引操作符的使用与 `QString` 类似于 `std::string`，返回一个副本或非常量引用到由 `QChar` 类表示的给定字符，如下面的代码所示：

```cpp
QString str = "foo";
QChar f = str[0]; // const
str[0] = 'g'; // non-const 
```

此外，Qt 还提供了一个专门的方法——`at()`，它返回字符的副本：

```cpp
QChar f = str.at(0); 
```

你应该优先使用 `at()` 而不是索引操作符来执行不修改字符的操作，因为这明确地使用了常量方法。

# 字符串搜索和查找

第二组功能与搜索字符串相关。你可以使用 `startsWith()`、`endsWith()` 和 `contains()` 等方法在字符串的开始、结束或任意位置搜索子字符串。可以使用 `count()` 方法检索字符串中子字符串的出现次数。

小心，还有一个不接收任何参数的 `count()` 方法，它返回字符串中的字符数。

如果你需要知道匹配的确切位置，你可以使用 `indexOf()` 或 `lastIndexOf()` 来接收字符串中匹配发生的位置。第一个调用通过向前搜索工作，而另一个调用通过向后搜索。这些调用中的每一个都接受两个可选参数——第二个参数确定搜索是否区分大小写（类似于 `remove` 的工作方式）。第一个参数是字符串中搜索开始的起始位置。它让你找到给定子字符串的所有出现：

```cpp
int pos = -1;
QString str = QStringLiteral("Orangutans like bananas.");
do {
    pos = str.indexOf("an", pos + 1);
    qDebug() << "'an' found starts at position" << pos;
} while(pos != -1); 
```

# 字符串分解

有另一组有用的字符串功能使得 `QString` 与 `std::string` 不同，那就是将字符串切割成更小的部分，并从更小的片段构建更大的字符串。

非常常见，一个字符串包含由重复分隔符粘合在一起的子字符串（例如，`"1,4,8,15"`）。虽然你可以使用你已知的函数（例如，`indexOf`）从记录中提取每个字段，但存在一种更简单的方法。`QString` 包含一个 `split()` 方法，它接受分隔符字符串作为其参数，并返回一个由 Qt 中的 `QStringList` 类表示的字符串列表。然后，将记录分解成单独的字段就像调用以下代码一样简单：

```cpp
QString record = "1,4,8,15,16,24,42";
QStringList items = record.split(",");
for(const QString& item: items) {
    qDebug() << item;
}
```

此方法的逆操作是 `QStringList` 类中存在的 `join()` 方法，它返回列表中的所有项作为一个由给定分隔符合并的单个字符串：

```cpp
QStringList fields = { "1", "4", "8", "15", "16", "24", "42" };
QString record = fields.join(","); 
```

# 在数字和字符串之间转换

`QString` 还提供了一些方便地在文本和数值之间进行转换的方法。例如 `toInt()`、`toDouble()` 或 `toLongLong()` 的方法使得从字符串中提取数值变得容易。所有这些方法都接受一个可选的 `bool *ok` 参数。如果你将一个指向 `bool` 变量的指针作为此参数传递，该变量将被设置为 `true` 或 `false`，具体取决于转换是否成功。返回整数的方法还接受第二个可选参数，该参数指定值的数值基数（例如，二进制、八进制、十进制或十六进制）：

```cpp
bool ok;
int v1 = QString("42").toInt(&ok, 10);
// v1 = 42, ok = true
long long v2 = QString("0xFFFFFF").toInt(&ok, 16);
// v2 = 16777215, ok = true
double v3 = QString("not really a number").toDouble(&ok);
//v3 = 0.0, ok = false
```

一个名为 `number()` 的静态方法执行相反方向的转换——它接受一个数值和数字基数，并返回该值的文本表示：

```cpp
QString txt = QString::number(42); // txt = "42" 
```

此函数有一些可选参数，允许您控制数字的字符串表示形式。对于整数，您可以指定数值基数。对于双精度浮点数，您可以选择科学格式`'e'`或传统格式`'f'`，并指定小数分隔符后的数字位数：

```cpp
QString s1 = QString::number(42, 16); // "2a"
QString s2 = QString::number(42.0, 'f', 6); // "42.000000"
QString s3 = QString::number(42.0, 'e', 6); // "4.200000e+1"
```

一些表示值的其他类也提供了与`QString`之间的转换功能。这样的一个例子是`QDate`，它表示日期并提供`fromString()`和`toString()`方法。

这些方法对于技术目的来说很棒且易于使用，例如，在配置文件中读取和写入数字。然而，当您需要向用户显示数字或解析用户输入时，它们并不适用，因为不同国家的数字书写方式不同。这引出了*国际化*这一主题。

# 国际化

大多数实际项目都有多个国家的目标受众。它们之间最显著的区别是 spoken language，但还有其他一些开发者可能没有考虑到的方面。例如，点`.`和逗号`,`在全球范围内都相当常见，作为小数分隔符。日期格式也非常不同且不兼容，使用错误的格式（例如，`mm/dd/yyyy`而不是`dd/mm/yyyy`）将导致日期完全不同。

Qt 提供了`QLocale`类来处理与区域设置相关的操作，包括字符串中数字之间的转换。在以下代码中，`text`和`number`的值可能因系统区域设置的不同而不同：

```cpp
QLocale locale = QLocale::system();
QString text = locale.toString(1.2);
double number = locale.toDouble(QStringLiteral("1,2"));
```

`QLocale`还提供了格式化日期和价格的方法，并允许我们请求有关本地约定的更多信息。

关于翻译，我们已提到，任何用户可见的文本都应该包裹在`tr()`函数中。现在我们将解释这一要求。

Qt 的翻译系统使得开发和翻译团队能够独立工作。项目经过以下步骤：

1.  开发者创建一个应用程序，并将所有应翻译的文本包裹在特殊的翻译函数中（例如`tr()`）。表单中的可见文本会自动包裹在翻译函数中。

1.  一个特殊的 Qt 工具（**lupdate**）搜索所有包裹在翻译函数中的字符串，并生成一个翻译文件（`.ts`）。

1.  翻译者在一个称为**Qt Linguist**的特殊应用程序中打开此文件。在该应用程序中，他们能够看到所有按*上下文*分组排列的字符串，这通常是指文本所属的类。他们可以在翻译文件中添加翻译并保存。

1.  当这个新的翻译文件被复制回项目并使用`QCoreApplication::installTranslator`函数应用时，翻译函数开始返回翻译后的文本，而不是简单地返回参数。

1.  随着应用程序的发展和新未翻译文本的出现，它默认显示为未翻译。然而，它可以自动添加到翻译文件中，翻译者可以为新内容添加新的翻译，而不会丢失现有的翻译。

我们不会深入这个过程的细节。作为一个开发者，你只需要确保所有可见的字符串都被包裹在翻译函数中，并提供适当的上下文。上下文是必要的，因为简短的文本（例如，按钮上的一个单词）可能不足以理解其含义并提供适当的翻译，但我们如何指定上下文呢？

主要的翻译函数是`QCoreApplication::translate()`。它接受三个参数：上下文、要翻译的文本和一个可选的歧义文本。歧义参数很少需要。它可以用来区分同一上下文中多个相同文本的实例，以及它们应该有不同的翻译。

通常，你应该使用`tr()`函数而不是`QCoreApplication::translate()`，该函数在每个继承自`QObject`的类中声明。`MyClass::tr(text, disambiguation)`是`QCoreApplication::translate(**"MyClass"**, text, disambiguation)`的快捷方式。因此，位于一个类中的所有可翻译文本将共享相同的`context`字符串，这样它们将在 Qt Linguist 中分组，以便使翻译者的工作更容易。

如果你有一个不在`QObject`子类之外的翻译文本，默认情况下`tr()`函数将不可用。在这种情况下，你有以下选项：

+   使用`QCoreApplication::translate()`函数并显式写出`context`参数

+   重新使用相关类的`tr()`函数（例如，`MyClass::tr()`）

+   在你的（非`QObject`基于的）类中声明`tr()`函数，通过在类声明顶部添加`Q_DECLARE_TR_FUNCTIONS(context)`宏

注意，翻译函数应该直接接收字符串字面量。否则，**lupdate**将无法理解正在翻译哪个文本。以下代码是不正确的，因为两个字符串将不会被翻译者看到：

```cpp
const char* text;
if (condition) {
    text = "translatable1";
} else {
    text = "translatable2";
}
QString result = tr(text); // not recognized!
```

解决这个问题的最简单方法是直接将`tr()`函数应用到每个字符串字面量上：

```cpp
QString result;
if (condition) {
    result = tr("translatable1");
} else {
    result = tr("translatable2");
}
```

另一个解决方案是使用`QT_TR_NOOP`宏标记可翻译文本：

```cpp
if (condition) {
    text = QT_TR_NOOP("translatable1");
} else {
    text = QT_TR_NOOP("translatable2");
}
QString result = tr(text);
```

`QT_TR_NOOP`宏返回其参数不变，但**lupdate**将识别这些字符串必须被翻译。

还可以通过使用特殊的 C++注释形式为翻译者添加注释：`//: ...`或`/*: ... */`。考虑以下示例：

```cpp
//: The button for sending attachment files
QPushButton *button = new QPushButton(tr("Send"));
```

在本节中，我们仅描述了在开始开发多语言游戏之前你需要了解的绝对最小知识。这些知识可以为你节省大量时间，因为在你编写时标记一些文本进行翻译比在大型代码库中后期进行要容易得多。然而，你需要学习更多才能在你的项目中实际实现国际化。我们将在稍后深入探讨这个主题（*在线*章节，*[`www.packtpub.com/sites/default/files/downloads/MiscellaneousandAdvancedConcepts.pdf`](https://www.packtpub.com/sites/default/files/downloads/MiscellaneousandAdvancedConcepts.pdf)）。

# 在字符串中使用参数

一个常见的任务是拥有一个需要动态变化的字符串，其内容取决于某些外部变量的值——例如，你可能想通知用户正在复制的文件数量，显示“正在复制文件 1/2”或“正在复制文件 2/5”，这取决于表示当前文件和文件总数的计数器的值。可能会诱使你通过使用可用的方法之一将所有片段组装在一起来完成这项任务：

```cpp
QString str = "Copying file " + QString::number(current)
            + " of " + QString::number(total);
```

采用这种方法存在一些缺点；最大的问题是将字符串翻译成其他语言时，不同语言的语法可能要求两个参数的位置与英语不同。

相反，Qt 允许我们在字符串中指定位置参数，然后使用实际值替换它们。这种方法称为**字符串插值**。字符串中的位置用 `%` 符号标记（例如，`%1`、`%2` 等等），并通过调用 `arg()` 并传递用于替换字符串中下一个最低标记的值来替换它们。然后我们的文件复制消息构建代码变为如下：

```cpp
QString str = tr("Copying file %1 of %2").arg(current).arg(total); 
```

与内置的 `printf()` 函数的行为相反，你不需要在占位符中指定值的类型（如 `%d` 或 `%s`）。相反，`arg()` 方法有几个重载，可以接受单个字符、字符串、整数和实数。`arg()` 方法具有与 `QString::number()` 相同的可选参数，允许你配置数字的格式。此外，`arg()` 方法还有一个 `fieldWidth` 参数，它强制它始终输出指定长度的字符串，这对于格式化表格来说很方便：

```cpp
const int fieldWidth = 4;
qDebug() << QStringLiteral("%1 | %2").arg(5, fieldWidth).arg(6, fieldWidth);
qDebug() << QStringLiteral("%1 | %2").arg(15, fieldWidth).arg(16, fieldWidth);
// output:
// "   5 |    6"
// "  15 |   16"
```

如果你想要使用除空格以外的字符来填充空白，请使用 `arg()` 的可选参数 `fillChar`。

# 正则表达式

让我们简要地谈谈**正则表达式**——通常简称为 "regex" 或 "regexp"。当您需要检查一个字符串或其部分是否与给定的模式匹配，或者当您想要在文本中找到特定的部分并可能提取它们时，您将需要这些正则表达式。验证有效性和查找/提取都基于所谓的正则表达式模式，它描述了字符串必须具有的格式才能有效、可找到或可提取。由于本书专注于 Qt，很遗憾没有时间深入探讨正则表达式。然而，这并不是一个大问题，因为您可以在互联网上找到许多提供正则表达式介绍的优质网站。

尽管正则表达式的语法有很多变体，但 Perl 使用的那个已经成为了*事实上的*标准。在 Qt 中，`QRegularExpression` 类提供了与 Perl 兼容的正则表达式。

`QRegularExpression` 首次在 Qt 5.0 中引入。在之前的版本中，唯一的正则异常类是 `QRegExp`，但为了兼容性它仍然可用。由于 `QRegularExpression` 更接近 Perl 标准，并且与 `QRegExp` 相比执行速度要快得多，我们建议尽可能使用 `QRegularExpression`。尽管如此，您仍然可以阅读 `QRegExp` 的文档，其中包含对正则表达式的一个很好的通用介绍。

# 行动时间 - 一个简单的问答游戏

为了向您介绍 `QRegularExpression` 的主要用法，让我们想象这个游戏：一张显示物体的照片被展示给多个玩家，他们中的每一个都必须估计物体的重量。估计值最接近实际重量的玩家获胜。估计将通过 `QLineEdit` 提交。由于您可以在行编辑中写入任何内容，我们必须确保内容是有效的。

那么“有效”是什么意思呢？在这个例子中，我们定义一个介于 1g 和 999kg 之间的值是有效的。了解这个规范后，我们可以构建一个验证格式的正则表达式。文本的第一部分是一个数字，可以是 1 到 999 之间的任何数字。因此，相应的模式看起来像 `[1-9]\d{0,2}`，其中 `[1-9]` 允许并且要求恰好一个数字，除了零。它可以选择性地后面跟多达两个数字，包括零。这通过 `\d{0,2}` 来表达，其中 `\d` 表示“任何数字”，0 是最小允许的计数，2 是最大允许的计数。输入的最后部分是重量的单位。使用如 `(mg|g|kg)` 这样的模式，我们允许重量以毫克（`mg`）、克（`g`）或千克（`kg`）输入。通过 `\s*`，我们最终允许在数字和单位之间有任意数量的空白字符。让我们将它们全部组合起来，并立即测试我们的正则表达式：

```cpp
QRegularExpression regex("[1-9]\\d{0,2}\\s*(mg|g|kg)");
regex.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
qDebug() << regex.match("100 kg").hasMatch();       // true
qDebug() << regex.match("I don't know").hasMatch(); // false
```

# 刚才发生了什么？

在第一行，我们构建了上述的 `QRegularExpression` 对象，并将正则表达式的模式作为参数传递给构造函数。请注意，我们必须转义 `\` 字符，因为它在 C++ 语法中有特殊含义。

默认情况下，正则表达式是区分大小写的。然而，我们希望允许输入为大写或混合大小写。为了实现这一点，我们当然可以写出 `(mg|mG|Mg|MG|g|G|kg|kG|Kg|KG)` 或者匹配之前将字符串转换为小写，但有一个更干净、更易读的解决方案。在代码示例的第二行，你可以看到答案——一个模式选项。我们使用了 `setPatternOptions()` 来设置 `QRegularExpression::CaseInsensitiveOption` 选项，它不尊重使用的字符的大小写。当然，你还可以在 Qt 的 `QRegularExpression::PatternOption` 文档中了解更多选项。我们也可以将选项作为 `QRegularExpression` 构造函数的第二个参数传递，而不是调用 `setPatternOptions()`：

```cpp
QRegularExpression regex("[1-9]\\d{0,2}\\s*(mg|g|kg)",
     QRegularExpression::CaseInsensitiveOption); 
```

当我们需要测试一个输入时，我们只需要调用 `match()`，传递我们想要检查的字符串。作为回报，我们得到一个 `QRegularExpressionMatch` 类型的对象，它包含进一步所需的所有信息——而不仅仅是检查有效性。然后，我们可以通过 `QRegularExpressionMatch::hasMatch()` 确定输入是否匹配我们的标准，因为它在找到模式时返回 `true`。否则，当然返回 `false`。

我们的模式还没有完成。如果我们将它匹配到 "foo 142g bar"，`hasMatch()` 方法也会返回 `true`。因此，我们必须定义模式是从匹配字符串的开始到结束进行检查的。这是通过 `\A` 和 `\z` 锚点完成的。前者标记字符串的开始，后者标记字符串的结束。当你使用这样的锚点时，不要忘记转义斜杠。正确的模式将看起来像这样：

```cpp
QRegularExpression regex("\\A[1-9]\\d{0,2}\\s*(mg|g|kg)\\z",
    QRegularExpression::CaseInsensitiveOption); 
```

# 从字符串中提取信息

在我们检查发送的猜测是否格式良好之后，我们必须从字符串中提取实际的重量。为了能够轻松比较不同的猜测，我们还需要将所有值转换为共同的参考单位。在这种情况下，应该是毫克，这是最低的单位。那么，让我们看看 `QRegularExpressionMatch` 可以为我们提供什么来完成任务。

使用 `capturedTexts()`，我们可以得到一个包含模式捕获组的字符串列表。在我们的例子中，这个列表将包含 "23kg" 和 "kg"。列表的第一个元素总是被模式完全匹配的字符串。接下来的元素都是被使用的括号捕获的子字符串。由于我们缺少实际的数字，我们必须将模式的开始更改为 `([1-9]\d{0,2})`。现在，列表的第二个元素是数字，第三个元素是单位。因此，我们可以写出以下内容：

```cpp
int getWeight(const QString &input) {
    QRegularExpression regex("\\A([1-9]\\d{0,2})\\s*(mg|g|kg)\\z");
    regex.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
    QRegularExpressionMatch match = regex.match(input);
    if(match.hasMatch()) {
        const QString number = match.captured(1);
        int weight = number.toInt();
        const QString unit = match.captured(2).toLower();
        if (unit == "g") {
            weight *= 1000;
        } else if (unit == "kg") {
            weight *= 1000000 ;
        }
        return weight;
    } else {
        return -1;
    }
}
```

在函数的前两行中，我们设置了模式和其选项。然后，我们将它与传递的参数进行匹配。如果`QRegularExpressionMatch::hasMatch()`返回`true`，则输入有效，我们提取数字和单位。我们不是通过调用`capturedTexts()`获取捕获文本的整个列表，而是通过调用`QRegularExpressionMatch::captured()`直接查询特定元素。传递的整数参数表示元素在列表中的位置。因此，调用`captured(1)`返回匹配的数字作为一个`QString`。

请注意，在以后的时间添加一个组将使所有后续组的索引增加 1，您将不得不调整您的代码！如果您有长的模式，或者有很高的概率未来会添加更多的括号，您可以使用**命名组**来使您的代码更易于维护。有一个`QRegularExpressionMatch::captured()`重载，允许您指定组名而不是索引。例如，如果您已经编写了`(?<number>[1-9][0-9]{0,2})`，那么您可以通过调用`match.captured("number")`来获取数字。

为了能够使用提取的数字进行计算，我们需要将`QString`转换为整数。这是通过调用`QString::toInt()`完成的。此转换的结果随后存储在`weight`变量中。接下来，我们获取单位并将其转换为小写字符。这样，我们可以轻松地确定用户的猜测是否以克为单位，只需将单位与小写的"g"进行比较。我们不需要关心大写的"G"或变体"KG"、"Kg"和奇特的"kG"（千克）。

为了得到标准化的毫克重量，我们需要将`weight`乘以 1,000 或 1,000,000，具体取决于这是否以 g 或 kg 表示。最后，我们返回这个标准化重量。如果字符串格式不正确，我们返回`-1`以指示给定的猜测无效。然后，调用者负责确定哪个玩家的猜测是最好的。

注意您选择的整数类型是否可以处理重量的值。在我们的例子中，9.99 亿是可能的最大结果，幸运的是，它小于有符号 32 位整数的最大可能值（2,147,483,647）。如果您不确定您使用的类型在所有目标系统上是否足够大，请使用固定宽度整数类型（例如，`int64_t`）。

作为练习，尝试通过允许小数来扩展示例，使得"23.5g"是一个有效的猜测。为了实现这一点，您必须更改模式以输入小数，并且您还必须处理`double`而不是`int`作为标准化重量。

# 查找所有模式出现

最后，让我们看看如何查找字符串中的所有数字，即使是那些以零开头的数字：

```cpp
QString input = QStringLiteral("123 foo 09 1a 3");
QRegularExpression regex("\\b\\d+\\b");
QRegularExpressionMatchIterator i = regex.globalMatch(input);
while (i.hasNext()) {
    QRegularExpressionMatch match = i.next();
    qDebug() << match.captured();
}
```

输入字符串包含一段示例文本，我们希望在其中找到所有数字。模式不应找到“foo”以及“1a”变量，因为这些不是有效的数字。因此，我们设置了模式，定义我们需要至少一个数字，`\d+`，并且这个数字——或者这些数字——应该被单词边界`\b`包围。请注意，您必须转义斜杠。使用这个模式，我们初始化`QRegularExpression`对象，并在其上调用`globalMatch()`。在传递的参数中，将搜索该模式。这次，我们没有返回`QRegularExpressionMatch`；相反，我们得到了`QRegularExpressionMatchIterator`类型的迭代器。由于`QRegularExpressionMatchIterator`有一个方便的`hasNext()`方法，我们检查是否存在进一步的匹配，如果有，就通过调用`next()`来获取下一个匹配。返回的匹配类型是`QRegularExpressionMatch`，您已经知道了。

如果您需要在`while`循环中了解下一个匹配项，您可以使用`QRegularExpressionMatchIterator::peekNext()`来获取它。这个函数的好处是它不会移动迭代器。

这样，您就可以遍历字符串中的所有模式出现。如果您，例如，想在文本中突出显示搜索字符串，这将很有帮助。

我们的示例将给出输出`"123"`、`"09"`和`"3"`。

考虑到这只是一个对正则表达式的简要介绍，我们鼓励您阅读文档中关于`QRegularExpression`、`QRegularExpressionMatch`和`QRegularExpressionMatchIterator`的详细描述部分。正则表达式非常强大且有用，因此，在您的日常编程生活中，您可以从正则表达式的深刻知识中受益！

# 容器

当您需要存储一组对象时，您需要一个容器来容纳它们。C++标准库提供了许多强大的容器，例如`std::vector`、`std::list`或`std::map`。然而，Qt 不使用这些容器（实际上，它几乎不使用任何标准库类）而是提供了自己的容器实现。当 Qt 容器被引入时，它们在不同平台上提供了比标准库实现显著更一致的性能，因此它们被要求创建可靠的跨平台应用程序。现在这并不是真的，因为 STL 实现和编译器已经发展并获得了新的优化和功能。然而，仍然有使用 Qt 容器的原因，尤其是在一个大量使用其他 Qt 类的应用程序中：

+   Qt API 始终使用 Qt 容器。当您收到一个`QList`时，几乎永远不会比将其转换为标准库容器更高效或方便。在调用接受`QList`的方法之前，您应该在`QList`中填充输入数据，而不是将其从 STL 容器中转换。

+   Qt 容器提供了独特的功能，如隐式共享（我们将在本章后面讨论）或 Java 风格的迭代器，以及 STL 容器缺乏的一些便利方法。

+   Qt 容器遵循 Qt 的命名方案和 API 习惯，因此在以 Qt 为中心的程序中看起来更自然。例如，`QVector::isEmpty()` 比 `std::vector::empty()` 更像 Qt 风格。

此外，Qt 容器提供了与 STL 兼容的 API（例如，`append()` 方法有 `push_back()` 别名），这使得我们可以在不改变代码大部分内容的情况下用 STL 容器替换 Qt 容器。基于范围的 `for` 循环和一些标准库算法也与 Qt 容器兼容。话虽如此，如果你需要 Qt 容器中不可用的某些功能，使用 STL 容器是个不错的选择。

# 主要容器类型

当你与 Qt API 方法交互时，你在容器类型上没有太多选择，因为你需要使用该方法使用的容器。然而，通常，你可以自由选择容器来存储你的数据。让我们了解一下主要的 Qt 容器以及何时使用它们。

我们将只提供一个简要概述 Qt 容器，不会深入到不同操作的算法复杂度等细节。对于大多数 Qt 容器，都有一个类似的 STL 容器，我们将命名它。选择正确容器的主题被广泛讨论，并且不难找到更多相关信息，特别是对于 STL 容器。你还可以在 Qt 容器类文档页面找到更多信息。

`QVector` 在内存的连续区域存储项目。项目是紧密打包的，这意味着这种类型是最节省内存和缓存友好的。它的 STL 等价物是 `std::vector`。`QVector` 应该是默认选择的容器，这意味着只有在你有理由这样做的情况下才应该使用不同的容器。`QVector` 提供了按项目编号快速查找、平均快速在末尾追加项目和从末尾删除项目。从向量开始或中间插入和删除项目较慢，因为这会导致右侧的所有项目在内存中移动。使用 `QVector` 是直接的：

```cpp
QVector<int> numbers;
numbers.append(1);
numbers.append(5);
numbers.append(7);
qDebug() << numbers.count(); // 3
qDebug() << numbers[1];      // 5
```

`QLinkedList` 容器，正如其名所示，实现了一个链表。它的 STL 等价物是 `std::list`。与 `QVector` 相比，它可以在任何位置（开始、中间或末尾）快速插入和删除项目，但按索引查找较慢，因为它需要从开始遍历项目以找到按索引的项目。`QLinkedList` 适用于需要多次在长列表中间插入或删除项目的情况。然而，请注意，在实际应用中，`QVector` 在这种情况下可能仍然更高效，因为 `QLinkedList` 在内存中不是紧密打包的，这增加了额外的开销。

`QSet` 是 Qt 的 `std::unordered_set` 等效物，是一个无序的唯一项目集合。它的优点是能够高效地添加项目、删除项目以及检查特定项目是否存在于集合中。其他列表类无法快速执行最后操作，因为它们需要遍历所有项目并将每个项目与参数进行比较。像任何其他集合一样，你可以遍历集合的项目，但迭代顺序未指定，也就是说，任何项目都可能出现在第一次迭代中，依此类推。以下代码展示了 `QSet` API 的一个示例：

```cpp
QSet<QString> names;
names.insert("Alice");
names.insert("Bob");
qDebug() << names.contains("Alice"); // true
qDebug() << names.contains("John"); // false
for(const QString &name: names) {
    qDebug() << "Hello," << name;
}
```

最后一个平面集合是 `QList`。目前不推荐使用它，除非与接受或生成 `QList` 对象的方法交互。它的性能和内存效率取决于项目类型，而定义“良好”项目类型的规则很复杂。对于“不良”类型，`QList` 表示为一个 `void *` 向量，每个项目都作为单独分配的对象存储在堆上。`QList` 实现可能在 Qt 6 中发生变化，但目前还没有官方信息。

有一些专门列表容器为特定项目类型提供了额外的功能：

+   已经熟悉的 `QString` 类本质上是一个 `QChar`（16 位 Unicode 字符）的向量

+   熟悉的 `QByteArray` 是一个 `char` 向量

+   `QStringList` 是一个带有额外便利操作的 `QList<QString>`

+   `QBitArray` 提供了一个具有一些有用 API 的内存高效位数组

接下来，有两个主要的键值集合：`QMap<K, T>` 和 `QHash<K, T>`。它们允许你将类型为 `T` 的值（或多个值）与类型为 `K` 的键关联起来。它们都提供了相对快速的键查找。当遍历 `QMap`（类似于 `std::map`）时，项目按键排序，而不考虑插入顺序：

```cpp
QMap<int, QString> map;
map[3] = "three";
map[1] = "one";
map[2] = "two";
for(auto i = map.begin(); i != map.end(); ++i) {
    qDebug() << i.key() << i.value();
}
// output:
// 1 "one"
// 2 "two"
// 3 "three"
```

`QHash`（类似于 `std::unordered_map`）与 `QMap` 有非常相似的 API，但会按未指定的顺序遍历项目，就像 `QSet`。你可以在前面的例子中将 `QMap` 替换为 `QHash`，并看到即使重复运行相同的程序，迭代顺序也会改变。作为交换，`QHash` 在平均插入和键查找方面比 `QMap` 更快。如果你不关心迭代顺序，你应该使用 `QHash` 而不是 `QMap`。

一个细心的读者可能会想知道看起来非常确定性的代码如何产生随机结果。这种随机性是故意引入的，以防止对 `QHash` 和 `QSet` 的 *算法复杂性攻击*。你可以阅读 `QHash` 文档页面的相应部分，了解更多关于攻击和配置随机化的方法。

最后，`QPair<T1, T2>` 是一个简单的类，可以持有两种不同类型的两个值，就像 `std::pair`。你可以使用 `qMakePair()` 函数从两个值中创建一个对。

# 便利容器

除了前面描述的容器之外，还有一些容器建立在它们之上，提供了一些在特殊情况下更方便的 API 和行为：

| **容器** | **描述** |
| --- | --- |
| `QStack` | 一个实现**后进先出**（LIFO）结构的`QVector`。它包含用于向栈中添加项目的`push()`函数，用于移除栈顶元素的`pop()`函数，以及用于读取栈顶元素而不移除它的`top()`函数。 |
| `QQueue` | 一个实现**先进先出**（FIFO）结构的`QList`。使用`enqueue()`将项目追加到队列中，使用`dequeue()`从队列中取出头部项目，使用`head()`读取头部项目而不移除它。 |
| `QMultiMap` | 一个针对具有多个键值的 API 定制的`QMap`。`QMap`已经允许我们这样做；例如，你可以使用`QMap::insertMulti()`方法使用一个键添加多个项目。然而，`QMultiMap`将其重命名为`insert()`，并隐藏了不允许每个键有多个值的原始`QMap::insert()`方法。 |
| `QMultiHash` | 与`QMultiMap`类似，它是一个`QHash`，具有更方便的 API，用于存储每个键的多个值。 |
| `QCache` | 一个类似于`QHash`的键值存储，允许你实现缓存。`QCache`将在元素未被最近使用时删除其元素，以保持缓存大小在最大允许大小之下。由于无法知道任意项目实际消耗的空间量，你可以为每个项目手动指定一个*成本*，以及特定`QCache`对象的最大总成本。 |
| `QContiguousCache` | 一个扁平容器，允许你缓存大列表的一个子列表。这在实现大表格的查看器时很有用，因为在当前滚动位置附近很可能发生读写操作。 |

当你的任务与它们的用例匹配时，使用这些类中的一个是个好主意。

# 允许的项目类型

并非所有类型都可以放入容器中。所有容器只能持有提供默认构造函数、拷贝构造函数和赋值运算符的类型。所有原始类型和大多数 Qt 数据类型（如`QString`或`QPointF`）都满足这些要求。简单的结构体也可以存储在容器中，因为根据 C++标准，会自动为它们生成所需的构造函数和运算符。|

特定类型通常不能放入容器中，因为它没有无参构造函数，或者故意禁用了此类型的复制。对于`QObject`及其所有子类来说，情况确实如此。`QObject`的使用模式表明，你通常想要存储指向`QObject`的指针以供以后引用。如果该对象被移动到容器中或容器内部移动，指针将被无效化，因此这些类型没有复制构造函数。然而，你可以将指向`QObject`的指针放入容器中（例如，`QVector<QObject *>`），因为指针是一种满足所有要求的基本类型。在这种情况下，你必须手动确保在对象被删除后，你的容器不会包含任何悬垂指针。

前面的限制适用于列表项和键值集合的*值*，但它们的键呢？事实证明，键类型有更多的限制，这取决于集合类型。

`QMap<K, T>`还要求键类型`K`具有比较运算符`operator<`，它提供了一种*全序*（即满足一组特定的公理）。作为一个例外，指针类型也被允许作为键类型。

`QHash<K, T>`和`QSet<K>`要求`K`类型具有`operator==`运算符，并且存在`qHash(K key)`函数重载。Qt 为大量可能实现这些重载的类型提供了支持，如果需要，你可以为你的自定义类型创建重载。

# 隐式共享

标准库容器和 Qt 容器之间最显著的区别是隐式共享功能。在 STL 中，创建容器副本会立即导致内存分配并复制数据缓冲区：

```cpp
std::vector<int> x { 1, 2, 3};
std::vector<int> y = x; // full copy
```

如果你没有打算编辑副本，这本质上是一种资源的浪费，你想要避免这种情况。在某些情况下，通过提供一个引用（`const std::vector<int> &`）而不是创建副本，可以轻松地做到这一点。然而，有时很难确保引用能够有效足够长的时间，例如，如果你想将其存储在类字段中。解决这个任务的另一种方法是使用`shared_ptr`包装一个向量，以显式地在多个对象之间共享它。当你使用 Qt 容器和一些其他 Qt 类型时，这变得不再必要。

在 Qt 中，所有主要的容器类型都实现了**隐式共享**或**写时复制**语义。复制一个`QVector`不会导致新的内存分配，直到两个向量中的任何一个发生变化：

```cpp
QVector<int> x { 1, 2, 3};
QVector<int> y = x;
// x and y share one buffer now
y[0] = 5; // new allocation happens here
// x and y have different buffers now
```

只要不对副本或原始对象进行编辑，复制成本非常低。这允许你以低成本轻松地在对象之间共享常量数据，而无需在代码中手动管理共享对象。此功能也适用于`QString`、`QPen`和许多其他 Qt 值类型。任何复制操作仍然有一些由引用计数引起的运行时开销，因此当容易时，你被鼓励传递引用而不是创建副本。然而，在大多数情况下，这种开销微不足道，除了计算密集型的地方。

如果你喜欢隐式共享，你可以使用`QSharedDataPointer`在自己的数据类型中实现它。请参阅其文档以获取详细说明。

在大多数情况下，你可以像它们没有实现隐式共享一样使用容器，但有一些情况下你必须注意这一点。

# 指针失效

首先，隐式共享意味着在有可能更改此对象或任何共享相同缓冲区的对象时，不允许持有对容器内容的任何引用或指针。以下小型示例说明了问题：

```cpp
// don't do this!
QVector<int> x { 1, 2, 3 };
int *x0 = x.begin();
QVector<int> y = x;
x[0] = 42;
qDebug() << *x0; // output: 1
```

我们用`x`向量的第一个元素的指针初始化了`x0`变量。然而，当我们为该元素设置新值然后尝试使用该指针读取它时，我们再次得到了旧值。

# 刚才发生了什么？

当我们将`x`向量复制到`y`时，两个向量的状态变得共享，原始缓冲区对它们两个都可用。然而，当我们使用`operator[]`修改`x`时，它变成了**分离的**，也就是说，为它分配了新的缓冲区，而`y`保留了原始缓冲区。`x0`指针继续指向原始缓冲区，现在它只对`y`可用。如果你删除`QVector<int> y = x;`这一行，输出将变为预期的 42。一般规则是，你应该避免在对象被修改或与另一个对象共享时存储指向其内容的指针或引用。

# 不必要的分配

接下来的问题是，对对象采取哪些操作会触发新缓冲区的实际分配？显然，`x[0] = 42`会触发分配，因为向量需要一个缓冲区来写入新数据。然而，如果`x`没有声明为`const`值或引用，`int i = x[0]`也会触发分配。这是因为即使在这次情况下并不必要，C++中的此代码也会触发可用的非`const`重载的`operator[]`。向量不知道请求的项目是否会更改，因此它必须假设它将会更改，并在返回新缓冲区中项目引用之前触发分配。

当使用具有 const 和非 const 重载的其他方法时，也会出现相同的问题，例如`begin()`或`data()`。基于范围的`for`循环也会调用`begin()`，所以如果你迭代非`const`值，它也会分离。

如果你显式地将容器变量声明为 const（例如，`const QVector<int> y` 或 `const QVector<int> &y`），则不可用非 const 方法，并且无法使用此变量触发分配。一个替代方案是使用仅对 const 版本可用的特殊方法别名，例如 `at()` 用于 `operator=`，`constBegin()` 用于 `begin()`，以及 `constData()` 用于 `data()`。然而，此解决方案不适用于基于范围的 `for` 循环。

# 基于范围的 `for` 循环和 Qt `foreach` 宏

Qt 提供了用于遍历 Qt 容器的 `foreach` 宏：

```cpp
QVector<int> x { 1, 2, 3 };
foreach(const int i, x) {
    qDebug() << i;
}
```

这个宏在基于范围的 `for` 循环进入 C++ 标准之前就已经可用，因此在 Qt 代码中仍然非常常见，你应该熟悉它。`foreach` 循环始终创建迭代对象的临时常量副本。由于它使用隐式共享，这非常便宜。如果你在遍历 `x` 时编辑它，更改将不会影响 `i` 的值，因为迭代使用的是副本，但这也意味着这种操作是安全的。请注意，当使用基于范围的 `for` 循环、STL 风格迭代器或 Java 风格迭代器时，编辑你正在遍历的同一容器通常是不安全的。例如，更改项目值可能是允许的，但删除项目可能会导致未定义的行为。

我们讨论了基于范围的 `for` 循环如何导致容器深拷贝。`foreach` 宏本身永远不会导致深拷贝。然而，如果你在遍历容器的同时编辑它，这将导致深拷贝，因为必须将两个数据版本存储在某个地方。

当使用基于范围的 `for` 循环时，你应该小心不要传递临时对象的引用。例如，此代码看起来合法，但它会导致未定义的行为：

```cpp
// don't do this!
for(QChar c: QString("abc").replace('a', 'z')) {
    qDebug() << c;
}
```

# 发生了什么？

我们创建了一个临时的 `QString` 对象并调用了它的 `replace()` 方法。此方法的返回类型是 `QString &`，因此它不拥有字符串的数据。如果我们立即将此值赋给拥有变量，则它是正确的，因为原始临时 `QString` 的生命周期持续到整个表达式结束（在这种情况下，赋值）：

```cpp
QString string = QString("abc").replace('a', 'z');
for(QChar c: string) { // correct
    qDebug() << c;
}
```

然而，原始示例中的临时对象在 `for` 循环结束时不会存活，因此这会导致使用后释放的漏洞。此代码的 `foreach` 版本将包含对变量的隐式赋值，因此它是正确的。

另一方面，`foreach` 的宏特性是其缺点。例如，以下代码无法编译，因为项目类型包含逗号：

```cpp
QVector<QPair<int, int>> x;
foreach(const QPair<int, int>& i, x) {
    //...
}
```

错误是 "宏 `Q_FOREACH` 传递了 3 个参数，但它只接受 2 个"。要修复此问题，你必须为项目类型创建一个 `typedef`。

自 C++11 以来，基于范围的 `for` 循环是 `foreach` 的本地、干净的替代方案，因此我们建议你优先选择本地构造而不是宏，但请记住我们描述的陷阱。

# 数据存储

在实现游戏时，你经常会需要处理持久数据；你需要存储保存的游戏数据、加载地图等。为此，你必须了解让你可以使用存储在数字媒体上的数据的机制。

# 文件和设备

用于访问数据的最低级和最基本机制是从文件中保存和加载它。虽然你可以使用 C 和 C++提供的经典文件访问方法，例如`stdio`或`iostream`，但 Qt 提供了自己的文件抽象，它隐藏了平台相关的细节，并提供了一个在所有平台上以统一方式工作的干净 API。

当使用文件时，你将与之合作的两个基本类是`QDir`和`QFile`。前者代表目录的内容，允许你遍历文件系统、创建和删除目录，最后，访问特定目录中的所有文件。

# 遍历目录

使用`QDir`遍历目录非常简单。首先，你需要有一个`QDir`实例。最简单的方法是将目录路径传递给`QDir`构造函数。

Qt 以平台无关的方式处理文件路径。尽管 Windows 上的常规目录分隔符是反斜杠字符（`\`），而其他平台上是正斜杠（`/`），Qt 内部始终使用正斜杠，并且大多数 Qt 方法返回的路径从不包含反斜杠。你始终可以使用正斜杠将路径传递给 Qt 方法，即使在 Windows 上也是如此。如果你需要将 Qt 的路径表示形式转换为本地形式（例如，传递给标准库或第三方库），可以使用`QDir::toNativeSeparators()`。`QDir::fromNativeSeparators()`执行相反的操作。

Qt 提供了一系列静态方法来访问一些特殊目录。以下表格列出了这些特殊目录及其访问函数：

| **访问函数** | **目录** |
| --- | --- |
| `QDir::current()` | 当前工作目录 |
| `QDir::home()` | 当前用户的家目录 |
| `QDir::root()` | 根目录——通常在 Unix 系统中为`/`，在 Windows 系统中为`C:\` |
| `QDir::temp()` | 系统临时目录 |

`QStandardPaths`类提供了关于系统中存在的其他标准位置的信息。例如，`QStandardPaths::writableLocation(QStandardPaths::MusicLocation)`返回用户音乐文件夹的路径。

请参考`QStandardPaths::StandardLocation`枚举文档以获取可用位置列表。

当你已经有一个有效的`QDir`对象时，你可以开始在不同目录之间移动。为此，你可以使用`cd()`和`cdUp()`方法。前者移动到命名的子目录，而后者移动到父目录。你应该始终检查这些命令是否成功。如果它们返回`false`，你的`QDir`对象将保持在同一目录！

要列出特定目录中的文件和子目录，您可以使用 `entryList()` 方法，该方法返回与 `entryList()` 传递的准则匹配的目录条目列表。`filters` 参数接受一个标志列表，这些标志对应于条目需要具有的不同属性才能包含在结果中。以下表格列出了最有用的标志：

| **过滤器** | **含义** |
| --- | --- |
| `QDir::Dirs`, `QDir::Files`, `QDir::Drives` | 列出目录、文件或 Windows 驱动器。您至少应指定这些过滤器之一以获取任何结果。 |
| `QDir::AllEntries` | 列出目录、文件和驱动器。这是 `Dirs | Files | Drives` 的快捷方式。 |
| `QDir::AllDirs` | 即使它们不匹配名称过滤器，也会列出目录。 |
| `QDir::NoDotAndDotDot` | 不要列出 `.`（当前目录）和 `..`（父目录）条目。如果存在 `Dirs` 标志且未指定 `NoDotAndDotDot`，则这些条目将始终列出。 |
| `QDir::Readable`, `QDir::Writable`, `QDir::Executable` | 仅列出可读、可写或可执行的项目。 |
| `QDir::Hidden`, `QDir::System` | 列出隐藏文件和系统文件。如果未指定这些标志，则不会列出隐藏和系统标志。 |

`entryList()` 的 `sort` 参数允许您选择结果的排序方式：

| **标志** | **含义** |
| --- | --- |
| `QDir::Unsorted` | 项目顺序未定义。如果您不关心顺序，使用它是个好主意，因为它可能更快。 |
| `QDir::Name`, `QDir::Time`, `QDir::Size`, `QDir::Type` | 按适当的条目属性排序。 |
| `QDir::DirsFirst`, `QDir::DirsLast` | 确定目录是否应在文件之前或之后列出。如果未指定任何标志，则目录将与文件混合在输出中。 |
| `QDir::Reversed` | 反转顺序。 |

此外，`entryList()` 方法还有一个重载版本，它接受一个以 `QStringList` 形式的文件名模式列表作为其第一个参数。以下是一个示例调用，它返回目录中所有按大小排序的 JPEG 文件：

```cpp
QStringList nameFilters = { QStringLiteral("*.jpg"), QStringLiteral("*.jpeg") };
QStringList entries = dir.entryList(nameFilters,
    QDir::Files | QDir::Readable, QDir::Size);
```

除了 `entryList()` 方法之外，还有一个 `entryInfoList()` 方法，它将每个返回的文件名包裹在一个具有许多方便功能的 `QFileInfo` 对象中。例如，`QFileInfo::absoluteFilePath()` 返回文件的绝对路径，而 `QFileInfo::suffix()` 返回文件的扩展名。

如果您需要递归遍历目录（例如，用于在所有子目录中查找所有文件），则可以使用 `QDirIterator` 类。

# 读取和写入文件

一旦您知道了文件的路径（例如，使用 `QDir::entryList()`、`QFileDialog::getOpenFileName()` 或某些外部来源），您就可以将其传递给 `QFile` 以接收一个作为文件句柄的对象。在可以访问文件内容之前，需要使用 `open()` 方法打开文件。此方法的基本变体需要一个模式，其中我们需要打开文件。以下表格解释了可用的模式：

| **模式** | **描述** |
| --- | --- |
| `只读` | 此文件可读取。 |
| `只写` | 此文件可写入。 |
| `读写` | 此文件可读取和写入。 |
| `追加` | 所有数据写入都将写入文件末尾。 |
| `截断` | 如果文件存在，在打开文件之前，其内容将被删除。 |
| `文本` | 读取时，所有行结束符都转换为 `\n`。写入时，所有 `\n` 符号都转换为本地格式（例如，Windows 上的 `\r\n` 或 Linux 上的 `\n`）。 |
| `无缓冲` | 该标志防止文件被缓冲。 |

`open()` 方法返回 `true` 或 `false`，取决于文件是否已打开。可以通过在文件对象上调用 `isOpen()` 来检查文件当前的状态。一旦文件打开，就可以根据打开文件时传递的选项进行读取或写入。读取和写入是通过 `read()` 和 `write()` 方法完成的。这些方法有许多重载，但我们建议您专注于使用那些接受或返回已熟悉的 `QByteArray` 对象的变体，因为它们自动管理内存。如果您正在处理纯文本，那么 `write` 的一个有用的重载是直接接受文本作为输入的变体。只需记住，文本必须以空字符终止。当从文件读取时，Qt 提供了其他一些可能在某些情况下很有用的方法。其中一种方法是 `readLine()`，它尝试从文件中读取，直到遇到新行字符。如果您与 `atEnd()` 方法一起使用，后者告诉您是否已到达文件末尾，您就可以实现逐行读取文本文件：

```cpp
QStringList lines;
while(!file.atEnd()) {
    QByteArray line = file.readLine();
    lines.append(QString::fromUtf8(line));
} 
```

另一个有用的方法是 `readAll()`，它简单地返回文件内容，从文件指针的当前位置开始，直到文件末尾。

然而，你必须记住，在使用这些辅助方法时，如果你不知道文件包含多少数据，你应该非常小心。可能会发生这种情况，当你逐行读取或尝试一次性将整个文件读入内存时，你会耗尽进程可用的内存量。如果你只想处理适合内存的小文件，你可以通过在`QFile`实例上调用`size()`来检查文件大小，如果文件太大则终止。然而，如果你需要处理任意大小的文件，你应该分步骤处理文件数据，每次只读取一小部分字节。这使得代码更复杂，但可以更好地管理可用资源。

如果你需要持续访问文件，你可以使用`map()`和`unmap()`调用，这些调用将文件的部分映射到内存地址，然后你可以像使用常规字节数组一样使用它：

```cpp
QFile f("myfile");
if(!f.open(QFile::ReadWrite)) {
    return;
}
uchar *addr = f.map(0, f.size());
if(!addr) {
    return;
}
f.close();
doSomeComplexOperationOn(addr);
```

当`QFile`对象被销毁时，映射将自动删除。

# 设备

`QFile`实际上是`QIODevice`（"输入/输出设备"）的子类，`QIODevice`是 Qt 接口，用于抽象与读取和写入数据块相关的实体。有两种类型的设备：顺序访问设备和随机访问设备。`QFile`属于后者；它具有起始、结束、大小和当前位置的概念，用户可以通过`seek()`方法更改这些概念。顺序设备，如套接字和管道，表示数据流——无法回滚流或检查其大小；你只能按顺序逐个读取数据——一次读取一部分，你可以检查你目前距离数据末尾有多远。我们将在第七章，*网络*中处理此类设备。

所有 I/O 设备都可以打开和关闭。它们都实现了`open()`、`read()`和`write()`接口。向设备写入数据会将数据排队等待写入；当数据实际写入时，会发出`bytesWritten()`信号，该信号携带写入设备的数据量。如果在顺序设备中还有更多数据可用，它会发出`readyRead()`信号，通知你如果现在调用`read`，你可以期待从设备接收一些数据。

# 行动时间 - 实现加密数据的设备

让我们实现一个非常简单的设备，该设备使用一个非常简单的算法——凯撒密码来加密或解密通过它的数据流。在加密时，它将明文中的每个字符按密钥定义的字符数进行移位。解密时执行相反操作。因此，如果密钥是`2`，明文字符是`a`，密文变为`c`。使用密钥`4`解密`z`将得到值`v`。

首先，通过从“其他项目”类别中选择“空 qmake 项目模板”来创建一个新的空项目。接下来，添加一个`main.cpp`文件和一个新的`CaesarCipherDevice`类，该类从`QIODevice`派生。该类的基本接口将接受一个整数密钥并设置一个作为数据源或目的地的底层设备。这些都是你应该已经理解的简单编码，因此不需要任何额外的解释，如下所示：

```cpp
class CaesarCipherDevice : public QIODevice
{
    Q_OBJECT
    Q_PROPERTY(int key READ key WRITE setKey)
public:
    explicit CaesarCipherDevice(QObject *parent = 0)
        : QIODevice(parent) {
        m_key = 0;
        m_baseDevice = 0;
    }
    void setBaseDevice(QIODevice *dev) {
        m_baseDevice = dev;
    }
    QIODevice *baseDevice() const {
        return m_baseDevice;
    }
    void setKey(int k) {
        m_key = k;
    }
    inline int key() const {
        return m_key;
    }
private:
    int m_key;
    QIODevice *m_baseDevice;
}; 
```

下一步是确保如果没有设备可以操作（即当`m_baseDevice == nullptr`时），则不能使用该设备。为此，我们必须重新实现`QIODevice::open()`方法，并在我们想要防止操作我们的设备时返回`false`：

```cpp
bool CaesarCipherDevice::open(OpenMode mode) {
    if(!m_baseDevice) {
        return false;
    }
    if(!m_baseDevice->isOpen()) {
        return false;
    }
    if(m_baseDevice->openMode() != mode) {
        return false;
    }
    return QIODevice::open(mode);
}
```

该方法接受用户想要以何种模式打开设备。我们在调用基类实现之前执行一个额外的检查，以验证基础设备是否以相同的模式打开，这将标记设备为打开。

调用`QIODevice::setErrorString`来让用户知道错误是一个好主意。此外，当发生错误时，可以使用`qWarning("message")`将警告打印到控制台。

要有一个完全功能性的设备，我们仍然需要实现两个受保护的纯虚方法，这些方法执行实际的读取和写入。这些方法在需要时由 Qt 从类的其他方法调用。让我们从`writeData()`开始，它接受一个包含数据的缓冲区的指针和与缓冲区大小相等的大小：

```cpp
qint64 CaesarCipherDevice::writeData(const char *data, qint64 len) {
    QByteArray byteArray;
    byteArray.resize(len);
    for(int i = 0; i < len; ++i) {
        byteArray[i] = data[i] + m_key;
    }
    int written = m_baseDevice->write(byteArray);
    emit bytesWritten(written);
    return written;
}
```

首先，我们创建一个局部字节数组并将其调整到输入的长度。然后，我们迭代输入的字节，将密钥的值添加到每个字节（这实际上执行了加密）并将其放入字节数组中。最后，我们尝试将字节数组写入底层设备。在通知调用者实际写入的数据量之前，我们发出一个携带相同信息的信号。

我们需要实现的最后一种方法是通过对基础设备进行读取并添加密钥到数据中的每个单元格来执行解密。这是通过实现`readData()`来完成的，它接受一个指向方法需要写入的缓冲区的指针以及缓冲区的大小。

代码与`writeData()`非常相似，只是我们是在减去密钥值而不是添加它：

```cpp
qint64 CaesarCipherDevice::readData(char *data, qint64 maxlen) {
    QByteArray baseData = m_baseDevice->read(maxlen);
    const int size = baseData.size();
    for(int i = 0; i < size; ++i) {
        data[i] = baseData[i] - m_key;
    }
    return size;
}
```

首先，我们尝试从底层设备读取`maxlen`个字节并将数据存储在字节数组中。请注意，字节数组可能包含少于`maxlen`个字节（例如，如果我们到达了文件的末尾），但它不能包含更多。然后，我们迭代数组并将数据缓冲区的后续字节设置为解密值。最后，我们返回实际读取的数据量。

一个简单的`main()`函数，可以测试该类，如下所示：

```cpp
int main(int argc, char **argv) {
    QByteArray ba = "plaintext";
    QBuffer buf;
    buf.open(QIODevice::WriteOnly);
    CaesarCipherDevice encrypt;
    encrypt.setKey(3);
    encrypt.setBaseDevice(&buf);
    encrypt.open(buf.openMode());
    encrypt.write(ba);
    qDebug() << buf.data();

    CaesarCipherDevice decrypt;
    decrypt.setKey(3);
    decrypt.setBaseDevice(&buf);
    buf.open(QIODevice::ReadOnly);
    decrypt.open(buf.openMode());
    qDebug() << decrypt.readAll();
    return 0;
} 
```

我们使用实现`QIODevice` API 并作为`QByteArray`或`QString`适配器的`QBuffer`类。

# 刚才发生了什么？

我们创建了一个加密对象，并将其密钥设置为`3`。我们还告诉它使用一个`QBuffer`实例来存储处理后的内容。在打开它以供写入后，我们向其中发送了一些数据，这些数据被加密并写入到基本设备中。然后，我们创建了一个类似的设备，再次将相同的缓冲区作为基本设备传递，但现在，我们打开设备以供读取。这意味着基本设备包含密文。在此之后，我们从设备中读取所有数据，这导致从缓冲区中读取数据，解密它，并将数据返回以便写入调试控制台。

# 尝试一下英雄——凯撒密码的 GUI

你可以通过实现一个完整的 GUI 应用程序来结合你已知的知识，该应用程序能够使用我们刚刚实现的凯撒密码`QIODevice`类加密或解密文件。记住，`QFile`也是`QIODevice`，所以你可以直接将其指针传递给`setBaseDevice()`。

这只是你的起点。`QIODevice` API 非常丰富，包含许多虚拟方法，因此你可以在子类中重新实现它们。

# 文本流

现今计算机产生的数据大多基于文本。你可以使用你已知的机制创建此类文件——打开`QFile`以供写入，使用`QString::arg()`将所有数据转换为字符串，可选地使用`QTextCodec`对字符串进行编码，并通过调用`write`将生成的字节写入文件。然而，Qt 提供了一个很好的机制，可以自动为你完成大部分工作，其工作方式类似于标准 C++ `iostream`类。`QTextStream`类以流式方式操作任何`QIODevice` API。你可以使用`<<`运算符向流中发送标记，它们将被转换为字符串，用空格分隔，并使用你选择的编解码器编码，然后写入底层设备。它也可以反过来工作；使用`>>`运算符，你可以从文本文件中流式传输数据，透明地将字符串转换为适当的变量类型。如果转换失败，你可以通过检查`status()`方法的结果来发现它——如果你得到`ReadPastEnd`或`ReadCorruptData`，这意味着读取失败。

虽然`QIODevice`是`QTextStream`操作的主要类，但它也可以操作`QString`或`QByteArray`，这使得它在为我们组合或解析字符串时非常有用。

使用`QTextStream`很简单——你只需将其设备传递给构造函数，然后就可以使用了。`QTextStream`对象将从这个设备中读取或写入。默认情况下，`QTextStream`使用当前区域设置的编码，但如果它遇到 UTF-16 或 UTF-32 的字节顺序标记（BOM），它将切换到由 BOM 指定的编码。流接受字符串和数值：

```cpp
QFile file("output.txt");
file.open(QFile::WriteOnly | QFile::Text);
QTextStream stream(&file);
stream << "Today is " << QDate::currentDate().toString() << endl;
QTime t = QTime::currentTime();
stream << "Current time is " << t.hour() << " h and " 
       << t.minute() << "m." << endl;
```

除了将内容定向到流中，流还可以接受多个操纵器，例如`endl`，它们直接或间接地影响流的行为。例如，你可以告诉流将一个数字以十进制显示，另一个以大写十六进制数字显示，如下面的代码所示（代码中突出显示的都是操纵器）：

```cpp
for(int i = 0;i < 10; ++i) {
    int num = qrand() % 100000;  // random number between 0 and 99999
    stream << dec << num
           << showbase << hex << uppercasedigits << num << endl;
} 
```

这并不是`QTextStream`功能的终点。它还允许我们通过定义列宽和对齐方式以表格形式显示数据。考虑以下由以下结构定义的游戏玩家记录：

```cpp
struct Player {
    QString name;
    qint64 experience;
    QPoint position;
    char direction;
};
```

假设你有一组存储在`QVector<Player> players`变量中的玩家记录。让我们以表格形式将这些信息写入文件：

```cpp
QFile file("players.txt");
file.open(QFile::WriteOnly | QFile::Text);
QTextStream stream(&file);
stream << center;
stream << qSetFieldWidth(16) << "Player" << qSetFieldWidth(0) << " ";
stream << qSetFieldWidth(10) << "Experience" << qSetFieldWidth(0) << " ";
stream << qSetFieldWidth(13) << "Position" << qSetFieldWidth(0) << " ";
stream << "Direction" << endl;

for(const Player &player: players) {
    stream << left << qSetFieldWidth(16) << player.name
           << qSetFieldWidth(0) << " ";
    stream << right << qSetFieldWidth(10) << player.experience
           << qSetFieldWidth(0) << " ";
    stream << right << qSetFieldWidth(6) << player.position.x()
           << qSetFieldWidth(0) << " ";
    stream << qSetFieldWidth(6) << player.position.y()
           << qSetFieldWidth(0) << " ";
    stream << center << qSetFieldWidth(10);

    switch(player.direction) {
    case 'n' : stream << "north"; break;
    case 's' : stream << "south"; break;
    case 'e' : stream << "east"; break;
    case 'w' : stream << "west"; break;
    default: stream << "unknown"; break;
    }
    stream << qSetFieldWidth(0) << endl;
}
```

程序创建的文件应该看起来像这样：

```cpp
     Player      Experience   Position    Direction
Gondael               46783     10     -5   north   
Olrael               123648     -5    103    east   
Nazaal             99372641     48    634   south   
```

关于`QTextStream`的最后一件事是，它可以操作标准的 C 文件结构，这使得我们能够使用`QTextStream`，例如，写入`stdout`或从`stdin`读取，如下面的代码所示：

```cpp
QTextStream stdoutStream(stdout);
stdoutStream << "This text goes to standard output." << endl; 
```

# 二进制流

更多的时候，我们必须以设备无关的方式存储对象数据，以便以后可以恢复，可能在不同的机器上，具有不同的数据布局等等。在计算机科学中，这被称为**序列化**。Qt 提供了几种序列化机制，现在我们将简要地看看其中的一些。

如果你从远处看`QTextStream`，你会注意到它真正做的是将数据序列化和反序列化到文本格式。它的近亲是`QDataStream`类，它处理任意数据的序列化和反序列化到二进制格式。它使用自定义数据格式以平台无关的方式从`QIODevice`存储和检索数据。它存储足够的数据，以便在一个平台上写入的流可以在不同的平台上成功读取。

`QDataStream`的使用方式与`QTextStream`类似——使用`<<`和`>>`运算符将数据重定向到或从流中。该类支持大多数内置的 Qt 数据类型，因此你可以直接操作`QColor`、`QPoint`或`QStringList`等类：

```cpp
QFile file("outfile.dat");
file.open(QFile::WriteOnly | QFile::Truncate);
QDataStream stream(&file);
double dbl = 3.14159265359;
QColor color = Qt::red;
QPoint point(10, -4);
QStringList stringList { "foo", "bar" };
stream << dbl << color << point << stringList; 
```

如果你想要序列化自定义数据类型，你可以通过实现适当的重定向运算符来教会`QDataStream`如何做到这一点。

# 行动时间 - 自定义结构的序列化

让我们通过实现使用`QDataStream`序列化包含我们用于文本流中的玩家信息的简单结构的函数来进行另一个小练习：

```cpp
struct Player {
    QString name;
    qint64 experience;
    QPoint position;
    char direction;
}; 
```

为了做到这一点，需要实现两个函数，这两个函数都返回一个`QDataStream`引用，该引用是作为调用参数传递的。除了流本身之外，序列化运算符接受一个对正在保存的类的常量引用。最简单的实现是将每个成员流式传输到流中，然后返回流：

```cpp
QDataStream& operator<<(QDataStream &stream, const Player &p) {
    stream << p.name;
    stream << p.experience;
    stream << p.position;
    stream << p.direction;
    return stream;
} 
```

与此同时，反序列化是通过实现一个接受从流中读取的数据填充的结构可变引用的重新定向运算符来完成的：

```cpp
QDataStream& operator>>(QDataStream &stream, Player &p) {
     stream >> p.name;
     stream >> p.experience;
     stream >> p.position;
     stream >> p.direction;
     return stream;
} 
```

同样，最后，流本身被返回。

现在，我们可以使用`QDataStream`将我们的对象写入任何 I/O 设备（例如，文件、缓冲区或网络套接字）：

```cpp
Player player = /* ... */;
QDataStream stream(device);
stream << player;
```

读取对象同样简单：

```cpp
Player player;
QDataStream stream(device);
stream >> player;
```

# 刚才发生了什么？

我们提供了两个独立的函数，用于为`Player`类定义到`QDataStream`实例的重新定向运算符。这使得你的类可以使用 Qt 提供和使用的机制进行序列化和反序列化。

# XML 流

XML 已经成为用于存储层次化数据的最受欢迎的标准之一。尽管它冗长且难以用肉眼阅读，但它几乎在需要数据持久化的任何领域都被使用，因为它非常容易由机器读取。Qt 提供了两个模块来支持读取和写入 XML 文档：

+   Qt Xml 模块通过`QDomDocument`、`QDomElement`等类提供使用**文档对象模型**（**DOM**）标准的访问。

+   Qt Core 模块包含`QXmlStreamReader`和`QXmlStreamWriter`类，它们实现了流式 API。

`QDomDocument`的一个缺点是它要求我们在解析之前将整个 XML 树加载到内存中。此外，Qt Xml 没有积极维护。因此，我们将专注于 Qt Core 提供的流式方法。

在某些情况下，与流式方法相比，DOM 方法的缺点可以通过其易用性得到补偿，所以如果你觉得你已经找到了适合它的任务，你可以考虑使用它。如果你想在 Qt 中使用 DOM 访问 XML，记得在项目配置文件中添加`QT += xml`行以启用`QtXml`模块。

# 是时候采取行动 - 实现玩家数据的 XML 解析器

在这个练习中，我们将创建一个解析器来填充表示 RPG 游戏中玩家及其库存的数据。首先，让我们创建将保存数据的类型：

```cpp
class InventoryItem {
    Q_GADGET
public:
    enum class Type {
        Weapon,
        Armor,
        Gem,
        Book,
        Other
    };
    Q_ENUM(Type)

    Type type;
    QString subType;
    int durability;

    static Type typeByName(const QStringRef &r);
};

class Player {
public:
    QString name;
    QString password;
    int experience;
    int hitPoints;
    QVector<InventoryItem> inventory;
    QString location;
    QPoint position;
};

struct PlayerInfo {
    QVector<Player> players;
};
```

# 刚才发生了什么？

我们想在我们的枚举上使用`Q_ENUM`宏，因为它将使我们能够轻松地将枚举值转换为字符串并反向转换，这对于序列化非常有用。由于`InventoryItem`不是`QObject`，我们需要在类声明开头添加`Q_GADGET`宏，以便`Q_ENUM`宏能够工作。将`Q_GADGET`视为`Q_OBJECT`的一个轻量级变体，它启用了一些功能但不是全部。

`typeByName()`方法将接收一个字符串并返回相应的枚举变体。我们可以按以下方式实现此方法：

```cpp
InventoryItem::Type InventoryItem::typeByName(const QStringRef &r) {
    QMetaEnum metaEnum = QMetaEnum::fromType<InventoryItem::Type>();
    QByteArray latin1 = r.toLatin1();
    int result = metaEnum.keyToValue(latin1.constData());
    return static_cast<InventoryItem::Type>(result);
}
```

实现可能看起来很复杂，但它比手动编写一大堆 `if` 语句来手动选择正确的返回值要少出错。首先，我们使用 `QMetaEnum::fromType<T>()` 模板方法来获取与我们的 `enum` 对应的 `QMetaEnum` 对象。此对象的 `keyToValue()` 方法执行我们需要的转换，但它需要伴随一些转换。

你可以注意到我们正在使用一个名为 `QStringRef` 的类。它代表一个字符串引用——现有字符串中的子串——并且以避免昂贵的字符串构造的方式实现；因此，它非常快。类似的 `std::string_view` 类型是在 C++17 中添加到标准库中的。我们将其用作参数类型，因为 `QXmlStreamReader` 将以这种格式提供字符串。

然而，`keyToValue()` 方法期望一个 `const char *` 参数，所以我们使用 `toLatin1()` 方法将我们的字符串转换为 `QByteArray`，然后使用 `constData()` 获取其缓冲区的 `const char *` 指针。最后，我们使用 `static_cast` 将结果从 `int` 转换为我们的 `enum` 类型。

将以下 XML 文档保存在某个地方。我们将使用它来测试解析器是否可以读取它：

```cpp
<PlayerInfo>
    <Player hp="40" exp="23456">
        <Name>Gandalf</Name>
        <Password>mithrandir</Password>
        <Inventory>
            <InvItem type="Weapon" durability="3">
                <SubType>Long sword</SubType>
            </InvItem>
            <InvItem type="Armor" durability="10">
                <SubType>Chain mail</SubType>
            </InvItem>
        </Inventory>
        <Location name="room1">
            <Position x="1" y="0"/>
        </Location>
    </Player>
</PlayerInfo> 
```

让我们创建一个名为 `PlayerInfoReader` 的类，该类将封装 `QXmlStreamReader` 并为 `PlayerInfo` 实例提供解析器接口：

```cpp
class PlayerInfoReader {
public:
    PlayerInfoReader(QIODevice *device);
    PlayerInfo read();
private:
    QXmlStreamReader reader;
}; 
```

类构造函数接受一个 `QIODevice` 指针，该指针将用于读取所需的数据。构造函数很简单，因为它只是将设备传递给 `reader` 对象：

```cpp
PlayerInfoReader(QIODevice *device) {
    reader.setDevice(device);
} 
```

在我们开始解析之前，让我们准备一些代码来帮助我们处理这个过程。首先，让我们向类中添加一个枚举类型，该类型将列出所有可能的令牌——我们希望在解析器中处理的标签名称：

```cpp
enum class Token {
    Invalid = -1,
    PlayerInfo, // root tag
    Player,     // in PlayerInfo
    Name, Password, Inventory, Location, // in Player
    Position,   // in Location
    InvItem     // in Inventory
};
```

然后，就像我们在 `InventoryItem` 类中所做的那样，我们使用 `Q_GADGET` 和 `Q_ENUM` 宏，并实现 `PlayerInfoReader::tokenByName()` 便利方法。

现在，让我们实现解析过程的入口点：

```cpp
PlayerInfo PlayerInfoReader::read() {
    if(!reader.readNextStartElement()) {
        return PlayerInfo();
    }
    if (tokenByName(reader.name()) != Token::PlayerInfo) {
        return PlayerInfo();
    }
    PlayerInfo info;
    while(reader.readNextStartElement()) {
        if(tokenByName(reader.name()) == Token::Player) {
            Player p = readPlayer();
            info.players.append(p);
        } else {
            reader.skipCurrentElement();
        }
    }
    return info;
}
```

首先，我们在读取器上调用 `readNextStartElement()`，使其找到第一个元素的起始标签，如果找到了，我们检查文档的根标签是否是我们期望的。如果不是，我们返回一个默认构造的 `PlayerInfo`，表示没有可用的数据。

接下来，我们创建一个 `PlayerInfo` 变量。我们遍历当前标签（`PlayerInfo`）中的所有起始子元素。对于每个子元素，我们检查它是否是 `Player` 标签，并调用 `readPlayer()` 下降到解析单个玩家数据的级别。否则，我们调用 `skipCurrentElement()`，这将快速前进流，直到遇到匹配的结束元素。

这个类中的其他方法通常遵循相同的模式。每个解析方法迭代所有起始元素，处理它所知道的元素，并忽略所有其他元素。这种做法使我们能够保持向前兼容性，因为较新版本的文档中引入的所有标签都会被较旧的解析器静默跳过。

`readPlayer()`的结构与之前相似；然而，它更复杂，因为我们还想要从`Player`标签本身的属性中读取数据。让我们逐部分查看这个函数。首先，我们获取与打开标签关联的属性列表，并请求我们感兴趣的两种属性的值：

```cpp
Player p;
const QXmlStreamAttributes& playerAttrs = reader.attributes();
p.hitPoints = playerAttrs.value("hp").toString().toInt();
p.experience = playerAttrs.value("exp").toString().toInt();
```

之后，我们循环所有子标签，并根据标签名称填充`Player`结构。通过将标签名称转换为标记，我们可以使用`switch`语句来整洁地组织代码，以便从不同的标签类型中提取信息，如下面的代码所示：

```cpp
while(reader.readNextStartElement()) {
    Token t = tokenByName(reader.name());
    switch(t) {
    case Token::Name:
        p.name = reader.readElementText();
        break;
    case Token::Password:
        p.password = reader.readElementText();
        break;
    case Token::Inventory:
        p.inventory = readInventory();
        break;
    //...
    }
}
```

如果我们对标签的文本内容感兴趣，我们可以使用`readElementText()`来提取它。此方法读取直到遇到关闭标签，并返回其内的文本。对于`Inventory`标签，我们调用专门的`readInventory()`方法。

对于`Location`标签，代码比之前更复杂，因为我们再次进入读取子标签，提取所需信息并跳过所有未知标签：

```cpp
case Token::Location:
    p.location = reader.attributes().value("name").toString();
    while(reader.readNextStartElement()) {
        if(tokenByName(reader.name()) == Token::Position) {
            const QXmlStreamAttributes& attrs = reader.attributes();
            p.position.setX(attrs.value("x").toString().toInt());
            p.position.setY(attrs.value("y").toString().toInt());
            reader.skipCurrentElement();
        } else {
            reader.skipCurrentElement();
        }
    }
    break;
```

接下来，我们再次跳过与任何已知标记不匹配的标签。在`readPlayer()`的末尾，我们简单地返回填充好的`Player`值。

最后一种方法在结构上与之前的方法类似——迭代所有标签，跳过我们不想处理的任何内容（即不是库存项目的所有内容），填充库存项目数据结构，并将项目追加到已解析项目列表中，如下所示：

```cpp
QVector<InventoryItem> PlayerInfoReader::readInventory() {
    QVector<InventoryItem> inventory;
    while(reader.readNextStartElement()) {
        if(tokenByName(reader.name()) != Token::InvItem) {
            reader.skipCurrentElement();
            continue;
        }
        InventoryItem item;
        const QXmlStreamAttributes& attrs = reader.attributes();
        item.durability = attrs.value("durability").toString().toInt();
        item.type = InventoryItem::typeByName(attrs.value("type"));
        while(reader.readNextStartElement()) {
            if(reader.name() == "SubType") {
                item.subType = reader.readElementText();
            }
            else {
                reader.skipCurrentElement();
            }
        }
        inventory << item;
    }
    return inventory;
}
```

在你的项目的`main()`函数中，编写一些代码来检查解析器是否工作正常。你可以使用`qDebug()`语句来输出列表的大小和变量的内容。以下代码是一个示例：

```cpp
QFile file(filePath);
file.open(QFile::ReadOnly | QFile::Text);
PlayerInfoReader reader(&file);
PlayerInfo playerInfo = reader.read();
if (!playerInfo.players.isEmpty()) {
    qDebug() << "Count:" << playerInfo.players.count();
    qDebug() << "Size of inventory:" <<
                playerInfo.players.first().inventory.size();
    qDebug() << "Inventory item:"
             << playerInfo.players.first().inventory[0].type
             << playerInfo.players.first().inventory[0].subType;
    qDebug() << "Room:" << playerInfo.players.first().location
             << playerInfo.players.first().position;
}
```

# 刚才发生了什么？

你刚才编写的代码实现了 XML 数据的完整自顶向下解析器。首先，数据通过一个标记化器，它返回比字符串更容易处理的标识符。然后，每个方法都可以轻松检查它接收到的标记是否是当前解析阶段的可接受输入。根据子标记，确定下一个解析函数，并解析器下降到较低级别，直到没有下降的地方。然后，流程向上回退一级并处理下一个子元素。如果在任何时刻发现未知标签，它将被忽略。这种方法支持一种情况，即新版本的软件引入了新的标签到文件格式规范中，但旧版本的软件仍然可以通过跳过它不理解的标签来读取文件。

# 尝试一下英雄——玩家数据的 XML 序列化器

现在你已经知道了如何解析 XML 数据，你可以创建互补的部分——一个模块，它将使用`QXmlStreamWriter`将`PlayerInfo`结构序列化到 XML 文档中。为此，你可以使用`writeStartDocument()`、`writeStartElement()`、`writeCharacters()`和`writeEndElement()`等方法。验证使用你的代码保存的文档是否可以使用我们共同实现的代码进行解析。

# QVariant

`QVariant`是一个可以持有多种类型值的类：

```cpp
QVariant intValue = 1;
int x = intValue.toInt();
QVariant stringValue = "ok";
QString y = stringValue.toString();
```

当你将一个值赋给一个`QVariant`对象时，该值及其类型信息将存储在其中。你可以使用它的`type()`方法来找出它持有哪种类型的值。`QVariant`的默认构造函数创建了一个无效值，你可以使用`isValid()`方法来检测它。

`QVariant`支持大量类型，包括 Qt 值类型，如`QDateTime`、`QColor`和`QPoint`。你还可以注册自己的类型以将它们存储在`QVariant`中。`QVariant`最强大的功能之一是能够存储值集合或值的层次结构。你可以使用`QVariantList`类型（它是`QList<QVariant>`的`typedef`）来创建一个`QVariant`对象的列表，并且你实际上可以将整个列表放入一个单一的`QVariant`对象中！你将能够检索列表并检查单个值：

```cpp
QVariant listValue = QVariantList { 1, "ok" };
for(QVariant item: listValue.toList()) {
  qDebug() << item.toInt() << item.toString();
}
```

同样，你可以使用`QVariantMap`或`QVariantHash`来创建一个具有`QString`键和`QVariant`值的键值集合。不用说，你还可以将这样的集合存储在一个单一的`QVariant`中。这允许你构建深度无限且结构任意的层次结构。

正如你所见，`QVariant`是一个相当强大的类，但我们如何用它进行序列化呢？首先，`QVariant`由`QDataStream`支持，因此你可以使用之前描述的二进制序列化来序列化和恢复你构造的任何`QVariant`值。例如，你不必将你的结构体中的每个字段放入`QDataStream`，你可以将它们放入一个`QVariantMap`，然后将其放入流中：

```cpp
Player player;
QVariantMap map;
map["name"] = player.name;
map["experience"] = player.experience;
//...
stream << map;
```

加载数据也是直截了当的：

```cpp
QVariantMap map;
stream >> map;
Player player;
player.name = map["name"].toString();
player.experience = map["experience"].toLongLong();
```

这种方法允许你在任意位置存储任意数据。然而，你也可以使用`QVariant`和`QSettings`一起方便地将数据存储在适当的位置。

# QSettings

虽然这并不是严格意义上的序列化问题，但存储应用程序设置的方面与描述的主题密切相关。Qt 为此提供了一个解决方案，即`QSettings`类。默认情况下，它在不同的平台上使用不同的后端，例如在 Windows 上使用系统注册表或在 Linux 上使用 INI 文件。`QSettings`的基本使用非常简单——你只需要创建对象并使用`setValue()`和`value()`来存储和加载数据：

```cpp
QSettings settings;
settings.setValue("level", 4);
settings.setValue("playerName", "Player1");
// ...
int level = settings.value("level").toInt(); 
```

你需要记住的唯一一件事是它操作的是`QVariant`，所以如果需要，返回值需要转换为适当的类型，就像前面代码中的`toInt()`一样。如果请求的键不在映射中，`value()`调用可以接受一个额外的参数，该参数包含要返回的值。这允许你在应用程序首次启动且设置尚未保存的情况下处理默认值，例如：

```cpp
int level = settings.value("level", 1).toInt(); 
```

如果你没有指定默认值，当没有存储任何内容时，将返回一个无效的`QVariant`，你可以使用`isValid()`方法来检查这一点。

为了确保默认设置位置正确，你需要设置组织名称和应用程序名称。它们决定了`QSettings`默认存储数据的确切位置，并确保存储的数据不会与其他应用程序冲突。这通常在`main()`函数的开始处完成：

```cpp
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("Packt");
    QCoreApplication::setApplicationName("Game Programming using Qt");
    //...
}
```

# 设置层次结构

最简单的情况假设设置是“扁平”的，即所有键都在同一级别上定义。然而，这不必是这种情况——相关的设置可以放入命名的组中。要操作一个组，你可以使用`beginGroup()`和`endGroup()`调用：

```cpp
settings.beginGroup("server");
QString serverIP = settings.value("host").toString();
int port = settings.value("port").toInt();
settings.endGroup();
```

当使用这种语法时，你必须记住在完成操作后结束组。做同样事情的一种替代方法是直接将组名传递给`value()`的调用，使用`/`来分隔它和值名：

```cpp
QString serverIP = settings.value("server/host").toString();
int port = settings.value("server/port").toInt(); 
```

你可以通过多次调用`beginGroup()`（或者，等价地，在值名中写入多个斜杠）来创建多个嵌套组。

向`QSettings`引入非扁平结构还有另一种方法。它可以处理复合`QVariant`值——`QVariantMap`和`QVariantList`。你可以简单地将你的数据转换为`QVariant`，就像我们之前将其转换为`QJsonValue`一样：

```cpp
QVariant inventoryItemToVariant(const InventoryItem &item) {
    QVariantMap map;
    map["type"]       = InventoryItem::typeToName(item.type);
    map["subtype"]    = item.subType;
    map["durability"] = item.durability;
    return map;
}
```

这个`QVariant`值可以传递给`QSettings::setValue()`。当然，你还需要实现逆操作。更重要的是，没有任何阻止你将数据转换为 JSON 并将其作为`QByteArray`保存到`QSettings`中。然而，这些方法可能比适当的序列化要慢，并且生成的设置文件难以手动编辑。

各种 Qt 类都有旨在与`QSettings`一起使用的方法，以便轻松保存一组属性。例如，`QWidget::saveGeometry()`和`QWidget::restoreGeometry()`辅助函数允许你将窗口的位置和大小保存到`QSettings`：

```cpp
settings.setValue("myWidget/geometry", myWidget->saveGeometry());
//...
myWidget->restoreGeometry(
    settings.value("myWidget/geometry").toByteArray());
```

类似地，多个小部件类有`saveState()`和`restoreState()`方法来保存小部件状态的信息：

+   `QMainWindow`可以保存工具栏和停靠小部件的位置

+   `QSplitter`可以保存其手柄的位置

+   `QHeaderView`可以保存表格的行或列的大小

+   `QFileDialog`可以保存对话框的布局、历史记录和当前目录

这些方法是保留用户在应用程序界面中做出的所有更改的绝佳方式。

# 自定义设置位置和格式

`QSettings` 类的构造函数有多个重载，允许您通过特定的 `QSettings` 对象更改数据存储的位置，而不是使用默认位置。首先，您可以覆盖组织名称和应用程序名称：

```cpp
QSettings settings("Packt", "Game Programming using Qt"); 
```

接下来，您可以通过传递 `QSettings::SystemScope` 作为 `scope` 参数来使用系统范围的存储位置：

```cpp
QSettings settings(QSettings::SystemScope, 
    "Packt", "Game Programming using Qt");
```

在此情况下，`QSettings` 将尝试读取所有用户的设置，然后回退到用户特定的位置。请注意，系统范围的存储位置可能不可写，因此在该位置上使用 `setValue()` 不会产生预期效果。

您还可以使用 `QSettings::setDefaultFormat()` 函数禁用首选格式检测。例如，在 Windows 上，使用以下代码禁用使用注册表：

```cpp
QSettings::setDefaultFormat(QSettings::IniFormat);
```

最后，还有一个选项可以完全控制设置数据的位置——直接告诉构造函数数据应该位于何处：

```cpp
QSettings settings(
    QStandardPaths::writableLocation(QStandardPaths::ConfigLocation) +
        "/myapp.ini", 
    QSettings::IniFormat
); 
```

如果您将 `QSettings::NativeFormat` 传递给此构造函数，路径的含义将取决于平台。例如，在 Windows 上，它将被解释为注册表路径。

由于您可以使用 `QSettings` 读取和写入任意 INI 文件，因此它是实现对象序列化为 INI 格式的方便且简单的方法，这在简单情况下是合适的。

`QSettings` 还允许您注册自己的格式，以便您可以控制设置存储的方式，例如，通过使用 XML 存储或添加即时加密。这是通过使用 `QSettings::registerFormat()` 实现的，您需要传递文件扩展名和两个函数指针，分别用于读取和写入设置，如下所示：

```cpp
bool readCCFile(QIODevice &device, QSettings::SettingsMap &map) {
    CeasarCipherDevice ccDevice;
    ccDevice.setBaseDevice(&device);
    // ...
    return true;
}
bool writeCCFile(QIODevice &device, const QSettings::SettingsMap &map) { 
    // ... 
}
const QSettings::Format CCFormat = QSettings::registerFormat(
    "ccph", readCCFile, writeCCFile);
```

# JSON 文件

**JSON** 代表“JavaScript 对象表示法”，这是一种流行的轻量级文本格式，用于以可读的形式存储面向对象的数据。它起源于 JavaScript，在那里它是存储对象信息的原生格式；然而，它被广泛应用于许多编程语言，并且是网络数据交换的流行格式。Qt Core 支持 JSON 格式，如下面的代码所示。一个简单的 JSON 格式定义如下：

```cpp
{
    "name": "Joe",
    "age": 14,
    "inventory": [
        { "type": "gold", "amount": "144000" },
        { "type": "short_sword", "material": "iron" }
    ]
} 
```

JSON 对象可以包含以下类型的值：

| **类型** | **描述** |
| --- | --- |
| bool | 布尔值（`true` 或 `false`）。 |
| double | 一个数值（例如，`42.1`）。 |
| string | 引号中的字符串（例如，`"Qt"`）。 |
| array | 用方括号括起来的任何类型的值集合（例如，`[42.1, "Qt"]`）。 |
| object | 用大括号括起来的键值对集合。键是字符串，值可以是任何类型（例如，`{ "key1": 42.1, "key2": [42.1, "Qt"] }`）。 |
| null | 表示数据缺失的特殊值（`null`）。 |

一个合适的 **JSON 文档** 必须在顶层有一个数组或对象。在前面的例子中，我们有一个包含三个属性的对象：name、age 和 inventory。前两个属性是简单值，最后一个属性是一个包含两个对象且每个对象有两个属性的数组。

Qt 可以使用 `QJsonDocument` 类创建和读取 JSON 描述。可以使用 `QJsonDocument::fromJson()` 静态方法从 UTF-8 编码的文本创建文档，并且可以使用 `toJson()` 方法再次将其存储为文本形式。一旦创建了一个 JSON 文档，就可以使用 `isArray()` 和 `isObject()` 调用之一来检查它是否表示一个对象或数组。然后，可以使用 `array()` 或 `object()` 方法将文档转换为 `QJsonArray` 或 `QJsonObject`。

由于 JSON 的结构紧密地类似于 `QVariant`（它也可以使用 `QVariantMap` 来存储键值对，使用 `QVariantList` 来存储数组），因此也存在转换方法 `QJsonDocument::fromVariant()` 和 `QJsonDocument::toVariant()`。

`QJsonObject` 是一种可迭代的类型，可以查询其键列表（使用 `keys()` 方法）或请求特定键的值（使用 `value()` 方法或 `operator[]`）。值使用 `QJsonValue` 类表示，它可以存储前面列出的任何值类型。可以使用 `insert()` 方法向对象添加新属性，该方法接受一个字符串键和一个作为 `QJsonValue` 的值。可以使用 `remove()` 方法删除现有属性。

`QJsonArray` 也是一种可迭代的类型，它包含一个经典列表 API；它包含 `append()`、`insert()`、`removeAt()`、`at()` 和 `size()` 等方法来操作数组中的条目，再次以 `QJsonValue` 作为项目类型。

# 行动时间 - 玩家数据 JSON 序列化器

我们接下来的练习是创建一个序列化器，其结构与我们在 XML 练习中使用的 `PlayerInfo` 结构相同，但这次的目标数据格式将是 JSON。

首先，创建一个 `PlayerInfoJson` 类，并给它一个类似于以下代码的接口：

```cpp
class PlayerInfoJson {
public:
    PlayerInfoJson() {}
    QByteArray playerInfoToJson(const PlayerInfo &pinfo);
}; 
```

实际上所需做的就是实现 `playerInfoToJson` 方法。通常，我们需要将我们的 `PlayerInfo` 数据转换为 `QJsonArray`，然后使用 `QJsonDocument` 将其编码为 JSON：

```cpp
QByteArray PlayerInfoJson::playerInfoToJson(const PlayerInfo &pinfo)
{
    QJsonDocument doc(toJson(pinfo));
    return doc.toJson();
}
```

现在，让我们开始实现 `toJson()` 方法：

```cpp
QJsonArray PlayerInfoJson::toJson(const PlayerInfo &pinfo) {
    QJsonArray array;
    for(const Player &p: pinfo.players) {
        array << toJson(p);
    }
    return array;
}
```

由于结构实际上是一个玩家列表，我们可以遍历它，将每个玩家转换为 `QJsonValue`，并将结果追加到 `QJsonArray` 中。有了这个函数，我们就可以向下级实现 `toJson()` 的重载，它接受一个 `Player` 对象：

```cpp
QJsonValue PlayerInfoJson::toJson(const Player &player) {
    QJsonObject object;
    object["name"]       = player.name;
    object["password"]   = player.password;
    object["experience"] = player.experience;
    object["hitpoints"]  = player.hitPoints;
    object["location"]   = player.location;
    object["position"]   = QJsonObject({ { "x", player.position.x() },
                                         { "y", player.position.y() } });
    object["inventory"]  = toJson(player.inventory);
    return object;
}
```

这次，我们使用 `QJsonObject` 作为我们的基本类型，因为我们想将值与键关联起来。对于每个键，我们使用索引操作符向对象添加条目。位置键包含一个 `QPoint` 值，这不是一个有效的 JSON 值，因此我们使用 C++11 初始化列表将点转换为包含两个键（`x` 和 `y`）的 `QJsonObject`。情况与存货不同——我们再次需要为 `toJson` 编写一个重载，以便执行转换：

```cpp
QJsonValue PlayerInfoJson::toJson(const QVector<InventoryItem> &items) {
    QJsonArray array;
    for(const InventoryItem &item: items) {
        array << toJson(item);
    }
    return array;
}
```

代码几乎与处理 `PlayerInfo` 对象的代码相同，所以让我们关注 `toVariant` 的最后一个重载——接受 `Item` 实例的那个：

```cpp
QJsonValue PlayerInfoJson::toJson(const InventoryItem &item) {
    QJsonObject object;
    object["type"] = InventoryItem::typeToName(item.type);
    object["subtype"] = item.subType;
    object["durability"] = item.durability;
    return object;
}
```

这里没有太多可评论的——我们向对象添加所有键，将项目类型转换为字符串。为此，我们必须添加一个静态的 `InventoryItem::typeToName()` 方法，它是 `typeByName()` 的反向操作，即它接受枚举变体并输出其名称作为字符串：

```cpp
const char *InventoryItem::typeToName(InventoryItem::Type value)
{
    QMetaEnum metaEnum = QMetaEnum::fromType<InventoryItem::Type>();
    return metaEnum.valueToKey(static_cast<int>(value));
}
```

这基本上是 `QMetaEnum::valueToKey()` 方法的包装，它执行所有不可能在没有 Qt 的情况下完成的魔法。

序列化器已经完成！现在你可以使用 `PlayerInfoJson::playerInfoToJson()` 将 `PlayerInfo` 转换为包含 JSON 的 `QByteArray`。它适合写入文件或通过网络发送。然而，为了使其更有用，我们需要实现反向操作（反序列化）。

# 行动时间 - 实现一个 JSON 解析器

让我们扩展 `PlayerInfoJSON` 类，并为其添加一个 `playerInfoFromJson()` 方法：

```cpp
PlayerInfo PlayerInfoJson::playerInfoFromJson(const QByteArray &ba) {
    QJsonDocument doc = QJsonDocument::fromJson(ba);
    if(!doc.isArray()) {
        return PlayerInfo();
    }
    QJsonArray array = doc.array();
    PlayerInfo pinfo;
    for(const QJsonValue &value: array) {
        pinfo.players << playerFromJson(value.toObject());
    }
    return pinfo;
}
```

首先，我们读取文档并检查它是否有效以及是否包含预期的数组。如果失败，则返回一个空结构；否则，我们遍历接收到的数组并将每个元素转换为对象。类似于序列化示例，我们为我们的数据结构中的每个复杂项创建一个辅助函数。因此，我们编写一个新的 `playerFromJson()` 方法，将 `QJsonObject` 转换为 `Player`，即与 `toJson(Player)` 相比执行反向操作：

```cpp
Player PlayerInfoJson::playerFromJson(const QJsonObject &object) {
    Player player;
    player.name       = object["name"].toString();
    player.password   = object["password"].toString();
    player.experience = object["experience"].toDouble();
    player.hitPoints  = object["hitpoints"].toDouble();
    player.location   = object["location"].toString();
    QJsonObject positionObject = object["position"].toObject();
    player.position   = QPoint(positionObject["x"].toInt(),
                               positionObject["y"].toInt());
    player.inventory  = inventoryFromJson(object["inventory"].toArray());
    return player;
}
```

在这个函数中，我们使用了 `operator[]` 从 `QJsonObject` 中提取数据，然后使用不同的函数将数据转换为所需的类型。请注意，为了转换为 `QPoint`，我们首先将其转换为 `QJsonObject`，然后提取值，在它们用于构建 `QPoint` 之前使用。在每种情况下，如果转换失败，我们将为该类型获取一个默认值（例如，一个空字符串或一个零数字）。为了读取存货，我们采用另一个自定义方法：

```cpp
QVector<InventoryItem> PlayerInfoJson::inventoryFromJson(
    const QJsonArray &array) 
{
    QVector<InventoryItem> inventory;
    for(const QJsonValue &value: array) {
      inventory << inventoryItemFromJson(value.toObject());
    }
    return inventory;
}
```

剩下的就是实现 `inventoryItemFromJson()`：

```cpp
InventoryItem PlayerInfoJson::inventoryItemFromJson(
    const QJsonObject &object) 
{
    InventoryItem item;
    item.type = InventoryItem::typeByName(object["type"].toString());
    item.subType = object["subtype"].toString();
    item.durability = object["durability"].toDouble();
    return item;
}
```

不幸的是，我们的 `typeByName()` 函数需要 `QStringRef`，而不是 `QString`。我们可以通过添加几个重载并将它们转发到单个实现来修复这个问题：

```cpp
InventoryItem::Type InventoryItem::typeByName(const QStringRef &r) {
 return typeByName(r.toLatin1());
}
InventoryItem::Type InventoryItem::typeByName(const QString &r) {
 return typeByName(r.toLatin1());
}
InventoryItem::Type InventoryItem::typeByName(const QByteArray &latin1) {
    QMetaEnum metaEnum = QMetaEnum::fromType<InventoryItem::Type>();
    int result = metaEnum.keyToValue(latin1.constData());
    return static_cast<InventoryItem::Type>(result);
}
```

# 刚才发生了什么？

实现的类可用于在`Item`实例和包含 JSON 格式对象数据的`QByteArray`对象之间进行双向转换。在这里我们没有进行任何错误检查；相反，我们依赖于 Qt 的规则，即错误会导致一个合理的默认值。

如果你想进行错误检查，在这种情况下最直接的方法是使用异常，因为它们会自动从多个嵌套调用传播到调用者的位置。确保你捕获你抛出的任何异常，否则应用程序将终止。一个更 Qt 的方法是在所有方法（包括内部方法）中创建一个`bool *ok`参数，并在发生任何错误时将该布尔值设置为`false`。

# 快速问答

Q1. 在 Qt 中，`std::string`最接近的等效是什么？

1.  `QString`

1.  `QByteArray`

1.  `QStringLiteral`

Q2. 哪些字符串与`\A\d\z`正则表达式匹配？

1.  由数字组成的字符串

1.  由单个数字组成的字符串

1.  这不是一个有效的正则表达式

Q3. 你可以使用哪种容器类型来存储小部件列表？

1.  `QVector<QWidget>`

1.  `QList<QWidget>`

1.  `QVector<QWidget*>`

Q4. 你可以使用哪个类将包含 JSON 的文本字符串转换为 Qt JSON 表示？

1.  `QJsonValue`

1.  `QJsonObject`

1.  `QJsonDocument`

# 摘要

在本章中，你学习了大量的核心 Qt 技术，从文本操作和容器到使用 XML 或 JSON 等流行技术访问可用于传输或存储数据的设备。你应该意识到，我们只是触及了 Qt 所能提供的皮毛，还有许多其他有趣的类你应该熟悉，但这个最小量的信息应该能让你领先一步，并展示你未来研究的方向。

在下一章中，我们将超越你电脑的边界，探索使用现代互联网强大世界的方法。你将学习如何与现有的网络服务交互，检查当前网络可用性，并实现你自己的服务器和客户端。如果你想要实现多人网络游戏，这些知识将非常有用。
