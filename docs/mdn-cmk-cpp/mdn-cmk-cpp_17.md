# 附录：杂项命令

每种语言都有许多实用的命令，CMake 在这方面也不例外：它提供了进行简单算术、位运算、字符串处理、列表和文件操作的工具。有趣的是，它们必要性相对较少（感谢过去几年中所有的改进和编写模块），但在更自动化的项目中仍然可能需要。

因此，本附录是对各种命令及其多种模式的一个简要总结。将其视为一个方便的离线参考或官方文档的简化版。如果您需要更多信息，请访问提供的链接。

在本章中，我们将涵盖以下主要主题：

+   `string()`命令

+   `list()`命令

+   `file()`命令

+   `math()`命令

# 字符串()命令

`string()`命令用于操作字符串。它有多种模式，可以执行字符串的不同操作：搜索和替换、 manipulation、比较、散列、生成和 JSON 操作（从 CMake 3.19 开始提供后一种）。

完整细节请在线查阅文档：[`cmake.org/cmake/help/latest/command/string.html`](https://cmake.org/cmake/help/latest/command/string.html)。

接受`<input>`参数的`string()`模式将接受多个`<input>`值，并在命令执行前将它们连接起来：

```cpp
string(PREPEND myVariable "a" "b" "c")
```

这相当于以下内容：

```cpp
string(PREPEND myVariable "abc")
```

让我们探索所有可用的`string()`模式。

## 搜索和替换

以下模式可供使用：

+   `string(FIND <haystack> <pattern> <out> [REVERSE])`在`<haystack>`字符串中搜索`<pattern>`，并将找到的位置作为整数写入`<out>`变量。如果使用了`REVERSE`标志，它从字符串的末尾开始搜索到开头。这仅适用于 ASCII 字符串（不提供多字节支持）。

+   `string(REPLACE <pattern> <replace> <out> <input>)`将`<input>`中的所有`<pattern>`替换为`<replace>`，并将结果存储在`<out>`变量中。

+   `string(REGEX MATCH <pattern> <out> <input>)`使用正则表达式在`<input>`中匹配第一个出现的`<pattern>`，并将其存储在`<out>`变量中。

+   `string(REGEX MATCHALL <pattern> <out> <input>)`使用正则表达式在`<input>`中匹配所有出现的`<pattern>`，并将其作为逗号分隔的列表存储在`<out>`变量中。

+   `string(REGEX REPLACE <pattern> <replace> <out> <input>)`命令使用正则表达式在`<input>`中替换所有出现的`<pattern>`，并将结果存储在`<out>`变量中。

正则表达式操作遵循 C++标准库中`<regex>`头文件定义的 C++语法。您可以使用捕获组将匹配项添加到带有数字占位符`\\1`、`\\2`...的`<replace>`表达式中：（需要使用双反斜杠，以便正确解析参数）。

## 操作

可用以下模式：

+   `string(APPEND <out> <input>)` 通过附加`<input>`字符串改变存储在`<out>`中的字符串。

+   `string(PREPEND <out> <input>)` 通过在`<out>`中的字符串前添加`<input>`字符串改变这些字符串。

+   `string(CONCAT <out> <input>)` 连接所有提供的`<input>`字符串，并将它们存储在`<out>`变量中。

+   `string(JOIN <glue> <out> <input>)` 使用`<glue>`值交错所有提供的`<input>`字符串，并将它们作为一个连接的字符串存储在`<out>`变量中（不要对列表变量使用此模式）。

+   `string(TOLOWER <string> <out>)` 将`<string>`转换为小写，并将其存储在`<out>`变量中。

+   `string(TOUPPER <string> <out>)` 将`<string>`转换为大写，并将其存储在`<out>`变量中。

+   `string(LENGTH <string> <out>)` 计算`<string>`的字节数，并将结果存储在`<out>`变量中。

+   `string(SUBSTRING <string> <begin> <length> <out>)` 提取`<string>`的`<length>`字节子字符串，从`<begin>`字节开始，并将其存储在`<out>`变量中。将长度设为`-1`则表示“直到字符串结束”。

+   `string(STRIP <string> <out>)` 从`<string>`中移除尾部和前导空白，并将结果存储在`<out>`变量中。

+   `string(GENEX_STRIP <string> <out>)` 移除`<string>`中使用的所有生成器表达式，并将结果存储在`<out>`变量中。

+   `string(REPEAT <string> <count> <out>)` 生成包含`<count>`个`<string>`重复的字符串，并将其存储在`<out>`变量中。

## 比较

字符串的比较采用以下形式：

```cpp
string(COMPARE <operation> <stringA> <stringB> <out>)
```

`<operation>`参数是以下之一：`LESS`、`GREATER`、`EQUAL`、`NOTEQUAL`、`LESS_EQUAL`或`GREATER_EQUAL`。它将用于比较`<stringA>`与`<stringB>`，并将结果（`true`或`false`）存储在`<out>`变量中。

## 散列

散列模式具有以下签名：

```cpp
string(<algorithm> <out> <string>)
```

使用`<algorithm>`散列`<string>`并将结果存储在`<out>`变量中。支持以下算法：

+   `MD5`: 消息摘要算法 5，RFC 1321

+   `SHA1`: 美国安全散列算法 1，RFC 3174

+   `SHA224`: 美国安全散列算法，RFC 4634

+   `SHA256`: 美国安全散列算法，RFC 4634

+   `SHA384`: 美国安全散列算法，RFC 4634

+   `SHA512`: 美国安全散列算法，RFC 4634

+   `SHA3_224`: 凯凯 SHA-3

+   `SHA3_256`: 凯凯 SHA-3

+   `SHA3_384`: 凯凯 SHA-3

+   `SHA3_512`: 凯凯 SHA-3

## 生成

可用以下模式：

+   `string(ASCII <number>... <out>)` 将给定`<number>`的 ASCII 字符存储在`<out>`变量中。

+   `string(HEX <string> <out>)` 将`<string>`转换为其十六进制表示，并将其存储在`<out>`变量中（自 CMake 3.18 起）。

+   `string(CONFIGURE <string> <out> [@ONLY] [ESCAPE_QUOTES])`完全像`configure_file()`一样工作，但用于字符串。结果存储在`<out>`变量中。

+   `string(MAKE_C_IDENTIFIER <string> <out>)` 将 `<string>` 中的非字母数字字符转换为下划线，并将结果存储在 `<out>` 变量中。

+   `string(RANDOM [LENGTH <len>] [ALPHABET <alphabet>] [RANDOM_SEED <seed>] <out>)` 生成一个 `<len>` 个字符（默认 `5`）的随机字符串，使用可选的 `<alphabet>` 从随机种子 `<seed>`，并将结果存储在 `<out>` 变量中。

+   `string(TIMESTAMP <out> [<format>] [UTC])` 生成一个表示当前日期和时间的字符串，并将其存储在 `<out>` 变量中。

+   `string(UUID <out> ...)` 生成一个全球唯一的标识符。这个模式使用起来有点复杂。

## JSON

JSON 格式的字符串操作使用以下签名：

```cpp
string(JSON <out> [ERROR_VARIABLE <error>] <operation +
args>)
```

several operations are available. They all store their results in the `<out>` variable, and errors in the `<error>` variable. Operations and their arguments are as follows:

+   `GET <json> <member|index>...` 返回使用 `<member>` 路径或 `<index>` 从 `<json>` 字符串中获取一个或多个元素的结果。

+   `TYPE <json> <member|index>...` 返回 `<json>` 字符串中使用 `<member>` 路径或 `<index>` 的一个或多个元素的类型。

+   `MEMBER <json> <member|index>... <array-index>` 返回 `<json>` 字符串中 `<array-index>` 位置的一个或多个数组类型元素的成员名称。

+   `LENGTH <json> <member|index>...` 返回 `<json>` 字符串中使用 `<member>` 路径或 `<index>` 的一个或多个数组类型元素的数量。

+   `REMOVE <json> <member|index>...` 返回使用 `<member>` 路径或 `<index>` 从 `<json>` 字符串中删除一个或多个元素的结果。

+   `SET <json> <member|index>... <value>` 返回将 `<value>` 插入到 `<json>` 字符串中一个或多个元素的结果。

+   `EQUAL <jsonA> <jsonB>` 评估 `<jsonA>` 和 `<jsonB>` 是否相等。

# `list()` 命令

该命令提供了列表的基本操作：阅读、搜索、修改和排序。有些模式会改变列表（改变原始值）。如果你之后需要它，请确保复制原始值。

完整详细信息可以在在线文档中找到：

[`cmake.org/cmake/help/latest/command/list.html`](https://cmake.org/cmake/help/latest/command/list.html)

## 阅读

以下模式可用：

+   `list(LENGTH <list> <out>)` 计算 `<list>` 变量的元素数量，并将结果存储在 `<out>` 变量中。

+   `list(GET <list> <index>... <out>)` 将 `<list>` 中指定的索引列表元素复制到 `<out>` 变量中。

+   `list(JOIN <list> <glue> <out>)` 将 `<list>` 元素与 `<glue>` 分隔符交织在一起，并将结果字符串存储在 `<out>` 变量中。

+   `list(SUBLIST <list> <begin> <length> <out>)` 类似于 `GET` 模式，但它操作范围而不是明确的索引。如果 `<length>` 是 `-1`，将从 `<begin>` 索引到 `<list>` 变量提供的列表末尾返回元素。

## 搜索

此模式简单地查找 `<list>` 变量中的 `<needle>` 元素的索引，并将结果存储在 `<out>` 变量中（如果未找到元素，则为 `-1`）：

```cpp
list(FIND <list> <needle> <out>)
```

## 修改

以下是一些可用模式：

+   `list(APPEND <list> <element>...)` 将一个或多个 `<element>` 值添加到 `<list>` 变量的末尾。

+   `list(PREPEND <list> [<element>...])` 类似于 `APPEND`，但它将元素添加到 `<list>` 变量的开头。

+   `list(FILTER <list> {INCLUDE | EXCLUDE} REGEX <pattern>)` 用于过滤 `<list>` 变量，以 `INCLUDE` 或 `EXCLUDE` 包含或排除与 `<pattern>` 值匹配的元素。

+   `list(INSERT <list> <index> [<element>...])` 在给定的 `<index>` 处向 `<list>` 变量添加一个或多个 `<element>` 值。

+   `list(POP_BACK <list> [<out>...])` 从 `<list>` 变量的末尾移除一个元素，并将其存储在可选的 `<out>` 变量中。如果提供了多个 `<out>` 变量，将移除更多元素以填充它们。

+   `list(POP_FRONT <list> [<out>...])` 类似于 `POP_BACK`，但它从 `<list>` 变量的开头移除元素。

+   `list(REMOVE_ITEM <list> <value>...)` 是 `FILTER EXCLUDE` 的简写，但不支持正则表达式。

+   `list(REMOVE_AT <list> <index>...)` 从 `<list>` 中移除特定的 `<index>` 处的元素。

+   `list(REMOVE_DUPLICATES <list>)` 从 `<list>` 中移除重复项。

+   `list(TRANSFORM <list> <action> [<selector>] [OUTPUT_VARIABLE <out>])` 对 `<list>` 元素应用特定的转换。默认情况下，该操作应用于所有元素，但我们也可以通过添加 `<selector>` 来限制其影响。除非提供了 `OUTPUT_VARIABLE` 关键字，否则列表将被突变（原地更改），在这种情况下，结果存储在 `<out>` 变量中。

以下是一些可用的选择器：`AT <index>`、`FOR <start> <stop> [<step>]` 和 `REGEX <pattern>`。

动作包括 `APPEND <string>`、`PREPEND <string>`、`TOLOWER`、`TOUPPER`、`STRIP`、`GENEX_STRIP` 和 `REPLACE <pattern> <expression>`。它们的工作方式与具有相同名称的 `string()` 模式完全相同。

## 排序

以下是一些可用模式：

+   `list(REVERSE <list>)` 简单地反转 `<list>` 的顺序。

+   `list(SORT <list>)` 按字母顺序对列表进行排序。有关更高级选项，请参阅在线手册。

# `file()` 命令

此命令提供了与文件相关的各种操作：读取、传输、锁定和归档。它还提供了检查文件系统和对表示路径的字符串执行操作的模式。

完整详细信息可以在在线文档中找到：

[链接](https://cmake.org/cmake/help/latest/command/file.html)

## 读取

以下是一些可用模式：

+   `file(READ <filename> <out> [OFFSET <o>] [LIMIT <max>] [HEX])` 从 `<filename>` 文件读取到 `<out>` 变量。读取可以从偏移量 `<o>` 开始，并具有可选的字节限制 `<max>`。`HEX` 标志指定输出应转换为十六进制表示。

+   `file(STRINGS <filename> <out>)` 从 `<filename>` 文件中读取字符串到 `<out>` 变量。

+   `file(<algorithm> <filename> <out>)` 从 `<filename>` 文件中计算 `<algorithm>` 哈希值，并将结果存储在 `<out>` 变量中。可用的算法与 `string()` 哈希函数相同。

+   `file(TIMESTAMP <filename> <out> [<format>])` 生成 `<filename>` 文件的时间戳字符串表示，并将其存储在 `<out>` 变量中。可选地接受一个 `<format>` 字符串。

+   `file(GET_RUNTIME_DEPENDENCIES [...])` 为指定文件获取运行时依赖项。这是一个仅在 `install(CODE)` 或 `install(SCRIPT)` 场景中使用的高级命令。

## 编写

以下模式可用：

+   `file({WRITE | APPEND} <filename> <content>...)` 将所有 `<content>` 参数写入或追加到 `<filename>` 文件中。如果提供的系统路径不存在，它将递归创建。

+   `file({TOUCH | TOUCH_NOCREATE} [<filename>...])` 更新 `<filename>` 的时间戳。如果文件不存在，只有在 `TOUCH` 模式下才会创建它。

+   `file(GENERATE OUTPUT <output-file> [...])` 是一个高级模式，为当前 CMake 生成器的每个构建配置生成一个输出文件。

+   `file(CONFIGURE OUTPUT <output-file> CONTENT <content> [...])` 类似于 `GENERATE_OUTPUT`，但它还会通过替换变量占位符来配置生成的文件。

## 文件系统

以下模式可用：

+   `file({GLOB | GLOB_RECURSE} <out> [...] [<globbing-expression>...])` 生成与 `<globbing-expression>` 匹配的文件列表，并将其存储在 `<out>` 变量中。`GLOB_RECURSE` 模式还将扫描嵌套目录。

+   `file(RENAME <oldname> <newname>)` 将文件从 `<oldname>` 移动到 `<newname>`。

+   `file({REMOVE | REMOVE_RECURSE } [<files>...])` 用于删除 `<files>`。`REMOVE_RECURSE` 选项还会删除目录。

+   `file(MAKE_DIRECTORY [<dir>...])` 创建一个目录。

+   `file(COPY <file>... DESTINATION <dir> [...])` 将 `files` 复制到 `<dir>` 目的地。提供过滤、设置权限、符号链接链跟随等功能选项。

+   `file(SIZE <filename> <out>)` 读取 `<filename>` 文件的字节大小，并将其存储在 `<out>` 变量中。

+   `file(READ_SYMLINK <linkname> <out>)` 从 `<linkname>` 符号链接中读取目标路径，并将其存储在 `<out>` 变量中。

+   `file(CREATE_LINK <original> <linkname> [...])` 在 `<linkname>` 位置创建指向 `<original>` 的符号链接。

+   `file({CHMOD|CHMOD_RECURSE} <files>... <directories>... PERMISSIONS <permissions>... [...])` 为文件和目录设置权限。

## 路径转换

以下模式可用：

+   `file(REAL_PATH <path> <out> [BASE_DIRECTORY <dir>])` 从相对路径计算绝对路径，并将其存储在 `<out>` 变量中。可选地接受 `<dir>` 基础目录。该功能自 CMake 3.19 起可用。

+   `file(RELATIVE_PATH <out> <directory> <file>)` 计算 `<file>` 相对于 `<directory>` 的路径，并将其存储在 `<out>` 变量中。

+   `file({TO_CMAKE_PATH | TO_NATIVE_PATH} <path> <out>)` 将 `<path>` 转换为 CMake 路径（目录用正斜杠分隔）或平台的本地路径，并将结果存储在 `<out>` 变量中。

## 传输

以下模式可用：

+   `file(DOWNLOAD <url> [<path>] [...])` 从 `<url>` 下载文件并将其存储在路径中。

+   `file(UPLOAD <file> <url> [...])` 将 `<file>` 上传到 URL。

## 锁定

锁定模式在 `<path>` 资源上放置一个建议锁：

```cpp
file(LOCK <path> [DIRECTORY] [RELEASE]
     [GUARD <FUNCTION|FILE|PROCESS>]
     [RESULT_VARIABLE <out>]
     [TIMEOUT <seconds>])
```

此锁可以可选地作用于 `FUNCTION`、`FILE` 或 `PROCESS`，并带有 `<seconds>` 的超时。要释放锁，请提供 `RELEASE` 关键字。结果将存储在 `<out>` 变量中。

## 归档

归档的创建提供以下签名：

```cpp
file(ARCHIVE_CREATE OUTPUT <destination> PATHS <source>...
  [FORMAT <format>]
  [COMPRESSION <type> [COMPRESSION_LEVEL <level>]]
  [MTIME <mtime>] [VERBOSE])
```

它在 `<destination>` 路径上创建一个存档，包含 `<source>` 文件中的一种支持格式：`7zip`、`gnutar`、`pax`、`paxr`、`raw` 或 `zip`（`paxr` 为默认值）。如果所选格式支持压缩级别，它可以是一个单个数字 `0-9`，其中 `0` 是默认值。

提取模式具有以下签名：

```cpp
file(ARCHIVE_EXTRACT INPUT <archive> [DESTINATION <dir>]
  [PATTERNS <patterns>...] [LIST_ONLY] [VERBOSE])
```

它从 `<archive>` 中提取匹配可选 `<patterns>` 值的文件到目标 `<dir>`。如果提供 `LIST_ONLY` 关键字，则不会提取文件，而只是列出文件。

# `math()` 命令

CMake 还支持一些简单的算术运算。有关完整详细信息，请参阅在线文档：

[`cmake.org/cmake/help/latest/command/math.html`](https://cmake.org/cmake/help/latest/command/math.html)

要评估一个数学表达式并将其作为可选的 `<format>`（`HEXADECIMAL` 或 `DECIMAL`）字符串存储在 `<out>` 变量中，请使用以下签名：

```cpp
math(EXPR <out> "<expression>" [OUTPUT_FORMAT <format>])
```

`<expression>` 的值是一个支持 C 代码中存在的运算符的字符串（这里的意义相同）：

+   算术：`+`、`-`、`*`、`/`、`%`（取模除法）

+   位运算：`|` 或，`&` 与，`^` 异或，`~` 非，`<<` 左移，`>>` 右移

+   圆括号 (...)

常数值可以以十进制或十六进制格式提供。
