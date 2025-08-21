# 附录

# 杂项命令

每种语言都包括一些用于各种任务的实用命令，CMake 也不例外。它提供了用于算术运算、按位操作、字符串操作以及列表和文件操作的工具。尽管由于功能增强和多个模块的发展，这些命令的需求有所减少，但在高度自动化的项目中，它们仍然是必不可少的。如今，您可能会发现它们在使用`cmake -P <filename>`调用的 CMake 脚本中更为有用。

因此，本附录总结了 CMake 命令和其多种模式，作为方便的离线参考或官方文档的简化版。要获取更详细的信息，请查阅提供的链接。

此参考适用于 CMake 3.26.6。

在本*附录*中，我们将涵盖以下主要内容：

+   `string()`命令

+   `list()`命令

+   `file()`命令

+   `math()`命令

# `string()`命令

`string()`命令用于操作字符串。它提供了多种模式，执行不同的操作：搜索和替换、操作、比较、哈希、生成和 JSON 操作（自 CMake 3.19 版本起提供最后一个）。

完整的详细信息可以在在线文档中找到：[`cmake.org/cmake/help/latest/command/string.html`](https://cmake.org/cmake/help/latest/command/string.html)。

请注意，接受`string()`模式的`<input>`参数将接受多个`<input>`值，并在执行命令之前将它们连接起来，因此：

```cpp
string(PREPEND myVariable "a" "b" "c") 
```

等同于以下内容：

```cpp
string(PREPEND myVariable "abc") 
```

可用的`string()`模式包括搜索和替换、操作、比较、哈希、生成和 JSON。

## 搜索和替换

以下模式可用：

+   `string(FIND <haystack> <pattern> <out> [REVERSE])`在`<haystack>`字符串中搜索`<pattern>`并将找到的位置以整数形式写入`<out>`变量。如果使用了`REVERSE`标志，它将从字符串的末尾向前搜索。此操作仅适用于 ASCII 字符串（不支持多字节字符）。

+   `string(REPLACE <pattern> <replace> <out> <input>)`将`<input>`中的所有`<pattern>`替换为`<replace>`，并将结果存储在`<out>`变量中。

+   `string(REGEX MATCH <pattern> <out> <input>)`使用正则表达式匹配`<input>`中第一次出现的`<pattern>`，并将其存储在`<out>`变量中。

+   `string(REGEX MATCHALL <pattern> <out> <input>)`使用正则表达式匹配`<input>`中所有出现的`<pattern>`并将其存储在`<out>`变量中，格式为逗号分隔的列表。

+   `string(REGEX REPLACE <pattern> <replace> <out> <input>)`正则替换`<input>`中的所有`<pattern>`出现，并使用`<replace>`表达式将它们替换，并将结果存储在`<out>`变量中。

正则表达式操作遵循 C++ 语法，如 `<regex>` 头文件中定义的标准库所示。你可以使用捕获组将匹配项添加到 `<replace>` 表达式中，并使用数字占位符：`\\1`、`\\2`...（需要使用双反斜杠，以确保参数被正确解析）。

## 操作

以下模式是可用的：

+   `string(APPEND <out> <input>)` 通过附加 `<input>` 字符串来修改存储在 `<out>` 中的字符串。

+   `string(PREPEND <out> <input>)` 通过在字符串前添加 `<input>` 字符串来修改存储在 `<out>` 中的字符串。

+   `string(CONCAT <out> <input>)` 将所有提供的 `<input>` 字符串连接在一起，并将其存储在 `<out>` 变量中。

+   `string(JOIN <glue> <out> <input>)` 使用 `<glue>` 值将所有提供的 `<input>` 字符串交织在一起，并将其作为连接的字符串存储在 `<out>` 变量中（不要在列表变量中使用此模式）。

+   `string(TOLOWER <string> <out>)` 将 `<string>` 转换为小写并将其存储在 `<out>` 变量中。

+   `string(TOUPPER <string> <out>)` 将 `<string>` 转换为大写并将其存储在 `<out>` 变量中。

+   `string(LENGTH <string> <out>)` 计算 `<string>` 的字节数，并将结果存储在 `<out>` 变量中。

+   `string(SUBSTRING <string> <begin> <length> <out>)` 从 `<string>` 中提取一个子字符串，长度为 `<length>` 字节，起始位置为 `<begin>` 字节，并将其存储在 `<out>` 变量中。提供 `-1` 作为长度表示“直到字符串的末尾”。

+   `string(STRIP <string> <out>)` 移除 `<string>` 的前导和尾随空白字符，并将结果存储在 `<out>` 变量中。

+   `string(GENEX_STRIP <string> <out>)` 移除 `<string>` 中所有使用的生成器表达式，并将结果存储在 `<out>` 变量中。

+   `string(REPEAT <string> <count> <out>)` 生成一个包含 `<count>` 次重复的 `<string>` 的字符串，并将其存储在 `<out>` 变量中。

## 比较

字符串比较采用以下形式：

```cpp
string(COMPARE <operation> <stringA> <stringB> <out>) 
```

`<operation>` 参数是以下之一：

+   `LESS`

+   `GREATER`

+   `EQUAL`

+   `NOTEQUAL`

+   `LESS_EQUAL`

+   `GREATER_EQUAL`

它将用于比较 `<stringA>` 和 `<stringB>`，并将结果（`true` 或 `false`）存储在 `<out>` 变量中。

## 哈希

哈希模式具有以下签名：

```cpp
string(<hashing-algorithm> <out> <string>) 
```

它使用 `<hashing-algorithm>` 对 `<string>` 进行哈希，并将结果存储在 `<out>` 变量中。支持以下算法：

+   `MD5`: 消息摘要算法 5，RFC 1321

+   `SHA1`: 美国安全哈希算法 1，RFC 3174

+   `SHA224`: 美国安全哈希算法，RFC 4634

+   `SHA256`: 美国安全哈希算法，RFC 4634

+   `SHA384`: 美国安全哈希算法，RFC 4634

+   `SHA512`: 美国安全哈希算法，RFC 4634

+   `SHA3_224`: Keccak SHA-3

+   `SHA3_256`: Keccak SHA-3

+   `SHA3_384`: Keccak SHA-3

+   `SHA3_512`: Keccak SHA-3

## 生成

以下模式是可用的：

+   `string(ASCII <number>... <out>)` 将给定的 `<number>` 的 ASCII 字符存储在 `<out>` 变量中。

+   `string(HEX <string> <out>)` 将 `<string>` 转换为其十六进制表示并将其存储在 `<out>` 变量中（从 CMake 3.18 起）。

+   `string(CONFIGURE <string> <out> [@ONLY] [ESCAPE_QUOTES])` 作用与 `configure_file()` 完全相同，但用于字符串。结果存储在 `<out>` 变量中。提醒一下，使用 `@ONLY` 关键字将替换限制为 `@VARIABLE@` 形式的变量。

+   `string(MAKE_C_IDENTIFIER <string> <out>)` 将 `<string>` 中的非字母数字字符转换为下划线，并将结果存储在 `<out>` 变量中。

+   `string(RANDOM [LENGTH <len>] [ALPHABET <alphabet>] [RANDOM_SEED <seed>] <out>)` 生成一个由 `<len>` 个字符（默认为 `5`）组成的随机字符串，使用来自随机种子 `<seed>` 的可选 `<alphabet>`，并将结果存储在 `<out>` 变量中。

+   `string(TIMESTAMP <out> [<format>] [UTC])` 生成一个表示当前日期和时间的字符串，并将其存储在 `<out>` 变量中。

+   `string(UUID <out> NAMESPACE <ns> NAME <name> TYPE <type>)` 生成一个全局唯一标识符。使用此模式稍微复杂一些；你需要提供一个命名空间（必须是 UUID）、一个名称（例如，域名）和一个类型（可以是 `MD5` 或 `SHA1`）。

## JSON

对 JSON 格式字符串的操作使用以下签名：

```cpp
string(JSON <out> [ERROR_VARIABLE <error>] <operation + args>) 
```

有几种操作可以使用。它们都将结果存储在 `<out>` 变量中，错误存储在 `<error>` 变量中。操作及其参数如下：

+   `GET <json> <member|index>...` 返回通过 `<member>` 路径或 `<index>` 对 `<json>` 字符串中的一个或多个元素提取值的结果。

+   `TYPE <json> <member|index>...` 返回通过 `<member>` 路径或 `<index>` 对 `<json>` 字符串中的一个或多个元素的类型。

+   `MEMBER <json> <member|index>... <array-index>` 返回通过 `<member>` 路径或 `<index>` 对 `<json>` 字符串中的一个或多个数组类型元素在 `<array-index>` 位置提取的成员名称。

+   `LENGTH <json> <member|index>...` 返回通过 `<member>` 路径或 `<index>` 对 `<json>` 字符串中的一个或多个数组类型元素的元素数量。

+   `REMOVE <json> <member|index>...` 返回通过 `<member>` 路径或 `<index>` 对 `<json>` 字符串中的一个或多个元素进行移除操作的结果。

+   `SET <json> <member|index>... <value>` 返回通过 `<member>` 路径或 `<index>` 对 `<json>` 字符串中的一个或多个元素进行上插入操作的结果，将 `<value>` 插入其中。

+   `EQUAL <jsonA> <jsonB>` 判断 `<jsonA>` 和 `<jsonB>` 是否相等。

# list() 命令

该命令提供基本的列表操作：读取、查找、修改和排序。一些模式会改变列表（修改原始值）。如果之后还需要使用原始值，请确保复制它。

完整的详细信息可以在在线文档中找到：

[`cmake.org/cmake/help/latest/command/list.html`](https://cmake.org/cmake/help/latest/command/list.html)

可用的 `list()` 模式类别包括读取、搜索、修改和排序。

## 读取

以下模式可用：

+   `list(LENGTH <list> <out>)` 计算 `<list>` 变量中的元素数量，并将结果存储在 `<out>` 变量中。

+   `list(GET <list> <index>... <out>)` 将 `<list>` 中通过 `<index>` 索引指定的元素复制到 `<out>` 变量中。

+   `list(JOIN <list> <glue> <out>)` 将 `<list>` 元素与 `<glue>` 分隔符交错连接，并将结果字符串存储在 `<out>` 变量中。

+   `list(SUBLIST <list> <begin> <length> <out>)` 的作用类似于 `GET` 模式，但操作的是范围而非显式索引。如果 `<length>` 为 `-1`，则返回从 `<begin>` 索引到 `<list>` 变量中提供的列表末尾的所有元素。

## 搜索

此模式简单地查找 `<needle>` 元素在 `<list>` 变量中的索引，并将结果存储在 `<out>` 变量中（如果元素未找到，则返回 `-1`）：

```cpp
list(FIND <list> <needle> <out>) 
```

## 修改

以下模式可用：

+   `list(APPEND <list> <element>...)` 将一个或多个 `<element>` 值添加到 `<list>` 变量的末尾。

+   `list(PREPEND <list> [<element>...])` 的作用类似于 `APPEND`，但将元素添加到 `<list>` 变量的开头。

+   `list(FILTER <list> {INCLUDE | EXCLUDE} REGEX <pattern>)` 根据 `<pattern>` 值筛选 `<list>` 变量中的元素，选择 `INCLUDE` 或 `EXCLUDE` 匹配的元素。

+   `list(INSERT <list> <index> [<element>...])` 将一个或多个 `<element>` 值添加到 `<list>` 变量的指定 `<index>` 位置。

+   `list(POP_BACK <list> [<out>...])` 从 `<list>` 变量的末尾移除一个元素，并将其存储在可选的 `<out>` 变量中。如果提供了多个 `<out>` 变量，将移除更多的元素以填充它们。

+   `list(POP_FRONT <list> [<out>...])` 与 `POP_BACK` 类似，但从 `<list>` 变量的开头移除一个元素。

+   `list(REMOVE_ITEM <list> <value>...)` 是 `FILTER EXCLUDE` 的简写，但不支持正则表达式。

+   `list(REMOVE_AT <list> <index>...)` 从 `<list>` 中指定的 `<index>` 位置移除元素。

+   `list(REMOVE_DUPLICATES <list>)` 移除 `<list>` 中的重复元素。

+   `list(TRANSFORM <list> <action> [<selector>] [OUTPUT_VARIABLE <out>])` 对 `<list>` 中的元素应用特定的变换。默认情况下，操作应用于所有元素，但我们可以通过添加 `<selector>` 来限制影响范围。如果没有提供 `OUTPUT_VARIABLE` 关键字，则提供的列表将被修改（就地改变）；如果提供了该关键字，结果将存储在 `<out>` 变量中。

以下选择器可用：`AT <index>`，`FOR <start> <stop> [<step>]` 和 `REGEX <pattern>`。

操作包括 `APPEND <string>`、`PREPEND <string>`、`TOLOWER`、`TOUPPER`、`STRIP`、`GENEX_STRIP` 和 `REPLACE <pattern> <expression>`。它们的功能与同名的 `string()` 模式完全相同。

## 排序

以下模式可用：

+   `list(REVERSE <list>)` 简单地反转 `<list>` 的顺序。

+   `list(SORT <list>)` 按字母顺序对列表进行排序。

请参考在线手册以获取更多高级选项。

# file()命令

该命令提供与文件相关的各种操作：读取、传输、锁定和归档。它还提供检查文件系统和操作表示路径的字符串的模式。

完整的详细信息可以在在线文档中找到：

[`cmake.org/cmake/help/latest/command/file.html`](https://cmake.org/cmake/help/latest/command/file.html)

可用的`file()`模式类别包括读取、写入、文件系统、路径转换、传输、锁定和归档。

## 阅读

可用的模式如下：

+   `file(READ <filename> <out> [OFFSET <o>] [LIMIT <max>] [HEX])` 从`<filename>`读取文件到`<out>`变量中。读取操作可选择从偏移量`<o>`开始，并遵循可选的`<max>`字节限制。`HEX flag`指定输出应转换为十六进制表示。

+   `file(STRINGS <filename> <out>)` 从`<filename>`文件读取字符串并将其存储到`<out>`变量中。

+   `file(<hashing-algorithm> <filename> <out>)` 计算来自`<filename>`文件的`<hashing-algorithm>`哈希值，并将结果存储到`<out>`变量中。可用的算法与`string()`哈希函数相同。

+   `file(TIMESTAMP <filename> <out> [<format>])` 生成`<filename>`文件的时间戳字符串表示，并将其存储到`<out>`变量中。可选接受一个`<format>`字符串。

+   `file(GET_RUNTIME_DEPENDENCIES [...])` 获取指定文件的运行时依赖项。这是一个高级命令，仅在`install(CODE)`或`install(SCRIPT)`场景中使用。从 CMake 3.21 版本开始可用。

## 写入

可用的模式如下：

+   `file({WRITE | APPEND} <filename> <content>...)` 将所有`<content>`参数写入或追加到`<filename>`文件中。如果提供的系统路径不存在，它将被递归创建。

+   `file({TOUCH | TOUCH_NOCREATE} [<filename>...])` 更新`<filename>`的时间戳。如果文件不存在，则仅在`TOUCH`模式下创建该文件。

+   `file(GENERATE OUTPUT <output-file> [...])` 是一个高级模式，它为当前 CMake 生成器的每个构建配置生成一个输出文件。

+   `file(CONFIGURE OUTPUT <output-file> CONTENT <content> [...])` 与`GENERATE_OUTPUT`类似，但还会通过将变量占位符替换为值来配置生成的文件。

## 文件系统

可用的模式如下：

+   `file({GLOB | GLOB_RECURSE} <out> [...] [<globbing-expression>...])` 生成与`<globbing-expression>`匹配的文件列表，并将其存储在`<out>`变量中。`GLOB_RECURSE`模式还会扫描嵌套目录。

+   `file(RENAME <oldname> <newname>)` 将文件从`<oldname>`移动到`<newname>`。

+   `file({REMOVE | REMOVE_RECURSE } [<files>...])` 删除`<files>`。`REMOVE_RECURSE`模式还会删除目录。

+   `file(MAKE_DIRECTORY [<dir>...])` 创建一个目录。

+   `file(COPY <file>... DESTINATION <dir> [...])`将文件复制到`<dir>`目标路径。它提供了过滤、设置权限、符号链接链跟踪等选项。

+   `file(COPY_FILE <file> <destination> [...])`将单个文件复制到`<destination>`路径。从 CMake 3.21 版本开始提供。

+   `file(SIZE <filename> <out>)`读取`<filename>`的字节大小，并将其存储在`<out>`变量中。

+   `file(READ_SYMLINK <linkname> <out>)`读取`<linkname>`符号链接的目标路径，并将其存储在`<out>`变量中。

+   `file(CREATE_LINK <original> <linkname> [...])`在`<linkname>`处创建指向`<original>`的符号链接。

+   `file({CHMOD|CHMOD_RECURSE} <files>... <directories>... PERMISSIONS <permissions>... [...])`设置文件和目录的权限。

+   `file(GET_RUNTIME_DEPENDENCIES [...])`收集各种文件类型的运行时依赖项：可执行文件、库文件和模块。与`install(RUNTIME_DEPENDENCY_SET)`一起使用。

## 路径转换

以下模式可用：

+   `file(REAL_PATH <path> <out> [BASE_DIRECTORY <dir>])`计算从相对路径到绝对路径，并将其存储在`<out>`变量中。它可以选择性地接受`<dir>`作为基础目录。从 CMake 3.19 版本开始提供。

+   `file(RELATIVE_PATH <out> <directory> <file>)`计算`<file>`相对于`<directory>`的路径，并将其存储在`<out>`变量中。

+   `file({TO_CMAKE_PATH | TO_NATIVE_PATH} <path> <out>)`将`<path>`转换为 CMake 路径（目录以正斜杠分隔），转换为平台的本地路径，并反向转换。结果存储在`<out>`变量中。

## 传输

以下模式可用：

+   `file(DOWNLOAD <url> [<path>] [...])`从`<url>`下载文件并将其存储在`<path>`中。

+   `file(UPLOAD <file> <url> [...])`将`<file>`上传到 URL。

## 锁定

锁定模式对`<path>`资源加上建议性锁：

```cpp
file(LOCK <path> [DIRECTORY] [RELEASE]
     [GUARD <FUNCTION|FILE|PROCESS>]
     [RESULT_VARIABLE <out>] [TIMEOUT <seconds>]
) 
```

此锁可以选择性地限定为`FUNCTION`、`FILE`或`PROCESS`，并限制超时时间为`<seconds>`。要释放锁，请提供`RELEASE`关键字。结果将存储在`<out>`变量中。

## 归档

创建归档提供了以下签名：

```cpp
file(ARCHIVE_CREATE OUTPUT <destination> PATHS <source>...
  [FORMAT <format>]
  [COMPRESSION <type> [COMPRESSION_LEVEL <level>]]
  [MTIME <mtime>] [VERBOSE]
) 
```

它将在`<destination>`路径创建一个包含`<source>`文件的归档，格式为支持的格式之一：`7zip`、`gnutar`、`pax`、`paxr`、`raw`或`zip`（默认格式为`paxr`）。如果所选格式支持压缩级别，则可以提供一个单数字符号`0-9`，其中`0`为默认值。

提取模式具有以下签名：

```cpp
file(ARCHIVE_EXTRACT INPUT <archive> [DESTINATION <dir>]
  [PATTERNS <patterns>...] [LIST_ONLY] [VERBOSE]
) 
```

它从`<archive>`中提取与可选的`<patterns>`值匹配的文件到目标`<dir>`。如果提供了`LIST_ONLY`关键字，则不会提取文件，而是仅列出文件。

# math()命令

CMake 还支持一些简单的算术运算。详细信息请参阅在线文档：

[`cmake.org/cmake/help/latest/command/math.html`](https://cmake.org/cmake/help/latest/command/math.html)

要评估一个数学表达式并将其作为字符串存储在 `<out>` 变量中，可以选择 `<format>`（`HEXADECIMAL` 或 `DECIMAL`），使用以下签名：

```cpp
math(EXPR <out> "<expression>" [OUTPUT_FORMAT <format>]) 
```

`<expression>` 值是一个字符串，支持 C 代码中存在的运算符（它们在这里具有相同的含义）：

+   算术运算：`+`，`-`，`*`，`/`，和 `%` 取模除法

+   位运算：`|` 或，`&` 与，`^` 异或，`~` 非，`<<` 左移，`>>` 右移

+   括号 (...)

常量值可以以十进制或十六进制格式提供。
