# 第十二章

# 最新的 C

变化是无法阻止的，C 语言也不例外。C 编程语言由 ISO 标准标准化，并且由一群试图使其更好并为其带来新特性的群体不断修订。这并不意味着语言一定会变得更容易；我们可能会看到随着新内容的添加，语言中出现新颖和复杂的功能。

在本章中，我们将简要地看看 C11 的特性。你可能知道 C11 已经取代了旧的 C99 标准，并且已经被 C18 标准所取代。换句话说，C18 是 C 标准的最新版本，而在那之前我们有 C11。

有趣的是，C18 没有提供任何新特性；它只是对 C11 中发现的问题进行了修复。因此，谈论 C11 基本上等同于谈论 C18，这将引导我们到最新的 C 标准。正如你所看到的，我们在 C 语言中观察到持续改进……这与它是一个已经死去很长时间的语言的看法相反！

本章将简要概述以下主题：

+   如何检测 C 版本以及如何编写兼容各种 C 版本的 C 代码

+   用于编写优化和安全的代码的新特性，如*不返回*函数和*边界检查*函数

+   新的数据类型和内存对齐技术

+   类型泛型函数

+   C11 中的 Unicode 支持，这在旧标准中缺失

+   匿名结构和联合

+   C11 中标准的多线程和同步技术支持

让我们从讨论 C11 及其新特性开始本章。

# C11

收集一个使用超过 30 年的技术的新的标准并非易事。数百万（如果不是数十亿！）行 C 代码存在，如果你即将引入新特性，这必须在保持现有代码或特性完整的情况下完成。新特性不应为现有程序带来新问题，并且应该是无错误的。虽然这种观点看似理想化，但这是我们应该致力于做到的。

以下 PDF 文档位于*开放标准*网站上，包含了在开始塑造 C11 之前 C 社区中人们的担忧和思考：http://www.open-std.org/JTC1/SC22/wg14/www/docs/n1250.pdf。阅读它会有所帮助，因为它会向你介绍为基于数千个软件构建的编程语言编写新标准时的经验。

最后，考虑到这些因素，我们考虑 C11 的发布。当 C11 发布时，它并非处于理想状态，实际上正遭受一些严重的缺陷。你可以看到这些缺陷的列表[这里：http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2244](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2244.htm)。

C11 发布七年之后，推出了 C18，这是为了修复在 C11 中发现的问题。请注意，C18 也非正式地被称为 C17，C17 和 C18 都指的是相同的 C 标准。如果您打开前面的链接，您将看到缺陷及其当前状态。如果缺陷的状态是“C17”，这意味着该缺陷作为 C18 的一部分得到了解决。这显示了构建一个像 C 一样拥有众多用户的标准的艰难和精细过程。

在接下来的几节中，我们将讨论 C11 的新特性。然而，在通过它们之前，我们需要一种方法来确保我们确实在编写 C11 代码，并且我们使用的是兼容的编译器。下一节将解决这个问题。

# 查找支持的 C 标准版本

在撰写本文时，C11 发布已有近 8 年。因此，可以预期许多编译器应该支持该标准，这确实是事实。开源编译器如`gcc`和`clang`都完美支持 C11，并且如果需要，它们可以切换回 C99 或更早的版本。在本节中，我们将展示如何使用特定的宏来检测 C 版本，以及根据版本如何使用支持的功能。

当使用支持不同 C 标准版本的编译器时，首先必要的是能够识别当前正在使用的 C 标准版本。每个 C 标准都定义了一个特殊的宏，可以用来找出正在使用哪个版本。到目前为止，我们在 Linux 系统中使用了`gcc`，在 macOS 系统中使用了`clang`。从版本 4.7 开始，`gcc`将其支持的标准之一提供为 C11。

让我们看看以下示例，看看已经定义的宏如何用于在运行时检测当前的 C 标准版本：

```cpp
#include <stdio.h>
int main(int argc, char** argv) {
#if __STDC_VERSION__ >=  201710L
  printf("Hello World from C18!\n");
#elif __STDC_VERSION__ >= 201112L
  printf("Hello World from C11!\n");
#elif __STDC_VERSION__ >= 199901L
  printf("Hello World from C99!\n");
#else
  printf("Hello World from C89/C90!\n");
#endif
  return 0;
}
```

Code Box 12-1 [ExtremeC_examples_chapter12_1.c]：检测 C 标准的版本

如您所见，前面的代码可以区分不同的 C 标准版本。为了看到不同的 C 版本如何导致不同的打印结果，我们必须多次使用编译器支持的 C 标准版本编译前面的源代码。

要让编译器使用特定的 C 标准版本，我们必须将`-std=CXX`选项传递给 C 编译器。查看以下命令和产生的输出：

```cpp
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out
$ ./ex12_1.out
Hello World from C11!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c11
$ ./ex12_1.out
Hello World from C11!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c99
$ ./ex12_1.out
Hello World from C99!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c90
$ ./ex12_1.out
Hello World from C89/C90!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c89
$ ./ex12_1.out
Hello World from C89/C90!
$
```

Shell Box 12-1：使用各种 C 标准版本编译示例 12.1

如您所见，较新编译器的默认 C 标准版本是 C11。在较旧版本中，如果您想启用 C11，必须使用`-std`选项指定版本。注意文件开头所做的注释。我使用了`/* ... */`多行注释而不是`//`单行注释。这是因为 C99 之前的标准中不支持单行注释。因此，我们必须使用多行注释，以便前面的代码能够与所有 C 版本兼容地编译。

# 移除 `gets` 函数

在 C11 中，著名的 `gets` 函数被移除。`gets` 函数曾受到 *缓冲区溢出* 攻击，在旧版本中，它被决定为 *已弃用*。后来，作为 C11 标准的一部分，它被移除。因此，使用 `gets` 函数的旧源代码将无法使用 C11 编译器编译。

可以使用 `fgets` 函数代替 `gets`。以下是从 macOS 中 `gets` 手册页（man 页）摘录的内容：

> 安全考虑
> 
> `gets()` 函数不能安全使用。由于其缺乏边界检查，以及调用程序无法可靠地确定下一行输入的长度，使用此函数会使恶意用户通过缓冲区溢出攻击任意更改正在运行的程序的功能。强烈建议在所有情况下使用 `fgets()` 函数。（参见 FSA。）

# `fopen` 函数的更改

`fopen` 函数通常用于打开文件并返回该文件的文件描述符。在 Unix 中，文件的概念非常通用，使用术语 *文件* 并不一定意味着位于文件系统上的文件。`fopen` 函数有以下签名：

```cpp
FILE* fopen(const char *pathname, const char *mode);
FILE* fdopen(int fd, const char *mode);
FILE* freopen(const char *pathname, const char *mode, FILE *stream);
```

Code Box 12-2：`fopen` 函数家族的各种签名

如您所见，所有前面的签名都接受一个 `mode` 输入。此输入参数是一个字符串，它决定了文件应该如何打开。*Shell Box 12-2* 中的以下描述来自 FreeBSD 手册中的 `fopen` 函数，并解释了如何使用 `mode`：

```cpp
$ man 3 fopen
...
The argument mode points to a string beginning with one of the following letters:
     "r"     Open for reading.  The stream is positioned at the beginning
             of the file.  Fail if the file does not exist.
     "w"     Open for writing.  The stream is positioned at the beginning
             of the file.  Create the file if it does not exist.
     "a"     Open for writing.  The stream is positioned at the end of
             the file. Subsequent writes to the file will always end up
             at the then current end of file, irrespective of 
             any intervening fseek(3) or similar. Create the file 
             if it does not exist.
     An optional "+" following "r", "w", or "a" opens the file
     for both reading and writing.  An optional "x" following "w" or
     "w+" causes the fopen() call to fail if the file already exists.
     An optional "e" following the above causes the fopen() call to set
     the FD_CLOEXEC flag on the underlying file descriptor.
     The mode string can also include the letter "b" after either 
     the "+" or the first letter.
...
$
```

Shell Box 12-2：FreeBSD 中 `fopen` 手册页的摘录

在 `fopen` 手册页的前面摘录中解释的 `x` 模式是作为 C11 的一部分引入的。为了写入文件，应向 `fopen` 提供模式 `w` 或 `w+`。问题是，如果文件已经存在，`w` 或 `w+` 模式将截断（清空）文件。

因此，如果程序员想要向文件追加内容并保留其当前内容，他们必须使用不同的模式，即 `a`。因此，他们必须在调用 `fopen` 之前使用文件系统 API（如 `stat`）检查文件是否存在，然后根据结果选择适当的模式。然而，现在有了新的模式 `x`，程序员首先尝试使用模式 `wx` 或 `w+x`，如果文件已存在，`fopen` 将失败。然后程序员可以继续使用 `a` 模式。

因此，在不需要使用文件系统 API 检查文件是否存在的情况下打开文件，需要编写的样板代码更少。从现在开始，`fopen` 就足以以每种所需的模式打开文件。

C11 的另一个变化是引入了 `fopen_s` API。这个函数作为安全的 `fopen`。根据位于 https://en.cppreference.com/w/c/io/fopen 的 `fopen_s` 文档，对提供的缓冲区和它们的边界进行额外检查，以检测其中的任何缺陷。

# 边界检查函数

C 程序在字符串和字节数组上操作时遇到的一个严重问题是容易超出为缓冲区或字节数组定义的边界。

作为提醒，缓冲区是内存中的一个区域，用作字节数组或字符串变量的占位符。超出缓冲区的边界会导致*缓冲区溢出*，基于此，恶意实体可以组织攻击（通常称为*缓冲区溢出攻击*）。这种攻击要么导致**拒绝服务**（**DOS**），要么在受影响的 C 程序中进行*利用*。

大多数此类攻击通常从一个操作字符或字节数组的函数开始。在`string.h`中找到的字符串操作函数，如`strcpy`和`strcat`，是缺乏边界检查机制以防止缓冲区溢出攻击的*易受攻击*函数。

然而，作为 C11 的一部分，引入了一套新的函数。*边界检查*函数从字符串操作函数借用相同的名称，但以`_s`结尾。后缀`_s`将它们区分开来，作为*安全*或*安全*版本的函数，这些函数在运行时进行更多的检查，以关闭漏洞。`strcpy_s`和`strcat_s`等函数作为 C11 中边界检查函数的一部分被引入。

这些函数接受一些额外的输入缓冲区参数，限制了它们执行危险操作的能力。例如，`strcpy_s`函数具有以下签名：

```cpp
errno_t strcpy_s(char *restrict dest, rsize_t destsz, const char *restrict src);
```

代码框 12-3：strcpy_s 函数的签名

如您所见，第二个参数是`dest`缓冲区的长度。使用它，该函数执行一些运行时检查，例如确保`src`字符串的长度短于或与`dest`缓冲区的大小相同，以防止写入未分配的内存。

# 无返回函数

函数调用可以通过使用`return`关键字或到达函数块的末尾来结束。也存在函数调用永远不会结束的情况，这通常是有意为之。看看以下包含在*代码框 12-4*中的代码示例：

```cpp
void main_loop() {
  while (1) {
    ...
  }
}

int main(int argc, char** argv) {
  ...
  main_loop();
  return 0;
}
```

代码框 12-4：永不返回的函数示例

如您所见，函数`main_loop`执行程序的主要任务，如果我们从函数返回，则程序可以被认为是结束的。在这些异常情况下，编译器可以执行一些额外的优化，但无论如何，它需要知道函数`main_loop`永远不会返回。

在 C11 中，您可以将一个函数标记为*无返回*函数。`stdnoreturn.h`头文件中的`_Noreturn`关键字可以用来指定一个函数永远不会退出。因此，*代码框 12-4*中的代码可以修改为 C11 的如下所示：

```cpp
_Noreturn void main_loop() {
  while (true) {
    ...
  }
}
```

代码框 12-5：使用 _Noreturn 关键字标记 main_loop 为永不结束的函数

还有其他函数，如`exit`、`quick_exit`（作为 C11 的一部分最近添加，用于快速终止程序），以及`abort`，被认为是不可返回的函数。此外，了解不可返回函数允许编译器识别那些无意中不会返回的函数调用，并产生适当的警告，因为这些可能是逻辑错误的迹象。请注意，如果一个标记为`_Noreturn`的函数返回，那么这将是一种*未定义的行为*，并且强烈不建议这样做。

# 输入通用宏

在 C11 中，引入了一个新的关键字：`_Generic`。它可以用来编写在编译时具有类型感知能力的宏。换句话说，你可以编写可以根据其参数类型改变其值的宏。这通常被称为*泛型选择*。请看以下代码示例在*代码框 12-6*：

```cpp
#include <stdio.h>
#define abs(x) _Generic((x), \
                        int: absi, \
                        double: absd)(x)
int absi(int a) {
  return a > 0 ? a : -a;
}
double absd(double a) {
  return a > 0 ? a : -a;
}
int main(int argc, char** argv) {
  printf("abs(-2): %d\n", abs(-2));
  printf("abs(2.5): %f\n", abs(2.5));;
  return 0;
}
```

代码框 12-6：通用宏示例

如您在宏定义中所见，我们根据参数`x`的类型使用了不同的表达式。如果它是整数值，我们使用`absi`；如果是双精度值，我们使用`absd`。这个特性对 C11 来说并不新鲜，您可以在较老的 C 编译器中找到它，但它不是 C 标准的一部分。截至 C11，它现在是标准的，您可以使用这种语法来编写类型感知宏。

# Unicode

C11 标准中添加的最伟大的特性之一是通过 UTF-8、UTF-16 和 UTF-32 编码支持 Unicode。C 长期以来缺少这个特性，C 程序员必须使用第三方库，如**IBM 国际组件 Unicode**（**ICU**），来满足他们的需求。

在 C11 之前，我们只有`char`和`unsigned char`类型，它们是 8 位变量，用于存储 ASCII 和扩展 ASCII 字符。通过创建这些 ASCII 字符的数组，我们可以创建 ASCII 字符串。

**注意**:

ASCII 标准有 128 个字符，可以用 7 位存储。扩展 ASCII 是 ASCII 的扩展，增加了另外 128 个字符，使总数达到 256 个。然后，一个 8 位或单字节变量足以存储所有这些字符。在即将到来的文本中，我们只会使用术语 ASCII，并且通过这个术语我们指的是 ASCII 标准和扩展 ASCII。

注意，对 ASCII 字符和字符串的支持是基本的，并且永远不会从 C 中移除。因此，我们可以确信我们将在 C 中始终拥有 ASCII 支持。从 C11 开始，它们添加了对新字符的支持，因此产生了使用不同字节数的新字符串，而不仅仅是每个字符一个字节。

为了进一步解释，在 ASCII 编码中，每个字符占用一个字节。因此，字节和字符可以互换使用，但这种情况并不普遍。不同的编码定义了在多个字节中存储更广泛字符的新方法。

在 ASCII 编码中，总共有 256 个字符。因此，一个单字节（8 位）的字符就足以存储所有这些字符。然而，如果我们需要超过 256 个字符，我们必须使用超过一个字节来存储超过 255 的数值。需要超过一个字节来存储其值的字符通常被称为*宽字符*。根据这个定义，ASCII 字符不被认为是宽字符。

Unicode 标准引入了多种使用超过一个字节来编码 ASCII、扩展 ASCII 和宽字符的方法。这些方法被称为*编码*。通过 Unicode，有三种著名的编码：UTF-8、UTF-16 和 UTF-32。UTF-8 使用第一个字节来存储 ASCII 字符的前半部分，接下来的字节，通常最多 4 个字节，用于存储 ASCII 字符的后半部分以及所有其他宽字符。因此，UTF-8 被认为是一种可变长度的编码。它使用字符的第一个字节中的某些位来表示应该读取多少实际字节才能完全检索字符。UTF-8 被认为是一个 ASCII 的超集，因为对于 ASCII 字符（不是扩展 ASCII 字符）的表示是相同的。

与 UTF-8 类似，UTF-16 使用一个或两个*字*（每个字内部有 16 位）来存储所有字符；因此，它也是一种可变长度的编码。UTF-32 使用恰好 4 字节来存储所有字符的值；因此，它是一种固定长度的编码。UTF-8 和 UTF-16 适用于需要为更频繁出现的字符使用更少字节的程序。

UTF-32 即使在 ASCII 字符上也使用固定数量的字节。因此，与使用其他编码相比，使用 UTF-32 编码存储字符串会消耗更多的内存空间；但使用 UTF-32 字符时所需的计算能力更少。UTF-8 和 UTF-16 可以被认为是压缩编码，但它们需要更多的计算来返回字符的实际值。

**注意**：

更多关于 UTF-8、UTF-16 和 UTF-32 字符串及其解码方式的信息可以在维基百科或其他来源找到，例如：

[`unicodebook.readth`](https://unicodebook.readthedocs.io/unicode_encodings.html)e[docs.io/unicode_encodings.html](https://javarevisited.blogspot.com/2015/02/difference-between-utf-8-utf-16-and-utf.html)

[`javarevisited.blogspot.com/2015/02/difference-be`](https://javarevisited.blogspot.com/2015/02/difference-between-utf-8-utf-16-and-utf.html)t[ween-utf-8-utf-16-and-utf.html](https://unicodebook.readthedocs.io/unicode_encodings.html).

在 C11 中，我们支持所有上述 Unicode 编码。查看以下示例，*example 12.3*。它定义了各种 ASCII、UTF-8、UTF-16 和 UTF-32 字符串，并计算存储它们的实际字节数和观察到的字符数。我们分多步展示代码，以便对代码进行额外的注释。以下代码框演示了所需的包含和声明：

```cpp
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __APPLE__
#include <stdint.h>
typedef uint16_t char16_t;
typedef uint32_t char32_t;
#else
#include <uchar.h> // Needed for char16_t and char32_t
#endif
```

Code Box 12-7 [ExtremeC_examples_chapter12_3.c]：示例 12.3 所需的包含和声明

前面的行是 *example 12.3* 的 `include` 语句。如您所见，在 macOS 上我们没有 `uchar.h` 头文件，我们必须为 `char16_t` 和 `char32_t` 类型定义新类型。尽管如此，Unicode 字符串的整个功能都得到了支持。在 Linux 上，我们没有 C11 中 Unicode 支持的问题。

代码的下一部分演示了用于计算各种类型 Unicode 字符串的字节数和字符数的函数。请注意，C11 没有提供用于操作 Unicode 字符串的实用函数，因此我们必须为它们编写新的 `strlen`。实际上，我们的 `strlen` 函数不仅返回字符数，还返回消耗的字节数。实现细节将不会描述，但强烈建议您阅读它们：

```cpp
typedef struct {
  long num_chars;
  long num_bytes;
} unicode_len_t;
unicode_len_t strlen_ascii(char* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  res.num_chars = strlen(str) + 1;
  res.num_bytes = strlen(str) + 1;
  return res;
}
unicode_len_t strlen_u8(char* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  // Last null character
  res.num_chars = 1;
  res.num_bytes = 1;
  while (*str) {
    if ((*str | 0x7f) == 0x7f) { // 0x7f = 0b01111111
      res.num_chars++;
      res.num_bytes++;
      str++;
    } else if ((*str & 0xc0) == 0xc0) { // 0xc0 = 0b11000000
      res.num_chars++;
      res.num_bytes += 2;
      str += 2;
    } else if ((*str & 0xe0) == 0xe0) { // 0xe0 = 0b11100000
      res.num_chars++;
      res.num_bytes += 3;
      str += 3;
    } else if ((*str & 0xf0) == 0xf0) { // 0xf0 = 0b11110000
      res.num_chars++;
      res.num_bytes += 4;
      str += 4;
    } else {
      fprintf(stderr, "UTF-8 string is not valid!\n");
      exit(1);
    }
  }
  return res;
}
unicode_len_t strlen_u16(char16_t* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  // Last null character
  res.num_chars = 1;
  res.num_bytes = 2;
  while (*str) {
    if (*str < 0xdc00 || *str > 0xdfff) {
      res.num_chars++;
      res.num_bytes += 2;
      str++;
    } else {
      res.num_chars++;
      res.num_bytes += 4;
      str += 2;
    }
  }
  return res;
}
unicode_len_t strlen_u32(char32_t* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  // Last null character
  res.num_chars = 1;
  res.num_bytes = 4;
  while (*str) {
      res.num_chars++;
      res.num_bytes += 4;
      str++;
  }
  return res;
}
```

Code Box 12-8 [ExtremeC_examples_chapter12_3.c]：示例 12.3 中使用的函数的定义

最后的部分是 `main` 函数。它声明了一些英文、波斯语和一些外星语言的字符串，以评估前面的函数：

```cpp
int main(int argc, char** argv) {
  char ascii_string[32] = "Hello World!";
  char utf8_string[32] = u8"Hello World!";
  char utf8_string_2[32] = u8"درود دنیا!";
  char16_t utf16_string[32] = u"Hello World!";
  char16_t utf16_string_2[32] = u"درود دنیا!";
  char16_t utf16_string_3[32] = u"হহহ!";
  char32_t utf32_string[32] = U"Hello World!";
  char32_t utf32_string_2[32] = U"درود دنیا!";
  char32_t utf32_string_3[32] = U"হহহ!";
  unicode_len_t len = strlen_ascii(ascii_string);
  printf("Length of ASCII string:\t\t\t %ld chars, %ld bytes\n\n",
      len.num_chars, len.num_bytes);
  len = strlen_u8(utf8_string);
  printf("Length of UTF-8 English string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u16(utf16_string);
  printf("Length of UTF-16 english string:\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u32(utf32_string);
  printf("Length of UTF-32 english string:\t %ld chars, %ld bytes\n\n",
      len.num_chars, len.num_bytes);
  len = strlen_u8(utf8_string_2);
  printf("Length of UTF-8 Persian string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u16(utf16_string_2);
  printf("Length of UTF-16 persian string:\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u32(utf32_string_2);
  printf("Length of UTF-32 persian string:\t %ld chars, %ld bytes\n\n",
      len.num_chars, len.num_bytes);
  len = strlen_u16(utf16_string_3);
  printf("Length of UTF-16 alien string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u32(utf32_string_3);
  printf("Length of UTF-32 alien string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  return 0;
}
```

Code Box 12-9 [ExtremeC_examples_chapter12_3.c]：示例 12.3 的主函数

现在，我们必须编译前面的示例。请注意，该示例只能使用 C11 编译器进行编译。您可以尝试使用较旧的编译器并查看产生的错误。以下命令编译并运行前面的程序：

```cpp
$ gcc ExtremeC_examples_chapter12_3.c -std=c11 -o ex12_3.out
$ ./ex12_3.out
Length of ASCII string:            13 chars, 13 bytes
Length of UTF-8 english string:      13 chars, 13 bytes
Length of UTF-16 english string:     13 chars, 26 bytes
Length of UTF-32 english string:     13 chars, 52 bytes
Length of UTF-8 persian string:      11 chars, 19 bytes
Length of UTF-16 persian string:     11 chars, 22 bytes
Length of UTF-32 persian string:     11 chars, 44 bytes
Length of UTF-16 alien string:       5 chars, 14 bytes
Length of UTF-32 alien string:       5 chars, 20 bytes
$
```

Shell Box 12-3：编译和运行示例 12.3

如您所见，具有相同字符数的相同字符串使用不同数量的字节来编码和存储相同的值。UTF-8 使用最少的字节，尤其是在文本中有大量 ASCII 字符时，因为大多数字符将仅使用一个字节。

当我们遇到与拉丁字符更不同的字符时，例如亚洲语言的字符，UTF-16 在字符数和使用的字节数之间有更好的平衡，因为大多数字符将使用最多两个字节。

UTF-32 很少使用，但它可以用于需要固定长度 *代码打印* 的字符的系统；例如，如果系统计算能力较低或受益于某些并行处理管道。因此，UTF-32 字符可以用作从字符到任何类型数据的映射中的键。换句话说，它们可以用来构建一些索引以快速查找数据。

# 匿名结构和匿名联合

匿名结构和匿名联合是没有名称的类型定义，通常用作其他类型的嵌套类型。用示例更容易解释它们。在这里，你可以看到一个类型，它在一个地方同时具有匿名结构和匿名联合，显示在代码框 12-10 中：

```cpp
typedef struct {
  union {
    struct {
      int x;
      int y;
    };
    int data[2];
  };
} point_t;
```

代码框 12-10：匿名结构和匿名联合的示例

前面的类型使用相同的内存来存储匿名结构和字节数组字段 `data`。以下代码框显示了它如何在实际示例中使用：

```cpp
#include <stdio.h>
typedef struct {
  union {
    struct {
      int x;
      int y;
    };
    int data[2];
  };
} point_t;
int main(int argc, char** argv) {
  point_t p;
  p.x = 10;
  p.data[1] = -5;
  printf("Point (%d, %d) using an anonymous structure inside an anonymous union.\n", p.x, p.y);
  printf("Point (%d, %d) using byte array inside an anonymous union.\n",
      p.data[0], p.data[1]);
  return 0;
}
```

代码框 12-11 [ExtremeC_examples_chapter12_4.c]：使用匿名结构和匿名联合的主函数

在此示例中，我们创建了一个包含匿名结构的匿名联合。因此，相同的内存区域用于存储匿名结构的一个实例和两个元素的整数数组。接下来，你可以看到前面程序的输出：

```cpp
$ gcc ExtremeC_examples_chapter12_4.c -std=c11 -o ex12_4.out
$ ./ex12_4.out
Point (10, -5) using anonymous structure.
Point (10, -5) using anonymous byte array.
$
```

脚本框 12-4：编译和运行示例 12.4

如你所见，对两个元素的整数数组的任何更改都可以在结构变量中看到，反之亦然。

# 多线程

C 语言通过 POSIX 线程函数或 `pthreads` 库长期以来一直支持多线程。我们在第十五章“线程执行”和第十六章“线程同步”中全面介绍了多线程。

如其名所示，POSIX 线程库仅在符合 POSIX 的系统（如 Linux 和其他类 Unix 系统）中可用。因此，如果你使用的是非 POSIX 兼容的操作系统，如 Microsoft Windows，你必须使用操作系统提供的库。作为 C11 的一部分，提供了一个标准线程库，可以在所有使用标准 C 的系统上使用，无论其是否符合 POSIX。这是我们在 C11 标准中看到的最大变化。

不幸的是，C11 线程在 Linux 和 macOS 上没有实现。因此，我们无法在撰写本文时提供工作示例。

# 关于 C18 的一些信息

如我们前面提到的，C18 标准包含了 C11 中所做的所有修复，并且没有在它中引入任何新功能。正如之前所说，以下链接带你到一个页面，你可以看到为 C11 创建并跟踪的问题以及围绕它们进行的讨论：http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2244.htm。

# 概述

在本章中，我们回顾了 C11、C18 以及最新的 C 标准，并探讨了 C11 的各种新特性。Unicode 支持、匿名结构和联合体，以及新的标准线程库（尽管到目前为止它尚未在最近的编译器和平台上可用）是现代 C 语言中引入的最重要特性之一。我们期待着未来看到 C 标准的新的版本。

在下一章中，我们将开始讨论并发以及并发系统的理论。这将开启一段长达六章的漫长旅程，我们将涵盖多线程和多进程，以实现我们编写并发系统的目的。
