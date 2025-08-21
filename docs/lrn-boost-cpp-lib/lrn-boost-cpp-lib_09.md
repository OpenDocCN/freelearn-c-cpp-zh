# 第九章：文件、目录和 IOStreams

为了与操作系统的各种子系统进行交互以利用它们的服务，编写真实世界系统的程序需要。从本章开始，我们将看看各种 Boost 库，这些库提供对操作系统子系统的编程访问。

在本章中，我们将介绍用于执行输入和输出以及与文件系统交互的 Boost 库。我们将在本章的以下部分中介绍这些库：

+   使用 Boost 文件系统管理文件和目录

+   使用 Boost IOStreams 进行可扩展 I/O

使用本章涵盖的库和技术，您将能够编写可移植的 C++程序，与文件系统交互，并使用标准接口执行各种 I/O 操作。本章不涵盖网络 I/O，而是专门讨论第十章*使用 Boost 进行并发*。

# 使用 Boost 文件系统管理文件和目录

使用 Boost 库编写的软件可以在多个操作系统上运行，包括 Linux、Microsoft Windows、Mac OS 和各种其他 BSD 变体。这些操作系统访问文件和目录的路径的方式可能在多种方面有所不同；例如，MS Windows 使用反斜杠作为目录分隔符，而所有 Unix 变体，包括 Linux、BSD 和 Mac，使用正斜杠。非英语操作系统可能使用其他字符作为目录分隔符，有时还支持多个目录分隔符。Boost 文件系统库隐藏了这些特定于平台的特性，并允许您编写更具可移植性的代码。使用 Boost 文件系统库中的函数和类型，您可以编写与操作系统无关的代码，执行应用程序运行所需的文件系统上的常见操作，如复制、重命名和删除文件，遍历目录，创建目录和链接等。

## 操作路径

文件系统路径使用`boost::filesystem::path`类型的对象表示。给定`boost::filesystem::path`类型的对象，我们可以从中获取有用的信息，并从中派生其他`path`对象。`path`对象允许我们对真实的文件系统路径进行建模并从中获取信息，但它不一定代表系统中真正存在的路径。

### 打印路径

让我们看看使用 Boost 文件系统打印进程的当前工作目录的第一个示例：

**清单 9.1：使用 Boost 文件系统的第一个示例**

```cpp
 1 #include <boost/filesystem.hpp>
 2 #include <iostream>
 3
 4 namespace fs = boost::filesystem;
 5
 6 int main() {
 7   // Get the current working directory
 8   fs::path cwd = fs::current_path();
 9
10   // Print the path to stdout
11   std::cout << "generic: " << cwd.generic_string() << '\n';
12   std::cout << "native: " << cwd.string() << '\n';
13   std::cout << "quoted: " << cwd << '\n';
14 
15   std::cout << "Components: \n";
16   for (const auto& dir : cwd) {
17     std::cout <<'[' <<dir.string() << ']'; // each part
18   }
19   std::cout << '\n';
20 }
```

在此示例中，程序通过调用`current_path`（第 8 行）确定其当前工作目录，这是`boost::filesystem`命名空间中的一个命名空间级函数。它返回一个表示当前工作目录的`boost::filesystem::path`类型的对象。`boost::filesystem`中的大多数函数都是在`boost::filesystem::path`对象上而不是字符串上工作。

我们通过调用`path`的`generic_string`成员函数（第 11 行），通过调用`string`成员函数（第 12 行），以及通过将`cwd`，路径对象，流式传输到输出流（第 13 行）来打印路径。`generic_string`成员以**通用格式**返回路径，该格式由 Boost 文件系统支持，使用正斜杠作为分隔符。`string`成员函数以**本机格式**返回路径，这是一个依赖于操作系统的实现定义格式。在 Windows 上，本机格式使用反斜杠作为路径分隔符，而在 UNIX 上，通用格式和本机格式之间没有区别。Boost 文件系统在 Windows 上识别正斜杠和反斜杠作为路径分隔符。

流式传输`path`对象也会以本机格式写入路径，但还会在路径周围加上双引号。在路径中有嵌入空格的情况下，加上双引号可以方便将结果用作命令的参数。如果路径中有嵌入的双引号字符（`"`），则会用和号（`&`）对其进行转义。

在 Windows 上，完整路径以宽字符（`wchar_t`）字符串存储，因此`generic_string`或`string`在执行转换*后*将路径作为`std::string`返回。根据路径中特定的 Unicode 字符，可能无法将路径有意义地转换为单字节字符字符串。在这种系统上，只能安全地调用`generic_wstring`或`wstring`成员函数，它们以通用或本机格式返回路径作为`std::wstring`。

我们使用 C++11 中的范围 for 循环迭代路径中的每个目录组件（第 15 行）。如果范围 for 循环不可用，我们应该使用`path`中的`begin`和`end`成员函数来迭代路径元素。在我的 Windows 系统上，该程序打印以下内容：

```cpp
generic: E:/DATA/Packt/Boost/Draft/Book/Chapter07/examples
native:E:\DATA\Packt\Boost\Draft\Book\Chapter07\examples
quoted: "E:\DATA\Packt\Boost\Draft\Book\Chapter07\examples"
Components:
[E:][/][DATA][Packt] [Boost][Draft][Book][Chapter07][examples]
```

在我的 Ubuntu 系统上，这是我得到的输出：

```cpp
generic: /home/amukher1/devel/c++/book/ch07
native: /home/amukher1/devel/c++/book/ch07
quoted: "/home/amukher1/devel/c++/book/ch07"
Components:
[/][home][amukher1] [devel][c++][book][ch07]
```

该程序以通用格式和本机格式打印其当前工作目录。您可以看到在 Ubuntu 上（以及通常在任何 Unix 系统上）两者之间没有区别。

在 Windows 上，路径的第一个组件是驱动器号，通常称为**根名称**。然后是/（根文件夹）和路径中的每个子目录。在 Unix 上，没有根名称（通常情况下），因此清单以/（根目录）开头，然后是路径中的每个子目录。

类型为`path`的`cwd`对象是可流式传输的（第 19 行），将其打印到标准输出会以本机格式带引号打印出来。

### 注意

**使用 Boost Filesystem 编译和链接示例**

Boost Filesystem 不是一个仅包含头文件的库。Boost Filesystem 共享库作为 Boost 操作系统包的一部分安装，或者根据第一章中描述的方式从源代码构建，*介绍 Boost*。

**在 Linux 上**

如果您使用本机包管理器安装 Boost 库，则可以使用以下命令构建您的程序。请注意，库名称采用系统布局。

```cpp
$ g++ <source>.c -o <executable> -lboost_filesystem -lboost_system

```

如果您按照第一章中所示的方式从源代码构建 Boost，并将其安装在`/opt/boost`下，您可以使用以下命令来编译和链接您的源代码：

```cpp
$ g++ <source>.cpp -c -I/opt/boost/include
$ g++ <source>.o -o <executable> -L/opt/boost/lib -lboost_filesystem-mt -lboost_system-mt -Wl,-rpath,/opt/boost/lib

```

由于我们使用标记布局构建了库，因此我们链接到适当命名的 Boost Filesystem 和 Boost System 版本。`-Wl,-rpath,/opt/boost/lib`部分将 Boost 共享库的路径嵌入生成的可执行文件中，以便运行时链接器知道从哪里获取可执行文件运行所需的共享库。

**在 Windows 上**

在 Windows 上，使用 Visual Studio 2012 或更高版本，您可以启用自动链接，无需显式指定要链接的库。为此，您需要在**项目属性**对话框中编辑**配置属性**设置（在 IDE 中使用*Alt* + *F7*打开）：

1\. 在**VC++目录**下，将`<boost-install-path>\include`追加到**包含目录**属性。

2\. 在**VC++目录**下，将`<boost-install-path>\lib`追加到**库目录**属性。

3\. 在**调试**下，将**环境**属性设置为`PATH=%PATH%;<boost-install-path>\lib`。

4\. 在**C/C++ > 预处理器**下，定义以下预处理器符号：

`BOOST_ALL_DYN_LINK`

`BOOST_AUTO_LINK_TAGGED`（仅在使用标记布局构建时）

5\. 通过从 Visual Studio IDE 中按下*F7*来构建，并通过从 IDE 中按下*Ctrl* + *F5*来运行程序。

### 构建路径

您可以使用`path`构造函数之一或以某种方式组合现有路径来构造`boost::filesystem::path`的实例。字符串和字符串字面值可以隐式转换为`path`对象。您可以构造相对路径和绝对路径，将相对路径转换为绝对路径，从路径中添加或删除元素，并“规范化”路径，如清单 9.2 所示：

**清单 9.2a：构造空路径对象**

```cpp
 1 #define BOOST_FILESYSTEM_NO_DEPRECATED
 2 #include <boost/filesystem.hpp>
 3 #include <iostream>
 4 #include <cassert>
 5 namespace fs = boost::filesystem;
 6 
 7 int main() {
 8   fs::path p1; // empty path
 9   assert(p1.empty());  // does not fire
10   p1 = "/opt/boost";   // assign an absolute path
11   assert(!p1.empty());
12   p1.clear();
13   assert(p1.empty());
14 }
```

一个默认构造的路径对象表示一个空路径，就像前面的例子所示。你可以将一个路径字符串赋给一个空的`path`对象（第 10 行），它就不再是空的了（第 11 行）。在路径上调用`clear`成员函数（第 12 行）后，它再次变为空（第 13 行）。多年来，Boost 文件系统库的一些部分已经被弃用，并被更好的替代品所取代。我们定义宏`BOOST_FILESYSTEM_NO_DEPRECATED`（第 1 行）以确保这些弃用的成员函数和类型不可访问。

**清单 9.2b：构造相对路径**

```cpp
15 void make_relative_paths() {
16   fs::path p2(".."); // relative path
17   p2 /= "..";
18   std::cout << "Relative path: " << p2.string() << '\n';
19
20   std::cout << "Absolute path: "
21      << fs::absolute(p2, "E:\\DATA\\photos").string() << '\n';
22   std::cout << "Absolute path wrt CWD: "
23             << fs::absolute(p2).string() << '\n';
24
25   std::cout << fs::canonical(p2).string() << '\n';
26 }
27
```

我们使用`..`（双点）构造了一个相对路径，这是一种在大多数文件系统上引用父目录的常见方式（第 16 行）。然后我们使用`operator/=`来将额外的`..`路径元素附加到相对路径（第 17 行）。然后我们以其原生格式打印相对路径（第 18 行），并使用这个相对路径创建绝对路径。

`boost::filesystem::absolute`函数根据相对路径构造绝对路径。你可以将一个绝对路径传递给它，以便将相对路径附加到构造一个新的绝对路径（第 21 行）。请注意，我们传递了一个 Windows 绝对路径，并确保转义了反斜杠。如果省略`absolute`的第二个参数，它将使用进程的当前工作目录作为基本路径从相对路径构造绝对路径（第 23 行）。

例如，文件路径`/opt/boost/lib/../include`可以被*规范化*为等效形式`/opt/boost/include`。函数`boost::filesystem::canonical`从给定路径生成一个**规范化的绝对路径**（第 25 行），但要求路径存在。否则，它会抛出一个需要处理的异常。它还会读取并遵循路径中的任何符号链接。前面的代码在我的 Windows 系统上打印了以下输出：

```cpp
Relative path: ..\..
Absolute path: E:\DATA\photos\..\..
Absolute path wrt CWD: E:\DATA\Packt\Boost\Draft\Book\Chapter07\examples\..\..
Canonical: E:/DATA\Packt\Boost\Draft\Book
```

请注意，规范路径的输出中双点已经被折叠。

**清单 9.2c：处理错误**

```cpp
28 void handle_canonical_errors() {
29   fs::path p3 = "E:\\DATA"; // absolute path
30   auto p4 = p3 / "boost" / "boost_1_56";  // append elements
31   std::cout << p4.string() << '\n';
32   std::cout.put('\n');
33
34   boost::system::error_code ec;
35   auto p5 = p4 / ".." / "boost_1_100";  // append elements
36   auto p6 = canonical(p5, ec);
37
38   if (ec.value() == 0) {
39     std::cout << "Normalized: " << p6.string() << '\n';
40   } else {
41     std::cout << "Error (file=" << p5.string()
42           << ") (code=" << ec.value() << "): "
43           << ec.message() << '\n';
44   }
45 }
```

这个例子说明了当`canonical`被传递一个不存在的路径时会出错。我们创建了一个路径对象`p3`，表示 Windows 上的绝对路径`E:\DATA`（第 29 行）。然后我们通过使用`operator/`为`path`对象（第 30 行）连续添加路径元素（`boost`和`boost_1_56`）来创建第二个路径对象`p4`。这构造了一个等同于`E:\DATA\boost\boost_1_56`的路径。

接下来，我们将相对路径`../boost_1_100`附加到`p4`（第 35 行），这构造了一个等同于`E:\DATA\boost\boost_1_56\..\boost_1_100`的路径。这个路径在我的系统上不存在，所以当我在这个路径上调用`canonical`时，它会出错。请注意，我们将`boost::system::error_code`类型的对象作为`canonical`的第二个参数传递，以捕获任何错误。我们使用`error_code`的`value`成员函数（第 38 行）来检查返回的非零错误代码。如果发生错误，我们还可以使用`message`成员函数（第 43 行）检索系统定义的描述性错误消息。或者，我们可以调用`canonical`的另一个重载，它不接受`error_code`引用作为参数，而是在路径不存在时抛出异常。抛出异常和不抛出异常的重载是在文件系统库和其他来自 Boost 的系统编程库中常见的模式。

### 将路径分解为组件

在前一节中，我们看到了如何通过调用`parent_path`成员函数来获取路径的父目录。实际上，在`boost::filesystem::path`中有一整套成员函数可以提取路径中的组件。让我们首先看一下路径及其组件。

我们将首先了解 Boost 文件系统术语中关于路径组件的概念，使用来自 UNIX 系统的以下路径：

`/opt/boost/include/boost/filesystem/path.hpp`

前导`/`称为**根目录**。最后一个组件`path.hpp`称为**文件名**，即使路径表示的是目录而不是常规文件。剥离了文件名的路径（`/opt/boost/include/boost/filesystem`）称为**父路径**。在前导斜杠之后的部分（`opt/boost/include/boost/filesystem/path.hpp`）称为**相对路径**。

在前面的示例中，`.hpp`是**扩展名**（包括句点或点），`path`是文件名的**主干**。对于具有多个嵌入点的文件名（例如，`libboost_filesystem-mt.so.1.56.0`），扩展名被认为从最后（最右边）的点开始。

现在考虑以下 Windows 路径：

`E:\DATA\boost\include\boost\filesystem\path.hpp`

组件`E:`称为**根名称**。在`E:`后面的前导反斜杠称为**根目录**。根名称与根目录（`E:\`）的连接称为**根路径**。以下是一个打印路径的不同组件的简短函数，使用`boost::filesystem::path`的成员函数：

**清单 9.3：将路径拆分为组件**

```cpp
 1 #include <boost/filesystem.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 namespace fs = boost::filesystem;
 5
 6 void printPathParts(const fs::path& p1)
 7 {
 8 std::cout << "For path: " << p1.string() << '\n';
 9
10   if (p1.is_relative()) {
11     std::cout << "\tPath is relative\n";
12   } else {
13     assert(p1.is_absolute());
14     std::cout << "\tPath is absolute\n";
15   }
16
17   if (p1.has_root_name())
18     std::cout << "Root name: "
19               << p1.root_name().string() << '\n';
20
21   if (p1.has_root_directory())
22     std::cout << "Root directory: "
23               << p1.root_directory().string() << '\n';
24
25   if (p1.has_root_path())
26     std::cout << "Root path: "
27               << p1.root_path().string() << '\n';
28
29   if (p1.has_parent_path())
30     std::cout << "Parent path: "
31               << p1.parent_path().string() << '\n';
32
33   if (p1.has_relative_path())
34     std::cout << "Relative path: "
35               << p1.relative_path().string() << '\n';
36
37   if (p1.has_filename())
38     std::cout << "File name: "
39               << p1.filename().string() << '\n';
40
41   if (p1.has_extension())
42     std::cout << "Extension: "
43               << p1.extension().string() << '\n';
44
45   if (p1.has_stem())
46     std::cout << "Stem: " << p1.stem().string() << '\n';
47
48   std::cout << '\n';
49 }
50
51 int main()
52 {
53   printPathParts ("");                    // no components
54   printPathParts ("E:\\DATA\\books.txt"); // all components
55   printPathParts ("/root/favs.txt");      // no root name
56   printPathParts ("\\DATA\\books.txt");   // Windows, relative
57   printPathParts ("boost");              // no rootdir, no extn
58   printPathParts (".boost");              // no stem, only extn
59   printPathParts ("..");                  // no extension
60   printPathParts (".");                   // no extension
61   printPathParts ("/opt/boost/");         // file name == .
62 }
```

在前面的示例中，函数`printPathParts`（第 6 行）打印路径的尽可能多的组件。要访问路径组件，它使用`path`的相应成员函数。要检查组件是否可用，它使用`path`的`has_`成员函数之一。它还使用`path`的`is_relative`和`is_absolute`成员函数（第 10 行，第 13 行）检查路径是相对路径还是绝对路径。

我们使用不同的相对路径和绝对路径调用`printPathParts`。结果可能因操作系统而异。例如，在 Windows 上，对`has_root_name`（第 17 行）的调用对除了 Windows 路径`E:\DATA\books.txt`（第 54 行）之外的所有路径返回`false`，这被认为是绝对路径。对此路径调用`root_name`返回`E:`。然而，在 UNIX 上，反斜杠不被识别为分隔符，被认为是路径组件的一部分，因此`E:\DATA\books.txt`将被解释为具有文件名`E:\DATA\books.txt`的相对路径，主干`E:\DATA\books`和扩展名`.txt`。这，再加上在 Windows 上正斜杠被识别为路径分隔符的事实，是绝对不要像我们在这里所做的那样在路径文字中使用反斜杠的一个很好的理由。

### 注意

为了最大的可移植性，在路径文字中始终使用正斜杠，或者使用重载的`operator/`和`operator/=`生成路径。

我们还可以比较两个路径，看它们是否**相等**和**等效**。可以使用重载的`operator==`来比较两个路径是否相等，只有当两个路径可以分解为相同的组件时才返回`true`。请注意，这意味着路径`/opt`和`/opt/`不相等；在前者中，文件名组件是`opt`，而在后者中，它是`.`（点）。如果两个路径不相等，但仍然可以等效，如果它们表示相同的底层文件系统条目。例如，`/opt/boost`和`/opt/cmake/../boost/`虽然不是相等路径，但它们是等效的。要计算等效性，我们可以使用`boost::filesystem::equivalent`函数，如果两个路径引用文件系统中的相同条目，则返回`true`：

```cpp
boost::filesystem::path p1("/opt/boost"), p2("/opt/cmake");
if (boost::filesystem::equivalent(p1, p2 / ".." / "boost") {
  std::cout << "The two paths are equivalent\n";
}
```

与`boost::filesystem::canonical`一样，`equivalent`函数实际上也检查路径的存在，并且如果任一路径不存在则抛出异常。还有一个不会抛出异常而是设置`boost::system::error_code`输出参数的重载。

`path`对象可以被视为路径元素的序列容器，这些元素可以通过`path`公开的迭代器接口进行迭代。这允许将几个标准算法轻松应用于`path`对象。要遍历每个路径元素，我们可以使用以下代码片段：

```cpp
boost::filesystem::path p1("/opt/boost/include/boost/thread.hpp");
for (const auto& pathElem: p1) {
  std::cout <<pathElem.string() <<"  ";
}
```

这将打印由一对空格分隔的组件：

`/ optboost include boost thread.hpp`

`boost::filesystem::path`的`begin`和`end`成员函数返回类型为`boost::filesystem::path::iterator`的随机访问迭代器，您可以以有趣的方式与标准库算法一起使用。例如，要找到路径中的组件数，您可以使用：

```cpp
size_t count = std::distance(p1.begin(), p1.end());

```

现在，考虑两个路径：`/opt/boost/include/boost/filesystem/path.hpp`和`/opt/boost/include/boost/thread/detail/thread.hpp`。我们现在将编写一个函数，计算这两个路径所在的公共子目录：

第 9.4 节：查找公共前缀路径

```cpp
 1 #include <boost/filesystem.hpp>
 2 #include <iostream>
 3 namespace fs = boost::filesystem;
 4
 5 fs::path commonPrefix(const fs::path& first,
 6                       const fs::path& second) {
 7   auto prefix =
 8     [](const fs::path& p1, const fs::path& p2) {
 9       auto result =
10         std::mismatch(p1.begin(), p1.end(), p2.begin());
11       fs::path ret;
12       std::for_each(p2.begin(), result.second,
13               &ret {
14               ret /= p;
15               });
16       return ret;
17     };
18
19   size_t n1 = std::distance(first.begin(), first.end());
20   size_t n2 = std::distance(second.begin(), second.end());
21 
22   return (n1 < n2) ? prefix(first, second)
23                    : prefix(second, first);
24 }
```

在这两个路径上调用`commonPrefix`函数会正确返回`/opt/boost/include/boost`。为了使该函数正确工作，我们应该传递不包含`.`或`..`组件的路径，一个更完整的实现可以处理这个问题。为了计算前缀，我们首先使用 lambda 表达式定义了一个名为`prefix`的嵌套函数（第 7-17 行），它执行实际的计算。我们计算了两个路径的元素计数（第 19、20 行），并将较短的路径作为第一个参数，较长的路径作为第二个参数传递给`prefix`函数（第 22-23 行）。在`prefix`函数中，我们使用`std::mismatch`算法在两个路径上计算它们不匹配的第一个组件（第 10 行）。然后我们构造公共前缀作为直到第一个不匹配的路径，并返回它（第 12-15 行）。

## 遍历目录

Boost Filesystem 提供了两个迭代器类，`directory_iterator`和`recursive_directory_iterator`，使得遍历目录变得相当简单。两者都符合**输入迭代器**概念，并提供了用于向前遍历的`operator++`。在这里的第一个例子中，我们看到了`directory_iterator`的使用：

第 9.5 节：迭代目录

```cpp
 1 #include <boost/filesystem.hpp>
 2 #include <iostream>
 3 #include <algorithm>
 4 namespace fs = boost::filesystem;
 5
 6 void traverse(const fs::path& dirpath) {
 7   if (!exists(dirpath) || !is_directory(dirpath)) {
 8     return;
 9   }
10
11   fs::directory_iterator dirit(dirpath), end;
12
13   std::for_each(dirit, end, [](const fs::directory_entry& entry) {
14           std::cout <<entry.path().string() << '\n';
15         });
16 }
17
18 int main(int argc, char *argv[1]) {
19   if (argc > 1) {
20     traverse(argv[1]);
21   }
22 }
```

`traverse`函数接受一个类型为`boost::filesystem::path`的参数`dirpath`，表示要遍历的目录。使用命名空间级别的函数`exists`和`is_directory`（第 7 行），函数检查`dirpath`是否实际存在并且是一个目录，然后再继续。

为了执行迭代，我们为路径创建了一个`boost::filesystem::directory_iterator`的实例`dirit`，并创建了一个名为`end`的第二个默认构造的`directory_iterator`实例（第 11 行）。默认构造的`directory_iterator`充当了序列结束标记。对类型为`directory_iterator`的有效迭代器进行解引用会返回一个类型为`boost::filesystem::directory_entry`的对象。由迭代器范围`dirit`，`end`)表示的序列是目录中的条目列表。为了遍历它们，我们使用熟悉的`std::for_each`标准算法。我们使用 lambda 来定义对每个条目执行的操作，即简单地将其打印到标准输出（第 13-14 行）。

虽然我们可以围绕`boost::directory_iterator`编写递归逻辑来递归地遍历目录树，但`boost::recursive_directory_iterator`提供了一个更简单的替代方法。我们可以在第 9.5 节中用`boost::recursive_directory_iterator`替换`boost::directory_iterator`，它仍然可以工作，对目录树进行深度优先遍历。但是`recursive_directory_iterator`接口提供了额外的功能，比如跳过特定目录的下降和跟踪下降的深度。手写循环更好地利用了这些功能，如下例所示：

第 9.6 节：递归迭代目录

```cpp
 1 void traverseRecursive(const fs::path& path)
 2 {
 3   if (!exists(path) || !is_directory(path)) {
 4     return;
 5   }
 6
 7   try {
 8     fs::recursive_directory_iterator it(path), end;
 9
10     while (it != end) {
11       printFileProperties(*it, it.level());
12
13       if (!is_symlink(it->path())
14           && is_directory(it->path())
15           && it->path().filename() == "foo") {
16           it.no_push();
17       }
18       boost::system::error_code ec;
19       it.increment(ec);
21       if (ec) {
22         std::cerr << "Skipping entry: "
23                   << ec.message() << '\n';
24       }
25     }
26   } catch (std::exception& e) {
27     std::cout << "Exception caught: " << e.what() << '\n';
28   }
29 }
```

我们创建了一个`recursive_directory_iterator`并用一个路径初始化它（第 8 行），就像我们在第 9.5 节中为`directory_iterator`做的那样。如果路径不存在或程序无法读取，`recursive_directory_iterator`构造函数可能会抛出异常。为了捕获这种异常，我们将代码放在`try-catch`块中。

我们使用 while 循环来遍历条目（第 10 行），并通过调用`increment`成员函数（第 19 行）来推进迭代器。当`increment`成员函数遇到目录时，它会尝试按深度优先顺序进入该目录。这有时可能会由于系统问题而失败，比如当程序没有足够的权限查看目录时。在这种情况下，我们希望继续到下一个可用的条目，而不是中止迭代。因此，我们不在迭代器上使用`operator++`，因为当它遇到错误时会抛出异常，处理这种情况会使代码变得更加复杂。`increment`函数接受一个`boost::system::error_code`参数，在出现错误时设置`error_code`并推进迭代器到下一个条目。在这种情况下，我们可以使用`error_code`的`message`成员函数获取与错误相关的系统定义的错误消息。

### 注意

**boost::filesystem::recursive_directory_iterator 的行为**

在 Boost 版本 1.56 之前，当`operator++`和`increment`成员函数遇到错误时，它们只会抛出异常或设置`error_code`，而不会推进迭代器。这使得编写一个正确的循环以跳过错误变得更加复杂。从 Boost 1.56 开始，这些函数还会将迭代器推进到下一个条目，使循环代码变得简单得多。

我们通过调用一个虚构的函数`printFileProperties`（第 11 行）来处理每个条目，该函数接受两个参数——解引用`recursive_directory_iterator`实例的结果，以及通过调用迭代器的`level`成员函数获得的遍历深度。`level`函数对于一级目录返回零，并且对于每个额外的下降级别，其返回值递增 1。`printFileProperties`函数可以利用这一点来缩进子目录中的条目，例如。我们将在下一节中实现`printFileProperties`函数。

为了给这个例子增加维度，我们决定不进入名为`foo`的目录。为此，我们检查名为`foo`的目录（第 13-15 行），并在`recursive_directory_iterator`上调用`no_push`成员函数以防止进入该目录（第 16 行）。同样，我们可以随时调用迭代器的`pop`成员函数来在目录树中上升一级，而不一定要在当前级别完成迭代。

在支持符号链接的系统上，如果`recursive_directory_iterator`遇到指向目录的符号链接，它不会跟随链接进入目录。如果我们想要覆盖这种行为，我们应该向`recursive_directory_iterator`构造函数传递`boost::filesystem::symlink_option`枚举类型的第二个参数。`symlink_option`枚举提供了`none`（或`no_recurse`）（默认值）和`recurse`两个值，表示应该跟随符号链接进入目录。

## 查询文件系统条目

Boost Filesystem 提供了一组函数来对文件和目录执行有用的操作。其中大多数是`boost::filesystem`命名空间中的函数。使用这些函数，我们可以检查文件是否存在、其大小（以字节为单位）、最后修改时间、文件类型、是否为空等等。我们使用这些函数来编写我们在前一节中使用的`printFileProperties`函数：

**清单 9.7：查询文件系统条目**

```cpp
 1 #include <boost/filesystem.hpp>
 2 #include <iostream>
 3 #include <boost/date_time.hpp>
 4 namespace fs = boost::filesystem;
 5 namespace pxtm = boost::posix_time;
 6
 7 void printFileProperties(const fs::directory_entry& entry,
 8                          int indent = 0) {
 9   const fs::path& path= entry.path();
10   fs::file_status stat = entry.symlink_status();
11   std::cout << std::string(2*indent, '');
12
13   try {
14     if (is_symlink(path)) {
15       auto origin = read_symlink(path);
16       std::cout <<" L " << " -  - "
17                 << path.filename().string() << " -> "
18                 << origin.string();
19     } else if (is_regular_file(path)) {
20       std::cout << " F " << " "
21          << file_size(path) << " " << " "
22          << pxtm::from_time_t(last_write_time(path))
23          << " " << path.filename().string();
24     } else if (is_directory(path)) {
25       std::cout << " D " << " – " << " "
26 << pxtm::from_time_t(last_write_time(path))
27 << " " << path.filename().string();
28     } else {
29       switch (stat.type()) {
30       case fs::character_file:
31         std::cout << " C ";
32         break;
33       case fs::block_file:
34         std::cout << " B ";
35         break;
36       case fs::fifo_file:
37         std::cout << " P ";
38         break;
39       case fs::socket_file:
40         std::cout << " S ";
41         break;
42       default:
43         std::cout << " - ";
44         break;
45       }
46       std::cout << pxtm::from_time_t(last_write_time(path))
47                 << " ";
48       std::cout << path.filename().string();
49     }
50     std::cout << '\n';
51   } catch (std::exception& e) {
52     std::cerr << "Exception caught: " <<e.what() << '\n';
53   }
54 }
```

`printFileProperties`用于打印给定文件的简短摘要，包括类型、大小、最后修改时间、名称，以及对于符号链接，目标文件。这个函数的第一个参数是`directory_entry`类型，是对`directory_iterator`或`recursive_directory_iterator`的解引用的结果。我们通过调用`directory_entry`的`path`成员函数（第 9 行）获取到`directory_entry`对象引用的文件的路径。我们通过调用`directory_entry`的`symlink_status`成员函数（第 10 行）获取到`file_status`对象的引用。`file_status`对象包含有关文件系统条目的其他详细信息，我们在示例中使用它来打印特殊文件的状态。`symlink_status`函数作用于所有类型的文件，而不仅仅是符号链接，但它返回的是符号链接本身的状态，而不是跟随它到目标的状态。如果你需要每次查询符号链接时都需要目标的状态，使用`status`成员函数而不是`symlink_status`。`status`和`symlink_status`成员函数比同名的全局函数更快，因为它们会缓存文件状态，而不是在每次调用时查询文件系统。

在打印适合类型的信息之前，我们确定每个条目的类型。为此，我们使用方便的函数`is_symlink`、`is_regular_file`和`is_directory`（第 14、19、24 行）。在像 Linux 这样的 POSIX 系统上，还有其他类型的文件，如块和字符设备、管道和 Unix 域套接字。为了识别这些文件，我们使用之前获得的`file_status`对象（第 10 行）。我们调用`file_status`对象的`type`成员函数来确定特殊文件的确切类型（第 29 行）。请注意，我们首先检查文件是否是符号链接，然后进行其他测试。这是因为`is_regular_file`或`is_directory`对于目标文件的类型也可能返回 true，基于目标文件的类型。

这个函数以以下格式打印每个条目：

```cpp
file_type  sizetime  name -> target
```

文件类型由单个字母表示（`D`：目录，`F`：普通文件，`L`：符号链接，`C`：字符设备，`B`：块设备，`P`：管道，`S`：Unix 域套接字）。大小以字节为单位打印，最后修改时间以长整数形式打印，文件名打印时不包含完整路径。只有对于符号链接，名称后面会附加一个指向目标路径的箭头。当文件大小或最后写入时间不可用时，缺少字段会显示为连字符（`-`）。对于每个下降级别，条目都会缩进两个额外的空格（第 11 行）。

这是在我的 Linux 系统上运行此函数的示例输出：

![查询文件系统条目](img/1217OT_09_04.jpg)

你也可以在 Linux 的`/dev`目录上运行这个程序，看看设备文件是如何列出的。

调用`read_symlink`函数（第 15 行）来获取符号链接指向的目标文件。调用`file_size`函数（第 21 行）获取文件的大小（以字节为单位），调用`last_write_time`函数（第 22、26 和 46 行）获取文件的最后修改时间。`last_write_time`函数返回文件最后修改的**Unix 时间**。

我们通过调用`boost::posix_time::from_time_t`函数将这个数字时间戳转换为可打印的日期时间字符串来打印这个时间戳的有意义的表示（参见第七章，“高阶和编译时编程”）。为了构建这个程序，你还必须链接 Boost DateTime 库，如下所示：

```cpp
$ g++ listing8_7.cpp -o listing8_7 -std=c++11 -lboost_filesystem -lboost_date_time
```

文件系统中有几个这样的函数，用于查询文件系统中对象的不同类型的信息，例如查找文件的硬链接数。我们可以查询`file_status`对象（第 10 行）以获取文件权限。请注意，我们不需要在命名空间级别函数中加上命名空间；它们会根据参数的类型正确解析，使用基于参数类型的参数相关查找（Argument Dependent Lookup）。

## 对文件执行操作

除了查询文件系统条目的信息之外，我们还可以使用 Boost 文件系统库对文件执行操作，如创建目录和链接，复制文件和移动文件等。

### 创建目录

使用函数`boost::filesystem::create_directory`很容易创建目录。传递一个路径给它，如果该路径上不存在目录，则会在该路径上创建一个目录；如果目录已经存在，则不会执行任何操作。如果路径存在但不是一个目录，`create_directory`会抛出一个异常。还有一个非抛出版本，它接受一个`boost::system::error_code`引用，在错误时设置错误代码。这些函数如果创建了目录则返回`true`，如果没有则返回`false`：

**清单 9.8：创建目录**

```cpp 
 1 #include <boost/filesystem.hpp> 
 2 #include <iostream> 
 3 #include <cassert>	 
 4 namespace fs = boost::filesystem; 
 5 
 6 int main() { 
 7   fs::path p1 = "notpresent/dirtest"; 
 8   boost::system::error_code ec; 
 9   if (!is_directory(p1.parent_path()) || exists(p1)) {
10     assert( !create_directory(p1, ec) );
11
12     if (is_directory(p1)) assert(!ec.value());
13     else assert(ec.value());
14   }
15
16   try {
17     if (create_directories(p1)) {
18       assert( !create_directory(p1) );
19     }
20   } catch (std::exception& e) {
21     std::cout << "Exception caught: " << e.what() << '\n';
22   }
23 }
```

在这个例子中，相对于当前目录在路径`notpresent/dirtest`上调用`create_directory`失败（第 10 行），如果当前目录中没有名为`notpresent`的目录，或者`notpresent/dirtest`已经存在。这是因为`create_directory`期望传递的路径的父目录存在，并且不会创建已经存在的路径。如果我们没有传递错误代码参数，这次对`create_directory`的调用将会抛出一个需要处理的异常。如果`notpresent/dirtest`已经存在并且是一个目录，那么`create_directory`会失败，但不会设置错误代码（第 12 行）。

函数`boost::filesystem::create_directories`创建所需的所有路径组件，类似于 Unix 系统上的`mkdir -p`。对它的调用（第 17 行）除非存在权限问题或路径已经存在，否则会成功。它创建目录，包括沿路径缺失的任何目录。对`create_directory`和`create_directories`的调用是幂等的；如果目标目录存在，不会返回错误或抛出异常，但函数会返回`false`，因为没有创建新目录。

### 创建符号链接

符号链接，有时被称为软链接，是文件系统中的条目，类似于其他文件的别名。它们可以引用文件以及目录，并经常用于为文件和目录提供替代的简化名称和路径。符号链接在UNIX系统上已经存在了相当长的时间，并且自Windows 2000以来在Windows上以某种形式可用。我们可以使用函数`boost::filesystem::create_symlink`来创建符号链接。对于创建指向目录的符号链接，建议使用函数`boost::filesystem::create_directory_symlink`以获得更好的可移植性。

**清单9.9：创建符号链接**

```cpp
 1 #include <boost/filesystem.hpp>
 2 namespace fs = boost::filesystem; 
 3 
 4 void makeSymLink(const fs::path& target, const fs::path& link) { 
 5   boost::system::error_code ec; 
 6  
 7   if (is_directory(target)) { 
 8     create_directory_symlink(target, link); 
 9   } else {
10     create_symlink(target, link);
11   }
12 }
```
这显示了一个名为`makeSymLink`的函数，它创建指向给定路径的符号链接。函数的第一个参数是链接必须别名的目标路径，第二个参数是链接本身的路径。这种参数顺序让人联想到UNIX的`ln`命令。如果目标是目录，此函数调用`create_directory_symlink`（第8行），而对于所有其他情况，它调用`create_symlink`（第10行）。请注意，目标路径在创建符号链接时不需要存在，在这种情况下将创建悬空的符号链接。调用这些函数的效果与在POSIX系统上运行`ln -s target link`命令相同。在Windows上，当`target`是目录时，通过运行`mklink /D link target`命令可以获得相同的效果，当`target`不是目录时，通过运行`mklink link target`命令可以获得相同的效果。如果`create_directory_symlink`或`create_symlink`抛出异常，函数`makeSymLink`将抛出异常。

### 复制文件

复制文件是Boost文件系统中的另一个常见任务。`boost::filesystem::copy_file`函数将常规文件从源复制到目标，并且如果目标处已存在该文件，则会失败。使用适当的覆盖，可以使其覆盖目标处的文件。`boost::filesystem::copy_symlink`接受源符号链接并在目标处创建第二个符号链接，它别名与源相同的文件。您不能将目录传递给任何一个函数作为目标。还有一个`boost::copy_directory`函数，似乎并不做其名称所示的事情。它创建目录并将源目录的属性复制到目标目录。因此，我们将推出我们自己的递归目录复制实用程序函数：

第9.10节：递归复制目录

```cpp 
 1 void copyDirectory(const fs::path& src, const fs::path& target) { 
 2   if (!is_directory(src) 
 3     || (exists(target) && !is_directory(target)) 
 4     || !is_directory(absolute(target).parent_path()) 
 5     || commonPrefix(src, target) == src) { 
 6     throw std::runtime_error("Preconditions not satisfied"); 
 7   } 
 8 
 9   boost::system::error_code ec;
10   fs::path effectiveTarget = target;
11   if (exists(target)) {
12     effectiveTarget /= src.filename();
13   }
14   create_directory(effectiveTarget);
15
16   fs::directory_iterator iter(src), end;
17   while (iter != end) {
18     auto status = iter->symlink_status();
19     auto currentTarget = effectiveTarget/
20                               iter->path().filename();
21
22     if (status.type() == fs::regular_file) {
23       copy_file(*iter, currentTarget,
24                     fs::copy_option::overwrite_if_exists);
25     } else if (status.type() == fs::symlink_file) {
26       copy_symlink(*iter, currentTarget);
27     } else if (status.type() == fs::directory_file) {
28       copyDirectory(*iter, effectiveTarget);
29     } // else do nothing
30     ++iter;
31   }
32 }
```
第9.10节定义了`copyDirectory`函数，该函数递归地将源目录复制到目标目录。它执行基本验证，并在不满足必要的初始条件时抛出异常（第6行）。如果以下任何条件为真，则违反了必要的前提条件：

1.  源路径不是目录（第2行）

1.  目标路径存在，但不是目录（第3行）

1.  目标路径的父目录不是目录（第4行）

1.  目标路径是源路径的子目录（第5行）

为了检测违反4，我们重用了第9.4节中定义的`commonPrefix`函数。如果目标路径已经存在，则在其下创建与源目录同名的子目录以容纳复制的内容（第11-12行，14行）。否则，将创建目标目录并将内容复制到其中。

除此之外，我们使用`directory_iterator`而不是`recursive_directory_iterator`（第17行）来递归迭代源目录。我们使用`copy_file`来复制常规文件，传递`copy_option::overwrite_if_exists`选项以确保已存在的目标文件被覆盖（第23-24行）。我们使用`copy_symlink`来复制符号链接（第26行）。每次遇到子目录时，我们递归调用`copyDirectory`（第28行）。如果从`copyDirectory`调用的Boost文件系统函数抛出异常，它将终止复制。

### 移动和删除文件

您可以使用`boost::filesystem::rename`函数移动或重命名文件和目录，该函数以旧路径和新路径作为参数。两个参数的重载如果失败会抛出异常，而三个参数的重载则设置错误代码：

```cpp
void rename(const path& old_path, const path& new_path);
void rename(const path& old_path, const path& new_path,
            error_code& ec);
```

如果`new_path`不存在，且其父目录存在，则会创建它；否则，重命名调用失败。如果`old_path`不是目录，则`new_path`如果存在，也不能是目录。如果`old_path`是目录，则`new_path`如果存在，必须是一个空目录，否则函数失败。当一个目录被移动到另一个空目录时，源目录的内容被复制到目标空目录内，然后源目录被删除。重命名符号链接会影响链接本身，而不是它们所指向的文件。

您可以通过调用`boost::filesystem::remove`并传递文件系统条目的路径来删除文件和空目录。要递归删除一个非空目录，必须调用`boost::filesystem::remove_all`。

```cpp
bool remove(const path& p);
bool remove(const path& p, error_code& ec);
uintmax_t remove_all(const path& p);
uintmax_t remove_all(const path& p, error_code& ec);
```

如果路径指定的文件不存在，`remove`函数返回false。这会删除符号链接而不影响它们所指向的文件。`remove_all`函数返回它删除的条目总数。在错误情况下，`remove`和`remove_all`的单参数重载会抛出异常，而双参数重载会设置传递给它的错误代码引用，而不会抛出异常。

### 路径感知的fstreams

此外，头文件`boost/filesystem/fstream.hpp`提供了与`boost::filesystem::path`对象一起工作的标准文件流类的版本。当您编写使用`boost::filesystem`并且需要读取和写入文件的代码时，这些非常方便。

### 注意

最近，基于Boost文件系统库的C++技术规范已被ISO批准。这为其包含在未来的C++标准库修订版中铺平了道路。

# 使用Boost IOStreams进行可扩展I/O

标准库IOStreams设施旨在为各种设备上的各种操作提供一个框架，但它并没有被证明是最容易扩展的框架。Boost IOStreams库通过一个更简单的接口来补充这个框架，以便将I/O功能扩展到新设备，并提供一些非常有用的类来满足在读取和写入数据时的常见需求。

## Boost IOStreams的架构

标准库IOStreams框架提供了两个基本抽象，**流**和**流缓冲区**。流为应用程序提供了一个统一的接口，用于在底层设备上读取或写入一系列字符。流缓冲区为实际设备提供了一个更低级别的抽象，这些设备被流所利用和进一步抽象。

Boost IOStreams框架提供了`boost::iostreams::stream`和`boost::iostreams::stream_buffer`模板，这些是流和流缓冲区抽象的通用实现。这两个模板根据一组进一步的概念实现其功能，这些概念描述如下：

+   **源**是一个抽象，用于从中读取一系列字符的对象。

+   **汇**是一个抽象，用于向其写入一系列字符。

+   **设备**是源、汇，或两者兼有。

+   **输入过滤器**修改从源读取的一系列字符，而**输出过滤器**修改写入到汇之前的一系列字符。

+   **过滤器**是输入过滤器或输出过滤器。可以编写一个既可以用作输入过滤器又可以用作输出过滤器的过滤器；这被称为**双用过滤器**。

要在设备上执行I/O，我们将零个或多个过滤器序列与设备关联到`boost::iostreams::stream`的实例或`boost::iostreams::stream_buffer`的实例。一系列过滤器称为**链**，一系列过滤器以设备结尾称为**完整链**。

以下图表是输入和输出操作的统一视图，说明了流对象和底层设备之间的I/O路径：

![Boost IOStreams的架构](img/1217OT_09_01.jpg)

Boost IOStreams 架构

输入从设备中读取，并通过一个可选的过滤器堆栈传递到流缓冲区，从那里可以通过流访问。输出从流通过流缓冲区写入，并通过一堆过滤器传递到设备。如果有的话，过滤器会对从设备读取的数据进行操作，以向流的读取者呈现一个转换后的序列。它们还会对要写入设备的数据进行操作，并在写入之前进行转换。上面的图表用于可视化这些交互，但略有不准确；在代码中，过滤器不能同时作为输入过滤器和输出过滤器。

Boost IOStreams 库配备了几个内置的设备和过滤器类，并且也很容易创建我们自己的设备和过滤器。在接下来的章节中，我们将通过代码示例来说明 Boost IOStreams 库的不同组件的使用。

## 使用设备

设备提供了一个接口，用于向底层介质读写字符。它抽象了像磁盘、内存或网络连接这样的真实介质。在本书中，我们将专注于使用作为 Boost IOStreams 库一部分提供的许多现成的设备。编写我们自己的设备类的方法超出了本书的范围，但一旦您熟悉了本章内容，您应该很容易从在线文档中学习它们。

### 文件 I/O 的设备

Boost 定义了许多用于在文件上执行 I/O 的设备，我们首先看的是一个抽象平台特定文件描述符的设备。每个平台都使用一些本机句柄来打开文件，与标准 C++使用`fstream`表示打开文件的方式不同。例如，这些可以是 POSIX 系统上的整数文件描述符和 Windows 上的 HANDLE。Boost IOStreams 库提供了`boost::iostreams::file_descriptor_source`、`boost::iostreams::file_descriptor_sink`和`boost::iostreams::file_descriptor`设备，它们将 POSIX 文件描述符和 Windows 文件句柄转换为输入和输出的设备。在下面的示例中，我们使用`file_descriptor_source`对象使用流接口从 POSIX 系统上的文件中读取连续的行。如果您想要使用流接口来处理使用文件描述符进行文件打开的 I/O，这将非常有用。

**清单 9.11：使用 file_descriptor 设备**

```cpp
 1 #include <boost/iostreams/stream.hpp>
 2 #include <boost/iostreams/device/file_descriptor.hpp>
 3 #include <iostream>
 4 #include <string>
 5 #include <cassert>
 6 #include <sys/types.h>
 7 #include <fcntl.h>
 8 namespace io = boost::iostreams;
 9
10 int main(int argc, char *argv[]) {
11   if (argc < 2) {
12     return 0;
13   }
14
15   int fdr = open(argv[1], O_RDONLY);
16   if (fdr >= 0) {
17     io::file_descriptor_source fdDevice(fdr,
18                    io::file_descriptor_flags::close_handle);
19     io::stream<io::file_descriptor_source> in(fdDevice);
20     assert(fdDevice.is_open());
21
22     std::string line;
23     while (std::getline(in, line))
24     std::cout << line << '\n';
25   }
26 }
```

使用这个程序，我们打开命令行中命名的第一个文件，并从中读取连续的行。我们首先使用 Unix 系统调用`open`（第 15 行）打开文件，为此我们包括 Unix 头文件`sys/types.h`和`fcntl.h`（第 6-7 行）。如果文件成功打开（由`open`返回的文件描述符的正值表示），那么我们创建一个`file_descriptor_source`的实例，将打开的文件描述符和一个`close_handle`标志传递给它，以指示在设备被销毁时应适当关闭描述符（第 17-18 行）。

如果我们不希望设备管理描述符的生命周期，那么我们必须传递`never_close_handle`标志。然后我们创建一个`boost::iostreams::stream<file_descriptor_source>`的实例（第 19 行），将设备对象传递给它，并使用`std::getline`函数从中读取连续的行，就像我们使用任何`std::istream`实例一样（第 23 行）。请注意，我们使用`is_open`成员函数断言设备已经打开以供读取（第 19 行）。这段代码旨在在 Unix 和类 Unix 系统上编译。在 Windows 上，Visual Studio C 运行时库提供了兼容的接口，因此您也可以通过包括一个额外的头文件`io.h`来在 Windows 上编译和运行它。

### 注意

Boost IOStreams 库中的类型和函数分为一组相当独立的头文件，并没有一个单一的头文件包含所有符号。设备头文件位于`boost/iostreams/device`目录下，过滤器头文件位于`boost/iostreams/filter`目录下。其余接口位于`boost/iostreams`目录下。

要构建此程序，我们必须将其与`libboost_iostreams`库链接。我在我的 Ubuntu 系统上使用以下命令行，使用本机包管理器在默认路径下安装的 Boost 库来构建程序：

```cpp
$ g++ listing8_11.cpp -o listing8_11 -std=c++11 -lboost_iostreams

```

我们可能还希望构建我们的程序，以使用我们在第一章中从源代码构建的 Boost 库，*介绍 Boost*。为此，我在我的 Ubuntu 系统上使用以下命令行来构建此程序，指定包含路径和库路径，以及要链接的`libboost_iostreams-mt`库：

```cpp
$ g++listing8_11.cpp -o listing8_11-I /opt/boost/include -std=c++11 -L /opt/boost/lib -lboost_iostreams-mt -Wl,-rpath,/opt/boost/lib

```

要通过文件描述符写入文件，我们需要使用`file_descriptor_sink`对象。我们还可以使用`file_descriptor`对象来同时读取和写入同一设备。还有其他允许写入文件的设备——`file_source`，`file_sink`和`file`设备允许您读取和写入命名文件。`mapped_file_source`，`mapped_file_sink`和`mapped_file`设备允许您通过内存映射读取和写入文件。

### 用于读写内存的设备

标准库`std::stringstream`类系列通常用于将格式化数据读写到内存。如果要从任何给定的连续内存区域（如数组或字节缓冲区）中读取和写入，Boost IOStreams 库中的`array`设备系列（`array_source`，`array_sink`和`array`）非常方便：

**清单 9.12：使用数组设备**

```cpp
 1 #include <boost/iostreams/device/array.hpp>
 2 #include <boost/iostreams/stream.hpp>
 3 #include <boost/iostreams/copy.hpp>
 4 #include <iostream>
 5 #include <vector>
 6 namespace io = boost::iostreams;
 7
 8 int main() {
 9   char out_array[256];
10   io::array_sink sink(out_array, out_array + sizeof(out_array));
11   io::stream<io::array_sink> out(sink);
12   out << "Size of out_array is " << sizeof(out_array)
13       << '\n' << std::ends << std::flush;
14
15   std::vector<char> vchars(out_array,
16                           out_array + strlen(out_array));
17   io::array_source src(vchars.data(),vchars.size());
18   io::stream<io::array_source> in(src);
19
20   io::copy(in, std::cout);
21 }
```

此示例遵循与清单 9.11 相同的模式，但我们使用了两个设备，一个汇和一个源，而不是一个。在每种情况下，我们都执行以下操作：

+   我们创建一个适当初始化的设备

+   我们创建一个流对象并将设备与其关联

+   在流上执行输入或输出

首先，我们定义了一个`array_sink`设备，用于写入连续的内存区域。内存区域作为一对指针传递给设备构造函数，指向一个`char`数组的第一个元素和最后一个元素的下一个位置（第 10 行）。我们将这个设备与流对象`out`关联（第 11 行），然后使用插入操作符(`<<`)向流中写入一些内容。请注意，这些内容可以是任何可流化的类型，不仅仅是文本。使用操纵器`std::ends`（第 13 行），我们确保数组在文本之后有一个终止空字符。使用`std::flush`操纵器，我们确保这些内容不会保留在设备缓冲区中，而是在调用`out_array`（第 16 行）上的`strlen`之前找到它们的方式到汇流设备的后备数组`out_array`中。

接下来，我们创建一个名为`vchars`的`char`向量，用`out_array`的内容进行初始化（第 15-16 行）。然后，我们定义一个由这个`vector`支持的`array_source`设备，向构造函数传递一个指向`vchars`第一个元素的迭代器和`vchars`中的字符数（第 17 行）。最后，我们构造一个与该设备关联的输入流（第 18 行），然后使用`boost::iostreams::copy`函数模板将字符从输入流复制到标准输出（第 20 行）。运行上述代码将通过`array_sink`设备向`out_array`写入以下行：

```cpp
The size of out_array is 256
```

然后它读取短语中的每个单词，并将其打印到新行的标准输出中。

除了`array`设备，`back_insert_device`设备还可以用于适配几个标准容器作为 sink。`back_insert_device`和`array_sink`之间的区别在于，`array_sink`需要一个固定的内存缓冲区来操作，而`back_insert_device`可以使用任何具有`insert`成员函数的标准容器作为其后备存储器。这允许`back_insert_device`的底层内存区域根据输入的大小而增长。我们使用`back_insert_device`替换`array_sink`重写列表 9.12：

**列表 9.13：使用 back_insert_device**

```cpp
 1 #include <boost/iostreams/device/array.hpp>
 2 #include <boost/iostreams/device/back_inserter.hpp>
 3 #include <boost/iostreams/stream.hpp>
 4 #include <boost/iostreams/copy.hpp>
 5 #include <iostream>
 6 #include <vector>
 7 namespace io = boost::iostreams;
 8
 9 int main() {
10   typedef std::vector<char> charvec;
11   charvec output;
12   io::back_insert_device<charvec> sink(output);
13   io::stream<io::back_insert_device<charvec>> out(sink);
14   out << "Size of outputis "<< output.size() << std::flush;
15
16   std::vector<char> vchars(output.begin(),
17                            output.begin() + output.size());
18   io::array_source src(vchars.data(),vchars.size());
19   io::stream<io::array_source> in(src);
20
21   io::copy(in, std::cout);
22 }
```

在这里，我们写入`out_vec`，它是一个`vector<char>`（第 11 行），并且使用`back_insert_device` sink（第 12 行）进行写入。我们将`out_vec`的大小写入流中，但这可能不会打印在那时已经写入设备的字符总数，因为设备可能会在将输出刷新到向量之前对其进行缓冲。由于我们打算将这些数据复制到另一个向量以供读取（第 16-17 行），我们使用`std::flush`操纵器确保所有数据都写入`out_vec`（第 14 行）。

还有其他有趣的设备，比如`tee_device`适配器，允许将字符序列写入两个不同的设备，类似于 Unix 的`tee`命令。现在我们将看一下如何编写自己的设备。

## 使用过滤器

过滤器作用于写入到汇或从源读取的字符流，可以在写入和读取之前对其进行转换，或者仅仅观察流的一些属性。转换可以做各种事情，比如标记关键字，翻译文本，执行正则表达式替换，以及执行压缩或解压缩。观察者过滤器可以计算行数和单词数，或者计算消息摘要等。

常规流和流缓冲区不支持过滤器，我们需要使用**过滤流**和**过滤流缓冲区**来使用过滤器。过滤流和流缓冲区维护一个过滤器堆栈，源或汇在顶部，最外层的过滤器在底部，称为**链**的数据结构。

现在我们将看一下 Boost IOStreams 库作为一部分提供的几个实用过滤器。编写自己的过滤器超出了本书的范围，但优秀的在线文档详细介绍了这个主题。

### 基本过滤器

在使用过滤器的第一个示例中，我们使用`boost::iostreams::counter`过滤器来计算从文件中读取的文本的字符和行数：

**列表 9.14：使用计数器过滤器**

```cpp
 1 #include <boost/iostreams/device/file.hpp>
 2 #include <boost/iostreams/filtering_stream.hpp>
 3 #include <boost/iostreams/filter/counter.hpp>
 4 #include <boost/iostreams/copy.hpp>
 5 #include <iostream>
 6 #include <vector>
 7 namespace io = boost::iostreams;
 8
 9 int main(int argc, char *argv[]) {
10   if (argc <= 1) {
11     return 0;
12   }
13
14   io::file_source infile(argv[1]);
15   io::counter counter;
16   io::filtering_istream fis;
17   fis.push(counter);
18   assert(!fis.is_complete());
19   fis.push(infile);
20   assert(fis.is_complete());
21
22   io::copy(fis, std::cout);
23
24   io::counter *ctr = fis.component<io::counter>(0);
25   std::cout << "Chars: " << ctr->characters() << '\n'
26             << "Lines: " << ctr->lines() << '\n';
27 }
```

我们创建一个`boost::iostream::file_source`设备来读取命令行中指定的文件的内容（第 14 行）。我们创建一个`counter`过滤器来计算读取的行数和字符数（第 15 行）。我们创建一个`filtering_istream`对象（第 16 行），并推送过滤器（第 17 行），然后是设备（第 19 行）。在设备被推送之前，我们可以断言过滤流是不完整的（第 18 行），一旦设备被推送，它就是完整的（第 20 行）。我们将从过滤输入流中读取的内容复制到标准输出（第 22 行），然后访问字符和行数。

要访问计数，我们需要引用过滤流内部的链中的`counter`过滤器对象。为了做到这一点，我们调用`filtering_istream`的`component`成员模板函数，传入我们想要的过滤器的索引和过滤器的类型。这将返回一个指向`counter`过滤器对象的指针（第 24 行），我们通过调用适当的成员函数（第 25-26 行）检索读取的字符和行数。

在下一个示例中，我们使用`boost::iostreams::grep_filter`来过滤掉空行。与不修改输入流的计数器过滤器不同，这个过滤器通过删除空行来转换输出流。

**列表 9.15：使用 grep_filter**

```cpp
 1 #include <boost/iostreams/device/file.hpp>
 2 #include <boost/iostreams/filtering_stream.hpp>
 3 #include <boost/iostreams/filter/grep.hpp>
 4 #include <boost/iostreams/copy.hpp>
 5 #include <boost/regex.hpp>
 6 #include <iostream>
 7 namespace io = boost::iostreams;
 8
 9 int main(int argc, char *argv[]) {
10   if (argc <= 1) {
11     return 0;
12   }
13
14   io::file_source infile(argv[1]);
15   io::filtering_istream fis;
16   io::grep_filter grep(boost::regex("^\\s*$"),
17       boost::regex_constants::match_default, io::grep::invert);
18   fis.push(grep);
19   fis.push(infile);
20
21   io::copy(fis, std::cout);
22 }
```

这个例子与列表 9.14 相同，只是我们使用了不同的过滤器`boost::iostreams::grep_filter`来过滤空行。我们创建了`grep_filter`对象的一个实例，并向其构造函数传递了三个参数。第一个参数是匹配空行的正则表达式`^\s*$`（第 16 行）。请注意，反斜杠在代码中被转义了。第二个参数是常量`match_default`，表示我们使用 Perl 正则表达式语法（第 17 行）。第三个参数`boost::iostreams::grep::invert`告诉过滤器只允许匹配正则表达式的行被过滤掉（第 17 行）。默认行为是只过滤掉不匹配正则表达式的行。

要在 Unix 上构建此程序，您还必须链接到 Boost Regex 库：

```cpp
$ g++ listing8_15.cpp -o listing8_15 -std=c++11 -lboost_iostreams-lboost_regex

```

在没有 Boost 本机包并且 Boost 安装在自定义位置的系统上，使用以下更详细的命令行：

```cpp
$ g++ listing8_15.cpp -o listing8_15-I /opt/boost/include -std=c++11 -L /opt/boost/lib -lboost_iostreams-mt-lboost_regex-mt -Wl,-rpath,/opt/boost/lib

```

在 Windows 上，使用 Visual Studio 并启用自动链接到 DLL，您不需要显式指定 Regex 或 IOStream DLL。

### 压缩和解压过滤器

Boost IOStreams 库配备了三种不同的数据压缩和解压过滤器，分别用于 gzip、zlib 和 bzip2 格式。gzip 和 zlib 格式实现了不同变种的 DEFLATE 算法进行压缩，而 bzip2 格式则使用更节省空间的 Burrows-Wheeler 算法。由于这些是外部库，如果我们使用这些压缩格式，它们必须被构建和链接到我们的可执行文件中。如果您已经按照第一章中概述的详细步骤构建了支持 zlib 和 bzip2 的 Boost 库，那么 zlib 和 bzip2 共享库应该已经与 Boost Iostreams 共享库一起构建了。

在下面的例子中，我们压缩了一个命令行中命名的文件，并将其写入磁盘。然后我们读取它，解压它，并将其写入标准输出。

**列表 9.16：使用 gzip 压缩器和解压器**

```cpp
 1 #include <boost/iostreams/device/file.hpp>
 2 #include <boost/iostreams/filtering_stream.hpp>
 3 #include <boost/iostreams/stream.hpp>
 4 #include <boost/iostreams/filter/gzip.hpp>
 5 #include <boost/iostreams/copy.hpp>
 6 #include <iostream>
 7 namespace io = boost::iostreams;
 8
 9 int main(int argc, char *argv[]) {
10   if (argc <= 1) {
11     return 0;
12   }
13   // compress
14   io::file_source infile(argv[1]);
15   io::filtering_istream fis;
16   io::gzip_compressor gzip;
17   fis.push(gzip);
18   fis.push(infile);
19
20   io::file_sink outfile(argv[1] + std::string(".gz"));
21   io::stream<io::file_sink> os(outfile);
22   io::copy(fis, os);
23
24   // decompress
25   io::file_source infile2(argv[1] + std::string(".gz"));
26   fis.reset();
27   io::gzip_decompressor gunzip;
28   fis.push(gunzip);
29   fis.push(infile2);
30   io::copy(fis, std::cout);
31 }
```

前面的代码首先使用`boost::iostreams::gzip_compressor`过滤器（第 16 行）在读取文件时解压文件（第 17 行）。然后使用`boost::iostreams::copy`将这个内容写入一个带有`.gz`扩展名的文件中，该扩展名附加到原始文件名上（第 20-22 行）。对`boost::iostreams::copy`的调用还会刷新和关闭传递给它的输出和输入流。因此，在`copy`返回后立即从文件中读取是安全的。为了读取这个压缩文件，我们使用一个带有`boost::iostreams::gzip_decompressor`的`boost::iostreams::file_source`设备（第 27-28 行），并将解压后的输出写入标准输出（第 30 行）。我们重用`filtering_istream`对象来读取原始文件，然后再次用于读取压缩文件。在过滤流上调用`reset`成员函数会关闭并删除与流相关的过滤器链和设备（第 26 行），因此我们可以关联一个新的过滤器链和设备（第 27-28 行）。

通过向压缩器或解压器过滤器的构造函数提供额外的参数，可以覆盖几个默认值，但基本结构不会改变。通过将头文件从`gzip.hpp`更改为`bzip2.hpp`（第 4 行），并在前面的代码中用`bzip2_compressor`和`bzip2_decompressor`替换`gzip_compressor`和`gzip_decompressor`，我们可以测试 bzip2 格式的代码；同样适用于 zlib 格式。理想情况下，扩展名应该适当更改（.bz2 用于 bzip2，.zlib 用于 zlib）。在大多数 Unix 系统上，值得测试生成的压缩文件，通过使用 gzip 和 bzip2 工具单独解压缩它们。对于 zlib 存档的命令行工具似乎很少，且标准化程度较低。在我的 Ubuntu 系统上，`qpdf`程序带有一个名为`zlib-flate`的原始 zlib 压缩/解压缩实用程序，可以压缩到 zlib 格式并从 zlib 格式解压缩。

构建此程序的步骤与构建清单 9.15 时的步骤相同。即使使用`zlib_compressor`或`bzip2_compressor`过滤器，只要在链接期间使用选项`-Wl,-rpath,/opt/boost/lib`，链接器（以及稍后的运行时链接器在执行期间）将自动选择必要的共享库，路径`/opt/boost/lib`包含 zlib 和 bzip2 的共享库。

### 组合过滤器

过滤流可以在管道中对字符序列应用多个过滤器。通过在过滤流上使用`push`方法，我们可以形成以最外层过滤器开始的管道，按所需顺序插入过滤器，并以设备结束。

这意味着对于过滤输出流，您首先推送首先应用的过滤器，然后向前推送每个连续的过滤器，最后是接收器。例如，为了过滤掉一些行并在写入接收器之前进行压缩，推送的顺序将如下所示：

```cpp
filtering_ostream fos;
fos.push(grep);
fos.push(gzip);
fos.push(sink);
```

对于过滤输入流，您需要推送过滤器，从最后应用的过滤器开始，然后逆向工作，推送每个前置过滤器，最后是源。例如，为了读取文件，解压缩它，然后执行行计数，推送的顺序将如下所示：

```cpp
filtering_istream fis;
fis.push(counter);
fis.push(gunzip);
fis.push(source);
```

#### 管道

原来一点点的操作符重载可以使这个过程更加具有表现力。我们可以使用管道操作符（`operator|`）以以下替代符号来编写前面的链：

```cpp
filtering_ostream fos;
fos.push(grep | gzip | sink);

filtering_istream fis;
fis.push(counter | gunzip | source);
```

前面的片段显然更具表现力，代码行数更少。从左到右，过滤器按照您将它们推入流中的顺序串联在一起，最后是设备。并非所有过滤器都可以以这种方式组合，但来自 Boost IOStreams 库的许多现成的过滤器可以；更明确地说，过滤器必须符合**可管道化概念**才能以这种方式组合。以下是一个完整的示例程序，该程序读取文件中的文本，删除空行，然后使用 bzip2 进行压缩：

**清单 9.17：使用管道过滤器**

```cpp
 1 #include <boost/iostreams/device/file.hpp>
 2 #include <boost/iostreams/filtering_stream.hpp>
 3 #include <boost/iostreams/stream.hpp>
 4 #include <boost/iostreams/filter/bzip2.hpp>
 5 #include <boost/iostreams/filter/grep.hpp>
 6 #include <boost/iostreams/copy.hpp>
 7 #include <boost/regex.hpp>
 8 #include <iostream>
 9 namespace io = boost::iostreams;
10
11 int main(int argc, char *argv[]) {
12   if (argc <= 1) { return 0; }
13
14   io::file_source infile(argv[1]);
15   io::bzip2_compressor bzip2;
16   io::grep_filter grep(boost::regex("^\\s*$"),
17         boost::regex_constants::match_default,
18         io::grep::invert);
19   io::filtering_istream fis;
20   fis.push(bzip2 | grep | infile);
21   io::file_sink outfile(argv[1] + std::string(".bz2"));
22   io::stream<io::file_sink> os(outfile);
23
24   io::copy(fis, os);
25 }
```

前面的示例将一个用于过滤空行的 grep 过滤器（第 16-18 行）和一个 bzip2 压缩器（第 15 行）与使用管道的文件源设备串联在一起（第 20 行）。代码的其余部分应该与清单 9.15 和 9.16 相似。

#### 使用 tee 分支数据流

在使用具有多个过滤器的过滤器链时，有时捕获两个过滤器之间流动的数据是有用的，特别是用于调试。`boost::iostreams::tee_filter`是一个输出过滤器，类似于 Unix 的`tee`命令，它位于两个过滤器之间，并提取两个过滤器之间流动的数据流的副本。基本上，当您想要在处理的不同中间阶段捕获数据时，可以使用`tee_filter`：

![使用 tee 分支数据流](img/1217OT_09_03.jpg)

您还可以复用两个接收设备来创建一个**tee 设备**，这样将一些内容写入 tee 设备会将其写入底层设备。`boost::iostream::tee_device`类模板结合了两个接收器以创建这样的 tee 设备。通过嵌套 tee 设备或管道化 tee 过滤器，我们可以生成几个可以以不同方式处理的并行流。`boost::iostreams::tee`函数模板可以生成 tee 过滤器和 tee 流。它有两个重载——一个单参数重载，接收一个接收器并生成一个`tee_filter`，另一个双参数重载，接收两个接收器并返回一个`tee_device`。以下示例显示了如何使用非常少的代码将文件压缩为三种不同的压缩格式（gzip、zlib 和 bzip2）：

**清单 9.18：使用 tee 分支输出流**

```cpp
 1 #include <boost/iostreams/device/file.hpp>
 2 #include <boost/iostreams/filtering_stream.hpp>
 3 #include <boost/iostreams/stream.hpp>
 4 #include <boost/iostreams/filter/gzip.hpp>
 5 #include <boost/iostreams/filter/bzip2.hpp>
 6 #include <boost/iostreams/filter/zlib.hpp>
 7 #include <boost/iostreams/copy.hpp>
 8 #include <boost/iostreams/tee.hpp>
 9 namespace io = boost::iostreams;
10
11 int main(int argc, char *argv[]) {
12   if (argc <= 1) { return 0; }
13
14   io::file_source infile(argv[1]);  // input
15   io::stream<io::file_source> ins(infile);
16
17   io::gzip_compressor gzip;
18   io::file_sink gzfile(argv[1] + std::string(".gz"));
19   io::filtering_ostream gzout;     // gz output
20   gzout.push(gzip | gzfile);
21   auto gztee = tee(gzout);
22
23   io::bzip2_compressor bzip2;
24   io::file_sink bz2file(argv[1] + std::string(".bz2"));
25   io::filtering_ostream bz2out;     // bz2 output
26   bz2out.push(bzip2 | bz2file);
27   auto bz2tee = tee(bz2out);
28
29   io::zlib_compressor zlib;
30   io::file_sink zlibfile(argv[1] + std::string(".zlib"));
31
32   io::filtering_ostream zlibout;
33   zlibout.push(gztee | bz2tee | zlib | zlibfile);
34
35   io::copy(ins, zlibout);
36 }
```

我们为 gzip、bzip2 和 zlib 设置了三个压缩过滤器（第 17、23 和 29 行）。我们需要为每个输出文件创建一个`filtering_ostream`。我们为 gzip 压缩输出创建了`gzout`流（第 20 行），为 bzip2 压缩输出创建了`bz2out`流（第 26 行）。我们在这两个流周围创建了 tee 过滤器（第 21 和 27 行）。最后，我们将 gztee、bz2tee 和 zlib 连接到 zlibfile 接收器前面，并将此链推入 zlibout 的`filtering_ostream`中，用于 zlib 文件（第 33 行）。从输入流`ins`复制到输出流`zlibout`会生成管道中的三个压缩输出文件，如下图所示：

![使用 tee 分支数据流](img/1217OT_09_02.jpg)

请注意，对 tee 的调用没有命名空间限定，但由于参数相关查找（见第二章，“使用 Boost 实用工具的第一次尝试”），它们得到了正确的解析。

Boost IOStreams 库提供了一个非常丰富的框架，用于编写和使用设备和过滤器。本章仅介绍了此库的基本用法，还有许多过滤器、设备和适配器可以组合成有用的 I/O 模式。

# 自测问题

对于多项选择题，选择所有适用的选项：

1.  对于操作路径的`canonical`和`equivalent`函数有什么独特之处？

a. 参数不能命名真实路径。

b. 两者都是命名空间级别的函数。

c. 参数必须命名真实路径。

1.  以下代码片段的问题是什么，假设路径的类型是`boost::filesystem::path`？

```cpp
if (is_regular_file(path)) { /* … */ }
else if (is_directory(path)) { /* … */ }
else if (is_symlink(path)) { /* … */ }
```

a. 它必须有静态的`value`字段。

b. 它必须有一个名为`type`的嵌入类型。

c. 它必须有静态的`type`字段。

d. 它必须有一个名为`result`的嵌入类型。

1.  考虑到这段代码：

```cpp
boost::filesystem::path p1("/opt/boost/include/boost/thread.hpp");
size_t n = std::distance(p1.begin(), p1.end());
```

n 的值是多少？

a. 5，路径中组件的总数。

b. 6，路径中组件的总数。

c. 10，斜杠和组件数量的总和。

d. 4，目录组件的总数。

1.  您想要读取一个文本文件，使用`grep_filter`删除所有空行，使用`regex_filter`替换特定关键词，并计算结果中的字符和行数。您将使用以下哪个管道？

a. `file_source | grep_filter| regex_filter | counter`

b. `grep_filter | regex_filter | counter | file_source`

c. `counter | regex_filter | grep_filter |file_source`

d. `file_source | counter | grep_filter | regex_filter`

1.  真或假：tee 过滤器不能与输入流一起使用。

a. 真。

b. 错误。

# 总结

在本章中，我们介绍了 Boost Filesystem 库，用于读取文件元数据和文件和目录状态，并对它们执行操作。我们还介绍了高级 Boost IOStreams 框架，用于执行具有丰富语义的类型安全 I/O。

处理文件和执行 I/O 操作是基本的系统编程任务，几乎任何有用的软件都需要执行这些任务，而我们在本章中介绍的 Boost 库通过一组可移植的接口简化了这些任务。在下一章中，我们将把注意力转向另一个系统编程主题——并发和多线程。
