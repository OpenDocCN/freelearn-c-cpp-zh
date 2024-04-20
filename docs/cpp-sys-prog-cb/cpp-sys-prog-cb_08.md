# 处理控制台 I/O 和文件

本章涵盖了基于 C++标准库的控制台、流和文件 I/O 的示例。我们在其他章节中已经读取了程序中的参数，但还有其他几种方法可以做到这一点。我们将深入研究这些主题，并学习每种主题的替代方法、技巧和最佳实践，具体而专门的实践示例。

我们的主要重点再次是尽可能多地使用 C++（及其标准库）来编写系统编程软件，因此代码将具有非常有限的 C 和 POSIX 解决方案。

本章将涵盖以下主题：

+   实现与控制台 I/O 的交互

+   操作 I/O 字符串

+   处理文件

# 技术要求

为了让您从一开始就尝试这些程序，我们设置了一个 Docker 镜像，其中包含了本书中将需要的所有工具和库。它基于 Ubuntu 19.04。

为了设置它，请按照以下步骤操作：

1.  从[www.docker.com](https://www.docker.com/)下载并安装 Docker Engine。

1.  从 Docker 中拉取图像

Hub：`docker pull kasperondocker/system_programming_cookbook:latest`

1.  现在应该可以使用图像。输入以下命令查看图像：`docker images`

1.  现在应该有这个镜像：`kasperondocker/system_programming_cookbook`

1.  使用以下命令运行 Docker 镜像，并使用交互式 shell：`docker run -it **-**-cap-add sys_ptrace kasperondocker/system_programming_cookbook:latest /bin/bash`

1.  正在运行的容器上现在可用 shell。使用`root@39a5a8934370/# cd /BOOK/`获取本书中开发的所有程序，按章节组织。

需要`--cap-add sys_ptrace`参数，以允许 Docker 容器中的 GDB 设置断点，Docker 默认情况下不允许。

# 实现与控制台 I/O 的交互

这个示例专注于控制台 I/O。我们编写的大多数程序都需要与用户进行某种交互：我们需要获取输入，进行一些处理，然后返回输出。例如，想象一下您可以在一个应用程序中收集的用户输入。在这个示例中，我们将编写代码，展示从控制台获取输入和返回输出的不同方法。

# 如何做...

让我们写一些代码：

1.  在运行 Docker 镜像的情况下，让我们创建一个名为`console_01.cpp`的新文件，并将以下代码输入其中：

```cpp
#include <iostream>
#include <string>
int main ()
{
    std::string name;
    std::cout << "name: ";
    std::cin >> name;

    std::string surname;
    std::cout << "surname: ";
    std::cin >> surname;

    int age;
    std::cout << "age: ";
    std::cin >> age;

    std::cout << "Hello " << name << ", " 
              << surname << ": " << age << std::endl;
    return 0;
}
```

1.  现在创建另一个名为`console_02.cpp`的文件，并输入以下代码以查看此方法的限制：

```cpp
#include <iostream>
#include <string>
int main ()
{
    std::string fullNameWithCin;
    std::cout << "full Name got with cin: ";
    std::cin >> fullNameWithCin;

    std::cout << "hello " << fullNameWithCin << std::endl;
    return 0;
}
```

1.  最后，让我们创建一个新文件并命名为`console_03.cpp`；让我们看看`std::getline`和`std::cin`如何克服这个先前的限制：

```cpp
#include <iostream>
#include <string>

int main ()
{
    std::string fullName;
    std::cout << "full Name: ";
    std::getline (std::cin, fullName);
    std::cout << "Hello " << fullName << std::endl;
    return 0;
}
```

尽管这些都是非常简单的示例，但它们展示了使用 C++与控制台标准输入和输出进行交互的方式。

# 工作原理...

在第一步中，`console_01.cpp`程序只使用`std::cin`和`std::cout`来获取用户的`name`和`surname`信息，并将其保存在`std::string`变量中。这些是在需要与标准输入和输出进行简单交互时要使用的第一件事情。通过构建和运行`console_01.cpp`文件，我们将得到以下输出：

![](img/2c32601b-b89d-43d4-857c-f37964704b56.png)

该示例的第二步显示了`std::cin`和`std::cout`的限制。用户在命令行中向正在运行的进程提供`name`和`surname`，但奇怪的是，`fullNameWithCin`变量中只存储了名字，完全跳过了姓氏。为什么？原因很简单：`std:cin`总是将空格、制表符或换行符视为从标准输入中捕获的值的分隔符。那么我们如何从标准输入中获取完整的行呢？通过编译和运行`console_02.cpp`，我们得到以下结果：

![](img/ebac7a09-1cfb-49bb-aa61-fe0dcce7482a.png)

第三步展示了`getline`函数与`std::cin`结合使用，从标准输入获取完整的行。`std::getline`从`std::cin`获取行并将其存储在`fullName`变量中。一般来说，`std::getline`接受任何`std::istream`作为输入，并有可能指定分隔符。标准库中可用的原型如下：

```cpp
istream& getline (istream& is, string& str, char delim);
istream& getline (istream&& is, string& str, char delim);
istream& getline (istream& is, string& str);
istream& getline (istream&& is, string& str);
```

这使得`getline`成为一个非常灵活的方法。通过构建和运行`console_03.cpp`，我们得到以下输出：

![](img/313c17da-22e0-4075-9502-54c86b4e5119.png)

让我们看看下面的例子，我们将一个流传递给方法，用于存储提取的信息片段的变量，以及分隔符：

```cpp
#include <iostream>
#include <string>
#include <sstream>

int main ()
{
    std::istringstream ss("ono, vaticone, 43");

    std::string token;
    while(std::getline(ss, token, ','))
    {
        std::cout << token << '\n';
    }

    return 0;
}
```

前面方法的输出如下：

![](img/6272f92b-7756-45a2-b4c5-a7fb7102e7b7.png)

这可以为构建自己的标记方法奠定基础。

# 还有更多...

`std::cin`和`std::cout`允许链式请求，这使得代码更易读和简洁：

```cpp
std::cin >> name >> surname;
std::cout << name << ", " << surname << std::endl;
```

`std::cin`期望用户传递他们的名字，然后是他们的姓氏。它们必须用空格、制表符或换行符分隔。

# 另请参阅

+   *学习如何操作 I/O 字符串*配方涵盖了如何操作字符串作为控制台 I/O 的补充。

# 学习如何操作 I/O 字符串

字符串操作是几乎任何软件的一个非常重要的方面。能够简单有效地操作字符串是软件开发的一个关键方面。你将如何读取应用程序的配置文件或解析它？这个配方将教你 C++提供了哪些工具，使这成为一个愉快的任务，使用`std::stringstream`类。

# 如何做...

在这一部分，我们将使用`std::stringstream`开发一个程序来解析流，这些流实际上可以来自任何来源：文件、字符串、输入参数等等。

1.  让我们开发一个程序，打印文件的所有条目。将以下代码输入到一个新的 CPP 文件`console_05.cpp`中：

```cpp
#include <iostream>
#include <string>
#include <fstream>

int main ()
{
    std::ifstream inFile ("file_console_05.txt", std::ifstream::in);
    std::string line;
    while( std::getline(inFile, line) )
        std::cout << line << std::endl;

    return 0;
}
```

1.  当我们需要将字符串解析为变量时，`std::stringstream`非常方便。让我们通过在一个新文件`console_06.cpp`中编写以下代码来看看它的作用：

```cpp
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

int main ()
{
    std::ifstream inFile ("file_console_05.txt",
        std::ifstream::in);
    std::string line;
    while( std::getline(inFile, line) )
    {
        std::stringstream sline(line);
        std::string name, surname; 
        int age{};
        sline >> name >> surname >> age;
        std::cout << name << "-" << surname << "-"<< age << 
            std::endl;
    }
    return 0;
}
```

1.  而且，为了补充第二步，解析和创建字符串流也很容易。让我们在`console_07.cpp`中做这个：

```cpp
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

int main ()
{
    std::stringstream sline;
    for (int i = 0; i < 10; ++i)
        sline << "name = name_" << i << ", age = " << i*7 << 
            std::endl;

    std::cout << sline.str();
    return 0;
}
```

前面的三个程序展示了在 C++中解析字符串是多么简单。下一节将逐步解释它们。

# 它是如何工作的...

*步骤 1*表明`std::getline`接受任何流作为输入，不仅仅是标准输入（即`std::cin`）。在这种情况下，它获取来自文件的流。我们包括`iostream`用于`std::cout`，`string`用于使用字符串，以及`fstream`用于读取文件。

然后，我们使用`std::fstream`（文件流）打开`file_console_05.txt`文件。在它的构造函数中，我们传递文件名和标志（在这种情况下，只是信息，它是一个带有`std::ifstream::in`的输入文件）。我们将文件流传递给`std::getline`，它将负责将每行从流中复制并存储在`std::string`变量`line`中，然后将其打印出来。这个程序的输出如下：

![](img/867418ac-6e1b-4f44-ba9c-7ce3e3c45e78.png)

*步骤 2*展示了相同的程序读取`file_console_05.txt`文件，但是这次我们想解析文件的每一行。我们通过将`line`字符串变量传递给`sline` `std::stringstream`变量来实现这一点。`std::stringstream`提供了方便和易于使用的解析能力。

只需写入一行`sline >> name >> surname >> age`，`std::stringstream`类的`operator>>`将把`name`、`surname`和`age`保存到相应的变量中，并处理类型转换（即对于`age`变量，从`string`到`int`），假设这些变量按照这个顺序出现在文件中。`operator>>`将解析字符串，并通过跳过前导**空格**，对每个标记调用适当的方法（例如`basic_istream& operator>>( short& value );`或`basic_istream& operator>>( long long& value );`等）。该程序的输出如下：

![](img/137a9b31-5a6d-45c2-9966-1de8b3c8cc6b.png)

*步骤 3*表明，将流解析为变量的简单性也适用于构建流。相同的`std::stringstream`变量`sline`与`<<`运算符一起使用，表示数据流现在流向`string stream`变量，该变量在以下截图中以两行打印到标准输出。该程序的输出如下：

![](img/0bfd621e-e616-449c-b2a4-5c4a1be90335.png)

`std::stringstream`使得解析字符串和流变得非常容易，无论它们来自何处。

# 还有更多...

如果您正在寻找低延迟，使用`std::stringstream`进行流操作可能不是您的首选。我们始终建议您测量性能并根据数据做出决定。如果是这种情况，您可以尝试不同的解决方案：

+   如果可以的话，只需专注于代码的低延迟部分进行优化。

+   使用标准的 C 或 C++方法编写您的层来解析数据，例如典型的`atoi()`方法。

+   使用任何开源低延迟框架。

# 另请参阅

+   *实现与控制台之间的 I/O*教程介绍了如何处理来自控制台的 I/O。

# 处理文件

这个教程将教会你处理文件所需的基本知识。C++标准库在历史上提供了一个非常好的接口，但 C++ 17 添加了一个名为`std::filesystem`的命名空间，进一步丰富了功能。尽管如此，我们不会利用 C++17 的`std::filesystem`命名空间，因为它已经在第二章中介绍过了，*重温 C++*。想想一个具体的用例，比如创建一个配置文件，或者你需要复制该配置文件的情况。这个教程将教会你如何使用 C++轻松完成这个任务。

# 如何做...

在本节中，我们将编写三个程序，学习如何使用`std::fstream`、`std::ofstream`和`std::ifstream`处理文件：

1.  让我们开发一个程序，通过使用`std::ofstream`打开并写入一个新文件`file_01.cpp`：

```cpp
#include <iostream>
#include <fstream>

int main ()
{
    std::ofstream fout;
    fout.open("file_01.txt");

    for (int i = 0; i < 10; ++i)
        fout << "User " << i << " => name_" << i << " surname_" 
            << i << std::endl;

    fout.close();
}
```

1.  在一个新的源文件`file_02.cpp`中，让我们从文件中读取并打印到标准输出：

```cpp
#include <iostream>
#include <fstream>

int main ()
{
    std::ifstream fiut;
    fiut.open("file_01.txt");

    std::string line;
    while (std::getline(fiut, line))
        std::cout << line << std::endl;

    fiut.close();
}
```

1.  现在我们想要结合打开文件进行读写的灵活性。我们将使用`std::fstream`将`file_01.txt`的内容复制到`file_03.txt`，然后打印其内容。在另一个源文件`file_03.cpp`中，输入以下代码：

```cpp
#include <iostream>
#include <fstream>

int main ()
{
    std::fstream fstr;
    fstr.open("file_03.txt", std::ios::trunc | std::ios::out | std::ios::in);

    std::ifstream fiut;
    fiut.open("file_01.txt");
    std::string line;
    while (std::getline(fiut, line))
        fstr << line << std::endl;
    fiut.close();

    fstr.seekg(0, std::ios::beg);
    while (std::getline(fstr, line))
        std::cout << line << std::endl; 
    fstr.close();
}

```

让我们看看这个教程是如何工作的。

# 它是如何工作的...

在深入研究前面三个程序之前，我们必须澄清标准库在文件流方面的结构。让我们看一下下表：

|  |  | `<fstream>` |
| --- | --- | --- |
| `<ios>` | <--`<ostream>` | <--`ofstream` |
| `<ios>` | <-- `<istream>` | <--`ifstream` |

让我们分解如下：

+   `<ostream>`：负责输出流的流类。

+   `<istream>`：负责输入流的流类。

+   `ofstream`：用于向文件写入的流类。在`fstream`头文件中存在。

+   `ifstream`：用于从文件读取的流类。在`fstream`头文件中存在。

`std::ofstream`和`std::ifstream`都继承自`std::ostream`和`std::istream`的通用流类。正如你可以想象的那样，`std::cin`和`std::cout`也继承自`std::istream`和`std::ostream`（在上表中未显示）。

*步骤 1*：我们首先包含`<iostream>`和`<fstream>`，以便使用`std::cout`和`std::ofstream`来读取`file_01.txt`文件。然后我们调用`open`方法，在这种情况下，打开文件以写入模式，因为我们使用`std::ofstream`类。现在我们准备使用`<<`运算符将字符串写入`fout`文件流中。最后，我们必须关闭流，这将关闭文件。通过编译和运行程序，我们将得到以下输出：

![](img/54d34028-d689-4189-be42-0a164bbe3750.png)

*步骤 2*：在这种情况下，我们做相反的操作：从`file_01.txt`文件中读取并打印到标准输出。唯一的区别在于，这种情况下我们使用`std::ifstream`类，它表示一个读取文件流。通过调用`open()`方法，文件以读取模式（`std::ios::in`）打开。通过使用`std::getline`方法，我们可以将文件的所有行打印到标准输出。输出如下所示：

![](img/d24e9dd6-2fda-49c5-905b-b1e690ba9987.png)

最后的第三步展示了`std::fstream`类的用法，它通过允许我们以读写模式（`std::ios::out` | `std::ios::in`）打开文件，给了我们更多的自由。我们还希望如果文件存在，则截断文件（`std::ios::trunc`）。有许多其他选项可用于传递给`std::fstream`构造函数。

# 还有更多...

C++17 通过将`std::filesystem`添加到标准库中取得了巨大的改进。这并不是完全新的 - 它受到 Boost 库的巨大启发。公开的主要成员如下：

| **方法名称** | **描述** |
| --- | --- |
| `path` | 表示路径 |
| `filesystem_error` | 文件系统错误的异常 |
| `directory_iterator` | 一个用于遍历目录内容的迭代器（递归版本也可用） |
| `space_info` | 文件系统上空闲和可用空间的信息 |
| `perms` | 标识文件系统权限系统 |

在`std::filesystem`命名空间中，还有一些辅助函数，可以提供有关文件的信息，例如`is_directory()`、`is_fifo()`、`is_regular_file()`、`is_socket()`等等。

# 另请参阅

+   第二章中的*理解文件系统*配方，*重温 C++*，对该主题进行了复习。
