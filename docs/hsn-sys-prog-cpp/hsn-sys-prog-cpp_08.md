# 第八章：学习编程文件输入/输出

文件输入/输出（I/O）是大多数系统级程序的重要部分。它可以用于调试、保存程序状态、处理特定于用户的数据，甚至与物理设备进行交互（由于 POSIX 块和字符设备）。

在 C++17 之前，处理文件 I/O 是困难的，因为文件系统管理必须使用非 C++ API 来处理，这些 API 通常不安全、特定于平台，甚至不完整。

在本章中，我们将提供一个实际操作的回顾，介绍如何打开、读取和写入文件，以及处理路径、目录和文件系统。最后，我们将提供三个不同的示例，演示如何记录到文件、追踪现有文件和对 C++文件输入/输出 API 进行基准测试。

本章将涵盖以下主题：

+   打开文件的方式

+   读取和写入文件

+   文件工具

# 技术要求

为了编译和执行本章中的示例，读者必须具备以下条件：

+   一个能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请参见以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter08`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter08)。

# 打开文件

打开文件的多种方式。我们将在以下部分讨论其中一些，并介绍如何使用`std::fstream` C++ API 来实现。

# 打开文件的不同方式

在 C++中打开文件就像提供一个`std::fstream`对象和你想要打开的对象的文件名和路径一样简单。示例如下：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::cout << "success\n";
    }
    else {
        std::cout << "failure\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在这个例子中，我们打开一个名为`test.txt`的文件，之前使用 POSIX 的`touch`命令创建过。这个文件以读/写权限打开（因为这是默认模式）。

文件存储在名为`file`的变量中，并使用`std::fstream`提供的 bool 运算符重载来确保它已经正确打开。如果文件成功打开，我们将`success`输出到`stdout`。

前面的例子利用了`std::fstream`对象具有重载的`bool`运算符，当文件成功打开时返回 true。更明确地执行此操作的另一种方法是使用`is_open()`函数，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt"); file.is_open()) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在前面的例子中，我们不是依赖于`bool`运算符重载，而是利用 C++17 在`if`语句中使用`is_open()`来检查文件是否打开。前面的例子通过使用构造函数初始化`std::fstream`来进一步简化，而不是显式调用`open()`，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    auto file = std::fstream();
    if (file.open("test.txt"); file.is_open()) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在这个例子中，`std::fstream`对象是使用默认构造函数创建的，这意味着还没有打开文件，允许我们在准备好时再打开文件。然后我们使用`open()`函数打开文件，然后，类似于前面的例子，我们利用 C++17 来检查文件是否打开，然后将`success`输出到`stdout`。

在所有前面的例子中，不需要在文件上调用`close()`。这是因为，像其他 C++类（如利用 RAII 的`std::unique_ptr`）一样，`std::fstream`对象在销毁时会自动关闭文件。

然而，如果需要的话，可以显式关闭文件，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    std::cout << std::boolalpha;

    if (auto file = std::fstream("test.txt")) {
        std::cout << file.is_open() << '\n';
        file.close();
        std::cout << file.is_open() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// true
// false
```

在这个例子中，我们打开一个文本文件并使用`is_open()`来检查文件是否打开。第一次使用`is_open()`返回 true，因为文件成功打开。然后我们使用`close()`显式关闭文件，然后再次使用`is_open()`检查文件是否打开，现在返回 false。

# 打开文件的模式

到目前为止，我们一直使用默认模式打开文件。有两种模式可以用来打开文件：

+   `std::ios::in`：打开文件以供读取

+   `std::ios::out`：打开文件以供写入

此外，还有几种其他模式可以与这两种模式结合使用，以修改文件的打开方式：

+   `std::ios::binary`：以二进制方式打开文件。默认情况下，`std::fstream`处于文本模式，该模式适用于使用换行符格式化文件以及可以读取/写入文件的字符类型的特定规则。这些规则通常适用于文本文件，但在尝试向文件读取/写入二进制数据时会导致问题。在这种情况下，应将`std::ios::binary`添加到模式说明符中。

+   `std::ios::app`：当此模式与`std::ios::out`一起使用时，对文件的所有写入都会追加到文件的末尾。

+   `std::ios::ate`：当此模式与`std::ios::in`或`std::ios::out`一起使用时，文件在成功打开后定位在文件的末尾。也就是说，对文件的读取和写入发生在文件的末尾，即使在文件打开后立即进行。

+   `std::ios::trunc`：当此模式与`std::ios::in`或`std::ios::out`一起使用时，打开文件之前会删除文件的内容。

为了演示这些模式，第一个示例以二进制模式打开文件进行读取：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::in | std::ios::binary;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

所有模式都是常量值，因此在前面的示例中，使用`constexpr`创建了一个名为`mode`的新常量，表示以只读、二进制模式打开文件。要以文本模式而不是二进制模式打开文件进行只读，请简单地删除`std::ios::binary`模式，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::in;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在前面的示例中，我们以只读、文本模式打开文件。相同的逻辑也可以用于只写，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::out | std::ios::binary;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在这里，我们以只写、二进制模式打开文件。要以只写、文本模式打开文件，请使用以下方法：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::out;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

再次，由于省略了`std::ios::binary`，这段代码以只写、文本模式打开文件。

要以只写、二进制模式在文件末尾（而不是默认的文件开头）打开文件，请使用以下方法：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::out | std::ios::binary | std::ios::ate;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在此示例中，我们通过将`std::ios::ate`添加到模式变量中，在只写、二进制模式下打开文件，将文件移动到文件末尾。这将文件中的输出指针移动到文件的末尾，但允许在文件中的任何位置进行写入。

为了确保文件始终追加到文件的末尾，使用`std::ios::app`而不是`std::ios::ate`来打开文件，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::out | std::ios::binary | std::ios::app;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在前面的示例中，由于使用了`std::ios::app`，文件的写入和添加总是追加到文件中。

应该注意，在所有先前使用`std::ios::out`的示例中，文件都是使用`std::ios::trunc`打开的。这是因为截断模式是在使用`std::ios::out`时的默认值，除非使用了`std::ios::ate`或`std::ios::app`。这样做的问题在于，没有办法在文件开头以只写模式打开文件而不截断文件。

为了解决这个问题，可以使用以下方法：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::out | std::ios::binary | std::ios::ate;
    if (auto file = std::fstream("test.txt", mode); file.seekp(0)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在此示例中，我们以只写、二进制模式在文件末尾打开文件，然后我们使用`seekp()`（稍后将解释的函数）将文件中的输出位置移动到文件的开头。

尽管`std::ios::trunc`是在使用`std::ios::out`时的默认值，但如果还使用了`std::ios::in`（即读/写模式），则必须显式添加`std::ios::trunc`，如果您希望在打开文件之前清除文件的内容，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    constexpr auto mode = std::ios::in | std::ios::out | std::ios::trunc;
    if (auto file = std::fstream("test.txt", mode)) {
        std::cout << "success\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; touch test.txt; ./a.out
// success
```

在这里，文件以读/写模式打开，并且在打开文件之前删除了文件的内容。

# 读取和写入文件

以下部分将帮助您了解如何使用`std::fstream` C++ API 读取和写入文件。

# 从文件中读取

C++提供了几种不同的方法来读取文件，包括按字段、按行和按字节数。

# 按字段读取

从文件中读取的最安全的方法是按字段，代码如下：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::string hello, world;
        file >> hello >> world;
        std::cout << hello << " " << world << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
```

在这个例子中，我们打开一个文件进行读写（因为这是默认模式）。如果文件成功打开，我们将两个字符串分别读入两个变量——`hello`和`world`。要读取这两个字符串，我们使用`>> operator()`，它的行为就像第六章中讨论的`std::cin`一样。

对于字符串，流会读取字符，直到发现第一个空格或换行符。与`std::cin`一样，也可以读取数值变量，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        int answer;
        file >> answer;
        std::cout << "The answer is: " << answer << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "42" > test.txt; ./a.out
// The answer is: 42
```

在这个例子中，我们读取的是一个整数而不是一个字符串，就像读取字符串一样，流会读取字节，直到发现空格或换行符，然后将输入解释为一个数字。当然，如果被读取的字段不是一个数字，就会读取`0`，如下所示：

```cpp
// > g++ -std=c++17 scratchpad.cpp; echo "not_a_number" > test.txt; ./a.out
// The answer is: 0
```

值得注意的是，当发生这种情况时会设置一个错误标志，我们将在本章后面讨论。

与其他 C++流一样，`std::fstream`可以被重载以支持用户定义的类型，如下所示：

```cpp
#include <fstream>
#include <iostream>

struct myclass
{
    std::string hello;
    std::string world;
};

std::fstream &operator >>(std::fstream &is, myclass &obj)
{
    is >> obj.hello;
    is >> obj.world;

    return is;
}

std::ostream &operator<<(std::ostream &os, const myclass &obj)
{
    os << obj.hello;
    os << ' ';
    os << obj.world;

    return os;
}

int main()
{
    if (auto file = std::fstream("test.txt")) {
        myclass obj;
        file >> obj;
        std::cout << obj << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
```

在这个例子中，我们创建了一个名为`myclass`的用户定义类型。在`main()`函数中，我们打开一个文件，如果文件成功打开，我们创建一个`myclass{}`对象，将文件读入`myclass{}`对象，然后将`myclass{}`对象的结果输出到`stdout`。

为了将文件读入`myclass{}`对象中，我们重载了`std::fstream{}`的`>> operator()`，它读取两个字符串，并将结果存储在`myclass{}`对象中。要将`myclass{}`对象输出到`stdout`，我们将在第六章中学到的内容进行扩展，即关于用户定义重载`std::ostream`的内容，并为我们的`myclass{}`对象提供用户定义的重载。

结果是从文件中读取`Hello World`并输出到`stdout`。

# 读取字节

除了从文件中读取字段外，C++还提供了直接从文件中读取字节的支持。要从流中读取一个字节，使用`get()`函数，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        char c = file.get();
        std::cout << c << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// H
```

在 C++17 中读取多个字节仍然是一种不安全的操作，因为没有能力直接将*x*个字节读入`std::string`。这意味着必须使用标准的 C 风格缓冲区，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        char buf[25] = {};
        file.read(buf, 11);
        std::cout << buf << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
```

在前面的例子中，我们创建了一个名为`buf`的标准 C 风格字符缓冲区，然后从文件中读取了 11 个字节到这个字符缓冲区中。最后，我们将结果输出到`stdout`。

我们需要确保被读取的字节数不超过缓冲区本身的总大小——这种操作通常会导致编码错误，产生难以调试的缓冲区溢出。

解决这个问题的简单方法是使用一个包装器来包围`read()`函数，以确保请求的字节数不超过缓冲区的总大小，如下所示：

```cpp
#include <fstream>
#include <iostream>

template<typename T, std::size_t N>
void myread(std::fstream &file, T (&str)[N], std::size_t count)
{
    if (count >= N) {
        throw std::out_of_range("file.read out of bounds");
    }

    file.read(static_cast<char *>(str), count);
}

int main()
{
    if (auto file = std::fstream("test.txt")) {
        char buf[25] = {};
        myread(file, buf, 11);
        std::cout << buf << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
```

在这个例子中，我们创建了一个名为`myread()`的模板函数，在编译期间将缓冲区的总大小编码到函数本身中。在读取发生之前，可以检查缓冲区的大小，以确保不会发生缓冲区溢出。

值得注意的是，这对于数组来说效果很好，但对于动态分配的数组来说存在问题，因为缓冲区的总大小也必须传递给我们的包装器函数，可能会导致难以调试的逻辑错误（即未提供正确的缓冲区大小，交换要读取的字节数和缓冲区大小等）。

为了克服这些问题，应该使用`gsl::span`。

当读取字节而不是字段时，了解当前正在读取文件的位置可能会有所帮助。当您从文件流中读取时，流内部会维护一个读指针和一个写指针。要获取当前的读位置，使用`tellg()`函数，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::cout << file.tellg() << '\n';
        char c = file.get();
        std::cout << file.tellg() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// 0
// 1
```

在这里，我们像往常一样打开一个文件并输出当前的读指针，预期的是`0`。然后我们从文件中读取一个字符，并再次输出读指针。这次，指针是`1`，表示我们已成功读取了一个字节。

另一种读取单个字节的方法是使用`peek`函数，它的功能类似于`get()`，只是内部读指针不会增加，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::cout << file.tellg() << '\n';
        char c = file.peek();
        std::cout << file.tellg() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// 0
// 0
```

这个例子与前一个例子相同，只是使用了`peek()`而不是`get()`。如所示，在使用`peek()`从缓冲区中读取一个字节之前和之后，读指针都是`0`，表明`peek()`不会增加流中的读指针。

C++也提供了相反的操作。除了从文件中读取一个字节而不移动读指针之外，还可以使用`ignore()`函数移动读指针而不从流中读取字节，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::cout << file.tellg() << '\n';
        file.ignore(1);
        std::cout << file.tellg() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// 0
// 1
```

在这个例子中，我们通过一个字节移动文件流中的读指针，并使用`tellg()`来验证读指针是否实际上已经移动。`ignore()`函数相对于当前读指针增加读指针。

C++还提供了`seekg()`函数，它将读指针设置为绝对位置，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::string hello, world;

        file >> hello >> world;
        std::cout << hello << " " << world << '\n';

        file.seekg(1);

        file >> hello >> world;
        std::cout << hello << " " << world << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
// ello World
```

在前面的例子中，`seekg()`函数用于在读取后将读指针设置为文件中的第 1 个字节，有效地倒带，使我们可以再次读取文件。

# 按行读取

最后，文件读取的最后一种类型是按行读取，这意味着您每次从文件中读取一行，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        char buf[25] = {};
        file.getline(buf, 25, '\n');
        std::cout << buf << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
```

在这个例子中，我们创建了一个标准的 C 字符缓冲区，从文件中读取一行，并将该行输出到`stdout`。与`read()`函数不同，`getline()`会一直读取，直到达到缓冲区的大小（第二个参数），或者看到一个分隔符。

由于行的定义取决于您使用的操作系统（尽管在这种情况下，我们将坚持使用 Unix），`getline()`函数接受一个分隔符参数，允许您定义行的结束位置。

与`read()`函数一样，这个操作是不安全的，因为它要求用户确保传递给`getline()`的总缓冲区大小实际上是缓冲区的总大小，从而提供了一个方便的机制来引入难以调试的缓冲区溢出。

与`read()`函数不同，C++提供了`getline()`的非成员版本，它接受任何流类型（包括`std::cin`）和`std::string`，而不是标准的 C 风格字符串，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::string str;
        std::getline(file, str);
        std::cout << str << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "Hello World" > test.txt; ./a.out
// Hello World
```

在前面的例子中，我们没有调用`file.getline()`，而是调用了`std::getline()`，并提供了`std::string`，它可以根据需要读取的字节数动态更改其大小，从而防止可能的缓冲区溢出。

应该注意的是，为了实现这一点，`std::string`将自动为您执行`new()`/`delete()`操作，这可能会引入不可接受的低效率（特别是在系统编程方面）。在这种情况下，应该使用`file.getline()`版本，使用一个包装类，类似于我们在`read()`函数中所做的。

最后，如果对已经打开的文件进行了更改，以下操作将使当前流与这些更改同步：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file.sync();
    }
}
```

如前面的代码所示，`sync()`函数可以用于将已经打开的文件与文件的更改重新同步。

# 写入文件

与`std::cin`和文件读取一样，C++还提供了文件写入，其行为类似于`std::cout`。与读取不同，文件写入只有两种不同的模式——按字段和按字节。

# 按字段写入

要按字段写入文件，使用`<< operator()`，类似于`std::cout`，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::string hello{"Hello"}, world{"World"};
        file << hello << " " << world << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// Hello World
```

在前面的例子中，我们像往常一样打开一个文件，然后创建了两个`std::string`对象，分别向这些字符串中添加了`hello`和`world`。最后，这些字符串被写入文件。请注意，不需要关闭或刷新文件，因为这在文件流对象销毁时会为我们完成。

与`std::cout`一样，C++本身支持标准 C 字符缓冲区和数字类型，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file << "The answer is: " << 42 << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// The answer is: 42
```

在前面的例子中，我们直接向文件写入了一个标准 C 字符缓冲区和一个整数。对于写入，也支持用户定义的类型，如下所示：

```cpp
#include <fstream>
#include <iostream>

struct myclass
{
    std::string hello{"Hello"};
    std::string world{"World"};
};

std::fstream &operator <<(std::fstream &os, const myclass &obj)
{
    os << obj.hello;
    os << ' ';
    os << obj.world;

    return os;
}

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file << myclass{} << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// Hello World
```

在这个例子中，我们打开一个文件，并向文件写入一个`myclass{}`对象。`myclass{}`对象是一个包含两个成员变量的结构体，这两个成员变量被初始化为`Hello`和`World`。然后提供了一个用户定义的`<< operator()`重载，用于向提供的文件流写入`myclass{}`对象的内容，结果是将`Hello World`写入文件。

# 写入字节

除了按字段写入，还支持写入一系列字节。在下面的例子中，我们使用`put()`函数向文件写入一个字节（以及一个换行符），该函数类似于`get()`，但用于写入而不是读取：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file.put('H');
        file.put('\n');
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// H
```

多个字节也可以使用`write()`函数进行写入，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file.write("Hello World\n", 12);
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// Hello World
```

在前面的例子中，我们向文件写入了`12`字节（字符串`Hello World`的 11 个字符和一个额外的换行符）。

与`read()`函数一样，`write()`函数是不安全的，应该进行包装，以确保写入文件的总字节数不超过缓冲区的总大小（否则会发生缓冲区溢出）。为了演示即使标准 C 风格的`const`字符缓冲区也是不安全的，可以参考以下内容：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file.write("Hello World\n", 100);
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// Hello World
// ;�����D���d)��������$=���DR����d���d�����[
```

正如本例所示，尝试从大小仅为`13`字节的标准 C `const`字符缓冲区中写入 100 个字节（`Hello World`的 11 个字节，`1`个换行符，`1`个`\0`空终止符），会导致缓冲区溢出。在这种情况下，缓冲区溢出会导致损坏的字节被写入文件，最好的情况下会泄漏程序的部分内容，但也可能导致不稳定性，包括难以调试的分段错误。

为了克服这个问题，无论何时使用这些不安全的函数，都应该使用一个包装器，如下所示：

```cpp
#include <string.h>

#include <fstream>
#include <iostream>

void
mywrite(std::fstream &file, const char *str, std::size_t count)
{
    if (count > strlen(str)) {
        throw std::out_of_range("file.write out of bounds");
    }

    file.write(str, count);
}

int main()
{
    if (auto file = std::fstream("test.txt")) {
        mywrite(file, "Hello World\n", 100);
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// terminate called after throwing an instance of 'std::out_of_range'
// what(): file.write out of bounds
// Aborted (core dumped)
```

在前面的例子中，我们创建了一个`write()`函数的包装器，类似于之前创建的`read()`函数包装器。当我们尝试写入的字节数超过了标准 C `const`字符缓冲区的总大小时，我们会生成一个异常，该异常可用于跟踪错误，以确定我们尝试写入 100 个字节。

应该注意的是，这个包装器只适用于编译器生成的标准 C `const`字符缓冲区。可以手动声明这种类型的缓冲区，这种类型的函数将失败，如下所示：

```cpp
#include <string.h>

#include <fstream>
#include <iostream>

void
mywrite(std::fstream &file, const char *str, std::size_t count)
{
    if (count > strlen(str)) {
    std::cerr << count << " " << strlen(str) << '\n';
        throw std::out_of_range("file.write out of bounds");
    }

    file.write(str, count);
}

int main()
{
    if (auto file = std::fstream("test.txt")) {
        const char str1[6] = {'H','e','l','l','o','\n'};
        const char str2[6] = {'#','#','#','#','#','\n'};
        mywrite(file, str1, 12);
        mywrite(file, str2, 6);
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// Hello
// World
// World
```

在这个例子中，我们创建了两个标准的 C `const` 字符缓冲区。第一个缓冲区由单词`Hello`和一个换行符组成，第二个缓冲区由单词`World`和一个换行符组成。然后我们向文件写入`Hello`，但是我们写入的不是`6`个字符，而是`12`个字符。最后，我们向文件写入`World`，并提供了正确的字节数，即`6`。

结果输出为`Hello World`，`World`被写入文件两次。这是因为精心设计的缓冲区溢出。向文件的第一次写入将`Hello`写入缓冲区，但是提供给`write()`函数的是`12`个字节，而不是`6`个字节。在这种情况下，我们的包装器正在寻找空终止符，但这个终止符不存在（因为我们手动定义了标准 C `const`字符缓冲区，删除了空终止符）。

因此，`mywrite()`函数无法检测到溢出，并写入了两个缓冲区。

没有安全的方法可以克服这种问题（`read()`函数存在类似的问题），除非使用指导支持库、勤勉和能够检测到这些类型的缓冲区不安全使用的静态分析器（这对于静态分析器来说并不是一件微不足道的事情）。因此，通常情况下，应尽可能避免使用`read()`和`write()`等函数，而应使用按字段和按行的替代方法。

与`tellg()`类似，写流也可以使用`tellp()`函数获取当前写指针位置，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::cout << file.tellp() << '\n';
        file << "Hello";
        std::cout << file.tellp() << '\n';
        file << ' ';
        std::cout << file.tellp() << '\n';
        file << "World";
        std::cout << file.tellp() << '\n';
        file << '\n';
        std::cout << file.tellp() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// 0
// 5
// 6
// 11
// 12
// Hello World
```

在上述示例中，`Hello World`被写入文件，并且使用`tellp()`函数输出写指针位置，结果为`0`、`5`、`6`、`11`和`12`。

还可以使用`seekp()`函数将写指针移动到文件中的绝对位置，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        std::cout << file.tellp() << '\n';
        file << "Hello World\n";
        std::cout << file.tellp() << '\n';
        file.seekp(0);
        std::cout << file.tellp() << '\n';
        file << "The answer is: " << 42 << '\n';
        std::cout << file.tellp() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// 0
// 12
// 0
// 18
// The answer is: 42
```

在此示例中，我们将`Hello World`写入文件，然后将流中的写指针移回文件的开头。然后我们将“答案是：42”写入文件。在此过程中，我们使用`tellp()`输出写指针的位置，显示了在执行这些操作时写指针的移动情况。

因此，文件包含“答案是：42”，而不是`Hello World`，因为`Hello World`被覆盖。

最后，与`sync()`函数一样，可以使用以下方法将文件的写入刷新到文件系统：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    if (auto file = std::fstream("test.txt")) {
        file.flush();
    }
}
```

应该注意的是，尽管可以手动刷新文件（例如，如果知道更改必须传输到文件系统），但是当`std::fstream`对象失去作用域并被销毁时，文件将自动关闭并刷新到文件系统。

在读取和写入时，可能会发生不同类型的错误。`std::fstream`提供了四个不同的函数来确定流的状态，如下所示：

+   `good()`: 如果此函数返回`true`，则没有发生错误，流也没有到达文件的末尾。

+   `eof()`: 如果此函数返回`true`，则已到达文件的末尾。内部错误不会影响此函数的结果。

+   `fail()`: 如果此函数返回`true`，则发生了内部错误，但流仍然可用，例如，如果发生数字转换错误。

+   `bad()`: 如果此函数返回`true`，则发生了错误，流不再可用，例如，如果流无法打开文件。

当正常的文件操作发生时，`good()`应该返回`true`，而其他三个状态函数应该返回`false`，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    std::cout << std::boolalpha;

    if (auto file = std::fstream("test.txt")) {
        std::string hello{"Hello"}, world{"World"};
        file << hello << " " << world << '\n';
        std::cout << "good: " << file.good() << '\n';
        std::cout << "fail: " << file.fail() << '\n';
        std::cout << "bad: " << file.bad() << '\n';
        std::cout << "eof: " << file.eof() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// good: true
// fail: false
// bad: false
// eof: false
// Hello World
```

在上述示例中，`Hello World`成功写入文件，导致`good()`返回`true`。

除了使用`good()`函数外，可以使用`! operator()`来检测是否发生了错误，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    std::cout << std::boolalpha;

    if (auto file = std::fstream("test.txt")) {
        std::string hello{"Hello"}, world{"World"};
        file << hello << " " << world << '\n';
        if (!file) {
            std::cout << "failed\n";
        }
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// Hello World
```

在这里，`Hello World`成功写入文件，因此`good()`函数返回`true`，这意味着`! operator()`返回`false`，导致`failed`字符串从未输出到`stdout`。

类似地，可以使用`bool`运算符，其返回与`good()`相同的结果，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    std::cout << std::boolalpha;

    if (auto file = std::fstream("test.txt")) {
        std::string hello{"Hello"}, world{"World"};
        file << hello << " " << world << '\n';
        if (file) {
            std::cout << "success\n";
        }
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "" > test.txt; ./a.out; cat test.txt
// success
// Hello World
```

在上述代码中，`Hello World`成功写入文件，导致`bool`运算符返回`true`；这意味着`good()`函数也会返回`true`，因为它们返回相同的结果。

如果发生错误，错误状态将保持触发状态，直到流关闭，或者使用`clear()`函数告诉流已处理错误，如下所示：

```cpp
#include <fstream>
#include <iostream>

int main()
{
    std::cout << std::boolalpha;

    if (auto file = std::fstream("test.txt")) {
        int answer;
        std::cout << file.good() << '\n';
        file >> answer;
        std::cout << file.good() << '\n';
        file.clear();
        std::cout << file.good() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; echo "not_a_number" > test.txt; ./a.out
// true
// false
// true
```

在上述示例中，将一个字符串写入文本文件。此测试文件被打开以进行读取，并读取一个整数。问题在于写入文件的值实际上不是一个数字，导致文件流报告错误。

然后使用`clear`函数清除错误，之后`good()`函数继续报告`true`。

# 理解文件实用程序

到目前为止，在 C++17 之前添加的所有 C++ API 都提供了描述的功能。尽管 C++提供了读写文件的能力，但它并没有提供管理文件系统所需的所有其他文件操作，包括文件路径、目录管理等。

本节将重点介绍 C++17 中的`std::filesystem`增强功能，以解决这些缺陷中的大部分。

# 路径

路径只不过是表示文件系统中节点的字符串。在 UNIX 系统上，这通常是一个由一系列目录名、`/`和文件名组成的字符串，通常带有扩展名。路径的目的是表示文件的名称和位置，然后可以用来对文件执行操作，如打开文件进行读写、更改文件的权限，甚至从文件系统中删除文件。

应该注意，路径可以表示文件系统中许多不同类型的节点，包括文件、目录、链接、设备等。更完整的列表将在本章后面呈现。考虑以下例子：

```cpp
/home/user/
```

这是一个指向名为`user`的目录的路径，位于名为`home`的根目录中。现在考虑以下内容：

```cpp
/home/user/test.txt
```

这指的是在同一目录中名为`test.txt`的文件。文件的主干是`test`，而文件的扩展名是`.txt`。此外，文件的根目录是`/`（这在大多数 UNIX 系统上都是这样）。

在 UNIX 系统上，路径可以采用不同的形式，包括以下内容：

+   **块设备**：路径指向 POSIX 风格的块设备，如`/dev/sda`

+   **字符设备**：路径指向 POSIX 风格的字符设备，如`/dev/random`

+   **目录**：路径指向常规目录

+   **Fifo**：路径指向管道或其他形式的 IPC

+   **套接字**：路径指向 POSIX 套接字

+   **符号链接**：路径指向 POSIX 符号链接

+   **文件**：路径指向常规文件

要确定路径的类型，C++17 提供了以下测试函数：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    using namespace std::filesystem;

    std::cout << std::boolalpha;
    std::cout << is_block_file("/dev/sda1") << '\n';
    std::cout << is_character_file("/dev/random") << '\n';
    std::cout << is_directory("/dev") << '\n';
    std::cout << is_empty("/dev") << '\n';
    std::cout << is_fifo("scratchpad.cpp") << '\n';
    std::cout << is_other("scratchpad.cpp") << '\n';
    std::cout << is_regular_file("scratchpad.cpp") << '\n';
    std::cout << is_socket("scratchpad.cpp") << '\n';
    std::cout << is_symlink("scratchpad.cpp") << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// true
// true
// false
// false
// false
// true
// false
// false
```

如前面的例子所示，`/dev/sda`是一个块设备，`/dev/random`是一个字符设备，`/dev`是一个非空目录，`scratchpad.cpp`文件用于编译本章中的所有示例，是一个常规文件。

要确定路径是否存在，C++17 提供了`exists（）`函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
 std::cout << std::boolalpha;
 std::cout << std::filesystem::exists("/dev") << '\n';
 std::cout << std::filesystem::exists("/dev/random") << '\n';
 std::cout << std::filesystem::exists("scratchpad.cpp") << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// true
// true
```

这里的目录`/dev`存在，字符设备`/dev/random`和常规文件`scratchpad.cpp`也存在。

每个执行的程序都必须从给定的目录执行。要确定这个目录，C++17 提供了`current_path（）`函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    std::cout << std::filesystem::current_path() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08"
```

在这个例子中，`current_path（）`用于获取`a.out`正在执行的当前目录。`current_path（）`提供的路径是绝对路径。要将绝对路径转换为相对路径，可以使用`relative（）`函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    std::cout << std::filesystem::relative(path) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "."
```

如本例所示，当前路径的相对路径只是（`.`）。

同样，要将相对路径转换为绝对路径，C++17 提供了`canonical（）`函数：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    std::cout << std::filesystem::canonical(".") << '\n';
    std::cout << std::filesystem::canonical("../Chapter08") << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08"
```

在这个例子中，我们使用`canonical（）`函数将相对路径转换为绝对路径。值得注意的是，获取`.`的绝对路径是返回`current_path（）`相同结果的另一种方法。

还要注意，`canonical（）`函数返回带有所有对`../`和`./`的引用解析的绝对路径，将绝对路径减少到其最小形式。如果不需要这种类型的路径，可以使用`absolute（）`函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    std::cout << std::filesystem::absolute("../Chapter08") << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/../Chapter08"
```

如本例所示，`../`不会被“absolute（）”函数移除。

由于有不同的表示相同路径的方式（即相对、规范和绝对），C++17 提供了`equivalent（）`函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
 auto path1 = std::filesystem::path{"."};
 auto path2 = std::filesystem::path{"../Chapter08"};
 auto path3 = std::filesystem::path{"../Chapter08/../Chapter08"};
 auto path4 = std::filesystem::current_path();
 auto path5 = std::filesystem::current_path() / "../Chapter08/";

 std::cout << std::boolalpha;
 std::cout << std::filesystem::equivalent(path1, path2) << '\n';
 std::cout << std::filesystem::equivalent(path1, path3) << '\n';
 std::cout << std::filesystem::equivalent(path1, path4) << '\n';
 std::cout << std::filesystem::equivalent(path1, path5) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// true
// true
// true
```

在本例中引用的所有路径都指向相同的目录，无论它们是相对的、规范的还是绝对的。

如果要确定两个路径在词法上是否相等（包含完全相同的字符），请使用`== operator()`，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path1 = std::filesystem::path{"."};
    auto path2 = std::filesystem::path{"../Chapter08"};
    auto path3 = std::filesystem::path{"../Chapter08/../Chapter08"};
    auto path4 = std::filesystem::current_path();
    auto path5 = std::filesystem::current_path() / "../Chapter08/";

    std::cout << std::boolalpha;
    std::cout << (path1 == path2) << '\n';
    std::cout << (path1 == path3) << '\n';
    std::cout << (path1 == path4) << '\n';
    std::cout << (path1 == path5) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// false
// false
// false
// false
```

这里的代码与前面的代码相同，只是使用了`== operator()`而不是`equivalent()`函数。前一个示例对所有路径返回`true`，因为它们都指向相同的路径，而前面的示例返回`false`，因为相同的路径在词法上不相等，即使它们在技术上是相同的路径。

请注意这些示例中的`/ operator()`的使用。C++17 为路径提供了不同的连接函数，方便地提供了一种清晰易读的方式来添加到现有路径中：`/`、`/=`和`+=`。`/ operator()`（以及自修改版本`/= operator()`）将两个路径连接在一起，并为您添加`/`，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << path << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
```

在这个例子中，使用`/= operator()`将`scratchpad.cpp`添加到路径中，并为我们添加了`/`。如果您希望自己添加`/`，或者根本不希望添加`/`，可以使用`+= operator()`，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path += "/scratchpad.cpp";

    std::cout << path << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
```

这里的结果与前一个示例中的结果相同，不同之处在于使用`+= operator()`而不是`/= operator()`，因此需要手动添加`/`。

除了连接，C++17 还提供了一些额外的路径修改器。其中一个函数是`remove_filename()`，它从路径中删除文件名，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << path << '\n';
    path.remove_filename();
    std::cout << path << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/"
```

如图所示，`remove_filename()`函数从路径中删除了文件名。

也可以用其他东西替换文件名，而不是删除它，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << path << '\n';
    path.replace_filename("test.cpp");
    std::cout << path << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/test.cpp"
```

如图所示，文件名`scratchpad.cpp`被替换为`test.cpp`。

除了替换文件名，还可以替换扩展名，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << path << '\n';
    path.replace_extension("txt");
    std::cout << path << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.txt"
```

如图所示，`scratchpad.cpp`的扩展名已更改为`.txt`。

最后，如果需要，可以使用`clear()`函数清除路径，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << path << '\n';
    path.clear();
    std::cout << path << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
// ""
```

如前面的代码所示，`clear()`函数删除了路径的内容（就像它是默认构造的一样）。

如前所述，路径由不同部分组成，包括根名称、目录、词干和扩展名。为了将路径分解为这些不同的组件，C++17 提供了一些辅助函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << std::boolalpha;
    std::cout << path.root_name() << '\n';
    std::cout << path.root_directory() << '\n';
    std::cout << path.root_path() << '\n';
    std::cout << path.relative_path() << '\n';
    std::cout << path.parent_path() << '\n';
    std::cout << path.filename() << '\n';
    std::cout << path.stem() << '\n';
    std::cout << path.extension() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// ""
// "/"
// "/"
// "home/user/Hands-On-System-Programming-with-CPP/Chapter08/scratchpad.cpp"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08"
// "scratchpad.cpp"
// "scratchpad"
// ".cpp"
```

在这个例子中，我们将`scratchpad.cpp`文件的路径分解为不同的部分。父路径是`/home/user/Hands-On-System-Programming-with-CPP/Chapter08`，文件名是`scratchpad.cpp`，词干是`scratchpad`，扩展名是`.cpp`。

并非所有路径都包含路径可能包含的所有部分。当路径指向目录或格式不正确时，可能会发生这种情况。

要找出路径包含的部分，使用以下辅助函数：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "scratchpad.cpp";

    std::cout << std::boolalpha;
    std::cout << path.empty() << '\n';
    std::cout << path.has_root_path() << '\n';
    std::cout << path.has_root_name() << '\n';
    std::cout << path.has_root_directory() << '\n';
    std::cout << path.has_relative_path() << '\n';
    std::cout << path.has_parent_path() << '\n';
    std::cout << path.has_filename() << '\n';
    std::cout << path.has_stem() << '\n';
    std::cout << path.has_extension() << '\n';
    std::cout << path.is_absolute() << '\n';
    std::cout << path.is_relative() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// false
// true
// false
// true
// true
// true
// true
// true
// true
// true
// false
```

如图所示，您可以确定路径是否具有根路径、根名称、根目录、相对路径、父路径、文件名、词干和扩展名。您还可以确定路径是绝对路径还是相对路径。

最后，C++17 提供了不同的机制来管理文件系统上的路径，具体取决于您使用的路径类型。例如，如果要创建目录或删除路径（无论其类型如何），可以分别使用`create_directory()`和`remove()`函数，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "test";

    std::cout << std::boolalpha;
    std::cout << std::filesystem::create_directory(path) << '\n';
    std::cout << std::filesystem::remove(path) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// true
```

在前面的示例中，我们使用`create_directory()`函数创建一个目录，然后使用`remove()`函数删除它。

我们还可以使用`rename()`函数重命名路径，如下所示：

```cpp
#include <iostream>
#include <filesystem>

int main()
{
    auto path1 = std::filesystem::current_path();
    auto path2 = std::filesystem::current_path();
    path1 /= "test1";
    path2 /= "test2";

    std::cout << std::boolalpha;
    std::cout << std::filesystem::create_directory(path1) << '\n';
    std::filesystem::rename(path1, path2);
    std::cout << std::filesystem::remove(path1) << '\n';
    std::cout << std::filesystem::remove(path2) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// false
// true
```

在这个例子中，我们使用`create_directory()`函数创建一个目录。然后我们使用`rename()`函数重命名目录，然后删除旧目录路径和新目录路径。如图所示，尝试删除已重命名的目录失败，因为该路径不再存在，而尝试删除新目录成功，因为该路径确实存在。

`remove()`函数将删除任何路径（假设程序具有适当的权限），除非路径指向一个非空的目录，在这种情况下它将失败。要删除一个非空的目录，请使用`remove_all()`函数，如下所示：

```cpp
#include <fstream>
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "test";

    std::cout << std::boolalpha;
    std::cout << std::filesystem::create_directory(path) << '\n';

    std::fstream(path / "test1.txt", std::ios::app);
    std::fstream(path / "test2.txt", std::ios::app);
    std::fstream(path / "test3.txt", std::ios::app);

    std::cout << std::filesystem::remove_all(path) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// 4
```

如此所示，我们创建一个目录并使用`std::fstream`向目录添加一些文件。然后，我们使用`remove_all()`而不是`remove()`来删除新创建的目录。如果我们使用`remove()`函数，程序将抛出异常，如下所示：

```cpp
terminate called after throwing an instance of 'std::filesystem::__cxx11::filesystem_error'
 what(): filesystem error: cannot remove: Directory not empty [/home/user/Hands-On-System-Programming-with-CPP/Chapter08/test]
Aborted (core dumped)
```

在文件系统上执行的另一个常见操作是遍历目录中的所有文件。为此，C++17 提供了一个目录迭代器，如下所示：

```cpp
#include <fstream>
#include <iostream>
#include <filesystem>

int main()
{
    auto path = std::filesystem::current_path();
    path /= "test";

    std::cout << std::boolalpha;
    std::cout << std::filesystem::create_directory(path) << '\n';

    std::fstream(path / "test1.txt", std::ios::app);
    std::fstream(path / "test2.txt", std::ios::app);
    std::fstream(path / "test3.txt", std::ios::app);

    for(const auto &p: std::filesystem::directory_iterator(path)) {
        std::cout << p << '\n';
    }

    std::cout << std::filesystem::remove_all(path) << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// true
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/test/test1.txt"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/test/test3.txt"
// "/home/user/Hands-On-System-Programming-with-CPP/Chapter08/test/test2.txt"
// 4
```

在前面的示例中，我们使用`create_directory()`函数创建一个目录，向目录添加一些文件，然后使用目录迭代器来遍历所有文件。

目录迭代器的功能与 C++中的任何其他迭代器一样，这意味着，如前面的示例所示，我们可以利用范围 for 语法。

最后，C++17 提供了一个方便的函数来确定临时目录的路径，可以用于根据需要为程序创建临时目录，如下所示：

```cpp
#include <fstream>
#include <iostream>
#include <filesystem>

int main()
{
    std::cout << std::filesystem::temp_directory_path() << '\n';
}

// > g++ -std=c++17 scratchpad.cpp -lstdc++fs; ./a.out
// "/tmp"

#endif
```

应该注意，在 POSIX 系统上，临时目录通常是`/tmp`，如此所示。然而，最好还是使用`temp_directory_path()`而不是硬编码这个路径。

# 理解记录器示例

在本节中，我们将扩展第六章中的调试示例，*学习编程控制台输入/输出*，以包括一个基本的记录器。这个记录器的目标是将对`std::clog`流的添加重定向到控制台之外的日志文件中。

就像[第六章](https://cdp.packtpub.com/hands_on_system_programming_with_c___/wp-admin/post.php?post=31&action=edit#post_29)中的调试函数一样，*学习编程控制台输入/输出*，如果调试级别不够，或者调试已被禁用，我们希望日志函数被编译出。

为了实现这一点，请参阅以下代码：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/example1.cpp`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/example1.cpp)。

首先，我们需要创建两个常量表达式——一个用于调试级别，一个用于启用或禁用调试，如下所示：

```cpp
#ifdef DEBUG_LEVEL
constexpr auto g_debug_level = DEBUG_LEVEL;
#else
constexpr auto g_debug_level = 0;
#endif

#ifdef NDEBUG
constexpr auto g_ndebug = true;
#else
constexpr auto g_ndebug = false;
#endif
```

接下来，我们需要创建一个全局变量，如下所示：

```cpp
std::fstream g_log{"log.txt", std::ios::out | std::ios::app};
```

全局变量是日志文件流。这将用于将对`std::clog`流的添加写入日志文件。由于这是一个日志文件，我们将其以只写、追加的方式打开，这意味着我们只能向日志文件写入，并且所有写入都必须追加到文件的末尾。

接下来，我们需要定义`log`函数本身。这个函数需要能够输出到`std::clog`和我们的日志文件流，而不会执行调试逻辑超过一次（因为这可能导致意外行为）。

以下实现了具有这一目标的`log`函数：

```cpp
template <std::size_t LEVEL>
constexpr void log(void(*func)()) {
    if constexpr (!g_ndebug && (LEVEL <= g_debug_level)) {
        std::stringstream buf;

        auto g_buf = std::clog.rdbuf();
        std::clog.rdbuf(buf.rdbuf());

        func();

        std::clog.rdbuf(g_buf);

        std::clog << "\0331;32mDEBUG\033[0m: ";
        std::clog << buf.str();

        g_log << "\033[1;32mDEBUG\033[0m: ";
        g_log << buf.str();
    };
}
```

与[第六章中的调试函数一样，*学习编程控制台输入/输出*，这个`log`函数首先通过`constexpr` `if`语句（C++17 的新特性）包装函数的业务逻辑，为编译器提供了一种在调试被禁用或者提供的调试级别大于当前调试级别时编译出代码的方法。

如果需要进行调试，第一步是创建一个字符串流，它的行为就像`std::clog`和日志文件流一样，但是将流的任何添加结果保存到`std::string`中。

然后保存`std::clog`的读取缓冲区，并将字符串流的读取缓冲区提供给`std::clog`。对`std::clog`流的任何添加都将重定向到我们的字符串流，而不是`stderr`。

接下来，我们执行用户提供的`debug`函数，收集调试字符串并将其存储在字符串流中。最后，将`std::clog`的`read()`缓冲区恢复为`stderr`，并将字符串流输出到`std::clog`和日志文件流。

最后一步是创建我们的`protected_main()`函数，记录`Hello World`。请注意，为了演示，我们还手动将`Hello World`添加到`std::clog`中，而不使用`log`函数，以演示`std::clog`在使用`log`函数时仍然正常工作，并且只在我们的日志文件中记录。下面的代码显示了这一点：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    log<0>([]{
        std::clog << "Hello World\n";
    });

    std::clog << "Hello World\n";

    return EXIT_SUCCESS;
}
```

要编译此代码，我们将利用我们一直在使用的相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/CMakeLists.txt)。

有了这段代码，我们可以使用以下方法编译和执行这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter08/
> mkdir build
> cd build

> cmake ..
> make
> ./example1
DEBUG: Hello World
Hello World

> cat log.txt
DEBUG: Hello World
```

请注意，`debug`语句都输出到`stderr`（`log`函数中的语句和手动执行的没有`log`函数的语句）。然而，日志文件中只有一个语句，演示了`log`函数负责将对`std::clog`的添加重定向到日志文件和`stderr`，同时保持`std::clog`完好无损以供将来使用。

# 学习关于 tail 文件的例子

在这个例子中，我们将创建一个简单的程序来 tail 一个文件。这个例子的目标是模仿`tail -f -n0`的行为，它输出文件的新添加。`-f`参数告诉 tail 跟踪文件，`-n0`告诉 tail 只将新添加输出到`stdout`。

第一步是定义我们打算在打开要 tail 的文件时使用的模式，如下所示：

```cpp
constexpr auto mode = std::ios::in | std::ios::ate;
```

在这种情况下，我们将以只读方式打开文件，并在打开时将读取指针移动到文件的末尾。

下一步是创建一个`tail`函数，用于监视文件的更改并将更改输出到`stdout`，如下所示：

```cpp
[[noreturn]] void
tail(std::fstream &file)
{
    while (true) {
        file.peek();
        while(!file.eof()) {
            auto pos = file.tellg();

            std::string buf;
            std::getline(file, buf, '\n');

            if (file.eof() && !file.good()) {
                file.seekg(pos);
                break;
            }

            std::cout << buf << '\n';
        }

        sleep(1);

        file.clear();
        file.sync();
    }
}
```

这个`tail`函数开始时告诉编译器这个函数不会返回，因为该函数包装在一个永不结束的`while(true)`循环中。

接下来，函数首先通过查看文件末尾来检查文件是否已到达末尾，然后使用`eof()`检查文件结束位。如果是，程序将休眠一秒钟，清除所有状态位，重新同步文件系统以查看是否有新的更改，然后再次循环。

如果读取指针不在文件末尾，则需要读取其当前位置，以便在需要时恢复其在文件中的位置。然后读取文件中的下一行并将其存储在缓冲区中。

尝试使用`getline`读取下一行可能会失败（例如，当文件中的最后一个字符不是换行符时）。如果发生这种情况，应忽略缓冲区的内容（因为它不是完整的一行），并且需要将读取指针恢复到其原始位置。

如果成功读取了下一行，它将输出到`stdout`，然后我们再次循环以查看是否需要读取更多行。

这个例子中的最后一个函数必须解析提供给我们程序的参数，以获取要 tail 的文件名，打开文件，然后使用新打开的文件调用`tail`函数，如下所示：

```cpp
int
protected_main(int argc, char **argv)
{
    std::string filename;
    auto args = make_span(argv, argc);

    if (args.size() < 2) {
        std::cin >> filename;
    }
    else {
        filename = ensure_z(args[1]).data();
    }

    if (auto file = std::fstream(filename, mode)) {
        tail(file);
    }

    throw std::runtime_error("failed to open file");
}
```

与以前的例子一样，我们使用`gsl::span`解析参数，以确保安全并符合 C++核心指南。如果没有为程序提供参数，我们将等待用户提供要 tail 的文件名。

如果提供了文件名，我们将打开文件并调用`tail()`。如果文件无法打开，我们会抛出异常。

为了编译这段代码，我们利用了同样的`CMakeLists.txt`文件，这是我们在其他示例中一直在使用的：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/CMakeLists.txt)。

有了这段代码，我们可以使用以下方式编译和执行这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter08/
> mkdir build
> cd build

> cmake ..
> make
> touch test.txt
> ./example2 test.txt
```

从另一个终端，我们可以对文件进行更改，如下所示：

```cpp
> cd Hands-On-System-Programming-with-CPP/Chapter08/build
> echo "Hello World" > test.txt
```

这将导致示例程序将以下内容输出到`stdout`：

```cpp
Hello World
```

为了确保程序忽略不完整的行，我们可以向文件中添加一个不完整的行，如下所示：

```cpp
> echo -n "Hello World" > test.txt
```

这导致示例程序没有输出。

# 比较 C++与 mmap 基准测试

在这个例子中，我们将对使用`std::fstream`和`mmap()`读取文件内容的差异进行基准测试。

值得注意的是，`mmap()`函数利用系统调用直接将文件映射到程序中，我们期望`mmap()`比本章中突出的 C++ API 更快。这是因为 C++ API 需要执行额外的内存复制，显然更慢。

我们将从定义我们打算读取的文件的大小开始，如下所示：

```cpp
constexpr auto size = 0x1000;
```

接下来，我们必须定义一个`benchmark`函数来记录执行操作所需的时间：

```cpp
template<typename FUNC>
auto benchmark(FUNC func) {
    auto stime = std::chrono::high_resolution_clock::now();
    func();
    auto etime = std::chrono::high_resolution_clock::now();

    return etime - stime;
}
```

在前面的函数中，我们利用高分辨率计时器来记录执行用户提供的函数所需的时间。值得注意的是，这个基准测试程序相对通用，可以用于许多非平凡的函数（因为即使使用高分辨率计时器，通常也很难对平凡函数进行基准测试）。

最后，我们需要创建一个文件读取，然后我们需要使用`std::fstream`和`mmap()`来读取文件，如下所示：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    using namespace std::chrono;

    {
        char buf[size] = {};
        if (auto file = std::fstream("test.txt", std::ios::out)) {
            file.write(buf, size);
        }
    }

    {
        char buf[size];
        if (auto file = std::fstream("test.txt", std::ios::in)) {
            auto time = benchmark([&file, &buf]{
                file.read(buf, size);
            });

            std::cout << "c++ time: "
                      << duration_cast<microseconds>(time).count()
                      << '\n';
        }
    }

    {
        void *buf;
        if (int fd = open("test.txt", O_RDONLY); fd != 0) {
            auto time = benchmark([&fd, &buf]{
                buf = mmap(NULL, size, PROT_READ, 0, fd, 0);
            });

            munmap(buf, size);

            std::cout << "mmap time: "
                      << duration_cast<microseconds>(time).count()
                      << '\n';
        }
    }

    return EXIT_SUCCESS;
}
```

`protected_main()`函数中的第一步是创建我们打算读取的文件，如下所示：

```cpp
char buf[size] = {};
if (auto file = std::fstream("test.txt", std::ios::out)) {
    file.write(buf, size);
}
```

为了做到这一点，我们以只写方式打开我们打算使用的文件，这也默认使用`std::ios::trunc`打开文件，以便在必要时为我们擦除文件的内容。最后，我们向文件写入`size`个零。

下一步是使用`std::fstream`读取文件，如下所示：

```cpp
char buf[size];
if (auto file = std::fstream("test.txt", std::ios::in)) {
    auto time = benchmark([&file, &buf]{
        file.read(buf, size);
    });

    std::cout << "c++ time: "
                << duration_cast<microseconds>(time).count()
                << '\n';
}
```

在使用`std::fstream`读取文件之前，我们首先以只读方式打开文件，这将文件打开到文件的开头。我们的文件读取然后封装在我们的基准函数中。基准测试的结果输出到`stdout`。

最后，最后一步是对`mmap()`做同样的操作，如下所示：

```cpp
void *buf;
if (int fd = open("test.txt", O_RDONLY); fd != 0) {
    auto time = benchmark([&fd, &buf]{
        buf = mmap(NULL, size, PROT_READ, 0, fd, 0);
    });

    munmap(buf, size);

    std::cout << "mmap time: "
                << duration_cast<microseconds>(time).count()
                << '\n';
}
```

与`std::fstream`一样，首先打开文件，然后在我们的基准函数中封装`mmap()`的使用。

为了编译这段代码，我们利用了同样的`CMakeLists.txt`文件，这是我们在其他示例中一直在使用的：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter08/CMakeLists.txt)。

有了这段代码，我们可以使用以下方式编译和执行这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter08/
> mkdir build
> cd build

> cmake ..
> make
> ./example3
c++ time: 16
mmap time: 3
```

如所示，`mmap()`的执行速度比`std::fstream`快。

# 摘要

在本章中，我们学习了如何以不同的方式打开文件，取决于我们打算如何使用文件本身。一旦打开，我们学习了如何使用`std::fstream` C++ API 读取和写入文件。

我们学习了字段和字节之间的区别，以及读写两种方法的优缺点，以及常见的不安全实践。此外，我们学习了支持函数，这些函数提供了在`std::fstream`API 中移动指针的能力，以操纵文件的读写方式。

此外，在本章中，我们对 C++17 新增的新文件系统 API 进行了广泛的概述，包括路径及其支持函数，用于操作文件和目录。

我们用三个示例结束了本章。在第一个示例中，我们编写了一个记录器，将`std::clog`的输出重定向到日志文件和`stdout`。第二个示例演示了如何使用 C++重写 tail POSIX 命令。

最后，在第三个示例中，我们编写了一些基准代码，以比较 POSIX、C 和 C++性能的差异。在下一章中，我们将介绍 C++分配器，包括如何创建有状态的分配器，例如在系统编程时可以提高内存性能和效率的内存池。

# 问题

1.  用于查看文件是否成功打开的函数的名称是什么？

1.  打开文件的默认模式是什么？

1.  如果您尝试从文件中读取非数字值到数字变量中会发生什么？

1.  使用`read()`或`write()`函数时可能会发生什么类型的错误？

1.  `/ = operator()`是否会自动为您的路径添加`/`？

1.  以下路径的 stem 是什么—`/home/user/test.txt`？

1.  以下路径的父目录是什么—`/home/user/test.txt`？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
