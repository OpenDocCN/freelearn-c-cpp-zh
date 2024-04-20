# 第一章：开始使用库开发

在本章中，我们将介绍一些有用的配方，用于创建我们自己的库，包括最少惊讶原则的解释，该原则鼓励我们使用用户已经熟悉的语义来实现库。我们还将看看如何对所有内容进行命名空间处理，以确保我们的自定义库不会与其他库发生冲突。此外，我们还将介绍如何创建仅包含头文件的库，以及与库开发相关的一些最佳实践。最后，我们将通过演示 boost 库来结束本章，以向您展示一个大型库的样子以及用户如何在自己的项目中使用它。

在本章中，我们将介绍以下配方：

+   理解最少惊讶原则

+   如何对所有内容进行命名空间处理

+   仅包含头文件的库

+   学习库开发的最佳实践

+   学习如何使用 boost API

让我们开始吧！

# 技术要求

要编译和运行本章中的示例，您必须具有管理访问权限，可以访问运行 Ubuntu 18.04 的计算机，并具有正常的互联网连接。在运行这些示例之前，您必须使用以下命令安装以下软件包：

```cpp
> sudo apt-get install build-essential git cmake
```

如果这个安装在除 Ubuntu 18.04 之外的任何操作系统上，那么将需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

# 理解最少惊讶原则

在使用现有的 C++库或创建自己的库时，理解**最少惊讶原则**（也称为**最少惊讶原则**）对于高效和有效地开发源代码至关重要。这个原则简单地指出，C++库提供的任何功能都应该是直观的，并且应该按照开发人员的期望进行操作。另一种说法是，库的 API 应该是自我记录的。尽管这个原则在设计库时至关重要，但它可以并且应该应用于所有形式的软件开发。在本教程中，我们将深入探讨这个原则。

# 准备工作

与本章中的所有配方一样，确保已满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有正确的工具来编译和执行本教程中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

执行以下步骤完成本教程：

1.  从新的终端运行以下代码来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter01
```

1.  要编译源代码，请运行以下代码：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe01_examples
```

1.  一旦源代码被编译，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe01_example01
The answer is: 42

> ./recipe01_example02
The answer is: 42

> ./recipe01_example03
The answer is: 42

> ./recipe01_example04
The answer is: 42
The answer is: 42

> ./recipe01_example05
The answer is: 42
The answer is: 42

> ./recipe01_example06
The answer is: 42
The answer is: 42

> ./recipe01_example07
The answer is: 42

> ./recipe01_example08
The answer is: 42

> ./recipe01_example09
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的功能以及它与本教程中所教授的课程的关系。

# 它是如何工作的...

如前一节所述，最少惊讶原则指出，库的 API 应该直观且自我记录，这个原则通常适用于所有形式的软件开发，而不仅仅是库设计。为了理解这一点，我们将看一些例子。

# 示例 1

示例 1 演示了最少惊讶原则，如下所示：

```cpp
#include <iostream>

int sub(int a, int b)
{ return a + b; }

int main(void)
{
    std::cout << "The answer is: " << sub(41, 1) << '\n';
    return 0;
}
```

如前面的示例所示，我们实现了一个库 API，它可以将两个整数相加并返回结果。问题在于我们将函数命名为`sub`，大多数开发人员会将其与减法而不是加法联系起来；尽管 API 按设计工作，但它违反了最少惊讶原则，因为 API 的名称不直观。

# 示例 2

示例 2 演示了最少惊讶原则，如下所示：

```cpp
#include <iostream>

void add(int a, int &b)
{ b += a; }

int main(void)
{
    int a = 41, b = 1;
    add(a, b);

    std::cout << "The answer is: " << b << '\n';
    return 0;
}
```

如前面的例子所示，我们已经实现了与上一个练习中实现的相同的库 API；它旨在添加两个数字并返回结果。这个例子的问题在于 API 实现了以下内容：

```cpp
b += a;
```

在这个例子中，最少惊讶原则以两种不同的方式被违反：

+   add 函数的参数是`a`，然后是`b`，尽管我们会将这个等式写成`b += a`，这意味着参数的顺序在直觉上是相反的。

+   对于这个 API 的用户来说，不会立即明显地意识到结果将在`b`中返回，而不必阅读源代码。

函数的签名应该使用用户已经习惯的语义来记录函数将如何执行，从而降低用户错误执行 API 的概率。

# 示例 3

示例 3 演示了最少惊讶原则如下：

```cpp
#include <iostream>

int add(int a, int b)
{ return a + b; }

int main(void)
{
    std::cout << "The answer is: " << add(41, 1) << '\n';
    return 0;
}
```

如前面的例子所示，我们在这里遵循了最少惊讶原则。API 旨在将两个整数相加并返回结果，API 直观地执行了预期的操作。

# 示例 4

示例 4 演示了最少惊讶原则如下：

```cpp
#include <stdio.h>
#include <iostream>

int main(void)
{
    printf("The answer is: %d\n", 42);
    std::cout << "The answer is: " << 42 << '\n';
    return 0;
}
```

如前面的例子所示，另一个很好的最少惊讶原则的例子是`printf()`和`std::cout`之间的区别。`printf()`函数需要添加格式说明符来将整数输出到`stdout`。`printf()`不直观的原因有很多：

+   对于初学者来说，`printf()`函数的名称，代表打印格式化，不直观（或者换句话说，函数的名称不是自我说明的）。其他语言通过选择更直观的打印函数名称来避免这个问题，比如`print()`或`console()`，这些名称更好地遵循了最少惊讶原则。

+   整数的格式说明符符号是`d`。对于初学者来说，这是不直观的。在这种特定情况下，`d`代表十进制，这是说*有符号整数*的另一种方式。更好的格式说明符可能是`i`，以匹配语言对`int`的使用。

与`std::cout`相比，它代表字符输出。虽然与`print()`或`console()`相比这不太直观，但比`printf()`更直观。此外，要将整数输出到`stdout`，用户不必记忆格式说明符表来完成任务。相反，他们可以简单地使用`<<`运算符。然后，API 会为您处理格式，这不仅更直观，而且更安全（特别是在使用`std::cin`而不是`scanf()`时）。

# 示例 5

示例 5 演示了最少惊讶原则如下：

```cpp
#include <iostream>

int main(void)
{
    auto answer = 41;

    std::cout << "The answer is: " << ++answer << '\n';
    std::cout << "The answer is: " << answer++ << '\n';

    return 0;
}
```

在前面的例子中，`++`运算符遵循最少惊讶原则。尽管初学者需要学习`++`代表递增运算符，意味着变量增加`1`，但`++`与变量的位置相当有帮助。

要理解`++variable`和`variable++`之间的区别，用户只需像平常一样从左到右阅读代码。当`++`在左边时，变量被递增，然后返回变量的内容。当`++`在右边时，返回变量的内容，然后递增变量。关于`++`位置的唯一问题是，左边的`++`通常更有效率（因为实现不需要额外的逻辑来存储递增操作之前的变量值）。

# 示例 6

示例 6 演示了最少惊讶原则如下：

```cpp
#include <iostream>

int add(int a, int b)
{ return a + b; }

int Sub(int a, int b)
{ return a - b; }

int main(void)
{
    std::cout << "The answer is: " << add(41, 1) << '\n';
    std::cout << "The answer is: " << Sub(43, 1) << '\n';

    return 0;
}
```

如前面的代码所示，我们实现了两个不同的 API。第一个是将两个整数相加并返回结果，而第二个是将两个整数相减并返回结果。减法函数的问题有两个：

+   加法函数是小写的，而减法函数是大写的。这不直观，API 的用户必须学习哪些 API 是小写的，哪些是大写的。

+   C++标准 API 都是蛇形命名法，意思是它们利用小写单词并使用`_`来表示空格。一般来说，最好设计 C++库 API 时使用蛇形命名法，因为初学者更有可能找到这种方式直观。值得注意的是，尽管这通常是这样，但蛇形命名法的使用是高度主观的，有几种语言不遵循这一指导。最重要的是选择一个约定并坚持下去。

再次确保您的 API 模仿现有语义，确保用户可以快速轻松地学会使用您的 API，同时降低用户错误编写 API 的可能性，从而导致编译错误。

# 示例 7

示例 7 演示了最小惊讶原则的如下内容：

```cpp
#include <queue>
#include <iostream>

int main(void)
{
    std::queue<int> my_queue;

    my_queue.emplace(42);
    std::cout << "The answer is: " << my_queue.front() << '\n';
    my_queue.pop();

    return 0;
}
```

在前面的例子中，我们向您展示了如何使用`std::queue`将整数添加到队列中，将队列输出到`stdout`，并从队列中删除元素。这个例子的重点是要突出 C++已经有一套标准的命名约定，应该在 C++库开发过程中加以利用。

如果您正在设计一个新的库，使用 C++已经定义的相同命名约定对您的库的用户是有帮助的。这样做将降低使用门槛，并提供更直观的 API。

# 示例 8

示例 8 演示了最小惊讶原则的如下内容：

```cpp
#include <iostream>

auto add(int a, int b)
{ return a + b; }

int main(void)
{
    std::cout << "The answer is: " << add(41, 1) << '\n';
    return 0;
}
```

如前面的例子所示，我们展示了`auto`的使用方式，告诉编译器自动确定函数的返回类型，这不符合最小惊讶原则。尽管`auto`对于编写通用代码非常有帮助，但在设计库 API 时应尽量避免使用。特别是为了让 API 的用户理解 API 的输入和输出，用户必须阅读 API 的实现，因为`auto`不指定输出类型。

# 示例 9

示例 9 演示了最小惊讶原则的如下内容：

```cpp
#include <iostream>

template <typename T>
T add(T a, T b)
{ return a + b; }

int main(void)
{
    std::cout << "The answer is: " << add(41, 1) << '\n';
    return 0;
}
```

如前面的例子所示，我们展示了一种更合适的方式来支持最小惊讶原则，同时支持通用编程。通用编程（也称为模板元编程或使用 C++模板进行编程）为程序员提供了一种在不声明算法中使用的类型的情况下创建算法的方法。在这种情况下，`add`函数不会规定输入类型，允许用户添加任何类型的两个值（在这种情况下，类型称为`T`，可以采用支持`add`运算符的任何类型）。我们返回一个类型`T`，而不是返回`auto`，因为`auto`不会声明输出类型。尽管`T`在这里没有定义，因为它代表任何类型，但它告诉 API 的用户，我们输入到这个函数中的任何类型也将被函数返回。这种逻辑在 C++标准库中大量使用。

# 如何对一切进行命名空间

创建库时，对一切进行命名空间是很重要的。这样做可以确保库提供的 API 不会与用户代码或其他库提供的设施发生名称冲突。在本示例中，我们将演示如何在我们自己的库中做到这一点。

# 准备工作

与本章中的所有示例一样，请确保已满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

要完成本文，您需要执行以下步骤：

1.  从新的终端中，运行以下命令下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter01
```

1.  要编译源代码，请运行以下代码：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe02_examples
```

1.  一旦源代码被编译，您可以通过运行以下命令来执行本文中的每个示例：

```cpp
> ./recipe02_example01
The answer is: 42

> ./recipe02_example02
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本文所教授的课程的关系。

# 它是如何工作的...

C++提供了将代码包裹在`namespace`中的能力，这简单地将`namespace`名称添加到`namespace`代码中的所有函数和变量（应该注意的是，C 风格的宏不包括在`namespace`中，并且应该谨慎使用，因为 C 宏是预处理器功能，不会对代码的编译语法产生影响）。为了解释为什么我们在创建自己的库时应该将所有东西都放在`namespace`中，我们将看一些例子。

# 示例 1

示例 1 演示了如何在 C++`namespace`中包裹库的 API：

```cpp
// Contents of library.h

namespace library_name
{
    int my_api() { return 42; }
    // ...
}

// Contents of main.cpp

#include <iostream>

int main(void)
{
    using namespace library_name;

    std::cout << "The answer is: " << my_api() << '\n';
    return 0;
}
```

如上例所示，库的内容被包裹在一个`namespace`中，并存储在头文件中（这个例子演示了一个头文件库，这是一种非常有用的设计方法，因为最终用户不需要编译库，将其安装在他/她的系统上，然后链接到它们）。库用户只需包含库头文件，并使用`using namespace library_name`语句来解开库的 API。如果用户有多个具有相同 API 名称的库，可以省略此语句以消除任何歧义。

# 示例 2

示例 2 扩展了上一个示例，并演示了如何在 C++命名空间头文件库中包裹库的 API，同时包括全局变量：

```cpp
// Contents of library.h

namespace library_name
{
    namespace details { inline int answer = 42; }

    int my_api() { return details::answer; }
    // ...
}

// Contents of main.cpp

#include <iostream>

int main(void)
{
    using namespace library_name;

    std::cout << "The answer is: " << my_api() << '\n';
    return 0;
}
```

在上面的例子中，利用 C++17 创建了一个包裹在我们库的`namespace`中的`inline`全局变量。`inline`变量是必需的，因为头文件库没有源文件来定义全局变量；没有`inline`关键字，在头文件中定义全局变量会导致变量被多次定义（也就是说，在编译过程中会出现链接错误）。C++17 通过添加`inline`全局变量解决了这个问题，这允许头文件库定义全局变量而无需使用 tricky magic（比如从单例样式函数返回静态变量的指针）。

除了库的`namespace`，我们还将全局变量包裹在`details namespace`中。这是为了在库的用户声明`using namespace library_name`的情况下，在库内创建一个`private`的地方。如果用户这样做，所有被`library_name`命名空间包裹的 API 和变量都会在`main()`函数的范围内变得全局可访问。因此，任何不希望用户访问的私有 API 或变量都应该被第二个`namespace`（通常称为`details`）包裹起来，以防止它们的全局可访问性。最后，利用 C++17 的`inline`关键字允许我们在库中创建全局变量，同时仍然支持头文件库的设计。

# 头文件库

头文件库就像它们的名字一样；整个库都是使用头文件实现的（通常是一个头文件）。头文件库的好处在于，它们很容易包含到您的项目中，只需包含头文件即可（不需要编译库，因为没有需要编译的源文件）。在本配方中，我们将学习在尝试创建头文件库时出现的一些问题以及如何克服这些问题。这个配方很重要，因为如果您计划创建自己的库，头文件库是一个很好的起点，并且可能会增加您的库被下游用户整合到他们的代码库中的几率。

# 准备工作

与本章中的所有配方一样，请确保已满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有正确的工具来编译和执行本配方中的示例。完成这些步骤后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 操作步骤...

完成此配方，您需要执行以下步骤：

1.  从新的终端中，运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter01
```

1.  要编译源代码，请运行以下代码：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe03_examples
```

1.  源代码编译完成后，您可以通过运行以下命令执行本配方中的每个示例：

```cpp
> ./recipe03_example01
The answer is: 42

> ./recipe03_example02
The answer is: 42

> ./recipe03_example03
The answer is: 42

> ./recipe03_example04
The answer is: 42
The answer is: 2a

> ./recipe03_example05

> ./recipe03_example06
The answer is: 42

> ./recipe03_example07
The answer is: 42
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它们与本配方中所教授的课程的关系。

# 工作原理...

要创建一个头文件库，只需确保所有代码都在头文件中实现，如下所示：

```cpp
#ifndef MY_LIBRARY
#define MY_LIBRARY

namespace library_name
{
    int my_api() { return 42; }
}

#endif
```

前面的示例实现了一个简单的库，其中有一个函数。这个库的整个实现可以在一个头文件中实现，并包含在我们的代码中，如下所示：

```cpp
#include "my_library.h"
#include <iostream>

int main(void)
{
    using namespace library_name;

    std::cout << "The answer is: " << my_api() << '\n';
    return 0;
}
```

尽管创建头文件库似乎很简单，但在尝试创建头文件库时会出现一些问题，这些问题应该考虑在内。

# 如何处理包含

在前面的示例中，您可能已经注意到，当我们使用我们的自定义头文件库时，我们首先包含了库。这是编写头文件库的一个基本步骤。在为头文件库编写示例或测试时，我们的库应该是我们包含的第一件事，以确保所有头文件的依赖关系都在头文件库中定义，而不是在我们的示例或测试中定义。

例如，假设我们将我们的库更改如下：

```cpp
#ifndef MY_LIBRARY
#define MY_LIBRARY

namespace library_name
{
    void my_api()
    {
        std::cout << "The answer is: 42" << '\n';
    }
}

#endif
```

如前面的代码片段所示，我们的 API 现在不再返回整数，而是输出到 `stdout`。我们可以如下使用我们的新 API：

```cpp
#include <iostream>
#include "my_library.h"

int main(void)
{
    library_name::my_api();
    return 0;
}
```

尽管前面的代码编译和运行如预期，但代码中存在一个错误，这个错误可能只有您的库的用户才能识别出来。具体来说，如果您的库的用户交换了包含的顺序或者没有`#include <iostream>`，代码将无法编译并产生以下错误：

![](img/14a94075-f946-458b-aa5c-a4bfd158978a.png)

这是因为头文件库本身没有包含所有的依赖关系。由于我们的示例将库放在其他包含之后，我们的示例意外地隐藏了这个问题。因此，当创建自己的头文件库时，始终在测试和示例中首先包含库，以确保这种类型的问题永远不会发生在您的用户身上。

# 全局变量

头文件库的最大限制之一是，在 C++17 之前，没有办法创建全局变量。尽管应尽量避免使用全局变量，但有些情况下是必需的。为了演示这一点，让我们创建一个简单的 API，输出到 `stdout` 如下：

```cpp
#ifndef MY_LIBRARY
#define MY_LIBRARY

#include <iostream>
#include <iomanip>

namespace library_name
{
    void my_api(bool show_hex = false)
    {
        if (show_hex) {
            std::cout << std::hex << "The answer is: " << 42 << '\n';
        }
        else {
            std::cout << std::dec << "The answer is: " << 42 << '\n';
        }
    }
}

#endif
```

前面的示例创建了一个 API，将输出到`stdout`。如果使用`true`而不是默认的`false`执行 API，则将以十六进制而不是十进制格式输出整数。在这个例子中，从十进制到十六进制的转换实际上是我们库中的一个配置设置。然而，如果没有全局变量，我们将不得不采用其他机制来实现这一点，包括宏或前面的示例中的函数参数；后者选择甚至更糟，因为它将库的配置与其 API 耦合在一起，这意味着任何额外的配置选项都会改变 API 本身。

解决这个问题的最佳方法之一是在 C++17 中使用全局变量，如下所示：

```cpp
#ifndef MY_LIBRARY
#define MY_LIBRARY

#include <iostream>
#include <iomanip>

namespace library_name
{
    namespace config
    {
        inline bool show_hex = false;
    }

    void my_api()
    {
        if (config::show_hex) {
            std::cout << std::hex << "The answer is: " << 42 << '\n';
        }
        else {
            std::cout << std::dec << "The answer is: " << 42 << '\n';
        }
    }
}

#endif
```

如前面的示例所示，我们在库中添加了一个名为`config`的新命名空间。我们的 API 不再需要任何参数，并根据内联全局变量确定如何运行。现在，我们可以按以下方式使用此 API：

```cpp
#include "my_library.h"
#include <iostream>

int main(void)
{
    library_name::my_api();
    library_name::config::show_hex = true;
    library_name::my_api();

    return 0;
}
```

以下是输出的结果：

![](img/2abe9bca-2a0e-4075-a8f8-e5fa3140663e.png)

需要注意的是，我们将配置设置放在`config`命名空间中，以确保我们的库命名空间不会因名称冲突而被污染，从而确保全局变量的意图是明显的。

# C 风格宏的问题

C 风格宏的最大问题在于，如果将它们放在 C++命名空间中，它们的名称不会被命名空间修饰。这意味着宏总是污染全局命名空间。例如，假设您正在编写一个需要检查变量值的库，如下所示：

```cpp
#ifndef MY_LIBRARY
#define MY_LIBRARY

#include <cassert>

namespace library_name
{
    #define CHECK(a) assert(a == 42)

    void my_api(int val)
    {
        CHECK(val);
    }
}

#endif
```

如前面的代码片段所示，我们创建了一个简单的 API，它在实现中使用了 C 风格的宏来检查整数值。前面示例的问题在于，如果您尝试在自己的库中使用单元测试库，很可能会遇到命名空间冲突。

C++20 可以通过使用 C++20 模块来解决这个问题，并且这是我们将在第十三章中更详细讨论的一个主题，*奖励-使用 C++20 功能*。具体来说，C++20 模块不会向库的用户公开 C 风格的宏。这样做的积极方面是，您将能够使用宏而不会出现命名空间问题，因为您的宏不会暴露给用户。这种方法的缺点是，许多库作者使用 C 风格的宏来配置库（例如，在包含库之前定义宏以更改其默认行为）。这种类型的库配置在 C++模块中将无法工作，除非在编译库时在命令行上定义了这些宏。

直到 C++20 可用，如果需要使用宏，请确保手动向宏名称添加修饰，如下所示：

```cpp
#define LIBRARY_NAME__CHECK(a) assert(a == 42)
```

前面的代码行将执行与宏位于 C++命名空间内相同的操作，确保您的宏不会与其他库的宏或用户可能定义的宏发生冲突。

# 如何将大型库实现为仅头文件

理想情况下，头文件库应使用单个头文件实现。也就是说，用户只需将单个头文件复制到其源代码中即可使用该库。这种方法的问题在于，对于非常大的项目，单个头文件可能会变得非常庞大。一个很好的例子是 C++中一个流行的 JSON 库，位于此处：[`github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp`](https://github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp)。

在撰写本文时，上述库的代码行数超过 22,000 行。尝试对一个有 22,000 行代码的文件进行修改将是非常糟糕的（即使您的编辑器能够处理）。一些项目通过使用多个头文件实现其仅包含头文件库，并使用单个头文件根据需要包含各个头文件来解决这个问题（例如，Microsoft 的 C++指南支持库就是这样实现的）。这种方法的问题在于用户必须复制和维护多个头文件，随着复杂性的增加，这开始破坏头文件库的目的。

另一种处理这个问题的方法是使用诸如 CMake 之类的工具从多个头文件中自动生成单个头文件。例如，在下面的示例中，我们有一个仅包含头文件的库，其中包含以下头文件：

```cpp
#include "config.h"

namespace library_name
{
    void my_api()
    {
        if (config::show_hex) {
            std::cout << std::hex << "The answer is: " << 42 << '\n';
        }
        else {
            std::cout << std::dec << "The answer is: " << 42 << '\n';
        }
    }
}
```

如前面的代码片段所示，这与我们的配置示例相同，唯一的区别是示例的配置部分已被替换为对`config.h`文件的包含。我们可以按照以下方式创建这个第二个头文件：

```cpp
namespace library_name
{
    namespace config
    {
        inline bool show_hex = false;
    }
}
```

这实现了示例的剩余部分。换句话说，我们已经将我们的头文件分成了两个头文件。我们仍然可以像下面这样使用我们的头文件：

```cpp
#include "apis.h"

int main(void)
{
    library_name::my_api();
    return 0;
}
```

然而，问题在于我们的库的用户需要拥有两个头文件的副本。为了解决这个问题，我们需要自动生成一个头文件。有许多方法可以做到这一点，但以下是使用 CMake 的一种方法：

```cpp
file(STRINGS "config.h" CONFIG_H)
file(STRINGS "apis.h" APIS_H)

list(APPEND MY_LIBRARY_SINGLE
    "${CONFIG_H}"
    ""
    "${APIS_H}"
)

file(REMOVE "my_library_single.h")
foreach(LINE IN LISTS MY_LIBRARY_SINGLE)
    if(LINE MATCHES "#include \"")
        file(APPEND "my_library_single.h" "// ${LINE}\n")
    else()
        file(APPEND "my_library_single.h" "${LINE}\n")
    endif()
endforeach()
```

上面的代码使用`file()`函数将两个头文件读入 CMake 变量。这个函数将每个变量转换为 CMake 字符串列表（每个字符串是文件中的一行）。然后，我们将两个文件合并成一个列表。为了创建我们的新的自动生成的单个头文件，我们遍历列表，并将每一行写入一个名为`my_library_single.h`的新头文件。最后，如果我们看到对本地包含的引用，我们将其注释掉，以确保没有引用我们的额外头文件。

现在，我们可以像下面这样使用我们的新单个头文件：

```cpp
#include "my_library_single.h"

int main(void)
{
    library_name::my_api();
    return 0;
}
```

使用上述方法，我们可以开发我们的库，使用尽可能多的包含，并且我们的构建系统可以自动生成我们的单个头文件，这将被最终用户使用，为我们提供了最好的两全其美。

# 学习库开发最佳实践

在编写自己的库时，所有库作者都应该遵循某些最佳实践。在本教程中，我们将探讨一些优先级较高的最佳实践，并总结一些关于一个专门定义这些最佳实践的项目的信息，包括一个注册系统，为您的库提供编译的评分。这个教程很重要，因为它将教会您如何制作最高质量的库，确保强大和充满活力的用户群体。

# 准备工作

与本章中的所有示例一样，请确保所有技术要求都已满足，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake clang-tidy valgrind
```

这将确保您的操作系统具有正确的工具来编译和执行本教程中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

您需要执行以下步骤来完成本教程：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter01
```

1.  要编译源代码，请运行以下代码：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe04_examples
```

1.  一旦源代码被编译，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe04_example01 
21862
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程中所教授的课程的关系。

# 它是如何工作的...

每个图书馆的作者都应该确保他们的图书馆易于使用并且可以整合到用户自己的项目中。这样做将确保您的用户继续使用您的图书馆，从而导致用户群随着时间的推移不断增长。让我们来看看其中一些最佳实践。

# 警告呢？

任何图书馆作者的最低挂果是确保您的代码尽可能多地编译。遗憾的是，GCC 并没有简化这个过程，因为没有一个警告标志可以统治所有警告，特别是因为 GCC 有许多对于现代 C ++版本来说并不有用的警告标志（换句话说，它们在某种程度上是相互排斥的）。开始的最佳地方是以下警告：

```cpp
-Wall -Wextra -pedantic -Werror
```

这将打开大部分重要的警告，同时确保您的示例或测试编译时生成错误的任何警告。然而，对于一些库来说，这还不够。在撰写本文时，微软的指南支持库使用以下标志：

```cpp
-Wall -Wcast-align -Wconversion -Wctor-dtor-privacy -Werror -Wextra -Wpedantic -Wshadow -Wsign-conversion
```

GSL 使用的另一个警告是转换警告，它会在您在不同的整数类型之间转换时告诉您。如果您使用 Clang，这个过程可能会更容易，因为它提供了`-Weverything`。如果筛选 GCC 提供的所有警告太麻烦，解决这个问题的一种方法是确保您的库在打开此警告的情况下与 Clang 编译器编译，这将确保您的代码与 GCC 提供的大部分警告一起编译。这样，当用户必须确保他们的代码中启用了特定警告时，您的用户在使用您的库时就不会遇到麻烦，因为您已经尽可能地测试了其中的许多。

# 静态和动态分析

除了测试警告之外，库还应该使用静态和动态分析工具进行测试。再次强调，作为图书馆的作者，您必须假设您的用户可能会使用静态和动态分析工具来加强他们自己应用程序的质量。如果您的库触发了这些工具，您的用户更有可能寻找经过更彻底测试的替代方案。

对于 C ++，有大量工具可用于分析您的库。在本教程中，我们将专注于 Clang Tidy 和 Valgrind，它们都是免费使用的。让我们看看以下简单的例子：

```cpp
#include <iostream>

int universe()
{
    auto i = new int;
    int the_answer;
    return the_answer;
}

int main()
{
    std::cout << universe() << '\n';
    return 0;
}
```

在前面的例子中，我们创建了一个名为`universe()`的函数，它返回一个整数并分配一个整数。在我们的主函数中，我们的`universe()`函数将结果输出到`stdout`。

要对前面的代码进行静态分析，我们可以使用 CMake，如下所示：

```cpp
set(CMAKE_CXX_CLANG_TIDY clang-tidy)
```

前面的代码告诉 CMake 在编译前面的示例时使用`clang-tidy`。当我们编译代码时，我们得到以下结果：

![](img/90475cde-2dd5-45e5-a2fe-e2a0ed81b304.png)

如果您的库的用户已经打开了使用 Clang Tidy 进行静态分析，这可能是他们会收到的错误，即使他们的代码完全正常。如果您正在使用别人的库并遇到了这个问题，克服这个问题的一种方法是将库包含为系统包含，这告诉 Clang Tidy 等工具忽略这些错误。然而，这并不总是有效，因为有些库需要使用宏，这会将库的逻辑暴露给您自己的代码，导致混乱。一般来说，如果您是库开发人员，尽可能多地对您的库进行静态分析，因为您不知道您的用户可能如何使用您的库。

动态分析也是一样。前面的分析没有检测到明显的内存泄漏。为了识别这一点，我们可以使用`valgrind`，如下所示：

![](img/e501ffa7-bfb7-46d1-b2e1-38630c9e0921.png)

如前面的屏幕截图所示，`valgrind`能够检测到我们代码中的内存泄漏。实际上，`valgrind`还检测到我们在`universe()`函数中从未初始化临时变量的事实，但输出内容过于冗长，无法在此展示。再次强调，如果你未能识别出这些类型的问题，你最终会暴露这些错误给你的用户。

# 文档

文档对于任何良好的库来说都是绝对必要的。除了有 bug 的代码，缺乏文档也会绝对阻止其他人使用你的库。库应该易于设置和安装，甚至更容易学习和融入到你自己的应用程序中。使用现有的 C++库最令人沮丧的一点就是缺乏文档。

# CII 最佳实践

在这个示例中，我们涉及了一些所有库开发者都应该在其项目中应用的常见最佳实践。除了这些最佳实践，CII 最佳实践项目在这里提供了更完整的最佳实践清单：[`bestpractices.coreinfrastructure.org/en`](https://bestpractices.coreinfrastructure.org/en)。

CII 最佳实践项目提供了一个全面的最佳实践清单，随着时间的推移进行更新，库开发者（以及一般的应用程序）可以利用这些最佳实践。这些最佳实践分为通过、银和金三个级别，金级实践是最难实现的。你的得分越高，用户使用你的库的可能性就越大，因为这显示了承诺和稳定性。

# 学习如何使用 boost API

boost 库是一组设计用于与标准 C++库配合使用的库。事实上，目前由 C++提供的许多库都起源于 boost 库。boost 库提供了从容器、时钟和定时器到更复杂的数学 API，如图形和 CRC 计算等一切。在这个示例中，我们将学习如何使用 boost 库，特别是演示一个大型库的样子以及如何将这样的库包含在用户的项目中。这个示例很重要，因为它将演示一个库可以变得多么复杂，教会你如何相应地编写你自己的库。

# 准备工作

与本章中的所有示例一样，请确保已满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake libboost-all-dev
```

这将确保你的操作系统具有编译和执行本教程中示例所需的正确工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做到...

你需要执行以下步骤来完成这个示例：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter01
```

1.  要编译源代码，请运行以下代码：

```cpp
> mkdir build && cd build
> cmake ..
> make recipe05_examples
```

1.  源代码编译完成后，你可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe05_example01
Date/Time: 1553894555446451393 nanoseconds since Jan 1, 1970
> ./recipe05_example02
[2019-03-29 15:22:36.756819] [0x00007f5ee158b740] [debug] debug message
[2019-03-29 15:22:36.756846] [0x00007f5ee158b740] [info] info message
```

在接下来的部分，我们将逐个介绍这些例子，并解释每个示例程序的作用，以及它们与本教程中所教授的课程的关系。

# 工作原理...

boost 库提供了一组用户 API，实现了大多数程序中常用的功能。这些库可以包含在你自己的项目中，简化你的代码，并提供一个完成的库可能是什么样子的示例。为了解释你自己的库如何被他人利用，让我们看一些如何使用 boost 库的示例。

# 例子 1

在这个例子中，我们使用 boost API 将当前日期和时间输出到`stdout`，如下所示：

```cpp
#include <iostream>
#include <boost/chrono.hpp>

int main(void)
{
    using namespace boost::chrono;

    std::cout << "Date/Time: " << system_clock::now() << '\n';
    return 0;
}
```

如前面的示例所示，当前日期和时间以自 Unix 纪元（1970 年 1 月 1 日）以来的纳秒总数的形式被输出到`stdout`。除了在源代码中包含 boost，你还必须将你的应用程序链接到 boost 库。在这种情况下，我们需要链接到以下内容：

```cpp
-lboost_chrono -lboost_system -lpthread
```

如何完成这一步骤的示例可以在随这个示例一起下载的`CMakeLists.txt`文件中看到。一旦这些库被链接到你的项目中，你的代码就能够利用它们内部的 API。这个额外的步骤就是为什么仅包含头文件的库在创建自己的库时可以如此有用，因为它们消除了额外链接的需要。

# 示例 2

在这个例子中，我们演示了如何使用 boost 的 trivial logging APIs 来记录到控制台，如下所示：

```cpp
#include <boost/log/trivial.hpp>

int main(void)
{
    BOOST_LOG_TRIVIAL(debug) << "debug message";
    BOOST_LOG_TRIVIAL(info) << "info message";
    return 0;
}
```

如前面的示例所示，`"debug message"`和`"info message"`消息被输出到`stdout`。除了链接正确的 boost 库，我们还必须在编译过程中包含以下定义：

```cpp
-DBOOST_LOG_DYN_LINK -lboost_log -lboost_system -lpthread
```

再次，链接这些库可以确保你在代码中使用的 API（如前面的示例所示）存在于可执行文件中。

# 另请参阅

有关 boost 库的更多信息，请查看[`www.boost.org/`](https://www.boost.org/)。
