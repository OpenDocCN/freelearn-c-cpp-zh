# 调试和测试

在本章中，您将学习如何正确测试和调试您的 C++应用程序。这很重要，因为没有良好的测试和调试，您的 C++应用程序很可能包含难以检测的错误，这将降低它们的整体可靠性、稳定性和安全性。

本章将从全面概述单元测试开始，这是在单元级别测试代码的行为，并且还将介绍如何利用现有库加快编写测试的过程。接下来，它将演示如何使用 ASAN 和 UBSAN 动态分析工具来检查内存损坏和未定义行为。最后，本章将简要介绍如何在自己的代码中利用`NDEBUG`宏来添加调试逻辑以解决问题。

本章包含以下教程：

+   掌握单元测试

+   使用 ASAN，地址检查器

+   使用 UBSAN，未定义行为检查器

+   使用`#ifndef NDEBUG`条件性地执行额外的检查

# 技术要求

要编译和运行本章中的示例，您必须具有管理访问权限的计算机，该计算机运行 Ubuntu 18.04，并具有功能正常的互联网连接。在运行这些示例之前，您必须安装以下内容：

```cpp
> sudo apt-get install build-essential git cmake
```

如果这是安装在 Ubuntu 18.04 以外的任何操作系统上，则需要 GCC 7.4 或更高版本和 CMake 3.6 或更高版本。

本章的代码文件可以在[`github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter07`](https://github.com/PacktPublishing/Advanced-CPP-CookBook/tree/master/chapter07)找到。

# 掌握单元测试

在这个教程中，我们将学习如何对我们的 C++代码进行单元测试。有几种不同的方法可以确保您的 C++代码以可靠性、稳定性、安全性和规范性执行。

单元测试是在基本单元级别测试代码的行为，是任何测试策略的关键组成部分。这个教程很重要，不仅因为它将教会您如何对代码进行单元测试，还因为它将解释为什么单元测试如此关键，以及如何利用现有库加快对 C++代码进行单元测试的过程。

# 准备工作

在开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本教程中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤进行教程：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter07
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe01_examples
```

1.  源代码编译完成后，您可以通过运行以下命令来执行本教程中的每个示例：

```cpp
> ./recipe01_example01
===========================================================================
All tests passed (1 assertion in 1 test case)

> ./recipe01_example02
===========================================================================
All tests passed (6 assertions in 1 test case)

> ./recipe01_example03
===========================================================================
All tests passed (8 assertions in 1 test case)

> ./recipe01_example04
===========================================================================
All tests passed (1 assertion in 1 test case)

> ./recipe01_example05
...
===========================================================================
test cases: 1 | 1 passed
assertions: - none -

> ./recipe01_example06
...
===========================================================================
test cases: 5 | 3 passed | 2 failed
assertions: 8 | 6 passed | 2 failed

> ./recipe01_example07
===========================================================================
test cases: 1 | 1 passed
assertions: - none -

> ./recipe01_example08
===========================================================================
All tests passed (3 assertions in 1 test case)
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本教程所教授的课程的关系。

# 它是如何工作的...

仅仅编写您的 C++应用程序，并希望它按预期工作而不进行任何测试，肯定会导致可靠性、稳定性和安全性相关的错误。这个教程很重要，因为在发布之前测试您的应用程序可以确保您的应用程序按预期执行，最终为您节省时间和金钱。

有几种不同的方法可以测试您的代码，包括系统级、集成、长期稳定性以及静态和动态分析等。在这个教程中，我们将专注于**单元测试**。单元测试将应用程序分解为功能**单元**，并测试每个单元以确保其按预期执行。通常，在实践中，每个函数和对象（即类）都是一个应该独立测试的单元。

有几种不同的理论，关于如何执行单元测试，整本书都是关于这个主题的。有些人认为应该测试函数或对象中的每一行代码，利用覆盖率工具来确保合规性，而另一些人认为单元测试应该是需求驱动的，采用黑盒方法。一种常见的开发过程称为**测试驱动开发**，它规定所有测试，包括单元测试，都应该在编写任何源代码之前编写，而**行为驱动开发**则进一步采用特定的、以故事为驱动的方法来进行单元测试。

每种测试模型都有其优缺点，您选择的方法将基于您正在编写的应用程序类型、您遵循的软件开发过程类型以及您可能需要或不需要遵循的任何政策。不管您做出什么选择，单元测试可能会成为您测试方案的一部分，这个示例将为您提供如何对 C++应用程序进行单元测试的基础。

尽管可以使用标准的 C++进行单元测试（例如，这就是`libc++`进行单元测试的方法），但单元测试库有助于简化这个过程。在这个示例中，我们将利用`Catch2`单元测试库，可以在以下网址找到

[`github.com/catchorg/Catch2.git`](https://github.com/catchorg/Catch2.git)。

尽管我们将回顾 Catch2，但正在讨论的原则适用于大多数可用的单元测试库，甚至适用于标准的 C++，如果您选择不使用辅助库。要利用 Catch2，只需执行以下操作：

```cpp
> git clone https://github.com/catchorg/Catch2.git catch
> cd catch
> mkdir build
> cd build
> cmake ..
> make
> sudo make install
```

您还可以使用 CMake 的`ExternalProject_Add`，就像我们在 GitHub 上的示例中所做的那样，来利用库的本地副本。

要了解如何使用 Catch2，让我们看下面这个简单的例子：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

TEST_CASE("the answer")
{
   CHECK(true);
}
```

运行时，我们看到以下输出：

![](img/82b7302a-7165-4cf8-92f1-83a6491e786f.png)

在前面的例子中，我们首先定义了`CATCH_CONFIG_MAIN`。这告诉 Catch2 库我们希望它为我们创建`main()`函数。这必须在我们包含 Catch2`include`语句之前定义，这是我们在前面的代码中所做的。

下一步是定义一个测试用例。每个单元都被分解成测试单元，测试所讨论的单元。每个测试用例的粒度由您决定：有些人选择为每个被测试的单元设置一个单独的测试用例，而其他人，例如，选择为每个被测试的函数设置一个测试用例。`TEST_CASE()`接受一个字符串，允许您提供测试用例的描述，当测试失败时，这对于帮助您确定测试代码中失败发生的位置是有帮助的，因为 Catch2 将输出这个字符串。我们简单示例中的最后一步是使用`CHECK()`宏。这个宏执行一个特定的测试。每个`TEST_CASE()`可能会有几个`CHECK()`宏，旨在为单元提供特定的输入，然后验证生成的输出。

一旦编译和执行，单元测试库将提供一些输出文本，描述如何执行测试。在这种情况下，库说明所有测试都通过了，这是期望的结果。

为了更好地理解如何在自己的代码中利用单元测试，让我们看下面这个更复杂的例子：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <vector>
#include <iostream>
#include <algorithm>

TEST_CASE("sort a vector")
{
    std::vector<int> v{4, 8, 15, 16, 23, 42};
    REQUIRE(v.size() == 6);

    SECTION("sort descending order") {
        std::sort(v.begin(), v.end(), std::greater<int>());

        CHECK(v.front() == 42);
        CHECK(v.back() == 4);
    }

    SECTION("sort ascending order") {
        std::sort(v.begin(), v.end(), std::less<int>());

        CHECK(v.front() == 4);
        CHECK(v.back() == 42);
    }
}
```

像前面的例子一样，我们使用`CATCH_CONFIG_MAIN`宏包含 Catch2，然后定义一个带有描述的单个测试用例。在这个例子中，我们正在测试对向量进行排序的能力，所以这是我们提供的描述。我们在测试中要做的第一件事是创建一个包含预定义整数列表的整数向量。

接下来我们使用`REQUIRE()`宏进行测试，确保向量中有`6`个元素。`REQUIRE()`宏类似于`CHECK()`，因为两者都检查宏内部的语句是否为真。不同之处在于，`CHECK()`宏将报告错误，然后继续执行，而`REQUIRE()`宏将停止执行，中止单元测试。这对于确保单元测试基于测试可能做出的任何假设正确构建是有用的。随着时间的推移，单元测试的成熟度越来越重要，其他程序员会添加和修改单元测试，以确保单元测试不会引入错误，因为没有比测试和调试单元测试更糟糕的事情了。

`SECTION()`宏用于进一步分解我们的测试，并提供添加每个测试的常见设置代码的能力。在前面的示例中，我们正在测试向量的`sort()`函数。`sort()`函数可以按不同的方向排序，这个单元测试必须验证。如果没有`SECTION()`宏，如果测试失败，将很难知道失败是由于按升序还是按降序排序。此外，`SECTION()`宏确保每个测试不会影响其他测试的结果。

最后，我们使用`CHECK()`宏来确保`sort()`函数按预期工作。单元测试也应该检查异常。在下面的示例中，我们将确保异常被正确抛出：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <vector>
#include <iostream>
#include <algorithm>

void foo(int val)
{
    if (val != 42) {
        throw std::invalid_argument("The answer is: 42");
    }
}

TEST_CASE("the answer")
{
    CHECK_NOTHROW(foo(42));
    REQUIRE_NOTHROW(foo(42));

    CHECK_THROWS(foo(0));
    CHECK_THROWS_AS(foo(0), std::invalid_argument);
    CHECK_THROWS_WITH(foo(0), "The answer is: 42");

    REQUIRE_THROWS(foo(0));
    REQUIRE_THROWS_AS(foo(0), std::invalid_argument);
    REQUIRE_THROWS_WITH(foo(0), "The answer is: 42");
}
```

与前面的示例一样，我们定义了`CATCH_CONFIG_MAIN`宏，添加了我们需要的包含文件，并定义了一个`TEST_CASE()`。我们还定义了一个`foo()`函数，如果`foo()`函数的输入无效，则会抛出异常。

在我们的测试用例中，我们首先使用有效的输入测试`foo()`函数。由于`foo()`函数没有输出（即函数返回`void`），我们通过使用`CHECK_NOTHROW()`宏来确保函数已经正确执行，确保没有抛出异常。值得注意的是，与`CHECK()`宏一样，`CHECK_NOTHROW()`宏有等效的`REQUIRE_NOTHROW()`，如果检查失败，将停止执行。

最后，我们确保`foo()`函数在其输入无效时抛出异常。有几种不同的方法可以做到这一点。`CHECK_THROWS()`宏只是确保抛出了异常。`CHECK_THROWS_AS()`宏确保不仅抛出了异常，而且异常是`std::runtime_error`类型。这两者都必须为测试通过。最后，`CHECK_THROWS_WITH()`宏确保抛出异常，并且异常的`what()`字符串返回与我们期望的异常匹配。与其他版本的`CHECK()`宏一样，每个宏也有`REQUIRE()`版本。

尽管 Catch2 库提供了宏，让您深入了解每种异常类型的具体细节，但应该注意，除非异常类型和字符串在您的 API 要求中明确定义，否则应该使用通用的`CHECK_THROWS()`宏。例如，规范中定义了`at()`函数在索引无效时始终返回`std::out_of_range`异常。在这种情况下，应该使用`CHECK_THROWS_AS()`宏来确保`at()`函数符合规范。规范中未指定此异常返回的字符串，因此应避免使用`CHECK_THROWS_WITH()`。这很重要，因为编写单元测试时常见的错误是编写过度规范的单元测试。过度规范的单元测试通常在被测试的代码更新时必须进行更新，这不仅成本高，而且容易出错。

单元测试应该足够详细，以确保单元按预期执行，但又足够通用，以确保对源代码的修改不需要更新单元测试本身，除非 API 的要求发生变化，从而产生一组能够长期使用的单元测试，同时仍然提供确保可靠性、稳定性、安全性甚至合规性所必需的测试。

一旦您有一组单元测试来验证每个单元是否按预期执行，下一步就是确保在修改代码时执行这些单元测试。这可以手动完成，也可以由**持续集成**（**CI**）服务器自动完成，例如 TravisCI；然而，当您决定这样做时，请确保单元测试返回正确的错误代码。在前面的例子中，当单元测试通过并打印简单的字符串表示所有测试都通过时，单元测试本身退出时使用了`EXIT_SUCCESS`。对于大多数 CI 来说，这已经足够了，但在某些情况下，让 Catch2 以易于解析的格式输出结果可能是有用的。

例如，考虑以下代码：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

TEST_CASE("the answer")
{
    CHECK(true);
}
```

让我们用以下方式运行：

```cpp
> ./recipe01_example01 -r xml
```

如果我们这样做，我们会得到以下结果：

![](img/181d1cbf-5814-44ae-8f95-b7577da6c8e5.png)

在前面的例子中，我们创建了一个简单的测试用例（与本配方中的第一个例子相同），并指示 Catch2 使用`-r xml`选项将测试结果输出为 XML。Catch2 有几种不同的输出格式，包括 XML 和 JSON。

除了输出格式之外，Catch2 还可以用来对我们的代码进行基准测试。例如，考虑以下代码片段：

```cpp
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch.hpp>

#include <vector>
#include <iostream>

TEST_CASE("the answer")
{
    std::vector<int> v{4, 8, 15, 16, 23, 42};

    BENCHMARK("sort vector") {
        std::sort(v.begin(), v.end());
    };
}
```

在上面的例子中，我们创建了一个简单的测试用例，对预定义的向量数字进行排序。然后我们在`BENCHMARK()`宏中对这个列表进行排序，当执行时会得到以下输出：

![](img/6ba0ee12-7624-4e52-897e-5182f5487f0e.png)

如前面的屏幕截图所示，Catch2 执行了该函数多次，平均花费`197`纳秒来对向量进行排序。`BENCHMARK()`宏对于确保代码不仅按预期执行并给出特定输入的正确输出，而且还确保代码在特定时间内执行非常有用。配合更详细的输出格式，比如 XML 或 JSON，这种类型的信息可以用来确保随着源代码的修改，生成的代码执行时间保持不变或更快。

为了更好地理解单元测试如何真正改进您的 C++，我们将用两个额外的例子来结束这个配方，这些例子旨在提供更真实的场景。

在第一个例子中，我们将创建一个**向量**。与 C++中的`std::vector`不同，它是一个动态的 C 风格数组，数学中的向量是*n*维空间中的一个点（在我们的例子中，我们将其限制为 2D 空间），其大小是点与原点（即 0,0）之间的距离。我们在示例中实现这个向量如下：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <cmath>
#include <climits>

class vector
{
    int m_x{};
    int m_y{};
```

除了通常的宏和包含之外，我们要做的第一件事是定义一个带有`x`和`y`坐标的类：

```cpp
public:

    vector() = default;

    vector(int x, int y) :
        m_x{x},
        m_y{y}
    { }

    auto x() const
    { return m_x; }

    auto y() const
    { return m_y; }

    void translate(const vector &p)
    {
        m_x += p.m_x;
        m_y += p.m_y;
    }

    auto magnitude()
    {
        auto a2 = m_x * m_x;
        auto b2 = m_y * m_y;

        return sqrt(a2 + b2);
    }
};
```

接下来，我们添加一些辅助函数和构造函数。默认构造函数创建一个没有方向或大小的向量，因为*x*和*y*被设置为原点。为了创建具有方向和大小的向量，我们还提供了另一个构造函数，允许您提供向量的初始*x*和*y*坐标。为了获取向量的方向，我们提供了返回向量*x*和*y*值的 getter。最后，我们提供了两个辅助函数。第一个辅助函数**translates**向量，在数学上是改变向量的*x*和*y*坐标的另一个术语。最后一个辅助函数返回向量的大小，即如果向量的*x*和*y*值用于构造三角形的斜边的长度（也就是说，我们必须使用勾股定理来计算向量的大小）。接下来，我们继续添加运算符，具体如下：

```cpp
bool operator== (const vector &p1, const vector &p2)
{ return p1.x() == p2.x() && p1.y() == p2.y(); }

bool operator!= (const vector &p1, const vector &p2)
{ return !(p1 == p2); }

constexpr const vector origin;
```

我们添加了一些等价运算符，用于检查两个向量是否相等。我们还定义了一个表示原点的向量，其*x*和*y*值都为 0。

为了测试这个向量，我们添加了以下测试：

```cpp
TEST_CASE("default constructor")
{
    vector p;

    CHECK(p.x() == 0);
    CHECK(p.y() == 0);
}

TEST_CASE("origin")
{
    CHECK(vector{0, 0} == origin);
    CHECK(vector{1, 1} != origin);
}

TEST_CASE("translate")
{
    vector p{-4, -8};
    p.translate({46, 50});

    CHECK(p.x() == 42);
    CHECK(p.y() == 42);
}

TEST_CASE("magnitude")
{
    vector p(1, 1);
    CHECK(Approx(p.magnitude()).epsilon(0.1) == 1.4);
}

TEST_CASE("magnitude overflow")
{
    vector p(INT_MAX, INT_MAX);
    CHECK(p.magnitude() == 65536);
}
```

第一个测试确保默认构造的向量实际上是原点。我们的下一个测试确保我们的全局**origin**向量是原点。这很重要，因为我们不应该假设原点是默认构造的，也就是说，未来有人可能会意外地将原点更改为`0,0`之外的其他值。这个测试用例确保原点实际上是`0,0`，这样在未来，如果有人意外更改了这个值，这个测试就会失败。由于原点必须导致*x*和*y*都为 0，所以这个测试并没有过度规定。

接下来，我们测试 translate 和 magnitude 函数。在 magnitude 测试用例中，我们使用`Approx()`宏。这是因为返回的大小是一个浮点数，其大小和精度取决于硬件，并且与我们的测试无关。`Approx()`宏允许我们声明要验证`magnitude()`函数结果的精度级别，该函数使用`epsilon()`修饰符来实际声明精度。在这种情况下，我们只希望验证到小数点后一位。

最后一个测试用例用于演示这些函数的所有输入应该被测试。如果一个函数接受一个整数，那么应该测试所有有效的、无效的和极端的输入。在这种情况下，我们为*x*和*y*都传递了`INT_MAX`。结果的`magnitude()`函数没有提供有效的结果。这是因为计算大小的过程溢出了整数类型。这种类型的错误应该在代码中考虑到（也就是说，您应该检查可能的溢出并抛出异常），或者 API 的规范应该指出这些类型的问题（也就是说，C++规范可能会声明这种类型输入的结果是未定义的）。无论哪种方式，如果一个函数接受一个整数，那么所有可能的整数值都应该被测试，并且这个过程应该对所有输入类型重复。

这个测试的结果如下：

![](img/64400718-2c26-405a-9f0f-f9581dec0119.png)

如前面的屏幕截图所示，该单元测试未通过最后一个测试。如前所述，为了解决这个问题，magnitude 函数应该被更改为在发生溢出时抛出异常，找到防止溢出的方法，或者删除测试并声明这样的输入是未定义的。

在我们的最后一个例子中，我们将演示如何处理不返回值而是操作输入的函数。

让我们通过创建一个写入文件的类和另一个使用第一个类将字符串写入该文件的类来开始这个例子，如下所示：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <string>
#include <fstream>

class file
{
    std::fstream m_file{"test.txt", std::fstream::out};

public:

    void write(const std::string &str)
    {
        m_file.write(str.c_str(), str.length());
    }
};

class the_answer
{
public:

    the_answer(file &f)
    {
        f.write("The answer is: 42\n");
    }
};
```

如前面的代码所示，第一个类写入一个名为`test.txt`的文件，而第二个类将第一个类作为输入，并使用它来向文件中写入一个字符串。

我们测试第二个类如下：

```cpp
TEST_CASE("the answer")
{
    file f;
    the_answer{f};
}
```

前面测试的问题在于我们没有任何`CHECK()`宏。这是因为除了`CHECK_NOTHROW()`之外，我们没有任何需要检查的东西。在这个测试中，我们测试以确保`the_answer{}`类调用`file{}`类和`write()`函数正确。我们可以打开`test.txt`文件并检查它是否用正确的字符串写入，但这是很多工作。这种类型的检查也会过度指定，因为我们不是在测试`file{}`类，我们只是在测试`the_answer{}`类。如果将来我们决定`file{}`类应该写入网络文件而不是磁盘上的文件，单元测试将不得不改变。

为了克服这个问题，我们可以利用一个叫做**mocking**的概念。`Mock`类是一个假装是输入类的类，为单元测试提供了**seams**，允许单元测试验证测试的结果。这与`Stub`不同，后者提供了虚假的输入。不幸的是，与其他语言相比，C++对 mocking 的支持并不好。辅助库，如 GoogleMock，试图解决这个问题，但需要所有可 mock 的类都包含一个 vTable（即继承纯虚拟基类）并在你的代码中定义每个可 mock 的类两次（一次在你的代码中，一次在你的测试中，使用 Google 定义的一组 API）。这远非最佳选择。像 Hippomocks 这样的库试图解决这些问题，但需要一些 vTable 黑魔法，只能在某些环境中工作，并且当出现问题时几乎不可能进行调试。尽管 Hippomocks 可能是最好的选择之一（即直到 C++启用本地 mocking），但以下示例是使用标准 C++进行 mocking 的另一种方法，唯一的缺点是冗长：

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <string>
#include <fstream>

class file
{
    std::fstream m_file{"test.txt", std::fstream::out};

public:
    VIRTUAL ~file() = default;

    VIRTUAL void write(const std::string &str)
    {
        m_file.write(str.c_str(), str.length());
    }
};

class the_answer
{
public:
    the_answer(file &f)
    {
        f.write("The answer is: 42\n");
    }
};
```

与我们之前的示例一样，我们创建了两个类。第一个类写入一个文件，而第二个类使用第一个类向该文件写入一个字符串。不同之处在于我们添加了`VIRTUAL`宏。当代码编译到我们的应用程序中时，`VIRTUAL`被设置为空，这意味着它被编译器从代码中移除。然而，当代码在我们的测试中编译时，它被设置为`virtual`，这告诉编译器给类一个 vTable。由于这只在我们的测试期间完成，所以额外的开销是可以接受的。

现在我们的类在我们的测试用例中支持继承，我们可以创建我们的`file{}`类的一个子类版本如下：

```cpp
class mock_file : public file
{
public:
    void write(const std::string &str)
    {
        if (str == "The answer is: 42\n") {
            passed = true;
        }
        else {
            passed = false;
        }
    }

    bool passed{};
};
```

前面的类定义了我们的 mock。我们的 mock 不是写入文件，而是检查特定的字符串是否被写入我们的假文件，并根据测试的结果设置一个全局变量为`true`或`false`。

然后我们可以测试我们的`the_answer{}`类如下：

```cpp
TEST_CASE("the answer")
{
    mock_file f;
    REQUIRE(f.passed == false);

    f.write("The answer is not: 43\n");
    REQUIRE(f.passed == false);

    the_answer{f};
    CHECK(f.passed);
}
```

当执行此操作时，我们会得到以下结果：

![](img/289a2554-23b1-4e41-9c45-6bc8b34eb163.png)

如前面的屏幕截图所示，我们现在可以检查我们的类是否按预期写入文件。值得注意的是，我们使用`REQUIRE()`宏来确保在执行我们的测试之前，mock 处于`false`状态。这确保了如果我们的实际测试被注册为通过，那么它确实已经通过，而不是因为我们测试逻辑中的错误而被注册为通过。

# 使用 ASAN，地址消毒剂

在这个示例中，我们将学习如何利用谷歌的**地址消毒剂**（**ASAN**）——这是一个动态分析工具——来检查代码中的内存损坏错误。这个示例很重要，因为它提供了一种简单的方法来确保你的代码既可靠又稳定，而对你的构建系统的更改数量很少。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有编译和执行本食谱中示例所需的适当工具。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何操作...

按照以下步骤执行该食谱：

1.  从新的终端运行以下命令以下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter07
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake -DCMAKE_BUILD_TYPE=ASAN ..
> make recipe02_examples
```

1.  编译源代码后，可以通过运行以下命令执行本食谱中的每个示例：

```cpp
> ./recipe02_example01
...

> ./recipe02_example02
...

> ./recipe02_example03
...

> ./recipe02_example04
...

> ./recipe02_example05
...
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本食谱中所教授的课程的关系。

# 它是如何工作的...

Google 的地址消毒剂是对 GCC 和 LLVM 编译器的一组修改，以及一组必须在测试时链接到应用程序中的库。为了实现这一点，我们在编译用于测试的代码时必须添加以下编译器标志（但不要将这些标志添加到生产版本中）：

```cpp
-fsanitize=address 
-fno-optimize-sibling-calls 
-fsanitize-address-use-after-scope 
-fno-omit-frame-pointer 
-g -O1
```

这里需要特别注意的最重要的标志是`-fsanitize=address`标志，它告诉编译器启用 ASAN。其余的标志是卫生间所需的，最值得注意的标志是`-g`和`-01`。`-g`标志启用调试，`-O1`标志将优化级别设置为 1，以提供一些性能改进。请注意，一旦启用 ASAN 工具，编译器将自动尝试链接到 ASAN 库，这些库必须存在于您的计算机上。

为了演示这个消毒剂是如何工作的，让我们看几个例子。

# 内存泄漏错误

`AddressSanitizer`是一种动态分析工具，旨在识别内存损坏错误。它类似于 Valgrind，但直接内置到您的可执行文件中。最容易用一个示例来演示这一点（也是最常见的错误类型之一）是内存泄漏，如下所示：

```cpp
int main(void)
{
    new int;
}
```

这导致以下输出：

![](img/4cdecfde-c17e-47ba-b109-8dd637af2a5d.png)

在上面的示例中，我们在程序中使用`new`运算符分配了一个整数，但在退出程序之前我们将永远不会释放这个分配的内存。ASAN 工具能够检测到这个问题，并在应用程序完成执行时输出错误。

# 内存两次删除

检测内存泄漏的能力非常有帮助，但这并不是 ASAN 能够检测到的唯一类型的错误。另一种常见的错误类型是多次删除内存。例如，考虑以下代码片段：

```cpp
int main(void)
{
    auto p = new int;
    delete p;

    delete p;
}
```

执行后，我们看到以下输出：

![](img/01d887f8-4fcc-4efc-8691-67831ec8b13a.png)

在上面的示例中，我们使用`new`运算符分配了一个整数，然后使用删除运算符删除了该整数。由于先前分配的内存的指针仍然在我们的`p`变量中，我们可以再次删除它，这是我们在退出程序之前所做的。在某些系统上，这将生成一个分段错误，因为这是未定义的行为。ASAN 工具能够检测到这个问题，并输出一个错误消息，指出发生了`double-free`错误。

# 访问无效内存

另一种错误类型是尝试访问从未分配的内存。这通常是由代码尝试对空指针进行解引用引起的，但也可能发生在指针损坏时，如下所示：

```cpp
int main(void)
{
    int *p = (int *)42;
    *p = 0;
}
```

这导致以下输出：

![](img/ebd30496-bc91-49e1-b20c-1e5f580db297.png)

在前面的示例中，我们创建了一个指向整数的指针，然后为它提供了一个损坏的值`42`（这不是一个有效的指针）。然后我们尝试对损坏的指针进行解引用，结果导致分段错误。应该注意的是，ASAN 工具能够检测到这个问题，但它无法提供任何有用的信息。这是因为 ASAN 工具是一个库，它钩入内存分配例程，跟踪每个分配以及分配的使用方式。如果一个分配从未发生过，它将不会有任何关于发生了什么的信息，除了典型的 Unix 信号处理程序已经提供的信息，其他动态分析工具，比如 Valgrind，更适合处理这些情况。

# 在删除后使用内存

为了进一步演示地址消毒剂的工作原理，让我们看看以下示例：

```cpp
int main(void)
{
    auto p = new int;
    delete p;

    *p = 0;
}
```

当我们执行这个时，我们会看到以下内容：

![](img/c01185c7-a10c-4464-be43-c60816cfcd63.png)

前面的示例分配了一个整数，然后删除了这个整数。然后我们尝试使用先前删除的内存。由于这个内存位置最初是分配的，ASAN 已经缓存了地址。当对先前删除的内存进行解引用时，ASAN 能够检测到这个问题，作为`heap-use-after-free`错误。它之所以能够检测到这个问题，是因为这块内存先前被分配过。

# 删除从未分配的内存

最后一个例子，让我们看看以下内容：

```cpp
int main(void)
{
    int *p = (int *)42;
    delete p;
}
```

这导致了以下结果：

![](img/ed7e5106-c3d4-478f-8085-45a6ed4f62fb.png)

在前面的示例中，我们创建了一个指向整数的指针，然后再次为它提供了一个损坏的值。与我们之前的示例不同，在这个示例中，我们尝试删除这个损坏的指针，结果导致分段错误。再一次，ASAN 能够检测到这个问题，但由于从未发生过分配，它没有任何有用的信息。

应该注意的是，C++核心指南——这是一个现代 C++的编码标准——在防止我们之前描述的问题类型方面非常有帮助。具体来说，核心指南规定`new()`、`delete()`、`malloc()`、`free()`和其他函数不应该直接使用，而应该使用`std::unique_ptr`和`std::shared_ptr`来进行*所有内存分配*。这些 API 会自动为您分配和释放内存。如果我们再次看一下前面的示例，很容易看出，使用这些 API 来分配内存而不是手动使用`new()`和`delete()`可以防止这些问题发生，因为大多数前面的示例都与无效使用`new()`和`delete()`有关。

# 使用 UBSAN，未定义行为消毒剂

在这个配方中，我们将学习如何在我们的 C++应用程序中使用 UBSAN 动态分析工具，它能够检测未定义的行为。在我们的应用程序中可能会引入许多不同类型的错误，未定义的行为很可能是最常见的类型，因为 C 和 C++规范定义了几种可能发生未定义行为的情况。

这个配方很重要，因为它将教会你如何启用这个简单的功能，以及它如何在你的应用程序中使用。

# 准备工作

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有适当的工具来编译和执行本配方中的示例。完成后，打开一个新的终端。我们将使用这个终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤进行配方：

1.  从一个新的终端，运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter07
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake -DCMAKE_BUILD_TYPE=UBSAN .
> make recipe03_examples
```

1.  源代码编译后，可以通过运行以下命令来执行本示例中的每个示例：

```cpp
> ./recipe03_example01
Floating point exception (core dumped)

> ./recipe03_example02
Segmentation fault (core dumped)

> ./recipe03_example03
Segmentation fault (core dumped)

> ./recipe03_example04

```

在下一节中，我们将逐个讲解这些示例，并解释每个示例程序的作用以及它与本示例中所教授的课程的关系。

# 工作原理...

UBSAN 工具能够检测到几种类型的未定义行为，包括以下内容：

+   越界错误

+   浮点错误

+   除零

+   整数溢出

+   空指针解引用

+   缺少返回值

+   有符号/无符号转换错误

+   不可达代码

在这个示例中，我们将看一些这样的例子，但首先，我们必须在我们的应用程序中启用 UBSAN 工具。为此，我们必须在应用程序的构建系统中启用以下标志：

```cpp
-fsanitize=undefined
```

这个标志将告诉 GCC 或 LLVM 使用 UBSAN 工具，它会向我们的应用程序添加额外的逻辑，并链接到 UBSAN 库。值得注意的是，UBSAN 工具的功能会随着时间的推移而增强。因此，GCC 和 LLVM 对 UBSAN 的支持水平不同。为了充分利用这个工具，你的应用程序应该同时针对 GCC 和 LLVM 进行编译，并且应该尽可能使用最新的编译器。

# 除零错误

使用 UBSAN 最容易演示的一个例子是除零错误，如下所示：

```cpp
int main(void)
{
    int n = 42;
    int d = 0;

    auto f = n/d;
}
```

当运行时，我们看到以下内容：

![](img/698ed489-e92b-4080-a0dc-fb224466ddf7.png)

在上面的示例中，我们创建了两个整数（一个分子和一个分母），分母设置为`0`。然后我们对分子和分母进行除法运算，导致除零错误，UBSAN 检测到并在程序崩溃时输出。

# 空指针解引用

在 C++中更常见的问题类型是空指针解引用，如下所示：

```cpp
int main(void)
{
    int *p = 0;
    *p = 42;
}
```

这导致了以下结果：

![](img/61d56d5b-161b-470f-8181-68dafa5ab7ab.png)

在上面的示例中，我们创建了一个指向整数的指针，并将其设置为`0`（即`NULL`指针）。然后我们对`NULL`指针进行解引用并设置其值，导致分段错误，UBSAN 能够检测到程序崩溃。

# 越界错误

前面的两个示例都可以使用 Unix 信号处理程序来检测。在下一个示例中，我们将访问一个超出边界的数组，这在 C++规范中是未定义的，而且更难以检测：

```cpp
int main(void)
{
    int numbers[] = {4, 8, 15, 16, 23, 42};
    numbers[10] = 0;
}
```

执行时，我们得到以下结果：

![](img/19eaa37d-90b8-4910-bc6a-bc96ec98f7dd.png)

如上面的示例所示，我们创建了一个有 6 个元素的数组，然后尝试访问数组中的第 10 个元素，这个元素并不存在。尝试访问数组中的这个元素并不一定会生成分段错误。不管怎样，UBSAN 能够检测到这种类型的错误，并在退出时将问题输出到`stderr`。

# 溢出错误

最后，我们还可以检测有符号整数溢出错误，这在 C++中是未定义的，但极不可能导致崩溃，而是会导致程序进入一个损坏的状态（通常产生无限循环、越界错误等）。考虑以下代码：

```cpp
#include <climits>

int main(void)
{
    int i = INT_MAX;
    i++;
}
```

这导致了以下结果：

![](img/0db50c0b-0249-4600-9a72-c62b3fc591b0.png)

如上面的示例所示，我们创建了一个整数，并将其设置为最大值。然后我们尝试增加这个整数，这通常会翻转整数的符号，这是 UBSAN 能够检测到的错误。

# 使用#ifndef NDEBUG 条件执行额外检查

在这个示例中，我们将学习如何利用`NDEBUG`宏，它代表*no debug*。这个示例很重要，因为大多数构建系统在编译*发布*或*生产*版本时会自动定义这个宏，这可以用来在创建这样的构建时禁用调试逻辑。

# 准备就绪

开始之前，请确保满足所有技术要求，包括安装 Ubuntu 18.04 或更高版本，并在终端窗口中运行以下命令：

```cpp
> sudo apt-get install build-essential git cmake
```

这将确保您的操作系统具有正确的工具来编译和执行本配方中的示例。完成后，打开一个新的终端。我们将使用此终端来下载、编译和运行我们的示例。

# 如何做...

按照以下步骤来完成这个配方：

1.  从新的终端运行以下命令来下载源代码：

```cpp
> cd ~/
> git clone https://github.com/PacktPublishing/Advanced-CPP-CookBook.git
> cd Advanced-CPP-CookBook/chapter07
```

1.  要编译源代码，请运行以下命令：

```cpp
> cmake .
> make recipe04_examples
```

1.  一旦源代码编译完成，您可以通过运行以下命令来执行本配方中的每个示例：

```cpp
> ./recipe04_example01
The answer is: 42

> ./recipe04_example02
recipe04_example02: /home/user/book/chapter07/recipe04.cpp:45: int main(): Assertion `42 == 0' failed.
Aborted (core dumped)
```

在下一节中，我们将逐个介绍这些示例，并解释每个示例程序的作用以及它与本配方中所教授的课程的关系。

# 工作原理...

`NDEBUG`宏源自 C 语言，用于更改`assert()`函数的行为。`assert()`函数可以编写如下：

```cpp
void __assert(int val, const char *str)
{
    if (val == 0) {
        fprintf(stderr, "Assertion '%s' failed.\n", str);
        abort();
    }
}

#ifndef NDEBUG
    #define assert(a) __assert(a, #a)
#else
    #define assert(a)
#endif 
```

如前面的代码所示，如果`__assert()`函数得到一个求值为`false`的布尔值（在 C 中，这是一个等于`0`的整数），则会向`stderr`输出错误消息，并中止应用程序。然后使用`NDEBUG`宏来确定`assert()`函数是否存在，如果应用程序处于发布模式，则会删除所有断言逻辑，从而减小应用程序的大小。在使用 CMake 时，我们可以使用以下命令启用`NDEBUG`标志：

```cpp
> cmake -DCMAKE_BUILD_TYPE=Release ..
```

这将自动定义`NDEBUG`宏并启用优化。要防止定义此宏，我们可以做相反的操作：

```cpp
> cmake -DCMAKE_BUILD_TYPE=Debug ..
```

上面的 CMake 代码将*不*定义`NDEBUG`宏，而是启用调试，并禁用大多数优化（尽管这取决于编译器）。

在我们自己的代码中，`assert`宏可以如下使用：

```cpp
#include <cassert>

int main(void)
{
    assert(42 == 0);
}
```

结果如下：

![](img/285a3fe5-641d-4c56-8521-9fe0e4ffbceb.png)

如前面的示例所示，我们创建了一个应用程序，该应用程序使用`assert()`宏来检查一个错误的语句，结果是应用程序中止。

尽管`NDEBUG`宏被`assert()`函数使用，但您也可以自己使用它，如下所示：

```cpp
int main(void)
{
#ifndef NDEBUG
    std::cout << "The answer is: 42\n";
#endif
}
```

如前面的代码所示，如果应用程序未以*release*模式编译（即在编译时未在命令行上定义`NDEBUG`宏），则应用程序将输出到`stdout`。您可以在整个代码中使用相同的逻辑来创建自己的调试宏和函数，以确保在*release*模式下删除调试逻辑，从而可以根据需要添加任意数量的调试逻辑，而无需修改交付给客户的最终应用程序。
