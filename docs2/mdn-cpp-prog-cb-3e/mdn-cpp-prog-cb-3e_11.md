

# 探索测试框架

测试代码是软件开发的重要部分。尽管 C++ 标准中没有对测试的支持，但存在大量用于单元测试 C++ 代码的框架。本章的目的是让您开始使用几个现代且广泛使用的测试框架，这些框架使您能够编写可移植的测试代码。本章将涵盖的框架是 **Boost.Test**、**Google Test** 和 **Catch2**。

本章包括以下食谱：

+   开始使用 Boost.Test

+   使用 Boost.Test 编写和调用测试

+   使用 Boost.Test 进行断言

+   使用 Boost.Test 的测试夹具

+   使用 Boost.Test 控制输出

+   开始使用 Google Test

+   使用 Google Test 编写和调用测试

+   使用 Google Test 进行断言

+   使用 Google Test 的测试夹具

+   使用 Google Test 控制输出

+   开始使用 Catch2

+   使用 Catch2 编写和调用测试

+   使用 Catch2 进行断言

+   使用 Catch2 控制输出

这三个框架被选择是因为它们广泛的使用、丰富的功能、易于编写和执行测试、可扩展性和可定制性。以下表格展示了这三个库功能的简要比较：

| **特性** | **Boost.Test** | **Google Test** | **Catch2 (v3)** |
| --- | --- | --- | --- |
| 容易安装 | 是 | 是 | 是 |
| 仅头文件 | 是 | 否 | 否 |
| 编译库 | 是 | 是 | 是 |
| 容易编写测试 | 是 | 是 | 是 |
| 自动测试注册 | 是 | 是 | 是 |
| 支持测试套件 | 是 | 是 | 否（间接通过标签） |
| 支持夹具 | 是（设置/清理） | 是（设置/清理） | 是（多种方式） |
| 丰富的断言集 | 是 | 是 | 是 |
| 非致命断言 | 是 | 是 | 是 |
| 多种输出格式 | 是（包括 HRF、XML） | 是（包括 HRF、XML） | 是（包括 HRF、XML） |
| 测试执行过滤 | 是 | 是 | 是 |
| 许可证 | Boost | Apache 2.0 | Boost |

表 11.1：Boost.Test、Google Test 和 Catch2 功能比较

所有这些功能将在每个框架的详细讨论中介绍。本章具有对称结构，有 4 个 5 个食谱专门针对每个测试框架。首先考虑的框架是 Boost.Test。

# 开始使用 Boost.Test

**Boost.Test** 是最古老且最受欢迎的 C++ 测试框架之一。它提供了一套易于使用的 API，用于编写测试并将它们组织成测试用例和测试套件。它对断言、异常处理、夹具和其他测试框架所需的重要功能提供了良好的支持。

在接下来的几个食谱中，我们将探索它最重要的功能，这些功能使您能够编写单元测试。在这个食谱中，我们将看到如何安装框架并创建一个简单的测试项目。

## 准备工作

Boost.Test 框架有一个基于宏的 API。虽然你只需要使用提供的宏来编写测试，但如果想很好地使用该框架，建议你了解宏。

## 如何实现...

为了设置你的环境以使用 Boost.Test，请执行以下操作：

1.  从 [`www.boost.org/`](http://www.boost.org/) 下载最新的 Boost 库版本。

1.  解压存档的内容。

1.  使用提供的工具和脚本构建库，以便使用静态库或共享库变体。如果你计划使用库的头文件版本，这一步是不必要的。

在 Linux 系统上，也可以使用包管理工具安装库。例如，在 Ubuntu 上，你可以使用 **app-get** 安装包含 Boost.Test 库的 `libboost-test-dev` 包，如下所示：

```cpp
sudo apt-get install libboost-test-dev 
```

建议你查阅库的在线文档，了解在不同系统上的安装步骤。

要使用 Boost.Test 库的头文件版本创建你的第一个测试程序，请执行以下操作：

1.  创建一个新的、空的 C++ 项目。

1.  根据你使用的开发环境进行必要的设置，以便将 Boost 的 `main` 文件夹对项目可用，以便包含头文件。

1.  向项目中添加一个新的源文件，内容如下：

    ```cpp
    #define BOOST_TEST_MODULE My first test module
    #include <boost/test/included/unit_test.hpp>
    BOOST_AUTO_TEST_CASE(first_test_function)
    {
      int a = 42;
      BOOST_TEST(a > 0);
    } 
    ```

1.  如果你想要链接到共享库版本，那么还需要定义 `BOOST_TEST_DYN_LINK` 宏。

1.  构建并运行项目。

## 它是如何工作的...

Boost.Test 库可以与其它 Boost 库一起从 [`www.boost.org/`](http://www.boost.org/) 下载。在这本书的这一版中，我使用了 1.83 版本，但讨论的这些功能可能适用于许多未来的版本。`Test` 库有三个变体：

+   **单个头文件**：这使你能够在不构建库的情况下编写测试程序；你只需要包含一个头文件。它的限制是，你只能为模块有一个翻译单元；然而，你仍然可以将模块分割成多个头文件，以便将不同的测试套件分开到不同的文件中。

+   **静态库**：这使你能够将模块分割到不同的翻译单元中，但库需要首先作为一个静态库来构建。

+   **共享库**：这提供了与静态库相同的场景。然而，它有一个优点，即对于具有许多测试模块的程序，这个库只需链接一次，而不是每个模块都链接一次，从而减小了二进制文件的大小。但是，在这种情况下，共享库必须在运行时可用。

为了简单起见，我们将在这本书中使用单个头文件变体。在静态库和共享库变体的情况下，你需要构建库。下载的存档包含构建库的脚本。然而，具体的步骤取决于平台和编译器；它们将不会在此处介绍，但可以在网上找到。

为了使用这个库，你需要理解几个术语和概念：

+   **测试模块**是一个执行测试的程序。有两种类型的模块：**单文件**（当你使用单头文件变体时）和**多文件**（当你使用静态或共享变体时）。

+   **测试断言**是测试模块检查的条件。

+   **测试用例**是一组一个或多个测试断言，它由测试模块独立执行和监控，以便如果它失败或泄漏未捕获的异常，其他测试的执行将不会停止。

+   **测试套件**是一组一个或多个测试用例或测试套件。

+   **测试单元**是一个测试用例或测试套件。

+   **测试树**是测试单元的分层结构。在这个结构中，测试用例是叶子节点，测试套件是非叶子节点。

+   **测试执行器**是一个组件，给定一个测试树，执行必要的初始化、测试执行和结果报告。

+   **测试报告**是测试执行器从执行测试产生的报告。

+   **测试日志**是记录测试模块执行期间发生的所有事件的记录。

+   **测试设置**是负责初始化框架、构建测试树和单个测试用例设置的测试模块的一部分。

+   **测试清理**是负责清理操作的测试模块的一部分。

+   **测试夹具**是一对设置和清理操作，用于多个测试单元以避免重复代码。

定义了这些概念之后，就可以解释前面列出的示例代码：

1.  `#define BOOST_TEST_MODULE My first test module`定义了一个模块初始化的占位符并为主测试套件设置了一个名称。这必须在包含任何库头文件之前定义。

1.  `#include <boost/test/included/unit_test.hpp>`包含单头文件库，该库包含所有其他必要的头文件。

1.  `BOOST_AUTO_TEST_CASE(first_test_function)`声明一个无参数的测试用例（`first_test_function`）并将其自动注册为包含在测试树中，作为封装测试套件的一部分。在这个例子中，测试套件是由`BOOST_TEST_MODULE`定义的主测试套件。

1.  `BOOST_TEST(true);`执行一个测试断言。

执行此测试模块的输出如下：

```cpp
Running 1 test case...
*** No errors detected 
```

## 还有更多...

如果你不想库生成`main()`函数但想自己编写，那么在包含任何库头文件之前，你需要定义几个额外的宏 - `BOOST_TEST_NO_MAIN`和`BOOST_TEST_ALTERNATIVE_INIT_API`。然后，在你提供的`main()`函数中，通过提供默认的初始化函数`init_unit_test()`作为参数，调用默认的测试执行器`unit_test_main()`，如下代码片段所示：

```cpp
#define BOOST_TEST_MODULE My first test module
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
BOOST_AUTO_TEST_CASE(first_test_function)
{
  int a = 42;
  BOOST_TEST(a > 0);
}
int main(int argc, char* argv[])
{
  return boost::unit_test::unit_test_main(init_unit_test, argc, argv);
} 
```

还可以自定义测试运行器的初始化函数。在这种情况下，必须删除 `BOOST_TEST_MODULE` 宏的定义，并编写一个不接受任何参数并返回 `bool` 值的初始化函数：

```cpp
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
#include <iostream>
BOOST_AUTO_TEST_CASE(first_test_function)
{
  int a = 42;
  BOOST_TEST(a > 0);
}
bool custom_init_unit_test()
{
  std::cout << "test runner custom init\n";
  return true;
}
int main(int argc, char* argv[])
{
  return boost::unit_test::unit_test_main(
    custom_init_unit_test, argc, argv);
} 
```

可以自定义初始化函数，而不必自己编写 `main()` 函数。在这种情况下，不应定义 `BOOST_TEST_NO_MAIN` 宏，并且初始化函数应命名为 `init_unit_test()`。

## 参见

+   *使用 Boost.Test 编写和调用测试*，以了解如何使用 Boost.Test 库的单头版本创建测试套件和测试用例，以及如何运行测试

# 使用 Boost.Test 编写和调用测试

库提供了自动和手动两种方式来注册测试用例和测试套件，以便测试运行器执行。自动注册是最简单的方式，因为它允许你仅通过声明测试单元来构建测试树。在本食谱中，我们将了解如何使用库的单头版本创建测试套件和测试用例，以及如何运行测试。

## 准备工作

为了说明测试套件和测试用例的创建，我们将使用以下类，它代表一个三维点。此实现包含访问点属性的方法、比较运算符、流输出运算符以及修改点位置的方法：

```cpp
class point3d
{
  int x_;
  int y_;
  int z_;
public:
  point3d(int const x = 0, 
          int const y = 0, 
          int const z = 0):x_(x), y_(y), z_(z) {}
  int x() const { return x_; }
  point3d& x(int const x) { x_ = x; return *this; }
  int y() const { return y_; }
  point3d& y(int const y) { y_ = y; return *this; }
  int z() const { return z_; }
  point3d& z(int const z) { z_ = z; return *this; }
  bool operator==(point3d const & pt) const
  {
    return x_ == pt.x_ && y_ == pt.y_ && z_ == pt.z_;
  }
  bool operator!=(point3d const & pt) const
  {
    return !(*this == pt);
  }
  bool operator<(point3d const & pt) const
  {
    return x_ < pt.x_ || y_ < pt.y_ || z_ < pt.z_;
  }
  friend std::ostream& operator<<(std::ostream& stream, 
                                  point3d const & pt)
  {
    stream << "(" << pt.x_ << "," << pt.y_ << "," << pt.z_ << ")";
    return stream;
  }
  void offset(int const offsetx, int const offsety, int const offsetz)
 {
    x_ += offsetx;
    y_ += offsety;
    z_ += offsetz;
  }
  static point3d origin() { return point3d{}; }
}; 
```

在继续之前，请注意，本食谱中的测试用例故意包含错误测试，以便它们产生失败。

## 如何操作...

使用以下宏来创建测试单元：

+   要创建测试套件，使用 `BOOST_AUTO_TEST_SUITE(name)` 和 `BOOST_AUTO_TEST_SUITE_END()`：

    ```cpp
    BOOST_AUTO_TEST_SUITE(test_construction)
    // test cases 
    BOOST_AUTO_TEST_SUITE_END() 
    ```

+   要创建测试用例，使用 `BOOST_AUTO_TEST_CASE(name)`。测试用例定义在 `BOOST_AUTO_TEST_SUITE(name)` 和 `BOOST_AUTO_TEST_SUITE_END()` 之间，如下面的代码片段所示：

    ```cpp
    BOOST_AUTO_TEST_CASE(test_constructor)
    {
      auto p = point3d{ 1,2,3 };
      BOOST_TEST(p.x() == 1);
      BOOST_TEST(p.y() == 2);
      BOOST_TEST(p.z() == 4); // will fail
    }
    BOOST_AUTO_TEST_CASE(test_origin)
    {
      auto p = point3d::origin();
      BOOST_TEST(p.x() == 0);
      BOOST_TEST(p.y() == 0);
      BOOST_TEST(p.z() == 0);
    } 
    ```

+   要创建嵌套测试套件，在另一个测试套件内部定义一个测试套件：

    ```cpp
    BOOST_AUTO_TEST_SUITE(test_operations)
    BOOST_AUTO_TEST_SUITE(test_methods)
    BOOST_AUTO_TEST_CASE(test_offset)
    {
      auto p = point3d{ 1,2,3 };
      p.offset(1, 1, 1);
      BOOST_TEST(p.x() == 2);
      BOOST_TEST(p.y() == 3);
      BOOST_TEST(p.z() == 3); // will fail
    }
    BOOST_AUTO_TEST_SUITE_END()
    BOOST_AUTO_TEST_SUITE_END() 
    ```

+   要向测试单元添加装饰器，向测试单元的宏添加一个额外的参数。装饰器可以包括描述、标签、先决条件、依赖项、固定装置等。请参考以下代码片段，它说明了这一点：

    ```cpp
    BOOST_AUTO_TEST_SUITE(test_operations)
    BOOST_AUTO_TEST_SUITE(test_operators)
    BOOST_AUTO_TEST_CASE(
      test_equal, 
      *boost::unit_test::description("test operator==")
      *boost::unit_test::label("opeq"))
    {
      auto p1 = point3d{ 1,2,3 };
      auto p2 = point3d{ 1,2,3 };
      auto p3 = point3d{ 3,2,1 };
      BOOST_TEST(p1 == p2);
      BOOST_TEST(p1 == p3); // will fail
    }
    BOOST_AUTO_TEST_CASE(
      test_not_equal, 
      *boost::unit_test::description("test operator!=")
      *boost::unit_test::label("opeq")
      *boost::unit_test::depends_on(
        "test_operations/test_operators/test_equal"))
    {
      auto p1 = point3d{ 1,2,3 };
      auto p2 = point3d{ 3,2,1 };
      BOOST_TEST(p1 != p2);
    }
    BOOST_AUTO_TEST_CASE(test_less)
    {
      auto p1 = point3d{ 1,2,3 };
      auto p2 = point3d{ 1,2,3 };
      auto p3 = point3d{ 3,2,1 };
      BOOST_TEST(!(p1 < p2));
      BOOST_TEST(p1 < p3);
    }
    BOOST_AUTO_TEST_SUITE_END()
    BOOST_AUTO_TEST_SUITE_END() 
    ```

要执行测试，执行以下操作（请注意，命令行是针对 Windows 的，但应该很容易替换为针对 Linux 或 macOS 的命令行）：

+   要执行整个测试树，不带任何参数运行程序（测试模块）：

    ```cpp
    chapter11bt_02.exe
    Running 6 test cases...
    f:/chapter11bt_02/main.cpp(12): error: in "test_construction/test_
    constructor": check p.z() == 4 has failed [3 != 4]
    f:/chapter11bt_02/main.cpp(35): error: in "test_operations/test_
    methods/test_offset": check p.z() == 3 has failed [4 != 3]
    f:/chapter11bt_02/main.cpp(55): error: in "test_operations/test_
    operators/test_equal": check p1 == p3 has failed [(1,2,3) != 
    (3,2,1)]
    *** 3 failures are detected in the test module "Testing point 3d" 
    ```

+   要执行单个测试套件，使用参数 `run_test` 运行程序，指定测试套件的路径：

    ```cpp
    chapter11bt_02.exe --run_test=test_construction
    Running 2 test cases...
    f:/chapter11bt_02/main.cpp(12): error: in "test_construction/test_
    constructor": check p.z() == 4 has failed [3 != 4]
    *** 1 failure is detected in the test module "Testing point 3d" 
    ```

+   要执行单个测试用例，使用参数 `run_test` 运行程序，指定测试用例的路径：

    ```cpp
    chapter11bt_02.exe --run_test=test_construction/test_origin
    Running 1 test case...
    *** No errors detected 
    ```

+   要执行在相同标签下定义的多个测试套件和测试用例，使用参数 `run_test` 运行程序，指定以 `@` 为前缀的标签名称：

    ```cpp
    chapter11bt_02.exe --run_test=@opeq
    Running 2 test cases...
    f:/chapter11bt_02/main.cpp(56): error: in "test_operations/test_
    operators/test_equal": check p1 == p3 has failed [(1,2,3) != 
    (3,2,1)]
    *** 1 failure is detected in the test module "Testing point 3d" 
    ```

## 它是如何工作的...

测试树是一系列测试用例和测试套件的层次结构，还包括了固定装置和额外的依赖项。测试套件可以包含一个或多个测试用例以及其他嵌套的测试套件。在相同文件或不同文件中，测试套件可以多次停止和重新启动，类似于命名空间。测试套件的自动注册使用宏`BOOST_AUTO_TEST_SUITE`，它需要一个名称，以及`BOOST_AUTO_TEST_SUITE_END`。测试用例的自动注册使用`BOOST_AUTO_TEST_CASE`。测试单元（无论是用例还是套件）成为最近测试套件的成员。在文件作用域级别定义的测试单元成为主测试套件的成员——由`BOOST_TEST_MODULE`声明创建的隐式测试套件。

测试套件和测试用例都可以用一系列属性装饰，这些属性会影响测试模块执行期间如何处理测试单元。目前支持的装饰器如下：

+   `depends_on`: 这表示当前测试单元与指定测试单元之间的依赖关系。

+   `description`: 这提供了测试单元的语义描述。

+   `enabled` / `disabled`: 这些将测试单元的默认运行状态设置为`true`或`false`。

+   `enable_if<bool>`: 这根据编译时表达式的评估结果，将测试单元的默认运行状态设置为`true`或`false`。

+   `expected_failures`: 这表示测试单元的预期失败情况。

+   `fixture`: 这指定了一对函数（启动和清理），在执行测试单元之前和之后调用。

+   `label`: 使用这个，你可以将测试单元与一个标签关联起来。相同的标签可以用于多个测试单元，并且一个测试单元可以有多个标签。

+   `precondition`: 这将一个谓词与测试单元关联起来，在运行时用于确定测试单元的运行状态。

+   `timeout`: 指定单元测试的超时时间，以墙钟时间为准。如果测试持续时间超过指定的超时时间，则测试失败。

+   `tolerance`: 这个装饰器指定了装饰测试单元中 FTP 浮点类型的默认比较容差。

如果测试用例的执行导致未处理的异常，框架将捕获该异常并以失败状态终止测试用例的执行。然而，框架提供了几个宏来测试特定的代码是否引发或未引发异常。有关更多信息，请参阅下一道菜谱，*使用 Boost.Test 进行断言*。

组成模块测试树的测试单元可以完全或部分执行。在两种情况下，要执行测试单元，请执行（二进制）程序，该程序代表测试模块。要仅执行某些测试单元，请使用`--run_test`命令行选项（或`--t`如果您想使用更短的名字）。此选项允许您过滤测试单元并指定路径或标签。路径是一系列测试套件和/或测试用例名称的序列，例如`test_construction`或`test_operations`/`test_methods`/`test_offset`。标签是与`label`装饰器定义的名称，并在`run_test`参数前加`@`。此参数是可重复的，这意味着您可以在其上指定多个过滤器。

## 参见

+   *开始使用 Boost.Test*，了解如何安装 Boost.Test 框架以及如何创建一个简单的测试项目

+   *使用 Boost.Test 进行断言*，探索 Boost.Test 库中的丰富断言宏集

# 使用 Boost.Test 进行断言

测试用例包含一个或多个测试。Boost.Test 库提供了一系列以宏形式存在的 API 来编写测试。在前面的配方中，您已经了解了一些关于`BOOST_TEST`宏的内容，这是库中最重要且最广泛使用的宏。在本配方中，我们将更详细地讨论如何使用`BOOST_TEST`宏。

## 准备就绪

您现在应该熟悉编写测试套件和测试用例，这是我们前面讨论的主题。

## 如何做到这一点...

以下列表显示了执行测试的一些最常用 API：

+   `BOOST_TEST`以其纯形式用于大多数测试：

    ```cpp
    int a = 2, b = 4;
    BOOST_TEST(a == b);
    BOOST_TEST(4.201 == 4.200);
    std::string s1{ "sample" };
    std::string s2{ "text" };
    BOOST_TEST(s1 == s2, "not equal"); 
    ```

+   `BOOST_TEST`，与`tolerance()`操作符一起使用，用于指示浮点数比较的容差：

    ```cpp
    BOOST_TEST(4.201 == 4.200, boost::test_tools::tolerance(0.001)); 
    ```

+   `BOOST_TEST`，与`per_element()`操作符一起使用，用于执行容器（即使是不同类型）的元素级比较：

    ```cpp
    std::vector<int> v{ 1,2,3 };
    std::list<short> l{ 1,2,3 };
    BOOST_TEST(v == l, boost::test_tools::per_element()); 
    ```

+   `BOOST_TEST`，与三元运算符和逻辑`||`或`&&`的复合语句一起使用时，需要额外的括号：

    ```cpp
    BOOST_TEST((a > 0 ? true : false));
    BOOST_TEST((a > 2 && b < 5)); 
    ```

+   `BOOST_ERROR`用于无条件失败测试并在报告中生成消息。这相当于`BOOST_TEST(false, message)`：

    ```cpp
    BOOST_ERROR("this test will fail"); 
    ```

+   `BOOST_TEST_WARN`用于在测试失败时在报告中生成警告，而不会增加遇到的错误数量并停止测试用例的执行：

    ```cpp
    BOOST_TEST_WARN(a == 4, "something is not right"); 
    ```

+   `BOOST_TEST_REQUIRE`用于确保测试用例的先决条件得到满足；否则，将停止测试用例的执行：

    ```cpp
    BOOST_TEST_REQUIRE(a == 4, "this is critical"); 
    ```

+   `BOOST_FAIL`用于无条件停止测试用例的执行，增加遇到的错误数量，并在报告中生成消息。这相当于`BOOST_TEST_REQUIRE(false, message)`：

    ```cpp
    BOOST_FAIL("must be implemented"); 
    ```

+   `BOOST_IS_DEFINED`用于检查在运行时是否定义了特定的预处理器符号。它与`BOOST_TEST`一起用于执行验证和记录：

    ```cpp
    BOOST_TEST(BOOST_IS_DEFINED(UNICODE)); 
    ```

## 它是如何工作的...

该库定义了各种宏和操作符，用于执行测试断言。最常用的是 `BOOST_TEST`。此宏简单地评估一个表达式；如果失败，它会增加错误计数但继续执行测试用例。实际上它有三个变体：

+   `BOOST_TEST_CHECK` 与 `BOOST_TEST` 相同，用于执行检查，如前文所述。

+   `BOOST_TEST_WARN` 用于旨在提供信息的断言，而不会增加错误计数并停止测试用例的执行。

+   `BOOST_TEST_REQUIRE` 的目的是确保测试用例继续执行所需的先决条件得到满足。如果失败，此宏会增加错误计数并停止测试用例的执行。

测试宏的一般形式是 `BOOST_TEST(statement)`。此宏提供了丰富和灵活的报告功能。默认情况下，它不仅显示语句，还显示操作数的值，以便快速识别失败的原因。

然而，用户可以提供替代的失败描述；在这种情况下，消息将记录在测试报告中：

```cpp
BOOST_TEST(a == b);
// error: in "regular_tests": check a == b has failed [2 != 4]
BOOST_TEST(a == b, "not equal");
// error: in "regular_tests": not equal 
```

此宏还允许您通过特殊支持来控制比较过程：

+   第一个是浮点数比较，可以定义容差来测试相等性。

+   其次，它支持使用多种方法对容器进行比较：默认比较（使用重载的运算符 `==`）、逐元素比较和字典序比较（使用字典序）。逐元素比较允许按容器的前向迭代器顺序比较不同类型的容器（如 vector 和 list），同时考虑容器的尺寸（这意味着它首先测试尺寸，只有当它们相等时，才会继续比较元素）。

+   最后，它支持操作数的位比较。如果失败，框架会报告比较失败的位索引。

`BOOST_TEST` 宏确实有一些限制。它不能与使用逗号分隔的复合语句一起使用，因为此类语句会被预处理器或三元运算符拦截和处理，以及使用逻辑运算符 `||` 和 `&&` 的复合语句。后者的解决方案是使用另一对括号，如 `BOOST_TEST((statement))`。

有几个宏可用于测试在表达式评估过程中是否抛出了特定异常。在以下列表中，`<level>` 可以是 `CHECK`、`WARN` 或 `REQUIRE`：

+   `BOOST_<level>_NO_THROW(expr)` 检查 `expr` 表达式是否抛出异常。在 `expr` 的评估过程中抛出的任何异常都会被此断言捕获，并且不会传播到测试主体。如果发生任何异常，断言将失败。

+   `BOOST_<level>_THROW(expr, exception_type)` 检查是否从 `expr` 表达式引发了 `exception_type` 类型的异常。如果表达式 `expr` 没有引发任何异常，则断言失败。除了 `exception_type` 类型之外的异常不会被此断言捕获，并且可以传播到测试主体。测试用例中的未捕获异常会被执行监视器捕获，但它们会导致测试用例失败。

+   `BOOST_<level>_EXCEPTION(expr, exception_type, predicate)` 检查是否从 `expr` 表达式引发了 `exception_type` 类型的异常。如果是这样，它将表达式传递给谓词以进行进一步检查。如果没有引发异常或引发了不同于 `exception_type` 类型的异常，则断言的行为类似于 `BOOST_<level>_THROW`。

本菜谱仅讨论了测试中最常见的 API 及其典型用法。然而，库提供了许多其他 API。有关进一步参考，请查阅在线文档。对于版本 1.83，请参阅 [`www.boost.org/doc/libs/1_83_0/libs/test/doc/html/index.html`](https://www.boost.org/doc/libs/1_83_0/libs/test/doc/html/index.html)。

## 参见

+   *使用 Boost.Test 编写和调用测试*，以了解如何使用 Boost.Test 库的单头版本创建测试套件和测试用例，以及如何运行测试

# 在 Boost.Test 中使用夹具

测试模块越大，测试用例越相似，就越有可能有需要相同设置、清理和可能相同数据的测试用例。包含这些的组件称为**测试夹具**或**测试上下文**。夹具对于建立运行测试的良好定义的环境非常重要，以便结果可重复。示例可以包括在执行测试之前将一组特定文件复制到某个位置，并在测试之后删除它们，或者从特定的数据源加载数据。

Boost.Test 为测试用例、测试套件或模块（全局）提供了定义测试夹具的几种方法。在本菜谱中，我们将探讨夹具的工作原理。

## 准备工作

本菜谱中的示例使用以下类和函数来指定测试单元夹具：

```cpp
struct global_fixture
{
   global_fixture()  { BOOST_TEST_MESSAGE("global setup"); }
   ~global_fixture() { BOOST_TEST_MESSAGE("global cleanup"); }
   int g{ 1 };
};
struct standard_fixture
{
  standard_fixture()  {BOOST_TEST_MESSAGE("setup");}
  ~standard_fixture() {BOOST_TEST_MESSAGE("cleanup");}
  int n {42};
};
struct extended_fixture
{
  std::string name;
  int         data;
  extended_fixture(std::string const & n = "") : name(n), data(0) 
  {
    BOOST_TEST_MESSAGE("setup "+ name);
  }
  ~extended_fixture()
  {
    BOOST_TEST_MESSAGE("cleanup "+ name);
  }
};
void fixture_setup()
{
  BOOST_TEST_MESSAGE("fixture setup");
}
void fixture_cleanup()
{
  BOOST_TEST_MESSAGE("fixture cleanup");
} 
```

前两个是类，其构造函数表示设置函数，析构函数表示清理函数。在示例末尾有一对函数，`fixture_setup()` 和 `fixture_cleanup()`，它们表示测试的设置和清理函数。

## 如何操作...

使用以下方法定义一个或多个测试单元的测试夹具：

+   要为特定的测试用例定义夹具，请使用 `BOOST_FIXTURE_TEST_CASE` 宏：

    ```cpp
    BOOST_FIXTURE_TEST_CASE(test_case, extended_fixture)
    {
      data++;
      BOOST_TEST(data == 1);
    } 
    ```

+   要为测试套件中的所有测试用例定义一个夹具，请使用 `BOOST_FIXTURE_TEST_SUITE`:

    ```cpp
    BOOST_FIXTURE_TEST_SUITE(suite1, extended_fixture)
    BOOST_AUTO_TEST_CASE(case1)
    {
      BOOST_TEST(data == 0);
    }
    BOOST_AUTO_TEST_CASE(case2)
    {
      data++;
      BOOST_TEST(data == 1);
    }
    BOOST_AUTO_TEST_SUITE_END() 
    ```

+   要为测试套件中的所有测试单元定义固定装置（除了一个或多个测试单元），请使用`BOOST_FIXTURE_TEST_SUITE`。您可以使用`BOOST_FIXTURE_TEST_CASE`覆盖特定测试用例，对于嵌套测试套件使用`BOOST_FIXTURE_TEST_SUITE`：

    ```cpp
    BOOST_FIXTURE_TEST_SUITE(suite2, extended_fixture)
    BOOST_AUTO_TEST_CASE(case1)
    {
      BOOST_TEST(data == 0);
    }
    BOOST_FIXTURE_TEST_CASE(case2, standard_fixture)
    {
      BOOST_TEST(n == 42);
    }
    BOOST_AUTO_TEST_SUITE_END() 
    ```

+   要为测试用例或测试套件定义多个固定装置，请使用带有`BOOST_AUTO_TEST_SUITE`和`BOOST_AUTO_TEST_CASE`宏的`boost::unit_test::fixture`：

    ```cpp
    BOOST_AUTO_TEST_CASE(test_case_multifix,
      * boost::unit_test::fixture<extended_fixture>(std::string("fix1"))
      * boost::unit_test::fixture<extended_fixture>(std::string("fix2"))
      * boost::unit_test::fixture<standard_fixture>())
    {
      BOOST_TEST(true);
    } 
    ```

+   在固定装置的情况下，要使用自由函数作为设置和拆卸操作，请使用`boost::unit_test::fixture`：

    ```cpp
    BOOST_AUTO_TEST_CASE(test_case_funcfix,
      * boost::unit_test::fixture(&fixture_setup, &fixture_cleanup))
    {
      BOOST_TEST(true);
    } 
    ```

+   要为模块定义固定装置，请使用`BOOST_GLOBAL_FIXTURE`：

    ```cpp
    BOOST_GLOBAL_FIXTURE(global_fixture); 
    ```

## 它是如何工作的...

该库支持几种固定装置模型：

+   **类模型**，其中构造函数充当设置函数，析构函数充当清理函数。扩展模型允许构造函数有一个参数。在前面的例子中，`standard_fixture`实现了第一种模型，而`extended_fixture`实现了第二种模型。

+   一对**自由函数**：一个定义设置，另一个是可选的，实现清理代码。在前面的例子中，我们在讨论`fixture_setup()`和`fixture_cleanup()`时遇到了这些。

将作为类实现的固定装置也可以有数据成员，并且这些成员对测试单元可用。如果为测试套件定义了固定装置，则它对该测试套件下所有分组测试单元隐式可用。然而，可能存在这种情况，即包含在这样一个测试套件中的测试单元可以重新定义固定装置。在这种情况下，定义在最近作用域中的固定装置是对测试单元可用的。

可以为测试单元定义多个固定装置。然而，这是通过`boost::unit_test::fixture()`装饰器完成的，而不是通过宏。在这种情况下，测试套件和测试用例是通过`BOOST_TEST_SUITE`/`BOOST_AUTO_TEST_SUITE`和`BOOST_TEST_CASE`/`BOOST_AUTO_TEST_CASE`宏定义的。多个`fixture()`装饰器可以通过`operator *`组合在一起，如前所述。此装饰器的目的是定义在测试单元执行前后要调用的设置和拆卸函数。它有几种形式，可以是成对的函数，也可以是类，其中构造函数和析构函数充当设置/拆卸函数。使用包含成员数据的类作为固定装置装饰器的缺点或可能是误导性的一部分是，这些成员将不可用于测试单元。

每当执行测试用例时，都会为每个测试用例构造一个新的固定装置对象，并在测试用例结束时销毁该对象。

固定状态不会在不同测试用例之间共享。因此，构造函数和析构函数会为每个测试用例调用一次。你必须确保这些特殊函数不包含仅应在每个模块中执行一次的代码。如果是这种情况，你应该为整个模块设置一个全局固定点。

全局固定点使用通用测试类模型（具有默认构造函数的模型）；你可以定义任意数量的全局固定点（如果需要，允许你按类别组织设置和清理）。全局固定点使用 `BOOST_GLOBAL_FIXTURE` 宏定义，并且必须在测试文件作用域内定义（不在任何测试单元内部）。它们的作用是定义设置和清理函数，由类的构造函数和析构函数表示。如果类还定义了其他成员，例如数据，这些成员在测试单元中不可用：

```cpp
BOOST_GLOBAL_FIXTURE(global_fixture);
BOOST_AUTO_TEST_CASE(test_case_globals)
{
   BOOST_TEST(g == 1); // error, g not accessible
BOOST_TEST(true);
} 
```

## 相关内容

+   *使用 Boost.Test 编写和调用测试*，了解如何使用 Boost.Test 库的单头版本创建测试套件和测试用例，以及如何运行测试

# 使用 Boost.Test 控制输出

框架为我们提供了自定义测试日志和测试报告中显示内容的能力，然后格式化结果。目前，支持两种格式：一种是人可读格式（或 HRF）和 XML（测试日志还有 JUNIT 格式）。然而，你可以创建并添加自己的格式。

人可读格式是指任何可以由人类自然读取的数据编码形式。用于此目的的文本，无论是以 ASCII 还是 Unicode 编码，都用于此目的。

输出中显示的配置可以在运行时通过命令行开关进行，也可以在编译时通过各种 API 进行。在测试执行期间，框架会收集日志中的所有事件。最后，它生成一个报告，该报告以不同级别的详细程度表示执行摘要。在失败的情况下，报告包含有关位置和原因的详细信息，包括实际值和预期值。这有助于开发者快速识别错误。在本菜谱中，我们将了解如何控制日志和报告中写入的内容以及格式；我们使用运行时的命令行选项来完成此操作。

## 准备工作

在本菜谱中展示的示例中，我们将使用以下测试模块：

```cpp
#define BOOST_TEST_MODULE Controlling output
#include <boost/test/included/unit_test.hpp>
BOOST_AUTO_TEST_CASE(test_case)
{
  BOOST_TEST(true);
}
BOOST_AUTO_TEST_SUITE(test_suite)
BOOST_AUTO_TEST_CASE(test_case)
{
  int a = 42;
  BOOST_TEST(a == 0);
}
BOOST_AUTO_TEST_SUITE_END() 
```

下一个部分将展示如何通过命令行选项控制测试日志和测试报告的输出。

## 如何操作...

要控制测试日志的输出，请执行以下操作：

+   使用 `--log_format=<format>` 或 `-f <format>` 命令行选项来指定日志格式。可能的格式有 `HRF`（默认值）、`XML` 和 `JUNIT`。

+   使用 `--log_level=<level>` 或 `-l <level>` 命令行选项来指定日志级别。可能的日志级别包括 `error`（HRF 和 XML 的默认值）、`warning`、`all` 和 `success`（JUNIT 的默认值）。

+   使用`--log_sink=<stream or file name>`或`-k <stream or file name>`命令行选项来指定框架应写入测试日志的位置。可能的选项是`stdout`（HRF 和 XML 的默认值）、`stderr`或任意文件名（JUNIT 的默认值）。

要控制测试报告的输出，请执行以下操作：

+   使用`--report_format=<format>`或`-m <format>`命令行选项来指定报告格式。可能的格式是`HRF`（默认值）和`XML`。

+   使用`--report_level=<format>`或`-r <format>`命令行选项来指定报告级别。可能的格式是`confirm`（默认值）、`no`（无报告）、`short`和`detailed`。

+   使用`--report_sink=<stream or file name>`或`-e <stream or file name>`命令行选项来指定框架应写入报告日志的位置。可能的选项是`stderr`（默认值）、`stdout`或任意文件名。

## 它是如何工作的...

当您从控制台/终端运行测试模块时，您将看到测试日志和测试报告，测试报告紧随测试日志之后。对于前面显示的测试模块，默认输出如下。前三条线代表测试日志，而最后一条线代表测试报告：

```cpp
Running 2 test cases...
f:/chapter11bt_05/main.cpp(14): error: in "test_suite/test_case": 
check a == 0 has failed [42 != 0]
*** 1 failure is detected in the test module "Controlling output" 
```

测试日志和测试报告的内容可以以多种格式提供。默认是 HRF；然而，框架也支持 XML，对于测试日志，支持 JUNIT 格式。这是一个为自动化工具设计的格式，例如持续构建或集成工具。除了这些选项之外，您可以通过实现自己的从`boost::unit_test::unit_test_log_formatter`派生的类来自定义测试日志的格式。

以下示例展示了如何使用 XML 格式化测试日志（第一个示例）和测试报告（第二个示例）（每个都加粗）：

```cpp
chapter11bt_05.exe -f XML
**<TestLog><Error file="f:/chapter11bt_05/main.cpp"** 
**line="14"><![CDATA[check a == 0 has failed [42 != 0]]]>**
**</Error></TestLog>**
*** 1 failure is detected in the test module "Controlling output"
chapter11bt_05.exe -m XML
Running 2 test cases...
f:/chapter11bt_05/main.cpp(14): error: in "test_suite/test_case": 
check a == 0 has failed [42 != 0]
**<TestResult><TestSuite name="Controlling output" result="failed"** 
**assertions_passed="1" assertions_failed="1" warnings_failed="0"** 
**expected_failures="0" test_cases_passed="1"** 
**test_cases_passed_with_warnings="0" test_cases_failed="1"** 
**test_cases_skipped="0" test_cases_aborted="0"></TestSuite>**
</TestResult> 
```

日志或报告级别表示输出的详细程度。以下表格显示了日志详细程度的可能值，按从低到高的顺序排列。表中的较高级别包括所有高于它的级别的消息：

| **级别** | **报告的消息** |
| --- | --- |
| `nothing` | 没有日志记录。 |
| `fatal_error` | 系统或用户致命错误以及所有在`REQUIRE`级别描述失败的断言的消息（例如`BOOST_TEST_REQUIRE`和`BOOST_REQUIRE_`）。 |
| `system_error` | 系统非致命错误。 |
| `cpp_exception` | 未捕获的 C++异常。 |
| `error` | `CHECK`级别失败的断言（`BOOST_TEST`和`BOOST_CHECK_`）。 |
| `warning` | `WARN`级别失败的断言（`BOOST_TEST_WARN`和`BOOST_WARN_`）。 |
| `message` | 由`BOOST_TEST_MESSAGE`生成的消息。 |
| `test_suite` | 每个测试单元的开始和结束状态的通知。 |
| `all` / `success` | 所有消息，包括通过断言。 |

表 11.2：日志详细程度的可能值

测试报告的可用格式在以下表格中描述：

| **级别** | **描述** |
| --- | --- |
| `no` | 不生成报告。 |
| `confirm` | **通过测试**：*** 未检测到错误。**跳过测试**：*** `<name>` 测试套件被跳过；请参阅标准输出以获取详细信息。**中止测试**：*** `<name>` 测试套件被中止；请参阅标准输出以获取详细信息。**无失败断言的失败测试**：*** 在 `<name>` 测试套件中检测到错误；请参阅标准输出以获取详细信息。**失败测试**：*** 在 `<name>` 测试套件中检测到 N 个失败。**预期失败的失败测试**：*** 在 `<name>` 测试套件中检测到 N 个失败（预期 M 个失败） |
| `detailed` | 结果以分层方式报告（每个测试单元作为父测试单元的一部分进行报告），但只显示相关信息。没有失败断言的测试用例不会在报告中产生条目。测试用例/套件 `<name>` 已通过/被跳过/被中止/失败，有 N 个断言中的 M 个通过/N 个断言中的 M 个失败/N 个警告中的 M 个失败/X 个预期的失败 |
| `short` | 与 `detailed` 类似，但只向主测试套件报告信息。 |

表 11.3：测试报告的可用格式

标准输出流 (`stdout`) 是测试日志的默认写入位置，标准错误流 (`stderr`) 是测试报告的默认位置。然而，测试日志和测试报告都可以重定向到另一个流或文件。

除了这些选项之外，还可以使用 `--report_memory_leaks_to=<文件名>` 命令行选项指定一个单独的文件来报告内存泄漏。如果此选项不存在且检测到内存泄漏，它们将被报告到标准错误流。

## 更多...

除了本配方中讨论的选项之外，框架还提供了额外的编译时 API 来控制输出。有关这些 API 的全面描述以及本配方中描述的功能，请查看框架文档[`www.boost.org/doc/libs/1_83_0/libs/test/doc/html/index.html`](https://www.boost.org/doc/libs/1_83_0/libs/test/doc/html/index.html)。

## 参见

+   *使用 Boost.Test 编写和调用测试*，以了解如何使用 Boost.Test 库的单头版本创建测试套件和测试用例，以及如何运行测试

+   *使用 Boost.Test 进行断言*，以探索 Boost.Test 库中丰富的断言宏集

# 开始使用 Google Test

**Google Test** 是 C++ 中最广泛使用的测试框架之一。**Chromium** 项目和 **LLVM** 编译器是使用它进行单元测试的项目之一。Google Test 允许开发者使用多个编译器在多个平台上编写单元测试。Google Test 是一个便携、轻量级的框架，它提供了一个简单而全面的 API，用于使用断言编写测试；在这里，测试被分组为测试套件，测试套件被分组为测试程序。

框架提供了有用的功能，例如重复执行测试多次，并在第一次失败时中断测试以调用调试器。其断言在启用或禁用异常的情况下都能正常工作。下一道菜将涵盖框架的最重要功能。这道菜将向您展示如何安装框架并设置您的第一个测试项目。

## 准备工作

Google Test 框架，就像 Boost.Test 一样，有一个基于宏的 API。尽管您只需要使用提供的宏来编写测试，但为了更好地使用框架，建议您了解宏。

## 如何操作...

为了设置您的环境以使用 Google Test，请执行以下操作：

1.  从 [`github.com/google/googletest`](https://github.com/google/googletest) 克隆或下载 Git 仓库。

1.  如果您选择下载，下载完成后，请解压存档内容。

1.  使用提供的构建脚本来构建框架。

要使用 Google Test 创建您的第一个测试程序，请执行以下操作：

1.  创建一个新的空 C++ 项目。

1.  根据您使用的开发环境进行必要的设置，以便将框架的头文件目录（称为 `include`）提供给项目以便包含头文件。

1.  将项目链接到 `gtest` 共享库。

1.  向项目中添加一个包含以下内容的源文件：

    ```cpp
    #include <gtest/gtest.h>
    TEST(FirstTestSuite, FirstTest)
    {
      int a = 42;
      ASSERT_TRUE(a > 0);
    }
    int main(int argc, char **argv) 
    {
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
    } 
    ```

1.  构建并运行项目。

## 它是如何工作的...

Google Test 框架提供了一套简单易用的宏，用于创建测试和编写断言。与 Boost.Test 等其他测试框架相比，测试的结构也得到了简化。测试被分组为测试套件，测试套件被分组为测试程序。

提及与术语相关的一些方面是很重要的。传统上，Google Test 并未使用 **测试套件** 这一术语。在 Google Test 中，**测试用例**基本上是一个测试套件，与 Boost.Test 中的测试套件相当。另一方面，**测试函数**相当于一个测试用例。由于这导致了混淆，Google Test 已经遵循了由 **国际软件测试资格认证委员会**（**ISTQB**）使用的通用术语，即测试用例和测试套件，并开始在其代码和文档中替换这些术语。在这本书中，我们将使用这些术语。

该框架提供了一套丰富的断言，包括致命和非致命断言，对异常处理提供了极大的支持，并且能够自定义测试执行方式和输出生成方式。然而，与 Boost.Test 库不同，Google Test 中的测试套件不能包含其他测试套件，只能包含测试函数。

框架的文档可在 GitHub 项目的页面上找到。对于本书的这一版，我使用了 Google Test 框架版本 1.14，但这里展示的代码与框架的先前版本兼容，并预期与框架的未来版本也兼容。*如何做…* 部分中展示的示例代码包含以下部分：

1.  `#include <gtest/gtest.h>` 包含框架的主要头文件。

1.  `TEST(FirstTestSuite, FirstTest)` 声明一个名为 `FirstTest` 的测试，作为名为 `FirstTestSuite` 的测试套件的一部分。这些名称必须是有效的 C++标识符，但不允许包含下划线。测试函数的实际名称是通过将测试套件的名称和测试名称连接起来，并在其中添加一个下划线来组成的。在我们的例子中，名称是 `FirstTestSuite_FirstTest`。来自不同测试套件的测试可能具有相同的单个名称。测试函数没有参数，并返回 `void`。可以将多个测试组合到同一个测试套件中。

1.  `ASSERT_TRUE(a > 0);` 是一个断言宏，当条件评估为 `false` 时会产生致命错误，并从当前函数返回。框架定义了许多其他断言宏，我们将在 *使用 Google Test 进行断言* 菜单中看到。

1.  `testing::InitGoogleTest(&argc, argv);` 初始化框架，必须在调用 `RUN_ALL_TESTS()` 之前执行。

1.  `return RUN_ALL_TESTS();` 自动检测并调用使用 `TEST()` 或 `TEST_F()` 宏定义的所有测试。宏返回的值用作 `main()` 函数的返回值。这很重要，因为自动化测试服务根据 `main()` 函数返回的值来确定测试程序的结果，而不是打印到 `stdout` 或 `stderr` 流的输出。`RUN_ALL_TESTS()` 宏只能调用一次；多次调用不支持，因为它与框架的一些高级功能冲突。

执行此测试程序将提供以下结果：

```cpp
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from FirstTestCase
[ RUN      ] FirstTestCase.FirstTestFunction
[       OK ] FirstTestCase.FirstTestFunction (1 ms)
[----------] 1 test from FirstTestCase (1 ms total)
[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2 ms total)
[  PASSED  ] 1 test. 
```

对于许多测试程序，`main()` 函数的内容与 *如何做…* 部分中显示的示例相同。为了避免编写这样的 `main()` 函数，框架提供了一个基本实现，您可以通过将程序与 `gtest_main` 共享库链接来使用它。

## 还有更多...

Google Test 框架也可以与其他测试框架一起使用。你可以使用其他测试框架，如 Boost.Test 或 CppUnit，来编写测试，并使用 Google Test 断言宏。为此，使用 `--gtest_throw_on_failure` 参数从代码或命令行设置 `throw_on_failure` 标志。或者，使用 `GTEST_THROW_ON_FAILURE` 环境变量并初始化框架，如下面的代码片段所示：

```cpp
#include "gtest/gtest.h"
int main(int argc, char** argv)
{
  testing::GTEST_FLAG(throw_on_failure) = true;
  testing::InitGoogleTest(&argc, argv);
} 
```

当你启用 `throw_on_failure` 选项时，失败的断言将打印错误消息并抛出异常，该异常将被宿主测试框架捕获并视为失败。如果未启用异常，则失败的 Google Test 断言将告诉你的程序以非零代码退出，这同样会被宿主测试框架视为失败。

## 参见

+   *使用 Google Test 编写和调用测试*，以了解如何使用 Google Test 库创建测试和测试套件，以及如何运行测试

+   *使用 Google Test 进行断言*，以探索 Google Test 库中的各种断言宏

# 使用 Google Test 编写和调用测试

在之前的菜谱中，我们瞥见了使用 Google Test 框架编写简单测试需要什么。多个测试可以组合成一个测试套件，一个或多个测试套件可以组合成一个测试程序。在这个菜谱中，我们将看到如何创建和运行测试。

## 准备工作

对于这个菜谱中的示例代码，我们将使用在 *使用 Boost.Test 编写和调用测试* 菜谱中讨论的 `point3d` 类。

## 如何做到...

使用以下宏来创建测试：

+   `TEST(TestSuiteName, TestName)` 定义了一个名为 `TestName` 的测试，作为名为 `TestSuiteName` 的测试套件的一部分：

    ```cpp
    TEST(TestConstruction, TestConstructor)
    {
      auto p = point3d{ 1,2,3 };
      ASSERT_EQ(p.x(), 1);
      ASSERT_EQ(p.x(), 2);
      ASSERT_EQ(p.x(), 3);
    }
    TEST(TestConstruction, TestOrigin)
    {
      auto p = point3d::origin();
      ASSERT_EQ(p.x(), 0);
      ASSERT_EQ(p.x(), 0);
      ASSERT_EQ(p.x(), 0);
    } 
    ```

+   `TEST_F(TestSuiteWithFixture, TestName)` 定义了一个名为 `TestName` 的测试，作为使用 `TestSuiteWithFixture` 固定装置的测试套件的一部分。你可以在 *使用 Google Test 的测试固定装置* 菜谱中找到关于它是如何工作的详细信息。

要执行测试，请执行以下操作：

1.  使用 `RUN_ALL_TESTS()` 宏来运行测试程序中定义的所有测试。这必须在框架初始化后从 `main()` 函数中只调用一次。

1.  使用 `--gtest_filter=<filter>` 命令行选项来过滤要运行的测试。

1.  使用 `--gtest_repeat=<count>` 命令行选项来重复执行所选测试指定的次数。

1.  使用 `--gtest_break_on_failure` 命令行选项，当第一个测试失败时，将调试器附加到测试程序进行调试。

## 它是如何工作的...

可用于定义测试的宏有几个（作为测试用例的一部分）。最常见的是 `TEST` 和 `TEST_F`。后者与 fixtures 一起使用，将在 *使用 Google Test 的测试 fixtures* 菜单中详细讨论。用于定义测试的其他宏包括 `TYPED_TEST` 用于编写类型测试和 `TYPED_TEST_P` 用于编写类型参数化测试。然而，这些是更高级的主题，超出了本书的范围。`TEST` 和 `TEST_F` 宏接受两个参数：第一个是测试套件名称，第二个是测试名称。这两个参数形成测试的完整名称，并且它们必须是有效的 C++ 标识符；它们不应该包含下划线。不同的测试套件可以包含具有相同名称的测试（因为完整名称仍然是唯一的）。这两个宏都会自动将测试注册到框架中；因此，用户不需要显式输入来完成此操作。

测试可以失败或成功。如果断言失败或发生未捕获的异常，则测试失败。除了这两种情况外，测试总是成功的。

要调用测试，请调用 `RUN_ALL_TESTS()`。然而，你只能在测试程序中调用一次，并且只能在调用 `testing::InitGoogleTest()` 初始化框架之后进行。此宏会运行测试程序中的所有测试。然而，你可能只想运行一些测试。你可以通过设置名为 `GTEST_FILTER` 的环境变量并使用适当的过滤器，或者通过使用 `--gtest_filter` 标志将过滤器作为命令行参数来做到这一点。如果这两个中的任何一个存在，框架只会运行名称与过滤器完全匹配的测试。过滤器可以包含通配符：`*` 匹配任何字符串，`?` 符号匹配任何字符。使用连字符（`-`）引入负模式（应该省略的内容）。以下是一些过滤器的示例：

| **过滤器** | **描述** |
| --- | --- |
| `--gtest_filter=*` | 运行所有测试 |
| `--gtest_filter=TestConstruction.*` | 运行名为 `TestConstruction` 的测试套件中的所有测试 |
| `--gtest_filter=TestOperations.*-TestOperations.TestLess` | 运行名为 `TestOperations` 的测试套件中的所有测试，除了名为 `TestLess` 的测试 |
| `--gtest_filter=*Operations*:*Construction*` | 运行所有名称中包含 `Operations` 或 `Construction` 的测试 |
| `--gtest_filter=Test?` | 运行所有名称有 5 个字符且以 `Test` 开头的测试，例如 `TestA`、`Test0` 或 `Test_`。 |
| `--gtest_filter=Test??` | 运行所有名称有 6 个字符且以 `Test` 开头的测试，例如 `TestAB`、`Test00` 或 `Test_Z`。 |

表 11.4：过滤器示例

以下列表是使用命令行参数 `--gtest_filter=TestConstruction.*-TestConstruction.TestConstructor` 调用包含前面显示的测试的测试程序时的输出：

```cpp
Note: Google Test filter = TestConstruction.*-TestConstruction.TestConstructor
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from TestConstruction
[ RUN      ] TestConstruction.TestOrigin
[       OK ] TestConstruction.TestOrigin (0 ms)
[----------] 1 test from TestConstruction (0 ms total)
[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2 ms total)
[  PASSED  ] 1 test. 
```

你可以通过在测试名称前加上 `DISABLED_` 或在具有相同标识符的测试套件名称前加上前缀来禁用一些测试。在这种情况下，测试套件中的所有测试都将被禁用。以下是一个示例：

```cpp
TEST(TestConstruction, DISABLED_TestConversionConstructor) 
{ /* ... */ }
TEST(DISABLED_TestComparisons, TestEquality) 
{ /* ... */ }
TEST(DISABLED_TestComparisons, TestInequality)
{ /* ... */ } 
```

这些测试都不会被执行。然而，你将在输出中收到一份报告，说明你有多个禁用的测试。

请记住，此功能仅用于临时禁用测试。当你需要执行一些会导致测试失败的代码更改，而你又没有时间立即修复它们时，这很有用。因此，应谨慎使用此功能。

## 参见

+   *Google Test 入门*，了解如何安装 Google Test 框架以及如何创建一个简单的测试项目

+   *使用 Google Test 进行断言*，探索 Google Test 库中的各种断言宏

+   *使用 Google Test 的测试夹具*，了解如何在使用 Google Test 库时定义测试夹具

# 使用 Google Test 进行断言

Google Test 框架提供了一套丰富的致命和非致命断言宏，它们类似于函数调用，用于验证测试代码。当这些断言失败时，框架会显示源文件、行号以及相关的错误信息（包括自定义错误信息），以帮助开发者快速识别失败的代码。我们已看到一些使用 `ASSERT_TRUE` 宏的简单示例；在本食谱中，我们将探讨其他可用的宏。

## 如何操作...

使用以下宏来验证测试代码：

+   使用 `ASSERT_TRUE(condition)` 或 `EXPECT_TRUE(condition)` 来检查条件是否为 `true`，以及使用 `ASSERT_FALSE(condition)` 或 `EXPECT_FALSE(condition)` 来检查条件是否为 `false`，以下代码展示了这一用法：

    ```cpp
    EXPECT_TRUE(2 + 2 == 2 * 2);
    EXPECT_FALSE(1 == 2);
    ASSERT_TRUE(2 + 2 == 2 * 2);
    ASSERT_FALSE(1 == 2); 
    ```

+   使用 `ASSERT_XX(val1, val2)` 或 `EXPECT_XX(val1, val2)` 来比较两个值，其中 `XX` 是以下之一：`EQ(val1 == val2)`、`NE(val1 != val2)`、`LT(val1 < val2)`、`LE(val1 <= val2)`、`GT(val1 > val2)` 或 `GE(val1 >= val2)`。以下代码展示了这一用法：

    ```cpp
    auto a = 42, b = 10;
    EXPECT_EQ(a, 42);
    EXPECT_NE(a, b);
    EXPECT_LT(b, a);
    EXPECT_LE(b, 11);
    EXPECT_GT(a, b);
    EXPECT_GE(b, 10); 
    ```

+   使用 `ASSERT_STRXX(str1, str2)` 或 `EXPECT_STRXX(str1, str2)` 来比较两个以 null 结尾的字符串，其中 `XX` 是以下之一：`EQ`（字符串内容相同）、`NE`（字符串内容不同）、`CASEEQ`（忽略大小写时字符串内容相同）和 `CASENE`（忽略大小写时字符串内容不同）。以下代码片段展示了这一用法：

    ```cpp
    auto str = "sample";
    EXPECT_STREQ(str, "sample");
    EXPECT_STRNE(str, "simple");
    ASSERT_STRCASEEQ(str, "SAMPLE");
    ASSERT_STRCASENE(str, "SIMPLE"); 
    ```

+   使用 `ASSERT_FLOAT_EQ(val1, val2)` 或 `EXPECT_FLOAT_EQ(val1, val2)` 来检查两个 `float` 值是否几乎相等，以及使用 `ASSERT_DOUBLE_EQ(val1, val2)` 或 `EXPECT_DOUBLE_EQ(val1, val2)` 来检查两个 `double` 值是否几乎相等；它们之间的差异不应超过 4 **ULP**（**最后一位单位**）。使用 `ASSERT_NEAR(val1, val2, abserr)` 来检查两个值之间的差异是否不大于指定的绝对值：

    ```cpp
    EXPECT_FLOAT_EQ(1.9999999f, 1.9999998f);
    ASSERT_FLOAT_EQ(1.9999999f, 1.9999998f); 
    ```

+   使用`ASSERT_THROW(statement, exception_type)`或`EXPECT_THROW(statement, exception_type)`来检查语句是否抛出指定类型的异常，使用`ASSERT_ANY_THROW(statement)`或`EXPECT_ANY_THROW(statement)`来检查语句是否抛出任何类型的异常，以及使用`ASSERT_NO_THROW(statement)`或`EXPECT_NO_THROW(statement)`来检查语句是否抛出任何异常：

    ```cpp
    void function_that_throws()
    {
      throw std::runtime_error("error");
    }
    void function_no_throw()
    {
    }
    TEST(TestAssertions, Exceptions)
    {
      EXPECT_THROW(function_that_throws(), std::runtime_error);
      EXPECT_ANY_THROW(function_that_throws());
      EXPECT_NO_THROW(function_no_throw());

      ASSERT_THROW(function_that_throws(), std::runtime_error);
      ASSERT_ANY_THROW(function_that_throws());
      ASSERT_NO_THROW(function_no_throw());
    } 
    ```

+   使用`ASSERT_PRED1(pred, val)`或`EXPECT_PRED1(pred, val)`来检查`pred(val)`是否返回`true`，使用`ASSERT_PRED2(pred, val1, val2)`或`EXPECT_PRED2(pred, val1, val2)`来检查`pred(val1, val2)`是否返回`true`，依此类推；用于*n*-元谓词函数或函数对象：

    ```cpp
    bool is_positive(int const val)
    {
      return val != 0;
    }
    bool is_double(int const val1, int const val2)
    {
      return val2 + val2 == val1;
    }
    TEST(TestAssertions, Predicates)
    {
      EXPECT_PRED1(is_positive, 42);
      EXPECT_PRED2(is_double, 42, 21);

      ASSERT_PRED1(is_positive, 42);
      ASSERT_PRED2(is_double, 42, 21);
    } 
    ```

+   使用`ASSERT_HRESULT_SUCCEEDED(expr)`或`EXPECT_HRESULT_SUCCEEDED(expr)`来检查`expr`是否是成功的`HRESULT`，以及使用`ASSERT_HRESULT_FAILED(expr)`或`EXPECT_HRESULT_FAILED(expr)`来检查`expr`是否是失败的`HRESULT`。这些断言旨在在 Windows 上使用。

+   使用`FAIL()`生成致命错误，使用`ADD_FAILURE()`或`ADD_FAILURE_AT(filename, line)`生成非致命错误：

    ```cpp
    ADD_FAILURE();
    ADD_FAILURE_AT(__FILE__, __LINE__); 
    ```

## 它是如何工作的……

所有这些断言都有两种版本：

+   `ASSERT_*`：这会生成致命错误，阻止当前测试函数的进一步执行。

+   `EXPECT_*`：这会生成非致命错误，这意味着即使断言失败，测试函数的执行也会继续。

如果不满足条件不是严重错误，或者您希望测试函数继续执行以获取尽可能多的错误信息，请使用`EXPECT_*`断言。在其他情况下，请使用测试断言的`ASSERT_*`版本。

您可以在框架的在线文档中找到这里展示的断言的详细信息，该文档可在 GitHub 上找到：[`github.com/google/googletest`](https://github.com/google/googletest)；这是项目所在的位置。然而，关于浮点数比较有一个特别的注意事项。由于舍入误差（分数部分不能表示为二的反幂的有限和），浮点数值不会完全匹配。因此，比较应该在相对误差范围内进行。宏`ASSERT_EQ`/`EXPECT_EQ`不适用于比较浮点数，框架提供了一套其他的断言。`ASSERT_FLOAT_EQ`/`ASSERT_DOUBLE_EQ`和`EXPECT_FLOAT_EQ`/`EXPECT_DOUBLE_EQ`使用默认误差为 4 ULP 进行比较。

ULP 是浮点数之间间隔的度量单位，即如果它是 1，则表示最不显著数字的值。有关更多信息，请阅读 Bruce Dawson 撰写的*比较浮点数，2012 年版*文章：[`randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/`](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)。

## 参见

+   *使用 Google Test 编写和调用测试*，了解如何使用 Google Test 库创建测试和测试套件，以及如何运行测试

# 使用 Google Test 的测试用例

框架提供了支持，将测试用例作为可重用组件用于测试套件中的所有测试。它还提供了支持，用于设置测试将运行的全球环境。在本食谱中，您将找到逐步说明如何定义和使用测试用例，以及如何设置测试环境。

## 准备工作

您现在应该熟悉使用 Google Test 框架编写和调用测试，这是在本章前面提到的主题，特别是在 *使用 Google Test 编写和调用测试* 食谱中。

## 如何操作...

要创建和使用测试用例，请执行以下操作：

1.  创建一个从 `testing::Test` 类派生的类：

    ```cpp
    class TestFixture : public testing::Test
    {
    }; 
    ```

1.  使用构造函数来初始化测试用例，并使用析构函数来清理它：

    ```cpp
    protected:
      TestFixture()
      {
        std::cout << "constructing fixture\n";
        data.resize(10);
        std::iota(std::begin(data), std::end(data), 1);
      }
      ~TestFixture()
      {
        std::cout << "destroying fixture\n";
      } 
    ```

1.  或者，您也可以重写虚拟方法 `SetUp()` 和 `TearDown()` 以达到相同的目的。

1.  向类中添加成员数据和函数，以便它们对测试可用：

    ```cpp
    protected:
      std::vector<int> data; 
    ```

1.  使用 `TEST_F` 宏定义使用测试用例的测试，并将测试用例类名指定为测试套件名称：

    ```cpp
    TEST_F(TestFixture, TestData)
    {
      ASSERT_EQ(data.size(), 10);
      ASSERT_EQ(data[0], 1);
      ASSERT_EQ(data[data.size()-1], data.size());
    } 
    ```

要自定义运行测试的环境设置，请执行以下操作：

1.  创建一个从 `testing::Environment` 派生的类：

    ```cpp
    class TestEnvironment : public testing::Environment
    {
    }; 
    ```

1.  重写虚拟方法 `SetUp()` 和 `TearDown()` 以执行设置和清理操作：

    ```cpp
    public:
      virtual void SetUp() override 
     {
        std::cout << "environment setup\n";
      }
      virtual void TearDown() override 
     {
        std::cout << "environment cleanup\n";
      }
      int n{ 42 }; 
    ```

1.  在调用 `RUN_ALL_TESTS()` 之前，通过调用 `testing::AddGlobalTestEnvironment()` 来注册环境：

    ```cpp
    int main(int argc, char **argv)
    {
      testing::InitGoogleTest(&argc, argv);
      testing::AddGlobalTestEnvironment(new TestEnvironment{});
      return RUN_ALL_TESTS();
    } 
    ```

## 它是如何工作的...

文本测试用例允许用户在多个测试之间共享数据配置。测试用例对象在测试之间不共享。对于与文本函数关联的每个测试，都会创建不同的测试用例对象。框架为来自测试用例的每个测试执行以下操作：

1.  创建一个新的测试用例对象。

1.  调用其 `SetUp()` 虚拟方法。

1.  运行测试。

1.  调用测试用例的 `TearDown()` 虚拟方法。

1.  销毁测试用例对象。

您可以通过两种方式设置和清理测试用例对象：使用构造函数和析构函数，或者使用 `SetUp()` 和 `TearDown()` 虚拟方法。在大多数情况下，前者是首选的方法。虚拟方法的使用适用于几种情况：

+   当清理操作抛出异常时，因为不允许异常离开析构函数。

+   如果在清理过程中需要使用断言宏，并且使用了 `--gtest_throw_on_failure` 标志，该标志用于确定在发生失败时抛出的宏。

+   如果需要调用虚拟方法（这些方法可能在派生类中被重写），因为虚拟调用不应从构造函数或析构函数中调用。

使用测试用例的测试必须使用 `TEST_F` 宏（其中 `_F` 代表测试用例）。尝试使用 `TEST` 宏声明它们将生成编译器错误。

运行测试的环境也可以进行定制。机制类似于测试夹具：您从基类 `testing::Environment` 派生，并重写 `SetUp()` 和 `TearDown()` 虚拟函数。这些派生环境类的实例必须通过调用 `testing::AddGlobalTestEnvironment()` 在框架中进行注册；然而，这必须在运行测试之前完成。您可以注册任意多个实例，在这种情况下，`SetUp()` 方法将按注册顺序调用对象，而 `TearDown()` 方法将按相反的顺序调用。您必须将动态实例化的对象传递给此函数。框架将接管对象，并在程序终止前删除它们；因此，请不要自行删除。

环境对象对测试不可用，也不打算为测试提供数据。它们的目的在于为运行测试定制全局环境。

## 参见

+   使用 Google Test 编写和调用测试，了解如何使用 Google Test 库创建测试和测试套件，以及如何运行测试

# 使用 Google Test 控制输出

默认情况下，Google Test 程序的输出流向标准流，以可读的格式打印。框架提供了几个选项来自定义输出，包括以基于 JUNIT 的格式将 XML 打印到磁盘文件。本菜谱将探讨可用于控制输出的选项。

## 准备工作

为了本菜谱的目的，让我们考虑以下测试程序：

```cpp
#include <gtest/gtest.h>
TEST(Sample, Test)
{
  auto a = 42;
  ASSERT_EQ(a, 0);
}
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 
```

其输出如下：

```cpp
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Sample
[ RUN      ] Sample.Test
f:\chapter11gt_05\main.cpp(6): error: Expected equality of these values:
  a
    Which is: 42
  0
[  FAILED  ] Sample.Test (1 ms)
[----------] 1 test from Sample (1 ms total)
[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (3 ms total)
[  PASSED  ] 0 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] Sample.Test
 1 FAILED TEST 
```

我们将使用这个简单的测试程序来演示我们可以用来控制程序输出的各种选项，这些选项在以下部分中进行了示例。

## 如何做...

要控制测试程序的输出，您可以：

+   使用 `--gtest_output` 命令行选项或带有 `xml:filepath` 字符串的 `GTEST_OUTPUT` 环境变量来指定要写入 XML 报告的文件位置：

    ```cpp
    chapter11gt_05.exe --gtest_output=xml:report.xml
    <?xml version="1.0" encoding="UTF-8"?>
    <testsuites tests="1" failures="1" disabled="0" errors="0" 
                time="0.007" timestamp="2020-05-18T19:00:17" 
                name="AllTests">
      <testsuite name="Sample" tests="1" failures="1" disabled="0" 
                 errors="0" time="0.002"
                 timestamp="2020-05-18T19:00:17">
        <testcase name="Test" status="run" result="completed" time="0"
                  timestamp="2020-05-18T19:00:17" classname="Sample">
          <failure message="f:\chapter11gt_05\main.cpp:6&#x0A;Expected equality of these values:&#x0A;  a&#x0A;    Which is: 42&#x0A;  0&#x0A;" type=""><![CDATA[f:\chapter11gt_05\main.cpp:6
    Expected equality of these values:
      a
        Which is: 42
      0
    ]]></failure> 
        </testcase>
      </testsuite>
    </testsuites> 
    ```

+   使用 `--gtest_color` 命令行选项或 `GTEST_COLOR` 环境变量，并指定 `auto`、`yes` 或 `no` 以指示报告是否应使用颜色打印到终端：

    ```cpp
    chapter11gt_05.exe --gtest_color=no 
    ```

+   使用 `--gtest_print_time` 命令行选项或带有值 `0` 的 `GTEST_PRINT_TIME` 环境变量来抑制打印每个测试执行所需的时间：

    ```cpp
    chapter11gt_05.exe --gtest_print_time=0
    [==========] Running 1 test from 1 test suite.
    [----------] Global test environment set-up.
    [----------] 1 test from Sample
    [ RUN      ] Sample.Test
    f:\chapter11gt_05\main.cpp(6): error: Expected equality of these values:
      a
        Which is: 42
      0
    [  FAILED  ] Sample.Test
    [----------] Global test environment tear-down
    [==========] 1 test from 1 test suite ran.
    [  PASSED  ] 0 tests.
    [  FAILED  ] 1 test, listed below:
    [  FAILED  ] Sample.Test
     1 FAILED TEST 
    ```

## 它是如何工作的...

以 XML 格式生成报告不会影响打印到终端的易读报告。输出路径可以指示文件、目录（在这种情况下，将创建一个以可执行文件命名的文件 - 如果之前运行已存在，则通过在后面添加数字创建一个新名称的文件），或者无，在这种情况下，报告将写入当前目录中名为 `test_detail.xml` 的文件。

XML 报告格式基于 JUNITReport Ant 任务，并包含以下主要元素：

+   `<testsuites>`：这是根元素，对应整个测试程序。

+   `<testsuite>`：这对应于一个测试套件。

+   `<testcase>`：这对应于一个测试函数，因为 Google Test 函数在其他框架中相当于测试用例。

默认情况下，框架会报告每个测试执行所需的时间。可以使用 `--gtest_print_time` 命令行选项或 `GTEST_PRINT_TIME` 环境变量来抑制此功能，如前所述。

## 参见

+   *使用 Google Test 编写和调用测试*，查看如何使用 Google Test 库创建测试和测试套件，以及如何运行测试

+   *使用 Google Test 的测试夹具*，学习如何在使用 Google Test 库时定义测试夹具

# 开始使用 Catch2

**Catch2** 是一个用于 C++ 和 Objective-C 的多范式测试框架。Catch2 的名字沿袭自 Catch，这是框架的第一个版本，代表 **C++ Automated Test Cases in Headers**。它允许开发者使用传统的测试函数分组在测试用例中的风格或带有 *given-when-then* 部分的 **行为驱动开发**(**BDD**) 风格来编写测试。测试是自动注册的，并且框架提供了几个断言宏；在这些宏中，使用得最多的是两个：一个是致命的（即，`REQUIRE`）和一个非致命的（即，`CHECK`）。它们对左右两边的表达式进行分解，并在失败时记录。与第一个版本不同，Catch2 不再支持 C++03。Catch2 的当前版本是 v3，与 Catch2 v2 相比有一些重大变化，例如，库不再是单头库，而是作为一个常规库（需要编译）工作，并需要一个 C++14 编译器。

在本章剩余的食谱中，我们将学习如何使用 Catch2 版本 3 编写单元测试。

## 准备工作

Catch2 测试框架有一个基于宏的 API。虽然你只需要使用提供的宏来编写测试，但如果想更好地使用该框架，建议对宏有一个良好的理解。

## 如何做...

为了设置你的环境以使用 Catch2 测试框架，请执行以下操作：

1.  从 [`github.com/catchorg/Catch2`](https://github.com/catchorg/Catch2) 克隆或下载 Git 仓库。

1.  下载仓库后，解压缩存档内容。

要使用 Catch 2 的 v3 版本，你有两种选择：

+   在你的测试项目中使用合并（混合）的库和源文件。这些文件被称为 `catch_amalgamated.hpp` 和 `catch_amalgamated.cpp`。它们位于 Catch2 库的 `extras` 文件夹中，如果你想的话，可以将它们复制到你的测试项目中。这样做的好处是，你不必处理 CMake 脚本，但代价是增加了构建时间。

+   使用 CMake 将 Catch2 添加为你的项目的静态库。

要使用 Catch2 和其合并文件创建您的第一个测试程序，请执行以下操作：

1.  创建一个新的空 C++ 项目。

1.  将 Catch2 库的 `extras` 文件夹中的 `catch_amalgamated.hpp` 和 `catch_amalgamated.cpp` 文件复制到您的测试项目中。

1.  将 `catch_amalgamated.cpp` 源文件添加到您的项目中，与其他源文件（包含测试）一起编译。

1.  向项目中添加一个新源文件，内容如下：

    ```cpp
    #include "catch_amalgamated.hpp"
    TEST_CASE("first_test_case", "[learn][catch]")
    {
      SECTION("first_test_function")
      {
        auto i{ 42 };
        REQUIRE(i == 42);
      }
    } 
    ```

1.  构建并运行项目。

要使用 CMake 集成创建您的第一个 Catch2 测试程序，请执行以下操作：

1.  打开控制台/命令提示符，并将目录更改为克隆/解压缩的 Catch2 文件的位置。

1.  使用以下命令构建库。在 Unix 系统上运行：

    ```cpp
    cmake -Bbuild -H. -DBUILD_TESTING=OFF
    sudo cmake --build build/ --target instal 
    ```

    在 Windows 系统上，从具有管理员权限的命令提示符中执行以下命令：

    ```cpp
    cmake -Bbuild -H. -DBUILD_TESTING=OFF
    cmake --build build/ --target instal 
    ```

1.  为 C++ 测试项目创建一个新的文件夹（称为 `Test`）。

1.  向此文件夹添加一个新源文件（称为 `main.cpp`），内容如下：

    ```cpp
    #include <catch2/catch_test_macros.hpp>
    TEST_CASE("first_test_case", "[learn][catch]")
    {
       SECTION("first_test_function")
       {
          auto i{ 42 };
          REQUIRE(i == 42);
       }
    } 
    ```

1.  在 `Test` 文件夹中添加一个新的 `CMakeLists.txt` CMake 文件，内容如下：

    ```cpp
    find_package(Catch2 3 REQUIRED)
    add_executable(Test main.cpp)
    target_link_libraries(Test PRIVATE Catch2::Catch2WithMain) 
    ```

1.  运行 `cmake.exe` 以生成/构建您的项目。

使用 CMake 设置项目有多种方式。在这个菜谱中，我提供了一个最小示例，它有效，您也可以在 GitHub 仓库的源文件中找到它。熟悉 CMake 的读者可能会找到比这里提供的方法更好的方法。您可以从在线资源中了解更多关于 CMake 的信息。

## 它是如何工作的...

Catch2 允许开发者将测试用例编写为自注册函数；它甚至可以提供 `main()` 函数的默认实现，这样您就可以专注于测试代码并编写更少的设置代码。测试用例被划分为单独运行的章节。该框架不遵循 **setup-test-teardown** 架构的风格。相反，测试用例部分（或者更确切地说，最内层的部分，因为部分可以嵌套）是执行的单位，以及它们的封装部分。这使得固定装置变得不再需要，因为数据和设置以及拆卸代码可以在多个级别上重用。

测试用例和部分使用字符串标识，而不是标识符（如大多数测试框架中那样）。测试用例也可以被标记，以便可以根据标记执行或列出测试。测试结果以文本可读格式打印；然而，它们也可以导出为 XML 格式，使用 Catch2 特定的模式或 JUNIT ANT 模式，以便轻松集成到持续交付系统中。测试的执行可以参数化，在失败时中断（在 Windows 和 macOS 上），这样您就可以附加调试器并检查程序。

该框架易于安装和使用。正如在 *如何做…* 部分中看到的那样，有两种替代方案：

+   使用合并的文件 `catch_amalgamated.hpp` 和 `catch_amalgamated_cpp`。这些是所有头文件和源文件的合并。使用它们的优点是，你不必担心构建 Catch2 库。你只需要将这些文件复制到你的目标位置（通常在项目内部），在你的包含测试的文件中包含 `catch_amalgamated.hpp` 头文件，并与其他源文件一起构建 `catch_amalgamated.cpp`。使用这种方法的不利之处是增加了构建时间。

+   将 Catch2 作为静态库使用。这要求你在使用之前构建库。你可以明确地将头文件和 `lib` 文件添加到你的项目中，或者你可以使用 CMake 来完成这个任务。这种方法的优势是减少了构建时间。

上一节中展示的示例代码有以下部分：

1.  `#include "catch_amalgamated.hpp"` 包含了库的合并头文件，这是一个所有库头文件的合并。另一方面，如果你使用的是构建版本，你只需要包含你需要的特定头文件，例如 `<catch2/catch_test_macros.hpp>`。你可以包含 `<cathc2/catch_all.hpp>`，但这将包含所有库头文件，这并不建议。一般来说，你应该只包含你需要的头文件。

1.  `TEST_CASE("first_test_case", "[learn][catch]")` 定义了一个名为 `first_test_case` 的测试用例，它有两个关联的标签：`learn` 和 `catch`。标签用于选择运行或仅列出测试用例。多个测试用例可以带有相同的标签。

1.  `SECTION("first_test_function")` 定义了一个部分，即一个测试函数，称为 `first_test_function`，作为外部测试用例的一部分。

1.  `REQUIRE(i == 42);` 是一个断言，告诉测试如果条件不满足则测试失败。

运行此程序的结果如下：

```cpp
=========================================================
All tests passed (1 assertion in 1 test cases) 
```

## 还有更多...

如前所述，该框架使我们能够使用带有 *给-当-然后* 部分的 BDD 风格编写测试。这是通过使用几个别名实现的：`SCENARIO` 对应于 `TEST_CASE` 和 `GIVE`、`WHEN`、`AND_WHEN`、`THEN` 和 `AND_THEN` 对应于 `SECTION`。使用这种风格，我们可以重写前面展示的测试，如下所示：

```cpp
SCENARIO("first_scenario", "[learn][catch]")
{
  GIVEN("an integer")
  {
    auto i = 0;
    WHEN("assigned a value")
    {
      i = 42;
      THEN("the value can be read back")
      {
        REQUIRE(i == 42);
      }
    }
  }
} 
```

当程序成功执行时，它将打印以下输出：

```cpp
=========================================================
All tests passed (1 assertion in 1 test cases) 
```

然而，在失败的情况下（假设我们得到了错误的条件：`i == 0`），失败的表达式以及左右两侧的值将在输出中打印出来，如下面的代码片段所示：

```cpp
---------------------------------------------------------------
f:\chapter11ca_01\main.cpp(11)
...............................................................
f:\chapter11ca_01\main.cpp(13): FAILED:
  REQUIRE( i == 0 )
with expansion:
  42 == 0
===============================================================
test cases: 1 | 1 failed
assertions: 1 | 1 failed 
```

这里展示的输出，以及在下述食谱中的其他代码片段，已经从实际的控制台输出中略微裁剪或压缩，以便更容易地在本书的页面中列出。

## 参见

+   *使用 Catch2 编写和调用测试*，以了解如何使用 Catch2 库创建测试，无论是基于测试用例的传统风格还是基于场景的 BDD 风格，以及如何运行测试

+   使用 *Catch2* 进行断言，以探索 Catch2 库中的各种断言宏

# 使用 Catch2 编写和调用测试

Catch2 框架允许你使用传统的测试用例和测试函数风格或带有场景和 *given-when-then* 部分的 BDD 风格来编写测试。测试被定义为测试用例的独立部分，可以嵌套到你想要的深度。无论你更喜欢哪种风格，测试都只使用两个基本宏来定义。这个配方将展示这些宏是什么以及它们是如何工作的。

## 如何操作...

要使用传统的测试用例和测试函数风格编写测试，请这样做：

+   使用 `TEST_CASE` 宏定义一个带有名称（作为字符串）的测试用例，可选地，一个与其关联的标签列表：

    ```cpp
    TEST_CASE("test construction", "[create]")
    {
      // define sections here
    } 
    ```

+   使用 `SECTION` 宏在测试用例内部定义一个测试函数，名称作为字符串：

    ```cpp
    TEST_CASE("test construction", "[create]")
    {
      SECTION("test constructor")
      {
        auto p = point3d{ 1,2,3 };
        REQUIRE(p.x() == 1);
        REQUIRE(p.y() == 2);
        REQUIRE(p.z() == 4);
      }
    } 
    ```

+   如果你想重用设置和清理代码或以分层结构组织测试，请定义嵌套部分：

    ```cpp
    TEST_CASE("test operations", "[modify]")
    {
      SECTION("test methods")
      {
        SECTION("test offset")
        {
          auto p = point3d{ 1,2,3 };
          p.offset(1, 1, 1);
          REQUIRE(p.x() == 2);
          REQUIRE(p.y() == 3);
          REQUIRE(p.z() == 3);
        }
      }
    } 
    ```

要使用 BDD 风格编写测试，请这样做：

+   使用 `SCENARIO` 宏定义场景，指定其名称：

    ```cpp
    SCENARIO("modify existing object")
    {
      // define sections here
    } 
    ```

+   在场景内部使用 `GIVEN`、`WHEN` 和 `THEN` 宏定义嵌套部分，为每个部分指定一个名称：

    ```cpp
    SCENARIO("modify existing object")
    {
      GIVEN("a default constructed point")
      {
        auto p = point3d{};
        REQUIRE(p.x() == 0);
        REQUIRE(p.y() == 0);
        REQUIRE(p.z() == 0);
        WHEN("increased with 1 unit on all dimensions")
        {
          p.offset(1, 1, 1);
          THEN("all coordinates are equal to 1")
          {
            REQUIRE(p.x() == 1);
            REQUIRE(p.y() == 1);
            REQUIRE(p.z() == 1);
          }
        }
      }
    } 
    ```

要执行测试，请执行以下操作：

+   要执行程序中的所有测试（除了隐藏的测试），运行测试程序而不带任何命令行参数（如下述代码中描述的）。

+   要执行特定的一组测试用例，提供一个过滤器作为命令行参数。这可以包含测试用例名称、通配符、标签名称和标签表达式：

    ```cpp
    chapter11ca_02.exe "test construction"
    test construction
       test constructor
    -------------------------------------------------
    f:\chapter11ca_02\main.cpp(7)
    .................................................
    f:\chapter11ca_02\main.cpp(12): FAILED:
      REQUIRE( p.z() == 4 )
    with expansion:
      3 == 4
    =================================================
    test cases: 1 | 1 failed
    assertions: 6 | 5 passed | 1 failed 
    ```

+   要执行特定的部分（或一系列部分），使用带有部分名称的命令行参数 `--section` 或 `-c`（可以多次使用以执行多个部分）：

    ```cpp
    chapter11ca_02.exe "test construction" --section "test origin"
    Filters: test construction
    ==================================================
    All tests passed (3 assertions in 1 test case) 
    ```

+   要指定测试用例应运行的顺序，使用命令行参数 `--order` 并选择以下值之一：`decl`（声明顺序），`lex`（按名称的字典顺序），或 `rand`（使用 `std::random_shuffle()` 确定的随机顺序）。以下是一个示例：

    ```cpp
    chapter11ca_02.exe --order lex 
    ```

## 它是如何工作的...

测试用例会自动注册，不需要开发人员做任何额外工作来设置测试程序，除了定义测试用例和测试函数。测试函数定义为测试用例的部分（使用 `SECTION` 宏），并且可以嵌套。

节的嵌套深度没有限制。测试用例和测试函数（从现在起将被称为节），形成一个树状结构，测试用例位于根节点，最内层的节作为叶子。当测试程序运行时，执行的是叶子节。每个叶子节都是独立于其他叶子节执行的。然而，执行路径从根测试用例开始，向下继续，直到最内层的节。路径上遇到的所有代码在每次运行时都会完全执行。这意味着当多个节共享公共代码（来自父节或测试用例）时，相同的代码为每个节执行一次，执行之间不共享任何数据。这在一方面消除了对特殊夹具方法的需求。另一方面，它为每个节（路径上遇到的所有内容）提供了多个夹具，这是许多测试框架所缺少的功能。

编写测试用例的 BDD 风格由相同的两个宏提供支持，即`TEST_CASE`和`SECTION`，以及测试节的能力。实际上，宏`SCENARIO`是对`TEST_CASE`的重定义，而`GIVEN`、`WHEN`、`AND_WHEN`、`THEN`和`AND_THEN`是对`SECTION`的重定义：

```cpp
#define SCENARIO( ... ) TEST_CASE( "Scenario: " __VA_ARGS__ )
#define GIVEN(desc)     INTERNAL_CATCH_DYNAMIC_SECTION("    Given: " << desc)
#define AND_GIVEN(desc) INTERNAL_CATCH_DYNAMIC_SECTION("And given: " << desc)
#define WHEN(desc)      INTERNAL_CATCH_DYNAMIC_SECTION("     When: " << desc)
#define AND_WHEN(desc)  INTERNAL_CATCH_DYNAMIC_SECTION(" And when: " << desc)
#define THEN(desc)      INTERNAL_CATCH_DYNAMIC_SECTION("     Then: " << desc)
#define AND_THEN(desc)  INTERNAL_CATCH_DYNAMIC_SECTION("      And: " << desc) 
```

当你执行测试程序时，所有定义的测试都会运行。然而，这排除了隐藏的测试，这些测试要么使用以`./`开头的名称指定，要么使用以点开头的标签指定。也可以通过提供命令行参数`[.]`或`[hide]`来强制运行隐藏的测试。

可以对要执行的测试用例进行过滤。这可以通过名称或标签来完成。以下表格显示了其中的一些可能选项：

| **参数** | **描述** |
| --- | --- |
| `"测试构建"` | 被称为`test construction`的测试用例 |
| `test*` | 所有以`test`开头的测试用例 |
| `~"测试构建"` | 除了被称为`test construction`的测试用例之外的所有测试用例 |
| `~*equal*` | 所有不包含单词`equal`的测试用例 |
| `a* ~ab* abc` | 所有以`a`开头的测试，除了以`ab`开头的，除了`abc`（包含在内） |
| `[修改]` | 所有带有标签`[修改]`的测试用例 |
| `[修改],[比较][操作]` | 所有带有标签`[修改]`或同时带有`[比较]`和`[操作]`的测试用例 |
| `-#sourcefile` | 来自`sourcefile.cpp`文件的全部测试 |

表 11.5：将要执行的测试用例的过滤器示例

通过指定命令行参数 `--section` 或 `-c` 中的一个或多个部分名称，也可以执行特定的测试函数。但是，此选项不支持通配符。如果你指定要运行的部分，请注意，将从根测试用例到所选部分的整个测试路径都将执行。此外，如果你首先没有指定测试用例或一组测试用例，则将执行所有测试用例，尽管只有它们中匹配的部分。

## 参见

+   *开始使用 Catch2*，学习如何安装 Catch2 框架以及如何创建一个简单的测试项目

+   *使用 Catch2 断言*，以探索 Catch2 库中的各种断言宏

# 使用 Catch2 断言

与其他测试框架不同，Catch2 不提供大量断言宏。它有两个主要的宏：`REQUIRE`，在失败时产生致命错误，停止测试用例的执行，和 `CHECK`，在失败时产生非致命错误，继续测试用例的执行。还定义了几个附加的宏；在本食谱中，我们将看到如何使用它们。

## 准备工作

你现在应该熟悉使用 Catch2 编写测试用例和测试函数，这是我们之前在 *使用 Catch2 编写和调用测试* 这一食谱中讨论的主题。

## 如何做...

以下列表包含使用 Catch2 框架进行断言的可用选项：

+   使用 `CHECK(expr)` 来检查 `expr` 是否评估为 `true`，在失败时继续执行，并使用 `REQUIRE(expr)` 来确保 `expr` 评估为 `true`，在失败时停止测试的执行：

    ```cpp
    int a = 42;
    CHECK(a == 42);
    REQUIRE(a == 42); 
    ```

+   使用 `CHECK_FALSE(expr)` 和 `REQUIRE_FALSE(expr)` 来确保 `expr` 评估为 `false`，并在失败时产生非致命或致命错误：

    ```cpp
    int a = 42;
    CHECK_FALSE(a > 100);
    REQUIRE_FALSE(a > 100); 
    ```

+   使用浮点数匹配器 `WithinAbs`、`WithinRel` 和 `WithinUPL` 来比较浮点数（这比过时的 `Approx` 类更受欢迎）：

    ```cpp
    double a = 42.5;
    CHECK_THAT(42.0, Catch::Matchers::WithinAbs(a, 0.5));
    REQUIRE_THAT(42.0, Catch::Matchers::WithinAbs(a, 0.5));
    CHECK_THAT(42.0, Catch::Matchers::WithinRel(a, 0.02));
    REQUIRE_THAT(42.0, Catch::Matchers::WithinRel(a, 0.02)); 
    ```

+   使用 `CHECK_NOTHROW(expr)`/`REQUIRE_NOTHROW(expr)` 来验证 `expr` 不抛出任何错误，`CHECK_THROWS(expr)`/`REQUIRE_THROWS(expr)` 来验证 `expr` 抛出任何类型的错误，`CHECK_THROWS_AS(expr, exctype)`/`REQUIRE_THROWS_AS(expr, exctype)` 来验证 `expr` 抛出类型为 `exctype` 的异常，或者 `CHECK_THROWS_WITH(expression, string or string matcher)`/`REQUIRE_THROWS_WITH(expression, string or string matcher)` 来验证 `expr` 抛出的异常描述与指定的字符串匹配：

    ```cpp
    void function_that_throws()
    {
      throw std::runtime_error("error");
    }
    void function_no_throw()
    {
    }
    SECTION("expressions")
    {
      CHECK_NOTHROW(function_no_throw());
      REQUIRE_NOTHROW(function_no_throw());

      CHECK_THROWS(function_that_throws());
      REQUIRE_THROWS(function_that_throws());

      CHECK_THROWS_AS(function_that_throws(), std::runtime_error);
      REQUIRE_THROWS_AS(function_that_throws(), std::runtime_error);

      CHECK_THROWS_WITH(function_that_throws(), "error");
      REQUIRE_THROWS_WITH(function_that_throws(), 
              Catch::Matchers::ContainsSubstring("error"));
    } 
    ```

+   使用 `CHECK_THAT(value, matcher expression)`/`REQUIRE_THAT(expr, matcher expression)` 来检查给定的匹配器表达式是否对指定的值评估为 `true`：

    ```cpp
    std::string text = "this is an example";
    CHECK_THAT(
      text,
      Catch::Matchers::ContainsSubstring("EXAMPLE", Catch::CaseSensitive::No));
    REQUIRE_THAT(
      text,
      Catch::Matchers::StartsWith("this") && 
      Catch::Matchers::ContainsSubstring("an")); 
    ```

+   使用 `FAIL(message)` 来报告 `message` 并使测试用例失败，`WARN(message)` 来记录消息而不停止测试用例的执行，以及 `INFO(message)` 来将消息记录到缓冲区，并且只在下一个会失败的断言中报告它。

## 它是如何工作的...

`REQUIRE`/`CATCH` 宏系列将表达式分解为其左右两侧的项，并在失败时报告失败的位置（源文件和行）、表达式以及左右两侧的值：

```cpp
f:\chapter11ca_03\main.cpp(19): FAILED:
  REQUIRE( a == 1 )
with expansion:
  42 == 1 
```

然而，这些宏不支持使用逻辑运算符（如 `&&` 和 `||`）组成的复杂表达式。以下示例是错误的：

```cpp
REQUIRE(a < 10 || a %2 == 0);   // error 
```

解决这个问题的方法是创建一个变量来保存表达式评估的结果，并在断言宏中使用它。然而，在这种情况下，打印表达式元素展开的能力丢失了：

```cpp
auto expr = a < 10 || a % 2 == 0;
REQUIRE(expr); 
```

另一个选择是使用另一组括号。然而，这也阻止了分解工作：

```cpp
REQUIRE((a < 10 || a %2 == 0)); // OK 
```

两套断言，即 `CHECK_THAT`/`REQUIRE_THAT` 和 `CHECK_THROWS_WITH`/`REQUIRE_THROWS_WITH`，与匹配器一起工作。匹配器是可扩展和可组合的组件，用于执行值匹配。框架提供了几个匹配器，包括：

+   字符串：`StartsWith`、`EndsWith`、`ContainsSubstring`、`Equals` 和 `Matches`

+   `std::vector`：`Contains`、`VectorContains`、`Equals`、`UnorderedEquals` 和 `Approx`

+   浮点值：`WithinAbs`、`WithinULP`、`WithinRel` 和 `IsNaN`

+   类似于范围类型（从版本 3.0.1 开始包含）：`IsEmpty`、`SizeIs`、`Contains`、`AllMatch`、`AnyMatch`、`NoneMatch`、`AllTrue`、`AnyTrue`、`NoneTrue`、`RangeEquals`、`UnorderedRangeEquals`

+   异常：`Message` 和 `MessageMatches`

`Contains()` 和 `VectorContains()` 之间的区别在于 `Contains()` 在另一个向量中搜索一个向量，而 `VectorContains()` 在向量内部搜索单个元素。

如前所述，有几个匹配器针对浮点数。这些匹配器包括：

+   `WithinAbs()`：创建一个接受小于或等于目标数且具有指定边缘（0 到 1 之间的数字表示的百分比）的浮点数的匹配器：

    ```cpp
    REQUIRE_THAT(42.0, WithinAbs(42.5, 0.5)); 
    ```

+   `WithinRel()`：创建一个接受近似等于目标值且具有给定容忍度的浮点数的匹配器：

    ```cpp
    REQUIRE_THAT(42.0, WithinRel(42.4, 0.01)); 
    ```

+   `WithinULP()`：创建一个接受目标值不超过给定 ULP 的浮点数的匹配器：

    ```cpp
    REQUIRE_THAT(42.0, WithinRel(target, 4)); 
    ```

这些匹配器也可以组合在一起，如下所示：

```cpp
REQUIRE_THAT(a,
  Catch::Matchers::WithinRel(42.0, 0.001) ||
  Catch::Matchers::WithinAbs(42.0, 0.000001)); 
```

一个过时的比较浮点数的方法由名为 `Approx` 的类表示，位于 `Catch` 命名空间中。这个类通过值重载了相等/不等和比较运算符，通过这些值可以构造一个 `double` 值。两个值可以相差的边缘或被认为是相等的边缘可以指定为给定值的百分比。这可以通过成员函数 `epsilon()` 来设置。值必须在 0 和 1 之间（例如，0.05 的值是 5%）。epsilon 的默认值设置为 `std::numeric_limits<float>::epsilon()*100`。

您可以创建自己的匹配器，无论是为了扩展现有框架的功能还是为了与您自己的类型一起工作。创建自定义匹配器有两种方式：旧版 v2 方式和新版 v3 方式。

要以旧方式创建自定义匹配器，有两个必要条件：

1.  从 `Catch::MatcherBase<T>` 派生出的匹配器类，其中 `T` 是正在比较的类型。必须重写两个虚拟函数：`match()`，它接受一个要匹配的值并返回一个布尔值，指示匹配是否成功，以及 `describe()`，它不接受任何参数但返回一个描述匹配器的字符串。

1.  从测试代码中调用的构建函数。

以下示例定义了一个匹配器，用于 `point3d` 类，这是我们在本章中看到过的，以检查给定的三维点是否位于三维空间中的一条直线上：

```cpp
class OnTheLine : public Catch::Matchers::MatcherBase<point3d>
{
  point3d const p1;
  point3d const p2;
public:
  OnTheLine(point3d const & p1, point3d const & p2):
    p1(p1), p2(p2)
  {}
  virtual bool match(point3d const & p) const override
 {
    auto rx = p2.x() - p1.x() != 0 ? 
             (p.x() - p1.x()) / (p2.x() - p1.x()) : 0;
    auto ry = p2.y() - p1.y() != 0 ? 
             (p.y() - p1.y()) / (p2.y() - p1.y()) : 0;
    auto rz = p2.z() - p1.z() != 0 ? 
             (p.z() - p1.z()) / (p2.z() - p1.z()) : 0;
    return 
      Catch::Approx(rx).epsilon(0.01) == ry &&
      Catch::Approx(ry).epsilon(0.01) == rz;
  }
protected:
  virtual std::string describe() const
 {
    std::ostringstream ss;
    ss << "on the line between " << p1 << " and " << p2;
    return ss.str();
  }
};
inline OnTheLine IsOnTheLine(point3d const & p1, point3d const & p2)
{
  return OnTheLine {p1, p2};
} 
```

要以新方式创建自定义匹配器，您需要以下内容：

1.  从 `Catch::Matchers::MatcherGenericBase` 派生出的匹配器类。这个类必须实现两个方法：`bool match(…) const`，它执行匹配，以及重写虚拟函数 `string describe() const`，它不接受任何参数但返回一个描述匹配器的字符串。尽管这些与旧式风格中使用的函数非常相似，但有一个关键区别：`match()` 函数对其参数的传递方式没有要求。这意味着它可以按值传递或通过可变引用传递参数。此外，它还可以是一个函数模板。优点是它使得编写更复杂的匹配器成为可能，例如可以比较类似范围的类型。

1.  从测试代码中调用的构建函数。

以新风格编写的比较 `point3d` 值的相同匹配器如下所示：

```cpp
class OnTheLine : public Catch::Matchers::MatcherGenericBase
{
   point3d const p1;
   point3d const p2;
public:
   OnTheLine(point3d const& p1, point3d const& p2) :
      p1(p1), p2(p2)
   {
   }
   bool match(point3d const& p) const
 {
      auto rx = p2.x() - p1.x() != 0 ? 
                (p.x() - p1.x()) / (p2.x() - p1.x()) : 0;
      auto ry = p2.y() - p1.y() != 0 ? 
                (p.y() - p1.y()) / (p2.y() - p1.y()) : 0;
      auto rz = p2.z() - p1.z() != 0 ? 
                (p.z() - p1.z()) / (p2.z() - p1.z()) : 0;
      return
         Catch::Approx(rx).epsilon(0.01) == ry &&
         Catch::Approx(ry).epsilon(0.01) == rz;
   }
protected:
   std::string describe() const override
 {
#ifdef __cpp_lib_format
return std::format("on the line between ({},{},{}) and ({},{},{})", p1.x(), p1.y(), p1.z(), p2.x(), p2.y(), p2.z());
#else
      std::ostringstream ss;
      ss << "on the line between " << p1 << " and " << p2;
      return ss.str();
#endif
   }
}; 
```

以下测试用例包含了一个如何使用此自定义匹配器的示例：

```cpp
TEST_CASE("matchers")
{
  SECTION("point origin")
  {
    point3d p { 2,2,2 };
    REQUIRE_THAT(p, IsOnTheLine(point3d{ 0,0,0 }, point3d{ 3,3,3 }));
  }
} 
```

此测试确保点 `{2,2,2}` 位于由点 `{0,0,0}` 和 `{3,3,3}` 定义的直线上，使用了之前实现的 `IsOnTheLine()` 自定义匹配器。

## 参见

+   *使用 Catch2 编写和调用测试*，了解如何使用 Catch2 库创建测试，无论是使用基于测试用例的传统风格还是基于场景的 BDD 风格，以及如何运行测试

# 使用 Catch2 控制输出

与本书中讨论的其他测试框架一样，Catch2 以人类可读的格式将测试程序执行的结果报告到 `stdout` 标准流。支持额外的选项，例如使用 XML 格式报告或写入文件。在本食谱中，我们将查看使用 Catch2 控制输出的主要选项。

## 准备工作

为了说明测试程序执行输出可能如何修改，请使用以下测试用例：

```cpp
TEST_CASE("case1")
{
  SECTION("function1")
  {
    REQUIRE(true);
  }
}
TEST_CASE("case2")
{
  SECTION("function2")
  {
    REQUIRE(false);
  }
} 
```

运行这两个测试用例的输出如下：

```cpp
----------------------------------------------------------
case2
  function2
----------------------------------------------------------
f:\chapter11ca_04\main.cpp(14)
..........................................................
f:\chapter11ca_04\main.cpp(16): FAILED:
  REQUIRE( false )
==========================================================
test cases: 2 | 1 passed | 1 failed
assertions: 2 | 1 passed | 1 failed 
```

在下一节中，我们将探讨控制 Catch2 测试程序输出的各种选项。

## 如何操作...

要控制使用 Catch2 时测试程序的输出，你可以：

+   使用命令行参数 `-r` 或 `--reporter <reporter>` 来指定用于格式化和结构化结果的报告器。框架提供的默认选项是 `console`、`compact`、`xml` 和 `junit`：

    ```cpp
    chapter11ca_04.exe -r junit
    <?xml version="1.0" encoding="UTF-8"?>
    <testsuites>
      <testsuite name="chapter11ca_04.exe" errors="0" 
                 failures="1"
                 tests="2" hostname="tbd" 
                 time="0.002039" 
                 timestamp="2020-05-02T21:17:04Z">
        <testcase classname="case1" name="function1" 
                  time="0.00016"/>
        <testcase classname="case2" 
                  name="function2" time="0.00024">
          <failure message="false" type="REQUIRE">
            at f:\chapter11ca_04\main.cpp(16)
          </failure>
        </testcase>
        <system-out/>
        <system-err/>
      </testsuite>
    </testsuites> 
    ```

+   使用命令行参数 `-s` 或 `--success` 来显示成功测试用例的结果：

    ```cpp
    chapter11ca_04.exe -s
    --------------------------------------------------
    case1
      function1
    --------------------------------------------------
    f:\chapter11ca_04\main.cpp(6)
    ..................................................
    f:\chapter11ca_04\main.cpp(8):
    PASSED:
      REQUIRE( true )
    --------------------------------------------------
    case2
      function2
    --------------------------------------------------
    f:\chapter11ca_04\main.cpp(14)
    ..................................................
    f:\chapter11ca_04\main.cpp(16): 
    FAILED:
      REQUIRE( false )
    ==================================================
    test cases: 2 | 1 passed | 1 failed
    assertions: 2 | 1 passed | 1 failed 
    ```

+   使用命令行参数 `-o` 或 `--out <filename>` 将所有输出发送到文件而不是标准流：

    ```cpp
    chapter11ca_04.exe -o test_report.log 
    ```

+   使用命令行参数 `-d` 或 `--durations <yes/no>` 来显示每个测试用例执行所需的时间：

    ```cpp
    chapter11ca_04.exe -d yes
    0.000 s: scenario1
    0.000 s: case1
    --------------------------------------------------
    case2
       scenario2
    --------------------------------------------------
    f:\chapter11ca_04\main.cpp(14)
    ..................................................
    f:\chapter11ca_04\main.cpp(16): 
    FAILED:
      REQUIRE( false )
    0.003 s: scenario2
    0.000 s: case2
    0.000 s: case2
    ==================================================
    test cases: 2 | 1 passed | 1 failed
    assertions: 2 | 1 passed | 1 failed 
    ```

## 它是如何工作的...

除了默认用于报告测试程序执行结果的易读格式外，Catch2 框架还支持两种 XML 格式：

+   一种特定的 Catch2 XML 格式（通过 `-r xml` 指定）

+   一种类似于 JUNIT 的 XML 格式，遵循 JUNIT ANT 任务的结构（通过 `-r junit` 指定）

前者报告器在单元测试执行和结果可用时流式传输 XML 内容，可以用作 XSLT 转换的输入以生成实例的 HTML 报告。后者报告器需要在打印报告之前收集程序的所有执行数据。JUNIT XML 格式对于被第三方工具（如持续集成服务器）消费很有用。

在独立头文件中提供了几个额外的报告器。它们需要包含在测试程序的源代码中（所有额外报告器的名称格式为 `catch_reporter_*.hpp`）。这些额外的可用报告器包括：

+   **TeamCity** 报告器（通过 `-r teamcity` 指定），它将 TeamCity 服务消息写入标准输出流。它仅适用于与 TeamCity 集成。它是一个流式报告器；数据在可用时即被写入。

+   **Automake** 报告器（通过 `-r automake` 指定），它通过 `make check` 写入 `automake` 所期望的元标签。

+   **Test Anything Protocol**（或简称 **TAP**）报告器（通过 `-r tap` 指定）。

+   **SonarQube** 报告器（通过 `-r sonarqube` 指定），它使用 SonarQube 通用测试数据 XML 格式进行写入。

以下示例展示了如何包含 TeamCity 头文件以使用 TeamCity 报告器生成报告：

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_teamcity.hpp> 
```

测试报告的默认目标是标准流 `stdout`（即使明确写入 `stderr` 的数据最终也会被重定向到 `stdout`）。然而，输出也可能被写入到文件中。这些格式化选项可以组合使用。请看以下命令：

```cpp
chapter11ca_04.exe -r junit -o test_report.xml 
```

此命令指定报告应使用 JUNIT XML 格式，并保存到名为 `test_report.xml` 的文件中。

## 参见

+   *开始使用 Catch2*，了解如何安装 Catch2 框架以及如何创建一个简单的测试项目

+   *使用 Catch2 编写和调用测试*，了解如何使用 Catch2 库创建测试，无论是基于测试用例的传统风格还是基于场景的 BDD 风格，以及如何运行测试

# 在 Discord 上了解更多

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![](img/QR_Code2659294082093549796.png)
