# 第一章. 开始编写你的应用程序

在本章中，我们将涵盖以下内容：

+   获取配置选项

+   在容器/变量中存储任何值

+   在容器/变量中存储多个选定的类型

+   使用一种更安全的方式来处理存储多个选定类型的数据容器

+   在没有值的情况下返回一个值或标志

+   从函数中返回一个数组

+   将多个值合并为一个

+   重新排序函数的参数

+   将值绑定为一个函数参数

+   使用 C++11 移动模拟

+   创建一个不可复制的类

+   创建一个不可复制但可移动的类

# 简介

Boost 是一组 C++ 库。每个库在被接受到 Boost 之前都经过了众多专业程序员的审查。库在多个平台上使用多个编译器和 C++ 标准库实现进行测试。在使用 Boost 时，你可以确信你正在使用一个最可移植、快速和可靠的解决方案，该解决方案适用于商业和开源项目。

Boost 的许多部分已经被包含在 C++11 中，甚至更多部分将被包含在下一个 C++ 标准中。你将在本书的每个菜谱中找到 C++11 特定的说明。

不进行长篇大论，让我们开始吧！

在本章中，我们将看到一些日常使用的菜谱。我们将了解如何从不同的来源获取配置选项，以及可以使用 Boost 库作者引入的一些数据类型制作什么。

# 获取配置选项

看一看一些控制台程序，例如 Linux 中的 `cp`。它们都有花哨的帮助，它们的输入参数不依赖于任何位置，并且具有人类可读的语法，例如：

```cpp
$ cp --help 

Usage: cp [OPTION]... [-T] SOURCE DEST 
 -a, --archive           same as -dR --preserve=all 
 -b                      like --backup but does not accept an argument

```

你可以在 10 分钟内为你的程序实现相同的功能。你所需要的只是 `Boost.ProgramOptions` 库。

## 准备工作

对于这个菜谱，你需要具备基本的 C++ 知识。记住，这个库不是仅头文件，所以你的程序需要链接到 `libboost_program_options` 库。

## 如何做到这一点...

让我们从一个小程序开始，该程序接受苹果和橙子的数量作为输入并计算水果的总数。我们希望达到以下结果：

```cpp
$ our_program –apples=10 –oranges=20
Fruits count: 30

```

执行以下步骤：

1.  首先，我们需要包含 `program_options` 头文件并为 `boost::program_options` 命名空间创建一个别名（它太长了，难以输入！）我们还需要一个 `<iostream>` 头文件：

    ```cpp
    #include <boost/program_options.hpp>
    #include <iostream>
    namespace opt = boost::program_options;
    ```

1.  现在，我们已经准备好描述我们的选项：

    ```cpp
    // Constructing an options describing variable and giving
    // it a textual description "All options" to it.
    opt::options_description desc("All options");

    // When we are adding options, first parameter is a name
    // to be used in command line. Second parameter is a type
    // of that option, wrapped in value<> class. 
    // Third parameter must be a short description of that 
    // option
    desc.add_options()
        ("apples", opt::value<int>(), "how many apples do you have")
        ("oranges", opt::value<int>(), "how many oranges do you have")
    ;
    ```

1.  我们将在稍后一点时间看到如何使用第三个参数，之后我们将处理解析命令行和输出结果：

    ```cpp
    // Variable to store our command line arguments
    opt::variables_map vm;

    // Parsing and storing arguments
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    std::cout << "Fruits count: "
        << vm["apples"].as<int>() + vm["oranges"].as<int>()
        << std::endl;
    ```

    这很简单，不是吗？

1.  让我们向我们的选项描述中添加 `--help` 参数：

    ```cpp
        ("help", "produce help message")
    ```

1.  现在在 `opt::notify(vm);` 之后添加以下行，你将为你的程序获得一个完全功能性的帮助：

    ```cpp
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    ```

    现在，如果我们用 `--help` 参数调用我们的程序，我们将得到以下输出：

    ```cpp
    All options: 
      --apples arg          how many apples do you have 
      --oranges arg         how many oranges do you have 
      --help                produce help message
    ```

    如你所见，我们没有为选项的值提供类型，因为我们不期望任何值传递给它。

1.  一旦我们掌握了所有基础知识，让我们为一些选项添加简短名称，为苹果设置默认值，添加一些字符串输入，并从配置文件中获取缺失的选项：

    ```cpp
    #include <boost/program_options.hpp>
    // 'reading_file' exception class is declared in errors.hpp
    #include <boost/program_options/errors.hpp>
    #include <iostream>
    namespace opt = boost::program_options;

    int main(int argc, char *argv[])
    {
        opt::options_description desc("All options");
        // 'a' and 'o' are short option names for apples and 
        // oranges 'name' option is not marked with 
        // 'required()', so user may not support it
        desc.add_options()
            ("apples,a", opt::value<int>()->default_value(10), "apples that you have")
            ("oranges,o", opt::value<int>(), "oranges that you have")
            ("name", opt::value<std::string>(), "your name")
            ("help", "produce help message")
        ;
        opt::variables_map vm;
     // Parsing command line options and storing values to 'vm'

       opt::store(opt::parse_command_line(argc, argv, desc), vm);
        // We can also parse environment variables using 
        // 'parse_environment' method
        opt::notify(vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }
        // Adding missing options from "aples_oranges.cfg" 
        // config file.
        // You can also provide an istreamable object as a 
        // first parameter for 'parse_config_file'
        // 'char' template parameter will be passed to
        // underlying std::basic_istream object
        try {
          opt::store(
            opt::parse_config_file<char>("apples_oranges.cfg", desc), 
            vm
          );
        } catch (const opt::reading_file& e) {
            std::cout 
               << "Failed to open file 'apples_oranges.cfg': "
               << e.what();
        }
        opt::notify(vm);
        if (vm.count("name")) {
          std::cout << "Hi," << vm["name"].as<std::string>() << "!\n";
        }

        std::cout << "Fruits count: "
            << vm["apples"].as<int>() + vm["oranges"].as<int>()
            << std::endl;
        return 0;
    }
    ```

    ### 注意

    当使用配置文件时，我们需要记住，其语法与命令行语法不同。我们不需要在选项前放置减号。因此，我们的`apples_oranges.cfg`选项必须看起来像这样：

    `oranges=20`

## 它是如何工作的...

从代码和注释中理解这个例子非常简单。更有趣的是我们在执行时得到的结果：

```cpp
$ ./our_program --help 
All options: 
  -a [ --apples ] arg (=10) how many apples do you have 
  -o [ --oranges ] arg      how many oranges do you have 
  --name arg                your name 
  --help                    produce help message 

$ ./our_program 
Fruits count: 30

$ ./our_program -a 10 -o 10 --name="Reader" 
Hi,Reader! 
Fruits count: 20
```

## 还有更多...

C++11 标准采用了许多 Boost 库；然而，你不会在其中找到`Boost.ProgramOptions`。

## 相关内容

+   Boost 的官方文档包含更多示例，并展示了`Boost.ProgramOptions`的更多高级特性，如位置相关的选项、非常规语法等。这可以在以下链接中找到：

    [`www.boost.org/doc/libs/1_53_0/doc/html/program_options.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/program_options.html)

### 小贴士

**下载示例代码**

你可以从你购买的所有 Packt 书籍的账户中下载示例代码文件。[`www.PacktPub.com`](http://www.PacktPub.com)。如果你在其他地方购买了这本书，你可以访问[`www.PacktPub.com/support`](http://www.PacktPub.com/support)并注册以直接将文件通过电子邮件发送给你。

# 在容器/变量中存储任何值

如果你一直在使用 Java、C#或 Delphi 编程，你肯定会怀念在 C++中创建具有`Object`值类型的容器的能力。在这些语言中，`Object`类是几乎所有类型的基本类，因此你可以在任何时候将其分配给（几乎）任何值。想象一下，如果 C++有这样一个特性会多么好：

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <auto_ptr.h>

int main()
{
    typedef std::auto_ptr<Object> object_ptr;
    std::vector<object_ptr> some_values;
    some_values.push_back(new Object(10));
    some_values.push_back(new Object("Hello there"));
    some_values.push_back(new Object(std::string("Wow!")));
    std::string* p = 
         dynamic_cast<std::string*>(some_values.back().get());
    assert(p);

    (*p) += " That is great!\n";
    std::cout << *p;
    return 0;
}
```

## 准备工作

我们将使用仅包含头文件的库。对于这个配方，你只需要具备基本的 C++知识。

## 如何实现...

在这种情况下，Boost 提供了一个解决方案，即`Boost.Any`库，它具有更好的语法：

```cpp
#include <boost/any.hpp>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    std::vector<boost::any> some_values;
    some_values.push_back(10);
    const char* c_str = "Hello there!";
    some_values.push_back(c_str);
    some_values.push_back(std::string("Wow!"));
    std::string& s = 
       boost::any_cast<std::string&>(some_values.back());
    s += " That is great!\n";
    std::cout << s;
    return 0;
}
```

太棒了，不是吗？顺便说一下，它有一个空状态，可以使用`empty()`成员函数进行检查（就像在 STL 容器中一样）。

你可以使用两种方法从`boost::any`获取值：

```cpp
    boost::any variable(std::string("Hello world!"));

    //#1: Following method may throw a boost::bad_any_cast exception
    // if actual value in variable is not a std::string
    std::string s1 = boost::any_cast<std::string>(variable);

    //#2: If actual value in variable is not a std::string
    // will return an NULL pointer
    std::string* s2 = boost::any_cast<std::string>(&variable);
```

## 它是如何工作的...

`boost::any`类只是存储任何值。为了实现这一点，它使用**类型擦除**技术（类似于 Java 或 C#对其所有类型所做的那样）。要使用这个库，您实际上并不需要了解其内部实现，所以我们只需快速浏览一下类型擦除技术。当对类型为`T`的某个变量进行赋值时，`Boost.Any`构造一个类型（让我们称它为`holder<T>`），它可以存储指定类型`T`的值，并且是从某个内部基类型占位符派生出来的。占位符有用于获取存储类型的`std::type_info`的虚拟函数和用于克隆存储类型的虚拟函数。当使用`any_cast<T>()`时，`boost::any`检查存储值的`std::type_info`是否等于`typeid(T)`（使用重载的占位符函数来获取`std::type_info`）。

## 还有更多...

这样的灵活性从来都不是没有代价的。复制构造、值构造、复制赋值以及将值赋给`boost::any`的实例将调用动态内存分配函数；所有的类型转换都需要获取**运行时类型信息**（**RTTI**）；`boost::any`大量使用虚拟函数。如果您对性能很感兴趣，请看下一个菜谱，它将给您一个在没有动态分配和 RTTI 使用的情况下实现几乎相同结果的想法。

`Boost.Any`的另一个缺点是它不能与禁用 RTTI 一起使用。有可能使这个库即使在禁用 RTTI 的情况下也能使用，但目前还没有实现。

### 注意

几乎所有的 Boost 异常都源自`std::exception`类或其派生类，例如，`boost::bad_any_cast`是从`std::bad_cast`派生出来的。这意味着您可以使用`catch (const std::exception& e)`捕获几乎所有的 Boost 异常。

## 参见

+   Boost 的官方文档可能给您一些更多的例子，您可以在[`www.boost.org/doc/libs/1_53_0/doc/html/any.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/any.html)找到它。

+   查看有关“使用更安全的方式来处理存储多个选定类型的容器”菜谱以获取更多关于该主题的信息

# 在变量/容器中存储多个选定的类型

您是否了解 C++11 中无限制联合的概念？让我简要地告诉您。**C++03 联合**只能存储称为 POD（普通旧数据）的非常简单的数据类型。因此，在 C++03 中，您不能在联合中存储`std::string`或`std::vector`。C++11 放宽了这一要求，但您必须自己管理这些类型的构造和析构，调用就地构造/析构，并记住存储在联合中的类型。这是一项巨大的工作，不是吗？

## 准备工作

我们将使用仅包含头文件的库进行工作，这个库使用起来很简单。您只需要具备基本的 C++知识就可以使用这个菜谱。

## 如何做到...

让我来向您介绍`Boost.Variant`库。

1.  `Boost.Variant` 库可以存储编译时指定的任何类型；它还管理就地构造/销毁，甚至不需要 C++11 标准：

    ```cpp
    #include <boost/variant.hpp>
    #include <iostream>
    #include <vector>
    #include <string>

    int main()
    {
        typedef boost::variant<int, const char*, std::string> 
          my_var_t;
        std::vector<my_var_t> some_values;
        some_values.push_back(10);
        some_values.push_back("Hello there!");
        some_values.push_back(std::string("Wow!"));
        std::string& s = boost::get<std::string>(some_values.back());
        s += " That is great!\n";
        std::cout << s;
        return 0;
    }
    ```

    太棒了，不是吗？

1.  `Boost.Variant` 没有空状态，但它有一个 `empty()` 函数，该函数始终返回 `false`。如果你确实需要表示一个空状态，只需在 `Boost.Variant` 库支持的类型中的第一个位置添加一些平凡类型。当 `Boost.Variant` 包含该类型时，将其解释为空状态。以下是一个示例，我们将使用 `boost::blank` 类型来表示空状态：

    ```cpp
        typedef boost::variant<boost::blank, int, const char*, std::string> my_var_t;
        // Default constructor will construct an 
        // instance of boost::blank
        my_var_t var;
        // 'which()' method returns an index of a type,
        // currently held by variant.
        assert(var.which() == 0); // Empty state
        var = "Hello, dear reader";
        assert(var.which() != 0);
    ```

1.  你可以使用两种方法从变体中获取值：

    ```cpp
        boost::variant<int, std::string> variable(0);
        // Following method may throw a boost::bad_get
        // exception if actual value in variable is not an int
        int s1 = boost::get<int>(variable);
        // If actual value in variable is not an int
        // will return an NULL pointer
        int* s2 = boost::get<int>(&variable);
    ```

## 它是如何工作的...

`boost::variant` 类持有一个字符数组，并在该数组中存储值。数组的大小在编译时使用 `sizeof()` 和获取对齐的函数确定。在赋值或构造 `boost::variant` 时，之前的值就地销毁，并在字符数组上使用新的放置构造新值。

## 还有更多...

`Boost.Variant` 变量通常不会在堆上分配内存，并且它们不需要启用 RTTI。`Boost.Variant` 非常快，并且被其他 Boost 库广泛使用。为了达到最佳性能，请确保支持类型列表中有一个平凡类型，并且该类型位于第一个位置。

### 注意

`Boost.Variant` 不是 C++11 标准的一部分。

## 参见

+   *使用更安全的方式来处理存储多个选定类型的容器* 菜谱

+   Boost 的官方文档包含了更多关于 `Boost.Variant` 的示例和一些其他特性的描述，可以在以下位置找到：

    [`www.boost.org/doc/libs/1_53_0/doc/html/variant.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/variant.html)

# 使用更安全的方式来处理存储多个选定类型的容器

想象一下，你正在创建一个围绕某些 SQL 数据库接口的包装器。你决定 `boost::any` 将完美地满足数据库表单单元格的要求。其他程序员将使用你的类，他的任务是从数据库中获取一行并计算行中算术类型的总和。

这就是代码的样貌：

```cpp
#include <boost/any.hpp>
#include <vector>
#include <string>
#include <typeinfo>
#include <algorithm>
#include <iostream>

// This typedefs and methods will be in our header,
// that wraps around native SQL interface
typedef boost::any cell_t;
typedef std::vector<cell_t> db_row_t;

// This is just an example, no actual work with database.
db_row_t get_row(const char* /*query*/) {
    // In real application 'query' parameter shall have a 'const
    // char*' or 'const std::string&' type? See recipe Using a 
    // reference to string type in Chapter 7, Manipulating Strings
    // for an answer.
    db_row_t row;
    row.push_back(10);
    row.push_back(10.1f);
    row.push_back(std::string("hello again"));
    return row;
}

// This is how a user will use your classes
struct db_sum: public std::unary_function<boost::any, void> {
private:
    double& sum_;
public:
    explicit db_sum(double& sum)
        : sum_(sum)
    {}

    void operator()(const cell_t& value) {
        const std::type_info& ti = value.type();
        if (ti == typeid(int)) {
            sum_ += boost::any_cast<int>(value);
        } else if (ti == typeid(float)) {
            sum_ += boost::any_cast<float>(value);
        }
    }
};

int main()
{
    db_row_t row = get_row("Query: Give me some row, please.");
    double res = 0.0;
    std::for_each(row.begin(), row.end(), db_sum(res));
    std::cout << "Sum of arithmetic types in database row is: " << res << std::endl;
    return 0;
}
```

如果你编译并运行这个示例，它将输出正确的结果：

```cpp
Sum of arithmetic types in database row is: 20.1
```

你还记得阅读 `operator()` 的实现时你的想法吗？我猜它们是，“那么 double、long、short、unsigned 以及其他类型怎么办？”。同样的想法也会出现在使用你的接口的程序员心中。所以你需要仔细记录你的 `cell_t` 存储的值，或者阅读以下章节中描述的更优雅的解决方案。

## 准备工作

如果你还不熟悉 `Boost.Variant` 和 `Boost.Any` 库，强烈建议你阅读前面的两个菜谱。

## 如何做到...

`Boost.Variant` 库实现了一种访问存储数据的访问者编程模式，这比通过 `boost::get<>` 获取值要安全得多。这种模式迫使程序员注意每个变体类型，否则代码将无法编译。您可以通过 `boost::apply_visitor` 函数使用此模式，该函数将访问者功能对象作为第一个参数，将变体作为第二个参数。访问者功能对象必须从 `boost::static_visitor<T>` 类派生，其中 `T` 是访问者返回的类型。访问者对象必须为变体存储的每个类型重载 `operator()`。

让我们将 `cell_t` 类型更改为 `boost::variant<int, float, string>` 并修改我们的示例：

```cpp
#include <boost/variant.hpp>
#include <vector>
#include <string>
#include <iostream>

// This typedefs and methods will be in header,
// that wraps around native SQL interface.
typedef boost::variant<int, float, std::string> cell_t;
typedef std::vector<cell_t> db_row_t;

// This is just an example, no actual work with database.
db_row_t get_row(const char* /*query*/) {
    // See the recipe "Using a reference to string type" 
    // in Chapter 7, Manipulating Strings
    // for a better type for 'query' parameter.
    db_row_t row;
    row.push_back(10);
    row.push_back(10.1f);
    row.push_back("hello again");
    return row;
}

// This is how code required to sum values
// We can provide no template parameter
// to boost::static_visitor<> if our visitor returns nothing.
struct db_sum_visitor: public boost::static_visitor<double> {
    double operator()(int value) const {
        return value;
    }
    double operator()(float value) const {
        return value;
    }
    double operator()(const std::string& /*value*/) const {
        return 0.0;
    }
};

int main()
{
    db_row_t row = get_row("Query: Give me some row, please.");
    double res = 0.0;
    db_row_t::const_iterator it = row.begin(), end = row.end();
    for (; it != end; ++it) {
         res += boost::apply_visitor(db_sum_visitor(), *it);
    }
    std::cout << "Sum of arithmetic types in database row is: " << res << std::endl;
    return 0;
}
```

## 它是如何工作的...

`Boost.Variant` 库将在编译时生成一个大的 `switch` 语句，每个 `case` 都将调用变体类型列表中的单个类型的访问者。在运行时，可以使用 `which()` 获取存储类型的索引，并跳转到 `switch` 语句中的正确 `case`。对于 `boost::variant<int, float, std::string>`，将生成类似以下的内容：

```cpp
switch (which())
{
case 0: return visitor(*reinterpret_cast<int*>(address()));
case 1: return visitor(*reinterpret_cast<float*>(address()));
case 2: return visitor(*reinterpret_cast<std::string*>(address()));
default: assert(false);
}
```

在这里，`address()` 函数返回 `boost::variant<int, float, std::string>` 的内部存储指针。

## 还有更多...

如果我们将此示例与配方中的第一个示例进行比较，我们将看到 `boost::variant` 的以下优点：

+   我们知道一个变量可以存储哪些类型

+   如果 SQL 接口库的编写者向变体中添加或修改类型，我们将得到编译时错误而不是不正确的行为。

## 参见

+   在阅读了 第四章 中的部分内容后，*编译时技巧*，您将能够使访问者对象如此通用，即使底层类型发生变化，它也能正确工作。

+   Boost 的官方文档包含了更多示例和 `Boost.Variant` 的某些其他特性的描述，可在以下链接找到：

    [`www.boost.org/doc/libs/1_53_0/doc/html/variant.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/variant.html)

# 在没有值的情况下返回值或标志

假设我们有一个不抛出异常并返回值或指示发生错误的函数。在 Java 或 C# 编程语言中，这些情况通过比较函数返回值与空指针来处理；如果是空指针，则表示发生了错误。在 C++ 中，从函数返回指针会混淆库用户，并且通常需要动态内存分配（这很慢）。

## 准备工作

此配方只需要基本的 C++ 知识。

## 如何实现...

女士们，先生们，让我通过以下示例向您介绍 `Boost.Optional` 库：

`try_lock_device()`函数尝试为设备获取锁，可能成功也可能不成功，这取决于不同的条件（在我们的例子中取决于`rand()`函数调用）。该函数返回一个可选变量，可以转换为布尔变量。如果返回值等于布尔`true`，则已获取锁，可以通过解引用返回的可选变量来获取用于处理设备的类的实例：

```cpp
#include <boost/optional.hpp>
#include <iostream>
#include <stdlib.h>

class locked_device {
    explicit locked_device(const char* /*param*/) {
        // We have unique access to device
        std::cout << "Device is locked\n";
    }
public:
    ~locked_device () {
        // Releasing device lock
    }

    void use() {
        std::cout << "Success!\n";
    }
    static boost::optional<locked_device> try_lock_device() {
        if (rand()%2) {
            // Failed to lock device
            return boost::none;
        }
        // Success!
        return locked_device("device name");
    }
};

int main()
{
    // Boost has a library called Random. If you wonder why it was 
    // written when stdlib.h has rand() function, see the recipe
    // "Using a true random number generator in Chapter 12, 
    // Scratching the Tip of the Iceberg
    srandom(5);
    for (unsigned i = 0; i < 10; ++i) {
        boost::optional<locked_device> t = locked_device::try_lock_device();
        // optional is convertible to bool
        if (t) {
            t->use();
            return 0;
        } else {
            std::cout << "...trying again\n";
        }
    }
    std::cout << "Failure!\n";
    return -1;
}
```

这个程序将输出以下内容：

```cpp
...trying again 
...trying again 
Device is locked 
Success! 
```

### 注意

默认构造的`optional`变量可以转换为持有`false`的布尔变量，并且不得解引用，因为它没有构造的底层类型。

## 它是如何工作的...

`Boost.Optional`类与`boost::variant`类非常相似，但只针对一种类型，`boost::optional<T>`有一个`chars`数组，其中类型为`T`的对象可以是一个就地构造器。它还有一个布尔变量来记住对象的状态（是否已构造）。

## 还有更多...

`Boost.Optional`类不使用动态分配，并且不需要底层类型的默认构造函数。它速度快，被认为将被纳入 C++的下一个标准。当前的`boost::optional`实现不能与 C++11 **右值引用**一起工作；然而，已经提出了一些补丁来修复这个问题。

C++11 标准不包括`Boost.Optional`类；然而，它目前正在被审查，以纳入下一个 C++标准或 C++14。

## 参见

+   Boost 的官方文档包含更多示例，并描述了`Boost.Optional`的先进特性（如使用工厂函数的就地构造）。文档可在以下链接找到：

    [`www.boost.org/doc/libs/1_53_0/libs/optional/doc/html/index.html`](http://www.boost.org/doc/libs/1_53_0/libs/optional/doc/html/index.html)

# 从函数返回数组

让我们玩一个猜谜游戏！你能从以下函数中了解到什么？

```cpp
char* vector_advance(char* val);
```

应该由程序员来释放返回值吗？函数是否尝试释放输入参数？输入参数应该是以零结尾，还是函数应该假设输入参数具有指定的宽度？

现在，让我们使任务更难！看看以下行：

```cpp
char ( &vector_advance( char (&val)[4] ) )[4];
```

请不要担心；在得到这里发生的事情的想法之前，我也一直在挠头半小时。`vector_advance`是一个接受并返回四个元素数组的函数。有没有办法清楚地写出这样的函数？

## 准备工作

本食谱只需要基本的 C++知识。

## 如何做到...

我们可以像这样重写函数：

```cpp
#include <boost/array.hpp>
typedef boost::array<char, 4> array4_t;array4_t& vector_advance(array4_t& val);
```

在这里，`boost::array<char, 4>`只是围绕四个字符元素的数组的一个简单包装器。

这段代码回答了我们第一个示例中的所有问题，并且比第二个示例更易于阅读。

## 它是如何工作的...

`boost::array` 的第一个模板参数是元素类型，第二个是数组的大小。`boost::array` 是一个固定大小的数组；如果需要在运行时更改数组大小，请使用 `std::vector` 或 `boost::container::vector`。

`Boost.Array` 库只包含一个数组。仅此而已。简单且高效。`boost::array<>` 类没有手写的构造函数，并且所有成员都是公共的，因此编译器会将其视为 POD 类型。

![工作原理...](img/4880OS_01_new.jpg)

## 还有更多...

让我们看看 `boost::array` 的一些更多使用示例：

```cpp
#include <boost/array.hpp>
#include <algorithm>

// Functional object to increment value by one
struct add_1 : public std::unary_function<char, void> {
    void operator()(char& c) const {
        ++ c;
    }
    // If you're not in a mood to write functional objects,
    // but don't know what does 'boost::bind(std::plus<char>(),
    // _1, 1)' do, then read recipe 'Binding a value as a function 
    // parameter'.
};

typedef boost::array<char, 4> array4_t;
array4_t& vector_advance(array4_t& val) {
    // boost::array has begin(), cbegin(), end(), cend(), 
    // rbegin(), size(), empty() and other functions that are 
    // common for STL containers.
    std::for_each(val.begin(), val.end(), add_1());
    return val;
}

int main() {
    // We can initialize boost::array just like an array in C++11:
    // array4_t val = {0, 1, 2, 3};
    // but in C++03 additional pair of curly brackets is required.
    array4_t val = {{0, 1, 2, 3}};

    // boost::array works like a usual array:
    array4_t val_res;       // it can be default constructible and
    val_res = vector_advance(val);  // assignable
    // if value type supports default construction and assignment

    assert(val.size() == 4);
    assert(val[0] == 1);
    /*val[4];*/ // Will trigger an assert because max index is 3
    // We can make this assert work at compile-time.
    // Interested? See recipe 'Checking sizes at compile time' 
    // in Chapter 4, Compile-time Tricks.'
    assert(sizeof(val) == sizeof(char) * array4_t::static_size);
    return 0;
}
```

`boost::array` 的最大优点之一是它提供了与普通 C 数组完全相同的性能。C++ 标准委员会的人也喜欢它，所以它被纳入了 C++11 标准。有可能你的 STL 库已经包含了它（你可以尝试包含 `<array>` 头文件并检查 `std::array<>` 的可用性）。

## 参见

+   Boost 的官方文档提供了 `Boost.Array` 方法的完整列表，包括方法的复杂性和抛出行为描述，并可在以下链接找到：

    [`www.boost.org/doc/libs/1_53_0/doc/html/boost/array.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/boost/array.html)

+   `boost::array` 函数在各个食谱中广泛使用；例如，可以参考 *将值作为函数参数绑定* 的食谱。

# 将多个值组合成一个

对于喜欢 `std::pair` 的人来说，有一个非常好的礼物。Boost 有一个名为 `Boost.Tuple` 的库，它就像 `std::pair` 一样，但它还可以与三元组、四元组以及更大的类型集合一起工作。

## 准备工作

本食谱只需要对 C++ 和 STL 有基本了解。

## 如何做到这一点...

执行以下步骤以将多个值组合成一个：

1.  要开始使用元组，你需要包含适当的头文件并声明一个变量：

    ```cpp
    #include <boost/tuple/tuple.hpp>
    #include <string>

    boost::tuple<int, std::string> almost_a_pair(10, "Hello");
    boost::tuple<int, float, double, int> quad(10, 1.0f, 10.0, 1);
    ```

1.  通过 `boost::get<N>()` 函数实现获取特定值，其中 `N` 是所需值的零基于索引：

    ```cpp
        int i = boost::get<0>(almost_a_pair);
        const std::string& str = boost::get<1>(almost_a_pair);
        double d = boost::get<2>(quad);
    ```

    `boost::get<>` 函数有许多重载，并在 Boost 中广泛使用。我们已经在 *在容器/变量中存储多个选择类型* 的食谱中看到了它是如何与其他库一起使用的。

1.  你可以使用 `boost::make_tuple()` 函数来构造元组，这比完全限定元组类型要短，因为你不需要完全限定元组类型：

    ```cpp
        using namespace boost;

        // Tuple comparison operators are
        // defined in header "boost/tuple/tuple_comparison.hpp"
        // Don't forget to include it!
        std::set<tuple<int, double, int> > s;
        s.insert(make_tuple(1, 1.0, 2));
        s.insert(make_tuple(2, 10.0, 2));
        s.insert(make_tuple(3, 100.0, 2));

        // Requires C++11
        auto t = make_tuple(0, -1.0, 2);
        assert(2 == get<2>(t));
        // We can make a compile-time assert for type
        // of t. Interested? See chapter 'compile-time tricks'
    ```

1.  另一个让生活变得容易的函数是 `boost::tie()`。它几乎与 `make_tuple` 一样工作，但为每个传递的类型添加了一个非 const 引用。这样的元组可以用来从另一个元组获取值。以下示例可以更好地理解它：

    ```cpp
        boost::tuple<int, float, double, int> quad(10, 1.0f, 10.0, 1);
        int i;
        float f;
        double d;
        int i2;

        // Passing values from 'quad' variables
        // to variables 'i', 'f', 'd', 'i2'
        boost::tie(i, f, d, i2) = quad;
        assert(i == 10);
        assert(i2 == 1);
    ```

## 工作原理...

一些读者可能会 wonder 为什么我们需要元组，因为我们总是可以编写自己的结构体，例如，而不是编写 `boost::tuple<int, std::string>`，我们可以创建一个结构体：

```cpp
struct id_name_pair {
    int id;
    std::string name;
};
```

好吧，这个结构肯定比 `boost::tuple<int, std::string>` 更清晰。但假设这个结构在代码中只使用两次呢？

元组库背后的主要思想是简化模板编程。

![如何工作...](img/4880OS_01_new.jpg)

## 还有更多...

一个元组的工作速度与 `std::pair` 相当（它不在堆上分配内存，也没有虚拟函数）。C++ 委员会认为这个类非常有用，并将其包含在 STL 中；你可以在 C++11 兼容的 STL 实现的 `<tuple>` 头文件中找到它（别忘了将所有 `boost::` 命名空间替换为 `std::`）。

当前 Boost 的元组实现不使用变长模板；它只是由脚本生成的一组类。有一个实验版本使用 C++11 的右值和 C++03 编译器的它们仿真，所以有可能 Boost 1.54 将包含更快的元组实现。

## 参见

+   可以在以下链接找到元组的实验版本：

    [`svn.boost.org/svn/boost/sandbox/tuple-move/`](http://svn.boost.org/svn/boost/sandbox/tuple-move/)

+   Boost 的官方文档包含了更多示例、关于性能和 `Boost.Tuple` 能力的信息。它可在以下链接找到：

    [`www.boost.org/doc/libs/1_53_0/libs/tuple/doc/tuple_users_guide.html`](http://www.boost.org/doc/libs/1_53_0/libs/tuple/doc/tuple_users_guide.html)

+   在 第八章 的 *元编程* 中，*将所有元组元素转换为字符串* 的菜谱展示了元组的某些高级用法

# 重新排序函数的参数

本菜谱和下一菜谱致力于一个非常有趣的库，其功能乍一看像某种魔法。这个库被称为 `Boost.Bind`，它允许您轻松地从函数、成员函数和功能对象创建新的功能对象，同时也允许重新排序初始函数的输入参数，并将某些值或引用绑定为函数参数。

## 准备工作

此菜谱需要具备 C++、STL 算法和功能对象的了解。

## 如何做到这一点...

1.  让我们从一个例子开始。你正在使用某个程序员提供的整数类型向量。这个整数类型只有一个操作符 `+`，但你的任务是乘以一个值。没有 `bind`，这可以通过使用功能对象来实现：

    ```cpp
    class Number{};
    inline Number operator + (Number, Number);

    // Your code starts here
    struct mul_2_func_obj: public std::unary_function<Number, Number> {
        Number operator()(Number n1) const {
            return n1 + n1;
        }
    };

    void mul_2_impl1(std::vector<Number>& values) {
        std::for_each(values.begin(), values.end(), mul_2_func_obj());
    }
    ```

    使用 `Boost.Bind`，可以这样：

    ```cpp
    #include <boost/bind.hpp>
    #include <functional>

    void mul_2_impl2(std::vector<Number>& values) {
       std::for_each(values.begin(), values.end(),
           boost::bind(std::plus<Number>(), _1, _1));
    }
    ```

1.  顺便说一下，我们可以轻松地使这个函数更通用：

    ```cpp
    template <class T>
    void mul_2_impl3(std::vector<T>& values) {
       std::for_each(values.begin(), values.end(),
           boost::bind(std::plus<T>(), _1, _1));
    }
    ```

## 如何工作...

让我们更仔细地看看 `mul_2` 函数。我们向它提供一个值的向量，并为每个值应用 `bind()` 函数返回的函数对象。`bind()` 函数接受三个参数；第一个参数是 `std::plus<Number>` 类的实例（它是一个函数对象）。第二个和第三个参数是占位符。占位符 `_1` 用结果函数对象的第一个输入参数替换参数。正如你可能猜到的，有许多占位符；占位符 `_2` 表示用结果函数对象的第二个输入参数替换参数，同样也适用于占位符 `_3`。嗯，看来你已经明白了这个概念。

## 还有更多...

为了确保你完全理解并知道 bind 可以在哪里使用，让我们看看另一个例子。

我们有两个类，它们与一些传感器设备一起工作。这些设备和类来自不同的供应商，因此它们提供了不同的 API。这两个类只有一个公共方法 `watch`，它接受一个函数对象：

```cpp
class Device1 {
private:
    short temperature();
    short wetness();
    int illumination();
    int atmospheric_pressure();
    void wait_for_data();
public:
    template <class FuncT>
    void watch(const FuncT& f) {
        for(;;) {
            wait_for_data();
            f(
                temperature(),
                wetness(),
                illumination(),
                atmospheric_pressure()
            );
        }
    }
};

class Device2 {
private:
    short temperature();
    short wetness();
    int illumination();
    int atmospheric_pressure();
    void wait_for_data();
public:
    template <class FuncT>
    void watch(const FuncT& f) {
        for(;;) {
            wait_for_data();
            f(
                wetness(),
                temperature(),
                atmospheric_pressure(),
                illumination()
            );
        }
    }
};
```

`Device1::watch` 和 `Device2::watch` 函数以不同的顺序将值传递给函数对象。

一些其他库提供了一个用于检测风暴的函数，当风暴风险足够高时，它会抛出一个异常：

```cpp
void detect_storm(int wetness, int temperature, int atmospheric_pressure);
```

你的任务是为这两个设备提供一个风暴检测函数。以下是使用 `bind` 函数实现的方法：

```cpp
    Device1 d1;
    // resulting functional object will silently ignore 
    // additional parameters passed to function call
    d1.watch(boost::bind(&detect_storm, _2, _1, _4));
    ...
    Device2 d2;
    d2.watch(boost::bind(&detect_storm, _1, _2, _3));
```

`Boost.Bind` 库提供了良好的性能，因为它不使用动态分配和虚函数。即使 C++11 的 lambda 函数不可用，它也非常有用：

```cpp
template <class FuncT>
void watch(const FuncT& f) {
    f(10, std::string("String"));
    f(10, "Char array");
    f(10, 10);
}

struct templated_foo {
    template <class T>
    void operator()(T, int) const {
        // No implementation, just showing that bound
        // functions still can be used as templated
    }
};

void check_templated_bind() {
    // We can directly specify return type of a functional object
    // when bind fails to do so
    watch(boost::bind<void>(templated_foo(), _2, _1));
}
```

Bind 是 C++11 标准的一部分。它在 `<functional>` 头文件中定义，并且可能与 `Boost.Bind` 实现略有不同（然而，它至少与 Boost 的实现一样有效）。

## 参考以下内容

+   “将值绑定为函数参数”食谱对 `Boost.Bind` 的特性有更多介绍

+   Boost 的官方文档包含更多示例和高级特性的描述。它可在以下链接找到：

    [`www.boost.org/doc/libs/1_53_0/libs/bind/bind.html`](http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html)

# 将值绑定为函数参数

如果你经常与 STL 库一起工作并使用 `<algorithm>` 头文件，你肯定会写很多函数对象。你可以使用一组 STL 适配器函数（如 `bind1st`、`bind2nd`、`ptr_fun`、`mem_fun` 和 `mem_fun_ref`）来构建它们，或者你可以手动编写它们（因为适配器函数看起来很吓人）。这里有一些好消息：`Boost.Bind` 可以替代所有这些函数，并提供更易读的语法。

## 准备工作

阅读前面的食谱以了解占位符的概念，或者确保你熟悉 C++11 占位符。了解 STL 函数和算法知识将受到欢迎。

## 如何做到这一点...

让我们看看 `Boost.Bind` 与传统 STL 类一起使用的示例：

1.  按照以下代码计算大于或等于 5 的值：

    ```cpp
    boost::array<int, 12> values = {{1, 2, 3, 4, 5, 6, 7, 100, 99, 98, 97, 96}};

    std::size_t count0 = std::count_if(values.begin(), values.end(),
          std::bind1st(std::less<int>(), 5));
    std::size_t count1 = std::count_if(values.begin(), values.end(),
          boost::bind(std::less<int>(), 5, _1));
    assert(count0 == count1);
    ```

1.  这是我们如何计算空字符串的方法：

    ```cpp
    boost::array<std::string, 3>  str_values = {{"We ", "are", " the champions!"}};
    count0 = std::count_if(str_values.begin(), str_values.end(),
          std::mem_fun_ref(&std::string::empty));
    count1 = std::count_if(str_values.begin(), str_values.end(),
          boost::bind(&std::string::empty, _1));
    assert(count0 == count1);
    ```

1.  现在让我们计算长度小于 `5` 的字符串：

    ```cpp
    // That code won't compile! And it is hard to understand
    //count0 = std::count_if(str_values.begin(), 
    //str_values.end(),
    //std::bind2nd(
    //    std::bind1st(
    //        std::less<std::size_t>(),
    //        std::mem_fun_ref(&std::string::size)
    //    )
    //, 5
    //));
    // This will become much more readable,
    // when you get used to bind
    count1 = std::count_if(str_values.begin(), str_values.end(),
        boost::bind(std::less<std::size_t>(), 
        boost::bind(&std::string::size, _1), 5));
    assert(2 == count1);
    ```

1.  比较字符串：

    ```cpp
    std::string s("Expensive copy constructor of std::string will be called when binding");
    count0 = std::count_if(str_values.begin(), str_values.end(), std::bind2nd(std::less<std::string>(), s));
    count1 = std::count_if(str_values.begin(), str_values.end(), boost::bind(std::less<std::string>(), _1, s));
    assert(count0 == count1);
    ```

## 它是如何工作的...

`boost::bind` 函数返回一个功能对象，该对象存储了绑定值的副本和原始功能对象的副本。当实际执行 `operator()` 调用时，存储的参数会传递给原始功能对象，同时也会传递调用时传递的参数。

## 还有更多...

看一下前面的例子。当我们绑定值时，我们会将一个值复制到一个功能对象中。对于某些类，这个操作可能很昂贵。有没有一种方法可以绕过复制？

是的！而且 `Boost.Ref` 库将帮助我们！它包含两个函数，`boost::ref()` 和 `boost::cref()`，前者允许我们将参数作为引用传递，后者将参数作为常量引用传递。`ref()` 和 `cref()` 函数只是构造了一个类型为 `reference_wrapper<T>` 或 `reference_wrapper<const T>` 的对象，它可以隐式转换为引用类型。让我们修改我们之前的例子：

```cpp
#include <boost/ref.hpp>
...
std::string s("Expensive copy constructor of std::string now "
             "won't be called when binding");
count0 = std::count_if(str_values.begin(), str_values.end(), std::bind2nd(std::less<std::string>(), boost::cref(s)));
count1 = std::count_if(str_values.begin(), str_values.end(), boost::bind(std::less<std::string>(), _1, boost::cref(s)));
assert(count0 == count1);
```

再举一个例子，展示如何使用 `boost::ref` 来连接字符串：

```cpp
void wierd_appender(std::string& to, const std::string& from) {
    to += from;
};

std::string result;
std::for_each(str_values.cbegin(), str_values.cend(), boost::bind(&wierd_appender, boost::ref(result), _1));
assert(result == "We are the champions!");
```

函数 `ref`、`cref`（以及 `bind`）被接受到 C++11 标准中，并在 `std::` 命名空间中的 `<functional>` 头文件中定义。这些函数都不在堆上动态分配内存，也不使用虚函数。它们返回的对象易于优化，并且不会为好的编译器应用任何优化屏障。

这些函数的 STL 实现可能有一些额外的优化，以减少编译时间或只是针对特定编译器的优化，但遗憾的是，一些 STL 实现缺少 Boost 版本的某些功能。你可以使用任何 Boost 库中的 STL 版本，甚至混合 Boost 和 STL 版本。

## 参见

+   `Boost.Bind` 库在这本书中得到了广泛的应用；请参阅第六章 “处理任务” 和第五章 “多线程”，以获取更多示例

+   官方文档包含更多示例和高级特性的描述，请参阅 [`www.boost.org/doc/libs/1_53_0/libs/bind/bind.html`](http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html)

# 使用 C++11 移动模拟

C++11 标准最伟大的特性之一是右值引用。这个特性允许我们修改临时对象，从它们那里“窃取”资源。正如你所猜到的，C++03 标准没有右值引用，但使用 `Boost.Move` 库，你可以编写一些可移植的代码来使用它们，甚至更多，你实际上可以开始模拟移动语义。

## 准备工作

强烈建议至少了解 C++11 rvalue references 的基础知识。

## 如何做到这一点...

现在，让我们看看以下示例：

1.  想象一下，你有一个具有多个字段（其中一些是 STL 容器）的类。

    ```cpp
    namespace other {
        // Its default construction is cheap/fast
        class characteristics{};
    } // namespace other

    struct person_info {
        // Fields declared here
        // ...
        bool is_male_;
        std::string name_;
        std::string second_name_;
        other::characteristics characteristic_;
    };
    ```

1.  是时候给它添加移动赋值和移动构造函数了！只需记住，在 C++03 中，STL 容器既没有移动操作符也没有移动构造函数。

1.  正确的移动赋值实现与 `swap` 和 `clear`（如果允许空状态）相同。正确的移动构造函数实现接近默认构造和 `swap`。所以，让我们从 `swap` 成员函数开始：

    ```cpp
    #include <boost/swap.hpp>

        void swap(person_info& rhs) {
            std::swap(is_male_, rhs.is_male_);
            name_.swap(rhs.name_);
            second_name_.swap(rhs.second_name_);
            boost::swap(characteristic_, rhs.characteristic_);
        }
    ```

1.  现在，将以下宏放在 `private` 部分：

    ```cpp
    BOOST_COPYABLE_AND_MOVABLE(classname)
    ```

1.  编写一个拷贝构造函数。

1.  编写一个拷贝赋值，参数为 `BOOST_COPY_ASSIGN_REF(classname)`。

1.  编写一个移动构造函数和一个移动赋值，参数为 `BOOST_RV_REF(classname)`：

    ```cpp
    struct person_info {
        // Fields declared here
        // ...
    private:
        BOOST_COPYABLE_AND_MOVABLE(person_info)
    public:
        // For the simplicity of example we will assume that 
        // person_info default constructor and swap are very 
        // fast/cheap to call
        person_info() {}

        person_info(const person_info& p)
            : is_male_(p.is_male_)
            , name_(p.name_)
            , second_name_(p.second_name_)
            , characteristic_(p.characteristic_)
        {}

        person_info(BOOST_RV_REF(person_info) person) {
            swap(person);
        }

        person_info& operator=(BOOST_COPY_ASSIGN_REF(person_info) person) {
            if (this != &person) {
                 person_info tmp(person);
                 swap(tmp);
             }
            return *this;
        }

        person_info& operator=(BOOST_RV_REF(person_info) person) {
            if (this != &person) {
                 swap(person);
                 person_info tmp;
                 tmp.swap(person);
             }
            return *this;
        }

        void swap(person_info& rhs) {
        // …
        }

    };
    ```

1.  现在，我们有了 `person_info` 类的移动赋值和移动构造函数的可移植、快速实现。

## 它是如何工作的...

下面是一个如何使用移动赋值的例子：

```cpp
    person_info vasya;
    vasya.name_ = "Vasya";
    vasya.second_name_ = "Snow";
    vasya.is_male_ = true;

    person_info new_vasya(boost::move(vasya));
    assert(new_vasya.name_ == "Vasya");
    assert(new_vasya.second_name_ == "Snow");
    assert(vasya.name_.empty());
    assert(vasya.second_name_.empty());

    vasya = boost::move(new_vasya);
    assert(vasya.name_ == "Vasya");
    assert(vasya.second_name_ == "Snow");
    assert(new_vasya.name_.empty());
    assert(new_vasya.second_name_.empty());
```

`Boost.Move` 库以非常高效的方式实现。当使用 C++11 编译器时，所有用于 rvalue 模拟的宏都将扩展为 C++11 特定功能，否则（在 C++03 编译器上）rvalue 将使用特定的数据类型和函数进行模拟，这些函数永远不会复制传递的值，也不会调用任何动态内存分配或虚拟函数。

## 还有更多...

你注意到 `boost::swap` 调用了吗？这是一个非常有用的实用函数，它将首先在变量的命名空间中（命名空间 `other::`）搜索 `swap` 函数，如果没有为 `characteristics` 类提供 `swap` 函数，它将使用 STL 的 `swap` 实现。

## 参见

+   更多关于模拟实现的信息可以在 Boost 网站上找到，也可以在 `Boost.Move` 库的源代码中找到，链接为 [`www.boost.org/doc/libs/1_53_0/doc/html/move.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/move.html)。

+   `Boost.Utility` 库是包含 `boost::utility` 的库，它有许多有用的函数和类。请参阅其文档，链接为 [`www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm`](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm)。

+   在第三章，*管理资源*中，*通过派生类的成员初始化基类*的配方。

+   *创建不可拷贝的类*配方。

+   在*创建不可拷贝但可移动的类*配方中，有更多关于 `Boost.Move` 的信息，以及一些关于如何在容器中以可移植和高效的方式使用可移动对象的示例。

# 创建不可拷贝的类

你几乎肯定遇到过这样的情况，为类提供拷贝构造函数和移动赋值操作符需要做太多工作，或者类拥有一些由于技术原因不能复制的资源：

```cpp
class descriptor_owner {
    void* descriptor_;

public:
    explicit descriptor_owner(const char* params);

    ~descriptor_owner() {
        system_api_free_descriptor(descriptor_);
    }
};
```

在前一个示例中，C++ 编译器将生成一个复制构造函数和赋值运算符，因此 `descriptor_owner` 类的潜在用户将能够创建以下糟糕的东西：

```cpp
    descriptor_owner d1("O_o");  
    descriptor_owner d2("^_^");

    // Descriptor of d2 was not correctly freed
    d2 = d1;

    // destructor of d2 will free the descriptor
    // destructor of d1 will try to free already freed descriptor
```

## 准备工作

对于这个配方，只需要非常基础的 C++ 知识。

## 如何做到...

为了避免这种情况，发明了 `boost::noncopyable` 类。如果你从它派生自己的类，C++ 编译器将不会生成复制构造函数和赋值运算符：

```cpp
#include <boost/noncopyable.hpp>

class descriptor_owner_fixed : private boost::noncopyable {
    …
```

现在，用户将无法做坏事：

```cpp
    descriptor_owner_fixed d1("O_o");
    descriptor_owner_fixed d2("^_^");
    // Won't compile
    d2 = d1;
    // Won't compile either
    descriptor_owner_fixed d3(d1);
```

## 它是如何工作的...

精通读者会告诉我，我们可以通过将 `descriptor_owning_fixed` 的复制构造函数和赋值运算符设为私有，或者只是定义它们而不实现它们来达到完全相同的结果。是的，你是正确的。此外，这是 `boost::noncopyable` 类的当前实现。但 `boost::noncopyable` 也为你的类提供了良好的文档。它永远不会提出诸如“复制构造函数体是否在其他地方定义？”或“它是否有非标准的复制构造函数（带有非 const 引用的参数）？”等问题。

## 参见

+   *创建一个不可复制但可移动的类* 配方将给你一些想法，如何在 C++03 中通过移动来允许资源的唯一所有权。

+   你可以在 [`www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm`](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm) 的 `Boost.Utility` 库的官方文档中找到很多有用的函数和类。

+   在 第三章 *管理资源* 中，*通过派生类的成员初始化基类* 的配方。

+   *使用 C++11 移动仿真的配方*

# 创建一个不可复制但可移动的类

现在想象以下情况：我们有一个不能复制的资源，它应该在析构函数中正确释放，并且我们希望从函数中返回它：

```cpp
descriptor_owner construct_descriptor() {
    return descriptor_owner("Construct using this string");
}
```

实际上，你可以使用 `swap` 方法来规避这种情况：

```cpp
void construct_descriptor1(descriptor_owner& ret) {
    descriptor_owner("Construct using this string").swap(ret);
}
```

但这样的规避方法不会允许我们在 STL 或 Boost 容器中使用 `descriptor_owner`。顺便说一下，这看起来很糟糕！

## 准备工作

强烈建议至少了解 C++11 右值引用的基础知识。阅读 *使用 C++11 移动仿真的配方* 也是推荐的。

## 如何做到...

那些已经使用 C++11 的读者已经知道关于只移动类（如 `std::unique_ptr` 或 `std::thread`）。使用这种方法，我们可以创建一个只移动的 `descriptor_owner` 类：

```cpp
class descriptor_owner1 {
    void* descriptor_;

public:
    descriptor_owner1()
        : descriptor_(NULL)
    {}

    explicit descriptor_owner1(const char* param)
        : descriptor_(strdup(param))
    {}

    descriptor_owner1(descriptor_owner1&& param)
        : descriptor_(param.descriptor_)
    {
        param.descriptor_ = NULL;
    }

    descriptor_owner1& operator=(descriptor_owner1&& param) {
        clear();
        std::swap(descriptor_, param.descriptor_);
        return *this;
    }

    void clear() {
        free(descriptor_);
        descriptor_ = NULL;
    }

    bool empty() const {
        return !descriptor_;
    }

    ~descriptor_owner1() {
        clear();
    }
};

// GCC compiles the following in with -std=c++0x
descriptor_owner1 construct_descriptor2() {
    return descriptor_owner1("Construct using this string");
}

void foo_rv() {
    std::cout << "C++11\n";
    descriptor_owner1 desc;
    desc = construct_descriptor2();
    assert(!desc.empty());
}
```

这只会在与 C++11 兼容的编译器上工作。这正是使用 `Boost.Move` 的正确时机！让我们修改我们的示例，使其可以在 C++03 编译器上使用。

根据文档，为了用可移植语法编写一个可移动但不可复制的类型，我们需要遵循以下简单的步骤：

1.  在 `private` 部分 put `BOOST_MOVABLE_BUT_NOT_COPYABLE(classname)` 宏：

    ```cpp
    class descriptor_owner_movable {
        void* descriptor_;
        BOOST_MOVABLE_BUT_NOT_COPYABLE(descriptor_owner_movable)
    ```

1.  编写一个移动构造函数和移动赋值运算符，参数为 `BOOST_RV_REF(classname)`：

    ```cpp
    #include <boost/move/move.hpp>

    public:
        descriptor_owner_movable()
            : descriptor_(NULL)
        {}

        explicit descriptor_owner_movable(const char* param)
            : descriptor_(strdup(param))
        {}

        descriptor_owner_movable(
          BOOST_RV_REF(descriptor_owner_movable) param)
           : descriptor_(param.descriptor_)
        { 
        param.descriptor_ = NULL;
        }

        descriptor_owner_movable& operator=(
          BOOST_RV_REF(descriptor_owner_movable) param)
        {
          clear();
          std::swap(descriptor_, param.descriptor_);
          return *this;
        }
        // ...
    };

    descriptor_owner_movable construct_descriptor3() {
        return descriptor_owner_movable("Construct using this string");
    }
    ```

## 它是如何工作的...

现在我们有一个可移动但不可复制的类，它甚至可以在 C++03 编译器和 `Boost.Containers` 中使用：

```cpp
#include <boost/container/vector.hpp>
...
    // Following code will work on C++11 and C++03 compilers
    descriptor_owner_movable movable;
    movable = construct_descriptor3();
    boost::container::vector<descriptor_owner_movable> vec;
    vec.resize(10);
    vec.push_back(construct_descriptor3());

    vec.back() = boost::move(vec.front());
```

但遗憾的是，C++03 STL 容器仍然无法使用它（这就是为什么我们在上一个示例中使用了 `Boost.Containers` 中的向量）。

## 还有更多...

如果你想在 C++03 编译器和 STL 容器中使用 `Boost.Containers`，在 C++11 编译器上，你可以使用以下简单的技巧。将以下内容的头文件添加到你的项目中：

```cpp
// your_project/vector.hpp
// Copyright and other stuff goes here

// include guards
#ifndef YOUR_PROJECT_VECTOR_HPP
#define YOUR_PROJECT_VECTOR_HPP

#include <boost/config.hpp>

// Those macro declared in boost/config.hpp header
// This is portable and can be used with any version of boost 
// libraries
#if !defined(BOOST_NO_RVALUE_REFERENCES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
// We do have rvalues
#include <vector>

namespace your_project_namespace {
  using std::vector;
} // your_project_namespace

#else
// We do NOT have rvalues
#include <boost/container/vector.hpp>

namespace your_project_namespace {
  using boost::container::vector;
} // your_project_namespace

#endif // !defined(BOOST_NO_RVALUE_REFERENCES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#endif // YOUR_PROJECT_VECTOR_HPP
```

现在，你可以包含 `<your_project/vector.hpp>` 并使用 `your_project_namespace` 命名空间中的向量：

```cpp
    your_project_namespace::vector<descriptor_owner_movable> v;
    v.resize(10);
    v.push_back(construct_descriptor3());
    v.back() = boost::move(v.front());
```

但请注意编译器和 STL 实现特定的问题！例如，只有当你将移动构造函数、析构函数和移动赋值运算符标记为 `noexcept` 时，这段代码才会在 GCC 4.7 的 C++11 模式下编译。

## 参见

+   在第十章 Chapter 10. Gathering Platform and Compiler Information 的 *Reducing code size and increasing performance of user-defined types (UDTs) in C++11* 节中，可以找到更多关于 `noexcept` 的信息。

+   关于 `Boost.Move` 的更多信息可以在 Boost 网站上找到 [`www.boost.org/doc/libs/1_53_0/doc/html/move.html`](http://www.boost.org/doc/libs/1_53_0/doc/html/move.html)
