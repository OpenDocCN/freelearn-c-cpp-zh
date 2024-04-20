# 第十九章：STL 支持和提案

自从 90 年代以来，**标准模板库**（**STL**）一直是 C++程序员的有用伴侣。从泛型编程和值语义等概念开始，它已经发展到支持许多有用的场景。在本章中，我们将看看 STL 如何支持 C++ 17 中的函数式编程，并了解一些在 C++ 20 中引入的新特性。

本章将涵盖以下主题：

+   使用`<functional>`头文件中的函数式特性

+   使用`<numeric>`头文件中的函数式特性

+   使用`<algorithm>`头文件中的函数式特性

+   `std::optional`和`std::variant`

+   C++20 和 ranges 库

# 技术要求

你需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.4.0c。

代码在 GitHub 上的[https:/​/​github.​com/​PacktPublishing/​Hands-​On-​Functional-Programming-​with-​Cpp](https://github.%E2%80%8Bcom/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)的`Chapter15`文件夹中。它包括并使用了`doctest`，这是一个单头开源单元测试库。你可以在它的 GitHub 仓库中找到它：[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# `<functional>`头文件

我们需要从 STL 中的函数式编程支持中的某个地方开始，而名为`<functional>`的头文件似乎是一个不错的起点。这个头文件定义了基本的`function<>`类型，我们可以用它来表示函数，并且在本书中的几个地方已经使用过了 lambda 表达式：

```cpp
TEST_CASE("Identity function"){
    function<int(int)> identity = [](int value) { return value;};

    CHECK_EQ(1, identity(1));
}
```

我们可以使用`function<>`类型来存储任何类型的函数，无论是自由函数、成员函数还是 lambda。让我们看一个自由函数的例子：

```cpp
TEST_CASE("Free function"){
    function<int()> f = freeFunctionReturns2;

    CHECK_EQ(2, f());
}
```

这里有一个成员函数的例子：

```cpp
class JustAClass{
    public:
        int functionReturns2() const { return 2; };
};

TEST_CASE("Class method"){
    function<int(const JustAClass&)> f = &JustAClass::functionReturns2;
    JustAClass justAClass;

    CHECK_EQ(2, f(justAClass));
}
```

正如你所看到的，为了通过`function<>`类型调用成员函数，需要传递一个有效的对象引用。可以把它看作是`*this`实例。

除了这种基本类型之外，`<functional>`头文件还提供了一些已定义的函数对象，当在集合上使用函数式转换时非常方便。让我们看一个简单的例子，使用`sort`算法与定义的`greater`函数结合，以便按降序对向量进行排序：

```cpp
TEST_CASE("Sort with predefined function"){
    vector<int> values{3, 1, 2, 20, 7, 5, 14};
    vector<int> expectedDescendingOrder{20, 14, 7, 5, 3,  2, 1};

    sort(values.begin(), values.end(), greater<int>());

    CHECK_EQ(expectedDescendingOrder, values);
}
```

`<functional>`头文件定义了以下有用的函数对象：

+   **算术操作**：`plus`，`minus`，`multiplies`，`divides`，`modulus`和`negate`

+   **比较**：`equal_to`，`not_equal_to`，`greater`，`less`，`greater_equal`和`less_equal`

+   **逻辑操作**：`logical_and`，`logical_or`和`logical_not`

+   **位操作**：`bit_and`，`bit_or`和`bit_xor`

当我们需要将常见操作封装在函数中以便在高阶函数中使用时，这些函数对象可以帮助我们省去麻烦。虽然这是一个很好的集合，但我敢于建议一个恒等函数同样有用，尽管这听起来有些奇怪。幸运的是，实现一个恒等函数很容易。

然而，`<functional>`头文件提供的不仅仅是这些。`bind`函数实现了部分函数应用。我们在本书中多次看到它的应用，你可以在第五章中详细了解它的用法，*部分应用和柯里化*。它的基本功能是接受一个函数，绑定一个或多个参数到值，并获得一个新的函数：

```cpp
TEST_CASE("Partial application using bind"){
    auto add = [](int first, int second){
        return first + second;
    };

    auto increment = bind(add, _1, 1);

    CHECK_EQ(3, add(1, 2));
    CHECK_EQ(3, increment(2));
}
```

有了`function<>`类型允许我们编写 lambda 表达式，预定义的函数对象减少了重复，以及`bind`允许部分应用，我们就有了以函数式方式构造代码的基础。但是如果没有高阶函数，我们就无法有效地这样做。

# <algorithm>头文件

`<algorithm>`头文件包含了一些算法，其中一些实现为高阶函数。在本书中，我们已经看到了许多它们的用法。以下是一些有用的算法列表：

+   `all_of`，`any_of`和`none_of`

+   `find_if`和`find_if_not`

+   `count_if`

+   `copy_if`

+   `generate_n`

+   `sort`

我们已经看到，专注于数据并结合这些高阶函数将输入数据转换为所需的输出是你可以思考的一种方式，这是小型、可组合、纯函数的一种方式。我们也看到了这种方法的缺点——需要复制数据，或者对相同的数据进行多次遍历——以及新的 ranges 库如何以一种优雅的方式解决了这些问题。

虽然所有这些函数都非常有用，但有一个来自`<algorithm>`命名空间的函数值得特别提及——函数式`map`操作`transform`的实现。`transform`函数接受一个输入集合，并对集合的每个元素应用一个 lambda，返回一个具有相同数量元素但其中存储了转换值的新集合。这为我们适应数据结构提供了无限的可能性。让我们看一些例子。

# 从集合中投影每个对象的一个属性

我们经常需要从集合中获取每个元素的属性值。在下面的例子中，我们使用`transform`来获取一个向量中所有人的姓名列表：

```cpp
TEST_CASE("Project names from a vector of people"){
    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14)
    };

    vector<string> expectedNames{"Alex", "John", "Jane"};
    vector<string> names = transformAll<vector<string>>(
            people, 
            [](Person person) { return person.name; } 
    );

    CHECK_EQ(expectedNames, names);
}
```

再次使用`transform`和`transformAll`的包装器，以避免编写样板代码：

```cpp
template<typename DestinationType>
auto transformAll = [](auto source, auto lambda){
    DestinationType result;
    transform(source.begin(), source.end(), back_inserter(result), 
        lambda);
    return result;
};
```

# 计算条件

有时，我们需要计算一组元素是否满足条件。在下面的例子中，我们将通过比较他们的年龄与`18`来计算人们是否未成年：

```cpp
TEST_CASE("Minor or major"){
    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14)
    };

    vector<bool> expectedIsMinor{false, false, true};
    vector<bool> isMinor = transformAll<vector<bool>>(
            people, 
            [](Person person) { return person.age < 18; } 
    );

    CHECK_EQ(expectedIsMinor, isMinor);
}
```

# 将所有内容转换为可显示或可序列化格式

我们经常需要保存或显示一个列表。为了做到这一点，我们需要将列表的每个元素转换为可显示或可序列化的格式。在下面的例子中，我们正在计算列表中的`Person`对象的 JSON 表示：

```cpp
TEST_CASE("String representation"){
    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14)
    };

    vector<string> expectedJSON{
        "{'person': {'name': 'Alex', 'age': '42'}}",
        "{'person': {'name': 'John', 'age': '21'}}",
        "{'person': {'name': 'Jane', 'age': '14'}}"
    };
    vector<string> peopleAsJson = transformAll<vector<string>>(
            people, 
            [](Person person) { 
            return 
            "{'person': {'name': '" + person.name + "', 'age': 
                '" + to_string(person.age) + "'}}"; } 
    );

    CHECK_EQ(expectedJSON, peopleAsJson);
}
```

即使`transform`函数打开了无限的可能性，但与`reduce`（在 C++中为`accumulate`）高阶函数结合使用时，它变得更加强大。

# `<numeric>`头文件 - accumulate

有趣的是，形成`map`/`reduce`模式的两个高阶函数之一，即函数式编程中最常见的模式之一，最终出现在 C++的两个不同的头文件中。`transform`/`accumulate`组合需要`<algorithm>`和`<numeric>`头文件，可以解决许多具有以下模式的问题：

+   提供了一个集合。

+   集合需要转换为其他形式。

+   需要计算一个聚合结果。

让我们看一些例子。

# 计算购物车的含税总价

假设我们有一个`Product`结构，如下所示：

```cpp
struct Product{
    string name;
    string category;
    double price;
    Product(string name, string category, double price): name(name), 
        category(category), price(price){}
};
```

假设我们根据产品类别有不同的税率：

```cpp
map<string, int> taxLevelByCategory = {
    {"book", 5},
    {"cosmetics", 20},
    {"food", 10},
    {"alcohol", 40}
};
```

假设我们有一个产品列表，如下所示：

```cpp
    vector<Product> products = {
        Product("Lord of the Rings", "book", 22.50),
        Product("Nivea", "cosmetics", 15.40),
        Product("apple", "food", 0.30),
        Product("Lagavulin", "alcohol", 75.35)
    };

```

让我们计算含税和不含税的总价。我们还有一个辅助包装器`accumulateAll`可供使用：

```cpp
auto accumulateAll = [](auto collection, auto initialValue,  auto 
    lambda){
        return accumulate(collection.begin(), collection.end(), 
            initialValue, lambda);
};
```

要计算不含税的价格，我们只需要获取所有产品的价格并相加。这是一个典型的`map`/`reduce`场景：

```cpp
   auto totalWithoutTax = accumulateAll(transformAll<vector<double>>
        (products, [](Product product) { return product.price; }), 0.0, 
            plus<double>());
     CHECK_EQ(113.55, doctest::Approx(totalWithoutTax));
```

首先，我们将`Products`列表转换为价格列表，然后将它们进行`reduce`（或`accumulate`）处理，得到一个单一的值——它的总价。

当我们需要含税的总价时，一个类似但更复杂的过程也适用：

```cpp
    auto pricesWithTax = transformAll<vector<double>>(products, 
            [](Product product){
                int taxPercentage = 
                    taxLevelByCategory[product.category];
                return product.price + product.price * 
                    taxPercentage/100;
            });
    auto totalWithTax = accumulateAll(pricesWithTax, 0.0, 
        plus<double> ());
    CHECK_EQ(147.925, doctest::Approx(totalWithTax));
```

首先，我们将`Products`列表与含税价格列表进行`map`（`transform`）处理，然后将所有值进行`reduce`（或`accumulate`）处理，得到含税总价。

如果你想知道，`doctest::Approx`函数允许对浮点数进行小的舍入误差比较。

# 将列表转换为 JSON

在前一节中，我们看到如何通过`transform`调用将列表中的每个项目转换为 JSON。通过`accumulate`的帮助，很容易将其转换为完整的 JSON 列表：

```cpp
    string expectedJSONList = "{people: {'person': {'name': 'Alex', 
        'age': '42'}}, {'person': {'name': 'John', 'age': '21'}}, 
            {'person': {'name': 'Jane', 'age': '14'}}}"; 
    string peopleAsJSONList = "{people: " + accumulateAll(peopleAsJson, 
        string(),
            [](string first, string second){
                return (first.empty()) ? second : (first + ", " + 
                    second);
            }) + "}";
    CHECK_EQ(expectedJSONList, peopleAsJSONList);
```

我们使用`transform`将人员列表转换为每个对象的 JSON 表示的列表，然后我们使用`accumulate`将它们连接起来，并使用一些额外的操作来添加 JSON 中列表表示的前后部分。

正如你所看到的，`transform`/`accumulate`（或`map`/`reduce`）组合可以根据我们传递给它的函数执行许多不同的用途。

# 回到<algorithm> – find_if 和 copy_if

我们可以通过`transform`、`accumulate`和`any_of`/`all_of`/`none_of`实现很多事情。然而，有时我们需要从集合中过滤掉一些数据。

通常的做法是使用`find_if`。然而，如果我们需要找到集合中符合特定条件的所有项目，`find_if`就显得很麻烦了。因此，使用 C++ 17 标准以函数式方式解决这个问题的最佳选择是`copy_if`。以下示例使用`copy_if`在人员列表中找到所有未成年人：

```cpp
TEST_CASE("Find all minors"){
    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14),
        Person("Diana", 9)
    };

    vector<Person> expectedMinors{Person("Jane", 14), 
                                  Person("Diana", 9)};

    vector<Person> minors;
    copy_if(people.begin(), people.end(), back_inserter(minors), []
        (Person& person){ return person.age < 18; });

    CHECK_EQ(minors, expectedMinors);
}
```

# <optional>和<variant>

我们已经讨论了很多快乐路径的情况，即数据对我们的数据转换是有效的情况。那么对于边缘情况和错误情况，我们该怎么办呢？当然，在特殊情况下，我们可以抛出异常或返回错误情况，但是在我们需要返回错误消息的情况下呢？

在这些情况下，功能性的方式是返回数据结构。毕竟，即使输入无效，我们也需要返回一个输出值。但我们面临一个挑战——在错误情况下我们需要返回的类型是错误类型，而在有效数据情况下我们需要返回的类型是更多的有效数据。

幸运的是，我们有两种结构在这些情况下支持我们——`std::optional`和`std::variant`。让我们以一个人员列表为例，其中一些人是有效的，一些人是无效的：

```cpp
    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14),
        Person("Diana", 0)
    };
```

最后一个人的年龄无效。让我们尝试以一种功能性的方式编写代码，以显示以下字符串：

```cpp
Alex, major
John, major
Jane, minor
Invalid person
```

要有一系列的转换，我们需要使用`optional`类型，如下所示：

```cpp
struct MajorOrMinorPerson{
    Person person;
    optional<string> majorOrMinor;

    MajorOrMinorPerson(Person person, string majorOrMinor) : 
        person(person), majorOrMinor(optional<string>(majorOrMinor)){};

    MajorOrMinorPerson(Person person) : person(person), 
        majorOrMinor(nullopt){};
};
    auto majorMinorPersons = transformAll<vector<MajorOrMinorPerson>>
        (people, [](Person& person){ 
            if(person.age <= 0) return MajorOrMinorPerson(person);
            if(person.age > 0 && person.age < 18) return 
                MajorOrMinorPerson(person, "minor");
            return MajorOrMinorPerson(person, "major");
            });
```

通过这个调用，我们得到了一个人和一个值之间的配对列表，该值要么是`nullopt`，要么是`minor`，要么是`major`。我们可以在下面的`transform`调用中使用它，以根据有效条件获取字符串列表：

```cpp
    auto majorMinorPersonsAsString = transformAll<vector<string>>
        (majorMinorPersons, [](MajorOrMinorPerson majorOrMinorPerson){
            return majorOrMinorPerson.majorOrMinor ? 
            majorOrMinorPerson.person.name + ", " + 
                majorOrMinorPerson.majorOrMinor.value() :
                    "Invalid person";
            });
```

最后，调用 accumulate 创建了预期的输出字符串：

```cpp
    auto completeString = accumulateAll(majorMinorPersonsAsString, 
        string(), [](string first, string second){
            return first.empty() ? second : (first + "\n" + second);
            });
```

我们可以通过测试来检查这一点：

```cpp
    string expectedString("Alex, major\nJohn, major\nJane, 
                                    minor\nInvalid person");

    CHECK_EQ(expectedString, completeString);
```

如果需要，可以使用`variant`来实现另一种方法，例如，返回与人员组合的错误代码。

# C++ 20 和范围库

我们在第十四章中详细讨论了范围库，*使用范围库进行惰性评估*。如果你可以使用它，要么是因为你使用 C++ 20，要么是因为你可以将它作为第三方库使用，那么前面的函数就变得非常简单且更快：

```cpp
TEST_CASE("Ranges"){
    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14),
        Person("Diana", 0)
    };
    using namespace ranges;

    string completeString = ranges::accumulate(
            people |
            view::transform(personToMajorMinor) | 
            view::transform(majorMinor),
            string(),
            combineWithNewline
           ); 
    string expectedString("Alex, major\nJohn, major\nJane, 
                                    minor\nInvalid person");

    CHECK_EQ(expectedString, completeString);
}
```

同样，从人员列表中找到未成年人的列表在范围的`view::filter`中非常容易：

```cpp
TEST_CASE("Find all minors with ranges"){
    using namespace ranges;

    vector<Person> people = {
        Person("Alex", 42),
        Person("John", 21),
        Person("Jane", 14),
        Person("Diana", 9)
    };
    vector<Person> expectedMinors{Person("Jane", 14),
                                   Person("Diana", 9)};

    vector<Person> minors = people | view::filter(isMinor);

    CHECK_EQ(minors, expectedMinors);
}
```

一旦我们有了`isMinor`谓词，我们可以将它传递给`view::filter`来从人员列表中找到未成年人。

# 摘要

在本章中，我们对 C++ 17 STL 中可用的函数式编程特性进行了介绍，以及 C++ 20 中的新特性。通过函数、算法、`variant`和`optional`在错误或边缘情况下提供的帮助，以及使用范围库可以实现的简化和优化代码，我们对函数式编程特性有了相当好的支持。

现在，是时候进入下一章，看看 C++ 17 对函数式编程的语言支持，以及 C++ 20 中即将出现的有趣的事情了。
