

# 在 C++ 中利用丰富的静态类型系统

在现代软件开发中，“类型”的概念已经超越了其原始定义，演变成一种丰富且富有表现力的语言特性，它不仅封装了数据表示，还包含了更多内容。在以性能和灵活性著称的 C++ 语言中，静态类型系统是一个强大的工具，它使开发者能够编写不仅健壮且高效的代码，而且具有自文档化和可维护性的代码。

类型在 C++ 中的重要性不仅限于对数据的分类。通过强制执行严格的编译时检查，语言的类型系统减少了运行时错误，提高了可读性，并促进了代码的直观理解。随着现代 C++ 标准的出现，利用类型的机会进一步扩大，为常见的编程挑战提供了优雅的解决方案。

然而，这些强大的功能往往没有得到充分利用。原始数据类型，如整数，经常被错误地用来表示时间长度等概念，导致代码缺乏表现力且容易出错。指针虽然灵活，但可能导致空指针解引用问题，使代码库变得脆弱。

在本章中，我们将探讨 C++ 静态类型系统的丰富景观，重点关注帮助减轻这些问题的先进和现代技术。从使用 `<chrono>` 库来表示时间长度到使用 `not_null` 包装器和 `std::optional` 进行更安全的指针处理，我们将深入研究体现强类型本质的实践。

我们还将探讨如 Boost 这样的外部库，它们提供了额外的实用工具来增强类型安全性。在整个章节中，现实世界的例子将说明这些工具和技术如何无缝集成到你的代码中，赋予你充分利用 C++ 类型系统的全部潜力。

到本章结束时，你将深入理解如何利用类型来编写更健壮、可读性和表现力更强的代码，挖掘 C++ 真正的潜力。

# 利用 Chrono 进行时间长度

C++ 类型系统如何被利用来编写更健壮的代码的最好例子之一是 `<chrono>` 库。自 C++11 引入以来，这个头文件提供了一套表示时间长度和时间点的实用工具，以及执行时间相关操作。

使用普通整数或如 `timespec` 这样的结构来管理时间相关函数可能是一种容易出错的途径，尤其是在处理不同时间单位时。想象一个接受表示秒数的整数的函数：

```cpp
void wait_for_data(int timeout_seconds) {
    sleep(timeout_seconds); // Sleeps for timeout_seconds seconds
}
```

这种方法缺乏灵活性，在处理各种时间单位时可能会导致混淆。例如，如果调用者错误地传递了毫秒而不是秒，可能会导致意外的行为。

相比之下，使用 `<chrono>` 定义相同的函数使代码更加健壮和具有表现力：

```cpp
#include <chrono>
#include <thread>
void wait_for_data(std::chrono::seconds timeout) {
    std::this_thread::sleep_for(timeout); // Sleeps for the specified timeout
}
```

调用者现在可以使用强类型持续时间传递超时，例如 `std::chrono::seconds(5)`，编译器确保使用正确的单位。此外，`<chrono>` 提供了不同时间单位之间的无缝转换，允许调用者以秒、毫秒或其他任何单位指定超时，而不存在歧义。以下代码片段展示了使用不同单位的用法：

```cpp
wait_for_data(std::chrono::milliseconds(150));
```

通过采用 `<chrono>` 提供的强类型，代码变得更加清晰、易于维护，并且不太可能受到与时间表示相关的常见错误的影响。

# 使用 `not_null` 和 `std::optional` 提高指针安全性

在 C++ 中，指针是语言的基本组成部分，允许直接访问和操作内存。然而，指针提供的灵活性也伴随着一定的风险和挑战。在这里，我们将探讨现代 C++ 技术如何增强指针安全性。

## 原始指针的陷阱

原始指针虽然强大，但可能是一把双刃剑。它们不会提供关于它们指向的对象所有权的任何信息，并且它们很容易成为“悬空”指针，指向已经被释放的内存。取消引用空指针或悬空指针会导致未定义的行为，这可能导致难以诊断的错误。

## 使用指南支持库中的 `not_null`

由 `not_null` 提供的包装器可以清楚地表明指针不应为空：

```cpp
#include <gsl/gsl>
void process_data(gsl::not_null<int*> data) {
    // Data is guaranteed not to be null here
}
```

如果用户以如下方式将空指针传递给此函数，应用程序将被终止：

```cpp
int main() {
    int* p = nullptr;
    process_data(p); // this will terminate the program
    return 0;
}
```

然而，如果指针作为 `process_data(nullptr)` 传递，应用程序将在编译时失败：

```cpp
source>: In function 'int main()':
<source>:9:16: error: use of deleted function 'gsl::not_null<T>::not_null(std::nullptr_t) [with T = int*; std::nullptr_t = std::nullptr_t]'
    9 |     process_data(nullptr);
      |     ~~~~~~~~~~~^~~~~~~~~
In file included from <source>:1:
/opt/compiler-explorer/libs/GSL/trunk/include/gsl/pointers:131:5: note: declared here
  131 |     not_null(std::nullptr_t) = delete;
      |     ^~~~~~~~
```

这通过在早期捕获潜在的空指针错误来促进代码的健壮性，从而减少运行时错误。

### 将 `not_null` 扩展到智能指针

`gsl::not_null` 不仅限于原始指针；它还可以与智能指针如 `std::unique_ptr` 和 `std::shared_ptr` 结合使用。这允许你结合现代内存管理的优点以及 `not_null` 提供的额外安全保证。

### 使用 `std::unique_ptr`

`std::unique_ptr` 确保动态分配的对象的所有权是唯一的，并且当不再需要对象时，它会自动删除该对象。通过使用 `not_null` 与 `unique_ptr` 结合，你也可以确保指针永远不会为空：

```cpp
#include <gsl/gsl>
#include <memory>
void process_data(gsl::not_null<std::unique_ptr<int>> data) {
    // Data is guaranteed not to be null here
}
int main() {
    auto data = std::make_unique<int>(42);
    process_data(std::move(data)); // Safely passed to the function
}
```

### 使用 `std::shared_ptr`

类似地，`gsl::not_null` 可以与 `std::shared_ptr` 结合使用，这使对象具有共享所有权。这允许你编写接受共享指针的函数，而无需担心空指针：

```cpp
#include <gsl/gsl>
#include <memory>
void process_data(gsl::not_null<std::shared_ptr<int>> data) {
    // Data is guaranteed not to be null here
}
int main() {
    auto data = std::make_shared<int>(42);
    process_data(data); // Safely passed to the function
}
```

这些示例展示了 `not_null` 如何无缝地与现代 C++ 内存管理技术集成。通过强制指针（无论是原始指针还是智能指针）不能为空，你进一步减少了运行时错误的可能性，并使代码更加健壮和易于表达。

## 利用 `std::optional` 处理可选值

有时，指针用于表示可选值，其中`nullptr`表示值的缺失。C++17 引入了`std::optional`，它提供了一种类型安全地表示可选值的方法：

```cpp
#include <optional>
std::optional<int> fetch_data() {
    if (/* some condition */)
        return 42;
    else
        return std::nullopt;
}
```

使用`std::optional`提供清晰的语义并避免使用指针进行此目的时的陷阱。

## 原始指针与 nullptr 的比较

`not_null`和`std::optional`都优于原始指针。虽然原始指针可以是 null 或悬空，导致未定义行为，但`not_null`在编译时防止 null 指针错误，而`std::optional`提供了一种清晰地表示可选值的方法。

考虑以下使用原始指针的示例：

```cpp
int* findValue() {
    // ...
    return nullptr; // No value found
}
```

这段代码可能会导致混淆和错误，尤其是如果调用者忘记检查`nullptr`。通过使用`not_null`和`std::optional`，可以使代码更具表达性且更不易出错。

## 利用 std::expected 获取预期结果和错误

虽然`std::optional`优雅地表示了可选值，但有时你需要传达更多关于值可能缺失的原因的信息。在这种情况下，`std::expected`提供了一种返回值或错误代码的方法，使代码更具表达性，错误处理更健壮。

考虑一个场景，你有一个从网络获取值的函数，并且你想处理网络错误。你可能为各种网络错误定义一个枚举：

```cpp
enum class NetworkError {
    Timeout,
    ConnectionLost,
    UnknownError
};
```

然后，你可以使用`std::expected`定义一个返回`int`值或`NetworkError`的函数：

```cpp
#include <expected>
#include <iostream>
std::expected<int, NetworkError> fetch_data_from_network() {
    // Simulating network operation...
    if (/* network timeout */) {
        return std::unexpected(NetworkError::Timeout);
    }
    if (/* connection lost */) {
        return std::unexpected(NetworkError::ConnectionLost);
    }
    return 42; // Successfully retrieved value
}
int main() {
    auto result = fetch_data_from_network();
    if (result) {
        std::cout << "Value retrieved: " << *result << '\n';
    } else {
        std::cout << "Network error: ";
        switch(result.error()) {
            case NetworkError::Timeout:
                std::cout << "Timeout\n";
                break;
            case NetworkError::ConnectionLost:
                std::cout << "Connection Lost\n";
                break;
            case NetworkError::UnknownError:
                std::cout << "Unknown Error\n";
                break;
        }
    }
}
```

在这里，`std::expected`同时捕获了成功情况和各种错误场景，允许进行清晰且类型安全的错误处理。这个例子说明了现代 C++类型如`std::expected`如何增强表达性和安全性，使你能够编写更精确地模拟复杂操作的代码。

通过采用这些现代 C++工具，你可以增强代码中的指针安全性，减少错误并使你的意图更清晰。

# 使用 enum class 和范围枚举的强类型

强类型是健壮、可维护的软件的基石，C++提供了多种机制来促进其实现。在这些机制中，C++11 中引入的`enum class`是一个特别有效的工具，用于创建强类型枚举，可以使你的程序更健壮且更容易理解。

## 对 enum class 的回顾

C++中的传统枚举存在一些限制——它们可以隐式转换为整数，如果误用可能会导致错误，并且它们的枚举符被引入到封装作用域中，导致名称冲突。`enum class`，也称为范围枚举，解决了这些限制：

```cpp
// Traditional enum
enum ColorOld { RED, GREEN, BLUE };
int color = RED; // Implicit conversion to int
// Scoped enum (enum class)
enum class Color { Red, Green, Blue };
// int anotherColor = Color::Red; // Compilation error: no implicit conversion
```

## 相比传统枚举的优势

范围枚举提供了一些优势：

+   `enum class`类型和整数，确保你不会意外地将枚举符用作整数

+   `enum class`，减少了名称冲突的可能性

+   `enum class` 允许你显式指定底层类型，从而让你对数据表示有精确的控制：

    ```cpp
    enum class StatusCode : uint8_t { Ok, Error, Pending };
    ```

能够指定底层类型对于将数据序列化为二进制格式特别有用。它确保你可以在字节级别对数据的表示有精细的控制，从而便于与可能具有特定二进制格式要求的系统进行数据交换。

## 实际应用场景

`enum class` 的优点使其成为各种场景中的强大工具：

+   `enum class` 提供了一种类型安全、具有表现力的方式来表示各种可能的状态

+   **选项集**：许多函数有多种行为选项，可以使用范围枚举清晰地和安全地封装

+   `enum class`：

    ```cpp
    enum class NetworkStatus { Connected, Disconnected, Error };
    ```

    ```cpp
    NetworkStatus check_connection() {
    ```

    ```cpp
        // Implementation
    ```

    ```cpp
    }
    ```

通过使用 `enum class` 创建强类型、范围枚举，你可以编写不仅更容易理解而且更不容易出错的代码。这一特性代表了 C++持续向结合高性能与现代编程便利性的语言发展的又一步。无论你是定义复杂的有限状态机还是仅仅尝试表示多个选项或状态，`enum class` 都提供了一个健壮、类型安全的解决方案。

# 利用标准库的类型实用工具

现代 C++在标准库中提供了一套丰富的类型实用工具，使开发者能够编写更具有表现力、类型安全和可维护的代码。两个突出的例子是 `std::variant` 和 `std::any`。

## std::variant – 一个类型安全的联合体

`std::variant` 提供了一种类型安全的方式来表示一个值，它可以属于几种可能类型中的一种。与允许程序员将存储的值视为其成员类型之一的传统 `union` 不同，这可能导致潜在的不确定行为，`std::variant` 跟踪当前持有的类型并确保适当的处理：

```cpp
#include <variant>
#include <iostream>
std::variant<int, double, std::string> value = 42;
// Using std::get with an index:
int intValue = std::get<0>(value); // Retrieves the int value
// Using std::get with a type:
try {
    double doubleValue = std::get<double>(value); // Throws std::bad_variant_access
} catch (const std::bad_variant_access& e) {
    std::cerr << "Bad variant access: " << e.what() << '\n';
}
// Using std::holds_alternative:
if (std::holds_alternative<int>(value)) {
    std::cout << "Holding int\n";
} else {
    std::cout << "Not holding int\n";
}
```

### 相比传统联合体的优势

+   `std::variant` 相反，跟踪当前类型并通过如 `std::get` 和 `std::holds_alternative` 等函数提供安全访问。

+   `std::variant` 在你分配新值时自动构造和销毁所持有的对象，正确地管理对象的生命周期。

+   `std::get`，如果抛出 `std::bad_variant_access` 异常，使得错误处理更加透明且易于管理。

+   `std::variant` 可以与标准库函数如 `std::visit` 一起使用，提供优雅地处理各种类型的方法。

## std::any – 任何类型的类型安全容器

`std::any` 是一个可以容纳任何类型的容器，但通过要求显式转换为正确的类型来保持类型安全。这允许灵活地处理数据，同时不牺牲类型完整性：

```cpp
#include <any>
#include <iostream>
#include <stdexcept>
std::any value = 42;
try {
    std::cout << std::any_cast<int>(value); // Outputs 42
    std::cout << std::any_cast<double>(value); // Throws std::bad_any_cast
} catch(const std::bad_any_cast& e) {
    std::cerr << "Bad any_cast: " << e.what();
}
```

使用 `std::any` 的优点包括以下：

+   **灵活性**：它可以存储任何类型，使其适合异构集合或灵活的 API

+   **类型安全**：需要显式转换，防止意外误解包含的值

+   **封装**：允许你传递值而不暴露它们的具体类型，支持更模块化和可维护的代码

## 高级类型技术

随着你对 C++ 的深入研究，你会发现该语言提供了一系列高级技术来增强类型安全、可读性和可维护性。在本节中，我们将探讨这些高级概念中的几个，并为每个提供实际示例。

### 模板 – 为类型安全而特化

模板是 C++ 中的一个强大功能，但你可能希望根据类型施加某些约束或特化。一种方法是通过模板特化来实现，这允许你为某些类型定义自定义行为。

例如，假设你有一个用于在数组中查找最大元素的泛型函数：

```cpp
template <typename T>
T find_max(const std::vector<T>& arr) {
    // generic implementation
    return *std::max_element(arr.begin(), arr.end());
}
```

现在，假设你想为 `std::string` 提供一个不区分大小写的专用实现：

```cpp
template <>
std::string find_max(const std::vector<std::string>& arr) {
    return *std::max_element(arr.begin(), arr.end(),
                              [](const std::string& a, const std::string& b) {
                                  return strcasecmp(a.c_str(), b.c_str()) < 0;
                              });
}
```

使用这个专用版本，调用 `find_max` 并使用 `std::string` 将进行不区分大小写的比较。

### 创建自定义类型特性

有时，标准类型特性可能不足以满足你的需求。你可以创建自己的自定义类型特性来封装基于类型的逻辑。例如，你可能需要一个类型特性来识别一个类是否具有特定的成员函数：

```cpp
template <typename T, typename = void>
struct has_custom_method : std::false_type {};
template <typename T>
struct has_custom_method<T, std::void_t<decltype(&T::customMethod)>> : std::true_type {};
```

你可以像使用任何其他类型特性一样使用这个自定义特性：

```cpp
static_assert(has_custom_method<MyClass>::value, "MyClass must have a customMethod");
```

### 类型别名以提高可读性和可维护性

类型别名可以通过为复杂类型提供有意义的名称来提高你代码的可读性和可维护性。例如，你不必反复写出 `std::unordered_map<std::string, std::vector<int>>`，你可以创建一个类型别名：

```cpp
using StringToIntVectorMap = std::unordered_map<std::string, std::vector<int>>;
```

现在，你可以在你的代码中使用 `StringToIntVectorMap`，这使得代码更易于阅读和维护：

```cpp
StringToIntVectorMap myMap;
```

类型别名也可以是模板化的，这提供了更大的灵活性：

```cpp
template <typename Value>
using StringToValueMap = std::unordered_map<std::string, Value>;
```

通过使用这些高级类型技术，你为你的 C++ 代码增加了另一层安全性、可读性和可维护性。这些方法让你能够更好地控制模板中类型的行为、检查方式和表示方式，确保你可以编写既健壮又高效的代码。

# 避免高级类型使用中的常见陷阱

## 通过类型检查编写健壮的代码

类型检查是构成程序健壮性和安全性的支柱之一。虽然 C++ 是强类型语言，但它确实允许一些灵活性（或宽容性，取决于你的观点），如果不小心管理，可能会导致错误。以下是一些技术和最佳实践，通过利用类型检查来编写健壮的 C++ 代码。

### 使用类型特性进行编译时检查

C++ 标准库在 `<type_traits>` 头文件中提供了一套类型特性，它允许你在编译时根据类型进行检查和做出决策。例如，如果你有一个只应接受无符号整型类型的泛型函数，你可以使用 `static_assert` 来强制执行这一点：

```cpp
#include <type_traits>
template <typename T>
void foo(T value) {
    static_assert(std::is_unsigned<T>::value, "foo() requires an unsigned integral type");
    // ... function body
}
```

### 利用 constexpr if

C++17 引入了 `constexpr if`，这使得您能够编写在编译时评估的条件代码。这在模板代码中的类型特定操作中非常有用：

```cpp
template <typename T>
void bar(T value) {
    if constexpr (std::is_floating_point<T>::value) {
        // Handle floating-point types
    } else if constexpr (std::is_integral<T>::value) {
        // Handle integral types
    }
}
```

### 函数参数的强类型

C++ 允许类型别名，这有时会使理解函数参数的目的变得困难。例如，声明为 `void process(int, int);` 的函数并不很有信息量。第一个整数是长度吗？第二个是索引吗？减轻这种困难的一种方法是通过使用强类型定义，如下所示：

```cpp
struct Length { int value; };
struct Index { int value; };
void process(Length l, Index i);
```

现在，函数签名提供了语义意义，使得开发者更不可能意外地交换参数。

## 隐式转换和类型强制

### 意外创建文件的情况

在 C++ 开发中，定义接受各种参数类型的构造函数的类是常见的，这样做可以提供灵活性。然而，这也伴随着无意中发生隐式转换的风险。为了说明这一点，考虑以下涉及 `File` 类和 `clean` 函数的代码片段：

```cpp
#include <iostream>
class File {
public:
    File(const std::string& path) : path_{path} {
        auto file = fopen(path_.c_str(), "w");
        // check if file is valid
        // handle errors, etc
        std::cout << "File ctor\n";
    }
    auto& path() const {
        return path_;
    }
    // other ctors, dtor, etc
private:
    FILE* file_ = nullptr;
    std::string path_;
};
void clean(const File& file) {
    std::cout << "Removing the file: " << file.path() << std::endl;
}
int main() {
    auto random_string = std::string{"blabla"};
    clean(random_string);
}
```

输出清楚地展示了问题：

```cpp
File ctor
Removing the file: blabla
```

由于构造函数中缺少 `explicit` 关键字，编译器会自动将 `std::string` 转换为 `File` 对象，从而触发一个意外的副作用——创建一个新的文件。

### 明确性的效用

为了减轻这些风险，可以使用 `explicit` 关键字。通过将构造函数标记为 `explicit`，您指示编译器不允许对该构造函数进行隐式转换。以下是修正后的 `File` 类的示例：

```cpp
class File {
public:
    explicit File(const std::string& path) : path_{path} {
        auto file = fopen(path_.c_str(), "w");
        // check if file is valid
        // handle errors, etc
        std::cout << "File ctor\n";
    }
    // ... rest of the class
};
```

通过这个更改，`clean(random_string);` 这一行将导致编译错误，从而有效地防止意外创建文件。

### 一个轻松的注意事项

虽然我们的示例可能为了教育目的而有所简化（是的，您不需要自己编写 `File` 类——我们有库可以做到这一点！），但它有助于强调 C++ 中类型安全的一个关键方面。一个看似无害的构造函数，如果没有明确防范隐式转换，可能会导致意外的行为。

因此，请记住，当您定义构造函数时，明确您的意图是值得的。您永远不知道何时可能会意外地开始一个您从未打算举办的“文件派对”。

# 摘要

在我们穿越了 C++ 丰富的静态类型系统广阔的领域后，值得花点时间反思我们已经走了多远。从 C++ 最早的那些日子里，原始指针和松散类型数组占据主导地位，到现代的 `std::optional`、`std::variant` 和 `enum class` 时代，语言在处理类型安全的方法上已经发生了显著的变化。

当我们考虑这些进步如何不仅改进单个代码片段，而且改进整个软件系统时，这些进步的真实力量才会显现出来。拥抱 C++ 的强大类型结构可以帮助我们编写更安全、更易读、最终更易于维护的代码。例如，`std::optional` 和 `not_null` 包装器可以减少空指针错误的可能性。高级技术，如模板特化和自定义类型特性，提供了对类型行为的无与伦比的控制。这些不仅仅是学术练习；它们是日常 C++ 程序员的实际工具。

展望未来，C++ 的轨迹表明类型系统将变得更加精细和强大。随着语言不断发展，谁知道未来版本的 C++ 将会带来哪些创新类型相关的特性？也许未来的 C++ 将提供更动态的类型检查，或者它们可能会引入我们目前还无法想象的新的结构。

在下一章中，我们将从类型的基础知识转向 C++ 中类、对象和面向对象编程的宏伟架构。虽然类型为我们提供了构建块，但正是这些更大的结构帮助我们将这些块组装成可持续的软件设计的摩天大楼。在此之前，愿你的类型强大，指针永不空，代码永远稳健。
