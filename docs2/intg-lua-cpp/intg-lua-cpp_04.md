# 4

# 将 Lua 类型映射到 C++

在上一章中，我们学习了如何调用一个接受单个字符串参数并返回一个字符串值的 Lua 函数。在本章中，我们将学习如何调用接受任何类型和任何数量的参数的 Lua 函数，并支持多个返回值。为此，我们需要找到一个方便的方法将 Lua 类型映射到 C++类型。然后，我们将在此基础上逐步改进我们的 Lua 执行器。在这个过程中，您将继续深化对 Lua 栈的理解，并学习如何使用一些现代 C++特性来集成 Lua。

在本章中，我们将涵盖以下主题：

+   映射 Lua 类型

+   支持不同的参数类型

+   支持可变数量的参数

+   支持多个返回值

# 技术要求

本章更注重 C++编码。为了更好地理解本章，请确保您理解以下内容：

+   您熟悉现代 C++标准。我们将开始使用**C++11**和**C++17**中的特性。如果您只使用过**C++03**，请在遇到本章中的新 C++特性时，花些时间自己学习。作为提醒，我们将使用**enum class**、**std::variant**、**std::visit**和**std::initializer_list**。

+   您可以在此处找到本章的源代码：[`github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter04`](https://github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter04)。

+   您可以从前面的 GitHub 链接中的`begin`文件夹中理解并执行代码。`begin`文件夹整合了上一章问题的必要解决方案，并作为本章的起点。我们将向其中添加本章实现的新功能。

# 映射 Lua 类型

在**第二章**，**Lua 基础**中，我们学习了 Lua 类型。它们与 C++类型不同。要在 C++中使用它们，我们需要进行一些映射。在**第三章**，**如何从 C++调用 Lua**中，我们将 Lua 字符串映射到 C++的`std::string`。这是通过在我们的 Lua 执行器中硬编码来实现的。

如果我们想在函数参数和函数返回值中支持所有可能的 Lua 类型怎么办？如果我们想以不同的参数数量调用 Lua 函数怎么办？为每种参数类型和参数数量组合创建一个 C++函数是不可行的。那样的话，我们的 Lua 执行器将受到数百个函数的困扰，只是为了调用 Lua 函数！

幸运的是，C++在**面向对象编程**和**泛型编程**方面非常强大。这两个范例导致了两种不同的方式，您可以在 C++中解决问题。

## 探索不同的映射选项

如前所述，C++支持面向对象编程和泛型编程。我们可以使用其中任何一个来设计类型系统。

### 使用面向对象类型

这种方法可能更容易理解。它自 C++诞生以来就得到了支持。我们可以定义一个表示所有可能类型的抽象基类，然后继承它并为每种类型实现一个具体类。

除了 C++，大多数编程语言都支持这种方法。如果你或你的团队使用多种编程语言，这种方法可能会在工作时减少概念切换。

但这种方法也更冗长。你还需要考虑其他因素。例如，在映射定义之后，你可能会想要防止创建 Lua 中不存在的类型。你必须将基类构造函数设为私有，并声明几个朋友。

### 使用泛型类型

这种方法依赖于一个新的 C++17 特性：*std::variant*。你可以为每个 Lua 类型定义并映射一个简单的 C++类，而不需要继承。然后你使用`std::variant`创建一个联合类型，表示这个联合类型可能并且只能是从预定义的 Lua 映射中来的。

这将导致代码更少。代码越少，出错的机会就越小。现代编程倾向于采用新的范式，而不仅仅是传统的面向对象方法。

这种方法的缺点是，并非所有组织都能如此迅速地采用新的 C++标准，这反过来又使得它们理解起来不那么广泛。

在本章中，我们将实现这种方法。在完成本章后，如果你愿意，你可以自己实现面向对象类型。但在我们继续之前，让我们看看用于本章的`Makefile`。

## 介绍一些新的 Makefile 技巧

在深入细节之前，让我们看一下`Makefile`。你可以在 GitHub 仓库中的`begin`文件夹中找到一个副本，如下所示：

```cpp
LUA_PATH = ../../lua
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Werror
CPPFLAGS = -I${LUA_PATH}/src
LDFLAGS = -L${LUA_PATH}/src
EXECUTABLE = executable
ALL_O = main.o LuaExecutor.o LoggingLuaExecutorListener.o
all: clean lua project
lua:
    @cd ${LUA_PATH} && make
project: ${ALL_O}
    $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $(EXECUTABLE) 
     ${ALL_O} -llua
clean:
    rm -f ${ALL_O} $(EXECUTABLE)
```

与*第三章*中使用的`Makefile`相比，只有四个不同之处：

+   在`CXXFLAGS`中，我们要求编译器通过添加`-std=c++17`将我们的代码编译为 C++17。如果没有这个，它将使用默认的标准，这可能是较旧的 C++版本。

+   新变量`ALL_O`定义了将生成的所有目标文件。每个 C++源文件将被编译成一个目标文件。记住，当你添加新的源文件时，在这里添加一个新的目标文件。否则，如果没有生成目标文件，链接器将找不到应该在缺失的目标文件中存在的符号，你将得到链接器错误。

+   现在的`project`目标依赖于所有目标文件。`Make`足够智能，可以自动为你编译源文件中的目标文件，使用相应的源文件作为目标文件的一个依赖项。

+   `all` 目标有一个额外的依赖项：`clean` 目标。这总是清理项目并重新构建它。当你手动编写目标文件时，你可以让它依赖于多个头文件。当 `Make` 为你这样做时，它无法告诉你哪些头文件需要依赖。所以，这是一个用于学习目的的小项目的技巧。对于更正式的项目，你应该考虑在不先清理的情况下正确编译所有内容。

如果你难以理解这个 `Makefile`，请查看 *第一章* 和 *第三章* 中的解释。更好的是，你可以在网上进行更多研究。如果你没有紧急需要学习关于 `Makefile` 的知识，仅仅使用它，并对其感到舒适，也是完全可以的。

记住

本书使用的 `Makefile` 示例更倾向于简单性，而不是生产灵活性。

我们通过解释一些新的 `Makefile` 机制，稍微分散了对 C++ 的注意力。这将是本书中最后一次介绍 `Makefile`。在接下来的章节中，请参考 GitHub 源代码。

解释是必要的，以防你从 C++ 编译器和链接器中得到难以理解的错误。现在我们可以回到我们的重点。我们将定义一些简单的 C++ 结构，这些结构映射到 Lua 类型。之后，我们可以使用 `std::variant` 来声明一个联合类型。拥有联合类型将使我们能够将任何类型的值传递给我们的 C++ 函数。现在，让我们在 C++ 中定义 Lua 类型。

## 定义 Lua 类型

第一项任务是我们在 C++ 中如何定义一个 Lua 类型。我们希望 Lua 类型有一个清晰的定义，因此像 `std::string` 这样的类型就不再足够独特了。

自 C++11 以来，我们有了 *enum class* 的支持。我们可以在 C++ 中使用枚举类来限制 Lua 类型，如下所示：

```cpp
enum class LuaType
{
    nil,
    boolean,
    number,
    string,
};
```

目前，我们只支持可以映射到简单 C++ 类型的 Lua 基本类型。你可以将这个声明放在一个名为 `LuaType.hpp` 的文件中，并像下面这样将其包含在 `LuaExecutor.h` 中：

```cpp
#include "LuaType.hpp"
```

我们称它为 `*.hpp` 因为我们将直接在头文件中放置类型实现并内联所有函数。这部分的理由是因为实现类将是简单的，部分是因为这是一本书，限制代码行数很重要。你可以将代码分离到头文件和源文件中，或者将包含实现的头文件命名为 `LuaType.h`。这取决于惯例，每个公司或组织都有自己的惯例，C++ 中有许多实现某种方式的方法。

## 在 C++ 中实现 Lua 类型

正如解释的那样，我们将使用没有继承的简单类。每个类将有两个字段：一个 `type` 字段，它是我们刚刚定义的 `LuaType`，以及一个 `value` 字段，用于在 C++ 中实际数据存储。

在 `LuaType.hpp` 中实现四个结构。在 C++ 中，结构与类相同，但默认情况下其成员对公共访问。当我们想要定义数据时，我们通常使用结构。首先，实现 `LuaType::nil`：

```cpp
#include <cstddef>
struct LuaNil final
{
    const LuaType type = LuaType::nil;
    const std::nullptr_t value = nullptr;
    static LuaNil make() { return LuaNil(); }
private:
    LuaNil() = default;
};
```

我们选择使用 `nullptr` 来表示 Lua 的 nil 值。它的类型是 `std::nullptr_t`。我们还将其构造函数设为私有，并提供了一个静态函数来创建新对象。

设计模式

在这里，我们使用了一个设计模式——私有构造函数的静态工厂方法。在我们的实现中，这将防止使用 `new` 在堆上创建对象。Lua 类型的 C++ 结构也不提供拷贝构造函数。这是一个设计选择——你可以完全支持传递和赋值，或者限制其使用。在这本书中，我们仅在其与 C++ 栈上的 Lua 执行器交互时限制其使用。如果你在 Lua 执行器之上还有其他层，你需要将结构转换为 C++ 基本类型或你自己的类型。这有助于抽象。

类似地，实现 `LuaType::boolean`：

```cpp
struct LuaBoolean final
{
    const LuaType type = LuaType::boolean;
    const bool value;
    static LuaBoolean make(const bool value)
    {
        return LuaBoolean(value);
    }
private:
    LuaBoolean(const bool value) : value(value) {}
};
```

静态的 `make` 函数接受一个 `boolean` 值来创建一个实例。在私有构造函数中，我们使用成员初始化列表来初始化 `value` 成员变量。

对于 `LuaType::number`，我们选择使用 C++ double 类型来存储值：

```cpp
struct LuaNumber final
{
    const LuaType type = LuaType::number;
    const double value;
    static LuaNumber make(const double value)
    {
        return LuaNumber(value);
    }
private:
    LuaNumber(const double value) : value(value) {}
};
```

Lua 本身在其基本 *number* 类型中不区分整数和浮点数，但如果你需要，你可以为整数和浮点数分别创建两个 C++ 类型。为此，你可以使用 Lua 的 `lua_isinteger` 库函数来检查数字是否为整数。如果不是，它就是一个 double。在这本书中，我们只实现了基本 Lua 类型的映射。在一个游戏系统中，你可能想强制使用浮点类型。在一个嵌入式系统中，你可能想强制使用整数类型。或者，你可以在项目中支持使用两者。通过引用 `LuaNumber` 的实现，这很容易实现。

知识链接

在 Lua 代码中，你可以使用 `math.type` 库函数来检查一个数字是整数还是浮点数。

最后，对于 `LuaType::string`，我们使用 `std::string` 来存储值：

```cpp
#include <string>
struct LuaString final
{
    const LuaType type = LuaType::string;
    const std::string value;
    static LuaString make(const std::string &value)
    {
        return LuaString(value);
    }
private:
    LuaString(const std::string &value) : value(value) {}
};
```

这就结束了我们的类型实现。接下来就是所有魔法发生的地方。

## 实现联合类型

我们定义了一个 `LuaType` 枚举类来标识 Lua 类型，以及结构来表示不同类型的 Lua 值。当我们想要传递 Lua 值时，我们需要一个类型来表示它们。不使用公共基类，我们可以使用 `std::variant`。它是一个模板类，接受一系列类型作为其参数。然后它可以安全地在代码中表示这些类型中的任何一种。要看到它的实际应用，请将以下内容添加到 `LuaType.hpp`：

```cpp
#include <variant>
using LuaValue = std::variant<
    LuaNil, LuaBoolean, LuaNumber, LuaString>;
```

`using` 关键字创建了一个类型别名，`LuaValue`。它可以代表模板参数中指定的四种类型中的任何一种。

## 与联合类型一起工作

如果您之前没有使用过 `std::variant`，您可能想知道我们如何判断它实际持有的类型。如果您传递 `LuaValue` 的值，您无法直接访问 `type` 或 `value` 字段。这是因为没有公共基类。在编译时，编译器无法仅通过查看 `std::variant` 变量来确定支持哪些字段。为此，我们需要一个小技巧。C++17 也提供了 `std::visit` 来帮助解决这个问题。让我们实现一个辅助函数来从 `LuaValue` 获取 `LuaType`。在 `LuaType.hpp` 中添加以下代码：

```cpp
inline LuaType getLuaType(const LuaValue &value)
{
    return std::visit(
        [](const auto &v) { return v.type; },
        value);
}
```

这个函数使调用站点更高效。此外，它需要是一个内联函数，因为我们直接在头文件中实现它。如果没有使用 `**inline**` 关键字，函数可能会被包含在不同的源文件中，具有相同的符号，从而导致链接错误。

**`std::visit` 接受两个参数。第一个是一个 C++ `std::visit` 使得类型信息可用。如果您之前从未遇到过这个概念或用法，可能需要一些时间来消化。您可以将这个可调用项视为 **lambda**。如果您在其他编程语言中使用过 lambda，例如 Java、Kotlin、Swift 或 Python，C++ 的 lambda 非常相似。在其他编程语言中，lambda 通常作为最后一个参数，称为 **尾随 lambda**，在某些情况下更容易阅读。了解 C++ lambda 的最佳方式是使用它，并尝试在完全掌握之前使其变得舒适。

因此，让我们实现另一个辅助函数来获取每种类型的字符串表示。在 `LuaTypp.hpp` 中添加以下函数：

```cpp
inline std::string
getLuaValueString(const LuaValue &value)
{
    switch (getLuaType(value))
    {
    case LuaType::nil:
        return "nil";
    case LuaType::boolean:
        return std::get<LuaBoolean>(value).value
            ? "true" : "false";
    case LuaType::number:
        return std::to_string(
            std::get<LuaNumber>(value).value);
    case LuaType::string:
        return  std::get<LuaString>(value).value;
    }
}
```

这将帮助我们通过获取存储在 `LuaValue` 中的内容来测试本章其余部分的实现。您可以使用 `std::get` 从 `std::variant` 联合中获取特定类型。

我们已经讨论了 `std::variant`、`std::visit` 和 `std::get`，但不足以成为该领域的 C++ 专家。在继续之前，请随意对这些内容进行更多研究。

接下来，让我们使用我们已实现的 Lua 映射来使调用 Lua 函数更加灵活。首先，我们将移除 Lua 执行器中 `call` 函数使用的硬编码的 `std::string`。

# 支持不同的参数类型

在上一章中，我们实现了以下方式调用 Lua 函数的 C++ 函数：

```cpp
std::string call(const std::string &function,
                 const std::string &param);
```

在这一步中，我们的目标是使其更通用，我们希望以下内容：

```cpp
LuaValue call(const std::string &function,
              const LuaValue &param);
```

实际上，请先在 `LuaExecutor.h` 中进行此更改。为了使其工作，我们将实现辅助函数来将 `LuaValue` C++ 类型推送到和从 Lua 栈中弹出，而不是 `std::string`。让我们首先处理推送到栈上的操作。

## 推送到栈上

在之前的调用函数中，我们将 `std::string` 类型的 `param` 参数以以下方式推送到 Lua 栈中：

```cpp
lua_pushstring(L, param.c_str());
```

为了支持更多的 Lua 类型，我们可以实现一个接受 `LuaValue` 作为参数的 `pushValue` 方法，并根据 `LuaValue` 的 `type` 字段调用不同的 `lua_pushX` Lua 库函数。

在 `LuaExecutor.h` 中，添加以下声明：

```cpp
class LuaExecutor
{
private:
    void pushValue(const LuaValue &value);
};
```

在 `LuaExecutor.cc` 中，实现 `pushValue` 函数：

```cpp
void LuaExecutor::pushValue(const LuaValue &value)
{
    switch (getLuaType(value))
    {
    case LuaType::nil:
        lua_pushnil(L);
        break;
    case LuaType::boolean:
        lua_pushboolean(L,
            std::get<LuaBoolean>(value).value ? 1 : 0);
        break;
    case LuaType::number:
        lua_pushnumber(L,
            std::get<LuaNumber>(value).value);
        break;
    case LuaType::string:
        lua_pushstring(L,
            std::get<LuaString>(value).value.c_str());
        break;
    }
}
```

我们的实现只在 `LuaType` 上使用一个 `switch` 语句。我们之前在本章中实现了 `getLuaType` 函数，位于 `LuaType.hpp` 中。在每一个 `case` 中，我们使用 `std::get` 从 `LuaValue` 类型联合中获取类型值。接下来，我们将查看弹出部分。

## 从栈中弹出

从 Lua 栈中弹出是推送部分的逆操作。我们将从 Lua 栈中获取值，并使用 Lua `lua_type` 库函数来检查其 Lua 类型，然后创建一个具有匹配 `LuaType` 的 C++ `LuaValue` 对象。

为了使事情更加模块化，我们将创建两个函数：

+   `getValue` 将 Lua 栈位置转换为 `LuaValue`

+   `popValue` 用于弹出并返回栈顶元素

将以下声明添加到 `LuaExecutor.h` 中：

```cpp
class LuaExecutor
{
private:
    LuaValue getValue(int index);
    LuaValue popValue();
};
```

在 `LuaExecutor.cc` 中，让我们首先实现 `getValue`：

```cpp
LuaValue LuaExecutor::getValue(int index)
{
    switch (lua_type(L, index))
    {
    case LUA_TNIL:
        return LuaNil::make();
    case LUA_TBOOLEAN:
        return LuaBoolean::make(
            lua_toboolean(L, index) == 1);
    case LUA_TNUMBER:
        return LuaNumber::make(
            (double)lua_tonumber(L, index));
    case LUA_TSTRING:
        return LuaString::make(lua_tostring(L, index));
    default:
        return LuaNil::make();
    }
}
```

代码相当直接。首先，我们检查请求的栈位置的 Lua 类型，然后相应地返回一个 `LuaValue`。对于不支持的 Lua 类型，例如表和函数，我们目前只返回 `LuaNil`。有了这个，我们可以如下实现 `popValue`：

```cpp
LuaValue LuaExecutor::popValue()
{
    auto value = getValue(-1);
    lua_pop(L, 1);
    return value;
}
```

我们首先使用 `-1` 作为栈位置调用 `getValue` 来获取栈顶元素。然后我们弹出栈顶元素。

在实现了栈操作之后，我们现在可以通过组合栈操作来实现新的 `call` 函数。

## 将其组合起来

花点时间再次阅读旧的 `call` 函数实现。如下所示：

```cpp
std::string LuaExecutor::call(
    const std::string &function,
    const std::string &param)
{
    int type = lua_getglobal(L, function.c_str());
    assert(LUA_TFUNCTION == type);
    lua_pushstring(L, param.c_str());
    pcall(1, 1);
    return popString();
}
```

要实现我们的新 `call` 函数，不需要做太多更改。我们只需要将执行栈操作的代码行替换为我们刚刚实现的新的辅助函数。在 `LuaExecutor.cc` 中编写新的 `call` 函数如下：

```cpp
LuaValue LuaExecutor::call(
    const std::string &function, const LuaValue &param)
{
    int type = lua_getglobal(L, function.c_str());
    assert(LUA_TFUNCTION == type);
    pushValue(param);
    pcall(1, 1);
    return popValue();
}
```

我们已经将处理 `std::string` 的行替换为处理 `LuaValue` 的新行。

由于我们有一个专门的 `getValue` 函数和 `popValue` 函数来将原始 Lua 值转换为 `LuaValue`，我们可以利用这个机会让 `popString` 也使用它们。重写如下：

```cpp
std::string LuaExecutor::popString()
{
    auto result = std::get<LuaString>(popValue());
    return result.value;
}
```

在这里，我们已经去掉了在 `popString` 中使用 Lua 库函数。限制对第三方库的依赖仅限于少数几个函数是一种良好的实践。另一种思考方式是，在一个类中，内部可以有低级函数和高级函数。

接下来，让我们测试我们改进的 Lua 执行器。

## 测试一下

由于我们使用了 C++17 特性来实现 `LuaValue`，因此我们将使用现代 C++ 编写测试代码。编写 `main.cpp` 如下：

```cpp
int main()
{
    auto listener = std::make_unique<
        LoggingLuaExecutorListener>();
    auto lua = std::make_unique<LuaExecutor>(*listener);
    lua->executeFile("script.lua");
    auto value1 = lua->call(
        "greetings", LuaString::make("C++"));
    std::cout << getLuaValueString(value1) << std::endl;
    auto value2 = lua->call(
        "greetings", LuaNumber::make(3.14));
    std::cout << getLuaValueString(value2) << std::endl;
    return 0;
}
```

在此测试代码中，我们首先使用 `std::unique_ptr` 来持有我们的 Lua 执行器和其监听器，然后使用 `greetings` Lua 函数加载 Lua 脚本。这个 Lua 函数来自上一章。实际操作是调用 Lua 函数两次：首先使用 `LuaString`，然后使用 `LuaNumber`。

编译并运行测试代码。如果你一切都做对了，你应该看到以下输出：

```cpp
Hello C++
Hello 3.14
```

如果你看到编译器或链接器错误，不要感到气馁。在构建新的 C++ 代码时，看到一些难以理解的错误信息是很常见的，尤其是在应用新知识时。追踪错误并尝试纠正它们。如果你需要，也可以与 GitHub 上的代码进行比较。

注意

到目前为止，我们已经学到了很多。我们改进的 Lua 执行器可以以更灵活的方式调用 Lua 函数，尽管它仍然只接受一个参数。现在，你应该对使用常见的 C++ 类型来表示不同的 Lua 类型感到舒适和自信。在继续进一步改进我们的 Lua 执行器以调用接受可变数量参数的 Lua 函数之前，先休息一下并反思。

现在，让我们继续改进我们的 Lua 执行器。

# 支持可变数量的参数

Lua 的 `function` 支持可变数量的参数。让我们在 `script.lua` 中实现一个：

```cpp
function greetings(...)
    local result = "Hello"
    for i, v in ipairs{...} do
        result = result .. " " .. v .. ","
    end
    return result
end
```

这将返回一个问候消息，并将所有参数包含在消息中。三个点（`...`）表示该函数接受可变数量的参数。我们可以使用 `ipairs` 遍历参数。

我们如何在 C++ 中支持这一点？对于堆栈操作，我们只需要推送更多的值。主要决定是如何声明 Lua 执行器 `call` 函数以接受可变数量的参数。

## 实现 C++ 函数

自 C++11 以来，我们可以使用 **可变参数函数模板** 来传递 **参数包**。参数包是任意大小的参数列表。

在 `LuaExecutor.h` 中，将 `call` 函数声明更改为以下内容：

```cpp
template <typename... Ts>
LuaValue call(const std::string &function,
              const Ts &...params);
```

`typename... Ts` 定义了一个模板参数包，函数将其作为 `params` 参数接受。

现在，让我们来实现它。删除 `LuaExecutor.cc` 中的 `call` 实现文件。由于我们现在正在使用模板，我们需要将实现放在头文件中。在 `LuaExecutor.h` 中添加以下代码：

```cpp
template <typename... Ts>
LuaValue LuaExecutor::call(const std::string &function,
                           const Ts &...params)
{
    int type = lua_getglobal(L, function.c_str());
    assert(LUA_TFUNCTION == type);
    for (auto param :
        std::initializer_list<LuaValue>{params...})
    {
        pushValue(param);
    }
    pcall(sizeof...(params), 1);
    return popValue();
}
```

此实现可以分为四个步骤，代码中通过空行分隔：

+   它获取要调用的 Lua 函数。这没有变化。

+   它推送 C++ 函数参数。在这里，我们选择从参数包中创建一个 `std::initializer_list` 并遍历它。

+   它调用 Lua 函数。我们使用 `sizeof...(params)` 获取参数包的大小，并告诉 Lua 我们将发送这么多参数。

+   它从 Lua 函数中获取返回值并将其返回。

完成第 2 步有不止一种方法。你可以使用 lambda 来解包参数包，甚至有不同选项来编写这个 lambda。当 **C++20** 逐渐被采用时，你将拥有更多选项。然而，这些选项超出了本书的范围。在这里，我们选择使用更传统的方式来实现，这样更多的人更容易理解。

接下来，让我们测试我们的实现是否有效。

## 测试它

在 `main.cpp` 中，替换调用 `lua->call` 并打印结果的行，如下所示：

```cpp
auto result = lua->call("greetings",
    LuaString::make("C++"), LuaString::make("Lua"));
std::cout << getLuaValueString(result) << std::endl;
```

在测试代码中，我们向 Lua 的 `greetings` 函数传递了两个字符串。由于我们支持可变数量的参数，你可以传递任意数量的参数，包括零。你应该看到类似 `Hello` `C++, Lua,` 的输出。

## 关于我们机制的更多说明

到目前为止，我们在 Lua 执行器中实现了一个通用的函数来调用任何 Lua 函数，并且可以接受任意数量的参数。请花点时间思考以下要点，这将加深你的理解：

+   *被调用的 Lua 函数不需要声明为接受可变数量的参数，而我们的 C++ 函数则需要这样做。* 当从 C++ 调用 Lua 函数时，你总是需要告诉 Lua 库已经推送到栈上的参数数量。

+   *Lua 函数不需要返回值。* 你可以尝试注释掉 `greetings` 函数中的返回语句。C++ 方面将得到一个 `LuaNil`，因为 Lua 库保证将请求的返回值数量推送到栈上，当 Lua 函数没有返回足够值时使用 nil。

+   *Lua 函数可以返回多个值。* 我们只会得到第一个值，Lua 库将丢弃其余的，因为当我们调用 Lua 函数时，我们只请求一个返回值。

我们当前的实现已经支持了调用普通 Lua 函数的大部分用例，除了上面提到的最后一点。接下来，我们将支持多个返回值以完成 Lua 函数调用机制。

# 支持多个返回值

为了处理获取多个返回值，让我们首先创建一个实际上执行这一操作的 Lua 函数。在 `script.lua` 中添加以下函数：

```cpp
function dump_params(...)
    local results = {}
    for i, v in ipairs{...} do
        results[i] = i .. ": " .. tostring(v) ..
            " [" .. type(v) .. "]"
    end
    return table.unpack(results)
end
```

这将获取每个参数并打印出其类型。我们首先将它们放入一个表中，然后解包这个表，使得每个表条目作为一个单独的值返回。

现在，我们有一些决定要做。我们对当前的 `call` 函数很满意，除了它的返回值。然而，在 C++ 中，我们不能为不同的返回类型重载一个函数。我们需要创建另一个返回值列表的函数。

我们如何从 Lua 获取多个返回值？与 `call` 相比，有两个差异需要我们解决：

+   我们如何告诉 Lua 库我们期望一个可变数量的返回值，而不是一个固定数量的？

+   我们如何在 C++ 中获取这个可变数量的返回值？

为了解决第一个问题，在调用 Lua 库的 `lua_pcall` 函数时，我们可以指定一个表示预期返回值数量的魔法数字：`LUA_MULTRET`。这意味着我们将接受 Lua 函数返回的任何内容，而库不会丢弃额外的返回值或用 `nil` 填充。这个魔法数字是唯一需要指定返回值数量的特殊情况。它在 `lua.h` 中内部定义为 `-1`。

为了解决第二个问题，我们只需要在调用 Lua 函数前后计算 Lua 栈中的元素数量。这是因为 Lua 库将所有返回值推入栈中，所以栈中的新元素就是返回值。我们已经实现了 `popValue` 来弹出栈顶元素。我们需要另一个函数来从栈中弹出多个值。

解决了这两个问题后，让我们开始实施。

## 实现 C++ 函数

在 `LuaExecutor.h` 中添加以下声明：

```cpp
class LuaExecutor
{
public:
    template <typename... Ts>
    std::vector<LuaValue> vcall(
        const std::string &function,
        const Ts &...params);
private:
    std::vector<LuaValue> popValues(int n);
};
```

我们添加了另一个函数来调用 Lua 函数。我们称它为 `vcall`，因为它返回一个 `std::vector`。我们还添加了一个 `popValues` 辅助函数，用于从 Lua 栈中弹出顶部 `n` 个元素。

首先，让我们在 `LuaExecutor.h` 中实现 `vcall`：

```cpp
template <typename... Ts>
std::vector<LuaValue> LuaExecutor::vcall(
    const std::string &function, const Ts &...params)
{
    int stackSz = lua_gettop(L);
    int type = lua_getglobal(L, function.c_str());
    assert(LUA_TFUNCTION == type);
    for (auto param :
        std::initializer_list<LuaValue>{params...})
    {
        pushValue(param);
    }
    if (pcall(sizeof...(params), LUA_MULTRET))
    {
        int nresults = lua_gettop(L) - stackSz;
        return popValues(nresults);
    }
    return std::vector<LuaValue>();
}
```

现在我们有五个步骤，具体说明如下：

1.  使用 `lua_gettop` 记录栈大小。

1.  使用 `lua_getglobal` 将 Lua 函数推入栈中。

1.  使用 `pushValue` 将所有参数推入栈中。

1.  使用 `pcall` 调用 Lua 函数，并传递 `LUA_MULTRET` 以指示我们将从 Lua 函数中获取所有返回值。Lua 库将保证弹出你在 *步骤 2* 和 *步骤 3* 中推入的所有元素。

1.  使用 `popValues` 弹出所有返回值并将它们返回。我们再次检查栈大小。新栈大小减去存储在 `stackSz` 中的原始栈大小就是返回值的数量。

接下来，我们将实现最后一部分，即辅助函数，用于从 Lua 栈中弹出所有返回值。在 `LuaExecutor.cc` 中添加以下代码：

```cpp
std::vector<LuaValue> LuaExecutor::popValues(int n)
{
    std::vector<LuaValue> results;
    for (int i = n; i > 0; --i)
    {
        results.push_back(getValue(-i));
    }
    lua_pop(L, n);
    return results;
}
```

Lua 将第一个返回值推入栈中，然后是第二个，依此类推。因此，栈顶需要存储在向量的末尾。在这里，我们按顺序读取返回值，从栈的中间开始，向栈顶移动。`-i` 是从栈顶开始计算的 `ith` 位置。

接下来，让我们测试一下。

## 测试

在 `main.cpp` 中，按照以下方式更改测试代码：

```cpp
auto results = lua->vcall(
    "dump_params",
    LuaString::make("C++"),
    LuaString::make("Lua"),
    LuaNumber::make(3.14),
    LuaBoolean::make(true),
    LuaNil::make());
for (auto result : results)
{
    std::cout << getLuaValueString(result) << std::endl;
}
```

我们向新函数传递了不同类型的 `LuaValue` 列表（`LuaString`、`LuaNumber`、`LuaBoolean` 和 `LuaNil`）。这将输出以下内容：

```cpp
1: C++ [string]
2: Lua [string]
3: 3.14 [number]
4: true [boolean]
```

你观察到任何异常情况吗？我们传递了五个参数，但只得到了四个返回值！`LuaNil` 没有打印出来。为什么？这是因为，在 `dump_params` 中，我们使用了 `table.unpack` 来返回多个值。Lua 的 `table.unpack` 会停止在它看到 nil 值时。如果你将 `LuaNil::make()` 移到列表的中间，你会错过更多的返回值。这是预期的。这是 Lua 的事情。类似于 C++ 的 `char*` 字符串，它会在第一次看到 `NULL` 字符时结束。

# 概述

在本章中，我们首先探讨了如何将 Lua 类型映射到 C++ 类型，目的是在 C++ 函数调用中易于使用。然后，我们了解了一种调用任何 Lua 函数的通用方法。

本章逐步推进。你继续改进 Lua 执行器。每一步都产生了一个里程碑。这反过来又基于上一章的工作。通过以下练习，你也将有机会通过实际编码回顾你所学的知识。我们将继续使用这种方法继续本书。 

在下一章中，我们将学习如何集成 Lua 表。

# 练习

1.  实现 `LuaType::function` 和 `LuaFunction` 以涵盖 Lua 函数类型。无需担心 `LuaFunction` 中的值字段。你可以使用 `nullptr`。为了测试它，你需要调用一个返回另一个函数的 Lua 函数，并在 C++ 中打印出返回值是一个函数。

1.  实现 `LuaType::table` 和 `LuaTable` 以涵盖 Lua 表类型。遵循与上一个问题相同的说明。

1.  在上一章中，我们实现了 `getGlobalString` 和 `setGlobal` 以与 Lua 全局值一起工作。重写这两个方法以支持更多类型。你可以使用新的名称 `getGlobal` 和 `setGlobal`，并使用 `LuaValue`。

1.  实现一个私有的 `dumpStack` 调试函数。此函数将输出当前 Lua 栈。你只需要支持 `LuaValue` 中当前支持的类型。在 `LuaExecutor` 的不同位置插入对该函数的调用。这将加深你对 Lua 栈的理解。**
