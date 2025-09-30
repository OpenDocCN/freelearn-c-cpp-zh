

# 抽象 C++类型导出器

在上一章中，我们学习了如何将 C++类导出到 Lua 作为用户定义的类型。类的实例被导出到 Lua 作为类型的原型。

上一章的练习要求你为类型创建一个工厂类。然而，由于这个要求，每种类型都需要自己的工厂类。

在本章中，我们将学习如何实现一个通用的 C++类型导出器，这样你就可以将其用作任何 C++类型的工厂类，而无需重复工作。

在本章中，我们将涵盖以下主题：

+   回顾工厂实现

+   设计类型导出器

+   模拟类型导出器

+   定义`LuaModuleDef`

+   重新实现`luaNew`

+   你是否足够灵活？

# 技术要求

你可以在此处访问本章的源代码：[`github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter08`](https://github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter08)。

# 回顾工厂实现

我们将回顾上一章的练习。本书采用的解决方案需要渐进和最小化更改。它也自然地引出了本章介绍的功能。

如果你还没有实现自己的解决方案并且愿意停下来思考一下，这是一个尝试的机会。许多技术在解释上很简单，但很难掌握。理解它们的最佳方式是通过反复回顾和实践，直到你获得“啊哈”的顿悟。

现在，我们将回顾一个解决方案。重点是变化和关键概念。

## 定义工厂

要创建一个工厂，我们只需要更改`Destinations.h`和`Destinations.cc`。在你的首选 IDE 中，你可以打开*第七章*的`end`项目以及本章的`begin`项目来检查差异。

让我们先看看工厂类声明的头文件。你可以在本章`begin`项目的`Destinations.h`中找到以下声明：

```cpp
class Destinations
{
public:
    Destinations();
    ~Destinations();
    void wish(const std::vector<std::string> &places);
    void went(const std::vector<std::string> &places);
    std::vector<std::string> listVisited() const;
    std::vector<std::string> listUnvisited() const;
private:
    std::map<std::string, bool> wishlist;
};
class DestinationsFactory : public LuaModule
{
public:
    const std::string &luaName() const override;
    const std::vector<luaL_Reg> &
    luaRegs() const override;
};
```

与上一章的变化是，我们创建了一个名为`DestinationsFactory`的工厂类，它实现了`LuaModule`接口。实际上，我们将`LuaModule`实现从`Destinations`移动到`DestinationsFactory`，这样`Destinations`类型就不知道任何关于 Lua 的事情。这是工厂类的一个好处。系统可以更好地分层。

你知道吗？

如果你使用 Linux 或 Mac，你也可以使用`diff` `Chapter07/end/Destinations.h Chapter08/begin/Destinations.h`。

接下来，我们将回顾工厂实现。

## 实现工厂

工厂只有两个成员函数，其实现如下：

```cpp
const std::string &
DestinationsFactory::luaName() const
{
    return NAME;
}
const std::vector<luaL_Reg> &
DestinationsFactory::luaRegs() const
{
    return FACTORY_REGS;
}
```

现在，`luaRegs`不再返回`REGS`，而是返回`FACTORY_REGS`，它被定义为如下：

```cpp
const std::vector<luaL_Reg> FACTORY_REGS = {
    {"new", luaNew},
    {NULL, NULL}};
```

这意味着，现在，我们只导出一个函数，`luaNew`，到 Lua。

如*第六章*所述，Lua 库期望数组的最后一个条目是`{NULL, NULL}`，以标记数组的结束。这是基于 C 的库的典型技术，因为它们通常将一个指向数组项的指针作为数组的输入，并需要确定数组在哪里结束。

此外，从`REGS`中删除`luaNew`，使其看起来像以下列表：

```cpp
const std::vector<luaL_Reg> REGS = {
    {"wish", luaWish},
    {"went", luaWent},
    {"list_visited", luaListVisited},
    {"list_unvisited", luaListUnvisited},
    {NULL, NULL}};
```

之前，`REGS`有两个用途：

+   从 Lua 创建的新实例的`__index`元表。这是在`luaNew`中通过`luaL_setfuncs(L, REGS.data(), 0)`来完成的。

+   已注册的 Lua 模块，这是一个普通的 Lua 表。这是通过调用`LuaExecutor::registerModule`来完成的。

现在，`REGS`只服务于第一个目的，并将第二个责任交给了`FACTORY_REGS`。这是另一个结构改进。

这些就是我们创建工厂所需的所有更改。您可以从 GitHub 获取完整的源代码。然而，代码更改并不多，对吧？我们只是移动了一些东西，现在我们有一个不同的对象创建机制。

现在，基于这个工厂概念，我们准备继续本章的主要焦点。从现在开始，您可以使用`begin`项目作为开发的基础。让我们开始设计一个通用的 C++类型导出器。

# 设计类型导出器

首先，让我们定义我们的范围。我们希望使刚刚创建的工厂通用化，并使其能够与任何 C++类一起工作——也就是说，C++类仍然需要以某种方式实现和提供`lua_CFunction`包装器。自动创建这些包装器是可能的，但这将需要实现一个重量级的 C++模板库，这与 Lua 没有直接关系，并且超出了本书的范围。

在定义了范围之后，让我们做一些高级设计。

## 选择设计模式

当我们谈论在 C++中使某物*通用*时，通常意味着我们需要使用模板。为了与我们的 Lua 执行器一起工作，我们需要导出`LuaModule`。因此，我们需要将导出器实现为一个模板类，它可以提供`LuaModule`。

我们如何提供`LuaModule`？我们可以使导出器继承自`LuaModule`接口，或者使其其中一个成员函数返回`LuaModule`。

后者选项中流行的设计模式之一是**建造者**模式。这可以通过以下伪代码来演示：

```cpp
class LuaModuleBuilder
{
    LuaModuleBuilder withOptionA(...);
    LuaModuleBuilder withOptionB(...);
    ...
    LuaModule build();
};
LuaModuleBuilder builder;
auto module = builder.withOptionX(...).build();
```

建造者通常有许多函数来定制它所创建的事物的不同属性，同时还有一个`build`函数来创建最终对象。

由于我们的目标仅仅是帮助进行对象创建，就像在工厂练习中一样，而不是定制对象，因此*建造者*模式是多余的。我们将选择纯 C++继承。导出器类型可以定义如下：

```cpp
template <typename T>
class LuaModuleExporter : public LuaModule;
```

这是一个模板类。它将导出 C++类型`T`为`LuaModule`。

现在，让我们模拟导出器。

# 模拟导出器

在导出器的设计中，我们有两个主要的考虑因素。首先，它是 `LuaModule`，因此需要实现它的纯虚函数。其次，我们希望它类似于我们在工厂练习中实现的内容，这意味着我们对 `luaRegs` 虚拟函数实现中要返回的内容有一个相当好的想法。

让我们开始。添加一个名为 `LuaModuleExporter.hpp` 的新文件，并定义 `LuaModuleExporter` 类，如下所示：

```cpp
template <typename T>
class LuaModuleExporter final : public LuaModule
{
public:
    LuaModuleExporter(
        const LuaModuleExporter &) = delete;
    ~LuaModuleExporter() = default;
    static LuaModuleExporter<T> make()
    {
        return LuaModuleExporter<T>();
    }
private:
    LuaModuleExporter() {}
};
```

这使得导出器成为一个最终类，并防止它被复制构造。因为导出器的目的是提供 `LuaModule`，我们没有逻辑让它通过值传递，所以添加一些限制可以防止未来的错误。我们通过将 `delete` 关键字赋给复制构造函数来实现这一点。我们还想控制对象创建，所以我们使构造函数私有。这还有一个副作用——你不能使用 `new` 操作符来创建类的实例。

现在，按照以下方式为 `LuaModule` 添加实现：

```cpp
class LuaModuleExporter final : public LuaModule
{
public:
    const std::string &luaName() const override
    {
        return name;
    }
    const std::vector<luaL_Reg> &luaRegs() const override
    {
        return factoryRegs;
    }
private:
    const std::string name = "TODO";
    const std::vector<luaL_Reg> factoryRegs = {
        {"new", luaNew},
        {NULL, NULL}};
    static int luaNew(lua_State *L)
    {
        return 0;
    }
};
```

这很简单。在 Lua 模块级别，我们只想导出一个函数来创建具体对象。因此，我们只会注册 `luaNew`。模块的名称需要传递进来。我们将在实现细节时找到一种方法。

因此，我们为导出器创建了一个占位符。这是一个系统级的设计合约。现在，让我们编写测试代码来查看它应该如何使用。

## 准备 C++ 测试代码

在 `main.cpp` 中编写 `main` 函数如下：

```cpp
int main()
{
    auto listener = std::make_unique<
        LoggingLuaExecutorListener>();
    auto lua = std::make_unique<LuaExecutor>(*listener);
    auto module = LuaModuleExporter<
        Destinations>::make();
    lua->registerModule(module);
    lua->executeFile("script.lua");
    return 0;
}
```

与上一章相比，唯一的区别是 `LuaModule` 的创建方式。现在，它是通过 `LuaModuleExporter<Destinations>::make()` 创建的。

到目前为止，项目应该可以编译。当你运行它时，它不应该在 C++ 端崩溃；尽管如此，在这个阶段，它将无法执行任何有意义的操作，你应该会看到 Lua 的错误信息。

现在，我们将看到我们需要什么 Lua 代码。

## 准备 Lua 测试脚本

将 `script.lua` 编写如下：

```cpp
dst = Destinations.new()
dst:wish("London", "Paris", "Amsterdam")
dst:went("Paris")
print("Visited:", dst:list_visited())
print("Unvisited:", dst:list_unvisited())
```

我们在上一章中使用了这个代码片段。这将帮助我们验证我们是否会在本章后面得到相同的结果。

接下来，让我们开始使导出器工作。

# 定义 LuaModuleDef

首先，我们需要提供模块的名称，然后是 `__index` 元表。最后，我们需要提供一个元表的名称。回想一下，在 `Destinations.cc` 中，元表的名称硬编码如下：

```cpp
const std::string METATABLE_NAME(
    "Destinations.Metatable");
```

现在，这需要传递给导出器。让我们定义一个结构来表示上述三块信息。在 `LuaModule.h` 中添加以下声明：

```cpp
template <typename T>
struct LuaModuleDef
{
    const std::string moduleName;
    const std::vector<luaL_Reg> moduleRegs;
    const std::string metatableName() const
    {
        return std::string(moduleName)
            .append(".Metatable");
    }
};
```

这定义了 `moduleName` 和 `moduleRegs`。元表名称基于模块名称，并在其后附加 `".Metatable"`。

注意，这个结构也是模板化的。这表明定义是为某种 C++ 类型。我们将在本章后面使用模板。

现在，我们可以将这个结构传递给导出器。

## 使用 LuaModuleDef

在`LuaModuleExporter.hpp`中，在导出器创建期间接受`LuaModuleDef`实例。重写相关代码如下：

```cpp
class LuaModuleExporter final : public LuaModule
{
public:
    static LuaModuleExporter<T> make(
        const LuaModuleDef<T> &luaModuleDef)
    {
        return LuaModuleExporter<T>(luaModuleDef);
    }
    const std::string &luaName() const override
    {
        return luaModuleDef.moduleName;
    }
private:
    LuaModuleExporter(
        const LuaModuleDef<T> &luaModuleDef)
        : luaModuleDef(luaModuleDef) {}
    const LuaModuleDef<T> luaModuleDef;
};
```

更改如下：

+   我们添加了一个私有成员变量`luaModuleDef`

+   我们向`make`和私有构造函数添加了一个类型为`LuaModuleDef`的参数

+   我们将`luaName`改为返回`luaModuleDef.moduleName`

+   我们删除了在模拟过程中引入的私有成员变量`name`

现在，我们可以为`Destinations`类定义`LuaModuleDef`。

在`Destinations.h`中，删除`DestinationsFactory`的声明并添加以下代码：

```cpp
struct DestinationsLuaModuleDef
{
    static LuaModuleDef<Destinations> def;
};
```

在`Destinations.cpp`中，删除`DestinationsFactory`的所有实现，并在匿名命名空间之后添加以下代码：

```cpp
LuaModuleDef DestinationsLuaModuleDef::def =
    LuaModuleDef<Destinations>{
        "Destinations",
        {{"wish", luaWish},
         {"went", luaWent},
         {"list_visited", luaListVisited},
         {"list_unvisited", luaListUnvisited},
         {NULL, NULL}},
    };
```

最后，在`main.cpp`中，将模块创建代码更改为以下语句：

```cpp
auto module = LuaModuleExporter<Destinations>::make(
    DestinationsLuaModuleDef::def);
```

这将`Destinations`类的`LuaModuleDef`泵送到导出器中。确保项目可以编译。

现在，我们将填补其余缺失的部分，使导出器真正工作。

# 重新实现`luaNew`

由于我们将在`LuaModuleExporter`中存储`LuaModuleDef`，为了访问它，我们需要找到`LuaModuleExporter`的实例。让我们首先实现一个辅助函数来完成这个任务。

由于导出器也是`LuaModule`，它已经具有第六章中实现的值机制。`LuaModule::pushLuaUpvalues`会将`LuaModule`实例的指针作为值推入。要检索它，我们可以添加以下函数：

```cpp
class LuaModuleExporter final : public LuaModule
{
private:
    static LuaModuleExporter<T> *getExporter(
        lua_State *L)
    {
        return reinterpret_cast<LuaModuleExporter<T> *>(
            lua_touserdata(L, lua_upvalueindex(1)));
    }
};
```

这与第六章中的`getObj`函数相同，但现在它是一个静态成员函数。

通过一种从静态成员函数访问导出器实例的方式，我们可以将`LuaModuleExporter::luaNew`编写如下：

```cpp
static int luaNew(lua_State *L)
{
    auto luaModuleDef = getExporter(L)->luaModuleDef;
    T *obj = new T();
    T **userdata = reinterpret_cast<T **>(
        lua_newuserdatauv(L, sizeof(obj), 0));
    *userdata = obj;
    auto metatableName = luaModuleDef.metatableName();
    int type = luaL_getmetatable(
        L, metatableName.c_str());
    if (type == LUA_TNIL)
    {
        lua_pop(L, 1);
        luaL_newmetatable(L, metatableName.c_str());
        lua_pushvalue(L, -1);
        lua_setfield(L, -2, "__index");
        luaL_setfuncs(
            L, luaModuleDef.moduleRegs.data(), 0);
        lua_pushcfunction(L, luaDelete);
        lua_setfield(L, -2, "__gc");
    }
    lua_setmetatable(L, 1);
    return 1;
}
```

这实际上是从`Destinations.cc`中复制的。除了使用`T` `typename`代替硬编码的类名之外，更改已在前面的代码中突出显示。你可以看到它们都是关于泵送`LuaModuleDef`。

如果你忘记了`luaNew`是如何工作的，你可以查看上一章，其中有一些图表显示了 Lua 堆栈如何变化。

最后，让我们实现`LuaModuleExporter::luaDelete`的存根如下：

```cpp
static int luaDelete(lua_State *L)
{
    T *obj = *reinterpret_cast<T **>(
        lua_touserdata(L, 1));
    delete obj;
    return 0;
}
```

`luaDelete`在`luaNew`中注册为`__gc`元方法。

你还记得吗？

如前一章所述，我们将`luaDelete`设置为`luaNew`中创建的用户数据的终结器。在 Lua 垃圾回收过程中，终结器将被调用，参数为用户数据引用。

你也可以在`Destinations.cc`中删除`REGS`、`FACTORY_REGS`、`luaNew`和`luaDelete`。它们不再使用。

现在，我们可以测试导出器。执行项目。如果你一切都做对了，你应该会看到以下输出：

```cpp
Destinations instance created: 0x12a704170
[Lua] Visited: Paris
[Lua] Unvisited: Amsterdam London
Destinations instance destroyed: 0x12a704170
```

我们并没有真正改变上一章中的测试代码，除了`Destinations`类如何导出到 Lua 的方式。

如果你遇到了任何错误，不要气馁。这是这本书中最复杂的一章，我们需要在多个文件中正确实现代码才能使其工作。回顾你的步骤并修复错误。你可以做到！此外，在 GitHub 上，有多个针对本章的检查点项目，你可以参考。如前所述，我们不会自动生成 `lua_CFunction` 包装器。泛化也需要有界限。

但是，让我们检查我们的实现有多通用。

# 你足够灵活吗？

为了回答这个问题，让我们将 `script.lua` 修改如下：

```cpp
dst = Destinations.new("Shanghai", "Tokyo")
dst:wish("London", "Paris", "Amsterdam")
dst:went("Paris")
print("Visited:", dst:list_visited())
print("Unvisited:", dst:list_unvisited())
```

是的，新的要求是在 Lua 代码中，当创建 `Destinations` 对象时，我们可以提供一个未访问地点的初始列表。

这意味着我们需要支持参数化对象创建。

我们的导出器支持这个功能吗？这应该是一个常见的用例。

现在是思考人生、喝杯咖啡或做任何其他事情的好时机。我们几乎接近这本书的第三部分结束。

如果你还记得，我们的对象创建代码如下：

```cpp
static int luaNew(lua_State *L)
{
   ...
   T *obj = new T();
   ...
}
```

作为一名经验丰富的 C++ 程序员，你可能认为，因为 `std::make_unique<T>` 可以将其参数传递给 `T` 的构造函数，所以一定有办法让 `LuaModuleExporter<T>::make` 做同样的事情。没错，但 `std::make_unique<T>` 的魔力在于 C++ 编译时。那么，当参数在 C++ 代码编译后通过 Lua 代码传递时，你将如何处理呢？

别担心。让我们来探索**工厂方法**设计模式。工厂方法是一个定义为一个方法或接口的合约，用于创建并返回一个对象。然而，对象的创建方式并不重要，也不属于合约的一部分。

为了看到它是如何工作的，让我们为 `LuaModuleDef` 实现一个。添加另一个成员变量，命名为 `createInstance`，如下所示：

```cpp
struct LuaModuleDef
{
    const std::function<T *(lua_State *)>
    createInstance =
    [](lua_State *) -> T* { return new T(); };
};
```

这是一点高级的 C++ 使用。因此，重要的是你要考虑以下几点：

+   `createInstance` 被声明为一个成员变量，而不是一个成员函数。这是因为你可以在对象构造期间简单地给成员变量赋一个不同的值以实现不同的行为，但如果你使用成员函数，你需要创建一个子类来覆盖行为。*我们应该在可能的情况下优先选择组合而不是继承*。

+   `createInstance` 的类型是 `std::function`。使用这种类型，你可以像使用函数一样使用这个变量。如果你在这方面更熟悉 Lua，你会明白一个命名的 Lua 函数也是一个变量。在这里，我们想要达到相同的效果。`T *(lua_State *)` 是函数的类型。这意味着该函数期望一个类型为 `lua_State*` 的参数，并将返回一个类型为 `T` 的指针。你可以查看 C++ 参考手册来了解更多关于 `std::function` 的信息。

+   然后，我们提供一个默认的实现，作为一个 C++ lambda 表达式。这个 lambda 表达式简单地创建一个堆中的实例，没有任何构造函数参数。

要使用这个工厂方法，按照以下方式更改 `LuaModuleExporter::luaNew`：

```cpp
static int luaNew(lua_State *L)
{
    ...
    T *obj = luaModuleDef.createInstance(L);
    ...
}
```

我们已经从 `new T()` 更改为 `luaModuleDef.createInstance(L)`，它仍然做同样的事情。

然而，请注意，我们不再在 `LuaModuleExporter` 中创建对象。

最后，为了回答这个问题，是的，我们足够灵活。

关于现代 C++

在 1998 年，C++ 首次标准化为 C++98。直到 2011 年的 C++11，它变化很小。从那时起，C++ 快速采用了语言规范中的现代编程技术。Lambda 和 `std::function` 只是众多例子中的两个。如果你了解其他语言（例如 Java），你可以做一些类比（lambda 和函数式接口），尽管语法不同。我以这种方式实现了 `LuaModuleDef`，而不是使用更传统的方法，以展示一些现代 C++ 特性的例子。这是未来，我鼓励你更深入地探索现代 C++。使用 Java、Kotlin 和 Swift 的人默认使用这些技术。通过采用这些新技术并帮助 C++ 赶上，你可以在这里扮演重要的角色。

在 `Destinations.cc` 中，按照以下方式更改 `LuaModuleDef` 实例：

```cpp
LuaModuleDef DestinationsLuaModuleDef::def =
LuaModuleDef<Destinations>{
    "Destinations",
    {{"wish", luaWish},
     ...
     {NULL, NULL}},
    [](lua_State *L) -> Destinations *
    {
        Destinations *obj = new Destinations();
        std::vector<std::string> places;
        int nArgs = lua_gettop(L);
        for (int i = 1; i <= nArgs; i++)
        {
            places.push_back(lua_tostring(L, i));
        }
        obj->wish(places);
        return obj;
    },
};
```

这将使用提供的 lambda 初始化 `createInstance` 字段，而不是默认的 lambda。新的 lambda 与 `luaWish` 包装器做类似的事情。这个优点在于你可以完全控制这个 lambda。你可以为 `Destinations` 类创建另一个构造函数，并简单地调用新的构造函数。

我们可以用新的 Lua 脚本测试项目。你应该能看到以下输出：

```cpp
Destinations instance created: 0x142004170
[Lua] Visited: Paris
[Lua] Unvisited: Amsterdam London Shanghai Tokyo
Destinations instance destroyed: 0x142004170
```

如你所见，`上海` 和 `东京` 已经被添加到未访问列表中。

更进一步的设计改进

我们在 `LuaModuleDef` 中创建对象，但在 `LuaModuleExporter` 中销毁它们，并且我们的用例不涉及对象所有权的转移。为了更好的设计，应该由创建对象的同一类销毁它们，这将在下一章中实现。

这次，真的是完成了。

# 摘要

在这一章中，我们实现了一个通用的 C++ 模块导出器，主要用于对象创建部分。这确保了你只需实现一次复杂对象创建逻辑，就可以与许多 C++ 类一起重用。此外，这一章标志着 *第三部分*，*从 Lua 调用 C++* 的结束。

在下一章中，我们将回顾 Lua 和 C++ 之间的不同通信机制，并进一步探讨它们。

# 练习

这是一个开放练习。你可以编写一个新的 C++ 类，或者找到你过去工作中的某个类，然后使用 `LuaModuleExporter` 将其导出到 Lua。尝试提供一个有趣的 `createInstance` 实现以及。

# 第四部分 – 高级主题

到这本书的这一部分，你将已经学会了所有将 Lua 与 C++ 集成的常见机制。

在这部分，你将回顾你所学的知识，这也可以作为快速参考的来源。你还将学习如何实现一个可以被 Lua 加载的独立 C++模块，作为一个可动态加载的库。然后，你将学习一些高级内存管理技术以及如何使用 Lua 实现多线程。

本部分包括以下章节：

+   *第九章*，*回顾 Lua-C++通信机制*

+   *第十章*，*管理资源*

+   *第十一章*，*使用 Lua 进行多线程*
