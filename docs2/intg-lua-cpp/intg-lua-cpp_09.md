# 9

# 回顾 Lua-C++ 通信机制

在本书的 *第二部分* 中，我们学习了如何从 C++ 调用 Lua。在 *第三部分* 中，我们学习了如何从 Lua 调用 C++。在本书的学习过程中，我们探讨了众多示例，其中一些示例依赖于高级 C++ 技术。

本章将总结 Lua 和 C++ 之间的所有通信机制，去除大部分 C++ 的细节。我们还将深入探讨一些在示例中尚未展示的主题。

您可以使用本章回顾您所学的内容。对于每个主题，我们将列出一些重要的 Lua 库函数。您可以在 Lua 参考手册中查找更多相关函数。

在未来的编程旅程中，随着您的进步，您可能会在项目中采用不同的 C++ 技术。在这种情况下，本章将是一个有用的快速参考来源。

我们将涵盖以下主题：

+   栈

+   从 C++ 调用 Lua

+   从 Lua 调用 C++

+   实现独立的 C++ 模块

+   在 Lua 中存储状态

+   用户数据

# 技术要求

您可以访问本章的源代码，网址为 [`github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter09`](https://github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter09)。

您可以访问 Lua 参考手册，并养成在 [`www.lua.org/manual/5.4/`](https://www.lua.org/manual/5.4/) 频繁检查 API 详细信息的习惯。

# 栈

Lua 栈可以服务于两个目的：

+   *在 C++ 和 Lua 之间交换数据**。传递函数参数和检索函数返回值适用于此用途。

+   *保留中间结果**。例如，我们可以在完成表格之前在栈上保留一个表格引用；我们可以将一些值推送到栈上，然后弹出并使用它们作为 upvalues。

Lua 栈有两种形式：

+   *Lua 状态附带的可公开访问的公共栈**。一旦通过 `luaL_newstate` 或 `lua_newstate` 创建了 Lua 状态，您就可以传递状态并在所有可以访问 Lua 状态的函数中访问相同的 Lua 栈。

+   *每个* `lua_CFunction` 调用的**私有栈**。栈仅对函数调用可访问。多次调用相同的 `lua_CFunction` 不会共享相同的栈。因此，传递给 `lua_CFunction` 调用的栈是函数调用的私有栈。

## 推送到栈上

您可以使用 `lua_pushXXX` 函数将值或对象引用推送到栈上——例如，`lua_pushstring`。

查看 Lua 参考手册以获取此类函数的列表。

## 查询栈

您可以使用 `lua_isXXX` 函数检查给定栈位置是否包含特定类型的项。

您可以使用 `lua_toXXX` 函数将给定栈位置转换为特定类型。这些函数总是会成功，尽管如果栈位置包含不同类型的项，则结果值可能会令人惊讶。

您可以在 Lua 参考手册中查找此类函数的列表。

## 其他栈操作

有一些其他常用的栈操作。

### 确保栈大小

Lua 栈以预定义的大小创建，应该足够大，可以满足大多数操作。如果您需要将大量项推送到栈上，您可以通过调用以下函数来确保栈大小可以满足您的需求：

```cpp
int lua_checkstack (lua_State *L, int n);
```

`n` 是所需的大小。

### 计数项

要检查栈中的项数，请使用 `lua_gettop`。返回值是计数。

### 重置栈顶

要将栈顶设置为特定索引，请使用 `lua_settop` 函数，其声明如下：

```cpp
void lua_settop (lua_State *L, int index);
```

这可以清除栈顶的一些项，或者用 nil 填充栈。我们可以使用它来有效地从栈中清除临时项，如我们示例中的 `LuaModuleExporter::luaNew` 所示：

```cpp
    T *obj = luaModuleDef.createInstance(L);
    lua_settop(L, 0);
```

在 `luaNew` 中，我们传递了 Lua 状态，即 Lua 栈，到一个外部工厂方法。因为我们不知道工厂方法将如何使用 Lua 栈，所以在工厂方法返回后我们清空了栈，以消除任何可能的副作用。

### 复制另一个项

如果一个项已经在栈中，您可以通过调用此函数快速将其副本推送到栈顶：

```cpp
void lua_pushvalue (lua_State *L, int index);
```

如果支持项的值或对象难以获取，这可以为您节省一些麻烦。

Lua 库支持一些其他栈操作，但它们很少用来实现复杂效果。您可以在参考手册中查看它们。

# 从 C++ 调用 Lua

要从 C++ 调用 Lua 代码，我们可以使用 `lua_pcall`，其声明如下：

```cpp
int lua_pcall(
    lua_State *L, int nargs, int nresults, int msgh);
```

这将调用一个 Lua 可调用项，它可以是一个函数或一个代码块。您可以将要调用的 Lua 函数推送到栈上，或者将文件或字符串编译成一个代码块并将其放置到栈上。`nargs` 是可调用项的参数数量。参数被推送到可调用项之上的栈上。`nresults` 是可调用项将返回的返回值数量。使用 `LUA_MULTRET` 来表示您期望一个可变数量的返回值。`msgh` 是错误消息处理器的栈索引。

`lua_pcall` 在 *保护模式* 下调用可调用项，这意味着调用链中可能发生的任何错误都不会传播。相反，`lua_pcall` 从中返回一个错误状态码。

在我们实现的 `LuaExecutor` 类中，您可以找到许多从 C++ 调用 Lua 的示例。

在 Lua 参考手册中，您可以找到其他类似于 `lua_pcall` 的库函数，尽管 `lua_pcall` 是最常用的一个。

# 从 Lua 调用 C++

要从 Lua 调用 C++ 代码，C++ 代码需要通过 `lua_CFunction` 实现导出，其定义如下：

```cpp
typedef int (*lua_CFunction) (lua_State *L);
```

例如，在 `LuaExecutor` 中，我们实现了一个函数：

```cpp
int luaGetExecutorVersionCode(lua_State *L)
{
   lua_pushinteger(L, LuaExecutor::versionCode);
   return 1;
}
```

这将返回一个整数给 Lua 代码。将此函数导出到全局表的一个简单方法可以如下实现：

```cpp
void registerHostFunctions(lua_State *L)
{
    lua_pushcfunction(L, luaGetExecutorVersionCode);
    lua_setglobal(L, "host_version");
}
```

您可以使用 `lua_pushcfunction` 将 `lua_CFunction` 推送到栈上，然后将其分配给您选择的变量。

然而，更有可能的是，您应该将一组函数作为一个模块导出。

## 导出 C++ 模块

要导出 C++ 模块，您只需将函数表导出到 Lua。在 `LuaExecutor` 中，我们是这样实现的：

```cpp
void LuaExecutor::registerModule(LuaModule &module)
{
    lua_createtable(L, 0, module.luaRegs().size() - 1);
    int nUpvalues = module.pushLuaUpvalues(L);
    luaL_setfuncs(L, module.luaRegs().data(), nUpvalues);
    lua_setglobal(L, module.luaName().c_str());
}
```

该过程是首先创建一个表，然后使用 `lua_createtable` 将引用推送到栈上。然后，您可以推送 *共享 upvalues*（我们将在本章后面回顾 upvalues），最后使用 `luaL_setfuncs` 将函数列表添加到表中。

如果您不需要 upvalues，您可以使用一个快捷方式：

```cpp
void luaL_newlib (lua_State *L, const luaL_Reg l[]);
```

`luaL_newlib` 和 `luaL_setfuncs` 都接受一个结构列表来描述函数：

```cpp
typedef struct luaL_Reg {
    const char *name;
    lua_CFunction func;
} luaL_Reg;
```

结构为 `lua_CFunction` 提供了一个 `name` 值，该值用作表条目的键。

# 实现独立 C++ 模块

到目前为止，在这本书中，我们只在 C++ 代码中显式将 C++ 模块注册到 Lua。然而，还有另一种方法向 Lua 提供一个 C++ 模块。

您可以为模块生成共享库并将其放置在 Lua 的搜索路径中。当 Lua 代码 *require* 模块时，Lua 将自动加载共享库。

通过重用我们的 `Destinations` 类，这很简单实现。创建一个名为 `DestinationsModule.cpp` 的文件，并按照以下内容填充：

```cpp
#include "Destinations.h"
#include "LuaModuleExporter.hpp"
#include <lua.hpp>
namespace {
    LuaModuleExporter module =
        LuaModuleExporter<Destinations>::make(
            DestinationsLuaModuleDef::def);
}
extern "C" {
int luaopen_destinations(lua_State *L)
{
    lua_createtable(L, 0, module.luaRegs().size() - 1);
    int nUpvalues = module.pushLuaUpvalues(L);
    luaL_setfuncs(L, module.luaRegs().data(), nUpvalues);
    return 1;
}
}
```

已实现的模块称为 `destinations`。Lua 所需的代码级合约如下：

+   您需要提供 `lua_CFunction`，其名称必须以 `luaopen_` 开头，然后附加模块名称。

+   `lua_CFunction` 需要将它创建的内容留在栈上。

`luaopen_destinations` 的代码几乎与我们在上一节中解释的 `LuaExecutor::registerModule` 相同。唯一的区别是我们将表引用留在了栈上，因为 Lua 的 `require` 函数会弹出它。

extern “C”

默认情况下，C++ 编译器会对 C++ 函数名进行名称修饰。这意味着在函数编译后，函数将具有比源代码中声明的更复杂的符号名。为了防止这种情况发生，您可以将函数声明放在一个 `extern "C"` 块中。否则，Lua 在编译后无法找到该函数，因为合约已被破坏。

## 编译独立模块

要编译共享库，请将以下行添加到您的 `Makefile` 中：

```cpp
DESTINATIONS_O = Destinations.o DestinationsModule.o
DESTINATIONS_SO = destinations.so
destinations: ${DESTINATIONS_O}
    $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -shared \
        -o $(DESTINATIONS_SO) ${DESTINATIONS_O} -llua
```

在终端中，执行 `make destinations` 命令以创建共享库。您将得到一个名为 `destinations.so` 的文件，这是 Lua 将加载的二进制文件。

## 测试独立模块

要测试独立模块，在 `destinations.so` 所在的文件夹中，启动 Lua 交互式解释器并执行以下语句：

```cpp
Chapter09 % ../lua/src/lua
Lua 5.4.6 Copyright (C) 1994-2023 Lua.org, PUC-Rio
> Destinations = require "destinations"
> dst = Destinations.new("Shanghai", "Tokyo")
Destinations instance created: 0x155a04210
> dst:wish("London", "Paris", "Amsterdam")
> dst:went("Paris")
> print("Visited:", dst:list_visited())
Visited: Paris
> print("Unvisited:", dst:list_unvisited())
Unvisited: Amsterdam London Shanghai Tokyo
> os.exit()
Destinations instance destroyed: 0x155a04210
```

最重要的语句是 `require` 语句。它加载 `destinations.so` 并将模块分配给 `Destinations` 全局变量。

我们在模块二进制文件所在的同一文件夹中启动了 Lua 交互式解释器，因为`require`将在当前工作目录中搜索模块。或者，你也可以将库放在系统搜索路径中。你可以查看参考手册了解更多关于`require`及其行为的信息。

当你需要跨多个项目重用二进制形式的模块或在 C++侧强制代码隔离时，一个独立的 C++模块很有用，但这只是一个设计选择。

# 在 Lua 中存储状态

在 Lua 中存储状态有两种方式为`lua_CFunction`：*upvalues*和*注册表*。让我们回顾一下它们，并更深入地了解 upvalues。

## Upvalues

为了引入 upvalue 的完整定义，我们需要同时引入**Lua C 闭包**。引用 Lua 参考手册：

当创建 C 函数时，可以将其与一些值关联起来，从而创建一个 C 闭包；这些值被称为 upvalue，并且每当函数被调用时都可以访问。

简单来说，闭包仍然是我们的老朋友`lua_CFunction`。当你将其与一些值关联时，它就变成了闭包，而这些值就变成了 upvalue。

重要的是要注意，Lua C 闭包和 upvalue 是不可分割的。

要创建一个闭包，请使用以下库函数：

```cpp
void lua_pushcclosure(
    lua_State *L, lua_CFunction fn, int n);
```

这从`lua_CFunction`创建了一个闭包，并将`n`个值与它关联。

要看到它的实际效果，让我们解决前一章的设计问题：

我们在`LuaModuleDef`中创建对象，但在`LuaModuleExporter`中销毁它们。为了更好的设计，应该由创建对象的同一类销毁它们。

### 实现 Lua C 闭包

以下特性是前一章的延续。如果你需要更好的理解，可以回顾前一章。

要做到这一点，我们可以为`LuaModuleDef`实现一个`destroyInstance`成员变量，如下所示：

```cpp
struct LuaModuleDef
{
    ...
    const std::function<void(T *)> destroyInstance =
        [](T *obj) { delete obj; };
    ...
};
```

现在，对象将在同一个`LuaModuleDef`实体中创建和销毁。要使用`destroyInstance`，修改`LuaModuleExporter::luaDelete`，如下所示：

```cpp
static int luaDelete(lua_State *L)
{
    auto luaModuleDef = getExporter(L)->luaModuleDef;
    T *obj = *reinterpret_cast<T **>(
        lua_touserdata(L, 1));
    luaModuleDef.destroyInstance(obj);
    return 0;
}
```

回想一下，`getExporter`用于检索第一个 upvalue，它是指向导出器的指针：

```cpp
static LuaModuleExporter<T> *getExporter(lua_State *L)
{
    return reinterpret_cast<LuaModuleExporter<T> *>(
        lua_touserdata(L, lua_upvalueindex(1)));
}
```

这对`luaNew`有效，因为`LuaModuleExporter`是从`LuaModule`继承的，它在默认实现中将其`this`作为 upvalue 压入栈中：

```cpp
class LuaModule
{
public:
    virtual int pushLuaUpvalues(lua_State *L)
    {
        lua_pushlightuserdata(L, this);
        return 1;
    }
};
```

然后，压入的 upvalue 被用作`LuaExecutor::registerModule`中所有导出函数的共享 upvalue：

```cpp
void LuaExecutor::registerModule(LuaModule &module)
{
    lua_createtable(L, 0, module.luaRegs().size() - 1);
    int nUpvalues = module.pushLuaUpvalues(L);
    luaL_setfuncs(L, module.luaRegs().data(), nUpvalues);
    lua_setglobal(L, module.luaName().c_str());
}
```

共享的 upvalue 只被压入栈中一次，并且与提供给`luaL_setfuncs`的所有函数相关联。

共享 upvalue 并不是真正共享

所说的共享 upvalues 在设置期间为每个函数复制。之后，函数访问它们自己的 upvalue 副本。在 Lua 参考手册中，这些被称为共享 upvalues，因为它们只被压入栈中一次，用于所有要注册的函数，这对于 API 调用来说只相关。我认为这个术语是误导性的。你应该把它们看作是普通的 upvalue。

然而，`getExporter` 对于 `luaDelete` 不会起作用，因为 `luaDelete` 不是一个导出函数，也没有传递给 `luaL_setfuncs`。为了支持 `luaDelete`，修改 `luaNew`，如下所示：

```cpp
static int luaNew(lua_State *L)
{
    auto exporter = getExporter(L);
    auto luaModuleDef = exporter->luaModuleDef;
    ...
    if (type == LUA_TNIL)
    {
        ...
        lua_pushlightuserdata(L, exporter);
        lua_pushcclosure(L, luaDelete, 1);
        lua_setfield(L, -2, "__gc");
    }
    ...
}
```

我们只需要将 `exporter` 作为 upvalue 压入 `luaDelete`，并将 `luaDelete` 转换为闭包。

现在，`LuaModuleExporter` 有了一个更好的设计，因为它将对象构造和对象销毁都委托给了 `LuaModuleDef`。同时，它在 `getExporter` 辅助函数中同时使用了 upvalues（用于 `luaDelete`）和共享 upvalues（用于 `luaNew`）。这表明设置后共享 upvalues 和 upvalues 没有区别。

## 注册表

注册表是一个预定义的 Lua 表，仅对 C/C++ 代码可访问。对于 Lua 状态，注册表对所有 C/C++ 函数是共享的，因此应仔细选择表键名以避免冲突。

值得注意的是，按照惯例，*完整用户数据*通过 `luaL_newmetatable` 将其元表放置在注册表中。

*简单来说，注册表是 Lua 语言特别对待的 Lua 表，并提供了一些辅助函数*。

# 用户数据

Lua 用户数据可以分为 *轻量级用户数据* 和 *完整用户数据*。

重要的是要注意，它们是不同的事物。在 Lua 库中，按照惯例，轻量级用户数据命名为 lightuserdata，而完整用户数据命名为 userdata。

## 轻量级用户数据

轻量级用户数据代表一个 C/C++ 指针。它是一个值类型，值在各个地方传递。你可以在 C/C++ 代码中使用 `lua_pushlightuserdata` 将一个指针压入栈中。你不能使用 Lua 库创建轻量级用户数据。

## 完整用户数据

完整用户数据是 Lua 库通过调用 `lua_newuserdatauv` 分配的原始内存区域。它是一个对象类型，只有其引用在各个地方传递。

因为完整用户数据是由 Lua 在堆中创建的，所以 Lua 垃圾回收就变得重要了。在 C++ 方面，你可以通过提供 `__gc` 元方法来提供一个 *终结器*。

有关如何利用完整用户数据在 Lua 中访问 C++ 对象的完整示例，请查看 `LuaModuleExporter`。

# 概述

在本章中，我们简要回顾了 Lua 和 C++ 之间的所有通信机制。这应该已经充分巩固了您到目前为止的学习。

我们还学习了如何生成一个独立的 C++ 模块作为共享库。这为您组织项目开辟了新的途径。

在下一章中，我们将更详细地讨论资源管理。
