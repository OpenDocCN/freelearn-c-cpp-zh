# 10

# 管理资源

在上一章中，我们回顾了 Lua 和 C++之间的通信机制。在本章中，我们将学习更多关于管理资源的内容。资源可以是对象使用的任何东西，例如内存、文件或网络套接字。

我们将涵盖以下主题：

+   自定义 Lua 内存分配

+   将 C++对象内存分配委托给 Lua

+   什么是 RAII？

# 技术要求

我们将使用第九章的源代码作为基础来开发本章的示例。请确保你可以访问本书的源代码：[`github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter10`](https://github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter10)。

# 自定义 Lua 内存分配

在 Lua 运行时，以下情况下在堆中分配、重新分配或释放内存：

+   **内存分配**：当创建对象时会发生这种情况。Lua 需要分配一块内存来存储它。

+   **内存重新分配**：当需要更改对象大小时会发生这种情况，例如，当表格没有更多预分配空间时，向表格中添加条目。

+   **内存释放**：在垃圾回收期间，当对象不再需要时发生。

在大多数情况下，你不需要关心这个问题。但有时，了解或自定义 Lua 内存分配是有帮助的。以下是一些示例：

+   你需要分析 Lua 对象的内存占用情况，以找到优化机会。

+   你需要自定义内存的分配位置。例如，为了提高运行时效率，你可能有一个内存池，你可以简单地让 Lua 使用它，而不需要在堆中每次都分配新的内存区域。

在本节中，我们将了解如何通过提供内存分配函数来自定义 Lua 的内存分配。

## Lua 内存分配函数是什么？

Lua 提供了一种简单的方式来自定义内存分配。当你创建 Lua 状态时，你可以提供一个内存分配函数，这样每当 Lua 需要管理内存时，它就会调用你提供的函数。

内存分配函数定义为以下类型：

```cpp
typedef void * (*lua_Alloc) (void *ud,
                             void *ptr,
                             size_t osize,
                             size_t nsize);
```

函数返回指向新分配内存的指针，或者如果调用用于释放一块内存，则返回`NULL`。其参数解释如下：

+   `ud`是 Lua 状态的用户定义数据的指针。你可以使用相同的内存分配函数与多个 Lua 状态一起使用。在这种情况下，你可以使用`ud`来识别每个 Lua 状态。Lua 对此是透明的。

+   `ptr`是要重新分配或释放的内存的指针。如果它是`NULL`，则调用内存分配器的目的是分配一个新的内存块。

+   `osize`是由`ptr`指向的先前分配的内存的原始大小。如果`ptr`是`NULL`，则`osize`具有特殊含义——正在分配的 Lua 对象的类型，可以是`LUA_TSTRING`、`LUA_TTABLE`等。

+   `nsize` 是要分配或重新分配的内存大小。如果 `nsize` 为 `0`，则表示要释放内存。

要注册你的内存分配函数，你可以使用 `lua_newstate` 来创建 Lua 状态，其声明如下：

```cpp
lua_State *lua_newstate (lua_Alloc f, void *ud)
```

通过这种方式，你为要创建的 Lua 状态提供了内存分配函数和用户数据。请注意，你可以向 `ud` 提供空指针，并且这个用户数据是 C++ 端的对象，而不是 Lua 用户数据。

接下来，我们将实现一个内存分配函数。

## 实现内存分配函数

我们将扩展 `LuaExecutor` 来练习实现内存分配函数。当我们创建执行器时，我们想要传递一个标志来指示是否应该使用我们自己的内存分配函数。

你可以基于 *第九章* 的源代码开始这项工作。在 `LuaExecutor.h` 中修改构造函数，如下所示：

```cpp
class LuaExecutor
{
public:
    LuaExecutor(const LuaExecutorListener &listener,
                bool overrideAllocator = false);
};
```

我们为构造函数添加了另一个布尔参数 `overrideAllocator`。我们还提供了一个默认值 `false`，因为在大多数情况下，我们不需要覆盖 Lua 内存分配器。

在 `LuaExecutor.cc` 中，在一个新的匿名命名空间中实现我们的内存分配函数，如下所示：

```cpp
namespace
{
void *luaAlloc(
    void *ud, void *ptr, size_t osize, size_t nsize)
{
    (void)ud;
    std::cout << "[luaAlloc] ptr=" << std::hex << ptr
              << std::dec << ", osize=" << osize
              << ", nsize=" << nsize;
    void *newPtr = NULL;
    if (nsize == 0)
    {
        free(ptr);
    }
    else
    {
        newPtr = realloc(ptr, nsize);
    }
    std::cout << std::dec << ", newPtr=" << newPtr
              << std::endl;
    return newPtr;
}
}
```

`luaAlloc` 依赖于标准的 `realloc` 和 `free` C 函数来分配、重新分配和释放内存。这正是默认 Lua 分配器所做的事情。但我们也记录了参数和返回值，以便更深入地了解内存使用情况。

要使用 `luaAlloc`，在 `LuaExecutor.cc` 中修改构造函数，如下所示：

```cpp
LuaExecutor::LuaExecutor(
    const LuaExecutorListener &listener,
    bool overrideAllocator)
    : L(overrideAllocator ? lua_newstate(luaAlloc, NULL)
                          : luaL_newstate()),
      listener(listener)
{ ... }
```

在这里，我们检查 `overrideAllocator` 是否为 `true`。如果是，我们通过调用 `lua_newstate` 使用我们的内存分配函数。如果不是，我们通过调用 `luaL_newstate` 使用默认分配器。

现在，让我们测试我们的分配器。

## 测试它

重新编写 `main.cpp`，如下所示：

```cpp
#include "LuaExecutor.h"
#include "LoggingLuaExecutorListener.h"
#include "LuaModuleExporter.hpp"
#include "Destinations.h"
int main()
{
    auto listener = std::make_unique<
        LoggingLuaExecutorListener>();
    auto lua = std::make_unique<LuaExecutor>(
        *listener, true);
    auto module = LuaModuleExporter<Destinations>::make(
        DestinationsLuaModuleDef::def);
    lua->registerModule(module);
    lua->executeFile("script.lua");
    return 0;
}
```

测试代码创建了一个 Lua 执行器，注册了 `Destinations` 模块，并执行了 `script.lua`。这与我们在前面的章节中所做的是相似的。唯一需要注意的是，我们在创建 `LuaExecutor` 实例时将 `overrideAllocator` 设置为 `true`。

重新编写 `script.lua`，如下所示：

```cpp
print("======script begin======")
dst = Destinations.new()
dst:wish("London", "Paris", "Amsterdam")
dst:went("Paris")
print("Visited:", dst:list_visited())
print("Unvisited:", dst:list_unvisited())
print("======script end======")
```

脚本创建了一个 `Destinations` 类型的对象并测试了其成员函数。这又与我们在前面的章节中所做的是相似的。

我们还打印出标记来标记脚本开始和结束执行的时间。这有助于我们定位感兴趣的事物，因为自定义的内存分配函数将会非常详细。

编译并执行项目。你应该得到一个类似于以下输出的结果：

```cpp
...
[Lua] ======script begin======
[luaAlloc] ptr=0x0, osize=7, nsize=56, newPtr=0x14e7060c0
Destinations instance created: 0x14e7060e0
[luaAlloc] ptr=0x0, osize=4, nsize=47, newPtr=0x14e706100
[luaAlloc] ptr=0x0, osize=5, nsize=56, newPtr=0x14e706130
[luaAlloc] ptr=0x0, osize=0, nsize=48, newPtr=0x14e706170
[luaAlloc] ptr=0x0, osize=0, nsize=96, newPtr=0x14e7061a0
[luaAlloc] ptr=0x14e706170, osize=48, nsize=0, newPtr=0x0
...
[Lua] Visited: Paris
[Lua] Unvisited: Amsterdam London
[Lua] ======script end======
Destinations instance destroyed: 0x14e7060e0
...
```

这里突出显示的两行是 `0x14e706170` 地址处的对象分配和释放。你还会看到很多无关的内存分配输出，因为 Lua 也会使用自定义的内存分配函数来管理其内部状态的内存。

尽管这个定制的内存分配函数并不复杂，但你可以将所学内容扩展以改变内存管理的方式。这对于运行时优化或资源受限的系统很有用。

在下一节中，我们将探讨一个更高级的场景——*如何让 Lua 为 *C++ 对象* 分配内存。

# 将 C++ 对象的内存分配委托给 Lua

到目前为止，我们一直在 C++ 中创建 C++ 对象，并让 Lua 在 userdata 中存储其指针。这是在 `LuaModuleExporter::luaNew` 中完成的，如下所示：

```cpp
static int luaNew(lua_State *L)
{
    ...
    T **userdata = reinterpret_cast<T **>(
        lua_newuserdatauv(L, sizeof(T *), 0));
    T *obj = luaModuleDef.createInstance(L, nullptr);
    *userdata = obj;
    ...
}
```

在这种情况下，Lua userdata 只存储一个指针。如您所回忆的，Lua userdata 可以表示更大的内存块，所以你可能想知道我们是否可以将整个 C++ 对象存储在 userdata 中，而不仅仅是指针。是的，我们可以。让我们学习如何做到这一点。

## 使用 C++ placement new

在 C++ 中，创建对象最常见的方式是调用 `new T()`。这做两件事：

+   它为 `T` 类型的对象创建一块内存。

+   它调用 `T` 类型的构造函数。在我们的例子中，我们调用默认构造函数。

同样，销毁对象最常见的方式是调用 `delete obj`。它也做两件事：

+   它调用 `T` 类型的析构函数。在这里，`obj` 是 `T` 类型的对象。

+   释放持有 `obj` 的内存。

C++ 还提供了一个只通过调用构造函数来创建对象的 *new 表达式*。它不会为对象分配内存。相反，你告诉 C++ 将对象放在哪里。这个 *new 表达式* 被称为 **placement new**。

要使用 *placement new*，我们需要提供一个已分配内存的地址。我们可以用以下方式使用它：

```cpp
T* obj = new (addr) T();
```

我们需要在 `new` 关键字和构造函数之间提供内存位置的地址。

现在我们已经找到了一种方法来解耦 C++ 内存分配和对象构造，让我们扩展我们的 C++ 模块导出器以支持将内存管理委托给 Lua。

## 扩展 LuaModuleDef

我们在这本书中实现了一个 C++ 模块导出系统。它有两个部分：

+   `LuaModuleExporter` 抽象了模块注册并实现了模块的 Lua 终结器。

+   `LuaModuleDef` 定义了模块名称、导出函数以及对象的构造和销毁。

首先，我们将向 `LuaModuleDef` 添加使用预分配内存的能力。

在 `LuaModule.h` 中，添加一个名为 `isManagingMemory` 的新成员变量，如下所示：

```cpp
struct LuaModuleDef
{
    const bool isManagingMemory;
};
```

当 `isManagingMemory` 为 `true` 时，我们表示 `LuaModuleDef` 实例正在管理内存分配和释放。当 `isManagingMemory` 为 `false` 时，我们表示 `LuaModuleDef` 不管理内存。在后一种情况下，`LuaModuleExporter` 应该让 Lua 管理内存，这将在我们扩展 `LuaModuleDef` 之后实现。

在添加了新标志后，修改 `createInstance`，如下所示：

```cpp
const std::function<T *(lua_State *, void *)>
createInstance = this -> T *
{
    if (isManagingMemory)
    {
        return new T();
    }
    else
    {
        return new (addr) T();
    }
};
```

我们添加了一个新参数 - `void *addr`。当 `LuaModuleDef` 实例管理内存时，它使用正常的 *new 操作符* 分配内存。当实例不管理内存时，它使用 *放置新表达式*，其中 `addr` 是对象应该构造的地址。

这是 `createInstance` 的默认实现。当你创建 `LuaModuleDef` 实例时，你可以覆盖它并调用非默认构造函数。

接下来，我们需要修改 `destroyInstance` 以支持 `isManagingMemory`。更改其默认实现，如下所示：

```cpp
const std::function<void(T *)>
destroyInstance = this
{
    if (isManagingMemory)
    {
        delete obj;
    }
    else
    {
        obj->~T();
    }
};
```

当 `LuaModuleDef` 实例不管理内存时，我们只需调用对象的析构函数，`obj->~T()`，来销毁它。

放置删除？

如果你想知道是否有与 *放置新* 匹配的 *放置删除*，答案是 no。要销毁对象而不释放其内存，你可以简单地调用其析构函数。

随着 `LuaModuleDef` 准备支持两种内存管理方式，接下来，我们将扩展 `LuaModuleExporter`。

## 扩展 LuaModuleExporter

在我们为 C++ 对象将内存管理委托给 Lua 支持之前，我们将强调主要架构差异：

+   当 C++ 分配对象的内存时，就像我们在这本书中一直做的那样，Lua userdata 持有分配的内存地址的指针

+   当 Lua 分配对象的内存作为 userdata 时，userdata 持有实际的 C++ 对象

让我们开始扩展 `LuaModuleExporter`。我们需要修改 `luaNew` 和 `luaDelete`，以便它们能与 `LuaModuleDef::isManagingMemory` 一起工作。

在 `LuaModuleExporter.hpp` 中，修改 `luaNew`，如下所示：

```cpp
static int luaNew(lua_State *L)
{
    auto exporter = getExporter(L);
    auto luaModuleDef = exporter->luaModuleDef;
    if (luaModuleDef.isManagingMemory)
    {
        T **userdata = reinterpret_cast<T **>(
            lua_newuserdatauv(L, sizeof(T *), 0));
        T *obj = luaModuleDef.createInstance(L, nullptr);
        *userdata = obj;
    }
    else
    {
        T *userdata = reinterpret_cast<T *>(
            lua_newuserdatauv(L, sizeof(T), 0));
        luaModuleDef.createInstance(L, userdata);
    }
    lua_copy(L, -1, 1);
    lua_settop(L, 1);
    ...
}
```

函数被分解为四个块，由新行分隔。这些块如下：

+   代码的前两行获取 `LuaModuleExporter` 实例和 `LuaModuleDef` 实例。如果你需要了解 `getExporter` 的工作原理，可以回顾 *第八章*。

+   `if` 子句创建 C++ 模块对象和 Lua userdata。当 `luaModuleDef.isManagingMemory` 为 `true` 时，执行的代码与 *第八章* 中的相同。当它是 `false` 时，代码创建一个大小为 `sizeof(T)` 的 userdata 来持有实际的 `T` 实例。请注意，在这种情况下，userdata 的类型是 `T*`，其地址传递给 `luaModuleDef.createInstance` 以用于 *放置新*。

+   它通过 `lua_copy(L, -1, 1)` 将 userdata 复制到栈底，并通过 `lua_settop`(`L, 1`) 清除栈上除底部以外的所有内容。对象构造委托给 `LuaModuleDef` 以清除栈上的临时项，以防 `LuaModuleDef` 推送了任何项。与 *第八章* 中的代码相比，这两行代码是一个改进版本，可以覆盖更多情况以及不同的对象创建方式。

+   函数其余部分省略的代码保持不变。

最后，为了完成这个特性，修改 `LuaModuleExporter::luaDelete`，如下所示：

```cpp
static int luaDelete(lua_State *L)
{
    auto luaModuleDef = getExporter(L)->luaModuleDef;
    T *obj = luaModuleDef.isManagingMemory
        ? *reinterpret_cast<T **>(lua_touserdata(L, 1))
        : reinterpret_cast<T *>(lua_touserdata(L, 1));
    luaModuleDef.destroyInstance(obj);
    return 0;
}
```

我们需要更改在终结器中获取 C++ 模块实例的方式。差异在于 userdata 是否持有实际对象或对象的指针。

接下来，让我们测试 Lua 分配 C++ 对象内存的机制是否工作。

## 使用 `Destinations.cc` 模块进行测试

我们只需要稍微调整 `Destinations.cc` 中的代码，以支持两种内存分配场景。修改 `getObj`，如下所示：

```cpp
inline Destinations *getObj(lua_State *L)
{
    luaL_checkudata(L, 1, DestinationsLuaModuleDef::def
        .metatableName().c_str());
    if (DestinationsLuaModuleDef::def.isManagingMemory)
    {
        return *reinterpret_cast<Destinations **>(
            lua_touserdata(L, 1));
    }
    else
    {
        return reinterpret_cast<Destinations *>(
            lua_touserdata(L, 1));
    }
}
```

我们在这里所做的更改与我们之前对 `LuaModuleExporter::luaDelete` 所做的更改类似，以便它支持 Lua userdata 所持有的不同内容。

要选择让 Lua 分配内存，更改 `DestinationsLuaModuleDef::def`，如下所示：

```cpp
LuaModuleDef DestinationsLuaModuleDef::def =
LuaModuleDef<Destinations>{
    "Destinations",
    {{"wish", luaWish},
     {"went", luaWent},
     {"list_visited", luaListVisited},
     {"list_unvisited", luaListUnvisited},
     {NULL, NULL}},
    false,
};
```

在这里，我们将 `LuaModuleDef::isManagingMemory` 设置为 `false`。

编译并执行项目。你应该看到以下输出：

```cpp
Chapter10 % ./executable
[Lua] ======script begin======
Destinations instance created: 0x135e0af10
[Lua] Visited: Paris
[Lua] Unvisited: Amsterdam London
[Lua] ======script end======
Destinations instance destroyed: 0x135e0af10
```

如果你将 `LuaModuleDef::isManagingMemory` 设置为 `true`，它也应该可以工作。

谁应该管理 C++ 对象的内存？

你可以选择让 C++ 或 Lua 管理 C++ 对象的内存分配。在复杂项目中管理内存可以提供更好的控制。在 Lua 中管理内存可以消除双重指针间接引用。还有心理上的考虑。对于一些来自 C++ 世界的人来说，让 Lua 分配 C++ 对象，尤其是当 Lua 只是项目中使用的库之一时，可能会感觉违反了资源所有权。对于一些来自 Lua 或 C 世界的人来说，让 Lua 做更多的事情可能更容易接受。然而，在现实世界的项目中，这些细节将被隐藏。正如本节所示，如果你有一个抽象，它很容易从一种方式转换为另一种方式。

接下来，我想向您介绍 RAII 资源管理惯用语。

# 什么是 RAII？

本章全部关于资源管理。资源可以是一块内存、一个打开的文件或一个网络套接字。尽管在本章中我们只使用了内存管理作为示例，但所有资源的原理都是相同的。

当然，所有获取的资源都需要释放。在 C++ 中，析构函数是一个释放资源的好地方。当与 Lua 一起工作时，Lua 终结器是一个释放资源的好触发器。

**资源获取即初始化**，或 **RAII**，是一个有用的资源管理惯用语。这意味着对象的创建和获取对象所需的资源应该是一个原子操作——一切都应该成功，或者部分获取的资源应该在引发错误之前释放。

通过使用这种技术，资源也与对象的生存周期相关联。这确保了在对象的整个生存周期内所有资源都可用。这将防止复杂的故障场景。例如，假设一项工作已经完成了一半，资源已经消耗，但由于某种新的资源不可用，它无法完成。

当设计 C++类时，你可以确保所有资源都在构造函数中获取，并在析构函数中释放。当与 Lua 集成时，确保你提供最终化器，并从最终化器中销毁对象。

最终化器将在 Lua 的垃圾回收周期之一中被调用。你不应该假设 Lua 何时这样做，因为这在不同平台和 Lua 版本之间是不可移植的。如果你内存受限，你可以通过从 C++中调用`lua_gc`或从 Lua 中调用`collectgarbage`来手动触发垃圾回收周期。

垃圾回收仅用于内存

记住，垃圾回收仅用于内存。如果你有一个不使用其他资源的简单类，你可能不想提供 Lua 最终化器。但如果后来类被修改为依赖于非内存资源，那么添加最终化器可能不是更改的一部分。然后，你会在以后的某个时候发现资源从奇怪的错误报告中泄露出来。

RAII 在多线程编程中获取共享资源时也非常有用。我们将在下一章中看到一个例子。

# 摘要

在本章中，我们更多地了解了资源管理。我们学习了如何为 Lua 提供一个定制的内存分配函数。我们还学习了如何在 Lua userdata 中持有实际的 C++对象。最后，我们熟悉了 RAII 资源管理技术。

在下一章中，我们将探讨将 Lua 集成到 C++时的多线程。
