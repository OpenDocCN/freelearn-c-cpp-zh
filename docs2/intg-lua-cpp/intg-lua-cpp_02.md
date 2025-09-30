

# 第二章：Lua 基础知识

在本章中，我们将学习 Lua 编程语言的基础知识。如果你只专注于 C++ 端，你不需要成为 Lua 专家，甚至不需要编写任何 Lua 代码。然而，了解基础知识将使你在将 Lua 集成到 C++ 中时更加高效。

如果你已经了解 Lua 编程，你可以跳过本章。如果你有一段时间没有使用 Lua，你可以使用本章来复习。如果你想了解更多关于 Lua 编程的知识，你可以获取官方 Lua 书籍：*Programming in Lua*。如果你不知道 Lua 编程，本章是为你准备的。如果你来自 C++，你可以阅读本书中关于 Lua 代码的简要说明来阅读任何 Lua 代码。当你需要时，你可以相信自己并在线上进行研究。

我们将讨论以下语言特性：

+   变量和类型

+   控制结构

# 技术要求

你将使用交互式 Lua 解释器来跟随本章中的代码示例。我们是从 *第一章* 中的 Lua 源代码构建的。你也可以使用来自其他渠道的 Lua 解释器，例如，由你的操作系统包管理器安装的那个。在继续之前，请确保你有访问权限。

当你在本章中看到代码示例，如以下示例，你应该在交互式 Lua 命令行中尝试该示例：

```cpp
Lua 5.4.6  Copyright (C) 1994-2022 Lua.org, PUC-Rio
> os.exit()
%
```

Lua 解释器启动时输出的第一行是。使用 `os.exit()` 退出解释器。

你可以在本书的 GitHub 仓库中找到本章的源代码：[`github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter02`](https://github.com/PacktPublishing/Integrate-Lua-with-CPP/tree/main/Chapter02)

# 变量和类型

虽然你可能很清楚 C++ 是一种静态类型语言，但 Lua 是一种动态类型语言。在 C++ 中，当你声明一个变量时，你会给它一个明确的类型。在 Lua 中，每个值都携带自己的类型，你不需要显式指定类型。此外，在引用它之前，你不需要定义全局变量。尽管鼓励你声明它——或者更好的是，使用局部变量。我们将在本章后面学习局部变量。

在 Lua 中，有八个基本类型：**nil**、**boolean**、**number**、**string**、**userdata**、**function**、**thread** 和 **table**。

在本章中，我们将学习其中的六个：`nil`、`boolean`、`number`、`string`、`table` 和 `function`。在我们深入了解之前，让我们尝试一些：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> a
nil
> type(a)
nil
> a = true
> type(a)
boolean
> a = 6
> type(a)
number
> a = "bonjour"
> type(a)
string
```

这里发生的情况如下：

1.  交互式地，输入 `a` 检查全局变量 `a` 的值。由于它尚未定义，其值为 `nil`。

1.  使用 `type(a)` 检查变量 `a` 的类型。在这种情况下，值是 `nil`，因为它尚未定义。

1.  将 `true` 赋值给 `a`。使用 `=` 进行赋值；与 C++ 中的用法相同。

1.  现在，它的类型是 `boolean`。

1.  将 `6` 赋值给 `a`。

1.  现在，它的类型是 `number`。

1.  将 `"bonjour"` 赋值给 `a`。

1.  现在，它的类型是 `string`。

在那里执行的每一行也是 Lua 语句。与 C++ 语句不同，你不需要在语句末尾放置分号。

接下来，我们将学习更多关于这些类型的内容。

## 空值

`nil` 类型只有一个值来表示非值：`nil`。它的意义与 C++ 中的 **nullptr**（或 **NULL**）类似。

## 布尔值

`boolean` 类型有两个值。它们是 *true* 和 *false*。它在 Lua 中的行为与 C++ 中的 **bool** 相同。

## 数字

`number` 类型涵盖了 C++ 的 **int**、**float** 和 **double** 以及它们的变体（例如 **long**）。

在这里，我们还将学习 Lua 的算术运算符和关系运算符，因为它们主要用于数字。

### 算术运算符

算术运算符是执行数字算术运算的运算符。Lua 支持 *六个* 算术运算符：

+   `+`: 加法

+   `-`: 减法

+   `*`: 乘法

+   `/`: 除法

+   `%`: 取模

+   `//`: 向下取整除法

C++ 有七个算术运算符：+、-、*、/、%、++ 和 --。Lua 的算术运算符与它们的 C++ 对应项类似，但有以下不同之处：

1.  没有自增（++）或自减（--）运算符。

1.  C++ 没有返回除法结果整数部分的 `//` 运算符。由于 C++ 是强类型语言，C++ 可以通过正常除法隐式地实现相同的功能。以下是一个例子：

    ```cpp
    Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
    ```

    ```cpp
    > 5 / 3
    ```

    ```cpp
    1.6666666666667
    ```

    ```cpp
    > 5 // 3
    ```

    ```cpp
    1
    ```

1.  注意 Lua 是一种动态类型语言。这意味着 5 / 3 不会像 C++ 那样产生 1。

### 关系运算符

关系运算符是测试两个值之间某种关系的运算符。Lua 支持 *六个* 关系运算符：

+   `<`: 小于

+   `>`: 大于

+   `<=`: 小于或等于

+   `>=`: 大于或等于

+   `==`: 等于

+   `~=`: 不等于

`~=` 运算符测试不等于。这与 C++ 中的 `!=` 运算符相同。其他运算符与 C++ 中的相同。

## 字符串

在 Lua 中，字符串始终是常量。你不能改变字符串中的一个字符并使其代表另一个字符串。你需要为那个新字符串创建一个新的字符串。

我们可以用双引号或单引号界定字面量字符串。其余部分与 C++ 字符串非常相似，如下例所示：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> a = "Hello\n" .. 'C++'
> a
Hello
C++
> #a
9
```

字符串上有两个在 C++ 中找不到的运算符。要连接两个字符串，可以使用 `..` 运算符。要检查字符串的长度，可以使用 `#` 运算符。

与 C++ 转义序列类似，使用 `\` 转义特殊字符，如前一个输出中的换行符所示。如果你不想插入换行转义序列 `\n`，可以使用长字符串。

### 长字符串

你可以使用 `[[` 和 `]]` 来界定多行字符串，例如：

```cpp
a = [[
Hello
C++
]]
```

这定义了一个等于单行定义 `"Hello\nC++\n"` 的字符串。

长字符串可以使字符串更易于阅读。你可以在 Lua 源代码中使用长字符串轻松定义 **XML** 或 **JSON** 字符串。

如果你的长字符串内容中包含 `[[` 或 `]]`，你可以在开括号之间添加一些等号，例如：

```cpp
a = [=[
x[y[1]]
]=]
b = [==[
x[y[2]]
]==]
```

你可以添加多少个等号取决于你。然而，通常一个就足够了。

## 表

Lua 表类似于 **C++ std::map** 容器，但更灵活。表是 Lua 中构建复杂数据结构的唯一方式。

让我们尝试一个带有一些操作的 Lua 表。在一个 Lua 解释器中，逐个输入以下语句并观察输出：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> a = {}
> a['x'] = 100
> a['x']
100
> a.x
100
```

使用 `{}` 构造函数创建一个表，并将值 `100` 赋给字符串键 `'x'`。另一种构建此表的方法是 `a = {x = 100}`。要使用更多键初始化表，请使用 `a = {x = 100, y = 200}`。

`a.x` 是 `a['x']` 的另一种语法。你使用哪种取决于你的风格偏好。但通常，点语法意味着表被用作记录或面向对象的方式。

除了 `nil` 之外，你可以使用所有 Lua 类型作为表键。你还可以在同一个表中使用不同类型的值。你拥有完全的控制权，如下所示：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> b = {"Monday", "Tomorrow"}
> b[1]
Monday
> b[2]
Tomorrow
> #b
2
> a = {}
> b[a] = "a table"
> b[a]
a table
> b.a
nil
> #b
2
```

此示例解释了与表相关的四个要点：

1.  它首先以数组的形式创建一个表。请注意，它是从 1 开始索引的，而不是从 0 开始。当你向表构造函数提供一个仅包含值的条目时，它将被视为数组的一部分。你还记得 `#` 是长度运算符吗？当它用于表示序列或数组时，它可以告诉表的长度。

1.  然后它使用另一个表 `a` 作为键和 `"a table"` 字符串作为值添加另一个条目。这是完全可以的。

1.  注意，`b.a` 是 `nil`，因为 `b.a` 表示使用 `'a'` 字符串键的 `b['a']`，而不是 `b[a]`。

1.  最后，我们再次尝试检查表长度。我们在表中添加了 3 个条目，但它输出长度为 2。来自 C++ 的你可能感到惊讶：Lua 不提供检查表长度的内置方式。长度运算符仅在表表示序列或数组时提供便利。你能够同时将表用作数组和映射，但你需要承担全部责任。

在本章的后面部分，当我们学习 `for` 控制结构时，我们将了解更多关于表遍历的内容。现在我们将学习 Lua 函数。

## 函数

Lua 函数与 C++ 函数具有类似的作用。但与 C++ 不同，它们也是基本数据类型之一的一等公民。

我们可以通过以下方式定义一个函数：

1.  从 `function` 关键字开始。

1.  后面跟着一个函数名和一对括号，其中可以定义所需的函数参数。

1.  实现函数体。

1.  使用 `end` 关键字结束函数定义。

例如，我们可以定义一个函数如下：

```cpp
function hello()
    print("Hello C++")
end
```

这将输出 `"Hello C++"`。

要定义一个用于 Lua 解释器的函数，你有两种选择：

+   在交互式解释器中，只需开始输入函数。当你结束每一行时，解释器会知道，你可以继续输入函数定义的下一行。

+   或者，你可以在另一个文件中定义你的函数，该文件可以在以后导入到解释器中。这样做更容易工作。从现在开始，我们将使用这种方法。尽量将你的函数放在名为 `1-functions.lua` 的文件中的这个部分。

要调用函数，使用其名称和一对括号。这和调用 C++ 函数的方式一样，例如：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> dofile("Chapter02/1-functions.lua")
> hello()
Hello C++
```

`dofile()` 是 Lua 库中用于加载另一个 Lua 脚本的方法。在这里，我们加载定义了 Lua 函数的文件。如果你已经更改了脚本文件，你可以再次执行它来加载最新的脚本。

接下来，我们将学习关于函数参数和函数返回值的内容。

### 函数参数

函数参数，也称为参数，是在函数被调用时提供给函数的值。

你可以通过在函数名后面的括号内提供参数声明来定义函数参数。这和在 C++ 中一样，但你不需要提供参数类型，例如：

```cpp
function bag(a, b, c)
    print(a)
    print(b)
    print(c)
end
```

当调用函数时，你可以传递比定义的更多或更少的参数。例如，你可以调用我们刚刚定义的 `bag` 函数：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> dofile("Chapter02/1-functions.lua")
> bag(1)
1
nil
nil
> bag(1, 2, 3, 4)
1
2
3
```

你可以看到当提供的参数数量与定义的数量不同时会发生什么：

+   当传递的参数不足时，剩余的参数将具有 `nil` 值。

+   当传递的参数多于定义的参数时，额外的参数将被丢弃。

你不能为函数参数定义默认值，因为 Lua 在语言级别不支持它。但你可以检查你的函数，如果参数是 `nil`，则分配给它默认值。

### 函数结果

你可以使用 `return` 关键字来返回函数结果。可以返回多个值。让我们定义两个函数，分别返回一个和两个值：

```cpp
function square(a)
    return a * a
end
function sincos(a)
    return math.sin(a), math.cos(a)
end
```

第一个函数返回给定参数的 `square`。第二个函数返回给定参数的 `sin` 和 `cos`。让我们尝试一下我们的两个函数：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> dofile("Chapter02/1-functions.lua")
> square(2)
4
> sincos(math.pi / 3)
0.86602540378444        0.5
```

你可以从输出中看到，函数分别返回一个和两个值。在这个例子中，`math.pi`、`math.sin` 和 `math.cos` 来自 Lua 的 `math` 库，该库默认在交互式解释器中加载。你有没有想过如何为我们的 `sincos` 函数创建一个基本库？

### 将函数放入表中

从整体的角度来看，Lua 的 `math` 库——以及任何其他库——只是包含函数和常量值的表。你可以定义自己的：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> dofile("Chapter02/1-functions.lua")
> mathx = {sincos = sincos}
> mathx.sincos(math.pi / 3)
0.86602540378444        0.5
> mathx["sincos"]
function: 0x13ca052d0
> mathx"sincos"
0.86602540378444        0.5
```

我们在这里创建了一个名为 `mathx` 的表，并将我们的 `sincos` 函数分配给 `"``sincos"` 键。

现在你已经知道如何创建自己的 Lua 库。为了完成我们对 Lua 类型的介绍，让我们看看为什么我们应该使用局部变量。

## 局部变量和作用域

到目前为止，我们一直在使用全局变量，因为我们只需要引用一个，对吧？是的，这很方便。但缺点是它们永远不会超出作用域，并且可以被所有函数访问，无论它们是否相关。如果你来自 C++ 背景，你不会同意这一点。

我们可以在 `for` 循环内使用 `if` 分支，或在函数内使用。

局部变量对于防止全局环境污染很有用。尝试定义两个函数来测试这一点：

```cpp
function test_variable_leakage()
    abc_leaked = 3
end
function test_local_variable()
    local abc_local = 4
end
```

在第一个函数中，没有使用局部变量，因此将创建一个名为 `abc_leaked` 的全局变量。在第二个函数中，使用了一个局部变量——`abc_local`——它将在其函数作用域之外不可知。让我们看看效果：

```cpp
Lua 5.4.6 Copyright (C) 1994-2022 Lua.org, PUC-Rio
> dofile("Chapter02/1-functions.lua")
> abc_leaked
nil
> test_variable_leakage()
> abc_leaked
3
> test_local_variable()
> abc_local
nil
```

从输出中，我们可以验证以下：

1.  首先，我们尝试第一个没有使用局部变量的函数。在调用函数之前，我们验证没有名为 `abc_leaked` 的全局变量。调用函数后，创建了一个全局变量——`abc_leaked`。

1.  然后我们尝试使用局部变量的第二个函数。在这种情况下，没有创建全局变量。

当你可以时，你应该始终使用局部变量。接下来，让我们熟悉 Lua 的控制结构。

# 控制结构

Lua 控制结构与 C++ 控制结构非常相似。在学习它们的时候，尝试将它们与它们的 C++ 对应物进行比较。

对于本节中显示的代码，你可以将它们放入另一个名为 `2-controls.lua` 的 Lua 脚本文件中，并在 Lua 解释器中使用 `dofile` 导入。你可以将每个示例放入一个单独的函数中，这样你就可以使用不同的参数测试代码。到现在为止，你应该已经熟悉了 Lua 解释器，所以我们不会在本章的其余部分展示如何使用它。

我们将首先探索如何在 Lua 中进行条件分支，然后我们将尝试循环。

## if then else

Lua 的 `if` 控制结构类似于 C++ 的。然而，你不需要在测试条件周围使用括号，也不使用花括号。相反，你需要使用 `then` 关键字和 `end` 关键字来界定代码分支，例如：

```cpp
if a < 0 then a = 0 end
if a > 0 then return a else return 0 end
if a < 0 then
    a = 0
    print("changed")
else
    print("unchanged")
end
```

如果没有操作，`else` 分支是可选的。如果你每个分支只有一个语句，你也可以选择将所有内容写在一行中。

Lua 语言设计强调简洁性，因此 `if` 控制结构是唯一的条件分支控制。如果你想要实现类似于 C++ 的 **switch** 控制结构，怎么办呢？

### 模拟 switch

Lua 中没有 switch 控制结构。为了模拟它，你可以使用 `elseif`。以下代码就是这样做的：

```cpp
if day == 1 then
    return "Monday"
elseif day == 2 then
    return "Tuesday"
elseif day == 3 then
    return "Wednesday"
elseif day == 4 then
    return "Thursday"
elseif day == 5 then
    return "Friday"
elseif day == 6 then
    return "Saturday"
elseif day == 7 then
    return "Sunday"
else
    return nil
end
```

这与 C++ 的 `if..else if` 控制结构行为相同。`if` 和 `elseif` 条件将逐个检查，直到满足一个条件并返回一周中某天的名称。

## while

Lua 的 `do` 关键字和 `end` 关键字。以下示例打印出一周中的日子：

```cpp
local days = {
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
}
local i = 1
while days[i] do
    print(days[i])
    i = i + 1
end
```

我们声明一个名为 `days` 的表，并使用它作为数组。当 `i` 索引达到 `8` 时，循环将结束，因为 `days[8]` 是 `nil` 并测试为 `false`。来自 C++ 的你可能想知道为什么我们可以访问一个七元素数组的第八个元素。在 Lua 中，以这种方式访问表时，没有索引越界的问题。

你可以使用 `break` 立即结束循环。这对 `repeat` 循环和 `for` 循环都适用，我们将在下面解释。

## repeat

Lua 的 `do..while` 控制结构也这样做，但结束条件被处理得不同。Lua 使用 `until` 条件来结束循环，而不是 C++ 的 `while`。

让我们实现之前为 `while` 控制结构展示的相同代码，但这次使用 `repeat`：

```cpp
local days = {
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
}
local i = 0
repeat
    i = i + 1
    print(days[i])
until i == #days
```

`#days` 返回 `day` 数组的长度。`repeat..until` 中的代码块将循环，直到 `i` 达到这个长度。

注意

请记住，对于 Lua 数组，索引从 1 开始。

要在 C++ 中使用 `do..while` 实现相同的代码，请执行以下操作：

```cpp
const std::vector<std::string> days {
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
};
size_t i = 0;
do {
    std::cout << days[i] << std::endl;
    i++;
} while (i < days.size());
```

C++ 的实现看起来与 Lua 版本非常相似，除了前面提到的结束条件：`i < days.size()`。我们检查的是小于，而不是等于。

## for, 数值

数值 **for** 循环遍历一个数字列表。它具有以下形式：

```cpp
for var = exp1, exp2, exp3 do
    do_something
end
```

+   `var` 被视为作用域限于 `for` 块的局部变量。

+   `exp1` 是起始值。

+   `exp2` 是结束值。

+   `exp3` 是步长，是可选的。如果没有提供，则默认步长为 1。

为了更好地理解这一点，让我们看一个例子：

```cpp
local days = {
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
}
for i = 1, #days, 4 do
    print(i, days[i])
end
```

`i` 是局部变量，初始值为 `1`。当 `i` 变得大于 `#days` 时，循环将结束。还提供了一个步长 `4`。因此，每次迭代后，效果是 `i = i + 4`。一旦运行此代码，你就会发现只有星期一和星期五被打印出来。

也许会让你惊讶，浮点类型也可以工作：

```cpp
Lua 5.4.6  Copyright (C) 1994-2022 Lua.org, PUC-Rio
> for a = 1.0, 4.0, 1.5 do print(a) end
1.0
2.5
4.0
```

如输出所示，`for` 循环从 `1.0` 开始打印，每次增加 `1.5`，只要值不大于 `4.0`。

## for, 通用

通用 `for` 循环遍历由 `迭代函数` 返回的所有值。这种形式的 `for` 循环在遍历表时非常方便。

当我们讨论数值 `for` 循环时，我们看到了它们如何遍历基于索引的表。然而，Lua 中的表可以不仅仅是数组。表上最常见的迭代器是 `pairs` 和 `ipairs`。它们返回表中的键值对。`pairs` 返回的键值对顺序未定义，就像大多数哈希表实现一样。`ipairs` 返回排序后的键值对。

即使对于基于索引的表，如果你想遍历所有内容，通用的 `for` 循环也可以更方便：

```cpp
local days = {
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
}
for index, day in pairs(days) do
    print(index, day)
end
```

这个循环遍历整个数组，而不需要引用数组长度。`pairs` 迭代器逐个返回键值对，直到枚举表中的所有元素。之后，循环结束。

# 摘要

在本章中，我们学习了 Lua 的八种数据类型中的六种和四种控制结构。我们还学习了局部变量以及为什么你应该使用它们。这些知识将为你阅读本书的其余部分做好准备。

到现在为止，你应该能够阅读和理解大多数 Lua 代码。一些细节和主题故意没有包含在本章中。当你遇到它们时，你可以了解更多关于它们的信息。

在下一章中，我们将学习如何从 C++ 调用 Lua 代码。

# 练习

1.  在 Lua 参考手册中定位标准字符串操作库。了解 `string.gmatch`、`string.gsub` 和模式匹配。什么模式代表所有非空格字符？

1.  使用 `string.gmatch` 和一个通用的 `for` 循环，反转句子 “C++ loves Lua.” 输出应该是 “Lua loves C++。”

1.  你能使用 `string.gsub` 并用一行代码实现相同的功能吗？

# 参考文献

官方 Lua 参考手册：[`www.lua.org/manual/5.4/`](https://www.lua.org/manual/5.4/)

# 第二部分 – 从 C++ 调用 Lua

现在你已经熟悉了使用 Lua 设置 C++ 项目，你将开始学习如何从 C++ 调用 Lua 代码。

你将开始实现一个通用的 C++ 工具类来加载和执行 Lua 代码。首先，你将学习如何加载 Lua 脚本和调用 Lua 函数。然后，你将探索如何向 Lua 函数传递参数和处理返回值。最后，你将深入了解如何与 Lua 表一起工作。

本部分包括以下章节：

+   *第三章*，*如何从 C++ 调用 Lua*

+   *第四章*，*将 Lua 类型映射到 C++*

+   *第五章*，*与 Lua 表一起工作*
