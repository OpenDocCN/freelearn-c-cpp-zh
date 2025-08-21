# 第十三章：异常处理

在这最后一章中，我们将学习如何在系统编程时执行错误处理。具体来说，将介绍三种不同的方法。第一种方法将演示如何使用 POSIX 风格的错误处理，而第二种方法将演示如何使用标准的 C 风格的 set jump 异常。第三种方法将演示如何使用 C++ 异常，并讨论每种方法的优缺点。最后，本章将以一个示例结束，演示了 C++ 异常如何优于 POSIX 风格的错误处理。

在本章中，我们将涵盖以下主题：

+   POSIX 风格的错误处理

+   C++ 中的异常支持

+   带异常基准的示例

# 技术要求

为了编译和执行本章中的示例，读者必须具备以下条件：

+   一个能够编译和执行 C++17 的基于 Linux 的系统（例如，Ubuntu 17.10+）

+   GCC 7+

+   CMake 3.6+

+   互联网连接

要下载本章中的所有代码，包括示例和代码片段，请参见以下链接：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter13`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/tree/master/Chapter13)。

# 错误处理 POSIX 风格

POSIX 风格的错误处理提供了可能的最基本的错误处理形式，几乎可以在任何系统的几乎任何程序中使用。以标准 C 为基础编写，POSIX 风格的错误处理采用以下形式：

```cpp
if (foo() != 0) {
    std::cout << errno << '\n';
}
```

通常，每个调用的函数要么在 `success` 时返回 `0`，要么在失败时返回 `-1`，并将错误代码存储在一个全局（非线程安全）的实现定义的宏中，称为 `errno`。使用 `0` 作为 `success` 的原因是，在大多数 CPU 上，将变量与 `0` 进行比较比将变量与任何其他值进行比较更快，而 `success` 情况是预期的情况。以下示例演示了如何使用这种模式：

```cpp
#include <cstring>
#include <iostream>

int myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;
        return -1;
    }

    return 0;
}

int main()
{
    if (myfunc(1) == 0) {
        std::cout << "success\n";
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }

    if (myfunc(42) == 0) {
        std::cout << "success\n";
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: Invalid argument
```

在这个例子中，我们创建了一个名为 `myfunc()` 的函数，它接受一个整数并返回一个整数。该函数接受任何值作为其参数，除了 `42`。如果将 `42` 作为输入函数，函数将返回 `-1` 并将 `errno` 设置为 `EINVAL`，表示函数提供了一个无效的参数。

在 `main` 函数中，我们调用 `myfunc()`，分别使用有效输入和无效输入进行测试，以查看是否发生了错误，有效输入返回 `success`，无效输入返回 `failure: Invalid argument`。值得注意的是，我们利用了 `strerror()` 函数，将 POSIX 定义的错误代码转换为它们的字符串等价物。还应该注意的是，这个简单的例子将在本章中被利用，并在此基础上进行改进。

从这个简单的例子中出现的第一个问题是函数的输出被用于错误处理，但如果函数需要输出除错误代码以外的值怎么办？有两种处理方法。处理这个问题的第一种方法是限制函数的有效输出（即，并非所有输出都被认为是有效的）。这通常是 POSIX 处理这个问题的方式。以下示例演示了这一点：

```cpp
#include <cstring>
#include <iostream>

int myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;
        return 0;
    }

    return 42;
}

int main()
{
    if (auto handle = myfunc(1); handle != 0) {
        std::cout << "success: " << handle << '\n';
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }

    if (auto handle = myfunc(42); handle != 0) {
        std::cout << "success: " << handle << '\n';
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success: 42
// failure: Invalid argument
```

在上面的示例中，我们创建了一个 `myfunc()` 函数，给定有效输入返回一个 `handle`，给定无效输入返回 `0`。这类似于很多返回文件句柄的 POSIX 函数。在这种情况下，`success` 的概念被颠倒了，此外，句柄可能永远不会取值为 `0`，因为这用于表示错误。另一种同时提供错误处理和函数输出的可能方法是返回多个值，如下所示：

```cpp
#include <utility>
#include <cstring>
#include <iostream>

std::pair<int, bool>
myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;
        return {0, false};
    }

    return {42, true};
}

int main()
{
    if (auto [handle, success] = myfunc(1); success) {
        std::cout << "success: " << handle << '\n';
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }

    if (auto [handle, success] = myfunc(42); success) {
        std::cout << "success: " << handle << '\n';
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success: 42
// failure: Invalid argument
```

在前面的例子中，我们返回了`std::pair{}`（实际上只是一个具有两个值的结构体）。对中的第一个值是我们的句柄，而对中的第二个值确定了句柄是否有效。使用这种机制，`0`可能是一个有效的句柄，因为我们有一种方法告诉这个函数的用户它是否有效。另一种方法是为函数提供一个作为*输出*而不是*输入*的参数，这种做法是 C++核心指南不推荐的。这通过以下代码表示：

```cpp
#include <cstring>
#include <iostream>

int myfunc(int val, int &error)
{
    if (val == 42) {
        error = EINVAL;
        return 0;
    }

    return 42;
}

int main()
{
    int error = 0;

    if (auto handle = myfunc(1, error); error == 0) {
        std::cout << "success: " << handle << '\n';
    }
    else {
        std::cout << "failure: " << strerror(error) << '\n';
    }

    if (auto handle = myfunc(42, error); error == 0) {
        std::cout << "success: " << handle << '\n';
    }
    else {
        std::cout << "failure: " << strerror(error) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success: 42
// failure: Invalid argument
```

在这个例子中，`myfunc()`接受两个参数，第二个参数接受一个整数，用于存储错误。如果错误整数保持为`0`，则表示没有发生错误。然而，如果错误整数被设置，就表示发生了错误，我们会检测并输出失败的结果。尽管这种方法不被 C++核心指南推荐（主要是因为在 C++中有更好的方法来处理错误），但这种方法的额外好处是错误整数是线程安全的，而不像`errno`的使用那样不是线程安全的。

除了 POSIX 风格错误处理的冗长和错误值被忽略的倾向之外，最大的问题是必须持续执行大量分支语句，以防错误可能发生的情况。下面的例子演示了这一点：

```cpp
#include <cstring>
#include <iostream>

int myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;
        return -1;
    }

    return 0;
}

int nested1(int val)
{
    if (auto ret = myfunc(val); ret != 0) {
        std::cout << "nested1 failure: " << strerror(errno) << '\n';
        return ret;
    }
    else {
        std::cout << "nested1 success\n";
    }

    return 0;
}

int nested2(int val)
{
    if (auto ret = nested1(val); ret != 0) {
        std::cout << "nested2 failure: " << strerror(errno) << '\n';
        return ret;
    }
    else {
        std::cout << "nested2 success\n";
    }

    return 0;
}

int main()
{
    if (nested2(1) == 0) {
        std::cout << "nested2(1) complete\n";
    }
    else {
        std::cout << "nested2(1) failure: " << strerror(errno) << '\n';
    }

    if (nested2(42) == 0) {
        std::cout << "nested2(42) complete\n";
    }
    else {
        std::cout << "nested2(42) complete: " << strerror(errno) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// nested1 success
// nested2 success
// nested2(1) complete
// nested1 failure: Invalid argument
// nested2 failure: Invalid argument
// nested2(42) failure: Invalid argument
```

在这个例子中，我们创建了相同的`myfunc()`函数，如果输入为`42`，则返回一个错误。然后我们从另一个函数中调用这个函数（也就是说，我们在`myfunc()`中进行了嵌套调用，这在系统编程中很可能会发生）。由于`myfunc()`可能返回一个错误，而我们的嵌套函数无法处理错误，它们也必须返回一个错误代码，然后必须对其进行检查。在这个例子中，大部分代码只提供了错误处理逻辑，旨在将错误的结果转发给下一个函数，希望下一个函数能够处理错误。

这种嵌套的错误转发可能被称为“堆栈展开”。每次调用可能返回错误的函数时，我们都会检查是否发生了错误，并将结果返回给堆栈中的下一个函数。这个展开调用堆栈的过程会重复，直到我们到达堆栈中能够处理错误的函数为止。在我们的情况下，这是`main()`函数。

POSIX 风格的错误处理存在的问题是必须手动执行堆栈展开，因此，在“成功”情况下，这段代码会持续执行，导致性能不佳、代码冗长，正如前面的示例所示，该示例仅在三个嵌套调用中检查了一个简单的整数值。

最后，应该指出，POSIX 风格的错误处理确实支持**资源获取即初始化**（**RAII**），这意味着在函数范围内定义的对象在函数退出时会被正确销毁，无论是在“成功”情况下还是错误情况下，如下例所示：

```cpp
#include <cstring>
#include <iostream>

class myclass
{
public:
    ~myclass()
    {
        std::cout << "destructor called\n";
    }
};

int myfunc(int val)
{
    myclass c{};

    if (val == 42) {
        errno = EINVAL;
        return -1;
    }

    return 0;
}

int main()
{
    if (myfunc(1) == 0) {
        std::cout << "success\n";
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }

    if (myfunc(42) == 0) {
        std::cout << "success\n";
    }
    else {
        std::cout << "failure: " << strerror(errno) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// destructor called
// success
// destructor called
// failure: Invalid argument
```

在前面的例子中，我们创建了一个简单的类，在销毁时向`stdout`输出一个字符串，并在我们的`myfunc()`函数中创建了这个类的一个实例。当调用`myfunc()`时，无论是在“成功”还是失败时，类的析构函数都会在退出时被正确调用。在我们下一个错误处理机制中，称为设置跳转，我们将演示如何解决 POSIX 风格错误处理的许多问题，同时也演示了设置跳转的关键限制是缺乏 RAII 支持，可能导致未定义的行为。

# 学习关于设置跳转异常

Set jump 异常可以看作是 C 风格的异常。与 C++风格的异常一样，set jump 异常提供了在出现错误时设置返回代码的位置以及执行跳转的异常生成方法。以下代码示例演示了这一点：

```cpp
#include <cstring>
#include <csetjmp>

#include <iostream>

std::jmp_buf jb;

void myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;   // Invalid argument
        std::longjmp(jb, -42);
    }
}

int main()
{
    if (setjmp(jb) == -42) {
        std::cout << "failure: " << strerror(errno) << '\n';
        std::exit(EXIT_FAILURE);
    }

    myfunc(1);
    std::cout << "success\n";

    myfunc(42);
    std::cout << "success\n";
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: Invalid argument
```

在这个例子中，我们创建了`myfunc()`函数，但是不返回错误代码，而是执行了 long jump，它像*goto*一样，跳转到调用`setjmp()`的调用栈中最后一次调用的位置。在我们的`main`函数中，我们首先调用`setjmp()`来设置返回点，然后使用有效输入和无效输入调用我们的`myfunc()`函数。

我们已经解决了 POSIX 风格错误处理的几个问题。如前面的例子所示，代码变得简单得多，不再需要检查错误条件。此外，`myfunc()`返回一个 void，不再需要返回错误代码，这意味着不再需要限制函数的输出以支持错误情况，如下例所示：

```cpp
#include <cstring>
#include <csetjmp>

#include <iostream>

std::jmp_buf jb;

int myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;
        std::longjmp(jb, -1);
    }

    return 42;
}

int main()
{
    if (setjmp(jb) == -1) {
        std::cout << "failure: " << strerror(errno) << '\n';
        std::exit(EXIT_FAILURE);
    }

    auto handle1 = myfunc(1);
    std::cout << "success: " << handle1 << '\n';

    auto handle2 = myfunc(42);
    std::cout << "success: " << handle2 << '\n';
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success: 42
// failure: Invalid argument
```

在这个例子中，`myfunc()`返回一个*handle*，并且使用 set jump 异常处理错误情况。因此，`myfunc()`可能返回任何值，函数的使用者根据是否调用了 long jump 来判断 handle 是否有效。

由于不再需要`myfunc()`的返回值，我们也不再需要检查`myfunc()`的返回值，这意味着我们的嵌套示例大大简化，如下所示：

```cpp
#include <cstring>
#include <csetjmp>

#include <iostream>

std::jmp_buf jb;

void myfunc(int val)
{
    if (val == 42) {
        errno = EINVAL;
        std::longjmp(jb, -1);
    }
}

void nested1(int val)
{
    myfunc(val);
    std::cout << "nested1 success\n";
}

void nested2(int val)
{
    nested1(val);
    std::cout << "nested2 success\n";
}

int main()
{
    if (setjmp(jb) == -1) {
        std::cout << "failure: " << strerror(errno) << '\n';
        exit(EXIT_FAILURE);
    }

    nested2(1);
    std::cout << "nested2(1) complete\n";

    nested2(42);
    std::cout << "nested2(42) complete\n";
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// nested1 success
// nested2 success
// nested2(1) complete
// failure: Invalid argument
```

正如所见，这个例子中唯一的错误逻辑存在于`myfunc()`中，用于确保输入有效。其余的错误逻辑已经被移除。这不仅使得代码更易于阅读和维护，而且由于不再执行分支语句，而是手动展开调用栈，因此结果代码的性能也更好。

使用 set jump 异常的另一个好处是可以创建线程安全的错误处理。在我们之前的例子中，我们在出现错误时设置了`errno`，然后在到达能够处理错误的代码时读取它。使用 set jump，不再需要`errno`，因为我们可以在 long jump 本身中返回错误代码，采用以下方法：

```cpp
#include <cstring>
#include <csetjmp>

#include <iostream>

void myfunc(int val, jmp_buf &jb)
{
    if (val == 42) {
        std::longjmp(jb, EINVAL);
    }
}

int main()
{
    std::jmp_buf jb;

    if (auto ret = setjmp(jb); ret > 0) {
        std::cout << "failure: " << strerror(ret) << '\n';
        std::exit(EXIT_FAILURE);
    }

    myfunc(1, jb);
    std::cout << "success\n";

    myfunc(42, jb);
    std::cout << "success\n";
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: Invalid argument
```

在前面的例子中，我们不再在 long jump 中设置`errno`并返回`-1`，而是在 long jump 中返回错误代码，并且使用 C++17 语法，在调用 set jump 时存储 long jump 的值，并确保这个值大于`0`。第一次调用 set jump 时，由于尚未发生错误，它返回`0`，意味着不会执行分支。然而，如果第二次调用 set jump（当我们的 long jump 被调用时），则返回 long jump 中放置的值，导致执行分支并以线程安全的方式报告错误。

请注意，我们需要对我们的例子进行的唯一修改是必须传递每个函数的跳转缓冲区，这非常不方便，特别是在嵌套函数调用的情况下。在我们之前的例子中，跳转缓冲区是全局存储的，这不是线程安全的，但更方便，代码更清晰。

除了提供线程安全的笨拙机制之外，使用 set jump 进行错误处理的主要缺点是不支持 RAII，这意味着在函数范围内创建的对象在退出时可能不会调用它们的析构函数（这实际上是特定于实现的问题）。析构函数不会被调用的原因是函数从技术上讲从未退出。set jump/long jump 在调用 set jump 时将指令指针和非易失性寄存器存储在跳转缓冲区中。

当执行长跳转时，应用程序会用跳转缓冲区中存储的值覆盖指令指针和 CPU 寄存器的值，然后继续执行，就好像调用`setjump()`后的代码从未执行过一样。因此，对象的析构函数永远不会被执行，就像下面的例子中所示的那样：

```cpp
#include <cstring>
#include <csetjmp>

#include <iostream>

jmp_buf jb;

class myclass
{
public:
    ~myclass()
    {
        std::cout << "destructor called\n";
    }
};

void myfunc(int val)
{
    myclass c{};

    if (val == 42) {
        errno = EINVAL;
        std::longjmp(jb, -1);
    }
}

int main()
{
    if (setjmp(jb) == -1) {
        std::cout << "failure: " << strerror(errno) << '\n';
        exit(EXIT_FAILURE);
    }

    myfunc(1);
    std::cout << "success\n";

    myfunc(42);
    std::cout << "success\n";
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// destructor called
// success
// failure: Invalid argument
```

在这个例子中，我们创建了一个简单的类，在类被销毁时向`stdout`输出一个字符串。然后我们在`myfunc()`中创建了这个类的一个实例。在`success`情况下，当`myfunc()`退出时，析构函数被调用，导致析构函数被调用。然而，在失败的情况下，`myfunc()`永远不会退出，导致析构函数不会被调用。

在下一节中，我们将讨论 C++异常，它建立在 set jump 异常的基础上，不仅提供了对 RAII 的支持，还提供了在发生错误时返回复杂数据类型的能力。

# 理解 C++中的异常支持

C++异常提供了一种在线程安全的方式报告错误的机制，无需手动展开调用堆栈，同时还支持 RAII 和复杂数据类型。要更好地理解这一点，请参考以下例子：

```cpp
#include <cstring>
#include <iostream>

void myfunc(int val)
{
    if (val == 42) {
        throw EINVAL;
    }
}

int main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(int ret) {
        std::cout << "failure: " << strerror(ret) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: Invalid argument
```

在上面的例子中，我们的`myfunc()`函数相对于其 POSIX 风格的等效函数大大简化了。就像我们之前的例子一样，如果提供给函数的输入是`42`，则返回错误（在这种情况下实际上是抛出）。如果提供的输入不是`42`，则函数成功返回。

与 set jump 一样，调用`myfunc()`不再需要检查函数的返回值，因为没有提供返回值。为了处理错误情况，我们将对`myfunc()`的调用包装在`try...catch`块中。如果`try{}`块中的任何代码导致抛出异常，将执行`catch{}`块。与大多数 C++一样，`catch`块是类型安全的，这意味着你必须声明在抛出异常时要接收的返回数据的类型。在这种情况下，我们抛出`EINVAL`，它是一个整数，所以我们捕获一个整数并将结果输出到`stdout`。

与 set jump 类似，`myfunc()`不再需要返回错误代码，这意味着它可以输出任何它想要的值（意味着输出不受限制），就像下一个例子中所示的那样：

```cpp
#include <cstring>
#include <iostream>

int myfunc(int val)
{
    if (val == 42) {
        throw EINVAL;
    }

    return 42;
}

int main()
{
    try {
       auto handle1 = myfunc(1);
        std::cout << "success: " << handle1 << '\n';

        auto handle2 = myfunc(42);
        std::cout << "success: " << handle2 << '\n';
    }
    catch(int ret) {
        std::cout << "failure: " << strerror(ret) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success: 42
// failure: Invalid argument
```

在上面的例子中，`myfunc()`返回一个句柄，它可以取任何值，因为如果抛出异常，这个函数的用户将知道句柄是否有效。

与 set jump 不同，我们的嵌套情况大大简化，因为我们不再需要手动展开调用堆栈：

```cpp
#include <cstring>
#include <iostream>
void myfunc(int val)
{
    if (val == 42) {
        throw EINVAL;
    }
}

void nested1(int val)
{
    myfunc(val);
    std::cout << "nested1 success\n";
}

void nested2(int val)
{
    nested1(val);
    std::cout << "nested2 success\n";
}

main()
{
    try {
        nested2(1);
        std::cout << "nested2(1) complete\n";

        nested2(42);
        std::cout << "nested2(42) complete\n";
    }
    catch(int ret) {
        std::cout << "failure: " << strerror(ret) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// nested1 success
// nested2 success
// nested2(1) complete
// failure: Invalid argument
```

上面的例子类似于我们的 set jump 例子，主要区别在于我们抛出异常而不是执行长跳转，并且我们使用`try...catch`块捕获异常。

与 set jump 不同，C++异常支持 RAII，这意味着在函数范围内定义的对象在函数退出时会被正确销毁：

```cpp
#include <cstring>
#include <iostream>

class myclass
{
public:
    ~myclass()
    {
        std::cout << "destructor called\n";
    }
};

void myfunc(int val)
{
    myclass c{};

    if (val == 42) {
        throw EINVAL;
    }
}

main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(int ret) {
        std::cout << "failure: " << strerror(ret) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// destructor called
// success
// destructor called
// failure: Invalid argument
```

正如在上面的例子中所看到的，析构函数在`success`情况和失败情况下都被调用。为了实现这一点，C++包括一个堆栈展开器，它能够自动展开堆栈，类似于我们使用 POSIX 风格的错误处理手动展开调用堆栈，但是自动进行，而不需要通过代码执行分支语句，从而实现最佳性能（就好像没有进行错误检查一样）。这被称为**零开销异常处理**。

自动展开器如何在不产生任何性能开销的情况下自动展开调用堆栈的细节，同时仍以线程安全的方式支持 RAII，这超出了本书的范围，因为这个过程非常复杂。然而，下面是一个简要的解释。

当启用 C++异常并编译代码时，每个函数还会为堆栈解开指令编译一组指令，并将其放置在可执行文件中，以便 C++异常解开器可以找到它们。然后编译器会编译代码，就好像没有进行错误处理一样，代码会按照这样执行。如果抛出异常，将创建一个线程安全的对象来包装被抛出的数据，并将其存储。然后，使用之前保存在可执行文件中的调用堆栈解开指令来逆转函数的执行，最终导致抛出异常的函数退出到其调用者。在函数退出之前，将执行所有析构函数，并且对调用堆栈中调用的每个函数都会继续执行这个过程，直到遇到一个能够处理被抛出的数据的`catch{}`块。

以下是一些需要记住的关键点：

+   解开指令存储在可执行文件的表中。每当需要从寄存器的角度逆转函数的执行时，解开器必须在表中查找下一个函数的这些指令。这个操作很慢（尽管已经添加了一些优化，包括使用哈希表）。因此，异常不应该用于控制流，因为它们在错误情况下很慢且低效，而在`成功`情况下非常高效。C++异常应该只用于错误处理。

+   程序中的函数越多，或者函数越大（即函数接触 CPU 寄存器越多），就需要在解开指令表中存储更多的信息，从而导致程序更大。如果程序中从未使用 C++异常，这些信息仍然会被编译并存储在应用程序中。因此，如果不使用异常，应该禁用异常。

除了线程安全、高性能和支持 RAII 之外，C++异常还支持复杂的数据类型。C++使用的典型数据类型包括字符串，如下所示：

```cpp
#include <cstring>
#include <iostream>

void myfunc(int val)
{
    if (val == 42) {
        throw std::runtime_error("invalid val");
    }
}

int main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(const std::runtime_error &e) {
        std::cout << "failure: " << e.what() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: invalid val
```

在前面的例子中，我们抛出了一个`std::runtime_error{}`异常。这个异常是 C++提供的许多异常之一，它继承了`std::exception`，支持除异常类型本身之外的字符串存储能力。在前面的例子中，我们存储了`invalid val`。前面的代码不仅能够检测到提供的字符串，还能检测到抛出了`std::runtime_exception{}`。

在某些情况下，您可能不知道抛出的异常类型是什么。当抛出一个不继承`std::exception`的异常时，比如原始字符串和整数，通常就会出现这种情况。要捕获任何异常，请使用以下方法：

```cpp
#include <cstring>
#include <iostream>

void myfunc(int val)
{
    if (val == 42) {
        throw -1;
    }
}

main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(...) {
        std::cout << "failure\n";
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure
```

在前面的例子中，我们抛出一个整数，并使用`...`语法来捕获它，表示我们希望捕获所有异常。在代码中至少有这种类型的`catch{}`语句是一个很好的做法，以确保捕获所有异常。在本书的所有示例中，我们都包含了这种`catch`语句，就是为了这个原因。这种类型的`catch{}`块的主要缺点是我们必须使用`std::current_exception()`来获取异常，例如：

```cpp
#include <cstring>
#include <iostream>
#include <stdexcept>

void myfunc1(int val)
{
    if (val == 42) {
        throw std::runtime_error("runtime_error");
    }
}

void myfunc2(int val)
{
    try {
        myfunc1(val);
    }
    catch(...) {
        auto e = std::current_exception();
        std::rethrow_exception(e);
    }
}

int main()
{
    try {
        myfunc2(42);
    }
    catch(const std::exception& e) {
        std::cout << "caught: " << e.what() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// caught: runtime_error
```

在前面的例子中，我们从`myfunc1()`抛出`std::runtime_error()`。在`myfunc2()`中，我们使用`...`语法捕获异常，表示我们希望捕获所有异常。要获取异常，我们必须使用`std::current_exception()`，它返回`std::exception_ptr{}`。`std::exception_ptr{}`是一个特定于实现的指针类型，可以使用`std::rethrow_exception()`重新抛出。使用这个函数，我们可以使用前面的标准方法捕获异常并输出消息。值得注意的是，如果您希望捕获异常，`std::current_exception()`不是推荐的方法，因为您需要重新抛出异常才能从中获取`what()`，因为`std::exception_ptr`不提供获取`what()`的接口。还应该注意，如果抛出的异常不是`std::exception{}`的子类，`std::current_exception()`也无济于事。

最后，可以用自定义数据替换`subclass std::exception`。要做到这一点，请参考以下示例：

```cpp
#include <cstring>
#include <iostream>
#include <stdexcept>

class myexception : public std::exception
{
    int m_error{0};

public:

    myexception(int error) noexcept :
        m_error{error}
    { }

    const char *
    what() const noexcept
    {
      return "error";
    }

    int error() const noexcept
    {
        return m_error;
    }
};

void myfunc(int val)
{
    if (val == 42) {
        throw myexception(42);
    }
}

int main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(const myexception &e) {
        std::cout << "failure: " << std::to_string(e.error()) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: 42
```

在前面的例子中，我们对`std::exception`进行子类化，以创建我们自己的异常，该异常能够存储错误编号。与所有`std::exception{}`的子类一样，`what()`函数应该被重载，以提供一个能够唯一标识你自定义异常的消息。在我们的情况下，我们还提供了一个函数来检索在创建和抛出异常时存储的错误代码。

另一个常见的任务是为您的异常创建自定义字符串。然而，这可能会导致一个常见的错误，即在`what()`函数中返回一个构造的字符串：

```cpp
const char *
what() const noexcept
{
    return ("error: " + std::to_string(m_error)).c_str();
}
```

前面的代码产生了未定义的行为和难以发现的错误。在前面的代码中，我们存储错误代码，就像在前面的例子中一样，但是我们不是返回错误代码，而是在`what()`函数中返回一个字符串中的错误代码。为此，我们利用`std::to_string()`函数将我们的错误代码转换为`std::string`。然后我们添加`error:`，并返回生成的标准 C 字符串。

前面例子的问题在于返回了指向标准 C 字符串的指针，然后在`what()`函数退出时销毁了`std::string{}`。试图使用此函数返回的字符串的代码最终会读取已删除的内存。这很难发现的原因是在某些情况下，这段代码会按预期执行，只是因为内存的内容可能没有变化得足够快。然而，经过足够长的时间，这段代码很可能会导致损坏。

相反，要创建输出相同消息的字符串，请将生成的错误代码放在现有异常的构造函数中：

```cpp
#include <cstring>
#include <iostream>

class myexception : public std::runtime_error
{
public:
    myexception(int error) noexcept :
        std::runtime_error("error: " + std::to_string(42))
    { }
};

void myfunc(int val)
{
    if (val == 42) {
        throw myexception(42);
    }
}

int main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(const std::exception &e) {
        std::cout << "failure: " << e.what() << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// success
// failure: error: 42
```

在前面的例子中，我们对`std::runtime_error{}`进行子类化，而不是直接对`std::exception`进行子类化，并在异常构造期间创建我们的`what()`消息。这样，当调用`what()`时，异常信息就可以在没有损坏的情况下使用。

我们将以关于 C++17 唯一真正的异常支持方面的说明结束本章。通常不鼓励在已经抛出异常时抛出异常。要实现这一点，您必须从已标记为`except()`的类的析构函数中抛出异常，并且在堆栈展开期间销毁。在 C++17 之前，析构函数可以通过利用`std::uncaught_exception()`函数来检测是否即将发生这种情况，该函数在正在抛出异常时返回 true。为了支持在已经抛出异常时抛出异常，C++17 将此函数更改为返回一个整数，该整数表示当前正在抛出的异常的总数：

```cpp
#include <cstring>
#include <iostream>

class myclass
{
public:
    ~myclass()
    {
        std::cout << "uncaught_exceptions: "
                  << std::uncaught_exceptions() << '\n';
    }
};

void myfunc(int val)
{
    myclass c{};

    if (val == 42) {
        throw EINVAL;
    }
}

int main()
{
    try {
        myfunc(1);
        std::cout << "success\n";

        myfunc(42);
        std::cout << "success\n";
    }
    catch(int ret) {
        std::cout << "failure: " << strerror(ret) << '\n';
    }
}

// > g++ -std=c++17 scratchpad.cpp; ./a.out
// uncaught_exceptions: 0
// success
// uncaught_exceptions: 1
// failure: Invalid argument
```

在前面的示例中，我们创建了一个类，输出当前正在抛出的异常总数到`stdout`。然后在`myfunc()`中实例化这个类。在成功案例中，当销毁类时，没有异常正在被抛出。在错误案例中，当销毁类时，报告有一个异常被抛出。

# 研究异常基准测试的示例

在最后一个示例中，我们将演示 C++异常优于 POSIX 风格异常（这一说法在很大程度上取决于您执行的硬件，因为编译器优化和激进的分支预测可以提高 POSIX 风格错误处理的性能）。

POSIX 风格的错误处理要求用户每次执行函数时都要检查结果。当函数嵌套发生时（这几乎肯定会发生），这个问题会进一步恶化。在这个示例中，我们将把这种情况推向极端，创建一个递归函数，检查自身的结果数千次，同时执行测试数十万次。每个测试都将进行基准测试，并比较结果。

有很多因素可能会改变这个测试的结果，包括分支预测、优化和操作系统。这个测试的目标是将示例推向极端，以便大部分这些问题都在噪音中消失，任何方法的性能相关问题都很容易识别。

首先，我们需要以下包含：

```cpp
#include <csetjmp>

#include <chrono>
#include <iostream>

```

我们还需要以下全局定义的跳转缓冲区，因为我们将比较 C++异常和 set jump 以及 POSIX 风格的错误处理：

```cpp
jmp_buf jb;
```

我们还将使用我们在之前章节中使用过的相同基准测试代码：

```cpp
template<typename FUNC>
auto benchmark(FUNC func) {
    auto stime = std::chrono::high_resolution_clock::now();
    func();
    auto etime = std::chrono::high_resolution_clock::now();

    return (etime - stime).count();
}
```

我们的第一个递归函数将使用 POSIX 风格的错误处理返回错误：

```cpp
int myfunc1(int val)
{
    if (val >= 0x10000000) {
        return -1;
    }

    if (val < 0x1000) {
        if (auto ret = myfunc1(val + 1); ret == -1) {
            return ret;
        }
    }

    return 0;
}
```

如图所示，函数的返回值与预期相比。第二个函数将使用 set jump 返回错误：

```cpp
void myfunc2(int val)
{
    if (val >= 0x10000000) {
        std::longjmp(jb, -1);
    }

    if (val < 0x1000) {
        myfunc2(val + 1);
    }
}
```

正如预期的那样，这个函数不那么复杂，因为不需要返回或比较返回值。最后，第三个函数将使用 C++异常返回错误：

```cpp
void myfunc3(int val)
{
    if (val >= 0x10000000) {
        throw -1;
    }

    if (val < 0x1000) {
        myfunc3(val + 1);
    }
}
```

正如预期的那样，这个函数与 set jump 几乎相同，唯一的区别是使用了 C++异常。由于我们不测试 RAII，我们期望 C++异常的执行速度与 set jump 一样快，因为两者都不需要进行比较。

最后，在我们的 protected `main`函数中，我们将以与之前示例相同的方式执行每个函数，以演示每个函数的执行结果如预期。

```cpp
void test_func1()
{
    if (auto ret = myfunc1(0); ret == 0) {
        std::cout << "myfunc1: success\n";
    }
    else {
        std::cout << "myfunc1: failure\n";
    }

    if (auto ret = myfunc1(bad); ret == 0) {
        std::cout << "myfunc1: success\n";
    }
    else {
        std::cout << "myfunc1: failure\n";
    }

    uint64_t total = 0;
    for (auto i = 0; i < num_iterations; i++) {
        total += benchmark([&] {
            myfunc1(0);
        });
    }

    std::cout << "time1: " << total << '\n';
}
```

第一个测试函数测试 C 风格的错误处理逻辑，以确保函数按预期返回成功和失败。然后，我们执行成功案例多次，并计算执行所需的时间，将结果输出到`stdout`：

```cpp
void test_func2()
{
    if (setjmp(jb) == -1) {
        std::cout << "myfunc2: failure\n";

        uint64_t total = 0;
        for (auto i = 0; i < num_iterations; i++) {
            total += benchmark([&] {
                myfunc2(0);
            });
        }

        std::cout << "time2: " << total << '\n';
        return;
    }

    myfunc2(0);
    std::cout << "myfunc2: success\n";

    myfunc2(bad);
    std::cout << "myfunc2: success\n";
}
```

如图所示，我们还确保第二个 C 风格异常示例也按预期返回成功和失败。然后，我们执行成功案例多次，以查看执行所需的时间：

```cpp
void test_func3()
{
    try {
        myfunc3(0);
        std::cout << "myfunc3: success\n";

        myfunc3(bad);
        std::cout << "myfunc3: success\n";
    }
    catch(...) {
        std::cout << "myfunc3: failure\n";
    }

    uint64_t total = 0;
    for (auto i = 0; i < num_iterations; i++) {
        total += benchmark([&] {
            myfunc3(0);
        });
    }

    std::cout << "time3: " << total << '\n';
}
```

我们对 C++异常示例做同样的事情。我们通过执行每个测试来完成我们的`protected_main()`函数，如下所示：

```cpp
int
protected_main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    test_func1();
    test_func2();
    test_func3();

    return EXIT_SUCCESS;
}
```

基准测试的结果将输出到`stdout`：

```cpp
int
main(int argc, char **argv)
{
    try {
        return protected_main(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << "Caught unhandled exception:\n";
        std::cerr << " - what(): " << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "Caught unknown exception\n";
    }

    return EXIT_FAILURE;
}
```

与我们的所有示例一样，`protected_main()`函数由`main()`函数执行，如果发生异常，则捕获异常。

# 编译和测试

要编译这段代码，我们利用了我们之前示例中使用的相同的`CMakeLists.txt`文件：[`github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter13/CMakeLists.txt`](https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP/blob/master/Chapter13/CMakeLists.txt)。

有了这个，我们可以使用以下命令编译这段代码：

```cpp
> git clone https://github.com/PacktPublishing/Hands-On-System-Programming-with-CPP.git
> cd Hands-On-System-Programming-with-CPP/Chapter13/
> mkdir build
> cd build

> cmake ..
> make
```

要执行示例，请运行以下代码：

```cpp
> ./example1
myfunc1: success
myfunc1: failure
time1: 1750637978
myfunc2: success
myfunc2: failure
time2: 1609691756
myfunc3: success
myfunc3: failure
time3: 1593301696
```

如前面的代码片段所示，C++异常优于 POSIX 风格的错误处理，并且 set jump 异常是可比较的。

# 总结

在本章中，我们学习了三种不同的方法来进行系统编程时的错误处理。第一种方法是 POSIX 风格的错误处理，它涉及从每个执行的函数返回一个错误代码，并检查每个函数的结果以检测错误。第二种方法涉及使用标准的 C 风格异常（即 set jump），演示了这种形式的异常处理如何解决了 POSIX 风格错误处理的许多问题，但引入了 RAII 支持和线程安全的问题。第三个例子讨论了使用 C++ 异常进行错误处理，以及这种错误处理形式如何解决了本章讨论的大部分问题，唯一的缺点是导致生成的可执行文件大小增加。最后，本章以一个示例结束，演示了 C++ 异常如何优于 POSIX 风格的错误处理。

# 问题

1.  为什么 C++ 异常优于 POSIX 风格的错误处理？

1.  使用 POSIX 风格的错误处理，函数如何返回输出？

1.  为什么 set jump 不支持 RAII？

1.  如何使用 `catch{}` 块捕获任何异常？

1.  为什么 C++ 异常会增加可执行文件的大小？

1.  为什么不应该将 C++ 异常用于控制流？

# 进一步阅读

+   [`www.packtpub.com/application-development/c17-example`](https://www.packtpub.com/application-development/c17-example)

+   [`www.packtpub.com/application-development/getting-started-c17-programming-video`](https://www.packtpub.com/application-development/getting-started-c17-programming-video)
