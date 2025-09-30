# 第二十一章

# 与其他语言的集成

了解如何编写 C 程序或库可能比你想象的更有价值。由于 C 在开发操作系统中的重要作用，C 并不仅限于其自身的世界。C 库有潜力在其他编程语言中加载和使用。当你从编写高级编程语言的代码中获得好处时，你可以在你的语言环境中作为加载的库拥有 C 的火箭动力。

在本章中，我们将更详细地讨论这个问题，并演示如何将 C 共享库与其他一些知名编程语言集成。

在本章中，我们将涵盖以下关键主题：

+   我们讨论了集成之所以可能的原因。这次讨论很重要，因为它为你提供了集成是如何工作的基本概念。

+   我们设计了一个 C 堆栈库。我们将其构建为一个共享对象文件。这个共享对象文件将被许多其他编程语言使用。

+   我们将探讨 C++、Java、Python 和 Golang，看看堆栈库是如何首先加载然后使用的。

作为本章的一般说明，由于我们将要处理五个不同的子项目，每个项目都有不同的编程语言，我们只展示了 Linux 的构建，以防止任何关于构建和执行的问题。当然，我们提供了足够关于 macOS 系统的信息，但我们的重点是构建和运行 Linux 上的源代码。本书的 GitHub 仓库中还有其他脚本，可以帮助你构建 macOS 的源代码。

第一部分讨论了集成本身。我们探讨了为什么与其他编程语言的集成是可能的，这为我们扩展在其他环境中的讨论而不是 C 环境中的讨论奠定了基础。

# 为什么集成是可能的？

如我们在*第十章*，*Unix – 历史 与 架构*中所述，C 彻底改变了我们开发操作系统的方法。这不仅仅是 C 的魔法；它还赋予了我们构建其他通用编程语言的能力，这些语言我们现在称之为高级编程语言。这些语言的编译器大多是用 C 编写的，如果不是，它们也是由用 C 编写的其他工具和编译器开发的。

一个不能使用或提供系统功能的通用编程语言实际上什么都没做。你可以用它写东西，但无法在任何系统上执行它。虽然从理论角度来看，这样的编程语言可能有用途，但从工业角度来看，这显然是不切实际的。因此，编程语言，特别是通过其编译器，应该能够生成可工作的程序。正如你所知，系统的功能是通过操作系统暴露的。无论操作系统本身如何，编程语言都应该能够提供这些功能，并且在该语言编写的程序，在运行在该系统上时，应该能够使用它们。

这就是 C 语言的作用所在。在类 Unix 操作系统中，C 标准库提供了使用系统可用功能的 API。如果编译器想要创建一个可工作的程序，它应该能够允许编译后的程序以间接的方式使用 C 标准库。无论编程语言是什么，以及它是否提供一些特定的和本机标准库，例如 Java 提供的**Java 标准版**（**Java SE**），任何由编写的程序提出的特定功能请求（如打开文件）都应该传递给 C 标准库，然后从那里，它可以到达内核并执行。

例如，让我们再详细谈谈 Java。Java 程序被编译成一种称为*字节码*的中间语言。为了执行 Java 字节码，需要安装**Java 运行时环境**（**JRE**）。JRE 的核心是一个虚拟机，它加载 Java 字节码并在其中运行。这个虚拟机必须能够模拟 C 标准库暴露的功能和服务，并将它们提供给运行在其内部的程序。由于每个平台在 C 标准库及其对 POSIX 和 SUS 标准的兼容性方面都可能不同，因此我们需要为每个平台构建一些特定的虚拟机。

最后关于在其他语言中可以加载的库的说明，我们只能加载共享对象文件，并且无法加载和使用静态库。静态库只能链接到可执行文件或共享对象文件。在大多数类 Unix 系统中，共享对象文件有`.so`扩展名，但在 macOS 中它们有`.dylib`扩展名。

在本节中，尽管篇幅很短，但我试图给你一个基本的概念，解释为什么我们能够加载 C 库，特别是共享库，以及大多数编程语言是如何已经使用 C 库的，因为它们中的大多数都存在加载共享对象库并使用它的能力。

下一步将是编写一个 C 库，然后将其加载到各种编程语言中以供使用。这正是我们很快就要做的事情，但在那之前，你需要知道如何获取章节材料以及如何运行在 shell 框中看到的命令。

# 获取必要的材料

由于本章充满了来自五种不同编程语言的源代码，而且我的希望是让你们所有人都能构建和运行示例，所以我将这一节专门用于介绍一些基本注意事项，这些注意事项你应该在构建源代码时注意。

首先，你需要获取章节材料。正如你现在应该知道的，这本书有一个仓库，其中这一章有一个名为`ch21-integration-with-other-languages`的特定目录。以下命令显示了如何克隆仓库并切换到章节的根目录：

```cpp
$ git clone https://github.com/PacktPublishing/Extreme-C.git
...
$ cd Extreme-C/ch21-integration-with-other-languages
$
```

Shell 代码 21-1：克隆书籍的 GitHub 仓库并切换到章节的根目录

关于本章中的 shell 框，我们假设在执行 shell 框中的命令之前，我们位于章节的根目录`ch21-integration-with-other-languages`文件夹中。如果我们需要切换到其他目录，我们将提供所需的命令，但所有操作都在章节目录内进行。

此外，为了能够构建源代码，你需要在你的机器上安装**Java 开发工具包**（**JDK**）、Python 和 Golang。根据你使用的是 Linux 还是 macOS，以及你的 Linux 发行版，安装命令可能会有所不同。

作为最后的注意事项，用除 C 以外的其他语言编写的源代码应该能够使用我们在下一节中讨论的 C 栈库。构建这些源代码需要你已经构建了 C 库。因此，请确保你首先阅读以下章节，并在继续下一章节之前构建其共享对象库。现在，既然你知道了如何获取章节材料，我们就可以继续讨论我们的目标 C 库。

# 栈库

在本节中，我们将编写一个小型库，该库将被其他编程语言编写的程序加载和使用。该库是关于一个栈类，它提供了一些基本的操作，如对栈对象进行*push*或*pop*。栈对象由库本身创建和销毁，并且有一个构造函数以及一个析构函数来满足这一目的。

接下来，你可以找到库的公共接口，它作为`cstack.h`头文件的一部分存在：

```cpp
#ifndef _CSTACK_H_
#define _CSTACK_H_
#include <unistd.h>
#ifdef __cplusplus
extern "C" {
#endif
#define TRUE 1
#define FALSE 0
typedef int bool_t;
typedef struct {
  char* data;
  size_t len;
} value_t;
typedef struct cstack_type cstack_t;
typedef void (*deleter_t)(value_t* value);
value_t make_value(char* data, size_t len);
value_t copy_value(char* data, size_t len);
void free_value(value_t* value);
cstack_t* cstack_new();
void cstack_delete(cstack_t*);
// Behavior functions
void cstack_ctor(cstack_t*, size_t);
void cstack_dtor(cstack_t*, deleter_t);
size_t cstack_size(const cstack_t*);
bool_t cstack_push(cstack_t*, value_t value);
bool_t cstack_pop(cstack_t*, value_t* value);
void cstack_clear(cstack_t*, deleter_t);
#ifdef __cplusplus
}
#endif
#endif
```

代码框 21-1 [cstack.h]：栈库的公共接口

正如在 *第六章*，*面向对象编程和封装* 中所解释的，前述声明引入了 Stack 类的公共接口。正如你所见，类的伴随属性结构是 `cstack_t`。我们使用 `cstack_t` 而不是 `stack_t`，因为后者在 C 标准库中使用，并且我更喜欢避免在此代码中产生任何歧义。通过前述声明，属性结构被前置声明且其中没有字段。相反，细节将在实际实现该结构的源文件中给出。该类还有一个构造函数、一个析构函数以及一些其他行为，如 push 和 pop。正如你所见，所有这些函数都将 `cstack_t` 类型的指针作为它们的第一个参数，该指针指示它们应该作用的对象。我们编写 Stack 类的方式在第六章，*面向对象编程和封装* 中的 *隐式封装* 部分进行了说明。

*代码框 21-2* 包含了栈类的实现。它还包含了 `cstack_t` 属性结构的实际定义：

```cpp
#include <stdlib.h>
#include <assert.h>
#include "cstack.h"
struct cstack_type {
  size_t top;
  size_t max_size;
  value_t* values;
};
value_t copy_value(char* data, size_t len) {
  char* buf = (char*)malloc(len * sizeof(char));
  for (size_t i = 0; i < len; i++) {
    buf[i] = data[i];
  }
  return make_value(buf, len);
}
value_t make_value(char* data, size_t len) {
  value_t value;
  value.data = data;
  value.len = len;
  return value;
}
void free_value(value_t* value) {
  if (value) {
    if (value->data) {
      free(value->data);
      value->data = NULL;
    }
  }
}
cstack_t* cstack_new() {
  return (cstack_t*)malloc(sizeof(cstack_t));
}
void cstack_delete(cstack_t* stack) {
  free(stack);
}
void cstack_ctor(cstack_t* cstack, size_t max_size) {
  cstack->top = 0;
  cstack->max_size = max_size;
  cstack->values = (value_t*)malloc(max_size * sizeof(value_t));
}
void cstack_dtor(cstack_t* cstack, deleter_t deleter) {
  cstack_clear(cstack, deleter);
  free(cstack->values);
}
size_t cstack_size(const cstack_t* cstack) {
  return cstack->top;
}
bool_t cstack_push(cstack_t* cstack, value_t value) {
  if (cstack->top < cstack->max_size) {
    cstack->values[cstack->top++] = value;
    return TRUE;
  }
  return FALSE;
}
bool_t cstack_pop(cstack_t* cstack, value_t* value) {
  if (cstack->top > 0) {
    *value = cstack->values[--cstack->top];
    return TRUE;
  }
  return FALSE;
}
void cstack_clear(cstack_t* cstack, deleter_t deleter) {
  value_t value;
  while (cstack_size(cstack) > 0) {
    bool_t popped = cstack_pop(cstack, &value);
    assert(popped);
    if (deleter) {
      deleter(&value);
    }
  }
}
```

代码框 21-2 [cstack.c]: 栈类的定义

正如你所见，定义暗示了每个栈对象都由一个数组支持，而且不仅如此，我们可以在栈中存储任何值。让我们构建库并从中生成一个共享对象库。这将是在接下来的章节中将被其他编程语言加载的库文件。

下面的 Shell 框展示了如何使用现有的源文件创建共享对象库。文本框中的命令在 Linux 上有效，并且为了在 macOS 上运行，它们应该进行轻微的修改。请注意，在运行构建命令之前，你应该处于该章节的根目录，正如之前所解释的：

```cpp
$ gcc -c -g -fPIC cstack.c -o cstack.o
$ gcc -shared cstack.o -o libcstack.so
$
```

Shell 框 21-2：在 Linux 中构建栈库并生成共享对象库文件

作为旁注，在 macOS 中，如果我们知道 `gcc` 是一个命令并且它指向 `clang` 编译器，我们可以运行前述的精确命令。否则，我们可以使用以下命令在 macOS 上构建库。请注意，在 macOS 中共享对象文件的扩展名是 .`dylib`：

```cpp
$ clang -c -g -fPIC cstack.c -o cstack.o
$ clang -dynamiclib cstack.o -o libcstack.dylib
$
```

Shell 框 21-3：在 macOS 中构建栈库并生成共享对象库文件

我们现在有了共享对象库文件，我们可以编写其他语言中的程序来加载它。在我们演示如何在前述库中加载和使用它在其他环境中的方法之前，我们需要编写一些测试来验证其功能。以下代码创建了一个栈并执行了一些可用的操作，并将结果与预期进行了比较：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cstack.h"
value_t make_int(int int_value) {
  value_t value;
  int* int_ptr = (int*)malloc(sizeof(int));
  *int_ptr = int_value;
  value.data = (char*)int_ptr;
  value.len = sizeof(int);
  return value;
}
int extract_int(value_t* value) {
  return *((int*)value->data);
}
void deleter(value_t* value) {
  if (value->data) {
    free(value->data);
  }
  value->data = NULL;
}
int main(int argc, char** argv) {
  cstack_t* cstack = cstack_new();
  cstack_ctor(cstack, 100);
  assert(cstack_size(cstack) == 0);
  int int_values[] = {5, 10, 20, 30};
  for (size_t i = 0; i < 4; i++) {
    cstack_push(cstack, make_int(int_values[i]));
  }
  assert(cstack_size(cstack) == 4);
  int counter = 3;
  value_t value;
  while (cstack_size(cstack) > 0) {
    bool_t popped = cstack_pop(cstack, &value);
    assert(popped);
    assert(extract_int(&value) == int_values[counter--]);
    deleter(&value);
  }
  assert(counter == -1);
  assert(cstack_size(cstack) == 0);
  cstack_push(cstack, make_int(10));
  cstack_push(cstack, make_int(20));
  assert(cstack_size(cstack) == 2);
  cstack_clear(cstack, deleter);
  assert(cstack_size(cstack) == 0);
   // In order to have something in the stack while
  // calling destructor.
  cstack_push(cstack, make_int(20));
  cstack_dtor(cstack, deleter);
  cstack_delete(cstack);
  printf("All tests were OK.\n");
  return 0;
}
```

代码框 21-3 [cstack_tests.c]: 测试 Stack 类功能性的代码

正如你所见，我们使用了断言来检查返回的值。以下是在 Linux 环境下构建并执行前述代码的输出。再次提醒，我们处于该章节的根目录：

```cpp
$ gcc -c -g cstack_tests.c -o tests.o
$ gcc tests.o -L$PWD -lcstack -o cstack_tests.out
$ LD_LIBRARY_PATH=$PWD ./cstack_tests.out
All tests were OK.
$
```

Shell 框 21-4: 构建和运行库测试

注意，在前面的 Shell 框中，当运行最终的可执行文件 `cstack_tests.out` 时，我们必须设置环境变量 `LD_LIBRARY_PATH` 以指向包含 `libcstack.so` 的目录，因为执行程序需要找到共享对象库并将它们加载。

如您在 *Shell 框 21-4* 中所见，所有测试都成功通过。这意味着从功能角度来看，我们的库运行正确。检查库是否符合非功能性要求，如内存使用或没有内存泄漏，将会很棒。

以下命令显示了如何使用 `valgrind` 检查测试的执行以检查任何可能的内存泄漏：

```cpp
$ LD_LIBRARY_PATH=$PWD valgrind --leak-check=full ./cstack_tests.out
==31291== Memcheck, a memory error detector
==31291== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==31291== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==31291== Command: ./cstack_tests.out
==31291==
All tests were OK.
==31291==
==31291== HEAP SUMMARY:
==31291==     in use at exit: 0 bytes in 0 blocks
==31291==   total heap usage: 10 allocs, 10 frees, 2,676 bytes allocated
==31291==
==31291== All heap blocks were freed -- no leaks are possible
==31291==
==31291== For counts of detected and suppressed errors, rerun with: -v
==31291== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
$
```

Shell 框 21-5: 使用 valgrind 运行测试

如您所见，我们没有任何内存泄漏，这使我们对我们编写的库更有信心。因此，如果我们看到另一个环境中的任何内存问题，首先应该在那里调查根本原因。

在下一章中，我们将介绍 C 的单元测试。作为 *代码框 21-3* 中看到的 `assert` 语句的合适替代品，我们可以编写单元测试并使用单元测试框架如 CMocka 来执行它们。

在接下来的几节中，我们将把堆栈库集成到用四种编程语言编写的程序中。我们将从 C++ 开始。

# 与 C++ 的集成

与 C++ 的集成可以认为是最容易的。C++ 可以被视为 C 的面向对象扩展。C++ 编译器生成的目标文件与 C 编译器生成的类似。因此，C++ 程序比其他任何编程语言更容易加载和使用 C 共享对象库。换句话说，共享对象文件是 C 还是 C++ 项目的输出并不重要；两者都可以被 C++ 程序消费。在某些情况下可能存在问题的唯一事情是 *第二章* 中描述的 C++ *名称修饰* 功能。作为提醒，我们将在以下部分简要回顾它。

## C++ 中的名称修饰

为了更详细地说明这一点，我们应该说，与函数（类中的全局函数和成员函数）对应的符号名称在 C++ 中会被修饰。名称修饰主要是为了支持 *命名空间* 和 *函数重载*，这些在 C 中是缺失的。名称修饰默认启用，因此如果 C 代码使用 C++ 编译器编译，我们期望看到修饰过的符号名称。看看以下 *代码框 21-4* 中的示例：

```cpp
int add(int a, int b) {
  return a + b;
}
```

代码框 21-4 [test.c]: C 语言中的一个简单函数

如果我们使用 C 编译器编译前面的文件，在这种情况下是 `clang`，我们将在生成的目标文件中看到以下符号，如 *Shell 框 21-6* 所示。请注意，书中的 GitHub 仓库中不存在 `test.c` 文件：

```cpp
$ clang -c test.c -o test.o
$ nm test.o
0000000000000000 T _add
$
```

Shell 框 21-6: 使用 C 编译器编译 test.c

正如你所见，我们有一个名为 `_add` 的符号，它指向上面定义的 `add` 函数。现在，让我们使用 C++ 编译器编译这个文件，在这种情况下是 `clang++`：

```cpp
$ clang++ -c test.c -o test.o
clang: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated [-Wdeprecated]
$ nm test.o
0000000000000000 T __Z3addii
$
```

Shell 框 21-7：使用 C++ 编译器编译 test.c

正如你所见，`clang++` 生成了一条警告，说明在不久的将来，将 C 代码编译为 C++ 代码的支持将被删除。但是，由于这种行为尚未被删除（并且只是已弃用），我们看到为前面函数生成的符号名称被修饰，并且与 `clang` 生成的不同。这肯定会在链接阶段查找特定符号时导致问题。

为了消除这个问题，需要将 C 代码封装在一个特殊的范围内，以防止 C++ 编译器对符号名称进行修饰。然后，使用 `clang` 和 `clang++` 编译它会产生相同的符号名称。看看 *代码框 21-5* 中的以下代码，这是 *代码框 21-4* 中引入的代码的修改版本：

```cpp
#ifdef __cplusplus
extern "C" {
#endif
int add(int a, int b) {
  return a + b;
}
#ifdef __cplusplus
}
#endif
```

代码框 21-5 [test.c]：将函数声明放入特殊的 C 范围

前面的函数仅在宏 `__cplusplus` 已经定义的情况下放入 `extern "C" { ... }` 范围内。拥有宏 `__cplusplus` 是代码正在由 C++ 编译器编译的标志。让我们再次使用 `clang++` 编译前面的代码：

```cpp
$ clang++ -c test.c -o test.o
clang: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated [-Wdeprecated]
$ nm test.o
0000000000000000 T _add
$
```

Shell 框 21-8：使用 clang++ 编译 test.c 的新版本

正如你所见，生成的符号不再被修饰。关于我们的堆栈库，根据我们到目前为止的解释，我们需要将所有声明放在 `extern "C" { … }` 范围内，这正是 *代码框 21-1* 中存在该范围的原因。因此，当将 C++ 程序与堆栈库链接时，符号可以在 `libcstack.so`（或 `libcstack.dylib`）中找到。

**注意**:

`extern "C"` 是一个 *链接规范*。更多信息可以通过以下链接找到：

https://isocpp.org/wiki/faq/mixing-c-and-cpp

https://stackoverflow.com/questions/1041866/what-is-the-effect-of-extern-c-in-c.

现在，是时候编写使用我们的堆栈库的 C++ 代码了。正如你很快就会看到的，这是一个简单的集成。

## C++ 代码

既然我们已经知道了如何在将 C 代码引入 C++ 项目时禁用名称修饰，我们可以通过编写一个使用堆栈库的 C++ 程序来继续。我们首先将堆栈库封装在一个 C++ 类中，这是面向对象 C++ 程序的主要构建块。以面向对象的方式暴露堆栈功能，而不是直接调用堆栈库的 C 函数，更为合适。

*代码框 21-6* 包含了从堆栈库派生出的封装堆栈功能的类：

```cpp
#include <string.h>
#include <iostream>
#include <string>
#include "cstack.h"
template<typename T>
value_t CreateValue(const T& pValue);
template<typename T>
T ExtractValue(const value_t& value);
template<typename T>
class Stack {
public:
  // Constructor
  Stack(int pMaxSize) {
    mStack = cstack_new();
    cstack_ctor(mStack, pMaxSize);
  }
  // Destructor
  ~Stack() {
    cstack_dtor(mStack, free_value);
    cstack_delete(mStack);
  }
  size_t Size() {
    return cstack_size(mStack);
  }
  void Push(const T& pItem) {
    value_t value = CreateValue(pItem);
    if (!cstack_push(mStack, value)) {
      throw "Stack is full!";
    }
  }
  const T Pop() {
    value_t value;
    if (!cstack_pop(mStack, &value)) {
      throw "Stack is empty!";
    }
    return ExtractValue<T>(value);
  }
  void Clear() {
    cstack_clear(mStack, free_value);
  }
private:
  cstack_t* mStack;
};
```

代码框 21-6 [c++/Stack.cpp]: 一个封装了堆栈库暴露的功能的 C++ 类

关于前面的类，我们可以指出以下重要注意事项：

+   前面的类保留了一个指向 `cstack_t` 变量的私有指针。这个指针指向由静态库的 `cstack_new` 函数创建的对象。这个指针可以被视为一个指向在 C 级别存在的对象的 *句柄*，由一个单独的 C 库创建和管理。指针 `mStack` 类似于文件描述符（或文件句柄），它指向一个文件。

+   该类封装了堆栈库公开的所有行为函数。这并不一定适用于围绕 C 库的任何面向对象的包装器，通常只公开有限的功能集。

+   前面的类是一个模板类。这意味着它可以操作多种数据类型。正如你所看到的，我们声明了两个模板函数用于序列化和反序列化具有各种类型的对象：`CreateValue` 和 `ExtractValue`。前面的类使用这些函数从 C++ 对象创建字节数组（序列化）以及从字节数组创建 C++ 对象（反序列化）。

+   我们为 `std::string` 类型定义了一个专门的模板函数。因此，我们可以使用前面的类来存储 `std::string` 类型的值。请注意，`std::string` 是 C++ 中用于字符串变量的标准类型。

+   作为堆栈库的一部分，你可以将来自不同类型的多个值推入单个堆栈实例。值可以转换为/从字符数组转换。查看 *代码框 21-1* 中的 `value_t` 结构。它只需要一个 `char` 指针，仅此而已。与堆栈库不同，前面的 C++ 类是 *类型安全的*，并且它的每个实例只能操作特定的数据类型。

+   在 C++ 中，每个类至少有一个构造函数和一个析构函数。因此，将底层堆栈对象作为构造函数的一部分进行初始化，并在析构函数中终止它是非常容易的。这正是前面代码所展示的。

我们希望我们的 C++ 类能够操作字符串值。因此，我们需要编写适当的序列化和反序列化函数，这些函数可以在类内部使用。以下代码包含将 C 字符数组转换为 `std::string` 对象以及相反转换的函数定义：

```cpp
template<>
value_t CreateValue(const std::string& pValue) {
  value_t value;
  value.len = pValue.size() + 1;
  value.data = new char[value.len];
  strcpy(value.data, pValue.c_str());
  return value;
}
template<>
std::string ExtractValue(const value_t& value) {
  return std::string(value.data, value.len);
}
```

代码框 21-7 [c++/Stack.cpp]：专门为 `std::string` 类型的序列化和反序列化设计的模板函数。这些函数作为 C++ 类的一部分被使用。

前面的函数是类中声明的模板函数的 `std::string` *特化*。正如你所看到的，它定义了如何将 `std::string` 对象转换为 C 字符数组，以及相反地，如何将 C 字符数组转换为 `std::string` 对象。

*代码框 21-8* 包含使用 C++ 类的 `main` 方法：

```cpp
int main(int argc, char** argv) {
  Stack<std::string> stringStack(100);
  stringStack.Push("Hello");
  stringStack.Push("World");
  stringStack.Push("!");
  std::cout << "Stack size: " << stringStack.Size() << std::endl;
  while (stringStack.Size() > 0) {
    std::cout << "Popped > " << stringStack.Pop() << std::endl;
  }
  std::cout << "Stack size after pops: " <<
      stringStack.Size() << std::endl;
  stringStack.Push("Bye");
  stringStack.Push("Bye");
  std::cout << "Stack size before clear: " <<
      stringStack.Size() << std::endl;
  stringStack.Clear();
  std::cout << "Stack size after clear: " <<
      stringStack.Size() << std::endl;
  return 0;
}
```

代码框 21-8 [c++/Stack.cpp]：使用 C++ 堆栈类的主体函数

上述场景涵盖了栈库公开的所有功能。我们执行了一系列操作并检查了它们的结果。请注意，前面的代码使用`Stack<std::string>`对象进行功能测试。因此，只能将`std::string`值推入或从栈中弹出。

下面的 shell box 展示了如何构建和运行前面的代码。请注意，本节中所有看到的 C++代码都是使用 C++11 编写的，因此应该使用兼容的编译器进行编译。正如我们之前所说的，当我们处于章节的根目录时，我们将运行以下命令：

```cpp
$ cd c++
$ g++ -c -g -std=c++11 -I$PWD/.. Stack.cpp -o Stack.o
$ g++ -L$PWD/.. Stack.o -lcstack -o cstack_cpp.out 
$ LD_LIBRARY_PATH=$PWD/.. ./cstack_cpp.out
Stack size: 3
Popped > !
Popped > World
Popped > Hello
Stack size after pops: 0
Stack size before clear: 2
Stack size after clear: 0
$
```

Shell Box 21-9：构建和运行 C++代码

如您所见，我们已经通过传递`-std=c++11`选项来指示我们将使用 C++11 编译器。请注意`-I`和`-L`选项，它们分别用于指定自定义包含和库目录。选项`-lcstack`要求链接器将 C++代码与库文件`libcstack.so`链接起来。请注意，在 macOS 系统上，共享对象库具有`.dylib`扩展名，因此您可能会找到`libcstack.dylib`而不是`libcstack.so`。

要运行`cstack_cpp.out`可执行文件，加载器需要找到`libcstack.so`。请注意，这与构建可执行文件不同。在这里，我们想要运行它，并且库文件必须在可执行文件运行之前位于正确的位置。因此，通过更改环境变量`LD_LIBRARY_PATH`，我们让加载器知道它应该在何处查找共享对象。我们已在*第二章*，*编译和链接*中对此进行了更多讨论。

C++代码也应该针对内存泄漏进行测试。`valgrind`帮助我们查看内存泄漏，我们用它来分析生成的可执行文件。下面的 shell box 展示了`valgrind`运行`cstack_cpp.out`可执行文件的输出：

```cpp
$ cd c++
$ LD_LIBRARY_PATH=$PWD/.. valgrind --leak-check=full ./cstack_cpp.out
==15061== Memcheck, a memory error detector
==15061== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==15061== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==15061== Command: ./cstack_cpp.out
==15061==
Stack size: 3
Popped > !
Popped > World
Popped > Hello
Stack size after pops: 0
Stack size before clear: 2
Stack size after clear: 0
==15061==
==15061== HEAP SUMMARY:
==15061==     in use at exit: 0 bytes in 0 blocks
==15061==   total heap usage: 9 allocs, 9 frees, 75,374 bytes allocated
==15061==
==15061== All heap blocks were freed -- no leaks are possible
==15061==
==15061== For counts of detected and suppressed errors, rerun with: -v
==15061== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0) 
$
```

Shell Box 21-10：使用 valgrind 构建和运行 C++代码

如前述输出所示，我们的代码中没有泄漏。请注意，`still reachable`部分中有 1081 字节并不意味着您的代码中存在泄漏。您可以在`valgrind`的手册中找到更多关于此的信息。

在本节中，我们解释了如何编写一个围绕我们的 C 栈库的 C++包装器。虽然混合 C 和 C++代码看起来很简单，但在 C++中应该注意一些额外的名称修饰规则。在下一节中，我们将简要介绍 Java 编程语言以及我们将如何在 Java 编写的程序中加载我们的 C 库。

# 与 Java 的集成

Java 程序由 Java 编译器编译成 Java 字节码。Java 字节码类似于在**应用程序二进制接口**（**ABI**）中指定的对象文件格式。包含 Java 字节码的文件不能像普通可执行文件那样执行，它们需要一个特殊的环境来运行。

Java 字节码只能在 **Java 虚拟机** (**JVM**) 中运行。JVM 本身是一个模拟 Java 字节码工作环境的进程。它通常用 C 或 C++ 编写，并且具有加载和使用 C 标准库以及在该层公开的功能的能力。

Java 编程语言不是唯一可以编译成 Java 字节码的语言。Scala、Kotlin 和 Groovy 等编程语言也可以编译成 Java 字节码，因此它们可以在 JVM 中运行。它们通常被称为 *JVM 语言*。

在本节中，我们将把已经构建的栈库加载到 Java 程序中。对于那些没有 Java 基础知识的人来说，我们采取的步骤可能看起来很复杂，难以理解。因此，强烈建议读者在进入本节之前对 Java 编程有一些基本了解。

## 编写 Java 部分

假设我们有一个构建成共享对象库的 C 项目。我们希望将其引入 Java 并使用其函数。幸运的是，我们可以编写和编译 Java 部分，而无需任何 C（或本地）代码。它们通过 Java 中的 *native 方法* 被很好地分离。显然，如果没有加载共享对象库文件，仅使用 Java 部分运行 Java 程序并调用 C 函数是不可能的。我们提供了必要的步骤和源代码来实现这一点，并成功运行了一个加载共享对象库并调用其函数的 Java 程序。

JVM 使用 **Java 本地接口** (**JNI**) 来加载共享对象库。请注意，JNI 不是 Java 编程语言的一部分；相反，它是 JVM 规范的一部分，因此导入的共享对象库可以在所有 JVM 语言（如 Scala）中使用。

在接下来的段落中，我们将展示如何使用 JNI 加载我们的目标共享对象库文件。

如我们之前所说，JNI 使用本地方法。本地方法在 Java 中没有定义；它们的实际定义是用 C 或 C++ 编写的，并驻留在外部共享库中。换句话说，本地方法是 Java 程序与 JVM 外部世界通信的端口。以下代码显示了一个包含多个静态本地方法的类，它应该公开我们的栈库提供的功能：

```cpp
package com.packt.extreme_c.ch21.ex1;
class NativeStack {
  static {
    System.loadLibrary("NativeStack");
  }
  public static native long newStack();
  public static native void deleteStack(long stackHandler);
  public static native void ctor(long stackHandler, int maxSize);
  public static native void dtor(long stackHandler);
  public static native int size(long stackHandler);
  public static native void push(long stackHandler, byte[] item);
  public static native byte[] pop(long stackHandler);
  public static native void clear(long stackHandler);
}
```

代码框 21-9 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: NativeStack 类

如方法签名所示，它们对应于我们在 C 栈库中的函数。请注意，第一个操作数是一个 `long` 变量。它包含从本地库中读取的本地地址，并作为指针传递给其他方法以表示栈实例。请注意，在编写前面的类时，我们不需要事先有一个完全工作的共享对象文件。我们所需的是定义栈 API 所需的声明列表。

前一个类还有一个*静态构造函数*。构造函数加载位于文件系统上的共享对象库文件，并尝试将本地方法与该共享对象库中找到的符号匹配。请注意，前面的共享对象库不是`libcstack.so`。换句话说，这并不是为我们自己的堆栈库生成的共享对象文件。JNI 有一个非常精确的配方来查找与本地方法相对应的符号。因此，我们不能使用定义在`libcstack.so`中的符号；相反，我们需要创建 JNI 正在寻找的符号，然后从那里使用我们的堆栈库。

这可能目前有点不清楚，但在下一节中，我们将澄清这一点，你将看到如何实现。让我们继续 Java 部分。我们仍然需要添加一些更多的 Java 代码。

下面的是一个名为`Stack<T>`的通用 Java 类，它封装了 JNI 公开的本地方法。通用 Java 类可以被视为与 C++中我们拥有的模板类的孪生概念。它们用于指定可以操作其他类型的某些通用类型。

正如你在`Stack<T>`类中看到的，有一个`marshaller`对象，其类型为`Marshaller<T>`，用于序列化和反序列化方法的输入参数（类型为`T`），以便将它们放入或从底层 C 堆栈中检索：

```cpp
interface Marshaller<T> {
  byte[] marshal(T obj);
  T unmarshal(byte[] data);
}
class Stack<T> implements AutoCloseable {
  private Marshaller<T> marshaller;
  private long stackHandler;
  public Stack(Marshaller<T> marshaller) {
    this.marshaller = marshaller;
    this.stackHandler = NativeStack.newStack();
    NativeStack.ctor(stackHandler, 100);
  }
  @Override
  public void close() {
    NativeStack.dtor(stackHandler);
    NativeStack.deleteStack(stackHandler);
  }
  public int size() {
    return NativeStack.size(stackHandler);
  }
  public void push(T item) {
    NativeStack.push(stackHandler, marshaller.marshal(item));
  }
  public T pop() {
    return marshaller.unmarshal(NativeStack.pop(stackHandler));
  }
  public void clear() {
    NativeStack.clear(stackHandler);
  }
}
```

代码框 21-10 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: `Stack<T>`类和`Marshaller<T>`接口

关于前面的代码，以下几点似乎值得关注：

+   `Stack<T>`类是一个通用类。这意味着它的不同实例可以操作各种类，如`String`、`Integer`、`Point`等，但每个实例只能操作在实例化时指定的类型。

+   在底层堆栈中存储任何数据类型的能力需要堆栈使用外部 marshaller 来执行对象的序列化和反序列化。C 堆栈库能够将字节数组存储在堆栈数据结构中，而愿意使用其功能的高级语言应该能够通过序列化输入对象来提供该字节数组。你很快就会看到`String`类的`Marshaller`接口的实现。

+   我们使用构造函数注入`Marshaller`实例。这意味着我们应该有一个已经创建的与类`T`的*通用*类型兼容的 marshaller 实例。

+   `Stack<T>`类实现了`AutoCloseable`接口。这仅仅意味着它有一些应该在销毁时释放的本地资源。请注意，实际的栈是在本地代码中创建的，而不是在 Java 代码中。因此，当不再需要栈时，JVM 的*垃圾回收器*无法释放栈。`AutoCloseable`对象可以用作具有特定作用域的资源，当它们不再需要时，它们的`close`方法会被自动调用。简而言之，你将看到我们如何在测试场景中使用前面的类。

+   如你所见，我们有一个构造函数，并且使用本地方法初始化了底层的栈。我们在类中保留了一个指向栈的`long`字段。请注意，与 C++不同，我们在这个类中没有任何析构函数。因此，有可能不释放底层的栈，并且最终可能导致内存泄漏。这就是为什么我们将这个类标记为`AutoCloseable`。当一个`AutoCloseable`对象不再需要时，它的`close`方法会被调用，正如你在前面的代码中所看到的，我们调用了 C 栈库的析构函数来释放由 C 栈分配的资源。

通常，你不能信任垃圾回收器机制在 Java 对象上调用*终结器方法*，使用`AutoCloseable`资源是管理本地资源的正确方式。

下面的内容是`StringMarshaller`的实现。由于`String`类在处理字节数组方面提供了很好的支持，所以实现非常直接：

```cpp
class StringMarshaller implements Marshaller<String> {
  @Override
  public byte[] marshal(String obj) {
    return obj.getBytes();
  }
  @Override
  public String unmarshal(byte[] data) {
    return new String(data);
  }
}
```

代码框 21-11 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: `StringMarshaller`类

以下代码是我们的`Main`类，它包含了通过 Java 代码演示 C 栈功能的测试场景：

```cpp
public class Main {
  public static void main(String[] args) {
    try (Stack<String> stack = new Stack<>(new StringMarshaller())) {
      stack.push("Hello");
      stack.push("World");
      stack.push("!");
      System.out.println("Size after pushes: " + stack.size());
      while (stack.size() > 0) {
        System.out.println(stack.pop());
      }
      System.out.println("Size after pops: " + stack.size());
      stack.push("Ba");
      stack.push("Bye!");
      System.out.println("Size after before clear: " + stack.size());
      stack.clear();
      System.out.println("Size after clear: " + stack.size());
    }
  }
}
```

代码框 21-12 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: 包含测试场景以检查 C 栈库功能的`Main`类

如你所见，引用变量`stack`是在一个`try`块内部创建和使用的。这种语法通常被称为*try-with-resources*，并且它是作为 Java 7 的一部分被引入的。当`try`块执行完毕后，会在资源对象上调用`close`方法，并且底层的栈被释放。测试场景与我们在上一节为 C++编写的场景相同，但这次是在 Java 中。

在本节中，我们涵盖了 Java 部分以及我们需要导入本地部分的所有 Java 代码。上述所有源代码都可以编译，但你不能运行它们，因为你还需要本地部分。只有两者结合才能生成可执行程序。在下一节中，我们将讨论我们应该采取的步骤来编写本地部分。

## 编写本地部分

在上一节中，我们介绍的最重要的事情是本地方法的概念。本地方法在 Java 中声明，但其定义位于 JVM 外部的共享对象库中。但是 JVM 如何在加载的共享对象文件中找到本地方法的定义呢？答案是简单的：通过在共享对象文件中查找某些符号名称。JVM 根据其各种属性（如包、包含的类和名称）为每个本地方法提取一个符号名称。然后，它在加载的共享对象库中查找该符号，如果找不到，它会给你一个错误。

根据我们在上一节中建立的内容，JVM 强制我们使用特定的符号名称来编写作为加载的共享对象文件一部分的函数。但我们在创建栈库时没有使用任何特定的约定。因此，JVM 将无法从栈库中找到我们公开的函数，我们必须想出另一种方法。通常，C 库是在没有任何假设会被用于 JVM 环境的情况下编写的。

*图 21-1* 展示了我们可以如何使用一个中间的 C 或 C++ 库作为 Java 部分和本地部分之间的粘合剂。我们给 JVM 它想要的符号，并将对代表这些符号的函数的调用委托给 C 库中的正确函数。这基本上就是 JNI 的工作方式。

我们将通过一个假设的例子来解释这一点。假设我们想从 Java 中调用一个 C 函数 `func`，该函数的定义可以在 `libfunc.so` 共享对象文件中找到。我们还在 Java 部分有一个名为 `Clazz` 的类，其中有一个名为 `doFunc` 的本地函数。我们知道 JVM 在尝试找到本地函数 `doFunc` 的定义时会查找符号 `Java_Clazz_doFunc`。我们创建一个中间共享对象库 `libNativeLibrary.so`，其中包含一个具有与 JVM 寻找的符号完全相同的函数。然后，在该函数内部，我们调用 `func` 函数。我们可以这样说，函数 `Java_Clazz_doFunc` 作为中继，将调用委托给底层的 C 库，最终是 `func` 函数。

![fig10-1](img/B11046_21_01.png)

图 21-1：中间共享对象库 libNativeStack.so，用于将函数调用从 Java 委托到实际的底层 C 栈库，libcstack.so。

为了与 JVM 符号名称保持一致，Java 编译器通常会将 Java 代码中找到的本地方法生成一个 C 头文件。这样，你只需要编写头文件中找到的函数的定义。这可以防止我们在符号名称上犯任何错误，这些错误最终会被 JVM 查找。

以下命令展示了如何编译一个 Java 源文件，以及如何请求编译器为其中找到的本地方法生成头文件。在这里，我们将编译我们唯一的 Java 文件 `Main.java`，它包含了之前代码框中引入的所有 Java 代码。请注意，在运行以下命令时，我们应该位于章节的根目录中：

```cpp
$ cd java
$ mkdir -p build/headers
$ mkdir -p build/classes
$ javac -cp src -h build/headers -d build/classes \
src/com/packt/extreme_c/ch21/ex1/Main.java
$ tree build
build
├── classes
│   └── com
│       └── packt
│           └── extreme_c
│               └── ch21
│                   └── ex1
│                       ├── Main.class
│                       ├── Marshaller.class
│                       ├── NativeStack.class
│                       ├── Stack.class
│                       └── StringMarshaller.class
└── headers
    └── com_packt_extreme_c_ch21_ex1_NativeStack.h
7 directories, 6 files
$
```

Shell Box 21-11：编译 Main.java 并为文件中找到的本地方法生成头文件

如前述 shell box 所示，我们向 Java 编译器 `javac` 传递了选项 `-h`。我们还指定了一个目录，所有头文件都应该放在那里。`tree` 工具以树形格式显示了 `build` 目录的内容。注意 `.class` 文件。它们包含 Java 字节码，当将这些类加载到 JVM 实例时将使用这些字节码。

除了类文件外，我们还看到了一个头文件 `com_packt_extreme_c_ch21_ex1_NativeStack.h`，它包含了在 `NativeStack` 类中找到的本地方法的相应 C 函数声明。

如果你打开头文件，你会看到类似于 *Code Box 21-13* 的内容。它包含了许多具有长而奇怪的名称的函数声明，每个名称都由包名、类名和相应的本地方法名称组成：

```cpp
/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_packt_extreme_c_ch21_ex1_NativeStack */
#ifndef _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#define _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_packt_extreme_c_ch21_ex1_NativeStack
 * Method:    newStack
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_packt_extreme_1c_ch21_ex1_NativeStack_newStack
  (JNIEnv *, jclass);
/*
 * Class:     com_packt_extreme_c_ch21_ex1_NativeStack
 * Method:    deleteStack
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_packt_extreme_1c_ch21_ex1_NativeStack_deleteStack
  (JNIEnv *, jclass, jlong);

...
...
...
#ifdef __cplusplus
}
#endif
#endif
```

Code Box 21-13：生成的 JNI 头文件的内容（不完整）

在前面的头文件中声明的函数携带了 JVM 在加载对应于本地方法的 C 函数时将寻找的符号名称。我们已经修改了前面的头文件，并使用宏使其紧凑，以便在更小的区域内放置所有函数声明。你可以在 *Code Box 21-14* 中看到它：

```cpp
// Filename: NativeStack.h
// Description: Modified JNI generated header file
#include <jni.h>
#ifndef _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#define _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#define JNI_FUNC(n) Java_com_packt_extreme_1c_ch21_ex1_NativeStack_##
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT jlong JNICALL JNI_FUNC(newStack)(JNIEnv* , jclass);
JNIEXPORT void JNICALL JNI_FUNC(deleteStack)(JNIEnv* , jclass, jlong);
JNIEXPORT void JNICALL JNI_FUNC(ctor)(JNIEnv* , jclass, jlong, jint);
JNIEXPORT void JNICALL JNI_FUNC(dtor)(JNIEnv* , jclass, jlong);
JNIEXPORT jint JNICALL JNI_FUNC(size)(JNIEnv* , jclass, jlong);
JNIEXPORT void JNICALL JNI_FUNC(push)(JNIEnv* , jclass, jlong, jbyteArray);
JNIEXPORT jbyteArray JNICALL JNI_FUNC(pop)(JNIEnv* , jclass, jlong);
JNIEXPORT void JNICALL JNI_FUNC(clear)(JNIEnv* , jclass, jlong);
#ifdef __cplusplus
}
#endif
#endif
```

Code Box 21-14 [java/native/NativeStack.h]：生成的 JNI 头文件的修改版本

正如你所见，我们创建了一个新的宏 `JNI_FUNC`，它提取了所有声明中通用的函数名的大部分。我们还移除了注释，以便使头文件更加紧凑。

我们将在头文件和随后的源文件中使用宏 `JNI_FUNC`，这些文件作为 *Code Box 21-15* 的一部分展示。

**注意**：

修改生成的头文件不是一个被接受的行为。我们这样做是因为教育目的。在真实的构建环境中，我们希望直接使用生成的文件，而不做任何修改。

在 *Code Box 21-15* 中，你可以找到前面函数的定义。正如你所见，定义仅将调用传递给从 C 栈库包含的底层 C 函数：

```cpp
#include <stdlib.h>
#include "NativeStack.h"
#include "cstack.h"
void defaultDeleter(value_t* value) {
  free_value(value);
}
void extractFromJByteArray(JNIEnv* env,
                           jbyteArray byteArray,
                           value_t* value) {
  jboolean isCopy = false;
  jbyte* buffer = env->GetByteArrayElements(byteArray, &isCopy);
  value->len = env->GetArrayLength(byteArray);
  value->data = (char*)malloc(value->len * sizeof(char));
  for (size_t i = 0; i < value->len; i++) {
    value->data[i] = buffer[i];
  }
  env->ReleaseByteArrayElements(byteArray, buffer, 0);
}
JNIEXPORT jlong JNICALL JNI_FUNC(newStack)(JNIEnv* env,
                                           jclass clazz) {
  return (long)cstack_new();
}
JNIEXPORT void JNICALL JNI_FUNC(deleteStack)(JNIEnv* env,
                                            jclass clazz,
                                            jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_delete(cstack);
}
JNIEXPORT void JNICALL JNI_FUNC(ctor)(JNIEnv *env,
                                      jclass clazz,
                                      jlong stackPtr,
                                      jint maxSize) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_ctor(cstack, maxSize);
}
JNIEXPORT void JNICALL JNI_FUNC(dtor)(JNIEnv* env,
                                      jclass clazz,
                                      jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_dtor(cstack, defaultDeleter);
}
JNIEXPORT jint JNICALL JNI_FUNC(size)(JNIEnv* env,
                                      jclass clazz,
                                      jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  return cstack_size(cstack);
}
JNIEXPORT void JNICALL JNI_FUNC(push)(JNIEnv* env,
                                      jclass clazz,
                                      jlong stackPtr,
                                      jbyteArray item) {
  value_t value;
  extractFromJByteArray(env, item, &value);
  cstack_t* cstack = (cstack_t*)stackPtr;
  bool_t pushed = cstack_push(cstack, value);
  if (!pushed) {
    jclass Exception = env->FindClass("java/lang/Exception");
    env->ThrowNew(Exception, "Stack is full!");
  }
}
JNIEXPORT jbyteArray JNICALL JNI_FUNC(pop)(JNIEnv* env,
                                           jclass clazz,
                                           jlong stackPtr) {
  value_t value;
  cstack_t* cstack = (cstack_t*)stackPtr;
  bool_t popped = cstack_pop(cstack, &value);
  if (!popped) {
    jclass Exception = env->FindClass("java/lang/Exception");
    env->ThrowNew(Exception, "Stack is empty!");
  }
  jbyteArray result = env->NewByteArray(value.len);
  env->SetByteArrayRegion(result, 0,
          value.len, (jbyte*)value.data);
  defaultDeleter(&value);
  return result;
}
JNIEXPORT void JNICALL JNI_FUNC(clear)(JNIEnv* env,
                                       jclass clazz,
                                       jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_clear(cstack, defaultDeleter);
}
```

Code Box 21-15 [java/native/NativeStack.cpp]：JNI 头文件中声明的函数的定义

上一段代码是用 C++ 编写的。也可以用 C 语言编写定义。唯一需要关注的是在 push 和 pop 函数中发生的从 C 字节数组到 Java 字节数组的转换。已经添加了 `extractFromJByteArray` 函数，用于根据从 Java 部分接收到的 Java 字节数组创建一个 C 字节数组。

以下命令在 Linux 中创建中间共享对象 `libNativeStack.so`，它将被 JVM 加载和使用。注意，在运行以下命令之前，您需要设置环境变量 `JAVA_HOME`：

```cpp
$ cd java/native
$ g++ -c -fPIC -I$PWD/../.. -I$JAVA_HOME/include \
 -I$JAVA_HOME/include/linux NativeStack.cpp -o NativeStack.o
$ g++ -shared -L$PWD/../.. NativeStack.o -lcstack -o libNativeStack.so
$
```

Shell Box 21-12: 构建中间共享对象库 libNativeStack.so

如您所见，最终的共享对象文件链接到 C 堆栈库的共享对象文件 `libcstack.so`，这仅仅意味着 `libNativeStack.so` 必须加载 `libcstack.so` 才能工作。因此，JVM 加载 `libNativeStack.so` 库，然后加载 `libcstack.so` 库，最终 Java 部分和本地部分可以合作，使 Java 程序得以执行。

以下命令运行 *代码框 21-12* 中显示的测试场景：

```cpp
$ cd java
$ LD_LIBRARY_PATH=$PWD/.. java -Djava.library.path=$PWD/native \
  -cp build/classes com.packt.extreme_c.ch21.ex1.Main
Size after pushes: 3
!
World
Hello
Size after pops: 0
Size after before clear: 2
Size after clear: 0
$
```

Shell Box 21-13: 运行 Java 测试场景

如您所见，我们已将选项 `-Djava.library.path=...` 传递给 JVM。它指定了共享对象库可以找到的位置。如您所见，我们已指定应包含 `libNativeStack.so` 共享对象库的目录。

在本节中，我们展示了如何将本地 C 库加载到 JVM 中，并与其他 Java 源代码一起使用。相同的机制也可以用于加载更大的多部分本地库。

现在，是时候通过 Python 集成来了解如何从 Python 代码中使用 C 堆栈库了。

# 与 Python 集成

Python 是一种 *解释型* 编程语言。这意味着 Python 代码是由一个称为 *解释器* 的中间程序读取和运行的。如果我们打算使用外部本地共享库，那么是解释器加载共享库并将其提供给 Python 代码。Python 有一个用于加载外部共享库的特殊框架。它被称为 *ctypes*，我们将在本节中使用它。

使用 `ctypes` 加载共享库非常简单。它只需要加载库并定义将要使用的函数的输入和输出。以下类封装了 ctypes 相关逻辑，并将其提供给我们的主 `Stack` 类，如即将显示的代码框所示：

```cpp
from ctypes import *
class value_t(Structure):
  _fields_ = [("data", c_char_p), ("len", c_int)]
class _NativeStack:
  def __init__(self):
    self.stackLib = cdll.LoadLibrary(
            "libcstack.dylib" if platform.system() == 'Darwin'
            else "libcstack.so")
    # value_t make_value(char*, size_t)
    self._makevalue_ = self.stackLib.make_value
    self._makevalue_.argtypes = [c_char_p, c_int]
    self._makevalue_.restype = value_t
    # value_t copy_value(char*, size_t)
    self._copyvalue_ = self.stackLib.copy_value
    self._copyvalue_.argtypes = [c_char_p, c_int]
    self._copyvalue_.restype = value_t
    # void free_value(value_t*)
    self._freevalue_ = self.stackLib.free_value
    self._freevalue_.argtypes = [POINTER(value_t)]
    # cstack_t* cstack_new()
    self._new_ = self.stackLib.cstack_new
    self._new_.argtypes = []
    self._new_.restype = c_void_p
    # void cstack_delete(cstack_t*)
    self._delete_ = self.stackLib.cstack_delete
    self._delete_.argtypes = [c_void_p]
    # void cstack_ctor(cstack_t*, int)
    self._ctor_ = self.stackLib.cstack_ctor
    self._ctor_.argtypes = [c_void_p, c_int]
    # void cstack_dtor(cstack_t*, deleter_t)
    self._dtor_ = self.stackLib.cstack_dtor
    self._dtor_.argtypes = [c_void_p, c_void_p]
    # size_t cstack_size(cstack_t*)
    self._size_ = self.stackLib.cstack_size
    self._size_.argtypes = [c_void_p]
    self._size_.restype = c_int
    # bool_t cstack_push(cstack_t*, value_t)
    self._push_ = self.stackLib.cstack_push
    self._push_.argtypes = [c_void_p, value_t]
    self._push_.restype = c_int
    # bool_t cstack_pop(cstack_t*, value_t*)
    self._pop_ = self.stackLib.cstack_pop
    self._pop_.argtypes = [c_void_p, POINTER(value_t)]
    self._pop_.restype = c_int
    # void cstack_clear(cstack_t*, deleter_t)
    self._clear_ = self.stackLib.cstack_clear
    self._clear_.argtypes = [c_void_p, c_void_p]
```

代码框 21-17 [python/stack.py]: 使堆栈库的 C 函数对 Python 的其余部分可用的 ctypes 相关代码

如你所见，所有需要在我们的 Python 代码中使用的函数都被放在了类定义中。C 函数的句柄存储在类的实例的私有字段中（私有字段在两边都有 `_`），并且可以用来调用底层的 C 函数。请注意，在上面的代码中，我们加载了 `libcstack.dylib`，因为我们是在 macOS 系统上。而对于 Linux 系统，我们需要加载 `libcstack.so`。

下面的类是主要的 Python 组件，它使用了上面的包装类。所有其他的 Python 代码都使用这个类来获得栈功能：

```cpp
class Stack:
  def __enter__(self):
    self._nativeApi_ = _NativeStack()
    self._handler_ = self._nativeApi_._new_()
    self._nativeApi_._ctor_(self._handler_, 100)
    return self
  def __exit__(self, type, value, traceback):
    self._nativeApi_._dtor_(self._handler_, self._nativeApi_._freevalue_)
    self._nativeApi_._delete_(self._handler_)
  def size(self):
    return self._nativeApi_._size_(self._handler_)
  def push(self, item):
    result = self._nativeApi_._push_(self._handler_,
            self._nativeApi_._copyvalue_(item.encode('utf-8'), len(item)));
    if result != 1:
      raise Exception("Stack is full!")
  def pop(self):
    value = value_t()
    result = self._nativeApi_._pop_(self._handler_, byref(value))
    if result != 1:
      raise Exception("Stack is empty!")
    item = string_at(value.data, value.len)
    self._nativeApi_._freevalue_(value)
    return item
  def clear(self):
    self._nativeApi_._clear_(self._handler_, self._nativeApi_._freevalue_)
```

代码框 21-16 [python/stack.py]：使用从栈库加载的 C 函数的 Python 中的 `Stack` 类

如你所见，`Stack` 类保持对 `_NativeStack` 类的引用，以便能够调用底层的 C 函数。请注意，前面的类覆盖了 `__enter__` 和 `__exit__` 函数。这使得该类可以用作资源类，并在 Python 的 `with` 语法中使用。你很快就会看到这种语法的样子。请注意，前面的 `Stack` 类仅对字符串项进行操作。

以下是一个测试场景，它与 Java 和 C++ 的测试场景非常相似：

```cpp
if __name__ == "__main__":
  with Stack() as stack:
    stack.push("Hello")
    stack.push("World")
    stack.push("!")
    print("Size after pushes:" + str(stack.size()))
    while stack.size() > 0:
      print(stack.pop())
    print("Size after pops:" + str(stack.size()))
    stack.push("Ba");
    stack.push("Bye!");
    print("Size before clear:" + str(stack.size()))
    stack.clear()
    print("Size after clear:" + str(stack.size()))
```

代码框 21-18 [python/stack.py]：使用 `Stack` 类编写的 Python 测试场景

在前面的代码中，你可以看到 Python 的 `with` 语句。

当进入 `with` 块时，会调用 `__enter__` 函数，并通过 `stack` 变量引用 `Stack` 类的实例。当离开 `with` 块时，会调用 `__exit__` 函数。这给了我们机会在不需要底层原生资源（在这种情况下是 C 栈对象）时释放它们。

接下来，你可以看到如何运行前面的代码。请注意，所有的 Python 代码框都存在于同一个名为 `stack.py` 的文件中。在运行以下命令之前，你需要位于章节的根目录中：

```cpp
$ cd python
$ LD_LIBRARY_PATH=$PWD/.. python stack.py
Size after pushes:3
!
World
Hello
Size after pops:0
Size before clear:2
Size after clear:0
$
```

Shell 框 21-14：运行 Python 测试场景

注意，解释器应该能够找到并加载 C 栈共享库；因此，我们将 `LD_LIBRARY_PATH` 环境变量设置为指向包含实际共享库文件的目录。

在下一节中，我们将展示如何在 Go 语言中加载和使用 C 栈库。

# 与 Go 的集成

Go 编程语言（或简称 Golang）与本地共享库的集成非常容易。它可以被认为是 C 和 C++ 编程语言的下一代，并且它将自己称为系统编程语言。因此，当我们使用 Golang 时，我们期望能够轻松地加载和使用本地库。

在 Golang 中，我们使用一个名为 *cgo* 的内置包来调用 C 代码和加载共享对象文件。在下面的 Go 代码中，你可以看到如何使用 `cgo` 包，并使用它来调用从 C 栈库文件加载的 C 函数。它还定义了一个新的类，`Stack`，该类被其他 Go 代码用来使用 C 栈功能：

```cpp
package main
/*
#cgo CFLAGS: -I..
#cgo LDFLAGS: -L.. -lcstack
#include "cstack.h"
*/
import "C"
import (
  "fmt"
)
type Stack struct {
  handler *C.cstack_t
}
func NewStack() *Stack {
  s := new(Stack)
  s.handler = C.cstack_new()
  C.cstack_ctor(s.handler, 100)
  return s
}
func (s *Stack) Destroy() {
  C.cstack_dtor(s.handler, C.deleter_t(C.free_value))
  C.cstack_delete(s.handler)
}
func (s *Stack) Size() int {
  return int(C.cstack_size(s.handler))
}
func (s *Stack) Push(item string) bool {
  value := C.make_value(C.CString(item), C.ulong(len(item) + 1))
  pushed := C.cstack_push(s.handler, value)
  return pushed == 1
}
func (s *Stack) Pop() (bool, string) {
  value := C.make_value(nil, 0)
  popped := C.cstack_pop(s.handler, &value)
  str := C.GoString(value.data)
  defer C.free_value(&value)
  return popped == 1, str
}
func (s *Stack) Clear() {
  C.cstack_clear(s.handler, C.deleter_t(C.free_value))
}
```

Code Box 21-19 [go/stack.go]：使用加载的 libcstack.so 共享对象文件的 Stack 类

为了使用 cgo 包，需要导入`C`包。它加载在伪`#cgo`指令中指定的共享对象库。正如你所看到的，我们指定了`libcstack.so`库作为指令`#cgo LDFLAGS: -L.. -lcstack`的一部分。请注意，`CFLAGS`和`LDFLAGS`包含直接传递给 C 编译器和链接器的标志。

我们还指出了应该搜索共享对象文件的路径。之后，我们可以使用`C`结构体来调用加载的本地函数。例如，我们使用了`C.cstack_new()`来调用栈库中的相应函数。使用 cgo 非常简单。请注意，前面的`Stack`类仅适用于字符串项。

以下代码展示了用 Golang 编写的测试场景。请注意，当退出`main`函数时，我们必须在`stack`对象上调用`Destroy`函数：

```cpp
func main() {
  var stack = NewStack()
  stack.Push("Hello")
  stack.Push("World")
  stack.Push("!")
  fmt.Println("Stack size:", stack.Size())
  for stack.Size() > 0 {
    _, str := stack.Pop()
    fmt.Println("Popped >", str)
  }
  fmt.Println("Stack size after pops:", stack.Size())
  stack.Push("Bye")
  stack.Push("Bye")
  fmt.Println("Stack size before clear:", stack.Size())
  stack.Clear()
  fmt.Println("Stack size after clear:", stack.Size())
  stack.Destroy()
}
```

Code Box 21-20 [go/stack.go]：使用 Stack 类的 Go 测试场景

以下 shell box 演示了如何构建和运行测试场景：

```cpp
$ cd go
$ go build -o stack.out stack.go
$ LD_LIBRARY_PATH=$PWD/.. ./stack.out
Stack size: 3
Popped > !
Popped > World
Popped > Hello
Stack size after pops: 0
Stack size before clear: 2
Stack size after clear: 0
$
```

Shell Box 21-15：运行 Go 测试场景

正如你在 Golang 中看到的，与 Python 不同，你需要首先编译你的程序，然后运行它。此外，我们仍然需要设置`LD_LIBRARY_PATH`环境变量，以便允许可执行文件定位`libcstack.so`库并将其加载。

在本节中，我们展示了如何使用 Golang 中的`cgo`包加载和使用共享对象库。由于 Golang 类似于 C 代码的薄包装器，因此它比使用 Python 和 Java 加载外部共享对象库并使用它更容易。

# 摘要

在本章中，我们介绍了 C 语言与其他编程语言的集成。作为本章的一部分：

+   我们设计了一个 C 库，该库暴露了一些栈功能，例如 push、pop 等。我们构建了库，并最终生成了一个共享对象库，供其他语言使用。

+   我们讨论了 C++中的名称混淆功能，以及我们在使用 C++编译器时应该如何避免在 C 中使用它。

+   我们编写了一个围绕栈库的 C++包装器，该包装器可以加载库的共享对象文件并在 C++中执行加载的功能。

+   我们继续编写了一个围绕 C 库的 JNI 包装器。我们使用了本地方法来实现这一点。

+   我们展示了如何使用 JNI 编写本地代码，并将本地部分和 Java 部分连接起来，最终运行一个使用 C 栈库的 Java 程序。

+   我们成功地编写了使用 ctypes 包加载和使用库的共享对象文件的 Python 代码。

+   作为最后一部分，我们用 Golang 编写了一个程序，该程序可以在`cgo`包的帮助下加载库的共享对象文件。

下一章将介绍 C 语言中的单元测试和调试。我们将介绍一些用于编写单元测试的 C 语言库。不仅如此，我们还将讨论 C 语言的调试，以及一些可用于调试或监控程序的现有工具。
