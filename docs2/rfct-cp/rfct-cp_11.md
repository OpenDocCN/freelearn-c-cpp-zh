

# 第十一章：动态分析

在软件开发的错综复杂世界中，确保代码的正确性、效率和安全性不仅是一个目标，更是一种必要性。这在 C++编程中尤其如此，因为该语言的力量和复杂性既提供了机会，也带来了挑战。在 C++中保持高代码质量的最有效方法之一是**动态代码分析**——这是一个审查程序在运行时行为的过程，以检测一系列潜在问题。

动态代码分析与静态分析形成对比，后者在执行代码之前检查源代码。虽然静态分析在开发周期早期捕获语法错误、代码异味和某些类型的错误方面非常有价值，但动态分析则更深入。它揭示了仅在程序实际执行过程中才会出现的问题，例如内存泄漏、竞态条件和可能导致崩溃、异常行为或安全漏洞的运行时错误。

本章旨在探索 C++中动态代码分析工具的领域，特别关注行业中一些最强大和最广泛使用的工具：一套基于编译器的清理器，包括**AddressSanitizer（ASan**）、**ThreadSanitizer（TSan**）和**UndefinedBehaviorSanitizer（UBSan**），以及 Valgrind，这是一个以其详尽的内存调试能力而闻名的多功能工具。

编译器清理器，作为 LLVM 项目和 GCC 项目的一部分，为动态分析提供了一系列选项。ASan 因其能够检测各种内存相关错误而著称，TSan 在识别多线程代码中的竞态条件方面表现出色，而 UBSan 有助于捕捉可能导致程序行为不可预测的未定义行为。这些工具因其效率、精确性和易于集成到现有开发工作流程中而受到赞誉。其中大多数都得到了 GCC 和 MSVC 的支持。

另一方面，Valgrind，一个用于构建动态分析工具的仪器框架，凭借其全面的内存泄漏检测和分析二进制可执行文件的能力（无需重新编译源代码），而显得格外耀眼。它是在深入内存分析至关重要的复杂场景下的首选解决方案，尽管这会带来更高的性能开销。

在本章中，我们将深入研究这些工具的每一个，了解它们的优点、缺点和适当的用例。我们将探讨如何有效地将它们集成到你的 C++开发过程中，以及它们如何相互补充，为确保 C++应用程序的质量和可靠性提供一个强大的框架。

到本章结束时，你将全面理解 C++中的动态代码分析，并具备选择和利用适合你特定开发需求工具的知识，最终导致编写更清洁、更高效、更可靠的 C++代码。

# 基于编译器的动态代码分析

基于编译器的清理器包含两部分：编译器仪器化和运行时诊断：

+   **编译器仪器化**: 当你使用清理器编译你的 C++ 代码时，编译器会对生成的二进制文件进行额外的检查。这些检查被策略性地插入到代码中，以监控特定类型的错误。例如，ASan 会添加代码来跟踪内存分配和访问，使其能够检测内存误用，如缓冲区溢出和内存泄漏。

+   **运行时诊断**: 当被仪器化的程序运行时，这些检查会积极监控程序的行为。当一个清理器检测到错误（如内存访问违规或数据竞争）时，它会立即报告，通常还会提供关于错误位置和性质的详细信息。这种实时反馈对于识别和修复难以通过传统测试捕获的隐蔽错误非常有价值。

尽管所有编译器团队都在不断努力添加新的清理器并改进现有的清理器，但基于编译器的清理器仍然存在一些限制：

+   **Clang 和 GCC**: 大多数清理器，包括 ASan、TSan 和 UBSan，都由 Clang 和 GCC 支持。这种广泛的支持使得它们对大量 C++ 开发社区成员可访问，无论他们偏好的编译器是什么。

+   **Microsoft Visual C++ (MSVC)**: MSVC 也支持一些清理器，尽管其范围和能力可能与 Clang 和 GCC 不同。例如，MSVC 支持 ASan，这对于 Windows 特定的 C++ 开发很有用。

+   **跨平台实用工具**: 这些工具的跨编译器和跨平台特性意味着它们可以在各种开发环境中使用，从 Linux 和 macOS 到 Windows，增强了它们在多样化的 C++ 项目中的实用性。

## ASan

ASan 是一个运行时内存错误检测器，是 LLVM 编译器基础设施、GCC 和 MSVC 的一部分。它作为开发者识别和解决各种内存相关错误的专用工具，包括但不限于缓冲区溢出、悬垂指针访问和内存泄漏。该工具通过在编译过程中对代码进行仪器化来实现这一点，使其能够在运行时监控内存访问和分配。

ASan 的一个关键优势是它能够提供详细的错误报告。当检测到内存错误时，ASan 会输出全面的信息，包括错误类型、涉及的内存位置和堆栈跟踪。这种详细程度大大有助于调试过程，使开发者能够快速定位问题的根源。

将 ASan 集成到 C++ 开发工作流程中非常简单。它需要对构建过程进行最小的更改，通常涉及在编译期间添加编译器标志（`-fsanitize=address`）。为了获得更好的结果，使用合理的性能选项（`-O1` 或更高）是有意义的。为了在错误消息中获得更好的堆栈跟踪，请添加 `-fno-omit-frame-pointer`。这种易于集成的便利性，加上其捕获内存错误的有效性，使 ASan 成为开发者增强其 C++ 应用程序可靠性和安全性的不可或缺的工具。

### 在 ASan 中符号化报告

当使用 ASan 在 C++ 应用程序中检测内存错误时，符号化错误报告至关重要。符号化将 ASan 输出的内存地址和偏移量转换为人类可读的函数名、文件名和行号。这个过程对于有效的调试至关重要，因为它允许开发者轻松地识别内存错误在源代码中的确切位置。

没有符号化，ASan 报告提供的是不太有意义的原始内存地址，这使得难以追踪到源代码中错误发生的确切位置。另一方面，符号化报告提供了清晰且可操作的信息，使开发者能够快速理解和修复代码中的潜在问题。

ASan 符号化的配置通常是自动的，不需要额外的步骤。然而，在某些情况下，你可能需要显式设置 `ASAN_SYMBOLIZER_PATH` 环境变量，以便指向符号化工具。这在非 Linux Unix 系统上尤其如此，在这些系统上可能需要额外的工具，如 `addr2line`，来进行符号化。如果它不能直接工作，请按照以下步骤进行，以确保符号化配置正确：

1.  在编译命令中使用 `-g` 标志。例如：

    ```cpp
    clang++ -fsanitize=address -g -o your_program your_file.cpp
    ```

1.  使用 `-g` 编译选项会在二进制文件中包含调试符号，这对于符号化是必不可少的。

1.  `llvm-symbolizer` 工具位于你的系统 `PATH` 中。

1.  `addr2line`（GNU Binutils 的一部分）可用于符号化堆栈跟踪。

1.  将 `ASAN_SYMBOLIZER_PATH` 环境变量指向符号化工具。例如：

    ```cpp
    export ASAN_SYMBOLIZER_PATH=/path/to/llvm-symbolizer
    ```

1.  这明确告诉 ASan 使用哪个符号化工具。

1.  **运行** **你的程序**：

    +   按照常规运行编译后的程序。如果检测到内存错误，ASan 将输出符号化的堆栈跟踪。

    +   报告将包括函数名、文件名和行号，这使得更容易定位和解决代码中的错误。

### 越界访问

让我们尝试捕捉 C++ 编程中最关键的错误之一：**越界访问**。这个问题跨越了内存管理的各个部分——堆、栈和全局变量，每个部分都提出了独特的挑战和风险。

#### 堆中的越界访问

我们首先探讨堆上的越界访问，动态内存分配可能导致指针超出分配的内存边界。考虑以下示例：

```cpp
int main() {
    int *heapArray = new int[5];
    heapArray[5]   = 10; // Out-of-bounds write on the heap
    delete[] heapArray;
    return 0;
}
```

此代码片段演示了越界写入，尝试访问超出分配范围的索引，导致未定义行为和潜在的内存损坏。

如果我们启用 ASan 运行此代码，我们得到以下输出：

```cpp
make && ./a.out
=================================================================
==3102850==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x603000000054 at pc 0x55af5525f222 bp 0x7ffde596fb60 sp 0x7ffde596fb50
WRITE of size 4 at 0x603000000054 thread T0
    #0 0x55af5525f221 in main /home/user/clang-sanitizers/main.cpp:3
    #1 0x7f1ad0a29d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
    #2 0x7f1ad0a29e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
    #3 0x55af5525f104 in _start (/home/user/clang-sanitizers/build/a.out+0x1104)
0x603000000054 is located 0 bytes to the right of 20-byte region 0x603000000040,0x603000000054)
allocated by thread T0 here:
    #0 0x7f1ad12b6357 in operator new[ ../../../../src/libsanitizer/asan/asan_new_delete.cpp:102
    #1 0x55af5525f1de in main /home/user/clang-sanitizers/main.cpp:2
    #2 0x7f1ad0a29d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
SUMMARY: AddressSanitizer: heap-buffer-overflow /home/user/clang-sanitizers/main.cpp:3 in main
Shadow bytes around the buggy address:
  0x0c067fff7fb0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x0c067fff7fc0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x0c067fff7fd0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x0c067fff7fe0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x0c067fff7ff0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x0c067fff8000: fa fa 00 00 00 fa fa fa 00 00[04]fa fa fa fa fa
  0x0c067fff8010: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c067fff8020: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c067fff8030: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c067fff8040: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c067fff8050: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==3102850==ABORTING
```

如您所见，报告包括详细的堆栈跟踪，突出显示源代码中错误的精确位置。这些信息对于调试和修复问题非常有价值。

#### 栈上的越界访问

接下来，我们关注栈。在这里，由于索引错误或不正确的缓冲区溢出，越界访问通常与局部变量有关。例如：

```cpp
int main() {
    int stackArray[5];
    stackArray[5] = 10; // Out-of-bounds write on the stack
    return 0;
}
```

在这种情况下，访问 `stackArray[5]` 超出了范围，因为有效的索引是从 `0` 到 `4`。此类错误可能导致崩溃或可利用的漏洞。ASan 对此示例的输出与上一个示例非常相似：

```cpp
==3190568==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7ffd166961e4 at pc 0x55b4cd113295 bp 0x7ffd166961a0 sp 0x7ffd16696190
WRITE of size 4 at 0x7ffd166961e4 thread T0
    #0 0x55b4cd113294 in main /home/user/clang-sanitizers/main.cpp:3
    #1 0x7f90fc829d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
    #2 0x7f90fc829e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
    #3 0x55b4cd113104 in _start (/home/user/clang-sanitizers/build/a.out+0x1104)
Address 0x7ffd166961e4 is located in stack of thread T0 at offset 52 in frame
    #0 0x55b4cd1131d8 in main /home/user/clang-sanitizers/main.cpp:1
  This frame has 1 object(s):
    [32, 52) ‘stackArray’ (line 2) <== Memory access at offset 52 overflows this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-buffer-overflow /home/user/clang-sanitizers/main.cpp:3 in main
Shadow bytes around the buggy address:
  0x100022ccabe0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccabf0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac00: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac10: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac20: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x100022ccac30: 00 00 00 00 00 00 f1 f1 f1 f1 00 00[04]f3 f3 f3
  0x100022ccac40: f3 f3 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac50: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac60: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac70: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100022ccac80: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==3190568==ABORTING
```

#### 全局变量的越界访问

最后，我们检查全局变量。当访问超出其定义边界时，它们也容易受到类似的风险。例如：

```cpp
int globalArray[5];
int main() {
    globalArray[5] = 10;  // Out-of-bounds access to a global array
    return 0;
}
```

在这里，尝试写入 `globalArray[5]` 的操作是一个越界操作，导致未定义行为。由于 ASan 的输出与之前的示例相似，我们在此不包括它。

### 解决 C++ 中的 Use-After-Free 漏洞

在下一节中，我们将解决 C++ 编程中的一个关键且经常具有挑战性的问题：**Use-After-Free 漏洞**。此类错误发生在程序在释放内存后继续使用该内存位置时，导致未定义行为、程序崩溃、安全漏洞和数据损坏。我们将从各种上下文中探讨此问题，提供有关其识别和预防的见解。

#### 动态内存（堆）中的 Use-After-Free

Use-After-Free 错误最常见的情况是在堆上的动态分配内存。考虑以下示例：

```cpp
#include <iostream>
template <typename T>
struct Node {
    T data;
    Node *next;
    Node(T val) : data(val), next(nullptr) {}
};
int main() {
    auto *head = new Node(1);
    auto *temp = head;
    head       = head->next;
    delete temp;
    std::cout << temp->data; // Use-after-free in a linked list
    return 0;
}
```

在此片段中，`ptr` 所指向的内存在使用 `delete` 释放后进行访问。这种访问可能导致不可预测的行为，因为已释放的内存可能被分配用于其他目的或被系统修改。

#### 使用对象引用的 Use-After-Free

Use-After-Free 也可以在面向对象编程中发生，尤其是在处理已销毁的对象的引用或指针时。例如：

```cpp
class Example {
public:
    int value;
    Example() : value(0) {}
};
Example* obj = new Example();
Example& ref = *obj;
delete obj;
std::cout << ref.value;  // Use-after-free through a reference
```

在这里，`ref` 指向一个已被删除的对象，并且在删除后对 `ref` 的任何操作都会导致 Use-After-Free。

#### 在复杂数据结构中的 Use-After-Free

复杂数据结构，如链表或树，也容易发生 Use-After-Free 错误，尤其是在删除或重构操作期间。例如：

```cpp
struct Node {
    int data;
    Node* next;
    Node(int val) : data(val), next(nullptr) {}
};
Node* head = new Node(1);
Node* temp = head;
head = head->next;
delete temp;
std::cout << temp->data;  // Use-after-free in a linked list
```

在这种情况下，`temp` 在释放后被使用，这可能导致严重问题，尤其是如果列表很大或属于关键系统组件的一部分。

ASan 可以帮助检测 C++ 程序中的使用后释放错误。例如，如果我们启用 ASan 运行前面的示例，我们得到以下输出：

```cpp
make && ./a.out
Consolidate compiler generated dependencies of target a.out
[100%] Built target a.out
=================================================================
==3448347==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010 at pc 0x55fbcc2ca3b2 bp 0x7fff2f3af7a0 sp 0x7fff2f3af790
READ of size 4 at 0x602000000010 thread T0
    #0 0x55fbcc2ca3b1 in main /home/user/clang-sanitizers/main.cpp:15
    #1 0x7efdb6429d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
    #2 0x7efdb6429e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
    #3 0x55fbcc2ca244 in _start (/home/user/clang-sanitizers/build/a.out+0x1244)
```

#### ASan 中的使用后返回检测

**使用后返回** 是 C++ 编程中的一种内存错误，其中函数返回一个指向局部（栈分配）变量的指针或引用。这个局部变量一旦函数返回就不再存在，通过返回的指针或引用的任何后续访问都是无效且危险的。这可能导致未定义行为和潜在的安全漏洞。

ASan 提供了一种检测使用后返回错误的机制。它可以在编译时使用 `-fsanitize-address-use-after-return` 标志进行控制，并在运行时使用 `ASAN_OPTIONS` 环境变量。

以下描述了使用后返回检测的配置：

+   `-fsanitize-address-use-after-return=(never|runtime|always)`

    该标志接受三个设置：

    +   `never`: 这禁用使用后返回检测

    +   `runtime`: 这启用检测，但可以在运行时被覆盖（默认设置）

    +   `always`: 这始终启用检测，不受运行时设置的影响

+   `ASAN_OPTIONS` 环境变量：

    +   `ASAN_OPTIONS=detect_stack_use_after_return=1`

    +   `ASAN_OPTIONS=detect_stack_use_after_return=0`

    +   在 Linux 上，默认启用检测

这里是其使用的一个示例：

1.  **启用** **使用后返回检测编译**：

    ```cpp
    clang++ -fsanitize=address -fsanitize-address-use-after-return=always -g -o your_program your_file.cpp
    ```

    此命令使用 ASan 编译 `your_file.cpp` 并显式启用使用后返回检测。

1.  **启用/禁用检测运行**：

    +   要在启用使用后返回检测的情况下运行程序（在默认情况下不是的平台）：

        ```cpp
        ASAN_OPTIONS=detect_stack_use_after_return=1 ./your_program
        ```

    +   要禁用检测，即使它在编译时已启用：

        ```cpp
        ASAN_OPTIONS=detect_stack_use_after_return=0 ./your_program
        ```

**示例代码** **演示使用后返回**

提供的 C++ 代码示例演示了使用后返回场景，这是一种由于从函数返回局部变量的引用而引起的未定义行为。让我们分析这个示例并了解其影响：

```cpp
#include <iostream>
const std::string &get_binary_name() {
    const std::string name = “main”;
    return name; // Returning address of a local variable
}
int main() {
    const auto &name = get_binary_name();
    // Use after return: accessing memory through name is undefined behavior
    std::cout << name << std::endl;
    return 0;
}
```

在给定的代码示例中，`get_binary_name` 函数被设计为创建一个名为 `name` 的局部 `std::string` 对象，并返回对其的引用。关键问题源于 `name` 是一个局部变量，它在函数作用域结束时就会被销毁。因此，`get_binary_name` 返回的引用在函数退出时立即变得无效。

在 `main` 函数中，现在存储在 `name` 中的返回引用被用来访问字符串值。然而，由于 `name` 指向一个已经被销毁的局部变量，以这种方式使用它会导致未定义行为。这是一个使用后返回错误的经典例子，其中程序试图访问不再有效的内存。

函数的预期功能似乎是要返回程序名称。然而，为了正确工作，`name` 变量应该具有静态或全局生命周期，而不是被限制在 `get_binary_name` 函数中的局部变量。这将确保返回的引用在函数作用域之外仍然有效，避免使用后返回错误。

现代编译器配备了发出关于潜在问题代码模式的警告的能力，例如返回对局部变量的引用。在我们的示例中，编译器可能会将返回局部变量引用标记为警告，表明可能存在使用后返回错误。

然而，为了有效地展示 ASan 捕获使用后返回错误的能力，有时需要绕过这些编译时警告。这可以通过显式禁用编译器的警告来实现。例如，通过在编译命令中添加 `-Wno-return-local-addr` 标志，我们可以防止编译器发出关于返回局部地址的警告。这样做使我们能够将重点从编译时检测转移到运行时检测，在那里 ASan 在识别使用后返回错误方面的能力可以更加突出和测试。这种方法强调了 ASan 的运行时诊断优势，尤其是在编译时分析可能不足的情况下。

**使用 ASan 编译**

要使用 ASan 的使用后返回检测编译此程序，您可以使用以下命令：

```cpp
clang++ -fsanitize=address -Wno-return-local-addr -g your_file.cpp -o your_program
```

此命令在启用 ASan 的同时抑制了关于返回局部变量地址的特定编译器警告。运行编译后的程序将允许 ASan 在运行时检测并报告使用后返回错误：

```cpp
Consolidate compiler generated dependencies of target a.out
[100%] Built target a.out
AddressSanitizer:DEADLYSIGNAL
=================================================================
==4104819==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000008 (pc 0x7f74e354f4c4 bp 0x7ffefcd298e0 sp 0x7ffefcd298c8 T0)
==4104819==The signal is caused by a READ memory access.
==4104819==Hint: address points to the zero page.
    #0 0x7f74e354f4c4 in std::basic_ostream<char, std::char_traits<char> >& std::operator<< <char, std::char_traits<char>, std::allocator<char> >(std::basic_ostream<char, std::char_traits<char> >&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) (/lib/x86_64-linux-gnu/libstdc++.so.6+0x14f4c4)
    #1 0x559799ab4785 in main /home/user/clang-sanitizers/main.cpp:11
    #2 0x7f74e3029d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
    #3 0x7f74e3029e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
    #4 0x559799ab4504 in _start (/home/user/clang-sanitizers/build/a.out+0x2504)
AddressSanitizer can not provide additional info.
SUMMARY: AddressSanitizer: SEGV (/lib/x86_64-linux-gnu/libstdc++.so.6+0x14f4c4) in std::basic_ostream<char, std::char_traits<char> >& std::operator<< <char, std::char_traits<char>, std::allocator<char> >(std::basic_ostream<char, std::char_traits<char> >&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)
==4104819==ABORTING
```

这个例子强调了理解 C++ 中对象生命周期的重要性以及误用可能导致未定义行为。虽然编译器警告在编译时捕获此类问题很有价值，但像 ASan 这样的工具提供了额外的运行时错误检测层，这在复杂场景中尤其有用，在这些场景中，编译时分析可能不足以满足需求。

#### 使用后返回检测

C++ 中使用后作用域的概念涉及在作用域结束后访问变量，导致未定义行为。这种类型的错误很微妙，可能特别难以检测和调试。ASan 提供了一种检测使用后作用域错误的功能，可以使用 `-fsanitize-address-use-after-scope` 编译标志来启用。

**理解使用后作用域**

使用后作用域发生在程序继续使用超出作用域的变量的指针或引用时。与使用后返回不同，后者的问题在于函数局部变量，使用后作用域可以发生在任何作用域内，例如在代码块中，如 `if` 语句或循环中。

当一个变量超出作用域时，其内存位置可能仍然保留旧数据一段时间，但这个内存随时可能被覆盖。访问这个内存是未定义行为，可能导致程序行为异常或崩溃。

**配置 ASan 以进行** **超出作用域检测**

`-fsanitize-address-use-after-scope`:

+   将此标志添加到您的编译命令中，指示 ASan 对代码进行操作以检测超出作用域错误。

+   重要的是要注意，这种检测默认是未启用的，必须显式启用。

**示例代码** **展示超出作用域**

提供的代码片段展示了 C++中超出作用域错误的经典案例。让我们分析代码并了解问题：

```cpp
int* create_array(bool condition) {
  int *p;
  if (condition) {
    int x[10];
    p = x;
  }
  *p = 1;
}
```

在给定的代码片段中，我们首先声明了一个未初始化的`p`指针。然后函数进入一个条件作用域，如果`condition`为真，则在栈上创建一个`x[10]`数组。在这个作用域内，`p`指针被分配为指向这个数组的起始位置，实际上使`p`指向`x`。

关键问题出现在条件块退出之后。此时，作为`if`块局部变量的`x`数组超出作用域，不再有效。然而，`p`指针仍然持有`x`之前所在位置的地址。当代码尝试使用`*p = 1;`写入这个内存位置时，它试图访问当前作用域内不再有效的`x`数组内存。这种行为导致了一个超出作用域错误，其中`p`被解引用以访问当前作用域内不再有效的内存。这种错误是超出作用域的经典例子，突出了通过指向超出作用域变量的指针访问内存的危险性。

通过指向超出作用域变量的指针访问内存，如提供的代码片段所示，会导致未定义行为。这是因为一旦`x`变量超出作用域，`p`指针指向的内存位置就变得不确定。由此场景产生的未定义行为存在几个问题。

首先，这对程序的安全性和稳定性构成了重大风险。行为的未定义性质意味着程序可能会崩溃或行为不可预测。程序执行中的这种不稳定性可能产生深远的影响，尤其是在可靠性至关重要的应用中。此外，如果`x`之前占用的内存位置被程序的其它部分覆盖，可能会潜在地导致安全漏洞。这些漏洞可能被利用来损害程序或其运行的系统。

总结来说，通过指向超出作用域变量的指针访问内存导致的未定义行为在软件开发中是一个严重的问题，需要仔细管理变量作用域和内存访问模式，以确保程序的安全性和稳定性。

要编译启用 ASan 使用范围之外检测的程序，你可以使用以下命令之一：

```cpp
g++ -fsanitize=address -fsanitize-address-use-after-scope -g your_file.cpp -o your_program
```

使用这些设置运行编译后的程序可以启用 ASan 在运行时检测并报告使用范围之外的错误。

由于它们依赖于程序的运行时状态和内存布局，使用范围之外的错误可能难以察觉和追踪。通过在 ASan 中启用使用范围之外的检测，开发者可以获得一个宝贵的工具，用于识别这些错误，从而创建更健壮和可靠的 C++ 应用程序。理解和预防此类问题是编写安全且正确 C++ 代码的关键。

#### 双重释放和无效释放检查在 ASan 中

ASan 是 LLVM 项目的一部分，提供了强大的机制来检测和诊断 C++ 程序中的两种关键类型的内存错误：双重释放和无效释放。这些错误不仅常见于复杂的 C++ 应用程序，还可能导致严重的程序崩溃、未定义行为和安全漏洞。

**理解双重释放和** **无效释放**

理解双重释放和无效释放错误对于有效地管理 C++ 程序中的内存至关重要。

当尝试使用 `delete` 或 `delete[]` 操作符多次释放内存块时，就会发生双重释放错误。这种情况通常发生在相同的内存分配被两次传递给 `delete` 或 `delete[]` 的情况下。第一次调用 `delete` 释放了内存，但第二次调用试图释放已经释放的内存。这可能导致堆损坏，因为程序可能会随后修改或重新分配已释放的内存以供其他用途。双重释放错误可能导致程序出现不可预测的行为，包括崩溃和数据损坏。

另一方面，当使用 `delete` 或 `delete[]` 操作一个未使用 `new` 或 `new[]` 分配，或已经释放的指针时，就会发生无效释放错误。这一类别包括尝试释放空指针、指向栈内存（这些不是动态分配的）或指向未初始化内存的指针。像双重释放错误一样，无效释放也可能导致堆损坏和不可预测的程序行为。它们尤其危险，因为它们可能会破坏 C++ 运行时的内存管理结构，导致微妙且难以诊断的错误。

这两种错误都源于对动态内存的不当处理，强调了遵守内存管理最佳实践的重要性，例如确保每个 `new` 都有一个相应的 `delete`，并在释放指针后避免其重用。

此列表概述了 ASan 检测机制的功能：

+   执行 `delete` 操作时，ASan 会检查该指针是否对应一个有效、先前分配且尚未释放的内存块。

+   **错误报告**：如果检测到双重释放或无效释放错误，ASan 会终止程序的执行并提供详细的错误报告。此报告包括错误发生的代码位置、涉及的内存地址以及该内存的分配历史（如果可用）。

这里有一些示例代码演示双重释放错误：

```cpp
int main() {
    int* ptr = new int(10);
    delete ptr;
    delete ptr;  // Double-free error
    return 0;
}
```

ASan 会报告以下错误：

```cpp
make && ./a.out
Consolidate compiler generated dependencies of target a.out
[ 50%] Building CXX object CMakeFiles/a.out.dir/main.cpp.o
[100%] Linking CXX executable a.out
[100%] Built target a.out
=================================================================
==765374==ERROR: AddressSanitizer: attempting double-free on 0x602000000010 in thread T0:
    #0 0x7f7ff5eb724f in operator delete(void*, unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:172
    #1 0x55839eca830b in main /home/user/clang-sanitizers/main.cpp:6
    #2 0x7f7ff5629d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
    #3 0x7f7ff5629e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
    #4 0x55839eca81c4 in _start (/home/user/clang-sanitizers/build/a.out+0x11c4)
0x602000000010 is located 0 bytes inside of 4-byte region [0x602000000010,0x602000000014)
freed by thread T0 here:
    #0 0x7f7ff5eb724f in operator delete(void*, unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:172
    #1 0x55839eca82f5 in main /home/user/clang-sanitizers/main.cpp:5
    #2 0x7f7ff5629d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
previously allocated by thread T0 here:
    #0 0x7f7ff5eb61e7 in operator new(unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:99
    #1 0x55839eca829e in main /home/user/clang-sanitizers/main.cpp:4
    #2 0x7f7ff5629d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
SUMMARY: AddressSanitizer: double-free ../../../../src/libsanitizer/asan/asan_new_delete.cpp:172 in operator delete(void*, unsigned long)
==765374==ABORTING
```

在这个例子中，`ptr` 所指向的同一内存被释放了两次，导致双重释放错误。

**演示无效释放的示例代码**

提供的代码片段演示了一个无效释放错误，这是一种可能在 C++ 编程中发生的内存管理错误。让我们分析这个例子，了解问题及其影响：

```cpp
int main() {
    int local_var = 42;
    int* ptr = &local_var;
    delete ptr;  // Invalid free error
    return 0;
}
```

在给定的代码段中，我们首先声明并初始化一个局部变量 `int local_var = 42;`。这创建了一个名为 `local_var` 的栈分配整数变量。随后，使用 `int* ptr = &local_var;` 进行指针赋值，其中 `ptr` 指针被设置为指向 `local_var` 的地址。这建立了指针和栈分配变量之间的联系。

然而，后续的操作出现了问题：`delete ptr;`。此行代码试图释放 `ptr` 所指向的内存。问题在于 `ptr` 指向的是栈分配的变量 `local_var`，而不是堆上的动态分配内存。在 C++ 中，`delete` 操作符仅应与使用 `new` 分配的指针一起使用。由于 `local_var` 没有使用 `new` 分配（它是一个栈分配的变量），在 `ptr` 上使用 `delete` 是无效的，并导致未定义的行为。在非堆指针上滥用 `delete` 操作符是 C++ 程序中可能导致严重运行时错误的常见错误。

这里有一些现代编译器的警告：

+   现代 C++ 编译器通常会在使用 `delete` 操作符对不指向动态分配内存的指针进行操作时发出警告或错误，因为这通常是错误的一个常见来源。

+   为了在不修改代码的情况下编译此代码并展示 ASan 捕获此类错误的能力，你可能需要抑制编译器警告。这可以通过在编译命令中添加 `-Wno-free-nonheap-object` 标志来实现。

**使用 ASan 编译以进行** **无效释放检测**

要使用 ASan 编译程序以检测无效释放操作，请使用以下命令：

```cpp
clang++ -fsanitize=address -Wno-free-nonheap-object -g your_file.cpp -o your_program
```

此命令以启用 ASan 并抑制关于释放非堆对象的特定编译器警告来编译程序。当你运行编译后的程序时，ASan 将检测并报告无效释放操作：

```cpp
=================================================================
==900629==ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed: 0x7fff390f21d0 in thread T0
    #0 0x7f30b82b724f in operator delete(void*, unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:172
    #1 0x563f21cd72c7 in main /home/user/clang-sanitizers/main.cpp:4
    #2 0x7f30b7a29d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
    #3 0x7f30b7a29e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
    #4 0x563f21cd7124 in _start (/home/user/clang-sanitizers/build/a.out+0x1124)
Address 0x7fff390f21d0 is located in stack of thread T0 at offset 32 in frame
    #0 0x563f21cd71f8 in main /home/user/clang-sanitizers/main.cpp:1
  This frame has 1 object(s):
    [32, 36) ‘local_var’ (line 2) <== Memory access at offset 32 is inside this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: bad-free ../../../../src/libsanitizer/asan/asan_new_delete.cpp:172 in operator delete(void*, unsigned long)
==900629==ABORTING
```

如示例所示，尝试删除指向非堆对象的指针是 C++ 中内存管理操作的误用。这种做法可能导致未定义的行为，并可能引发崩溃或其他异常程序行为。ASan 作为一种宝贵的工具，在检测这类错误方面发挥了重要作用，对开发健壮且无错误的 C++ 应用程序做出了重大贡献。

#### **调整 ASan 以增强控制**

虽然 ASan 是检测 C++ 程序中内存错误的强大工具，但在某些情况下，其行为需要微调。这种微调对于有效地管理分析过程至关重要，尤其是在处理涉及外部库、遗留代码或特定代码模式的复杂项目时。

**抑制来自外部库的警告**

在许多项目的背景下，使用外部库是一种常见的做法。然而，这些你可能无法控制的库有时可能包含内存问题。当运行 ASan 等工具时，这些外部库中的问题可能会被标记出来，导致诊断信息中充满了与你的项目代码不直接相关的警告。这可能会成为问题，因为它可能会掩盖你自己的代码库中需要关注的真正问题。

为了减轻这种情况，ASan 提供了一个有用的功能，允许你抑制来自这些外部库的特定警告。这种过滤掉无关警告的能力对于专注于修复自己代码库范围内的问题非常有价值。该功能的实现通常涉及使用 sanitizer 特殊情况列表或在编译过程中指定某些链接器标志。这些机制提供了一种方法，告诉 ASan 忽略某些路径或模式在诊断中的信息，从而有效地减少外部来源的噪音，并有助于更精确和高效的调试过程。

**条件编译**

在软件开发中，有些情况下你可能希望在编译程序时仅包含与 ASan 相关的特定代码段。这种方法在多种用途上特别有用，例如，整合额外的诊断信息或修改内存分配以使其更兼容或友好地与 ASan 的操作。

要实现这种策略，你可以利用条件编译技术，根据特定条件包含或排除代码的一部分。在 ASan 的情况下，你可以使用 `__has_feature` 宏来检查其是否存在。这个宏在编译时评估当前编译上下文中是否存在特定的功能（在这种情况下，即 ASan）。如果正在使用 ASan，条件编译块中的代码将被包含在最终的可执行文件中；否则，它将被排除：

```cpp
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
// Do something specific for AddressSanitizer
#  endif
#endif
```

这种条件编译的方法允许开发者针对 ASan 使用的情况专门调整他们的代码，从而提高清理器的有效性，并可能避免仅在 ASan 存在时出现的问题。它提供了一种灵活的方式来根据构建配置调整程序的行为，这在开发、测试和生产阶段使用不同配置的复杂开发环境中非常有价值。

**禁用特定代码行的清理器**

在开发复杂软件的过程中，有时可能会故意执行某些操作，即使它们可能会被 ASan 标记为错误。或者，你可能希望出于特定原因将代码库的某些部分排除在 ASan 的分析之外。这可能是由于你的代码中已知的好行为，ASan 可能会错误地将其解释为错误，或者由于 ASan 引入的开销不希望出现在某些代码部分。

为了应对这些场景，GCC 和 Clang 编译器都提供了一种方法，可以针对特定的函数或代码块选择性地禁用 ASan。这是通过使用 `__attribute__((no_sanitize(“address”)))` 属性来实现的。通过将此属性应用于函数或特定的代码块，你可以指示编译器省略该特定段落的 ASan 仪器设置。

这个特性特别有用，因为它允许对代码的哪些部分受到 ASan 的审查进行细粒度控制。它使开发者能够微调彻底的错误检测与代码行为或性能要求的实际现实之间的平衡。通过审慎地应用此属性，你可以确保 ASan 的分析既有效又高效，将精力集中在最有益的地方。

**利用清理器特殊案例列表**

+   **源文件和函数（src 和 fun）**：ASan 允许你在指定的源文件或函数中抑制错误报告。这在你想忽略某些已知问题或第三方代码时特别有用。

+   **全局变量和类型（global 和 type）**：此外，ASan 引入了抑制对具有特定名称和类型的全局变量越界访问错误的能力。这个特性对于全局变量和类/结构体类型特别有用，允许更精确的错误抑制。

**清理器特殊案例列表的示例条目**

微调 ASan 是将其集成到大型、复杂开发环境中的关键方面。它允许开发者根据项目的具体需求自定义 ASan 的行为，无论是通过排除外部库、为 ASan 构建条件化代码，还是忽略某些错误以关注更关键的问题。通过有效利用这些微调能力，团队可以利用 ASan 的全部力量，确保 C++应用程序的稳健和可靠。抑制规则可以按照以下方式设置在文本文件中：

```cpp
fun:FunctionName  # Suppresses errors from FunctionName
global:GlobalVarName  # Suppresses out-of-bound errors on GlobalVarName
type:TypeName  # Suppresses errors for TypeName objects
```

此文件可以通过`ASAN_OPTIONS`环境变量传递给运行时，例如`ASAN_OPTIONS=suppressions=path/to/suppressionfile`。

#### ASan 的性能开销

在检测内存管理问题，如无效的释放操作时，ASan 的使用对于识别和解决 C++应用程序中潜在的 bug 非常有益。然而，重要的是要意识到使用 ASan 的性能影响。

**性能影响、限制和推荐**

将 ASan 集成到开发和测试过程中会带来一定程度的性能开销。通常，ASan 引入的减速在 2 倍左右，这意味着经过 ASan 工具化的程序可能比其非工具化版本慢大约两倍。这种增加的执行时间主要是由于 ASan 执行的额外检查和监控，以细致地检测内存错误。每次内存访问，以及每次内存分配和释放操作，都受到这些检查的影响，不可避免地导致额外的 CPU 周期消耗。

由于这种性能影响，ASan 主要在软件生命周期的开发和测试阶段使用。这种使用模式代表了一种权衡：虽然使用 ASan 会有性能成本，但在开发早期阶段捕捉和修复关键内存相关错误的好处是显著的。早期发现这些问题有助于保持代码质量，并且可以显著减少在生命周期后期调试和修复 bug 所需的时间和资源。

然而，在生产环境中部署经过 ASan 工具化的二进制文件通常不推荐，尤其是在性能是关键因素的场景中。ASan 引入的开销可能会影响应用程序的响应性和效率。尽管如此，在某些情况下，尤其是在可靠性和安全性至关重要的应用程序中，并且性能考虑是次要的，为了彻底测试，在类似生产环境中使用 ASan 可能是合理的。在这种情况下，ASan 提供的额外稳定性和安全性保障可能超过性能下降的担忧。

ASan 支持以下平台：

+   Linux i386/x86_64（在 Ubuntu 12.04 上测试）

+   macOS 10.7 – 10.11 (i386/x86_64)

+   iOS Simulator

+   Android ARM

+   NetBSD i386/x86_64

+   FreeBSD i386/x86_64（在 FreeBSD 11-current 上测试）

+   Windows 8.1+ (i386/x86_64)

## LeakSanitizer (LSan)

**LSan**是 ASan 套件的一部分的专用内存泄漏检测工具，也可以独立使用。它专门设计用于识别 C++程序中的内存泄漏——即未释放分配的内存，导致内存消耗随时间增加。

### 与 ASan 的集成

LSan 通常与 ASan 一起使用。当你在你的构建中启用 ASan 时，LSan 也会自动启用，为内存错误和泄漏提供全面的分析。

### 独立模式

如果你希望在不使用 ASan 的情况下使用 LSan，可以通过编译程序时加上`-fsanitize=leak`标志来启用它。这在只想专注于内存泄漏检测而不想承受其他地址清理开销时特别有用。

### 内存泄漏检测示例

考虑以下具有内存泄漏的 C++代码：

```cpp
int main() {
    int* leaky_memory = new int[100]; // Memory allocated and never freed
    leaky_memory      = nullptr;      // Memory leaked
    (void)leaky_memory;
    return 0;
}
```

在这个例子中，一个整数数组被动态分配但没有释放，导致内存泄漏。

当你使用 LSan 编译并运行此代码时，输出可能看起来像这样：

```cpp
=================================================================
==1743181==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 400 byte(s) in 1 object(s) allocated from:
    #0 0x7fa14b6b6357 in operator new[](unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:102
    #1 0x55888aabd19e in main /home/user/clang-sanitizers/main.cpp:2
    #2 0x7fa14ae29d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
SUMMARY: AddressSanitizer: 400 byte(s) leaked in 1 allocation(s).
```

这个输出指出了内存泄漏的位置和大小，有助于快速有效地进行调试。

### 平台支持

根据最新信息，LSan 支持 Linux、macOS 和 Android。支持可能根据工具链和使用的编译器版本而有所不同。

LSan 是 C++开发者识别和解决应用程序中内存泄漏的有价值工具。它能够与 ASan 一起使用，也可以独立使用，这为解决特定的内存相关问题提供了灵活性。通过将 LSan 集成到开发和测试过程中，开发者可以确保更有效的内存使用和整体的应用程序稳定性。

## MemorySanitizer (MSan)

**MSan**是一个动态分析工具，是 LLVM 项目的一部分，旨在检测 C++程序中未初始化内存的使用。未初始化内存的使用是导致不可预测行为、安全漏洞和难以诊断的错误等常见错误的原因。

要使用 MSan，请使用`-fsanitize=memory`标志编译你的程序。这指示编译器在代码中插入检查未初始化内存使用的检查。例如：

```cpp
clang++ -fsanitize=memory -g -o your_program your_file.cpp
```

### 展示未初始化内存使用的示例代码

考虑以下简单的 C++示例：

```cpp
#include <iostream>
int main() {
    int* ptr = new int[10];
    if (ptr[1]) {
        std::cout << “xx\n”;
    }
    delete[] ptr;
    return 0;
}
```

在此代码中，整数是在堆上分配的但未初始化。

当与 MSan 一起编译和运行时，输出可能看起来像这样：

```cpp
==48607==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x560a37e0f557 in main /home/user/clang-sanitizers/main.cpp:5:9
    #1 0x7fa118029d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f) (BuildId: c289da5071a3399de893d2af81d6a30c62646e1e)
    #2 0x7fa118029e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f) (BuildId: c289da5071a3399de893d2af81d6a30c62646e1e)
    #3 0x560a37d87354 in _start (/home/user/clang-sanitizers/build/a.out+0x1e354) (BuildId: 5a727e2c09217ae0a9d72b8a7ec767ce03f4e6ce)
SUMMARY: MemorySanitizer: use-of-uninitialized-value /home/user/clang-sanitizers/main.cpp:5:9 in main
```

MSan 检测到未初始化变量的使用，并指向代码中发生此情况的精确位置。

在这种情况下，修复可能只需要初始化数组：

```cpp
    int* ptr = new int[10]{};
```

### 微调、性能影响和限制

+   **微调**：MSan 的微调选项与 ASan 类似。用户可以参考官方文档以获取详细的定制选项。

+   **性能影响**：通常，使用 MSan 会引入大约 3 倍的运行时减速。这种开销是由于 MSan 执行的额外检查，以检测未初始化内存的使用。

+   **支持的平台**：MSan 支持 Linux、NetBSD 和 FreeBSD。它在检测未初始化内存使用方面的有效性使其成为在这些平台上工作的开发人员的强大工具。

+   **局限性**：与其他清理器一样，MSan 的运行时开销使其最适合用于测试环境，而不是生产环境。此外，MSan 要求整个程序（包括它使用的所有库）都进行仪器化。在无法获得某些库的源代码的情况下，这可能是一个限制。

MSan 是检测 C++ 程序中难以捉摸但可能至关重要的未初始化内存使用问题的基本工具。通过提供关于此类问题发生位置和方式的详细报告，MSan 使开发人员能够识别和修复这些错误，显著提高其应用程序的可靠性和安全性。尽管其性能影响，但将 MSan 集成到开发和测试阶段，是一个谨慎的步骤，以确保稳健的软件质量。

## TSan

在 C++ 编程领域，有效地管理并发和多线程既是关键也是挑战。与线程相关的问题，尤其是数据竞争，臭名昭著地难以检测和调试。与其他可以通过确定性测试方法（如单元测试）发现的错误不同，线程问题本质上是难以捉摸和非确定性的。这些问题可能不会在程序的每次运行中都表现出来，导致不可预测和混乱的行为，这可能非常难以复制和诊断。

### 线程相关问题的复杂性

+   **非确定性行为**：包括数据竞争、死锁和线程泄漏在内的并发问题本质上是非确定性的。这意味着它们在相同条件下并不一致地重现，使它们难以捉摸和不可预测。

+   **检测挑战**：传统的测试方法，包括全面的单元测试，往往无法检测到这些问题。涉及并发的测试结果可能会因时间、线程调度和系统负载等因素而有所不同。

+   **微妙且严重的错误**：与线程相关的错误可能处于休眠状态，仅在特定条件下在生产环境中出现，可能导致严重的影响，如数据损坏、性能下降和系统崩溃。

### TSan 的必要性

由于 C++ 中管理并发的固有挑战，Clang 和 GCC 提供的 TSan 等工具变得至关重要。TSan 是一个旨在检测线程问题的复杂工具，特别关注数据竞争。

### 启用 TSan

+   `-fsanitize=thread` 标志。这指示 Clang 和 GCC 为运行时检测线程问题对你的代码进行仪器化。

+   **编译示例**：

    ```cpp
    clang++ -fsanitize=thread -g -o your_program your_file.cpp
    ```

此命令将使用 TSan 启用编译`your_file.cpp`，以便检测和报告线程问题。请注意，无法同时开启线程和 ASan。

### C++中的数据竞争示例

考虑这个简单但具有说明性的例子：

```cpp
#include <iostream>
#include <thread>
int shared_counter = 0;
void increment_counter() {
    for (int i = 0; i < 10000; ++i) {
        shared_counter++; // Potential data race
    }
}
int main() {
    std::thread t1(increment_counter);
    std::thread t2(increment_counter);
    t1.join();
    t2.join();
    std::cout << “Shared counter: “ << shared_counter << std::endl;
    return 0;
}
```

在这里，两个线程在不进行同步的情况下修改相同的共享资源，导致数据竞争。

如果我们启用 TSan 并构建和运行此代码，我们将得到以下输出：

```cpp
==================
WARNING: ThreadSanitizer: data race (pid=2560038)
  Read of size 4 at 0x555fd304f154 by thread T2:
    #0 increment_counter() /home/user/clang-sanitizers/main.cpp:8 (a.out+0x13f9)
    #1 void std::__invoke_impl<void, void (*)()>(std::__invoke_other, void (*&&)()) /usr/include/c++/11/bits/invoke.h:61 (a.out+0x228a)
    #2 std::__invoke_result<void (*)()>::type std::__invoke<void (*)()>(void (*&&)()) /usr/include/c++/11/bits/invoke.h:96 (a.out+0x21df)
    #3 void std::thread::_Invoker<std::tuple<void (*)()> >::_M_invoke<0ul>(std::_Index_tuple<0ul>) /usr/include/c++/11/bits/std_thread.h:259 (a.out+0x2134)
    #4 std::thread::_Invoker<std::tuple<void (*)()> >::operator()() /usr/include/c++/11/bits/std_thread.h:266 (a.out+0x20d6)
    #5 std::thread::_State_impl<std::thread::_Invoker<std::tuple<void (*)()> > >::_M_run() /usr/include/c++/11/bits/std_thread.h:211 (a.out+0x2088)
    #6 <null> <null> (libstdc++.so.6+0xdc252)
  Previous write of size 4 at 0x555fd304f154 by thread T1:
    #0 increment_counter() /home/user/clang-sanitizers/main.cpp:8 (a.out+0x1411)
    #1 void std::__invoke_impl<void, void (*)()>(std::__invoke_other, void (*&&)()) /usr/include/c++/11/bits/invoke.h:61 (a.out+0x228a)
    #2 std::__invoke_result<void (*)()>::type std::__invoke<void (*)()>(void (*&&)()) /usr/include/c++/11/bits/invoke.h:96 (a.out+0x21df)
    #3 void std::thread::_Invoker<std::tuple<void (*)()> >::_M_invoke<0ul>(std::_Index_tuple<0ul>) /usr/include/c++/11/bits/std_thread.h:259 (a.out+0x2134)
    #4 std::thread::_Invoker<std::tuple<void (*)()> >::operator()() /usr/include/c++/11/bits/std_thread.h:266 (a.out+0x20d6)
    #5 std::thread::_State_impl<std::thread::_Invoker<std::tuple<void (*)()> > >::_M_run() /usr/include/c++/11/bits/std_thread.h:211 (a.out+0x2088)
    #6 <null> <null> (libstdc++.so.6+0xdc252)
  Location is global ‘shared_counter’ of size 4 at 0x555fd304f154 (a.out+0x000000005154)
  Thread T2 (tid=2560041, running) created by main thread at:
    #0 pthread_create ../../../../src/libsanitizer/tsan/tsan_interceptors_posix.cpp:969 (libtsan.so.0+0x605b8)
    #1 std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)()) <null> (libstdc++.so.6+0xdc328)
    #2 main /home/user/clang-sanitizers/main.cpp:14 (a.out+0x1484)
  Thread T1 (tid=2560040, finished) created by main thread at:
    #0 pthread_create ../../../../src/libsanitizer/tsan/tsan_interceptors_posix.cpp:969 (libtsan.so.0+0x605b8)
    #1 std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)()) <null> (libstdc++.so.6+0xdc328)
    #2 main /home/user/clang-sanitizers/main.cpp:13 (a.out+0x146e)
SUMMARY: ThreadSanitizer: data race /home/user/clang-sanitizers/main.cpp:8 in increment_counter()
==================
Shared counter: 20000
ThreadSanitizer: reported 1 warnings
```

TSan 的此输出表明 C++程序中存在数据竞争条件。让我们分析此报告的关键元素，以了解它告诉我们什么：

+   `WARNING: ThreadSanitizer:` `data race`）。

+   `0x555fd304f154`内存地址，被识别为全局`shared_counter`变量。

+   `increment_counter() /home/user/clang-sanitizers/main.cpp:8`。这意味着数据竞争读取发生在`increment_counter`函数中，具体在`main.cpp`的第*8*行。

+   报告还提供了导致此读取的堆栈跟踪，显示了函数调用的序列。

+   `increment_counter`函数位于`main.cpp`的第*8*行。*   `main.cpp`在第*13*和*14*行，分别)。这有助于理解导致数据竞争的程序流程。*   `SUMMARY: ThreadSanitizer: data race /home/user/clang-sanitizers/main.cpp:8` `in increment_counter()`。

    这简要指出了检测到数据竞争的函数和文件。

### TSan 的微调、性能影响、限制和建议

TSan 通常引入大约 5x-15x 的运行时减速。这种显著的执行时间增加是由于 TSan 执行的综合检查，以检测数据竞争和其他线程问题。除了减速外，TSan 还增加了内存使用，通常约为 5x-10x。这种开销是由于 TSan 使用的额外数据结构来监控线程交互和识别潜在的竞争条件。

此列表概述了 TSan 的限制和当前状态：

+   **Beta 阶段**：TSan 目前处于 Beta 阶段。虽然它在使用 pthread 的大 C++程序中已经有效，但无法保证其在每个场景中的有效性。

+   **支持的线程模型**：当使用 llvm 的 libc++编译时，TSan 支持 C++11 线程。这种兼容性包括 C++11 标准引入的线程功能。

TSan 由多个操作系统和架构支持：

+   **Android**：aarch64, x86_64

+   **Darwin (macOS)**：arm64, x86_64

+   FreeBSD

+   **Linux**：aarch64, x86_64, powerpc64, powerpc64le

+   NetBSD

主要支持 64 位架构。对 32 位平台的支持存在问题，且未计划支持。

### 微调 TSan

TSan 的微调与 ASan 的微调非常相似。对详细微调选项感兴趣的用户可以参考官方文档，该文档提供了全面指导，以定制 TSan 的行为以满足特定需求和场景。

### 使用 TSan 的建议

由于性能和内存开销，TSan 最好在项目的开发和测试阶段使用。在评估性能要求时，应谨慎考虑其在生产环境中的使用。TSan 特别适用于具有大量多线程组件的项目，其中数据竞争和线程问题的可能性更高。将 TSan 集成到 **持续集成**（**CI**）管道中可以帮助在开发周期早期捕获线程问题，从而降低这些错误进入生产的风险。

TSan 是处理 C++ 中并发复杂性的开发者的关键工具。它提供了在检测传统测试方法往往忽略的难以捉摸的线程问题时无价的服务。通过将 TSan 集成到开发和测试过程中，开发者可以显著提高其多线程 C++ 应用程序的可靠性和稳定性。

## UBSan

UBSan 是一种动态分析工具，旨在检测 C++ 程序中的未定义行为。根据 C++ 标准，未定义行为是指其行为未规定的代码，导致程序执行不可预测。这可能包括整数溢出、除以零或对空指针的误用等问题。未定义行为可能导致程序行为异常、崩溃和安全漏洞。然而，它通常被编译器开发者用于优化代码。UBSan 对于识别这些问题至关重要，这些问题通常很微妙且难以通过标准测试检测到，但可能在软件可靠性和安全性方面引起重大问题。

### 配置 UBSan

要使用 UBSan，请使用 `-fsanitize=undefined` 标志编译您的程序。这指示编译器使用对各种形式的未定义行为的检查来对代码进行操作。这些命令使用 Clang 或 GCC 启用 UBSan 编译程序。

### 展示未定义行为的示例代码

考虑这个简单的例子：

```cpp
#include <iostream>
int main() {
    int x = 0;
    std::cout << 10 / x << std::endl;  // Division by zero, undefined behavior
    return 0;
}
```

在此代码中，尝试除以零（`10 / x`）是未定义行为的实例。

当与 UBSan 一起编译和运行时，输出可能包括如下内容：

```cpp
/home/user/clang-sanitizers/main.cpp:5:21: runtime error: division by zero
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior /home/user/clang-sanitizers/main.cpp:5:21 in
0
```

UBSan 检测到除以零，并报告代码中发生此情况的精确位置。

微调、性能影响和限制

+   **微调**：UBSan 提供了各种选项来控制其行为，允许开发者专注于特定类型的未定义行为。对详细定制感兴趣的用户可以参考官方文档。

+   **性能影响**：与 ASan 和 TSan 等工具相比，UBSan 的运行时性能影响通常较低，但会根据启用的检查类型而变化。典型的减速通常很小。

+   **支持的平台**：UBSan 支持主要平台，如 Linux、macOS 和 Windows，使其对 C++ 开发者广泛可用。

+   **局限性**：虽然 UBSan 在检测未定义行为方面非常强大，但它无法捕获每个实例，尤其是那些高度依赖于特定程序状态或硬件配置的实例。

UBSan 是 C++ 开发者的无价之宝，有助于早期发现可能导致软件不稳定和不安全的微妙但关键问题。将其集成到开发和测试过程中是确保 C++ 应用程序健壮性和可靠性的主动步骤。由于其最小的性能影响和广泛的支持平台，UBSan 是任何 C++ 开发者工具包的实用补充。

# 使用 Valgrind 进行动态代码分析

**Valgrind** 是一个强大的内存调试、内存泄漏检测和性能分析工具。它在识别内存管理错误和访问错误等常见问题方面至关重要，这些问题在复杂的 C++ 程序中很常见。与基于编译器的工具（如 Sanitizers）不同，Valgrind 通过在类似虚拟机的环境中运行程序来检查内存相关错误。

## 设置 Valgrind

Valgrind 通常可以通过您的系统包管理器进行安装。例如，在 Ubuntu 上，您可以使用 `sudo apt-get install valgrind` 命令进行安装。要在 Valgrind 下运行程序，请使用 `valgrind ./your_program` 命令。此命令在 Valgrind 环境中执行您的程序，并执行其分析。对于 Valgrind 的基本内存检查，不需要特殊的编译标志，但包含调试符号（使用 `-g`）可以帮助使其输出更有用。

## Memcheck – 全面的内存调试器

**Memcheck**，Valgrind 套件的核心工具，是一个针对 C++ 应用的复杂内存调试器。它结合了地址、内存和 LSans 的功能，提供了对内存使用的全面分析。Memcheck 检测与内存相关的错误，例如使用未初始化的内存、不正确使用内存分配和释放函数以及内存泄漏。

要使用 Memcheck，不需要特殊的编译标志，但使用带有调试信息（使用 `-g`）的编译可以增强 Memcheck 报告的有用性。通过使用 `valgrind ./your_program` 命令执行您的程序。对于检测内存泄漏，添加 `--leak-check=full` 以获取更详细的信息。以下是一个示例命令：

```cpp
valgrind --leak-check=full ./your_program
```

由于 Memcheck 覆盖了广泛的内存相关问题，我将只展示检测内存泄漏的示例，因为它们通常是最难检测的。让我们考虑以下具有内存泄漏的 C++ 代码：

```cpp
int main() {
    int* ptr = new int(10); // Memory allocated but not freed
    return 0; // Memory leak occurs here
}
```

Memcheck 将检测并报告内存泄漏，指示内存是在哪里分配的以及它没有被释放：

```cpp
==12345== Memcheck, a memory error detector
==12345== 4 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x...: operator new(unsigned long) (vg_replace_malloc.c:...)
==12345==    by 0x...: main (your_file.cpp:2)
...
==12345== LEAK SUMMARY:
==12345==    definitely lost: 4 bytes in 1 blocks
...
```

性能影响，微调和局限性

重要的是要记住，Memcheck 可以显著减慢程序执行速度，通常慢 10-30 倍，并增加内存使用。这是由于对每个内存操作进行的广泛检查。

Memcheck 提供了几个选项来控制其行为。例如，`--track-origins=yes`可以帮助找到未初始化内存使用的来源，尽管这可能会进一步减慢分析速度。

Memcheck 的主要局限性是其性能开销，这使得它不适合生产环境。此外，尽管它在内存泄漏检测方面非常彻底，但它可能无法捕捉到所有未初始化内存使用的实例，尤其是在复杂场景或应用特定编译器优化时。

Memcheck 是 C++开发者工具箱中用于内存调试的重要工具。通过提供对内存错误和泄漏的详细分析，它在提高 C++应用程序的可靠性和正确性方面发挥着关键作用。尽管存在性能开销，但 Memcheck 在识别和解决内存问题方面的好处使其在软件开发的开发和测试阶段变得不可或缺。

## Helgrind – 线程错误检测器

**Helgrind**是 Valgrind 套件中的一个工具，专门设计用于检测 C++多线程应用程序中的同步错误。它专注于识别竞争条件、死锁和对 pthreads API 的误用。Helgrind 通过监控线程之间的交互来运行，确保共享资源被安全且正确地访问。它检测线程错误的能力使其与 TSan 相当，但具有不同的底层方法和用法。

要使用 Helgrind，你不需要用特殊标志重新编译你的程序（尽管建议使用`-g`标志来包含调试符号）。使用`--tool=helgrind`选项运行你的程序。以下是一个示例命令：

```cpp
valgrind --tool=helgrind ./your_program
```

让我们考虑我们之前用 TSan 分析过的数据竞争示例：

```cpp
#include <iostream>
#include <thread>
int shared_counter = 0;
void increment_counter() {
    for (int i = 0; i < 10000; ++i) {
        shared_counter++; // Potential data race
    }
}
int main() {
    std::thread t1(increment_counter);
    std::thread t2(increment_counter);
    t1.join();
    t2.join();
    std::cout << “Shared counter: “ << shared_counter << std::endl;
    return 0;
}
```

Helgrind 将检测并报告数据竞争，显示线程在没有适当同步的情况下并发修改`shared_counter`的位置。除了识别数据竞争外，Helgrind 的输出还包含线程创建公告、堆栈跟踪和其他详细信息：

```cpp
valgrind --tool=helgrind ./a.out
==178401== Helgrind, a thread error detector
==178401== Copyright (C) 2007-2017, and GNU GPL’d, by OpenWorks LLP et al.
==178401== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==178401== Command: ./a.out
==178401== ---Thread-Announcement------------------------------------------
==178401==
==178401== Thread #3 was created
==178401==    at 0x4CCE9F3: clone (clone.S:76)
==178401==    by 0x4CCF8EE: __clone_internal (clone-internal.c:83)
==178401==    by 0x4C3D6D8: create_thread (pthread_create.c:295)
==178401==    by 0x4C3E1FF: pthread_create@@GLIBC_2.34 (pthread_create.c:828)
==178401==    by 0x4853767: ??? (in /usr/libexec/valgrind/vgpreload_helgrind-amd64-linux.so)
==178401==    by 0x4952328: std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)()) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==178401==    by 0x1093F9: std::thread::thread<void (&)(), , void>(void (&)()) (std_thread.h:143)
==178401==    by 0x1092AF: main (main.cpp:14)
==178401==
==178401== ---Thread-Announcement------------------------------------------
==178401==
==178401== Thread #2 was created
==178401== ----------------------------------------------------------------
==178401==
==178401== Possible data race during read of size 4 at 0x10C0A0 by thread #3
==178401== Locks held: none
==178401==    at 0x109258: increment_counter() (main.cpp:8)
==178401==    by 0x109866: void std::__invoke_impl<void, void (*)()>(std::__invoke_other, void (*&&)()) (invoke.h:61)
==178401==    by 0x1097FC: std::__invoke_result<void (*)()>::type std::__invoke<void (*)()>(void (*&&)()) (invoke.h:96)
==178401==    by 0x1097D4: void std::thread::_Invoker<std::tuple<void (*)()> >::_M_invoke<0ul>(std::_Index_tuple<0ul>) (std_thread.h:259)
==178401==    by 0x1097A4: std::thread::_Invoker<std::tuple<void (*)()> >::operator()() (std_thread.h:266)
==178401==    by 0x1096F8: std::thread::_State_impl<std::thread::_Invoker<std::tuple<void (*)()> > >::_M_run() (std_thread.h:211)
==178401==    by 0x4952252: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==178401==    by 0x485396A: ??? (in /usr/libexec/valgrind/vgpreload_helgrind-amd64-linux.so)
==178401==    by 0x4C3DAC2: start_thread (pthread_create.c:442)
==178401==    by 0x4CCEA03: clone (clone.S:100)
==178401==
==178401== This conflicts with a previous write of size 4 by thread #2
==178401== Locks held: none
==178401==    at 0x109261: increment_counter() (main.cpp:8)
==178401==    by 0x109866: void std::__invoke_impl<void, void (*)()>(std::__invoke_other, void (*&&)()) (invoke.h:61)
==178401==    by 0x1097FC: std::__invoke_result<void (*)()>::type std::__invoke<void (*)()>(void (*&&)()) (invoke.h:96)
==178401==    by 0x1097D4: void std::thread::_Invoker<std::tuple<void (*)()> >::_M_invoke<0ul>(std::_Index_tuple<0ul>) (std_thread.h:259)
==178401==    by 0x1097A4: std::thread::_Invoker<std::tuple<void (*)()> >::operator()() (std_thread.h:266)
==178401==    by 0x1096F8: std::thread::_State_impl<std::thread::_Invoker<std::tuple<void (*)()> > >::_M_run() (std_thread.h:211)
==178401==    by 0x4952252: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30)
==178401==    by 0x485396A: ??? (in /usr/libexec/valgrind/vgpreload_helgrind-amd64-linux.so)
==178401==  Address 0x10c0a0 is 0 bytes inside data symbol “shared_counter”
==178401==
Shared counter: 20000
==178401==
==178401== Use --history-level=approx or =none to gain increased speed, at
==178401== the cost of reduced accuracy of conflicting-access information
==178401== For lists of detected and suppressed errors, rerun with: -s
==178401== ERROR SUMMARY: 2 errors from 2 contexts (suppressed: 0 from 0)`
```

## 性能影响、微调和局限性

使用 Helgrind 可能会显著减慢你的程序执行速度（通常慢 20 倍或更多），这是由于对线程交互的详细分析。这使得它最适合测试环境。Helgrind 提供了几个选项来自定义其行为，例如控制检查级别或忽略某些错误。其主要局限性是性能开销，这使得它在生产中使用不切实际。此外，Helgrind 可能会产生假阳性，尤其是在复杂的线程场景或使用 Helgrind 不完全理解的先进同步原语时。

Helgrind 是开发多线程 C++应用程序的开发者的重要工具，它提供了对具有挑战性的并发问题的见解。它通过检测和帮助解决复杂的同步问题，有助于创建更可靠和线程安全的应用程序。尽管由于其性能开销，其使用可能仅限于开发和测试阶段，但它为提高多线程代码的正确性提供的益处是无价的。

# Valgrind 套件中的其他知名工具

除了 Helgrind，Valgrind 套件还包括其他几个工具，每个工具都有独特的功能，旨在满足程序分析和性能分析的不同方面。

## 数据竞争检测器（DRD）- 线程错误检测器

**DRD** 是另一个用于检测线程错误的工具，类似于 Helgrind。它专注于在多线程程序中识别数据竞争。虽然 Helgrind 和 DRD 都旨在检测线程问题，但 DRD 在检测数据竞争方面进行了优化，通常比 Helgrind 具有更低的性能开销。在某些情况下，DRD 可能产生较少的误报，但在检测所有类型的同步错误方面可能不如 Helgrind 彻底。

## Cachegrind

**Cachegrind** 是一个缓存和分支预测分析器。它提供了关于您的程序如何与计算机的缓存层次结构交互以及分支预测效率的详细信息。这个工具对于优化程序性能非常有价值，尤其是在 CPU 密集型应用程序中。它有助于识别低效的内存访问模式以及可以通过优化来提高缓存利用率的代码区域。

## Callgrind

**Callgrind** 通过添加调用图生成功能扩展了 Cachegrind 的功能。它记录程序中函数之间的调用历史，使开发者能够分析执行流程并识别性能瓶颈。Callgrind 特别适用于理解复杂应用程序的整体结构和交互。

## Massif

Massif 是一个堆分析器，它提供了关于程序内存使用的见解。它帮助开发者理解和优化内存消耗，追踪内存泄漏，并确定程序中内存分配发生的位置和方式。

## 动态堆分析工具（DHAT）

**DHAT** 专注于分析堆分配模式。它特别适用于查找堆内存的低效使用，例如过多的微小分配或可能优化的小型短期分配。

Valgrind 套件中的每个工具都提供了分析程序性能和行为不同方面的独特功能。从线程问题到内存使用和 CPU 优化，这些工具为增强 C++应用程序的效率、可靠性和正确性提供了一套全面的函数。它们集成到开发和测试过程中，使开发者能够深入了解其代码，从而得到优化良好且稳健的软件解决方案。

# 摘要

基于编译器的清理器和 Valgrind 在调试和性能分析过程中带来了不同的优势和挑战。

基于编译器的工具，如 ASan、TSan 和 UBSan，通常更容易访问，并且更容易集成到开发工作流程中。在引入的性能开销方面，它们“更便宜”，配置和使用相对简单。这些清理器直接集成到编译过程中，使开发者能够经常使用它们。它们的主要优势在于能够在开发阶段提供即时反馈，在编写和测试代码时捕捉错误和问题。然而，由于这些工具在运行时进行分析，其有效性直接与测试覆盖范围的程度相关。测试越全面，动态分析就越有效，因为只有执行的代码路径才会被分析。这一点突出了彻底测试的重要性：测试覆盖范围越好，这些工具可以潜在地揭示的问题就越多。

相反，Valgrind 提供了更强大和彻底的分析，能够检测更广泛的问题，尤其是在内存管理和线程方面。其工具套件——Memcheck、Helgrind、DRD、Cachegrind、Callgrind、Massif 和 DHAT——对程序性能和行为的多方面进行了全面分析。然而，这种力量是有代价的：与基于编译器的工具相比，Valgrind 通常更复杂，引入了显著的性能开销。是否使用 Valgrind 或基于编译器的清理器的选择通常取决于项目的具体需求和要解决的问题。虽然 Valgrind 的广泛诊断提供了对程序的深入洞察，但基于编译器的清理器的易用性和较低的性能成本使它们更适合在 CI 管道中常规使用。

总结来说，虽然基于编译器的工具和 Valgrind 在动态分析领域都有其位置，但它们在诊断、易用性和性能影响方面的差异使它们适合软件开发过程的各个阶段和方面。将它们作为常规持续集成（CI）管道的一部分使用是非常推荐的，因为它允许早期检测和解决问题，对软件的整体质量和鲁棒性贡献显著。下一章将深入探讨测量测试覆盖率的工具，提供关于代码库测试有效性的见解，从而补充动态分析过程。
