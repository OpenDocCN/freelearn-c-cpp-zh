# 12

# 清理和测试异步软件

**测试**是评估和验证软件解决方案是否按预期工作，验证其质量并确保满足用户需求的过程。通过适当的测试，我们可以预防错误的发生并提高性能。

在本章中，我们将探讨几种测试异步软件的技术，主要使用**GoogleTest**库以及来自**GNU 编译器集合**（**GCC**）和**Clang**编译器的清理器。需要一些单元测试的先验知识。在本章末尾的**进一步阅读**部分，您可以找到一些可能有助于刷新和扩展这些领域知识的参考资料。

在本章中，我们将涵盖以下主要主题：

+   清理代码以分析软件并查找潜在问题

+   测试异步代码

# 技术要求

对于本章，我们需要安装**GoogleTest**（[`google.github.io/googletest`](https://google.github.io/googletest)）来编译一些示例。

一些示例需要支持 C++20 的编译器。因此，请参阅*第三章*中的**技术要求**部分，因为它包含有关如何安装 GCC 13 和 Clang 8 编译器的指导。

您可以在以下 GitHub 仓库中找到所有完整代码：

[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)

本章的示例位于**Chapter_12**文件夹下。所有源代码文件都可以使用以下 CMake 编译：

```cpp
$ cmake . && cmake —build .
```

可执行二进制文件将在**bin**目录下生成。

# 清理代码以分析软件并查找潜在问题

**清理器**是工具，最初由 Google 开发，用于检测和预防代码中各种类型的问题或安全漏洞，帮助开发者尽早在开发过程中捕捉到错误，减少后期修复问题的成本，并提高软件的稳定性和安全性。

清理器通常集成到开发环境中，并在手动测试或运行单元测试、**持续集成**（**CI**）管道或代码审查管道时启用。

C++ 编译器，如 GCC 和 Clang，在构建程序时提供编译器选项以生成代码，以跟踪运行时的执行并报告错误和漏洞。这些功能从 Clang 3.1 版本和 GCC 4.8 版本开始实现。

由于向程序的二进制代码中注入了额外的指令，根据清理器类型，性能惩罚约为 1.5 倍到 4 倍减慢。此外，总体内存开销为 2 倍到 4 倍，堆栈大小增加最多 3 倍。但请注意，减慢程度远低于使用其他仪器框架或动态分析工具（如**Valgrind** [`valgrind.org`](https://valgrind.org)）时遇到的减慢，后者比生产二进制文件慢高达 50 倍。另一方面，使用 Valgrind 的好处是不需要重新编译。两种方法都仅在程序运行时检测问题，并且仅在执行遍历的代码路径上检测。因此，我们需要确保足够的覆盖率。

此外，还有静态分析工具和代码检查器，它们在编译期间检测问题并检查程序中包含的所有代码，非常有用。例如，通过启用**–Werror**、**–Wall**和**–pedantic**选项，编译器如 GCC 和 Clang 可以执行额外的检查并提供有用的信息。

此外，还有开源替代方案，如**Cppcheck**或**Flawfinder**，或免费提供给开源项目的商业解决方案，如**PVS-Studio**或**Coverity Scan**。其他解决方案，如**SonarQube**、**CodeSonar**或**OCLint**，可用于**持续集成/持续交付**（**CI/CD**）管道中的持续质量跟踪。

在本节中，我们将重点关注可以通过向编译器传递一些特殊选项来启用的清理器。

## 编译器选项

要启用清理器，我们需要在编译程序时传递一些编译器选项。

主要选项是**--fsanitize=sanitizer_name**，其中**sanitizer_name**是以下选项之一：

+   **地址**：这是针对**AddressSanitizer**（**ASan**），用于检测内存错误，如缓冲区溢出和使用后释放错误

+   **线程**：这是针对**ThreadSanitizer**（**TSan**），通过监控线程交互来识别多线程程序中的数据竞争和其他线程同步问题

+   **泄露**：这是针对**LeakSanitizer**（**LSan**），通过跟踪内存分配并确保所有分配的内存都得到适当释放来发现内存泄露

+   **内存**：这是针对**MemorySanitizer**（**MSan**），用于揭示未初始化内存的使用

+   **未定义**：这是针对**UndefinedBehaviorSanitizer**（**UBSan**），用于检测未定义行为，例如整数溢出、无效类型转换和其他错误操作

Clang 还包括**dataflow**、**cfi**（控制流完整性）、**safe_stack**和**realtime**。

GCC 增加了**kernel-address**、**hwaddress**、**kernel-hwaddress**、**pointer-compare**、**pointer-subtract**和**shadow-call-stack**。

由于此列表和标志行为可能会随时间而变化，建议检查编译器的官方文档。

可能需要额外的标志：

+   **-fno-omit-frame-pointer**：帧指针是编译器用来跟踪当前堆栈帧的寄存器，其中包含其他信息，如当前函数的基址。省略帧指针可能会提高程序的性能，但代价是使调试变得非常困难；它使得定位局部变量和重建堆栈跟踪更加困难。

+   **-g**：包含调试信息，并在警告消息中显示文件名和行号。如果使用调试器 GDB，则可能希望使用**–ggdb**选项，因为编译器可以生成更易于调试的符号。还可以通过使用**–g[level]**指定一个级别，其中**[level]**是一个从**0**到**3**的值，每次级别增加都会添加更多的调试信息。默认级别是**2**。

+   **–fsanitize-recover**：这些选项会导致清理器尝试继续运行程序，就像没有检测到错误一样。

+   **–fno-sanitize-recover**：清理器将仅检测到第一个错误，并且程序将以非零退出码退出。

为了保持合理的性能，我们可能需要通过指定**–O[num]**选项来调整优化级别。不同的清理器在一定的优化级别上表现最佳。最好从**–O0**开始，如果减速显著，尝试增加到**–O1**、**–O2**等。此外，由于不同的清理器和编译器推荐特定的优化级别，请检查它们的文档。

当使用 Clang 时，为了使堆栈跟踪易于理解，并让清理器将地址转换为源代码位置，除了使用前面提到的标志外，我们还可以将特定的环境变量**[X]SAN_SYMBOLIZER_PATH**设置为**llvm-symbolizer**的位置（其中**[X]**为**A**表示 AddressSanitizer，**L**表示 LSan，**M**表示 MSan 等）。我们还可以将此位置包含在**PATH**环境变量中。以下是在使用**AddressSanitizer**时设置**PATH**变量的示例：

```cpp
export ASAN_SYMBOLIZER_PATH=`which llvm-symbolizer`
export PATH=$ASAN_SYMBOLIZER_PATH:$PATH
```

注意，启用**–Werror**与某些清理器一起可能会导致误报。此外，可能还需要其他编译器标志，但执行期间的警告消息将显示正在发生问题，并且将明显表明需要某个标志。请检查清理器和编译器的文档，以找到在那些情况下应使用的标志。

### 避免对代码部分进行清理

有时，我们可能希望静音某些清理器警告，并跳过某些函数的清理，原因如下：这是一个已知问题，该函数是正确的，这是一个误报，该函数需要加速，或者这是一个第三方库的问题。在这些情况下，我们可以使用抑制文件或通过使用某些宏指令排除代码区域。还有一个黑名单机制，但由于它已被抑制文件取代，我们在此不做评论。

使用抑制文件，我们只需要创建一个文本文件，列出我们不希望清理器运行的代码区域。每一行都包含一个模式，该模式根据清理器的不同而有所不同，但通常结构如下：

```cpp
type:location_pattern
```

在这里，**type** 表示抑制的类型，例如，**leak** 和 **race** 值，而 **location_pattern** 是匹配要抑制的函数或库名的正则表达式。下面是一个 ASan 的抑制文件示例，将在下一节中解释：

```cpp
# Suppress known memory leaks in third-party function Func1 in library Lib1
leak:Lib1::Func1
# Ignore false-positive from function Func2 in library Lib2
race:Lib2::Func2
# Suppress issue from libc
leak:/usr/lib/libc.so.*
```

让我们称这个文件为 **myasan.supp**。然后，编译并使用以下命令将抑制文件传递给清理器通过 **[X]SAN_OPTIONS**：

```cpp
$ clang++ -O0 -g -fsanitize=address -fno-omit-frame-pointer test.cpp –o test
$ ASAN_OPTIONS=suppressions=myasan.supp ./test
```

我们还可以在源代码中使用宏来排除特定的函数，使其不被清理器清理，如下所示使用 **__attribute__((no_sanitize("<sanitizer_name>")))**：

```cpp
#if defined(__clang__) || defined (__GNUC__)
# define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
# define ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif
...
ATTRIBUTE_NO_SANITIZE_ADDRESS
void ThisFunctionWillNotBeInstrumented() {...}
```

这种技术提供了对清理器应该对什么进行插装的细粒度编译时控制。

现在，让我们探索最常见的代码清理器类型，从与检查地址误用最相关的一种开始。

## AddressSanitizer

ASan 的目的是检测由于数组越界访问、使用释放的内存块（堆、栈和全局）以及其他内存泄漏而发生的内存相关错误。

除了设置 **-fsanitize=address** 和之前推荐的其他标志外，我们还可以使用 **–fsanitize-address-use-after-scope** 来检测移出作用域后使用的内存，或者设置环境变量 **ASAN_OPTIONS=option detect_stack_use_after_return=1** 来检测返回后使用。

**ASAN_OPTIONS** 也可以用来指示 ASan 打印堆栈跟踪或设置日志文件，如下所示：

```cpp
ASAN_OPTIONS=detect_stack_use_after_return=1,print_stacktrace=1,log_path=asan.log
```

Linux 上的 Clang 完全支持 ASan，其次是 Linux 上的 GCC。默认情况下，ASan 是禁用的，因为它会增加额外的运行时开销。

此外，ASan 处理所有对 **glibc** 的调用——这是为 GNU 系统提供核心库的 GNU C 库。然而，其他库的情况并非如此，因此建议使用 **–fsanitize=address** 选项重新编译此类库。如前所述，使用 Valgrind 不需要重新编译。

ASan 可以与 UBSan 结合使用，我们将在后面看到，但这会降低性能约 50%。

如果我们想要更激进的诊断清理，可以使用以下标志组合：

```cpp
ASAN_OPTIONS=strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
```

让我们看看使用 ASan 检测常见软件问题的两个示例，包括释放内存后继续使用和检测缓冲区溢出。

### 释放内存后的内存使用

软件中常见的一个问题是释放内存后继续使用。在这个例子中，堆中分配的内存被删除后仍在使用：

```cpp
#include <iostream>
#include <memory>
int main() {
  auto arr = new int[100];
  delete[] arr;
  std::cout << "arr[0] = " << arr[0] << '\n';
  return 0;
}
```

假设之前的源代码在一个名为 **test.cpp** 的文件中。要启用 ASan，我们只需使用以下命令编译文件：

```cpp
$ clang++ -fsanitize=address -fno-omit-frame-pointer -g -O0 –o test test.cpp
```

然后，执行生成的输出 **test** 程序，我们得到以下输出（注意，输出已简化，仅显示相关内容，可能因不同的编译器版本和执行环境而有所不同）：

```cpp
ERROR: AddressSanitizer: heap-use-after-free on address 0x514000000040 at pc 0x63acc82a0bec bp 0x7fff2d096c60 sp 0x7fff2d096c58
READ of size 4 at 0x514000000040 thread T0
    #0 0x63acc82a0beb in main test.cpp:7:31
0x514000000040 is located 0 bytes inside of 400-byte region 0x514000000040,0x5140000001d0)
freed by thread T0 here:
    #0 0x63acc829f161 in operator delete[ (/mnt/StorePCIE/Projects/Books/Packt/Book/Code/build/bin/Chapter_11/11x02-ASAN_heap_use_after_free+0x106161) (BuildId: 7bf8fe6b1f86a8b587fbee39ae3a5ced3e866931)
previously allocated by thread T0 here:
    #0 0x63acc829e901 in operator new[](unsigned long) (/mnt/StorePCIE/Projects/Books/Packt/Book/Code/build/bin/Chapter_11/11x02-ASAN_heap_use_after_free+0x105901) (BuildId: 7bf8fe6b1f86a8b587fbee39ae3a5ced3e866931)
SUMMARY: AddressSanitizer: heap-use-after-free test.cpp:7:31 in main
```

输出显示 ASan 已应用并检测到一个堆使用后释放错误。这个错误发生在 **T0** 线程（主线程）。输出还指向了分配该内存区域的代码，稍后释放，以及其大小（400 字节区域）。

这类错误不仅发生在堆内存中，也发生在栈或全局区域分配的内存区域中。ASan 可以用来检测这类问题，例如内存溢出。

### 内存溢出

内存溢出，也称为缓冲区溢出或越界，发生在将某些数据写入超出缓冲区分配内存的地址时。

以下示例显示了一个堆内存溢出：

```cpp
#include <iostream>
int main() {
  auto arr = new int[100];
  arr[0] = 0;
  int res = arr[100];
  std::cout << "res = " << res << '\n';
  delete[] arr;
  return 0;
}
```

编译并运行生成的程序后，这是输出：

```cpp
ERROR: AddressSanitizer: heap-buffer-overflow on address 0x5140000001d0 at pc 0x582953d2ac07 bp 0x7ffde9d58910 sp 0x7ffde9d58908
READ of size 4 at 0x5140000001d0 thread T0
    #0 0x582953d2ac06 in main test.cpp:6:13
0x5140000001d0 is located 0 bytes after 400-byte region 0x514000000040,0x5140000001d0)
allocated by thread T0 here:
    #0 0x582953d28901 in operator new[ (test+0x105901) (BuildId: 82a16fc86e01bc81f6392d4cbcad0fe8f78422c0)
    #1 0x582953d2ab78 in main test.cpp:4:14
(test+0x2c374) (BuildId: 82a16fc86e01bc81f6392d4cbcad0fe8f78422c0)
SUMMARY: AddressSanitizer: heap-buffer-overflow test.cpp:6:13 in main
```

从输出中我们可以看到，现在 ASan 报告了主线程（**T0**）在访问超过 400 字节区域（**arr** 变量）的内存地址时的堆缓冲区溢出错误。

集成到 ASan 中的清理器是 LSan。现在让我们学习如何使用这个清理器来检测内存泄漏。

## LeakSanitizer

LSan 用于检测内存泄漏，当内存已分配但未正确释放时发生。

LSan 集成到 ASan 中，并在 Linux 系统上默认启用。在 macOS 上可以通过使用 **ASAN_OPTIONS=detect_leaks=1** 来启用它。要禁用它，只需设置 **detect_leaks=0** 。

如果使用 **–fsanitize=leak** 选项，程序将链接到支持 LSan 的 ASan 的子集，禁用编译时仪器并减少 ASan 的减速。请注意，此模式不如默认模式经过充分测试。

让我们看看一个内存泄漏的例子：

```cpp
#include <string.h>
#include <iostream>
#include <memory>
int main() {
    auto arr = new char[100];
    strcpy(arr, "Hello world!");
    std::cout << "String = " << arr << '\n';
    return 0;
}
```

在这个例子中，分配了 100 字节（**arr** 变量），但从未释放。

要启用 LSan，我们只需使用以下命令编译文件：

```cpp
$ clang++ -fsanitize=leak -fno-omit-frame-pointer -g -O2 –o test test.cpp
```

运行生成的测试二进制文件，我们得到以下结果：

```cpp
ERROR: LeakSanitizer: detected memory leaks
Direct leak of 100 byte(s) in 1 object(s) allocated from:
    #0 0x5560ba9a017c in operator new[](unsigned long) (test+0x3417c) (BuildId: 2cc47a28bb898b4305d90c048c66fdeec440b621)
    #1 0x5560ba9a2564 in main test.cpp:6:16
SUMMARY: LeakSanitizer: 100 byte(s) leaked in 1 allocation(s).
```

LSan 正确报告了一个 100 字节大小的内存区域是通过使用操作符 **new** 分配的，但从未被删除。

由于本书探讨了多线程和异步编程，现在让我们了解一个用于检测数据竞争和其他线程问题的清理器：TSan。

## ThreadSanitizer

TSan 用于检测线程问题，特别是数据竞争和同步问题。它不能与 ASan 或 LSan 结合使用。TSan 是与本书内容最一致的清理器。

通过指定 **–fsanitize=thread** 编译器选项启用此清理器，可以通过使用 **TSAN_OPTIONS** 环境变量来修改其行为。例如，如果我们想在第一次错误后停止，只需使用以下命令：

```cpp
TSAN_OPTIONS=halt_on_error=1
```

此外，为了合理的性能，使用编译器的 **–** **O2** 选项。

TSan 只报告在运行时发生的竞争条件，因此它不会在未在运行时执行的代码路径中存在的竞争条件上发出警报。因此，我们需要设计提供良好覆盖率和使用真实工作负载的测试。

让我们看看 TSan 检测数据竞争的一些示例。在下一个示例中，我们将通过使用一个全局变量而不使用互斥锁来保护其访问来实现这一点：

```cpp
#include <thread>
int globalVar{0};
void increase() {
  globalVar++;
}
void decrease() {
  globalVar--;
}
int main() {
  std::thread t1(increase);
  std::thread t2(decrease);
  t1.join();
  t2.join();
  return 0;
}
```

编译程序后，使用以下命令启用 TSan：

```cpp
$ clang++ -fsanitize=thread -fno-omit-frame-pointer -g -O2 –o test test.cpp
```

运行生成的程序会生成以下输出：

```cpp
WARNING: ThreadSanitizer: data race (pid=31692)
  Write of size 4 at 0x5932b0585ae8 by thread T2:
    #0 decrease() test.cpp:10:12 (test+0xe0b32) (BuildId: 895b75ef540c7b44daa517a874d99d06bd27c8f7)
  Previous write of size 4 at 0x5932b0585ae8 by thread T1:
    #0 increase() test.cpp:6:12 (test+0xe0af2) (BuildId: 895b75ef540c7b44daa517a874d99d06bd27c8f7)
  Thread T2 (tid=31695, running) created by main thread at:
    #0 pthread_create <null> (test+0x6062f) (BuildId: 895b75ef540c7b44daa517a874d99d06bd27c8f7)
  Thread T1 (tid=31694, finished) created by main thread at:
    #0 pthread_create <null> (test+0x6062f) (BuildId: 895b75ef540c7b44daa517a874d99d06bd27c8f7)
SUMMARY: ThreadSanitizer: data race test.cpp:10:12 in decrease()
ThreadSanitizer: reported 1 warnings
```

从输出中可以看出，在**increase()**和**decrease()**函数访问**globalVar**时存在数据竞争。

如果我们决定使用 GCC 而不是 Clang，在运行生成的程序时可能会报告以下错误：

```cpp
FATAL: ThreadSanitizer: unexpected memory mapping 0x603709d10000-0x603709d11000
```

这种内存映射问题是由称为**地址空间布局随机化**（**ASLR**）的安全功能引起的，这是一种操作系统使用的内存保护技术，通过随机化进程的地址空间来防止缓冲区溢出攻击。

一种解决方案是使用以下命令减少 ASLR：

```cpp
$ sudo sysctl vm.mmap_rnd_bits=30
```

如果错误仍然发生，传递给**vm.mmap_rnd_bits**（在先前的命令中为**30**）的值可以进一步降低。为了检查该值是否正确设置，只需运行以下命令：

```cpp
$ sudo sysctl vm.mmap_rnd_bits
vm.mmap_rnd_bits = 30
```

注意，此更改不是永久的。因此，当机器重新启动时，其值将设置为默认值。要持久化此更改，请将**m.mmap_rnd_bits=30**添加到**/etc/sysctl.conf**。

但这降低了系统的安全性，因此可能更倾向于使用以下命令临时禁用特定程序的 ASLR：

```cpp
$ setarch `uname -m` -R ./test
```

运行上述命令将显示与使用 Clang 编译时显示的类似输出。

让我们转到另一个示例，其中**std::map**对象在没有互斥锁的情况下被访问。即使映射被用于不同的键值，因为写入**std::map**会使其迭代器无效，这也可能导致数据竞争：

```cpp
#include <map>
#include <thread>
std::map<int,int> m;
void Thread1() {
  m[123] = 1;
}
void Thread2() {
  m[345] = 0;
}
int main() {
  std::jthread t1(Thread1);
  std::jthread t2(Thread1);
  return 0;
}
```

编译并运行生成的二进制文件会生成大量输出，包含三个警告。在这里，我们只显示第一个警告中最相关的行（其他警告类似）：

```cpp
WARNING: ThreadSanitizer: data race (pid=8907)
  Read of size 4 at 0x720c00000020 by thread T2:
  Previous write of size 8 at 0x720c00000020 by thread T1:
  Location is heap block of size 40 at 0x720c00000000 allocated by thread T1:
  Thread T2 (tid=8910, running) created by main thread at:
  Thread T1 (tid=8909, finished) created by main thread at:
SUMMARY: ThreadSanitizer: data race test.cpp:11:3 in Thread2()
```

当**t1**和**t2**线程都在向映射，**m**写入时，TSan 警告会标记。

在下一个示例中，只有一个辅助线程通过指针访问映射，但此线程与主线程竞争以访问和使用映射。**t**线程访问映射，**m**，以更改**foo**键的值；同时，主线程将其值打印到控制台：

```cpp
#include <iostream>
#include <thread>
#include <map>
#include <string>
typedef std::map<std::string, std::string> map_t;
void *func(void *p) {
  map_t& m = *static_cast<map_t*>(p);
  m["foo"] = "bar";
  return 0;
}
int main() {
  map_t m;
  std::thread t(func, &m);
  std::cout << "foo = " << m["foo"] << '\n';
  t.join();
  return 0;
}
```

编译并运行此示例会生成大量输出，包含七个 TSan 警告。在这里，我们只显示第一个警告。您可以自由地通过在 GitHub 存储库中编译和运行示例来检查完整的报告：

```cpp
WARNING: ThreadSanitizer: data race (pid=10505)
  Read of size 8 at 0x721800003028 by main thread:
    #8 main test.cpp:17:28 (test+0xe1d75) (BuildId: 8eef80df1b5c81ce996f7ef2c44a6c8a11a9304f)
  Previous write of size 8 at 0x721800003028 by thread T1:
    #0 operator new(unsigned long) <null> (test+0xe0c3b) (BuildId: 8eef80df1b5c81ce996f7ef2c44a6c8a11a9304f)
    #9 func(void*) test.cpp:10:3 (test+0xe1bb7) (BuildId: 8eef80df1b5c81ce996f7ef2c44a6c8a11a9304f)
  Location is heap block of size 96 at 0x721800003000 allocated by thread T1:
    #0 operator new(unsigned long) <null> (test+0xe0c3b) (BuildId: 8eef80df1b5c81ce996f7ef2c44a6c8a11a9304f)
    #9 func(void*) test.cpp:10:3 (test+0xe1bb7) (BuildId: 8eef80df1b5c81ce996f7ef2c44a6c8a11a9304f)
  Thread T1 (tid=10507, finished) created by main thread at:
    #0 pthread_create <null> (test+0x616bf) (BuildId: 8eef80df1b5c81ce996f7ef2c44a6c8a11a9304f)
SUMMARY: ThreadSanitizer: data race test.cpp:17:28 in main
ThreadSanitizer: reported 7 warnings
```

从输出中，TSan 正在警告访问在堆中分配的**std::map**对象时存在数据竞争。该对象是映射**m**。

然而，TSan 不仅可以通过缺少互斥锁来检测数据竞争，还可以报告何时变量必须是原子的。

下一个示例展示了这种情况。**RefCountedObject** 类定义了可以保持该类已创建对象数量的引用计数的对象。智能指针遵循这个想法，当计数器达到值 **0** 时，在销毁时删除底层分配的内存。在这个例子中，我们只展示了 **Ref()** 和 **Unref()** 函数，它们增加和减少引用计数变量 **ref_**。为了避免多线程环境中的问题，**ref_** 必须是一个原子变量。正如这里所示，这并不是这种情况，**t1** 和 **t2** 线程正在修改 **ref_**，可能发生数据竞争：

```cpp
#include <iostream>
#include <thread>
class RefCountedObject {
   public:
    void Ref() {
        ++ref_;
    }
    void Unref() {
        --ref_;
    }
   private:
    // ref_ should be atomic to avoid synchronization issues
    int ref_{0};
};
int main() {
  RefCountedObject obj;
  std::jthread t1(&RefCountedObject::Ref, &obj);
  std::jthread t2(&RefCountedObject::Unref, &obj);
  return 0;
}
```

编译并运行此示例会产生以下输出：

```cpp
WARNING: ThreadSanitizer: data race (pid=32574)
  Write of size 4 at 0x7fffffffcc04 by thread T2:
    #0 RefCountedObject::Unref() test.cpp:12:9 (test+0xe1dd0) (BuildId: 448eb3f3d1602e21efa9b653e4760efe46b621e6)
  Previous write of size 4 at 0x7fffffffcc04 by thread T1:
    #0 RefCountedObject::Ref() test.cpp:8:9 (test+0xe1c00) (BuildId: 448eb3f3d1602e21efa9b653e4760efe46b621e6)
  Location is stack of main thread.
  Location is global '??' at 0x7ffffffdd000 ([stack]+0x1fc04)
  Thread T2 (tid=32577, running) created by main thread at:
    #0 pthread_create <null> (test+0x6164f) (BuildId: 448eb3f3d1602e21efa9b653e4760efe46b621e6)
    #2 main test.cpp:23:16 (test+0xe1b94) (BuildId: 448eb3f3d1602e21efa9b653e4760efe46b621e6)
  Thread T1 (tid=32576, finished) created by main thread at:
    #0 pthread_create <null> (test+0x6164f) (BuildId: 448eb3f3d1602e21efa9b653e4760efe46b621e6)
    #2 main test.cpp:22:16 (test+0xe1b56) (BuildId: 448eb3f3d1602e21efa9b653e4760efe46b621e6)
SUMMARY: ThreadSanitizer: data race test.cpp:12:9 in RefCountedObject::Unref()
ThreadSanitizer: reported 1 warnings
```

TSan 输出显示，当访问之前由 **Ref()** 函数修改的内存位置时，**Unref()** 函数中发生了数据竞争条件。

数据竞争也可能发生在没有同步机制的情况下从多个线程初始化的对象中。在以下示例中，**MyObj** 类型的对象在 **init_object()** 函数中被创建，全局静态指针 **obj** 被分配其地址。由于此指针没有由互斥锁保护，当 **t1** 和 **t2** 线程分别从 **func1()** 和 **func2()** 函数尝试创建对象并更新 **obj** 指针时，会发生数据竞争：

```cpp
#include <iostream>
#include <thread>
class MyObj {};
static MyObj *obj = nullptr;
void init_object() {
  if (!obj) {
    obj = new MyObj();
  }
}
void func1() {
  init_object();
}
void func2() {
  init_object();
}
int main() {
  std::thread t1(func1);
  std::thread t2(func2);
  t1.join();
  t2.join();
  return 0;
}
```

这是编译并运行此示例后的输出：

```cpp
WARNING: ThreadSanitizer: data race (pid=32826)
  Read of size 1 at 0x5663912cbae8 by thread T2:
    #0 func2() test.cpp (test+0xe0b68) (BuildId: 12f32c1505033f9839d17802d271fc869b7a3e38)
  Previous write of size 1 at 0x5663912cbae8 by thread T1:
    #0 func1() test.cpp (test+0xe0b3d) (BuildId: 12f32c1505033f9839d17802d271fc869b7a3e38)
  Location is global 'obj (.init)' of size 1 at 0x5663912cbae8 (test+0x150cae8)
  Thread T2 (tid=32829, running) created by main thread at:
    #0 pthread_create <null> (test+0x6062f) (BuildId: 12f32c1505033f9839d17802d271fc869b7a3e38)
  Thread T1 (tid=32828, finished) created by main thread at:
    #0 pthread_create <null> (test+0x6062f) (BuildId: 12f32c1505033f9839d17802d271fc869b7a3e38)
SUMMARY: ThreadSanitizer: data race test.cpp in func2()
ThreadSanitizer: reported 1 warnings
```

输出显示了我们之前描述的情况，由于从 **func1()** 和 **func2()** 访问 **obj** 全局变量而导致的数据竞争。

由于 C++11 标准已正式将数据竞争视为未定义行为，现在让我们看看如何使用 UBSan 来检测程序中的未定义行为问题。

## UndefinedBehaviorSanitizer

UBSan 可以检测代码中的未定义行为，例如，当通过过多的位移操作、整数溢出或误用空指针时。可以通过指定 **–fsanitize=undefined** 选项来启用它。其行为可以通过设置 **UBSAN_OPTIONS** 变量在运行时进行修改。

许多 UBSan 可以检测到的错误也可以在编译期间由编译器检测到。

让我们看看一个简单的例子：

```cpp
int main() {
  int val = 0x7fffffff;
  val += 1;
  return 0;
}
```

要编译程序并启用 UBSan，请使用以下命令：

```cpp
$ clang++ -fsanitize=undefined -fno-omit-frame-pointer -g -O2 –o test test.cpp
```

运行生成的程序会产生以下输出：

```cpp
test.cpp:3:7: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior test.cpp:3:7
```

输出非常简单且易于理解；存在一个有符号整数溢出操作。

现在，让我们了解另一个有用的 C++ 检查器，用于检测未初始化的内存和其他内存使用问题：MSan。

## MemorySanitizer

MSan 可以检测未初始化的内存使用，例如，在使用变量或指针之前没有分配值或地址时。它还可以跟踪位域中的未初始化位。

要启用 MSan，请使用以下编译器标志：

```cpp
-fsanitize=memory -fPIE -pie -fno-omit-frame-pointer
```

它还可以通过指定**-** **fsanitize-memory-track-origins**选项将每个未初始化的值追踪到其创建的内存分配。

GCC 不支持 MSan，因此当使用此编译器时，**-fsanitize=memory**标志是无效的。

在以下示例中，创建了**arr**整数数组，但只初始化了其位置**5**。在向控制台打印消息时使用位置**0**的值，但此值仍然是未初始化的：

```cpp
#include <iostream>
int main() {
  auto arr = new int[10];
  arr[5] = 0;
  std::cout << "Value at position 0 = " << arr[0] << '\n';
  return 0;
}
```

要编译程序并启用 MSan，请使用以下命令：

```cpp
$ clang++ -fsanitize=memory -fno-omit-frame-pointer -g -O2 –o test test.cpp
```

运行生成的程序将生成以下输出：

```cpp
==20932==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x5b9fa2bed38f in main test.cpp:6:41
    #3 0x5b9fa2b53324 in _start (test+0x32324) (BuildId: c0a0d31f01272c3ed59d4ac66b8700e9f457629f)
SUMMARY: MemorySanitizer: use-of-uninitialized-value test.cpp:6:41 in main
```

再次，输出清楚地显示，在读取**arr**数组中位置**0**的值时，在第 6 行使用了未初始化的值。

最后，让我们在下一节总结其他检查器。

## 其他检查器

在为某些系统（如内核或实时开发）开发时，还有其他有用的检查器：

+   **硬件辅助地址检查器 (HWASan)**：ASan 的一个新变体，通过使用硬件能力忽略指针的最高字节来消耗更少的内存。可以通过指定**–** **fsanitize=hwaddress**选项来启用。

+   **实时检查器 (RTSan)**：实时测试工具，用于检测在调用具有确定运行时要求的函数中不安全的函数时发生的实时违规。

+   **Fuzzer 检查器**：一种检查器，通过向程序输入大量随机数据来检测潜在漏洞，检查程序是否崩溃，并寻找内存损坏或其他安全漏洞。

+   **内核相关检查器**：还有其他检查器可用于通过内核开发者跟踪问题。出于好奇，以下是一些例子：

    +   **内核地址** **检查器** ( **KASAN** )

    +   **内核并发** **检查器** ( **KCSAN** )

    +   **内核** **电栅栏** ( **KFENCE** )

    +   **内核内存** **检查器** ( **KMSAN** )

    +   **内核线程** **检查器** ( **KTSAN** )

检查器可以自动在我们的代码中找到许多问题。一旦我们找到并调试了一些错误，并且可以重现导致这些特定错误的场景，设计一些涵盖这些情况的测试将非常方便，以避免未来代码中的更改可能导致类似问题或事件。

让我们在下一节学习如何测试多线程和异步代码。

# 测试异步代码

最后，让我们探索一些测试异步代码的技术。本节中显示的示例需要**GoogleTest**和**GoogleTest Mock** ( **gMock** )库来编译。如果您不熟悉这些库，请查阅官方文档了解如何安装和使用它们。

正如我们所知，**单元测试**是一种编写小型且独立的测试的实践，用于验证单个代码单元的功能和行为。单元测试有助于发现和修复错误，重构和改进代码质量，记录和传达底层代码设计，并促进协作和集成。

本节不会涵盖将测试分组到逻辑和描述性套件的最佳方式，或者何时应该使用断言或期望来验证不同变量和测试方法结果的值。本节的目的在于提供一些关于如何创建单元测试以测试异步代码的指南。因此，对单元测试或**测试驱动开发**（**TDD**）有一些先前的知识是可取的。

处理异步代码的主要困难在于它可能在另一个线程中执行，通常不知道何时会发生，或何时完成。

测试异步代码时，主要遵循的方法是将功能与多线程分离，这意味着我们可能希望以同步方式测试异步代码，尝试在一个特定的线程中执行它，移除上下文切换、线程创建和销毁以及其他可能影响测试结果和时序的活动。有时，也会使用计时器，在超时前等待回调被调用。

## 测试一个简单的异步函数

让我们从测试一个异步操作的小例子开始。此示例展示了**asyncFunc()**函数，它通过使用**std::async**异步运行来测试，如第七章中所示：

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include <future>
using namespace std::chrono_literals;
int asyncFunc() {
    std::this_thread::sleep_for(100ms);
    return 42;
}
TEST(AsyncTests, TestHandleAsyncOperation) {
    std::future<int> result = std::async(
                         std::launch::async,
                         asyncFunc);
    EXPECT_EQ(result.get(), 42);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**std::async**返回一个 future，用于检索计算值。在这种情况下，**asyncFunc**只是等待**100ms**然后返回值**42**。如果异步任务运行正常，测试将通过，因为有一个期望指令检查返回的值确实是**42**。

只定义了一个测试，使用**TEST()**宏，其中第一个参数是测试套件名称（在这个例子中，**AsyncTests**），第二个参数是测试名称（**TestHandleAsyncOperation**）。

在**main()**函数中，通过调用**::testing::InitGoogleTest()**初始化 GoogleTest 库。此函数解析命令行以获取 GoogleTest 识别的标志。然后调用**RUN_ALL_TESTS()**，该函数收集并运行所有测试，如果所有测试都成功则返回**0**，否则返回**1**。这个函数最初是一个宏，这就是为什么它的名字是大写的。

## 通过使用超时限制测试时长

这种方法可能出现的一个问题是，异步任务可能由于任何原因而未能被调度，完成时间超过预期，或者由于任何原因未能完成。为了处理这种情况，可以使用计时器，将其超时时间设置为合理的值，以便给测试足够的时间成功完成。因此，如果计时器超时，测试将失败。以下示例通过在 **std::async** 返回的 future 上使用定时等待来展示这种方法：

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include <future>
using namespace std::chrono;
using namespace std::chrono_literals;
int asyncFunc() {
    std::this_thread::sleep_for(100ms);
    return 42;
}
TEST(AsyncTest, TestTimeOut) {
    auto start = steady_clock::now();
    std::future<int> result = std::async(
                         std::launch::async,
                         asyncFunc);
    if (result.wait_for(200ms) ==
               std::future_status::timeout) {
        FAIL() << "Test timed out!";
    }
    EXPECT_EQ(result.get(), 42);
    auto end = steady_clock::now();
    auto elapsed = duration_cast<milliseconds>(
                                end - start);
    EXPECT_LT(elapsed.count(), 200);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

现在，调用 future 对象 **result** 的 **wait_for()** 函数，等待 200 毫秒以完成异步任务。由于任务将在 100 毫秒内完成，超时不会过期。如果由于任何原因，**wait_for()** 被调用时的时间值低于 100 毫秒，它将超时，并调用 **FAIL()** 宏，使测试失败。

测试继续运行并检查返回的值是否为 **42**，正如前一个示例中所示，并且还检查执行异步任务所花费的时间是否少于使用的超时时间。

## 测试回调

测试回调是一个相关任务，尤其是在实现库和 **应用程序编程接口** ( **API** ) 时。以下示例展示了如何测试回调已被调用及其结果：

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
void asyncFunc(std::function<void(int)> callback) {
    std::thread([callback]() {
        std::this_thread::sleep_for(1s);
        callback(42);
    }).detach();
}
TEST(AsyncTest, TestCallback) {
    int result = 0;
    bool callback_called = false;
    auto callback = & {
        callback_called = true;
        result = value;
    };
    asyncFunc(callback);
    std::this_thread::sleep_for(2s);
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(result, 42);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**TestCallback** 测试仅定义了一个作为 lambda 函数的回调，该 lambda 函数接受一个参数。这个 lambda 函数通过引用捕获存储 **value** 参数的 **result** 变量，以及默认为 **false** 并在回调被调用时设置为 **true** 的 **callback_called** 布尔变量。

然后，测试调用 **asyncFunc()** 函数，该函数启动一个线程，该线程在调用回调并传递值 **42** 之前等待一秒钟。测试在等待两秒钟后使用 **EXPECT_TRUE** 宏检查是否调用了回调，并检查 **callback_called** 的值，以及 **result** 是否具有预期的值 **42**。

## 测试事件驱动软件

我们在 *第九章* 中看到了如何使用 **Boost.Asio** 和其事件队列来调度异步任务。在事件驱动编程中，通常还需要测试回调，如前一个示例所示。我们可以设置测试以注入回调并在它们被调用后验证结果。以下示例展示了如何在 Boost.Asio 程序中测试异步任务：

```cpp
#include <gtest/gtest.h>
#include <boost/asio.hpp>
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
void asyncFunc(boost::asio::io_context& io_context,
               std::function<void(int)> callback) {
    io_context.post([callback]() {
        std::this_thread::sleep_for(100ms);
        callback(42);
    });
}
TEST(AsyncTest, BoostAsio) {
    boost::asio::io_context io_context;
    int result = 0;
    asyncFunc(io_context, &result {
        result = value;
    });
    std::jthread io_thread([&io_context]() {
        io_context.run();
    });
    std::this_thread::sleep_for(150ms);
    EXPECT_EQ(result, 42);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**BoostAsio** 测试首先创建一个 I/O 执行上下文对象 **io_context**，并将其传递给 **asyncFunc()** 函数，同时传递一个 lambda 函数，该 lambda 函数实现一个在后台运行的任务或回调。这个回调简单地设置由 lambda 函数捕获的 **result** 变量的值，将其设置为传递给它的值。

**asyncFunc()** 函数仅使用 **io_context** 来发布一个任务，该任务由一个 lambda 函数组成，该函数在等待 100 毫秒后调用回调并传递值 **42**。

然后，测试只是等待 150 毫秒，直到后台任务完成，并检查结果值是否为 **42**，以标记测试通过。

## 模拟外部资源

如果异步代码还依赖于外部资源，例如文件访问、网络服务器、计时器或其他模块，我们可能需要模拟它们，以避免由于任何资源问题导致的测试失败。模拟和存根是用于在测试目的下用假或简化的对象或函数替换或修改真实对象或函数行为的技巧。这样，我们可以控制异步代码的输入和输出，并避免副作用或其他因素的干扰。

例如，如果测试的代码依赖于服务器，服务器可能无法连接或执行其任务，导致测试失败。在这些情况下，失败是由于资源问题，而不是由于测试的异步代码，导致了一个错误，通常是一个短暂的错误。我们可以通过使用我们自己的模拟类来模拟外部资源，这些模拟类模仿它们的接口。让我们看看如何使用模拟类和使用依赖注入来测试该类的示例。

在这个例子中，有一个外部资源 **AsyncTaskScheduler**，其 **runTask()** 方法用于执行异步任务。因为我们只想测试异步任务并消除异步任务调度器可能产生的任何不期望的副作用，我们可以使用模拟类模仿 **AsyncScheduler** 接口。这个类是 **MockTaskScheduler**，它继承自 **AsyncTaskScheduler** 并实现了其 **runTask()** 基类方法，其中任务是同步运行的：

```cpp
#include <gtest/gtest.h>
#include <functional>
class AsyncTaskScheduler {
   public:
    virtual int runTask(std::function<int()> task) = 0;
};
class MockTaskScheduler : public AsyncTaskScheduler {
   public:
    int runTask(std::function<int()> task) override {
        return task();
    }
};
TEST(AsyncTests, TestDependencyInjection) {
    MockTaskScheduler scheduler;
    auto task = []() -> int {
        return 42;
    };
    int result = scheduler.runTask(task);
    EXPECT_EQ(result, 42);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**TestDependencyInjection** 测试仅创建一个 **MockTaskScheduler** 对象和一个 lambda 函数形式的任务，并使用模拟对象通过运行 **runTask()** 函数来执行任务。一旦任务运行，**result** 将具有值 **42**。

我们不仅可以用 gMock 库完全定义模拟类，还可以只模拟所需的方法。以下示例展示了 gMock 的应用：

```cpp
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <functional>
class AsyncTaskScheduler {
   public:
    virtual int runTask(std::function<int()> task) = 0;
};
class MockTaskScheduler : public AsyncTaskScheduler {
   public:
    MOCK_METHOD(int, runTask, (std::function<int()> task), (override));
};
TEST(AsyncTests, TestDependencyInjection) {
    using namespace testing;
    MockTaskScheduler scheduler;
    auto task = []() -> int {
        return 42;
    };
    EXPECT_CALL(scheduler, runTask(_)).WillOnce(
        Invoke(task)
    );
    auto result = scheduler.runTask(task);
    EXPECT_EQ(result, 42);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

现在，**MockTaskScheduler** 也继承自 **AsyncTaskScheduler**，其中定义了接口，但不是通过重写其方法，而是使用 **MOCK_METHOD** 宏，其中传递了返回类型、模拟方法名称及其参数。

然后，**TestMockMethod** 测试使用 **EXPECT_CALL** 宏来定义对 **MockTaskScheduler** 中 **runTask()** 模拟方法的预期调用，该调用只会发生一次，并调用 lambda 函数任务，该任务返回值 **42**。

该调用仅在下一个指令中发生，其中调用 **scheduler.runTask()**，并将返回值存储在结果中。测试通过检查 **result** 是否是预期的 **42** 值来完成。

## 测试异常和失败

异步任务并不总是成功并生成有效的结果。有时可能会出错（网络故障、超时、异常等），返回错误或抛出异常是通知用户这种情况的方式。我们应该模拟失败以确保代码能够优雅地处理这些情况。

测试错误或异常可以像通常那样进行，通过使用 try-catch 块和使用断言或期望来检查是否抛出了错误，并使测试成功或失败。GoogleTest 还提供了 **EXPECT_ANY_THROW()** 宏，它简化了检查是否发生了异常。以下示例展示了这两种方法：

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
using namespace std::chrono_literals;
int asyncFunc(bool should_fail) {
    std::this_thread::sleep_for(100ms);
    if (should_fail) {
        throw std::runtime_error("Simulated failure");
    }
    return 42;
}
TEST(AsyncTest, TestAsyncFailure1) {
    try {
        std::future<int> result = std::async(
                             std::launch::async,
                             asyncFunc, true);
        result.get();
        FAIL() << "No expected exception thrown";
    } catch (const std::exception& e) {
        SUCCEED();
    }
}
TEST(AsyncTest, TestAsyncFailure2) {
    std::future<int> result = std::async(
                         std::launch::async,
                         asyncFunc, true);
    EXPECT_ANY_THROW(result.get());
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**TestAsyncFailure1** 和 **TestAsyncFailure2** 这两个测试非常相似。它们都异步执行了 **asyncFunc()** 函数，该函数现在接受一个 **should_fail** 布尔参数，指示任务是否应该成功并返回值 **42**，或者失败并抛出异常。两个测试都使任务失败，区别在于 **TestAsyncFailure1** 在没有抛出异常的情况下使用 **FAIL()** 宏使测试失败，或者在 try-catch 块捕获到异常时使用 **SUCCEED()**，而 **TestAsyncFailure2** 使用 **EXPECT_ANY_THROW()** 宏来检查在尝试通过调用其 **get()** 方法从 future result 获取结果时是否发生了异常。

## 测试多个线程

在 C++ 中测试涉及多个线程的异步软件时，一个常见且有效的技术是使用条件变量来同步线程。正如我们在 *第四章* 中所看到的，条件变量允许线程在满足某些条件之前等待，这使得它们对于管理线程间的通信和协调至关重要。

接下来是一个示例，其中多个线程执行一些任务，而主线程等待所有其他线程完成。

让我们先定义一些必要的全局变量，例如线程总数（ **num_threads** ），**counter** 作为每次异步任务被调用时都会增加的原子变量，以及条件变量 **cv** 和其关联的互斥锁 **mtx**，这将有助于在所有异步任务完成后解锁主线程：

```cpp
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <syncstream>
#include <thread>
#include <vector>
using namespace std::chrono_literals;
#define sync_cout std::osyncstream(std::cout)
std::condition_variable cv;
std::mutex mtx;
bool ready = false;
std::atomic<unsigned> counter = 0;
const std::size_t num_threads = 5;
```

**asyncTask()** 函数将在增加 **counter** 原子变量并通过 **cv** 条件变量通知主线程其工作已完成之前执行异步任务（在这个例子中简单等待 100 毫秒）：

```cpp
void asyncTask(int id) {
    sync_cout << "Thread " << id << ": Starting work..."
              << std::endl;
    std::this_thread::sleep_for(100ms);
    sync_cout << "Thread " << id << ": Work finished."
              << std::endl;
    ++counter;
    cv.notify_one();
}
```

**TestMultipleThreads** 测试将首先启动多个线程，每个线程将异步运行 **asyncTask()** 任务。然后，它将等待，使用一个条件变量，其中 **counter** 的值与线程数相同，这意味着所有后台任务都已完成工作。条件变量使用 **wait_for()** 函数设置 150 毫秒的超时时间，以限制测试可以运行的时间，但为所有后台任务成功完成留出一些空间：

```cpp
TEST(AsyncTest, TestMultipleThreads) {
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(asyncTask, i + 1);
    }
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock, 150ms, [] {
            return counter == num_threads;
        });
        sync_cout << "All threads have finished."
                  << std::endl;
    }
    EXPECT_EQ(counter, num_threads);
}
```

测试通过检查确实 **counter** 的值与 **num_threads** 相同来结束。

最后，实现 **main()** 函数：

```cpp
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

如前所述，程序通过调用 **::testing::InitGoogleTest()** 来初始化 GoogleTest 库，然后调用 **RUN_ALL_TESTS()** 来收集和运行所有测试。

## 测试协程

随着 C++20 的推出，协程提供了一种编写和管理异步代码的新方法。基于协程的代码可以通过使用与其他异步代码类似的方法进行测试，但有一个细微的区别，即协程可以挂起和恢复。

让我们用一个简单的协程示例来看看。

我们在 *第八章* 中看到，协程有一些样板代码来定义它们的承诺类型和可等待方法。让我们先实现定义协程的 **Task** 结构。请重新阅读 *第八章* 以全面理解这段代码。

让我们先定义 **Task** 结构：

```cpp
#include <gtest/gtest.h>
#include <coroutine>
#include <exception>
#include <iostream>
struct Task {
    struct promise_type;
    using handle_type =
              std::coroutine_handle<promise_type>;
    handle_type handle_;
    Task(handle_type h) : handle_(h) {}
    ~Task() {
        if (handle_) handle_.destroy();
    }
    // struct promise_type definition
    // and await methods
};
```

在 **Task** 中，我们定义 **promise_type**，它描述了协程是如何管理的。此类型提供了一些预定义的方法（钩子），用于控制值的返回方式、协程的挂起方式以及协程完成后资源的管理方式：

```cpp
struct Task {
    // ...
    struct promise_type {
        int result_;
        std::exception_ptr exception_;
        Task get_return_object() {
            return Task(handle_type::from_promise(*this));
        }
        std::suspend_always initial_suspend() {
            return {};
        }
        std::suspend_always final_suspend() noexcept {
            return {};
        }
        void return_value(int value) {
            result_ = value;
        }
        void unhandled_exception() {
            exception_ = std::current_exception();
        }
    };
    // ....
};
```

然后，实现用于控制协程挂起和恢复的方法：

```cpp
struct Task {
    // ...
    bool await_ready() const noexcept {
        return handle_.done();
    }
    void await_suspend(std::coroutine_handle<>
                           awaiting_handle) {
        handle_.resume();
        awaiting_handle.resume();
    }
    int await_resume() {
        if (handle_.promise().exception_) {
            std::rethrow_exception(
                handle_.promise().exception_);
        }
        return handle_.promise().result_;
    }
    int result() {
        if (handle_.promise().exception_) {
            std::rethrow_exception(
                    handle_.promise().exception_);
        }
        return handle_.promise().result_;
    }
    // ....
};
```

在有了 **Task** 结构之后，让我们定义两个协程，一个用于计算有效值，另一个用于抛出异常：

```cpp
Task asyncFunc(int x) {
    co_return 2 * x;
}
Task asyncFuncWithException() {
    throw std::runtime_error("Exception from coroutine");
    co_return 0;
}
```

由于 GoogleTest 中的 **TEST()** 宏内的测试函数不能直接是协程，因为它们没有与它们关联的 **promise_type** 结构，我们需要定义一些辅助函数：

```cpp
Task testCoroutineHelper(int value) {
    co_return co_await asyncFunc(value);
}
Task testCoroutineWithExceptionHelper() {
    co_return co_await asyncFuncWithException();
}
```

在此基础上，我们现在可以实施测试：

```cpp
TEST(AsyncTest, TestCoroutine) {
    auto task = testCoroutineHelper(5);
    task.handle_.resume();
    EXPECT_EQ(task.result(), 10);
}
TEST(AsyncTest, TestCoroutineWithException) {
    auto task = testCoroutineWithExceptionHelper();
    EXPECT_THROW({
            task.handle_.resume();
            task.result();
        },
        std::runtime_error);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**TestCoroutine** 测试使用 **testCoroutineHelper()** 辅助函数定义了一个任务，并传递了值 **5**。在恢复协程时，预期它将返回双倍值，即 **10**，这通过 **EXPECT_EQ()** 进行测试。

**TestCoroutineWithException** 测试使用类似的方法，但现在使用 **testCoroutineWithExceptionHelper()** 辅助函数，当协程恢复时将抛出异常。这正是 **EXPECT_THROW()** 断言宏在检查确实异常是 **std::runtime_error** 类型之前所发生的事情。

## 压力测试

通过执行压力测试可以实现竞态条件检测。对于高度并发或多线程异步代码，压力测试至关重要。我们可以通过多个异步任务来模拟高负载，以检查系统在压力下的行为是否正确。此外，使用随机延迟、线程交错或压力测试工具也很重要，以减少确定性条件，增加测试覆盖率。

下一个示例展示了实现一个压力测试，该测试启动 100（**total_nums**）个线程执行异步任务，其中原子变量计数器在每个运行后随机等待增加：

```cpp
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
std::atomic<int> counter(0);
const std::size_t total_runs = 100;
void asyncIncrement() {
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 100));
    counter.fetch_add(1);
}
TEST(AsyncTest, StressTest) {
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < total_runs; ++i) {
        threads.emplace_back(asyncIncrement);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(counter, total_runs);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

如果计数器的值与线程总数相同，则测试成功。

## 并行化测试

为了更快地运行测试套件，我们可以并行化在不同线程中运行的测试，但测试必须是独立的，每个测试都在特定的线程中作为一个同步的单线程解决方案运行。此外，它们还需要设置和拆除任何必要的对象，而不会保留之前测试运行的状态。

当使用 CMake 与 GoogleTest 一起时，我们可以通过指定以下命令来并行运行所有检测到的测试：

```cpp
$ ctest –j <num_jobs>
```

本节中展示的所有示例只是测试异步代码可以进行的很小一部分。我们希望这些技术能提供足够的洞察力和知识，以开发进一步的处理特定场景的测试技术。

# 摘要

在本章中，我们学习了如何清理和测试异步程序。

我们首先学习了如何使用 sanitizers 来清理代码，以帮助找到多线程和异步问题，例如竞态条件、内存泄漏和作用域后使用错误等问题。

然后，描述了一些旨在处理异步软件的测试技术，使用 GoogleTest 作为测试库。

使用这些工具和技术有助于检测和预防未定义行为、内存错误和安全漏洞，同时确保并发操作正确执行，正确处理时序问题，并在各种条件下代码按预期执行。这提高了整个程序的整体可靠性和稳定性。

在下一章中，我们将学习可以用来提高异步程序运行时间和资源使用的性能和优化技术。

# 进一步阅读

+   Sanitizers: [`github.com/google/sanitizers`](https://github.com/google/sanitizers)

+   Clang 20.0 ASan: [`clang.llvm.org/docs/AddressSanitizer.html`](https://clang.llvm.org/docs/AddressSanitizer.html)

+   Clang 20.0 硬件辅助 ASan: [`clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html`](https://clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html)

+   Clang 20.0 TSan: [`clang.llvm.org/docs/ThreadSanitizer.html`](https://clang.llvm.org/docs/ThreadSanitizer.html)

+   Clang 20.0 MSan: [`clang.llvm.org/docs/MemorySanitizer.html`](https://clang.llvm.org/docs/MemorySanitizer.html)

+   Clang 20.0 UBSan: [`clang.llvm.org/docs/UndefinedBehaviorSanitizer.html`](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)

+   Clang 20.0 DataFlowSanitizer: [`clang.llvm.org/docs/DataFlowSanitizer.html`](https://clang.llvm.org/docs/DataFlowSanitizer.html)

+   Clang 20.0 LSan: [`clang.llvm.org/docs/LeakSanitizer.html`](https://clang.llvm.org/docs/LeakSanitizer.html)

+   Clang 20.0 RealtimeSanitizer: [`clang.llvm.org/docs/RealtimeSanitizer.html`](https://clang.llvm.org/docs/RealtimeSanitizer.html)

+   Clang 20.0 SanitizerCoverage: [`clang.llvm.org/docs/SanitizerCoverage.html`](https://clang.llvm.org/docs/SanitizerCoverage.html)

+   Clang 20.0 SanitizerStats: [`clang.llvm.org/docs/SanitizerStats.html`](https://clang.llvm.org/docs/SanitizerStats.html)

+   GCC: *程序仪器* *选项*: [`gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html`](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html)

+   Apple 开发者: *早期诊断内存、线程和崩溃问题*: [`developer.apple.com/documentation/xcode/diagnosing-memory-thread-and-crash-issues-early`](https://developer.apple.com/documentation/xcode/diagnosing-memory-thread-and-crash-issues-early)

+   GCC: *调试你的* *程序* 的选项: [`gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html`](https://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html)

+   OpenSSL: *C 和 C++编译器选项加固指南*: [`best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html`](https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html)

+   C 和 C++中的内存错误检查：比较 Sanitizers 和 Valgrind: [`developers.redhat.com/blog/2021/05/05/memory-error-checking-in-c-and-c-comparing-sanitizers-and-valgrind`](https://developers.redhat.com/blog/2021/05/05/memory-error-checking-in-c-and-c-comparing-sanitizers-and-valgrind)

+   GNU C 库: [`www.gnu.org/software/libc`](https://www.gnu.org/software/libc)

+   Sanitizers: 常见标志: [`github.com/google/sanitizers/wiki/SanitizerCommonFlags`](https://github.com/google/sanitizers/wiki/SanitizerCommonFlags)

+   AddressSanitizer 标志: [`github.com/google/sanitizers/wiki/AddressSanitizerFlags`](https://github.com/google/sanitizers/wiki/AddressSanitizerFlags)

+   AddressSanitizer: 快速地址检查器: [`www.usenix.org/system/files/conference/atc12/atc12-final39.pdf`](https://www.usenix.org/system/files/conference/atc12/atc12-final39.pdf)

+   MemorySanitizer: C++中未初始化内存使用的快速检测器: [`static.googleusercontent.com/media/research.google.com/en//pubs/archive/43308.pdf`](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43308.pdf)

+   Linux 内核 Sanitizers: [`github.com/google/kernel-sanitizers`](https://github.com/google/kernel-sanitizers)

+   TSan 标志: [`github.com/google/sanitizers/wiki/ThreadSanitizerFlags`](https://github.com/google/sanitizers/wiki/ThreadSanitizerFlags)

+   TSan：常见的数据竞争: [`github.com/google/sanitizers/wiki/ThreadSanitizerPopularDataRaces`](https://github.com/google/sanitizers/wiki/ThreadSanitizerPopularDataRaces)

+   TSan 报告格式: [`github.com/google/sanitizers/wiki/ThreadSanitizerReportFormat`](https://github.com/google/sanitizers/wiki/ThreadSanitizerReportFormat)

+   TSan 算法: [`github.com/google/sanitizers/wiki/ThreadSanitizerAlgorithm`](https://github.com/google/sanitizers/wiki/ThreadSanitizerAlgorithm)

+   地址空间布局随机化: [`en.wikipedia.org/wiki/Address_space_layout_randomization`](https://en.wikipedia.org/wiki/Address_space_layout_randomization)

+   GoogleTest 用户指南: [`google.github.io/googletest`](https://google.github.io/googletest)
