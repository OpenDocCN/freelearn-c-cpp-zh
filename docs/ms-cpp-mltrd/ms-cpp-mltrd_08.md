# 第八章：原子操作 - 与硬件一起工作

许多优化和线程安全性取决于对底层硬件的理解：从某些架构上的对齐内存访问，到知道哪些数据大小和因此 C++类型可以安全地进行访问而不会有性能惩罚或需要互斥锁等。

本章将介绍如何利用多个处理器架构的特性，例如防止使用互斥锁，其中原子操作将防止任何访问冲突。还将研究诸如 GCC 中的特定于编译器的扩展。

本章主题包括：

+   原子操作的类型以及如何使用它们

+   如何针对特定的处理器架构

+   基于编译器的原子操作

# 原子操作

简而言之，原子操作是处理器可以用单个指令执行的操作。这使得它在没有任何干扰（除了中断）的情况下是原子的，或者可以更改任何变量或数据。

应用包括保证指令执行顺序，无锁实现以及相关用途，其中指令执行顺序和内存访问保证是重要的。

在 2011 年之前的 C++标准中，对处理器提供的原子操作的访问仅由编译器使用扩展提供。

# Visual C++

对于微软的 MSVC 编译器，有 interlocked 函数，从 MSDN 文档总结而来，从添加功能开始：

| **Interlocked 函数** | **描述** |
| --- | --- |
| `InterlockedAdd` | 对指定的`LONG`值执行原子加法操作。 |
| `InterlockedAddAcquire` | 对指定的`LONG`值执行原子加法操作。该操作使用获取内存排序语义执行。 |
| `InterlockedAddRelease` | 对指定的`LONG`值执行原子加法操作。该操作使用释放内存排序语义执行。 |
| `InterlockedAddNoFence` | 对指定的`LONG`值执行原子加法操作。该操作是原子执行的，但不使用内存屏障（在本章中介绍）。 |

这些是该特性的 32 位版本。API 中还有其他方法的 64 位版本。原子函数往往专注于特定的变量类型，但本摘要中省略了此 API 中的变体，以保持简洁。

我们还可以看到获取和释放的变体。这些提供了保证，即相应的读取或写入访问将受到内存重排序（在硬件级别）的保护，以及任何后续的读取或写入操作。最后，无栅栏变体（也称为内存屏障）执行操作而不使用任何内存屏障。

通常，CPU 执行指令（包括内存读取和写入）是为了优化性能而无序执行的。由于这种行为并不总是理想的，因此添加了内存屏障以防止此指令重排序。

接下来是原子`AND`特性：

| **Interlocked 函数** | **描述** |
| --- | --- |
| `InterlockedAnd` | 对指定的`LONG`值执行原子`AND`操作。 |
| `InterlockedAndAcquire` | 对指定的`LONG`值执行原子`AND`操作。该操作使用获取内存排序语义执行。 |
| `InterlockedAndRelease` | 对指定的`LONG`值执行原子`AND`操作。该操作使用释放内存排序语义执行。 |
| `InterlockedAndNoFence` | 对指定的`LONG`值执行原子`AND`操作。该操作是原子执行的，但不使用内存屏障。 |

位测试功能如下：

| **Interlocked 函数** | **描述** |
| --- | --- |
| `InterlockedBitTestAndComplement` | 测试指定的`LONG`值的指定位并对其进行补码。 |
| `InterlockedBitTestAndResetAcquire` | 测试指定的`LONG`值的指定位，并将其设置为`0`。该操作是`原子`的，并且使用获取内存排序语义执行。 |
| `InterlockedBitTestAndResetRelease` | 测试指定的`LONG`值的指定位，并将其设置为`0`。该操作是`原子`的，并且使用内存释放语义执行。 |
| `InterlockedBitTestAndSetAcquire` | 测试指定的`LONG`值的指定位，并将其设置为`1`。该操作是`原子`的，并且使用获取内存排序语义执行。 |
| `InterlockedBitTestAndSetRelease` | 测试指定的`LONG`值的指定位，并将其设置为`1`。该操作是`原子`的，并且使用释放内存排序语义执行。 |
| `InterlockedBitTestAndReset` | 测试指定的`LONG`值的指定位，并将其设置为`0`。 |
| `InterlockedBitTestAndSet` | 测试指定的`LONG`值的指定位，并将其设置为`1`。 |

比较功能可以列举如下：

| **Interlocked function** | **描述** |
| --- | --- |
| `InterlockedCompareExchange` | 对指定的值执行原子比较和交换操作。该函数比较两个指定的 32 位值，并根据比较的结果与另一个 32 位值进行交换。 |
| `InterlockedCompareExchangeAcquire` | 对指定的值执行原子比较和交换操作。该函数比较两个指定的 32 位值，并根据比较的结果与另一个 32 位值进行交换。操作使用获取内存排序语义执行。 |
| `InterlockedCompareExchangeRelease` | 对指定的值执行原子比较和交换操作。该函数比较两个指定的 32 位值，并根据比较的结果与另一个 32 位值进行交换。交换使用释放内存排序语义执行。 |
| `InterlockedCompareExchangeNoFence` | 对指定的值执行原子比较和交换操作。该函数比较两个指定的 32 位值，并根据比较的结果与另一个 32 位值进行交换。操作是原子的，但不使用内存屏障。 |
| `InterlockedCompareExchangePointer` | 对指定的指针值执行原子比较和交换操作。该函数比较两个指定的指针值，并根据比较的结果与另一个指针值进行交换。 |
| `InterlockedCompareExchangePointerAcquire` | 对指定的指针值执行原子比较和交换操作。该函数比较两个指定的指针值，并根据比较的结果与另一个指针值进行交换。操作使用获取内存排序语义执行。 |
| `InterlockedCompareExchangePointerRelease` | 对指定的指针值执行原子比较和交换操作。该函数比较两个指定的指针值，并根据比较的结果与另一个指针值进行交换。操作使用释放内存排序语义执行。 |
| `InterlockedCompareExchangePointerNoFence` | 对指定的值执行原子比较和交换操作。该函数比较两个指定的指针值，并根据比较的结果与另一个指针值进行交换。操作是原子的，但不使用内存屏障。 |

递减功能如下：

| **Interlocked function** | **描述** |
| --- | --- |
| `InterlockedDecrement` | 递减（减少一个）指定 32 位变量的值作为`原子`操作。 |
| `InterlockedDecrementAcquire` | 递减（减少一个）指定 32 位变量的值作为`原子`操作。操作使用获取内存排序语义执行。 |
| `InterlockedDecrementRelease` | 将指定的 32 位变量的值减 1 作为原子操作。操作使用释放内存排序语义执行。 |
| `InterlockedDecrementNoFence` | 将指定的 32 位变量的值减 1 作为原子操作。操作是原子执行的，但不使用内存屏障。 |

交换（交换）功能包括：

| **Interlocked function** | **描述** |
| --- | --- |
| --- |
| `InterlockedExchange` | 将 32 位变量设置为指定值作为原子操作。 |
| `InterlockedExchangeAcquire` | 将 32 位变量设置为指定值作为原子操作。操作使用获取内存排序语义执行。 |
| `InterlockedExchangeNoFence` | 将 32 位变量设置为指定值作为原子操作。操作是原子执行的，但不使用内存屏障。 |
| `InterlockedExchangePointer` | 原子交换一对指针值。 |
| `InterlockedExchangePointerAcquire` | 原子交换一对指针值。操作使用获取内存排序语义执行。 |
| `InterlockedExchangePointerNoFence` | 原子交换一对地址。操作是原子执行的，但不使用内存屏障。 |
| `InterlockedExchangeSubtract` | 执行两个值的原子减法。 |
| `InterlockedExchangeAdd` | 执行两个 32 位值的原子加法。 |
| `InterlockedExchangeAddAcquire` | 执行两个 32 位值的原子加法。操作使用获取内存排序语义执行。 |
| `InterlockedExchangeAddRelease` | 执行两个 32 位值的原子加法。操作使用释放内存排序语义执行。 |
| `InterlockedExchangeAddNoFence` | 执行两个 32 位值的原子加法。操作是原子执行的，但不使用内存屏障。 |

增量功能包括：

| **Interlocked function** | **描述** |
| --- | --- |
| --- |
| `InterlockedIncrement` | 将指定的 32 位变量的值增加 1 作为原子操作。 |
| `InterlockedIncrementAcquire` | 将指定的 32 位变量的值增加 1 作为原子操作。操作使用获取内存排序语义执行。 |
| `InterlockedIncrementRelease` | 将指定的 32 位变量的值增加 1 作为原子操作。操作使用释放内存排序语义执行。 |
| `InterlockedIncrementNoFence` | 将指定的 32 位变量的值增加 1 作为原子操作。操作是原子执行的，但不使用内存屏障。 |

`OR`功能：

| **Interlocked function** | **描述** |
| --- | --- |
| --- |
| `InterlockedOr` | 对指定的`LONG`值执行原子`OR`操作。 |
| `InterlockedOrAcquire` | 对指定的`LONG`值执行原子`OR`操作。操作使用获取内存排序语义执行。 |
| `InterlockedOrRelease` | 对指定的`LONG`值执行原子`OR`操作。操作使用释放内存排序语义执行。 |
| `InterlockedOrNoFence` | 对指定的`LONG`值执行原子`OR`操作。操作是原子执行的，但不使用内存屏障。 |

最后，独占`OR`（`XOR`）功能包括：

| **Interlocked function** | **描述** |
| --- | --- |
| --- |
| `InterlockedXor` | 对指定的`LONG`值执行原子`XOR`操作。 |
| `InterlockedXorAcquire` | 对指定的`LONG`值执行原子`XOR`操作。操作使用获取内存排序语义执行。 |
| `InterlockedXorRelease` | 对指定的`LONG`值执行原子`XOR`操作。操作使用释放内存排序语义执行。 |
| `InterlockedXorNoFence` | 对指定的`LONG`值执行原子`XOR`操作。操作是原子执行的，但不使用内存屏障。 |

# GCC

与 Visual C++一样，GCC 也带有一组内置的原子函数。这些函数根据 GCC 版本和标准库的底层架构而异。由于 GCC 在许多平台和操作系统上的使用要比 VC++多得多，这在考虑可移植性时绝对是一个重要因素。

例如，在 x86 平台上提供的每个内置原子函数都可能不会在 ARM 上可用，部分原因是由于架构差异，包括特定 ARM 架构的变化。例如，ARMv6、ARMv7 或当前的 ARMv8，以及 Thumb 指令集等。

在 C++11 标准之前，GCC 使用`__sync-prefixed`扩展来进行原子操作：

```cpp
type __sync_fetch_and_add (type *ptr, type value, ...) 
type __sync_fetch_and_sub (type *ptr, type value, ...) 
type __sync_fetch_and_or (type *ptr, type value, ...) 
type __sync_fetch_and_and (type *ptr, type value, ...) 
type __sync_fetch_and_xor (type *ptr, type value, ...) 
type __sync_fetch_and_nand (type *ptr, type value, ...) 

```

这些操作从内存中获取一个值并对其执行指定操作，返回内存中的值。这些操作都使用内存屏障。

```cpp
type __sync_add_and_fetch (type *ptr, type value, ...) 
type __sync_sub_and_fetch (type *ptr, type value, ...) 
type __sync_or_and_fetch (type *ptr, type value, ...) 
type __sync_and_and_fetch (type *ptr, type value, ...) 
type __sync_xor_and_fetch (type *ptr, type value, ...) 
type __sync_nand_and_fetch (type *ptr, type value, ...) 

```

这些操作与第一组类似，只是在指定操作后返回新值。

```cpp
bool __sync_bool_compare_and_swap (type *ptr, type oldval, type newval, ...) 
type __sync_val_compare_and_swap (type *ptr, type oldval, type newval, ...) 

```

如果旧值与提供的值匹配，这些比较操作将写入新值。布尔变体在新值被写入时返回 true。

```cpp
__sync_synchronize (...) 

```

该函数创建一个完整的内存屏障。

```cpp
type __sync_lock_test_and_set (type *ptr, type value, ...) 

```

这种方法实际上是一种交换操作，与名称所示不同。它更新指针值并返回先前的值。这不使用完整的内存屏障，而是使用获取屏障，这意味着它不会释放屏障。

```cpp
void __sync_lock_release (type *ptr, ...) 

```

该函数释放了先前方法获得的屏障。

为了适应 C++11 内存模型，GCC 添加了`__atomic`内置方法，这也大大改变了 API：

```cpp
type __atomic_load_n (type *ptr, int memorder) 
void __atomic_load (type *ptr, type *ret, int memorder) 
void __atomic_store_n (type *ptr, type val, int memorder) 
void __atomic_store (type *ptr, type *val, int memorder) 
type __atomic_exchange_n (type *ptr, type val, int memorder) 
void __atomic_exchange (type *ptr, type *val, type *ret, int memorder) 
bool __atomic_compare_exchange_n (type *ptr, type *expected, type desired, bool weak, int success_memorder, int failure_memorder) 
bool __atomic_compare_exchange (type *ptr, type *expected, type *desired, bool weak, int success_memorder, int failure_memorder) 

```

首先是通用的加载、存储和交换函数。它们都相当容易理解。加载函数读取内存中的值，存储函数将值存储在内存中，交换函数将现有值与新值交换。比较和交换函数使交换有条件。

```cpp
type __atomic_add_fetch (type *ptr, type val, int memorder) 
type __atomic_sub_fetch (type *ptr, type val, int memorder) 
type __atomic_and_fetch (type *ptr, type val, int memorder) 
type __atomic_xor_fetch (type *ptr, type val, int memorder) 
type __atomic_or_fetch (type *ptr, type val, int memorder) 
type __atomic_nand_fetch (type *ptr, type val, int memorder) 

```

这些函数基本上与旧 API 中的函数相同，返回特定操作的结果。

```cpp
type __atomic_fetch_add (type *ptr, type val, int memorder) 
type __atomic_fetch_sub (type *ptr, type val, int memorder) 
type __atomic_fetch_and (type *ptr, type val, int memorder) 
type __atomic_fetch_xor (type *ptr, type val, int memorder) 
type __atomic_fetch_or (type *ptr, type val, int memorder) 
type __atomic_fetch_nand (type *ptr, type val, int memorder) 

```

再次，相同的函数，针对新 API 进行了更新。这些函数返回原始值（在操作之前获取）。

```cpp
bool __atomic_test_and_set (void *ptr, int memorder) 

```

与旧 API 中同名的函数不同，该函数执行的是真正的测试和设置操作，而不是旧 API 函数的交换操作，后者仍然需要在之后释放内存屏障。测试是针对某个定义的值。

```cpp
void __atomic_clear (bool *ptr, int memorder) 

```

该函数清除指针地址，将其设置为`0`。

```cpp
void __atomic_thread_fence (int memorder) 

```

可以使用该函数在线程之间创建同步内存屏障（栅栏）。

```cpp
void __atomic_signal_fence (int memorder) 

```

该函数在线程和同一线程内的信号处理程序之间创建内存屏障。

```cpp
bool __atomic_always_lock_free (size_t size, void *ptr) 

```

该函数检查指定大小的对象是否总是为当前处理器架构创建无锁原子指令。

```cpp
bool __atomic_is_lock_free (size_t size, void *ptr) 

```

这基本上与以前的函数相同。

# 内存顺序

在 C++11 内存模型中，并不总是使用内存屏障（栅栏）进行原子操作。在 GCC 内置的原子 API 中，这反映在其函数中的`memorder`参数中。此参数的可能值直接映射到 C++11 原子 API 中的值：

+   `__ATOMIC_RELAXED`：意味着没有线程间的排序约束。

+   `__ATOMIC_CONSUME`：由于 C++11 对`memory_order_consume`的语义存在缺陷，目前使用更强的`__ATOMIC_ACQUIRE`内存顺序来实现。

+   `__ATOMIC_ACQUIRE`：从释放（或更强）语义存储到此获取加载创建线程间的 happens-before 约束

+   `__ATOMIC_RELEASE`：创建一个线程间 happens-before 约束，以获取（或更强）语义加载，从此发布存储读取

+   `__ATOMIC_ACQ_REL`：结合了 `__ATOMIC_ACQUIRE` 和 `__ATOMIC_RELEASE` 的效果。

+   `__ATOMIC_SEQ_CST`：强制与所有其他 `__ATOMIC_SEQ_CST` 操作进行完全排序。

上述列表是从 GCC 手册的关于 GCC 7.1 版本原子的章节中复制的。连同该章节中的注释，这清楚地表明在实现 C++11 原子支持及编译器实现中都做出了权衡。

由于原子依赖于底层硬件支持，永远不会有一个使用原子的代码可以在各种架构上运行。

# 其他编译器

当然，C/C++ 有很多其他编译器工具链，不仅仅是 VC++ 和 GCC，包括英特尔编译器集合（ICC）和其他通常是专有工具。所有这些都有自己的内置原子函数集。幸运的是，由于 C++11 标准，我们现在在编译器之间有了一个完全可移植的原子标准。一般来说，这意味着除了非常特定的用例（或维护现有代码）之外，人们会使用 C++ 标准而不是特定于编译器的扩展。

# C++11 原子

为了使用本机 C++11 原子特性，所有人只需包含 `<atomic>` 头文件。这样就可以使用 `atomic` 类，它使用模板来使自己适应所需的类型，并具有大量预定义的 typedef：

| **类型定义名称** **完全特化** |
| --- |
| `std::atomic_bool` `std::atomic<bool>` |
| `std::atomic_char` `std::atomic<char>` |
| `std::atomic_schar` `std::atomic<signed char>` |
| `std::atomic_uchar` `std::atomic<unsigned char>` |
| `std::atomic_short` `std::atomic<short>` |
| `std::atomic_ushort` `std::atomic<unsigned short>` |
| `std::atomic_int` `std::atomic<int>` |
| `std::atomic_uint` `std::atomic<unsigned int>` |
| `std::atomic_long` `std::atomic<long>` |
| `std::atomic_ulong` `std::atomic<unsigned long>` |
| `std::atomic_llong` `std::atomic<long long>` |
| `std::atomic_ullong` `std::atomic<unsigned long long>` |
| `std::atomic_char16_t` `std::atomic<char16_t>` |
| `std::atomic_char32_t` `std::atomic<char32_t>` |
| `std::atomic_wchar_t` `std::atomic<wchar_t>` |
| `std::atomic_int8_t` `std::atomic<std::int8_t>` |
| `std::atomic_uint8_t` `std::atomic<std::uint8_t>` |
| `std::atomic_int16_t` `std::atomic<std::int16_t>` |
| `std::atomic_uint16_t` `std::atomic<std::uint16_t>` |
| `std::atomic_int32_t` `std::atomic<std::int32_t>` |
| `std::atomic_uint32_t` `std::atomic<std::uint32_t>` |
| `std::atomic_int64_t` `std::atomic<std::int64_t>` |
| `std::atomic_uint64_t` `std::atomic<std::uint64_t>` |
| `std::atomic_int_least8_t` `std::atomic<std::int_least8_t>` |
| `std::atomic_uint_least8_t` `std::atomic<std::uint_least8_t>` |
| `std::atomic_int_least16_t` `std::atomic<std::int_least16_t>` |
| `std::atomic_uint_least16_t` `std::atomic<std::uint_least16_t>` |
| `std::atomic_int_least32_t` `std::atomic<std::int_least32_t>` |
| `std::atomic_uint_least32_t` `std::atomic<std::uint_least32_t>` |
| `std::atomic_int_least64_t` `std::atomic<std::int_least64_t>` |
| `std::atomic_uint_least64_t` `std::atomic<std::uint_least64_t>` |
| `std::atomic_int_fast8_t` `std::atomic<std::int_fast8_t>` |
| `std::atomic_uint_fast8_t` `std::atomic<std::uint_fast8_t>` |
| `std::atomic_int_fast16_t` `std::atomic<std::int_fast16_t>` |
| `std::atomic_uint_fast16_t` `std::atomic<std::uint_fast16_t>` |
| `std::atomic_int_fast32_t` `std::atomic<std::int_fast32_t>` |
| `std::atomic_uint_fast32_t` `std::atomic<std::uint_fast32_t>` |
| `std::atomic_int_fast64_t` `std::atomic<std::int_fast64_t>` |
| `std::atomic_uint_fast64_t` `std::atomic<std::uint_fast64_t>` |
| `std::atomic_intptr_t` `std::atomic<std::intptr_t>` |
| `std::atomic_uintptr_t` | `std::atomic<std::uintptr_t>` |
| `std::atomic_size_t` | `std::atomic<std::size_t>` |
| `std::atomic_ptrdiff_t` | `std::atomic<std::ptrdiff_t>` |
| `std::atomic_intmax_t` | `std::atomic<std::intmax_t>` |
| `std::atomic_uintmax_t` | `std::atomic<std::uintmax_t>` |

这个`atomic`类定义了以下通用函数：

| **函数** | **描述** |
| --- | --- |
| `operator=` | 将值赋给原子对象。 |
| `is_lock_free` | 如果原子对象是无锁的，则返回 true。 |
| `store` | 用非原子参数原子地替换原子对象的值。 |
| `load` | 原子地获取原子对象的值。 |
| `operator T` | 从原子对象中加载值。 |
| `exchange` | 原子地用新值替换对象的值，并返回旧值。 |
| `compare_exchange_weak``compare_exchange_strong` | 原子地比较对象的值，如果相等则交换值，否则返回当前值。 |

使用 C++17 更新，添加了`is_always_lock_free`常量。这允许我们查询类型是否总是无锁。

最后，我们有专门的`atomic`函数：

| **函数** | **描述** |
| --- | --- |
| `fetch_add` | 原子地将参数添加到存储在`atomic`对象中的值，并返回旧值。 |
| `fetch_sub` | 原子地从存储在`atomic`对象中的值中减去参数并返回旧值。 |
| `fetch_and` | 原子地执行参数和`atomic`对象的值之间的按位`AND`操作，并返回旧值。 |
| `fetch_or` | 原子地执行参数和`atomic`对象的值之间的按位`OR`操作，并返回旧值。 |
| `fetch_xor` | 原子地执行参数和`atomic`对象的值之间的按位`XOR`操作，并返回旧值。 |
| `operator++``operator++(int)``operator--``operator--(int)` | 将原子值增加或减少一。 |
| `operator+=``operator-=``operator&=``operator&#124;=``operator^=` | 增加、减少或执行按位`AND`、`OR`、`XOR`操作与原子值。 |

# 示例

使用`fetch_add`的基本示例如下：

```cpp
#include <iostream> 
#include <thread> 
#include <atomic> 

std::atomic<long long> count; 
void worker() { 
         count.fetch_add(1, std::memory_order_relaxed); 
} 

int main() { 
         std::thread t1(worker); 
         std::thread t2(worker); 
         std::thread t3(worker); 
         std::thread t4(worker); 
         std::thread t5(worker); 

         t1.join(); 
         t2.join(); 
         t3.join(); 
         t4.join(); 
         t5.join(); 

         std::cout << "Count value:" << count << '\n'; 
} 

```

这个示例代码的结果将是`5`。正如我们在这里看到的，我们可以用原子方式实现一个基本的计数器，而不必使用任何互斥锁或类似的东西来提供线程同步。

# 非类函数

除了`atomic`类之外，在`<atomic>`头文件中还定义了许多基于模板的函数，我们可以以更类似于编译器内置的原子函数的方式使用。 

| **函数** | **描述** |
| --- | --- |
| `atomic_is_lock_free` | 检查原子类型的操作是否是无锁的。 |
| `atomic_storeatomic_store_explicit` | 原子地用非原子参数替换`atomic`对象的值。 |
| `atomic_load``atomic_load_explicit` | 原子地获取存储在`atomic`对象中的值。 |
| `atomic_exchange``atomic_exchange_explicit` | 原子地用非原子参数替换`atomic`对象的值，并返回`atomic`的旧值。 |
| `atomic_compare_exchange_weak``atomic_compare_exchange_weak_explicit``atomic_compare_exchange_strong``atomic_compare_exchange_strong_explicit` | 原子地比较`atomic`对象的值与非原子参数，并在相等时执行原子交换，否则执行原子加载。 |
| `atomic_fetch_add``atomic_fetch_add_explicit` | 将非原子值添加到`atomic`对象中并获取`atomic`的先前值。 |
| `atomic_fetch_sub``atomic_fetch_sub_explicit` | 从`atomic`对象中减去非原子值并获取`atomic`的先前值。 |
| `atomic_fetch_and``atomic_fetch_and_explicit` | 用非原子参数的逻辑`AND`结果替换`atomic`对象，并获取原子的先前值。 |
| `atomic_fetch_or``atomic_fetch_or_explicit` | 用非原子参数的逻辑`OR`结果替换`atomic`对象，并获取`atomic`的先前值。 |
| `atomic_fetch_xor``atomic_fetch_xor_explicit` | 用非原子参数的逻辑`XOR`结果替换`atomic`对象，并获取`atomic`的先前值。 |
| `atomic_flag_test_and_set``atomic_flag_test_and_set_explicit` | 原子地将标志设置为`true`并返回其先前的值。 |
| `atomic_flag_clear``atomic_flag_clear_explicit` | 原子地将标志的值设置为`false`。 |
| `atomic_init` | 默认构造的`atomic`对象的非原子初始化。 |
| `kill_dependency` | 从`std::memory_order_consume`依赖树中移除指定的对象。 |
| `atomic_thread_fence` | 通用的内存顺序相关的栅栏同步原语。 |
| `atomic_signal_fence` | 线程和在同一线程中执行的信号处理程序之间的栅栏。 |

常规和显式函数之间的区别在于后者允许设置要使用的内存顺序。前者总是使用`memory_order_seq_cst`作为内存顺序。

# 例子

在这个使用`atomic_fetch_sub`的例子中，一个索引容器被多个线程同时处理，而不使用锁：

```cpp
#include <string> 
#include <thread> 
#include <vector> 
#include <iostream> 
#include <atomic> 
#include <numeric> 

const int N = 10000; 
std::atomic<int> cnt; 
std::vector<int> data(N); 

void reader(int id) { 
         for (;;) { 
               int idx = atomic_fetch_sub_explicit(&cnt, 1, std::memory_order_relaxed); 
               if (idx >= 0) { 
                           std::cout << "reader " << std::to_string(id) << " processed item " 
                                       << std::to_string(data[idx]) << '\n'; 
               }  
         else { 
                           std::cout << "reader " << std::to_string(id) << " done.\n"; 
                           break; 
               } 
         } 
} 

int main() { 
         std::iota(data.begin(), data.end(), 1); 
         cnt = data.size() - 1; 

         std::vector<std::thread> v; 
         for (int n = 0; n < 10; ++n) { 
               v.emplace_back(reader, n); 
         } 

         for (std::thread& t : v) { 
               t.join(); 
         } 
} 

```

这个例子代码使用了一个大小为*N*的整数向量作为数据源，用 1 填充它。原子计数器对象设置为数据向量的大小。之后，创建了 10 个线程（使用向量的`emplace_back` C++11 特性在原地初始化），运行`reader`函数。

在那个函数中，我们使用`atomic_fetch_sub_explicit`函数从内存中读取索引计数器的当前值，这使我们能够使用`memory_order_relaxed`内存顺序。这个函数还从这个旧值中减去我们传递的值，将索引减少 1。

只要我们以这种方式获得的索引号大于或等于零，函数就会继续，否则它将退出。一旦所有线程都完成，应用程序就会退出。

# 原子标志

`std::atomic_flag`是一种原子布尔类型。与`atomic`类的其他特化不同，它保证是无锁的。但它不提供任何加载或存储操作。

相反，它提供了赋值运算符，并提供了清除或`test_and_set`标志的函数。前者将标志设置为`false`，后者将测试并将其设置为`true`。

# 内存顺序

这个属性在`<atomic>`头文件中被定义为一个枚举：

```cpp
enum memory_order { 
    memory_order_relaxed, 
    memory_order_consume, 
    memory_order_acquire, 
    memory_order_release, 
    memory_order_acq_rel, 
    memory_order_seq_cst 
}; 

```

在 GCC 部分，我们已经简要涉及了内存顺序的主题。如前所述，这是底层硬件架构特性的一部分。

基本上，内存顺序决定了如何对原子操作周围的非原子内存访问进行排序（内存访问顺序）。这会影响不同线程在执行其指令时如何看到内存中的数据：

| **枚举** | **描述** |
| --- | --- |
| `memory_order_relaxed` | 松散操作：对其他读取或写入没有同步或排序约束，只保证了这个操作的原子性。 |
| `memory_order_consume` | 具有这个内存顺序的加载操作在受影响的内存位置上执行*consume 操作*：当前加载之前不能对当前线程中依赖当前加载的值的读取或写入进行重新排序。释放相同原子变量的其他线程对数据相关变量的写入在当前线程中可见。在大多数平台上，这只影响编译器优化。 |
| `memory_order_acquire` | 具有这种内存顺序的加载操作在受影响的内存位置上执行*获取操作*：在此加载之前，当前线程中的任何读取或写入都不能被重新排序。释放相同原子变量的其他线程中的所有写入在当前线程中是可见的。 |
| `memory_order_release` | 具有这种内存顺序的存储操作执行*释放操作*：在此存储之后，当前线程中的任何读取或写入都不能被重新排序。当前线程中的所有写入对于获取相同原子变量的其他线程是可见的，并且对原子变量进行依赖的写入对于消费相同原子的其他线程是可见的。 |
| `memory_order_acq_rel` | 具有这种内存顺序的读取-修改-写入操作既是*获取操作*又是*释放操作*。当前线程中的任何内存读取或写入都不能在此存储之前或之后重新排序。释放相同原子变量的其他线程中的所有写入在修改之前可见，并且在获取相同原子变量的其他线程中修改是可见的。 |
| `memory_order_seq_cst` | 具有这种内存顺序的任何操作既是*获取操作*又是*释放操作*，并且存在一个单一的总顺序，所有线程以相同的顺序观察到所有修改。 |

# 松散排序

在松散内存排序中，不对并发内存访问强制执行任何顺序。这种类型的排序仅保证原子性和修改顺序。

这种类型的排序的典型用途是用于计数器，无论是递增还是递减，就像我们在上一节的示例代码中看到的那样。

# 释放-获取排序

如果线程 A 中的原子存储被标记为`memory_order_release`，并且线程 B 中从相同变量的原子加载被标记为`memory_order_acquire`，则从线程 A 的视角来看，所有在原子存储之前发生的内存写入（非原子和松散原子）都会在线程 B 中变为*可见副作用*。也就是说，一旦原子加载完成，线程 B 将保证看到线程 A 写入内存的所有内容。

这种类型的操作在所谓的强排序架构上是自动的，包括 x86、SPARC 和 POWER。弱排序架构，如 ARM、PowerPC 和 Itanium，将需要在这里使用内存屏障。

这种类型的内存排序的典型应用包括互斥机制，如互斥锁或原子自旋锁。

# 释放-获取排序

如果线程 A 中的原子存储被标记为`memory_order_release`，并且线程 B 中从相同变量的原子加载被标记为`memory_order_consume`，则从线程 A 的视角来看，所有在原子存储之前*依赖排序*的内存写入（非原子和松散原子）都会在线程 B 的操作中变为*可见副作用*，这些操作使用了从加载操作中获得的值。也就是说，一旦原子加载完成，线程 B 中使用从加载中获得的值的运算符和函数将保证看到线程 A 写入内存的内容。

这种类型的排序在几乎所有架构上都是自动的。唯一的主要例外是（过时的）Alpha 架构。这种类型排序的典型用例是对很少更改的数据进行读取访问。

截至 C++17，这种内存排序正在进行修订，暂时不鼓励使用`memory_order_consume`。

# 顺序一致性排序

标记为`memory_order_seq_cst`的原子操作不仅以与释放-获取排序相同的方式对内存进行排序（在一个线程中存储之前发生的所有事情都成为*可见副作用*在执行加载的线程中），而且还建立了所有标记为这种方式的原子操作的*单一总修改顺序*。

这种排序可能在所有消费者必须以完全相同的顺序观察其他线程所做的更改的情况下是必要的。这会导致在多核或多 CPU 系统上需要完整的内存屏障。

由于这种复杂的设置，这种排序比其他类型要慢得多。它还要求每个原子操作都必须带有这种类型的内存排序标记，否则顺序排序将会丢失。

# Volatile 关键字

`volatile`关键字对于任何曾经编写过复杂多线程代码的人来说可能非常熟悉。它的基本用途是告诉编译器相关变量应始终从内存中加载，不要对其值进行任何假设。它还确保编译器不会对变量进行任何激进的优化。

对于多线程应用程序来说，它通常是无效的，因此不建议使用。`volatile`关键字的主要问题在于它没有定义多线程内存模型，这意味着这个关键字的结果可能在不同平台、CPU 甚至工具链上都不确定。

在原子操作领域，不需要使用这个关键字，事实上，使用它可能并不会有帮助。为了确保获取在多个 CPU 核心和它们的缓存之间共享的变量的当前版本，必须使用像`atomic_compare_exchange_strong`、`atomic_fetch_add`或`atomic_exchange`这样的操作，让硬件获取正确和当前的值。

对于多线程代码，建议不要使用`volatile`关键字，而是使用原子操作来保证正确的行为。

# 总结

在本章中，我们看了原子操作以及它们是如何被集成到编译器中的，以使代码尽可能地与底层硬件配合。读者现在将熟悉原子操作的类型、内存屏障（fencing）的使用，以及各种内存排序及其影响。

读者现在能够在自己的代码中使用原子操作来实现无锁设计，并正确使用 C++11 内存模型。

在下一章中，我们将把迄今为止学到的一切都放在一起，远离 CPU，转而看看 GPGPU，即在显卡（GPU）上对数据进行通用处理。
