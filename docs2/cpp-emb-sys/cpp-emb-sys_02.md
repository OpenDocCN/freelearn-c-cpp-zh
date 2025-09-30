

# 第一章：澄清关于 C++的常见误解

为微控制器和嵌入式系统编写软件具有挑战性。为了充分利用资源受限的系统，嵌入式开发者需要对平台架构有良好的了解。他们需要了解可用的资源，包括处理器能力、可用内存和外设。通过内存映射外设直接访问硬件的需求使得**C**成为嵌入式系统半个世纪以来的首选语言。

任何编程语言的目标都是将应用特定的抽象转换为可转换为机器代码的代码。例如，**面向商业的通用语言**（**COBOL**）用于银行应用，而**Fortran**用于科学研究和大型的数学计算。另一方面，C 是一种通用编程语言，常用于**操作系统**（**OSs**）和嵌入式系统应用。

C 是一种语法简单、易于学习的语言。语法简单意味着它无法表达复杂的思想。与抽象这些细节的高级语言相比，C 允许进行复杂操作，但需要更明确和详细的代码来管理复杂性。

在 20 世纪 70 年代末，高级语言的性能无法达到 C 的水平。这促使丹麦计算机科学家 Bjarne Stroustrup 开始研究**带类的 C**，它是 C++的前身。如今，C++是一种以性能为设计目标的泛型语言。C++的起源仍然是某些神话的来源，这常常导致人们在嵌入式系统编程中对其犹豫不决。本章将向你介绍这些神话，并对其进行澄清。本章将涵盖以下主题：

+   C++简史

+   带类的 C

+   肥大和运行时开销

# 技术要求

为了充分利用本章内容，我强烈建议你在阅读示例时使用编译器探索器([`godbolt.org/`](https://godbolt.org/))。选择 GCC 作为你的编译器，并针对 x86 架构。这将允许你看到标准输出（stdio）结果，并更好地观察代码的行为。本章的示例可在 GitHub 上找到([`github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter01`](https://github.com/PacktPublishing/Cpp-in-Embedded-Systems/tree/main/Chapter01))。

# C++简史

在 20 世纪 60 年代中期，模拟编程语言**SIMULA**将类和对象引入了软件开发的世界。**类**是一种抽象，它允许我们以简洁的方式在编程中表示现实世界概念，使代码更易于人类阅读。在嵌入式开发中，**UART**、**SPI**、**TemperatureSensor**、**PidController**和**TemperatureController**是一些可以以类形式实现的概念。SIMULA 还引入了类之间的层次关系。例如，`PT100`类也是`TemperatureSensor`类，而`TemperatureController`类有一个`TemperatureSensor`成员实例（对象）和一个`PidController`。这被称为**面向对象编程**（**OOP**）。

在反思编程语言的演变时，C++的创造者 Bjarne Stroustrup 分享了他设计 C++的方法。Stroustrup 的目标是在高级抽象和低级效率之间架起桥梁。他说了以下内容：

> 我的想法非常简单。从 SIMULA 中汲取一般抽象的想法，以利于人类表示事物，使人类能够理解，同时使用低级的东西，当时最好的语言是 C，这是在贝尔实验室由 Dennis Ritchie 完成的。将这两个想法结合起来，以便可以进行高级抽象，同时足够高效且足够接近硬件，以处理真正要求高的计算任务。

C++最初是由 Bjarne Stroustrup 以 C with Classes 开始，演变成一种现代编程语言，它仍然提供了对硬件和内存映射外设的直接访问。使用强大的抽象，C++使得编写表达性和高度模块化的代码成为可能。C++是一种通用、多范式的语言，支持过程式、面向对象编程，并在一定程度上支持函数式编程范式。

虽然 C 语言仍然是嵌入式开发的首选语言，占嵌入式项目的 60%，但 C++的采用率稳步增长。在嵌入式开发领域的估计使用率为 20-30%，C++提供了类、改进的类型安全和编译时计算等功能。

尽管 C++提供了许多功能，但在嵌入式编程中，C 语言仍然占据主导地位。有许多原因，本章将讨论其中的一些。C++比 C 语言更复杂，这使得初学者开发者更难上手。C 语言更容易学习，并使得初学者开发者能够更快地参与到项目中。

C 语言的简洁性很好，因为它允许初学者开发者更快地开始为项目做出贡献，但它也使得编写复杂的逻辑过于冗长。这通常会导致代码库更大，因为缺乏表现力。这就是 C++介入的地方，它提供了更高的抽象层次，如果得到充分利用，可以使代码更容易阅读和理解。

C++未被更广泛采用的其他原因与关于 C++的神话有关。人们仍然认为 C++仅仅是“带有类的 C”，或者由于标准库中的动态内存分配，使用 C++对于安全性至关重要的系统是完全不可接受的，或者它会产生膨胀代码并增加空间和时间开销。本章将在嵌入式开发背景下解决一些关于 C++最普遍的神话。让我们揭开这些神话，为嵌入式系统中的 C++带来新的光芒！

# C with Classes

从历史的角度来看，C++最初是 C with Classes。第一个 C++编译器**Cfront**将 C++转换为 C，但这已经是很久以前的事情了。随着时间的推移，C 和 C++分别发展，现在由不同的语言标准定义。C 保持了其简洁性，而 C++已经发展成为一门现代语言，它能够为问题提供抽象解决方案，而不牺牲性能水平。但 C++有时仍然被称为 C with Classes，这暗示 C++除了类之外没有增加任何价值。

C++11 标准于 2011 年发布，是 C++的第二大版本。它包含了许多使语言现代化的功能，如基于范围的循环、lambda 和`constexpr`。随后的版本，C++14、C++17、C++20 和 C++23，继续使语言现代化并引入了使 C with Classes 仅仅成为现代 C++的一个遥远前辈的功能。

## Modern C++

为了证明 C++不仅仅是 C with Classes，让我们探索几个简短的 C 代码示例及其现代 C++等价物。让我们从一个简单的示例开始，即从整数缓冲区打印元素：

```cpp
#define N 20
int buffer[N];
for(int i = 0; i < N; i ++) {
    printf("%d ", buffer[i]);
} 
```

上述 C 代码可以转换为以下 C++代码：

```cpp
std::array<int, 20> buffer;
for(const auto& element : buffer) {
    printf("%d ", element);
} 
```

我们首先注意到的是 C++版本的长度更短。它包含的单词更少，并且比 C 代码更接近英语。它更容易阅读。现在，如果你来自 C 背景并且没有接触过高级语言，第一个版本可能看起来更容易阅读，但让我们比较一下。我们首先注意到的是 C 代码定义了常量`N`，它决定了`buffer`的大小。这个常量用于定义`buffer`，并作为`for`循环的边界。

C++11 中引入的基于范围的循环消除了在循环停止条件中使用容器大小的认知负担。大小信息已经包含在`std::array`容器中，基于范围的循环利用这个容器轻松地遍历数组。此外，没有对缓冲区的索引，因为元素是通过常量引用访问的，确保在循环内部不会修改元素。

让我们看看一些简单的 C 代码，它将`array_a`整数中的所有元素复制到`array_b`，如果元素小于`10`：

```cpp
int w_idx = 0;
for(int i = 0; i < sizeof(array_a)/sizeof(int); i++) {
    if(array_a[i] < 10) {
        array_b[w_idx++] = array_a[i];
    }
} 
```

下面是具有相同功能的 C++代码：

```cpp
auto less_than_10 =  [](auto x) -> bool {
    return x < 10;
};
std::copy_if(std::begin(array_a), std::end(array_a), std::begin(array_b), less_than_10); 
```

而不是手动遍历`array_a`并将超过`10`的元素复制到`array_b`中，我们可以使用 C++标准模板库中的`copy_if`函数。`std::copy_if`的前两个参数是迭代器，它们定义了在`array_a`中要考虑的元素范围：第一个迭代器指向数组的开始，第二个迭代器指向最后一个元素之后的位子。第三个参数是指向`array_b`起始位置的迭代器，第四个是`less_than_10` lambda 表达式。

Lambda 表达式是一个匿名函数对象，可以在调用它的位置声明，或者将其作为参数传递给函数。请注意，Lambda 将在*第十章*中更详细地介绍。在`std::copy_if`的情况下，`less_than_10` lambda 用于确定`array_a`中的元素是否要复制到`array_b`。我们也可以定义一个独立的`less_than_10`函数，该函数接受一个整数并返回一个布尔值，如果它大于 10，但使用 lambda，我们可以将此功能编写得接近我们将其传递给算法的位置，这使得代码更加紧凑和表达。

## 泛型类型

之前的示例使用了`std::array`标准库容器。它是一个类模板，它包装了一个 C 风格数组及其大小信息。请注意，模板将在*第八章*中更详细地介绍。当你使用具有特定底层类型和大小的`std::array`时，编译器在实例化的过程中定义了一个新类型。

`std::array<int, 10>`创建了一个容器类型，它有一个底层大小为`10`的整数 C 风格数组。`std::array<int, 20>`是一个容器类型，它有一个底层大小为`20`的整数 C 风格数组。`std::array<int, 10>`和`std::array<int, 20>`是不同的类型。它们具有相同的底层类型，但大小不同。

`std::array<float, 10>`会产生第三种类型，因为它与`std::array<int, 10>`在底层类型上不同。使用不同的参数会产生不同的类型。模板类型是泛型类型，只有在实例化时才会成为具体类型。

为了更好地理解泛型类型并欣赏它们，让我们检查 C 语言中环形缓冲区的实现，并将其与 C++中基于模板的解决方案进行比较。

### C 语言中的环形缓冲区

**环形**或**循环缓冲区**是嵌入式编程中常用的数据结构。它通常通过一组函数实现，这些函数围绕一个数组，使用写和读索引来访问数组的元素。`count`变量用于数组空间管理。接口由 push 和 pop 函数组成，这里将进行解释：

+   **推送**函数用于将元素存储在环形缓冲区中。在每次推送时，数据元素存储在数组中，并增加写入索引。如果写入索引等于数据数组中的元素数量，则将其重置为 0。

+   **弹出**函数用于从环形缓冲区中检索一个元素。在每次弹出时，如果底层数组不为空，我们返回数组中由读取索引索引的元素。我们增加读取索引。

在每次推送时，我们增加`count`变量，在弹出时减少它。如果计数等于数据数组的大小，我们需要将读取索引向前移动。

让我们定义我们想在 C 模块中实现的环形缓冲区的实现要求：

+   它不应该使用动态内存分配

+   当缓冲区满时，我们将覆盖最旧的元素

+   为存储数据到缓冲区并检索它提供推送和弹出功能

+   整数将被存储在环形缓冲区中

这里是满足先前要求的 C 语言简单解决方案：

```cpp
#include <stdio.h>
#define BUFFER_SIZE 5
typedef struct {
int arr[BUFFER_SIZE]; // Array to store int values directly
size_t write_idx;     // Index of the next element to write (push)
size_t read_idx;      // Index of the next element to read (pop)
size_t count;         // Number of elements in the buffer
} int_ring_buffer;
void int_ring_buffer_init(int_ring_buffer *rb) {
  rb->write_idx = 0;
  rb->read_idx = 0;
  rb->count = 0;
}
void int_ring_buffer_push(int_ring_buffer *rb, int value) {
  rb->arr[rb->write_idx] = value;
  rb->write_idx = (rb->write_idx + 1) % BUFFER_SIZE;
  if (rb->count < BUFFER_SIZE) {
    rb->count++;
  } else {
    // Buffer is full, move read_idx forward
    rb->read_idx = (rb->read_idx + 1) % BUFFER_SIZE;
  }
}
int int_ring_buffer_pop(int_ring_buffer *rb) {
  if (rb->count == 0) {
    return 0;
  }
  int value = rb->arr[rb->read_idx];
  rb->read_idx = (rb->read_idx + 1) % BUFFER_SIZE;
  rb->count--;
  return value;
}
int main() {
  int_ring_buffer rb;
  int_ring_buffer_init(&rb);
  for (int i = 0; i < 10; i++) {
    int_ring_buffer_push(&rb, i);
  }
  while (rb.count > 0) {
    int value = int_ring_buffer_pop(&rb);
    printf("%d\n", value);
  }
  return 0;
} 
```

我们使用`for`循环来初始化缓冲区。由于缓冲区大小为`5`，值从`5`到`9`将存储在缓冲区中，因为环形缓冲区会覆盖现有数据。现在，如果我们想在环形缓冲区中存储浮点数、字符或用户定义的数据结构怎么办？我们可以为不同类型实现相同的逻辑，并创建一组新的数据结构和函数，称为`float_ring_buffer`或`char_ring_buffer`。我们能否创建一个可以存储不同数据类型并使用相同函数的解决方案？

我们可以使用`unsigned char`数组作为不同数据类型的存储，并使用`void`指针将不同数据类型传递给推送和弹出函数。唯一缺少的是知道数据类型的大小，我们可以通过向`ring_buffer`结构添加`size_t elem_size`成员来解决：

```cpp
#include <stdio.h>
#include <string.h>
#define BUFFER_SIZE 20 // Total bytes available in the buffer
typedef struct {
unsigned char data[BUFFER_SIZE]; // Array to store byte values
size_t write_idx;                // Index of the next byte to write
size_t read_idx;                 // Index of the next byte to read
size_t count;     // Number of bytes currently used in the buffer
size_t elem_size; // Size of each element in bytes
} ring_buffer;
void ring_buffer_init(ring_buffer *rb, size_t elem_size) {
  rb->write_idx = 0;
  rb->read_idx = 0;
  rb->count = 0;
  rb->elem_size = elem_size;
}
void ring_buffer_push(ring_buffer *rb, void *value) {
  if (rb->count + rb->elem_size <= BUFFER_SIZE) {
    rb->count += rb->elem_size;
  } else {
    rb->read_idx = (rb->read_idx + rb->elem_size) % BUFFER_SIZE;
  }
  memcpy(&rb->data[rb->write_idx], value, rb->elem_size);
  rb->write_idx = (rb->write_idx + rb->elem_size) % BUFFER_SIZE;
}
int ring_buffer_pop(ring_buffer *rb, void *value) {
  if (rb->count < rb->elem_size) {
    // Not enough data to pop
return 0;
  }
  memcpy(value, &rb->data[rb->read_idx], rb->elem_size);
  rb->read_idx = (rb->read_idx + rb->elem_size) % BUFFER_SIZE;
  rb->count -= rb->elem_size;
  return 1; // Success
}
int main() {
  ring_buffer rb;
  ring_buffer_init(&rb, sizeof(int)); // Initialize buffer for int values
for (int i = 0; i < 10; i++) {
    int val = i;
    ring_buffer_push(&rb, &val);
  }
  int pop_value;
  while (ring_buffer_pop(&rb, &pop_value)) {
    printf("%d\n", pop_value);
  }
  return 0;
} 
```

此环形缓冲区解决方案可以用于存储不同数据类型。由于我们避免了使用动态内存分配，并且`data`缓冲区大小是在编译时确定的，因此我们在定义环形缓冲区不同实例所需的内存大小时不够灵活。我们遇到的另一个问题是类型安全。我们可以轻松地用指向浮点数的指针调用`ring_buffer_push`，用指向整数的指针调用`ring_buffer_pop`。编译器无法解决这个问题，灾难的可能性是真实的。此外，通过使用`void`指针，我们增加了一层间接引用，因为我们必须依赖内存从数据缓冲区检索数据。

我们能否解决类型安全问题，并使在 C 中定义环形缓冲区的大小成为可能？我们可以使用标记粘贴（`##`）运算符为不同类型和大小创建一组函数，使用宏。在跳入使用此技术实现的环形缓冲区实现之前，让我们快速通过使用`##`运算符的简单示例：

```cpp
#include <stdio.h>
// Macro to define a function for summing two numbers
#define DEFINE_SUM_FUNCTION(TYPE) \
TYPE sum_##TYPE(TYPE a, TYPE b) { \
    return a + b; \
}
// Define sum functions for int and float
DEFINE_SUM_FUNCTION(int)
DEFINE_SUM_FUNCTION(float)
int main() {
    int result_int = sum_int(5, 3);
    printf("Sum of integers: %d\n", result_int);
    float result_float = sum_float(3.5f, 2.5f);
    printf("Sum of floats: %.2f\n", result_float);
    return 0;
} 
```

`DEFINE_SUM_FUNCTION(int)`将创建一个接受并返回整数的`sum_int`函数。如果我们用`float`调用`DEFINE_SUM_FUNCTION`宏，它将导致创建`sum_float`。现在我们已经很好地理解了标记粘贴操作符，让我们继续环形缓冲区的实现：

```cpp
#include <stdio.h>
#include <string.h>
// Macro to declare ring buffer type and functions for a specific type and size
#define DECLARE_RING_BUFFER(TYPE, SIZE) \
typedef struct { \
    TYPE data[SIZE]; \
    size_t write_idx; \
    size_t read_idx; \
    size_t count; \
} ring_buffer_##TYPE##_##SIZE; \
void ring_buffer_init_##TYPE##_##SIZE(ring_buffer_##TYPE##_##SIZE *rb) { \
    rb->write_idx = 0; \
    rb->read_idx = 0; \
    rb->count = 0; \
} \
void ring_buffer_push_##TYPE##_##SIZE(ring_buffer_##TYPE##_##SIZE *rb, TYPE value) { \
    rb->data[rb->write_idx] = value; \
    rb->write_idx = (rb->write_idx + 1) % SIZE; \
    if (rb->count < SIZE) { \
        rb->count++; \
    } else { \
        rb->read_idx = (rb->read_idx + 1) % SIZE; \
    } \
} \
int ring_buffer_pop_##TYPE##_##SIZE(ring_buffer_##TYPE##_##SIZE *rb, TYPE *value) { \
    if (rb->count == 0) { \
        return 0; /* Buffer is empty */ \
    } \
    *value = rb->data[rb->read_idx]; \
    rb->read_idx = (rb->read_idx + 1) % SIZE; \
    rb->count--; \
    return 1; /* Success */ \
}
// Example usage with int type and size 5
DECLARE_RING_BUFFER(int, 5) // Declare the ring buffer type and functions for integers
int main() {
    ring_buffer_int_5 rb;
    ring_buffer_init_int_5(&rb); // Initialize the ring buffer
// Push values into the ring buffer
for (int i = 0; i < 10; ++i) {
        ring_buffer_push_int_5(&rb, i);
    }
    // Pop values from the ring buffer and print them
int value;
    while (ring_buffer_pop_int_5(&rb, &value)) {
        printf("%d\n", value);
    }
    return 0;
} 
```

现在，这个解决方案解决了我们的类型安全和定义环形缓冲区大小的难题，但它在实现和使用时都存在可读性问题。我们需要在任意函数之外“调用”`DECLARE_RING_BUFFER`，因为它基本上是一个定义了一组函数的宏。我们还需要了解它所执行的操作以及它将生成的函数的签名。我们可以通过模板做得更好。让我们看看 C++中环形缓冲区的实现是什么样的。

### C++中的环形缓冲区

让我们使用模板制作一个通用的环形缓冲区实现。我们可以使用`std::array`类模板作为底层类型，并将我们的推入和弹出逻辑围绕它包装。以下是在 C++中`ring_buffer`类型可能看起来如何的代码示例：

```cpp
#include <array>
#include <cstdio>
template <class T, std::size_t N> struct ring_buffer {
  std::array<T, N> arr;
  std::size_t write_idx = 0; // Index of the next element to write (push)
  std::size_t read_idx = 0;  // Index of the next element to read (pop)
  std::size_t count = 0;     // Number of elements in the buffer
void push(T t) {
    arr.at(write_idx) = t;
    write_idx = (write_idx + 1) % N;
    if (count < N) {
      count++;
    } else {
      // buffer is full, move forward read_idx
      read_idx = (read_idx + 1) % N;
    }
  }
  T pop() {
    if (count == 0) {
      // Buffer is empty, return a default-constructed T.
return T{};
    }
    T value = arr.at(read_idx);
    read_idx = (read_idx + 1) % N;
    --count;
    return value;
  }
  bool is_empty() const { return count == 0; }
};
int main() {
  ring_buffer<int, 5> rb;
  for (int i = 0; i < 10; ++i) {
    rb.push(i);
  }
  while (!rb.is_empty()) {
    printf("%d\n", rb.pop());
  }
  return 0;
} 
```

使用模板在 C++中实现的环形缓冲区比 C 中基于标记粘贴的解决方案更易于阅读和使用。`ring_buffer`模板类可以用来实例化具有不同大小的整数、float 或其他底层类型的环形缓冲区类型。相同的推入和弹出逻辑可以应用于具有不同底层类型的环形缓冲区。我们可以通过模板将**DRY**原则应用于不同的类型。**模板**使得泛型类型的实现变得简单，这在 C 中相当具有挑战性和冗长。

模板也被用于**模板元编程**（**TMP**），这是一种编程技术，其中编译器使用模板生成临时源代码，然后编译器将其与源代码的其余部分合并，并最终编译。TMP 最著名的例子之一是在编译时计算**阶乘**。TMP 是一种高级技术，将在*第八章*中介绍。现代 C++还引入了`constexpr`指定符，这是一种更易于初学者使用的编译时计算技术。

## `constexpr`

C++11 引入了`constexpr`指定符，它声明了在编译时评估函数或变量的值是可能的。指定符随着时间的推移而演变，扩展了其功能。一个`constexpr`变量必须立即初始化，并且其类型必须是`literal`类型（int、float 等）。这就是我们声明`constexpr`变量的方式：

```cpp
constexpr double pi = 3.14159265359; 
```

使用`constexpr`指定符是声明 C++中编译时常量的首选方法，而不是使用 C 风格的宏方法。让我们分析一个使用 C 风格宏的简单示例：

```cpp
#include <cstdio>
#define VOLTAGE 3300
#define CURRENT 1000
int main () {
    const float resistance = VOLTAGE / CURRENT;
    printf("resistance = %.2f\r\n", resistance);
    return 0;
} 
```

```cpp
The output of this simple program might be surprising:
resistance = 3.00 
```

`VOLTAGE`和`CURRENT`都被解析为整数`字面量`，除法的结果也是如此。使用`f`后缀声明浮点`字面量`，在这个例子中省略了。使用`constexpr`来定义编译时常量更安全，因为它允许我们指定常量的类型。这就是我们如何使用`constexpr`编写相同示例的方法：

```cpp
#include <cstdio>
constexpr float voltage = 3300;
constexpr float current = 1000;
int main () {
    const float resistance = voltage / current;
    printf("resistance = %.2f\r\n", resistance);
    return 0;
} 
```

```cpp
This would result in
resistance = 3.30 
```

这个简单的例子表明，`constexpr`编译时常量比传统的 C 风格宏常量更安全、更容易阅读。`constexpr`指定符的另一个主要用途是向编译器暗示一个函数可以在编译时评估。一个`constexpr`函数必须满足的一些要求如下：

+   返回类型必须是一个`字面量`类型

+   函数的每个参数必须是一个`字面量`类型

+   如果`constexpr`函数不是一个构造函数，它需要恰好有一个`return`语句

让我们考察一个使用`constexpr`函数的简单例子：

```cpp
int square(int a) {
    return a*a;
}
int main () {
    int ret = square(2);
    return ret;
} 
```

为了更好地理解底层发生了什么，我们将检查前面代码的汇编输出。汇编代码非常接近机器代码，或者将在我们的目标上执行的指令，因此检查它给我们提供了处理器执行的工作（指令数）的估计。以下是在没有优化的情况下，使用 ARM GCC 编译器为 ARM 架构编译前面程序生成的汇编输出：

```cpp
square(int):
        push    {r7}
        sub sp, sp, #12
add r7, sp, #0
str r0, [r7, #4]
        ldr r3, [r7, #4]
        mul r3, r3, r3
mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr
main:
push    {r7, lr}
        sub sp, sp, #8
add r7, sp, #0
movs r0, #2
bl      square(int)
        str r0, [r7, #4]
        ldr r3, [r7, #4]
        mov r0, r3
adds r7, r7, #8
mov sp, r7
pop     {r7, pc} 
```

生成的汇编代码正在执行以下操作：

+   操作栈指针

+   调用`square`函数

+   将`r0`返回的值存储到`r7`地址中，偏移量为`4`

+   从偏移量为`4`的`r7`地址中加载值到`r3`

+   将`r3`的值移动到`r0`，这是 ARM 调用约定用于存储返回值的指定寄存器

我们可以看到输出二进制文件中存在一些不必要的操作，这既增加了二进制文件的大小，又影响了性能。这个例子是有效的 C 和有效的 C++代码，使用 C 和 C++编译器编译它将产生相同的汇编代码。

如果我们为`square`函数使用`constexpr`指定符，我们是在指示编译器它在编译时可以评估它：

```cpp
constexpr int square(int a) {
    return a*a;
}
int main() {
    constexpr int val = square(2);
    return ret;
} 
```

这段代码导致`square(2)`表达式的编译时评估，使`val`整数成为一个`constexpr`变量，即编译时常量。以下是将生成的汇编代码：

```cpp
main:
push    {r7}
        sub sp, sp, #12
add r7, sp, #0
movs r3, #4
str r3, [r7, #4]
        movs r3, #4
mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr 
```

如我们所见，程序返回的值是`4`，这是`square(2)`编译时计算的结果。生成的汇编代码中没有`square`函数，只有编译器为我们执行的计算结果。这个简单的例子展示了编译时计算的力量。当我们知道所有计算参数时，我们可以将重计算从运行时移到编译时，这通常是可能的。这种方法可以用来生成查找表或复杂的数学信号，这将在本书的后续章节中演示。

自从 C with Classes 以来，C++已经走了很长的路。本章中的示例展示了 C++相对于 C 所能提供的功能——表达性强、可读性高、紧凑的代码；标准模板库容器；算法；用户定义的泛型类型；以及编译时计算，仅举几例。我希望我已经成功地打破了 C++只是 C 带类这一神话。关于 C++的下一个常见神话是它会产生臃肿的代码并增加运行时开销。让我们继续打破关于 C++的神话！

# 肿胀和运行时开销

**bloatware**这个术语描述的是在设备上预装操作系统的不需要的软件。在编程世界中，不需要的软件描述的是框架、库或语言构造本身插入到二进制中的代码。在 C++中，被指责为导致代码臃肿的语言构造是构造函数、析构函数和模板。我们将通过检查从 C++代码生成的汇编输出来分析这些误解。

## 构造函数和析构函数

当你提到 C++时，非 C++开发者首先想到的可能是它是一种面向对象的语言，并且你必然会实例化对象。对象是类的实例。它们是占用内存的变量。称为**构造函数**的特殊函数用于构建或实例化对象。

构造函数用于初始化对象，包括类成员的初始化，而析构函数用于清理资源。它们与对象的生存周期紧密相关。对象通过构造函数创建，当对象变量超出作用域时，会调用**析构函数**。

构造函数和析构函数都会增加二进制文件的大小并增加运行时开销，因为它们的执行需要时间。我们将通过一个简单的类示例来检查构造函数和析构函数的影响，该类有一个私有成员、一个构造函数、一个析构函数和一个获取器：

```cpp
class MyClass
{
    private:
         int num;
    public:
        MyClass(int t_num):num(t_num){}
        ~MyClass(){}
        int getNum() const {
            return num;
        }
};
int main () {
   MyClass obj(1);
   return obj.getNum();
} 
```

`MyClass`是一个非常简单的类，它有一个私有成员，我们通过构造函数设置它。我们可以通过获取器访问它，为了保险起见，我们还声明了一个空的析构函数。以下是没有启用优化编译的上述代码的汇编等价代码：

```cpp
MyClass::MyClass(int) [base object constructor]:
        push    {r7}
        sub sp, sp, #12
add r7, sp, #0
str r0, [r7, #4]
        str r1, [r7]
        ldr r3, [r7, #4]
        ldr r2, [r7]
        str r2, [r3]
        ldr r3, [r7, #4]
        mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr
MyClass::~MyClass() [base object destructor]:
        push    {r7}
        sub sp, sp, #12
add r7, sp, #0
str r0, [r7, #4]
        ldr r3, [r7, #4]
        mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr
MyClass::getNum() const:
        push    {r7}
        sub sp, sp, #12
add r7, sp, #0
str r0, [r7, #4]
        ldr r3, [r7, #4]
        ldr r3, [r3]
        mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr
main:
push    {r4, r7, lr}
        sub sp, sp, #12
add r7, sp, #0
adds r3, r7, #4
movs r1, #1
mov r0, r3
bl      MyClass::MyClass(int) [complete object constructor]
        adds r3, r7, #4
mov r0, r3
bl      MyClass::getNum() const
        mov r4, r0
nop
adds r3, r7, #4
mov r0, r3
bl      MyClass::~MyClass() [complete object destructor]
        mov r3, r4
mov r0, r3
adds r7, r7, #12
mov sp, r7
pop     {r4, r7, pc} 
```

如果你不懂汇编，不必担心。我们可以看到有一些用于函数的标签和大量的指令。对于一个简单的类抽象来说，这有很多指令；这是我们不想在我们的二进制文件中出现的冗余代码。更精确地说，我们有 59 行汇编代码。如果我们启用优化，生成的汇编代码将只有几行长，但让我们不进行优化来分析这个问题。我们首先注意到的是析构函数没有做任何有用的事情。如果我们从 C++ 代码中移除它，生成的汇编代码将是 44 行长：

```cpp
MyClass::MyClass(int) [base object constructor]:
        push    {r7}
        sub sp, sp, #12
add r7, sp, #0
str r0, [r7, #4]
        str r1, [r7]
        ldr r3, [r7, #4]
        ldr r2, [r7]
        str r2, [r3]
        ldr r3, [r7, #4]
        mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr
MyClass::getNum() const:
        push    {r7}
        sub sp, sp, #12
add r7, sp, #0
str r0, [r7, #4]
        ldr r3, [r7, #4]
        ldr r3, [r3]
        mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr
main:
push    {r7, lr}
        sub sp, sp, #8
add r7, sp, #0
adds r3, r7, #4
movs r1, #1
mov r0, r3
bl      MyClass::MyClass(int) [complete object constructor]
        adds r3, r7, #4
mov r0, r3
bl      MyClass::getNum() const
        mov r3, r0
nop
mov r0, r3
adds r7, r7, #8
mov sp, r7
pop     {r7, pc} 
```

如我们所见，没有调用析构函数，二进制文件中也没有析构函数代码。教训是*你不为不使用的东西付费*。这是 C++ 的设计原则之一。通过删除析构函数，编译器不需要为它生成任何代码，也不需要在对象变量超出作用域时调用它。

我们必须认识到的是，C++ 不是一个面向对象的编程语言。它是一种多范式语言。它是过程性的、面向对象的、泛型的，甚至在某种程度上是函数式的。如果我们想有只能通过构造函数设置的私有成员，那么我们需要为此付出代价。C++ 中的结构体默认是公有成员，所以让我们将 `MyClass` 类改为没有构造函数的 `MyClass` 结构体：

```cpp
struct MyClass
{
    int num;
};
int main () {
   MyClass obj(1);

   return obj.num;
} 
```

设置器和获取器函数在面向对象范式中很常见，但 C++ 不是一个（仅仅是）面向对象的编程语言，我们也不必局限于使用设置器和获取器。当我们移除 `getNum` 获取器时，我们有一个只有一个成员的非常基本的结构体示例。生成的汇编代码只有 14 行长：

```cpp
main:
push    {r7}
        sub sp, sp, #12
add r7, sp, #0
movs r3, #1
str r3, [r7, #4]
        ldr r3, [r7, #4]
        mov r0, r3
adds r7, r7, #12
mov sp, r7
ldr r7, [sp], #4
bx lr 
```

尽管这个例子很简单，但其目的是确立两个基本事实：

+   你不需要为不使用的东西付费

+   使用 C++ 并不意味着你必然被绑定到面向对象（OOP）范式

如果我们想使用诸如构造函数和析构函数之类的抽象，我们必须为二进制大小付出代价。在 C++ 中，不实例化对象而使用类型（类和结构体）可以为嵌入式软件设计提供比传统面向对象方法更显著的好处。我们将在接下来的章节中通过详细的例子来探讨这一点。

在这个和之前的例子中，我们以禁用优化的方式编译了 C++ 代码，并能够看到生成的汇编代码结果中存在可以移除的不必要操作。让我们检查最后一个例子在启用 `O3` 优化级别时的汇编代码：

```cpp
main:
movs r0, #1
bx lr 
```

上述汇编是包含类、构造函数、析构函数和获取函数的原例程输出。生成的程序只有两条指令。`obj` 变量的 `num` 成员值存储在 `r0` 寄存器中作为返回值。汇编代码去除了所有与栈操作和将值存储在偏移量为 `4` 的栈指针中的 `r3` 相关的必要指令，并将它重新加载到 `r3`，然后移动到 `r0`。生成的汇编代码只有几行。

移除不必要的指令是优化过程的工作。然而，在嵌入式项目中，有些人声称优化会破坏代码，因此优化常常被避免。但这真的吗？

## 优化

未优化的代码会导致不必要的指令影响二进制大小和性能。然而，许多嵌入式项目仍然使用禁用优化的方式构建，因为开发者 *不相信编译器*，并担心它将 *破坏程序*。这确实有一定的道理，但事实是，这种情况发生在程序结构不佳时。如果程序包含未定义的行为，则程序结构不佳。

未定义行为的最佳例子之一是带符号的 **整数溢出**。标准没有定义如果你在你的平台上将 `1` 添加到带符号整数的最大值会发生什么。编译后的程序不需要执行任何有意义的操作。程序结构不佳。让我们检查以下代码：

```cpp
#include <cstdio>
#include <limits>
int foo(int x) {
    int y = x + 1;
    return y > x;
}
int main() {
    if(foo(std::numeric_limits<int>::max())) {
        printf("X is larger than X + 1\r\n");
    }
    else {
        printf("X is NOT larger than X + 1\. Oh nooo !\r\n");
    }
    return 0;
} 
```

使用 GCC 为 x86 和 Arm Cortex-M4 编译代码将产生相同的结果。如果程序未启用优化编译，`foo` 函数返回 `0`，你可以在输出中看到 **X 不大于 X + 1\. 哦不！**。编译器执行整数溢出，如果我们传递最大整数值给 `foo`，它将返回 `0`。请注意，标准没有指定这一点，这种行为取决于编译器。

如果我们启用优化编译程序，输出将是 **X 大于 X + 1**`,` 这意味着 `foo` 返回 `1`。让我们检查使用优化编译的程序汇编输出：

```cpp
foo(int):
        movs r0, #1
bx lr
.LC0:
.ascii "X is larger then X + 1\015\000"
main:
push    {r3, lr}
        movw    r0, #:lower16:.LC0
        movt r0, #:upper16:.LC0
        bl      puts
        movs r0, #0
pop     {r3, pc} 
```

如我们所见，`foo` 不执行任何计算。编译器假设程序结构良好，并且没有未定义的行为。`foo` 总是返回 `1`。确保程序中没有未定义的行为是开发者的责任。这正是优化会破坏程序的神话仍然存在的原因。将未定义的行为归咎于编译器不处理它更容易。

当然，如果使用优化，编译器中可能存在一个错误，会破坏程序的功能，而如果禁用优化，程序则可以正常工作。这种情况非常罕见，但并非没有发生过，这就是为什么存在诸如单元和集成测试之类的验证技术，以确保代码的功能，无论是否启用优化。

优化通过从机器代码中删除不必要的指令来减少二进制大小并提高性能。未定义行为是编译器依赖的，必须由开发者处理以确保程序结构良好。应实施单元和集成测试等技术来验证程序的功能，以减轻编译器损坏程序的风险。优化过程对于在 C++代码中使用抽象同时保持最小的二进制大小和最大性能至关重要。本书的其余部分我们将使用最高的优化级别`O3`。

我们将要检查的下一个代码膨胀的嫌疑者是模板。它们是如何导致代码膨胀的，它们又给我们的嵌入式代码库带来了什么价值？

## 模板

使用不同参数实例化**模板**将导致编译器生成不同的类型，这实际上会增加二进制大小。这是可以预料的。我们在使用占位符操作符和宏在 C 中实现环形缓冲区的泛型实现时也有完全相同的情况。一个替代方案是类型擦除，我们在 C 实现中使用空指针。如果我们施加静态数据分配的限制，它会在灵活性上受损，并且由于指针间接引用而影响性能。

使用泛型类型是设计选择之一。我们可以使用它们，并为此付出二进制大小增加的代价，但如果我们分别实现不同数据类型的环形缓冲区（例如`ring_buffer_int`、`ring_buffer_float`等），这也会发生。维护单个模板类型比在代码库的几个不同地方修复相同的错误要容易得多。泛型类型的使用不会导致二进制大小超过等效单个类型实现的尺寸。让我们通过`ring_buffer`示例来检查模板对二进制大小的影响：

```cpp
int main() {
#ifdef USE_TEMPLATES
  ring_buffer<int, 10> buffer1;
  ring_buffer<float, 10> buffer2;
#else
  ring_buffer_int buffer1;
  ring_buffer_float buffer2;
#endif
for (int i = 0; i < 20; i++) {
    buffer1.push(i);
    buffer2.push(i + 0.2f);
  }
  for (int i = 0; i < 10; i++) {
    printf("%d, %.2f\r\n", buffer1.pop(), buffer2.pop());
  }
  return 0;
} 
```

如果使用`USE_TEMPLATES`定义构建程序，它将使用泛型`ring_buffer type`，否则将使用`ring_buffer_int`和`ring_buffer_float`类型。如果我们使用没有启用优化的 GCC 构建此示例，模板版本将导致稍微更大的二进制大小（24 字节）。这是由于使用模板版本时符号表中的符号更大。如果我们从目标文件中删除符号表，它们将具有相同的大小。此外，使用`O3`构建两个版本将产生相同的二进制大小。

泛型类型不会比我们手动编写实例化类型作为单独类型时增加的二进制大小更多。模板由于在不同编译单元中实例化具体类型而影响构建时间，如果需要，有技术可以避免这种情况。所有与具有相同参数的实例化类型相关的函数都将导致二进制中只有一个函数，因为链接器将删除重复的符号。

## RTTI 和异常

C++中的**运行时类型信息**（**RTTI**）是一种允许在运行时确定对象类型的机制。大多数编译器使用虚表来实现 RTTI。每个具有至少一个虚函数的多态类都有一个虚表，其中包含运行时类型识别的类型信息。RTTI 既增加了时间成本也增加了空间成本。如果使用类型识别，它会增加二进制文件大小并影响运行时性能。这就是为什么编译器有禁用 RTTI 的方法。让我们通过一个基类和派生类的简单例子来考察：

```cpp
#include <cstdio>
struct Base {
    virtual void print () {
        printf("Base\r\n");
    }
};
struct Derived : public Base {
    void print () override {
        printf("Derived\r\n");
    }
};
void printer (Base &base) {
    base.print();
}
int main() {
    Base base;
    Derived derived;
    printer(base);
    printer(derived);
  return 0;
} 
```

程序的输出如下：

```cpp
Base
Derived 
```

具有虚函数的类有用于动态分发的 v 表。动态分发是一个选择多态函数实现的过程。`printer`函数接受`Base`类的引用。根据传递给`printer`的引用类型（`Base`或`Derived`），动态分发过程将选择`Base`或`Derived`类中的`print`方法。v 表也用于存储类型信息。

通过使用作为 RTTI 机制一部分的`dynamic_cast`，我们可以使用对超类引用或指针来找到类型信息。让我们修改前一个例子中的`printer`方法：

```cpp
void printer (Base &base) {
    base.print();
    if(Derived *derived = dynamic_cast<Derived*>(&base); derived!=nullptr) {
        printf("We found Base using RTTI!\r\n");
    }
} 
```

输出如下：

```cpp
Base
Derived
We found Base using RTTI! 
```

正如我们之前提到的，RTTI 可以被禁用。在 GCC 中，我们可以通过向编译器传递`-fno-rtti`标志来实现这一点。如果我们尝试使用这个标志编译修改后的示例，编译器将报错`error: dynamic_cast' not permitted with '-fno-rtti'`。如果我们将`printer`方法恢复到原始实现，删除`if`语句，并分别启用和禁用 RTTI 来构建它，我们可以注意到当 RTTI 启用时，二进制文件的大小更大。RTTI 在特定场景下很有用，但它会给资源受限的设备增加巨大的开销，因此我们将它保持禁用状态。

另一个在 C++嵌入式项目中经常禁用的 C++特性是异常。**异常**是一种基于 try-catch 块的错误处理机制。让我们通过一个简单的例子来利用异常来更好地理解它们：

```cpp
#include <cstdio>
struct A {
  A() { printf("A is created!\r\n"); }
  ~A() { printf("A is destroyed!\r\n"); }
};
struct B {
  B() { printf("B is created!\r\n"); }
  ~B() { printf("B is destroyed!\r\n"); }
};
void bar() {
    B b;
    throw 0;
}
void foo() {
  A a;
  bar();
  A a1;
}
int main() {
  try {
    foo();
  } catch (int &p) {
    printf("Catching an exception!\r\n");
  }
  return 0;
} 
```

程序的输出如下：

```cpp
A is created!
B is created!
B is destroyed!
A is destroyed!
Catching an exception! 
```

在这个简单的例子中，`foo` 在 `try` 块中被调用。它创建了一个局部对象 `a` 并调用 `bar`。`bar` 函数创建了一个局部对象 `b` 并抛出一个异常。在输出中，我们看到 `A` 和 `B` 被创建，然后 `B` 被销毁，接着 `A` 被销毁，最后我们看到 `catch` 块被执行。这被称为**栈展开**，为了使其发生，标准实现通常最常用的是 unwind tables，它们存储有关捕获处理程序、将被调用的析构函数等信息。unwind tables 可以变得很大且复杂，这增加了应用程序的内存占用，并由于运行时用于异常处理的机制而引入了非确定性。这就是为什么异常通常在嵌入式系统项目中被禁用。

# 摘要

C++ 遵循**零开销原则**。唯一不遵循此原则的两个语言特性是 RTTI 和异常，这也是为什么编译器支持一个开关来关闭它们。

零开销原则基于我们在本章中确立的两个陈述：

+   你不需要为不使用的功能付费

+   你使用的功能与你可以合理手动编写的功能一样高效

在大多数嵌入式项目中，RTTI 和异常都被禁用，所以你不需要为它们付费。使用泛型类型和模板是一种设计选择，并且不比手动编写单个类型（如 `ring_buffer_int`、`ring_buffer_float` 等）更昂贵，但它允许你重用不同类型的代码逻辑，使代码更易于阅读和维护。

在高风险系统中工作不是禁用编译器优化能力的理由。无论我们是在启用或禁用优化的程序中构建，代码功能都需要经过验证。当启用优化时，最常见的错误来源是未定义行为。理解未定义行为并防止它取决于开发者。

现代 C++是一种对嵌入式世界有很多贡献的语言。本书的使命是帮助你发现 C++以及它可以为你的嵌入式项目做什么，所以让我们踏上发现 C++并利用它来解决嵌入式领域问题的道路。

在下一章中，我们将讨论嵌入式系统中的资源限制挑战和 C++中的动态内存管理。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/embeddedsystems`](https://packt.link/embeddedsystems)

![Discord 二维码](img/QR_code_Discord.png)
