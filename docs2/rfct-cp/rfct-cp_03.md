

# 第三章：糟糕代码的原因

在前面的章节中，我们讨论了 C++的编码标准和核心开发原则。当我们深入到重构现有代码时，理解导致代码质量低下或糟糕的原因至关重要。识别这些原因使我们能够避免重复相同的错误，解决现有问题，并有效地优先考虑未来的改进。

恶劣代码可能由各种因素造成，从外部压力到内部团队动态。一个重要因素是快速交付产品的需求，尤其是在快节奏的环境，如初创公司。在这里，快速发布功能的压力往往导致代码质量的妥协，开发者可能会为了满足紧迫的截止日期而走捷径或跳过重要的最佳实践。

另一个影响因素是 C++中解决同一问题的多种方式。语言的灵活性和丰富性，虽然强大，但可能导致不一致性和维护连贯代码库的困难。不同的开发者可能会以不同的方式处理相同的问题，导致代码库碎片化且难以维护。

开发者的个人品味也起着作用。个人偏好和编码风格可能会影响代码的整体质量和可读性。一个开发者认为优雅的，另一个可能觉得复杂，导致主观差异影响代码的一致性和清晰度。

最后，对现代 C++特性的缺乏可能导致代码效率低下或存在错误。随着 C++的发展，它引入了新的特性和范式，这些特性需要深入理解才能有效使用。当开发者没有跟上这些进步时，他们可能会退回到过时的做法，错失可以提高代码质量和性能的改进。

通过探讨这些方面，我们的目标是提供一个对导致糟糕代码的因素的全面理解。这种知识对于任何希望有效地重构和改进现有代码库的开发者来说都是必不可少的。让我们深入探讨，揭示 C++开发中糟糕代码的根本原因。

# 交付产品的需求

当开发者检查现有代码时，他们可能会质疑为什么代码以不那么优雅或缺乏可扩展性的方式编写。批评他人完成的工作通常很容易，但理解原始开发者的背景至关重要。假设项目最初是在一家初创公司开发的。在这种情况下，重要的是要考虑到初创文化显著强调快速产品交付和超越竞争对手的需求。虽然这可能是优势，但也可能导致糟糕的代码。其中一个主要原因是快速交付的压力，这可能导致开发者为了赶工期而走捷径或跳过必要的编码实践（例如，前几章中提到的 SOLID 原则）。这可能导致代码缺乏适当的文档，难以维护，并且可能容易出错。

此外，初创公司有限的资源和小型开发团队可能会加剧对速度的需求，因为开发者可能没有足够的人手来专注于优化和精炼代码库。结果，代码可能会变得杂乱无章且效率低下，导致性能下降和错误增加。

此外，创业文化中注重快速交付的特点可能会让开发者难以跟上 C++ 语言的最新进展。这可能会导致代码过时，缺乏重要功能，使用低效或已弃用的函数，并且没有针对性能进行优化。

# 开发者的个人品味

导致糟糕代码的另一个重要因素是开发者的个人品味。个人偏好和编码风格可能差异很大，导致主观差异影响代码的一致性和可读性。例如，考虑两位开发者鲍勃和爱丽丝。鲍勃偏好使用简洁、紧凑的代码，利用高级 C++ 特性，而爱丽丝则更喜欢更明确和冗长的代码，优先考虑清晰和简单。

鲍勃可能会使用现代 C++ 特性，如 lambda 表达式和 `auto` 关键字来编写函数：

```cpp
auto process_data = [](const std::vector<int>& data) {
    return std::accumulate(data.begin(), data.end(), 0L);
};
```

相反，爱丽丝可能会偏好更传统的方法，避免使用 lambda 表达式并使用显式类型：

```cpp
long process_data(const std::vector<int>& data) {
    long sum = 0;
    for (int value : data) {
        sum += value;
    }
    return sum;
}
```

虽然这两种方法都是有效的，并且可以达到相同的结果，但风格上的差异可能导致代码库中的混淆和不一致。如果鲍勃和爱丽丝在没有遵循共同编码标准的情况下共同工作，代码可能会成为不同风格的大杂烩，使得维护和理解变得更加困难。

此外，鲍勃使用现代特性的做法可能会引入团队成员不熟悉这些特性时难以应对的复杂性，而爱丽丝的冗长风格可能会被那些偏好更简洁代码的人视为过于简单和低效。这些差异源于个人品味，强调了建立和遵循团队编码标准的重要性，以确保代码库的一致性和可维护性。

通过识别和解决个人编码偏好的影响，团队可以共同努力创建一个统一且易于阅读的代码库，与最佳实践保持一致，并提高整体代码质量。

# C++中解决相同问题的多种方法

C++是一种多才多艺的语言，提供了多种解决相同问题的方法，这一特性既能赋予开发者力量，也可能使他们感到困惑。这种灵活性往往会导致代码库中的不一致性，尤其是在不同开发者的专业水平和偏好不同时。在本章中，我们将展示一些示例，说明相同问题可以以不同的方式处理，突出每种方法的潜在优点和缺点。正如在*开发者的个人品味*部分所讨论的，像鲍勃和爱丽丝这样的开发者可能会使用不同的技术来处理相同的问题，从而导致代码库碎片化。

## 重温鲍勃和爱丽丝的例子

回顾一下，鲍勃使用了现代 C++特性，如 lambda 表达式和`auto`来简洁地处理数据，而爱丽丝则更喜欢更明确和冗长的方法。两种方法都能达到相同的结果，但风格上的差异可能导致代码库中的混淆和不一致。虽然鲍勃的方法更紧凑并利用了现代 C++特性，但爱丽丝的方法对那些不熟悉 lambda 的人来说更易于理解。

## 原始指针和 C 函数与标准库函数

考虑一个项目，它大量使用原始指针和 C 函数来复制数据，这是较老 C++代码库中的常见做法：

```cpp
void copy_array(const char* source, char* destination, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        destination[i] = source[i];
    }
}
```

这种方法虽然可行，但容易发生缓冲区溢出等错误，并且需要手动管理内存。相比之下，现代 C++方法会使用标准库函数，例如`std::copy`：

```cpp
void copy_array(const std::vector<char>& source, std::vector<char>& destination) {
    std::copy(source.begin(), source.end(), std::back_inserter(destination));
}
```

使用`std::copy`不仅简化了代码，而且利用了经过良好测试的库函数，这些函数可以处理边缘情况并提高安全性。

## 继承与模板

C++提供多种解决方案的另一个领域是代码重用和抽象。一些项目更喜欢使用继承，这可能导致僵化和复杂的层次结构：

```cpp
class Shape {
public:
    virtual void draw() const = 0;
    virtual ~Shape() = default;
};
class Circle : public Shape {
public:
    void draw() const override {
        // Draw circle
    }
};
class Square : public Shape {
public:
    void draw() const override {
        // Draw square
    }
};
class ShapeDrawer {
public:
    explicit ShapeDrawer(std::unique_ptr<Shape> shape) : shape_(std::move(shape)) {}
    void draw() const {
        shape_->draw();
    }
private:
    std::unique_ptr<Shape> shape_;
};
```

虽然继承提供了清晰的架构并允许多态行为，但随着层次结构的增长，它可能会变得繁琐。一种替代方法是使用模板来实现多态，而不需要虚拟函数的开销。以下是模板如何实现类似功能的方法：

```cpp
template<typename ShapeType>
class ShapeDrawer {
public:
    explicit ShapeDrawer(ShapeType shape) : shape_(std::move(shape)) {}
    void draw() const {
        shape_.draw();
    }
private:
    ShapeType shape_;
};
class Circle {
public:
    void draw() const {
        // Draw circle
    }
};
class Square {
public:
    void draw() const {
        // Draw square
    }
};
```

在这个例子中，`ShapeDrawer`使用模板来实现多态行为。`ShapeDrawer`可以与任何提供`draw`方法的类型一起工作。这种方法避免了与虚拟函数调用相关的开销，并且可以更高效，尤其是在性能关键的应用中。

## 示例 - 处理错误

另一个例子是不同方式解决相同问题，比如错误处理。考虑一个项目中鲍勃使用传统的错误码：

```cpp
int process_file(const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        return -1; // Error opening file
    }
    // Process file
    return fclose(file);
}
```

另一方面，爱丽丝更倾向于使用异常进行错误处理：

```cpp
void process_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Error opening file");
    }
    // Process file
}
```

使用异常可以使代码更简洁，因为它将错误处理与主逻辑分离，但需要理解异常安全性和处理。错误代码虽然更简单，但可能会使代码因重复检查而变得杂乱，并且可能提供的信息较少。

## 采用不同方法的项目

在现实世界的项目中，你可能会遇到这些方法的混合使用，反映了不同开发者的不同背景和偏好，例如以下示例：

+   **项目 A**在性能关键部分使用原始指针和 C 函数，依赖开发者的专业知识来安全地管理内存

+   **项目 B**更倾向于使用标准库容器和算法，优先考虑安全性和可读性，而不是原始性能

+   **项目 C**采用深度继承层次结构来建模其领域，强调实体之间的清晰关系

+   **项目 D**广泛使用模板来实现高性能和灵活性，尽管学习曲线更陡峭，且可能更复杂

每种方法都有其优缺点，选择正确的方法取决于项目的需求、团队的专长以及要解决的问题的具体性。然而，如果不小心管理，解决同一问题的多种方法可能会导致代码库碎片化和不一致。

C++提供了多种解决同一问题的方法，从原始指针和 C 函数到标准库容器和模板。虽然这种灵活性非常强大，但也可能导致代码库中的不一致性和复杂性。理解每种方法的优缺点，并通过编码标准和团队协议努力保持一致性，对于维护高质量、可维护的代码至关重要。通过采用现代 C++特性和最佳实践，开发者可以编写既高效又健壮的代码，降低错误发生的可能性，并提高整体代码质量。

# C++知识不足

导致糟糕代码的一个主要原因是 C++知识不足。C++是一种复杂且不断发展的语言，具有广泛的功能，保持对其最新标准的更新需要持续学习。不熟悉现代 C++实践的开发者可能会无意中编写低效或容易出错的代码。本节探讨了 C++理解上的差距如何导致各种问题，并使用示例来说明常见的陷阱。

考虑两位开发者，鲍勃和爱丽丝。鲍勃对较老版本的 C++有丰富的经验，但 hasn’t kept up with recent updates，而爱丽丝对现代 C++特性非常熟悉。

## 使用原始指针和手动内存管理

鲍勃可能会使用原始指针和手动内存管理，这是较老 C++代码中的常见做法：

```cpp
void process() {
    int* data = new int[100];
    // ... perform operations on data
    delete[] data;
}
```

如果错过或错误地匹配`delete[]`与`new`，这种方法容易出错，如内存泄漏和未定义行为。例如，如果在分配之后但在`delete[]`之前抛出异常，内存将会泄漏。爱丽丝熟悉现代 C++，会使用`std::vector`来安全有效地管理内存：

```cpp
void process() {
    std::vector<int> data(100);
    // ... perform operations on data
}
```

使用`std::vector`消除了手动内存管理的需要，降低了内存泄漏的风险，并使代码更健壮、更容易维护。

## 智能指针使用不当

鲍勃试图采用现代实践，但错误地使用了`std::shared_ptr`，导致潜在的性能问题：

```cpp
std::shared_ptr<int> create() {
    std::shared_ptr<int> ptr(new int(42));
    return ptr;
}
```

这种方法涉及两个独立的分配：一个用于整数，另一个用于`std::shared_ptr`的控制块。爱丽丝了解到`std::make_shared`的好处，因此使用它来优化内存分配：

```cpp
std::shared_ptr<int> create() {
    return std::make_shared<int>(42);
}
```

`std::make_shared`将分配合并到单个内存块中，提高了性能和缓存局部性。

## 移动语义的有效使用

鲍勃可能不完全理解移动语义及其在处理临时对象时提高性能的能力。考虑一个将元素追加到`std::vector`的函数：

```cpp
void append_data(std::vector<int>& target, const std::vector<int>& source) {
    for (const int& value : source) {
        target.push_back(value); // Copies each element
    }
}
```

这种方法涉及将每个元素从`source`复制到`target`，这可能效率低下。爱丽丝了解移动语义，通过使用`std::move`来优化：

```cpp
void append_data(std::vector<int>& target, std::vector<int>&& source) {
    for (int& value : source) {
        target.push_back(std::move(value)); // Moves each element
    }
}
```

通过使用`std::move`，爱丽丝确保每个元素是移动而不是复制，这更有效率。此外，如果`source`不再需要，爱丽丝还可能考虑对整个容器使用`std::move`：

```cpp
void append_data(std::vector<int>& target, std::vector<int>&& source) {
    target.insert(target.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
}
```

这种方法有效地移动整个容器的元素，利用移动语义来避免不必要的复制。

## const 正确性使用不当

鲍勃可能会忽视 const 的正确性，导致潜在的 bug 和不清晰的代码：

```cpp
class MyClass {
public:
    int get_value() { return value; }
    void set_value(int v) { value = v; }
private:
    int value;
};
```

没有 const 正确性，`get_value`是否修改对象的状态并不明确。爱丽丝应用 const 正确性来阐明意图并提高安全性：

```cpp
class MyClass {
public:
    int get_value() const { return value; }
    void set_value(int v) { value = v; }
private:
    int value;
};
```

将`get_value`标记为`const`确保它不会修改对象，使代码更清晰并防止意外修改。

## 不高效的字符串处理

鲍勃可能会使用 C 风格的字符数组来处理字符串，这可能导致缓冲区溢出和复杂的代码：

```cpp
char message[100];
strcpy(message, "Hello, world!");
std::cout << message << std::endl;
```

这种方法容易出错且难以管理。爱丽丝了解到`std::string`的能力，简化了代码并避免了潜在的错误：

```cpp
std::string message = "Hello, world!";
std::cout << message << std::endl;
```

使用`std::string`提供自动内存管理和丰富的字符串操作函数，使代码更安全、更易于表达。

## 使用 lambda 表达式时的未定义行为

C++11 中引入的 lambda 函数提供了强大的功能，但如果不正确使用，可能会导致未定义行为。鲍勃可能会编写一个 lambda 函数，通过引用捕获局部变量并返回它，从而导致悬垂引用：

```cpp
auto create_lambda() {
    int value = 42;
    return [&]() { return value; };
}
auto lambda = create_lambda();
int result = lambda(); // Undefined behavior
```

爱丽丝理解风险，通过值捕获变量以确保其有效性：

```cpp
auto create_lambda() {
    int value = 42;
    return [=]() { return value; };
}
auto lambda = create_lambda();
int result = lambda(); // Safe
```

通过值捕获避免了悬垂引用的风险，并确保 lambda 可以安全使用。

## 对未定义行为的误解

鲍勃可能会无意中编写依赖于未初始化变量的代码，从而导致未定义的行为：

```cpp
int sum() {
    int x;
    int y = 5;
    return x + y; // Undefined behavior: x is uninitialized
}
```

访问未初始化的变量可能导致不可预测的行为和难以调试的问题。爱丽丝理解初始化的重要性，确保所有变量都得到适当的初始化：

```cpp
int sum() {
    int x = 0;
    int y = 5;
    return x + y; // Defined behavior
}
```

正确初始化变量可以防止未定义的行为，并使代码更可靠。

## C 风格数组的误用

使用 C 风格数组可能导致各种问题，例如缺乏边界检查和管理数组大小的困难。考虑以下示例，其中函数在栈上创建一个 C 数组并返回它：

```cpp
int* create_array() {
    int arr[5] = {1, 2, 3, 4, 5};
    return arr; // Undefined behavior: returning a pointer to a local array
}
```

返回指向局部数组的指针会导致未定义的行为，因为数组在函数返回时超出作用域。一种更安全的方法是使用 `std::array`，它可以安全地从函数返回。它提供了 `size` 方法，并与 C++ 算法（如 `std::sort`）兼容：

```cpp
std::array<int, 5> create_array() {
    return {1, 2, 3, 4, 5};
}
```

使用 `std::array` 不仅避免了未定义的行为，还增强了安全性和与 C++ 标准库的互操作性。例如，排序数组变得简单：

```cpp
std::array<int, 5> arr = create_array();
std::sort(arr.begin(), arr.end());
```

# 指针使用不足

现代 C++提供了如 `std::unique_ptr` 和 `std::shared_ptr` 这样的智能指针，以更安全、更有效地管理动态内存。通常，使用 `std::unique_ptr` 而不是原始指针来拥有独占所有权更好。当多个参与者需要共享资源的所有权时，可以使用 `std::shared_ptr`。然而，与 `std::shared_ptr` 的误用相关的问题很常见。

## 构建 `std::shared_ptr`

使用 `std::shared_ptr` 构造函数创建对象会导致控制块和管理对象分别进行分配：

```cpp
std::shared_ptr<int> create() {
    std::shared_ptr<int> ptr(new int(42));
    return ptr;
}
```

更好的方法是使用 `std::make_shared`，它将分配合并到单个内存块中，提高性能和缓存局部性：

```cpp
std::shared_ptr<int> create() {
    return std::make_shared<int>(42);
}
```

## 通过值复制 `std::shared_ptr`

在同一线程栈内通过值复制 `std::shared_ptr` 不是很高效，因为引用计数是原子的。建议通过引用传递 `std::shared_ptr`：

```cpp
void process_shared_ptr(std::shared_ptr<int> ptr) {
    // Inefficient: copies shared_ptr by value
}
void process_shared_ptr(const std::shared_ptr<int>& ptr) {
    // Efficient: passes shared_ptr by reference
}
```

## `std::shared_ptr` 的循环依赖

当两个或更多 `std::shared_ptr` 实例互相引用时，可能会发生循环依赖，阻止引用计数达到零并导致内存泄漏。考虑以下示例：

```cpp
struct B;
struct A {
    std::shared_ptr<B> b_ptr;
    ~A() { std::cout << "A destroyed\n"; }
};
struct B {
    std::shared_ptr<A> a_ptr;
    ~B() { std::cout << "B destroyed\n"; }
};
void create_cycle() {
    auto a = std::make_shared<A>();
    auto b = std::make_shared<B>();
    a->b_ptr = b;
    b->a_ptr = a;
}
```

在这种情况下，`A` 和 `B` 互相引用，形成一个循环，阻止它们的销毁。这个问题可以使用 `std::weak_ptr` 来打破循环：

```cpp
struct B;
struct A {
    std::weak_ptr<B> b_ptr; // Use weak_ptr to break the cycle
    ~A() { std::cout << "A destroyed\n"; }
};
struct B {
    std::shared_ptr<A> a_ptr;
    ~B() { std::cout << "B destroyed\n"; }
};
void create_cycle() {
    auto a = std::make_shared<A>();
    auto b = std::make_shared<B>();
    a->b_ptr = b;
    b->a_ptr = a;
}
```

## 检查 `std::weak_ptr` 状态

使用 `std::weak_ptr` 的一个常见错误是使用 `expired()` 检查其状态，然后锁定它，这并不是线程安全的：

```cpp
std::weak_ptr<int> weak_ptr = some_shared_ptr;
void check_and_use_weak_ptr() {
    if (!weak_ptr.expired()) {
        // This is not thread-safe
        auto shared_ptr = weak_ptr.lock();
        shared_ptr->do_something();
    }
}
```

正确的做法是锁定 `std::weak_ptr` 并检查返回的 `std::shared_ptr` 不是 `null`：

```cpp
void check_and_use_weak_ptr_correctly() {
    // This is thread-safe
    if (auto shared_ptr = weak_ptr.lock()) {
        // Use shared_ptr
        shared_ptr->do_something();
    }
}
```

C++知识的缺乏可能导致各种问题，从内存管理错误到低效且难以阅读的代码。通过保持对现代 C++特性和最佳实践的更新，开发者可以编写更安全、更高效且易于维护的代码。持续学习和适应是克服这些挑战并提高整体代码质量的关键。Bob 和 Alice 的例子突出了理解和应用现代 C++实践的重要性，以避免常见陷阱并生成高质量的代码。

# 摘要

在本章中，我们探讨了 C++中不良代码的多种原因，以及缺乏现代 C++实践知识如何导致低效、易出错或未定义的行为。通过检查具体示例，我们强调了持续学习和适应以跟上 C++功能演变的重要性。

我们首先讨论了使用原始指针和手动内存管理的陷阱，展示了现代 C++实践，如`std::vector`如何消除手动内存管理的需求并降低内存泄漏的风险。强调了使用`std::unique_ptr`进行独占所有权和`std::shared_ptr`进行共享所有权的优势，同时突出了常见问题，如低效的内存分配、不必要的复制和循环依赖。

在`std::shared_ptr`的上下文中，我们展示了使用`std::make_shared`而非构造函数来减少内存分配并提高性能的优势。由于原子引用计数器，通过引用而非值传递`std::shared_ptr`所获得的效率提升也得到了解释。我们还阐述了循环依赖的问题以及如何使用`std::weak_ptr`来打破循环并防止内存泄漏。同时，还介绍了通过锁定并检查结果`std::shared_ptr`来确保线程安全的正确方式。

讨论了有效使用移动语义来优化性能，通过减少临时对象的非必要复制。使用`std::move`和`std::make_move_iterator`可以显著提高程序性能。强调了 const 正确性的重要性，展示了将`const`应用于方法如何阐明意图并提高代码安全性。

我们讨论了使用 C 风格字符数组的危险，以及`std::string`如何简化字符串处理、减少错误并提供更好的内存管理。探讨了 C 风格数组的误用，并介绍了`std::array`作为更安全、更健壮的替代方案。通过使用`std::array`，我们可以避免未定义的行为，并利用 C++标准库算法，如`std::sort`。

最后，我们讨论了 lambda 函数的正确使用，以及通过引用捕获变量可能导致的潜在陷阱，这可能导致悬垂引用。通过值捕获变量确保 lambda 函数的安全使用。

通过这些例子，我们了解到采用现代 C++特性和最佳实践对于编写更安全、更高效和可维护的代码具有至关重要的意义。通过紧跟最新标准并不断深化我们对 C++的理解，我们可以避免常见的陷阱，并生产出高质量的软件。
