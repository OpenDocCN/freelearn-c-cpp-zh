# 第二章：基本的 C++技术

在本章中，我们将深入研究一些基本的 C++技术，如移动语义、错误处理和 lambda 表达式，这些技术将贯穿本书使用。即使是经验丰富的 C++程序员，有些概念仍然会让人困惑，因此我们将探讨它们的用例和工作原理。

本章将涵盖以下主题：

+   自动类型推导以及在声明函数和变量时如何使用`auto`关键字。

+   移动语义和*五法则*和*零法则*。

+   错误处理和契约。虽然这些主题并没有提供可以被视为现代 C++的任何内容，但异常和契约在当今的 C++中都是高度争议的领域。

+   使用 lambda 表达式创建函数对象，这是 C++11 中最重要的功能之一。

让我们首先来看一下自动类型推导。

# 使用 auto 关键字进行自动类型推导

自从 C++11 引入了`auto`关键字以来，C++社区对如何使用不同类型的`auto`（如`const` `auto&`、`auto&`、`auto&&`和`decltype(auto)`）产生了很多困惑。

## 在函数签名中使用 auto

尽管一些 C++程序员不赞成，但在我的经验中，在函数签名中使用`auto`可以增加可读性，方便浏览和查看头文件。

以下是`auto`语法与显式类型的传统语法相比的样子：

| 显式类型的传统语法： | 使用 auto 的新语法： |
| --- | --- |

|

```cpp
struct Foo {
  int val() const {    return m_;   }  const int& cref() const {    return m_;   }  int& mref() {    return m_;   }  int m_{};}; 
```

|

```cpp
struct Foo {
  auto val() const {    return m_;   }  auto& cref() const {    return m_;   }  auto& mref() {    return m_;   }  int m_{};}; 
```

|

`auto`语法可以在有或没有尾随返回类型的情况下使用。在某些情境下，尾随返回类型是必要的。例如，如果我们正在编写虚函数，或者函数声明放在头文件中，函数定义在`.cpp`文件中。

请注意，`auto`语法也可以用于自由函数：

| 返回类型 | 语法变体（a、b 和 c 对应相同的结果）： |
| --- | --- |
| 值 |

```cpp
auto val() const                // a) auto, deduced type
auto val() const -> int         // b) auto, trailing type
int val() const                 // c) explicit type 
```

|

| 常量引用 |
| --- |

```cpp
auto& cref() const              // a) auto, deduced type
auto cref() const -> const int& // b) auto, trailing type
const int& cref() const         // c) explicit type 
```

|

| 可变引用 |
| --- |

```cpp
auto& mref()                    // a) auto, deduced type
auto mref() -> int&             // b) auto, trailing type
int& mref()                     // c) explicit type 
```

|

### 使用 decltype(auto)进行返回类型转发

还有一种相对罕见的自动类型推导版本称为`decltype(auto)`。它最常见的用途是从函数中转发确切的类型。想象一下，我们正在为前面表格中声明的`val()`和`mref()`编写包装函数，就像这样：

```cpp
int val_wrapper() { return val(); }    // Returns int
int& mref_wrapper() { return mref(); } // Returns int& 
```

现在，如果我们希望对包装函数使用返回类型推导，`auto`关键字将在两种情况下推导返回类型为`int`：

```cpp
auto val_wrapper() { return val(); }   // Returns int
auto mref_wrapper() { return mref(); } // Also returns int 
```

如果我们希望`mref_wrapper()`返回`int&`，我们需要写`auto&`。在这个例子中，这是可以的，因为我们知道`mref()`的返回类型。然而，并非总是如此。因此，如果我们希望编译器选择与`int&`或`auto&`相同的类型而不明确指定`mref_wrapper()`的返回类型，我们可以使用`decltype(auto)`：

```cpp
decltype(auto) val_wrapper() { return val(); }   // Returns int
decltype(auto) mref_wrapper() { return mref(); } // Returns int& 
```

通过这种方式，我们可以避免在不知道函数`val()`或`mref()`返回的类型时明确选择写`auto`或`auto&`。这通常发生在泛型代码中，其中被包装的函数的类型是模板参数。

## 使用 auto 声明变量

C++11 引入`auto`关键字引发了 C++程序员之间的激烈辩论。许多人认为它降低了可读性，甚至使 C++变得类似于动态类型语言。我倾向于不参与这些辩论，但我个人认为你应该（几乎）总是使用`auto`，因为在我的经验中，它使代码更安全，减少了混乱。

过度使用`auto`可能会使代码难以理解。在阅读代码时，我们通常想知道某个对象支持哪些操作。一个好的 IDE 可以为我们提供这些信息，但在源代码中并没有明确显示。C++20 概念通过关注对象的行为来解决这个问题。有关 C++概念的更多信息，请参阅*第八章*，*编译时编程*。

我喜欢使用`auto`来定义使用从左到右的初始化样式的局部变量。这意味着将变量保留在左侧，后跟一个等号，然后在右侧是类型，就像这样：

```cpp
auto i = 0;
auto x = Foo{};
auto y = create_object();
auto z = std::mutex{};     // OK since C++17 
```

在 C++17 中引入了*保证的拷贝省略*，语句`auto x = Foo{}`与`Foo x{}`是相同的；也就是说，语言保证在这种情况下没有需要移动或复制的临时对象。这意味着我们现在可以使用从左到右的初始化样式，而不用担心性能，我们还可以用于不可移动/不可复制的类型，如`std::atomic`或`std::mutex`。

使用`auto`定义变量的一个很大的优势是，您永远不会留下未初始化的变量，因为`auto x;`不会编译。未初始化的变量是未定义行为的一个常见来源，您可以通过遵循这里建议的样式完全消除。

使用`auto`将帮助您使用正确的类型来定义变量。但您仍然需要通过指定需要引用还是副本，以及是否要修改变量或仅从中读取来表达您打算如何使用变量。

### 一个 const 引用

`const`引用，用`const auto&`表示，具有绑定到任何东西的能力。原始对象永远不会通过这样的引用发生变异。我认为`const`引用应该是潜在昂贵的对象的默认选择。

如果`const`引用绑定到临时对象，则临时对象的生命周期将延长到引用的生命周期。这在以下示例中得到了证明：

```cpp
void some_func(const std::string& a, const std::string& b) {
  const auto& str = a + b;  // a + b returns a temporary
  // ...
} // str goes out of scope, temporary will be destroyed 
```

也可以通过使用`auto&`得到一个`const`引用。可以在以下示例中看到：

```cpp
 auto foo = Foo{};
 auto& cref = foo.cref(); // cref is a const reference
 auto& mref = foo.mref(); // mref is a mutable reference 
```

尽管这是完全有效的，但最好始终明确表示我们正在处理`const`引用，使用`const auto&`，更重要的是，我们应该使用`auto&`仅表示可变引用。

### 一个可变引用

与`const`引用相反，可变引用不能绑定到临时对象。如前所述，我们使用`auto&`来表示可变引用。只有在打算更改引用的对象时才使用可变引用。

### 转发引用

`auto&&`被称为转发引用（也称为*通用引用*）。它可以绑定到任何东西，这对某些情况很有用。转发引用将像`const`引用一样，延长临时对象的生命周期。但与`const`引用相反，`auto&&`允许我们改变它引用的对象，包括临时对象。

对于只转发到其他代码的变量，请使用`auto&&`。在这些转发情况下，您很少关心变量是`const`还是可变的；您只是想将其传递给实际要使用变量的一些代码。

重要的是要注意，只有在函数模板中使用`T`作为该函数模板的模板参数时，`auto&&`和`T&&`才是转发引用。使用显式类型，例如`std::string&&`，带有`&&`语法表示**右值**引用，并且不具有转发引用的属性（右值和移动语义将在本章后面讨论）。

### 便于使用的实践

尽管这是我的个人意见，我建议对基本类型（`int`，`float`等）和小的非基本类型（如`std::pair`和`std::complex`）使用`const auto`。对于潜在昂贵的大型类型，使用`const auto&`。这应该涵盖 C++代码库中大多数变量声明。

只有在需要可变引用或显式复制的行为时，才应使用`auto&`和`auto`；这向代码的读者传达了这些变量的重要性，因为它们要么复制一个对象，要么改变一个引用的对象。最后，只在转发代码时使用`auto&&`。

遵循这些规则可以使您的代码库更易于阅读、调试和理解。

也许会觉得奇怪，虽然我建议在大多数变量声明中使用`const auto`和`const auto&`，但在本书的某些地方我倾向于使用简单的`auto`。使用普通的`auto`的原因是书籍格式提供的有限空间。

在继续之前，我们将花一点时间讨论`const`以及在使用指针时如何传播`const`。

## 指针的 const 传播

通过使用关键字`const`，我们可以告诉编译器哪些对象是不可变的。然后编译器可以检查我们是否尝试改变不打算改变的对象。换句话说，编译器检查我们的代码是否符合`const`-correctness。在 C++中编写`const`-correct 代码时的一个常见错误是，`const`初始化的对象仍然可以操作成员指针指向的值。以下示例说明了这个问题：

```cpp
class Foo {
public:
  Foo(int* ptr) : ptr_{ptr} {} 
  auto set_ptr_val(int v) const { 
    *ptr_ = v; // Compiles despite function being declared const!
  }
private:
  int* ptr_{};
};
int main() {
  auto i = 0;
  const auto foo = Foo{&i};
  foo.set_ptr_val(42);
} 
```

虽然函数`set_ptr_val()`正在改变`int`值，但声明它为`const`是有效的，因为指针`ptr_`本身没有被改变，只有指针指向的`int`对象被改变。

为了以一种可读的方式防止这种情况，标准库扩展中添加了一个名为`std::experimental::propagate_const`的包装器（在撰写本文时，已包含在最新版本的 Clang 和 GCC 中）。使用`propagate_const`，函数`set_ptr_val()`将无法编译。请注意，`propagate_const`仅适用于指针和类似指针的类，如`std::shared_ptr`和`std::unique_ptr`，而不适用于`std::function`。

以下示例演示了如何使用`propagate_const`在尝试在`const`函数内部改变对象时生成编译错误：

```cpp
#include <experimental/propagate_const>
class Foo { 
public: 
  Foo(int* ptr) : ptr_{ptr} {}
  auto set_ptr(int* p) const { 
    ptr_ = p;  // Will not compile, as expected
  }
  auto set_val(int v) const { 
    val_ = v;  // Will not compile, as expected
  }
  auto set_ptr_val(int v) const { 
    *ptr_ = v; // Will not compile, const is propagated
  }
private:
  std::experimental::propagate_const<int*> ptr_ = nullptr; 
  int val_{}; 
}; 
```

在大型代码库中正确使用`const`的重要性不言而喻，而引入`propagate_const`使`const`-correctness 变得更加有效。

接下来，我们将看一下移动语义以及处理类内部资源的一些重要规则。

# 解释移动语义

移动语义是 C++11 中引入的一个概念，在我看来，即使是经验丰富的程序员也很难理解。因此，我将尝试为您深入解释它的工作原理，编译器如何利用它，以及为什么它是必要的。

基本上，C++之所以有移动语义的概念，而大多数其他语言没有，是因为它是一种基于值的语言，正如在《第一章 C++简介》中讨论的那样。如果 C++没有内置移动语义，那么基于值的语义的优势在许多情况下将会丢失，程序员将不得不进行以下折衷之一：

+   执行性能成本高的冗余深克隆操作

+   像 Java 一样使用对象指针，失去值语义的健壮性

+   以牺牲可读性为代价进行容易出错的交换操作

我们不希望出现这些情况，所以让我们看看移动语义如何帮助我们。

## 复制构造、交换和移动

在深入了解移动的细节之前，我将首先解释并说明复制构造对象、交换两个对象和移动构造对象之间的区别。

### 复制构造对象

在复制处理资源的对象时，需要分配新资源，并且需要复制源对象的资源，以便使这两个对象完全分离。想象一下，我们有一个类`Widget`，它引用需要在构造时分配的某种资源。以下代码默认构造了一个`Widget`对象，然后复制构造了一个新实例：

```cpp
auto a = Widget{}; 
auto b = a;        // Copy-construction 
```

所进行的资源分配如下图所示：

![](img/B15619_02_01.png)

图 2.1：复制具有资源的对象

分配和复制是缓慢的过程，在许多情况下，源对象不再需要。使用移动语义，编译器会检测到这样的情况，其中旧对象不与变量绑定，而是执行移动操作。

#### 交换两个对象

在 C++11 中添加移动语义之前，交换两个对象的内容是一种常见的在不分配和复制的情况下传输数据的方式。如下所示，对象只是互相交换它们的内容：

```cpp
auto a = Widget{};
auto b = Widget{};
std::swap(a, b); 
```

以下图示说明了这个过程：

![](img/B15619_02_02.png)

图 2.2：在两个对象之间交换资源

`std::swap()`函数是一个简单但有用的实用程序，在本章后面将介绍的复制和交换习语中使用。

#### 移动构造对象

移动对象时，目标对象直接从源对象中夺取资源，而源对象被重置。

正如您所见，这与交换非常相似，只是*移出*的对象不必从*移入*对象那里接收资源：

```cpp
auto a = Widget{}; 
auto b = std::move(a); // Tell the compiler to move the resource into b 
```

以下图示说明了这个过程：

![](img/B15619_02_03.png)

图 2.3：将资源从一个对象移动到另一个对象

尽管源对象被重置，但它仍处于有效状态。源对象的重置不是编译器自动为我们执行的。相反，我们需要在移动构造函数中实现重置，以确保对象处于可以被销毁或赋值的有效状态。我们将在本章后面更多地讨论有效状态。

只有在对象类型拥有某种资源（最常见的情况是堆分配的内存）时，移动对象才有意义。如果所有数据都包含在对象内部，移动对象的最有效方式就是简单地复制它。

现在您已经基本掌握了移动语义，让我们深入了解一下细节。

## 资源获取和五法则

要完全理解移动语义，我们需要回到 C++中类和资源获取的基础概念。C++中的一个基本概念是，一个类应该完全处理其资源。这意味着当一个类被复制、移动、复制赋值、移动赋值或销毁时，类应该确保其资源得到适当处理。实现这五个函数的必要性通常被称为**五法则**。

```cpp
floats pointed at by the raw pointer ptr_:
```

```cpp
class Buffer { 
public: 
  // Constructor 
  Buffer(const std::initializer_list<float>& values)       : size_{values.size()} { 
    ptr_ = new float[values.size()]; 
    std::copy(values.begin(), values.end(), ptr_); 
  }
  auto begin() const { return ptr_; } 
  auto end() const { return ptr_ + size_; } 
  /* The 5 special functions are defined below */
private: 
  size_t size_{0}; 
  float* ptr_{nullptr};
}; 
```

在这种情况下，处理的资源是在`Buffer`类的构造函数中分配的一块内存。内存可能是类处理的最常见资源，但资源可以是更多：互斥锁、图形卡上纹理的句柄、线程句柄等等。

在“五法则”中提到的五个函数已被省略，将在下文中介绍。我们将从复制构造函数、复制赋值运算符和析构函数开始，这些函数都需要参与资源处理：

```cpp
// 1\. Copy constructor 
Buffer::Buffer(const Buffer& other) : size_{other.size_} { 
  ptr_ = new float[size_]; 
  std::copy(other.ptr_, other.ptr_ + size_, ptr_); 
} 
// 2\. Copy assignment 
auto& Buffer::operator=(const Buffer& other) {
  delete [] ptr_;
  ptr_ = new float[other.size_];
  size_ = other.size_;
  std::copy(other.ptr_, other.ptr_ + size_, ptr_);
  return *this;
} 
// 3\. Destructor 
Buffer::~Buffer() { 
  delete [] ptr_; // OK, it is valid to delete a nullptr
  ptr_ = nullptr;  
} 
```

在 C++11 中引入移动语义之前，这三个函数通常被称为**三法则**。复制构造函数、复制赋值运算符和析构函数在以下情况下被调用：

```cpp
auto func() { 
  // Construct 
  auto b0 = Buffer({0.0f, 0.5f, 1.0f, 1.5f}); 
  // 1\. Copy-construct 
  auto b1 = b0; 
  // 2\. Copy-assignment as b0 is already initialized 
  b0 = b1; 
} // 3\. End of scope, the destructors are automatically invoked 
```

虽然正确实现这三个函数是类处理内部资源所需的全部内容，但会出现两个问题：

+   **无法复制的资源**：在`Buffer`类示例中，我们的资源可以被复制，但还有其他类型的资源，复制是没有意义的。例如，类中包含的资源可能是`std::thread`、网络连接或其他无法复制的资源。在这些情况下，我们无法传递对象。

+   **不必要的复制**：如果我们从函数中返回我们的`Buffer`类，整个数组都需要被复制。（编译器在某些情况下会优化掉复制，但现在让我们忽略这一点。）

解决这些问题的方法是移动语义。除了复制构造函数和复制赋值，我们还可以在我们的类中添加移动构造函数和移动赋值运算符。移动版本不是以`const`引用（`const Buffer&`）作为参数，而是接受`Buffer&&`对象。

`&&`修饰符表示参数是我们打算从中移动而不是复制的对象。用 C++术语来说，这被称为 rvalue，我们稍后会更详细地讨论这些。

而`copy()`函数复制对象，移动等效函数旨在将资源从一个对象移动到另一个对象，释放被移动对象的资源。

这就是我们如何通过移动构造函数和移动赋值来扩展我们的`Buffer`类。如您所见，这些函数不会抛出任何异常，因此可以标记为`noexcept`。这是因为，与复制构造函数/复制赋值相反，它们不会分配内存或执行可能引发异常的操作：

```cpp
// 4\. Move constructor
Buffer::Buffer(Buffer&& other) noexcept     : size_{other.size_}, ptr_{other.ptr_} {
  other.ptr_ = nullptr;
  other.size_ = 0;
}
// 5\. Move assignment
auto& Buffer::operator=(Buffer&& other) noexcept {
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
  return *this;
} 
```

现在，当编译器检测到我们执行了似乎是复制的操作，例如从函数返回一个`Buffer`，但复制的值不再被使用时，它将使用不抛出异常的移动构造函数/移动赋值代替复制。

这非常棒；接口保持与复制时一样清晰，但在底层，编译器执行了一个简单的移动。因此，程序员不需要使用任何奇怪的指针或输出参数来避免复制；因为类已经实现了移动语义，编译器会自动处理这个问题。

不要忘记将您的移动构造函数和移动赋值运算符标记为`noexcept`（除非它们可能抛出异常）。不标记它们为`noexcept`会阻止标准库容器和算法在某些条件下使用它们，而是转而使用常规的复制/赋值。

为了能够知道编译器何时允许移动对象而不是复制，需要了解 rvalue。

## 命名变量和 rvalue

那么，编译器何时允许移动对象而不是复制呢？简短的答案是，当对象可以被归类为 rvalue 时，编译器会移动对象。术语**rvalue**听起来可能很复杂，但本质上它只是一个不与命名变量绑定的对象，原因如下：

+   它直接来自函数

+   通过使用`std::move()`，我们可以将变量变成 rvalue

以下示例演示了这两种情况：

```cpp
// The object returned by make_buffer is not tied to a variable
x = make_buffer();  // move-assigned
// The variable "x" is passed into std::move()
y = std::move(x);   // move-assigned 
```

在本书中，我还将交替使用术语**lvalue**和**命名变量**。lvalue 对应于我们在代码中可以通过名称引用的对象。

现在，我们将通过在类中使用`std::string`类型的成员变量来使其更加高级。以下的`Button`类将作为一个例子：

```cpp
class Button { 
public: 
  Button() {} 
  auto set_title(const std::string& s) { 
    title_ = s; 
  } 
  auto set_title(std::string&& s) { 
    title_ = std::move(s); 
  } 
  std::string title_; 
}; 
```

我们还需要一个返回标题和`Button`变量的自由函数：

```cpp
auto get_ok() {
  return std::string("OK");
}
auto button = Button{}; 
```

在满足这些先决条件的情况下，让我们详细看一些复制和移动的案例：

+   **Case 1**：`Button::title_`被移动赋值，因为`string`对象通过`std::move()`传递：

```cpp
auto str = std::string{"OK"};
button.set_title(str);              // copy-assigned 
```

+   **Case 2**：`Button::title_`被移动赋值，因为`str`通过`std::move()`传递：

```cpp
auto str = std::string{"OK"};
button.set_title(std::move(str));   // move-assigned 
```

+   **Case 3**：`Button::title_`被移动赋值，因为新的`std::string`对象直接来自函数：

```cpp
button.set_title(get_ok());        // move-assigned 
```

+   **Case 4**：`Button::title_`被复制赋值，因为`string`对象与`s`绑定（这与*Case 1*相同）：

```cpp
auto str = get_ok();
button.set_title(str);             // copy-assigned 
```

+   **Case 5**：`Button::title_`被复制赋值，因为`str`被声明为`const`，因此不允许改变：

```cpp
const auto str = get_ok();
button.set_title(std::move(str));  // copy-assigned 
```

如您所见，确定对象是移动还是复制非常简单。如果它有一个变量名，它就会被复制；否则，它就会被移动。如果您正在使用`std::move()`来移动一个命名对象，那么该对象就不能被声明为`const`。

## 默认移动语义和零规则

本节讨论自动生成的复制赋值运算符。重要的是要知道生成的函数没有强异常保证。因此，如果在复制赋值期间抛出异常，对象可能最终处于部分复制的状态。

与复制构造函数和复制赋值一样，移动构造函数和移动赋值可以由编译器生成。尽管一些编译器允许在某些条件下自动生成这些函数（稍后会详细介绍），但我们可以通过使用`default`关键字简单地强制编译器生成它们。

对于不手动处理任何资源的`Button`类，我们可以简单地扩展它如下：

```cpp
class Button {
public: 
  Button() {} // Same as before

  // Copy-constructor/copy-assignment 
  Button(const Button&) = default; 
  auto operator=(const Button&) -> Button& = default;
  // Move-constructor/move-assignment 
  Button(Button&&) noexcept = default; 
  auto operator=(Button&&) noexcept -> Button& = default; 
  // Destructor
  ~Button() = default; 
  // ...
}; 
```

更简单的是，如果我们不声明*任何*自定义复制构造函数/复制赋值或析构函数，移动构造函数/移动赋值将被隐式声明，这意味着第一个`Button`类实际上处理了一切：

```cpp
class Button {
public: 
  Button() {} // Same as before

  // Nothing here, the compiler generates everything automatically! 
  // ...
}; 
```

很容易忘记只添加五个函数中的一个会阻止编译器生成其他函数。以下版本的`Button`类具有自定义析构函数。因此，移动运算符不会生成，并且该类将始终被复制：

```cpp
class Button {
public: 
  Button() {} 
  ~Button() 
    std::cout << "destructed\n"
  }
  // ...
}; 
```

让我们看看在实现应用程序类时如何使用这些生成函数的见解。

### 实际代码库中的零规则

实际上，必须编写自己的复制/移动构造函数、复制/移动赋值和构造函数的情况应该非常少。编写类，使其不需要显式编写任何这些特殊成员函数（或声明为`default`）通常被称为**零规则**。这意味着如果应用程序代码库中的类需要显式编写任何这些函数，那么该代码片段可能更适合于代码库的一部分。

在本书的后面，我们将讨论`std::optional`，这是一个方便的实用类，用于处理可选成员，同时应用零规则。

#### 关于空析构函数的说明

编写空析构函数可以防止编译器实现某些优化。如下片段所示，使用具有空析构函数的平凡类的数组复制产生与使用手工制作的`for`循环复制相同（非优化）的汇编代码。第一个版本使用具有`std::copy()`的空析构函数：

```cpp
struct Point {
 int x_, y_;
 ~Point() {}     // Empty destructor, don't use!
};
auto copy(Point* src, Point* dst) {
  std::copy(src, src+64, dst);
} 
```

第二个版本使用了一个没有析构函数但有手工制作的`for`循环的`Point`类：

```cpp
struct Point {
  int x_, y_;
};
auto copy(Point* src, Point* dst) {
  const auto end = src + 64;
  for (; src != end; ++src, ++dst) {
    *dst = *src;
  }
} 
```

两个版本生成以下 x86 汇编代码，对应一个简单的循环：

```cpp
 xor eax, eax
.L2:
 mov rdx, QWORD PTR [rdi+rax]
 mov QWORD PTR [rsi+rax], rdx
 add rax, 8
 cmp rax, 512
 jne .L2
 rep ret 
```

但是，如果我们删除析构函数或声明析构函数为`default`，编译器将优化`std::copy()`以利用`memmove()`而不是循环：

```cpp
struct Point { 
  int x_, y_; 
  ~Point() = default; // OK: Use default or no constructor at all
};
auto copy(Point* src, Point* dst) {
  std::copy(src, src+64, dst);
} 
```

前面的代码生成以下 x86 汇编代码，带有`memmove()`优化：

```cpp
 mov rax, rdi
 mov edx, 512
 mov rdi, rsi
 mov rsi, rax
 jmp memmove 
```

汇编是使用*Compiler Explorer*中的 GCC 7.1 生成的，可在[`godbolt.org/`](https://godbolt.org/)上找到。

总之，使用`default`析构函数或根本不使用析构函数，以便在应用程序中挤出更多性能。

### 一个常见的陷阱-移动非资源

在使用默认创建的移动赋值时存在一个常见的陷阱：将基本类型与更高级的复合类型混合使用。与复合类型相反，基本类型（如`int`、`float`和`bool`）在移动时只是被复制，因为它们不处理任何资源。

当简单类型与拥有资源的类型混合在一起时，移动赋值成为移动和复制的混合。

这是一个将失败的类的示例：

```cpp
class Menu {
public:
  Menu(const std::initializer_list<std::string>& items)       : items_{items} {}
  auto select(int i) {
    index_ = i;
  }
  auto selected_item() const {
     return index_ != -1 ? items_[index_] : "";
  }
  // ...
private:
  int index_{-1}; // Currently selected item
  std::vector<std::string> items_; 
}; 
```

如果像这样使用`Menu`类，它将具有未定义的行为：

```cpp
auto a = Menu{"New", "Open", "Close", "Save"};
a.select(2);
auto b = std::move(a);
auto selected = a.selected_item(); // crash 
```

未定义的行为发生在`items_`向量被移动并且因此为空。另一方面，`index_`被复制，因此在移动的对象`a`中仍然具有值`2`。当调用`selected_item()`时，函数将尝试访问索引`2`处的`items_`，程序将崩溃。

在这些情况下，移动构造函数/赋值最好通过简单交换成员来实现，就像这样：

```cpp
Menu(Menu&& other) noexcept { 
  std::swap(items_, other.items_); 
  std::swap(index_, other.index_); 
} 
auto& operator=(Menu&& other) noexcept { 
  std::swap(items_, other.items_); 
  std::swap(index_, other.index_); 
  return *this; 
} 
```

这种方式，`Menu`类可以安全地移动，同时保留无抛出保证。在*第八章*，*编译时编程*中，您将学习如何利用 C++中的反射技术来自动创建交换元素的移动构造函数/赋值函数。

## 将`&&`修饰符应用于类成员函数

除了应用于对象之外，您还可以向类的成员函数添加`&&`修饰符，就像您可以向成员函数应用`const`修饰符一样。与`const`修饰符一样，具有`&&`修饰符的成员函数只有在对象是右值时才会被重载解析考虑：

```cpp
struct Foo { 
  auto func() && {} 
}; 
auto a = Foo{}; 
a.func();            // Doesn't compile, 'a' is not an rvalue 
std::move(a).func(); // Compiles 
Foo{}.func();        // Compiles 
```

也许有些奇怪，有人会想要这种行为，但确实有用例。我们将在*第十章*，*代理对象和延迟评估*中调查其中之一。

## 当复制被省略时不要移动

当从函数返回值时，可能会诱人使用`std::move()`，就像这样：

```cpp
auto func() {
  auto x = X{};
  // ...
  return std::move(x);  // Don't, RVO is prevented
} 
```

然而，除非`x`是一个仅移动类型，否则不应该这样做。使用`std::move()`会阻止编译器使用**返回值优化**（**RVO**），从而完全省略了`x`的复制，这比移动更有效。因此，当通过值返回新创建的对象时，不要使用`std::move()`；而是直接返回对象：

```cpp
auto func() {
  auto x = X{};
  // ...
  return x;  // OK
} 
```

这种特定的例子，其中*命名*对象被省略，通常称为**NRVO**或**Named-RVO**。 RVO 和 NRVO 由今天所有主要的 C++编译器实现。如果您想了解更多关于 RVO 和复制省略的信息，您可以在[`en.cppreference.com/w/cpp/language/copy_elision`](https://en.cppreference.com/w/cpp/language/copy_elision)找到详细的摘要。

## 在适用时传递值

考虑一个将`std::string`转换为小写的函数。为了在适用时使用移动构造函数，在不适用时使用复制构造函数，似乎需要两个函数：

```cpp
// Argument s is a const reference
auto str_to_lower(const std::string& s) -> std::string {
  auto clone = s;
  for (auto& c: clone) c = std::tolower(c);
  return clone;
}
// Argument s is an rvalue reference
auto str_to_lower(std ::string&& s) -> std::string {
  for (auto& c: s) c = std::tolower(c);
  return s;
} 
```

然而，通过按值传递`std::string`，我们可以编写一个函数来涵盖这两种情况：

```cpp
auto str_to_lower(std::string s) -> std::string {
  for (auto& c: s) c = std::tolower(c);
  return s;
} 
```

让我们看看`str_to_lower()`的这种实现如何避免可能的不必要的复制。当传递一个常规变量时，如下所示，函数调用之前`str`的内容被复制构造到`s`中，然后在函数返回时移动分配回`str`：

```cpp
auto str = std::string{"ABC"};
str = str_to_lower(str); 
```

当传递一个右值时，如下所示，函数调用之前`str`的内容被移动构造到`s`中，然后在函数返回时移动分配回`str`。因此，没有通过函数调用进行复制：

```cpp
auto str = std::string{"ABC"};
str = str_to_lower(std::move(str)); 
```

乍一看，这种技术似乎适用于所有参数。然而，这种模式并不总是最佳的，接下来您将看到。

### 不适用传值的情况

有时，接受按值然后移动的模式实际上是一种悲观化。例如，考虑以下类，其中函数`set_data()`将保留传递给它的参数的副本：

```cpp
class Widget {
  std::vector<int> data_{};
  // ...
public:
  void set_data(std::vector<int> x) { 
    data_ = std::move(x);               
  }
}; 
```

假设我们调用`set_data()`并将一个左值传递给它，就像这样：

```cpp
auto v = std::vector<int>{1, 2, 3, 4};
widget.set_data(v);                  // Pass an lvalue 
```

由于我们传递了一个命名对象`v`，代码将复制构造一个新的`std::vector`对象`x`，然后将该对象移动分配到`data_`成员中。除非我们将一个空的向量对象传递给`set_data()`，否则`std::vector`复制构造函数将为其内部缓冲区执行堆分配。

现在将其与`set_data()`的以下版本进行比较，该版本针对左值进行了优化：

```cpp
void set_data(const std::vector<int>& x) { 
    data_ = x;  // Reuse internal buffer in data_ if possible
} 
```

在这里，如果当前向量`data_`的容量小于源对象`x`的大小，那么赋值运算符内部将只有一个堆分配。换句话说，在许多情况下，`data_`的内部预分配缓冲区可以在赋值运算符中被重用，从而避免额外的堆分配。

如果我们发现有必要优化`set_data()`以适应 lvalues 和 rvalues，最好在这种情况下提供两个重载：

```cpp
void set_data(const std::vector<int>& x) {
  data_ = x;
}
void set_data(std::vector<int>&& x) noexcept { 
  data_ = std::move(x);
} 
```

第一个版本对于 lvalues 是最佳的，第二个版本对于 rvalues 是最佳的。

最后，我们现在将看一个场景，在这个场景中我们可以安全地传值，而不用担心刚刚演示的悲观情况。

### 移动构造函数参数

在构造函数中初始化类成员时，我们可以安全地使用传值然后移动的模式。在构造新对象时，没有机会利用预分配的缓冲区来避免堆分配。接下来是一个具有一个`std::vector`成员和一个构造函数的类的示例，用于演示这种模式：

```cpp
class Widget {
  std::vector<int> data_;
public:
  Widget(std::vector<int> x)       // By value
      : data_{std::move(x)} {}     // Move-construct
  // ...
}; 
```

我们现在将把焦点转移到一个不能被视为*现代 C++*但即使在今天也经常被讨论的话题。

# 设计带有错误处理的接口

错误处理是函数和类接口中重要但经常被忽视的部分。错误处理是 C++中一个备受争议的话题，但讨论往往倾向于异常与其他错误机制之间的对比。虽然这是一个有趣的领域，但在关注错误处理的实际实现之前，还有其他更重要的错误处理方面需要理解。显然，异常和错误码在许多成功的软件项目中都被使用过，而且经常会遇到将两者结合在一起的项目。

无论编程语言如何，错误处理的一个基本方面是区分**编程错误**（也称为错误）和**运行时错误**。运行时错误可以进一步分为**可恢复的运行时错误**和**不可恢复的运行时错误**。不可恢复的运行时错误的一个例子是*堆栈溢出*（见*第七章*，*内存管理*）。当发生不可恢复的错误时，程序通常会立即终止，因此没有必要发出这些类型的错误。然而，一些错误在某种类型的应用程序中可能被认为是可恢复的，但在其他应用程序中是不可恢复的。

讨论可恢复和不可恢复错误时经常出现的一个边缘情况是 C++标准库在内存耗尽时的不太幸运的行为。当程序耗尽内存时，这通常是不可恢复的，但标准库在这种情况下会尝试抛出`std::bad_alloc`异常。我们不会在这里花时间讨论不可恢复的错误，但是 Herb Sutter 的演讲《De-fragmenting C++: Making Exceptions and RTTI More Affordable and Usable》（[`sched.co/SiVW`](https://sched.co/SiVW)）非常推荐，如果你想深入了解这个话题。

在设计和实现 API 时，您应该始终反思您正在处理的错误类型，因为不同类别的错误应该以完全不同的方式处理。决定错误是编程错误还是运行时错误可以通过使用一种称为**设计契约**的方法来完成；这是一个值得一本书的话题。然而，我在这里将介绍足够我们目的的基本原则。

有关在 C++中添加契约语言支持的提案，但目前契约尚未成为标准的一部分。然而，许多 C++ API 和指南都假定您了解契约的基础知识，因为契约使用的术语使得更容易讨论和记录类和函数的接口。

## 契约

**合同**是调用某个函数的调用者和函数本身（被调用者）之间的一组规则。C++允许我们使用 C++类型系统明确指定一些规则。例如，考虑以下函数签名：

```cpp
int func(float x, float y) 
```

它指定`func()`返回一个整数（除非它抛出异常），并且调用者必须传递两个浮点值。但它并没有说明允许使用什么浮点值。例如，我们可以传递值 0.0 或负值吗？此外，`x`和`y`之间可能存在一些必需的关系，这些关系不能很容易地使用 C++类型系统来表达。当我们谈论 C++中的合同时，通常指的是调用者和被调用者之间存在的一些规则，这些规则不能很容易地使用类型系统来表达。

在不太正式的情况下，这里将介绍与设计合同相关的一些概念，以便为您提供一些可以用来推理接口和错误处理的术语：

+   前置条件指定了函数的*调用者*的*责任*。对函数传递的参数可能有约束。或者，如果它是一个成员函数，在调用函数之前对象可能必须处于特定状态。例如，在`std::vector`上调用`pop_back()`时的前置条件是向量不为空。确保向量不为空是`pop_back()`的*调用者*的责任。

+   后置条件指定了函数返回时的*职责*。如果它是一个成员函数，函数在什么状态下离开对象？例如，`std::list::sort()`的后置条件是列表中的元素按升序排序。

+   不变量是一个应该始终成立的条件。不变量可以在许多情境中使用。循环不变量是每次循环迭代开始时必须为真的条件。此外，类不变量定义了对象的有效状态。例如，`std::vector`的不变量是`size() <= capacity()`。明确陈述某些代码周围的不变量使我们更好地理解代码。不变量也是一种工具，可以用来证明某些算法是否按预期运行。

类不变量非常重要；因此，我们将花费更多时间讨论它们是什么以及它们如何影响类的设计。

### 类不变量

如前所述，**类不变量**定义了对象的有效状态。它指定了类内部数据成员之间的关系。在执行成员函数时，对象可能暂时处于无效状态。重要的是，当函数将控制权传递给可以观察对象状态的其他代码时，不变量得到维持。这可能发生在函数：

+   返回

+   抛出异常

+   调用回调函数

+   调用可能观察当前调用对象状态的其他函数；一个常见的情况是将`this`的引用传递给其他函数

重要的是要意识到类不变量是类的每个成员函数的前置条件和后置条件的隐含部分。如果成员函数使对象处于无效状态，则未满足后置条件。类似地，成员函数在调用函数时始终可以假定对象处于有效状态。这条规则的例外是类的构造函数和析构函数。如果我们想要插入代码来检查类不变量是否成立，我们可以在以下点进行：

```cpp
struct Widget {
  Widget() {
    // Initialize object…
    // Check class invariant
  }
  ~Widget() {
    // Check class invariant
    // Destroy object…
   }
   auto some_func() {
     // Check precondition (including class invariant)
     // Do the actual work…
     // Check postcondition (including class invariant)
   }
}; 
```

复制/移动构造函数和复制/移动赋值运算符在这里没有提到，但它们遵循与构造函数和`some_func()`相同的模式。

当对象已被移动后，对象可能处于某种空或重置状态。这也是对象的有效状态之一，因此是类不变式的一部分。然而，通常只有少数成员函数可以在对象处于此状态时调用。例如，您不能在已移动的`std::vector`上调用`push_back()`、`empty()`或`size()`，但可以调用`clear()`，这将使向量处于准备再次使用的状态。

您应该知道，这种额外的重置状态使类不变式变得更弱，也更不实用。为了完全避免这种状态，您应该以这样的方式实现您的类，使得已移动的对象被重置为对象在默认构造后的状态。我的建议是总是这样做，除非在很少的情况下，将已移动的状态重置为默认状态会带来无法接受的性能损失。这样，您可以更好地推理有关已移动状态的情况，而且类的使用更安全，因为在该对象上调用成员函数是可以的。

如果您可以确保对象始终处于有效状态（类不变式成立），那么您可能会拥有一个难以被误用的类，如果实现中存在错误，通常很容易发现。您最不希望的是在代码库中找到一个类，并想知道该类的某些行为是一个错误还是一个特性。违反合同始终是一个严重的错误。

为了能够编写有意义的类不变式，我们需要编写具有高内聚性和少可能状态的类。如果您曾经为自己编写的类编写单元测试，您可能会注意到，在编写单元测试时，很明显可以从初始版本改进 API。单元测试迫使您使用和反思类的接口而不是实现细节。同样，类不变式使您考虑对象可能处于的所有有效状态。如果您发现很难定义类不变式，通常是因为您的类承担了太多的责任并处理了太多的状态。因此，定义类不变式通常意味着您最终会得到设计良好的类。

### 维护合同

合同是您设计和实现的 API 的一部分。但是，您如何维护和向使用您的 API 的客户端传达合同呢？C++尚没有内置支持合同的功能，但正在进行工作以将其添加到未来的 C++版本中。不过，也有一些选择：

+   使用诸如 Boost.Contract 之类的库。

+   记录合同。这样做的缺点是在运行程序时不会检查合同。此外，文档往往在代码更改时过时。

+   使用`static_assert()`和`<cassert>`中定义的`assert()`宏。断言是可移植的，标准的 C++。

+   构建一个自定义库，其中包含类似断言的自定义宏，但对失败合同的行为具有更好的控制。

在本书中，我们将使用断言，这是检查合同违规的最原始的方式之一。然而，断言可以非常有效，并对代码质量产生巨大影响。

#### 启用和禁用断言

从技术上讲，在 C++中有两种标准的断言方式：使用`<cassert>`头文件中的`static_assert()`或`assert()`宏。`static_assert()`在代码编译期间进行验证，因此需要一个可以在编译时而不是运行时进行检查的表达式。失败的`static_assert()`会导致编译错误。

对于只能在运行时评估的断言，您需要使用`assert()`宏。`assert()`宏是一种运行时检查，通常在调试和测试期间处于活动状态，并在以发布模式构建程序时完全禁用。`assert()`宏通常定义如下：

```cpp
#ifdef NDEBUG
#define assert(condition) ((void)0)
#else
#define assert(condition) /* implementation defined */
#endif 
```

这意味着您可以通过定义`NDEBUG`完全删除所有断言和用于检查条件的代码。

现在，有了一些设计合同的术语，让我们专注于合同违反（错误）以及如何在您的代码中处理它们。

## 错误处理

在设计具有适当错误处理的 API 时，首先要做的是区分编程错误和运行时错误。因此，在我们深入讨论错误处理策略之前，我们将使用设计合同来定义我们正在处理的错误类型。

### 编程错误还是运行时错误？

如果我们发现合同违反，我们也发现了我们程序中的错误。例如，如果我们可以检测到有人在空向量上调用`pop_back()`，我们知道我们的源代码中至少有一个错误需要修复。每当前提条件不满足时，我们知道我们正在处理一个*编程错误*。

另一方面，如果我们有一个从磁盘加载某个记录的函数，并且由于磁盘上的读取错误而无法返回记录，那么我们已经检测到了一个*运行时错误*：

```cpp
auto load_record(std::uint32_t id) {
  assert(id != 0);           // Precondition
  auto record = read(id);    // Read from disk, may throw
  assert(record.is_valid()); // Postcondition
  return record;
} 
```

前提条件得到满足，但由于程序外部的某些原因，后置条件无法满足。源代码中没有错误，但由于某些与磁盘相关的错误，函数无法返回在磁盘上找到的记录。由于无法满足后置条件，必须将运行时错误报告给调用者，除非调用者可以自行通过重试等方式恢复。

### 编程错误（错误）

一般来说，编写代码来发出并处理代码中的错误没有意义。相反，使用断言（或先前提到的其他一些替代方案）来使开发人员意识到代码中的问题。您应该只对可恢复的运行时错误使用异常或错误代码。

#### 通过假设缩小问题空间

断言指定了您作为某些代码的作者所做的假设。只有在您的代码中的所有断言都为真时，您才能保证代码按预期工作。这使编码变得更容易，因为您可以有效地限制需要处理的情况数量。断言在您的团队使用、阅读和修改您编写的代码时也是巨大的帮助。所有假设都以断言语句的形式清楚地记录下来。

#### 使用断言查找错误

失败的断言总是严重的错误。当您在测试过程中发现一个失败的断言时，基本上有三种选择：

+   断言是正确的，但代码是错误的（要么是因为函数实现中的错误，要么是因为调用站点上的错误）。根据我的经验，这是最常见的情况。通常情况下，使断言正确比使其周围的代码正确要容易得多。修复代码并重新测试。

+   代码是正确的，但断言是错误的。有时会发生这种情况，如果您看的是旧代码，通常会感到非常不舒服。更改或删除失败的断言可能会耗费时间，因为您需要确保代码实际上是有效的，并理解为什么旧断言突然开始失败。通常，这是因为原始作者没有考虑到一个新的用例。

+   断言和代码都是错误的。这通常需要重新设计类或函数。也许要求已经改变，程序员所做的假设不再成立。但不要绝望；相反，您应该高兴那些假设是明确地使用断言写出来的；现在您知道为什么代码不再起作用了。

运行时断言需要测试，否则断言将不会被执行。新编写的带有许多断言的代码通常在测试时会出现故障。这并不意味着您是一个糟糕的程序员；这意味着您添加了有意义的断言，可以捕获一些本来可能会进入生产的错误。此外，使测试版本的程序终止的错误也很可能会被修复。

#### 性能影响

在代码中有许多运行时断言很可能会降低测试构建的性能。然而，断言从不应该在优化程序的最终版本中使用。如果您的断言使您的测试构建速度太慢而无法使用，通常很容易在分析器中跟踪到减慢代码速度的断言集（有关分析器的更多信息，请参见*第三章*，*分析和测量性能*）。

通过使程序的发布构建完全忽略由错误引起的错误状态，程序将不会花时间检查由错误引起的错误状态。相反，您的代码将运行得更快，只花时间解决它本来要解决的实际问题。它只会检查需要恢复的运行时错误。

总结一下，编程错误应该在测试程序时被检测出来。没有必要使用异常或其他错误处理机制来处理编程错误。相反，编程错误应该记录一些有意义的东西，并终止程序，以通知程序员需要修复错误。遵循这一准则显著减少了我们需要在代码中处理异常的地方。我们在优化构建中会有更好的性能，希望由于断言失败而检测到的错误会更少。然而，有些情况下可能会发生运行时错误，这些错误需要被我们实现的代码处理和恢复。

### 可恢复的运行时错误

如果一个函数无法履行其合同的一部分（即后置条件），则发生了运行时错误，需要将其通知到可以处理并恢复有效状态的代码中。处理可恢复错误的目的是将错误从发生错误的地方传递到可以恢复有效状态的地方。有许多方法可以实现这一点。这是一个硬币的两面：

+   对于信号部分，我们可以选择 C++异常、错误代码、返回`std::optional`或`std::pair`，或使用`boost::outcome`或`std::experimental::expected`。

+   保持程序的有效状态而不泄漏任何资源。确定性析构函数和自动存储期是 C++中使这成为可能的工具。

实用类`std::optional`和`std::pair`将在*第九章*，*基本实用程序*中介绍。现在我们将专注于 C++异常以及如何在从错误中恢复时避免泄漏资源。

#### 异常

异常是 C++提供的标准错误处理机制。该语言设计用于与异常一起使用。一个例子是构造函数失败；从构造函数中发出错误的唯一方法是使用异常。

根据我的经验，异常以许多不同的方式使用。造成这种情况的一个原因是不同的应用在处理运行时错误时可能有非常不同的要求。对于一些应用，比如起搏器或发电厂控制系统，如果它们崩溃可能会产生严重影响，我们可能必须处理每种可能的异常情况，比如内存耗尽，并保持应用程序处于运行状态。有些应用甚至完全不使用堆内存，要么是因为平台根本没有可用的堆，要么是因为堆引入了无法控制的不确定性，因为分配新内存的机制超出了应用程序的控制。

我假设您已经知道抛出和捕获异常的语法，并且不会在这里涵盖它。可以标记为`noexcept`的函数保证不会抛出异常。重要的是要理解编译器*不*验证这一点；相反，这取决于代码的作者来弄清楚他们的函数是否可能抛出异常。

标记为`noexcept`的函数在某些情况下可以使编译器生成更快的代码。如果从标记为`noexcept`的函数中抛出异常，程序将调用`std::terminate()`而不是展开堆栈。以下代码演示了如何将函数标记为不抛出异常：

```cpp
auto add(int a, int b) noexcept {
  return a + b;
} 
```

您可能会注意到，本书中的许多代码示例即使在生产代码中也适用`noexcept`（或`const`），也没有使用。这仅仅是因为书的格式；如果在我通常会添加`noexcept`和`const`的所有地方添加它们，会使代码难以阅读。

#### 保持有效状态

异常处理要求我们程序员考虑异常安全性保证；也就是说，在异常发生之前和之后程序的状态是什么？强异常安全性可以被视为一个事务。一个函数要么提交所有状态更改，要么在发生异常时执行完全回滚。

为了使这更具体化，让我们来看一个简单的函数：

```cpp
void func(std::string& str) {
  str += f1();  // Could throw
  str += f2();  // Could throw
} 
```

该函数将`f1()`和`f2()`的结果附加到字符串`str`。现在考虑一下，如果调用函数`f2()`时抛出异常会发生什么；只有`f1()`的结果会附加到`str`。相反，我们希望在发生异常时`str`保持不变。这可以通过使用一种称为**复制和交换**的惯用法来解决。这意味着我们在让应用程序状态被非抛出`swap()`函数修改之前，在临时副本上执行可能引发异常的操作：

```cpp
void func(std::string& str) {
  auto tmp = std::string{str};  // Copy
  tmp += f1();                  // Mutate copy, may throw
  tmp += f2();                  // Mutate copy, may throw
  std::swap(tmp, str);          // Swap, never throws
} 
```

相同的模式可以在成员函数中使用，以保持对象的有效状态。假设我们有一个类，其中包含两个数据成员和一个类不变式，该不变式规定数据成员不能相等，如下所示：

```cpp
class Number { /* ... */ };
class Widget {
public:
  Widget(const Number& x, const Number& y) : x_{x}, y_{y} {
    assert(is_valid());           // Check class invariant
  }
private:
  Number x_{};
  Number y_{};
  bool is_valid() const {         // Class invariant
   return x_ != y_;               // x_ and y_ must not be equal
  }
}; 
```

接下来，假设我们正在添加一个成员函数，该函数更新两个数据成员，如下所示：

```cpp
void Widget::update(const Number& x, const Number& y) {
  assert(x != y && is_valid());   // Precondition
  x_ = x;
  y_ = y;          
  assert(is_valid());             // Postcondition
} 
```

前提条件规定`x`和`y`不能相等。如果`x_`和`y_`的赋值可能会抛出异常，`x_`可能会被更新，但`y_`不会。这可能导致破坏类不变式；也就是说，对象处于无效状态。如果发生错误，我们希望函数保持对象在赋值操作之前的有效状态。再次，一个可能的解决方案是使用复制和交换惯用法：

```cpp
void Widget::update(const Number& x, const Number& y) {
    assert(x != y && is_valid());     // Precondition
    auto x_tmp = x;  
    auto y_tmp = y;  
    std::swap(x_tmp, x_); 
    std::swap(y_tmp, y_); 
    assert(is_valid());               // Postcondition
  } 
```

首先，创建本地副本，而不修改对象的状态。然后，如果没有抛出异常，可以使用非抛出`swap()`来更改对象的状态。复制和交换惯用法也可以在实现赋值运算符时使用，以实现强异常安全性保证。

错误处理的另一个重要方面是避免在发生错误时泄漏资源。

#### 资源获取

C++对象的销毁是可预测的，这意味着我们完全控制我们何时以及以何种顺序释放我们获取的资源。在下面的示例中进一步说明了这一点，当退出函数时，互斥变量`m`总是被解锁，因为作用域锁在我们退出作用域时释放它，无论我们如何以及在何处退出：

```cpp
auto func(std::mutex& m, bool x, bool y) {
  auto guard = std::scoped_lock{m}; // Lock mutex 
  if (x) { 
    // The guard automatically releases the mutex at early exit
    return; 
  }
  if (y) {
    // The guard automatically releases if an exception is thrown
    throw std::exception{};
  }
  // The guard automatically releases the mutex at function exit
} 
```

所有权、对象的生命周期和资源获取是 C++中的基本概念，我们将在*第七章* *内存管理*中进行讨论。

#### 性能

不幸的是，异常在性能方面声誉不佳。一些担忧是合理的，而一些是基于历史观察的，当时编译器没有有效地实现异常。然而，今天人们放弃异常的两个主要原因是：

+   即使不抛出异常，二进制程序的大小也会增加。尽管这通常不是问题，但它并不遵循零开销原则，因为我们为我们不使用的东西付费。

+   抛出和捕获异常相对昂贵。抛出和捕获异常的运行时成本是不确定的。这使得异常在具有硬实时要求的情况下不适用。在这种情况下，其他替代方案，如返回带有返回值和错误代码的`std::pair`可能更好。

另一方面，当没有抛出异常时，异常的性能表现非常出色；也就是说，当程序遵循成功路径时。其他错误报告机制，如错误代码，即使在程序没有任何错误时也需要在`if-else`语句中检查返回代码。

异常情况应该很少发生，通常当异常发生时，异常处理所增加的额外性能损耗通常不是这些情况的问题。通常可以在一些性能关键代码运行之前或之后执行可能引发异常的计算。这样，我们可以避免在程序中不能容忍异常的地方抛出和捕获异常。

为了公平比较异常和其他错误报告机制，重要的是要指定要比较的内容。有时异常与根本没有错误处理的情况进行比较是不公平的；异常需要与提供相同功能的机制进行比较。在你测量它们可能产生的影响之前，不要因为性能原因而放弃异常。你可以在下一章中了解更多关于分析和测量性能的内容。

现在我们将远离错误处理，探讨如何使用 lambda 表达式创建函数对象。

# 函数对象和 lambda 表达式

Lambda 表达式，引入于 C++11，并在每个 C++版本中进一步增强，是现代 C++中最有用的功能之一。它们的多功能性不仅来自于轻松地将函数传递给算法，还来自于在许多需要传递代码的情况下的使用，特别是可以将 lambda 存储在`std::function`中。

尽管 lambda 使得这些编程技术变得更加简单易用，但本节提到的所有内容都可以在没有 lambda 的情况下执行。lambda，或者更正式地说，lambda 表达式是构造函数对象的一种便捷方式。但是，我们可以不使用 lambda 表达式，而是实现重载了`operator()`的类，然后实例化这些类来创建函数对象。

我们将在稍后探讨 lambda 与这些类的相似之处，但首先我将在一个简单的用例中介绍 lambda 表达式。

## C++ lambda 的基本语法

简而言之，lambda 使程序员能够像传递变量一样轻松地将函数传递给其他函数。

让我们比较将 lambda 传递给算法和将变量传递给算法：

```cpp
// Prerequisite 
auto v = std::vector{1, 3, 2, 5, 4}; 

// Look for number three 
auto three = 3; 
auto num_threes = std::count(v.begin(), v.end(), three); 
// num_threes is 1 

// Look for numbers which is larger than three 
auto is_above_3 = [](int v) { return v > 3; }; 
auto num_above_3 = std::count_if(v.begin(), v.end(), is_above_3);
// num_above_3 is 2 
```

在第一种情况下，我们将一个变量传递给`std::count()`，而在后一种情况下，我们将一个函数对象传递给`std::count_if()`。这是 lambda 的典型用例；我们传递一个函数，由另一个函数（在本例中是`std::count_if()`）多次评估。

此外，lambda 不需要与变量绑定；就像我们可以将变量直接放入表达式中一样，我们也可以将 lambda 放入表达式中：

```cpp
auto num_3 = std::count(v.begin(), v.end(), 3); 
auto num_above_3 = std::count_if(v.begin(), v.end(), [](int i) { 
  return i > 3; 
}); 
```

到目前为止，你看到的 lambda 被称为**无状态 lambda**；它们不复制或引用 lambda 外部的任何变量，因此不需要任何内部状态。让我们通过使用捕获块引入**有状态 lambda**来使其更加高级。

## 捕获子句

在前面的例子中，我们在 lambda 中硬编码了值`3`，以便我们始终计算大于三的数字。如果我们想在 lambda 中使用外部变量怎么办？我们通过将外部变量放入**捕获子句**（即 lambda 的`[]`部分）来捕获外部变量：

```cpp
auto count_value_above(const std::vector<int>& v, int x) { 
  auto is_above = x { return i > x; }; 
  return std::count_if(v.begin(), v.end(), is_above); 
} 
```

在这个例子中，我们通过将变量`x`复制到 lambda 中来捕获它。如果我们想要将`x`声明为引用，我们在开头加上`&`，像这样：

```cpp
auto is_above = &x { return i > x; }; 
```

该变量现在只是外部`x`变量的引用，就像 C++中的常规引用变量一样。当然，我们需要非常小心引用到 lambda 中的对象的生命周期，因为 lambda 可能在引用的对象已经不存在的情况下执行。因此，通过值捕获更安全。

### 通过引用捕获与通过值捕获

使用捕获子句引用和复制变量的工作方式与常规变量一样。看看这两个例子，看看你能否发现区别：

| 通过值捕获 | 通过引用捕获 |
| --- | --- |

|

```cpp
auto func() {
  auto vals = {1,2,3,4,5,6};
  auto x = 3;
  auto is_above = x {
    return v > x;
  };
  x = 4;
  auto count_b = std::count_if(
    vals.begin(),
    vals.end(),
    is_above
   );  // count_b equals 3 } 
```

|

```cpp
auto func() {
  auto vals = {1,2,3,4,5,6};
  auto x = 3;
  auto is_above = &x {
    return v > x;
  };
  x = 4;
  auto count_b = std::count_if(
    vals.begin(),
    vals.end(),
    is_above
   );  // count_b equals 2 } 
```

|

在第一个例子中，`x`被*复制*到 lambda 中，因此当`x`被改变时不受影响；因此`std::count_if()`计算的是大于 3 的值的数量。

在第二个例子中，`x`被*引用捕获*，因此`std::count_if()`实际上计算的是大于 4 的值的数量。

### lambda 和类之间的相似之处

我之前提到过，lambda 表达式生成函数对象。函数对象是一个具有调用运算符`operator()()`定义的类的实例。

要理解 lambda 表达式的组成，你可以将其视为具有限制的常规类：

+   该类只包含一个成员函数

+   捕获子句是类的成员变量和其构造函数的组合

下表显示了 lambda 表达式和相应的类。左列使用*通过值捕获*，右列使用*通过引用捕获*：

| 通过值捕获的 lambda... | 通过引用捕获的 lambda... |
| --- | --- |

|

```cpp
auto x = 3;auto is_above = x { return y > x;};auto test = is_above(5); 
```

|

```cpp
auto x = 3;auto is_above = &x { return y > x;};auto test = is_above(5); 
```

|

| ...对应于这个类： | ...对应于这个类： |
| --- | --- |

|

```cpp
auto x = 3;class IsAbove {
public: IsAbove(int x) : x{x} {} auto operator()(int y) const {   return y > x; }private: int x{}; // Value };auto is_above = IsAbove{x};
auto test = is_above(5); 
```

|

```cpp
auto x = 3;class IsAbove {
public: IsAbove(int& x) : x{x} {} auto operator()(int y) const {   return y > x; }private: int& x; // Reference };
auto is_above = IsAbove{x};
auto test = is_above(5); 
```

|

由于 lambda 表达式，我们不必手动实现这些函数对象类型作为类。

### 初始化捕获变量

如前面的例子所示，捕获子句初始化了相应类中的成员变量。这意味着我们也可以在 lambda 中初始化成员变量。这些变量只能在 lambda 内部可见。下面是一个初始化名为`numbers`的捕获变量的 lambda 的示例：

```cpp
auto some_func = [numbers = std::list<int>{4,2}]() {
  for (auto i : numbers)
    std::cout << i;
};
some_func();  // Output: 42 
```

相应的类看起来像这样：

```cpp
class SomeFunc {
public:
 SomeFunc() : numbers{4, 2} {}
 void operator()() const {
  for (auto i : numbers)
    std::cout << i;
 }
private:
 std::list<int> numbers;
};
auto some_func = SomeFunc{};
some_func(); // Output: 42 
```

在捕获中初始化变量时，你可以想象在变量名前面有一个隐藏的`auto`关键字。在这种情况下，你可以将`numbers`视为被定义为`auto numbers = std::list<int>{4, 2}`。如果你想初始化一个引用，你可以在名称前面使用一个`&`，这对应于`auto&`。下面是一个例子：

```cpp
auto x = 1;
auto some_func = [&y = x]() {
  // y is a reference to x
}; 
```

同样，当引用（而不是复制）lambda 外部的对象时，你必须非常小心对象的生命周期。

在 lambda 中也可以移动对象，这在使用`std::unique_ptr`等移动类型时是必要的。以下是如何实现的：

```cpp
auto x = std::make_unique<int>(); 
auto some_func = [x = std::move(x)]() {
  // Use x here..
}; 
```

这也表明在 lambda 中使用相同的名称（`x`）是可能的。这并非必须。相反，我们可以在 lambda 内部使用其他名称，例如`[y = std::move(x)]`。

### 改变 lambda 成员变量

由于 lambda 的工作方式就像一个具有成员变量的类，它也可以改变它们。然而，lambda 的函数调用运算符默认为`const`，因此我们需要使用`mutable`关键字明确指定 lambda 可以改变其成员。在下面的示例中，lambda 在每次调用时改变`counter`变量：

```cpp
auto counter_func = [counter = 1]() mutable {
  std::cout << counter++;
};
counter_func(); // Output: 1
counter_func(); // Output: 2
counter_func(); // Output: 3 
```

如果 lambda 只通过引用捕获变量，我们不必在声明中添加`mutable`修饰符，因为 lambda 本身不会改变。可变和不可变 lambda 之间的区别在下面的代码片段中进行了演示：

| 通过值捕获 | 通过引用捕获 |
| --- | --- |

|

```cpp
auto some_func() {
  auto v = 7;
  auto lambda = [v]() mutable {
    std::cout << v << " ";
    ++v;
  };
  assert(v == 7);
  lambda();  lambda();
  assert(v == 7);
  std::cout << v;
} 
```

|

```cpp
auto some_func() {
  auto v = 7;
  auto lambda = [&v]() {
    std::cout << v << " ";
    ++v;
  };
  assert(v == 7);
  lambda();
  lambda();
  assert(v == 9);
  std::cout << v;
} 
```

|

| 输出：`7 8 7` | 输出：`7 8 9` |
| --- | --- |

在右侧的示例中，`v`被引用捕获，lambda 将改变`some_func()`作用域拥有的变量`v`。左侧列中的可变 lambda 只会改变 lambda 本身拥有的`v`的副本。这就是为什么我们会得到两个版本中不同的输出的原因。

#### 从编译器的角度改变成员变量

要理解前面示例中发生了什么，看一下编译器如何看待前面的 lambda 对象：

| 通过值捕获 | 通过引用捕获 |
| --- | --- |

|

```cpp
class Lambda {
 public:
 Lambda(int m) : v{m} {}
 auto operator()() {
   std::cout<< v << " ";
   ++v;
 }
private:
  int v{};
}; 
```

|

```cpp
class Lambda {
 public:
 Lambda(int& m) : v{m} {}
 auto operator()() const {
   std::cout<< v << " ";
   ++v;
 }
private:
 int& v;
}; 
```

|

正如你所看到的，第一种情况对应于具有常规成员的类，而通过引用捕获的情况只是对应于成员变量是引用的类。

你可能已经注意到我们在通过引用捕获类的`operator()`成员函数上添加了`const`修饰符，并且在相应的 lambda 上也没有指定`mutable`。这个类仍然被认为是`const`的原因是我们没有在实际的类/lambda 内部改变任何东西；实际的改变应用于引用的值，因此函数仍然被认为是`const`的。

### 捕获所有

除了逐个捕获变量，还可以通过简单地写`[=]`或`[&]`来捕获作用域中的所有变量。

使用`[=]`意味着每个变量都将被值捕获，而`[&]`则通过引用捕获所有变量。

如果我们在成员函数内部使用 lambda，也可以通过使用`[this]`来通过引用捕获整个对象，或者通过写`[*this]`来通过复制捕获整个对象：

```cpp
class Foo { 
public: 
 auto member_function() { 
   auto a = 0; 
   auto b = 1.0f;
   // Capture all variables by copy 
   auto lambda_0 = [=]() { std::cout << a << b; }; 
   // Capture all variables by reference 
   auto lambda_1 = [&]() { std::cout << a << b; }; 
   // Capture object by reference 
   auto lambda_2 = [this]() { std::cout << m_; }; 
   // Capture object by copy 
   auto lambda_3 = [*this]() { std::cout << m_; }; 
 }
private: 
 int m_{}; 
}; 
```

请注意，使用`[=]`并不意味着作用域内的所有变量都会被复制到 lambda 中；只有实际在 lambda 内部使用的变量才会被复制。

当通过值捕获所有变量时，可以指定通过引用捕获变量（反之亦然）。以下表格显示了捕获块中不同组合的结果：

| 捕获块 | 结果捕获类型 |
| --- | --- |

|

```cpp
int a, b, c;auto func = [=] { /*...*/ }; 
```

| 通过值捕获`a`、`b`、`c`。 |
| --- |

|

```cpp
int a, b, c;auto func = [&] { /*...*/ }; 
```

| 通过引用捕获`a`、`b`、`c`。 |
| --- |

|

```cpp
int a, b, c;auto func = [=, &c] { /*...*/ }; 
```

| 通过值捕获`a`、`b`。通过引用捕获`c`。 |
| --- |

|

```cpp
int a, b, c;auto func = [&, c] { /*...*/ }; 
```

| 通过引用捕获`a`、`b`。通过值捕获`c`。 |
| --- |

虽然使用`[&]`或`[=]`捕获所有变量很方便，但我建议逐个捕获变量，因为这样可以通过明确指出 lambda 作用域内使用了哪些变量来提高代码的可读性。

## 将 C 函数指针分配给 lambda

没有捕获的 lambda 可以隐式转换为函数指针。假设你正在使用一个 C 库，或者一个旧的 C++库，它使用回调函数作为参数，就像这样：

```cpp
extern void download_webpage(const char* url,
                              void (*callback)(int, const char*)); 
```

回调函数将以返回代码和一些下载内容的形式被调用。在调用`download_webpage()`时，可以将 lambda 作为参数传递。由于回调是常规函数指针，lambda 不能有任何捕获，必须在 lambda 前面加上加号（`+`）：

```cpp
auto lambda = +[](int result, const char* str) {
  // Process result and str
};
download_webpage("http://www.packt.com", lambda); 
```

这样，lambda 就转换为常规函数指针。请注意，lambda 不能有任何捕获，以便使用此功能。

## Lambda 类型

自 C++20 以来，没有捕获的 lambda 是可默认构造和可赋值的。通过使用`decltype`，现在可以轻松构造具有相同类型的不同 lambda 对象：

```cpp
auto x = [] {};   // A lambda without captures
auto y = x;       // Assignable
decltype(y) z;    // Default-constructible
static_assert(std::is_same_v<decltype(x), decltype(y)>); // passes
static_assert(std::is_same_v<decltype(x), decltype(z)>); // passes 
```

然而，这仅适用于没有捕获的 lambda。具有捕获的 lambda 有它们自己的唯一类型。即使两个具有捕获的 lambda 函数是彼此的克隆，它们仍然具有自己的唯一类型。因此，不可能将一个具有捕获的 lambda 分配给另一个 lambda。

## Lambda 和 std::function

如前一节所述，具有捕获的 lambda（有状态的 lambda）不能相互赋值，因为它们具有唯一的类型，即使它们看起来完全相同。为了能够存储和传递具有捕获的 lambda，我们可以使用`std::function`来保存由 lambda 表达式构造的函数对象。

`std::function`的签名定义如下：

```cpp
std::function< return_type ( parameter0, parameter1...) > 
```

因此，返回空并且没有参数的`std::function`定义如下：

```cpp
auto func = std::function<void(void)>{}; 
```

返回`bool`类型，带有`int`和`std::string`作为参数的`std::function`定义如下：

```cpp
auto func = std::function<bool(int, std::string)>{}; 
```

共享相同签名（相同参数和相同返回类型）的 lambda 函数可以由相同类型的`std::function`对象持有。`std::function`也可以在运行时重新分配。

重要的是，lambda 捕获的内容不会影响其签名，因此具有捕获和不捕获的 lambda 可以分配给相同的`std::function`变量。以下代码展示了如何将不同的 lambda 分配给同一个名为`func`的`std::function`对象：

```cpp
// Create an unassigned std::function object 
auto func = std::function<void(int)>{}; 
// Assign a lambda without capture to the std::function object 
func = [](int v) { std::cout << v; }; 
func(12); // Prints 12 
// Assign a lambda with capture to the same std::function object 
auto forty_two = 42; 
func = forty_two { std::cout << (v + forty_two); }; 
func(12); // Prints 54 
```

让我们在接下来的一个类似真实世界的例子中使用`std::function`。

### 使用 std::function 实现一个简单的 Button 类

假设我们着手实现一个`Button`类。然后我们可以使用`std::function`来存储与点击按钮对应的动作，这样当我们调用`on_click()`成员函数时，相应的代码就会被执行。

我们可以这样声明`Button`类：

```cpp
class Button {
public: 
  Button(std::function<void(void)> click) : handler_{click} {} 
  auto on_click() const { handler_(); } 
private: 
  std::function<void(void)> handler_{};
}; 
```

然后我们可以使用它来创建多种具有不同动作的按钮。这些按钮可以方便地存储在容器中，因为它们都具有相同的类型：

```cpp
auto create_buttons () { 
  auto beep = Button([counter = 0]() mutable {  
    std::cout << "Beep:" << counter << "! "; 
    ++counter; 
  }); 
  auto bop = Button([] { std::cout << "Bop. "; }); 
  auto silent = Button([] {});
  return std::vector<Button>{beep, bop, silent}; 
} 
```

在列表上进行迭代，并对每个按钮调用`on_click()`将执行相应的函数：

```cpp
const auto& buttons = create_buttons();
for (const auto& b: buttons) {
  b.on_click();
}
buttons.front().on_click(); // counter has been incremented
// Output: "Beep:0! Bop. Beep:1!" 
```

前面的按钮和点击处理程序示例展示了在 lambda 与`std::function`结合使用时的一些好处；即使每个有状态的 lambda 都有其自己独特的类型，一个`std::function`类型可以包装共享相同签名（返回类型和参数）的 lambda。

顺便说一句，你可能已经注意到`on_click()`成员函数被声明为`const`。然而，它通过增加一个点击处理程序中的`counter`变量来改变成员变量`handler_`。这可能看起来违反了 const 正确性规则，因为`Button`的 const 成员函数允许调用其类成员的变异函数。之所以允许这样做，是因为成员指针在 const 上下文中允许改变其指向的值。在本章的前面，我们讨论了如何传播指针数据成员的 const 性。

### std::function 的性能考虑

与通过 lambda 表达式直接构造的函数对象相比，`std::function`有一些性能损失。本节将讨论在使用`std::function`时需要考虑的一些与性能相关的事项。

#### 阻止内联优化

在谈到 lambda 时，编译器有能力内联函数调用；也就是说，函数调用的开销被消除了。`std::function`的灵活设计使得编译器几乎不可能内联包装在`std::function`中的函数。如果非常频繁地调用包装在`std::function`中的小函数，那么阻止内联优化可能会对性能产生负面影响。

#### 捕获变量的动态分配内存

如果将`std::function`分配给带有捕获变量/引用的 lambda，那么`std::function`在大多数情况下将使用堆分配的内存来存储捕获的变量。如果捕获变量的大小低于某个阈值，一些`std::function`的实现将不分配额外的内存。

这意味着不仅由于额外的动态内存分配而产生性能损失，而且由于堆分配的内存可能增加缓存未命中的次数（在*第四章*的*数据结构*中了解更多关于缓存未命中的信息）。

#### 额外的运行时计算

调用`std::function`通常比执行 lambda 慢一点，因为涉及到更多的代码。对于小而频繁调用的`std::function`来说，这种开销可能变得很大。想象一下，我们定义了一个非常小的 lambda：

```cpp
auto lambda = [](int v) { return v * 3; }; 
```

接下来的基准测试演示了对于一个`std::vector`的 1000 万次函数调用，使用显式 lambda 类型与相应的`std::function`的`std::vector`之间的差异。我们将从使用显式 lambda 的版本开始：

```cpp
auto use_lambda() { 
  using T = decltype(lambda);
  auto fs = std::vector<T>(10'000'000, lambda);
  auto res = 1;
  // Start clock
  for (const auto& f: fs)
    res = f(res);
  // Stop clock here
  return res;
} 
```

我们只测量执行函数内部循环所需的时间。下一个版本将我们的 lambda 包装在`std::function`中，如下所示：

```cpp
auto use_std_function() { 
  using T = std::function<int(int)>;
  auto fs = std::vector<T>(10'000'000, T{lambda});
  auto res = 1;
  // Start clock
  for (const auto& f: fs)
    res = f(res);
  // Stop clock here
  return res;
} 
```

我正在使用 2018 年的 MacBook Pro 上使用 Clang 编译此代码，并打开了优化（`-O3`）。第一个版本`use_lambda()`在大约 2 毫秒内执行循环，而第二个版本`use_std_function()`则需要近 36 毫秒来执行循环。

## 通用 lambda

通用 lambda 是一个接受`auto`参数的 lambda，使得可以用任何类型调用它。它的工作原理与常规 lambda 一样，但`operator()`已被定义为成员函数模板。

只有参数是模板变量，而不是捕获的值。换句话说，以下示例中捕获的值`v`将始终是`int`类型，而不管`v0`和`v1`的类型如何：

```cpp
auto v = 3; // int
auto lambda = v {
  return v + v0*v1;
}; 
```

如果我们将上述 lambda 表达式转换为一个类，它将对应于以下内容：

```cpp
class Lambda {
public:
  Lambda(int v) : v_{v} {}
  template <typename T0, typename T1>
  auto operator()(T0 v0, T1 v1) const { 
    return v_ + v0*v1; 
  }
private:
  int v_{};
};
auto v = 3;
auto lambda = Lambda{v}; 
```

就像模板化版本一样，直到调用 lambda 表达式，编译器才会生成实际的函数。因此，如果我们像这样调用之前的 lambda：

```cpp
auto res_int = lambda(1, 2);
auto res_float = lambda(1.0f, 2.0f); 
```

编译器将生成类似于以下 lambda 表达式：

```cpp
auto lambda_int = v { return v + v0*v1; };
auto lambda_float = v { return v + v0*v1; };
auto res_int = lambda_int(1, 2);
auto res_float = lambda_float(1.0f, 2.0f); 
```

正如您可能已经发现的那样，这些版本将进一步处理，就像常规 lambda 一样。

C++20 的一个新特性是，我们可以在通用 lambda 的参数类型中使用`typename`而不仅仅是`auto`。以下通用 lambda 是相同的：

```cpp
// Using auto
auto x = [](auto v) { return v + 1; };
// Using typename
auto y = []<typename Val>(Val v) { return v + 1; }; 
```

这使得在 lambda 的主体内部命名类型或引用类型成为可能。

# 总结

在本章中，您已经学会了如何使用现代 C++特性，这些特性将在整本书中使用。自动类型推导、移动语义和 lambda 表达式是每个 C++程序员今天都需要熟悉的基本技术。

我们还花了一些时间来研究错误处理以及如何思考错误和有效状态，以及如何从运行时错误中恢复。错误处理是编程中极其重要的一部分，很容易被忽视。考虑调用方和被调用方之间的契约是使您的代码正确并避免在程序的发布版本中进行不必要的防御性检查的一种方法。

在下一章中，我们将探讨在 C++中分析和测量性能的策略。
