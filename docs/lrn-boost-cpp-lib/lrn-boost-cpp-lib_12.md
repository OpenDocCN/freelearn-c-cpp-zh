# 附录 A：C++11 语言特性模拟

在本节中，我们将回顾一些 C++ 编程中的概念，这些概念在理解本书涵盖的几个主题中具有概念上的重要性。其中许多概念是作为 C++11 的一部分相对较新地引入的。我们将研究：RAII、复制和移动语义、`auto`、基于范围的 for 循环以及 C++11 异常处理增强。我们将看看如何在预 C++11 编译器下使用 Boost 库的部分来模拟这些特性。

# RAII

C++ 程序经常处理系统资源，如内存、文件和套接字句柄、共享内存段、互斥锁等。有明确定义的原语，一些来自 C 标准库，还有更多来自本地系统编程接口，用于请求和释放这些资源。未能保证已获取资源的释放可能会对应用程序的性能和正确性造成严重问题。

C++ 对象 *在堆栈上* 的析构函数在堆栈展开时会自动调用。展开发生在由于控制达到作用域的末尾而退出作用域，或者通过执行 `return`、`goto`、`break` 或 `continue`。由于抛出异常而导致作用域退出也会发生展开。在任何情况下，都保证调用析构函数。这个保证仅限于堆栈上的 C++ 对象。它不适用于堆上的 C++ 对象，因为它们不与词法作用域相关。此外，它也不适用于前面提到的资源，如内存和文件描述符，它们是平凡旧数据类型（POD 类型）的对象，因此没有析构函数。

考虑以下使用 `new[]` 和 `delete[]` 运算符的 C++ 代码：

```cpp
char *buffer = new char[BUFSIZ];
… …
delete [] buffer;
```

程序员小心释放了分配的缓冲区。但是，如果另一个程序员随意编写代码，在调用 `new` 和 `delete` 之间的某个地方退出作用域，那么 `buffer` 就永远不会被释放，您将会泄漏内存。异常也可能在介入的代码中出现，导致相同的结果。这不仅适用于内存，还适用于任何需要手动释放的资源，比如在这种情况下的 `delete[]`。

这是我们可以利用在退出作用域时保证调用析构函数来保证资源的清理。我们可以创建一个包装类，其构造函数获取资源的所有权，其析构函数释放资源。几行代码可以解释这种通常被称为**资源获取即初始化**或**RAII**的技术。

**清单 A.1：RAII 的实际应用**

```cpp
 1 class String
 2 {
 3 public:
 4   String(const char *str = 0)
 5   {  buffer_ = dupstr(str, len_);  }
 6 
 7   ~String() { delete [] buffer_; }
 8
 9 private:
10   char *buffer_;
11   size_t len_;
12 };
13
14 // dupstr returns a copy of s, allocated dynamically.
15 //   Sets len to the length of s.
16 char *dupstr(const char *str, size_t& len) {
17   char *ret = nullptr;
18
19   if (!str) {
20     len = 0;
21     return ret;
22   }
23   len = strlen(str);
24   ret = new char[len + 1];
25   strncpy(ret, str, len + 1);
26
27   return ret;
28 }
```

`String` 类封装了一个 C 风格的字符串。我们在构造过程中传递了一个 C 风格的字符串，并且如果它不为空，它会在自由存储器上创建传递的字符串的副本。辅助函数 `dupstr` 使用 `new[]` 运算符（第 24 行）在自由存储器上为 `String` 对象分配内存。如果分配失败，`operator new[]` 抛出 `std::bad_alloc`，并且 `String` 对象永远不会存在。换句话说，资源获取必须成功才能使初始化成功。这是 RAII 的另一个关键方面。

我们在代码中使用 `String` 类，如下所示：

```cpp
 {
   String favBand("Led Zeppelin");
 ...   ...
 } // end of scope. favBand.~String() called.
```

我们创建了一个名为 `favBand` 的 `String` 实例，它在内部动态分配了一个字符缓冲区。当 `favBand` 正常或由于异常而超出范围时，它的析构函数被调用并释放这个缓冲区。您可以将这种技术应用于所有需要手动释放的资源形式，并且它永远不会让资源泄漏。`String` 类被认为拥有缓冲区资源，即它具有*独占所有权语义*。

# 复制语义

一个对象在其数据成员中保留状态信息，这些成员本身可以是 POD 类型或类类型。如果你没有为你的类定义一个复制构造函数，那么编译器会隐式为你定义一个。这个隐式定义的复制构造函数依次复制每个成员，调用类类型成员的复制构造函数，并对 POD 类型成员执行位拷贝。赋值运算符也是如此。如果你没有定义自己的赋值运算符，编译器会生成一个，并执行成员逐个赋值，调用类类型成员对象的赋值运算符，并对 POD 类型成员执行位拷贝。

以下示例说明了这一点：

**清单 A.2：隐式析构函数、复制构造函数和赋值运算符**

```cpp
 1 #include <iostream>
 2
 3 class Foo {
 4 public:
 5   Foo() {}
 6
 7   Foo(const Foo&) {
 8     std::cout << "Foo(const Foo&)\n";
 9   }
10
11   ~Foo() {
12     std::cout << "~Foo()\n";
13   }
14
15   Foo& operator=(const Foo&) {
16     std::cout << "operator=(const Foo&)\n";
17     return *this;
18   }
19 };
20
21 class Bar {
22 public:
23   Bar() {}
24
25 private:
26   Foo f;
27 };
28
29 int main() {
30   std::cout << "Creating b1\n";
31   Bar b1;
32   std::cout << "Creating b2 as a copy of b1\n";
33   Bar b2(b1);
34
35   std::cout << "Assigning b1 to b2\n";
36   b2 = b1;
37 }
```

类`Bar`包含类`Foo`的一个实例作为成员（第 25 行）。类`Foo`定义了一个析构函数（第 11 行），一个复制构造函数（第 7 行）和一个赋值运算符（第 15 行），每个函数都打印一些消息。类`Bar`没有定义任何这些特殊函数。我们创建了一个名为`b1`的`Bar`实例（第 30 行），以及`b1`的一个副本`b2`（第 33 行）。然后我们将`b1`赋值给`b2`（第 36 行）。当程序运行时，输出如下：

```cpp
Creating b1
Creating b2 as a copy of b1
Foo(const Foo&)
Assigning b1 to b2
operator=(const Foo&)
~Foo()
~Foo()
```

通过打印的消息，我们可以追踪从`Bar`的隐式生成的特殊函数调用`Foo`的特殊函数。

这对所有情况都有效，除了当你在类中封装指针或非类类型句柄到某些资源时。隐式定义的复制构造函数或赋值运算符将复制指针或句柄，但不会复制底层资源，生成一个**浅复制**的对象。这很少是需要的，这就是需要用户定义复制构造函数和赋值运算符来定义正确的复制语义的地方。如果这样的复制语义对于类没有意义，复制构造函数和赋值运算符应该被禁用。此外，您还需要使用 RAII 来管理资源的生命周期，因此需要定义一个析构函数，而不是依赖于编译器生成的析构函数。

有一个众所周知的规则叫做**三规则**，它规范了这个常见的习惯用法。它说如果你需要为一个类定义自己的析构函数，你也应该定义自己的复制构造函数和赋值运算符，或者禁用它们。我们在 A.1 清单中定义的`String`类就是这样一个候选者，我们将很快添加剩下的三个规范方法。正如我们所指出的，并不是所有的类都需要定义这些函数，只有封装资源的类才需要。事实上，建议使用这些资源的类应该与管理这些资源的类不同。因此，我们应该为每个资源创建一个包装器，使用专门的类型来管理这些资源，比如智能指针（第三章，“内存管理和异常安全性”），`boost::ptr_container`（第五章，“超出 STL 的有效数据结构”），`std::vector`等等。使用资源的类应该有包装器而不是原始资源作为成员。这样，使用资源的类就不必再担心管理资源的生命周期，隐式定义的析构函数、复制构造函数和赋值运算符对它的目的就足够了。这就被称为**零规则**。

## 不抛出交换

感谢零规则，你应该很少需要担心三规则。但是当你确实需要使用三规则时，有一些细枝末节需要注意。让我们首先了解如何在 A.1 清单中为`String`类定义一个复制操作：

**清单 A.1a：复制构造函数**

```cpp
 1 String::String(const String &str) : buffer_(0), len_(0)
 2 {
 3   buffer_ = dupstr(str.buffer_, len_);
 4 }
```

复制构造函数的实现与清单 A.1 中的构造函数没有区别。赋值运算符需要更多的注意。考虑以下示例中如何对`String`对象进行赋值：

```cpp
 1 String band1("Deep Purple");
 2 String band2("Rainbow");
 3 band1 = band2;
```

在第 3 行，我们将`band2`赋值给`band1`。作为此过程的一部分，应释放`band1`的旧状态，然后用`band2`的内部状态的副本进行覆盖。问题在于复制`band2`的内部状态可能会失败，因此在成功复制`band2`的状态之前，不应销毁`band1`的旧状态。以下是实现此目的的简洁方法：

**清单 A.1b：赋值运算符**

```cpp
 1 String& String::operator=(const String& rhs)
 2 {
 3   String tmp(rhs);   // copy the rhs in a temp variable
 4   swap(tmp);         // swap tmp's state with this' state.
 5   return *this;      // tmp goes out of scope, releases this'
 6                      // old state
 7 }
```

我们将`tmp`作为`rhs`的副本创建（第 3 行），如果此复制失败，它应该抛出异常，赋值操作将失败。被赋值对象`this`的内部状态不应更改。对`swap`的调用（第 4 行）仅在复制成功时执行（第 3 行）。对`swap`的调用交换了`this`和`tmp`对象的内部状态。因此，`this`现在包含`rhs`的副本，而`tmp`包含`this`的旧状态。在此函数结束时，`tmp`超出范围并释放了`this`的旧状态。

### 提示

通过考虑特殊情况，可以进一步优化此实现。如果被赋值对象（左侧）已经具有至少与`rhs`的内容相同大小的存储空间，那么我们可以简单地将`rhs`的内容复制到被赋值对象中，而无需额外的分配和释放。

这是`swap`成员函数的实现：

**清单 A.1c：nothrow swap**

```cpp
 1 void String::swap(String&rhs) noexcept
 2 {
 3   using std::swap;
 3   swap(buffer_, rhs.buffer_);
 4   swap(len_, rhs.len_);
 5 }
```

交换原始类型变量（整数、指针等）不应引发任何异常，这一事实我们使用 C++11 关键字`noexcept`来宣传。我们可以使用`throw()`代替`noexcept`，但异常规范在 C++11 中已被弃用，而`noexcept`比`throw()`子句更有效。这个`swap`函数完全是用交换原始数据类型来写的，保证成功并且永远不会使被赋值对象处于不一致的状态。

# 移动语义和右值引用

复制语义用于创建对象的克隆。有时很有用，但并非总是需要或有意义。考虑封装 TCP 客户端套接字的以下类。TCP 套接字是一个整数，表示 TCP 连接的一个端点，通过它可以向另一个端点发送或接收数据。TCP 套接字类可以有以下接口：

```cpp
class TCPSocket
{
public:
  TCPSocket(const std::string& host, const std::string& port);
  ~TCPSocket();

  bool is_open();
  vector<char> read(size_t to_read);
  size_t write(vector<char> payload);

private:
  int socket_fd_;

  TCPSocket(const TCPSocket&);
  TCPSocket& operator = (const TCPSocket&);
};
```

构造函数打开到指定端口上主机的连接并初始化`socket_fd_`成员变量。析构函数关闭连接。TCP 不定义一种克隆打开套接字的方法（不像具有`dup`/`dup2`的文件描述符），因此克隆`TCPSocket`也没有意义。因此，通过将复制构造函数和复制赋值运算符声明为私有来禁用复制语义。在 C++11 中，这样做的首选方法是将这些成员声明为已删除：

```cpp
TCPSocket(const TCPSocket&) = delete;
TCPSocket& operator = (const TCPSocket&) = delete;
```

虽然不可复制，但在一个函数中创建`TCPSocket`对象然后返回给调用函数是完全合理的。考虑一个创建到某个远程 TCP 服务的连接的工厂函数：

```cpp
TCPSocket connectToService()
{
  TCPSocket socket(get_service_host(),  // function gets hostname
                   get_service_port()); // function gets port
  return socket;
}
```

这样的函数将封装关于连接到哪个主机和端口的详细信息，并创建一个要返回给调用者的`TCPSocket`对象。这实际上根本不需要复制语义，而是需要移动语义，在`connectToService`函数中创建的`TCPSocket`对象的内容将被转移到调用点的另一个`TCPSocket`对象中：

```cpp
TCPSocket socket = connectToService();
```

在 C++03 中，如果不启用复制构造函数，将无法编写此代码。我们可以通过曲线救国复制构造函数来提供移动语义，但这种方法存在许多问题：

```cpp
TCPSocket::TCPSocket(TCPSocket& that) {
  socket_fd_ = that.socket_fd_;
  that.socket_fd_ = -1;
}
```

请注意，这个“复制”构造函数实际上将其参数的内容移出，这就是为什么参数是非 const 的原因。有了这个定义，我们实际上可以实现`connectToService`函数，并像之前那样使用它。但是没有什么可以阻止以下情况发生：

```cpp
 1 void performIO(TCPSocket socket)
 2 {
 3   socket.write(...);
 4   socket.read(...);
 5   // etc.
 6 }
 7
 8 TCPSocket socket = connectToService();
 9 performIO(socket);   // moves TCPSocket into performIO
10 // now socket.socket_fd_ == -1
11 performIO(socket);   // OOPs: not a valid socket
```

我们通过调用`connectToService`（第 8 行）获得了名为`socket`的`TCPSocket`实例，并将此实例传递给`performIO`（第 9 行）。但用于将`socket`按值传递给`performIO`的复制构造函数移出了其内容，当`performIO`返回时，`socket`不再封装有效的 TCP 套接字。通过将移动伪装成复制，我们创建了一个令人费解且容易出错的接口；如果您熟悉`std::auto_ptr`，您以前可能已经见过这种情况。

## 右值引用

为了更好地支持移动语义，我们必须首先回答一个问题：哪些对象可以被移动？再次考虑`TCPSocket`示例。在函数`connectToService`中，表达式`TCPSocket(get_service_host(), get_service_port())`是`TCPSocket`的*无名临时*对象，其唯一目的是传递到调用者的上下文。没有人可以在创建该语句之后引用此对象。从这样的对象中移出内容是完全合理的。但在以下代码片段中：

```cpp
TCPSocket socket = connectToService();
performIO(socket);
```

从`socket`对象中移出内容是危险的，因为在调用上下文中，对象仍然绑定到名称`socket`，并且可以在进一步的操作中使用。表达式`socket`被称为**左值表达式**——具有标识并且其地址可以通过在表达式前加上&-运算符来获取。非左值表达式被称为**右值表达式**。这些是无名表达式，其地址不能使用&-运算符在表达式上计算。例如，`TCPSocket(get_service_host(), get_service_port())`是一个右值表达式。

一般来说，从左值表达式中移动内容是危险的，但从右值表达式中移动内容是安全的。因此，以下是危险的：

```cpp
TCPSocket socket = connectToService();
performIO(socket);
```

但以下是可以的：

```cpp
performIO(connectToService());
```

请注意，表达式`connectToService()`不是左值表达式，因此符合右值表达式的条件。为了区分左值和右值表达式，C++11 引入了一种新的引用类别，称为**右值引用**，它可以引用右值表达式但不能引用左值表达式。这些引用使用双和符号的新语法声明，如下所示：

```cpp
socket&& socketref = TCPSocket(get_service_host(), 
                               get_service_port());
```

早期被简单称为*引用*的引用的另一类现在称为**左值引用**。非 const 左值引用只能引用左值表达式，而 const 左值引用也可以引用右值表达式：

```cpp
/* ill-formed */
socket& socketref = TCPSocket(get_service_host(), 
                              get_service_port());

/* well-formed */
const socket& socketref = TCPSocket(get_service_host(), 
                                    get_service_port());
```

右值引用可以是非 const 的，通常是非 const 的：

```cpp
socket&& socketref = TCPSocket(...);
socketref.read(...);
```

在上面的代码片段中，表达式`socketref`本身是一个左值表达式，因为可以使用&-运算符计算其地址。但它绑定到一个右值表达式，并且通过非 const 右值引用引用的对象可以通过它进行修改。

### 右值引用重载

我们可以根据它们是否接受左值表达式或右值表达式来创建函数的重载。特别是，我们可以重载复制构造函数以接受右值表达式。对于`TCPSocket`类，我们可以编写以下内容：

```cpp
TCPSocket(const TCPSocket&) = delete;

TCPSocket(TCPSocket&& rvref) : socket_fd_(-1)
{
  std::swap(socket_fd_, rvref.socket_fd_);
}
```

虽然左值重载是删除的复制构造函数，但右值重载被称为移动构造函数，因为它被实现为篡夺或“窃取”传递给它的右值表达式的内容。它将源的内容移动到目标，将源（`rvref`）留在某种未指定的状态中，可以安全地销毁。在这种情况下，这相当于将`rvref`的`socket_fd_`成员设置为-1。

使用移动构造函数的定义，`TCPSocket` 可以移动，但不能复制。`connectToService` 的实现将正常工作：

```cpp
TCPSocket connectToService()
{
  return TCPSocket(get_service_host(),get_service_port());
}
```

这将把临时对象移回到调用者。但是，对 `performIO` 的后续调用将是不合法的，因为 `socket` 是一个左值表达式，而 `TCPSocket` 仅为其定义了需要右值表达式的移动语义：

```cpp
TCPSocket socket = connectToService();
performIO(socket);
```

这是一个好事，因为您不能移动像 `socket` 这样的对象的内容，而您可能稍后还会使用它。可移动类型的右值表达式可以通过值传递，因此以下内容将是合法的：

```cpp
performIO(connectToService());
```

请注意，表达式 `connectToService()` 是一个右值表达式，因为它未绑定到名称，其地址也无法被获取。

类型可以既可复制又可移动。例如，我们可以为 `String` 类实现一个移动构造函数，除了它的复制构造函数：

```cpp
 1 // move-constructor
 2 String::String(String&& source) noexcept
 3       : buffer_(0), len_(0)
 4 {
 5   swap(source); // See listing A.1c
 6 }
```

nothrow `swap` 在移动语义的实现中起着核心作用。源对象和目标对象的内容被交换。因此，当源对象在调用范围内超出范围时，它释放其新内容（目标对象的旧状态）。目标对象继续存在，具有其新状态（源对象的原始状态）。移动是基于 nothrow `swap` 实现的，它只交换原始类型的指针和值，并且保证成功；因此，使用了 `noexcept` 说明。实际上，移动对象通常需要更少的工作，涉及交换指针和其他数据位，而复制通常需要可能失败的新分配。

### 移动赋值

就像我们可以通过窃取另一个对象的内容来构造对象一样，我们也可以在两者都构造之后将一个对象的内容移动到另一个对象。为此，我们可以定义一个**移动赋值运算符**，即复制赋值运算符的右值重载：

```cpp
 1 // move assignment
 2 String& String::operator=(String&& rhs) noexcept
 3 {
 4   swap(rhs);
 5   return *this;
 6 }
```

或者，我们可以定义一个**通用赋值运算符**，适用于左值和右值表达式：

```cpp
 1 // move assignment
 2 String& String::operator=(String rhs)
 3 {
 4   swap(rhs);
 5   return *this;
 6 }
```

请注意，通用赋值运算符不能与左值或右值重载共存，否则在重载解析中会存在歧义。

### xvalues

当您使用右值表达式调用函数时，如果有可用的右值重载函数，则编译器会将函数调用解析为右值重载函数。但是，如果您使用命名变量调用函数，则会将其解析为左值重载（如果有的话），否则程序将是不合法的。现在，您可能有一个命名变量，可以从中移动，因为您以后不需要使用它：

```cpp
void performIO(TCPSocket socket);

TCPSocket socket = connectToService();
// do stuff on socket
performIO(socket);  // ill-formed because socket is lvalue
```

前面的示例将无法编译，因为 `performIO` 以值传递其唯一参数，而 `socket` 是一个仅移动类型，但它不是右值表达式。通过使用 `std::move`，您可以将左值表达式转换为右值表达式，并将其传递给期望右值表达式的函数。`std::move` 函数模板在标准头文件 `utility` 中定义。

```cpp
#include <utility> // for std::moves
performIO(std::move(socket));
```

对 `std::move(socket)` 的调用给我们一个对 `socket` 的右值引用；它不会导致任何数据从 `socket` 中移出。当我们将这种右值引用类型的表达式传递给以值传递其参数的函数 `performIO` 时，在 `performIO` 函数中创建了一个新的 `TCPSocket` 对象，对应于其按值参数。它是从 `socket` 进行**移动初始化**的，也就是说，它的移动构造函数窃取了 `socket` 的内容。在调用 `performIO` 后，变量 `socket` 失去了其内容，因此不应在后续操作中使用。如果 `TCPSocket` 的移动构造函数正确实现，那么 `socket` 应该仍然可以安全地销毁。

表达式 `std::move(socket)` 共享 `socket` 的标识，但它可能在传递给函数时被移动。这种表达式称为**xvalues**，*x* 代表 *expired*。

### 提示

**xvalues**像 lvalues 一样有明确定义的标识，但可以像 rvalues 一样移动。**xvalues**绑定到函数的 rvalue 引用参数。

如果`performIO`没有按值接受其参数，而是按 rvalue-reference，那么有一件事会改变：

```cpp
void performIO(TCPSocket&& socket);
performIO(std::move(socket));
```

对`performIO(std::move(socket))`的调用仍然是良好的形式，但不会自动移出`socket`的内容。这是因为我们在这里传递了一个现有对象的引用，而当我们按值传递时，我们创建了一个从`socket`移动初始化的新对象。在这种情况下，除非`performIO`函数的实现明确地移出`socket`的内容，否则在调用`performIO`之后，它仍将在调用上下文中保持有效。

### 提示

一般来说，如果您将对象转换为 rvalue 表达式并将其传递给期望 rvalue 引用的函数，您应该假设它已经被移动，并且在调用之后不再使用它。

如果 T 类型的对象是*函数内部的本地对象*，并且 T 具有可访问的移动或复制构造函数，则可以从该函数中返回该对象的值。如果有移动构造函数，则返回值将被移动初始化，否则将被复制初始化。但是，如果对象不是函数内部的本地对象，则必须具有可访问的复制构造函数才能按值返回。此外，编译器在可能的情况下会优化掉复制和移动。

考虑`connectToService`的实现以及它的使用方式：

```cpp
 1 TCPSocket connectToService()
 2 {
 3   return TCPSocket(get_service_host(),get_service_port());
 4 }
 5
 6 TCPSocket socket = connectToService();
```

在这种情况下，编译器实际上会直接在`socket`对象的存储空间（第 3 行）中构造临时对象，而`connectToService`的返回值原本是要移动到的地方（第 6 行）。这样，它会简单地优化掉`socket`的移动初始化（第 6 行）。即使移动构造函数具有副作用，这种优化也会生效，这意味着这些副作用可能不会因此优化而产生效果。同样，编译器可以优化掉复制初始化，并直接在目标位置构造返回的对象。这被称为**返回值优化**（**RVO**），自 C++03 以来一直是所有主要编译器的标准，当时它只优化了复制。尽管在 RVO 生效时不会调用复制或移动构造函数，但它们仍然必须被定义和可访问才能使 RVO 生效。

当返回 rvalue 表达式时，RVO 适用，但是即使从函数中返回了命名的*本地*堆栈对象，编译器有时也可以优化掉复制或移动。这被称为**命名返回值优化**（**NRVO**）。

返回值优化是**复制省略**的一个特例，其中编译器优化掉 rvalue 表达式的移动或复制，直接在目标存储中构造它：

```cpp
std::string reverse(std::string input);

std::string a = "Hello";
std::string b = "World";
reverse(a + b);
```

在前面的示例中，表达式`a + b`是一个 rvalue 表达式，它生成了一个`std::string`类型的临时对象。这个对象不会被复制到函数`reverse`中，而是*省略*了复制，并且由表达式`a + b`生成的对象会直接在`reverse`的参数的存储空间中构造。

### 提示

通过值传递和返回类型 T 的对象需要为 T 定义移动或复制语义。如果有移动构造函数，则使用它，否则使用复制构造函数。在可能的情况下，编译器会优化掉复制或移动操作，并直接在调用或被调用函数的目标位置构造对象。

## 使用 Boost.Move 进行移动模拟

在本节中，我们将看看如何使用 Boost.Move 库相对容易地为自己的传统类实际上实现了大部分移动语义的后期改造。首先，考虑 C++ 11 语法中`String`类的接口：

```cpp
 1 class String
 2 {
 3 public:
 4   // Constructor
 5   String(const char *str = 0);
 6
 7   // Destructor
 8   ~String();
 9
10   // Copy constructor
11   String(const String& that);
12
13   // Copy assignment operator
14   String& operator=(const String& rhs);
15
16   // Move constructor
17   String(String&& that);
18
19   // Move assignment
20   String& operator=(String&& rhs);
21   …
22 };
```

现在让我们看看如何使用 Boost 的工具定义等效的接口：

**清单 A.2a：使用 Boost.Move 进行移动模拟**

```cpp
 1 #include <boost/move/move.hpp>
 2 #include <boost/swap.hpp>
 3
 4 class String {
 5 private:
 6   BOOST_COPYABLE_AND_MOVABLE(String);
 7
 8 public:
 9   // Constructor
10   String(const char *str = 0);
11
12   // Destructor
13   ~String();
14
15   // Copy constructor
16   String(const String& that);
17
18   // Copy assignment operator
19   String& operator=(BOOST_COPY_ASSIGN_REF(String) rhs);
20
21   // Move constructor
22   String(BOOST_RV_REF(String) that);
23
24   // Move assignment
25   String& operator=(BOOST_RV_REF(String) rhs);
26 
27   void swap(String& rhs);
28
29 private:
30   char *buffer_;
31   size_t size_;
32 };
```

关键更改如下：

+   第 6 行：宏`BOOST_COPYABLE_AND_MOVABLE(String)`定义了一些内部基础设施，以支持`String`类型的拷贝和移动语义，并区分`String`类型的左值和右值。这被声明为私有。

+   第 19 行：一个拷贝赋值运算符，它接受类型`BOOST_COPY_ASSIGN_REF(String)`。这是`String`的包装类型，可以隐式转换为`String`的左值。

+   第 22 行和 25 行：接受包装类型`BOOST_RV_REF(String)`的移动构造函数和移动赋值运算符。`String`的右值隐式转换为此类型。

+   请注意，第 16 行的拷贝构造函数不会改变。

在 C++ 03 编译器下，移动语义的模拟是在没有语言或编译器的特殊支持的情况下提供的。使用 C++ 11 编译器，宏自动使用 C++ 11 本机构造来支持移动语义。

实现与 C++ 11 版本基本相同，只是参数类型不同。

**清单 A.2b：使用 Boost Move 进行移动模拟**

```cpp
 1 // Copy constructor
 2 String::String(const String& that) : buffer_(0), len_(0)
 3 {
 4   buffer_ = dupstr(that.buffer_, len_);
 5 }
 6 
 7 // Copy assignment operator
 8 String& String::operator=(BOOST_COPY_ASSIGN_REF(String)rhs)
 9 {
10   String tmp(rhs);
11   swap(tmp);        // calls String::swap member
12   return *this;
13 }
14 
15 // Move constructor
16 String::String(BOOST_RV_REF(String) that) : buffer_(0), 
17                                             size_(0) 
18 { 
19   swap(that);      // calls String::swap member 
20 }
21 // Move assignment operator
22 String& String::operator=(BOOST_RV_REF(String)rhs)
23 {
24   swap(rhs);
25   String tmp;
26   rhs.swap(tmp);
27
28   return *this;
29 }
30 
31 void String::swap(String& that)
32 {
33   boost::swap(buffer_, that.buffer_);
34   boost::swap(size_, that.size_);
35 }
```

如果我们只想使我们的类支持移动语义而不支持拷贝语义，那么我们应该使用宏`BOOST_MOVABLE_NOT_COPYABLE`代替`BOOST_COPYABLE_AND_MOVABLE`，并且不应该定义拷贝构造函数和拷贝赋值运算符。

在拷贝/移动赋值运算符中，如果需要，我们可以通过将执行交换/复制的代码放在 if 块内来检查自赋值，如下所示：

```cpp
if (this != &rhs) {
  …
}
```

只要拷贝/移动的实现是异常安全的，这不会改变代码的正确性。但是，通过避免对自身进行赋值的进一步操作，可以提高性能。

因此，总之，以下宏帮助我们在 C++ 03 中模拟移动语义：

```cpp
#include <boost/move/move.hpp>

BOOST_COPYABLE_AND_MOVABLE(classname)
BOOST_MOVABLE_BUT_NOT_COPYABLE(classname)
BOOST_COPY_ASSIGN_REF(classname)
BOOST_RV_REF(classname)
```

除了移动构造函数和赋值运算符之外，还可以使用`BOOST_RV_REF(…)`封装类型作为其他成员方法的参数。

如果要从左值移动，自然需要将其转换为“模拟右值”的表达式。您可以使用`boost::move`来实现这一点，它对应于 C++ 11 中的`std::move`。以下是使用 Boost 移动模拟在`String`对象上调用不同的拷贝和移动操作的一些示例：

```cpp
 1 String getName();                       // return by value
 2 void setName(BOOST_RV_REF(String) str); // rvalue ref overload
 3 void setName(const String&str);        // lvalue ref overload
 4 
 5 String str1("Hello");                 
 6 String str2(str1);                      // copy ctor
 7 str2 = getName();                       // move assignment
 8 String str3(boost::move(str2));         // move ctor
 9 String str4;
10 str4 = boost::move(str1);               // move assignment
11 setName(String("Hello"));               // rvalue ref overload
12 setName(str4);                          // lvalue ref overload
13 setName(boost::move(str4));             // rvalue ref overload

```

# C++11 auto 和 Boost.Auto

考虑如何声明指向字符串向量的迭代器：

```cpp
std::vector<std::string> names;
std::vector<std::string>::iterator iter = vec.begin();

```

`iter`的声明类型很大且笨重，每次显式写出来都很麻烦。鉴于编译器知道右侧初始化表达式的类型，即`vec.begin()`，这也是多余的。从 C++11 开始，您可以使用`auto`关键字要求编译器使用初始化表达式的类型来推导已声明变量的类型。因此，前面的繁琐被以下内容替换：

```cpp
std::vector<std::string> names;
auto iter = vec.begin();

```

考虑以下语句：

```cpp
auto var = expr;
```

当使用参数`expr`调用以下函数模板时，`var`的推导类型与推导类型`T`相同：

```cpp
template <typename T>
void foo(T);

foo(expr);
```

## 类型推导规则

有一些规则需要记住。首先，如果初始化表达式是引用，则在推导类型中引用被剥离：

```cpp
int x = 5;
int& y = x;
auto z = y;  // deduced type of z is int, not int&

```

如果要声明左值引用，必须将`auto`关键字明确加上`&`，如下所示：

```cpp
int x = 5;
auto& y = x;     // deduced type of y is int&

```

如果初始化表达式不可复制，必须以这种方式使被赋值者成为引用。

第二条规则是，初始化表达式的`const`和`volatile`限定符在推导类型中被剥离，除非使用`auto`声明的变量被显式声明为引用：

```cpp
int constx = 5;
auto y = x;     // deduced type of y is int
auto& z = x;    // deduced type of z is constint

```

同样，如果要添加`const`或`volatile`限定符，必须显式这样做，如下所示：

```cpp
intconst x = 5;
auto const y = x;    // deduced type of y is constint

```

## 常见用法

`auto`关键字在许多情况下非常方便。它让您摆脱了不得不输入长模板 ID 的困扰，特别是当初始化表达式是函数调用时。以下是一些示例，以说明其优点：

```cpp
auto strptr = boost::make_shared<std::string>("Hello");
// type of strptr is boost::shared_ptr<std::string>

auto coords(boost::make_tuple(1.0, 2.0, 3.0));
// type of coords is boost::tuple<double, double, double>
```

请注意通过使用`auto`实现的类型名称的节省。另外，请注意，在创建名为`coords`的`tuple`时，我们没有使用赋值语法进行初始化。

## Boost.Auto

如果您使用的是 C++11 之前的编译器，可以使用`BOOST_AUTO`和`BOOST_AUTO_TPL`宏来模拟这种效果。因此，您可以将最后一小节写成如下形式：

```cpp
#include <boost/typeof/typeof.hpp>

BOOST_AUTO(strptr, boost::make_shared<std::string>("Hello"));
// type of strptr is boost::shared_ptr<std::string>

BOOST_AUTO(coords, boost::make_tuple(1.0, 2.0, 3.0));
// type of coords is boost::tuple<double, double, double>
```

请注意需要包含的头文件`boost/typeof/typeof.hpp`以使用该宏。

如果要声明引用类型，可以在变量前加上引导符号（&）。同样，要为变量添加`const`或`volatile`限定符，应在变量名之前添加`const`或`volatile`限定符。以下是一个示例：

```cpp
BOOST_AUTO(const& strptr, boost::make_shared<std::string>("Hello"));
// type of strptr is boost::shared_ptr<std::string>
```

# 基于范围的 for 循环

基于范围的 for 循环是 C++11 中引入的另一个语法便利。基于范围的 for 循环允许您遍历值的序列，如数组、容器、迭代器范围等，而无需显式指定边界条件。它通过消除了需要指定边界条件来使迭代更不容易出错。

基于范围的 for 循环的一般语法是：

```cpp
for (range-declaration : sequence-expression) {
 statements;
}
```

**序列表达式**标识要遍历的值序列，如数组或容器。**范围声明**标识一个变量，该变量将在循环的连续迭代中代表序列中的每个元素。基于范围的 for 循环自动识别数组、大括号包围的表达式序列和具有返回前向迭代器的`begin`和`end`成员函数的容器。要遍历数组中的所有元素，可以这样写：

```cpp
T arr[N];
...
for (const auto& elem : arr) {
  // do something on each elem
}
```

您还可以遍历大括号括起来的表达式序列：

```cpp
for (const auto& elem: {"Aragorn", "Gandalf", "Frodo Baggins"}) {
  // do something on each elem
}
```

遍历通过`begin`和`end`成员函数公开前向迭代器的容器中的元素并没有太大不同：

```cpp
std::vector<T> vector;
...
for (const auto& elem: vector) {
  // do something on each elem
}
```

范围表达式使用`auto`声明了一个名为`elem`的循环变量来推断其类型。基于范围的 for 循环中使用`auto`的这种方式是惯用的和常见的。要遍历封装在其他类型对象中的序列，基于范围的 for 循环要求两个命名空间级别的方法`begin`和`end`可用，并且可以通过*参数相关查找*（见第二章，*Boost 实用工具的第一次接触*）来解析。基于范围的 for 循环非常适合遍历在遍历期间长度保持不变的序列。

## Boost.Foreach

您可以使用`BOOST_FOREACH`宏来模拟 C++11 基于范围的 for 循环的基本用法：

```cpp
#include <boost/foreach.hpp>

std::vector<std::string> names;
...
BOOST_FOREACH(std::string& name, names) {
  // process each elem
}
```

在前面的示例中，我们使用`BOOST_FOREACH`宏来遍历名为`names`的字符串向量的元素，使用名为`name`的`string`类型的循环变量。使用`BOOST_FOREACH`，您可以遍历数组、具有返回前向迭代器的成员函数`begin`和`end`的容器、迭代器对和以空字符结尾的字符数组。请注意，C++11 基于范围的 for 循环不容易支持最后两种类型的序列。另一方面，使用`BOOST_FOREACH`，您无法使用`auto`关键字推断循环变量的类型。

# C++11 异常处理改进

C++11 引入了捕获和存储异常的能力，可以在稍后传递并重新抛出。这对于在线程之间传播异常特别有用。

## 存储和重新抛出异常

为了存储异常，使用类型`std::exception_ptr`。`std::exception_ptr`是一种具有共享所有权语义的智能指针类型，类似于`std::shared_ptr`（参见第三章，“内存管理和异常安全性”）。`std::exception_ptr`的实例是可复制和可移动的，并且可以传递给其他函数，可能跨线程。默认构造的`std::exception_ptr`是一个空对象，不指向任何异常。复制`std::exception_ptr`对象会创建两个管理相同底层异常对象的实例。只要包含它的最后一个`exception_ptr`实例存在，底层异常对象就会继续存在。

函数`std::current_exception`在 catch 块内调用时，返回执行该 catch 块的活动异常，包装在`std::exception_ptr`的实例中。在 catch 块外调用时，返回一个空的`std::exception_ptr`实例。

函数`std::rethrow_exception`接收一个`std::exception_ptr`的实例（不能为 null），并抛出`std::exception_ptr`实例中包含的异常。

**清单 A.3：使用 std::exception_ptr**

```cpp
 1 #include <stdexcept>
 2 #include <iostream>
 3 #include <string>
 4 #include <vector>
 5
 6 void do_work()
 7 {
 8   throw std::runtime_error("Exception in do_work");
 9 }
10
11 std::vector<std::exception_ptr> exceptions;
12
13 void do_more_work()
14 {
15   std::exception_ptr eptr;
16
17   try {
18     do_work();
19   } catch (...) {
20     eptr = std::current_exception();
21   }
22
23   std::exception_ptr eptr2(eptr);
24   exceptions.push_back(eptr);
25   exceptions.push_back(eptr2);
26 }
27
28 int main()
29 {
30   do_more_work();
31
32   for (auto& eptr: exceptions) try {
33     std::rethrow_exception(eptr);
34   } catch (std::exception& e) {
35     std::cout << e.what() << '\n';
36   }
37 }
```

运行上述示例会打印以下内容：

```cpp
Exception in do_work
Exception in do_work
```

`main`函数调用`do_more_work`（第 30 行），然后调用`do_work`（第 18 行），后者只是抛出一个`runtime_error`异常（第 8 行），该异常最终到达`do_more_work`（第 19 行）中的 catch 块。我们在`do_more_work`（第 15 行）中声明了一个类型为`std::exception_ptr`的对象`eptr`，并在 catch 块内调用`std::current_exception`，并将结果赋给`eptr`。稍后，我们创建了`eptr`的副本（第 23 行），并将两个实例推入全局`exception_ptr`向量（第 24-25 行）。

在`main`函数中，我们遍历全局向量中的`exception_ptr`实例，使用`std::rethrow_exception`（第 33 行）抛出每个异常，并捕获并打印其消息。请注意，在此过程中，我们打印相同异常的消息两次，因为我们有两个包含相同异常的`exception_ptr`实例。

## 使用 Boost 存储和重新抛出异常

在 C++11 之前的环境中，可以使用`boost::exception_ptr`类型来存储异常，并使用`boost::rethrow_exception`来抛出存储在`boost::exception_ptr`中的异常。还有`boost::current_exception`函数，其工作方式类似于`std::current_exception`。但是在没有底层语言支持的情况下，它需要程序员的帮助才能运行。

为了使`boost::current_exception`返回当前活动的异常，包装在`boost::exception_ptr`中，我们必须修改异常，然后才能抛出它，以便使用这种机制进行处理。为此，我们在要抛出的异常上调用`boost::enable_current_exception`。以下代码片段说明了这一点：

**清单 A.4：使用 boost::exception_ptr**

```cpp
 1 #include <boost/exception_ptr.hpp>
 2 #include <iostream>
 3
 4 void do_work()
 5 {
 6   throw boost::enable_current_exception(
 7             std::runtime_error("Exception in do_work"));
 8 }
 9
10 void do_more_work()
11 {
12   boost::exception_ptr eptr;
13 
14   try {
15     do_work();
16   } catch (...) {
17     eptr = boost::current_exception();
18   }
19
20   boost::rethrow_exception(eptr);
21 }
22
23 int main() {
24   try {
25     do_more_work();
26   } catch (std::exception& e) {
27     std::cout << e.what() << '\n';
28   }
29 }
```

# 自测问题

1.  三大法则规定，如果为类定义自己的析构函数，则还应定义：

a. 您自己的复制构造函数

b. 您自己的赋值运算符

c. 两者都是

d. 两者中的任意一个

1.  假设类`String`既有复制构造函数又有移动构造函数，以下哪个不会调用移动构造函数：

a. `String s1(getName());`

b. `String s2(s1);`

c. `String s2(std::move(s1));`

d. `String s3("Hello");`

1.  `std::move`函数的目的是：

a. 移动其参数的内容

b. 从右值引用创建左值引用

c. 从左值表达式创建 xvalue

d. 交换其参数的内容与另一个对象

1.  以下哪种情况适用于返回值优化？：

a. `return std::string("Hello");`

b. `string reverse(string);string a, b;reverse(a + b);`

c. `std::string s("Hello");return s;`

d. `std::string a, b;return a + b.`

# 参考资料

+   *《Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 and C++14*，*Scott Meyers*，*O'Reilly Media*》

+   《C++之旅》，Bjarne Stroustrup，Addison Wesley Professional

+   《C++程序设计语言（第 4 版）》，Bjarne Stroustrup，Addison Wesley Professional
