

# 第四章：使用析构函数

我们对 C++内存管理的更好和更深入的理解之旅现在进入了干净代码和当代实践的领域。在前面的章节中，我们探讨了内存表示的基本概念（什么是对象、引用、指针等等），如果我们以不适当的方式偏离良好的编程实践，会面临哪些陷阱，以及我们如何以受控和有纪律的方式欺骗类型系统，所有这些都将有助于本书的其余部分。现在，我们将讨论我们语言中资源管理的根本方面；内存作为一种特殊的资源，本章中找到的思想和技术将帮助我们编写干净和健壮的代码，包括执行内存管理任务的代码。

C++是一种支持（包括其他范例）面向对象编程的编程语言，但使用实际的对象。这听起来像是一种玩笑，但实际上这是一个正确的陈述：许多语言只提供对对象的间接访问（通过指针或引用），这意味着在这些语言中，赋值的语义通常是共享所引用的对象（*目标*）。当然，这也有其优点：例如，复制一个引用通常不会失败，而复制一个对象可能会失败，如果复制构造函数或复制赋值（根据情况而定）抛出异常。

在 C++中，默认情况下，程序使用对象、复制对象、赋值给对象等，间接访问是可选的，需要为指针和引用提供额外的语法。这要求 C++程序员考虑对象的生命周期，复制对象意味着什么，从对象移动意味着什么……这些话题可能很深，取决于涉及的类型。

注意

参见*第一章*了解更多关于对象和对象生命周期的信息，包括构造函数和析构函数的作用。

即使在源代码中实际使用对象需要调整编程时的思维方式，但它也提供了一个显著的优势：当自动对象达到它们声明的范围结束时（当它们达到该范围的闭合花括号时）以及当一个对象被销毁时，会调用一个特殊函数，即类型的`}`，闭合花括号。

在本章中，我们将探讨析构函数的作用，它们不应该做什么，何时应该编写（以及何时我们应该坚持编译器默认的行为），以及我们的代码如何有效地使用析构函数来管理资源，一般而言……以及更具体地是内存。然后，我们将快速查看一些标准库中的关键类型，这些类型利用析构函数为我们带来便利。

更详细地说，在他的章节中，我们将：

+   提供一个概述，说明如何在 C++中安全地管理资源；

+   仔细研究 RAII 习语，这是一种众所周知的惯用实践，它使用对象的生存期来确保该对象管理的资源得到适当释放；

+   检查与自动化资源管理相关的一些陷阱；

+   快速概述标准库提供的某些自动化资源管理工具。

到本章结束时，我们将了解与 C++ 资源管理相关的一些最常见思想和实践。这将使我们能够在本书的剩余部分构建更强大的抽象。

# 技术要求

您可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter4`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter4)。

# 关于析构函数：简要回顾

本章旨在讨论使用析构函数来管理资源，特别是内存，但由于我们之前已经讨论过析构函数（在第第一章中），我们将快速回顾一下这个强大想法背后的基本概念：

+   当一个对象到达其生命周期的末尾时，会调用一个特殊的成员函数，称为析构函数。对于某些类 `X`，该成员函数的名称为 `X::~X()`。这个函数是类型 `X` 在结束其生命周期之前执行一些“最后时刻”行动的机会。正如我们将在本章中讨论的，析构函数的一种惯用用法是释放正在销毁的对象所持有的资源；

+   在类层次结构中，当一个对象到达其生命周期的末尾时，发生的情况是（a）调用该对象的析构函数，然后（b）按照声明顺序调用每个非 `static` 数据成员的析构函数，最后（c）按照声明顺序调用每个基类子对象（其“父类”，非正式地）的析构函数；

+   当通过在指针上应用 `operator delete` 来显式销毁对象时，涉及的过程是先销毁指针指向的对象，然后释放对象所在内存块的分配。不出所料，这里有一些注意事项，我们将在第七章中看到；

+   在某些情况下，特别是当某个类 `X` 至少公开一个 `virtual` 成员函数时，这表明 `X*` 实际上可能指向一个从 `X` 直接或间接派生的类 `Y` 的对象。为了确保调用 `Y` 的析构函数而不是 `X` 的析构函数，通常也将 `X::~X()` 标记为 `virtual`。如果不这样做，可能会不调用正确的析构函数，从而导致资源泄露。

以一个小例子为例，考虑以下内容：

```cpp
#include <iostream>
struct Base {
    ~Base() { std::cout << "~Base()\n"; }
};
struct DerivedA : Base {
    ~DerivedA() { std::cout << "~DerivedA()\n"; }
};
struct VirtBase {
    virtual ~VirtBase() {
       std::cout << "~VirtBase()\n";
    }
};
struct DerivedB : VirtBase {
    ~DerivedB() {
       std::cout << "~DerivedB()\n";
    }
};
int main() {
   {
      Base base;
   }
   {
      DerivedA derivedA;
   }
   std::cout << "----\n";
   Base *pBase = new DerivedA;
   delete pBase; // bad
   VirtBase *pVirtBase = new DerivedB;
   delete pVirtBase; // Ok
}
```

如果您运行这段代码，您将看到为 `base` 调用一个析构函数，为 `derivedA` 调用两个析构函数：派生类的析构函数后跟基类的析构函数。这是预期的，并且这段代码是正确的。

有问题的案例是`pBase`，一个指向`Base*`类型的指针，它指向一个从`Base`派生出来的类的对象，因为`Base`的析构函数不是`virtual`，这表明尝试通过基类指针删除派生对象可能是意图的违规：`delete pBase`只调用`Base::~Base()`，永远不会调用`DerivedA::~DerivedA()`。通过`pVirtBase`这个问题可以避免，因为`VirtBase::~VirtBase()`是`virtual`。

当然，在 C++中，我们有选择，因为总会有一些令人惊讶的使用场景出现，我们将在*第七章*中看到其中一个，我们将删除一个指向派生类的指针，而无需通过`virtual`析构函数进行中介，这是出于（如果专门化）的良好（如果专门化）原因。

注意，`virtual`成员函数是有用的，但它们也有成本：一个典型的实现将为每个类型创建一个包含至少一个`virtual`成员函数的函数指针表，并将该表的指针存储在每个这样的对象中，这使得对象稍微大一些。因此，当你期望从一个基类的指针使用派生类的指针时，尤其是在你期望通过基类指针调用析构函数时，应该使用`virtual`析构函数。

话虽如此，让我们来探讨一下这一切与资源管理之间的关系。

# 资源管理

假设你正在编写一个函数，该函数打开一个文件，从中读取数据，然后关闭它。你在一个过程式平台上进行开发（就像大多数操作系统 API 一样），该平台提供了一组执行这些任务的函数。请注意，在这个例子中，所有的“操作系统”函数都是故意虚构的，但与现实世界的对应物相似。在这个 API 中，对我们来说有趣的函数是：

```cpp
// opens the file called "name", returns a pointer
// to a file descriptor for that file (nullptr on failure)
FILE *open_file(const char *name);
// returns the number of bytes read from the file into
// buf. Preconditions: file is non-null and valid, buf
// points to a buffer of at least capacity bytes, and
// capacity >= 0
int read_from(FILE *file, char *buf, int capacity);
// closes file. Precondition: file is non-null and valid,
void close_file(FILE *file);
```

假设你的代码需要处理从文件中读取的数据，但这种处理可能会抛出一个异常。这里异常的原因并不重要：可能是数据损坏、内存分配失败、调用某个会抛出异常的辅助函数，等等。关键点是，函数可能会抛出异常的风险。

如果我们尝试为该函数天真地编写代码，它可能看起来像这样：

```cpp
void f(const char *name) {
   FILE *file = open_file(name);
   if(!file) return false; // failure
   vector<char> v;
   char buf[N]; // N is a positive integral constant
   for(int n = read_from(file, buf, N); n != 0;
       n = read_from(file, buf, N))
      v.insert(end(v), buf + 0, buf + n);
   process(v); // our processing function
   close_file(file);
}
```

那段代码是可行的，在没有异常的情况下，基本上能完成我们想要的功能。现在，假设`process(v)`抛出了一个异常…会发生什么？

在这种情况下，函数`f()`退出，未能满足其后置条件。对`process(v)`的调用从未结束…并且`close_file(file);`也从未被调用。我们有一个泄漏。不一定是*内存*泄漏，但确实是泄漏，因为`file`从未被关闭，因为从`process()`抛出的异常在调用代码`f()`中没有被捕获，这将结束`f()`并让异常流经`f()`的调用者（等等，直到被捕获或程序崩溃，哪个先到来）。

有一些方法可以绕过这种情况。一种方法是“手动”进行，并在可能抛出异常的代码周围添加一个`try` … `catch`块：

```cpp
void f(const char *name) {
   FILE *file = open_file(name);
   if(!file) return; // failure
   vector<char> v;
   char buf[N]; // N is a positive integral constant
   try {
      for(int n = read_from(file, buf, N); n != 0;
          n = read_from(file, buf, N))
         v.insert(end(v), buf + 0, buf + n);
      process(v); // our processing function
      close_file(file);
   } catch(...) { // catch anything
      close_file(file);
      throw; // re-throw what we caught
   }
}
```

我同意这有点“笨拙”，有两个`close_file(file)`的调用，一个在`try`块的末尾，以在正常情况下关闭文件，另一个在`catch`块的末尾，以避免文件资源的泄露。

手动方法可以使其工作，但这是一种脆弱的解决问题的方法：在 C++中，任何既不是`noexcept`也不是`noexcept(true)`的函数都可能抛出异常；这意味着在实践中，几乎任何表达式都可能抛出异常。

捕获任何东西

在 C++中，与某些其他语言中可以看到的相比，没有为所有异常类型指定一个单一的基类。确实，`throw 3;`是完全合法的 C++代码。除此之外，C++拥有极其强大的泛型编程机制，这使得泛型代码在我们的语言中很普遍。因此，我们经常发现自己调用可能会抛出异常但无法真正知道会抛出什么的函数。要知道`catch(...)`会捕获任何用于表示异常的 C++对象：你不知道你捕获了什么，但你确实捕获了它。

在这种情况下，我们通常会想要拦截异常，可能为了做一些清理工作，然后让那个异常保持不变地继续其路径，以便让客户端代码按需处理它。清理部分是因为我们希望我们的函数能够成为`catch(...)`块，简单地使用`throw;`，这被称为“重新抛出”。

## 异常处理…还是不处理？

这引出了另一个问题：在一个像`f()`这样的函数中，我们只旨在消费数据并为我们自己的目的处理它，我们真的应该寻求处理异常吗？想想看：抛出异常的要求与处理异常的要求显著不同。

的确，我们从函数中抛出异常是为了表示我们的函数无法满足其后置条件（它无法完成其预期要做的任务）：可能是内存不足，可能是要读取的文件不存在，可能是执行你要求的那个积分除法会导致除以零，从而摧毁宇宙（我们不想发生这种情况），可能是我们调用的某个函数无法以我们没有预见或不想处理的方式满足其自己的后置条件……函数失败有很多原因。许多情况下，函数可能会发现自己处于进一步执行会导致严重问题的位置，在某些情况下（构造函数和重载运算符就是例子），异常确实是向客户端代码发出问题的唯一合理方式。

处理异常本身是一种较为罕见的情况：抛出异常需要识别问题，但处理异常则需要理解上下文。确实，在交互式控制台应用程序中针对异常采取的行动与在人们跳舞时针对音频应用程序采取的行动不同，或者与面对核反应堆熔毁时所需的行动也不同。

大多数函数在某种程度上需要异常安全性（这一点有多种形式），而不仅仅是处理问题。在我们的例子中，困难源于在异常发生时手动关闭 `file`。避免这种手动资源处理的最简单方法就是自动化它，而函数结束时发生的事情，无论该函数是否正常完成（到达函数的结束括号，遇到 `return` 语句，看到异常“飞过”），最好用析构函数来模拟。这种做法已经深深植根于 C++ 程序员的实践中，以至于被认为是惯用法，并被赋予了一个名称：*RAII 习语*。

# RAII 习语

C++ 程序员倾向于使用析构函数来自动释放资源，这确实可以称得上是我们语言中的惯用编程技术，以至于我们给它起了一个名字。可能不是最好的名字，但无论如何是一个众所周知的名字：**RAII**，代表**资源获取即初始化**（有些人也建议**责任获取即初始化**，这也适用，并且有相似的含义）。一般想法是，对象倾向于在构造时间（或之后）获取资源，但（更重要的是！）释放对象持有的资源通常应该在对象生命周期的末尾完成。因此，RAII 更多地与析构函数有关，而不是与构造函数有关，但正如我所说的，我们往往在名称和缩写上做得不好。

回顾本章早期“管理资源”部分中的文件读取和处理示例，我们可以构建一个 RAII 资源处理器，以便无论函数如何结束都能方便地关闭文件：

```cpp
class FileCloser { // perfectible, as we will see
   FILE * file;
public:
   FileCloser(FILE *file) : file{ file } {
   }
   ~FileCloser() {
      close_file(file);
}
};
void f(const char *name) {
   FILE *file = open_file(name);
   if(!file) return; // failure
   FileCloser fc{ file }; // <-- fc manages file now
   vector<char> v;
   char buf[N]; // N is a positive integral constant
   for(int n = read_from(file, buf, N); n != 0;
       n = read_from(file, buf, N))
      v.insert(end(v), buf + 0, buf + n);
   process(v); // our processing function
} FileCloser does will vary with our perception of its role: does this class just manage the closing of the file or does it actually represent the file with all of its services? I went for the former in this case but both options are reasonable: it all depends on the semantics you are seeking to implement. The key point is that by using a FileCloser object, we are relieving client code of a responsibility, instead delegating the responsibility of closing a file to an object that automates this task, simplifying our own code and reducing the risks of inadvertently leaving it open.
This `FileCloser` object is very specific to our task. We could generalize it in many ways, for example through a generic object that performs a user-supplied set of actions when destroyed:

```

template <class F> class scoped_finalizer { // 简化版

F f;

public:

scoped_finalizer(F f) : f{ f } {

}

~scoped_finalizer() {

f();

}

};

void f(const char *name) {

FILE *file = open_file(name);

if(!file) return; // 失败

auto sf = scoped_finalizer{ [&file] {

close_file(file);

} }; // <-- 文件现在由 sf 管理

vector<char> v;

char buf[N]; // N 是一个正整数常量

for(int n = read_from(file, buf, N); n != 0;

n = read_from(file, buf, N))

v.insert(end(v), buf + 0, buf + n);

process(v); // 我们的处理函数

} 使用代码块，Java 有 try-with 语句，Go 有 defer 关键字等，但在 C++中，使用作用域来自动化与资源管理相关的操作的可能性直接来自类型系统，使得对象而不是用户代码成为习惯性地管理资源的一方。

RAII 和 C++的特殊成员函数

*第一章*描述了六个特殊成员函数（默认构造函数、析构函数、复制构造函数、复制赋值运算符、移动构造函数和移动赋值运算符）。当一个类实现这些函数时，通常意味着该类负责某些资源。如*第一章*中所述，当一个类没有明确管理资源时，我们通常可以将这些函数留给编译器，并且结果的行为通常会导致更简单、更高效的代码。

现在考虑一下，RAII 习语主要关于资源管理，因为我们把对象的销毁时刻与释放之前获取的资源的行为联系起来。许多 RAII 对象（包括前面示例中的`FileCloser`和`scoped_finalizer`类）可以说对它们提供的资源负责，这意味着复制这些对象可能会引入错误（谁将负责资源，原始对象还是副本？）。因此，除非你有充分的理由明确实现它们，否则请考虑删除你的 RAII 类型的复制操作：

```cpp
template <class F> class scoped_finalizer {
   F f;
public:
   scoped_finalizer(const scoped_finalizer&) = delete;
   scoped_finalizer& operator=
      (const scoped_finalizer&) = delete;
   scoped_finalizer(F f) : f{ f } {
   }
   ~scoped_finalizer() {
      f();
   }
};
```

就像大多数习语一样，RAII 是一种普遍接受的优秀编程实践，但它并非万能良药，析构函数的使用也是如此。我们将探讨与析构函数相关的风险，以及如何避免陷入这样的困境。

一些陷阱

析构函数很棒。它们使我们能够自动化任务，简化代码，并在一般情况下使代码更安全。尽管如此，还有一些注意事项，使用析构函数的一些方面需要特别注意。

析构函数不应该抛出异常

本节的标题简单明了：析构函数不应该抛出异常。它们*可以*抛出异常，但这样做是个坏主意。

这可能一开始看起来有些令人惊讶。毕竟，构造函数可以（并且确实！）抛出异常。当构造函数抛出异常时，这意味着构造函数无法满足其后置条件：正在构建的对象没有被构建（构造函数没有完成！）因此，该对象不存在。这是一个简单、有效的模型。

如果析构函数抛出异常……嗯，这可能是你程序的终结。确实，析构函数是隐式`noexcept`的，这意味着从析构函数中抛出异常将调用`std::terminate()`，这将导致你的程序结束。

好吧，你可能想，如果我明确地将我的析构函数标记为`noexcept(false)`，从而覆盖默认行为呢？好吧，这可以工作，但要注意，如果析构函数在栈回溯期间抛出异常，比如当异常已经在飞行中时，这仍然会调用`std::terminate()`，因为你已经做错了事，违反了规则，编译器可以优化掉你的一些代码。例如，在以下程序中，即使此时`Evil`的析构函数尚未被调用，也有可能既不会打印`"A\n"`也不会打印`"B\n"`：

```cpp
#include <iostream>
class Darn {};
void f() { throw 3; }
struct Evil {
   Evil() { std::cout << "Evil::Evil()\n"; }
   ~Evil() noexcept(false) {
      std::cout << "Evil::~Evil()\n";
      throw Darn {};
   }
};
void g() {
    std::cout << "A\n";
    Evil e;
    std::cout << "B\n";
    f();
    std::cout << "C\n";
}
int main() {
   try {
      g();
   } catch(int) {
      std::cerr << "catch(int)\n";
   } catch(Darn) {
      std::cerr << "darn...\n";
   }
}
```

从这段代码可能得到的一个结果是，程序将什么也不显示，并且会输出一些类似“抛出`Darn`导致调用`std::terminate()`”的信息。为什么一些代码（特别是我们试图输出的消息）会被编译器明显地移除呢？答案是，未捕获的异常会进入实现定义的行为，而在这个例子中，抛出`Darn`无法被捕获（因为它在栈回溯期间直接调用`std::terminate()`），这使得编译器可以显著优化我们的代码。

总结一下：除非你真的知道自己在做什么，否则不要从析构函数中抛出异常，控制它将被调用的上下文，并且与其他人讨论以确保即使所有证据都指向相反的方向，这也是合理的。即便如此，寻找替代方案可能更好。

了解你的析构顺序

这个小节的标题可能看起来像是一个有趣的告诫。为什么了解我们的对象将被销毁的顺序很重要呢？毕竟，基本规则很简单：对象的构造和析构是对称的，因此对象将以构造的相反顺序被销毁…对吗？

好吧，这就是局部、自动对象的情况。如果你编写以下代码：

```cpp
void f() {
   A a; // a's ctor
   B b; // b's ctor
   {
      C c; // c's ctor
   } // c's dtor
   D d; // d's ctor
} // d's dtor, b's dtor, a's dtor (in that order)
```

…然后构造和析构的顺序将如注释中所述：作用域内的自动对象将以构造的相反顺序被销毁，嵌套的作用域会按预期行为工作。

如果混合使用非自动对象，情况会变得更加复杂。C++ 允许在函数内声明`static`对象：这些对象在函数第一次被调用时构造，并从那时起一直存活到程序执行结束。C++ 允许声明全局变量（这里有很多细微差别，例如`static`或`extern`链接说明），C++ 允许在类中有`static`数据成员：这些本质上也是全局变量。我不会在这里提到`thread_local`变量，因为它们超出了这本书的范围，但如果你使用它们，要知道它们可以被延迟初始化，这增加了整体图景的复杂性。全局对象将以构造的相反顺序被销毁，但这个构造顺序并不总是可以从我们的角度来看轻易预测。

考虑以下示例，它使用`Verbose`对象，这些对象会告诉我们它们的构造时刻以及销毁时刻：

```cpp
#include <iostream>
#include <format>
struct Verbose {
   int n;
   Verbose(int n) : n{ n } {
      std::cout << std::format(«Verbose({})\n», n);
   }
   ~Verbose(){
      std::cout << std::format(«~Verbose({})\n», n);
   }
};
class X {
   static inline Verbose v0 { 0 };
   Verbose v1{ 1 };
};
Verbose v2{ 2 };
static void f() {
    static Verbose v3 { 3 };
    Verbose v4{ 4 };
}
static void g() { // note : never called
    static Verbose v5 { 5 };
}
int main() {
   Verbose v6{ 6 };
   {
      Verbose v7{ 7 };
      f();
      X x;
   }
   f();
   X x;
}
```

仔细思考这个示例，并试图弄清楚将会显示什么。我们有一个全局对象，一个类中的`static`和`inline`数据成员，两个局部于函数的`static`对象，以及一些局部自动对象。

那么，如果我们运行这个程序，将会显示什么？如果你尝试运行它，你应该会看到：

```cpp
Verbose(0)
Verbose(2)
Verbose(6)
Verbose(7)
Verbose(3)
Verbose(4)
~Verbose(4)
Verbose(1)
~Verbose(1)
~Verbose(7)
Verbose(4)
~Verbose(4)
Verbose(1)
~Verbose(1)
~Verbose(6)
~Verbose(3)
~Verbose(2)
~Verbose(0)
```

首先被构造（也是最后被销毁）的是`v0`，即`static`的`inline`数据成员。它也恰好是我们的第一个全局对象，接着是`v2`（我们的第二个全局对象）。然后我们进入`main()`并创建`v6`，它将在`main()`结束时被销毁。

现在，如果你查看该程序的输出，你会看到在这一点上对称性被打破了，因为`v6`构造之后，我们构造了`v7`（在一个内部更窄的作用域中；`v7`将在之后很快被销毁），然后第一次调用`f()`，这构造了`v3`，但`v3`是一个全局对象，因此它将在`v6`和`v7`之后被销毁。

整个过程是机械的和确定的，但理解它需要一些思考和解析。如果我们使用对象的析构函数来释放资源，如果未能理解发生了什么以及何时发生，可能会导致我们的代码尝试使用已经释放的资源。

对于一个涉及自动和手动资源管理的具体示例，让我们看看 C++标准一无所知的东西：动态链接库（`.dll`文件）。这里我不会深入细节，所以知道如果你在 Linux 机器上（使用共享对象，`.so`文件）或在 Mac 上（`.dylib`文件），总体思路是相同的，但函数名称将不同。

我们程序将（a）加载一个动态链接库，（b）获取一个函数的地址，（c）调用这个函数，然后（d）卸载库。假设库的名称为`Lib`，我们想要调用的函数名为`factory`，它返回一个`X*`，我们想要调用其成员函数`f()`：

```cpp
#include "Lib.h"
#include <Windows.h> // LoadLibrary, GetProcAddress
int main() {
   using namespace std;
   HMODULE hMod = LoadLibrary(L"Lib.dll");
   // suppose the signature of factory is in Lib.h
   auto factory_ptr = reinterpret_cast<
      decltype(&factory)
   >(GetProcAddress(hMod, "factory"));
   X *p = factory_ptr();
   p->f();
   delete p;
   FreeLibrary(hMod);
}
```

你可能已经注意到了其中的手动内存管理：我们通过`factory_ptr`调用`factory()`来获取一个资源（一个指向至少是`X`的`X*`），然后我们使用（在`pointee`上调用`f()`）并手动释放该*指针*。

到目前为止，你可能正在告诉自己手动资源管理并不是一个好主意（这里：如果`p->f()`抛出异常，资源会发生什么？），所以你查阅了标准，发现`std::unique_ptr`类型的对象将负责*指针*，并在其析构函数被调用时销毁它。这很美，不是吗？事实上，它可能确实如此，但考虑以下摘录，重新编写以使用`std::unique_ptr`并自动化资源管理过程：

```cpp
#include "Lib.h"
#include <memory> // std::unique_ptr
#include <Windows.h> // LoadLibrary, GetProcAddress
int main() {
   using namespace std;
   HMODULE hMod = LoadLibrary(L"Lib.dll");
   // suppose the signature of factory is in Lib.h
   auto factory_ptr = reinterpret_cast<
      decltype(&factory)
   >(GetProcAddress(hMod, "factory"));
   std::unique_ptr<X> p { factory_ptr() };
   p->f();
   // delete p; // not needed anymore
   FreeLibrary(hMod);
} p is now an RAII object responsible for the destruction of the *pointee*. Being destroyed at the closing brace of our main() function, we know that the destructor of the *pointee* will be called even if p->f() throws, so we consider ourselves more exception-safe than before…
… except that this code crashes on that closing brace! If you investigate the source of the crash, you will probably end up realizing that the crash happens at the point where the destructor of `p` calls operator `delete` on the `X*` it has stored internally. Reading further, you will notice that the reason why this crash happens is that the library the object came from has been freed (call to `FreeLibrary()`) before the destructor ran.
Does that mean we cannot use an automated memory management tool here? Of course not, but we need to be more careful with the way in which we put object lifetime to contribution. In this example, we want to make sure that `p` is destroyed before the call to `FreeLibrary()` happens; this can be achieved through the simple introduction of a scope in our function:

```

#include "Lib.h"

#include <memory> // std::unique_ptr

#include <Windows.h> // LoadLibrary, GetProcAddress

int main() {

using namespace std;

HMODULE hMod = LoadLibrary(L"Lib.dll");

// 假设 factory 的签名在 Lib.h 中

auto factory_ptr = reinterpret_cast<

decltype(&factory)

>(GetProcAddress(hMod, "factory"));

{

std::unique_ptr<X> p { factory_ptr() };

p->f();

} // p 被销毁在这里

FreeLibrary(hMod);

}

```cpp

 In this specific example, we could find a simple solution; in other cases we might have to move some declarations around to make sure the scopes in which our objects find themselves don’t alter the intended semantics of our function. Understanding the order in which objects are destroyed is essential to properly using this precious resource management facility that is the destructor.
Standard resource management automation tools
The standard library offers a significant number of classes that manage memory efficiently. One needs only consider the standard containers to see shining examples of the sort. In this section, we will take a quick look at a few examples of types useful for resource management. Far from providing an exhaustive list, we’ll try to show different ways to benefit from the RAII idiom.
As mentioned before, when expressing a type that provides automated resource management, the key aspects of that type’s behavior are expressed through its six special member functions. For that reason, with each of the following types, we will take a brief look at what the semantics of these functions are.
unique_ptr<T> and shared_ptr<T>
This short section aims to provide a brief overview of the two main standard smart pointers types in the C++ standard library: `std::unique_ptr<T>` and `std::shared_ptr<T>`. It is meant to provide a broad overview of each type’s role; a more detailed examination of how these types can be used appears in *Chapter 5*, and we will implement simplified versions of both types (as well as of a few other smart pointer types) in *Chapter 6*.
We have seen an example using `std::unique_ptr<T>` earlier in this chapter. An object of this type implements “single ownership of the resource” semantics: an object of type `std::unique_ptr<T>` is uncopiable, and when provided with a `T*` to manage, it destroys the *pointee* at the end of its lifetime. By default, this type will call `delete` on the pointer it manages, but it can be made to use some other means of disposal if needed.
A default `std::unique_ptr<T>` represents an empty object and mostly behaves like a null pointer. Since this type expresses exclusive ownership of a resource, it is uncopiable. Moving from a `std::unique_ptr<T>` transfers ownership of the resource, leaving the moved-from object into an empty state conceptually analogous to a null pointer. The destructor of this type destroys the resource managed by the object, if any.
Type `std::shared_ptr<T>` implements “shared ownership of the resource” semantics. With this type, each `std::shared_ptr<T>` object that co-owns a given pointer shares responsibilities with respect to the pointee’s lifetime and the last co-owner of the resource is responsible for freeing it; as is the case with most smart pointers, this responsibility falls on the object’s destructor. This type is surprisingly complicated to write, even in a somewhat naïve implementation like the one we will write in *Chapter 6*, and is less frequently useful than some people think, as the main use case (expressing ownership in the type system for cases where the last owner of the pointee is a priori unknown, something most frequently seen in multithreaded code) is more specialized than many would believe, but when one needs to fill this niche, it’s the kind of type that’s immensely useful.
A default `std::shared_ptr<T>` also represents an empty object and mostly behaves like a null pointer. Since this type expresses shared ownership of a resource, it is copyable but copying an object means sharing the *pointee*; copy assignment releases the resource held by the object on the left hand of the assignment and then shares the resource held by the object on the right side of the assignment between both objects. Moving from a `std::unique_ptr<T>` transfers ownership of the resource, leaving the moved-from object into an empty state. The destructor of this type releases ownership of the shared resource, destroying the resource managed by the object if that object was the last owner thereof.
What does the “shared” in shared_ptr mean?
There can be confusion with respect to what the word “shared” in the name of the `std::shared_ptr` type actually means. For example, should we use that type whenever we want to share a pointer between caller and callee? Should we use it when whenever client code makes a copy of a pointer with the intent of sharing the pointee, such as when passing a pointer by value to a function or sharing resources stored in a global manager object?
The short answer is that this is the wrong way to approach smart pointers. Sharing a dynamically allocated resource does not mean co-owning that resource: only the latter is what `std::shared_ptr` models, whereas the former can be done with much more lightweight types. We will examine this idea in detail in *Chapter 5* from a usage perspective, then reexamine it in *Chapter 6* with our implementer eyes, hopefully building a more comprehensive understanding of these deep and subtle issues.
lock_guard and scoped_lock
Owning a resource is not limited to owning memory. Indeed, consider the following code excerpt and suppose that `string_mutator` is a class used to perform arbitrary transformations to characters in a `string`, but is expected to be used in a multithreaded context in the sense that one needs to synchronize accesses to that `string` object:

```

#include <thread>

#include <mutex>

#include <string>

#include <algorithm>

#include <string_view>

class string_mutator {

std::string text;

mutable std::mutex m;

public:

// note: m in uncopiable so string_mutator

// also is uncopiable

string_mutator(std::string_view src)

: text{ src.begin(), src.end() } {

}

template <class F> void operator()(F f) {

m.lock();

std::transform(text.begin(), text.end(),

text.begin(), f);

m.unlock();

}

std::string grab_snapshot() const {

m.lock();

std::string s = text;

m.unlock();

return s;

}

};

```cpp

 In this example, a `string_mutator` object’s function call operator accepts an arbitrary function `f` applicable to a `char` and that returns something that can be converted to a `char`, then applies `f` to each `char` in the sequence. For example, the following call would display `"I LOVE` `MY INSTRUCTOR"`:

```

// ...

string_mutator sm{ "I love my instructor" };

sm([](char c) {

return static_cast<char>(std::toupper(c));

});

std::cout << sm.grab_snapshot();

// ...

```cpp

 Now, since `string_mutator::operator()(F)` accepts any function of the appropriate signature as argument, it could among other things accept a function that could throw an exception. Looking at the implementation of that operator, you will notice that with the current (naïve) implementation, this would lock `m` but never unlock it, a bad situation indeed.
There are languages that offer specialized language constructs to solve this problem. In C++, there’s no need for such specialized support as robust code just flows from the fact that one could write an object that locks a mutex at construction time and unlocks it when destroyed… and that’s pretty much all we need. In C++, the simplest such type is `std::lock_guard<M>`, where a simple implementation could look like:

```

template <class M>

class lock_guard { // 简化版本

M &m;

public:

lock_guard(M &m) : m { m } { m.lock(); }

~lock_guard() { m.unlock(); }

lock_guard(const lock_guard&) = delete;

lock_guard& operator=(const lock_guard&) = delete;

};

```cpp

 The simplest types are often the best. Indeed, applying this type to our `string_mutator` example, we end up with a simpler, yet much more robust implementation:

```

#include <thread>

#include <mutex>

#include <string>

#include <algorithm>

#include <string_view>

class string_mutator {

std::string text;

mutable std::mutex m;

public:

// note: m in uncopiable so string_mutator

// also is uncopiable

string_mutator(std::string_view src)

: text{ src.begin(), src.end() } {

}

template <class F> void operator()(F f) {

std::lock_guard lck{ m };

std::transform(text.begin(), text.end(),

text.begin(), f);

} // 隐式 m.unlock

std::string grab_snapshot() const {

std::lock_guard lck{ m };

return text;

} // 隐式 m.unlock

};

```cpp

 Clearly, using destructors to automate unlocking our mutex is advantageous for cases such as this: it simplifies code and helps make it exception-safe.
stream objects
In C++, stream objects are also resource owners. Consider the following code example where we copy each byte from file `in.txt` to the standard output stream:

```

#include <fstream>

#include <iostream>

int main() {

std::ifstream in{ "in.txt" };

for(char c; in.get(c); )

std::cout << c;

}

```cpp

 You might notice a few interesting details in this code: we never call `close()`, there’s no `try` block where we would be preparing ourselves for exception management, there’s no call to `open()` in order to open the file, there’s no explicit check for some end-of-file state… yet, this code works correctly, does what it’s supposed to do, and does not leak resources.
How can such a simple program do all that? Through “the magic of destructors”, or (more precisely) the magic of a good API. Think about it:

*   The constructor’s role is to put the object in a correct initial state. Thus, we use it to open the file as it would be both pointless and inefficient to default-construct the stream, then open it later.
*   Errors when reading from a stream are not exceptional at all… Think about it, how often do we face errors when reading from a stream? In C++, reading from a stream (here: calling `in.get(c)`) returns a reference to the stream after reading from it, and that stream behaves like a `false` Boolean value if the stream is in an error state.
*   Finally, the destructor of a stream object closes whatever representation of a stream it is responsible for. Calling `close()` on a stream in C++ is unnecessary most of the time; just using the stream object in a limited scope generally suffices.

Destructors (and constructors!), when used appropriately, lead to more robust and simpler code.
vector<T> and other containers
We will not write a full-blown comparison of containers with raw arrays or other low-level constructs such as linked lists with manually managed nodes or dynamic arrays maintained explicitly through client code. We will however examine how one can write containers such as `std::vector` or `std::list` in later chapters of this book (*Chapters 12*, *13*, and *14*) when we know a bit more on memory management techniques.
Please note, still, that using `std::vector<T>` (for example) is not only significantly simpler and safer than managing a dynamically allocated array of `T`: in practice, it’s most probably significantly *faster*, at least if used knowledgeably. As we will come to see, there’s no way users can invest the care and attention that goes into memory management and object creation, destruction and copying or movement that goes in a standard container when writing day-to-day code. The destructor of these types, coupled with the way their other special member functions are implemented, make them almost as easy to use as `int` objects, a worthy goal if there ever was one!
Summary
In this chapter, we have discussed some safety-related issues, with a focus on those involving exceptions. We have seen that some standard library types offer specialized semantics with respect to resource management, where “resource” includes but is not limited to memory. In *Chapter 5*, we will spend some time examining how to use and benefit from standard smart pointer; then, in *Chapter 6*, we will go further and look at some of the challenges behind writing your own versions of these smart pointers, as well as some other smart pointer-inspired types with other semantics. Then, we will delve into deeper memory management-related concerns.

```

```cpp

```
