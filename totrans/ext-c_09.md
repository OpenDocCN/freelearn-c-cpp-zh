# 第九章

# C++中的抽象和 OOP

这是 C 中面向对象编程的最后一章。在本章中，我们将涵盖剩余的主题，并介绍一个新的编程范式。此外，我们将探索 C++，并查看它如何在幕后实现面向对象的概念。

作为本章的一部分，我们将涵盖以下主题：

+   首先，我们讨论*抽象*。这继续了我们关于继承和多态的讨论，并将是我们作为 C 中面向对象（OOP）的一部分要覆盖的最后一个主题。我们展示了抽象如何帮助我们设计具有最大可扩展性和最小组件之间依赖性的对象模型。

+   我们讨论了面向对象的概念是如何在一个著名的 C++编译器`g++`中实现的。作为这部分内容的一部分，我们看到我们之前讨论的方法与`g++`提供相同概念的方法是多么接近。

让我们通过讨论抽象来开始本章。

# 抽象

抽象在科学和工程学的各个领域可以有一个非常广泛的意义。但在编程中，尤其是在面向对象编程（OOP）中，抽象本质上处理的是*抽象数据类型*。在基于类的面向对象中，抽象数据类型等同于*抽象类*。抽象类是特殊的类，我们不能从它们中创建对象；它们还没有准备好或足够完整，不能用于对象创建。那么，为什么我们需要这样的类或数据类型呢？这是因为当我们与抽象和通用数据类型一起工作时，我们避免了在代码的各个部分之间创建强烈的依赖关系。

例如，我们可以有如下*人类*和*苹果*类之间的关系：

*人类类的一个对象吃的是苹果类的一个对象*。

*人类类的一个对象吃的是橙子类的一个对象*。

如果一个人类对象可以吃的类被扩展到不仅仅是*苹果*和*橙子*，我们就需要向*人类*类添加更多关系。然而，我们可以创建一个名为*水果*的抽象类，它是*苹果*和*橙子*类的父类，并且我们可以将关系设置为*人类*和*水果*之间。因此，我们可以将前面的两个陈述合并为一个：

*人类类的一个对象吃的是水果类子类型的一个对象*。

*水果*类是抽象的，因为它缺少关于形状、味道、气味、颜色以及更多特定水果属性的详细信息。只有当我们拥有一个苹果或一个橙子时，我们才知道不同属性的确切值。*苹果*和*橙子*类被称为*具体类型*。

我们甚至可以添加更多的抽象。*人类*类可以吃*沙拉*或*巧克力*。因此，我们可以这样说：

*人类类型的一个对象吃的是可食用类子类型的一个对象*。

正如你所见，*Eatable* 的抽象级别甚至高于 *Fruit*。抽象是设计一个具有最小具体类型依赖的对象模型的一种伟大技术，它允许在系统中引入更多具体类型时，对对象模型进行最大程度的未来扩展。

关于前面的例子，我们还可以通过使用 *Human* 是一个 *Eater* 的这一事实来进一步抽象。然后，我们可以使我们的声明更加抽象：

*来自 Eater 类子类的对象会吃来自 Eatable 类子类的对象*。

我们可以继续在对象模型中抽象一切，并找到比我们解决问题所需的级别更抽象的抽象数据类型。这通常被称为 *过度抽象*。这发生在你试图创建没有实际应用（无论是当前还是未来的需求）的抽象数据类型时。无论如何都应该避免这种情况，因为尽管抽象提供了许多好处，但它也可能引起问题。

关于我们需要多少抽象的一般指南可以在 *抽象原则* 中找到。我从其维基百科 [页面](https://en.wikipedia.org/wiki/Abstraction_principle_(computer_programming)) 中获得了以下引言。它简单地陈述：

*程序中每个重要的功能部分都应该在源代码的单一位置实现。当相似的功能由不同的代码块执行时，通常通过抽象出不同的部分将它们组合在一起是有益的*。

虽然乍一看你可能看不到任何面向对象或继承的迹象，但通过进一步思考，你会注意到我们使用继承所做的是基于这个原则。因此，作为一般规则，当你不期望在特定逻辑中存在变化时，在那个点引入抽象是没有必要的。

在一种编程语言中，继承和多态是创建抽象所必需的两个能力。例如，一个名为 *Eatable* 的抽象类相对于其具体类，如 *Apple*，是一个超类型，这是通过继承实现的。

多态也扮演着重要的角色。在抽象类型中，有一些行为在该抽象级别上 *不能* 有默认实现。例如，作为使用行为函数（如 `eatable_get_taste`）实现的属性 *taste*，在谈论 *Eatable* 对象时不能有一个确切值。换句话说，如果我们不知道如何定义 `eatable_get_taste` 行为函数，我们就不能直接从 *Eatable* 类创建对象。

上述函数只能在子类足够具体时才能定义。例如，我们知道 *Apple* 对象应该返回 *甜* 作为它们的味道（我们在这里假设所有苹果都是甜的）。这正是多态发挥作用的地方。它允许子类覆盖其父类的行为并返回适当的味道，例如。

如果您还记得上一章的内容，可以被子类重写的函数称为 *虚函数*。请注意，一个虚函数可能根本没有任何定义。当然，这会使拥有该函数的类成为抽象类。

通过不断增加抽象层次，我们最终会到达没有任何属性且只包含没有默认定义的虚函数的类。这些类被称为 *接口*。换句话说，它们暴露了功能，但根本不提供任何实现，并且通常用于在软件项目中创建各种组件之间的依赖关系。例如，在我们前面的例子中，*Eater* 和 *Eatable* 类是接口。请注意，就像抽象类一样，您绝对不应该从接口创建对象。以下代码展示了为什么在 C 代码中不能这样做。

以下代码框是使用我们在上一章中介绍的技术为前面提到的接口 *Eatable* 在 C 中编写的等效代码，以实现继承和多态：

```cpp
typedef enum {SWEET, SOUR} taste_t;
// Function pointer type
typedef taste_t (*get_taste_func_t)(void*);
typedef struct {
  // Pointer to the definition of the virtual function
  get_taste_func_t get_taste_func;
} eatable_t;
eatable_t* eatable_new() { ... }
void eatable_ctor(eatable_t* eatable) {
  // We don't have any default definition for the virtual function
  eatable->get_taste_func = NULL;
}
// Virtual behavior function
taste_t eatable_get_taste(eatable_t* eatable) {
  return eatable->get_taste_func(eatable);
}
```

代码框 9-1：C 中的 Eatable 接口

如您所见，在构造函数中，我们将 `get_taste_func` 指针设置为 `NULL`。因此，调用 `eatable_get_taste` 虚函数可能导致段错误。从编码的角度来看，这基本上是我们为什么不能从 *Eatable* 接口创建对象，除了我们从接口的定义和设计角度知道的原因之外。

以下代码框展示了从 *Eatable* 接口创建对象，这在 C 的角度来看是完全可能且允许的，但它可能导致崩溃，并且绝对不应该这样做：

```cpp
eatable_t *eatable = eatable_new();
eatable_ctor(eatable);
taste_t taste = eatable_get_taste(eatable); // Segmentation fault!
free(eatable);
```

代码框 9-2：从 Eatable 接口创建对象并调用其纯虚函数时发生段错误

为了防止我们从一个抽象类型创建对象，我们可以从类的公共接口中移除 *分配器函数*。如果您还记得我们在上一章中用于在 C 中实现继承的方法，通过移除分配器函数，只有子类能够从父类的属性结构中创建对象。

外部代码随后将不再能够这样做。例如，在先前的例子中，我们不希望任何外部代码能够从结构 `eatable_t` 中创建任何对象。为了做到这一点，我们需要将属性结构提前声明并使其成为一个不完整类型。然后，我们需要从类中移除公共内存分配器 `eatable_new`。

总结在 C 中创建抽象类所需做的事情，你需要将那些在该抽象级别不应该有默认定义的虚函数指针置为空。在极其高层次的抽象中，我们有一个所有函数指针都为空的接口。为了防止任何外部代码从抽象类型创建对象，我们应该从公共接口中移除分配器函数。

在下一节中，我们将比较 C 和 C++ 中的类似面向对象特性。这让我们了解 C++ 是如何从纯 C 发展而来的。

# C++ 中的面向对象结构

在本节中，我们将比较我们在 C 中所做的工作以及著名 C++ 编译器 `g++` 使用的底层机制来支持封装、继承、多态和抽象。

我们想展示在 C 和 C++ 中实现面向对象概念的方法之间有紧密的一致性。请注意，从现在开始，每当提到 C++ 时，我们实际上是在指 `g++` 作为 C++ 编译器之一的具体实现，而不是 C++ 标准。当然，不同编译器的底层实现可能有所不同，但我们不期望看到很多差异。我们还将使用 `g++` 在 64 位 Linux 环境中。

我们将使用之前讨论的技术在 C 中编写面向对象的代码，然后我们将用 C++ 编写相同的程序，最后得出最终结论。

## 封装

深入研究 C++ 编译器并查看它是如何使用我们迄今为止探索的技术来生成最终可执行文件是困难的，但我们可以使用一个巧妙的技巧来实际看到这一点。这样做的方法是比较两个类似 C 和 C++ 程序生成的汇编指令。

这正是我们将要做的，以证明 C++ 编译器最终生成的汇编指令与使用我们在前几章中讨论的 OOP 技术的 C 程序相同。

*示例 9.1* 讲述了两个 C 和 C++ 程序实现相同的简单面向对象逻辑。在这个例子中有一个 `Rectangle` 类，它有一个用于计算面积的行为函数。我们想查看并比较两个程序中相同行为函数生成的汇编代码。以下代码框展示了 C 版本：

```cpp
#include <stdio.h>
typedef struct {
  int width;
  int length;
} rect_t;
int rect_area(rect_t* rect) {
  return rect->width * rect->length;
}
int main(int argc, char** argv) {
  rect_t r;
  r.width = 10;
  r.length = 25;
  int area = rect_area(&r);
  printf("Area: %d\n", area);
  return 0;
}
```

代码框 9-3 [ExtremeC_examples_chapter9_1.c]：C 中的封装示例

以下代码框显示了前面程序的 C++ 版本：

```cpp
#include <iostream>
class Rect {
public:
  int Area() {
    return width * length;
  }
  int width;
  int length;
};
int main(int argc, char** argv) {
  Rect r;
  r.width = 10;
  r.length = 25;
  int area = r.Area();
  std::cout << "Area: " << area << std::endl;
  return 0;
}
```

代码框 9-4 [ExtremeC_examples_chapter9_1.cpp]：C++ 中的封装示例

因此，让我们生成前面 C 和 C++ 程序的汇编代码：

```cpp
$ gcc -S ExtremeC_examples_chapter9_1.c -o ex9_1_c.s
$ g++ -S ExtremeC_examples_chapter9_1.cpp -o ex9_1_cpp.s
$
```

Shell 框 9-1：生成 C 和 C++ 代码的汇编输出

现在，让我们查看`ex9_1_c.s`和`ex9_1_cpp.s`文件，并寻找行为函数的定义。在`ex9_1_c.s`中，我们应该寻找`rect_area`符号，而在`ex9_1_cpp.s`中，我们应该寻找`_ZN4Rect4AreaEv`符号。请注意，C++会对符号名称进行名称修饰，这就是为什么你需要搜索这个奇怪的符号。C++中的名称修饰已在*第二章*，*编译和链接*中讨论过。

对于 C 程序，以下是为`rect_area`函数生成的汇编代码：

```cpp
$ cat ex9_1_c.s
...
rect_area:
.LFB0:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset 6, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register 6
 movq    %rdi, -8(%rbp)
 movq    -8(%rbp), %rax
 movl    (%rax), %edx
 movq    -8(%rbp), %rax
 movl    4(%rax), %eax
    imull   %edx, %eax
    popq    %rbp
    .cfi_def_cfa 7, 8
    Ret
    .cfi_endproc
...
$
```

Shell Box 9-2：rect_area 函数生成的汇编代码

以下是为`Rect::Area`函数生成的汇编指令：

```cpp
$ cat ex9_1_cpp.s
...
_ZN4Rect4AreaEv:
.LFB1493:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset 6, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register 6
 movq    %rdi, -8(%rbp)
 movq    -8(%rbp), %rax
 movl    (%rax), %edx
 movq    -8(%rbp), %rax
 movl    4(%rax), %eax
    imull   %edx, %eax
    popq    %rbp
    .cfi_def_cfa 7, 8
    Ret
    .cfi_endproc
...
$
```

Shell Box 9-3：Rect::Area 函数生成的汇编代码

令人难以置信的是，它们确实是相同的！我不确定 C++代码是如何变成前面的汇编代码的，但我确信为前面的 C 函数生成的汇编代码几乎在高度精确的程度上与为 C++函数生成的汇编代码相当。

从这个例子中，我们可以得出结论，C++编译器使用了与我们用于 C 的方法类似的方法，这作为*第六章*，*面向对象编程和封装*中*隐式封装*的一部分来实现封装。就像我们处理隐式封装一样，你可以在*代码框 9-3*中看到，将属性结构的指针传递给`rect_area`函数作为第一个参数。

在两个 shell 框中加粗的汇编指令部分，`width`和`length`变量是通过向第一个参数传递的内存地址添加来读取的。根据*System V ABI*，第一个指针参数可以在`%rdi`寄存器中找到。因此，我们可以推断 C++已经将`Area`函数修改为接受一个指针参数作为其第一个参数，该参数指向对象本身。

关于封装的最后一句话，我们看到了 C 和 C++在封装方面密切相关，至少在这个简单的例子中是这样。让我们看看关于继承是否也是同样的情况。

## 继承

调查继承比封装更容易。在 C++中，子类的指针可以被分配给父类的指针。此外，子类应该能够访问父类的私有定义。

这两种行为都表明 C++正在使用我们之前章节中讨论的第一种实现继承的方法，以及第二种方法。如果你需要提醒自己这两种方法，请参阅上一章。

然而，C++的继承似乎更复杂，因为 C++支持多重继承，而我们的第一种方法无法支持。在本节中，我们将检查从 C 和 C++中两个类似类实例化的两个对象的内存布局，如*示例 9.2*所示。

*示例 9.2* 是关于一个简单的类从另一个简单的类继承，这两个类都没有行为函数。C 版本如下：

```cpp
#include <string.h>
typedef struct {
  char c;
  char d;
} a_t;
typedef struct {
  a_t parent;
  char str[5];
} b_t;
int main(int argc, char** argv) {
  b_t b;
  b.parent.c = 'A';
  b.parent.d = 'B';
  strcpy(b.str, "1234");
  // We need to set a break point at this line to see the memory layout.
  return 0;
}
```

代码框 9-5 [ExtremeC_examples_chapter9_2.c]：C 中的继承示例

C++ 版本如下所示：

```cpp
#include <string.h>
class A {
public:
  char c;
  char d;
};
class B : public A {
public:
  char str[5];
};
int main(int argc, char** argv) {
  B b;
  b.c = 'A';
  b.d = 'B';
  strcpy(b.str, "1234");
  // We need to set a break point at this line to see the memory layout.
  return 0;
}
```

代码框 9-6 [ExtremeC_examples_chapter9_2.cpp]：C++ 中的继承示例

首先，我们需要编译 C 程序并使用 `gdb` 在 `main` 函数的最后一行设置断点。当执行暂停时，我们可以检查内存布局以及现有的值：

```cpp
$ gcc -g ExtremeC_examples_chapter9_2.c -o ex9_2_c.out
$ gdb ./ex9_2_c.out
...
(gdb) b ExtremeC_examples_chapter9_2.c:19
Breakpoint 1 at 0x69e: file ExtremeC_examples_chapter9_2.c, line 19.
(gdb) r
Starting program: .../ex9_2_c.out
Breakpoint 1, main (argc=1, argv=0x7fffffffe358) at ExtremeC_examples_chapter9_2.c:20
20    return 0;
(gdb) x/7c &b
0x7fffffffe261: 65 'A'  66 'B'  49 '1'  50 '2'  51 '3'  52 '4'  0 '\000'
(qdb) c
[Inferior 1 (process 3759) exited normally]
(qdb) q
$
```

Shell 框 9-4：在 gdb 中运行示例 9.2 的 C 版本

如您所见，我们已从 `b` 对象的地址开始打印了七个字符，如下所示：`'A'`、`'B'`、`'1'`、`'2'`、`'3'`、`'4'`、`'\0'`。让我们对 C++ 代码也做同样的操作：

```cpp
$ g++ -g ExtremeC_examples_chapter9_2.cpp -o ex9_2_cpp.out
$ gdb ./ex9_2_cpp.out
...
(gdb) b ExtremeC_examples_chapter9_2.cpp:20
Breakpoint 1 at 0x69b: file ExtremeC_examples_chapter9_2.cpp, line 20.
(gdb) r
Starting program: .../ex9_2_cpp.out
Breakpoint 1, main (argc=1, argv=0x7fffffffe358) at ExtremeC_examples_chapter9_2.cpp:21
21    return 0;
(gdb) x/7c &b
0x7fffffffe251: 65 'A'  66 'B'  49 '1'  50 '2'  51 '3'  52 '4'  0 '\000'
(qdb) c
[Inferior 1 (process 3804) exited normally]
(qdb) q
$
```

Shell 框 9-5：在 gdb 中运行示例 9.2 的 C++ 版本

如您在前面的两个 Shell 框中看到的，内存布局和存储在属性中的值是相同的。您不应该因为 C++ 中类中行为函数和属性一起出现而感到困惑；它们将作为类外部分别处理。在 C++ 中，无论您在类中将属性放在哪里，它们总是收集在特定对象的同一内存块中，而函数将始终独立于属性，正如我们在 *第六章*、*面向对象编程和封装* 中查看 *隐式封装* 时所看到的。

之前的示例演示了 *单继承*。那么，*多继承*又是如何呢？在前一章中，我们解释了为什么我们的 C 中实现继承的第一种方法不能支持多继承。我们再次在以下代码框中演示了原因：

```cpp
typedef struct { ... } a_t;
typedef struct { ... } b_t;
typedef struct {
  a_t a;
  b_t b;
  ...
} c_t;
c_t c_obj;
a_t* a_ptr = (a_ptr*)&c_obj;
b_t* b_ptr = (b_ptr*)&c_obj;
c_t* c_ptr = &c_obj;
```

代码框 9-7：演示为什么多继承不能与我们在 C 中实现继承的提议的第一种方法一起工作

在前面的代码框中，`c_t` 类希望继承 `a_t` 和 `b_t` 类。在声明这些类之后，我们创建了 `c_obj` 对象。在前面代码的以下行中，我们创建了不同的指针。

这里的一个重要注意事项是，*所有这些指针都必须指向相同的地址*。`a_ptr` 和 `c_ptr` 指针可以安全地与 `a_t` 和 `c_t` 类的任何行为函数一起使用，但 `b_ptr` 指针的使用是危险的，因为它指向 `c_t` 类中的 a 字段，这是一个 `a_t` 对象。尝试通过 `b_ptr` 访问 `b_t` 内部的字段会导致未定义的行为。

以下代码是前面代码的正确版本，其中所有指针都可以安全使用：

```cpp
c_t c_obj;
a_t* a_ptr = (a_ptr*)&c_obj;
b_t* b_ptr = (b_ptr*)(&c_obj + sizeof(a_t));
c_t* c_ptr = &c_obj;
```

代码框 9-8：演示如何更新类型转换以指向正确的字段

如你在 *Code Box 9-8* 的第三行所见，我们已将 `a_t` 对象的大小添加到 `c_obj` 的地址中；这最终导致一个指向 `c_t` 中 `b` 字段的指针。请注意，C 中的类型转换并不做任何魔法；它在那里是为了转换类型，并且不会修改传递的值，即前一个案例中的内存地址。最终，在赋值之后，右侧的地址会被复制到左侧。

现在，让我们看看 C++ 中的相同示例，并查看 *example 9.3*。假设我们有一个 `D` 类，它从三个不同的类 `A`、`B` 和 `C` 继承。以下是为 *example 9.3* 编写的代码：

```cpp
#include <string.h>
class A {
public:
  char a;
  char b[4];
};
class B {
public:
  char c;
  char d;
};
class C {
public:
  char e;
  char f;
};
class D : public A, public B, public C {
public:
  char str[5];
};
int main(int argc, char** argv) {
  D d;
  d.a = 'A';
  strcpy(d.b, "BBB");
  d.c = 'C';
  d.d = 'D';
  d.e = 'E';
  d.f = 'F';
  strcpy(d.str, "1234");
  A* ap = &d;
  B* bp = &d;
  C* cp = &d;
  D* dp = &d;
  // We need to set a break point at this line.
  return 0;
}
```

Code Box 9-9 [ExtremeC_examples_chapter9_3.cpp]：C++ 中的多重继承

让我们编译这个示例，并用 `gdb` 运行它：

```cpp
$ g++ -g ExtremeC_examples_chapter9_3.cpp -o ex9_3.out
$ gdb ./ex9_3.out
...
(gdb) b ExtremeC_examples_chapter9_3.cpp:40
Breakpoint 1 at 0x100000f78: file ExtremeC_examples_chapter9_3.cpp, line 40.
(gdb) r
Starting program: .../ex9_3.out
Breakpoint 1, main (argc=1, argv=0x7fffffffe358) at ExtremeC_examples_chapter9_3.cpp:41
41    return 0;
(gdb) x/14c &d
0x7fffffffe25a: 65 'A'  66 'B'  66 'B'  66 'B'  0 '\000'    67 'C'  68 'D'  69 'E'
0x7fffffffe262: 70 'F'  49 '1'  50 '2'  51 '3'  52 '4'  0 '\000'
(gdb)
$
```

Shell Box 9-6：在 gdb 中编译和运行示例 9.3

如你所见，属性被放置在彼此相邻的位置。这表明父类中的多个对象被保存在 `d` 对象相同的内存布局中。那么 `ap`、`bp`、`cp` 和 `dp` 指针呢？如你所见，在 C++ 中，当我们将子指针赋值给父指针（向上转型）时，可以隐式地进行类型转换。

让我们检查当前执行中这些指针的值：

```cpp
(gdb) print ap
$1 = (A *) 0x7fffffffe25a
(gdb) print bp
$2 = (B *) 0x7fffffffe25f
(gdb) print cp
$3 = (C *) 0x7fffffffe261
(gdb) print dp
$4 = (D *) 0x7fffffffe25a
(gdb)
```

Shell Box 9-7：打印指针中存储的地址，作为示例 9.3 的一部分

前面的 shell box 显示，`d` 对象的起始地址，显示为 `$4`，与 `ap` 所指向的地址相同，显示为 `$1`。因此，这清楚地表明 C++ 将类型 *A* 的对象作为 *D* 类相应属性结构中的第一个字段。基于指针中的地址和从 `x` 命令得到的结果，类型为 *B* 的对象然后是类型为 *C* 的对象，被放入属于对象 `d` 的相同内存布局中。

此外，前面的地址显示，C++ 中的类型转换不是一个被动的操作，它可以在转换类型的同时对传递的地址执行一些指针算术。例如，在 *Code Box 9-9* 中，当在 `main` 函数中赋值 `bp` 指针时，地址上增加了五个字节或 `sizeof(A)`。这是为了克服我们在 C 中实现多重继承时遇到的问题。现在，这些指针可以很容易地用于所有行为函数，而无需你自己进行算术运算。作为一个重要的注意事项，C 的类型转换和 C++ 的类型转换是不同的，如果你假设 C++ 的类型转换与 C 的类型转换一样被动，你可能会看到不同的行为。

现在是时候看看 C 和 C++ 在多态情况下的相似之处了。

## 多态

比较 C 和 C++中实现多态的底层技术并不是一件容易的事情。在前一章中，我们提出了在 C 中实现多态行为函数的简单方法，但 C++使用了一种更复杂的多态实现机制，尽管基本理念仍然是相同的。如果我们想将我们的方法推广到 C 中的多态实现，我们可以像以下代码框中的伪代码那样做：

```cpp
// Typedefing function pointer types
typedef void* (*func_1_t)(void*, ...);
typedef void* (*func_2_t)(void*, ...);
...
typedef void* (*func_n_t)(void*, ...);
// Attribute structure of the parent class
typedef struct {
  // Attributes
  ...
  // Pointers to functions
  func_1_t func_1;
  func_2_t func_2;
  ...
  func_n_t func_t;
} parent_t;
// Default private definitions for the
// virtual behavior functions
void* __default_func_1(void* parent, ...) {  // Default definition }
void* __default_func_2(void* parent, ...) {  // Default definition }
...
void* __default_func_n(void* parent, ...) {  // Default definition }
// Constructor
void parent_ctor(parent_t *parent) {
  // Initializing attributes
  ...
  // Setting default definitions for virtual
  // behavior functions
  parent->func_1 = __default_func_1;
  parent->func_2 = __default_func_2;
  ...
  parent->func_n = __default_func_n;
}
// Public and non-virtual behavior functions
void* parent_non_virt_func_1(parent_t* parent, ...) { // Code }
void* parent_non_virt_func_2(parent_t* parent, ...) { // Code }
...
void* parent_non_virt_func_m(parent_t* parent, ...) { // Code }
// Actual public virtual behavior functions
void* parent_func_1(parent_t* parent, ...) {
  return parent->func_1(parent, ...); 
}
void* parent_func_2(parent_t* parent, ...) {
  return parent->func_2(parent, ...); 
}
...
void* parent_func_n(parent_t* parent, ...) { 
  return parent->func_n(parent, ...); 
}
```

代码框 9-10：伪代码，演示了如何在 C 代码中声明和定义虚函数

如您在前面伪代码中看到的，父类必须在它的属性结构中维护一个函数指针列表。这些函数指针（在父类中）要么指向虚函数的默认定义，要么是空的。作为*代码框 9-10*的一部分定义的伪类有`m`个非虚行为函数和`n`个虚行为函数。

**注意**：

并非所有行为函数都是多态的。多态行为函数被称为虚行为函数或简单地称为虚函数。在某些语言中，例如 Java，它们被称为*虚方法*。

非虚函数不是多态的，调用它们永远不会得到各种行为。换句话说，对非虚函数的调用只是一个简单的函数调用，它只是执行定义中的逻辑，并不将调用传递给另一个函数。然而，虚函数需要将调用重定向到由父类或子类构造函数设置的适当函数。如果一个子类想要覆盖一些继承的虚函数，它应该更新虚函数指针。

**注意**：

输出变量的`void*`类型可以被替换为任何其他指针类型。我使用了一个通用指针来表明伪代码中的函数可以返回任何东西。

以下伪代码显示了子类如何覆盖*代码框 9-10*中找到的一些虚函数：

```cpp
Include everything related to parent class ...
typedef struct {
  parent_t parent;
  // Child attributes
  ...
} child_t;
void* __child_func_4(void* parent, ...) { // Overriding definition }
void* __child_func_7(void* parent, ...) { // Overriding definition }
void child_ctor(child_t* child) {
  parent_ctor((parent_t*)child);
  // Initialize child attributes
  ...
  // Update pointers to functions
  child->parent.func_4 = __child_func_4;
  child->parent.func_7 = __child_func_7;
}
// Child's behavior functions
...
```

代码框 9-11：C 语言中的伪代码，演示了子类如何覆盖从父类继承的一些虚函数

如您在*代码框 9-11*中看到的，子类只需要更新父类属性结构中的几个指针。C++采取了类似的方法。当你将行为函数声明为虚函数（使用`virtual`关键字）时，C++创建一个函数指针数组，这与我们在*代码框 9-10*中做的方式非常相似。

如您所见，我们为每个虚函数添加了一个函数指针属性，但 C++有更智能的方式来保持这些指针。它只是使用一个名为*虚表*或*vtable*的数组。虚表是在创建对象之前创建的。它首先在调用基类的构造函数时填充，然后作为子类构造函数的一部分，正如我们在*代码框 9-10*和*9-11*中所示。

由于虚表仅在构造函数中填充，因此应避免在父类或子类构造函数中调用多态方法，因为其指针可能尚未更新，可能指向错误的定义。

作为我们关于 C 和 C++中实现各种面向对象概念的底层机制的最后一次讨论，我们将讨论抽象。

## 抽象类

在 C++中，可以使用*纯虚*函数来实现抽象。在 C++中，如果你将成员函数定义为虚函数并将其设置为零，你就声明了一个纯虚函数。看看以下示例：

```cpp
enum class Taste { Sweet, Sour };
// This is an interface
class Eatable {
public:
  virtual Taste GetTaste() = 0;
};
```

代码框 9-12：C++中的`Eatable`接口

在类`Eatable`内部，我们有一个被设置为零的`GetTaste`虚函数。`GetTaste`是一个纯虚函数，这使得整个类成为抽象类。你不能再从`*Eatable*`类型创建对象，C++不允许这样做。此外，`*Eatable*`是一个接口，因为它的所有成员函数都是纯虚的。这个函数可以在子类中被重写。

以下是一个重写`GetTaste`函数的类的示例：

```cpp
enum class Taste { Sweet, Sour };
// This is an interface
class Eatable {
public:
  virtual Taste GetTaste() = 0;
};
class Apple : public Eatable {
public:
  Taste GetTaste() override {
    return Taste::Sweet;
  }
};
```

代码框 9-13：实现`Eatable`接口的两个子类

纯虚函数与虚函数非常相似。实际定义的地址以与虚函数相同的方式保存在虚表中，但有一个区别。纯虚函数指针的初始值是 null，而正常虚函数的指针需要在构造过程中指向默认定义。

与不知道抽象类型的 C 编译器不同，C++编译器知道抽象类型，如果你尝试从抽象类型创建对象，它会生成编译错误。

在本节中，我们比较了使用过去三章中介绍的技术在 C 中使用和`g++`编译器在 C++中使用的各种面向对象的概念，并比较了它们。我们展示了我们采用的方法在大多数情况下与`g++`之类的编译器使用的技巧是一致的。

# 摘要

在本章中，我们总结了面向对象编程（OOP）主题的探索，从抽象开始，通过展示 C 和 C++在面向对象概念方面的相似性继续前进。

在本章中，我们讨论了以下主题：

+   我们最初讨论了抽象类和接口。使用它们，我们可以有一个接口或部分抽象的类，这可以用来创建具有多态和不同行为的具体子类。

+   我们然后将我们在 C 中使用的技术带来的面向对象特征的输出与`g++`产生的输出进行了比较。这是为了展示结果是多么相似。我们得出结论，我们采用的技术在结果上可以非常相似。

+   我们更深入地讨论了虚拟表。

+   我们展示了如何使用纯虚函数（这是一个 C++概念，但确实有一个 C 语言的对应物）来声明没有默认定义的虚拟行为。

下一章将介绍 Unix 及其与 C 的关系。它将回顾 Unix 的历史和 C 的发明。它还将解释 Unix 系统的分层架构。
