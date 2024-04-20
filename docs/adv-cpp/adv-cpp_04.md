# 第四章：不允许泄漏-异常和资源

## 学习目标

在本章结束时，您将能够：

+   开发管理资源的类

+   开发异常健壮的代码，以防止资源通过 RAII 泄漏

+   实现可以通过移动语义传递资源所有权的类

+   实现控制隐式转换的类

在本章中，您将学习如何使用类来管理资源，防止泄漏，并防止复制大量数据。

## 介绍

在*第 2A 章*中，*不允许鸭子-类型和推断*，我们简要涉及了一些概念，如智能指针和移动语义。在本章中，我们将进一步探讨它们。事实证明，这些主题与资源管理和编写健壮的代码（经常运行并长时间运行而没有问题的代码）非常密切相关。

为了理解发生了什么，我们将探讨变量在内存中的放置位置，以及当它们超出范围时发生了什么。

我们将查看编译器为我们输入的汇编代码生成了什么，并探讨当异常发生时所有这些都受到了什么影响。

### 变量范围和生命周期

在*第 2B 章*中，*不允许鸭子-模板和推断*，我们讨论了变量范围和生命周期。让我们快速浏览它们的不同类型：

**范围**：

+   `{}`）。

+   **全局/文件范围**：这适用于在普通函数或类之外声明的变量，也适用于普通函数。

**寿命**：

+   **自动寿命**：在这里，局部变量在声明时创建，并在退出其所在范围时销毁。这些由堆栈管理。

+   **永久寿命**：在这里，全局和静态局部变量具有永久寿命。

+   `new`和`delete`操作符）。这些变量的内存是从堆中分配的。

我们将使用以下程序来澄清`局部变量`的行为-具有`自动寿命`和具有`动态寿命`的变量：

![](img/C14583_03_01.jpg)

###### 图 3.1：变量范围和生命周期的测试程序

当我们运行上述程序时，将生成以下输出：

![图 3.2：生命周期测试程序的输出](img/C14583_03_02.jpg)

###### 图 3.2：生命周期测试程序的输出

在上述输出中的十六进制数字（`0xNNNNNNNN`）是正在构造或销毁的 Int 对象的地址。我们的程序从`第 46 行`进入`main()`函数开始。此时，程序已经进行了大量初始化，以便我们随时可以使用一切。下面的图表指的是两个堆栈-**PC 堆栈**和**数据堆栈**。

这些是帮助我们解释幕后发生的事情的抽象概念。`PC 堆栈`（`程序计数器堆栈`）用于记住程序计数器的值（指向需要运行的下一条指令的寄存器），而`数据堆栈`保存我们正在操作的值或地址。尽管这是两个单独的堆栈，在实际 CPU 上，它很可能会被实现为一个堆栈。让我们看看以下表格，其中我们使用缩写`OLn`来引用上述程序输出的行号：

![图 3.3：测试程序执行的详细分析（第 1 部分）](img/C14583_03_03.jpg)

###### 图 3.3：测试程序执行的详细分析（第 1 部分）

以下是测试程序执行详细分析的第二部分：

![图 3.4：测试程序执行的详细分析（第 2 部分）](img/C14583_03_04.jpg)

###### 图 3.4：测试程序执行的详细分析（第 2 部分）

以下是测试程序执行详细分析的第三部分：

![图 3.5：测试程序执行的详细分析（第 3 部分）](img/C14583_03_05.jpg)

###### 图 3.5：测试程序执行的详细分析（第 3 部分）

从这个简单的程序中，我们学到了一些重要的事实：

+   当我们按值传递时，会调用复制构造函数（就像在这种情况下所做的那样）。

+   返回类型只会调用一个构造函数（不是两个构造函数 - 一个用于创建返回对象，一个用于存储返回的数据） - C++将其称为**复制省略**，现在在标准中是强制性的。

+   在作用域终止时（闭合大括号'`}`'），任何超出作用域的变量都会调用其析构函数。如果这是真的，那么为什么地址`0x6000004d0`没有显示析构函数调用（`~Int()`）？这引出了下一个事实。

+   在`calculate()`方法的析构函数中，我们泄漏了一些内存。

了解和解决资源泄漏问题的最后两个事实是重要的。在我们处理 C++中的异常之后，我们将研究资源管理。

## C++中的异常

我们已经看到了 C++如何管理具有自动和动态生命周期的局部作用域变量。当变量超出作用域时，它调用具有自动生命周期的变量的析构函数。我们还看到了原始指针在超出作用域时被销毁。由于它不清理动态生命周期变量，我们会失去它们。这是我们后来构建**资源获取即初始化**（**RAII**）的故事的一部分。但首先，我们需要了解异常如何改变程序的流程。

### 异常的必要性

在*第 2A 章*，*不允许鸭子 - 类型和推断*中，我们介绍了枚举作为处理`check_file()`函数的魔术数字的一种方式：

```cpp
FileCheckStatus check_file(const char* name)
{
  FILE* fptr{fopen(name,"r")};
  if ( fptr == nullptr)
    return FileCheckStatus::NotFound;
  char buffer[30];
  auto numberRead = fread(buffer, 1, 30, fptr);
  fclose(fptr);
  if (numberRead != 30)
    return FileCheckStatus::IncorrectSize;
  if(is_valid(buffer))
    return FileCheckStatus::InvalidContents;
  return FileCheckStatus::Good;
}
```

前面的函数使用了一种称为**状态**或**错误代码**的技术来报告操作的结果。这是 C 风格编程中使用的方法，其中与**POSIX API**和**Windows API**相关的错误被处理。

#### 注意

`POSIX`代表`可移植操作系统接口`。这是 Unix 变体和其他操作系统之间软件兼容性的 IEEE 标准。

这意味着，方法的调用者必须检查返回值，并针对每种错误类型采取适当的操作。当您可以推断代码将生成的错误类型时，这种方法效果很好。但并非总是如此。例如，可能存在输入到程序的数据存在问题。这会导致程序中的异常状态无法处理。具有处理错误逻辑的代码部分被从检测问题的代码部分中移除。

虽然可能编写处理此类问题的代码，但这会增加处理所有错误条件的复杂性，从而使程序难以阅读，难以推断函数应该执行的操作，并因此难以维护。

对于错误处理，异常比错误代码提供以下优点：

+   错误代码可以被忽略 - 异常强制处理错误（或程序终止）。

+   异常可以沿着堆栈流向最佳方法来响应错误。错误代码需要传播到每个中间方法之外。

+   异常将错误处理与主程序流程分离，使软件易于阅读和维护。

+   异常将检测错误的代码与处理错误的代码分离。

只要遵循最佳实践并将异常用于异常条件，使用异常不会有（时间）开销。这是因为一个实现良好的编译器将提供 C++的口号 - 你不为你不使用的东西付费。它可能会消耗一些内存，你的代码可能会变得稍微庞大，但运行时间不应受影响。

C++使用异常来处理运行时异常。通过使用异常，我们可以检测错误，抛出异常，并将错误传播回可以处理它的位置。让我们修改前面的程序，引入`divide()`函数并更改`calculate()`函数以调用它。我们还将在`main()`函数中添加日志记录，以便探索异常的行为方式：

![图 3.6：用于调查异常的修改测试程序](img/C14583_03_06.jpg)

###### 图 3.6：用于调查异常的修改测试程序

当我们编译并运行前面的程序时，将生成以下输出：

![图 3.7：测试程序的输出](img/C14583_03_07.jpg)

###### 图 3.7：测试程序的输出

在前面的代码中，您可以看到注释已添加到右侧。现在，我们从程序中的`result2`行中删除注释，重新编译程序并重新运行。生成的新输出如下所示：

![图 3.8：测试程序的输出 - result2](img/C14583_03_08.jpg)

###### 图 3.8：测试程序的输出 - result2

通过比较输出，我们可以看到每个输出的前八行是相同的。前面输出的接下来两行是因为`divide()`函数被调用了两次。最后一行指示抛出了异常并且程序被终止。

第二次调用`divide()`函数尝试除以零 - 这是一种异常操作。这导致异常。如果整数被零除，那么会导致浮点异常。这与在`POSIX`系统中生成异常的方式有关 - 它使用了称为信号的东西（我们不会在这里详细介绍信号的细节）。当整数被零除时，`POSIX`系统将其映射到称为`浮点错误`的信号，但现在是更通用的`算术错误`。

#### **注意**

根据 C++标准，如果零出现为除数，无论是'/'运算符（除法）还是'%'运算符（取模），行为都是未定义的。大多数系统会选择抛出异常。

因此，我们从前面的解释中学到了一个重要的事实：未处理的异常将终止程序（在内部调用`std::terminate()`）。我们将修复`未定义行为`，捕获异常，并查看输出中的变化。为了修复`未定义行为`，我们需要在文件顶部添加`#include <stdexcept>`并修改`divide()`函数：

```cpp
Int divide(Int a, Int b )
{
    if (b.m_value == 0)
        throw std::domain_error("divide by zero error!");
    return a.m_value/b.m_value;
}
```

当我们重新编译并运行程序时，我们得到以下输出：

![图 3.9：当我们抛出异常时的输出](img/C14583_03_09.jpg)

###### 图 3.9：当我们抛出异常时的输出

从前面的输出中可以看到，没有太多变化。只是我们不再得到`浮点异常`（核心转储）- 程序仍然终止但不会转储核心。然后我们在`main()`函数中添加了一个`try/catch`块，以确保异常不再是未处理的。

![图 3.10：捕获异常](img/C14583_03_10.jpg)

###### 图 3.10：捕获异常

重新编译程序并运行以获得以下输出：

![图 3.11：捕获异常的程序输出](img/C14583_03_11.jpg)

###### 图 3.11：捕获异常的程序输出

在前面的输出中，异常在第二行抛出，注释为“**复制 a 以调用 divide**”。之后的所有输出都是异常处理的结果。

我们的代码已将程序控制转移到`main()`函数中的`catch()`语句，并执行了在`try`子句中进行调用时在堆栈上构造的所有变量的析构函数。

### 堆栈展开

C++语言所保证的销毁所有本地函数变量的过程被称为**堆栈展开**。在异常出现时，堆栈展开时，C++使用其明确定义的规则来销毁作用域中的所有对象。

当异常发生时，函数调用堆栈从当前函数开始线性搜索，直到找到与异常匹配的异常处理程序（由`catch`块表示）。

如果找到异常处理程序，则进行堆栈展开，销毁堆栈中所有函数的本地变量。对象按创建顺序的相反顺序销毁。如果找不到处理抛出异常的处理程序，则程序将终止（通常不会警告用户）。

### 练习 1：在 Fraction 和 Stack 中实现异常

在这个练习中，我们将回到*第 2A 章*和*第 2B 章*中我们所做的两个类，*不允许鸭子 - 类型和推断*和*不允许鸭子 - 模板和推断* - `Fraction`和`Stack`，它们都可能出现运行时异常。我们将更新它们的代码，以便在检测到任何问题时都能引发异常。按照以下步骤执行此练习：

1.  打开 Eclipse，并使用**Lesson3**示例文件夹中的文件创建一个名为**Lesson3**的新项目。

1.  由于这是一个**基于 CMake 的项目**，因此将当前构建器更改为**CMake Build (portable)**。

1.  转到**项目** | **构建所有**菜单以构建所有练习。默认情况下，屏幕底部的控制台将显示**CMake Console [Lesson3]**。

1.  配置一个新的**启动配置**，**L3Exercise1**，以运行名称为**Exercise1**的项目。

1.  还要配置一个新的 C/C++单元运行配置，**L3Ex1Tests**，以运行**L3Ex1tests**。设置**Google Tests Runner**。

1.  点击**运行**选项，对现有的**18**个测试进行运行和通过。![图 3.12：现有测试全部通过（运行次数：18）](img/C14583_03_12.jpg)

###### 图 3.12：现有测试全部通过（运行次数：18）

1.  在编辑器中打开**Fraction.hpp**，并更改文件顶部的行，使其读起来像这样：

```cpp
#define EXERCISE1_STEP  14
```

1.  点击`Fraction`，其中分母为零。测试期望抛出异常：![图 3.13：新的失败测试 ThrowsDomainErrorForZeroDenominator](img/C14583_03_13.jpg)

###### 图 3.13：新的失败测试 ThrowsDomainErrorForZeroDenominator

1.  点击失败的测试名称 - “预期…抛出 std::domain_error 类型的异常”，下一行显示“实际：它没有抛出任何异常”。

1.  双击消息，它将带您到以下测试：![图 3.14：失败的测试](img/C14583_03_14.jpg)

###### 图 3.14：失败的测试

`ASSERT_THROW()`宏需要两个参数。由于`Fraction 初始化器`中有一个逗号，因此需要在第一个参数的外面再加一组括号。第二个参数预期从这个构造函数中获得一个`std::domain_error`。内部的`try/catch`结构用于确认预期的字符串是否被捕获在异常对象中。如果我们不想检查这一点，那么我们可以简单地这样编写测试：

```cpp
ASSERT_THROW(({Fraction f1{1,0}; }), std::domain_error);
```

1.  在编辑器中打开文件**Fraction.cpp**。在文件顶部附近插入以下行：

```cpp
#include <stdexcept> 
```

1.  修改构造函数，如果使用零分母创建，则抛出异常：

```cpp
Fraction::Fraction(int numerator, int denominator) 
                       : m_numerator{numerator}, m_denominator{denominator}
{
    if(m_denominator == 0) 
    {
        throw std::domain_error("Zero Denominator");
    }
}
```

1.  点击**运行**按钮重新运行测试。现在有**19**个测试通过。

1.  在编辑器中打开**Fraction.hpp**，并更改文件顶部附近的行，使其读起来像这样：

```cpp
#define EXERCISE1_STEP  20
```

1.  点击`ThrowsRunTimeErrorForZeroDenominator`失败。

1.  点击失败的测试名称 - “预期…抛出 std::runtime_error 类型的异常”，下一行显示“实际：它抛出了不同类型的异常”。

1.  再次双击消息以打开失败的测试：![图 3.15：另一个失败的测试](img/C14583_03_15.jpg)

###### 图 3.15：另一个失败的测试

此测试验证除法赋值运算符对零进行除法时会抛出异常。

1.  打开`operator/=()`函数。您会看到，在这个函数内部，它实际上使用了`std::domain_error`的构造函数。

1.  现在修改`operator/=()`以在调用构造函数之前检测此问题，以便抛出带有预期消息的`std::runtime_error`。

1.  通过添加一个将检测除法运算符的域错误来修改**Fraction.cpp**：

```cpp
Fraction& Fraction::operator/=(const Fraction& rhs)
{
    if (rhs.m_numerator == 0)
    {
        throw std::runtime_error("Fraction Divide By Zero");
    }
    Fraction tmp(m_numerator*rhs.m_denominator, 
m_denominator*rhs.m_numerator);
    *this = tmp;
    return *this;
}
```

1.  点击**Run**按钮重新运行测试。所有**20**个测试通过。

1.  在编辑器中打开**Stack.hpp**并更改文件顶部附近的行，使其读起来像这样：

```cpp
#define EXERCISE1_STEP  27
```

1.  点击`FractionTest`以折叠测试列表并显示`StackTest`：![图 3.16：pop Stack 测试失败](img/C14583_03_16.jpg)

###### 图 3.16：pop Stack 测试失败

1.  在文件顶部使用`#include <stdexcept>`，然后更新`pop()`函数，使其如下所示：

```cpp
void pop()
{
    if(empty())
        throw std::underflow_error("Pop from empty stack");
    m_stack.pop_back();
} 
```

1.  点击**Run**按钮重新运行测试。现在**21**个测试通过了。

1.  在编辑器中打开**Stack.hpp**并更改文件顶部的行，使其读起来像这样：

```cpp
#define EXERCISE1_STEP  31
```

1.  点击`TopEmptyStackThrowsUnderFlowException`，失败。

1.  使用`top()`方法，使其如下所示：

```cpp
reference top()
{
    if(empty())
        throw std::underflow_error("Top from empty stack");
    return m_stack.back();
}
```

1.  点击**Run**按钮重新运行测试。**22**个测试通过。

1.  在编辑器中打开**Stack.hpp**并更改文件顶部的行，使其读起来像这样：

```cpp
#define EXERCISE1_STEP  35
```

1.  点击`TopEmptyConstStackThrowsUnderFlowException`，失败。

1.  使用`top()`方法，使其如下所示：

```cpp
const_reference top() const
{
    if(empty())
        throw std::underflow_error("Top from empty stack");
    return m_stack.back();
}
```

1.  点击**Run**按钮重新运行测试。现在所有**23**个测试都通过了。

在这个练习中，我们为使用我们的`Fraction`和`Stack`类的正常操作的前提条件添加了运行时检查。当违反前提条件之一时，此代码将仅执行以抛出异常，表明数据或程序执行方式存在问题。

### 当抛出异常时会发生什么？

在某个时刻，我们的程序执行以下语句：

```cpp
throw expression;
```

通过执行此操作，我们正在发出发生错误的条件，并且我们希望它得到处理。接下来发生的事情是一个**临时**对象，称为**异常对象**，在未指定的存储中构造，并从表达式进行复制初始化（可能调用移动构造函数，并可能受到复制省略的影响）。异常对象的类型从表达式中静态确定，去除 const 和 volatile 限定符。数组类型会衰减为指针，而函数类型会转换为函数的指针。如果表达式的类型格式不正确或抽象，则会发生编译器错误。

在异常对象构造之后，控制权连同异常对象一起转移到异常处理程序。被选择的异常处理程序是与异常对象最匹配的类型，因为堆栈展开。异常对象存在直到最后一个 catch 子句退出，除非它被重新抛出。表达式的类型必须具有可访问的`复制构造函数`和`析构函数`。

### 按值抛出还是按指针抛出

知道临时异常对象被创建，传递，然后销毁，抛出表达式应该使用什么类型？一个`值`还是一个`指针`？

我们还没有详细讨论在 catch 语句中指定类型。我们很快会做到。但是现在，请注意，要捕获指针类型（被抛出的），catch 模式也需要是指针类型。

如果抛出对象的指针，那么抛出方必须确保异常对象将指向的内容（因为它将是指针的副本）在异常处理之前保持活动，即使通过`堆栈展开`也是如此。

指针可以指向静态变量、全局变量或从堆中分配的内存，以确保在处理异常时被指向的对象仍然存在。现在，我们已经解决了保持异常对象存活的问题。但是当处理程序完成后，捕获者该怎么办？

异常的捕获者不知道异常对象的创建（`全局`，`静态`或`堆`），因此不知道是否应该删除接收到的指针。因此，通过指针抛出异常不是推荐的异常抛出方法。

被抛出的对象将被复制到创建的临时异常对象中，并交给处理程序。当异常被处理后，临时对象将被简单地销毁，程序将继续执行。对于如何处理它没有歧义。因此，最佳实践是通过值抛出异常。

### 标准库异常

C++标准库将`std::exception`定义为所有标准库异常的基类。标准定义了以下第一级层次的`异常`/`错误`（括号中的数字表示从该类派生的异常数量）：

![图 3.17：标准库异常层次结构（两级）](img/C14583_03_17.jpg)

###### 图 3.17：标准库异常层次结构（两级）

这些异常在 C++标准库中被使用，包括 STL。创建自己的异常类的最佳实践是从标准异常中派生它。接下来我们会看到，你的特殊异常可以被标准异常的处理程序捕获。

### 捕获异常

在讨论异常的需要时，我们介绍了抛出异常的概念，但并没有真正看看 C++如何支持捕获异常。异常处理的过程始于将代码段放在`try`块中以进行**异常检查**。try 块后面是一个或多个 catch 块，它们是异常处理程序。当在 try 块内执行代码时发生异常情况时，异常被抛出，控制转移到异常处理程序。如果没有抛出异常，那么所有异常处理程序都将被跳过，try 块中的代码完成，正常执行继续。让我们在代码片段中表达这些概念：

```cpp
void SomeFunction()
{
  try {
    // code under exception inspection
  }
  catch(myexception e)         // first handler – catch by value
  {
    // some error handling steps
  }
  catch(std::exception* e)     // second handler – catch by pointer
  {
    // some other error handling steps
  }
  catch(std::runtime_error& e) // third handler – catch by reference
  {
    // some other error handling steps
  }
  catch(...)                   // default exception handler – catch any exception
  {
    // some other error handling steps
  }
  // Normal programming continues from here
}
```

前面的片段展示了必要的关键字 - `try`和`catch`，并介绍了三种不同类型的捕获模式（不包括默认处理程序）：

+   **通过值捕获异常**：这是一种昂贵的机制，因为异常处理程序像任何其他函数一样被处理。通过值捕获意味着必须创建异常对象的副本，然后传递给处理程序。第二个副本的创建减慢了异常处理过程。这种类型也可能受到对象切片的影响，其中子类被抛出，而 catch 子句是超类。然后 catch 子句只会接收到失去原始异常对象属性的超类对象的副本。因此，我们应避免使用通过值捕获异常处理程序。

+   **通过指针捕获异常**：如在讨论通过值抛出时所述，通过指针抛出，这种异常处理程序只能捕获指针抛出的异常。由于我们只想通过值抛出，应避免使用通过指针捕获异常处理程序。

+   `通过值抛出`和`通过引用捕获`。

当存在多个 catch 块时，异常对象类型用于匹配按指定顺序的处理程序。一旦找到匹配的处理程序，它就会被执行，并且剩余的异常处理程序将被忽略。这与函数解析不同，编译器将找到最佳匹配的参数。因此，异常处理程序（catch 块）应该从更具体到更一般的定义。例如，默认处理程序（`catch(...)`）应该始终在定义中的最后一个。

### 练习 2：实现异常处理程序

在这个练习中，我们将实现一系列异常处理程序的层次结构，以管理异常的处理方式。按照以下步骤实现这个练习：

1.  打开`e`。该变量的作用域仅限于它声明的 catch 块。

1.  单击**启动配置**下拉菜单，然后选择**新启动配置…**。从**搜索项目**菜单配置**L3Exercise2**应用程序以使用名称**L3Exercise2**运行它。

1.  完成后，它将是当前选择的**启动配置**。

1.  点击**运行**按钮。练习 2 将运行并产生以下输出：![图 3.18：练习 2 输出-默认处理程序捕获了异常](img/C14583_03_18.jpg)

###### 图 3.18：练习 2 输出-默认处理程序捕获了异常

1.  在控制台窗口中，单击`CMake`文件设置`-fpermissive`标志，当它编译此目标时。）

1.  在编辑器中，将默认异常处理程序`catch(...)`移动到`std::domain_error`处理程序后面。点击**运行**按钮。练习 2 将运行并产生以下输出：![图 3.19：已使用 std::exception 处理程序](img/C14583_03_19.jpg)

###### 图 3.19：已使用 std::exception 处理程序

1.  在编辑器中，将`std::exception`处理程序移动到`std::domain_error`处理程序后面。点击`std::logic_error`处理程序按预期执行。

1.  在编辑器中，将`std:: logic_error`处理程序移动到`std::domain_error`处理程序后面。点击`std:: domain_error`处理程序被执行，这实际上是我们所期望的。

1.  现在将`throw`行更改为`std::logic_error`异常。点击`std::logic_error`处理程序按预期执行。

1.  现在将`throw`行更改为`std::underflow_error`异常。点击`std::exception`处理程序按预期执行。`std::exception`是所有标准库异常的基类。

在这个练习中，我们实现了一系列异常处理程序，并观察了异常处理程序的顺序如何影响异常的捕获以及异常层次结构如何被使用。

### CMake 生成器表达式

在使用`CMake`时，有时需要调整变量的值。`CMake`是一个构建生成系统，可以为许多构建工具和编译器工具链生成构建文件。由于这种灵活性，如果要在编译器中启用某些功能，只需将其应用于特定类型。这是因为不同供应商之间的命令行选项是不同的。例如，g++编译器启用 C++17 支持的命令行选项是`-std=c++17`，但对于`msvc`来说是`/std:c++17`。如果打开`add_excutable`，那么以下行将在其后：

```cpp
target_compile_options(L3Exercise2 PRIVATE $<$<CXX_COMPILER_ID:GNU>:-fpermissive>)
```

这使用`$<CXX_COMPILER_ID:GNU>`变量查询来检查它是否是 GCC 编译器。如果是，则生成 1（true），否则生成 0（false）。它还使用`$<condition:true_string>`条件表达式将`-fpermissive`添加到`target_compile_options`的编译器选项或通过一个调用。

#### 注意

有关生成器表达式的更多信息，请查看以下链接：[`cmake.org/cmake/help/v3.15/manual/cmake-generator-expressions.7.html`](https://cmake.org/cmake/help/v3.15/manual/cmake-generator-expressions.7.html)。

### 异常使用指南

在 C++代码中使用异常时，请记住以下几点：

+   口号：**按值抛出，按引用捕获**

+   **不要将异常用于正常程序流**。如果函数遇到异常情况并且无法满足其（功能性）义务，那么只有在这种情况下才抛出异常。如果函数可以解决异常情况并履行其义务，那么这不是异常。它们之所以被称为异常，是有原因的，如果不使用它们，就不会产生任何处理开销。

+   **不要在析构函数中抛出异常**。请记住，由于堆栈展开，局部变量的析构函数将被执行。如果在堆栈展开过程中调用了析构函数并抛出了异常，那么程序将终止。

+   **不要吞没异常**。不要使用默认的 catch 处理程序，也不要对异常做任何处理。异常被抛出是为了指示存在问题，你应该对此做些什么。忽视异常可能会导致以后难以排查的故障。这是因为任何有用的信息都真正丢失了。

+   **异常对象是从抛出中复制的**。

## 资源管理（在异常世界中）

到目前为止，我们已经看过局部变量作用域，以及当变量超出作用域时如何处理`自动`和`动态生命周期变量` - 自动生命周期变量（放在堆栈上的变量）将被完全析构，而`动态生命周期变量`（由程序员分配到堆上的变量）不会被析构：我们只是失去了对它们的任何访问。我们也看到，当抛出异常时，会找到最近匹配的处理程序，并且在堆栈展开过程中将析构抛出点和处理程序之间的所有局部变量。

我们可以利用这些知识编写健壮的资源管理类，这些类将使我们不必跟踪资源（动态生命周期变量、文件句柄、系统句柄等），以确保在使用完它们后将它们释放（释放到野外）。在正常操作和异常情况下管理资源的技术被称为**资源获取即初始化**（**RAII**）。

### 资源获取即初始化

RAII 是另一个命名不好的概念的好例子（另一个是`SFINAE`）。`RAII`或`Resource Acquisition is Initialization`描述了一个用于管理资源的类的行为。如果它被命名为`File`类并展示了 RAII 如何提高可读性和我们对函数操作的理解能力，可能会更好。

考虑以下代码：

```cpp
void do_something()
{
    FILE* out{};
    FILE* in = fopen("input.txt", "r");
    try 
    {
        if (in != nullptr)
        {
            // UNSAFE – an exception here will create a resource leak
            out = fopen("output.txt", "w");
            if (out != nullptr)
            {
                // Do some work
                // UNSAFE – an exception here will create resource leaks
                fclose(out);
            }
            fclose(in);
        }
    }
    catch(std::exception& e)
    {
        // Respond to the exception
    }
}
```

这段代码展示了资源管理的两个潜在问题：

+   最重要的是，在打开和关闭文件之间发生异常会导致资源泄漏。如果这是系统资源，许多这样的情况可能导致系统不稳定或应用程序性能受到不利影响，因为它会因资源匮乏而受到影响。

+   此外，在一个方法中管理多个资源可能会导致由于错误处理而产生深度嵌套的子句。这对代码的可读性有害，因此也影响了代码的理解和可维护性。很容易忘记释放资源，特别是当有多个退出点时。

那么，我们如何管理资源，以便有异常安全和更简单的代码？这个问题不仅仅是 C++独有的，不同的语言以不同的方式处理它。`Java`、`C#`和`Python`使用垃圾回收方法，在对象创建后清理它们，当它们不再被引用时。但是 C++没有垃圾回收，那么解决方案是什么呢？

考虑以下类：

```cpp
class File {
public:
    File(const char* name, const char* access) {
        m_file = fopen(name, access);
        if (m_file == nullptr) {
            throw std::ios_base::failure("failed to open file");
        }
    }
    ~File() {
        fclose(m_file);
    }
    operator FILE*() {
        return m_file;
    }
private:
    FILE* m_file{};
};
```

这个类实现了以下特征：

+   构造函数获取资源。

+   如果资源没有在构造函数中获取，那么会抛出异常。

+   当类被销毁时，资源被释放。

如果我们在`do_something()`方法中使用这个类，那么它看起来像这样：

```cpp
void do_something()
{
    try 
    {
        File in("input.txt", "r");
        File out("output.txt", "w");
        // Do some work
    }
    catch(std::exception& e)
    {
        // Respond to the exception
    }
}
```

如果在执行此操作时发生异常，那么 C++保证将调用所有基于堆栈的对象的析构函数（`堆栈展开`），从而确保文件被关闭。这解决了在发生异常时资源泄漏的问题，因为现在资源会自动清理。此外，这种方法非常容易阅读，因此我们可以理解逻辑流程，而不必担心错误处理。

这种技术利用`File`对象的生命周期来获取和释放资源，确保资源不会泄漏。资源在管理类的构造（初始化）期间获取，并在管理类的销毁期间释放。正是这种作用域绑定资源的行为导致了`Resource Acquisition Is Initialization`的名称。

前面的例子涉及管理系统资源的文件句柄。它适用于任何在使用前需要获取，然后在完成后放弃的资源。RAII 技术可以应用于各种资源 - 打开文件，打开管道，分配的堆内存，打开套接字，执行线程，数据库连接，互斥锁/临界区的锁定 - 基本上是主机系统中供应不足的任何资源，并且需要进行管理。

### 练习 3：为内存和文件句柄实现 RAII

在这个练习中，我们将实现两个不同的类，使用 RAII 技术来管理内存或文件。按照以下步骤来实现这个练习：

1.  在 Eclipse 中打开**Lesson3**项目。然后在**Project Explorer**中展开**Lesson3**，然后展开**Exercise03**，双击**Exercise3.cpp**以打开此练习的文件到编辑器中。

1.  点击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。从搜索项目菜单中配置**L3Exercise3**应用程序以使用名称**L3Exercise3**运行它。

1.  当`monitor`被析构时，点击`main()`函数，它会转储分配和释放的内存报告，以及打开但从未关闭的文件。

1.  在编辑器中，输入以下内容到`File`类中：

```cpp
class File {
public:
    File(const char* name, const char* access) {
        m_file = fopen(name, access);
        if (m_file == nullptr) {
            throw std::ios_base::failure(""failed to open file"");
        }
    }
    ~File() {
        fclose(m_file);
    }
    operator FILE*() {
        return m_file;
    }
private:
    FILE* m_file{};
};
```

1.  点击**Run**按钮运行 Exercise 3 - 它仍然泄漏文件和内存，但代码是正确的。

1.  找到`LeakFiles()`函数，并修改它以使用新的`File`类（就像前面的代码一样）以防止文件泄漏：

```cpp
void LeakFiles()
{
    File fh1{"HelloB1.txt", "w"};
    fprintf(fh1, "Hello B2\n");
    File fh2{"HelloB2.txt", "w"};
    fprintf(fh2, "Hello B1\n");
}
```

1.  正确点击`LeakFiles()`，然后输出将如下所示：![图 3.21：没有文件泄漏](img/C14583_03_21.jpg)

###### 图 3.21：没有文件泄漏

1.  现在在`CharPointer`类中：

```cpp
class CharPointer
{
public:
    void allocate(size_t size)
    {
        m_memory = new char[size];
    }
    operator char*() { return m_memory;}
private:
    char* m_memory{};
};
```

1.  修改`LeakPointers()`如下所示：

```cpp
void LeakPointers()
{
    CharPointer memory[5];
    for (auto i{0} ; i<5 ; i++)
    {
        memory[i].allocate(20); 
        std::cout << "allocated 20 bytes @ " << (void *)memory[i] << "\n";
    }
}
```

1.  点击**Run**按钮运行 Exercise 3 - 它仍然有内存泄漏，但代码是正确的。

1.  现在，向`CharPointer`添加以下析构函数。请注意，`delete`操作符使用数组`[]`语法：

```cpp
~CharPointer()
{
    delete [] m_memory;
}
```

1.  再次点击**Run**按钮运行 Exercise 3 - 这次，您应该看到监视器报告没有泄漏：

![图 3.22：没有泄漏 - 内存或文件](img/C14583_03_22.jpg)

###### 图 3.22：没有泄漏 - 内存或文件

`File`和`CharPointer`的实现符合`RAII`设计方法，但在设计这些方法时还有其他考虑因素。例如，我们是否需要复制构造函数或复制赋值函数？在这两种情况下，仅仅从一个对象复制资源到另一个对象可能会导致关闭文件句柄或删除内存的两次尝试。通常，这会导致未定义的行为。接下来，我们将重新审视特殊成员函数，以实现`File`或`CharPointer`等资源管理对象。

### 特殊编码技术

*练习 3*的代码，*为内存和文件句柄实现 RAII*，已经特别编写，以便我们可以监视内存和文件句柄的使用，并在退出时报告任何泄漏。访问**monitor.h**和**monitor.cpp**文件，并检查用于使监视器可能的两种技术：

+   如果包括`SendMessageA`或`SendMessageW`，则`SendMessage`

+   **定义我们自己的新处理程序**：这是一种高级技术，除非你编写嵌入式代码，否则你不太可能需要它。

### C++不需要最终

其他支持异常抛出机制的语言（`C#`、`Java`和`Visual Basic.NET`）具有`try/catch/finally`范式，其中`finally`块中的代码在退出 try 块时被调用 - 无论是正常退出还是异常退出。C++没有`finally`块，因为它有更好的机制，可以确保我们不会忘记释放资源 - RAII。由于资源由本地对象表示，本地对象的析构函数将释放资源。

这种设计模式的附加优势是，如果正在管理大量资源，则`finally`块的大小也相应较大。RAII 消除了对 finally 的需求，并导致更易于维护的代码。

### RAII 和 STL

标准模板库（STL）在许多模板和类中使用 RAII。例如，C++11 中引入的智能指针，即`std::unique_ptr`和`std::shared_ptr`，通过确保在使用完毕后释放内存，或者确保在其他地方使用时不释放内存，帮助避免了许多问题。STL 中的其他示例包括`std::string`（内存）、`std::vector`（内存）和`std::fstream`（文件句柄）。

### 谁拥有这个对象？

通过前面对`File`和`CharPointer`的实现，我们已经测试了使用 RAII 进行资源管理。让我们进一步探讨。首先，我们将定义一个不仅拥有一个资源的类：

```cpp
class BufferedWriter
{
public:
    BufferedWriter(const char* filename);
    ~BufferedWriter();
    bool write(const char* data, size_t length);
private:
    const size_t BufferSize{4096};
    FILE* m_file{nullptr};
    size_t m_writePos{0};
    char* m_buffer{new char[BufferSize]};
};
```

该类用于缓冲写入文件。

#### 注意

当使用 iostream 派生类时，这通常是不必要的，因为它们已经提供了缓冲。

每次调用`write()`函数都会将数据添加到分配的缓冲区，直到达到`BufferSize`，此时数据实际写入文件，并且缓冲区被重置。

但是如果我们想要将`BufferedWriter`的这个实例分配给另一个实例或复制它呢？什么是正确的行为？

如果我们只是让默认的复制构造函数/复制赋值做它们的事情，我们会得到项目的成员复制。这意味着我们有两个`BufferedWriter`的实例，它们持有相同的文件句柄和缓冲区指针。当对象的第一个实例被销毁时，作为优秀的程序员，我们将通过关闭文件和删除内存来清理文件。第二个实例现在有一个失效的文件句柄和一个指向我们已告诉操作系统为下一个用户恢复的内存的指针。任何尝试使用这些资源，包括销毁它们，都将导致未定义的行为，很可能是程序崩溃。默认的复制构造函数/复制赋值运算符执行所谓的浅复制 - 也就是说，它按位复制所有成员（但不是它们所指的内容）。

我们拥有的两个资源可以被不同对待。首先，应该只有一个类拥有`m_buffer`。在处理这个问题时有两个选择：

+   防止类的复制，因此也防止内存。

+   执行`深复制`，其中第二个实例中的缓冲区是由构造函数分配的，并且复制了第一个缓冲区的内容

其次，应该只有一个类拥有文件句柄（`m_file`）。在处理这个问题时有两个选择：

+   防止类的复制，因此也防止文件句柄的复制

+   将`所有权`从原始实例转移到第二个实例，并将原始实例标记为无效或空（无论这意味着什么）

实现深拷贝很容易，但如何转移资源的所有权呢？为了回答这个问题，我们需要再次看看临时对象和值类别。

### 临时对象

在将结果存储到变量（或者只是忘记）之前，创建临时对象来存储表达式的中间结果。表达式是任何返回值的代码，包括按值传递给函数，从函数返回值，隐式转换，文字和二进制运算符。临时对象是`rvalue 表达式`，它们有内存，为它们分配了临时位置，以放置表达式结果。正是这种创建临时对象和在它们之间复制数据导致了 C++11 之前的一些性能问题。为了解决这个问题，C++11 引入了`rvalue 引用`，以实现所谓的移动语义。

### 移动语义

一个`rvalue 引用`（用双`&&`表示）是一个只分配给`rvalue`的引用，它将延长`rvalue`的生命周期，直到`rvalue 引用`完成为止。因此，`rvalues`可以在定义它的表达式之外存在。有了`rvalue 引用`，我们现在可以通过移动构造函数和移动赋值运算符来实现移动语义。移动语义的目的是从被引用对象中窃取资源，从而避免昂贵的复制操作。当移动完成时，被引用对象必须保持在稳定状态。换句话说，被移动的对象必须保持在一个状态，不会在销毁时引起任何未定义的行为或程序崩溃，也不应该影响从中窃取的资源。

C++11 还引入了一个转换运算符`std::move()`，它将一个`lvalue`转换为一个`rvalue`，以便调用移动构造函数或移动赋值运算符来'移动'资源。`std::move()`方法实际上并不移动数据。

一个意外的事情要注意的是，在移动构造函数和移动赋值运算符中，`rvalue`引用实际上是一个`lvalue`。这意味着如果你想确保在方法内发生移动语义，那么你可能需要再次在成员变量上使用`std::move()`。

随着 C++11 引入了移动语义，它还更新了标准库以利用这种新的能力。例如，`std::string`和`std::vector`已经更新以包括移动语义。要获得移动语义的好处，你只需要用最新的 C++编译器重新编译你的代码。

### 实现智能指针

智能指针是一个资源管理类，它在资源超出范围时持有指向资源的指针并释放它。在本节中，我们将实现一个智能指针，观察它作为一个支持复制的类的行为，使其支持移动语义，最后移除其对复制操作的支持：

```cpp
#include <iostream>
template<class T>
class smart_ptr
{
public:
  smart_ptr(T* ptr = nullptr) :m_ptr(ptr)
  {
  }
  ~smart_ptr()
  {
    delete m_ptr;
  }
  // Copy constructor --> Do deep copy
  smart_ptr(const smart_ptr& a)
  {
    m_ptr = new T;
    *m_ptr = *a.m_ptr;      // use operator=() to do deep copy
  }
  // Copy assignment --> Do deep copy 
  smart_ptr& operator=(const smart_ptr& a)
  {
    // Self-assignment detection
    if (&a == this)
      return *this;
    // Release any resource we're holding
    delete m_ptr;
    // Copy the resource
    m_ptr = new T;
    *m_ptr = *a.m_ptr;
    return *this;
  }
  T& operator*() const { return *m_ptr; }
  T* operator->() const { return m_ptr; }
  bool is_null() const { return m_ptr == nullptr; }
private:
  T* m_ptr{nullptr};
};
class Resource
{
public:
  Resource() { std::cout << "Resource acquired\n"; }
  ~Resource() { std::cout << "Resource released\n"; }
};
smart_ptr<Resource> createResource()
{
    smart_ptr<Resource> res(new Resource);                       // Step 1
    return res; // return value invokes the copy constructor     // Step 2
}
int main()
{
  smart_ptr<Resource> the_res;
  the_res = createResource(); // assignment invokes the copy assignment Step 3/4

  return 0; // Step 5
}
```

当我们运行这个程序时，会生成以下输出：

![图 3.23：智能指针程序输出](img/C14583_03_23.jpg)

###### 图 3.23：智能指针程序输出

对于这样一个简单的程序，获取和释放资源的操作很多。让我们来分析一下：

1.  在`createResource()`内部的局部变量 res 是在堆上创建并初始化的（动态生命周期），导致第一个“`获取资源`”消息。

1.  编译器可能创建另一个临时对象来返回值。然而，编译器已经执行了`复制省略`来删除复制（也就是说，它能够直接在调用函数分配的堆栈位置上构建对象）。编译器有`返回值优化`（`RVO`）和`命名返回值优化`（`NRVO`）优化，它可以应用，并且在 C++17 中，在某些情况下这些优化已经成为强制性的。

1.  临时对象通过复制赋值分配给`main()`函数中的`the_res`变量。由于复制赋值正在进行深拷贝，因此会获取资源的另一个副本。

1.  当赋值完成时，临时对象超出范围，我们得到第一个"资源释放"消息。

1.  当`main()`函数返回时，`the_res`超出范围，释放第二个 Resource。

因此，如果资源很大，我们在`main()`中创建`the_res`局部变量的方法非常低效，因为我们正在创建和复制大块内存，这是由于复制赋值中的深拷贝。然而，我们知道当`createResource()`创建的临时变量不再需要时，我们将丢弃它并释放其资源。在这些情况下，将资源从临时变量转移（或移动）到类型的另一个实例中将更有效。移动语义使我们能够重写我们的`smart_ptr`模板，而不是进行深拷贝，而是转移资源。

让我们为我们的`smart_ptr`类添加移动语义：

```cpp
// Move constructor --> transfer resource
smart_ptr(smart_ptr&& a) : m_ptr(a.m_ptr)
{
  a.m_ptr = nullptr;    // Put into safe state
}
// Move assignment --> transfer resource
smart_ptr& operator=(smart_ptr&& a)
{
  // Self-assignment detection
  if (&a == this)
    return *this;
  // Release any resource we're holding
  delete m_ptr;
  // Transfer the resource
  m_ptr = a.m_ptr;
  a.m_ptr = nullptr;    // Put into safe state
  return *this;
}
```

重新运行程序后，我们得到以下输出：

![图 3.24：使用移动语义的智能指针程序输出](img/C14583_03_24.jpg)

###### 图 3.24：使用移动语义的智能指针程序输出

现在，因为移动赋值现在可用，编译器在这一行上使用它：

```cpp
the_res = createResource(); // assignment invokes the copy assignment Step 3/4
```

第 3 步现在已经被移动赋值所取代，这意味着深拷贝现在已经被移除。

`第 4 步`不再释放资源，因为带有注释“//”的行将其置于安全状态——它不再具有要释放的资源，因为其所有权已转移。

另一个需要注意的地方是`移动构造函数`和`移动赋值`的参数在它们的拷贝版本中是 const 的，而在它们的移动版本中是`非 const`的。这被称为`所有权的转移`，这意味着我们需要修改传入的参数。

移动构造函数的另一种实现可能如下所示：

```cpp
// Move constructor --> transfer resource
smart_ptr(smart_ptr&& a) 
{
  std::swap(this->m_ptr, a.m_ptr);
}
```

基本上，我们正在交换资源，C++ STL 支持许多特化的模板交换。这是因为我们使用成员初始化将`m_ptr`设置为`nullptr`。因此，我们正在交换`nullptr`和存储在`a`中的值。

现在我们已经解决了不必要的深拷贝问题，我们实际上可以从`smart_ptr()`中删除复制操作，因为实际上我们想要的是所有权的转移。如果我们将非临时`smart_ptr`的实例复制到另一个非临时`smart_ptr`实例中，那么当它们超出范围时会删除资源，这不是期望的行为。为了删除（深）复制操作，我们改变了成员函数的定义，如下所示：

```cpp
smart_ptr(const smart_ptr& a) = delete;
smart_ptr& operator=(const smart_ptr& a) = delete;
```

我们在*第 2A 章*中看到的`= delete`的后缀告诉编译器，尝试访问具有该原型的函数现在不是有效的代码，并导致错误。

### STL 智能指针

与其编写自己的`smart_ptr`，不如使用 STL 提供的类来实现我们对象的 RAII。最初的是`std::auto_ptr()`，它在 C++ 11 中被弃用，并在 C++ 17 中被移除。它是在`rvalue`引用支持之前创建的，并且因为它使用复制来实现移动语义而导致问题。C++ 11 引入了三个新模板来管理资源的生命周期和所有权：

+   通过指针管理`单个对象`，并在`unique_ptr`超出范围时销毁该对象。它有两个版本：用`new`创建的单个对象和用`new[]`创建的对象数组。`unique_ptr`与直接使用底层指针一样高效。

+   **std::shared_ptr**：通过指针保留对象的共享所有权。它通过引用计数管理资源。每个分配给 shared_ptr 的 shared_ptr 的副本都会更新引用计数。当引用计数变为零时，这意味着没有剩余所有者，资源被释放/销毁。

+   `shared_ptr`，但不修改计数器。可以检查资源是否仍然存在，但不会阻止资源被销毁。如果确定资源仍然存在，那么可以用它来获得资源的`shared_ptr`。一个使用场景是多个`shared_ptrs`最终形成循环引用的情况。循环引用会阻止资源的自动释放。`weak_ptr`用于打破循环并允许资源在应该被释放时被释放。

### std::unique_ptr

`std::unique_ptr()`在 C++ 11 中引入，以取代`std::auto_ptr()`，并为我们提供了`smart_ptr`所做的一切（以及更多）。我们可以将我们的`smart_ptr`程序重写如下：

```cpp
#include <iostream>
#include <memory>
class Resource
{
public:
  Resource() { std::cout << "Resource acquired\n"; }
  ~Resource() { std::cout << "Resource released\n"; }
};
std::unique_ptr<Resource> createResource()
{
  std::unique_ptr<Resource> res(new Resource);
  return res; 
}
int main()
{
  std::unique_ptr<Resource> the_res;
  the_res = createResource(); // assignment invokes the copy assignment
  return 0;
}
```

我们可以进一步进行，因为 C++ 14 引入了一个辅助方法，以确保在处理`unique_ptrs`时具有异常安全性：

```cpp
std::unique_ptr<Resource> createResource()
{
  return std::make_unique<Resource>(); 
}
```

*为什么这是必要的？*考虑以下函数调用：

```cpp
some_function(std::unique_ptr<T>(new T), std::unique_ptr<U>(new U));
```

问题在于编译器可以自由地以任何顺序对参数列表中的操作进行排序。它可以调用`new T`，然后`new U`，然后`std::unique_ptr<T>()`，最后`std::unique_ptr<U>()`。这个顺序的问题在于，如果`new U`抛出异常，那么由调用`new T`分配的资源就没有被放入`unique_ptr`中，并且不会自动清理。使用`std::make_unique<>()`可以保证调用的顺序，以便资源的构建和`unique_ptr`的构建将一起发生，不会泄漏资源。在 C++17 中，对这些情况下的评估顺序的规则已经得到了加强，因此不再需要`make_unique`。然而，使用`make_unique<T>()`方法仍然可能是一个好主意，因为将来转换为 shared_ptr 会更容易。

名称`unique_ptr`清楚地表明了模板的意图，即它是指向对象的唯一所有者。这在`auto_ptr`中并不明显。同样，`shared_ptr`清楚地表明了它的意图是共享资源。`unique_ptr`模板提供了对以下操作符的访问：

+   **T* get()**：返回指向托管资源的指针。

+   如果实例管理资源，则为`true`（`get() != nullptr`）。

+   对托管资源的`lvalue`引用。与`*get()`相同。

+   `get()`。

+   `unique_ptr(new [])`，它提供对托管数组的访问，就像它本来是一个数组一样。返回一个`lvalue`引用，以便可以设置和获取值。

### std::shared_ptr

当您想要共享资源的所有权时，可以使用共享指针。为什么要这样做？有几种情况适合共享资源，比如在 GUI 程序中，您可能希望共享字体对象、位图对象等。**GoF 飞行权重设计模式**就是另一个例子。

`std::shared_ptr`提供了与`std::unique_ptr`相同的所有功能，但因为现在必须为对象跟踪引用计数，所以有更多的开销。所有在`std::unique_ptr`中描述的操作符都可以用在`std::shared_ptr`上。一个区别是创建`std::shared_ptr`的推荐方法是调用`std::make_shared<>()`。

在编写库或工厂时，库的作者并不总是知道用户将如何使用已创建的对象，因此建议从工厂方法返回`unique_ptr<T>`。原因是用户可以通过赋值轻松地将`std::unique_ptr`转换为`std::shared_ptr`：

```cpp
std::unique_ptr<MyClass> unique_obj = std::make_unique<MyClass>();
std::shared_ptr<MyClass> shared_obj = unique_obj;
```

这将转移所有权并使`unique_obj`为空。

#### 注意

一旦资源被作为共享资源，就不能将其恢复为唯一对象。

### std::weak_ptr

弱指针是共享指针的一种变体，但它不持有资源的引用计数。因此，当计数降为零时，它不会阻止资源被释放。考虑以下程序结构，它可能出现在正常的图形用户界面（GUI）中：

```cpp
#include <iostream>
#include <memory>
struct ScrollBar;
struct TextWindow;
struct Panel
{
    ~Panel() {
        std::cout << "--Panel destroyed\n";
    }
    void setScroll(const std::shared_ptr<ScrollBar> sb) {
        m_scrollbar = sb;
    }
    void setText(const std::shared_ptr<TextWindow> tw) {
        m_text = tw;
    }
    std::weak_ptr<ScrollBar> m_scrollbar;
    std::shared_ptr<TextWindow> m_text;
};
struct ScrollBar
{
    ~ScrollBar() {
        std::cout << "--ScrollBar destroyed\n";
    }
    void setPanel(const std::shared_ptr<Panel> panel) {
        m_panel=panel;
    }
    std::shared_ptr<Panel> m_panel;
};
struct TextWindow
{
    ~TextWindow() {
        std::cout << "--TextWindow destroyed\n";
    }
    void setPanel(const std::shared_ptr<Panel> panel) {
        m_panel=panel;
    }
    std::shared_ptr<Panel> m_panel;
};
void run_app()
{
    std::shared_ptr<Panel> panel = std::make_shared<Panel>();
    std::shared_ptr<ScrollBar> scrollbar = std::make_shared<ScrollBar>();
    std::shared_ptr<TextWindow> textwindow = std::make_shared<TextWindow>();
    scrollbar->setPanel(panel);
    textwindow->setPanel(panel);
    panel->setScroll(scrollbar);
    panel->setText(textwindow);
}
int main()
{
    std::cout << "Starting app\n";
    run_app();
    std::cout << "Exited app\n";
    return 0;
}
```

执行时，输出如下：

![图 3.25：弱指针程序输出](img/C14583_03_25.jpg)

###### 图 3.25：弱指针程序输出

这表明当应用程序退出时，面板和`textwindow`都没有被销毁。这是因为它们彼此持有`shared_ptr`，因此两者的引用计数不会降为零并触发销毁。如果我们用图表表示结构，那么我们可以看到它有一个`shared_ptr`循环：

![图 3.26：弱指针和共享指针循环](img/C14583_03_26.jpg)

###### 图 3.26：弱指针和共享指针循环

### 智能指针和调用函数

现在我们可以管理我们的资源了，我们如何使用它们？我们传递智能指针吗？当我们有一个智能指针（`unique_ptr`或`shared_ptr`）时，在调用函数时有四个选项：

+   通过值传递智能指针

+   通过引用传递智能指针

+   通过指针传递托管资源

+   通过引用传递托管资源

这不是一个详尽的列表，但是主要考虑的。我们如何传递智能指针或其资源的答案取决于我们对函数调用的意图：

+   函数的意图是仅仅使用资源吗？

+   函数是否接管资源的所有权？

+   函数是否替换托管对象？

如果函数只是要`使用资源`，那么它甚至不需要知道它正在使用托管资源。它只需要使用它，并且应该通过指针、引用（甚至值）调用资源：

```cpp
do_something(Resource* resource);
do_something(Resource& resource);
do_something(Resource resource);
```

如果你想要将资源的所有权传递给函数，那么函数应该通过智能指针按值调用，并使用`std::move()`调用：

```cpp
do_something(std::unique_ptr<Resource> resource);
auto res = std::make_unique<Resource>();
do_something (std::move(res));
```

当`do_something()`返回时，`res`变量将为空，资源现在由`do_something()`拥有。

如果你想要`替换托管对象`（一个称为**重新安置**的过程），那么你通过引用传递智能指针：

```cpp
do_something(std::unique_ptr<Resource>& resource);
```

以下程序将所有内容整合在一起，演示了每种情况以及如何调用函数：

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
class Resource
{
public:
  Resource() { std::cout << "+++Resource acquired ["<< m_id <<"]\n"; }
  ~Resource() { std::cout << "---Resource released ["<< m_id <<"]\n"; }
  std::string name() const {
      std::ostringstream ss;
      ss << "the resource [" << m_id <<"]";
      return ss.str();
  }
  int m_id{++m_count};
  static int m_count;
};
int Resource::m_count{0};
void use_resource(Resource& res)
{
    std::cout << "Enter use_resource\n";
    std::cout << "...using " << res.name() << "\n";
    std::cout << "Exit use_resource\n";
}
void take_ownership(std::unique_ptr<Resource> res)
{
    std::cout << "Enter take_ownership\n";
    if (res)
        std::cout << "...taken " << res->name() << "\n";
    std::cout << "Exit take_ownership\n";
}
void reseat(std::unique_ptr<Resource>& res)
{
    std::cout << "Enter reseat\n";
    res.reset(new Resource);
    if (res)
        std::cout << "...reseated " << res->name() << "\n";
    std::cout << "Exit reseat\n";
}
int main()
{
  std::cout << "Starting...\n";
  auto res = std::make_unique<Resource>();
  // Use - pass resource by reference
  use_resource(*res);               
  if (res)
    std::cout << "We HAVE the resource " << res->name() << "\n\n";
  else
    std::cout << "We have LOST the resource\n\n";
  // Pass ownership - pass smart pointer by value
  take_ownership(std::move(res));    
  if (res)
    std::cout << "We HAVE the resource " << res->name() << "\n\n";
  else
    std::cout << "We have LOST the resource\n\n";
  // Replace (reseat) resource - pass smart pointer by reference
  reseat(res);                      
  if (res)
    std::cout << "We HAVE the resource " << res->name() << "\n\n";
  else
    std::cout << "We have LOST the resource\n\n";
  std::cout << "Exiting...\n";
  return 0;
}
```

当我们运行这个程序时，我们会收到以下输出：

![](img/C14583_03_27.jpg)

###### 图 3.27：所有权传递程序输出

#### 注意

*C++核心指南*有一个完整的部分涉及*资源管理*、智能指针以及如何在这里使用它们：[`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-resource`](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-resource)。我们只触及了指南涵盖的最重要的方面。

### 练习 4：使用 STL 智能指针实现 RAII

在这个练习中，我们将实现一个传感器工厂方法，通过`unique_ptr`返回传感器资源。我们将实现一个`unique_ptr`来持有一个数组，然后开发代码将`unique_ptr`转换为共享指针，然后再分享它。按照以下步骤实现这个练习：

1.  在 Eclipse 中打开**Lesson3**项目。然后在**项目资源管理器**中展开**Lesson3**，然后**Exercise04**，双击**Exercise4.cpp**以将此练习的文件打开到编辑器中。

1.  单击**启动配置**下拉菜单，选择**新启动配置...**。从**搜索项目**菜单中配置**L3Exercise4**应用程序，以便它以名称**L3Exercise4**运行。

1.  单击**运行**按钮运行练习 4。这将产生以下输出：![图 3.28：练习 4 输出](img/C14583_03_28.jpg)

###### 图 3.28：练习 4 输出

1.  在编辑器中，检查代码，特别是工厂方法，即`createSensor(type)`。

```cpp
std::unique_ptr<ISensor>
createSensor(SensorType type)
{
    std::unique_ptr<ISensor> sensor;
    if (type == SensorType::Light)
    {
        sensor.reset(new LightSensor);
    }
    else if (type == SensorType::Temperature)
    {
        sensor.reset(new TemperatureSensor);
    }
    else if (type == SensorType::Pressure)
    {
        sensor.reset(new PressureSensor);
    }
    return sensor;
}
```

这将创建一个名为 sensor 的空 unique 指针，然后根据传入的`type`重置包含的指针以获取所需的传感器。

1.  在编辑器中打开 Exercise4.cpp，并将文件顶部附近的行更改为以下内容：

```cpp
#define EXERCISE4_STEP  5
```

1.  点击`unique_ptr`到`shared_ptr`是不允许的。

1.  找到报告错误的行，并将其更改为以下内容：

```cpp
SensorSPtr light2 = std::move(light);
```

1.  点击`light`（一个`unique_ptr`）到`light2`（一个`shared_ptr`）。问题实际上是模板方法：

```cpp
template<typename SP>
void printSharedPointer(SP sp, const char* message)
```

第一个参数是按值传递的，这意味着将创建`shared_ptr`的新副本并传递给方法进行打印。

1.  现在让我们通过将模板更改为按引用传递来修复这个问题。点击**Run**按钮编译和运行程序。生成以下输出：![图 3.31：已更正的 printSharedPointer 输出](img/C14583_03_31.jpg)

###### 图 3.31：已更正的 printSharedPointer 输出

1.  在编辑器中打开**Exercise4.cpp**，并将文件顶部附近的行更改为以下内容：

```cpp
#define EXERCISE4_STEP  12
```

1.  点击**Run**按钮编译和运行程序。生成以下输出：![](img/C14583_03_32.jpg)

###### 图 3.32：Exercise 4 的注释步骤 12 输出

1.  将输出与`testSensors()`方法中的代码进行比较。我们会发现可以轻松地将空的`unique_ptr`（`light`）分配给另一个，并且可以在不需要在任何情况下使用`std::move()`的情况下从一个`shared_ptr`分配给另一个（`light3 = light2`）。

1.  在编辑器中打开**Exercise4.cpp**，并将文件顶部附近的行更改为以下内容：

```cpp
#define EXERCISE4_STEP  15
```

1.  点击**Run**按钮编译和运行程序。输出切换为以下内容：![图 3.33：在 unique_ptr 中管理数组](img/C14583_03_33.jpg)

###### 图 3.33：在 unique_ptr 中管理数组

1.  在编辑器中找到`testArrays()`方法：

```cpp
void testArrays()
{
    std::unique_ptr<int []> board = std::make_unique<int []>(8*8);
    for(int i=0  ; i<8 ; i++)
        for(int j=0 ; j<8 ; j++)
            board[i*8+j] = 10*(i+1)+j+1;
    for(int i=0  ; i<8 ; i++)
    {
        char sep{' '};
        for(int j=0 ; j<8 ; j++)
            std::cout << board[i*8+j] << sep;
        std::cout << "\n";
    }
}
```

在这段代码中有几点需要注意。首先，类型声明为`int[]`。我们在这个练习中选择了`int`，但它可以是几乎任何类型。其次，当使用`unique_ptr`（自 C++ 17 以来也是`shared_ptr`）来管理数组时，定义了`operator[]`。因此，我们通过从二维索引的`board[i*8+j]`计算出一维索引来模拟二维数组。

1.  编辑方法的第一行并声明`auto`类型：

```cpp
auto board = std::make_unique<int []>(8*8);
```

1.  点击`make_unique()`调用。

在这个练习中，我们实现了一个工厂函数，使用`unique_ptr`来管理传感器的生命周期。然后，我们实现了将其从`unique_ptr`更改为共享到多个对象。最后，我们开发了一种使用单一维数组来管理多维数组的`unique_ptr`技术。

### 零/五法则-不同的视角

当我们引入`BufferedWriter`时，它管理了两个资源：内存和文件。然后我们讨论了默认编译器生成的浅拷贝操作。我们谈到了我们可以以不同的方式管理资源-停止复制，执行深拷贝，或者转移所有权。在这些情况下我们决定如何做被称为资源管理策略。您选择的策略将影响您如何执行`零/五法则`。

在资源管理方面，一个类可以管理零个资源，管理可以复制但不能移动的资源，管理可以移动但不能复制的资源，或者管理不应复制也不应移动的资源。以下类显示了如何表达这些类别：

```cpp
struct NoResourceToManage
{
    // use compiler generated copy & move constructors and operators
};
struct CopyOnlyResource
{
    ~CopyOnlyResource()                                      {/* defined */ }
    CopyOnlyResource(const CopyOnlyResource& rhs)            {/* defined */ }
    CopyOnlyResource& operator=(const CopyOnlyResource& rhs) {/* defined */ }
    CopyOnlyResource(CopyOnlyResource&& rhs) = delete;
    CopyOnlyResource& operator=(CopyOnlyResource&& rhs) = delete;
};
struct MoveOnlyResource
{
    ~MoveOnlyResource()                                      {/* defined */ }
    MoveOnlyResource(const MoveOnlyResource& rhs)             = delete;
    MoveOnlyResource& operator=(const MoveOnlyResource& rhs)  = delete;
    MoveOnlyResource(MoveOnlyResource&& rhs)                 {/* defined */ }  
    MoveOnlyResource& operator=(MoveOnlyResource&& rhs)      {/* defined */ }
};
struct NoMoveOrCopyResource
{
    ~NoMoveOrCopyResource()                                  {/* defined */ }
    NoMoveOrCopyResource(const NoMoveOrCopyResource& rhs)             = delete;
    NoMoveOrCopyResource& operator=(const NoMoveOrCopyResource& rhs)  = delete;
    NoMoveOrCopyResource(NoMoveOrCopyResource&& rhs)                  = delete;
    NoMoveOrCopyResource& operator=(NoMoveOrCopyResource&& rhs)       = delete;
};
```

由于在所有上下文和异常下管理资源的复杂性，最佳实践是，如果一个类负责管理资源，那么该类只负责管理该资源。

### 活动 1：使用 RAII 和 Move 实现图形处理

在*第 2A 章*，*不允许鸭子-类型和推断*中，您的团队努力工作并实现了`Point3d`和`Matrix3d`。现在，您的公司希望在推出之前对库进行两项重大改进：

+   公司的类必须在一个命名空间中，即 Advanced C Plus Plus Inc.因此，图形的命名空间将是`accp::gfx`。

+   `Point3d`和`Matrix3d`中矩阵的存储是类的固有部分，因此它是从堆栈而不是堆中分配的。作为库矩阵支持的演变，我们需要从堆中分配内存。因为我们正在努力实现更大的矩阵在未来的版本中，我们还希望在我们的类中引入移动语义。

按照以下步骤实现这一点：

1.  从我们当前版本的库开始（可以在`acpp::gfx`命名空间中找到。

1.  修复所有因更改而失败的测试。（失败可能意味着编译失败，而不仅仅是运行测试。）

1.  在`Matrix3d`中，从在类中直接声明矩阵切换到堆分配的存储器。

1.  通过实现复制构造函数和复制赋值运算符的深度复制实现来修复失败的测试。进行其他必要的更改以适应新的内部表示。请注意，您不需要修改任何测试来使其通过，因为它们只访问公共接口，这意味着我们可以更改内部结构而不影响客户端。

1.  通过在`CreateTranslationMatrix()`中使用`std::move`强制调用移动构造函数来触发另一个失败。在`Matrix3d`类中引入所需的移动操作以使测试能够编译并通过。

1.  重复步骤 3 到 4，针对`Point3d`。

在实现上述步骤后，预期的输出看起来与开始时没有变化：

![图 3.34：成功转换为使用 RAII 后的活动 1 输出](img/C14583_03_34.jpg)

###### 图 3.34：成功转换为使用 RAII 后的活动 1 输出

#### 注意

此活动的解决方案可以在第 657 页找到。

### 何时调用函数？

C++程序执行的所有操作本质上都是函数调用（尽管编译器可能会将这些优化为内联操作序列）。但是，由于`a = 2 + 5`，你可能不会意识到自己在进行函数调用，实际上你在调用`operator=(&a, operator+(2, 5))`。只是语言允许我们写第一种形式，但第二种形式允许我们重载运算符并将这些功能扩展到用户定义的类型。

以下机制会导致对函数的调用：

+   显式调用函数。

+   所有运算符，如+，-，*，/，%，以及 new/delete。

+   变量的声明-如果存在初始化值，则会导致对带有参数的构造函数的调用。

+   用户定义的字面量-我们还没有处理这些，但基本上，我们为`type operator "" name(argument)`定义了一个重载。然后我们可以写诸如 10_km 之类的东西，这样可以使我们的代码更容易理解，因为它携带了语义信息。

+   从一个值转换为另一个值（`static_cast<>`，`const_cast<>`，`reinterpret_cast<>`和`dynamic_cast<>`）。再次，我们有另一个运算符重载，允许我们将一种类型转换为另一种类型。

+   在函数重载期间，可能需要将一种类型转换为另一种类型，以使其与函数原型匹配。它可以通过调用具有正确参数类型的构造函数来创建临时对象，或者通过隐式调用的转换运算符来实现。

每一个结果都会让编译器确定必须调用一个函数。确定需要调用一个函数后，必须找到与名称和参数匹配的函数。这是我们将在下一节讨论的内容。

### 调用哪个函数

在*第 2A 章*，*不允许鸭子 - 类型和推断*中，我们看到函数重载解析是按以下方式执行的：

![图 3.35：函数重载解析](img/C14583_03_35.jpg)

###### 图 3.35：函数重载解析

我们真正没有深入研究的是名称查找的概念。在某个时刻，编译器将遇到对`func`函数的以下调用：

```cpp
func(a, b);
```

当这种情况发生时，它必须将其名称与引入它的声明关联起来。这个过程称为**名称查找**。这种名称查找对程序中的所有项目（变量、命名空间、类、函数、函数模板和模板）都适用。为了使程序编译通过，变量、命名空间和类的名称查找过程必须产生一个单一的声明。然而，对于函数和函数模板，编译器可以将多个声明与相同的名称关联起来 - 主要是通过函数重载，可以通过**参数依赖查找**（**ADL**）考虑到额外的函数。

### 标识符

根据 C++标准的定义，**标识符**是一系列大写和小写拉丁字母、数字、下划线和大多数 Unicode 字符。有效的标识符必须以非数字字符开头，长度任意长且区分大小写。每个字符都是有意义的。

### 名称

**名称**用于引用实体或标签。名称可以是以下形式之一：

+   标识符

+   函数符号重载的运算符名称（例如 operator-，operator delete）

+   模板名称后跟其参数列表（vector<int>）

+   用户定义的转换函数名称（operator float）

+   用户定义的字面量运算符名称（operator ""_ms）

每个实体及其名称都是由声明引入的，而标签的名称是由**goto**语句或标记语句引入的。一个名称可以在一个文件（或翻译单元）中多次使用，以依赖于作用域而引用不同的实体。一个名称也可以用来引用跨多个文件（翻译单元）相同的实体，或者根据链接性引用不同的实体。编译器使用名称查找通过**名称查找**将引入名称的声明与程序中的未知名称关联起来。

### 名称查找

名称查找过程是两种之一，并且是根据上下文选择的：

+   `::`，或者可能在`::`之后，跟着`template`关键字。限定名可以指代命名空间成员、类成员或枚举器。`::`运算符左边的名称定义了要从中查找名称的作用域。如果没有名称，那么就使用全局命名空间。

+   **未经限定名称查找**：其他所有情况。在这种情况下，名称查找检查当前作用域和所有封闭作用域。

如果未经限定的名称位于函数调用运算符'`()`'的左侧，则使用参数依赖查找。

### 依赖参数的查找

查找未经限定的函数名的规则集称为“参数依赖查找”（简称 ADL），或者“Koenig 查找”（以 Andrew Koenig 命名，他定义了它，并且是 C++标准委员会的资深成员）。未经限定的函数名可以出现为函数调用表达式，也可以作为对重载运算符的隐式函数调用的一部分。

ADL 基本上表示，在未经限定名称查找期间考虑的作用域和命名空间之外，还考虑所有参数和模板参数的“关联命名空间”。考虑以下代码：

```cpp
#include <iostream>
#include <string>
int main()
{
    std::string welcome{"Hello there"};
    std::cout << welcome;
    endl(std::cout);
}
```

当我们编译这段代码并运行它时，输出结果如预期的那样：

```cpp
$ ./adl.exe
Hello there
$
```

这是一种不寻常的编写程序的方式。通常，它会被这样编写：

```cpp
#include <iostream>
#include <string>
int main()
{
    std::string welcome{"Hello there"};
    std::cout << welcome << std::endl;
}
```

我们使用调用`endl()`来展示 ADL 的奇怪方法。但是这里发生了两次 ADL 查找。

第一个经历 ADL 的函数调用是`std::cout << welcome`，编译器认为这是`operator<<(std::cout, welcome)`。现在，操作符<<在可用范围和其参数的命名空间`std`中被查找。这个额外的命名空间将名称解析为自由方法，即在字符串头文件中声明的`std::operator<<(ostream& os, string& s)`。

第二个调用更明显`endl(std::cout)`。同样，编译器可以访问 std 命名空间来解析这个名称查找，并在头文件`ostream`（包含在`iostream`中）中找到`std::endl`模板。

没有 ADL，编译器无法找到这两个函数，因为它们是由 iostream 和 string 包提供的自由函数。插入操作符（<<）的魔力将会丢失，如果我们被迫写`std::operator<<(std::cout, welcome)`，对程序员来说将会很繁琐。如果考虑到链式插入，情况会更糟。或者，您可以写"`using namespace std;`"。这两种选项都不理想，这就是为什么我们需要 ADL（Koenig 查找）。

### 买家当心

我们已经看到 ADL 通过包含与函数参数类型相关的命名空间，使程序员的生活更加轻松。然而，这种查找能力并非没有风险，大部分情况下我们可以将风险降到最低。考虑以下示例代码：

```cpp
#include <iostream>
namespace mylib 
{
void is_substring(std::string superstring, std::string substring)
{
    std::cout << "mylib::is_substring()\n";
}
void contains(std::string superstring, const char* substring) {
    is_substring(superstring, substring);
}
}
int main() {
    mylib::contains("Really long reference", "included");
}
```

当我们编译和运行上述程序时，我们得到了预期的输出：

![图 3.36：ADL 示例程序输出](img/C14583_03_36.jpg)

###### 图 3.36：ADL 示例程序输出

C++标准委员会随后决定引入一个`is_substring()`函数，看起来像这样：

```cpp
namespace std {
void is_substring(std::string superstring, const char* substring)
{
    std::cout << "std::is_substring()\n";
}
}
```

如果我们将其添加到文件顶部，编译并重新运行，现在我们得到以下输出：

![图 3.37：ADL 问题程序输出](img/C14583_03_37.jpg)

###### 图 3.37：ADL 问题程序输出

由于 ADL，（下一个 C++标准）编译器选择了不同的实现作为`is_substring()`的未限定函数调用的更好选择。并且由于参数的隐式转换，它不会导致歧义和编译器错误。它只是悄悄地采用了新的方法，这可能会导致细微且难以发现的错误，如果参数顺序不同。编译器只能检测类型和语法差异，而不能检测语义差异。

#### 注意

为了演示 ADL 的工作原理，我们已将我们的函数添加到 std 命名空间中。命名空间有一个分离关注点的目的，特别是添加到别人的命名空间，特别是`标准库命名空间`（`std`）是不好的做法。

那么，为什么要买家注意（买家当心）？如果您在开发中使用第三方库（包括 C++标准库），那么当您升级库时，您需要确保接口的更改不会因为 ADL 而导致问题。

### 练习 5：实现模板以防止 ADL 问题

在这个练习中，我们将演示 C++17 STL 中的一个破坏性变化，这可能会在实际中引起问题。C++11 引入了`std::begin(type)`和`std::end(type)`的模板。作为开发人员，这是一种对通用接口的吸引人的表达，您可能已经为 size(type)和 empty(type)编写了自己的版本。按照以下步骤实现这个练习：

1.  在 Eclipse 中打开**Lesson3**项目。然后在**Project Explorer**中展开**Lesson3**，然后**Exercise05**，双击**Exercise5.cpp**以将此练习的文件打开到编辑器中。

1.  单击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。从搜索项目菜单配置**L3Exercise5**应用程序，以便以**L3Exercise5**的名称运行。

1.  单击**Run**按钮运行 Exercise 5。这将产生以下输出：![图 3:38：Exercise 5 成功执行](img/C14583_03_38.jpg)

###### 图 3:38：练习 5 的成功执行

1.  代码检查发现了两个辅助模板：

```cpp
template<class T>
bool empty(const T& x)
{
    return x.empty();
}
template<class T>
int size(const T& x)
{
    return x.size();
}
```

1.  与所有其他练习不同，此练习已配置为在 C++ 14 下构建。打开**Lesson3**下的**CMakeLists.txt**文件，并找到以下行：

```cpp
set_property(TARGET L3Exercise5 PROPERTY CXX_STANDARD 14)
```

1.  将`14`改为`17`。

1.  单击**Run**按钮编译练习，现在失败：![图 3.39：C++ 17 下编译失败-模棱两可的函数调用](img/C14583_03_39.jpg)

###### 图 3.39：C++ 17 下编译失败-模棱两可的函数调用

1.  因为`empty()`和`size()`模板的参数是 std::vector，ADL 引入了新包含的 STL 版本的这些模板，破坏了我们的代码。

1.  在`empty()`和两个生成错误的`size()`出现之前，在它们（作用域限定符）之前插入两个冒号“`::`”。

1.  单击`empty()`和`size()`函数现在已经有了限定。我们也可以指定`std::`作用域。

在这个练习中，我们在全局命名空间中实现了两个模板函数，如果我们在 C++ 14 标准下编译程序，它们就可以正常工作。然而，当我们在 C++17 下编译时，我们的实现就会出问题，因为 STL 库发生了变化，我们必须改变我们的实现，以确保编译器定位并使用我们编写的模板。

### 隐式转换

在确定*图 3.36*中的函数候选集时，编译器必须查看所有在名称查找期间找到的可用函数，并确定参数数量和类型是否匹配调用点。在确定类型是否匹配时，它还将检查所有可用的转换，以确定是否有一种机制可以将类型 T1 类型（传递的参数类型）转换为 T2 类型（函数参数指定的类型）。如果它可以将所有参数从 T1 转换为 T2，那么它将把函数添加到候选集中。

从类型 T1 到类型 T2 的这种转换被称为**隐式转换**，当某种类型 T1 在不接受该类型但接受其他类型 T2 的表达式或上下文中使用时发生。这发生在以下情境中：

+   T1 作为参数传递时调用以 T2 为参数声明的函数。

+   T1 用作期望 T2 的运算符的操作数。

+   T1 用于初始化 T2 的新对象（包括返回语句）。

+   T1 在`switch`语句中使用（在这种情况下，T2 是 int）。

+   T1 在`if`语句或`do-while`或`while`循环中使用（其中 T2 为 bool）。

如果存在从 T1 到 2 的明确转换序列，则程序将编译。内置类型之间的转换通常由通常的算术转换确定。

### 显式-防止隐式转换

隐式转换是一个很好的特性，使程序员能够表达他们的意图，并且大多数时候都能正常工作。然而，编译器在程序员没有提供提示的情况下将一种类型转换为另一种类型的能力并不总是理想的。考虑以下小程序：

```cpp
#include <iostream>
class Real
{
public:
    Real(double value) : m_value{value} {}
    operator float() {return m_value;}
    float getValue() const {return m_value;}
private:
    double m_value {0.0};
};
void test(bool result)
{
    std::cout << std::boolalpha;
    std::cout << "Test => " << result << "\n";
}
int main()
{
    Real real{3.14159};
    test(real);
    if ( real ) 
    {
        std::cout << "true: " << real.getValue() << "\n";
    }
    else
    {
        std::cout << "false: " << real.getValue() << "\n";
    }
}
```

当我们编译并运行上述程序时，我们得到以下输出：

![图 3.40：隐式转换示例程序输出](img/C14583_03_40.jpg)

###### 图 3.40：隐式转换示例程序输出

嗯，这可能有点出乎意料，它编译并实际产生了输出。`real`变量是`Real`类型，它有一个到 float 的转换运算符- `operator float()`。`test()`函数以`bool`作为参数，并且`if`条件也必须产生一个`bool`。如果值为零，则编译器将任何数值类型转换为值为 false 的`boolean`类型，如果值不为零，则转换为 true。但是，如果这不是我们想要的行为，我们可以通过在函数声明前加上 explicit 关键字来阻止它。假设我们更改行，使其读起来像这样：

```cpp
explicit operator float() {return m_value;}
```

如果我们现在尝试编译它，我们会得到两个错误：

![图 3.41：因为隐式转换被移除而导致的编译错误。](img/C14583_03_41.jpg)

###### 图 3.41：因为隐式转换被移除而导致的编译错误。

两者都与无法将 Real 类型转换为 bool 有关 - 首先是对`test()`的调用位置，然后是 if 条件中。

现在，让我们引入一个 bool 转换操作符来解决这个问题。

```cpp
operator bool() {return m_value == 0.0;}
```

现在我们可以再次构建程序。我们将收到以下输出：

![图 3.42：引入 bool 运算符替换隐式转换](img/C14583_03_42.jpg)

###### 图 3.42：引入 bool 运算符替换隐式转换

`boolean`值现在为 false，而以前为 true。这是因为浮点转换返回的值的隐式转换不为零，然后转换为 true。

自 C++ 11 以来，所有构造函数（除了复制和移动构造函数）都被认为是转换构造函数。这意味着如果它们没有声明为显式，则它们可用于隐式转换。同样，任何未声明为显式的转换操作符都可用于隐式转换。

`C++核心指南`有两条与隐式转换相关的规则：

+   **C.46**：默认情况下，将单参数构造函数声明为显式

+   **C.164**：避免隐式转换操作符

### 上下文转换

如果我们现在对我们的小程序进行进一步的更改，我们就可以进入所谓的上下文转换。让我们将 bool 运算符设置为显式，并尝试编译程序：

```cpp
explicit operator bool() {return m_value == 0.0;}
```

我们将收到以下输出：

![图 3.43：使用显式 bool 运算符的编译错误](img/C14583_03_43.jpg)

###### 图 3.43：使用显式 bool 运算符的编译错误

这次我们只有一个错误，即对`test()`的调用位置，但对 if 条件没有错误。我们可以通过使用 C 风格的转换（bool）或 C++ `static_cast<bool>(real)`（这是首选方法）来修复此错误。当我们添加转换时，程序再次编译和运行。

因此，如果 bool 转换是显式的，那么为什么 if 表达式的条件不需要转换？

C++标准允许在某些情况下，如果期望`bool`类型并且存在 bool 转换的声明（无论是否标记为显式），则允许隐式转换。这被称为**上下文转换为 bool**，并且可以出现在以下上下文中：

+   `if`、`while`、`for`的条件（或控制表达式）

+   内置逻辑运算符的操作数：`!`（非）、`&&`（与）和`||`（或）

+   条件（或条件）运算符`?:`的第一个操作数。

### 练习 6：隐式和显式转换

在这个练习中，我们将尝试调用函数、隐式转换、阻止它们以及启用它们。按照以下步骤实施这个练习：

1.  在 Eclipse 中打开**Lesson3**项目。然后在**Project Explorer**中展开**Lesson3**，然后展开**Exercise06**，双击**Exercise6.cpp**以在编辑器中打开此练习的文件。

1.  单击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。从**Search Project**菜单中配置**L3Exercise6**应用程序，以便以**L3Exercise6**的名称运行。

1.  单击**Run**按钮运行练习 6。这将产生以下输出：![图 3.44：练习 6 的默认输出](img/C14583_03_44.jpg)

###### 图 3.44：练习 6 的默认输出

1.  在文本编辑器中，将`Voltage`的构造函数更改为`explicit`：

```cpp
struct Voltage
{
    explicit Voltage(float emf) : m_emf(emf) 
    {
    }
    float m_emf;
};
```

1.  单击**Run**按钮重新编译代码 - 现在我们得到以下错误：![图 3.45：int 转换为 Voltage 失败](img/C14583_03_45.jpg)

###### 图 3.45：int 转换为 Voltage 失败

1.  从构造函数中删除显式，并将`calculate`函数更改为引用：

```cpp
void calculate(Voltage& v)
```

1.  单击**Run**按钮重新编译代码 - 现在，我们得到以下错误：![](img/C14583_03_46.jpg)

###### 图 3.46：将整数转换为电压&失败

同一行出现了我们之前遇到的问题，但原因不同。因此，*隐式转换仅适用于值类型*。

1.  注释掉生成错误的行，然后在调用`use_float(42)`之后，添加以下行：

```cpp
use_float(volts);
```

1.  单击**Run**按钮重新编译代码-现在我们得到以下错误：![图 3.47：电压转换为浮点数失败](img/C14583_03_47.jpg)

###### 图 3.47：电压转换为浮点数失败

1.  现在，将以下转换运算符添加到`Voltage`类中：

```cpp
operator float() const
{
    return m_emf;
}
```

1.  单击**Run**按钮重新编译代码并运行它：![图 3.48：成功将电压转换为浮点数](img/C14583_03_48.jpg)

###### 图 3.48：成功将电压转换为浮点数

1.  现在，在我们刚刚添加的转换前面放置`explicit`关键字，然后单击**Run**按钮重新编译代码。再次出现错误：![图 3.49：无法将电压转换为浮点数](img/C14583_03_49.jpg)

###### 图 3.49：无法将电压转换为浮点数

1.  通过在转换中添加显式声明，我们可以防止编译器使用转换运算符。将出错的行更改为将电压变量转换为浮点数：

```cpp
use_float(static_cast<float>(volts));
```

1.  单击**Run**按钮重新编译代码并运行它。

![图 3.50：使用转换将电压转换为浮点数再次成功](img/C14583_03_50.jpg)

###### 图 3.50：使用转换将电压转换为浮点数再次成功

在这个练习中，我们已经看到了类型（而不是引用）之间可以发生隐式转换，并且我们可以控制它们何时发生。现在我们知道如何控制这些转换，我们可以努力满足先前引用的指南`C.46`和`C.164`。

### 活动 2：实现日期计算的类

您的团队负责开发一个库，以帮助处理与日期相关的计算。特别是，我们希望能够确定两个日期之间的天数，并且给定一个日期，添加（或从中减去）一定数量的天数以获得一个新日期。此活动将开发两种新类型并增强它们，以确保程序员不能意外地使它们与内置类型交互。按照以下步骤来实现这一点：

1.  设计和实现一个`Date`类，将`day`、`month`和`year`作为整数存储。

1.  添加方法来访问内部的天、月和年值。

1.  定义一个类型`date_t`来表示自 1970 年 1 月 1 日`纪元日期`以来的天数。

1.  向`Date`类添加一个方法，将其转换为`date_t`。

1.  添加一个方法来从`date_t`值设置`Date`类。

1.  创建一个存储天数值的`Days`类。

1.  为`Date`添加一个接受`Days`作为参数的`加法`运算符。

1.  使用`explicit`来防止数字的相加。

1.  添加`减法`运算符以从两个`日期`的`差异`返回`Days`值。

在按照这些步骤之后，您应该收到以下输出：

![图 3.51：成功的日期示例应用程序输出](img/C14583_03_51.jpg)

###### 图 3.51：成功的日期示例应用程序输出

#### 注意

此活动的解决方案可在第 664 页找到。

## 总结

在本章中，我们探讨了变量的生命周期 - 包括自动变量和动态变量，它们存储在何处，以及它们何时被销毁。然后，我们利用这些信息开发了`RAII`技术，使我们几乎可以忽略资源管理，因为自动变量在被销毁时会清理它们，即使在出现异常的情况下也是如此。然后，我们研究了抛出异常和捕获异常，以便我们可以在正确的级别处理异常情况。从`RAII`开始，我们进入了关于资源所有权的讨论，以及`STL`智能指针如何帮助我们在这个领域。我们发现几乎所有东西都被视为函数调用，从而允许操作符重载和隐式转换。我们发现了“参数相关查找”（`ADL`）的奇妙（或者说糟糕？）世界，以及它如何潜在地在未来使我们陷入困境。我们现在对 C++的基本特性有了很好的理解。在下一章中，我们将开始探讨函数对象以及它们如何使用 lambda 函数实现和实现。我们将进一步深入研究 STL 的功能，并在重新访问封装时探索 PIMPLs。
