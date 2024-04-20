# 2A. 禁止鸭子 - 类型和推断

## 学习目标

通过本章结束时，您将能够：

+   实现自己的类，使其行为类似于内置类型

+   实现控制编译器创建的函数的类（零规则/五规则）

+   使用 auto 变量开发函数，就像你一直做的那样

+   通过使用强类型编写更安全的代码来实现类和函数

本章将为您提供对 C++类型系统的良好基础，并使您能够编写适用于该系统的自己的类型。

## 引言

C++是一种强类型、静态类型的语言。编译器使用与使用的变量相关的类型信息以及它们所用的上下文来检测和防止某些类别的编程错误。这意味着每个对象都有一个类型，而且该类型永远不会改变。相比之下，Python 和 PHP 等动态类型语言将类型检查推迟到运行时（也称为后期绑定），变量的类型可能在应用程序执行过程中发生变化。这些语言使用鸭子测试而不是变量类型 - 也就是说，“如果它走起来像鸭子，叫起来像鸭子，那么它一定是鸭子。”C++等静态类型语言依赖于类型来确定变量是否可以用于特定目的，而动态类型语言依赖于某些方法和属性的存在来确定其适用性。

C++最初被描述为“带类的 C”。这是什么意思？基本上，C 提供了一组内置的基本类型 - int、float、char 等 - 以及这些项的指针和数组。您可以使用 struct 将这些聚合成相关项的数据结构。C++将此扩展到类，以便您可以完全定义自己的类型，包括可以用来操作它们的运算符，从而使它们成为语言中的一等公民。自其谦卑的开始以来，C++已经发展成为不仅仅是“带类的 C”，因为它现在可以表达面向对象范式（封装、多态、抽象和继承）、函数范式和泛型编程（模板）。

在本书中，我们将重点关注 C++支持面向对象范式的含义。随着您作为开发人员的经验增长，并且接触到像 Clojure、Haskell、Lisp 和其他函数式语言，它们将帮助您编写健壮的 C++代码。动态类型语言如 Python、PHP 和 Ruby 已经影响了我们编写 C++代码的方式。随着 C++17 的到来，引入了`std::variant`类 - 一个在编译时保存我们选择的任何类型，并且在动态语言中的变量类似。

在上一章中，我们学习了如何使用 CMake 创建可移植和可维护的 C++项目。我们学习了如何在项目中加入单元测试，以帮助编写正确的代码，并在出现问题时进行调试。我们了解了工具链如何将我们的代码通过一系列程序流水线处理，以生成可执行文件。最后，我们总结了一些经验法则，帮助我们创建可读性强、理解性好、易于维护的代码。

在本章中，我们将快速浏览 C++类型系统，声明和使用我们自己的类型。

## C++类型

作为一种强类型和静态类型的语言，C++提供了几种基本类型，并能够根据需要定义自己的类型，以解决手头的问题。本节将首先介绍基本类型，初始化它们，声明变量，并将类型与之关联。然后我们将探讨如何声明和定义新类型。

### C++基本类型

C++包括几种*基本类型*或*内置类型*。C++标准定义了每种类型在内存中的最小大小和它们的相对大小。编译器识别这些基本类型，并具有内置规则来定义可以对它们执行哪些操作和不能执行哪些操作。还有关于类型之间的隐式转换的规则；例如，从 int 类型到 float 类型的转换。

#### 注意

有关所有内置类型的简要描述，请参阅[`en.cppreference.com/w/cpp/language/types`](https://en.cppreference.com/w/cpp/language/types)中的**基本类型**部分。

### C++文字量

C++文字量用于告诉编译器您希望在声明变量或对其进行赋值时与变量关联的值。前一节中的每种内置类型都有与之关联的文字量形式。

#### 注意

有关每种类型的文字量的简要描述，请参阅[`en.cppreference.com/w/cpp/language/expressions`](https://en.cppreference.com/w/cpp/language/expressions)中的**文字量**部分。

## 指定类型 - 变量

由于 C++是一种静态类型语言，在声明变量时需要指定变量的类型。当声明函数时，需要指定返回类型和传递给它的参数的类型。在声明变量时，有两种选择可以指定类型：

+   **显式**：您作为程序员正在明确指定类型。

+   **隐式**（使用 auto）：您告诉编译器查看用于初始化变量的值并确定其类型。这被称为（auto）**类型推导**。

标量变量的声明一般形式如下之一：

```cpp
type-specifier var;                       // 1\. Default-initialized variable
type-specifier var = init-value;          // 2\. Assignment initialized variable
type-specifier var{init-value};           // 3\. Brace-initialize variable
```

`type-specifier`指示您希望将`var`变量与之关联的类型（基本类型或用户定义类型）。所有三种形式都会导致编译器分配一些存储空间来保存值，并且将来对`var`的所有引用都将引用该位置。`init-value`用于初始化存储位置。默认初始化对内置类型无效，并将根据函数重载解析调用用户定义类型的构造函数来初始化存储。

编译器必须知道要分配多少内存，并提供一个运算符来确定类型或变量有多大 - `sizeof`。

根据我们的声明，编译器将在计算机的内存中留出空间来存储变量引用的数据项。考虑以下声明：

```cpp
int value = 42;     // declare value to be an integer and initialize to 42
short a_value{64};  // declare a_value to be a short integer and initialize
                    //    to 64
int bad_idea;       // declare bad_idea to be an integer and DO NOT 
                    // initialize it. Use of this variable before setting
                    // it is UNDEFINED BEHAVIOUR.
float pi = 3.1415F; // declare pi to be a single precision floating point
                    // number and initialize it to pi.
double e{2.71828};  // declare e to be a double precision floating point
                    // number and initialize it to natural number e.
auto title = "Sir Robin of Loxley"; // Let the compiler determine the type
```

如果这些是在函数范围内声明的，那么编译器会从所谓的堆栈中为它们分配内存。这可能看起来像以下的内存布局：

![图 2A.1：变量的内存布局](img/C14583_02A_01.jpg)

###### 图 2A.1：变量的内存布局

编译器将按照我们声明变量的顺序分配内存。未使用的内存是因为编译器分配内存，以便基本类型通常是原子访问的，并且为了效率而对齐到适当的内存边界。请注意，`title`是`const char *`类型，是`const`。**"Sir Robin of Loxley"**字符串将存储在程序加载时初始化的内存的不同部分。我们将在后面讨论程序内存。

标量声明语法的轻微修改给我们提供了声明值数组的语法：

```cpp
type-specifier ary[count];                          // 1\. Default-initialized 
type-specifier ary[count] = {comma-separated list}; // 2\. Assignment initialized 
type-specifier ary[count]{comma-separated list};    // 3\. Brace-initialized
```

这可以用于多维数组，如下所示：

```cpp
type-specifier ary2d[countX][countY]; 
type-specifier ary3d[countX][countY][countZ];
// etc...
```

请注意，前述声明中的`count`、`countX`和其他项目在编译时必须评估为常量，否则将导致错误。此外，逗号分隔的初始化列表中的项目数必须小于或等于`count`，否则将再次出现编译错误。在下一节中，我们将在练习中应用到目前为止学到的概念。

#### 注意

在本章的任何实际操作之前，下载本书的 GitHub 存储库（[`github.com/TrainingByPackt/Advanced-CPlusPlus`](https://github.com/TrainingByPackt/Advanced-CPlusPlus)），并在 Eclipse 中导入 Lesson 2A 文件夹，以便您可以查看每个练习和活动的代码。

### 练习 1：声明变量和探索大小

这个练习将为本章的所有练习设置，并让您熟悉声明和初始化内置类型的变量。您还将介绍`auto 声明`，`数组`和`sizeof`。让我们开始吧：

1.  打开 Eclipse（在*第一章* *可移植 C++软件的解剖*中使用），如果出现启动窗口，请点击启动。

1.  转到**File**，在**New** **►**下选择**Project…**，然后转到选择 C++ Project（而不是 C/C++ Project）。

1.  点击**Next >**，清除**Use default location**复选框，并输入**Lesson2A**作为**Project name**。

1.  选择**Empty Project**作为**Project Type**。然后，点击**Browse…**并导航到包含 Lesson2A 示例的文件夹。

1.  点击**打开**以选择文件夹并关闭对话框。

1.  点击**Next >**，**Next >**，然后点击**Finish**。

1.  为了帮助您进行练习，我们将配置工作区在构建之前自动保存文件。转到**Window**，选择**Preferences**。在**General**下，打开**Workspace**并选择**Build**。

1.  勾选**Save automatically before build**框，然后点击**Apply and Close**。

1.  就像*第一章* *可移植 C++软件的解剖*一样，这是一个基于 CMake 的项目，所以我们需要更改当前的构建器。在**Project**资源管理器中点击**Lesson2A**，然后在**Project**菜单下点击**Properties**。在左侧窗格中选择 C/C++ Build 下的 Tool Chain Editor，并将 Current builder 设置为 Cmake Build（portable）。

1.  点击**Apply and Close**。然后，选择**Project** | **Build All**菜单项来构建所有练习。默认情况下，屏幕底部的控制台将显示**CMake Console [Lesson2A]**：![图 2A.2：CMake 控制台输出](img/C14583_02A_02.jpg)

###### 图 2A.2：CMake 控制台输出

1.  在控制台的右上角，点击**Display Selected Console**按钮，然后从列表中选择**CDT Global Build Console**：![图 2A.3：选择不同的控制台](img/C14583_02A_03.jpg)

###### 图 2A.3：选择不同的控制台

这将显示构建的结果 - 应该显示 0 个错误和 3 个警告：

![图 2A.4：构建过程控制台输出](img/C14583_02A_04.jpg)

###### 图 2A.4：构建过程控制台输出

1.  由于构建成功，我们希望运行 Exercise1。在窗口顶部，点击下拉列表，选择**No Launch Configurations**：![图 2A.5：启动配置菜单](img/C14583_02A_05.jpg)

###### 图 2A.5：启动配置菜单

1.  点击**New Launch Configuration…**。保持默认设置，然后点击**Next >**。

1.  将**Name**更改为**Exercise1**，然后点击**Search Project**：![图 2A.6：Exercise1 启动配置](img/C14583_02A_06.jpg)

###### 图 2A.6：Exercise1 启动配置

1.  从 Binaries 窗口中显示的程序列表中，点击**Exercise1**，然后点击**OK**。

1.  点击**Finish**。这将导致 exercise1 显示在启动配置下拉框中：![图 2A.7：更改启动配置](img/C14583_02A_07.jpg)

###### 图 2A.7：更改启动配置

1.  要运行**Exercise1**，点击**Run**按钮。Exercise1 将在控制台中执行并显示其输出：![图 2A.8：exercise1 的输出](img/C14583_02A_08.jpg)

###### 图 2A.8：exercise1 的输出

该程序没有任何价值 - 它只输出系统中各种类型的大小。但这表明程序是有效的并且可以编译。请注意，您系统的数字可能会有所不同（特别是 sizeof（title）的值）。

1.  在“项目资源管理器”中，展开“Lesson2A”，然后展开“Exercise01”，双击“Exercise1.cpp”以在编辑器中打开此练习的文件：

```cpp
int main(int argc, char**argv)
{
    std::cout << "\n\n------ Exercise 1 ------\n";
    int value = 42;     // declare value to be an integer & initialize to 42
    short a_value{64};  // declare a_value to be a short integer & 
                        // initialize to 64
    int bad_idea;       // declare bad_idea to be an integer and DO NOT 
                        // initialize it. Use of this variable before 
                        // setting it is UNDEFINED BEHAVIOUR.
    float pi = 3.1415F; // declare pi to be a single precision floating 
                        // point number and initialize it to pi.

    double e{2.71828};  // declare e to be a double precision floating point
                        // number and initialize it to natural number e.
    auto title = "Sir Robin of Loxley"; 
                        // Let the compiler determine the type
    int ary[15]{};      // array of 15 integers - zero initialized
    // double pi = 3.14159;  // step 24 - remove comment at front
    // auto speed;           // step 25 - remove comment at front
    // value = "Hello world";// step 26 - remove comment at front
    // title = 123456789;    // step 27 - remove comment at front
    // short sh_int{32768};  // step 28 - remove comment at front
    std::cout << "sizeof(int) = " << sizeof(int) << "\n";
    std::cout << "sizeof(short) = " << sizeof(short) << "\n";
    std::cout << "sizeof(float) = " << sizeof(float) << "\n";
    std::cout << "sizeof(double) = " << sizeof(double) << "\n";
    std::cout << "sizeof(title) = " << sizeof(title) << "\n";
    std::cout << "sizeof(ary) = " << sizeof(ary)
              << " = " << sizeof(ary)/sizeof(ary[0]) 
              << " * " << sizeof(ary[0]) << "\n";
    std::cout << "Complete.\n";
    return 0;
}
```

关于前面的程序，需要注意的一点是，主函数的第一条语句实际上是可执行语句，而不是声明。 C++允许您几乎可以在任何地方声明变量。 它的前身 C 最初要求所有变量必须在任何可执行语句之前声明。

#### 最佳实践

尽可能靠近将要使用的位置声明变量并初始化它。

1.  在编辑器中，通过删除行开头的分隔符（//）取消注释标记为“步骤 24”的行：

```cpp
double pi = 3.14159;  // step 24 - remove comment at front    
// auto speed;           // step 25 - remove comment at front
// value = "Hello world";// step 26 - remove comment at front
// title = 123456789;    // step 27 - remove comment at front
// short sh_int{32768};  // step 28 - remove comment at front
```

1.  再次单击“运行”按钮。 这将导致再次构建程序。 这一次，构建将失败，并显示错误：![图 2A.9：工作区中的错误对话框](img/C14583_02A_09.jpg)

###### 图 2A.9：工作区中的错误对话框

1.  单击“取消”关闭对话框。 如果未显示“CDT 构建控制台[Lesson2A]”，则将其选择为活动控制台：![图 2A.10：重复声明错误](img/C14583_02A_10.jpg)

###### 图 2A.10：重复声明错误

这一次，构建失败，因为我们尝试重新定义变量 pi 的类型。 编译器提供了有关我们需要查找以修复错误的位置的有用信息。

1.  将注释分隔符恢复到行的开头。 在编辑器中，通过删除行开头的分隔符（//）取消注释标记为“步骤 25”的行：

```cpp
// double pi = 3.14159;  // step 24 - remove comment at front    
auto speed;           // step 25 - remove comment at front
// value = "Hello world";// step 26 - remove comment at front
// title = 123456789;    // step 27 - remove comment at front
// short sh_int{32768};  // step 28 - remove comment at front
```

1.  再次单击“运行”按钮。 当“工作区中的错误”对话框出现时，单击“取消”：![图 2A.11：自动声明错误-无初始化](img/C14583_02A_11.jpg)

###### 图 2A.11：自动声明错误-无初始化

再次构建失败，但这次我们没有给编译器足够的信息来推断速度的类型-自动类型的变量必须初始化。

1.  将注释分隔符恢复到行的开头。 在编辑器中，通过删除注释起始分隔符（//）取消注释标记为“步骤 26”的行：

```cpp
// double pi = 3.14159;  // step 24 - remove comment at front    
// auto speed;           // step 25 - remove comment at front
value = "Hello world";// step 26 - remove comment at front
// title = 123456789;    // step 27 - remove comment at front
// short sh_int{32768};  // step 28 - remove comment at front
```

1.  单击“值”。

1.  将注释分隔符恢复到行的开头。 在编辑器中，通过删除行开头的分隔符（//）取消注释标记为“步骤 27”的行：

```cpp
// double pi = 3.14159;  // step 24 - remove comment at front    
// auto speed;           // step 25 - remove comment at front
// value = "Hello world";// step 26 - remove comment at front
title = 123456789;    // step 27 - remove comment at front
// short sh_int{32768};  // step 28 - remove comment at front
```

1.  单击`int`，以标题，这是一个`const char*`。 这里非常重要的一点是，`title`是用`auto`类型声明的。 编译器生成的错误消息告诉我们，`title`被推断为`const char*`类型。

1.  将注释分隔符恢复到行的开头。 在编辑器中，通过删除行开头的分隔符（//）取消注释标记为“步骤 28”的行：

```cpp
// double pi = 3.14159;  // step 24 - remove comment at front    
// auto speed;           // step 25 - remove comment at front
// value = "Hello world";// step 26 - remove comment at front
// title = 123456789;    // step 27 - remove comment at front
short sh_int{32768};  // step 28 - remove comment at front
```

1.  单击`sh_int`与（`short`类型。 短占用两个字节的内存，被认为是 16 位的有符号数量。 这意味着可以存储在短中的值的范围是`-2^(16-1)`到`2^(16-1)-1`，或**-32768**到**32767**。

1.  将值从`short`更改。

1.  将值从`short`更改。

1.  将注释分隔符恢复到行的开头。 在编辑器中，尝试使用任何基本类型及其相关文字来探索变量声明，然后尽可能多地单击“运行”按钮。 检查“构建控制台”的输出是否有任何错误消息，因为这可能会帮助您找到错误。

在这个练习中，我们学习了如何设置 Eclipse 开发，实现变量声明，并解决声明中的问题。

## 指定类型-函数

现在我们可以声明一个变量为某种类型，我们需要对这些变量做些什么。 在 C++中，我们通过调用函数来做事情。 函数是一系列语句，产生结果。 结果可能是数学计算（例如，指数）然后发送到文件或写入终端。

函数允许我们将解决方案分解为更易于管理和理解的语句序列。当我们编写这些打包的语句时，我们可以在合适的地方重复使用它们。如果我们需要根据上下文使其以不同方式运行，那么我们会传入一个参数。如果它返回一个结果，那么函数需要一个返回类型。

由于 C++是一种强类型语言，我们需要指定与我们实现的函数相关的类型 - 函数返回的值的类型（包括无返回）以及传递给它的参数的类型（如果有的话）。

以下是一个典型的 hello world 程序：

```cpp
#include <iostream>
void hello_world()
{
  std::cout << "Hello world\n"; 
}
int main(int argc, char** argv)
{
  std::cout << "Starting program\n";
  hello_world();
  std::cout << "Exiting program\n";
  return 0;
}
```

在上面的例子中声明了两个函数 - `hello_world()`和`main()`。`main()`函数是每个 C++程序的入口点，并返回一个传递给主机系统的`int`值。它被称为退出代码。

从返回类型的声明到开括号（{）之间的所有内容都被称为**函数原型**。它定义了三件事，即返回类型、函数的名称和参数的数量和类型。

对于第一个函数，返回类型是`void` - 也就是说，它不返回任何值；它的名称是`hello_world`，不需要参数：

![图 2A.15：不带参数并且不返回任何内容的函数声明](img/C14583_02A_15.jpg)

###### 图 2A.15：不带参数并且不返回任何内容的函数声明

第二个函数返回一个`int`值，名称为`main`，并带有两个参数。这些参数分别是`argc`和`argv`，类型分别为`int`和`char`类型的*指针的指针*：

![图 2A.16：带有两个参数并返回 int 的函数声明](img/C14583_02A_16.jpg)

###### 图 2A.16：带有两个参数并返回 int 的函数声明

函数原型之后的所有内容都被称为**函数体**。函数体包含变量声明和要执行的语句。

函数在使用之前必须声明 - 也就是说，编译器需要知道它的参数和返回类型。如果函数在调用它的文件中定义在它之后，那么可以通过在使用之前提供函数的前向声明来解决这个问题。

通过在调用之前的文件中放置以分号终止的函数原型来进行前向声明。对于`hello_world()`，可以这样做：

```cpp
void hello_world();
```

对于主函数，可以这样做：

```cpp
int main(int, char**);
```

函数原型不需要参数的名称，只需要类型。但是，为了帮助函数的用户，保留参数是个好主意。

在 C++中，函数的定义可以在一个文件中，需要从另一个文件中调用。那么，第二个文件如何知道它希望调用的函数的原型？这是通过将前向声明放入一个名为头文件的单独文件中并在第二个文件中包含它来实现的。

### 练习 2：声明函数

在这个练习中，我们将测试编译器在遇到函数调用时需要了解的内容，并实现一个前向声明来解析未知的函数。让我们开始吧。

1.  在 Eclipse 中打开**Lesson2A**项目，然后在**Project Explorer**中展开**Lesson2A**，然后展开**Exercise02**，双击**Exercise2.cpp**以在编辑器中打开此练习的文件。

1.  单击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。

1.  将**Exercise2**配置为以名称**Exercise2**运行。完成后，它将成为当前选择的启动配置。

1.  单击**Run**按钮。练习 2 将运行并产生以下输出：![图 2A.17：exercise2 程序的输出](img/C14583_02A_17.jpg)

###### 图 2A.17：exercise2 程序的输出

1.  进入编辑器，通过将`gcd`函数移动到`main`之后来更改代码。它应该如下所示：

```cpp
int main(int argc, char**argv)
{
    std::cout << "\n\n------ Exercise 2 ------\n";
    std::cout << "The greatest common divisor of 44 and 121 is " << gcd(44, 121) << "\n";
    std::cout << "Complete.\n";
    return 0;
}
int gcd(int x, int y)
{
    while(y!=0)
    {
        auto c{x%y};
        x = y;
        y = c;
    }
    return x;
}
```

1.  点击`gcd()`函数。在需要调用它的时候，它对该函数没有任何了解，即使它在相同的文件中定义，但是在调用之后。

1.  在编辑器中，将前向声明放在主函数定义之前。同时在末尾添加一个分号（;）：

```cpp
int gcd(int x, int y);
```

1.  再次点击**运行**按钮。这次，程序编译并恢复原始输出。

在这个练习中，我们学习了如何提前声明函数并解决编译器错误，这些错误发生在使用函数之前未声明的情况下。

在早期的 C 编译器版本中，这是可以接受的。程序会假定函数存在并返回一个 int。函数的参数可以从调用中推断出来。然而，在现代 C++中并非如此，因为您必须在使用之前声明函数、类、变量等。在下一节中，我们将学习指针类型。

### 指针类型

由于 C 语言的起源，即编写高效的系统并直接访问硬件，C++允许您将变量声明为指针类型。其格式如下：

```cpp
type-specifier* pvar = &var;
```

这与以前一样，只有两个不同之处：

+   使用特殊声明符星号（`*`）指示名为 pvar 的变量指向内存中的位置或地址。

+   它使用特殊运算符和号（`&`）进行初始化，在这种情况下告诉编译器返回`var`变量的地址。

由于 C 是一种高级语言，但具有低级访问权限，指针允许用户直接访问内存，这在我们希望向硬件提供输入/输出并控制硬件时非常有帮助。指针的另一个用途是允许函数访问共同的数据项，并在调用函数时消除大量数据的复制需求，因为它默认为按值传递。要访问指针指向的值，使用特殊运算符星号（`*`）来**解引用**位置：

```cpp
int five = 5;                // declare five and initialize it
int *pvalue = &five;         // declare pvalue as pointer to int and have it
                            // point to the location of five
*pvalue = 6;                // Assign 6 into the location five.
```

下图显示了编译器分配内存的方式。`pvalue`需要内存来存储指针，而`five`需要内存来存储整数值 5：

![图 2A.19：指针变量的内存布局](img/C14583_02A_19.jpg)

###### 图 2A.19：指针变量的内存布局

当通过指针访问用户定义的类型时，还有第二个特殊运算符（->）用于解引用成员变量和函数。在现代 C++中，这些指针被称为**原始指针**，它们的使用方式发生了显著变化。在 C 和 C++中使用指针一直是程序员面临的挑战，它们的错误使用是许多问题的根源，最常见的是资源泄漏。资源泄漏是指程序获取了资源（内存、文件句柄或其他系统资源）供其使用，但在使用完毕后未释放。这些资源泄漏可能导致性能问题、程序失败，甚至系统崩溃。在现代 C++中使用原始指针来管理资源的所有权现已被弃用，因为智能指针在 C++11 中出现。智能指针（在 STL 中实现为类）现在执行所需的清理工作，以成为主机系统中的良好组成部分。关于这一点将在*第三章*，*能与应该之间的距离-对象、指针和继承*中进行更多介绍。

在上面的代码中，当声明`pvalue`时，编译器只分配内存来存储它将引用的内存的地址。与其他变量一样，您应始终确保在使用指针之前对其进行初始化，因为对未初始化的指针进行解引用会导致未定义的行为。存储指针的内存量取决于编译器设计的系统以及处理器支持的位数。但是，无论它们指向什么类型，所有指针的大小都将相同。

指针也可以传递给函数。这允许函数访问指向的数据并可能修改它。考虑以下 swap 的实现：

```cpp
void swap(int* data1, int* data2)
{
    int temp{*data1};         // Initialize temp from value pointed to by data1
    *data1 = *data2;          // Copy data pointed to by data2 into location 
                              // pointed to by data1
    *data2 = temp;            // Store the temporarily cached value from temp
                              // into the location pointed to by data2
}
```

这展示了如何将指针声明为函数的参数，如何使用解引用运算符`*`从指针获取值，以及如何通过解引用运算符设置值。

以下示例使用 new 运算符从主机系统中分配内存，并使用 delete 运算符将其释放回主机系统：

```cpp
char* name = new char[20];    // Allocate 20 chars worth of memory and assign it
                              // to name.
  Do something with name
delete [] name;
```

在上面的代码中，第一行使用 new 运算符的数组分配形式创建了一个包含 20 个字符的数组。它向主机系统发出调用，为我们分配 20 * sizeof(char)字节的内存。分配多少内存取决于主机系统，但保证至少为 20 * sizeof(char)字节。如果无法分配所需的内存，则会发生以下两种情况之一：

+   它会抛出一个异常

+   它将返回`nullptr`。这是 C++11 中引入的特殊文字。早期，C++使用 0 或 NULL 表示无效指针。C++11 也将其作为强类型值。

在大多数系统上，第一个结果将是结果，并且您需要处理异常。第二个结果可能来自两种情况——调用 new 的 nothrow 变体，即`new(std::nothrow) int [250]`，或者在嵌入式系统上，异常处理的开销不够确定。

最后，请注意，delete 的调用使用了 delete 运算符的数组形式，即带有方括号[]。重要的是确保与 new 和 delete 运算符一起使用相同的形式。当 new 用于用户定义的类型（将在下一节中讨论）时，它不仅仅是分配内存：

```cpp
MyClass* object = new MyClass;
```

在上面的代码中，对 new 的调用分配了足够的内存来存储 MyClass，如果成功，它会继续调用构造函数来初始化数据：

```cpp
MyClass* objects = new MyClass[12];
```

在上面的代码中，对 new 的调用分配了足够的内存来存储 12 个 MyClass 的副本，如果成功，它会继续调用构造函数 12 次来初始化每个对象的数据。

请注意，在上面代码片段中声明的`object`和`objects`，`objects`应该是指向 MyClass 数组的指针，但实际上它是 MyClass 实例的指针。`objects`指向 MyClass 数组中的第一个实例。

考虑以下代码摘录：

```cpp
void printMyClasses(MyClass* objects, size_t number)
{
  for( auto i{0U} ; i<number ; i++ ) { 
    std::cout << objects[i] << "\n";
  }
}
void process()
{
    MyClass objects[12];

    // Do something with objects
    printMyClasses(objects, sizeof(objects)/sizeof(MyClass));
}
```

在 process()函数中，`objects`是"包含 12 个 MyClass 项的数组"类型，但当它传递给`printMyClasses()`时，它被（由编译器）转换为"指向 MyClass 的指针"类型。这是有意设计的（从 C 继承而来），并且被称为`printMyClasses()`如下：

```cpp
void printMyClasses(MyClass objects[12], size_t number)
```

这仍然会受到数组衰减的影响，因为编译器将参数对象更改为 MyClass*；在这种情况下，它不保留维度信息。数组衰减是我们需要将数字传递给`printMyClasses()`函数的原因：这样我们就知道数组中有多少项。C++提供了两种处理数组衰减的机制：

+   使用迭代器将范围传递到方法中。STL 容器（参见*第 2B 章*中的*C++预打包模板*部分，*不允许鸭子-模板和推断*）提供`begin()`和`end()`方法，以便我们可以获得允许算法遍历数组或其部分的迭代器。

#### 注意

对于 C++20，ISO 标准委员会正在考虑包含一种称为 Ranges 的概念，它将允许同时捕获起始和结束迭代器的对象。

+   使用模板（参见*第 2B 章，不允许鸭子-模板和推断*中的*非类型模板参数*部分）。

### 练习 3：声明和使用指针

在这个练习中，我们将实现接受指针和数组作为参数并比较它们的行为，同时考虑数组衰减的函数。让我们开始吧：

1.  在 Eclipse 中打开**Lesson2A**项目，然后在项目资源管理器中展开**Lesson2A**，然后**Exercise03**，双击**Exercise3.cpp**以在编辑器中打开此练习的文件。

1.  点击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。配置**Exercise3**以运行名称**Exercise3**。完成后，它将成为当前选择的 Launch Configuration。

1.  点击**Run**按钮。练习 3 将运行并产生以下输出：![图 2A.20：练习 3 输出](img/C14583_02A_20.jpg)

###### 图 2A.20：练习 3 输出

1.  在编辑器中的某个地方插入一行空行，然后点击**Run**按钮。（通过更改文件，它将强制构建系统重新编译**Exercise3.cpp**。）

1.  如果我们现在看`print_array_size2()`是`int*`类型，并且由警告说明`sizeof`将返回'int*'的大小所证实：![图 2A.22：练习 3 部分输出](img/C14583_02A_22.jpg)

###### 图 2A.22：练习 3 部分输出

`sizeof(ary)/sizeof(arg[0])`的计算应返回数组中的元素数。`elements in (ary) = 10`是从 main 函数生成的，ary 声明为`ary[10]`，所以是正确的。在---print_array_size2---横幅下的`elements in (ary) = 2`显示了数组衰减的问题，以及为什么编译器生成了警告。为什么值是 2？在测试 PC 上，指针占用 8 字节（64 位），而 int 只占用 4 字节，所以我们得到 8/4 = 2。

1.  在编辑器中，找到 main()中声明 ary 的行，并将其更改为以下内容：

```cpp
int ary[15]{};
```

1.  点击`int ary[15]`会导致错误或至少警告，因为参数原型不匹配。正如我们之前所述，编译器将参数视为`int* ary`，因此函数也可以声明如下：

```cpp
void print_array_size2(int* ary)
```

1.  在编辑器中，将`print_array_size2`的名称全部更改为`print_array_size`。点击`int* ary`和`int ary[10]`。这是确认，当作为函数参数使用时，`int ary[10]`生成的结果与声明`int*` ary 时相同。

1.  将文件恢复到其原始状态。

1.  在`main()`函数中，找到带有`Step 11`注释的行，并删除该行开头的注释。点击`title`以使其为`const char*`，p 的类型为`char*`。const 很重要。p 指针允许我们更改其指向的值。

1.  看一下以下行：

```cpp
p = title; 
```

将其更改为以下内容：

```cpp
title = p;
```

1.  点击**Run**按钮。这次，它构建并正确运行。将非 const 指针分配给 const 指针是可以的。

在这个练习中，我们学到了当将数组传递到函数中时，需要小心处理数组，因为关键信息（数组的大小）将在调用中丢失。

## 创建用户类型

C++的伟大之处在于您可以使用**struct**、**class**、**enum**或**union**创建自己的类型，编译器将在整个代码中将其视为基本类型。在本节中，我们将探讨创建自己的类型以及我们需要编写的方法来操纵它，以及编译器将为我们创建的一些方法。

### 枚举

最简单的用户定义类型是枚举。C++11 对枚举进行了改进，使它们更加类型安全，因此我们必须考虑两种不同的声明语法。在看如何声明它们之前，让我们弄清楚为什么需要它们。考虑以下代码：

```cpp
int check_file(const char* name)
{
  FILE* fptr{fopen(name,"r")};
  if ( fptr == nullptr)
    return -1;
  char buffer[120];
  auto numberRead = fread(buffer, 1, 30, fptr);
  fclose(fptr);
  if (numberRead != 30)
    return -2;
  if(is_valid(buffer))
    return -3;
  return 0;
}
```

这是许多 C 库函数的典型特征，其中返回状态代码，您需要主页知道它们的含义。在前述代码中，`-1`、`-2`、`-3`和`0`被称为**魔术数字**。您需要阅读代码以了解每个数字的含义。现在，考虑以下版本的代码：

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

这使用枚举类来传达结果并将含义附加到值的名称上。函数的用户现在可以使用枚举，因为代码更容易理解和使用。因此，魔术数字（与状态相关）已被替换为具有描述性标题的枚举值。让我们通过以下代码片段了解`FileCheckStatus`的声明：

```cpp
enum FileCheckStatus             // Old-style enum declaration
{
  Good,                         // = 0 - Value defaults to 0
  NotFound,                     // = 1 - Value set to one more than previous
  IncorrectSize,                // = 2 - Value set to one more than previous
  InvalidContents,              // = 3 - Value set to one more than previous
};
```

如果我们想使用魔术数字的值，那么我们会这样声明它们：

```cpp
enum FileCheckStatus             // Old-style enum declaration
{
  Good = 0, 
  NotFound = -1,
  IncorrectSize = -2,
  InvalidContents = -3,
};
```

或者，通过改变顺序，我们可以设置第一个值，编译器会完成其余部分：

```cpp
enum FileCheckStatus             // Old-style enum declaration
{
  InvalidContents = -3,          // Force to -3
  IncorrectSize,                 // set to -2(=-3+1)
  NotFound,                      // Set to -1(=-2+1)
  Good,                          // Set to  0(=-1+1)
};
```

前述函数也可以写成如下形式：

```cpp
FileCheckStatus check_file(const char* name)
{
  FILE* fptr{fopen(name,"r")};
  if ( fptr == nullptr)
    return NotFound;
  char buffer[30];
  auto numberRead = fread(buffer, 1, 30, fptr);
  fclose(fptr);
  if (numberRead != 30)
    return IncorrectSize;
  if(is_valid(buffer))
    return InvalidContents;
  return Good;
}
```

请注意，代码中缺少作用域指令`FileCheckStatus::`，但它仍将编译并工作。这引发了作用域的问题，我们将在*第 2B 章*的*可见性、生命周期和访问*部分中详细讨论。现在，知道每种类型和变量都有一个作用域，旧式枚举的问题在于它们的枚举器被添加到与枚举相同的作用域中。假设我们有两个枚举定义如下：

```cpp
enum Result 
{
    Pass,
    Fail,
    Unknown,
};
enum Option
{
    Keep,
    Discard,
    Pass,
    Play
};
```

现在我们有一个问题，`Pass`枚举器被定义两次并具有两个不同的值。旧式枚举还允许我们编写有效的编译器，但显然毫无意义的代码，例如以下代码：

```cpp
Option option{Keep};
Result result{Unknown};
if (option == result)
{
    // Do something
}
```

由于我们试图开发清晰明了的代码，易于理解，将结果与选项进行比较是没有意义的。问题在于编译器会隐式将值转换为整数，从而能够进行比较。

C++11 引入了一个被称为**枚举类**或**作用域枚举**的新概念。前述代码的作用域枚举定义如下：

```cpp
enum class Result 
{
    Pass,
    Fail,
    Unknown,
};
enum class Option
{
    Keep,
    Discard,
    Pass,
    Play
};
```

这意味着前述代码将不再编译：

```cpp
Option option{Keep};          // error: must use scope specifier Option::Keep
Result result{Unknown};       // error: must use scope specifier Result::Unknown
if (option == result)         // error: can no longer compare the different types
{
    // Do something
}
```

正如其名称所示，**作用域枚举**将枚举器放置在枚举名称的作用域内。此外，作用域枚举将不再被隐式转换为整数（因此 if 语句将无法编译通过）。您仍然可以将枚举器转换为整数，但需要进行类型转换：

```cpp
int value = static_cast<int>(Option::Play);
```

### 练习 4：枚举-新旧学校

在这个练习中，我们将实现一个程序，使用枚举来表示预定义的值，并确定当它们更改为作用域枚举时所需的后续更改。让我们开始吧：

1.  在 Eclipse 中打开**Lesson2A**项目，然后在**Project Explorer**中展开**Lesson2A**，然后展开**Exercise04**，双击**Exercise4.cpp**以在编辑器中打开此练习的文件。

1.  单击**启动配置**下拉菜单，然后选择**新建启动配置…**。配置**Exercise4**以使用名称**Exercise4**运行。

1.  完成后，它将成为当前选择的启动配置。

1.  单击**运行**按钮。练习 4 将运行并产生以下输出：![图 2A.25：练习 4 输出](img/C14583_02A_25.jpg)

###### 图 2A.25：练习 4 输出

1.  检查编辑器中的代码。目前，我们可以比较苹果和橙子。在`printOrange()`的定义处，将参数更改为`Orange`：

```cpp
void printOrange(Orange orange)
```

1.  单击**运行**按钮。当出现工作区中的错误对话框时，单击**取消**：![图 2A.26：无法转换错误](img/C14583_02A_26.jpg)

###### 图 2A.26：无法转换错误

通过更改参数类型，我们迫使编译器强制执行传递给函数的值的类型。

1.  通过在初始调用中传递`orange` `enum`变量并在第二次调用中传递`apple`变量，两次调用`printOrange()`函数：

```cpp
printOrange(orange);
printOrange(apple);
```

这表明编译器会将橙色和苹果隐式转换为`int`，以便调用该函数。还要注意关于比较`Apple`和`Orange`的警告。

1.  通过采用 int 参数并将`orange` `enum`的定义更改为以下内容来恢复`printOrange()`函数：

```cpp
enum class Orange;
```

1.  单击**运行**按钮。当出现工作区中的错误对话框时，单击**取消**：![图 2A.27：作用域枚举更改的多个错误](img/C14583_02A_27.jpg)

###### 图 2A.27：作用域枚举更改的多个错误

1.  找到此构建的第一个错误：![图 2A.28：第一个作用域枚举错误](img/C14583_02A_28.jpg)

###### 图 2A.28：第一个作用域枚举错误

1.  关于作用域枚举的第一件事是，当您引用枚举器时，它们必须具有作用域限定符。因此，在编辑器中，转到并更改此行如下：

```cpp
Orange orange{Orange::Hamlin};
```

1.  单击`Orange`类型。因为这涉及基于模板的类（我们稍后会讨论），错误消息变得非常冗长。花一分钟时间查看从此错误到下一个错误（红线）出现的所有消息。它向您展示了编译器试图做什么以能够编译该行。

1.  更改指定的行以读取如下内容：

```cpp
std::cout << "orange = " << static_cast<int>(orange) << "\n";
```

1.  单击`Orange::`作用域限定符。

1.  留给你的练习是使用`orange`作为作用域枚举重新编译文件。

在这个练习中，我们发现作用域枚举改进了 C++的强类型检查，如果我们希望将它们用作整数值，那么我们需要对它们进行转换，而非作用域枚举则会隐式转换。

#### 故障排除编译器错误

从前面的练习中可以看出，编译器可以从一个错误生成大量的错误和警告消息。这就是为什么建议找到第一个错误并首先修复它。在 IDE 中开发或使用着色错误的构建系统可以使这更容易。

### 结构和类

枚举是用户定义类型中的第一个，但它们并没有真正扩展语言，以便我们可以以适当的抽象级别表达问题的解决方案。然而，结构和类允许我们捕获和组合数据，然后关联方法以一致和有意义的方式来操作这些数据。

如果我们考虑两个矩阵的乘法，*A（m x n）*和*B（n x p）*，其结果是矩阵*C（m x p）*，那么 C 的第 i 行和第 j 列的方程如下：

![](img/C14583_02A_31.jpg)

###### 图 2A.31：第 i 行和第 j 列的方程

如果我们每次都必须这样写来乘两个矩阵，我们最终会得到许多嵌套的 for 循环。但是，如果我们可以将矩阵抽象成一个类，那么我们可以像表达两个整数或两个浮点数的乘法一样来表达它：

```cpp
Matrix a;
Matrix b;
// Code to initialize the matrices
auto c = a * b;
```

这就是面向对象设计的美妙之处 - 数据封装和概念的抽象被解释在这样一个层次上，以至于我们可以轻松理解程序试图实现的目标，而不会陷入细节。一旦我们确定矩阵乘法被正确实现，那么我们就可以自由地专注于以更高层次解决我们的问题。

接下来的讨论涉及类，但同样适用于结构体，大部分适用于联合体。在学习如何定义和使用类之后，我们将概述类、结构体和联合体之间的区别。

### 分数类

为了向您展示如何定义和使用类，我们将致力于开发`Fraction`类来实现有理数。一旦定义，我们可以像使用任何其他内置类型一样使用`Fraction`（加法、减法、乘法、除法），而不必担心细节 - 这就是抽象。现在我们只需在更高的抽象层次上思考和推理分数。

`Fraction`类将执行以下操作：

+   包含两个整数成员变量，`m_numerator`和`m_denominator`

+   提供方法来复制自身，分配给自身，相乘，相除，相加和相减

+   提供一种方法写入输出流

为了实现上述目标，我们有以下定义：

![图 2A.32：操作的定义](img/C14583_02A_32.jpg)

###### 图 2A.32：操作的定义

此外，我们执行的操作将需要将分数归一化为最低项。为此，分子和分母都要除以它们的最大公约数（GCD）。

### 构造函数、初始化和析构函数

类定义在 C++代码中表达的是用于在内存中创建对象并通过它们的方法操作对象的模式。我们需要做的第一件事是告诉编译器我们希望声明一个新类型 - 一个类。要声明`Fraction`类，我们从以下开始：

```cpp
class Fraction
{
};
```

我们将这放在一个头文件**Fraction.h**中，因为我们希望在代码的其他地方重用这个类规范。

我们需要做的下一件事是引入要存储在类中的数据，在这种情况下是`m_numerator`和`m_denominator`。这两者都是 int 类型：

```cpp
class Fraction
{
  int m_numerator;
  int m_denominator;
};
```

我们现在已经声明了要存储的数据，并为它们赋予了任何熟悉数学的人都能理解的名称，以了解每个成员变量存储的内容：

![](img/C14583_02A_33.jpg)

###### 图 2A.33：分数的公式

由于这是一个类，默认情况下，声明的任何项目都被假定为`private`。这意味着没有外部实体可以访问这些变量。正是这种隐藏（使数据私有，以及某些方法）使得 C++中的封装成为可能。C++有三种类访问修饰符：

+   **public**：这意味着成员（变量或函数）可以从类外部的任何地方访问。

+   **private**：这意味着成员（变量或函数）无法从类外部访问。事实上，甚至无法查看。私有变量和函数只能从类内部或通过友元方法或类访问。私有成员（变量和函数）由公共函数使用以实现所需的功能。

+   **protected**：这是私有和公共之间的交叉。从类外部来看，变量或函数是私有的。但是，对于从声明受保护成员的类派生的任何类，它们被视为公共的。

在我们定义类的这一点上，这并不是很有用。让我们将声明更改为以下内容：

```cpp
class Fraction
{
public:
  int m_numerator;
  int m_denominator;
};
```

通过这样做，我们可以访问内部变量。`Fraction number;`变量声明将导致编译器执行两件事：

+   分配足够的内存来容纳数据项（取决于类型，这可能涉及填充，即包括或添加未使用的内存以对齐成员以实现最有效的访问）。`sizeof`运算符可以告诉我们为我们的类分配了多少内存。

+   通过调用**默认构造函数**来初始化数据项。

这些步骤与编译器为内置类型执行的步骤相同，即步骤 2 什么也不做，导致未初始化的变量。但是默认构造函数是什么？它做什么？

首先，默认构造函数是一个特殊成员函数。它是许多可能构造函数中的一个，其中三个被视为特殊成员函数。构造函数可以声明零个、一个或多个参数，就像任何其他函数一样，但它们不指定返回类型。构造函数的特殊目的是将所有成员变量初始化，将对象置于一个明确定义的状态。如果成员变量本身是一个类，那么可能不需要指定如何初始化变量。如果成员变量是内置类型，那么我们需要为它们提供初始值。

### 类特殊成员函数

当我们定义一个新类型（结构体或类）时，编译器会为我们创建多达六个（6）个特殊成员函数：

+   `Fraction::Fraction()`): 当没有提供参数时调用（例如在前面的部分中）。这可以通过构造函数没有参数列表或为所有参数定义默认值来实现，例如`Fraction(int numerator=0, denominator=1)`。编译器提供了一个`implicit` `inline`默认构造函数，执行成员变量的默认初始化 - 对于内置类型，这意味着什么也不做。

+   `Fraction::~Fraction()`): 这是一个特殊成员函数，当对象的生命周期结束时调用。它的目的是释放对象在其生命周期中分配和保留的任何资源。编译器提供了一个`public` `inline`成员函数，调用成员变量的析构函数。

+   `Fraction::Fraction(const Fraction&)`): 这是另一个构造函数，其中第一个参数是`Fraction&`的形式，没有其他参数，或者其余参数具有默认值。第一个参数的形式是`Fraction&`、`const Fraction&`、`volatile Fraction&`或`const volatile Fraction&`。我们将在后面处理`const`，但在本书中不处理`volatile`。编译器提供了一个`non-explicit` `public` `inline`成员函数，通常形式为`Fraction::Fraction(const Fraction&)`，按初始化顺序复制每个成员变量。

+   `Fraction& Fraction::operator=(Fraction&)`): 这是一个成员函数，名称为`operator=`，第一个参数可以是值，也可以是类的任何引用类型，在这种情况下是`Fraction`、`Fraction&`、`const Fraction&`、`volatile Fraction&`或`const volatile Fraction&`。编译器提供了一个`public` `inline`成员函数，通常形式为`Fraction::Fraction(const Fraction&)`，按初始化顺序复制每个成员变量。

+   `Fraction::Fraction(Fraction&&)`): 这是 C++11 中引入的一种新类型的构造函数，第一个参数是`Fraction&&`的形式，没有其他参数，或者其余参数具有默认值。第一个参数的形式是`Fraction&&`、`const Fraction&&`、`volatile Fraction&&`或`const volatile Fraction&&`。编译器提供了一个`non-explicit` `public` `inline`成员函数，通常形式为`Fraction::Fraction(Fraction&&)`，按初始化顺序移动每个成员变量。

+   `Fraction& Fraction::operator=(Fraction&&)`): 这是 C++11 中引入的一种新类型的赋值运算符，是一个名为`operator=`的成员函数，第一个参数是允许移动构造函数的任何形式之一。编译器提供了一个`public` `inline`成员函数，通常采用`Fraction::Fraction(Fraction&&)`的形式，按初始化顺序复制每个成员变量。

除了默认构造函数外，这些函数处理了该类拥有的资源的管理-即如何复制/移动它们以及如何处理它们。另一方面，默认构造函数更像是接受值的任何其他构造函数-它只初始化资源。

我们可以声明任何这些特殊函数，强制它们被默认（即，让编译器生成默认版本），或者强制它们不被创建。关于这些特殊函数在其他特殊函数存在时何时自动生成也有一些规则。前四个函数在概念上相对直接，但是两个“移动”特殊成员函数需要额外的解释。我们将在第三章“可以和应该之间的距离-对象、指针和继承”中详细讨论移动语义，但现在它基本上就是它所指示的意思-将某物从一个对象移动到另一个对象。

### 隐式构造函数与显式构造函数

前面的描述讨论了编译器生成隐式或非显式构造函数。如果存在可以用一个参数调用的构造函数，例如复制构造函数或移动构造函数，默认情况下，编译器可以在必要时调用它，以便将其从一种类型转换为另一种类型，从而允许对表达式、函数调用或赋值进行编码。这并不总是期望的行为，我们可能希望阻止隐式转换，并确保如果我们类的用户真的希望进行转换，那么他们必须在程序中写出来。为了实现这一点，我们可以在构造函数的声明前加上`explicit`关键字，如下所示：

```cpp
explicit Fraction(int numerator, int denominator = 1);
```

`explicit`关键字也可以应用于其他运算符，编译器可能会用它进行类型转换。

### 类特殊成员函数-编译器生成规则

首先，如果我们声明了任何其他形式的构造函数-默认、复制、移动或用户定义的构造函数，就不会生成`Default Constructor`。其他特殊成员函数都不会影响它的生成。

其次，如果声明了析构函数，则不会生成`Destructor`。其他特殊成员函数都不会影响它的生成。

其他四个特殊函数的生成取决于析构函数或其他特殊函数的声明的存在，如下表所示：

![](img/C14583_02A_34.jpg)

###### 图 2A.34：特殊成员函数生成规则

### 默认和删除特殊成员函数

在 C++11 之前，如果我们想要阻止使用复制构造函数或复制赋值成员函数，那么我们必须将函数声明为私有，并且不提供函数的定义：

```cpp
class Fraction
{
public:
  Fraction();
private:
  Fraction(const Fraction&);
  Fraction& operator=(const Fraction&);
};
```

通过这种方式，我们确保如果有人试图从类外部访问复制构造函数或复制赋值，那么编译器将生成一个错误，说明该函数不可访问。这仍然声明了这些函数，并且它们可以从类内部访问。这是一种有效的方法，但并不完美，以防止使用这些特殊成员函数。

但是自 C++11 引入了两种新的声明形式，允许我们覆盖编译器的默认行为，如前述规则所定义。

首先，我们可以通过使用`= delete`后缀来声明方法，强制编译器不生成该方法，如下所示：

```cpp
Fraction(const Fraction&) = delete;
```

#### 注意

如果参数没有被使用，我们可以省略参数的名称。对于任何函数或成员函数都是如此。实际上，根据编译器设置的警告级别，它甚至可能会生成一个警告，表明参数没有被使用。

或者，我们可以通过使用`= default`后缀来强制编译器生成特殊成员函数的默认实现，就像这样：

```cpp
Fraction(const Fraction&) = default;
```

如果这只是函数的声明，那么我们也可以省略参数的名称。尽管如此，良好的实践规定我们应该命名参数以指示其用途。这样，我们类的用户就不需要查看调用函数的实现。

#### 注意

使用默认后缀声明特殊成员函数被视为用户定义的成员函数，用于上述规则的目的。

### 三五法则和零法则

正如我们之前讨论过的，除了默认构造函数之外，特殊成员函数处理了管理该类拥有的资源的语义 - 即如何复制/移动它们以及如何处理它们。这导致了 C++社区内关于处理特殊函数的两个“规则”。

在 C++11 之前，有“三法则”，它涉及复制构造函数、复制赋值运算符和析构函数。基本上它表明我们需要实现其中一个方法，因为封装资源的管理是非平凡的。

随着 C++11 中移动构造函数和移动赋值运算符的引入，这个规则扩展为“五法则”。规则的本质没有发生变化。简单地说，特殊成员函数的数量增加到了五个。记住编译器生成规则，确保所有五个特殊方法都被实现（或通过= default 强制），这是一个额外的原因，如果编译器无法访问移动语义函数，它将尝试使用复制语义函数，这可能不是所期望的。

#### 注意

有关更多详细信息，请参阅 C.ctor：C++核心指南中的构造函数、赋值和析构函数部分，网址为：[`isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines`](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)。

### 构造函数 - 初始化对象

构造函数的主要任务是将对象置于稳定状态，以便通过其成员函数对对象执行的任何操作都会产生一致的定义行为。虽然前面的陈述对于复制和移动构造函数是正确的，但它们通过不同的语义（从另一个对象复制或移动）来实现这一点。

我们有四种不同的机制可以控制对象的初始状态。C++对于在这种情况下使用哪种初始化有很多规则。我们不会详细讨论 C++标准的默认初始化、零初始化、值初始化、常量初始化等等。只需知道最好的方法是明确地初始化您的变量。

第一种，也是最不受欢迎的初始化机制是在构造函数的主体中为成员变量赋值，就像这样：

```cpp
Fraction::Fraction()
{
  this->m_numerator = 0;
  this->m_denominator = 1;
}
Fraction::Fraction(int numerator, int denominator)
{
  m_numerator = numerator;
  m_denominator = denominator;
}
```

清楚地知道了用于初始化变量的值。严格来说，这不是类的初始化 - 根据标准，当构造函数的主体被调用时，初始化才算完成。这在这个类中很容易维护。对于有多个构造函数和许多成员变量的较大类，这可能是一个维护问题。如果更改一个构造函数，您将需要更改它们所有。它还有一个问题，如果成员变量是引用类型（我们稍后会讨论），那么它就不能在构造函数的主体中完成。

默认构造函数使用`this`指针。每个成员函数，包括构造函数和析构函数，都带有一个隐式参数（即使它从未声明过）- `this`指针。`this`指向对象的当前实例。`->`操作符是另一个解引用操作符，在这种情况下是简写，即`*(this).m_numerator`。使用`this->`是可选的，可以省略。其他语言，如 Python，要求声明和使用隐式指针/引用（Python 中的约定是称为*self*）。

**第二**种机制是使用成员初始化列表，其在使用中有一个警告。对于我们的 Fraction 类，我们有以下内容：

```cpp
Fraction::Fraction() : m_numerator(0), m_denominator(1)
{
}
Fraction::Fraction(int numerator, int denominator) :
  m_numerator(numerator), m_denominator(denominator)
{
}
```

冒号:后面和左花括号{前面的代码部分（`m_numerator(0), m_denominator(1)`和`m_numerator(numerator), m_denominator(denominator)`）是成员初始化列表。我们可以在成员初始化列表中初始化引用类型。

#### 成员初始化列表顺序

无论您在成员初始化列表中放置成员的顺序如何，编译器都将按照它们在类中声明的顺序初始化成员。

**第三**种和**推荐**的初始化是 C++11 中引入的默认成员初始化。我们在变量声明时使用赋值或大括号初始化器定义默认初始值：

```cpp
class Fraction
{
public:
  int m_numerator = 0;     // equals initializer
  int m_denominator{1};    // brace initializer
};
```

如果构造函数没有定义成员变量的初始值，则将使用此默认值来初始化变量。这样做的好处是确保所有构造函数产生相同的初始化，除非它们在构造函数的定义中被明确修改。

C++11 还引入了第四种初始化样式，称为构造函数委托。它是成员初始化列表的修改，其中不是列出成员变量及其初始值，而是调用另一个构造函数。以下示例是人为的，您不会以这种方式编写类，但它显示了构造函数委托的语法：

```cpp
Fraction::Fraction(int numerator) : m_numerator(numerator), m_denominator(1)
{
}
Fraction::Fraction(int numerator, int denominator) : Fraction(numerator)
{
  auto factor = std::gcd(numerator, denominator);
  m_numerator /= factor;
  m_denominator = denominator / factor;
}
```

您从具有两个参数的构造函数中调用单参数构造函数。

### 练习 5：声明和初始化分数

在这个练习中，我们将使用不同的技术实现类成员初始化，包括构造函数委托。让我们开始吧：

1.  在 Eclipse 中打开**Lesson2A**项目，然后在**Project Explorer**中展开**Lesson2A**，然后展开**Exercise05**，双击**Exercise5.cpp**以在编辑器中打开此练习的文件。

1.  单击**启动配置**下拉菜单，然后选择**新启动配置…**。将**Exercise5**配置为以名称 Exercise5 运行。

1.  完成后，它将成为当前选择的启动配置。

1.  单击**运行**按钮。**练习 5**将运行并产生类似以下输出：![](img/C14583_02A_35.jpg)

###### 图 2A.35：练习 5 典型输出

报告的分数值来自以任何方式初始化成员变量。如果再次运行，您很可能会得到不同的分数。

1.  点击**运行**按钮几次。您会看到分数发生变化。

1.  在编辑器中，将构造函数更改为如下所示：

```cpp
Fraction() : m_numerator{0}, m_denominator{1}
{
}
```

1.  单击**运行**按钮并观察输出：![](img/C14583_02A_36.jpg)

###### 图 2A.36：修改后的练习 5 输出

这次，分数值由我们在成员初始化列表中指定的值定义。 

1.  在编辑器中，添加以下两个`构造函数`：

```cpp
Fraction(int numerator) : m_numerator(numerator), m_denominator(1)
{
}
Fraction(int numerator, int denominator) : Fraction(numerator)
{
  auto factor = std::gcd(numerator, denominator);
  m_numerator /= factor;
  m_denominator = denominator / factor;
}
```

1.  在主函数中，更改`fraction`的声明以包括初始化：

```cpp
Fraction fraction{3,2};
```

1.  点击**运行**按钮并观察输出：![](img/C14583_02A_37.jpg)

###### 图 2A.37：构造函数委托示例

在这个练习中，我们使用成员初始化列表和构造函数委托实现了成员变量的初始化。*我们将在练习 7 中返回到分数，为分数类添加运算符。*

### 值与引用和常量

到目前为止，我们只处理了值类型，也就是变量保存了对象的值。指针保存了我们感兴趣的值（即对象的地址）（或 nullptr）。但这可能导致效率低下和资源管理问题。我们将在这里讨论如何解决效率低下的问题，但在*第三章*，*可以和应该之间的距离-对象、指针和继承*中解决资源管理问题。

考虑以下问题..我们有一个 10×10 的双精度矩阵，我们希望为其编写一个反转函数。该类声明如下：

```cpp
class Matrix10x10
{
private:
  double m_data[10][10];
};
```

如果我们要取`sizeof(Matrix10x10)`，我们会得到`sizeof(double)` x 10 x 10 = 800 字节。现在，如果我们要为此实现一个矩阵反转函数，其签名可能如下所示：

```cpp
Matrix10x10 invert(Matrix10x10 lhs);
Matrix10x10 mat;
// set up mat
Matrix10x10 inv = invert(mat);
```

首先，这意味着编译器需要将`mat`持有的值传递给`invert()`函数，并将 800 字节复制到堆栈上。然后函数执行其需要执行的操作来反转矩阵（L-U 分解、计算行列式-无论实现者选择的方法是什么），然后将 800 字节的结果复制回`inv`变量。在堆栈上传递大量值从来都不是一个好主意，原因有两个：

+   堆栈是主机操作系统给我们程序的有限资源。

+   在系统中复制大量值是低效的。

这种方法被称为按值传递。也就是说，我们希望处理的项目的值被复制到函数中。

在 C（和 C++）中，通过使用指针来解决这个限制。上面的代码可能变成下面这样：

```cpp
void invert(Matrix10x10* src, Matrix10x10* inv);
Matrix10x10 mat;
Matrix10x10 inv;
// set up mat
invert(&mat, &inv);
```

在这里，我们只是传递了 src 和 target 的地址作为两个指针的逆结果（这是少量字节）。不幸的是，这导致函数内部的代码在每次使用`src`或`inv`时都必须使用解引用操作符（`*`），使得代码更难阅读。此外，指针的使用导致了许多问题。

C++引入了一个更好的方法-变量别名或引用。引用类型是用和号（`&`）操作符声明的。因此，我们可以将 invert 方法声明如下：

```cpp
void invert(Matrix10x10& src, Matrix10x10& inv);
Matrix10x10 mat;
Matrix10x10 inv;
// set up mat
invert(mat, inv);
```

请注意，调用该方法不需要特殊的操作符来传递引用。从编译器的角度来看，引用仍然是一个带有一个限制的指针-它不能保存 nullptr。从程序员的角度来看，引用允许我们在不必担心在正确的位置使用解引用操作符的情况下推理我们的代码。这被称为**按引用传递**。

我们看到引用被传递给了复制构造函数和复制赋值方法。当用于它们的移动等价物时，引用的类型被称为**右值引用运算符**，将在*第三章*，*可以和应该之间的距离-对象、指针和继承*中解释。

`按值传递`的一个优点是我们不能无意中修改传递给方法的变量的值。现在，如果我们`按引用传递`，我们就不能再保证我们调用的方法不会修改原始变量。为了解决这个问题，我们可以将 invert 方法的签名更改为如下所示：

```cpp
void invert(const Matrix10x10& src, Matrix10x10& inv);
```

const 关键字告诉编译器，在处理`invert()`函数的定义时，将值引用到`src`的任何部分都是非法的。如果该方法尝试修改 src，编译器将生成一个错误。

在指定类型-变量部分，我们发现`auto title`的声明导致`title`是`const char *`类型。现在，我们可以解释`const`部分了。

`title`变量是**指向常量字符的指针**。换句话说，我们不能改变指向的内存中存储的数据的值。因此，我们不能执行以下操作：

```cpp
*title = 's';
```

这是因为编译器将生成与更改常量值相关的错误。然而，我们可以改变指针中存储的值。我们可以执行以下操作：

```cpp
title = "Maid Marian";
```

我们现在已经介绍了引用作为函数参数类型的用法，但它们也可以用作成员变量而不是指针。引用和指针之间有区别：

引用必须引用实际对象（没有 nullptr 的等价物）。一旦初始化，引用就不能被改变（这意味着引用必须要么是默认成员初始化的，要么出现在成员初始化列表中）。对象必须存在，只要对它的引用存在（如果对象可以在引用被销毁之前被销毁，那么如果尝试访问对象就有潜在的未定义行为）。

### 练习 6：声明和使用引用类型

在这个练习中，我们将声明和使用引用类型，以使代码更高效、更易读。让我们开始吧：

1.  在 Eclipse 中打开**Lesson2A**项目，然后在**Project Explorer**中展开**Lesson2A**，然后展开**Exercise06**，双击**Exercise6.cpp**以在编辑器中打开此练习的文件。

1.  点击**Launch Configuration**下拉菜单，选择**New Launch Configuration…**。配置**Exercise6**以使用名称 Exercise6 运行。

1.  完成后，它将成为当前选择的启动配置。

1.  点击`rvalue`变量允许我们操纵（读取和写入）存储在`value`变量中的数据。我们有一个对`value`变量的引用`rvalue`。我们还可以看到`swap()`函数交换了`a`和`b`变量中存储的值。

1.  在编辑器中，更改 swap 函数的函数定义：

```cpp
void swap(const int& lhs, const int& rhs)
```

1.  点击**Run**按钮。当出现工作区中的错误对话框时，点击**Cancel**。编译器报告的第一个错误如下所示：

![图 2A.39：赋值时的只读错误](img/C14583_02A_39.jpg)

###### 图 2A.39：赋值时的只读错误

通过将参数从`int& lhs`更改为`const int& lhs`，我们告诉编译器在此函数内部参数不应该被改变。因为我们在函数中对 lhs 进行了赋值，所以编译器生成了关于 lhs 为只读的错误并终止了程序。

### 实现标准运算符

要像内置类一样使用分数，我们需要使它们能够使用标准数学运算符（`+，-，*，/`）及其赋值对应物（`+=，-=，*=，/=`）。如果您不熟悉赋值运算符，请考虑以下两个表达式 - 它们产生相同的输出：

```cpp
a = a + b;
a += b;
```

为 Fraction 声明这两个运算符的语法如下：

```cpp
// member function declarations
Fraction& operator+=(const Fraction& rhs);
Fraction operator+(const Fraction& rhs) const;
// normal function declaration of operator+
Fraction operator+(const Fraction& lhs, const Fraction& rhs);
```

因为`operator+=`方法修改了左侧变量的内容（将 a 添加到 b 然后再次存储在 a 中），建议将其实现为成员变量。在这种情况下，由于我们没有创建新值，我们可以直接返回对现有 lhs 的引用。

另一方面，`operator+`方法不应修改 lhs 或 rhs 并返回一个新对象。实现者可以自由地将其实现为成员函数或自由函数。在前面的代码中都展示了这两种方法，但只应存在一种。关于成员函数实现的有趣之处在于声明末尾的 const 关键字。这告诉编译器，当调用这个成员函数时，它不会修改对象的内部状态。虽然这两种方法都是有效的，但如果可能的话，`operator+`应该作为一个普通函数实现，而不是类的一部分。

相同的方法也可以用于其他运算符`-（减法）`，`*（乘法）`和`/（除法）`。前面的方法实现了标准数学运算符的语义，并使我们的类型像内置类型一样工作。

### 实现输出流操作符（<<）

C++将输入/输出（I/O）抽象为标准库中的流类层次结构（我们将在*第 2B 章*，*不允许鸭子 - 模板和推断*中讨论）。在*练习 5*，*声明和初始化分数*中，我们看到我们可以将分数插入到输出流中，如下所示：

```cpp
std::cout << "fraction = " << fraction.getNumerator() << "/" 
                           << fraction.getDenominator() << "\n";
```

到目前为止，对于我们的分数类，我们已经通过使用`getNumerator()`和`getDenominator()`方法从外部访问数据值来写出了分子和分母的值，但有更好的方法。作为使我们的类在 C++中成为一等公民的一部分，在合适的情况下，我们应该重载 I/O 运算符。在本章中，我们只会看输出运算符<<，也称为插入运算符。这样，我们可以用更清晰的版本替换以前的代码：

```cpp
std::cout << "fraction = " << fraction << "\n";
```

我们可以将运算符重载为友元函数或普通函数（如果类提供我们需要插入的数据的 getter 函数）。对于我们的目的，我们将其定义为普通函数：

```cpp
inline std::ostream& operator<< (std::ostream &out, const Fraction &rhs)
{
    out << rhs.getNumerator() << " / " << rhs.getDenominator();
    return out;
}
```

### 我们的代码结构

在我们深入练习之前，我们需要讨论一下我们的类的各个部分放在哪里 - 声明和定义。声明是我们的类的蓝图，指示它需要什么数据存储和将实现的方法。定义是每个方法的实际实现细节。

在 Java 和 C#等语言中，声明和定义是一样的，它们必须存在于一个文件（Java）或跨多个文件（C#部分类）中。在 C++中，取决于类和您希望向其他类公开多少，声明必须出现在头文件中（可以在其他文件中`#include`使用），定义可以出现在三个地方之一 - 内联在定义中，在相同文件中的`inline`定义，或在单独的实现文件中。

头文件通常以.hpp 扩展名命名，而实现文件通常是`*.cpp`或`*.cxx`之一。实现文件也称为**翻译单元**。通过将函数定义为内联，我们允许编译器以函数可能甚至不存在于最终程序中的方式优化代码 - 它已经将我们放入函数中的步骤替换为我们从中调用函数的位置。

### 练习 7：为分数类添加运算符

在这个练习中，我们的目标是使用单元测试在我们的分数类中实现运算符功能。这使我们的分数类成为一个真正的类型。让我们开始吧：

1.  在 Eclipse 中打开**Lesson2A**项目，然后在**项目资源管理器**中展开**Lesson2A**，然后**Exercise07**，双击**Exercise7.cpp**以在编辑器中打开此练习的文件。

1.  单击**启动配置**下拉菜单，然后选择**新启动配置…**。配置 Exercise7 以使用名称 Exercise7 运行。

1.  完成后，它将成为当前选择的启动配置。

1.  我们还需要配置一个单元测试。在 Eclipse 中，单击名为**运行** | **运行配置…**的菜单项，在左侧右键单击**C/C++单元**，然后选择**新配置**。

1.  将名称从`Lesson2A Debug`更改为`Exercise7 Tests`。

1.  在**C/C++应用程序**下，选择**搜索项目**选项，并在新对话框中选择**tests**。

1.  接下来，转到**C/C++测试**选项卡，并在下拉菜单中选择**Google 测试运行器**。点击对话框底部的**应用**，然后点击我们第一次运行的测试选项：![图 2A.40：失败的测试 - 乘法](img/C14583_02A_40.jpg)

###### 图 2A.40：失败的测试 - 乘法

1.  打开`operator*=`函数。更新它的代码如下：

```cpp
Fraction& Fraction::operator*=(const Fraction& rhs)
{
  Fraction tmp(m_numerator*rhs.m_numerator, m_denominator*rhs.m_denominator);
  *this = tmp;
  return *this;
}
```

1.  点击**运行**按钮重新运行测试。这次，所有的测试都通过了：![图 2A.41：通过测试](img/C14583_02A_41.jpg)

###### 图 2A.41：通过测试

1.  在 IDE 中打开`operator*=()`，同时测试其他的`operator*()`。修复`operator*=()`如何修复`operator*()`？如果在编辑器中打开 Fraction.hpp，你会发现`operator*()`函数是通过调用`operator*=()`来实现的，也就是说，它被标记为内联函数，是一个普通函数而不是成员函数。一般来说，当重载这些运算符时，修改调用它的对象的函数是成员函数，而生成新值的函数是调用成员函数的普通函数。

1.  在编辑器中打开**Fraction.hpp**，并将文件顶部的行更改为以下内容：

```cpp
#define EXERCISE7_STEP  11
```

1.  点击**AddFractions**和**AddFractions2**：![图 2A.42：额外的失败测试](img/C14583_02A_42.jpg)

###### 图 2A.42：额外的失败测试

1.  在**Function.cpp**文件中找到`operator+=`函数。

1.  对函数进行必要的更改，然后点击实现`operator*=()`。

1.  在编辑器中打开**Fraction.hpp**，并将文件顶部的行更改为以下内容：

```cpp
#define EXERCISE7_STEP  15
```

1.  点击**SubtractFractions**和**SubtractFractions2**。

1.  在 Function.cpp 文件中找到`operator-=`函数。

1.  对函数进行必要的更改，然后点击**运行**按钮，直到测试通过。

1.  在编辑器中打开**Fraction.hpp**，并将文件顶部的行更改为以下内容：

```cpp
#define EXERCISE7_STEP  19
```

1.  点击**运行**按钮重新运行测试 - 这次，我们添加了两个失败的测试 - **DivideFractions**和**DivideFractions2**。

1.  在**Function.cpp**文件中找到`operator/=`函数。

1.  对函数进行必要的更改，然后点击**运行**按钮，直到测试通过。

1.  在编辑器中打开**Fraction.hpp**，并将文件顶部的行更改为以下内容：

```cpp
#define EXERCISE7_STEP  23
```

1.  点击**插入运算符**。

1.  在 Function.hpp 文件中找到`operator<<`函数。

1.  对函数进行必要的更改，然后点击**运行**按钮，直到测试通过。

1.  从**启动配置**中选择**Exercise7**，然后点击**运行**按钮。这将产生以下输出：

![图 2A.43：功能性分数类](img/C14583_02A_43.jpg)

###### 图 2A.43：功能性分数类

这完成了我们对`Fraction`类的实现。当我们考虑*第三章*中的异常时，我们将再次返回它，*可以和应该之间的距离 - 对象、指针和继承*，这样我们就可以处理分数中的非法值（分母为 0）。

### 函数重载

C++支持一种称为函数重载的特性，即两个或多个函数具有相同的名称，但它们的参数列表不同。参数的数量可以相同，但至少一个参数类型必须不同。或者，它们可以具有不同数量的参数。因此，多个函数的函数原型是不同的。但是，两个函数不能具有相同的函数名称、相同的参数类型和不同的返回类型。以下是一个重载的示例：

```cpp
std::ostream& print(std::ostream& os, int value) {
   os << value << " is an int\n";
   return os;
}
std::ostream& print(std::ostream& os, float value) {
   os << value << " is a single precision float\n";
   return os;
}
std::ostream& print(std::ostream& os, double value) {
   os << value << " is a double precision float \n";
   return os;
}
// The next function causes the compiler to generate an error
// as it only differs by return type.
void print(std::ostream& os, double value) {
   os << value << " is a double precision float!\n";
}
```

到目前为止，`Fraction`上的多个构造函数和重载的算术运算符都是编译器在遇到这些函数时必须引用的重载函数的示例。考虑以下代码：

```cpp
int main(int argc, char** argv) {
   print(42);
}
```

当编译器遇到`print(42)`这一行时，它需要确定调用先前定义的函数中的哪一个，因此执行以下过程（大大简化）：

![图 2A.44：函数重载解析（简化）](img/C14583_02A_44.jpg)

###### 图 2A.44：函数重载解析（简化）

C++标准定义了编译器根据如何操作（即转换）参数来确定最佳候选函数的规则。如果不需要转换，则该函数是最佳匹配。

### 类，结构体和联合

当您定义一个类并且不指定访问修饰符（public，protected，private）时，默认情况下所有成员都将是 private 的：

```cpp
class Fraction
{
  Fraction() {};            // All of these are private
  int m_numerator;
  int m_denominator;
};
```

当您定义一个结构体并且不指定访问修饰符（public，protected，private）时，默认情况下所有成员都将是 public 的：

```cpp
struct Fraction
{
  Fraction() {};            // All of these are public
  int m_numerator;
  int m_denominator;
};
```

还有另一个区别，我们将在解释继承和多态性之后进行讨论。联合是一种与结构体和类不同但又相同的数据构造类型。联合是一种特殊类型的结构声明，其中所有成员占用相同的内存，并且在给定时间只有一个成员是有效的。`union`声明的一个示例如下：

```cpp
union variant
{
  int m_ivalue;
  float m_fvalue;
  double m_dvalue;
};
```

当您定义一个联合并且不指定访问修饰符（public，protected，private）时，默认情况下所有成员都将是 public 的。

联合的主要问题是没有内在的方法来知道在任何给定时间哪个值是有效的。这通过定义所谓的*标记联合*来解决 - 即一个包含联合和一个枚举的结构，用于标识它是有效值。联合还有其他限制（例如，只有一个成员可以有默认成员初始化程序）。我们不会在本书中深入探讨联合。

### 活动 1：图形处理

在现代计算环境中，矩阵被广泛用于解决各种问题 - 解决同时方程，分析电力网格或电路，对图形渲染对象进行操作，并提供机器学习的实现。在图形世界中，无论是二维（2D）还是三维（3D），您希望对对象执行的所有操作都可以通过矩阵乘法来完成。您的团队被要求开发点，变换矩阵的表示以及您可能希望对它们执行的操作。按照以下步骤来实现这一点：

1.  从**Lesson2A/Activity01**文件夹加载准备好的项目。

1.  创建一个名为**Point3d**的类，可以默认构造为原点，或使用三个或四个值的初始化列表（数据直接存储在类中）来构造。

1.  创建一个名为**Matrix3d**的类，可以默认构造为单位矩阵，或使用嵌套初始化列表来提供所有值（数据直接存储在类中）来构造。

1.  在`operator()`上，以便它接受（`index`）参数以返回`x(0)`，`y(1)`，`z(2)`和`w(3)`处的值。

1.  在`operator()`上接受（`row, col`）参数，以便返回该值。

1.  添加单元测试以验证所有上述功能。

1.  在**Matrix3d**类中添加`operator*=(const Matrix3d&)`和`operator==(const Matrix3d&)`，以及它们的单元测试。

1.  添加用于将两个**Matrix3d**对象相乘以及将**Matrix3d**对象乘以**Point3d**对象的自由函数，并进行单元测试。

1.  添加用于创建平移，缩放和旋转矩阵（围绕 x，y，z 轴）及其单元测试的独立方法。

在实现上述步骤之后，预期输出如下：

![](img/C14583_02A_45.jpg)

###### 图 2A.45：成功运行活动程序

在本次活动中，我们不会担心索引超出范围的可能性。我们将在*第三章*“能与应该之间的距离-对象、指针和继承”中讨论这个问题。单位矩阵是一个方阵（在我们的例子中是 4x4），对角线上的所有值都设置为 1，其他值都为 0。

在处理 3D 图形时，我们使用增广矩阵来表示点（顶点）和变换，以便所有的变换（平移、缩放、旋转）都可以通过乘法来实现。

一个`n × m`矩阵是一个包含 n 行 m 个数字的数组。例如，一个`2 x 3`矩阵可能如下所示：

![图 2A.46：2x3 矩阵](img/C14583_02A_46.jpg)

###### 图 2A.46：2x3 矩阵

三维空间中的顶点可以表示为一个三元组（x，y，z）。然而，我们用另一个坐标`w（对于顶点为 1，对于方向为 0）`来增强它，使其成为一个四元组（x，y，z，1）。我们不使用元组，而是将其放在一个`4 x 1`矩阵中，如下所示：

![图 2A.47：4x1 矩阵](img/C14583_02A_47.jpg)

###### 图 2A.47：4x1 矩阵

如果我们将`4 x 1`矩阵（点）乘以`4 x 4`矩阵（变换），我们可以操纵这个点。如果`Ti`表示一个变换，那么我们可以将变换相乘，以实现对点的某种操纵。

![图 2A.48：乘法变换](img/C14583_02A_48.jpg)

###### 图 2A.48：乘法变换

要将一个转换矩阵相乘，`A x P = B`，我们需要做以下操作：

![图 2A.49：乘法变换矩阵](img/C14583_02A_49.jpg)

###### 图 2A.49：乘法变换矩阵

我们也可以这样表达：

![图 2A.50：乘法变换表达式](img/C14583_02A_50.jpg)

###### 图 2A.50：乘法变换表达式

同样，两个`4 x 4`矩阵也可以相乘，`AxB=C`：

![图 2A.51：4x4 矩阵乘法表达式：](img/C14583_02A_51.jpg)

###### 图 2A.51：4x4 矩阵乘法表达式：

变换的矩阵如下：

![图 2A.52：变换矩阵列表](img/C14583_02A_52.jpg)

###### 图 2A.52：变换矩阵列表

#### 注意

本次活动的解决方案可以在第 635 页找到。

## 总结

在本章中，我们学习了 C++中的类型。首先，我们介绍了内置类型，然后学习了如何创建行为类似于内置类型的自定义类型。我们学习了如何声明和初始化变量，了解了编译器从源代码生成的内容，变量的存储位置，链接器如何将其组合，以及在计算机内存中的样子。我们学习了一些关于 C++的部落智慧，比如零规则和五规则。这些构成了 C++的基本组成部分。在下一章中，我们将学习如何使用 C++模板创建函数和类，并探索模板类型推导的更多内容。
