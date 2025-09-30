# *第二章*：微控制器板软件设置和 C 编程

在本章中，你将回顾用于编程 Blue Pill 和 Curiosity Nano 微控制器板的 IDE 的基本配置，以及学习为 Blue Pill 和 Curiosity Nano 编写应用程序所需的 C 编程语言基础知识。这绝对不是一份全面的 C 教程。它包含了理解和完成本书所有章节中解释的练习的重要信息。在本章中，我们将涵盖以下主要主题：

+   介绍 C 编程语言

+   介绍 Curiosity Nano 微控制器板编程

+   介绍 Blue Pill 微控制器板编程

+   示例 – 编程和使用微控制器板内部 LED

在本章结束时，你将获得对 C 编程语言的坚实基础介绍，包括一组编程指令，这些指令对于使用 Blue Pill 和 Curiosity Nano 微控制器板开发许多小型和中型微控制器项目非常有用。本章还涵盖了内部 LED 的使用，Blue Pill 和 Curiosity Nano 都具备这一功能。这可以非常方便地快速显示数字结果（例如，确认项目中的操作）。

# 技术要求

我们在本章中将使用的软件是用于编程 Blue Pill 和 Curiosity Nano 的 Arduino 和 MPLAB X IDE。它们的安装过程在*第一章**,* *微控制器和微控制器板简介*中已有描述。我们还将使用之前章节中使用的相同代码示例。

在本章中，我们还将使用以下硬件：

+   一个无焊点面包板。

+   Blue Pill 和 Curiosity Nano 微控制器板。

+   一条用于将微控制器板连接到计算机的微型 USB 线。

+   需要上传编译代码到 Blue Pill 的 ST-LINK/V2 电子接口。请记住，ST-Link/V2 需要四根公对公的杜邦线。

这些是本章中描述的示例所需的基本硬件组件，它们也将证明在其他章节中解释的更复杂项目中非常有用。

本章中使用的代码可以在本书的 GitHub 仓库中找到：

[`github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter02`](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter02)

本章的“代码在行动”视频可以在此找到：[`bit.ly/3xwFvPA`](https://bit.ly/3xwFvPA)

下一节将简要介绍 C 编程语言。

# 介绍 C 编程语言

**C 编程语言**最初在七十年代初被创建，用于开发 UNIX 操作系统，但自那时起它已经被移植到几乎所有操作系统上。它是一种中级编程语言，因为它从高级语言（例如 Python）和低级语言（例如汇编语言）中共享属性。C 语言通常比低级语言更容易编程，因为它非常*易于阅读*，并且有许多库可用，这些库有助于软件开发，以及其他原因。它也非常适合编程嵌入式系统。C 是最受欢迎的编程语言之一，几乎所有微控制器都可以用 C 编译器编程——Blue Pill 和 Curiosity Nano 也不例外。

C 语言在不同微控制器系列和制造商之间并不完全可移植。例如，在 Blue Pill 和 Curiosity Nano 中，I/O 端口和中断的编程方式并不相同。这就是为什么需要两种类型的 C 编译器和不同的库来编程这两个微控制器板。实际上，用于编程 Blue Pill 的 Arduino IDE 使用的是 C 语言的变体，称为**C++**。C++是 C 编程语言的强大扩展，它结合了面向对象和低内存级编程等特性。

以下部分解释了 C 语言结构的基础知识。本节包括对`#include`指令的解释、编写注释、理解变量、使用常量、关键字列表、声明函数、评估表达式以及在 C 中编写循环。

## C 语言的基本结构

与其他编程语言一样，C 语言使得可以在称为`.h`的单独文件中声明程序元素，如常量、类型、函数和变量。这有助于组织 C 指令并减少主 C 代码中的混乱。库是一个包含程序元素（如函数）的头文件，可以与其他 C 程序员共享或在不同的 C 程序中反复使用。C 语言编译器包含我们在本书中将使用的重要库。头文件可以通过`#include`指令（即链接和编译）与主程序一起包含；因此，在头文件中声明的编程元素将在 C 程序中被调用和使用。

有许多有用的标准和非标准库。我们将回顾并使用两者。`#include`指令是 C 编译器的特殊指令，而不是常规的 C 指令。它应该写在程序的开头，并且末尾没有分号。只有 C 语句在末尾有分号。有三种编写和应用`#include`指令的方法。具体如下：

+   `#include <file_name.h>`：此类指令使用小于和大于符号，意味着头文件（`.h`）位于编译器路径中。你不需要写出头文件的完整路径。

+   `#include "file_name.h"`：此类指令使用双引号。头文件存储在项目目录中。

+   `#include "sub_directory_name/file_name.h"`：此类指令类型告诉编译器头文件位于子目录中。请注意，斜杠符号的使用取决于你使用的操作系统。例如，Windows 计算机使用反斜杠（*\\*）符号作为目录分隔符。Linux 和 Mac 计算机使用正斜杠（*/*）符号。

下一个子节将展示如何定义和使用头文件。

### #include 指令示例

以下程序示例展示了如何包含位于项目目录中的头文件：

```cpp
#include "main_file.h"
int main(void)
{
    x = 1;
    y = 2;
    z = x+y;
}
```

在前面的例子中，`x`、`y` 和 `z` 变量是在 `main_file.h` 头文件中声明的，因此它们在主程序中没有声明。头文件 (`file.h`) 包含以下代码，声明了在主代码中使用的三个变量：

```cpp
int x;
int y;
int z;
```

我们可以在主程序中声明变量，而不在头文件（`.h`）中声明变量。是否要在头文件中编写程序元素取决于你。我们将在本章后面学习更多关于变量的内容。

注意

C 语言是区分大小写的，因此在编写 C 代码时要小心。大多数 C 语言指令都是用非大写字母编写的。声明变量时也要小心。例如，C 中的变量 *x* 和 *X* 是不同的。

C 语言附带了一些标准库，许多程序员很好地利用了它们。`stdio.h` 库（存储为头文件）在 C 编程中被广泛使用。它定义了多个宏、变量类型，以及用于执行数据输入和输出的专用函数；例如，从键盘读取输入字母或将文本写入控制台。控制台是 IDE 提供的一个基于文本的区域，在这里从键盘读取数据或写入文本或特殊字符。

这是一个使用 `<stdio.h>` 指令的简短 C 程序示例：

```cpp
// program file name: helloworld.c
#include <stdio.h>  
int main()  
{  // start main block of instructions
     printf("Hello world!"); 
     return 0;
} 
```

C 程序文件以 `.c` 扩展名存储（例如 `mainprogram.c`）。C++ 程序文件通常以 `.cpp` 扩展名存储（例如，`mainprogram.cpp`）。

`printf()` 函数用于在 IDE 的控制台上显示字符（例如，文本消息）。从前面的程序示例中可以看出，我们编写了一些注释来解释每一行代码。下一节将展示在 C 语言中编写注释的不同方式。

### 在 C 中使用注释

**注释**是不影响 C 程序功能的文本块或行。在 C 编程中编写注释很有用，因为它们可以用来解释和阐明指令、函数、变量等的含义或功能。我们编写的所有注释都会被编译器忽略。在 C 中编写注释有几种方法：

+   使用双斜杠（`//`）：这会创建单行注释。

+   使用斜杠和星号（`/*  */`）：这会创建一个包含文本块的注释。

此代码示例演示了如何使用两种类型的注释：

```cpp
/**********************************************************
Program: Helloworld.c 
Purpose: It shows the text "Hello, world!!" on the IDE's console. 
Author: M. Garcia.
Program creation date: September 9, 2020.
Program version: 1.0 
**********************************************************/
#include <stdio.h>  //standard I/O library
int main(void)
{
    int x; // we declare an integer variable

    printf("Hello, world!!"); 
    x=1; // we assign the value of 1 to variable x.
}
```

小贴士

在 C 程序的开头添加代码的目的、版本号和日期以及作者的姓名作为注释是一种良好的编程实践。

下一节将描述如何在 C 编程中声明和使用变量。变量非常有用，你将在本书的大部分章节中使用它们。

### 理解 C 语言中的变量

**变量**是一个通过编程分配给微控制器内存存储区域（也称为标识符）的名字，用于临时存储数据。在 C 语言中存在特定类型的变量，用于存储不同类型的数据。变量类型决定了分配给变量的微控制器内存的布局和大小（通常是其内部随机访问内存或 RAM）。

我们必须在 C 语言中首先声明一个变量，才能在代码中使用它。变量声明有两个部分——数据类型和标识符，使用以下语法：`<数据类型> <标识符>`。以下是对这两部分的解释：

+   **数据类型**（或简称类型）定义了要存储在变量中的数据类型（例如，整数）。有许多数据类型及其修饰符。以下表格描述了四种主要类型：

![表 2.1 – C 语言中使用的四种主要数据类型](img/Table_2.1_B16413.jpg)

表 2.1 – C 语言中使用的四种主要数据类型

+   *表 2.1*中的每种类型都有`unsigned`、`signed`、`short`和`long`等修饰符。例如，我们可以声明一个存储无符号整数的变量，如`unsigned int x;`。

+   还有一种名为`void`的类型。这种类型没有值，通常用于定义返回无值的函数类型。

+   标识符是唯一标识变量的名字。标识符可以用字母 a..z 或 A..Z、数字 0..9 和下划线字符：_ 来编写。标识符不能有空格，并且第一个字符不能是数字。记住，标识符是区分大小写的。此外，根据 ANSI C 标准，标识符的长度应少于 32 个字符。

例如，让我们声明一个名为 x 的变量，它可以存储浮点数：

`float x;`

在前面的代码示例中，C 编译器将为变量*x*分配一个特定的内存分配，该分配仅包含浮点数。

现在，让我们在以下代码行中使用这个变量：

`x=1.10;`

如您所见，我们将浮点值 1.10 存储在名为*x*的变量中。以下示例演示了如何在 C 程序中使用变量：

```cpp
/* program that converts from Fahrenheit degrees to Celsius degrees. Written by Miguel Garcia-Ruiz. Version 1.0\. Date: Sept. 9, 2020
*/
#include <stdio.h> // standard I/O library to write text
int main(void) // It won't return any value
{
    float celsius_degrees;
    float fahrenheit_degrees=75.0;
    // Calculate the conversion:
    celsius_degrees=(fahrenheit_degrees-32)*5/9;
    // printf displays the result on the console:
    printf("%f",celsius_degrees); 
}
```

您可以在声明变量时初始化其值，就像前面示例中的`fahrenheit_degrees`变量所示。

我们也可以使用字符串开头和结尾的双引号将字符串存储在变量中。以下是一个示例：

`char name = "Michael";`

上述示例展示了如何将字符串存储在字符变量类型中，它是一个字符数组。

### 声明局部和全局变量

在 C 语言中，根据变量的声明位置，存在两种类型的变量。它们可以有不同的值和用途：

+   **全局变量**：这些是在您的代码的所有函数外部声明的。这些变量可以在任何函数和整个程序中使用。

+   **局部变量**：局部变量是在函数内部声明的。它们仅在声明它们的函数内部工作，因此它们的值不能在函数外部使用。请看以下包含全局和局部变量的示例：

```cpp
#include<stdio.h>
// These are global variables:
int y;
int m;
int x;
int b;
int straight_line_equation() {
    y=m*x+b;
    return y;
}
int main(){
    int answer;  // this is a local variable
    m=2;
    x=3;
    b=5;
    answer = straight_line_equation();
    printf(" %d\n  ",answer);
    return 0;  // this terminates  program
}
```

在前面的示例中，全局变量*y*、*m*、*x*和*b*在所有程序中工作，包括在`straight_line_equation()`函数内部。

### 使用常量

**常量**（也称为常量变量）可以用来定义一个在整个程序中值不会改变的变量。C 语言中的常量对于定义数学常数很有用。声明常量的语法如下：

`const <数据类型> <标识符>=<值>;`

在这里，数据类型可以是`int`、`float`、`char`或`double`，或者它们的修饰符，例如：

```cpp
const float euler_constant=2.7183;
const char A_character='a';
```

您还可以使用`#define`指令声明变量。它位于程序的开头，紧随`#include`指令之后，行尾没有分号，使用以下语法：`#define <标识符> <值>`。

我们不需要声明常量的数据类型。编译器将动态确定。以下示例展示了如何声明常量：

```cpp
#define PI 3.1416
#define value1 11
#define char_Val 'z'
```

下一节将处理在 C 程序中广泛使用的 C 语言关键字。

### 使用关键字

ANSI C 标准定义了多个在 C 编程中具有特定目的的**关键字**。这些关键字不能用于命名变量或常量。这些是在您的 C 代码中可以使用的关键字（语句）：

`auto, break, case, char, const, continue, default, do, double, else, enum, extern, float, for, goto, if, int, long, register, return, short, signed sizeof, static, struct, switch, typedef, union, unsigned, void, volatile, while.`

用于编译 Blue Pill 和 Curiosity Nano 板程序的编译器具有额外的关键字。我们将在本章中列出它们。下一节将解释 C 语言中的函数是什么。

### 声明 C 语言中的函数

`main()` 函数。这个函数在 C 程序中编写，其他函数都从它调用。你可以逻辑地将代码划分为函数，使其更易于阅读，并将与同一任务相关的指令分组，为指令提供一些结构。C 语言中的函数定义大致类似于代数函数，其中包含函数名、函数定义和函数参数（s）。

在 C 语言中定义函数的一般形式如下：

```cpp
<return_data_type> <function_name> (parameter list) {    <list of instructions>
    return <expression>; //optional
}
```

`return`语句允许从函数返回一个值，并且这个返回值被用于程序的其它部分。`return`语句是可选的，因为你可以编写一个不返回值的函数。

小贴士

在函数块中缩进包含的指令是一种良好的编程实践。这为函数提供了更多的视觉结构和可读性。

以下函数示例展示了如何使用参数以及如何从函数中返回数据，其中`number1`和`number2`是函数参数：

```cpp
int maxnumber(int number1, int number2) {
    /* Declaring a local variable to store the result: */
    int result1;
    if (number1 > number2)
        result1 = number1;
    else
        result1 = number2;
    return result1; 
}
```

在前面的例子中，函数返回了两个数字比较的结果。

小贴士

确保函数的数据类型与`return`语句中使用的变量类型相同。

如果出于某种原因，你不需要从函数中返回值，你可以使用`void`语句而不是定义函数的数据类型，例如：

```cpp
void error_message ()
{
    printf("Error.");
}
```

在前面的例子中，我们没有在函数中使用`return 0`语句，因为它没有返回任何值。然后我们可以`error_message();`。

### 调用函数

一旦我们声明了一个函数，我们就需要**调用**它，即在代码的另一个部分运行它。这会将程序控制权转移到被调用的函数，并执行其中的指令。执行完函数中的所有指令后，程序控制权恢复，继续运行主程序中的指令。

要调用一个函数，你需要写出函数名和参数所需的值。如果你的函数返回一个值，你可以将其存储在一个变量中。例如，让我们调用我们之前解释过的`max()`函数：

```cpp
int result2;
result2=maxnumber(4,3);
```

在这个例子中，`maxnumber()`函数进行的数字比较的结果将被存储在`result2`变量中。

### 评估表达式（决策语句）

C 语言提供了一种声明一个或多个**逻辑条件**的方法，这些条件可以被程序评估（测试），以及一些根据评估结果需要执行的语句，即条件为真或假。

C 编程语言假定任何非空或非零值都是真值。如果值为零或空，则为假。C 语言有以下决策语句：

+   `if` (expression_to_evaluate) {statements}: 这里的决策包含一个布尔表达式，后面跟着一个或多个语句，如果决策为真，则执行这些语句，例如：

    ```cpp
    #include <stdio.h>
    void main(){
    	int x;
    	x=11;
    	if (x>10) {
    		printf("yes, x is greater than 10");
    	}
    }
    ```

+   `if` (决策) {语句} `else` {语句}：`else`部分可以在`if`语句之后使用，并且当决策为假时，运行一个或多个语句可能很有用，例如：

    ```cpp
    #include <stdio.h>
    void main(){
      int x;
      x=5;
      if (x>10) {
         printf("yes, x is greater than 10");
      }
      else {
        printf("no, x is not greater than 10");
      }
    }
    ```

    在上述示例中，分析了 x 变量，如果 x 大于 10，它将在 IDE 控制台打印出此消息：`yes, x is greater than 10`，否则它将打印出`no, x is not greater than 10`。

    小贴士

    评估两个变量时请小心使用`if`语句。为此请使用双等号（==）。如果你只使用一个等号，编译器将引发错误。请这样做：`if` (x==y) {语句}

+   `switch`语句将变量的值与多个可能的值进行比较，这些值称为情况。`switch`语句中的每个情况都有一个唯一的名称（标识符）。如果在情况列表中没有找到匹配项，则将执行默认语句，并且程序控制将离开`switch`，并带有情况列表。可选的`break`语句用于在`switch`块之外终止程序控制。这在某些原因导致你不想让`switch`语句继续评估其余情况时很有用。以下是`switch`语句的语法：

```cpp
switch( expression_to_evaluate)
{
    case value1:
        <statement(s)>;
        break;
    case value_n:
        <statement(s)>;
        break;
}
```

上述代码显示了`switch`语句的语法，包括其`break`语句。以下代码是使用`switch`的示例，它将变量年龄与三个情况进行比较。如果变量的值为`10`，它将打印出以下文本：`the person is a child`：

```cpp
#include <stdio.h>
void main(){
	int age;
	age=10;
	switch (age)
	{
		case 10:
			printf ("the person is a child");
			break;
		case 30:
			printf ("the person is an adult");
			break;
		case 80:
			printf ("the person is a senior citizen");
			break;
	}	
}
```

到目前为止，我们已经回顾了如何逻辑地评估一个表达式。下一节将解释如何重复运行一个或多个语句。这对于微控制器板的一些重复性任务可能很有用，例如连续从输入微控制器端口读取数据。

### 理解循环

`for`、`while`和`do..while`：

#### for 循环

`for`循环在其测试表达式变为假之前重复其块中的一个或多个语句。这是`for`循环的语法：

```cpp
for (<initialization_variable>;         <test_expression_with_variable>; <update_variable>)
{
    <statement(s)_to_run>;
} 
```

在上述语法中，`counter_variable`的初始化只执行一次。然后，使用`counter_variable`评估表达式。如果测试的表达式为假，则循环终止。如果评估的表达式为真，则执行块语句，并更新`counter_variable`。`counter_variable`是局部变量，仅在`for`循环中有效。此示例在 IDE 控制台打印出从 1 到 10 的数字列表：

```cpp
for (int x=1; x<=10; x++)
{
    printf("%d ", x);
}
```

请注意，x++语句与写作 x=x+1 相同。

#### while 循环

`while`循环在给定条件为真时重复其块中的一个或多个语句，在执行语句之前测试其条件。当条件测试为假时，循环终止。以下是`while`循环语句的语法：

```cpp
while (<test_expression>) 
{
    statement(s); 
}
```

上述代码是`while`循环的语法。以下是一个使用`while`循环的示例代码，从 0 计数到 10：

```cpp
int x = 0;
while (x <= 10)
{
    // \n it will display the next number in a new 
    // line of text:
    printf("%d \n", x); 
    x=x+1;
}
```

#### do..while 循环：

这种循环与`while`循环非常相似。`do..while`循环至少执行一次其块语句。表达式在块末尾评估。过程将继续，直到评估的表达式为假。

以下为`do..while`循环的语法：

```cpp
do
{
    statement(s);
}
while (<test_expression>);
```

以下示例使用`do..while`循环，从 5 计数到 50，当总和小于 50 时：

```cpp
int number=5;
do
{
    number=number+5;
    printf("%d ", number);
}
while (number < 50);
```

在前面的代码中，名为`number`的变量值增加了 5，并且该变量至少在 IDE 的控制台上打印一次，然后变量被评估。

### 无限循环

你也可以编写一个**无限循环**，当然，它将无限期地运行（循环不会终止）直到我们终止程序（或从微控制器板上断开电源！）无限循环可以用于持续显示微控制器的结果，连续从微控制器板上读取数据而不停止，等等。

你可以使用三种类型的循环中的任何一种来做这件事。以下是一些无限循环的示例：

```cpp
for(; ;)
{
    printf("this text will be displayed endlessly!");
}
while(1) 
{
    printf("this text will be displayed endlessly!");
}
do
{
    printf("this text will be displayed endlessly!");
}
while (1);
```

如您从前面的代码中看到的，编写无限循环既简单又容易。

### 循环中的 break 和 continue 关键字

你可以使用`break`关键字。以下示例中的`break`语句将停止`for`循环，但该语句只会运行一次：

```cpp
for (int x=1; x<=10; x++)
{
    printf("%d ", x);
    break;
}
```

你可以在三种类型的循环中的任何一种中使用`break`语句。

`continue`关键字。以下示例将不会打印出第二行文本：

```cpp
for (int x=1; x<=10; x++)
{
    printf("%d ", x);
    continue;
    printf("this line won't be displayed.");
}
```

前面的代码由于`continue`语句而没有显示下一行文本，因为它将程序控制移动到`for`循环的开始。

下一节将处理一些专门为 Curiosity Nano 微控制器板创建的 C 语句和函数，这些语句和函数与 Blue Pill 板上的略有不同。

# 介绍 Curiosity Nano 微控制器板编程

如您在*第一章*中学习到的，*微控制器和微控制器板简介*，Curiosity Nano 可以使用本章中解释的 ANSI C 语言编程，使用 MPLAB X IDE。

Curiosity Nano 的 C 程序的基本结构与上述使用`main()`函数解释的类似，但其声明有所改变。你必须包含关键字 void，如下所示：

```cpp
//necessary IDE's library defining input-output ports:
#include "mcc_generated_files/mcc.h"
void main(void) //main program function
{
    // statements
}
```

书籍的 GitHub 页面中的文件`16F15376_Curiosity_Nano_IOPorts.zip`包含了必要的`IO_RD1_GetValue()`函数，该函数将从 Curiosity Nano 的 RD1 端口读取模拟值。

以下是一些有用的函数，您可以在编程 Curiosity Nano 时使用，这些函数已经由 MPLAB X 编译器定义。请注意，`xxx`表示 Curiosity Nano 的端口名称。请阅读*第一章*，*微控制器和微控制器板简介*，以熟悉 Curiosity Nano 的 I/O 端口名称及其相应的芯片引脚：

+   `IO_xxx_SetHigh();`: 这个函数在指定的引脚（端口）上写入逻辑高（3.3 V）值。

+   `IO_xxx_SetLow();`: 这个函数在指定的引脚（端口）上写入逻辑低（0 V）值。

+   `IO_xxx_GetValue();`: 这个函数返回从指定端口读取的逻辑（数字）值（要么是高要么是低）。高返回为 1，低返回为 0。

+   `ADC_GetConversion(xxx);`: 这个函数从指定的端口读取模拟值，并返回一个从 0 到 1023 的值，该值对应于对读取值的模拟-数字转换。

+   `SYSTEM_Initialize();`: 这个函数初始化微控制器端口。

+   `__delay_ms(number_milliseconds);`: 这个函数使程序暂停一定数量的毫秒（一秒中有 1,000 毫秒）。

+   `IO_xxx_Toggle();`: 这个函数将指定端口的值切换到其相反的状态。如果端口逻辑为高（1），则此函数将将其切换到 0，反之亦然。

我们将在本章后面解释的示例中用到一些前面的函数。

*图 2.1*显示了 Curiosity Nano 的引脚。请注意，其中许多是 I/O 端口：

![图 2.1 – Curiosity Nano 的引脚配置](img/Figure_2.1_B16413.jpg)

图 2.1 – Curiosity Nano 的引脚配置

我们已经将以下端口从 Curiosity Nano 微控制器板配置为 I/O 端口。我们在本书的所有 Curiosity Nano 软件项目文件中做了这件事。端口的引脚可以在*图 2.1*中看到。其中一些在本书中被广泛使用：

RA0, RA1, RA2, RA3, RA4, RA5, RB0, RB3, RB4, RB5, RC0, RC1, RC7, RD0, RD1, RD2, RD3, RD5, RD6, RD7, RE0, RE1 和 SW0。

下一个部分解释了 Blue Pill 板微控制器板编程的基本编程结构和重要函数，这些函数与 Curiosity Nano 板略有不同。

# 介绍 Blue Pill 微控制器板编程

如您在*第一章*，*微控制器和微控制器板简介*中学习的那样，您可以使用 Arduino IDE 编程 Blue Pill 板，同时安装 IDE 中的特殊库。请记住，这个 IDE 使用 C++语言，它是 C 语言的扩展。在 Arduino IDE 编程中，程序也被称为 sketch。所有 sketch 都必须有两个函数，称为`setup()`和`loop()`。

`setup()`函数用于定义变量、定义输入或输出端口（板引脚）、定义和打开串行端口等，并且此函数将只运行一次。它必须在`loop()`函数之前声明。

`loop()`函数是您代码的主要块，将运行程序的主要语句。这个`loop()`函数将反复无限地运行。草图不需要`main()`函数。

这是您草图（程序）的主要结构：

```cpp
void setup() 
{
    statement(s);
}
void loop() 
{
    statement(s);
}
```

这里是如何将引脚（微控制器板的端口）定义为输入或输出的示例：

```cpp
void setup ( ) 
{
 // it sets the pin as output.
    pinMode (pin_number1, OUTPUT);
 // it sets the pin as input 
    pinMode (pin_number2, INPUT); 
}
```

输入端口将用于从传感器或开关读取数据，输出端口将用于向其他设备或组件发送数据，打开 LED 等。

提示

在 Arduino IDE 中编程时区分大小写。在编写函数名、定义变量等时请小心。

如前述代码所示，每个语句块都包含在大括号内，每个语句都以分号结束，类似于 ANSI C。这些是有用的函数，可用于编程 Blue Pill：

+   `digitalWrite(pin_number, value);`：此函数在指定的引脚（端口）上写入高（3.3 V）或低（0 V）值；例如，`digitalWrite(13,HIGH);`将向引脚（端口）号 13 发送高值。

    注意

    您必须在`setup()`函数中先前声明`pin_number`为`OUTPUT`。

+   `digitalRead(pin_number);`：此函数返回从指定引脚（端口）读取的逻辑高（3.3 V）或逻辑低（0 V）值，例如，`val = digitalRead(pin_number);`。

    注意

    您必须在`setup()`函数中先前声明`pin_number`为`INPUT`。

+   `analogWrite(pin_number, value);`：此函数将（发送）一个模拟值（0..65535）到 Blue Pill 指定的 PIN（输出端口）。

+   `analogRead(pin_number);`：此函数返回从指定 PIN 读取的模拟值。Blue Pill 有 10 个通道（可以用于模拟输入的端口或引脚），12 位的`analogRead()`函数将输入电压在 0 到 3.3 伏之间映射到 0 到 4095 的整数之间，例如：

+   `int val = analogRead(A7);`

+   `delay(number_of_milliseconds);`：此函数使程序暂停指定的时间，以毫秒为单位定义（记住一秒有一千毫秒）。

    提示

    您还可以使用本节中解释的 C 语言结构来编程 Arduino 微控制器板，唯一的区别是`analogWrite()`的值范围将是 0...255 而不是 0...65535，`analogRead()`的值范围将是 0 到 1023 而不是 0 到 4095。

*图 2.2*显示了 Blue Pill 的 I/O 端口和其他引脚：

![图 2.2 – Blue Pill 的引脚配置](img/Figure_2.2_B16413.jpg)

图 2.2 – Blue Pill 的引脚配置

可以在 *图 2.2* 中看到端口的引脚。其中一些在本书的章节中被使用。Blue Pill 有以下模拟端口：A0, A1, A2, A3, A4, A5, A6, A7, B0 和 B1。以下为数字 I/O 端口：C13, C14, C15, B10, B11, B12, B13, B14, B15, A8, A9, A10, A11, A12, A15, B3, B4, B5, B6, B7, B8 和 B9。

只需记住，在代码中，端口被引用为 `PA0`、`PA1` 等，添加一个字母 `P`。

我们将在下一节的示例中使用一些前面的函数。

# 示例 – 编程和使用微控制器板内部 LED

在本节中，我们将使用 C/C++ 语言的常见语句来控制 Blue Pill 和 Curiosity Nano 板的内部 LED。内部 LED 可以非常方便地快速验证 I/O 端口的状态，显示传感器数据等，而无需将带有相应电阻的 LED 连接到端口。下一节将展示如何使用内部 LED 编译并发送代码到微控制器板。

## 编程 Blue Pill 的内部 LED

本节涵盖了编程内部 LED 的步骤。您不需要连接任何外部电子组件，例如外部 LED。使用 Blue Pill 的内部 LED 对于快速测试和显示程序的结果或变量值非常有用。您只需要使用微控制器板。以下步骤演示了如何上传和运行程序到 Blue Pill：

1.  按照在 *第一章* “微控制器和微控制器板简介”中解释的步骤，将 ST-LINK/V2 接口连接到 Blue Pill。

1.  将 USB 线缆连接到 Blue Pill 和您的计算机。将 Blue Pill 插入无焊点面包板。*图 2.3* 显示了 Curiosity Nano 和 Blue Pill 板的内部 LED：![图 2.3 – Blue Pill（顶部）和 Curiosity Nano 的内部 LED](img/Figure_2.3_B16413.jpg)

    图 2.3 – Blue Pill（顶部）和 Curiosity Nano 的内部 LED

1.  打开 Arduino IDE。在其编辑器中编写以下程序：

    ```cpp
    /*
      Blink
      This program turns on the Blue Pill's internal LED   on for one second, then off for two seconds,   repeatedly.
      Version number: 1.
      Date: Sept. 18, 2020.
      Note: the internal LED is internally connected to   port PC13.
      Written by Miguel Garcia-Ruiz.
     */
    void setup() 
    {
      pinMode(PC13, OUTPUT);
    }
    void loop() 
    {
      digitalWrite(PC13, HIGH);
      delay(1000);
      digitalWrite(PC13, LOW);
      delay(2000);             // it waits for two seconds
    }
    ```

1.  点击 `PC13` 以选择 `LED_BUILTIN`。

您可以将 Blue Pill 留在未插入无焊点面包板的状态，因为我们在此前的示例中没有将任何组件或线缆连接到 Blue Pill 的端口。

### 编程 Curiosity Nano 的内部 LED

与 Blue Pill 类似，您可以使用 Curiosity Nano 的内部 LED 快速显示传感器数据等，而无需将 LED 连接到端口。包含此示例和其他支持文件（这些文件对于在 MPLAB X IDE 上编译程序是必要的）的整个项目存储在 GitHub 页面上。它是一个名为 `16F15376_Curiosity_Nano_LED_Blink_Delay.zip` 的 zip 文件。

按照以下步骤在 MPLAB X IDE 上运行程序：

1.  将 USB 线缆连接到 Curiosity Nano 并将板插入无焊点面包板。解压 `16F15376_Curiosity_Nano_LED_Blink_Delay.zip` 文件。

1.  在 MPLAB X IDE 中，点击**文件/打开项目**然后打开项目。

1.  双击项目文件夹，然后点击`源文件`文件夹。

1.  点击`main.c`，你将看到以下源代码：

    ```cpp
    /*
    This program makes the on-board LED to blink once a second (1000 milliseconds).
    Ver. 1\. July, 2020\. Written by Miguel Garcia-Ruiz
    */
    //necessary library generated by MCC:
    #include "mcc_generated_files/mcc.h" 
    void main(void) //main program function
    {
        // initializing the microcontroller board:
        SYSTEM_Initialize(); 
        //it sets up LED0 as output: 
        LED0_SetDigitalOutput();
        while (1) //infinite loop
        {
            LED0_SetLow(); //it turns off the on-board LED
            __delay_ms(1000); //it pauses the program for                           //1 second 
            LED0_SetHigh(); //it turns on on-board LED and                         //RE0 pin
            __delay_ms(1000); //it pauses the program for                           //1 second 
        }
    }
    ```

1.  通过点击顶部的绿色运行图标来编译和运行代码，该图标位于菜单栏上。如果一切顺利，你将看到 Curiosity Nano 的内部 LED 闪烁。

如前例所示，它具有专门为 Curiosity Nano 板创建的实用 C 函数，例如以下内容：

`SetLow()`, `SetHigh()` 和 `__delay_ms()`。

这些功能对于使用微控制器板制作项目是必不可少的，并且它们被用于本书的其他章节。

# 摘要

在本章中，我们学习了如何正确配置和设置 MPLAB X 和 Arduino IDE 以进行 C 微控制器板编程。我们介绍了 C 编程语言，特别是用于编程 Blue Pill 和微控制器板的 C 语言指令集。为了练习你学到的 C 语言知识，我们查看了一些使用板内和板外 LED 的实际电路。本章中学到的指令和结构可以应用于本书的其余部分。

*第三章*，*使用按钮开关 LED*，将重点介绍如何将按钮与上拉电阻连接到微控制器板上，以及在使用按钮时如何最小化电气噪声。它还将解释如何通过软件设置微控制器板的输入端口，以及按钮的可能应用。

# 进一步阅读

+   Gay, W. (2018). *开始 STM32：使用 FreeRTOS、libopencm3 和 GCC 进行开发*。St. Catharines, ON: Apress。

+   Microchip Technology (2019). *MPLAB X IDE 用户指南.* 从[`ww1.microchip.com/downloads/en/DeviceDoc/50002027E.pdf`](https://ww1.microchip.com/downloads/en/DeviceDoc/50002027E.pdf)获取。
