# 使用元编程优化代码

在上一章中，我们讨论了使用延迟评估的优化技术，并使用了延迟过程、缓存技术和记忆化来使我们的代码运行得更快。在本章中，我们将使用**元编程**来优化代码，其中我们将创建一个将创建更多代码的代码。本章我们将讨论以下主题：

+   元编程简介

+   构建模板元编程的部分

+   将流程控制重构到模板元编程中

+   在编译时执行代码

+   模板元编程的优缺点

# 元编程简介

这样说最简单，即元编程是一种通过使用代码来创建代码的技术。实现元编程，我们编写一个计算机程序，它操作其他程序并将它们视为其数据。此外，模板是 C++中的编译时机制，它是**图灵完备**的，这意味着任何可以用计算机程序表示的计算都可以通过模板元程序在运行前以某种形式进行计算。它还大量使用递归和不可变变量。因此，在元编程中，我们创建在代码编译时运行的代码。

# 使用宏预处理代码

为了开始我们对元编程的讨论，让我们回到 ANSI C 编程语言流行的时代。为了简单起见，我们通过创建宏来使用 C 预处理器。C 参数化宏也称为**元函数**，是元编程的一个例子。考虑以下参数化宏：

```cpp
    #define MAX(a,b) (((a) > (b)) ? (a) : (b))

```

由于 C++编程语言对 C 语言有兼容性的缺点，我们可以使用我们的 C++编译器编译前面的宏。让我们创建代码来消费前面的宏，如下所示：

```cpp
    /* macro.cpp */
    #include <iostream>

    using namespace std;

    // Defining macro
    #define MAX(a,b) (((a) > (b)) ? (a) : (b))

    auto main() -> int
    {
      cout << "[macro.cpp]" << endl;

      // Initializing two int variables
      int x = 10;
      int y = 20;

      // Consuming the MAX macro
      // and assign the result to z variable
      int z = MAX(x,y);

      // Displaying the result
      cout << "Max number of " << x << " and " << y;
      cout << " is " << z << endl;

      return 0;
    }

```

正如我们在前面的`macro.cpp`代码中看到的，我们向`MAX`宏传递了两个参数，因为它是一个参数化宏，这意味着参数可以从用户那里获取。如果我们运行前面的代码，我们应该在控制台上看到以下输出：

![图片](img/c3415c09-ba2c-4b3c-9377-afd18f2d9d30.png)

正如我们在本章开头所讨论的，元编程是一种在编译时运行的代码。通过在前面的代码中使用宏，我们可以演示由`MAX`宏生成的新代码。预处理程序将在编译时解析宏并引入新代码。在编译时，编译器将按照以下方式修改代码：

```cpp
    auto main() -> int
    {
      // same code
      // ...

      int z = (((a) > (b)) ? (a) : (b)); // <-- Notice this section

      // same code
      // ...

      return 0;
    }

```

除了单行宏预处理器之外，我们还可以生成多行宏元函数。为了实现这一点，我们可以在行尾使用反斜杠字符。假设我们需要交换两个值。我们可以创建一个名为`SWAP`的参数化宏，并像以下代码那样使用它：

```cpp
    /* macroswap.cpp */
    #include <iostream>

    using namespace std;

    // Defining multi line macro
    #define SWAP(a,b) { \
      (a) ^= (b); \
      (b) ^= (a); \
      (a) ^= (b); \
    }

    auto main() -> int
    {
      cout << "[macroswap.cpp]" << endl;

      // Initializing two int variables
      int x = 10;
      int y = 20;

      // Displaying original variable value
      cout << "before swapping" << endl;
      cout << "x = " << x << ", y = " << y ;
      cout << endl << endl;

      // Consuming the SWAP macro
      SWAP(x,y);

      // Displaying swapped variable value
      cout << "after swapping" << endl;
      cout << "x = " << x << ", y = " << y;
      cout << endl;

      return 0;
    }

```

如前述代码所示，我们将创建一个多行预处理器宏，并在每行的末尾使用反斜杠字符。每次我们调用参数化的`SWAP`宏时，它就会被替换为宏的实现。如果我们运行前面的代码，控制台将显示以下输出：

![](img/502d5c3f-23b1-4998-b4c8-04ea9ef089bf.png)

现在我们对元编程有了基本了解，特别是在元函数方面，我们可以进一步探讨下一个主题。

我们在宏预处理器中的每个变量实现中都使用括号，因为预处理器只是用宏的实现来替换我们的代码。假设我们有一个以下的宏：

`MULTIPLY(a,b) (a * b)` 如果我们将数字作为参数传递，这不会是问题。然而，如果我们传递一个操作作为参数，就会出问题。例如，如果我们如下使用`MULTIPLY`宏：

`MULTIPLY(x+2,y+5);`

然后，编译器会将其替换为`(x+2*y+5)`。这是因为宏只是将`a`变量替换为`x + 2`表达式，将`b`变量替换为`y + 5`表达式，并添加任何额外的括号。由于乘法的优先级高于加法，我们将得到以下结果：

`(x+2y+5)`

这并不是我们预期的结果。因此，最好的方法是使用括号在每个参数变量中。

# 在标准库中剖析模板元编程

我们在第一章“深入现代 C++”中讨论了标准库，并在上一章中也对其进行了处理。C++语言提供的标准库主要是包含不完整函数的模板。然而，它将被用来生成完整的函数。模板元编程是 C++模板，在编译时生成 C++类型和代码。

让我们挑选标准库中的一个类——`Array`类。在`Array`类中，我们可以为它定义一个数据类型。当我们实例化数组时，编译器实际上会生成我们定义的数据类型的数组代码。现在，让我们尝试构建一个简单的`Array`模板实现，如下所示：

```cpp
    template<typename T>
    class Array
    {
      T element;
    };

```

然后，我们实例化`char`和`int`数组如下：

```cpp
    Array<char> arrChar;
    Array<int> arrInt;

```

编译器所做的是根据我们定义的数据类型创建这两个模板实现。尽管我们不会在代码中看到这一点，但编译器实际上创建了以下代码：

```cpp
    class ArrayChar
    {
      char element;
    };

    class ArrayInt
    {
      int element;
    };

    ArrayChar arrChar;
    ArrayInt arrInt;

```

如前述代码片段所示，模板元编程是在编译时创建另一段代码的代码。

# 构建模板元编程

在我们进一步讨论模板元编程之前，最好先讨论构建模板元编程的框架。模板元编程由四个因素组成——**类型**、**值**、**分支**和**递归**。在本主题中，我们将深入研究构成模板的因素。

# 在模板中向变量添加值

在本章的开头，当我们讨论宏预处理器时，我们讨论了元函数的概念。在宏预处理器中，我们明确地操作源代码；在这种情况下，宏（元函数）操作源代码。相比之下，我们在 C++模板元编程中处理类型。这意味着元函数是一个与类型一起工作的函数。因此，使用模板元编程的更好方法是尽可能只与类型参数一起工作。当我们谈论模板元编程中的变量时，实际上它不是一个变量，因为其上的值不能被修改。我们需要从变量中获取的是它的名称，这样我们就可以访问它。因为我们将以类型进行编码，所以命名值是`typedef`，如下面的代码片段所示：

```cpp
    struct ValueDataType
    {
      typedef int valueDataType;
    };

```

通过使用前面的代码，我们将`int`类型存储到`valueDataType`别名中，这样我们就可以通过`valueDataType`变量来访问数据类型。如果我们需要将值而不是数据类型存储到变量中，我们可以使用`enum`，这样它将成为`enum`本身的数据成员。如果我们想存储值，让我们看一下以下代码片段：

```cpp
    struct ValuePlaceHolder
    {
      enum 
       { 
        value = 1 
       };
    };

```

根据前面的代码片段，我们现在可以访问`value`变量以获取其值。

# 将函数映射到输入参数

我们可以将变量添加到模板元编程中。接下来，我们必须做的是检索用户参数并将它们映射到一个函数上。假设我们想要开发一个`Multiplexer`函数，该函数将乘以两个值，并且我们必须使用模板元编程。以下代码片段可以用来解决这个问题：

```cpp
    template<int A, int B>
    struct Multiplexer
    {
      enum 
      {
        result = A * B 
      };
    };

```

如前所述的代码片段所示，模板需要用户提供的两个参数`A`和`B`，并且它将通过乘以这两个参数来获取`result`变量的值。我们可以使用以下代码来访问结果变量：

```cpp
    int i = Multiplexer<2, 3>::result;

```

如果我们运行前面的代码片段，`i`变量将存储`6`，因为它将计算`2`乘以`3`。

# 根据条件选择正确的流程

当我们有多个函数时，我们必须根据某些条件选择一个而不是其他。我们可以通过提供`template`类的两个替代特化来构建条件分支，如下所示：

```cpp
    template<typename A, typename B>
    struct CheckingType
    {
      enum 
      { 
        result = 0 
      };
    };

    template<typename X>
    struct CheckingType<X, X>
    {
      enum 
      { 
        result = 1 
      };
    };

```

如前所述的`template`代码所示，我们有两个模板，它们的类型是`X`和`A`/`B`。当模板只有一个类型时，即`typename X`，这意味着我们比较的两个类型（`CheckingType <X, X>》）是相同的。否则，这两个数据类型是不同的。以下代码片段可以用来消费前两个模板：

```cpp
    if (CheckingType<UnknownType, int>::result)
    {
      // run the function if the UnknownType is int
    } 
    else 
    { 
      // otherwise run any function 
    }

```

正如我们可以在前面的代码片段中看到的那样，我们试图将`UnknownType`数据类型与`int`类型进行比较。`UnknownType`数据类型可能来自其他进程。然后，我们可以通过使用模板比较这两种类型来决定我们想要运行的下一个进程。

到这里，你可能想知道模板多道程序设计如何帮助我们进行代码优化。很快，我们将使用模板元编程来优化代码。然而，我们需要讨论其他一些事情，这些事情将巩固我们在模板多道程序设计方面的知识。现在，请耐心等待，继续阅读。

# 递归地重复过程

我们已经成功地将值和数据类型添加到模板中，然后创建了一个分支，根据当前条件决定下一个进程。在基本模板中，我们还需要考虑重复的过程。然而，由于模板中的变量是不可变的，我们无法迭代序列。相反，我们必须像在第四章“使用递归算法重复方法调用”中讨论的那样，递归地执行这个过程。

假设我们正在开发一个模板来计算阶乘值。我们必须做的第一件事是开发一个通用模板，将`I`值传递给函数，如下所示：

```cpp
    template <int I>
    struct Factorial
    {
      enum 
      { 
        value = I * Factorial<I-1>::value 
      };
    };

```

正如我们可以在前面的代码中看到的那样，我们可以通过运行以下代码来获取阶乘的值：

```cpp
    Factorial<I>::value;

```

在前面的代码中，`I`是一个整数。

接下来，我们必须开发一个模板来确保它不会陷入无限循环。我们可以创建以下模板，将其参数传递为零（`0`）：

```cpp
    template <>
    struct Factorial<0>
    {
      enum 
      { 
        value = 1 
      };
    };

```

现在我们有一对模板，它们将在编译时生成阶乘的值。以下是一个示例代码，用于在编译时获取`Factorial(10)`的值：

```cpp
    int main()
    {
      int fact10 = Factorial<10>::value;
    }

```

如果我们运行前面的代码，我们将得到`10`的阶乘结果`3628800`。

# 编译时选择类型

正如我们在前面的话题中讨论的那样，`type`是模板的一个基本部分。然而，我们可以根据用户的输入选择一个特定的类型。让我们创建一个模板，它可以决定在变量中应该使用哪种类型。下面的`types.cpp`代码将展示模板的实现：

```cpp
    /* types.cpp */
    #include <iostream>

    using namespace std;

 // Defining a data type
 // in template
 template<typename T>
 struct datatype
 {
 using type = T;
 };

    auto main() -> int
    {
      cout << "[types.cpp]" << endl;

      // Selecting a data type in compile time
      using t = typename datatype<int>::type;

      // Using the selected data type
      t myVar = 123;

      // Displaying the selected data type
      cout << "myVar = " << myVar;

      return 0;
    }

```

正如我们可以在前面的代码中看到的那样，我们有一个名为`datatype`的模板。这个模板可以用来选择传递给它的`type`。我们可以使用`using`关键字将变量赋值给`type`。从前面的`types.cpp`代码中，我们将`t`变量赋值给`datatype`模板中的`type`。现在`t`变量将是`int`，因为我们向模板传递了`int`数据类型。

我们也可以创建一个代码来根据当前条件选择正确的数据类型。我们将有一个`IfElseDataType`模板，它接受三个参数，分别是`predicate`、当`predicate`参数为真时的数据类型以及当`predicate`参数为假时的数据类型。代码如下：

```cpp
    /* selectingtype.cpp */
    #include <iostream>

    using namespace std;

    // Defining IfElseDataType template
    template<
      bool predicate,
      typename TrueType,
      typename FalseType>
      struct IfElseDataType
      {
      };

    // Defining template for TRUE condition
    // passed to 'predicate' parameter
    template<
      typename TrueType,
      typename FalseType>
      struct IfElseDataType<
       true,
       TrueType,
       FalseType>
       {
         typedef TrueType type;
       };

    // Defining template for FALSE condition
    // passed to 'predicate' parameter
    template<
      typename TrueType,
      typename FalseType>
      struct IfElseDataType<
      false,
      TrueType,
      FalseType>
      {
         typedef FalseType type;
      };

    auto main() -> int
    {
      cout << "[types.cpp]" << endl;

      // Consuming template and passing
      // 'SHRT_MAX == 2147483647'
      // It will be FALSE
      // since the maximum value of short
      // is 32767
      // so the data type for myVar
      // will be 'int'
      IfElseDataType<
        SHRT_MAX == 2147483647,
        short,
        int>::type myVar;

      // Assigning myVar to maximum value
      // of 'short' type
      myVar = 2147483647;

      // Displaying the data type of myVar
      cout << "myVar has type ";
      cout << typeid(myVar).name() << endl;

      return 0;
    }

```

现在，通过拥有`IfElseDataType`模板，我们可以根据我们拥有的条件选择正确的变量类型。假设我们想将`2147483647`赋值给一个变量，这样我们就可以检查它是否是一个短数。如果是，`myVar`将是`short`类型，否则，它将是`int`类型。此外，由于`short`类型的最大值是`32767`，通过将谓词设置为`SHRT_MAX == 2147483647`将得到`FALSE`。因此，`myVar`的类型将是`int`类型，正如我们可以在以下控制台输出的以下输出中看到：

![](img/c9f65a57-9612-429f-82b7-55741f74cc50.png)

# 使用模板元编程进行流程控制

代码流程是编写程序的一个重要方面。在许多编程语言中，它们都有`if-else`、`switch`和`do-while`语句来安排代码的流程。现在，让我们重构通常的代码流程，使其成为基于模板的流程。我们将从使用`if-else`语句开始，然后是`switch`语句，最后以`do-while`语句结束，所有这些都在模板中完成。

# 根据当前条件决定下一个流程

现在是时候使用我们之前讨论过的模板了。假设我们有两个函数，我们必须根据某个条件来选择。我们通常的做法是使用如下`if-else`语句：

```cpp
    /* condition.cpp */
    #include <iostream>

    using namespace std;

    // Function that will run
    // if the condition is TRUE
    void TrueStatement()
    {
      cout << "True Statement is run." << endl;
    }

    // Function that will run
    // if the condition is FALSE
    void FalseStatement()
    {
      cout << "False Statement is run." << endl;
    }

    auto main() -> int
    {
      cout << "[condition.cpp]" << endl;

      // Choosing the function
      // based on the condition
      if (2 + 3 == 5)
        TrueStatement();
      else
        FalseStatement();

      return 0;
    }

```

如前所述的代码所示，我们有两个函数——`TrueStatement()`和`FalseStatement()`。我们还在代码中有一个条件——`2 + 3 == 5`。由于条件是`TRUE`，因此`TrueStatement()`函数将被运行，正如我们可以在以下屏幕截图中所看到的：

![](img/9e5c6760-bbad-42e6-9208-48d2213aca67.png)

现在，让我们重构前面的`condition.cpp`代码。在这里，我们将创建三个模板。首先，模板初始化将条件输入如下：

```cpp
    template<bool predicate> class IfElse

```

然后，我们为每个条件创建两个模板——`TRUE`或`FALSE`。名称将如下：

```cpp
    template<> class IfElse<true>
    template<> class IfElse<false> 

```

前面的代码片段中的每个模板都将运行我们之前创建的函数——`TrueStatement()`和`FalseStatement()`函数。我们将得到以下`conditionmeta.cpp`代码的完整代码：

```cpp
    /* conditionmeta.cpp */
    #include <iostream>

    using namespace std;

    // Function that will run
    // if the condition is TRUE
    void TrueStatement()
    {
      cout << "True Statement is run." << endl;
    }

    // Function that will run
    // if the condition is FALSE
    void FalseStatement()
    {
      cout << "False Statement is run." << endl;
    }

    // Defining IfElse template
    template<bool predicate>
    class IfElse
    {
    };

    // Defining template for TRUE condition
    // passed to 'predicate' parameter
    template<>
    class IfElse<true>
    {
      public:
        static inline void func()
        {
          TrueStatement();
        }
    };

    // Defining template for FALSE condition
    // passed to 'predicate' parameter
    template<>
    class IfElse<false>
    {
      public:
        static inline void func()
        {
          FalseStatement();
        }
    };

    auto main() -> int
    {
      cout << "[conditionmeta.cpp]" << endl;

      // Consuming IfElse template
      IfElse<(2 + 3 == 5)>::func();

      return 0;
    }

```

如我们所见，我们将条件放在`IfElse`模板的括号中，然后在模板内调用`func()`方法。如果我们运行`conditionmeta.cpp`代码，我们将得到与`condition.cpp`代码完全相同的输出，如下所示：

![](img/52dfab6d-8e80-4036-9e19-95f40d013725.png)

我们现在有了`if-else`语句来在模板元编程中控制代码的流程。

# 选择正确的语句

在 C++编程以及其他编程语言中，我们使用`switch`语句根据我们提供给`switch`语句的值来选择某个流程。如果值与某个`switch` case 的值匹配，它将运行该 case 下的流程。让我们看看以下实现`switch`语句的`switch.cpp`代码：

```cpp
    /* switch.cpp */
    #include <iostream>

    using namespace std;

    // Function to find out
    // the square of an int
    int Square(int a)
    {
      return a * a;
    }

    auto main() -> int
    {
      cout << "[switch.cpp]" << endl;

      // Initializing two int variables
      int input = 2;
      int output = 0;

      // Passing the correct argument
      // to the function
      switch (input)
      {
        case 1:
            output = Square(1);
            break;
        case 2:
            output = Square(2);
            break;
        default:
            output = Square(0);
            break;
      }

      // Displaying the result
      cout << "The result is " << output << endl;

      return 0;
    }

```

如前一个代码所示，我们有一个名为`Square()`的函数，它接受一个参数。我们传递给它的参数是基于我们给`switch`语句的值。由于我们传递给`switch`的值是`2`，因此将运行`Square(2)`方法。以下屏幕截图是我们将在控制台屏幕上看到的：

![](img/bb934a6e-293e-47b9-b771-cb7d9da4832f.png)

要将`switch.cpp`代码重构为模板元编程，我们必须创建三个模板，这些模板包含我们计划运行的函数。首先，我们将创建初始化模板以从用户那里检索值，如下所示：

```cpp
    template<int val> class SwitchTemplate 

```

前面的初始化模板也将用于默认值。接下来，我们将为每个可能的值添加两个模板，如下所示：

```cpp
    template<> class SwitchTemplate<1>
    template<> class SwitchTemplate<2> 

```

每个前面的模板都会运行`Square()`函数，并根据模板的值传递参数。完整的代码如下：

```cpp
    /* switchmeta.cpp */
    #include <iostream>

    using namespace std;

    // Function to find out
    // the square of an int
    int Square(int a)
    {
      return a * a;
    }

    // Defining template for
    // default output
    // for any input value
    template<int val>
    class SwitchTemplate
    {
      public:
        static inline int func()
        {
          return Square(0);
        }
    };

    // Defining template for
    // specific input value
    // 'val' = 1
    template<>
    class SwitchTemplate<1>
    {
       public:
         static inline int func()
         {
           return Square(1);
         }
    };

    // Defining template for
    // specific input value
    // 'val' = 2
    template<>
    class SwitchTemplate<2>
    {
       public:
         static inline int func()
         {
            return Square(2);
         }
    };

    auto main() -> int
    {
      cout << "[switchmeta.cpp]" << endl;

      // Defining a constant variable
      const int i = 2;

      // Consuming the SwitchTemplate template
      int output = SwitchTemplate<i>::func();

      // Displaying the result
      cout << "The result is " << output << endl;

      return 0;
    }

```

如前所述，我们与`conditionmeta.cpp`做同样的事情--我们在模板内部调用`func()`方法来运行选定的函数。这个`switch-case`条件的值是我们放在尖括号中的模板。如果我们运行前面的`switchmeta.cpp`代码，我们将在控制台上看到以下输出：

![](img/eeb896aa-2598-4995-b3b0-bbb61386f762.png)

如前一个屏幕截图所示，与`switch.cpp`代码相比，我们得到了`switchmeta.cpp`代码的完全相同的输出。因此，我们已经成功地将`switch.cpp`代码重构为模板元编程。

# 循环过程

当我们需要迭代某些内容时，通常使用`do-while`循环。假设我们需要打印某些数字，直到它们达到零（`0`）。代码如下：

```cpp
    /* loop.cpp */
    #include <iostream>

    using namespace std;

    // Function for printing
    // given number
    void PrintNumber(int i)
    {
      cout << i << "\t";
    }

    auto main() -> int
    {
      cout << "[loop.cpp]" << endl;

      // Initializing an int variable
      // marking as maximum number
      int i = 100;

      // Looping to print out
      // the numbers below i variable
      cout << "List of numbers between 100 and 1";
      cout << endl;
      do
      {
        PrintNumber(i);
      }
      while (--i > 0);
      cout << endl;

      return 0;
    }

```

如前一个代码所示，我们将打印数字`100`，减少其值，然后再次打印。它将一直运行，直到数字达到零（`0`）。控制台上的输出应该如下所示：

![](img/1a84c23b-dd08-4dfc-99a6-273c36918e06.png)

现在，让我们将其重构为模板元编程。在这里，我们只需要两个模板就可以在模板元编程中实现`do-while`循环。首先，我们将创建以下模板：

```cpp
    template<int limit> class DoWhile

```

前面的代码中的限制是传递给`do-while`循环的值。为了不使循环变成无限循环，我们必须在它达到零（`0`）时设计`DoWhile`模板，如下所示：

```cpp
    template<> class DoWhile<0>

```

前面的模板将不执行任何操作，因为它仅用于中断循环。`do-while`循环的完整重构如下`loopmeta.cpp`代码：

```cpp
    /* loopmeta.cpp */
    #include <iostream>

    using namespace std;

    // Function for printing
    // given number
    void PrintNumber(int i)
    {
      cout << i << "\t";
    }

    // Defining template for printing number
    // passing to its 'limit' parameter
    // It's only run
    // if the 'limit' has not been reached
    template<int limit>
    class DoWhile
    {
       private:
         enum
         {
           run = (limit-1) != 0
         };

       public:
         static inline void func()
         {
           PrintNumber(limit);
           DoWhile<run == true ? (limit-1) : 0>
            ::func();
         }
    };

    // Defining template for doing nothing
    // when the 'limit' reaches 0
    template<>
    class DoWhile<0>
    {
      public:
        static inline void func()
        {
        }
    };

    auto main() -> int
    {
      cout << "[loopmeta.cpp]" << endl;

      // Defining a constant variable
      const int i = 100;

      // Looping to print out
      // the numbers below i variable
      // by consuming the DoWhile
      cout << "List of numbers between 100 and 1";
      cout << endl;
      DoWhile<i>::func();
      cout << endl;

      return 0;
    }

```

我们然后在模板内部调用`func()`方法来运行我们想要的函数。如果我们运行代码，我们将在屏幕上看到以下输出：

![](img/b5fd496c-ddac-483e-b0db-523307b3c91e.png)

再次，我们已经成功地将`loop.cpp`代码重构为`loopmeta.cpp`代码，因为它们具有完全相同的输出。

# 在编译时执行代码

正如我们之前讨论的，模板元编程将通过创建新代码在编译时运行代码。现在，让我们看看我们如何在本节中获取编译时常量并生成编译时类。

# 获取编译时常量

要检索编译时常量，让我们创建一个包含斐波那契算法模板的代码。我们将消耗模板，以便编译器在编译时提供值。代码应如下所示：

```cpp
    /* fibonaccimeta.cpp */
    #include <iostream>

    using namespace std;

    // Defining Fibonacci template
    // to calculate the Fibonacci sequence
    template <int number>
    struct Fibonacci
    {
      enum
      {
        value =
            Fibonacci<number - 1>::value +
            Fibonacci<number - 2>::value
      };
    };

    // Defining template for
    // specific input value
    // 'number' = 1
    template <>
    struct Fibonacci<1>
    {
      enum
      {
        value = 1
      };
    };

    // Defining template for
    // specific input value
    // 'number' = 0
    template <>
    struct Fibonacci<0>
    {
      enum
      {
        value = 0
      };
    };

    auto main() -> int
    {
      cout << "[fibonaccimeta.cpp]" << endl;

      // Displaying the compile-time constant
      cout << "Getting compile-time constant:";
      cout << endl;
      cout << "Fibonacci(25) = ";
      cout << Fibonacci<25>::value;
      cout << endl;

      return 0;
    }

```

如前所述的代码所示，斐波那契模板中的值变量将提供一个编译时常量。如果我们运行前面的代码，我们将在控制台屏幕上看到以下输出：

![](img/015e83df-4905-4a3c-aa49-c8f233a2282c.png)

现在，我们有了由编译器生成的编译时常量 `75025`。

# 使用编译时类生成生成类

除了生成编译时常量外，我们还将生成编译时的类。假设我们有一个模板，用于在 `0` 到 `X` 的范围内找出素数。以下 `isprimemeta.cpp` 代码将解释模板元编程的实现，以找出素数：

```cpp
    /* isprimemeta.cpp */
    #include <iostream>

    using namespace std;

    // Defining template that decide
    // whether or not the passed argument
    // is a prime number
    template <
      int lastNumber,
      int secondLastNumber>
    class IsPrime
    {
      public:
        enum
        {
          primeNumber = (
            (lastNumber % secondLastNumber) &&
            IsPrime<lastNumber, secondLastNumber - 1>
                ::primeNumber)
        };
     };

    // Defining template for checking
    // the number passed to the 'number' parameter
    // is a prime number
    template <int number>
    class IsPrime<number, 1>
    {
      public:
        enum
        {
          primeNumber = 1
        };
    };

    // Defining template to print out
    // the passed argument is it's a prime number
    template <int number>
    class PrimeNumberPrinter
    {
      public:
        PrimeNumberPrinter<number - 1> printer;

      enum
      {
        primeNumber = IsPrime<number, number - 1>
            ::primeNumber
      };

      void func()
      {
        printer.func();

        if (primeNumber)
        {
            cout << number << "\t";
        }
      }
    };

    // Defining template to just ignoring the number
    // we pass 1 as argument to the parameter
    // since 1 is not prime number
    template<>
    class PrimeNumberPrinter<1>
    {
      public:
        enum
        {
          primeNumber = 0
        };

        void func()
        {
        }
    };

    int main()
    {
      cout << "[isprimemeta.cpp]" << endl;

      // Displaying the prime numbers between 1 and 500
      cout << "Filtering the numbers between 1 and 500 ";
      cout << "for of the prime numbers:" << endl;

      // Consuming PrimeNumberPrinter template
      PrimeNumberPrinter<500> printer;

      // invoking func() method from the template
      printer.func();

      cout << endl;
      return 0;
    }

```

有两种模板，具有不同的角色--**素数检查器**，确保传递的数字是一个素数，以及**打印器**，将素数显示到控制台。当代码访问 `PrimeNumberPrinter<500> printer` 和 `printer.func()` 时，编译器将在编译时生成类。当我们运行前面的 `isprimemeta.cpp` 代码时，我们将在控制台屏幕上看到以下输出：

![](img/1377d14f-9287-4e54-93b8-39bf5906fe43.png)

由于我们将 `500` 传递给模板，我们将从 `0` 到 `500` 获取素数。前面的输出已经证明编译器已成功生成编译时类，因此我们可以获取正确的值。

# 元编程的优缺点

在我们讨论模板元编程之后，以下是我们从中获得的优点：

+   模板元编程没有副作用，因为它是不变的，所以我们不能修改现有的类型

+   与未实现元编程的代码相比，代码的可读性更好

+   它减少了代码的重复性

尽管我们可以从模板元编程中获得好处，但也有一些缺点，如下所示：

+   语法相当复杂。

+   编译时间变长，因为我们现在在编译时执行代码。

+   编译器可以更好地优化生成的代码并执行内联，例如，C 语言的`qsort()`函数和 C++的`sort`模板。在 C 语言中，`qsort()`函数接受一个比较函数的指针，因此将有一个不内联的`qsort`代码副本。它将通过指针调用比较例程。在 C++中，`std::sort`是一个模板，它可以接受一个作为比较器的`functor`对象。对于用作比较器的每种不同类型，都有一个不同的`std::sort`副本。如果我们使用一个具有重载`operator()`函数的`functor`类，比较器的调用可以很容易地内联到这个`std::sort`副本中。

# 摘要

元编程，尤其是模板元编程，可以自动为我们创建新代码，因此我们不需要在我们的源代码中编写很多代码。通过使用模板元编程，我们可以重构代码的流程控制，以及在编译时执行代码。

在下一章中，我们将讨论并发技术，这些技术将使我们所构建的应用程序变得更加响应。我们可以使用并行技术同时运行代码中的进程。
