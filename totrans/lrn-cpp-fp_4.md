# 使用递归算法重复方法调用

在上一章中，你学习了不可变状态，这些状态使我们不必处理副作用。在这一章中，让我们看看递归的概念。作为面向对象编程的程序员，我们通常使用迭代来重复过程而不是递归。然而，递归比迭代有更多的优势。例如，一些问题（尤其是数学）可以通过递归轻松解决，幸运的是，所有算法都可以递归定义。这使得可视化证明变得容易得多。为了了解更多关于递归的知识，本章将讨论以下主题：

+   区分迭代和递归调用

+   重复不可变函数

+   在递归中使用尾递归找到更好的方法

+   列举三种递归类型--函数式递归、过程式递归和回溯递归

# 递归重复函数调用

作为程序员，尤其是在面向对象编程中，我们通常使用迭代技术来重复我们的过程。现在，我们将讨论递归方法来重复我们的过程，并在函数式方法中使用它。基本上，递归和迭代执行相同的任务，即逐步解决复杂任务然后组合结果。然而，它们有一个区别。迭代过程强调我们应该重复过程直到任务完成，而递归强调需要将任务分解成更小的部分，直到我们可以解决任务，然后组合结果。当我们需要运行某个过程直到达到限制或读取流直到达到`eof()`时，我们可以使用迭代过程。此外，递归可以在我们使用它时提供最佳值，例如，在计算阶乘时。

# 执行迭代过程以重复该过程

我们将从迭代过程开始。正如我们之前讨论的，如果使用递归方法设计，计算阶乘将更好。然而，也可以使用迭代方法来设计它。在这里，我们将有一个`factorial_iteration_do_while.cpp`代码，我们可以用它来计算阶乘。我们将有一个名为`factorial()`的函数，它传递一个参数，将计算我们传递给参数的阶乘值。代码应如下所示：

```cpp
    /* factorial_iteration_do_while.cpp */
    #include <iostream>

    using namespace std;

    // Function containing
    // do-while loop iteration

    int factorial (int n)
    {
      int result = 1;
      int i = 1;

      // Running iteration using do-while loop
      do
       {
         result *= i;
       }
       while(++i <= n);

       return result;
    }

    auto main() -> int
    {
      cout << "[factorial_iteration_do_while.cpp]" << endl;

      // Invoking factorial() function nine times
      for(int i = 1; i < 10; ++i)
      {
        cout << i << "! = " << factorial(i) << endl;
      }

      return 0;
    } 

```

如前述代码所示，我们依赖于`n`的值，这是我们传递给`factorial()`函数的值，以确定将发生多少次迭代。每次迭代执行时，`result`变量将被乘以计数器`i`。最后，`result`变量将通过结合迭代的最终结果值来持有最后的值。我们应在屏幕上获得的输出如下：

![](img/444549f8-8a38-48bb-b1e5-2be38e9f7bee.png)

迭代中的另一种技术是使用另一个迭代过程。我们可以重构前面的代码，在`factorial()`函数中使用`for`循环。以下是从我们前面的`factorial_iteration_do_while.cpp`代码重构的`factorial_iteration_for.cpp`代码：

```cpp
    /* factorial_iteration_do_while.cpp */
    #include <iostream>

    using namespace std;

    // Function containing
    // for loop iteration
    int factorial (int n)
    {
      int result = 1;

      // Running iteration using for loop
 for(int i = 1; i <= n; ++i)
 {
 result *= i;
 }

      return result;
     }

     auto main() -> int
     {
      cout << "[factorial_iteration_for.cpp]" << endl;

      // Invoking factorial() function nine times
      for(int i = 1; i < 10; ++i)
       {
         cout << i << "! = " << factorial(i) << endl;
       }

      return 0;
    }

```

如我们所见，我们将`do-while`循环替换为`for`循环。然而，程序的行为将完全相同，因为它每次迭代时也会将当前结果与`i`计数器相乘。在这个迭代的末尾，我们将从这个乘法过程中获得最终结果。屏幕应该显示以下输出：

![](img/c604f1cb-6277-4b29-a074-4bbf92bd02bc.png)

现在我们已经成功执行了迭代以获得阶乘目的，可以使用`do-while`循环或`for`循环。

当我们尝试将`do-while`循环重构为`for`循环时，这似乎太微不足道了。正如我们所知，`for`循环允许我们在知道要运行多少次循环时运行循环，而`do-while`循环则提供了更多的灵活性，例如`while(i > 0)`或使用布尔值如`while(true)`。然而，基于前面的例子，我们现在可以说我们可以将`for`循环或`do-while`循环转换为递归。

# 执行递归过程以重复该过程

我们之前讨论过，递归在函数式编程中提供更好的性能。我们也用迭代方法开发了`factorial()`函数。现在，让我们将之前的代码重构为`factorial_recursion.cpp`，它将使用递归方法而不是迭代方法。代码将执行与之前代码相同的任务。然而，我们将修改`factorial()`函数，使其在函数末尾调用自身。代码如下所示：

```cpp
    /* factorial_recursion.cpp */
    #include <iostream>

    using namespace std;

    int factorial(int n)
    {
      // Running recursion here
      if (n == 0)
        return 1;
      else
        return n * factorial (n - 1);
    }

    auto main() -> int
    {
       cout << "[factorial_recursion.cpp]" << endl;

      for(int i = 1; i < 10; ++i)
      {
        cout << i << "! = " << factorial(i) << endl;
      }

      return 0;
    }

```

如我们所见，前面的代码中的`factorial()`函数会一直调用自身，直到`n`为`0`。每次函数调用自身时，它都会递减`n`参数。一旦传递的参数为`0`，函数将很快返回`1`。我们也将得到与之前两个代码块相同的输出，如下面的截图所示：

![](img/b9c6c1c3-250a-4a02-961f-1ae358de0e60.png)

虽然递归给我们提供了易于维护代码所需的简单性，但我们必须注意传递给递归函数的参数。例如，在`factorial_recursion.cpp`代码中的`factorial()`函数中，如果我们传递负数给`n < 0`函数，我们将得到无限循环，并且它可能会使我们的设备崩溃。

# 递归不可变函数

正如我们在上一章中讨论的，我们需要递归地循环不可变函数。假设我们有一个不可变的`fibonacci()`函数。然后我们需要将其重构为递归函数。`fibonacci_iteration.cpp`代码以迭代方式实现了`fibonacci()`函数。代码如下所示：

```cpp
    /* fibonacci_iteration.cpp */
    #include <iostream>

    using namespace std;

    // Function for generating
    // Fibonacci sequence using iteration
    int fibonacci(int n)
    {
      if (n == 0)
        return 0;

      int previous = 0;
      int current = 1;

      for (int i = 1; i < n; ++i)
      {
        int next = previous + current;
        previous = current;
        current = next;
      }

      return current;
    }

    auto main() -> int
    {
      cout << "[fibonacci_iteration.cpp]" << endl;

      // Invoking fibonacci() function ten times
      for(int i = 0; i < 10; ++i)
       {
         cout << fibonacci(i) << " ";
       }
      cout << endl;

      return 0;
    }

```

如我们所见，前面的代码中`fibonacci()`函数是不可变的，因为它每次接收到完全相同的`n`输入时都会返回相同的值。输出应该看起来像以下截图：

![](img/bb62a358-6e52-44fb-9628-d55839169d48.png)

如果我们需要将其重构为递归函数，我们可以使用以下`fibonacci_recursion.cpp`代码：

```cpp
    /* fibonacci_recursion.cpp */
    #include <iostream>

    using namespace std;

    // Function for generating
    // Fibonacci sequence using recursion
    int fibonacci(int n)
    {
      if(n <= 1)
        return n;

      return fibonacci(n-1) + fibonacci(n-2);
    }

    auto main() -> int
    {
      cout << "[fibonacci_recursion.cpp]" << endl;

      // Invoking fibonacci() function ten times
      for(int i = 0; i < 10; ++i)
      {
        cout << fibonacci(i) << " ";
      }
      cout << endl;

      return 0;
    }

```

如我们所见，前面的代码具有递归方法，因为它在函数的末尾调用自身。现在我们有了递归的`fibonacci()`函数，它将在控制台上给出以下输出：

![](img/cb853f00-201a-493a-b561-5eed14833299.png)

现在，与`fibonacci_iteration.cpp`代码相比，`fibonacci_recursion.cpp`代码显示了完全相同的输出。

# 接近尾递归

当递归调用在函数的末尾执行时，发生尾递归。它被认为比我们之前开发的非尾递归代码更好，因为编译器可以更好地优化代码。由于递归调用是函数执行的最后一个语句，这个函数就没有其他事情可做了。结果是，编译器不需要保存当前函数的栈帧。让我们看看以下实现尾递归的`tail_recursion.cpp`代码：

```cpp
    /* tail_recursion.cpp */
    #include <iostream>

    using namespace std;

    void displayNumber(long long n)
    {
      // Displaying the current n value
      cout << n << endl;

      // The last executed statement 
      // is the recursive call
      displayNumber(n + 1);
    }

    auto main() -> int
    {
      cout << "[tail_recursion.cpp]" << endl;

      // Invoking the displayNumber() function
      // containing tail recursion
      displayNumber(0);

      return 0;
    }

```

如我们所见，前面的代码中`displayNumber()`函数是一个尾递归调用函数，因为它在过程的末尾调用自身。实际上，如果我们运行前面的`tail_recursion.cpp`代码，程序将不会结束，因为它会在`displayNumber()`函数中增加`n`的值。当`n`的值达到`long long`数据类型的最大值时，程序可能会崩溃。然而，由于尾递归不存储栈中的值，程序将不会出现栈溢出问题。

此外，我们还可以将`tail_recursion.cpp`代码中的前一个`displayNumber()`函数重构，使用`goto`关键字而不是反复调用函数。重构后的代码可以在下面的`tail_recursion_goto.cpp`代码中看到：

```cpp
    /* tail_recursion_goto.cpp */
    #include <iostream>

    using namespace std;

    void displayNumber(long long n)
    {
 loop:
        // Displaying the current n value
        cout << n << endl;

       // Update parameters of recursive call
 // and replace recursive call with goto
 n++;
 goto loop;
    }

    auto main() -> int
    {
      cout << "[tail_recursion_goto.cpp]" << endl;

      // Invoking the displayNumber() function
      // containing tail recursion
      displayNumber(0);

      return 0;
    }

```

如我们所见，在前面的代码中，我们可以使用`goto`关键字删除`displayNumber()`函数中的最后一个调用。这就是编译器通过执行尾调用消除来优化尾递归的方式，将最后一个调用替换为`goto`关键字。我们还将看到在`displayNumber()`函数中不需要栈。

不要忘记使用编译器提供的优化选项编译包含尾递归的代码。由于我们使用 GCC，始终启用优化级别 2（`-O2`）以获得优化代码。如果不启用优化编译，我们的前两个程序（`tail_recursion.cpp`和`tail_recursion_goto.cpp`）将因栈溢出而崩溃。有关 GCC 中优化选项的更多信息，请参阅[`gcc.gnu.org/onlinedocs/gcc-7.1.0/gcc/Optimize-Options.html`](https://gcc.gnu.org/onlinedocs/gcc-7.1.0/gcc/Optimize-Options.html)。

现在，让我们创建一个有用的尾递归调用。在前一节中，我们已经成功地将迭代函数重构为递归函数。`factorial()`函数现在已成为递归函数，并在函数的末尾调用自身。然而，它不是尾递归，尽管它在函数的末尾调用自身。如果我们仔细观察，`factorial(n-1)`返回的值被用于`factorial(n)`中，因此`factorial(n-1)`的调用不是`factorial(n)`完成的最后一件事。

我们可以创建我们的`factorial_recursion.cpp`代码，使其成为尾递归函数。我们将开发以下`factorial_recursion_tail.cpp`代码，修改`factorial()`函数，并添加一个名为`factorialTail()`的新函数。代码如下：

```cpp
    /* factorial_recursion_tail.cpp */
    #include <iostream>

    using namespace std;

 // Function for calculating factorial
 // tail recursion
 int factorialTail(int n, int i)
 {
 if (n == 0)
 return i;

 return factorialTail(n - 1, n * i);
 } 
 // The caller of tail recursion function
 int factorial(int n)
 {
 return factorialTail(n, 1);
 }

    auto main() -> int
    {
      cout << "[factorial_recursion_tail.cpp]" << endl;

      // Invoking fibonacci() function ten times
      for(int i = 1; i < 10; ++i)
      {
        cout << i << "! = " << factorial(i) << endl;
      }

     return 0;
    }

```

如我们所见，我们将`factorial_recursion.cpp`代码中的`factorial()`函数移动到了`factorial_recursion_tail.cpp`代码中的`factorialTail()`函数，该函数需要两个参数。因此，在调用`factorial(i)`之后，它将调用`factorialTail()`函数。在这个函数的末尾，`factorialTail()`函数是唯一被调用的函数。以下图像是`factorial_recursion_tail.cpp`代码的输出，它与`factorial_recursion.cpp`代码完全相同。这也证明了我们已经成功地将`factorial_recursion.cpp`代码重构为尾递归。

![](img/de6f4d4b-d59c-40cc-8ed7-f041a58a3d1b.png)

# 了解函数递归、过程递归和回溯递归

既然我们已经对递归有了一定的了解，递归函数将在其体内调用自身。递归只有在达到某个特定值时才会停止。我们将立即讨论三种递归类型——**函数递归**、**过程递归**和**回溯递归**；然而，这三种递归类型可能不是标准术语。函数递归是一种返回某些值的递归过程。过程递归是一种不返回值，但在每次递归中执行操作的递归过程。回溯递归是一种将任务分解成一组小子任务的过程，如果这些子任务不起作用，则可以取消它们。让我们在接下来的讨论中考虑这些递归类型。

# 期望函数递归的结果

在函数递归中，过程试图通过组合子问题的结果来解决问题。我们组合的结果来自子问题的返回值。假设我们有一个计算一个数的幂的问题，例如，`2` 的 `2` 次幂是 `4` (`2² = 4`)。通过使用迭代，我们可以构建如下 `exponential_iteration.cpp` 代码。我们有一个名为 `power()` 的函数，它将通过两个参数传递--`base` 和 `exp`。表示法将是 `base^(exp)`，代码如下所示：

```cpp
    /* exponential_iteration.cpp */
    #include <iostream>

    using namespace std;

    // Calculating the power of number
    // using iteration
    int power(int base, int exp)
    {
      int result = 1;

      for(int i = 0; i < exp; ++i)
       {
         result *= base;
       }

       return(result);
    } 

    auto main() -> int
    {
      cout << "[exponential_iteration.cpp]" << endl;

      // Invoking power() function six times
      for(int i = 0; i <= 5; ++i)
      {
        cout << "power (2, " << i << ") = ";
        cout << power(2, i) << endl;
      }

      return 0;
    }

```

如我们可以在前面的代码中看到，我们首先使用迭代版本，然后再转向递归版本，因为我们通常在日常生活中更多地使用迭代。我们通过将每个迭代的 `result` 值乘以 `base` 值来组合每个迭代中的 `result` 值。如果我们运行前面的代码，我们将在控制台上得到以下输出：

![图片](img/84a5e00c-542f-40cc-b398-d5124ddb94e3.png)

现在，让我们将前面的代码重构为递归版本。我们将有一个 `exponential_recursion.cpp` 代码，它将具有相同的 `power()` 函数签名。然而，我们将不会使用 `for` 循环，而是使用递归，在函数的末尾函数调用自身。代码应该写成如下所示：

```cpp
    /* exponential_recursion.cpp */
    #include <iostream>

    using namespace std;

    // Calculating the power of number
    // using recursion
    int power(int base, int exp)
    {
      if(exp == 0)
        return 1;
      else
        return base * power(base, exp - 1);
    }

    auto main() -> int
    {
      cout << "[exponential_recursion.cpp]" << endl;

      // Invoking power() function six times
      for(int i = 0; i <= 5; ++i)
      {
        cout << "power (2, " << i << ") = ";
        cout << power(2, i) << endl;
      }

      return 0;
    }

```

如我们之前讨论的那样，函数递归返回值，`power()` 函数就是一个函数递归，因为它返回一个 `int` 类型的值。我们将从每个子函数返回的值中获取最终结果。因此，我们在控制台上将得到以下输出：

![图片](img/2488e227-b4d4-4275-98c4-a499df6d85de.png)

# 在过程递归中递归运行任务

因此，我们有一个期望从函数返回值的函数递归。有时，我们不需要返回值，因为我们从函数内部运行任务。为了达到这个目的，我们可以使用过程递归。假设我们想要排列一个短字符串以找到它的所有可能的排列。我们不需要返回值，只需要在每次递归执行时打印结果。

我们有以下 `permutation.cpp` 代码来演示这个任务。它有一个 `permute()` 函数，它将被调用一次，然后它将递归地调用 `doPermute()` 函数。代码应该写成如下所示：

```cpp
    /* permutation.cpp */
    #include <iostream>

    using namespace std;

    // Calculation the permutation
    // of the given string
    void doPermute(
      const string &chosen,
      const string &remaining)
      {
       if(remaining == "")
       {
          cout << chosen << endl;
       }
       else
       {
         for(uint32_t u = 0; u < remaining.length(); ++u)
         {
            doPermute(
              chosen + remaining[u],
              remaining.substr(0, u)
              + remaining.substr(u + 1));
         }
       }
    }     

    // The caller of doPermute() function
    void permute(
      const string &s)
    {
      doPermute("", s);
    }

    auto main() -> int
    {
      cout << "[permutation.cpp]" << endl;

      // Initializing str variable
      // then ask user to fill in
      string str;
      cout << "Permutation of a string" << endl;
      cout << "Enter a string: ";
      getline(cin, str);

      // Finding the possibility of the permutation
      // by calling permute() function
      cout << endl << "The possibility permutation of ";
      cout << str << endl;
      permute(str);

      return 0;
    }

```

如我们可以在前面的代码中看到，我们要求用户输入一个字符串，然后代码将使用 `permute()` 函数找到这种排列的可能性。由于用户给出的字符串也是可能的，`doPermute()` 将从空字符串开始。控制台上的输出应该如下所示：

![图片](img/7b9a4176-1af8-412e-a860-110982c72225.png)

# 回溯递归

如我们之前讨论的，如果子任务不起作用，我们可以撤销这个过程。让我们尝试一个迷宫，我们需要从起点找到终点。假设我们必须要找到从`S`到`F`的路径，如下面的迷宫所示：

```cpp
    # # # # # # # #
    # S           #
    # # #   # # # #
    #   #   # # # #
    #             #
    #   # # # # # #
    #           F #
    # # # # # # # #

```

为了解决这个问题，我们必须决定我们需要走的路线，以找到终点。然而，我们将假设每个选择都是好的，直到我们证明它不是。递归将返回一个布尔值来标记这是否是正确的路线。如果我们选择了错误的路线，调用栈将展开，并将撤销这个选择。首先，我们将在代码中绘制`labyrinth`。在下面的代码中，将会有`createLabyrinth()`和`displayLabyrinth()`函数。代码看起来是这样的：

```cpp
    /* labyrinth.cpp */
    #include <iostream>
    #include <vector>

    using namespace std;

    vector<vector<char>> createLabyrinth()
    {
      // Initializing the multidimensional vector
      // labyrinth 
      // # is a wall
      // S is the starting point
      // E is the finishing point
      vector<vector<char>> labyrinth = 
      {
        {'#', '#', '#', '#', '#', '#', '#', '#'},
        {'#', 'S', ' ', ' ', ' ', ' ', ' ', '#'},
        {'#', '#', '#', ' ', '#', '#', '#', '#'},
        {'#', ' ', '#', ' ', '#', '#', '#', '#'},
        {'#', ' ', ' ', ' ', ' ', ' ', ' ', '#'},
        {'#', ' ', '#', '#', '#', '#', '#', '#'},
        {'#', ' ', ' ', ' ', ' ', ' ', 'F', '#'},
        {'#', '#', '#', '#', '#', '#', '#', '#'}
     };

     return labyrinth;
    }

    void displayLabyrinth(vector<vector<char>> labyrinth)
    {
      cout << endl;
      cout << "====================" << endl;
      cout << "The Labyrinth" << endl;
      cout << "====================" << endl;

      // Displaying all characters in labyrinth vector
      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
        {
            cout << labyrinth[i][j] << " ";
        }
        cout << endl;
      }
      cout << "====================" << endl << endl;
    }

    auto main() -> int
    {
      vector<vector<char>> labyrinth = createLabyrinth();
      displayLabyrinth(labyrinth);

      string line;
      cout << endl << "Press enter to continue..." << endl;
      getline(cin, line);

      return 0;
    }

```

如我们所见，前面的代码中没有递归。`createLabyrinth()`函数只是创建一个包含迷宫模式的二维数组，而`displayLabyrinth()`函数只是将数组显示到控制台。如果我们运行前面的代码，我们将在控制台上看到以下输出：

![](img/9e5caa72-94f1-4458-9bf4-e723d64e3fa7.png)

从前面的截图，我们可以看到那里有两个点--`S`是起点，`F`是终点。代码必须找到从`S`到`F`的路径。预期的路线应该是这样的：

![](img/bb949478-e271-4a23-96c1-fdc57ad1d0cf.png)

前面的截图中的白色箭头是我们期望从`S`到`F`的路径。现在，让我们开发代码来解决这个迷宫问题。我们将创建一个名为`navigate`的函数，通过确定以下三个状态来找到可能的路径：

+   如果我们在`[*x*,*y*]`位置找到`F`，例如`labyrinth[2][4]`，那么我们就解决了问题，然后只需返回`true`作为返回值。

+   如果`[*x*,*y*]`位置是`#`，这意味着我们遇到了墙壁，必须重新访问其他`[*x*,*y*]`位置。

+   否则，我们在该位置打印`*`以标记我们已经访问过它。

在我们分析了这三个状态之后，我们将从以下递归情况开始：

+   路径寻找者将向上走，如果它可以导航到`row - 1`，并且它大于或等于`0`（`row - 1 >= 0 && navigate(labyrinth, row - 1, col)`）

+   路径寻找者将向下走，如果它可以导航到`row + 1`，并且它小于`8`（`row + 1 < 8 && navigate(labyrinth, row + 1, col)`）

+   路径寻找者将向左走，如果它可以导航到`col - 1`，并且它大于或等于`0`（`col - 1 >= 0 && navigate(labyrinth, row, col - 1)`）

+   路径寻找者将向右走，如果它可以导航到`col + 1`，并且它小于`8`（`col + 1 < 8 && navigate(labyrinth, row, col + 1)`）

我们将会有以下`navigate()`函数：

```cpp
    bool navigate(
      vector<vector<char>> labyrinth,
      int row,
      int col)
    {
      // Displaying labyrinth
      displayLabyrinth(labyrinth);

      cout << "Checking cell (";
      cout << row << "," << col << ")" << endl;

      // Pause 1 millisecond
      // before navigating
      sleep(1);

      if (labyrinth[row][col] == 'F')
      {
        cout << "Yeayy.. ";
        cout << "Found the finish flag ";
        cout << "at point (" << row << ",";
        cout << col << ")" << endl;
        return (true);
      }
      else if (
        labyrinth[row][col] == '#' ||
        labyrinth[row][col] == '*')
      {
        return (false);
      }
      else if (labyrinth[row][col] == ' ')
      {
        labyrinth[row][col] = '*';
      }

      if ((row + 1 < rows) &&
        navigate(labyrinth, row + 1, col))
        return (true);

      if ((col + 1 < cols) &&
        navigate(labyrinth, row, col + 1))
        return (true);

      if ((row - 1 >= 0) &&
        navigate(labyrinth, row - 1, col))
        return (true);

      if ((col - 1 >= 0) &&
        navigate(labyrinth, row, col - 1))
        return (true);

        return (false);
    }

```

我们现在有了`navigate()`函数来找出找到`F`的正确路径。然而，在我们运行`navigate()`函数之前，我们必须确保`S`在那里。然后我们必须开发一个名为`isLabyrinthSolvable()`的辅助函数。它将遍历迷宫数组，并告知`S`是否在那里。以下代码片段是`isLabyrinthSolvable()`函数的实现：

```cpp
    bool isLabyrinthSolvable(
      vector<vector<char>> labyrinth)
    {
      int start_row = -1;
      int start_col = -1;
      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
        {
            if (labyrinth[i][j] == 'S')
            {
                start_row = i;
                start_col = j;
                break;
            }
        }
      }

      if (start_row == -1 || start_col == -1)
      {
        cout << "No valid starting point found!" << endl;
        return (false);
      }

      cout << "Starting at point (" << start_row << ",";
      cout << start_col << ")" << endl;

      return navigate(labyrinth, start_row, start_col);
    }

```

如我们可以在先前的代码片段中看到，我们提到了`rows`和`cols`变量。我们将它们初始化为全局变量，正如我们可以在以下代码片段中看到：

```cpp
    const int rows = 8;
    const int cols = 8;

```

现在，如果我们向`labyrinth.cpp`代码中插入`navigate()`和`isLabyrinthSolvable()`函数，让我们看一下以下代码：

```cpp
    /* labyrinth.cpp */
    #include <iostream>
    #include <vector>
 #include <unistd.h>

    using namespace std;

 const int rows = 8;
 const int cols = 8;

    vector<vector<char>> createLabyrinth()
    {
      // Initializing the multidimensional vector
      // labyrinth
      // # is a wall
      // S is the starting point
      // E is the finishing point
      vector<vector<char>> labyrinth =
      {
        {'#', '#', '#', '#', '#', '#', '#', '#'},
        {'#', 'S', ' ', ' ', ' ', ' ', ' ', '#'},
        {'#', '#', '#', ' ', '#', '#', '#', '#'},
        {'#', ' ', '#', ' ', '#', '#', '#', '#'},
        {'#', ' ', ' ', ' ', ' ', ' ', ' ', '#'},
        {'#', ' ', '#', '#', '#', '#', '#', '#'},
        {'#', ' ', ' ', ' ', ' ', ' ', 'F', '#'},
        {'#', '#', '#', '#', '#', '#', '#', '#'}
       };

     return labyrinth;
    }

    void displayLabyrinth(
      vector<vector<char>> labyrinth)
    {
      cout << endl;
      cout << "====================" << endl;
      cout << "The Labyrinth" << endl;
      cout << "====================" << endl;
      // Displaying all characters in labyrinth vector
      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
        {
            cout << labyrinth[i][j] << " ";
        }
        cout << endl;
       }
      cout << "====================" << endl << endl;
    }

 bool navigate(
 vector<vector<char>> labyrinth,
 int row,
 int col)
 {
 // Displaying labyrinth
 displayLabyrinth(labyrinth);

 cout << "Checking cell (";
 cout << row << "," << col << ")" << endl;

 // Pause 1 millisecond
 // before navigating
 sleep(1);

 if (labyrinth[row][col] == 'F')
 {
 cout << "Yeayy.. ";
 cout << "Found the finish flag ";
        cout << "at point (" << row << ",";
 cout << col << ")" << endl;
 return (true);
 }
 else if (
 labyrinth[row][col] == '#' ||
 labyrinth[row][col] == '*')
 {
 return (false);
 }
 else if (labyrinth[row][col] == ' ')
 {
 labyrinth[row][col] = '*';
 }

 if ((row + 1 < rows) &&
 navigate(labyrinth, row + 1, col))
 return (true); 
 if ((col + 1 < cols) &&
 navigate(labyrinth, row, col + 1))
 return (true); 
 if ((row - 1 >= 0) &&
 navigate(labyrinth, row - 1, col))
 return (true); 
 if ((col - 1 >= 0) &&
 navigate(labyrinth, row, col - 1))
 return (true); 
 return (false);
 } 
 bool isLabyrinthSolvable(
 vector<vector<char>> labyrinth)
 {
 int start_row = -1;
 int start_col = -1;
 for (int i = 0; i < rows; i++)
 {
 for (int j = 0; j < cols; j++)
 {
 if (labyrinth[i][j] == 'S')
 {
 start_row = i;
 start_col = j;
 break;
 }
 }
 }

 if (start_row == -1 || start_col == -1)
 {
 cerr << "No valid starting point found!" << endl;
 return (false);
 }

 cout << "Starting at point (" << start_row << ",";
 cout << start_col << ")" << endl;

 return navigate(labyrinth, start_row, start_col);
 }

    auto main() -> int
    {
      vector<vector<char>> labyrinth = createLabyrinth();
      displayLabyrinth(labyrinth);

      string line;
      cout << endl << "Press enter to continue..." << endl;
      getline(cin, line);

 if (isLabyrinthSolvable(labyrinth))
 cout << "Labyrinth solved!" << endl;
 else
 cout << "Labyrinth could not be solved!" << endl;

     return 0;
    }

```

如我们可以在先前的引言中看到，在`main()`函数中，我们首先运行`isLabyrinthSolvable()`函数，该函数反过来调用`navigate()`函数。然后`navigate()`函数将遍历迷宫以找到正确的路径。以下是代码的输出：

![](img/529db26b-9492-4c16-bfba-ea14a2a6061d.png)

然而，如果我们追踪程序解决迷宫的过程，当它找到结束标志时，它会面临错误路径，正如我们可以在以下截图中所见：

![](img/be30241d-2365-4325-907c-8761c1460abd.png)

如我们所见，在先前的截图中有白色方块。当它在寻找正确路径时，这是一个错误的选择。一旦遇到障碍，它会退回并寻找其他路径。它也会撤销它所做的选择。让我们看看以下截图，它展示了当递归找到另一条路径并撤销之前的选项时的情况：

![](img/b6bd29c3-e866-4819-8eaf-eb23403b3dc4.png)

在先前的截图中，我们可以看到递归尝试了另一条路径，由于回溯递归撤销了路径，之前失败的路径已经消失。现在递归有了正确的路径，它只需继续直到找到结束标志。因此，我们现在已经成功开发了回溯递归。

# 摘要

本章向我们介绍了通过迭代和递归重复函数调用的技术。然而，由于递归比迭代更具有函数性，我们强调了我们对递归而不是迭代的讨论。我们首先讨论了迭代和递归之间的区别。然后我们继续讨论将不可变函数重构为递归不可变函数的讨论。

在我们学习了递归（recursion）之后，我们发现了一些更好的递归技术。我们还讨论了尾递归（tail recursion）以获得这种改进的技术。最后，我们列举了三种递归类型——函数式递归（functional recursion）、过程式递归（procedural recursion）和回溯递归（backtracking recursion）。当我们期望递归的返回值时，我们通常使用函数式递归。否则，我们使用过程式递归。而且，如果我们需要分解问题并在递归不起作用时撤销递归性能，我们可以使用回溯递归来解决问题。

在下一章中，我们将讨论懒加载（lazy evaluation）以使代码运行更快。这将使代码变得更加高效，因为它将确保不会执行不必要的代码。
