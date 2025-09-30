# 使用惰性评估延迟执行过程

在上一章中，我们讨论了在函数式方法中重复函数调用的递归。现在，我们将讨论惰性评估，它可以使我们的代码更加高效，因为它只有在需要时才会运行。我们还将应用我们在上一章中讨论的递归主题来生成惰性代码。

在本章中，我们讨论**惰性评估**以使代码运行更快。这将使代码变得高效，因为它将确保不会执行不必要的代码。以下是我们将讨论的主题，以深入了解惰性评估：

+   区分贪婪评估和惰性评估的区别

+   使用缓存技术优化代码

+   将贪婪评估重构为惰性评估

+   设计可重用于他人函数式代码的有用类

# 评估表达式

每种编程语言都有其自己的策略来确定何时评估函数调用的参数以及必须传递给参数的类型值。在编程语言中，有两种主要的评估策略被广泛使用--**严格**（贪婪）评估和**非严格**（惰性）评估。

# 使用严格评估立即运行表达式

严格评估在大多数命令式编程语言中使用。它将立即执行我们已有的代码。假设我们有以下等式：

```cpp
    int i = (x + (y * z));

```

在严格评估中，最内层的括号将首先计算，然后向外计算前面的等式。这意味着我们将计算`y * z`，然后将结果加到`x`上。为了更清楚地说明，让我们看看以下`strict.cpp`代码：

```cpp
    /* strict.cpp */
    #include <iostream>

    using namespace std;

    int OuterFormula(int x, int yz)
    {
      // For logging purpose only
      cout << "Calculate " << x << " + ";
      cout << "InnerFormula(" << yz << ")";
      cout << endl;

      // Returning the calculation result
      return x * yz;
    }

    int InnerFormula(int y, int z)
    {
      // For logging purpose only
      cout << "Calculate " << y << " * ";
      cout << z << endl;

      // Returning the calculation result
      return y * z;
    }

    auto main() -> int
    {
      cout << "[strict.cpp]" << endl;

      // Initializing three int variables
      // for the calculation
      int x = 4;
      int y = 3;
      int z = 2;

      // Calculating the expression
      cout << "Calculate " << x <<" + ";
      cout << "(" << y << " * " << z << ")";
      cout << endl;
      int result = OuterFormula(x, InnerFormula(y, z));

      // For logging purpose only
      cout << x << " + ";
      cout << "(" << y << " * " << z << ")";
      cout << " = " << result << endl;

      return 0;
    }

```

如我们之前讨论的，上述代码的执行顺序将是先计算`y * z`，然后我们将结果加到`x`上，正如我们可以在以下输出中看到：

![](img/5df7ea63-3801-47d1-9fbd-e50d30c1f8ec.png)

上述执行顺序是我们通常期望的。然而，在非严格评估中，我们将重新排序这个执行过程。

# 使用非严格评估延迟表达式

在非严格评估中，首先计算`+`运算符，然后计算内部公式，即`(y * z)`。我们将看到评估将从外部开始向内部进行。我们将重构我们之前的`strict.cpp`代码，使其成为非严格评估。代码应该如下所示`non_strict.cpp`代码：

```cpp
    /* non_strict.cpp */
    #include <functional>
    #include <iostream>

    using namespace std;

 int OuterFormulaNonStrict(
 int x,
 int y,
 int z,
 function<int(int, int)> yzFunc)
 {
 // For logging purpose only
 cout << "Calculate " << x << " + ";
 cout << "InnerFormula(" << y << ", ";
 cout << z << ")" << endl;

 // Returning the calculation result
 return x * yzFunc(y, z);
 }

     int InnerFormula(int y, int z)
     {
       // For logging purpose only
       cout << "Calculate " << y << " * ";
       cout << z << endl;

       // Returning the calculation result
       return y * z;
     }

     auto main() -> int
     {
       cout << "[non_strict.cpp]" << endl;

       // Initializing three int variables
       // for the calculation
       int x = 4;
       int y = 3;
       int z = 2;

       // Calculating the expression
       cout << "Calculate " << x <<" + ";
       cout << "(" << y << " * " << z << ")";
       cout << endl;
       int result = OuterFormulaNonStrict(x, y, z, InnerFormula);

       // For logging purpose only
       cout << x << " + ";
       cout << "(" << y << " * " << z << ")";
       cout << " = " << result << endl;

       return 0;
    }

```

如我们所见，我们将`strict.cpp`代码中的`OuterFormula()`函数修改为`non_strict.cpp`代码中的`OuterFormulaNonStrict()`函数。在`OuterFormulaNonStrict()`函数中，我们除了传递三个变量`x`、`y`和`z`之外，还传递了一个函数作为参数。因此，前面表达式的执行顺序发生了改变。当我们运行`non_strict.cpp`代码时，在控制台屏幕上我们应该看到以下内容：

![](img/ce639e69-1325-47bc-b524-6894d77af3b5.png)

从前面的输出中，我们已经证明了我们的代码正在执行非严格评估，因为它现在首先计算加法运算符（`+`），而不是乘法运算符（`*`）。然而，结果仍然是正确的，尽管顺序已经改变。

# 懒加载的基本概念

在我们创建懒代码之前，让我们讨论懒加载的基本概念。我们将使用延迟过程使我们的代码变得懒，使用缓存技术通过避免不必要的计算来提高代码性能，以及使用优化技术通过存储昂贵函数调用的结果并在相同的输入再次出现时返回缓存的值来加快代码速度。在查看这些技术之后，我们将尝试开发真正的懒代码。

# 延迟过程

懒的基本概念是延迟一个过程。在本节中，我们将讨论如何延迟特定过程的执行。我们将创建一个名为 `Delay` 的新类。当我们构造这个类时，我们会将一个函数传递给它。除非我们调用 `Fetch()` 方法，否则该函数不会运行。函数的实现如下：

```cpp
    template<class T> class Delay
    {
      private:
        function<T()> m_func;

      public:
        Delay(
          function<T()> func)
          : m_func(func)
          {
          }

        T Fetch()
        {
          return m_func();
        }
    };

```

现在，让我们消耗`Delay`类来推迟执行。我们将创建一个名为`delaying.cpp`的文件，该文件将运行两个函数--`multiply`和`division`。然而，这两个函数只有在调用`Fetch()`方法后才会运行。文件的内容如下：

```cpp
    /* delaying.cpp */
    #include <iostream>
    #include <functional>

    using namespace std;

    template<class T> class Delay
    {
      private:
        function<T()> m_func;

      public:
        Delay(function<T()> func) : m_func(func)
        {
        }

        T Fetch()
        {
          return m_func();
        }
    };

    auto main() -> int
    {
      cout << "[delaying.cpp]" << endl;

      // Initializing several int variables
      int a = 10;
      int b = 5;

      cout << "Constructing Delay<> named multiply";
      cout << endl;
      Delay<int> multiply([a, b]()
      {
        cout << "Delay<> named multiply";
        cout << " is constructed." << endl;
        return a * b;
      });

     cout << "Constructing Delay<> named division";
     cout << endl;
     Delay<int> division([a, b]()
     {
       cout << "Delay<> named division ";
       cout << "is constructed." << endl;
       return a / b; 
     });

     cout << "Invoking Fetch() method in ";
     cout << "multiply instance." << endl;
     int c = multiply.Fetch();

     cout << "Invoking Fetch() method in ";
     cout << "division instance." << endl;
     int d = division.Fetch();

     // Displaying the result
     cout << "The result of a * b = " << c << endl;
     cout << "The result of a / b = " << d << endl;

     return 0;
    }

```

正如我们在第一章“深入现代 C++”中讨论的那样，我们可以使用 Lambda 表达式来构建`multiply`和`division`函数。然后我们将它们传递给每个`Delay`构造函数。在这个阶段，函数还没有运行。它将在调用`Fetch()`方法后运行--`multiply.Fetch()`和`division.Fetch()`。屏幕上显示的输出应该如下截图所示：

![](img/5ade918f-d163-4d2b-9d84-d41ddc962d86.png)

正如我们在前面的输出截图中所见，`multiply`和`division`实例是在调用`Fetch()`方法时构建的（见两个白色箭头），而不是在调用`Delay`类的构造函数时。现在，我们已经成功延迟了执行，我们可以这样说，这个过程只有在需要时才会执行。

# 使用记忆化技术缓存值

现在我们已经通过消耗`Delay`类成功延迟了函数的执行。然而，由于`Delay`类实例的函数将在每次调用`Fetch()`方法时运行，如果函数不是纯函数或具有副作用，可能会出现意外结果。让我们通过修改`multiply`函数来重构我们之前的`delaying.cpp`代码。这个函数现在变成了一个非纯函数，因为它依赖于外部变量。代码应该如下所示：

```cpp
    /* delaying_non_pure.cpp */
    #include <iostream>
    #include <functional>

    using namespace std;

    template<class T> class Delay
    {
      private:
        function<T()> m_func;

      public:
        Delay(function<T()> func) : m_func(func)
        {
        }

        T Fetch()
        {
          return m_func();
        }
    };

    auto main() -> int
    {
      cout << "[delaying_non_pure.cpp]" << endl;

      // Initializing several int variables
      int a = 10;
      int b = 5;
      int multiplexer = 0;

      // Constructing Delay<> named multiply_impure
      Delay<int> multiply_impure([&]()
      {
        return multiplexer * a * b;
      });

      // Invoking Fetch() method in multiply_impure instance
      // multiple times
      for (int i = 0; i < 5; ++i)
      {
        ++multiplexer;
        cout << "Multiplexer = " << multiplexer << endl;
        cout << "a * b = " << multiply_impure.Fetch();
        cout << endl;
      }

      return 0;
    }

```

如前述代码所示，我们现在有一个名为 `multiply_impure` 的新 Lambda 表达式，它是我们在 `delaying.cpp` 代码中创建的 `multiply` 函数的重构版本。`multiply_impure` 函数依赖于 `multiplexer` 变量，其值将在我们调用 `Fetch()` 方法之前每次增加。以下是我们应该在屏幕上看到的截图输出：

![截图](img/f1e15207-1b93-4d22-b534-54ce6b0bba97.png)

如我们所见，`Fetch()` 方法每次被调用时都会给出不同的结果。我们现在必须重构 `Delay` 类，以确保它在每次使用相同的传递参数调用 `Fetch()` 方法时都会返回完全相同的结果。为了实现这一点，我们将使用存储函数调用结果的记忆化技术，并在相同的输入再次出现时返回缓存的結果。

我们将把 `Delay` 类重命名为 `Memoization` 类。这不仅会延迟函数调用，还会记录带有特定传递参数的函数。因此，下次出现带有这些参数的函数时，函数本身将不会运行，而是直接返回缓存的結果。为了便于讨论，让我们看一下以下 `Memoization` 类的实现：

```cpp
    template<class T> class Memoization
    {
      private:
        T const & (*m_subRoutine)(Memoization *);
        mutable T m_recordedFunc;
        function<T()> m_func;

        static T const & ForceSubroutine(Memoization * d)
        {
          return d->DoRecording();
        }

        static T const & FetchSubroutine(Memoization * d)
        {
          return d->FetchRecording();
        }

        T const & FetchRecording()
        {
          return m_recordedFunc;
        }

        T const & DoRecording()
        {
          m_recordedFunc = m_func();
          m_subRoutine = &FetchSubroutine;
          return FetchRecording();
        }

     public:
        Memoization(function<T()> func) : m_func(func),
         m_subRoutine(&ForceSubroutine),
         m_recordedFunc(T())
        {
        }

       T Fetch()
       {
         return m_subRoutine(this);
       }
    };

```

如前述代码片段所示，我们现在有 `FetchRecording()` 和 `DoRecording()` 来获取和设置我们存储的函数。此外，当类被构造时，它将记录传递的函数并将其保存到 `m_subRoutine`。当调用 `Fetch()` 方法时，类将检查 `m_subRoutine`，以确定它是否具有带有当前传递参数的函数的值。如果是，它将直接从 `m_subRoutine` 返回值而不是运行函数。现在，让我们看看以下 `delaying_non_pure_memoization.cpp` 代码，它使用了 `Memoization` 类：

```cpp
    /* delaying_non_pure_memoization.cpp */
    #include <iostream>
    #include <functional>

    using namespace std;

    template<class T> class Memoization
    {
      private:
        T const & (*m_subRoutine)(Memoization *);
        mutable T m_recordedFunc;
        function<T()> m_func;

        static T const & ForceSubroutine(Memoization * d)
        {
          return d->DoRecording();
        }

       static T const & FetchSubroutine(Memoization * d)
       {
          return d->FetchRecording();
       }

       T const & FetchRecording()
       {
          return m_recordedFunc;
       }

       T const & DoRecording()
       {
          m_recordedFunc = m_func();
          m_subRoutine = &FetchSubroutine;
          return FetchRecording();
       }

     public:
       Memoization(function<T()> func) : m_func(func),
        m_subRoutine(&ForceSubroutine),
        m_recordedFunc(T())
       {
       }

      T Fetch()
      {
        return m_subRoutine(this);
      }
    };

    auto main() -> int
    {
      cout << "[delaying_non_pure_memoization.cpp]" << endl;

      // Initializing several int variables
      int a = 10;
      int b = 5;
      int multiplexer = 0;

 // Constructing Memoization<> named multiply_impure
 Memoization<int> multiply_impure([&]()
 {
 return multiplexer * a * b;
 });

      // Invoking Fetch() method in multiply_impure instance
      // multiple times
      for (int i = 0; i < 5; ++i)
      {
        ++multiplexer;
        cout << "Multiplexer = " << multiplexer << endl;
        cout << "a * b = " << multiply_impure.Fetch();
        cout << endl;
      }

      return 0;
    }

```

从前述代码片段中，我们看到在 `main()` 函数中我们没有进行太多修改。我们修改的只是 `multiply_impure` 变量所使用的类类型，从 `Delay` 改为 `Memoization`。然而，现在结果已经改变，因为我们将从 `multiply_impure()` 函数的五次调用中获得完全相同的返回值。让我们看一下以下截图以证明这一点：

![截图](img/eb597eec-cfe2-4c06-b0eb-a546e3a26fb5.png)

从前述截图可以看出，即使 `Multiplexer` 的值增加了，计算的返回值始终相同。这是因为第一次函数调用的返回值被记录下来，因此对于剩余的调用不需要再次运行函数。

正如我们在第二章中讨论的，*在函数式编程中操作函数*，在函数式编程中，一个不纯的函数看起来是不正确的。如果代码确实需要不同的结果（非缓存结果），那么在缓存后面隐藏一个不纯的函数可能会导致错误。请谨慎使用前面的技术来缓存不纯的函数。

# 使用缓存技术优化代码

缓存（Memoization）在应用于非纯函数或具有副作用（side effect）的函数时非常有用。然而，它也可以用来优化代码。通过使用缓存，我们开发的代码将运行得更快。假设我们需要多次运行完全相同的函数，并且传递的参数也完全相同。如果代码从我们记录值的地方获取值而不是运行函数，这将更快。对于昂贵的函数调用来说，这也更好，因为通过使用缓存，我们不需要反复执行不必要的昂贵函数调用。

让我们创建一个代码来讨论进一步的优化。我们将使用`Delay`类来演示它相对于`Memoization`类不是优化过的代码。我们将有一个`not_optimize_code.cpp`代码，它将消耗`Delay`类。在这个未优化的代码中，我们将调用我们在第四章中创建的`fibonacci()`函数，即使用递归算法重复方法调用。我们将`40`作为参数传递给`fibonacci()`函数，并从`fib40`类实例中调用`Fetch()`方法五次。我们还将使用`chrono`头文件中的`high_resolution_clock`类计算方法调用的耗时，以记录**开始**和**结束**时间，通过减去结束值与开始值来获取耗时。除了每个`Fetch()`方法调用的耗时外，我们还计算整个代码的耗时。`not_optimize_code.cpp`代码的实现如下：

```cpp
    /* not_optimize_code.cpp */
    #include <iostream>
    #include <functional>
    #include <chrono>

    using namespace std;

    template<class T> class Delay
    {
      private:
        function<T()> m_func;

      public:
        Delay(function<T()> func): m_func(func)
        {
        }

        T Fetch()
        {
          return m_func();
        }
    };

    // Function for calculating Fibonacci sequence
    int fibonacci(int n)
    {
      if(n <= 1)
         return n;
      return fibonacci(n-1) + fibonacci(n-2);
    }

    auto main() -> int
    {
      cout << "[not_optimize_code.cpp]" << endl;

      // Recording start time for the program
      auto start = chrono::high_resolution_clock::now();

      // Initializing int variable to store the result
      // from Fibonacci calculation
      int fib40Result = 0;

      // Constructing Delay<> named fib40
      Delay<int> fib40([]()
      {
        return fibonacci(40);
      });

      for (int i = 1; i <= 5; ++i)
      {
        cout << "Invocation " << i << ". ";

        // Recording start time
        auto start = chrono::high_resolution_clock::now();

        // Invoking the Fetch() method
        // in fib40 instance
        fib40Result = fib40.Fetch();

        // Recording end time
        auto finish = chrono::high_resolution_clock::now();

        // Calculating the elapsed time
        chrono::duration<double, milli> elapsed = finish - start;

        // Displaying the result
        cout << "Result = " << fib40Result << ". ";

        // Displaying elapsed time
        // for each fib40.Fetch() invocation
        cout << "Consuming time = " << elapsed.count();
        cout << " milliseconds" << endl;
      }

       // Recording end time for the program
       auto finish = chrono::high_resolution_clock::now();

       // Calculating the elapsed time for the program
       chrono::duration<double, milli> elapsed = finish - start;

       // Displaying elapsed time for the program
       cout << "Total consuming time = ";
       cout << elapsed.count() << " milliseconds" << endl;

       return 0;
    }

```

现在，让我们运行代码以获取先前代码处理过程的耗时。以下截图是我们将在屏幕上看到的内容：

![图片](img/d2ef7cc8-601b-47af-ae9f-19797be02341.png)

从前面的截图可以看出，我们处理代码大约需要`2357.79`毫秒。并且每次调用`fib40.Fetch()`方法时，平均需要大约`470`毫秒，尽管我们传递给`fibonacci()`函数的确切参数是`40`。现在，让我们看看如果我们对前面的代码使用缓存技术会发生什么。我们不会对代码做太多修改，只是重构`fib40`的实例化。现在，它不是从`Delay`类实例化，而是从`Memoization`类实例化。代码应该如下所示：

```cpp
    /* optimizing_memoization.cpp */
    #include <iostream>
    #include <functional>
    #include <chrono>

    using namespace std;

    template<class T> class Memoization
    {
      private:
        T const & (*m_subRoutine)(Memoization *);
        mutable T m_recordedFunc;
        function<T()> m_func;

        static T const & ForceSubroutine(Memoization * d)
        {
          return d->DoRecording();
        }

        static T const & FetchSubroutine(Memoization * d)
        {
          return d->FetchRecording();
        }

        T const & FetchRecording()
        {
          return m_recordedFunc;
        }

        T const & DoRecording()
        {
          m_recordedFunc = m_func();
          m_subRoutine = &FetchSubroutine;
          return FetchRecording();
        }

      public:
        Memoization(function<T()> func): m_func(func),
          m_subRoutine(&ForceSubroutine),
          m_recordedFunc(T())
          {
          }

        T Fetch()
        {
          return m_subRoutine(this);
        }
     };

       // Function for calculating Fibonacci sequence
       int fibonacci(int n)
       {
         if(n <= 1)
           return n;
           return fibonacci(n-1) + fibonacci(n-2);
       }

       auto main() -> int
       {
         cout << "[optimizing_memoization.cpp]" << endl;

         // Recording start time for the program
         auto start = chrono::high_resolution_clock::now();

         // Initializing int variable to store the result
         // from Fibonacci calculation
         int fib40Result = 0;

         // Constructing Memoization<> named fib40
 Memoization<int> fib40([]()
 {
 return fibonacci(40);
 });

         for (int i = 1; i <= 5; ++i)
         {
           cout << "Invocation " << i << ". ";

           // Recording start time
           auto start = chrono::high_resolution_clock::now();

           // Invoking the Fetch() method
           // in fib40 instance
           fib40Result = fib40.Fetch();

           // Recording end time
           auto finish = chrono::high_resolution_clock::now();

           // Calculating the elapsed time
           chrono::duration<double, milli> elapsed = finish - start;

           // Displaying the result
           cout << "Result = " << fib40Result << ". ";

           // Displaying elapsed time
           // for each fib40.Fetch() invocation
           cout << "Consuming time = " << elapsed.count();
           cout << " milliseconds" << endl;
       }

          // Recording end time for the program
          auto finish = chrono::high_resolution_clock::now();

          // Calculating the elapsed time for the program
          chrono::duration<double, milli> elapsed = finish - start;

          // Displaying elapsed time for the program
          cout << "Total consuming time = ";
          cout << elapsed.count() << " milliseconds" << endl;

          return 0;
     }

```

当我们运行`optimizing_memoization.cpp`代码时，控制台屏幕上会出现以下内容：

![图片](img/0c030b41-e024-44bc-b1fc-986e0504eae2.png)

令人惊讶的是，我们只需 `494.681` 毫秒就能执行 `optimizing_memoization.cpp` 代码。与 `not_optimize_code.cpp` 代码相比，代码的速度大约快了 `4.7` 倍。这是因为代码在参数传递 `40` 给 `fibonacci()` 函数时成功缓存了结果。每次我们再次调用 `fib40.Fetch()` 方法时，它将再次调用 `fibonacci()` 函数，输入完全相同。代码将直接返回缓存的結果，从而避免了运行不必要的昂贵函数调用。

# 惰性评估实践

在讨论了惰性评估的基本概念之后，让我们通过以惰性方式设计代码来深入了解惰性评估。在本节中，我们将首先开发一个急切评估代码，然后将该代码重构为惰性评估代码。我们开发的代码将生成一系列素数。首先，我们将使用 `for` 循环迭代整数以在急切评估中获得素数。下面是我们要讨论的 `prime.cpp` 代码：

```cpp
    /* prime.cpp */
    #include <iostream>
    #include <cmath>

    using namespace std;

    bool PrimeCheck(int i)
    {
      // All even numbers are not prime number
      // except 2
      if ((i % 2) == 0)
      {
        return i == 2;
      }

      // Calculating the square root of i
      // and store in int data type variable
      // if the argument i is not even number,
      int sqr = sqrt(i);

      // For numbers 9 and below,
      // the prime numbers is simply the odd numbers
      // For number above 9
      // the prime numbers is all of odd numbers
      // except the square number
      for (int t = 3; t <= sqr; t += 2)
      {
        if (i % t == 0)
        {
            return false;
        }
      }

       // The number 1 is not prime number
       // but still passing the preceding test
       return i != 1;
    }

    auto main() -> int
    {
      cout << "[delaying.cpp]" << endl;

      // Initializing a counting variable
      int n = 0;

      // Displaying the first 100 prime numbers
      cout << "List of the first 100 prime numbers:" << endl;
      for (int i = 0; ; ++i)
      {
        if (PrimeCheck(i))
        {
            cout << i << "\t";

            if (++n == 100)
                return 0;
        }
      }

      return 0;
    }

```

如前所述的代码所示，我们有一个简单的 `PrimeCheck()` 函数来分析整数是否为素数。之后，代码使用 `for` 循环迭代无限整数，然后检查它是否为素数。如果得到了一百个素数，循环将结束。以下是我们应该在控制台上看到的输出截图：

![图片](img/3263a17d-40a8-4065-82ad-48cfa4601359.png)

我们现在有一个使用急切评估生成素数的代码。如前所述的截图所示，我们使用 `for` 循环生成了百个素数。接下来，我们将将其重构为惰性代码。

# 设计 Chunk 和 Row 类

在 `prime.cpp` 代码中，我们使用 `for` 循环生成一系列整数。在这个序列中，有几个数字被称为 **Chunk**。现在，在我们重构代码之前，我们将准备一个名为 `Row` 和 `Chunk` 的类，以便我们进一步讨论。从我们之前的类比中，`Row` 类将持有整数序列，而 `Chunk` 类将持有单个数字。我们将从数据的最小部分开始，即块。以下是 `Chunk` 类的实现：

```cpp
    template<class T> class Chunk
    {
      private:
        T m_value;
        Row<T> m_lastRow;

      public:
        Chunk()
         {
         }

        Chunk(T value, Row<T> lastRow): m_value(value),
         m_lastRow(std::move(lastRow))
        {
        }

        explicit Chunk(T value) : m_value(value)
        {
        }

        T Value() const
        {
          return m_value;
        }

        Row<T> ShiftLastToFirst() const
        {
          return m_lastRow;
        }
    };

```

由于 `Row` 类是由几个 `Chunk` 类构成的，除了 `Chunk` 本身的值外，`Chunk` 类还在当前 `Row` 中通过 `m_lastRow` 成员变量表示 `Chunk` 的下一个值。我们也可以通过调用 `ShiftLastToFirst()` 方法来获取 `m_lastRow` 的值。现在，让我们转向 `Row` 类。类的实现如下：

```cpp
    template<class T> class Row
    {
      private:
        std::shared_ptr <Memoization<Chunk<T>>>
        m_lazyChunk;

      public:
         Row()
         {
         }

         explicit Row(T value)
         {
           auto chunk = ChunkPreparation<T>(value);
           m_lazyChunk = std::make_shared<Memoization<Chunk<T>>> 
           (chunk);
         }

         Row(T value, Row row)
         {
           auto chunk = ChunkPreparation<T>( value, std::move(row));

           m_lazyChunk = std::make_shared<Memoization<Chunk<T>>>(
           chunk);
         }

         Row(std::function<Chunk<T>()> func): m_lazyChunk(
         std::make_shared<Memoization<Chunk<T>>>(func))
         {
         }

         bool IsEmpty() const
         {
           return !m_lazyChunk;
         }

         T Fetch() const
         {
           return m_lazyChunk->Fetch().Value();
         }

         Row<T> ShiftLastToFirst() const
         {
          return m_lazyChunk->Fetch().ShiftLastToFirst();
         }

         Row Pick(int n) const
         {
           if (n == 0 || IsEmpty())
            return Row();

          auto chunk = m_lazyChunk;
          return Row([chunk, n]()
          {
            auto val = chunk->Fetch().Value();
            auto row = chunk->Fetch().ShiftLastToFirst();
            return Chunk<T>(val, row.Pick(n - 1));
          });
         }
    };

```

如前述代码片段所示，`Row`类只有一个私有成员用于存储`Chunk`数据的缓存。`Row`类有四个构造函数，我们将在下一段代码中使用它们全部。它还包含一个`Fetch()`方法，这是我们在上一节设计`Memoization`类时得到的，用于获取`m_lazyChunk`值。其他方法对我们编写下一部分懒加载代码也是很有用的。`IsEmpty()`方法将检查`m_lazyChunk`值是否为空，`ShiftLastToFirst()`方法将取`m_lazyChunk`的最后一行，而`Pick(int n)`方法将取出我们将在稍后需要取出的一百个整数质数的第一个`n`行元素。

我们还可以看到，`Row`构造函数之一调用了`ChunkPreparation`类的构造函数。`ChunkPreparation`类将使用给定的值和最后一行的值初始化一个新的`Chunk`类构造函数。类的实现如下：

```cpp
    template<class T> class ChunkPreparation
    {
      public:
        T m_value;
        Row<T> m_row;

        ChunkPreparation(T value, Row<T> row) :
          m_value(value),
          m_row(std::move(row))
          {
          }

        explicit ChunkPreparation(T value) :
          m_value(value)
          {
          }

        Chunk<T> operator()()
        {
          return Chunk<T>(
            m_value,
            m_row);
        }
    };

```

如我们所见，通过调用`operator ()`，新的`Chunk`将使用给定的`m_value`和`m_row`值生成。

# 连接多行

当我们计划生成一个质数行时，我们必须能够将当前行与代码生成的新行连接起来。为了满足这一需求，以下是对`ConcatenateRows()`函数的实现，该函数将连接两个行：

```cpp
    template<class T> Row<T> ConcatenateRows(
      Row<T> leftRow,
      Row<T> rightRow)
      {
        if (leftRow.IsEmpty())
          return rightRow;

        return Row<T>([=]()
        {
          return Chunk<T>(
            leftRow.Fetch(),
            ConcatenateRows<T>(
             leftRow.ShiftLastToFirst(),
             rightRow));
         });
       }

```

当我们查看前述代码片段时，`ConcatenateRows()`函数的作用非常清晰。如果`leftRow`仍然是空的，就只返回第二行，即`rightRow`。如果`leftRow`和`rightRow`都可用，我们可以返回给定行的已形成的块作为一行。

# 迭代`Row`类的每个元素

在我们构建质数行之后，我们需要迭代每一行的元素来操作它，例如，将值打印到控制台。为此，我们必须开发以下`ForEach()`方法：

```cpp
    template<class T, class U> void ForEach( Row<T> row, U func)
     {
        while (!row.IsEmpty())
        {
          func(row.Fetch());
          row = row.ShiftLastToFirst();
         }
     }

```

我们将传递行本身和一个函数到`ForEach()`方法中。我们传递给它的函数将会运行到行的每一个元素上。

为了方便我们在本章开发懒加载代码，我将我们之前的讨论`template`类打包成一个名为`lazyevaluation.h`的单个头文件；我们也可以在其他项目中重用它。该头文件将包含`Memoization`、`Row`、`Chunk`、`ChunkPreparation`、`ConcatenateRows`和`ForEach`模板类。您可以自己创建头文件，或者从 Packt 网站上的代码仓库下载它（[`github.com/PacktPublishing/LearningCPPFunctionalProgramming`](https://github.com/PacktPublishing/LearningCPPFunctionalProgramming)）。

# 生成无限整数行

现在是时候生成无限整数行，就像我们在之前的 `prime.cpp` 代码中使用 `for` 循环所做的那样。然而，我们现在将创建一个名为 `GenerateInfiniteIntRow()` 的新函数，用于从几个整数块生成整数行。以下代码片段是该函数的实现：

```cpp
    Row<int> GenerateInfiniteIntRow( int initialNumber)
    {
      return Row<int>([initialNumber]()
      {
        return Chunk<int>(
            initialNumber,
            GenerateInfinityIntRow(
             initialNumber + 1));
      });
    }

```

如我们所见，首先，我们从 `initialNumber` 创建 `Chunk` 直到无穷大。最后，这些块将被转换为 `Row` 数据类型。为了停止这个递归函数，我们可以在 `Row` 类中调用 `Pick()` 方法。

# 生成无限质数行

在成功生成无限数字之后，我们现在必须限制行，只生成质数。我们将修改 `prime.cpp` 代码中的 `CheckPrime()` 函数。我们将更改函数的返回值，如果不是质数则返回 `Row<void*>(nullptr)`，如果是质数则返回 `Row<void*>()`。函数的实现应该如下所示：

```cpp
    Row<void*> PrimeCheck(int i)
    {
      if ((i % 2) == 0)
      {
        if (i == 2)
            return Row<void*>(nullptr);
        else
            return Row<void*>();
      }

      int sqr = sqrt(i);

      for (int t = 3; t <= sqr; t = t + 2)
      {
        if (i % t == 0)
        {
            return Row<void*>();
        }
      }

      if (i == 1)
        return Row<void*>();
      else
        return Row<void*>(nullptr);
    }

```

为什么我们需要更改函数的返回值？因为我们想将返回值传递给 `JoiningPrimeNumber()` 函数，该函数将生成的 Chunk 与以下实现连接起来：

```cpp
    template<class T, class U> 
    auto JoiningPrimeNumber(
      Row<T> row, U func) -> decltype(func())
      {
         return JoiningAllRows(
           MappingRowByValue(row, func));
      }

```

此外，`MappingRowByValue()` 函数将给定的行映射到给定的函数。函数的实现如下所示：

```cpp
    template<class T, class U> 
    auto MappingRowByValue(
      Row<T> row, U func) -> Row<decltype(func())>
    {
      using V = decltype(func());

      if (row.IsEmpty())
        return Row<V>();

      return Row<V>([row, func]()
      {
        return Chunk<V>(
          func(),
          MappingRowByValue(
            row.ShiftLastToFirst(),
            func));
      });
    }

```

在我们使用 `JoiningPrimeNumber()` 函数成功连接所有质数之后，我们必须使用 `Binding()` 函数将其绑定到现有的行上，如下所示：

```cpp
    template<class T, class U> Row<T> 
    Binding( Row<T> row, U func)
    {
       return JoiningAllRows( MappingRow( row, func));
    }

```

从前面的代码片段中，`MappingRow()` 函数将给定的行映射到给定的函数，然后 `JoiningAllRows()` 将从 `MappingRow()` 返回值连接所有行。`MappingRow()` 和 `JoiningAllRows()` 函数的实现如下所示：

```cpp
    template<class T, class U>
    auto MappingRow(
      Row<T> row, U func) -> Row<decltype(
        func(row.Fetch()))>
      {
        using V = decltype(func(row.Fetch()));

        if (row.IsEmpty())
          return Row<V>();

        return Row<V>([row, func]()
        {
          return Chunk<V>(func(
            row.Fetch()),
            MappingRow(
              row.ShiftLastToFirst(),
              func));
       });
    }

    template<class T> Row<T> 
    JoiningAllRows(
      Row<Row<T>> rowOfRows)
    {
      while (!rowOfRows.IsEmpty() && 
        rowOfRows.Fetch().IsEmpty())
      {
        rowOfRows = rowOfRows.ShiftLastToFirst();
      }

     if (rowOfRows.IsEmpty()) 
        return Row<T>();

     return Row<T>([rowOfRows]()
     {
        Row<T> row = rowOfRows.Fetch();

        return Chunk<T>(
          row.Fetch(), 
          ConcatenateRows(
            row.ShiftLastToFirst(), 
            JoiningAllRows(
              rowOfRows.ShiftLastToFirst())));
     });
    }

```

现在我们可以创建一个函数来限制无限整数行，如下所示：

```cpp
    Row<int> GenerateInfinitePrimeRow()
    {
      return Binding(
        GenerateInfiniteIntRow(1),
        [](int i)
        {
          return JoiningPrimeNumber(
            PrimeCheck(i),
            [i]()
            {
              return ConvertChunkToRow(i);
            });
        });
     }

```

由于 `JoiningPrimeNumber()` 函数的第二个参数需要一个行作为数据类型，我们需要使用 `ConvertChunkToRow()` 函数将 `Chunk` 转换为 `Row`，如下所示：

```cpp
    template<class T> Row<T> 
    ConvertChunkToRow(
      T value)
      {
        return Row<T>([value]()
        {
          return Chunk<T>(value);
        });
      }

```

现在我们可以消费所有前面的类和函数来重构我们的 `prime.cpp` 代码。

# 将急切评估重构为懒加载评估

我们已经拥有了所有需要的功能来将 `prime.cpp` 代码重构为懒加载代码。我们将创建一个 `prime_lazy.cpp` 代码，它首先生成无限个整数，然后选择其前一百个元素。之后，我们迭代一百个元素并将它们传递给将在控制台上打印值的函数。代码应该看起来像这样：

```cpp
    /* prime_lazy.cpp */
    #include <iostream>
    #include <cmath>
    #include "../lazyevaluation/lazyevaluation.h"

    using namespace std;

    Row<void*> PrimeCheck(int i)
    {
      // Use preceding implementation
    }

    Row<int> GenerateInfiniteIntRow(
      int initialNumber)
    {
      // Use preceding implementation
    }

    template<class T, class U>
    auto MappingRow(
      Row<T> row, U func) -> Row<decltype(
        func(row.Fetch()))>
      {     
        // Use preceding implementation
      }

    template<class T, class U>
    auto MappingRowByValue(
      Row<T> row, U func) -> Row<decltype(func())>
      {
        // Use preceding implementation
      }

    template<class T> Row<T>
    ConvertChunkToRow(
      T value)
    {
      // Use preceding implementation
    }

    template<class T> Row<T>
    JoiningAllRows(
      Row<Row<T>> rowOfRows)
    {
      // Use preceding implementation
    }

    template<class T, class U> Row<T>
    Binding(
      Row<T> row, U func)
      {
        // Use preceding implementation
      }

    template<class T, class U>
    auto JoiningPrimeNumber(
      Row<T> row, U func) -> decltype(func())
      {
        // Use preceding implementation
      }

    Row<int> GenerateInfinitePrimeRow()
    {
      // Use preceding implementation
    }

    auto main() -> int
    {
      cout << "[prime_lazy.cpp]" << endl;

      // Generating infinite prime numbers list
      Row<int> r = GenerateInfinitePrimeRow();

      // Picking the first 100 elements from preceding list
      Row<int> firstAHundredPrimeNumbers = r.Pick(100);

      // Displaying the first 100 prime numbers
      cout << "List of the first 100 prime numbers:" << endl;
      ForEach(
        move(firstAHundredPrimeNumbers),
        [](int const & i)
        {
            cout << i << "\t";
        });

      return 0;
    }

```

如前述代码所示，我们有一个名为`r`的变量，它包含无限个数，然后我们选取前一百个素数并将它们存储到`firstAHundredPrimeNumbers`中。为了将元素的值打印到控制台，我们使用`ForEach()`函数并将 Lambda 表达式传递给它。如果我们运行代码，结果将与`prime.cpp`代码完全相同，只是使用的标题不同。以下是我们运行`prime_lazy.cpp`代码时应在控制台上看到的输出：

![](img/c60d9bc0-433e-4a8a-ad78-6b0691382add.png)

通过使用`template`类，我们在本章中揭示了我们可以开发其他惰性代码以获得惰性的好处。

在前面的`prime_lazy.cpp`代码中，我省略了上一节中编写的几行代码，以避免代码冗余。如果您发现由于代码不完整而难以理解，请访问[`github.com/PacktPublishing/LearningCPPFunctionalProgramming`](https://github.com/PacktPublishing/LearningCPPFunctionalProgramming)。

# 摘要

惰性求值不仅对函数式编程有用，实际上对命令式编程也有好处。通过使用惰性求值，我们可以通过实现缓存和优化技术来编写高效且快速的代码。

在下一章中，我们将讨论在函数式方法中可以使用的元编程。我们将讨论如何使用元编程来获得所有好处，包括代码优化。
