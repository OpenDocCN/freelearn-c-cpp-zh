# 函数式编程中的函数操作

在上一章中，我们深入讨论了现代 C++，特别是关于 C++11 的新特性——Lambda 表达式。正如我们之前讨论的，Lambda 表达式在简化函数表示法方面很有用。因此，在本章中，我们将再次应用 Lambda 表达式的力量，它将用于函数式代码，尤其是在我们讨论柯里化——将当前函数拆分和简化的技术时。 

在本章中，我们将讨论以下主题：

+   应用一级函数和高阶函数，以便我们的函数不仅可以作为函数调用，还可以分配给任何变量，传递一个函数，并返回一个函数

+   纯函数，以避免在函数中产生副作用，因为它不再接触外部状态

+   柯里化，如本章开头所述，用于简化多参数函数，以便我们可以评估一系列函数，每个函数只有一个参数。

# 在所有函数中应用一级函数

一级函数只是一个普通的类。我们可以像对待任何其他数据类型一样对待一级函数。然而，在支持一级函数的语言中，我们可以执行以下任务而无需递归调用编译器：

+   将函数作为另一个函数的参数传递

+   将函数分配给变量

+   将函数存储在集合中

+   在运行时从现有函数创建新函数

幸运的是，C++可以用来解决上述任务。我们将在以下主题中深入讨论。

# 将函数作为另一个函数的参数传递

让我们开始传递一个函数作为函数参数。我们将选择四个函数之一，并从其主函数中调用该函数。代码将看起来像这样：

```cpp
    /* first_class_1.cpp */
    #include <functional>
    #include <iostream>

    using namespace std;

    // Defining a type of function named FuncType
    // representing a function
    // that pass two int arguments
    // and return an int value
    typedef function<int(int, int)> FuncType;

    int addition(int x, int y)
    {
      return x + y;
    }

    int subtraction(int x, int y)
    {
      return x - y;
    }

    int multiplication(int x, int y)
    {
      return x * y;
    }

    int division(int x, int y)
    {
      return x / y;
    }

    void PassingFunc(FuncType fn, int x, int y)
    {
      cout << "Result = " << fn(x, y) << endl;
    }

    auto main() -> int
    {
      cout << "[first_class_1.cpp]" << endl;
      int i, a, b;
      FuncType func;

      // Displaying menu for user
      cout << "Select mode:" << endl;
      cout << "1\. Addition" << endl;
      cout << "2\. Subtraction" << endl;
      cout << "3\. Multiplication" << endl;
      cout << "4\. Division" << endl;
      cout << "Choice: ";
      cin >> i;

      // Preventing user to select
      // unavailable modes
      if(i < 1 || i > 4)
      {
         cout << "Please select available mode!";
         return 1;
      }

      // Getting input from user for variable a
      cout << "a -> ";
      cin >> a;

      // Input validation for variable a
      while (cin.fail())
      {
        // Clearing input buffer to restore cin to a usable state
        cin.clear();

        // Ignoring last input
        cin.ignore(INT_MAX, '\n');

        cout << "You can only enter numbers.\n";
        cout << "Enter a number for variable a -> ";
        cin >> a;
      }

      // Getting input from user for variable b
      cout << "b -> ";
      cin >> b;

      // Input validation for variable b
      while (cin.fail())
      {
        // Clearing input buffer to restore cin to a usable state
        cin.clear();

        // Ignoring last input
        cin.ignore(INT_MAX, '\n');

        cout << "You can only enter numbers.\n";
        cout << "Enter a number for variable b -> ";
        cin >> b;
      }
      switch(i)
      {
        case 1: PassingFunc(addition, a, b); break;
        case 2: PassingFunc(subtraction, a, b); break;
        case 3: PassingFunc(multiplication, a, b); break;
        case 4: PassingFunc(division, a, b); break;
      }

      return 0;
    }

```

从前面的代码中，我们可以看到我们有四个函数，我们希望用户选择一个，然后运行它。在 `switch` 语句中，我们将根据用户的选择调用四个函数之一。我们将选定的函数传递给 `PassingFunc()`，如下面的代码片段所示：

```cpp
    case 1: PassingFunc(addition, a, b); break;
    case 2: PassingFunc(subtraction, a, b); break;
    case 3: PassingFunc(multiplication, a, b); break;
    case 4: PassingFunc(division, a, b); break;

```

我们还有输入验证来防止用户选择不可用的模式，以及为变量 `a` 和 `b` 输入非整数值。屏幕上显示的输出应该看起来像这样：

![图片](img/e7a276f3-b116-4d3e-80fd-12b4fe9e197f.png)

上一张截图显示，我们从可用模式中选择 `乘法` 模式。然后，我们尝试为变量 `a` 输入 `r` 和 `e` 变量。幸运的是，由于我们进行了输入验证，程序拒绝了它。然后，我们将 `4` 分配给变量 `a`，将 `2` 分配给变量 `b`。正如我们所预期的那样，程序给出了 `8` 作为结果。

正如我们在`first_class_1.cpp`程序中看到的，我们使用`std::function`类和`typedef`关键字来简化代码。`std::function`类用于存储、复制和调用任何可调用函数、Lambda 表达式或其他函数对象，以及成员函数指针和数据成员指针。然而，`typedef`关键字用作其他类型或函数的别名。

# 将函数分配给变量

我们还可以将函数分配给变量，这样我们就可以通过调用变量来调用函数。我们将重构`first_class_1.cpp`，如下所示：

```cpp
    /* first_class_2.cpp */
    #include <functional>
    #include <iostream>

    using namespace std;

    // Defining a type of function named FuncType
    // representing a function
    // that pass two int arguments
    // and return an int value
    typedef function<int(int, int)> FuncType;

    int addition(int x, int y)
    {
      return x + y;
    }

    int subtraction(int x, int y)
    {
      return x - y;
    }

    int multiplication(int x, int y)
    {
      return x * y;
    }

    int division(int x, int y)
    {
      return x / y;
    }

    auto main() -> int
    {
      cout << "[first_class_2.cpp]" << endl;

      int i, a, b;
      FuncType func;

      // Displaying menu for user
      cout << "Select mode:" << endl;
      cout << "1\. Addition" << endl;
      cout << "2\. Subtraction" << endl;
      cout << "3\. Multiplication" << endl;
      cout << "4\. Division" << endl;
      cout << "Choice: ";
      cin >> i;

      // Preventing user to select
      // unavailable modes
      if(i < 1 || i > 4)
      {
        cout << "Please select available mode!";
        return 1;
      }

      // Getting input from user for variable a
      cout << "a -> ";
      cin >> a;

      // Input validation for variable a
      while (cin.fail())
      {
        // Clearing input buffer to restore cin to a usable state
        cin.clear();

        // Ignoring last input
        cin.ignore(INT_MAX, '\n');

        cout << "You can only enter numbers.\n";
        cout << "Enter a number for variable a -> ";
        cin >> a;
      }

      // Getting input from user for variable b
      cout << "b -> ";
      cin >> b;

      // Input validation for variable b
      while (cin.fail())
      {
        // Clearing input buffer to restore cin to a usable state
        cin.clear();

        // Ignoring last input
        cin.ignore(INT_MAX, '\n');

        cout << "You can only enter numbers.\n";
        cout << "Enter a number for variable b -> ";
        cin >> b;
      }

      switch(i)
      {
        case 1: func = addition; break;
        case 2: func = subtraction; break;
        case 3: func = multiplication; break;
        case 4: func = division; break;
      }

      cout << "Result = " << func(a, b) << endl;

      return 0;
    }

```

现在，我们将根据用户的选择分配四个函数，并在 switch 语句中将选定的函数存储在`func`变量中，如下所示：

```cpp
    case 1: func = addition; break;
    case 2: func = subtraction; break;
    case 3: func = multiplication; break;
    case 4: func = division; break;

```

在`func`变量被分配了用户的选项后，代码将像调用函数一样调用该变量，如下面的代码行所示：

```cpp
    cout << "Result = " << func(a, b) << endl;

```

然后，如果我们运行代码，我们将在控制台上获得相同的输出。

# 在容器中存储函数

现在，让我们将函数保存到容器中。在这里，我们将使用**vector**作为容器。代码如下：

```cpp
    /* first_class_3.cpp */
    #include <vector>
    #include <functional>
    #include <iostream>

    using namespace std;

    // Defining a type of function named FuncType
    // representing a function
    // that pass two int arguments
    // and return an int value
    typedef function<int(int, int)> FuncType;

    int addition(int x, int y)
    {
      return x + y;
    }

    int subtraction(int x, int y)
    {
      return x - y;
    }

    int multiplication(int x, int y)
    {
      return x * y;
    }

    int division(int x, int y)
    {
      return x / y;
    }

    auto main() -> int
    {
      cout << "[first_class_3.cpp]" << endl;

      // Declaring a vector containing FuncType element
      vector<FuncType> functions;

      // Assigning several FuncType elements to the vector
      functions.push_back(addition);
      functions.push_back(subtraction);
      functions.push_back(multiplication);
      functions.push_back(division);

      int i, a, b;
      function<int(int, int)> func;

      // Displaying menu for user
      cout << "Select mode:" << endl;
      cout << "1\. Addition" << endl;
      cout << "2\. Subtraction" << endl;
      cout << "3\. Multiplication" << endl;
      cout << "4\. Division" << endl;
      cout << "Choice: ";
      cin >> i;

      // Preventing user to select
      // unavailable modes
      if(i < 1 || i > 4)
      {
        cout << "Please select available mode!";
        return 1;
      }

      // Getting input from user for variable a
      cout << "a -> ";
      cin >> a;

      // Input validation for variable a
      while (cin.fail())
      {
        // Clearing input buffer to restore cin to a usable state
        cin.clear();

        // Ignoring last input
        cin.ignore(INT_MAX, '\n');

        cout << "You can only enter numbers.\n";
        cout << "Enter a number for variable a -> ";
        cin >> a;
      }

      // Getting input from user for variable b
      cout << "b -> ";
      cin >> b;

      // Input validation for variable b
      while (cin.fail())
      {
        // Clearing input buffer to restore cin to a usable state
        cin.clear();

        // Ignoring last input
        cin.ignore(INT_MAX, '\n');

        cout << "You can only enter numbers.\n";
        cout << "Enter a number for variable b -> ";
        cin >> b;
      }

      // Invoking the function inside the vector
      cout << "Result = " << functions.at(i - 1)(a, b) << endl;

      return 0;
    }

```

从前面的代码中，我们可以看到我们创建了一个名为`functions`的新向量，然后将其存储了四个不同的函数。就像我们在前两个代码示例中所做的那样，我们要求用户选择模式。然而，现在代码变得更简单，因为我们不需要添加 switch 语句；我们可以通过选择向量索引直接选择函数，如下面的代码片段所示：

```cpp
    cout << "Result = " << functions.at(i - 1)(a, b) << endl;

```

然而，由于向量是**基于零的索引**，我们必须根据菜单选择调整索引。结果将与我们的前两个代码示例相同。

# 在运行时从现有函数创建新函数

现在，让我们在运行时从现有函数创建一个新的函数。假设我们有两个函数集合，第一个是双曲函数，第二个是第一个的逆。除了这些内置函数之外，我们还在第一个集合中添加了一个用户定义的函数来计算平方数，并在第二个集合中计算平方数的逆。然后，我们将实现函数组合并从两个现有函数构建一个新的函数。

**函数组合**是将两个或更多简单函数组合成一个更复杂函数的过程。每个函数的结果作为下一个函数的参数传递。最终结果是从最后一个函数的结果获得的。在数学方法中，我们通常使用以下符号来表示函数组合：`compose(f, g) (x) = f(g(x))`。让我们假设我们有以下代码：

`double x, y, z; // ... y = g(x); z = f(y);`

因此，为了简化符号，我们可以使用函数组合，并为*z*使用以下符号：

`z = f(g(x));`

如果我们运行双曲函数，然后将结果传递给其逆函数，我们会看到我们确实得到了传递给双曲函数的原始值。现在，让我们看一下以下代码：

```cpp
    /* first_class_4.cpp */
    #include <vector>
    #include <cmath>
    #include <algorithm>
    #include <functional>
    #include <iostream>

    using std::vector;
    using std::function;
    using std::transform;
    using std::back_inserter;
    using std::cout;
    using std::endl;

    // Defining a type of function named HyperbolicFunc
    // representing a function
    // that pass a double argument
    // and return an double value
    typedef function<double(double)> HyperbolicFunc;

    // Initializing a vector containing four functions
    vector<HyperbolicFunc> funcs = {
      sinh,
      cosh,
      tanh,
      [](double x) {
        return x*x; }
    };

    // Initializing a vector containing four functions
    vector<HyperbolicFunc> inverseFuncs = {
      asinh,
      acosh,
      atanh,
      [](double x) {
        return exp(log(x)/2); }
    };

    // Declaring a template to be able to be reused
    template <typename A, typename B, typename C>
    function<C(A)> compose(
      function<C(B)> f,
      function<B(A)> g) {
        return f,g {
            return f(g(x));
      };
    }

    auto main() -> int
    {
      cout << "[first_class_4.cpp]" << endl;

      // Declaring a template to be able to be reused
      vector<HyperbolicFunc> composedFuncs;

      // Initializing a vector containing several double elements
      vector<double> nums;
      for (int i = 1; i <= 5; ++i)
        nums.push_back(i * 0.2);

      // Transforming the element inside the vector
      transform(
        begin(inverseFuncs),
        end(inverseFuncs),
        begin(funcs),
        back_inserter(composedFuncs),
        compose<double, double, double>);

      for (auto num: nums)
      {
        for (auto func: composedFuncs)
            cout << "f(g(" << num << ")) = " << func(num) << endl;

        cout << "---------------" << endl;
      }

      return 0;
    }

```

正如我们可以在前面的代码中看到的那样，我们有两个函数集合--`funcs`和`inverseFuncs`。此外，正如我们之前讨论的那样，`inverseFuncs`函数是`funcs`函数的逆函数。`funcs`函数包含三个内置的双曲函数，以及一个用户定义的函数来计算平方数，而`inverseFuncs`包含三个内置的双曲逆函数，以及一个用户定义的函数来计算平方数的逆。

正如我们可以在前面的`first_class_4.cpp`代码中看到的那样，当我们使用`using`关键字时，我们会使用单独的类/函数。与本章中的其他代码示例相比，单独的类/函数中使用`using`关键字是不一致的，因为我们使用了`using namespace std`。这是因为`std`命名空间中存在冲突的函数名，所以我们必须单独调用它们。

通过使用这两个函数集合，我们将从它们中构建一个新的函数。为了达到这个目的，我们将使用`transform()`函数将来自两个不同集合的两个函数结合起来。代码片段如下：

```cpp
 transform(
 begin(inverseFuncs), 
 inverseFuncs.end(inverseFuncs), 
 begin(funcs), 
 back_inserter(composedFuncs), 
 compose<double, double, double>);

```

现在，我们在`composedFuncs`向量中存储了一个新的函数集合。我们可以遍历这个集合，并将我们在`nums`变量中提供的值传递给这个新函数。如果我们运行代码，应该在控制台上获得以下输出：

![](img/020e5ce8-760c-4446-82c2-7aeca8fddc1a.png)

正如我们从前面的输出中可以看到的那样，无论我们传递什么给转换函数，我们都会得到与输入相同的输出。在这里，我们可以证明 C++编程可以用来从两个或更多现有函数中组合一个函数。

在前面的`first_class_4.cpp`代码中，我们使用了`template<>`。如果您需要关于`template<>`的更详细解释，请参阅第七章，*使用并发运行并行执行*。

# 在高阶函数中熟悉三种函数式技术

我们讨论了在第一类函数中，C++语言将函数视为值，这意味着我们可以将它们传递给其他函数，赋值给变量等等。然而，在函数式编程中，我们还有一个术语，那就是高阶函数，这些函数作用于其他函数。这意味着高阶函数可以将函数作为参数传递，也可以返回一个函数。

高阶函数的概念可以应用于一般函数，就像数学函数一样，而不是只能应用于函数式编程语言中的第一类函数概念。现在，让我们来检查函数式编程中最有用的三个高阶函数--**map**、**filter**和**fold**。

# 使用 map 执行每个元素列表

我们不会讨论 C++语言中将 map 作为一个容器，而是将其作为一个高阶函数的特性。这个特性用于将给定的函数应用到列表的每个元素上，并按相同的顺序返回一个结果列表。我们可以使用`transform()`函数来实现这个目的。正如你所知，我们之前已经讨论过这个函数。然而，我们可以看一下以下代码片段来查看`transform()`函数的使用：

```cpp
    /* transform_1.cpp */
    #include <vector>
    #include <algorithm>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[transform_1.cpp]" << endl;

      // Initializing a vector containing integer element
      vector<int> v1;
      for (int i = 0; i < 5; ++i)
        v1.push_back(i);

      // Creating another v2 vector
      vector<int> v2;
      // Resizing the size of v2 exactly same with v1
      v2.resize(v1.size());

      // Transforming the element inside the vector
      transform (
        begin(v1),
        end(v1),
        begin(v2),
        [](int i){
            return i * i;});

      // Displaying the elements of v1
      std::cout << "v1 contains:";
      for (auto v : v1)
        std::cout << " " << v;
      std::cout << endl;

      // Displaying the elements of v2
      std::cout << "v2 contains:";
      for (auto v : v2)
        std::cout << " " << v;
      std::cout << endl;

      return 0;
    }

```

如我们之前在高阶函数中对 map 的定义所示，它将给定的函数应用到列表的每个元素上。在之前的代码中，我们尝试使用 Lambda 表达式中的给定函数将`v1`向量映射到`v2`向量，如下所示：

```cpp
 transform (
      begin(v1), 
      end(v1), 
      begin(v2), 
      [](int i){
        return i * i;});

```

如果我们运行代码，我们应该在控制台屏幕上得到以下输出：

![图片](img/30abb408-577c-4cfd-82c7-8b1030998fc0.png)

如我们在输出显示中看到的那样，我们使用 Lambda 表达式中的给定函数将`v1`转换为`v2`，该函数是对输入值的双倍。

# 使用 filter 提取数据

在高阶函数中，过滤是一个函数，它从现有数据结构中产生一个新的数据结构，该数据结构中的每个元素都与给定的返回布尔值的谓词精确匹配。在 C++语言中，我们可以应用 C++11 中添加的`copy_if()`函数来获得过滤过程。让我们看一下以下代码片段，以分析使用`copy_if()`函数的过滤过程：

```cpp
    /* filter_1.cpp */
    #include <vector>
    #include <algorithm>
    #include <iterator>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[filter_1.cpp]" << endl;

      // Initializing a vector containing integer elements
      vector<int> numbers;
      for (int i = 0; i < 20; ++i)
        numbers.push_back(i);

       // Displaying the elements of numbers
       cout << "The original numbers: " << endl;
       copy(
        begin(numbers),
        end(numbers),
        ostream_iterator<int>(cout, " "));
       cout << endl;

       // Declaring a vector containing int elements
       vector<int> primes;

      // Filtering the vector
      copy_if(
        begin(numbers),
        end(numbers),
        back_inserter(primes),
        [](int n) {
            if(n < 2) {
                return (n != 0) ? true : false;}
            else {
                for (int j = 2; j < n; ++j) {
                    if (n % j == 0){
                        return false;}
            }

            return true;
         }});

        // Displaying the elements of primes
        // using copy() function
        cout << "The primes numbers: " << endl;
        copy(
         begin(primes),
         end(primes),
         ostream_iterator<int>(cout, " "));
         cout << endl;

         return 0;
    }

```

如我们在前面的代码中看到的那样，我们使用`copy_if()`函数将`numbers`向量过滤到`0`素数向量。我们将传递 Lambda 表达式来决定选定的元素是否是素数，就像我们在第一章，《深入现代 C++》中的`lambda_multiline_func.cpp`代码中使用的 Lambda 表达式一样。我们还将使用`copy()`函数将选定的向量中的所有元素复制出来以打印。当我们运行前面的代码时，结果应该是这样的：

![图片](img/84a91b01-5fb5-4111-afaa-c2ca77c868ca.png)

除了`copy_if()`函数之外，我们还可以使用`remove_copy_if()`函数来过滤数据结构。使用`remove_copy_if()`函数不是从现有数据结构中选择匹配谓词的元素，而是省略匹配谓词的元素，选择不匹配的元素，并将其存储在新数据结构中。让我们重构我们的`filter_1.cpp`代码，创建一个新的非素数向量。代码如下：

```cpp
    /* filter_2.cpp */
    #include <vector>
    #include <algorithm>
    #include <iterator>
    #include <iostream>

    using namespace std;

    int main()
   {
      cout << "[filter_2.cpp]" << endl;

      // Initializing a vector containing integer elements
      vector<int> numbers;
      for (int i = 0; i < 20; ++i)
        numbers.push_back(i);

      // Displaying the elements of numbers
      cout << "The original numbers: " << endl;
      copy(
        begin(numbers),
        end(numbers),
        ostream_iterator<int>(cout, " "));
      cout << endl;

      // Declaring a vector containing int elements
      vector<int> nonPrimes;

      // Filtering the vector
      remove_copy_if(
        numbers.begin(),
        numbers.end(),
        back_inserter(nonPrimes),
        [](int n) {
            if(n < 2){
                return (n != 0) ? true : false;}
            else {
                for (int j = 2; j < n; ++j){
                    if (n % j == 0) {
                        return false;}
            }

            return true;
        }});

      // Displaying the elements of nonPrimes
      // using copy() function
      cout << "The non-primes numbers: " << endl;
      copy(
        begin(nonPrimes),
        end(nonPrimes),
        ostream_iterator<int>(cout, " "));
      cout << endl;

      return 0;
    }

```

正如我们在前面高亮的代码中看到的，我们重构了之前的代码，并使用`remove_copy_if()`函数来选择非素数。正如我们所预期的，控制台窗口将显示以下输出：

![](img/c0b50b7a-1cfb-48ed-b552-c4bfb806a152.png)

现在我们有了非素数，而不是素数，就像我们在`filter_1.cpp`代码中所看到的那样。

# 使用 fold 组合列表的所有元素

在函数式编程中，fold 是一种将数据结构归约到单个值的技术。有两种类型的 fold--左 fold（`foldl`）和右 fold（`foldr`）。假设我们有一个包含 0, 1, 2, 3 和 4 的列表。让我们使用 fold 技术来添加列表中的所有内容，首先使用`foldl`，然后使用`foldr`。然而，两者之间有一个显著的区别--`foldl`是左结合的，这意味着我们首先组合最左边的元素，然后向右移动。例如，根据我们拥有的列表，我们将得到以下括号：

```cpp
    ((((0 + 1) + 2) + 3) + 4)

```

虽然`foldr`是右结合的，这意味着我们将从最右边的元素开始组合，然后向左移动。括号将像以下代码行所示：

```cpp
    (0 + (1 + (2 + (3 + 4))))

```

现在，让我们看看以下代码：

```cpp
    /* fold_1.cpp */
    #include <vector>
    #include <numeric>
    #include <functional>
    #include <iostream>

    using namespace std;

    auto main() -> int
    {
      cout << "[fold_1.cpp]" << endl;

      // Initializing a vector containing integer elements
      vector<int> numbers = {0, 1, 2, 3, 4};

      // Calculating the sum of the value
      // in the vector
      auto foldl = accumulate(
        begin(numbers),
        end(numbers),
        0,
        std::plus<int>());

      // Calculating the sum of the value
      // in the vector
      auto foldr = accumulate(
        rbegin(numbers),
        rend(numbers),
        0,
        std::plus<int>());

      // Displaying the calculating result
      cout << "foldl result = " << foldl << endl;
      cout << "foldr result = " << foldr << endl;

      return 0;
    }

```

在 C++编程中，我们可以使用`accumulate()`函数应用`fold`技术。正如我们可以在前面的代码中看到，我们在`foldl`中使用前向迭代器，而在`foldr`中使用后向迭代器。控制台上的输出应该像以下截图所示：

![](img/9cb482af-59c6-4f00-b396-d4fbabd686c9.png)

正如我们在前面的输出截图中所看到的，对于`foldl`和`foldr`两种技术，我们都得到了相同的结果。对于那些对求和顺序感到好奇的人，我们可以将前面的代码重构为以下形式：

```cpp
    /* fold_2.cpp */
    #include <vector>
    #include <numeric>
    #include <functional>
    #include <iostream>

    using namespace std;

    // Function for logging the flow
    int addition(const int& x, const int& y)
    {
      cout << x << " + " << y << endl;
      return x + y;
    }

    int main()
    {
      cout << "[fold_2.cpp]" << endl;

      // Initializing a vector containing integer elements
      vector<int> numbers = {0, 1, 2, 3, 4};

      // Calculating the sum of the value
      // in the vector
      // from left to right
      cout << "foldl" << endl;
      auto foldl = accumulate(
          begin(numbers),
          end(numbers),
          0,
          addition);

      // Calculating the sum of the value
      // in the vector
      // from right to left
      cout << endl << "foldr" << endl;
      auto foldr = accumulate(
          rbegin(numbers),
          rend(numbers),
          0,
          addition);

      cout << endl;

      // Displaying the calculating result
      cout << "foldl result = " << foldl << endl;
      cout << "foldr result = " << foldr << endl;

      return 0;
    }

```

在前面的代码中，我们传递了一个新的`addition()`函数并将其传递给`accumulate()`函数。从`addition()`函数中，我们将跟踪每个元素的操作。现在，让我们运行前面的代码，其输出如下：

![](img/5b662182-a4b8-4a3f-a969-29983904dcbb.png)

从前面的输出截图可以看出，尽管`foldl`和`foldr`给出了完全相同的结果，但它们执行的操作顺序不同。由于我们设置了初始值为`0`，加法操作在`foldl`技术中从将`0`加到第一个元素开始，在`foldr`技术中从将`0`加到最后一个元素开始。

我们将初始值设为`0`，因为`0`是加法的单位元，它不会影响加法的结果。然而，在乘法中，我们必须考虑将初始值改为`1`，因为`1`是乘法的单位元。

# 使用纯函数避免副作用

**纯函数**是一个函数，每次给定相同的输入时都会返回相同的结果。结果不依赖于任何信息或状态，并且不会产生**副作用**，即函数外部系统状态的变化。让我们看看以下代码片段：

```cpp
    /* pure_function_1.cpp */
    #include <iostream>

    using namespace std;

    float circleArea(float r)
    {
      return 3.14 * r * r;
    }

    auto main() -> int
    {
      cout << "[pure_function_1.cpp]" << endl;

      // Initializing a float variable
      float f = 2.5f;

      // Invoking the circleArea() function
      // passing the f variable five times
      for(int i = 1; i <= 5; ++i)
      {
        cout << "Invocation " << i << " -> ";
        cout << "Result of circleArea(" << f << ") = ";
        cout << circleArea(f) << endl;
      }

      return 0;
    }

```

从上述代码中，我们可以看到一个名为 `circleArea()` 的函数，它根据给定的半径计算圆的面积。然后我们调用该函数五次，并传递相同的半径值。控制台上的输出应该如下所示：

![截图](img/363316b8-632e-4346-a2f6-50003c5a8c14.png)

正如我们所见，在五次调用中传递相同的输入，函数也返回相同的输出。因此，我们可以说 `circleArea()` 是一个纯函数。现在，让我们看看不纯函数在以下代码片段中的样子：

```cpp
    /* impure_function_1.cpp */
    #include <iostream>

    using namespace std;

    // Initializing a global variable
    int currentState = 0;

    int increment(int i)
    {
      currentState += i;
      return currentState;
    }

    auto main() -> int
    {
      cout << "[impure_function_1.cpp]" << endl;

      // Initializing a local variable
      int fix = 5;

      // Involving the global variable
      // in the calculation
      for(int i = 1; i <= 5; ++i)
      {
        cout << "Invocation " << i << " -> ";
        cout << "Result of increment(" << fix << ") = ";
        cout << increment(fix) << endl;
      }

       return 0;
    }

```

在上述代码中，我们看到一个名为 `increment()` 的函数增加了 `currentState` 变量的值。正如我们所见，`increment()` 函数依赖于 `currentState` 变量的值，因此它不是一个纯函数。让我们通过运行上述代码来证明这一点。控制台窗口应该显示以下截图：

![代码截图](img/d0f1d544-925c-447a-9780-1ae4db9ca6f1.png)

我们看到，即使我们传递相同的输入，`increment()` 函数也会给出不同的结果。这是不纯函数在依赖于外部状态或改变外部状态值时的副作用。

我们已经能够区分纯函数和不纯函数。然而，考虑以下代码：

```cpp
    /* im_pure_function_1.cpp */
    #include <iostream>

    using namespace std;

    // Initializing a global variable
    float phi = 3.14f;

    float circleArea(float r)
    {
      return phi * r * r;
    }

    auto main() -> int
    {
      cout << "[im_pure_function_1.cpp]" << endl;

      // Initializing a float variable
      float f = 2.5f;

      // Involving the global variable
      // in the calculation
      for(int i = 1; i <= 5; ++i)
      {
        cout << "Invocation " << i << " -> ";
        cout << "Result of circleArea(" << f << ") = ";
        cout << circleArea(f) << endl;
      }

      return 0;
    }

```

上述代码来自 `pure_function_1.cpp`，但我们添加了一个全局状态，`phi`。如果我们运行上述代码，我们肯定会获得与 `pure_function_1.cpp` 相同的结果。尽管函数在五次调用中返回相同的结果，但 `im_pure_function_1.cpp` 中的 `circleArea()` 不是一个纯函数，因为它依赖于 `phi` 变量。

副作用不仅包括函数所做的全局状态的变化。向屏幕打印也是副作用。然而，由于我们需要展示我们创建的每一行代码的结果，我们无法避免在代码中存在打印到屏幕的情况。在下一章中，我们还将讨论不可变状态，这是我们将不纯函数转换为纯函数的方法。

# 使用柯里化减少多参数函数

柯里化是一种技术，它将接受多个参数的函数分解为评估一系列函数，每个函数只有一个参数。换句话说，我们通过减少当前函数来创建基于当前函数的其他函数。假设我们有一个名为 `areaOfRectangle()` 的函数，它接受两个参数，`length` 和 `width`。代码将如下所示：

```cpp
    /* curry_1.cpp */

    #include <functional>
    #include <iostream>

    using namespace std;

    // Variadic template for currying
    template<typename Func, typename... Args>
    auto curry(Func func, Args... args)
    {
      return =
      {
        return func(args..., lastParam...);
      };
    }

    int areaOfRectangle(int length, int width)
    {
      return length * width;
    }

    auto main() -> int
    {
      cout << "[curry_1.cpp]" << endl;

      // Currying the areaOfRectangle() function
      auto length5 = curry(areaOfRectangle, 5);

      // Invoking the curried function
      cout << "Curried with spesific length = 5" << endl;
      for(int i = 0; i <= 5; ++i)
      {
        cout << "length5(" << i << ") = ";
        cout << length5(i) << endl;
      }

      return 0;
    }

```

如前述代码所示，我们有一个名为`curry`的可变模板和函数。我们将使用此模板来构建 currying 函数。在常规函数调用中，我们可以像以下这样调用`areaOfRectangle()`函数：

```cpp
    int i = areaOfRectangle(5, 2);

```

如前述代码片段所示，我们将`5`和`2`作为参数传递给`areaOfRectangle()`函数。然而，使用 curried 函数，我们可以简化`areaOfRectangle()`函数，使其只有一个参数。我们只需像以下一样调用 curry 函数模板：

```cpp
 auto length5 = curry(areaOfRectangle, 5);

```

现在，我们有一个名为`areaOfRectangle()`的函数，它具有名为`length`的参数值`length5`。对我们来说，调用该函数并仅添加`width`参数要容易得多，如下面的代码片段所示：

```cpp
 length5(i) // where i is the width parameter we want to pass

```

让我们看看运行前述代码时在控制台上将看到的输出：

![图片](img/1097f10c-3b57-448c-8d71-fc69bf9f4918.png)

可变模板和函数帮助我们简化了`areaOfRectangle()`函数成为`length5()`函数。然而，它也可以帮助我们简化具有两个以上参数的函数。假设我们有一个名为`volumeOfRectanglular()`的函数，它传递三个参数。我们将同样简化该函数，如下面的代码所示：

```cpp
    /* curry_2.cpp */

    #include <functional>
    #include <iostream>

    using namespace std;

    // Variadic template for currying
    template<typename Func, typename... Args>
    auto curry(Func func, Args... args)
    {
      return =
      {
        return func(args..., lastParam...);
      };
    }

    int volumeOfRectanglular(
      int length,
      int width,
      int height)
     {
        return length * width * height;
     }

    auto main() -> int
    {
      cout << "[curry_2.cpp]" << endl;

      // Currying the volumeOfRectanglular() function
      auto length5width4 = curry(volumeOfRectanglular, 5, 4);

      // Invoking the curried function
      cout << "Curried with spesific data:" << endl;
      cout << "length = 5, width 4" << endl;
      for(int i = 0; i <= 5; ++i)
      {
        cout << "length5width4(" << i << ") = ";
        cout << length5width4(i) << endl;
      }

      return 0;
    }

```

如前述代码所示，我们已经成功地将`length`和`width`参数传递给`volumeOfRectanglular()`函数，然后将其简化为`length5width4()`。我们可以调用`length5width4()`函数，并仅传递剩余的参数，即`height`。以下是在我们运行前述代码时将在控制台屏幕上看到的输出：

![图片](img/dbbc74e9-74e1-4a84-af5e-59a4b8494f0c.png)

通过使用 currying 技术，我们可以通过简化函数，使其只传递单个参数，部分评估一个多参数函数。

# 摘要

我们已经讨论过有一些技术可以用来操作一个函数。我们可以从中获得许多优势。由于我们可以在 C++语言中实现一等函数，因此我们可以将一个函数作为另一个函数的参数传递。我们可以将函数视为一个数据对象，因此我们可以将其分配给一个变量并将其存储在容器中。此外，我们可以从现有的函数中组合出一个新的函数。此外，通过使用 map、filter 和 fold，我们可以在我们创建的每个函数中实现高阶函数。

在获取更好的函数式代码方面，我们必须实现另一种技术，即纯函数以避免副作用。我们可以重构我们所有的函数，使其不会与外部变量或状态通信，也不会从外部状态更改和检索值。此外，为了减少多个参数的函数以便我们可以评估其序列，我们可以在我们的函数中实现 currying 技术。

在下一章中，我们将讨论另一种避免副作用的技术。我们将使代码中的所有状态都是不可变的，因此每次函数被调用时，都没有状态会被突变。
