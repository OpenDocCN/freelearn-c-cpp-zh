

# 使用 Boost.Cobalt 的协程

前几章介绍了 C++20 协程和 Boost.Asio 库，后者是使用 Boost 编写异步 **输入/输出** ( **I/O** ) 操作的基础。在本章中，我们将探讨 Boost.Cobalt，这是一个基于 Boost.Asio 的高级抽象，它简化了使用协程的异步编程。

Boost.Cobalt 允许你编写清晰、可维护的异步代码，同时避免在 C++ 中手动实现协程的复杂性（如第 *第八章* 中所述）。Boost.Cobalt 与 Boost.Asio 完全兼容，允许你在项目中无缝结合这两个库。通过使用 Boost.Cobalt，你可以专注于构建你的应用程序，而无需担心协程的低级细节。

在本章中，我们将涵盖以下 Boost.Cobalt 主题：

+   介绍 Boost.Cobalt 库

+   Boost.Cobalt 生成器

+   Boost.Cobalt 任务和承诺

+   Boost.Cobalt 通道

+   Boost.Cobalt 同步函数

# 技术要求

要构建和执行本章的代码示例，需要一个支持 C++20 的编译器。我们使用了 Clang **18** 和 GCC **14.2**。

确保你使用的是 Boost 版本 1.84 或更高版本，并且你的 Boost 库是用 C++20 支持编译的。在撰写本书时，Cobalt 在 Boost 中的支持相对较新，并非所有预编译的分发版都可能提供此组件。在阅读本书时，情况通常会得到改善。如果由于任何原因，你系统中的 Boost 库不满足这些要求，你必须从其源代码构建它。使用更早的版本，如 C++17，编译将不会包含 Boost.Cobalt，因为它严重依赖于 C++20 协程。

你可以在以下 GitHub 仓库中找到完整的代码：

[`github.com/PacktPublishing/Asynchronous-Programming-with-CPP`](https://github.com/PacktPublishing/Asynchronous-Programming-with-CPP)

本章的示例位于 **Chapter_10** 文件夹下。

# 介绍 Boost.Cobalt 库

我们在 *第八章* 中介绍了 C++20 对协程的支持。很明显，由于两个主要原因，编写协程并不是一件容易的事情：

+   在 C++ 中编写协程需要一定量的代码才能使协程工作，但这与我们要实现的功能无关。例如，我们编写的用于生成斐波那契序列的协程相当简单，但我们必须实现包装类型、承诺以及所有使其可用的函数。

+   开发 plain C++20 协程需要了解 C++ 中协程实现的底层细节，包括编译器如何将我们的代码转换为实现保持协程状态所需的所有机制，以及我们必须实现的功能的调用方式和时机。

异步编程本身就足够复杂，无需那么多细节。如果我们能专注于我们的程序，并从底层概念和代码中隔离出来，那就更好了。我们看到了 C++23 如何引入 **std::generator** 来实现这一点。让我们只写生成器代码，让 C++ 标准库和编译器处理其余部分。预计在下一个 C++ 版本中，这种协程支持将得到改进。

Boost.Cobalt 是 Boost C++ 库中包含的库之一，它允许我们做到这一点——避免协程的细节。Boost.Cobalt 在 Boost 1.84 中引入，并需要 C++20，因为它依赖于语言协程功能。它基于 Boost.Asio，我们可以在程序中使用这两个库。

Boost.Cobalt 的目标是让我们能够使用协程编写简单的单线程异步代码——可以在单个线程中同时执行多项任务的应用程序。当然，当我们说“同时”时，我们是指并发，而不是并行，因为只有一个线程。通过使用 Boost.Asio 的多线程功能，我们可以在不同的线程中执行协程，但在这个章节中，我们将专注于单线程应用程序。

## **急切协程和懒协程**

在介绍 Boost.Cobalt 实现的协程类型之前，我们需要定义两种协程类型：

+   **急切协程**：急切协程在调用时立即开始执行。这意味着协程逻辑会立即开始运行，并一直运行到遇到挂起点（例如 **co_await** 或 **co_yield**）。协程的创建实际上启动了其处理过程，并且其主体中的任何副作用都会立即执行。

    当你希望协程在创建时立即开始其工作，急切协程是有益的，例如启动异步网络操作或准备数据。

+   **懒协程**：懒协程会延迟其执行，直到被显式地等待或使用。协程对象可以在其主体中的任何代码运行之前被创建，直到调用者决定与之交互（通常是通过使用**co_await**来等待它）。

    当你需要设置一个协程但希望延迟其执行，直到满足某个条件，或者需要与其他任务协调其执行时，懒协程非常有用。

在定义了急切协程和懒协程之后，我们将描述 Boost.Cobalt 中实现的不同类型的协程。

## Boost.Cobalt 协程类型

Boost.Cobalt 实现了四种类型的协程。我们将在本节中介绍它们，并在本章后面的部分给出一些示例：

+   **承诺**：这是 Boost.Cobalt 中的主要协程类型。它用于实现返回单个值的异步操作（调用 **co_return**）。它是一个急切协程。它支持 **co_await**，允许异步挂起和继续。例如，承诺可以用来执行网络调用，当完成时，将返回其结果而不会阻塞其他操作。

+   **任务**：任务是对承诺的懒实现。它将不会开始执行，直到被显式等待。它提供了更多的灵活性来控制协程何时以及如何运行。当被等待时，任务开始执行，允许延迟处理异步操作。

+   **生成器**：在 Boost.Cobalt 中，生成器是唯一可以产生值的协程类型。每个值都是通过 **co_yield** 单独产生的。它的功能类似于 C++23 中的 **std::generator**，但它允许使用 **co_await** 等待（**std::generator** 不支持）。

+   **分离的**：这是一个急切协程，可以使用 **co_await** 但不能返回 **co_return** 值。它不能被恢复，通常也不被等待。

到目前为止，我们介绍了 Boost.Cobalt。我们定义了急切和懒协程是什么，然后我们定义了库中的四种主要协程类型。

在下一节中，我们将深入探讨与 Boost.Cobalt 相关的最重要的主题之一——生成器。我们还将实现一些简单的生成器示例。

# Boost.Cobalt 生成器

如在第 *第八章* 中所述，**生成器协程**是专门设计的协程，用于逐步产生值。在产生每个值之后，协程会暂停自身，直到调用者请求下一个值。在 Boost.Cobalt 中，生成器以相同的方式工作。它们是唯一可以产生值的协程类型。这使得生成器在您需要协程在一段时间内产生多个值时变得至关重要。

Boost.Cobalt 生成器的一个关键特性是它们默认是急切的，这意味着它们在被调用后立即开始执行。此外，这些生成器是异步的，允许它们使用 **co_await**，这是与 C++23 中引入的 **std::generator** 的重要区别，后者是懒的且不支持 **co_await**。

## 查看基本示例

让我们从最简单的 Boost.Cobalt 程序开始。这个例子不是生成器的例子，但我们将借助它解释一些重要细节：

```cpp
#include <iostream>
#include <boost/cobalt.hpp>
boost::cobalt::main co_main(int argc, char* argv[]) {
    std::cout << "Hello Boost.Cobalt\n";
    co_return 0;
}
```

在前面的代码中，我们观察到以下内容：

+   要使用 Boost.Cobalt，必须包含 **<boost/cobalt.hpp>** 头文件。

+   您还必须将 Boost.Cobalt 库链接到您的应用程序。我们提供了一个 **CMakeLists.txt** 文件来完成这项工作，不仅适用于 Boost.Cobalt，还适用于所有必需的 Boost 库。要显式地链接 Boost.Cobalt（即不是所有必需的 Boost 库），只需将以下行添加到您的 **CMakeLists.txt** 文件中：

    ```cpp
    target_link_libraries(${EXEC_NAME} Boost::cobalt)
    ```

+   使用**co_main**函数。与常规的**main**函数不同，Boost.Cobalt 引入了一个基于协程的入口点，称为**co_main**。这个函数可以使用协程特定的关键字，如**co_return**。Boost.Cobalt 内部实现了所需的**main**函数。

    使用**co_main**将允许您将程序的**main**函数（入口点）实现为协程，从而能够调用**co_await**和**co_return**。记住，从*第八章*中，**main**函数不能是协程。

    如果您无法更改当前的**主**函数，可以使用 Boost.Cobalt。您只需从**main**函数中调用一个函数，这个函数将成为您使用 Boost.Cobalt 的异步代码的最高级函数。实际上，这正是 Boost.Cobalt 所做的事情：它实现了一个**主**函数，这是程序的入口点，并且（对您隐藏的）这个**主**函数调用了**co_main**。

    使用您自己的**main**函数的最简单方法可能如下所示：

    ```cpp
    cobalt::task<int> async_task() {
        // your code here
        // …
        return 0;
    }
    int main() {
        // main function code
        // …
        return cobalt::run(async_code();
    }
    ```

示例简单地打印一条问候消息，然后通过调用**co_await**返回 0。在所有未来的例子中，我们将遵循这个模式：包含**<boost/cobalt.hpp>**头文件，并使用**co_main**而不是**main**。

## Boost.Cobalt 简单生成器

在我们之前的基本例子中获得的知识的基础上，我们将实现一个非常简单的生成器协程：

```cpp
#include <chrono>
#include <iostream>
#include <boost/cobalt.hpp>
using namespace std::chrono_literals;
using namespace boost;
cobalt::generator<int> basic_generator()
{
    std::this_thread::sleep_for(1s);
    co_yield 1;
    std::this_thread::sleep_for(1s);
    co_return 0;
}
cobalt::main co_main(int argc, char* argv[]) {
    auto g = basic_generator();
    std::cout << co_await g << std::endl;
    std::cout << co_await g << std::endl;
    co_return 0;
}
```

上述代码展示了一个简单的生成器，它产生一个整数值（使用**co_yield**）并返回另一个值（使用**co_return**）。

**cobalt::generator**是一个**struct**模板：

```cpp
template<typename Yield, typename Push = void>
struct generator
```

两个参数类型如下：

+   **Yield**：生成的对象类型

+   **Push**：输入参数类型（默认为**void**）

**co_main**函数在通过**co_await**获取数值后打印这两个数（调用者等待数值可用）。我们已经引入了一些延迟来模拟生成器必须执行的处理以生成这些数字。

我们的第二个生成器将产生一个整数的平方：

```cpp
#include <chrono>
#include <iostream>
#include <boost/cobalt.hpp>
using namespace std::chrono_literals;
using namespace boost;
cobalt::generator<int, int> square_generator(int x){
    while (x != 0) {
        x = co_yield x * x;
    }
    co_return 0;
}
cobalt::main co_main(int argc, char* argv[]){
    auto g = square_generator(10);
    std::cout << co_await g(4) << std::endl;
    std::cout << co_await g(12) << std::endl;
    std::cout << co_await g(0) << std::endl;
    co_return 0;
}
```

在这个例子中，**square_generator**产生**x**参数的平方。这展示了我们如何将值推送到 Boost.Cobalt 生成器。在 Boost.Cobalt 中，将值推送到生成器意味着传递参数（在先前的例子中，传递的参数是整数）。

在这个例子中，尽管生成器是正确的，但可能会令人困惑。请看以下代码行：

```cpp
auto g = square_generator(10);
```

这创建了一个初始值为**10**的生成器对象。然后，看看以下代码行：

```cpp
std::cout << co_await g(4) << std::endl;
```

这将打印**10**的平方并将**4**推送到生成器。正如你所看到的，打印的值不是传递给生成器的值的平方。这是因为生成器初始化时有一个值（在这个例子中，**10**），当调用者调用**co_await**传递另一个值时，它将生成平方值。当接收到新值**4**时，生成器将产生**100**，然后当接收到**12**的值时，它将产生**16**，依此类推。

我们说过，Boost.Cobalt 生成器是急切的，但它们在开始执行时可以等待（**co_await**）。以下示例展示了如何做到这一点：

```cpp
#include <iostream>
#include <boost/cobalt.hpp>
boost::cobalt::generator<int, int> square_generator() {
    auto x = co_await boost::cobalt::this_coro::initial;
    while (x != 0) {
        x = co_yield x * x;
    }
    co_return 0;
}
boost::cobalt::main co_main(int, char*[]) {
    auto g = square_generator();
    std::cout << co_await g(4) << std::endl;
    std::cout << co_await g(10) << std::endl;
    std::cout << co_await g(12) << std::endl;
    std::cout << co_await g(0) << std::endl;
    co_return 0;
}
```

代码与上一个示例非常相似，但有一些不同：

+   我们创建生成器时没有传递任何参数给它：

    ```cpp
    auto g = square_generator();
    ```

+   看一下生成器代码的第一行：

    ```cpp
    auto x = co_await boost::cobalt::this_coro::initial;
    ```

    这使得生成器等待第一个推入的整数。这表现得像一个惰性生成器（实际上，它立即开始执行，因为生成器是急切的，但它首先做的事情是等待一个整数）。

+   产生的值是我们从代码中期望得到的：

    ```cpp
    std::cout << co_await g(10) << std::endl;
    ```

    这将打印**100**而不是之前推入整数的平方。

让我们在这里总结一下示例做了什么：**co_main**函数调用**square_generator**协程生成整数的平方。生成器协程在开始时挂起等待第一个整数，并在产生每个平方后再次挂起。这个例子故意简单，只是为了说明如何使用 Boost.Cobalt 编写生成器。

前一个程序的一个重要特性是它在单个线程中运行。这意味着**co_main**和生成器协程一个接一个地运行。

## 一个斐波那契数列生成器

在本节中，我们将实现一个类似于我们在*第八章*中实现的斐波那契数列生成器。这将让我们看到使用 Boost.Cobalt 编写生成器协程比使用纯 C++20（不使用任何协程库）要容易多少。

我们编写了两个版本的生成器。第一个计算斐波那契数列的任意项。我们推入我们想要生成的项，然后我们得到它。这个生成器使用 lambda 作为斐波那契计算器：

```cpp
boost::cobalt::generator<int, int> fibonacci_term() {
    auto fibonacci = [](int n) {
        if (n < 2) {
            return n;
        }
        int f0 = 0;
        int f1 = 1;
        int f;
        for (int i = 2; i <= n; ++i) {
            f = f0 + f1;
            f0 = f1;
            f1 = f;
        }
        return f;
    };
    auto x = co_await boost::cobalt::this_coro::initial;
    while (x != -1) {
        x = co_yield fibonacci(x);
    }
    co_return 0;
 }
```

在前面的代码中，我们看到这个生成器与我们之前章节中用于计算数字平方的生成器非常相似。在协程的开始，我们有以下内容：

```cpp
auto x = co_await boost::cobalt::this_coro::initial;
```

这行代码使协程挂起以等待第一个输入值。

然后我们有以下内容：

```cpp
while (x != -1) {
        x = co_yield fibonacci(x);
    }
```

这生成所需的斐波那契数列项，并在请求下一个项之前挂起。当请求的项不等于**-1**时，我们可以继续请求更多值，直到推入**-1**终止协程。

下一个版本的斐波那契生成器将在需要时产生无限多个项。当我们说“无限”时，我们是指“潜在无限”。将这个生成器想象成总是准备好产生下一个斐波那契数列的数字：

```cpp
boost::cobalt::generator<int> fibonacci_sequence() {
    int f0 = 0;
    int f1 = 1;
    int f = 0;
    while (true) {
        co_yield f0;
        f = f0 + f1;
        f0 = f1;
        f1 = f;
    }
}
```

前面的代码很容易理解：协程产生一个值并暂停自己，直到另一个值被请求，然后协程计算新值并产生它，再次在无限循环中暂停自己。

在这种情况下，我们可以看到协程的优势：我们可以在需要时逐个生成斐波那契数列的项。我们不需要保持任何状态来生成下一个项，因为状态被保存在协程中。

还要注意，即使函数执行了无限循环，因为它是一个协程，它会暂停并再次恢复，从而避免阻塞当前线程。

# Boost.Cobalt 任务和承诺

正如我们在本章中已经看到的，Boost.Cobalt 的承诺是急切协程，它们返回一个值，而 Boost.Cobalt 的任务是承诺的懒版本。

我们可以将其视为只是函数，不像生成器那样产生多个值。我们可以多次调用承诺以获取多个值，但调用之间不会保持状态（就像生成器中那样）。基本上，承诺是一个可以使用**co_await**（它也可以使用**co_return**）的协程。

承诺的不同用法可能是一个套接字监听器，用于接收网络数据包，处理它们，对数据库进行查询，然后从数据中生成一些结果。一般来说，它们的功能需要异步等待某个结果，然后对该结果进行一些处理（或者可能只是将其返回给调用者）。

我们的第一个例子是一个简单的承诺，它生成一个随机数（这也可以用生成器来完成）：

```cpp
#include <iostream>
#include <random>
#include <boost/cobalt.hpp>
boost::cobalt::promise<int> random_number(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(min, max);
    co_return dist(gen);
}
boost::cobalt::promise<int> random(int min, int max) {
    int res = co_await random_number(min, max);
    co_return res;
}
boost::cobalt::main co_main(int, char*[]) {
    for (int i = 0; i < 10; ++i) {
        auto r = random(1, 100);
        std::cout << "random number between 1 and 100: "
                  << co_await r << std::endl;
    }
    co_return 0;
}
```

在前面的代码中，我们已经编写了三个协程：

+   **co_main**：记住在 Boost.Cobalt 中，**co_main**是一个协程，它调用**co_return**来返回一个值。

+   **random()**：这个协程返回一个随机数给调用者。它使用**co_await**调用**random()**来生成随机数。它异步等待随机数的生成。

+   **random_number()**：这个协程生成两个值**min**和**max**之间的均匀分布随机数，并将其返回给调用者。**random_number()**也是一个承诺。

下面的协程返回一个包含随机数的**std::vector<int>**。在循环中调用**co_await random_number()**来生成一个包含**n**个随机数的向量：

```cpp
boost::cobalt::promise<std::vector<int>> random_vector(int min, int max, int n) {
    std::vector<int> rv(n);
    for (int i = 0; i < n; ++i) {
        rv[i] = co_await random_number(min, max);
    }
    co_return rv;
}
```

前面的函数返回一个**std::vector<int>**的承诺。要访问这个向量，我们需要调用**get()**：

```cpp
auto v = random_vector(1, 100, 20);
for (int n : v.get()) {
    std::cout << n << " ";
}
std::cout << std::endl;
```

之前的代码打印了**v**向量的元素。要访问这个向量，我们需要调用**v.get()**。

我们将实现第二个示例来展示承诺和任务的执行有何不同：

```cpp
#include <chrono>
#include <iostream>
#include <thread>
#include <boost/cobalt.hpp>
void sleep(){
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
boost::cobalt::promise<int> eager_promise(){
    std::cout << "Eager promise started\n";
    sleep();
    std::cout << "Eager promise done\n";
    co_return 1;
}
boost::cobalt::task<int> lazy_task(){
    std::cout << "Lazy task started\n";
    sleep();
    std::cout << "Lazy task done\n";
    co_return 2;
}
boost::cobalt::main co_main(int, char*[]){
    std::cout << "Calling eager_promise...\n";
    auto promise_result = eager_promise();
    std::cout << "Promise called, but not yet awaited.\n";
    std::cout << "Calling lazy_task...\n";
    auto task_result = lazy_task();
    std::cout << "Task called, but not yet awaited.\n";
    std::cout << "Awaiting both results...\n";
    int promise_value = co_await promise_result;
    std::cout << "Promise value: " << promise_value
              << std::endl;
    int task_value = co_await task_result;
    std::cout << "Task value: " << task_value
              << std::endl;
    co_return 0;
}
```

在这个例子中，我们实现了两个协程：一个承诺（promise）和一个任务（task）。正如我们之前所说的，承诺是急切的，它一旦被调用就开始执行。任务则是懒加载的，在被调用后会被挂起。

当我们运行程序时，它会打印出所有消息，这让我们确切地知道协程是如何执行的。

执行完 **co_main()** 的前三行后，打印的输出如下：

```cpp
Calling eager_promise...
Eager promise started
Eager promise done
Promise called, but not yet awaited.
```

从这些消息中，我们知道承诺已经执行到调用 **co_return** 的位置。

执行完 **co_main()** 的下一三行后，打印的输出有这些新消息：

```cpp
Calling lazy_task...
Task called, but not yet awaited.
```

在这里，我们看到任务尚未执行。它是一个懒加载的协程，因此，在被调用后立即挂起，并且这个协程还没有打印任何消息。

执行了三行更多的 **co_main()** 代码，这些是新消息，程序输出的内容如下：

```cpp
Awaiting both results...
Promise value: 1
```

在承诺上调用 **co_await** 会给我们其结果（在这个例子中，设置为 **1**）并且执行结束。

最后，我们在任务上调用 **co_await**，然后它执行并返回其值（在这个例子中，设置为 **2**）。输出如下：

```cpp
Lazy task started
Lazy task done
Task value: 2
```

这个例子展示了任务是如何懒加载的，开始时是挂起的，并且只有在调用者对它们调用 **co_await** 时才会恢复执行。

在本节中，我们看到了，就像生成器的情况一样，使用 Boost.Cobalt 比仅仅使用纯 C++ 更容易编写承诺和任务协程。我们不需要编写 C++ 实现协程所需的所有支持代码。我们也看到了任务和承诺之间的主要区别。

在下一节中，我们将研究一个通道的例子，这是一个在生产者/消费者模型中两个协程之间的通信机制。

# Boost.Cobalt 通道

在 Boost.Cobalt 中，通道为协程提供了异步通信的方式，允许生产者和消费者协程之间以安全且高效的方式进行数据传输。它们受到了 Golang 通道的启发，并允许通过消息传递进行通信，促进了一种“通过通信共享内存”的范式。

**通道**是一种机制，通过它，值可以从一个协程（生产者）异步地传递到另一个协程（消费者）。这种通信是非阻塞的，这意味着协程在等待通道上有可用数据时可以挂起它们的执行，或者在向具有有限容量的通道写入数据时也可以挂起。让我们澄清一下：如果“阻塞”意味着协程被挂起，那么读取和写入操作可能会根据缓冲区大小而阻塞，但另一方面，从线程的角度来看，这些操作不会阻塞线程。

如果缓冲区大小为零，读取和写入将需要同时发生并作为 rendezvous（同步通信）。如果通道大小大于零且缓冲区未满，写入操作不会挂起协程。同样，如果缓冲区不为空，读取操作也不会挂起。

类似于 Golang 的通道，Boost.Cobalt 的通道是强类型的。通道为特定类型定义，并且只能通过它发送该类型的数据。例如，**int** 类型的通道（**boost::cobalt::channel<int>**）只能传输整数。

现在让我们看看一个通道的示例：

```cpp
#include <iostream>
#include <boost/cobalt.hpp>
#include <boost/asio.hpp>
boost::cobalt::promise<void> producer(boost::cobalt::channel<int>& ch) {
    for (int i = 1; i <= 10; ++i) {
        std::cout << "Producer waiting for request\n";
        co_await ch.write(i);
        std::cout << "Producing value " << i << std::endl;
    }
    std::cout << "Producer end\n";
    ch.close();
    co_return;
}
boost::cobalt::main co_main(int, char*[]) {
    boost::cobalt::channel<int> ch;
    auto p = producer(ch);
    while (ch.is_open()) {
        std::cout << "Consumer waiting for next number \n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
        auto n = co_await ch.read();
        std::cout << "Consuming value " << n << std::endl;
        std::cout << n * n << std::endl;
    }
    co_await p;
    co_return 0;
}
```

在这个示例中，我们创建了一个大小为 **0** 的通道和两个协程：**生产者**承诺和作为消费者的 **co_main()**。生产者将整数写入通道，消费者读取它们并将它们平方后打印出来。

我们添加了 `**std::this_thread::sleep**` 来延迟程序执行，从而能够看到程序运行时的状态。让我们看看示例输出的摘录，看看它是如何工作的：

```cpp
Producer waiting for request
Consumer waiting for next number
Producing value 1
Producer waiting for request
Consuming value 1
1
Consumer waiting for next number
Producing value 2
Producer waiting for request
Consuming value 2
4
Consumer waiting for next number
Producing value 3
Producer waiting for request
Consuming value 3
9
Consumer waiting for next number
```

消费者和生产者都等待下一个动作发生。生产者将始终等待消费者请求下一个项目。这基本上是生成器的工作方式，并且在使用协程的异步代码中是一个非常常见的模式。

消费者执行以下代码行：

```cpp
auto n = co_await ch.read();
```

然后，生产者将下一个数字写入通道并等待下一个请求。这是在以下代码行中完成的：

```cpp
co_await ch.write(i);
```

你可以在上一段输出摘录的第四行中看到生产者如何返回等待下一个请求。

Boost.Cobalt 通道使得编写这种异步代码非常清晰且易于理解。

示例显示了两个协程通过通道进行通信。

这部分内容就到这里。下一部分将介绍同步函数——等待多个协程的机制。

# Boost.Cobalt 同步函数

之前，我们实现了协程，并且在每次调用 `**co_await**` 的时候，我们只为一个协程调用。这意味着我们只等待一个协程的结果。Boost.Cobalt 有机制允许我们等待多个协程。这些机制被称为 **同步函数**。

Boost.Cobalt 实现了四个同步函数：

+   **race**：**race** 函数等待一组协程中的一个完成，但它以伪随机的方式进行。这种机制有助于避免协程的饥饿，确保一个协程不会在执行流程上主导其他协程。当你有多个异步操作，并且想要第一个完成以确定流程时，**race** 将允许任何准备就绪的协程以非确定性的顺序继续执行。

    当你有多个任务（在通用意义上，不是 Boost.Cobalt 任务）并且对完成其中一个感兴趣，没有偏好哪个，但想要防止在准备就绪同时发生的情况下一个协程总是获胜时，你会使用**race**。

+   **join**：**join**函数等待给定集合中的所有协程完成，并返回它们的值。如果任何一个协程抛出异常，**join**将把异常传播给调用者。这是一种从多个异步操作中收集结果的方法，这些操作必须在继续之前全部完成。

    当你需要多个异步操作的结果一起，并且如果任何一个操作失败则想要抛出错误时，你会使用**join**。

+   **gather**：与**join**类似，**gather**函数等待一组协程完成，但它处理异常的方式不同。当其中一个协程失败时，**gather**不会立即抛出异常，而是单独捕获每个协程的结果。这意味着你可以独立检查每个协程的输出（成功或失败）。

    当你需要所有异步操作都完成，但想要单独捕获所有结果和异常以分别处理时，你会使用**gather**。

+   **left_race**：**left_race**函数类似于**race**，但具有确定性行为。它从左到右评估协程，并等待第一个协程准备好。当协程完成的顺序很重要，并且你想要基于它们提供的顺序确保可预测的结果时，这可能很有用。

    当你有多个潜在的结果，并且需要优先考虑提供的顺序中的第一个可用的协程，使行为比**race**更可预测时，你会使用**left_race**。

在本节中，我们将探讨**join**和**gather**函数的示例。正如我们所见，这两个函数都等待一组协程完成。它们之间的区别在于，如果任何一个协程抛出异常，**join**将抛出一个异常，而**gather**总是返回所有等待的协程的结果。在**gather**函数的情况下，每个协程的结果将要么是一个错误（缺失值），要么是一个值。**join**返回一个值元组或抛出一个异常；**gather**返回一个可选值元组，在发生异常的情况下没有值（可选变量未初始化）。

以下示例的完整代码在 GitHub 仓库中。在这里，我们将关注主要部分。

我们定义了一个简单的函数来模拟数据处理，它仅仅是一个延迟。如果传递的延迟大于 5,000 毫秒，该函数将抛出一个异常：

```cpp
boost::cobalt::promise<std::chrono::milliseconds::rep> process(std::chrono::milliseconds ms) {
    if (ms > std::chrono::milliseconds(5000)) {
        throw std::runtime_error("delay throw");
    }
    boost::asio::steady_timer tmr{ co_await boost::cobalt::this_coro::executor, ms };
    co_await tmr.async_wait(boost::cobalt::use_op);
    co_return ms.count();
}
```

该函数是一个 Boost.Cobalt 承诺。

现在，在代码的下一节中，我们将等待这个承诺的三个实例运行：

```cpp
auto result = co_await boost::cobalt::join(process(100ms),
                                           process(200ms),
                                           process(300ms));
std::cout << "First coroutine finished in: "
          <<  std::get<0>(result) << "ms\n";
std::cout << "Second coroutine took finished in: "
          <<  std::get<1>(result) << "ms\n";
std::cout << "Third coroutine took finished in: "
         <<  std::get<2>(result) << "ms\n";
```

前面的代码调用**join**等待三个协程完成，然后打印它们所花费的时间。正如你所看到的，结果是元组，为了使代码尽可能简单，我们只为每个元素调用**std::get<i>(result)**。在这种情况下，所有处理时间都在有效范围内，没有抛出异常，因此我们可以获取所有已执行协程的结果。

如果抛出异常，则我们不会得到任何值：

```cpp
try {
    auto result throw = co_await
    boost::cobalt::join(process(100ms),
                        process(20000ms),
                        process(300ms));
}
catch (...) {
    std::cout << "An exception was thrown\n";
}
```

前面的代码将抛出异常，因为第二个协程接收到的处理时间超出了有效范围。它将打印一条错误信息。

当调用**join**函数时，我们希望所有协程都被视为处理的一部分，并且在发生异常的情况下，整个处理失败。

如果我们需要获取每个协程的所有结果，我们将使用**gather**函数：

```cpp
try
    auto result throw =
    boost::cobalt::co_await lt::gather(process(100ms),
                                       process(20000ms),
                                       process(300ms));
    if (std::get<0>(result throw).has value()) {
        std::cout << "First coroutine took: "
                  <<  *std::get<0>(result throw)
                  << "msec\n";
    }
    else {
        std::cout << "First coroutine threw an exception\n";
    }
    if (std::get<1>(result throw).has value()) {
        std::cout << "Second coroutine took: "
                  <<  *std::get<1>(result throw)
                  << "msec\n";
    }
    else {
        std::cout << "Second coroutine threw an exception\n";
    }
    if (std::get<2>(result throw).has value()) {
        std::cout << "Third coroutine took: "
                  <<  *std::get<2>(result throw)
                  << "msec\n";
    }
    else {
        std::cout << "Third coroutine threw an exception\n";
    }
}
catch (...) {
    // this is never reached because gather doesn't throw exceptions
    std::cout << "An exception was thrown\n";
}
```

我们将代码放在了**try-catch**块中，但没有抛出异常。**gather**函数返回一个可选值的元组，我们需要检查每个协程是否返回了值（可选值是否有值）。

当我们希望协程在成功执行时返回一个值时，我们使用**gather**。

这些**join**和**gather**函数的例子结束了我们对 Boost.Cobalt 同步函数的介绍。

# 摘要

在本章中，我们看到了如何使用 Boost.Cobalt 库实现协程。它最近才被添加到 Boost 中，关于它的信息并不多。它简化了使用协程异步代码的开发，避免了编写 C++20 协程所需的底层代码。

我们研究了主要库概念，并开发了一些简单的示例来理解它们。

使用 Boost.Cobalt，使用协程编写异步代码变得简单。C++中编写协程的所有底层细节都由库实现，我们可以专注于我们想要在程序中实现的功能。

在下一章中，我们将看到如何调试异步代码。

# 进一步阅读

+   Boost.Cobalt 参考：*Boost.Cobalt 参考* *指南* ([`www.boost.org/doc/libs/1_86_0/libs/cobalt/doc/html/index.html#overview`](https://www.boost.org/doc/libs/1_86_0/libs/cobalt/doc/html/index.html#overview) )

+   一个关于 Boost.Cobalt 的 YouTube 视频：*使用 Boost.Cobalt 进行协程* ([`www.youtube.com/watch?v=yElSdUqEvME`](https://www.youtube.com/watch?v=yElSdUqEvME) )

# 第五部分：异步编程中的调试、测试和性能优化

在本最终部分，我们专注于调试、测试和优化多线程和异步程序性能的基本实践。我们将首先使用日志记录和高级调试工具和技术，包括反向调试和代码清理器，来识别和解决异步应用程序中的微妙错误，例如崩溃、死锁、竞态条件、内存泄漏和线程安全问题，随后使用 GoogleTest 框架针对异步代码制定测试策略。最后，我们将深入性能优化，理解诸如缓存共享、伪共享以及如何缓解性能瓶颈等关键概念。掌握这些技术将为我们提供一套全面的工具集，用于识别、诊断和改进异步应用程序的质量和性能。

本部分包含以下章节：

+   *第十一章* ，*异步软件的日志记录和调试*

+   *第十二章* ，*清理和测试异步软件*

+   *第十三章* ，*提高异步软件性能*
