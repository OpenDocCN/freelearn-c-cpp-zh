# 第二十二章

# 单元测试和调试

实际上，您使用哪种编程语言或开发什么类型的应用程序并不重要，在交付给客户之前彻底测试它始终很重要。

编写测试并不是一件新鲜事，截至今天，您几乎可以在每个软件项目中找到数百甚至数千个测试。如今，编写软件测试是必须的，没有经过适当测试的代码或功能被强烈反对。这就是为什么我们有一个专门的章节来讨论用 C 编写的软件测试，以及今天存在用于此目的的各种库。

然而，本章不仅仅讨论测试。我们还将讨论可用于调试 C 程序的调试工具和技术。测试和调试从一开始就相互补充，每当测试失败时，都会进行一系列调查，调试目标代码是常见的后续行动。

在本章中，我们不会讲解测试的哲学，而是假设测试是好的。相反，我们将为您提供一个关于基本术语和开发者在编写可测试代码时应遵循的指南的简要介绍。

本章分为两部分。第一部分，我们讨论测试和可用于现代 C 开发的现有库。本章的第二部分将讨论调试，从讨论各种类型的错误开始。内存问题、并发问题和性能问题是似乎需要进一步调试以建立成功调查的最常见情况。

我们还将涵盖适用于 C（和 C++）的最常用的调试工具。本章的最终目标是让您了解 C 和调试工具，并为您提供一些基本背景知识。

第一部分向您介绍了软件测试的基本术语。它不仅限于 C，这些思想和概念也可以应用于其他编程语言和技术。

# 软件测试

软件测试是计算机编程中的一个庞大且重要的主题，它有自己的特定术语和许多概念。在本节中，我们将为您提供一个关于软件测试的非常基础的介绍。我们的目的是定义一些我们将在本章前半部分使用的术语。因此，您应该意识到这不是一个关于测试的详尽章节，强烈建议进一步学习。

当谈到测试软件时，首先想到的问题是，我们在测试什么，这次测试是关于什么的？一般来说，我们测试软件系统的一个方面。这个方面可以是 *功能* 或 *非功能* 的。换句话说，这个方面可能与系统的某个功能相关，或者当执行功能时，可能与系统的某个变量相关。接下来，我们给出一些例子。

*功能测试* 是关于测试作为 *功能需求* 部分请求的特定功能。这些测试向一个 *软件元素*，例如一个 *函数*、一个 *模块*、一个 *组件* 或一个 *软件系统*，提供一定的输入，并期望从它们那里获得一定的输出。只有当期望的输出被视为测试的一部分时，该测试才被认为是 *通过* 的。

*非功能测试* 是关于软件元素，例如一个函数、一个模块、一个组件或整个软件系统，完成特定功能的质量水平。这些测试通常旨在 *测量* 各种 *变量*，如 *内存使用*、*完成时间*、*锁竞争* 和 *安全性级别*，并评估该元素完成其工作的程度。只有当测量的变量在预期的范围内时，测试才被认为是 *通过* 的。这些变量的 *预期值* 来自为系统定义的 *非功能需求*。

除了功能和非功能测试之外，我们还可以有不同的 *测试级别*。这些级别的设计方式是为了覆盖一些正交方面。这些方面中的一些是测试元素的大小、测试的参与者以及应该测试的功能范围的广度。

例如，就元素的大小而言，这些级别是从可能的最小功能块定义的，我们称之为函数（或方法），到从整个软件系统暴露出的可能最大的功能块。

在以下部分，我们将更深入地介绍这些级别。

测试级别

对于每个软件系统，可以考虑和计划以下测试级别。这些并不是唯一的测试级别，你可以在其他参考资料中找到更多：

+   单元测试

+   集成测试

+   系统测试

+   接受测试

+   回归测试

在 *单元测试* 中，我们测试一个 *功能单元*。这个单元可以是一个执行特定任务的函数，或者是一组函数组合起来满足需求，或者是一个有最终目标执行特定功能的类，甚至是一个有特定任务要完成的组件。一个 *组件* 是软件系统的一部分，它有一组定义良好的功能，并且与其他组件一起结合，成为整个软件系统。

在组件作为单元的情况下，我们将测试过程称为*组件测试*。功能和非功能测试都可以在单元层面进行。在测试单元时，该单元应与其周围单元隔离，为此，周围环境应以某种方式模拟。这个层面将是本章涵盖的唯一层面，我们提供实际代码来演示如何在 C 中进行单元测试和组件测试。

当单元组合在一起时，它们形成一个组件。在组件测试中，组件单独进行测试。但当我们将这些组件分组时，我们需要不同层面的测试来检查该特定组件组的函数性或变量。这个层面称为*集成测试*。正如其名所示，这个层面的测试检查某些组件的集成是否良好，并且它们一起仍然满足系统定义的要求。

在不同层面，我们测试整个系统的功能。这将包含所有完全集成的组件的完整集合。这样，我们测试暴露的系统功能性和系统变量是否与为软件系统定义的要求一致。

在不同层面，我们评估一个软件系统是否与从*利益相关者*或*最终用户*角度定义的业务需求一致。这个层面被称为*验收测试*。虽然系统测试和验收测试都是关于整个软件系统，但它们实际上相当不同。以下是一些差异：

+   系统测试由开发人员和测试人员执行，但验收测试通常由最终用户或利益相关者执行。

+   系统测试是关于检查功能和非功能需求，但验收测试只关注功能需求。

+   在系统测试中，我们通常使用准备好的小数据集作为输入，但在验收测试中，实际实时数据被输入到系统中。

一个很好的链接，解释了所有这些差异，可以在[`www.javatpoint.com/acceptance-tes`](https://www.javatpoint.com/acceptance-testing)ting 找到。

当向软件系统引入变更时，需要检查当前的功能和非功能测试是否仍然处于良好状态。这在不同层面进行，称为*回归测试*。回归测试的目的是确认在引入变更后没有发生*回归*。作为回归测试的一部分，所有作为单元测试、集成测试和端到端（系统）测试发现的测试都会再次运行，以查看是否有任何测试在变更后失败。

在本节中，我们介绍了各种测试级别。在本章的其余部分，我们将讨论单元测试。在接下来的部分，我们将通过给出一个 C 示例并尝试为其编写测试用例来开始讨论它。

# 单元测试

如我们在上一节中解释的，作为单元测试的一部分，我们测试独立的单元，一个单元可以小到函数，也可以大到组件。在 C 中，它可以是一个函数，也可以是整个用 C 编写的组件。同样的讨论也适用于 C++，但在那里我们可以有其他单元，如类。

单元测试最重要的地方是单元应该单独进行测试。例如，如果目标函数依赖于另一个函数的输出，我们需要找到一种方法来单独测试目标函数。我们将通过一个真实示例来解释这一点。

*示例 22.1* 打印小于 10 的偶数的阶乘，但不是以通常的方式。代码在一个头文件和两个源文件中组织得很好。本例涉及两个函数；其中一个函数生成小于 10 的偶数，另一个函数接收一个函数指针并将其用作读取整数数的源，并最终计算其阶乘。

以下代码框包含包含函数声明的头文件：

```cpp
#ifndef _EXTREME_C_EXAMPLE_22_1_
#define _EXTREME_C_EXAMPLE_22_1_
#include <stdint.h>
#include <unistd.h>
typedef int64_t (*int64_feed_t)();
int64_t next_even_number();
int64_t calc_factorial(int64_feed_t feed);
#endif
```

Code Box 22-1 [ExtremeC_examples_chapter22_1.h]：示例 22.1 的头文件

如你所见，`calc_factorial` 函数接受一个返回整数的函数指针。它将使用该函数指针来读取一个整数并计算其阶乘。以下代码是前面函数的定义：

```cpp
#include "ExtremeC_examples_chapter22_1.h"
int64_t next_even_number() {
  static int feed = -2;
  feed += 2;
  if (feed >= 10) {
    feed = 0;
  }
  return feed;
}
int64_t calc_factorial(int64_feed_t feed) {
  int64_t fact = 1;
  int64_t number = feed();
  for (int64_t i = 1; i <= number; i++) {
    fact *= i;
  }
  return fact;
}
```

Code Box 22-2 [ExtremeC_examples_chapter22_1.c]：示例 22.1 中使用的函数的定义

`next_even_number` 函数有一个内部静态变量，它作为调用函数的输入。请注意，它永远不会超过 8，之后它回到 0。因此，你可以简单地多次调用这个函数，而永远不会得到一个大于 8 且小于零的数字。以下代码框包含包含 `main` 函数的源文件的内容：

```cpp
#include <stdio.h>
#include "ExtremeC_examples_chapter22_1.h"
int main(int argc, char** argv) {
  for (size_t i = 1; i <= 12; i++) {
    printf("%lu\n", calc_factorial(next_even_number));
  }
  return 0;
}
```

Code Box 22-3 [ExtremeC_examples_chapter22_1_main.c]：示例 22.1 的主函数

如你所见，`main` 函数调用了 `calc_function` 12 次，并打印了返回的阶乘。为了运行前面的示例，你需要首先编译这两个源文件，然后将它们相应的可重定位目标文件链接在一起。以下 shell box 包含构建和运行示例所需的命令：

```cpp
$ gcc -c ExtremeC_examples_chapter22_1.c -o impl.o
$ gcc -c ExtremeC_examples_chapter22_1_main.c -o main.o
$ gcc impl.o main.o -o ex22_1.out
$ ./ex22_1.out
1
2
24
720
40320
1
2
24
720
40320
1
2
$
```

Shell Box 22-1：构建和运行示例 22.1

为了编写前面函数的测试，我们首先需要做一些介绍。正如你所见，示例中有两个函数（不包括`main`函数）。因此，存在两个不同的单元，在这种情况下是函数，应该分别且独立于彼此进行测试；一个是`next_even_number`函数，另一个是`calc_factorial`函数。但是，正如主函数中所示，`calc_factorial`函数依赖于`next_even_number`函数，有人可能会认为这种依赖会使`calc_factorial`函数的隔离比我们预期的要困难得多。但这并不是真的。

事实上，`calc_factorial`函数根本不依赖于`next_even_number`函数。它只依赖于`next_even_number`的*签名*，而不是其定义。因此，我们可以用一个遵循相同签名的函数来替换`next_even_number`，但总是返回一个固定的整数。换句话说，我们可以提供一个简化的`next_even_number`版本，这个版本仅打算在*测试用例*中使用。

那么，什么是测试用例呢？正如你所知，有各种场景可以用来测试特定的单元。最简单的例子是向某个单元提供各种输入并*期望*得到预定的输出。在先前的例子中，我们可以为`calc_factorial`函数提供`0`作为输入，并等待其输出为`1`。我们也可以提供`-1`并等待其输出为`1`。

这些场景中的每一个都可以成为一个测试用例。因此，针对单个单元，我们可以有多个测试用例，以解决该单元的所有不同边界情况。测试用例的集合称为*测试套件*。测试套件中找到的所有测试用例不一定与同一个单元相关。

我们首先为`next_even_number`函数创建一个测试套件。由于`next_even_number`可以很容易地独立测试，因此不需要额外的工作。以下是为`next_even_number`函数编写的测试用例：

```cpp
#include <assert.h>
#include "ExtremeC_examples_chapter22_1.h"
void TESTCASE_next_even_number__even_numbers_should_be_returned() {
  assert(next_even_number() == 0);
  assert(next_even_number() == 2);
  assert(next_even_number() == 4);
  assert(next_even_number() == 6);
  assert(next_even_number() == 8);
}
void TESTCASE_next_even_number__numbers_should_rotate() {
  int64_t number = next_even_number();
  next_even_number();
  next_even_number();
  next_even_number();
  next_even_number();
  int64_t number2 = next_even_number();
  assert(number == number2);
}
```

代码框 22-4 [ExtremeC_examples_chapter22_1 __next_even_number__tests.c]：为`next_even_number`函数编写的测试用例

如你所见，我们在先前的测试套件中定义了两个测试用例。请注意，我使用了自己的约定来命名上述测试用例；然而，这并没有标准。整个命名测试用例的目的是从其名称中了解测试用例的作用，更重要的是，当测试用例失败或需要修改时，可以在代码中轻松找到它。

我使用大写`TESTCASE`作为函数名称的前缀，以便与其它普通函数区分开来。函数的名称也试图描述测试用例及其所关注的问题。

两个测试用例都在最后使用了 `assert`。这是所有测试用例函数在评估期望时都会做的事情。如果 `assert` 括号内的条件不成立，*测试运行器*，一个正在运行测试的程序，会退出并打印错误信息。不仅如此，测试运行器还会返回一个非零的 *退出码*，这表明一个或多个测试用例失败了。当所有测试都成功时，测试运行器程序必须返回 0。

很好，你可以自己走一遍测试用例，尝试理解它们是如何通过调用前面两个场景中的 `next_even_number` 函数来评估我们的期望的。

现在，是时候为 `calc_factorial` 函数编写测试用例了。为 `calc_factorial` 函数编写测试用例需要一个 *存根函数* 作为其输入。我们简要解释一下存根是什么。

以下是三个仅测试 `calc_factorial` 单元的测试用例：

```cpp
#include <assert.h>
#include "ExtremeC_examples_chapter22_1.h"
int64_t input_value = -1;
int64_t feed_stub() {
  return input_value;
}
void TESTCASE_calc_factorial__fact_of_zero_is_one() {
  input_value = 0;
  int64_t fact = calc_factorial(feed_stub);
  assert(fact == 1);
}
void TESTCASE_calc_factorial__fact_of_negative_is_one() {
  input_value = -10;
  int64_t fact = calc_factorial(feed_stub);
  assert(fact == 1);
}
void TESTCASE_calc_factorial__fact_of_5_is_120() {
  input_value = 5;
  int64_t fact = calc_factorial(feed_stub);
  assert(fact == 120);
}
```

代码框 22-5 [ExtremeC_examples_chapter22_1 __calc_factorial__tests.c]：为 `calc_factorial` 函数编写的测试用例

如你所见，我们为 `calc_factorial` 函数定义了三个测试用例。注意 `feed_stub` 函数。它遵循与 `next_even_number` 相同的契约，如 *代码框 22-2* 所示，但它有一个非常简单的定义。它只是返回存储在静态变量 `input_value` 中的值。这个变量可以在调用 `calc_factorial` 函数之前由测试用例设置。

使用前面的存根函数，我们可以隔离 `calc_factorial` 并单独测试它。同样的方法也适用于 C++ 或 Java 这样的面向对象编程语言，但我们在那里定义 *存根类* 和 *存根对象*。

在 C 语言中，*存根* 是一个符合目标单元逻辑中使用的函数声明的函数定义，更重要的是，存根没有复杂的逻辑，它只是返回一个将被测试用例使用的值。

在 C++ 中，存根仍然可以是一个符合函数声明的函数定义，或者是一个实现接口的类。在其他无法有独立函数的对象导向语言中，例如 Java，存根只能是一个实现接口的类。然后，存根对象是从这样的存根类中创建的对象。请注意，在所有情况下，存根都应该有一个简单的定义，仅适用于测试，而不适用于生产。

最后，我们需要能够运行测试用例。正如我们之前所说的，我们需要一个测试运行器来运行测试。因此，我们需要一个包含 `main` 函数的特定源文件，该函数只依次运行测试用例。下面的代码框包含了测试运行器的代码：

```cpp
#include <stdio.h>
void TESTCASE_next_even_number__even_numbers_should_be_returned();
void TESTCASE_next_even_number__numbers_should_rotate();
void TESTCASE_calc_factorial__fact_of_zero_is_one();
void TESTCASE_calc_factorial__fact_of_negative_is_one();
void TESTCASE_calc_factorial__fact_of_5_is_120();
int main(int argc, char** argv) {
  TESTCASE_next_even_number__even_numbers_should_be_returned();
  TESTCASE_next_even_number__numbers_should_rotate();
  TESTCASE_calc_factorial__fact_of_zero_is_one();
  TESTCASE_calc_factorial__fact_of_negative_is_one();
  TESTCASE_calc_factorial__fact_of_5_is_120();
  printf("All tests are run successfully.\n");
  return 0;
}
```

代码框 22-6 [ExtremeC_examples_chapter22_1 _tests.c]：示例 22.1 中使用的测试运行器

上述代码仅在`main`函数中的所有测试用例都成功执行时返回`0`。为了构建测试运行器，我们需要运行以下命令。注意`-g`选项，它将调试符号添加到最终的测试运行器可执行文件中。进行*调试构建*是构建测试的最常见方式，因为如果测试用例失败，我们立即需要精确的*堆栈跟踪*和进一步的调试信息以继续调查。更重要的是，`assert`语句通常从*发布构建*中删除，但我们需要在测试运行器可执行文件中保留它们：

```cpp
$ gcc -g -c ExtremeC_examples_chapter22_1.c -o impl.o
$ gcc -g -c ExtremeC_examples_chapter22_1__next_even_number__tests.c -o tests1.o
$ gcc -g -c ExtremeC_examples_chapter22_1__calc_factorial__tests.c -o tests2.o
$ gcc -g -c ExtremeC_examples_chapter22_1_tests.c -o main.o
$ gcc impl.o tests1.o tests2.o main.o -o ex22_1_tests.out
$ ./ex22_1_tests.out
All tests are run successfully.
$ echo $?
0
$
```

Shell Box 22-2：构建和运行示例 22.1 的测试运行器

前面的 shell 框显示所有测试都已通过。您也可以通过使用`echo $?`命令来检查测试运行进程的退出代码，并看到它已返回零。

现在，通过在其中一个函数中应用简单的更改，我们可以使测试失败。让我们看看当我们按照以下方式更改`calc_factorial`时会发生什么：

```cpp
int64_t calc_factorial(int64_feed_t feed) {
  int64_t fact = 1;
  int64_t number = feed();
  for (int64_t i = 1; i <= (number + 1); i++) {
    fact *= i;
  }
  return fact;
}
```

代码框 22-7：将`calc_factorial`函数更改为使测试失败

通过前面的更改，以粗体显示，关于`0`和负输入的测试用例仍然通过，但最后一个测试用例，即关于计算`5`的阶乘的测试用例失败了。我们将再次构建测试运行器，以下是在 macOS 机器上执行的输出：

```cpp
$ gcc -g -c ExtremeC_examples_chapter22_1.c -o impl.o
$ gcc -g -c ExtremeC_examples_chapter22_1_tests.c -o main.o
$ ./ex22_1_tests.out
Assertion failed: (fact == 120), function TESTCASE_calc_factorial__fact_of_5_is_120, 
file .../22.1/ExtremeC_examples_chapter22_1__calc_factorial__tests.c, line 29.
Abort trap: 6
$ echo $?
134
$
```

Shell Box 22-3：更改`calc_factorial`函数后构建和运行测试运行器

如您所见，输出中出现了`Assertion failed`，退出代码为`134`。这个退出代码通常由定期运行测试的系统使用和报告，例如*Jenkins*，以检查测试是否成功运行。

作为一项经验法则，每当您有一个需要独立测试的单元时，您需要找到一种方法来提供其依赖项作为某种输入。因此，单元本身应该以使其*可测试*的方式编写。并非所有代码都是可测试的，可测试性不仅限于单元测试，这一点非常重要，需要意识到。此链接提供了有关如何编写可测试代码的良好信息：[`blog.gurock.com/highly-testable-code/`](https://blog.gurock.com/highly-testable-code/)。

为了澄清上述讨论，假设我们已经像下面这样编写了`calc_factorial`函数，直接使用`next_even_number`函数而不是使用函数指针。请注意，在以下代码框中，函数不接收函数指针参数，并且它直接调用`next_even_number`函数：

```cpp
int64_t calc_factorial() {
  int64_t fact = 1;
  int64_t number = next_even_number();
  for (int64_t i = 1; i <= number; i++) {
    fact *= i;
  }
  return fact;
}
```

代码框 22-8：将`calc_factorial`函数的签名更改为不接受函数指针

上述代码的可测试性较低。没有方法在不调用`next_even_number`的情况下测试`calc_factorial`——也就是说，没有使用一些技巧来更改最终可执行文件中符号`next_even_number`背后的定义，就像我们在*示例 22.2*中所做的那样。

事实上，`calc_factorial`的两个版本都做了同样的事情，但*代码框 22-2*中的定义更易于测试，因为我们可以在隔离的情况下对其进行测试。编写可测试的代码并不容易，你应该始终仔细思考，以便实现代码并使其可测试。

编写可测试的代码通常需要更多的工作。关于编写可测试代码的额外开销百分比存在各种观点，但可以肯定的是，编写测试确实会在时间和精力上带来一些额外的成本。但这种额外的成本确实带来了巨大的好处。如果没有为单元编写测试，随着时间的推移和单元中引入的更多更改，你将失去对它的跟踪。

## 测试双胞胎

在前面的例子中，在编写测试用例时，我们引入了存根函数。还有一些其他术语是关于试图模仿单元依赖的对象。这些对象被称为*测试双胞胎*。接下来，我们将介绍另外两种测试双胞胎：*模拟*和*伪造*函数。首先，让我们再次简要解释一下存根函数是什么。

在这个简短的部分中，请注意两点。首先，关于这些测试双胞胎的定义永远存在争论，我们试图给出一个符合本章使用的适当定义。其次，我们只将讨论与 C 语言相关的内容，因此没有对象，我们有的都是函数。

当一个单元依赖于另一个函数时，它只是依赖于该函数的签名，因此该函数可以被一个新的函数所替代。这个新函数，基于它可能具有的一些属性，可以被称作存根、模拟或伪造函数。这些函数只是编写来满足测试要求，它们不能在生产环境中使用。

我们将存根（stub）解释为一个非常简单的函数，通常只是返回一个常量值。正如你在*示例 22.1*中看到的，它间接地返回了由正在运行的测试用例设置的值。在以下链接中，你可以了解更多关于我们正在讨论的测试双胞胎以及一些其他的：[`en.wikipedia.org/wiki/Test_double`](https://en.wikipedia.org/wiki/Test_double)。如果你打开链接，存根被定义为向测试代码提供*间接输入*的东西。如果你接受这个定义，*代码框 22-5*中看到的`feed_stub`函数就是一个存根函数。

模拟函数，或者更普遍地说，作为面向对象语言的一部分的模拟对象，可以通过指定某个输入的输出来进行操作。这样，在运行测试逻辑之前，你可以设置模拟函数对于某个输入应该返回的内容，在逻辑运行期间，它将按照你事先设置的方式行动。一般来说，模拟对象也可以有期望，并且它们将相应地执行所需的断言。正如前一个链接中所述，对于模拟对象，我们在运行测试之前设置期望。我们将在组件测试部分给出一个 C 语言的模拟函数示例。

最后，可以使用一个假函数来为运行测试中的真实且可能复杂的函数提供非常简化的功能。例如，而不是使用真实的文件系统，可以使用一些简化的内存存储。在组件测试中，例如，具有复杂功能的其他组件可以在测试中用假实现替换。

在结束本节之前，我想谈谈 *代码覆盖率*。在理论上，所有单位都应该有相应的测试套件，并且每个测试套件都应该包含通过所有可能代码分支的所有测试用例。正如我们所说的，这是在理论上，但在实践中，您通常只为一部分单位有测试单元。通常，您没有覆盖所有可能代码分支的测试用例。

拥有适当测试用例的单位比例称为代码覆盖率或 *测试覆盖率*。比例越高，您就越有可能被通知关于不希望修改的情况。这些不希望修改通常不是由糟糕的开发者引入的。事实上，这些破坏性更改通常是在某人正在修复代码中的错误或实现新功能时引入的。

在讨论了测试替身之后，我们将在下一节中讨论组件测试。

# 组件测试

正如我们在上一节中解释的，单位可以被定义为单个函数、一组函数或整个组件。因此，组件测试是单元测试的一种特殊类型。在本节中，我们想要定义一个假设的组件作为 *示例 22.1* 的一部分，并将示例中找到的两个函数放入该组件中。请注意，组件通常会产生一个可执行文件或库。我们可以假设我们的假设组件将产生一个包含两个函数的库。

正如我们之前所说的，我们必须能够测试组件的功能。在本节中，我们仍然想要编写测试用例，但本节中编写的测试用例与上一节的不同之处在于应该隔离的单位。在上一节中，我们有应该隔离的函数，但在本节中，我们有一个由两个协同工作的函数组成的组件，需要隔离。因此，当它们一起工作时，必须对这些函数进行测试。

接下来，您可以找到我们为作为 *示例 22.1* 部分定义的组件编写的测试用例：

```cpp
#include <assert.h>
#include "ExtremeC_examples_chapter22_1.h"
void TESTCASE_component_test__factorials_from_0_to_8() {
  assert(calc_factorial(next_even_number) == 1);
  assert(calc_factorial(next_even_number) == 2);
  assert(calc_factorial(next_even_number) == 24);
  assert(calc_factorial(next_even_number) == 720);
  assert(calc_factorial(next_even_number) == 40320);
}
void TESTCASE_component_test__factorials_should_rotate() {
  int64_t number = calc_factorial(next_even_number);
  for (size_t i = 1; i <= 4; i++) {
    calc_factorial(next_even_number);
  }
  int64_t number2 = calc_factorial(next_even_number);
  assert(number == number2);
}
int main(int argc, char** argv) {
  TESTCASE_component_test__factorials_from_0_to_8();
  TESTCASE_component_test__factorials_should_rotate();
  return 0;
}
```

代码框 22-9 [ExtremeC_examples_chapter22_1_component_tests.c]: 为我们假设的组件作为示例 22.1 的一部分编写的某些组件测试

正如您所见，我们已经编写了两个测试用例。正如我们之前所说的，在我们的假设组件中，函数 `calc_factorial` 和 `next_even_number` 必须协同工作，如您所见，我们已经将 `next_even_number` 作为 `calc_factorial` 的输入。前面的测试用例和其他类似的测试用例应该保证组件正常工作。

准备编写测试用例的基础需要付出很多努力。因此，使用测试库来完成此目的非常常见。这些库为测试用例准备舞台；它们初始化每个测试用例，运行测试用例，并最终拆除测试用例。在下一节中，我们将讨论 C 可用的一些测试库。

# C 的测试库

在本节中，我们将演示两个用于为 C 程序编写测试的知名库。对于 C 的单元测试，我们使用用 C 或 C++编写的库，因为我们可以轻松地将它们集成并直接从 C 或 C++测试环境中使用单元。在本节中，我们的重点是 C 的单元测试和组件测试。

对于集成测试，我们可以自由选择其他编程语言。通常，集成和系统测试要复杂得多，因此我们需要使用一些测试自动化框架来更容易地编写测试并轻松运行它们。使用**领域特定语言**（**DSL**）是这一自动化过程的一部分，以便更容易地编写测试场景并使测试执行更加简单。许多语言都可以用于此目的，但像 Unix shell、Python、JavaScript 和 Ruby 这样的脚本语言是最受欢迎的。一些其他编程语言，如 Java，也在测试自动化中得到了广泛使用。

以下是一些用于为 C 程序编写单元测试的知名单元测试框架列表。以下列表可以在以下链接中找到：http://check.sourceforge.net/doc/check_html/check_2.html#SEC3：

+   Check（来自前一个链接的作者）

+   AceUnit

+   GNU Autounit

+   cUnit

+   CUnit

+   CppUnit

+   CuTest

+   embUnit

+   MinUnit

+   Google Test

+   CMocka

在以下几节中，我们将介绍两个流行的测试框架：用 C 编写的*CMocka*和用 C++编写的*Google Test*。我们不会探索这些框架的所有功能，但这是为了给你一个单元测试框架的初步感觉。在这个领域进一步学习是非常鼓励的。

在下一节中，我们将使用 CMocka 为*example 22.1*编写单元测试。

## CMocka

CMocka 的第一个优点是它完全用 C 编写，并且只依赖于 C 标准库——不依赖于任何其他库。因此，你可以使用 C 编译器编译测试，这让你有信心测试环境非常接近实际的生产环境。CMocka 可在许多平台如 macOS、Linux 甚至 Microsoft Windows 上使用。

CMocka 是 C 语言单元测试的**事实上的**框架。它支持**测试夹具**。测试夹具可能允许你在每个测试用例之前和之后初始化和清理测试环境。CMocka 还支持**函数模拟**，这在尝试模拟任何 C 函数时非常有用。作为提醒，模拟函数可以被配置为在提供特定输入时返回特定值。我们将给出模拟 *example 22.2* 中使用的 `rand` 标准函数的示例。

以下代码框包含与 *example 22.1* 中看到的相同的测试用例，但这次是用 CMocka 编写的。我们将所有测试用例都放在了一个文件中，该文件有自己的 `main` 函数：

```cpp
// Required by CMocka
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "ExtremeC_examples_chapter22_1.h"
int64_t input_value = -1;
int64_t feed_stub() {
  return input_value;
}
void calc_factorial__fact_of_zero_is_one(void** state) {
  input_value = 0;
  int64_t fact = calc_factorial(feed_stub);
  assert_int_equal(fact, 1);
}
void calc_factorial__fact_of_negative_is_one(void** state) {
  input_value = -10;
  int64_t fact = calc_factorial(feed_stub);
  assert_int_equal(fact, 1);
}
void calc_factorial__fact_of_5_is_120(void** state) {
  input_value = 5;
  int64_t fact = calc_factorial(feed_stub);
  assert_int_equal(fact, 120);
}
void next_even_number__even_numbers_should_be_returned(void** state) {
  assert_int_equal(next_even_number(), 0);
  assert_int_equal(next_even_number(), 2);
  assert_int_equal(next_even_number(), 4);
  assert_int_equal(next_even_number(), 6);
  assert_int_equal(next_even_number(), 8);
}
void next_even_number__numbers_should_rotate(void** state) {
  int64_t number = next_even_number();
  for (size_t i = 1; i <= 4; i++) {
    next_even_number();
  }
  int64_t number2 = next_even_number();
  assert_int_equal(number, number2);
}
int setup(void** state) {
  return 0;
}
int tear_down(void** state) {
  return 0;
}
int main(int argc, char** argv) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(calc_factorial__fact_of_zero_is_one),
    cmocka_unit_test(calc_factorial__fact_of_negative_is_one),
    cmocka_unit_test(calc_factorial__fact_of_5_is_120),
    cmocka_unit_test(next_even_number__even_numbers_should_be_returned),
    cmocka_unit_test(next_even_number__numbers_should_rotate),
  };
  return cmocka_run_group_tests(tests, setup, tear_down);
}
```

代码框 22-10 [ExtremeC_examples_chapter22_1_cmocka_tests.c]：示例 22.1 的 CMocka 测试用例

在 CMocka 中，每个测试用例都应该返回 `void` 并接收一个 `void**` 参数。指针参数将被用来接收一段信息，称为 `state`，它对于每个测试用例是特定的。在 `main` 函数中，我们创建一个测试用例列表，然后最终调用 `cmocka_run_group_tests` 函数来运行所有单元测试。

除了测试用例函数外，你还会看到两个新的函数：`setup` 和 `tear_down`。正如我们之前所说的，这些函数被称为测试夹具。测试夹具在每次测试用例之前和之后被调用，其责任是设置和清理测试用例。夹具 `setup` 在每个测试用例之前被调用，而夹具 `tear_down` 在每个测试用例之后被调用。请注意，名称是可选的，它们可以命名为任何名称，但我们使用 `setup` 和 `tear_down` 以便清晰。

我们之前编写的测试用例和用 CMocka 编写的测试用例之间的重要区别在于使用了不同的断言函数。这是使用单元测试框架的优点之一。测试库中包含了一系列断言函数，可以提供更多关于它们失败的信息，而不是标准的 `assert` 函数，后者会立即终止程序且不提供太多信息。正如你所看到的，我们已经在前面的代码中使用了 `assert_int_equal`，它检查两个整数的相等性。

为了编译前面的程序，你首先需要安装 CMocka。在基于 Debian 的 Linux 系统上，只需运行 `sudo apt-get install libcmocka-dev` 即可，而在 macOS 系统上，只需使用命令 `brew install cmocka` 进行安装。网上将会有很多帮助信息，可以帮助你完成安装过程。

在安装了 CMocka 之后，你可以使用以下命令来构建前面的代码：

```cpp
$ gcc -g -c ExtremeC_examples_chapter22_1.c -o impl.o
$ gcc -g -c ExtremeC_examples_chapter22_1_cmocka_tests.c -o cmocka_tests.o
$ gcc impl.o cmocka_tests.o -lcmocka -o ex22_1_cmocka_tests.out
$ ./ex22_1_cmocka_tests.out
[==========] Running 5 test(s).
[ RUN      ] calc_factorial__fact_of_zero_is_one
[       OK ] calc_factorial__fact_of_zero_is_one
[ RUN      ] calc_factorial__fact_of_negative_is_one
[       OK ] calc_factorial__fact_of_negative_is_one
[ RUN      ] calc_factorial__fact_of_5_is_120
[       OK ] calc_factorial__fact_of_5_is_120
[ RUN      ] next_even_number__even_numbers_should_be_returned
[       OK ] next_even_number__even_numbers_should_be_returned
[ RUN      ] next_even_number__numbers_should_rotate
[       OK ] next_even_number__numbers_should_rotate
[==========] 5 test(s) run.
[  PASSED  ] 5 test(s).
$
```

Shell 框 22-4：构建和运行为示例 22.1 编写的 CMocka 单元测试

如您所见，我们必须使用 `-lcmocka` 来将前面的程序与已安装的 CMocka 库链接。输出显示了测试用例名称和通过测试的数量。接下来，我们更改一个测试用例使其失败。我们只是修改了 `next_even_number__even_numbers_should_be_returned` 测试用例中的第一个断言：

```cpp
void next_even_number__even_numbers_should_be_returned(void** state) {
  assert_int_equal(next_even_number(), 1);
  ...
}
```

代码框 22-11: 修改示例 22.1 中的一个 CMocka 测试用例

现在，构建测试并再次运行它们：

```cpp
$ gcc -g -c ExtremeC_examples_chapter22_1_cmocka_tests.c -o cmocka_tests.o
$ gcc impl.o cmocka_tests.o -lcmocka -o ex22_1_cmocka_tests.out
$ ./ex22_1_cmocka_tests.out
[==========] Running 5 test(s).
[ RUN      ] calc_factorial__fact_of_zero_is_one
[       OK ] calc_factorial__fact_of_zero_is_one
[ RUN      ] calc_factorial__fact_of_negative_is_one
[       OK ] calc_factorial__fact_of_negative_is_one
[ RUN      ] calc_factorial__fact_of_5_is_120
[       OK ] calc_factorial__fact_of_5_is_120
[ RUN      ] next_even_number__even_numbers_should_be_returned
[  ERROR   ] --- 0 != 0x1
[   LINE   ] --- .../ExtremeC_examples_chapter22_1_cmocka_tests.c:37: error: Failure!
[  FAILED  ] next_even_number__even_numbers_should_be_returned
[ RUN      ] next_even_number__numbers_should_rotate
[       OK ] next_even_number__numbers_should_rotate
[==========] 5 test(s) run.
[  PASSED  ] 4 test(s).
[  FAILED  ] 1 test(s), listed below:
[  FAILED  ] next_even_number__even_numbers_should_be_returned
 1 FAILED TEST(S)
 $
```

Shell 框 22-5：修改其中一个测试用例后构建和运行 CMocka 单元测试

在前面的输出中，您可以看到有一个测试用例失败了，原因在日志中间显示为一个错误。它显示了一个整数相等断言失败。正如我们之前解释的，使用 `assert_int_equal` 而不是使用普通的 `assert` 调用允许 CMocka 在执行日志中打印出有用的消息，而不是仅仅终止程序。

我们接下来的示例是关于使用 CMocka 的函数模拟功能。CMocka 允许您模拟一个函数，这样，您可以在提供特定输入时使函数返回特定的结果。

在下一个示例，即 *示例 22.2* 中，我们想展示如何使用模拟功能。在这个示例中，标准函数 `rand` 用于生成随机数。还有一个名为 `random_boolean` 的函数，它根据 `rand` 函数返回的数字的奇偶性返回一个布尔值。在展示 CMocka 的模拟功能之前，我们想展示如何为 `rand` 函数创建存根。您可以看到这个示例与 *示例 22.1* 不同。接下来，您可以看到 `random_boolean` 函数的声明：

```cpp
#ifndef _EXTREME_C_EXAMPLE_22_2_
#define _EXTREME_C_EXAMPLE_22_2_
#define TRUE 1
#define FALSE 0
typedef int bool_t;
bool_t random_boolean();
#endif
```

代码框 22-12 [ExtremeC_examples_chapter22_2.h]: 示例 22.2 的头文件

以下代码框包含定义：

```cpp
#include <stdlib.h>
#include <stdio.h>
#include "ExtremeC_examples_chapter22_2.h"
bool_t random_boolean() {
  int number = rand();
  return (number % 2);
}
```

代码框 22-13 [ExtremeC_examples_chapter22_2.c]: 示例 22.2 中 random_boolean 函数的定义

首先，我们不能让 `random_boolean` 在测试中使用实际的 `rand` 定义，因为，正如其名称所暗示的，它生成随机数，我们测试中不能有随机元素。测试是关于检查预期的，而预期和提供的输入必须是可预测的。更重要的是，`rand` 函数的定义是 C 标准库的一部分，例如 Linux 中的 *glibc*，使用存根函数对它进行操作不会像我们在 *示例 22.1* 中做的那样简单。

在上一个示例中，我们可以非常容易地将函数指针发送到存根定义。但在这个示例中，我们直接使用 `rand` 函数。我们不能更改 `random_boolean` 的定义，我们必须想出另一个技巧来使用存根函数 `rand`。

为了使用 `rand` 函数的不同定义，在 C 中最简单的方法是玩弄最终目标文件的 *symbols*。在结果目标文件的 *symbol table* 中，有一个指向 `rand` 的条目，它引用了其在 C 标准库中的实际定义。如果我们更改此条目以引用测试二进制文件中 `rand` 函数的不同定义，我们就可以轻松地用我们的存根定义替换 `rand` 的定义。

在以下代码框中，您可以看到我们如何定义存根函数和测试。这会非常类似于我们在 *example 22.1* 中所做的那样：

```cpp
#include <stdlib.h>
// Required by CMocka
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "ExtremeC_examples_chapter22_2.h"
int next_random_num = 0;
int __wrap_rand() {
  return next_random_num;
}
void test_even_random_number(void** state) {
  next_random_num = 10;
  assert_false(random_boolean());
}
void test_odd_random_number(void** state) {
  next_random_num = 13;
  assert_true(random_boolean());
}
int main(int argc, char** argv) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_even_random_number),
    cmocka_unit_test(test_odd_random_number)
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
```

Code Box 22-14 [ExtremeC_examples_chapter22_2_cmocka_tests_with_stub.c]：使用存根函数编写 CMocka 测试用例

如您所见，前面的代码主要遵循我们在 *Code Box 22-10* 中的 *example 22.1* 编写的 CMocka 测试中看到的相同模式。让我们构建前面的文件并运行测试。我们期望所有测试都失败，因为无论您如何定义存根函数，`random_boolean` 都会从 C 标准库中选取 `rand`：

```cpp
$ gcc -g -c ExtremeC_examples_chapter22_2.c -o impl.o
$ gcc -g -c ExtremeC_examples_chapter22_2_cmocka_tests_with_stub.c -o tests.o
$ gcc impl.o tests.o -lcmocka -o ex22_2_cmocka_tests_with_stub.out
$ ./ex22_2_cmocka_tests_with_stub.out
[==========] Running 2 test(s).
[ RUN      ] test_even_random_number
[  ERROR   ] --- random_boolean()
[   LINE   ] --- ExtremeC_examples_chapter22_2_cmocka_tests_with_stub.c:23: error: Failure!
[  FAILED  ] test_even_random_number
[ RUN      ] test_odd_random_number
[  ERROR   ] --- random_boolean()
[   LINE   ] --- ExtremeC_examples_chapter22_2_cmocka_tests_with_stub.c:28: error: Failure!
[  FAILED  ] test_odd_random_number
[==========] 2 test(s) run.
[  PASSED  ] 0 test(s).
[  FAILED  ] 2 test(s), listed below:
[  FAILED  ] test_even_random_number
[  FAILED  ] test_odd_random_number
 2 FAILED TEST(S)
$
```

Shell Box 22-6：构建和运行示例 22.2 的 CMocka 单元测试

现在是时候施展技巧，更改 `rand` 符号背后的定义，该定义作为 `ex22_2_cmocka_tests_with_stub.out` 可执行文件的一部分。请注意，以下命令仅适用于 Linux 系统。我们这样做：

```cpp
$ gcc impl.o tests.o -lcmocka -Wl,--wrap=rand -o ex22_2_cmocka_tests_with_stub.out
$ ./ex22_2_cmocka_tests_with_stub.out
[==========] Running 2 test(s).
[ RUN      ] test_even_random_number
[       OK ] test_even_random_number
[ RUN      ] test_odd_random_number
[       OK ] test_odd_random_number
[==========] 2 test(s) run.
[  PASSED  ] 2 test(s).
$
```

Shell Box 22-7：在包装 rand 符号后构建和运行示例 22.2 的 CMocka 单元测试

如您在输出中看到的，标准的 `rand` 函数不再被调用，取而代之的是存根函数返回我们告诉它返回的内容。使函数 `__wrap_rand` 被调用而不是标准 `rand` 函数的技巧主要在于在 `gcc` 链接命令中使用选项 `-Wl`,`--wrap=rand`。

注意，此选项仅适用于 Linux 中的 `ld` 程序，您必须使用其他技巧，如 *inter-positioning*，在 macOS 或使用非 GNU 链接器的其他系统中调用不同的函数。

选项 `--wrap=rand` 告诉链接器更新最终可执行文件符号表中 `rand` 符号的条目，这将引用 `__wrap_rand` 函数的定义。请注意，这不是一个自定义名称，您必须将存根函数命名为这样。函数 `__wrap_rand` 被称为 *wrapper function*。更新符号表后，对 `rand` 函数的任何调用都会导致调用 `__wrap_func` 函数。这可以通过查看最终测试二进制的符号表来验证。

除了在符号表中更新 `rand` 符号外，链接器还创建另一个条目。新条目具有符号 `__real_rand`，它指向标准 `rand` 函数的实际定义。因此，如果我们需要运行标准的 `rand`，我们仍然可以使用函数名 `__real_rand`。这是符号表及其符号的出色用法，以便调用包装函数，尽管有些人不喜欢这样做，他们更喜欢预加载一个包装实际 `rand` 函数的共享对象。无论你使用哪种方法，你最终都需要将调用重定向到 `rand` 符号的另一个存根函数。

上述机制将是演示 CMocka 中函数模拟如何工作的基础。与 *代码框 22-14* 中看到的全局变量 `next_random_num` 不同，我们可以使用一个模拟函数来返回指定的值。接下来，你可以看到相同的 CMocka 测试，但使用模拟函数来读取测试输入：

```cpp
#include <stdlib.h>
// Required by CMocka
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "ExtremeC_examples_chapter22_2.h"
int __wrap_rand() {
  return mock_type(int);
}
void test_even_random_number(void** state) {
  will_return(__wrap_rand, 10);
  assert_false(random_boolean());
}
void test_odd_random_number(void** state) {
  will_return(__wrap_rand, 13);
  assert_true(random_boolean());
}
int main(int argc, char** argv) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_even_random_number),
    cmocka_unit_test(test_odd_random_number)
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
```

代码框 22-15 [ExtremeC_examples_chapter22_2_cmocka_tests_with_mock.c]：使用模拟函数编写 CMocka 测试用例

现在我们知道了包装函数 `__wrap_rand` 的调用方式，我们可以解释模拟部分。模拟功能由 `will_return` 和 `mock_type` 函数对提供。首先，应该调用 `will_return`，指定模拟函数应返回的值。然后，当模拟函数（在这种情况下为 `__wrap_rand`）被调用时，`mock_type` 函数返回指定的值。

例如，我们通过使用 `will_return(__wrap_rand, 10)` 将 `__wrap_rand` 定义为返回 `10`，然后在 `__wrap_rand` 内部调用 `mock_type` 函数时返回值 `10`。请注意，每个 `will_return` 都必须与一个 `mock_type` 调用配对；否则，测试将失败。因此，如果由于任何原因没有调用 `__wrap_rand`，则测试将失败。

作为本节的最后一条注释，前面代码的输出将与我们在 Shell Boxes *22-6* 和 *22-7* 中看到的一样。此外，当然对于源文件 `ExtremeC_examples_chapter22_2_cmocka_tests_with_mock.c`，必须使用相同的命令来构建代码并运行测试。

在本节中，我们展示了如何使用 CMocka 库编写测试用例、执行断言和编写模拟函数。在下一节中，我们将讨论 Google Test，这是另一个可以用于单元测试 C 程序的测试框架。

## Google Test

Google Test 是一个 C++ 测试框架，可用于单元测试 C 和 C++ 程序。尽管它是用 C++ 开发的，但它可以用于测试 C 代码。有些人认为这是一种不好的做法，因为测试环境不是使用你将用于设置生产环境的相同编译器和链接器来设置的。

在能够使用 Google Test 为 *示例 22.1* 编写测试用例之前，我们需要稍微修改 *示例 22.1* 中的头文件。以下是新头文件：

```cpp
#ifndef _EXTREME_C_EXAMPLE_22_1_
#define _EXTREME_C_EXAMPLE_22_1_
#include <stdint.h>
#include <unistd.h>
#if __cplusplus
extern "C" {
#endif
typedef int64_t (*int64_feed_t)();
int64_t next_even_number();
int64_t calc_factorial(int64_feed_t feed);
#if __cplusplus
}
#endif
#endif
```

Code Box 22-16 [ExtremeC_examples_chapter22_1.h]: 作为示例 22.1 一部分修改的头文件

正如你所见，我们将声明放在了`extern C { ... }`块中。我们只在定义了宏`_cplusplus`时这样做。前面的更改简单地说，就是当编译器是 C++时，我们希望在生成的目标文件中拥有未混淆的符号，否则当链接器尝试查找*混淆符号*的定义时，我们将得到链接错误。如果你不了解 C++的*名称混淆*，请参阅*第二章*的最后部分，*编译和链接*。

现在，让我们继续使用 Google Test 编写测试用例：

```cpp
// Required by Google Test
#include <gtest/gtest.h>
#include "ExtremeC_examples_chapter22_1.h"
int64_t input_value = -1;
int64_t feed_stub() {
  return input_value;
}
TEST(calc_factorial, fact_of_zero_is_one) {
  input_value = 0;
  int64_t fact = calc_factorial(feed_stub);
  ASSERT_EQ(fact, 1);
}
TEST(calc_factorial, fact_of_negative_is_one) {
  input_value = -10;
  int64_t fact = calc_factorial(feed_stub);
  ASSERT_EQ(fact, 1);
}
TEST(calc_factorial, fact_of_5_is_120) {
  input_value = 5;
  int64_t fact = calc_factorial(feed_stub);
  ASSERT_EQ(fact, 120);
}
TEST(next_even_number, even_numbers_should_be_returned) {
  ASSERT_EQ(next_even_number(), 0);
  ASSERT_EQ(next_even_number(), 2);
  ASSERT_EQ(next_even_number(), 4);
  ASSERT_EQ(next_even_number(), 6);
  ASSERT_EQ(next_even_number(), 8);
}
TEST(next_even_number, numbers_should_rotate) {
  int64_t number = next_even_number();
  for (size_t i = 1; i <= 4; i++) {
    next_even_number();
  }
  int64_t number2 = next_even_number();
   ASSERT_EQ(number, number2);
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

Code Box 22-17 [ExtremeC_examples_chapter22_1_gtests.cpp]: 使用 Google Test 为示例 22.1 编写的测试用例

测试用例使用`TEST(...)`宏定义。这是一个如何有效地使用宏来形成领域特定语言的例子。还有其他宏，如`TEST_F(...)`和`TEST_P(...)`, 这些是 C++特定的。传递给宏的第一个参数是测试类的名称（Google Test 是为面向对象的 C++编写的），可以将其视为包含多个测试用例的测试套件。第二个参数是测试用例的名称。

注意`ASSERT_EQ`宏，它用于断言对象的相等性，而不仅仅是整数。Google Test 中有大量的期望检查宏，使其成为一个完整的单元测试框架。最后一部分是`main`函数，它运行所有定义的测试。请注意，上述代码应该使用符合 C++11 规范的编译器（如`g++`和`clang++`）进行编译。

以下命令构建前面的代码。注意使用`g++`编译器和传递给它的选项`-std=c++11`，这表示应使用 C++11：

```cpp
$ gcc -g -c ExtremeC_examples_chapter22_1.c -o impl.o
$ g++ -std=c++11 -g -c ExtremeC_examples_chapter22_1_gtests.cpp -o gtests.o
$ g++ impl.o gtests.o -lgtest -lpthread -o ex19_1_gtests.out
$ ./ex19_1_gtests.out
[==========] Running 5 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 3 tests from calc_factorial
[ RUN      ] calc_factorial.fact_of_zero_is_one
[       OK ] calc_factorial.fact_of_zero_is_one (0 ms)
[ RUN      ] calc_factorial.fact_of_negative_is_one
[       OK ] calc_factorial.fact_of_negative_is_one (0 ms)
[ RUN      ] calc_factorial.fact_of_5_is_120
[       OK ] calc_factorial.fact_of_5_is_120 (0 ms)
[----------] 3 tests from calc_factorial (0 ms total)
[----------] 2 tests from next_even_number
[ RUN      ] next_even_number.even_numbers_should_be_returned
[       OK ] next_even_number.even_numbers_should_be_returned (0 ms)
[ RUN      ] next_even_number.numbers_should_rotate
[       OK ] next_even_number.numbers_should_rotate (0 ms)
[----------] 2 tests from next_even_number (0 ms total)
[----------] Global test environment tear-down
[==========] 5 tests from 2 test suites ran. (1 ms total)
[  PASSED  ] 5 tests.
$
```

Shell Box 22-8：构建和运行示例 22.1 的 Google Test 单元测试

上述输出显示与 CMocka 输出类似。它表明有五个测试用例已经通过。让我们改变与 CMocka 相同的测试用例来破坏测试套件：

```cpp
TEST(next_even_number, even_numbers_should_be_returned) {
  ASSERT_EQ(next_even_number(), 1);
  ...
}
```

Code Box 22-18：修改 Google Test 编写的测试用例之一

让我们再次构建测试并运行它们：

```cpp
$ g++ -std=c++11 -g -c ExtremeC_examples_chapter22_1_gtests.cpp -o gtests.o
$ g++ impl.o gtests.o -lgtest -lpthread -o ex22_1_gtests.out
$ ./ex22_1_gtests.out
[==========] Running 5 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 3 tests from calc_factorial
[ RUN      ] calc_factorial.fact_of_zero_is_one
[       OK ] calc_factorial.fact_of_zero_is_one (0 ms)
[ RUN      ] calc_factorial.fact_of_negative_is_one
[       OK ] calc_factorial.fact_of_negative_is_one (0 ms)
[ RUN      ] calc_factorial.fact_of_5_is_120
[       OK ] calc_factorial.fact_of_5_is_120 (0 ms)
[----------] 3 tests from calc_factorial (0 ms total)
[----------] 2 tests from next_even_number
[ RUN      ] next_even_number.even_numbers_should_be_returned
.../ExtremeC_examples_chapter22_1_gtests.cpp:34: Failure
Expected equality of these values:
  next_even_number()
    Which is: 0
  1
[  FAILED  ] next_even_number.even_numbers_should_be_returned (0 ms)
[ RUN      ] next_even_number.numbers_should_rotate
[       OK ] next_even_number.numbers_should_rotate (0 ms)
[----------] 2 tests from next_even_number (0 ms total)
[----------] Global test environment tear-down
[==========] 5 tests from 2 test suites ran. (0 ms total)
[  PASSED  ] 4 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] next_even_number.even_numbers_should_be_returned
 1 FAILED TEST
$
```

Shell Box 22-9：修改一个测试用例后构建和运行示例 22.1 的 Google Test 单元测试

正如你所见，并且与 CMocka 完全一样，Google Test 也会打印出测试失败的位置，并显示一个有用的报告。关于 Google Test 的最后一句话，它支持测试固定值，但不是像 CMocka 那样支持。测试固定值应该在*测试类*中定义。

**注意**：

为了拥有模拟对象和模拟功能，可以使用*Google Mock*（或*gmock*）库，但我们在本书中不涉及它。

在本节中，我们介绍了 C 语言中最著名的两个单元测试库。在章节的下一部分，我们将深入探讨调试这一主题，这对于每一位程序员来说当然是一项必要的技能。

# 调试

有时候一个测试或一组测试会失败。也有时候你会发现一个错误。在这两种情况下，都存在错误，你需要找到根本原因并修复它。这涉及到许多调试会话，通过查看源代码来寻找错误的原因并规划所需的修复。但“调试”一段软件究竟意味着什么呢？

**注意**：

人们普遍认为，“调试”这个术语起源于计算机如此庞大，以至于真正的虫子（如蛾子）可以卡在系统机械中并导致故障的时代。因此，一些人，官方称为*调试器*，被派到硬件室去从设备中移除虫子。更多信息请见此链接：https://en.wikipedia.org/wiki/Debugging。

调试是一项调查任务，通过查看程序内部和/或外部来找到观察到的错误的根本原因。当运行程序时，你通常将其视为一个黑盒。然而，当结果出现问题时或执行被中断时，你需要更深入地查看并了解问题是如何产生的。这意味着你必须将程序视为一个白盒，其中一切都可以被看到。

这基本上是我们可以为程序拥有两种不同构建的原因：*发布*和*构建*。在发布构建中，重点是执行和功能，程序主要被视为一个黑盒，但在调试构建中，我们可以跟踪所有发生的事件，并将程序视为一个白盒。调试构建通常用于开发和测试环境，而发布构建则针对部署和生产环境。

为了拥有调试构建版本，软件项目的所有产品或其中的一部分需要包含*调试*符号，这些符号允许开发者跟踪和查看程序的*堆栈跟踪*和执行流程。通常，发布产品（可执行文件或库）不适合调试目的，因为它不够透明，无法让观察者检查程序的内部结构。在*第四章*，*进程内存结构*和*第五章*，*栈和堆*中，我们讨论了如何为调试目的构建 C 源代码。

为了调试程序，我们主要使用调试器。调试器是独立程序，它们附着到目标进程上以控制或监视它。当我们在处理问题时，调试器是我们调查的主要工具，但其他调试工具也可以用来研究内存、并发执行流程或程序的性能。我们将在接下来的章节中讨论这些工具。

大多数错误都是*可复现的*，但也有一些错误无法复现或在调试会话中观察到；这主要是因为*观察者效应*。它说，当你想查看程序的内幕时，你会改变它的工作方式，这可能会阻止一些错误发生。这类问题非常严重，通常很难修复，因为你不能使用你的调试工具来调查问题的根本原因！

在高性能环境中，一些线程错误可以归入这一类。

在接下来的章节中，我们将讨论不同类别的错误。然后，我们将介绍我们在现代 C/C++开发中使用的工具，以调查错误。

## 错误类别

在软件被客户使用的过程中，可能会有成千上万的错误被报告。但如果你看看这些错误的类型，它们并不多。接下来，你可以看到我们认为重要且需要特殊技能来处理的一些错误类别列表。当然，这个列表并不完整，可能还有我们遗漏的其他类型的错误：

+   **逻辑错误**：为了调查这些错误，你需要了解代码和代码的执行流程。为了看到程序的实际执行流程，应该将调试器附加到正在运行的过程中。只有这样，才能*追踪*和分析执行流程。在调试程序时，*执行日志*也可以使用，尤其是在最终二进制文件中没有调试符号或调试器无法附加到程序的实际运行实例时。

+   **内存错误**：这些错误与内存相关。它们通常是由于悬挂指针、缓冲区溢出、双重释放等原因引起的。这些错误应该使用*内存分析器*进行调查，它作为一种调试工具，用于观察和监控内存。

+   **并发错误**：多进程和多线程程序一直是软件行业中一些最难以解决的错误的发源地。你需要特殊的工具，如*线程检查器*，来检测诸如竞态条件和数据竞争等特别困难的问题。

+   **性能错误**：新的发展可能会导致*性能下降*或性能错误。这些错误应该使用更深入和更专注的测试甚至调试来调查。包含先前执行的历史数据的*执行日志*在寻找导致下降的确切变化或变化时可能很有用。

在接下来的章节中，我们将讨论前面列表中介绍的各种工具。

## 调试器

我们在*第四章*，*进程内存结构*中讨论了调试器，特别是`gdb`，我们用它来查看进程的内存。在本节中，我们将再次审视调试器，并描述它们在日常软件开发中的作用。以下是由大多数现代调试器提供的常见功能列表：

+   调试器是一个程序，就像所有其他程序一样，它作为一个进程运行。调试器进程可以附加到另一个进程，前提是给出目标进程 ID。

+   调试器可以在成功附加到目标进程后控制目标进程中的指令执行；因此，用户可以使用交互式调试会话暂停并继续目标进程的执行流程。

+   调试器可以查看进程的保护内存。它们还可以修改内容，因此开发者可以在故意更改内存内容的同时运行相同的指令组。

+   几乎所有已知的调试器，如果在编译源代码到可重定位目标文件时提供了调试符号，都可以追踪指令到源代码。换句话说，当你暂停在一条指令上时，你可以转到源文件中对应的代码行。

+   如果目标对象文件中没有提供调试符号，调试器可以显示目标指令的汇编代码，这仍然可能是有用的。

+   一些调试器是针对特定语言的，但大多数不是。**Java 虚拟机**（**JVM**）语言，如 Java、Scala 和 Groovy，必须使用 JVM 调试器才能查看和控制 JVM 实例的内部结构。

+   解释型语言如 Python 也有它们自己的调试器，可以用来暂停和控制脚本。虽然像`gdb`这样的低级调试器仍然可用于 JVM 或脚本语言，但它们试图调试 JVM 或解释器进程，而不是执行 Java 字节码或 Python 脚本。

可以在以下链接的维基百科上找到调试器的列表：https://en.wikipedia.org/wiki/List_of_debuggers。从这个列表中，以下调试器引人注目：

1.  **高级调试器**（**adb**）：默认的 Unix 调试器。它根据实际的 Unix 实现有不同的实现。它一直是 Solaris Unix 的默认调试器。

1.  **GNU 调试器**（**gdb**）：Unix 调试器的 GNU 版本，它是许多类 Unix 操作系统的默认调试器，包括 Linux。

1.  **LLDB**：主要设计用于调试由 LLVM 编译器生成的目标文件的调试器。

1.  **Python 调试器**：用于 Python 调试 Python 脚本。

1.  **Java 平台调试架构**（**JPDA**）：这不是一个调试器，但它是一个为在 JVM 实例中运行的程序设计的 API。

1.  **OllyDbg**：用于 Microsoft Windows 调试 GUI 应用的调试器和反汇编器。

1.  **Microsoft Visual Studio 调试器**：Microsoft Visual Studio 使用的调试器。

除了`gdb`，还可以使用`cgdb`。`cgdb`程序在`gdb`交互式 shell 旁边显示一个终端代码编辑器，这使得你更容易在代码行之间移动。

在本节中，我们讨论了调试器作为调查问题的主要工具。在下一节中，我们将讨论内存分析器，这对于调查内存相关错误至关重要。

## 内存检查器

有时候当你遇到与内存相关的错误或崩溃时，仅使用调试器并不能提供太多帮助。你需要另一个工具来检测内存损坏以及对内存单元的无效读写。你需要的是*内存检查器*或*内存分析器*。它可能是调试器的一部分，但通常作为一个独立的程序提供，并且它检测内存异常行为的方式与调试器不同。

我们通常可以期待内存检查器具有以下功能：

+   报告分配的内存总量、释放的内存、使用的静态内存、堆分配、栈分配等。

+   内存泄漏检测，这可以被认为是内存检查器提供的最重要的功能。

+   检测无效的内存读写操作，如缓冲区和数组越界访问、写入已释放的内存区域等。

+   检测*双重释放*问题。当程序尝试释放已释放的内存区域时会发生这种情况。

到目前为止，我们在一些章节中看到了内存检查器，如*Memcheck*（Valgrind 的工具之一），尤其是在*第五章*，*栈和堆*。我们在第五章也讨论了不同类型的内存检查器和内存分析器。在这里，我们再次解释它们，并给出每个的更多细节。

内存检查器都做同样的事情，但它们用于监控内存操作的技术可能不同。因此，我们根据它们使用的技术将它们分组：

1.  **编译时覆盖**：对于使用这种技术的内存检查器，你需要对你的源代码进行一些通常很小的修改，比如包含内存检查器库的头文件。然后，你需要重新编译你的二进制文件。有时，有必要将二进制文件链接到内存检查器提供的库。优点是执行二进制文件的性能下降小于其他技术，但缺点是需要重新编译你的二进制文件。**LLVM AddressSanitizer**（**ASan**）、Memwatch、Dmalloc 和 Mtrace 都是使用这种技术的内存分析器。

1.  **链接时覆盖**：这个内存检查器组类似于之前的内存检查器组，但不同之处在于你不需要更改源代码。相反，你只需要将生成的二进制文件与内存检查器提供的库链接起来，而无需更改源代码。*gperftools*中的*heap checker*实用程序可以用作链接时内存检查器。

1.  **运行时拦截**：使用这种技术的内存检查器位于程序和操作系统之间，试图拦截和跟踪所有与内存相关的操作，并在发现任何不当行为或无效访问时报告。它还可以根据总分配和释放的内存块生成泄漏报告。使用这种技术的最大优点是您无需重新编译或重新链接程序即可使用内存检查器。其重大缺点是它给程序执行引入了显著的开销。此外，内存占用会比在没有内存检查器运行程序时高得多。这绝对不是调试高性能和嵌入式程序的理想环境。Valgrind 中的 Memcheck 工具可以用作运行时拦截内存检查器。这些内存分析器应该与代码库的调试构建一起使用。

1.  **预加载库**：一些内存检查器使用*插入位置*来包装标准内存函数。因此，通过使用`LD_PRELOAD`环境变量预加载内存检查器的共享库，程序可以使用包装函数，内存检查器可以拦截对底层标准内存函数的调用。*堆检查器*实用程序在*gperftools*中可以这样使用。

通常，仅使用特定工具来解决所有内存问题是不够的，因为每个工具都有其自身的优缺点，这使得该工具特定于某个特定环境。

在本节中，我们介绍了可用的内存分析器，并根据它们记录内存分配和释放的技术进行了分类。在下一节中，我们将讨论线程清理器。

## 线程调试器

*线程清理器*或*线程调试器*是用于在程序运行时调试多线程程序以查找并发相关问题的程序。它们可以找到的一些问题如下：

+   数据竞争，以及在不同线程中读写操作导致数据竞争的确切位置

+   错误使用线程 API，尤其是在 POSIX 兼容系统中的 POSIX 线程 API

+   可能的死锁

+   锁定顺序问题

线程调试器和内存检查器都可能检测到假阳性问题。换句话说，它们可能会找到并报告一些问题，但在调查后，它们变得明显不是问题。这实际上取决于这些库用于跟踪事件的技巧以及对该事件的最终决定。

在以下列表中，您可以找到许多知名的可用线程调试器：

+   **Helgrind**（**来自 Valgrind**）：它是 Valgrind 内的另一个工具，主要用于线程调试。DRD 也是 Valgrind 工具包的一部分，另一个线程调试器。功能和差异的列表可以在以下链接中查看：http://valgrind.org/docs/manual/hg-manual.html 和 http://valgrind.org/docs/manual/drd-manual.html。像 Valgrind 的所有其他工具一样，使用 Helgrind 不需要您修改源代码。要运行 Helgrind，您需要运行命令 `valgrind --tool=helgrind [path-to-executable]`。

+   **Intel Inspector**：这是 *Intel Thread Checker* 的继任者，它执行线程错误和内存问题的分析。因此，它既是线程调试器也是内存检查器。与 Valgrind 不同，它不是免费的，使用此工具需要购买适当的许可证。

+   **LLVM ThreadSanitizer**（**TSan**）：这是 LLVM 工具包的一部分，并附带 LLVM AddressSanitizer，这在前面章节中已描述。为了使用调试器和重新编译代码库，需要进行一些轻微的编译时修改。

在本节中，我们讨论了线程调试器，并介绍了一些可用的线程调试器，以便调试线程问题。在下一节中，我们将提供用于调整程序性能的程序和工具包。

## 性能分析器

有时，一组非功能性测试的结果表明性能有所下降。有一些专门的工具用于调查性能下降的原因。在本节中，我们将快速查看可用于分析性能和找到性能瓶颈的工具。

这些性能调试器通常提供以下功能的子集：

+   收集每个单独函数调用的统计数据

+   提供一个用于跟踪函数调用的 *函数调用图*

+   收集每个函数调用的内存相关统计数据

+   收集锁竞争统计数据

+   收集内存分配/释放统计数据

+   缓存分析，提供缓存使用统计数据，并显示不友好的代码部分

+   收集关于线程和同步事件的统计数据

以下是可以用于性能分析的最知名程序和工具包列表：

+   **Google 性能工具**（**gperftools**）：这实际上是一个高性能的 `malloc` 实现，但正如其主页上所述，它提供了一些性能分析工具，如 *heap checker*，这在前面章节中作为内存分析器被介绍。为了使用它，需要将其链接到最终二进制文件。

+   **Callgrind**（**作为 Valgrind 的一部分**）：主要收集关于函数调用以及两个函数之间调用者/被调用者关系的统计数据。无需更改源代码或链接最终二进制文件，它可以在运行时使用，当然，前提是使用调试构建。

+   **Intel VTune**：这是一个来自 Intel 的性能分析套件，包含了前面列表中提到的所有功能。为了使用它，必须购买适当的许可证。

# 摘要

本章是关于单元测试和调试 C 程序。作为总结，在本章中：

+   我们讨论了测试，以及为什么它对我们作为软件工程师和开发团队来说很重要。

+   我们还讨论了不同级别的测试，如单元测试、集成测试和系统测试。

+   功能性和非功能性测试也被涵盖。

+   回归测试被解释了。

+   CMocka 和 Google Test，作为两个著名的 C 语言测试库，被探索，并给出了一些示例。

+   我们讨论了调试以及各种类型的错误。

+   我们讨论了调试器、内存分析器、线程调试器和性能调试器，这些可以帮助我们在处理错误时进行更成功的调查。

下一章将介绍适用于 C 项目的*构建系统*。我们将讨论构建系统是什么以及它能够带来哪些功能，这最终将帮助我们自动化构建大型 C 项目的流程。
