# 第八章：7. 每个人都会跌倒，重要的是你如何重新站起来——测试和调试

## 学习目标

通过本章结束时，您将能够：

+   描述不同类型的断言

+   实施编译时和运行时断言

+   实施异常处理

+   描述并实施单元测试和模拟测试

+   使用断点和监视点调试 C++代码

+   在调试器中检查数据变量和 C++对象

在本章中，您将学习如何适当地添加断言，添加单元测试用例以使代码按照要求运行，并学习调试技术，以便您可以找到代码中的错误并追踪其根本原因。

## 介绍

在**软件开发生命周期**（**SDLC**）中，一旦需求收集阶段完成，通常会进入设计和架构阶段，在这个阶段，项目的高级流程被定义并分解成模块的较小组件。当项目中有许多团队成员时，每个团队成员清楚地被分配了模块的特定部分，并且他们了解自己的要求是必要的。这样，他们可以在隔离的环境中独立编写他们的代码部分，并确保它能正常运行。一旦他们的工作部分完成，他们可以将他们的模块与其他开发人员的模块集成，并确保整个项目按照要求执行。

这个概念也可以应用于小型项目，其中开发人员完全致力于一个需求，将其分解为较小的组件，在隔离的环境中开发组件，确保它按计划执行，集成所有小模块以完成项目，并最终测试以确保整个项目正常运行。

整合整个项目并执行时需要大量的测试。可能会有一个单独的团队（称为`IP 地址`作为`字符串`，然后开发人员需要确保它的格式为`XXX.XXX.XXX.XXX`，其中`X`是`0`-`9`之间的数字。字符串的长度必须是有限的。

在这里，开发人员可以创建一个测试程序来执行他们的代码部分：解析文件，提取`IP 地址`作为字符串，并测试它是否处于正确的格式。同样，如果配置有其他需要解析的参数，并且它们需要以特定格式出现，比如`userid`/`password`，日志文件的位置或挂载点等，那么所有这些都将成为该模块的单元测试的一部分。在本章中，我们将解释诸如`断言`、`安全嵌套`（`异常处理`）、`单元测试`、`模拟`、`断点`、`监视点`和`数据可视化`等技术，以确定错误的来源并限制其增长。在下一节中，我们将探讨断言技术。

### 断言

对于上述情景使用测试条件将有助于项目更好地发展，因为缺陷将在基本层面被捕捉到，而不是在后期的 QA 阶段。可能会出现这样的情况，即使编写了单元测试用例并成功执行了代码，也可能会发现问题，比如应用程序崩溃、程序意外退出或行为不如预期。为了克服这种情况，通常开发人员使用调试模式二进制文件来重新创建问题。`断言`用于确保条件被检查，否则程序的执行将被终止。

这样，问题可以被迅速追踪。此外，在`调试模式`中，开发人员可以逐行遍历程序的实际执行，并检查代码流程是否如预期那样，或者变量是否设置如预期那样并且是否被正确访问。有时，访问指针变量会导致意外行为，如果它们没有指向有效的内存位置。

在编写代码时，我们可以检查是否满足必要条件。如果不满足，程序员可能不希望继续执行代码。这可以很容易地通过断言来实现。断言是一个宏，用于检查特定条件，如果不满足条件，则调用 abort（停止程序执行）并打印错误消息作为标准错误。这通常是**运行时断言**。还可以在编译时进行断言。我们将在后面讨论这一点。在下一节中，我们将解决一个练习，其中我们将编写和测试我们的第一个断言。

### 练习 1：编写和测试我们的第一个断言

在这个练习中，我们将编写一个函数来解析 IP 地址并检查它是否有效。作为我们的要求的一部分，IP 地址将作为字符串文字以`XXX.XXX.XXX.XXX`的格式传递。在这种格式中，`X`代表从`0`到`9`的数字。因此，作为测试的一部分，我们需要确保解析的字符串不为空，并且长度小于 16。按照以下步骤来实现这个练习：

1.  创建一个名为**AssertSample.cpp**的新文件。

1.  打开文件并写入以下代码以包括头文件：

```cpp
#include<iostream>
#include<cassert>
#include<cstring>
using std::cout;
using std::endl;
```

在上述代码中，`#include<cassert>`显示我们需要包括定义 assert 的 cassert 文件。

1.  创建一个名为 checkValidIp（）的函数，它将以 IP 地址作为输入，并在 IP 地址满足我们的要求时返回 true 值。编写以下代码来定义该函数：

```cpp
bool checkValidIp(const char * ip){
    assert(ip != NULL);
    assert(strlen(ip) < 16);
    cout << "strlen: " << strlen(ip) << endl;
    return true;
}
```

在这里，“assert（ip！= NULL）”显示 assert 宏用于检查传递的`ip`变量是否不为`NULL`。如果是`NULL`，那么它将中止并显示错误消息。另外，“assert（strlen（ip）<16）”显示 assert 用于检查`ip`是否为 16 个字符或更少。如果不是，则中止并显示错误消息。

1.  现在，创建一个 main 函数，向我们的 checkValidIp（）函数传递一个不同的字符串文字，并确保可以适当地进行测试。编写以下代码以实现 main 函数：

```cpp
int main(){
    const char * ip;
    ip = NULL;
    bool check = checkValidIp(ip);
    cout << " IP address is validated as :" << (check ? "true" : "false") << endl;
    return 0;
}
```

在上述代码中，我们故意将 NULL 传递给 ip 变量，以确保调用 assert。

1.  打开命令提示符并转到 g++编译器的位置，方法是键入以下命令：

```cpp
g++ AssertSample.cpp
```

使用此命令生成 a.out 二进制文件。

1.  通过在编译器中键入以下命令来运行 a.out 二进制文件：

```cpp
./a.out
```

您将看到以下输出：

![图 7.1：在命令提示符上运行断言二进制文件](img/C14583_07_01.jpg)

###### 图 7.1：在命令提示符上运行断言二进制文件

在上面的屏幕截图中，您可以看到用红色圈出的三段代码。第一个高亮部分显示了.cpp 文件的编译。第二个高亮部分显示了前面编译生成的 a.out 二进制文件。第三个高亮部分显示了对传递的 NULL 值抛出错误的断言。它指示了断言被调用的行号和函数名。

1.  现在，在 main 函数中，我们将传递长度大于 16 的 ip，并检查这里是否也调用了 assert。编写以下代码来实现这一点：

```cpp
ip = "111.111.111.11111";
```

再次打开编译器，编译传递的 ip 长度大于 16。

1.  现在，为了满足 assert 条件，使二进制文件正常运行，我们需要在 main 函数中更新 ip 的值。编写以下代码来实现这一点：

```cpp
ip = "111.111.111.111"; 
```

再次打开编译器，在这里编译 assert，我们没有向 checkValidIP（）函数添加任何额外的功能。但是，在*异常处理*和*单元测试*部分中，我们将使用相同的示例添加更多功能到我们的函数中。

1.  如果我们不希望可执行文件因为生产或发布环境中的断言而中止，就从代码中删除`assert`宏调用。首先，我们将更新`ip`的值，其长度大于`16`。将以下代码添加到文件中：

```cpp
ip = "111.111.111.11111";
```

1.  现在，在编译时，传递`-DNDEBUG`宏。这将确保断言在二进制文件中不被调用。在终端中写入以下命令来编译我们的`.cpp`文件：

```cpp
g++ -DNDEBUG AssertSample.cpp
```

在这之后，当我们执行二进制文件时，会生成以下输出：

![](img/C14583_07_04.jpg)

###### 图 7.4：在命令提示符上运行断言二进制文件

在上述截图中，由于未调用`assert`，它将显示字符串长度为**17**，并且**true**值为 IP 地址将被验证。在这个练习中，我们看到了在执行二进制文件时调用了断言。我们也可以在代码编译时进行断言。这是在 C++ 11 中引入的。它被称为**静态断言**，我们将在下一节中探讨它。

### 静态断言

有时，我们可以在编译时进行条件检查，以避免任何未来的错误。例如，在一个项目中，我们可能会使用一个第三方库，其中声明了一些数据结构。我们可以使用这些信息来正确分配或释放内存，并处理其成员变量。这个结构属性可能会在第三方库的不同版本中发生变化。然而，如果我们的项目代码仍然使用早期版本的结构，那么在使用它时就会出现问题。我们可能会在运行二进制文件时的后期阶段遇到错误。我们可以使用`static assertion`在编译时捕获这个错误。我们可以对静态数据进行比较，比如库的版本号，从而确保我们的代码不会遇到任何问题。在下一节中，我们将解决一个基于此的练习。

### 练习 2：测试静态断言

在这个练习中，我们将通过进行`静态断言`来比较两个头文件的版本号。如果`版本号`小于`1`，那么静态断言错误将被抛出。执行以下步骤来实现这个练习：

1.  创建一个名为`name`、`age`和`address`的头文件。它还有版本号`1`。

1.  创建另一个名为`struct person`的头文件，其中包含以下属性：`name`、`age`、`address`和`Mobile_No`。它还有`版本号 2`。现在，`版本 1`是旧版本，`版本 2`是新版本。以下是两个头文件并排的截图：![图 7.5：具有不同版本的库文件](img/C14583_07_05.jpg)

###### 图 7.5：具有不同版本的库文件

1.  创建一个名为`doSanityCheck()`的文件，用于对库进行版本检查。它使用静态断言，并在编译时执行。代码的第二行显示了`doSanityCheck()`函数，`static_assert()`函数检查此库的版本是否大于 1。

#### 注意

如果您的项目需要在`版本 2`或更高版本的库中定义的`person`结构才能正确执行，我们需要匹配`版本 2`的文件，即`PERSON_LIB_VERSION`至少应设置为`2`。如果开发人员获得了库的`版本 1`并尝试为项目创建二进制文件，可能会在执行时出现问题。为了避免这种情况，在项目的主代码中，在构建和执行之前对项目进行健全性检查。

1.  要在我们的`版本 1`中包含库的`版本 1`。

1.  编译我们的`static_assert`错误，因为库的版本不匹配。

1.  现在，为了正确编译程序，删除`ProgramLibrary`的软链接，并创建一个指向`version2`的新链接，然后再次编译。这次，它将编译成功。在终端中输入以下命令以删除软链接：

```cpp
rm PersonLibrary.h 
ln -s PersonLibrary_ver2.h PersonLibrary.h
g++ StaticAssertionSample.cpp
```

以下是相同的屏幕截图：

![图 7.7：静态断言编译文件](img/C14583_07_07.jpg)

###### 图 7.7：静态断言编译文件

如您所见，红色标记的区域显示使用了正确版本的`PersonLibrary`，编译进行顺利。编译后，将创建一个名为“**a.exe**”的二进制文件。在这个练习中，我们通过比较两个头文件的版本号执行了静态断言。在下一节中，我们将探讨异常处理的概念。

### 理解异常处理

正如我们之前在调试模式二进制中看到的，我们可以使用运行时断言来中止程序，当某个条件不满足时。但是在发布模式二进制或生产环境中，当客户使用此产品时，突然中止程序并不是一个好主意。最好处理这样的错误条件，并继续执行二进制的下一部分。

最坏的情况发生在二进制需要退出时。它会通过添加正确的日志消息和清理为该进程分配的所有内存来优雅地退出。对于这种情况，使用异常处理。在这里，当发生错误条件时，执行会转移到一个特殊的代码块。异常包括三个部分，如下所示：

+   **try 块**：在这里，我们检查条件是否符合必要的条件。

+   **throw 块**：如果条件不符合，它会抛出异常。

+   **catch 块**：它捕获异常并对该错误条件执行必要的执行。

在下一节中，我们将解决一个练习，在其中我们将对我们的代码执行异常处理。

### 练习 3：执行异常处理

在这个练习中，我们将在我们的**AssertSample.cpp**代码上执行异常处理。我们将用我们的异常替换断言条件。执行以下步骤来实现这个练习：

1.  创建一个名为`ExceptionSample.cpp`的文件。

1.  添加以下代码以添加头文件：

```cpp
#include<iostream>
#include<cstring>
using std::cout;
using std::endl; 
```

1.  创建一个`checkValidIp()`函数，在其中有一个 try-catch 块。如果 try 块中的条件不满足，将抛出异常，并打印 catch 块中的消息。添加以下代码来完成这个操作：

```cpp
bool checkValidIp(const char * ip){
    try{
        if(ip == NULL)
            throw ("ip is NULL");
        if(strlen(ip) > 15)
            throw int(strlen(ip));
    }
    catch(const char * str){
        cout << "Error in checkValidIp :"<< str << endl;
        return false;
    }
    catch(int len){
        cout << "Error in checkValidIp, ip len:" << len <<" greater than 15 characters, condition fail" << endl;
        return false;
    }
    cout << "strlen: " << strlen(ip) << endl;
    return true;
}
```

在前面的代码中，您可以看到 try 块，其中检查条件。在 try 块内，如果`ip`是`NULL`，那么它将抛出(`const char *`)类型的异常。在下一个条件中，如果`ip`大于 15，则它将抛出带有 int 参数类型的异常。这个抛出被正确的 catch 捕获，匹配参数（`int`或`const char *`）。两个异常都返回带有一些错误消息的`false`。或者，在`catch`块中，如果需要进行任何清理或使用在异常中用于比较的变量的默认值，可以执行额外的步骤。

#### 注意

有一个默认的异常；例如，如果有一个嵌套函数抛出一个带有不同参数的错误，它可以作为具有参数的更高级函数捕获（…）。同样，在通用 catch 中，您可以为异常处理创建默认行为。

1.  创建`main()`函数，并在其中写入以下代码：

```cpp
int main(){
    const char * ip;
    ip = NULL;
    if (checkValidIp(ip)) 
        cout << "IP address is correctly validated" << endl;
    else {
        /// work on error condition 
        // if needed exit program gracefully.
        return -1;
    }
    return 0;
}
```

1.  打开终端，编译我们的文件，并运行二进制文件。您将看到以下输出：![图 7.8：带有异常处理的示例执行代码](img/C14583_07_08.jpg)

###### 图 7.8：带有异常处理的示例执行代码

前面的示例对`ip`为`NULL`抛出异常并优雅退出。

1.  现在，在`main`函数中修改`ip`的值，提供超过 15 个字符。编写以下代码来执行此操作：

```cpp
ip = "111.111.111.11111";
```

1.  打开终端，编译我们的文件，然后运行二进制文件。您将看到以下输出：![图 7.9：异常处理的另一个例子](img/C14583_07_09.jpg)

###### 图 7.9：异常处理的另一个例子

它为“ip 字符串”的“长度不匹配”抛出错误。

1.  再次修改`main`函数中`ip`的值，提供少于`15`个字符。编写以下代码来实现这一点：

```cpp
ip = "111.111.111.111";
```

1.  打开终端，编译我们的文件，然后运行二进制文件。您将看到以下输出：

![图 7.10：二进制文件正常运行，没有抛出异常](img/C14583_07_10.jpg)

###### 图 7.10：二进制文件正常运行，没有抛出异常

如前面的截图所示，二进制文件正常执行，没有抛出任何异常。现在您已经了解了如何处理异常，在下一节中，我们将探讨“单元测试”和“模拟测试”的概念。

## 单元测试和模拟测试

当开发人员开始编写代码时，他们需要确保在单元级别正确测试代码。可能会出现边界条件被忽略的情况，当代码在客户端站点运行时可能会出现故障。为了避免这种情况，通常最好对代码进行“单元测试”。“单元测试”是在代码的单元级别或基本级别进行的测试，在这里开发人员可以在隔离的环境中测试他们的代码，假设已经满足了运行代码功能所需的设置。通常，将模块分解为小函数并分别测试每个函数是一个很好的实践。

例如，假设功能的一部分是读取配置文件并使用配置文件中的参数设置环境。我们可以创建一个专门的函数来编写这个功能。因此，为了测试这个功能，我们可以创建一组单元测试用例，检查可能失败或行为不正确的各种组合。一旦确定了这些测试用例，开发人员可以编写代码来覆盖功能，并确保它通过所有单元测试用例。这是开发的一个良好实践，您首先不断添加测试用例，然后相应地添加代码，然后运行该函数的所有测试用例，并确保它们的行为是适当的。

有许多可用于编写和集成项目的单元测试用例的工具。其中一些是“Google 测试框架”。它是免费提供的，并且可以与项目集成。它使用**xUnit 测试框架**，并具有一系列断言，可用于测试用例的条件。在下一节中，我们将解决一个练习，其中我们将创建我们的第一个单元测试用例。

### 练习 4：创建我们的第一个单元测试用例

在这个练习中，我们将处理与上一节讨论过的相同场景，即开发人员被要求编写一个函数来解析“配置文件”。配置文件中传递了不同的有效参数，例如“产品可执行文件名”、“版本号”、“数据库连接信息”、“连接到服务器的 IP 地址”等。假设开发人员将分解解析文件的所有功能，并在单独的函数中设置和测试各个属性的参数。在我们的情况下，我们假设开发人员正在编写功能，他们已经将“IP 地址”解析为“字符串”，并希望推断出该“字符串”是否是有效的“IP 地址”。目前，使“IP 地址”有效的标准需要满足以下条件：

+   “字符串”不应为空。

+   “字符串”不应包含超过`16`个字符

+   “字符串”应该是`XXX.XXX.XXX.XXX`的格式，其中`X`必须是`0`-`9`的数字。

执行以下步骤来实现这个练习：

1.  创建`checkValidIp()`来检查`IP 地址`是否有效。再次，为了理解`Google 单元测试`，我们将编写最少的代码来理解这个功能。

1.  创建一个`ip`不为空，并且长度小于`16`：

```cpp
#include "CheckIp.h"
#include<string>
#include<sstream>
bool checkValidIp(const char * ip){
    if(ip == NULL){
        cout << "Error : IP passes is NULL " << endl;
        return false;
    }
    if(strlen(ip) > 15){
        cout << "Error: IP size is greater than 15" << endl;
        return false;
    }
    cout << "strlen: " << strlen(ip) << endl;
    return true;
} 
```

在前面的代码中，如果两个条件都失败，函数将返回`false`。

1.  调用`checkValidIp()`函数来创建一个名为`checkValidIP()`函数的新文件。在其中添加以下代码：

```cpp
#include"CheckIp.h"
int main(){
    const char * ip;
    //ip = "111.111.111.111";
    ip = "111.111.111.11111";
    if (checkValidIp(ip)) 
        cout << "IP address is correctly validated" << endl;
    else {
        /// work on error condition 
        // if needed exit program gracefully.
        cout << " Got error in valid ip " << endl;
        return -1;
    }
    return 0;
} 
```

1.  要创建测试代码，我们将创建我们的第一个`checkValidIp`函数。在其中写入以下代码：

```cpp
#include"CheckIp.h"
#include<gtest/gtest.h>
using namespace std;
const char * testIp;
TEST(CheckIp, testNull){
    testIp=NULL;
    ASSERT_FALSE(checkValidIp(testIp));
}
TEST(CheckIp, BadLength){
    testIp = "232.13.1231.1321.123";
    ASSERT_FALSE(checkValidIp(testIp));
}
```

在前面代码的第二行，我们包含了`TEST`函数，它接受两个参数：第一个是`testsuite`名称，第二个是`testcase`名称。对于我们的情况，我们创建了`TestSuite` `CheckIp`。在`TEST`块中，您将看到我们有`Google 测试`定义了一个名为`ASSERT_FALSE`的`assert`，它将检查条件是否为`false`。如果不是，它将使测试用例失败，并在结果中显示相同的内容。

#### 注意

通常，对于`Google 测试`用例和测试套件，您可以将它们分组在一个公共命名空间中，并调用`RUN_ALL_TESTS`宏，该宏运行附加到测试二进制文件的所有测试用例。对于每个测试用例，它调用`SetUp`函数来初始化（类中的构造函数），然后调用实际的测试用例，最后调用`TearDown`函数（类中的析构函数）。除非您必须为测试用例初始化某些内容，否则不需要编写`SetUp`和`TearDown`函数。

1.  现在，要运行测试用例，我们将创建主`RUN_ALL_TESTS`宏。或者，我们可以创建一个可执行文件，链接`Google Test 库`，并调用`RUN_ALL_TESTS`。对于我们的情况，我们将选择后者。打开终端并运行以下命令以创建一个测试运行二进制文件：

```cpp
g++ -c CheckIp.cpp
```

这将包括`CheckValidIp`函数的对象文件在其中定义。

1.  现在，输入以下命令以添加必要的库，这些库将被链接以创建一个二进制文件：

```cpp
g++ CheckIp.o TestCases.cpp -lgtest -lgtest_main -pthread -o TestRun 
```

1.  现在，使用以下命令运行二进制文件：

```cpp
./TestRun
```

这显示了通过`CheckIp` `testsuite`的两个测试用例。第一个测试用例`CheckIp.testNull`被调用并通过了。第二个测试用例`CheckIp.BadLength`也被调用并通过了。这个结果在以下截图中可见：

![图 7.11：编译和执行测试用例](img/C14583_07_11.jpg)

###### 图 7.11：编译和执行测试用例

#### 注意

在`Google 测试`中，我们也可以使用其他断言，但对于我们的测试用例，我们满意于`ASSERT_FALSE`，因为我们只检查我们传递的 IP 地址的假条件。

1.  现在，我们将添加更多的测试用例来使我们的代码更加健壮。这通常是编写代码的良好实践。首先，创建测试用例，并确保代码对新测试用例和旧测试用例以及代码的正确功能都能正常运行。要添加更多的测试用例，将以下代码添加到`IP`以"."开头。如果`IP`以"."结尾，则第四个案例应该失败。如果`IP`之间有空格，则第五个案例应该失败。如果`IP`包含任何非数字字符，则第六个案例应该失败。如果`IP`的令牌值小于`0`且大于`255`，则第七个案例应该失败。如果`IP`的令牌计数错误，则最后一个案例应该失败。

1.  现在，在**CheckIp.cpp**文件的`CheckValidIp()`函数中添加以下代码。这段代码是处理新测试用例所必需的：

```cpp
if(ip[strlen(ip)-1] == '.'){
    cout<<"ERROR : Incorrect token at end"<<endl;
    return false;
}
isstringstream istrstr(ip);
vector<string> tokens;
string token;
regex expression("[⁰-9]");
smatch m;
while(getline(istrstr, token, '.')){
    if(token.empty()){
        cout<<"ERROR : Got empty token"<<endl;
        return false;
    }
    if(token.find(' ') != string::npos){
        cout<<"ERROR : Space character in token"<<endl;
        return false;
    }
    if(regex_search(token,m,expression)){
        cout<<"ERROR : NonDigit character in token"<<endl;
        return false;
    }
    int val = atoi(token.c_str());
    if(val<0 || val>255){
        cout<<"ERROR : Invalid digit in token"<<endl;
        return false;
    }
    tokens.push_back(token);
}
if(tokens.size()!=4){
    cout<<"ERROR : Incorrect IP tokens used"<<endl;
    return false;
}
cout<<"strlen: "<<strlen(ip)<<endl;
return true;
}
```

1.  打开终端并输入以下命令以运行二进制文件：

```cpp
./TestRun
```

所有测试用例都已执行，如下截图所示：

![图 7.12：测试用例运行的输出](img/C14583_07_12.jpg)

###### 图 7.12：测试用例运行的输出

前面的截图显示了`CheckIp`测试套件中有`10`个测试用例，并且所有测试用例都运行正常。在下一节中，我们将学习使用模拟对象进行单元测试。

### 使用模拟对象进行单元测试

当开发人员进行单元测试时，可能会出现在具体操作发生后调用某些接口的情况。例如，正如我们在前面的情景中讨论的，假设项目设计成在执行之前从数据库中获取所有配置信息。它查询数据库以获取特定参数，例如 Web 服务器的`IP 地址`，`用户`和`密码`。然后尝试连接到 Web 服务器（也许有另一个模块处理与网络相关的任务）或开始对实际项目所需的项目进行操作。之前，我们测试了 IP 地址的有效性。现在，我们将更进一步。假设 IP 地址是从数据库中获取的，并且我们有一个实用类来处理连接到`DB`和查询`IP 地址`。

现在，为了测试 IP 地址的有效性，我们需要假设数据库连接已经设置好。这意味着应用程序可以正确地查询数据库并获取查询结果，其中之一是`IP 地址`。只有这样，我们才能测试 IP 地址的有效性。现在，为了进行这样的测试，我们必须假设所有必要的活动都已经完成，并且我们已经得到了一个`IP 地址`来测试。这就是模拟对象的作用，它就像真实对象一样。它提供了单元测试的功能，以便应用程序认为 IP 地址已经从数据库中获取，但实际上我们是模拟的。要创建一个模拟对象，我们需要从它需要模拟的类中继承。在下一节中，我们将进行一个练习，以更好地理解模拟对象。

### 练习 5：创建模拟对象

在这个练习中，我们将通过假设所有接口都按预期工作来创建模拟对象。使用这些对象，我们将测试一些功能，比如验证`IP 地址`，检查数据库连接性，以及检查`用户名`和`密码`是否格式正确。一旦所有测试都通过了，我们将确认应用程序，并准备好进行`QA`。执行以下步骤来实现这个练习：

1.  创建一个名为**Misc.h**的头文件，并包含必要的库：

```cpp
#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<iterator>
#include<regex>
using namespace std;
```

1.  创建一个名为`ConnectDatabase`的类，它将连接到数据库并返回查询结果。在类内部，声明`Dbname`，user 和 passwd 变量。还声明一个构造函数和两个虚函数。在这两个虚函数中，第一个必须是析构函数，第二个必须是`getResult()`函数，它从数据库返回查询结果。添加以下代码来实现这一点：

```cpp
class ConnectDatabase{
    string DBname;
    string user;
    string passwd;
    public:
        ConnectDatabase() {} 
        ConnectDatabase(string _dbname, string _uname, string _passwd) :
            DBname(_dbname), user(_uname), passwd(_passwd) { }
        virtual ~ConnectDatabase() {} 
        virtual string getResult(string query);
};
```

1.  创建另一个名为`WebServerConnect`的类。在`class`内部声明三个`string`变量，分别是`Webserver`，`uname`和`passwd`。创建构造函数和两个虚函数。在这两个虚函数中，第一个必须是析构函数，第二个必须是`getRequest()`函数。添加以下代码来实现这一点：

```cpp
class WebServerConnect{
    string Webserver;
    string uname;
    string passwd;
    public :
    WebServerConnect(string _sname, string _uname, string _passwd) :
            Webserver(_sname), uname(_uname), passwd(_passwd) { }
        virtual ~WebServerConnect() {}
        virtual string getRequest(string req);
};
```

#### 注意

由于我们将从前面的类创建一个`模拟类`并调用这些函数，所以需要`虚函数`。

1.  创建一个名为`App`的类。创建构造函数和析构函数并调用所有函数。添加以下代码来实现这一点：

```cpp
class App {
    ConnectDatabase *DB;
    WebServerConnect *WB;
    public : 
        App():DB(NULL), WB(NULL) {} 
        ~App() { 
            if ( DB )  delete DB;
            if ( WB )  delete WB;
        }
        bool checkValidIp(string ip);
        string getDBResult(string query);
        string getWebResult(string query);
        void connectDB(string, string, string);
        void connectDB(ConnectDatabase *db);
        void connectWeb(string, string, string);
        void run();
};
```

在前面的代码中，应用程序将首先查询数据库并获取`IP 地址`。然后，它使用必要的信息连接到 Web 服务器并查询以获取所需的信息。

1.  创建一个名为`gmock`的类头文件，这是创建模拟类所需的。此外，`MockDB`类是从`ConnectDatabase`类继承的。`MOCK_METHOD1(getResult, string(string));`这一行表示我们将模拟`getResult`接口。因此，在单元测试期间，我们可以直接调用`getResult`函数，并传递所需的结果，而无需创建`ConnectDatabase`类并运行实际的数据库查询。需要注意的一个重要点是，我们需要模拟的函数必须使用`MOCK_METHOD[N]`宏进行定义，其中 N 是接口将接受的参数数量。在我们的情况下，`getResult`接口接受一个参数。因此，它使用`MOCK_METHOD1`宏进行模拟。

1.  创建一个名为`getResult()`和`getRequest()`的函数，其中 DB 查询和`WebServer`查询返回默认字符串。在这里，`App::run()`函数假设 DB 连接和 web 服务器连接已经执行，现在它可以定期执行 web 查询。在每次查询结束时，它将默认返回"`Webserver returned success`"字符串。

1.  现在，创建一个名为`dbname`、`dbuser`和`dbpasswd`的文件。然后，我们查询数据库以获取 IP 地址和其他配置参数。我们已经注释掉了`app.checkValidIp(ip)`这一行，因为我们假设从数据库中获取的 IP 地址需要进行验证。此外，这个函数需要进行单元测试。使用`connectWeb()`函数，我们可以通过传递虚拟参数如`webname`、`user`和`passwd`来连接到 web 服务器。最后，我们调用`run()`函数，它将迭代运行，从而查询 web 服务器并给出默认输出。

1.  保存所有文件并打开终端。为了获得执行项目所需的基本功能，我们将构建二进制文件并执行它以查看结果。在终端中运行以下命令：

```cpp
g++ Misc.cpp RunApp.cpp -o RunApp
```

上述代码将在当前文件夹中创建一个名为`RunApp`的二进制文件。

1.  现在，编写以下命令来运行可执行文件：

```cpp
./RunApp
```

上述命令在终端中生成以下输出：

![图 7.13：运行应用程序](img/C14583_07_13.jpg)

###### 图 7.13：运行应用程序

如前面的截图所示，二进制文件及时显示输出"`Webserver returned success`"。到目前为止，我们的应用程序正常运行，因为它假设所有接口都按预期工作。但在将其准备好供 QA 测试之前，我们仍需测试一些功能，如验证`IP 地址`、`DB 连接性`、检查`用户名`和`密码`是否符合正确格式等。

1.  使用相同的基础设施，开始对每个功能进行单元测试。在我们的练习中，我们假设`DB 连接`已经完成，并已查询以获取`IP 地址`。之后，我们可以开始单元测试`IP 地址`的有效性。因此，在我们的测试用例中，需要模拟数据库类，并且`getDBResult`函数必须返回`IP 地址`。稍后，这个`IP 地址`将传递给`checkValidIP`函数进行测试。为了实现这一点，创建一个名为`checkValidIP`的类：

```cpp
#include"MockMisc.h"
using ::testing::_;
using ::testing::Return;
class TestApp : public ::testing::Test {
    protected : 
        App testApp;
        MockDB *mdb;
        void SetUp(){
            mdb = new MockDB();
            testApp.connectDB(mdb);
        }
        void TearDown(){
        }
};
TEST_F(TestApp, NullIP){
    EXPECT_CALL(*mdb, getResult(_)).
                 WillOnce(Return(""));
    ASSERT_FALSE(testApp.checkValidIp(testApp.getDBResult("")));
}
TEST_F(TestApp, SpaceTokenIP){
    EXPECT_CALL(*mdb, getResult(_)).
                 WillOnce(Return("13\. 21.31.68"));
    ASSERT_FALSE(testApp.checkValidIp(testApp.getDBResult("")));
}
TEST_F(TestApp, NonValidDigitIP){
    EXPECT_CALL(*mdb, getResult(_)).
                 WillOnce(Return("13.521.31.68"));
    ASSERT_FALSE(testApp.checkValidIp(testApp.getDBResult("")));
}
TEST_F(TestApp, CorrectIP){
    EXPECT_CALL(*mdb, getResult(_)).
                 WillOnce(Return("212.121.21.45"));
    ASSERT_TRUE(testApp.checkValidIp(testApp.getDBResult("")));
}
```

在这里，我们使用了测试和`testing::Return`命名空间来调用模拟类接口，并返回用于测试用例的用户定义的值。在`TEST_F`函数中，我们使用了`EXPECT_CALL`函数，其中我们将模拟对象的实例作为第一个参数传递，并将`getResult()`函数作为第二个参数传递。`WillOnce(Return(""))`行表示需要调用接口一次，并将返回""和一个空字符串。这是需要传递给`checkValidIP`函数以测试空字符串的值。这通过`ASSERT_FALSE`宏进行检查。类似地，可以使用 DB 的模拟对象创建其他测试用例，并将 IP 地址传递给`checkValidIP`函数。为了创建各种测试用例，`TestApp`类从`testing::Test`类继承，其中包含 App 实例和 Database 的模拟对象。在`TestApp`类中，我们定义了两个函数，即`SetUp()`和`TearDown()`。在`SetUp()`函数中，我们创建了一个`MockDB`实例并将其标记为 testApp 实例。由于`TearDown()`函数不需要执行任何操作，我们将其保持为空。它的析构函数在`App`类的析构函数中被调用。此外，我们在`TEST_F`函数中传递了两个参数。第一个参数是测试类，而第二个参数是测试用例的名称。

1.  保存所有文件并打开终端。运行以下命令：

```cpp
g++ Misc.cpp TestApp.cpp -lgtest -lgmock -lgtest_main -pthread -o TestApp
```

在前面的命令中，我们还链接了`gmock 库`。现在，输入以下命令来运行测试用例：

```cpp
./TestApp
```

前面的命令生成了以下输出：

![图 7.14：运行 Gmock 测试](img/C14583_07_14.jpg)

###### 图 7.14：运行 Gmock 测试

从前面的命令中，我们可以看到所有的测试用例都执行并成功通过了。在下一节中，我们将讨论`断点`、`观察点`和`数据可视化`。

### 断点、观察点和数据可视化

在前面的部分中，我们讨论了在开发人员将代码检入存储库分支之前需要进行单元测试，并且其他团队成员可以看到它，以便他们可以将其与其他模块集成。虽然单元测试做得很好，开发人员检查了代码，但在集成代码并且 QA 团队开始测试时，可能会发现代码中存在错误的机会。通常，在这种情况下，可能会在由于其他模块的更改而导致的模块中抛出错误。团队可能会很难找出这些问题的真正原因。在这种情况下，**调试**就出现了。它告诉我们代码的行为如何，开发人员可以获得代码执行的细粒度信息。开发人员可以看到函数正在接收的参数以及它返回的值。它可以准确地告诉一个变量或指针分配了什么值，或者内存中的内容是什么。这对于开发人员来说非常有帮助，可以确定代码的哪一部分存在问题。在下一节中，我们将实现一个堆栈并对其执行一些操作。

### 与堆栈数据结构一起工作

考虑这样一个场景，其中开发人员被要求开发自己的堆栈结构，可以接受任何参数。在这里，要求是堆栈结构必须遵循**后进先出**（**LIFO**）原则，其中元素被放置在彼此之上，当它们从堆栈中移除时，最后一个元素应该首先被移除。它应该具有以下功能：

+   **push()**将新元素放置在堆栈顶部

+   **top()**显示堆栈的顶部元素（如果有）

+   **pop()**从堆栈中移除最后插入的元素

+   **is_empty()**检查堆栈是否为空

+   **size()**显示堆栈中存在的元素数量

+   **clean()**清空堆栈（如果有任何元素）

以下代码行显示了如何在**Stack.h**头文件中包含必要的库：

```cpp
#ifndef STACK_H__
#define STACK_H__
#include<iostream>
using namespace std;
```

正如我们已经知道的，栈由各种操作组成。为了定义这些函数中的每一个，我们将编写以下代码：

```cpp
template<typename T>
struct Node{
    T element;
    Node<T> *next;
};
template<typename T>
class Stack{
    Node<T> *head;
    int sz;
    public :
        Stack():head(nullptr), sz(0){}
        ~Stack();

        bool is_empty();
        int size();
        T top();
        void pop();
        void push(T);
        void clean();
};
template<typename T>
Stack<T>::~Stack(){
    if ( head ) clean();
}
template<typename T>
void Stack<T>::clean(){
    Node<T> *tmp;
    while( head ){
        tmp = head;
        head = head -> next;
        delete tmp;
        sz--;
    }
}
template<typename T>
int Stack<T>::size(){
    return sz;
}
template<typename T>
bool Stack<T>::is_empty(){
        return (head == nullptr) ? true : false;
}
template<typename T>
T Stack<T>::top(){
    if ( head == nullptr){
        // throw error ...
        throw(string("Cannot see top of empty stack"));
    }else {
        return head -> element;
    }
}
template<typename T>
void Stack<T>::pop(){
    if ( head == nullptr ){
        // throw error
        throw(string("Cannot pop empty stack"));
    }else {
        Node<T> *tmp = head ;
        head = head -> next;
        delete tmp;
        sz--;
    }
}
template<typename T>
void Stack<T>::push(T val){
    Node<T> *tmp = new Node<T>();
    tmp -> element = val;
    tmp -> next = head;
    head = tmp;
    sz++;
}
// Miscellaneous functions for stack.. 
template<typename T>
void displayStackStats(Stack<T> &st){
    cout << endl << "------------------------------" << endl;
    cout << "Showing Stack basic Stats ...  " << endl;
    cout << "Stack is empty : " << (st.is_empty() ? "true" : "false") << endl;
    cout << "Stack size :" << st.size() << endl;
    cout << "--------------------------------" << endl << endl;
}
#endif 
```

到目前为止，我们已经看到了如何使用`单链表`实现栈。每次在 Stack 中调用`push`时，都会创建一个给定值的新元素，并将其附加到栈的开头。我们称之为头成员变量，它是头部将指向栈中的下一个元素等等。当调用`pop`时，头部将从栈中移除，并指向栈的下一个元素。

让我们在`22`、`426`和`57`中编写先前创建的 Stack 的实现。当调用`displayStackStats()`函数时，它应该声明栈的大小为`3`。然后，我们从栈中弹出`57`，顶部元素必须显示`426`。我们将对 char 栈执行相同的操作。以下是栈的完整实现：

```cpp
#include"Stack.h"
int main(){
    try {
        Stack<int> si;
        displayStackStats<int>(si);
        si.push(22);
        si.push(426);
        cout << "Top of stack contains " << si.top() << endl;
        si.push(57);
        displayStackStats<int>(si);
        cout << "Top of stack contains " << si.top() << endl;
        si.pop();
        cout << "Top of stack contains " << si.top() << endl;
        si.pop();
        displayStackStats<int>(si);
        Stack<char> sc;
        sc.push('d');
        sc.push('l');
        displayStackStats<char>(sc);
        cout << "Top of char stack contains:" << sc.top() << endl;
    }
    catch(string str){
        cout << "Error : " << str << endl;
    }
    catch(...){
        cout << "Error : Unexpected exception caught " << endl;
    }
    return 0;
}
```

当我们编译时（使用了`-g`选项）。因此，如果需要，您可以调试二进制文件：

```cpp
g++ -g Main.cpp -o Main
```

我们将写以下命令来执行二进制文件：

```cpp
./Main
```

前面的命令生成了以下输出：

![图 7.15：使用 Stack 类的主函数](img/C14583_07_15.jpg)

###### 图 7.15：使用 Stack 类的主函数

在前面的输出中，统计函数的第二次调用中的红色墨水显示了在 int 栈中显示三个元素的正确信息。然而，int 栈顶部的红色墨水调用显示了随机或垃圾值。如果程序再次运行，它将显示一些其他随机数字，而不是预期的值`57`和`426`。同样，对于 char 栈，红色墨水突出显示的部分，即`char`的顶部，显示了垃圾值，而不是预期的值，即"l"。后来，执行显示了双重释放或损坏的错误，这意味着再次调用了相同的内存位置。最后，可执行文件产生了核心转储。程序没有按预期执行，从显示中可能不清楚实际错误所在。为了调试`Main`，我们将编写以下命令：

```cpp
gdb ./Main 
```

前面的命令生成了以下输出：

![图 7.16：调试器显示 – I](img/C14583_07_16.jpg)

###### 图 7.16：调试器显示 – I

在前面的屏幕截图中，蓝色突出显示的标记显示了调试器的使用方式以及它显示的内容。第一个标记显示了使用`gdb`命令调用调试器。输入`gdb`命令后，用户进入调试器的命令模式。以下是命令模式中使用的命令的简要信息：

+   **b main**：这告诉调试器在主函数调用时中断。

+   **r**：这是用于运行可执行文件的简写。也可以通过传递参数来运行。

+   **n**：这是下一个命令的简写，告诉我们执行下一个语句。

+   `si`变量在代码中被调用时，其值会发生变化。调试器将显示使用此变量的代码的内容。

+   `step in`"命令。

将执行的下一个语句是`si.push(22)`。由于`si`已经更新，观察点调用并显示了`si`的旧值和一个新值，其中显示了`si`的旧值是带有 NULL 的头部和`sz`为 0。在`si.push`之后，头部将更新为新值，并且其执行到了`Stack.h`文件的第 75 行，这是`sz`变量增加的地方。如果再次按下*Enter*键，它将执行。

请注意，执行已自动从主函数移动到`Stack::push`函数。以下是调试器上继续命令的屏幕截图：

![](img/C14583_07_17.jpg)

###### 图 7.17：调试器显示 – II

下一个命令显示`sz`已更新为新值`1`。按*Enter*后，代码的执行从`Stack::push`的`第 76 行`返回到主函数的`第 8 行`。这在下面的屏幕截图中有所突出。它显示执行停在`si.push(426)`的调用处。一旦我们进入，`Stack::push`将被调用。执行移动到`Stack.h`程序的`第 71 行`，如红色墨水所示。一旦执行到达`第 74 行`，如红色墨水所示，watch 被调用，显示`si`已更新为新值。您可以看到在`Stack::push`函数完成后，流程回到了主代码。以下是调试器中执行的步骤的屏幕截图：

![](img/C14583_07_18.jpg)

###### 图 7.18：调试器显示-III

按*Enter*后，您会看到`displayStackStats`在`第 11 行`被调用。然而，在`第 12 行`，显示的值是`0`，而不是预期的值`57`。这是一个错误，我们仍然无法弄清楚-为什么值会改变？但是，很明显，值可能在前面对主函数的调用中的某个地方发生了变化。因此，这可能不会让我们对继续进行调试感兴趣。但是，我们需要继续并从头开始调试。

以下屏幕截图显示了将用于调试代码的命令：

![](img/C14583_07_19.jpg)

###### 图 7.19：调试器显示-IV

要从头重新运行程序，我们必须按*r*，然后按*y*进行确认和继续，这意味着我们从头重新运行程序。它会要求确认；按*y*继续。在前面的屏幕截图中，所有这些命令都用蓝色标出。在第 7 行执行时，我们需要运行'`display *si.head`'命令，它将在执行每条语句后持续显示`si.head`内存位置的内容。如红色墨水所示，在将`22`推入堆栈后，head 会更新为正确的值。类似地，对于值`426`和`57`，在使用 push 将其插入堆栈时，对 head 的调用也会正确更新。

稍后，当调用`displayStackStats`时，它显示了正确的`size`为`3`。但是当调用 top 命令时，head 显示了错误的值。这在红色墨水中有所突出。现在，top 命令的代码不会改变 head 的值，因此很明显错误发生在前一条执行语句中，也就是在`displayStackStats`处。

因此，我们已经缩小了可能存在问题的代码范围。我们可以运行调试器指向`displayStackStats`并移动到`displayStackStats`内部，以找出导致堆栈内部值发生变化的原因。以下是同一屏幕截图，用户需要从头开始启动调试器：

![图 7.20：调试器显示-IV](img/C14583_07_20.jpg)

###### 图 7.20：调试器显示-IV

重新启动调试器并到达调用`displayStackStats`的第 11 行执行点后，我们需要进入。流程是进入`displayStackStats`函数的开头。此外，我们需要执行下一条语句。由于函数中的初始检查是清晰的，它们不会改变 head 的值，我们可以按*Enter*执行下一步。当我们怀疑下一步可能会改变我们正在寻找的变量的值时，我们需要进入。这是在前面的快照中完成的，用红色标出。后面的执行到达`第 97 行`，也就是`displayStackStats`函数的最后一行。

在输入*s*后，执行移动到析构堆栈并在第 81 行调用清理函数。此清理命令删除了与头部相同值的`tmp`变量。该函数清空了堆栈，这是不希望发生的。只有`displayStackStats`函数应该被调用和执行，最终返回到主函数。但是，由于局部变量超出范围，析构函数可能会被调用。在这里，局部变量是在`line 92`处作为`displayStackStats`函数的参数声明的变量。因此，当调用`displayStackStats`函数时，会创建来自主函数的`si`变量的局部副本。当`displayStackStats`函数被调用时，该变量调用了 Stack 的析构函数。现在，`si`变量的指针已被复制到临时变量，并且错误地在最后删除了指针。这不是开发人员的意图。因此，在代码执行结束时，会报告双重释放错误。`si`变量在超出范围时必须调用 Stack 析构函数，因为它将尝试再次释放相同的内存。为了解决这个问题，很明显`displayStackStats`函数必须以传递参数作为引用的方式进行调用。为此，我们必须更新`Stack.h`文件中`displayStackStats`函数的代码：

```cpp
template<typename T>
void displayStackStats(Stack<T> &st){
    cout << endl << "------------------------------" << endl;
    cout << "Showing Stack basic Stats ...  " << endl;
    cout << "Stack is empty : " << (st.is_empty() ? "true" : "false") << endl;
    cout << "Stack size :" << st.size() << endl;
    cout << "--------------------------------" << endl << endl;
}
```

现在，当我们保存并编译**Main.cpp**文件时，将生成二进制文件：

```cpp
./Main
```

前面的命令在终端中生成以下输出：

![图 7.21：调试器显示 - IV](img/C14583_07_21.jpg)

###### 图 7.21：调试器显示 - IV

从前面的屏幕截图中，我们可以看到`57`和`426`的预期值显示在堆栈顶部。`displayStackStats`函数还显示了 int 和 char 堆栈的正确信息。最后，我们使用调试器找到了错误并进行了修复。在下一节中，我们将解决一个活动，我们将开发用于解析文件并编写测试用例以检查函数准确性的函数。

### 活动 1：使用测试用例检查函数的准确性并了解测试驱动开发（TDD）

在这个活动中，我们将开发函数，以便我们可以解析文件，然后编写测试用例来检查我们开发的函数的正确性。

一个大型零售组织的 IT 团队希望通过在其数据库中存储产品详情和客户详情来跟踪产品销售作为其对账的一部分。定期，销售部门将以简单的文本格式向 IT 团队提供这些数据。作为开发人员，您需要确保在公司将记录存储在数据库之前，对数据进行基本的合理性检查，并正确解析所有记录。销售部门将提供两个包含客户信息和货币信息的文本文件。您需要编写解析函数来处理这些文件。这两个文件是`Currency`和`ConversionRatio`。

此项目环境设置的所有必要信息都保存在配置文件中。这也将保存文件名，以及其他参数（如`DB`，`RESTAPI`等）和文件`recordFile`中的变量值，以及货币文件，变量名为`currencyFile`。

以下是我们将编写的测试条件，以检查用于解析**CurrencyConversion.txt**文件的函数的准确性：

+   第一行应该是标题行，其第一个字段应包含"`Currency`"字符串。

+   `Currency`字段应由三个字符组成。例如："`USD`"，"`GBP`"是有效的。

+   `ConversionRatio`字段应由浮点数组成。例如，`1.2`，`0.06`是有效的。

+   每行应该恰好有两个字段。

+   用于记录的分隔符是"|"。

以下是我们将编写的测试条件，用于检查用于解析**RecordFile.txt**文件的函数的准确性：

+   第一行应包含标题行，其第一个字段应包含"`Customer Id`"字符串。

+   `Customer Id`，`Order Id`，`Product Id`和`Quantity`应该都是整数值。例如，`12312`，`4531134`是有效的。

+   `TotalPrice (Regional Currency)`和`TotalPrice (USD)`应该是浮点值。例如，`2433.34`，`3434.11`是有效的。

+   `RegionalCurrency`字段的值应该存在于`std::map`中。

+   每行应该有九个字段，如文件的`HEADER`信息中定义的那样。

+   记录的分隔符是"|"。

按照以下步骤执行此活动：

1.  解析**parse.conf**配置文件，其中包括项目运行的环境变量。

1.  从步骤 1 正确设置`recordFile`和`currencyFile`变量。

1.  使用从配置文件中检索的这些变量，解析满足所有条件的货币文件。如果条件不满足，返回适当的错误消息。

1.  解析满足的所有条件的记录文件。如果不满足条件，则返回错误消息。

1.  创建一个名为`CommonHeader.h`的头文件，并声明所有实用函数，即`isAllNumbers()`，`isDigit()`，`parseLine()`，`checkFile()`，`parseConfig()`，`parseCurrencyParameters()`，`fillCurrencyMap()`，`parseRecordFile()`，`checkRecord()`，`displayCurrencyMap()`和`displayRecords()`。

1.  创建一个名为`Util.cpp`的文件，并定义所有实用函数。

1.  创建一个名为`ParseFiles.cpp`的文件，并调用`parseConfig()`，`fillCurrencyMap()`和`parseRecordFile()`函数。

1.  编译并执行`Util.cpp`和`ParseFiles.cpp`文件。

1.  创建一个名为`ParseFileTestCases.cpp`的文件，并为函数编写测试用例，即`trim()`，`isAllNumbers()`，`isDigit()`，`parseCurrencyParameters()`，`checkFile()`，`parseConfig()`，`fillCurrencyMap()`和`parseRecordFile()`。

1.  编译并执行`Util.cpp`和`ParseFileTestCases.cpp`文件。

以下是解析不同文件并显示信息的流程图：

![](img/C14583_07_22.jpg)

###### 图 7.22：流程图

从上面的流程图中，我们大致了解了执行流程。在编写代码之前，让我们看看更细节的内容，以便清楚地理解。这将有助于为每个执行块定义测试用例。

对于解析配置文件块，我们可以将步骤分解如下：

1.  检查配置文件是否存在并具有读取权限。

1.  检查是否有适当的标题。

1.  逐行解析整个文件。

1.  对于每一行，使用'='作为分隔符解析字段。

1.  如果从上一步中有 2 个字段，则处理以查看它是`Currency file`还是`Record file`变量，并适当存储。

1.  如果从步骤 4 中没有 2 个字段，则转到下一行。

1.  完全解析文件后，检查上述步骤中的两个变量是否不为空。

1.  如果为空，则返回错误。

对于解析`Currency File`块，我们可以将步骤分解如下：

1.  读取`CurrencyFile`的变量，看看文件是否存在并且具有读取权限。

1.  检查是否有适当的标题。

1.  逐行解析整个文件，使用'|'作为分隔符。

1.  如果每行找到确切的 2 个字段，将第一个视为`Currency field`，第二个视为`conversion field`。

1.  如果从步骤 3 中没有找到 2 个字段，则返回适当的错误消息。

1.  从步骤 4 开始，对`Currency field`（应为 3 个字符）和`Conversion Field`（应为数字）进行所有检查。

1.  如果从步骤 6 通过，将`currency`/`conversion`值存储为具有`Currency`作为键和数字作为值的映射对。

1.  如果未从步骤 6 通过，返回说明`currency`的错误。

1.  解析完整的`Currency`文件后，将创建一个映射，其中将为所有货币的转换值。

对于解析`Record File`块，我们可以将步骤分解为以下步骤：

1.  读取`RecordFile`的变量，并查看文件是否存在并具有读取权限。

1.  检查是否有适当的头部

1.  逐行解析整个文件，以'|'作为分隔符。

1.  如果从上述步骤中找不到 9 个字段，请返回适当的错误消息。

1.  如果找到 9 个字段，请对活动开始时列出的所有字段进行相应的检查。

1.  如果步骤 5 未通过，请返回适当的错误消息。

1.  如果步骤 5 通过，请将记录存储在记录的向量中。

1.  在完全解析记录文件后，所有记录将存储在记录的向量中。

在创建解析所有三个文件的流程时，我们看到所有 3 个文件都重复了一些步骤，例如：

检查文件是否存在且可读

检查文件是否具有正确的头部信息

使用分隔符解析记录

检查字段是否为`Digit`在`Currency`和`Record file`中是常见的

检查字段是否为`Numeric`在`Currency`和`Record file`中是常见的

上述要点将有助于重构代码。此外，将有一个用于使用分隔符解析字段的常见函数，即`trim`函数。因此，当我们使用分隔符解析记录时，我们可能会得到带有空格或制表符的值，这可能是不需要的，因此我们需要在解析记录时修剪它一次。

现在我们知道我们有上述常见的步骤，我们可以为它们编写单独的函数。为了开始 TDD，我们首先了解函数的要求，并首先编写单元测试用例来测试这些功能。然后我们编写函数，使其通过单元测试用例。如果有几个测试用例失败，我们迭代更新函数并执行测试用例的步骤，直到它们全部通过。

对于我们的示例，我们可以编写`trim`函数，

现在我们知道在修剪函数中，我们需要删除第一个和最后一个额外的空格/制表符。例如，如果字符串包含"AA"，则修剪应返回"AA"删除所有空格。

修剪函数可以返回具有预期值的新字符串，也可以更新传递给它的相同字符串。

所以现在我们可以编写修剪函数的签名：`string trim(string&);`

我们可以为此编写以下测试用例：

+   仅有额外字符(" ")，返回空字符串()。

+   仅以开头的空字符("AA")返回带有结束字符("AA")的字符串

+   仅以结尾的空字符("AA ")，应返回带有开始字符("AA")的字符串

+   在中间有字符("AA")，返回带有字符("AA")的字符串

+   在中间有空格("AA BB")，返回相同的字符串("AA BB")

+   所有步骤 3,4,5 都是单个字符。应返回具有单个字符的字符串。

要创建测试用例，请检查文件`trim`函数是否在测试套件`trim`中编写。现在在文件中编写具有上述签名的`trim`函数。执行`trim`函数的测试用例并检查是否通过。如果没有适当更改函数并再次测试。重复直到所有测试用例通过。

现在我们有信心在项目中使用`trim`函数。对于其余的常见函数（`isDigit`，`isNumeric`，`parseHeader`等），请参考**Util.cpp**文件和**ParseFiletestCases.cpp**，并测试所有常见函数。

完成常见功能后，我们可以分别编写解析每个文件的函数。要理解和学习的主要内容是如何将模块分解为小函数。找到小的重复任务，并为每个创建小函数，以便进行重构。了解这些小函数的详细功能，并创建适当的单元测试用例。

完整测试单个函数，如果失败，则更新函数直到通过所有测试用例。类似地，完成其他函数。然后编写并执行更大函数的测试用例，这应该相对容易，因为我们在这些更大函数中调用了上面测试过的小函数。

在实施了上述步骤之后，我们将得到以下输出：

![图 7.23：所有测试都正常运行](img/C14583_07_23.jpg)

###### 图 7.23：所有测试都正常运行

以下是下一步的屏幕截图：

![图 7.24：所有测试都正常运行](img/C14583_07_24.jpg)

###### 图 7.24：所有测试都正常运行

#### 注意

此活动的解决方案可以在第 706 页找到。

## 摘要

在本章中，我们看了各种通过可执行文件抛出的错误可以在编译时和运行时使用断言来捕获的方法。我们还学习了静态断言。我们了解了异常是如何生成的，以及如何在代码中处理它们。我们还看到单元测试如何可以成为开发人员的救星，因为他们可以在开始时识别代码中的任何问题。我们为需要在测试用例中使用的类使用了模拟对象。然后我们学习了调试器、断点、观察点和数据可视化。我们能够使用调试器找到代码中的问题并修复它们。我们还解决了一个活动，其中我们编写了必要的测试用例来检查用于解析文件的函数的准确性。

在下一章中，我们将学习如何优化我们的代码。我们将回顾处理器如何执行代码并访问内存。我们还将学习如何确定软件执行所需的额外时间。最后，我们将学习内存对齐和缓存访问。
