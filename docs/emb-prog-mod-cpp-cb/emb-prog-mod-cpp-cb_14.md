# 第十四章：安全关键系统的指南

嵌入式系统的代码质量要求通常比其他软件领域更高。由于许多嵌入式系统在没有监督或控制的情况下工作，或者控制昂贵的工业设备，错误的成本很高。在安全关键系统中，软件或硬件故障可能导致受伤甚至死亡，错误的成本甚至更高。这种系统的软件必须遵循特定的指南，旨在最大程度地减少在调试和测试阶段未发现错误的机会。

在本章中，我们将通过以下示例探讨安全关键系统的一些要求和最佳实践：

+   使用所有函数的返回值

+   使用静态代码分析器

+   使用前置条件和后置条件

+   探索代码正确性的正式验证

这些示例将帮助您了解安全关键系统的要求和指南，以及用于认证和一致性测试的工具和方法。

# 使用所有函数的返回值

C 语言和 C++语言都不要求开发人员使用任何函数的返回值。完全可以定义一个返回整数的函数，然后在代码中调用它，忽略其返回值。

这种灵活性经常导致软件错误，可能难以诊断和修复。最常见的情况是函数返回错误代码。开发人员可能会忘记为经常使用且很少失败的函数添加错误条件检查，比如`close`。

对于安全关键系统，最广泛使用的编码标准之一是 MISRA。它分别为 C 和 C++语言定义了要求——MISRA C 和 MISRA C++。最近引入的自适应 AUTOSAR 为汽车行业定义了编码指南。预计自适应 AUTOSAR 指南将作为更新后的 MISRA C++指南的基础。

MISRA 和 AUTOSAR 的 C++编码指南（[`www.autosar.org/fileadmin/user_upload/standards/adaptive/17-03/AUTOSAR_RS_CPP14Guidelines.pdf`](https://www.autosar.org/fileadmin/user_upload/standards/adaptive/17-03/AUTOSAR_RS_CPP14Guidelines.pdf)）要求开发人员使用所有非 void 函数和方法的返回值。相应的规则定义如下：

"规则 A0-1-2（必需，实现，自动化）：具有非 void 返回类型的函数返回值应该被使用。"

在这个示例中，我们将学习如何在我们的代码中使用这个规则。

# 如何做...

我们将创建两个类，它们在文件中保存两个时间戳。一个时间戳表示实例创建的时间，另一个表示实例销毁的时间。这对于代码性能分析很有用，可以测量我们在函数或其他感兴趣的代码块中花费了多少时间。按照以下步骤进行：

1.  在您的工作目录中，即`~/test`，创建一个名为`returns`的子目录。

1.  使用您喜欢的文本编辑器在`returns`子目录中创建一个名为`returns.cpp`的文件。

1.  在`returns.cpp`文件中添加第一个类：

```cpp
#include <system_error>

#include <unistd.h>
#include <sys/fcntl.h>
#include <time.h>

[[nodiscard]] ssize_t Write(int fd, const void* buffer,
                            ssize_t size) {
  return ::write(fd, buffer, size);
}

class TimeSaver1 {
  int fd;

public:
  TimeSaver1(const char* name) {
    int fd = open(name, O_RDWR|O_CREAT|O_TRUNC, 0600);
    if (fd < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to open file");
    }
    Update();
  }

  ~TimeSaver1() {
    Update();
    close(fd);
  }

private:
  void Update() {
    time_t tm;
    time(&tm);
    Write(fd, &tm, sizeof(tm));
  }
};
```

1.  接下来，我们添加第二个类：

```cpp
class TimeSaver2 {
  int fd;

public:
  TimeSaver2(const char* name) {
    fd = open(name, O_RDWR|O_CREAT|O_TRUNC, 0600);
    if (fd < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to open file");
    }
    Update();
  }

  ~TimeSaver2() {
    Update();
    if (close(fd) < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to close file");
    }
  }

private:
  void Update() {
    time_t tm = time(&tm);
    int rv = Write(fd, &tm, sizeof(tm));
    if (rv < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to write to file");
    }
  }
};
```

1.  `main`函数创建了两个类的实例：

```cpp
int main() {
  TimeSaver1 ts1("timestamp1.bin");
  TimeSaver2 ts2("timestamp2.bin");
  return 0;
}
```

1.  最后，我们创建一个`CMakeLists.txt`文件，其中包含程序的构建规则：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(returns)
add_executable(returns returns.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++17")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在可以构建和运行应用程序了。

# 它是如何工作的...

我们现在创建了两个类，`TimeSaver1`和`TimeSaver2`，它们看起来几乎相同，并且执行相同的工作。这两个类都在它们的构造函数中打开一个文件，并调用`Update`函数，该函数将时间戳写入打开的文件。

同样，它们的析构函数调用相同的`Update`函数来添加第二个时间戳并关闭文件描述符。

然而，`TimeSaver1`违反了*A0-1-2*规则，是不安全的。让我们仔细看看这一点。它的`Update`函数调用了两个函数，`time`和`write`。这两个函数可能失败，返回适当的错误代码，但我们的实现忽略了它：

```cpp
    time(&tm);
    Write(fd, &tm, sizeof(tm));
```

此外，`TimeSaver1`的析构函数通过调用`close`函数关闭打开的文件。这也可能失败，返回错误代码，我们忽略了它：

```cpp
    close(fd);
```

第二个类`TimeSaver2`符合要求。我们将时间调用的结果分配给`tm`变量：

```cpp
    time_t tm = time(&tm);
```

如果`Write`返回错误，我们会抛出异常：

```cpp
    int rv = Write(fd, &tm, sizeof(tm));
    if (rv < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to write to file");
    }
```

同样，如果`close`返回错误，我们会抛出异常：

```cpp
    if (close(fd) < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to close file");
    }
```

为了减轻这种问题，C++17 标准引入了一个特殊的属性称为`[[nodiscard]]`。如果一个函数声明了这个属性，或者它返回一个标记为`nodiscard`的类或枚举，那么如果其返回值被丢弃，编译器应该显示警告。为了使用这个特性，我们创建了一个围绕`write`函数的自定义包装器，并声明它为`nodiscard`：

```cpp
[[nodiscard]] ssize_t Write(int fd, const void* buffer,
                            ssize_t size) {
  return ::write(fd, buffer, size);
}
```

当我们构建应用程序时，我们可以在编译器输出中看到这一点，这也意味着我们有机会修复它：

![](img/0d3ff757-ae38-48be-b05c-de4b55b2ed2c.png)

事实上，编译器能够识别并报告我们代码中的另一个问题，我们将在下一个示例中讨论。

如果我们构建并运行应用程序，我们不会看到任何输出，因为所有写入都会写入文件。我们可以运行`ls`命令来检查程序是否产生结果，如下所示：

```cpp
$ ls timestamp*
```

从中，我们得到以下输出：

![](img/bad36a2a-3f7f-40a6-855b-345fba095e31.png)

如预期的那样，我们的程序创建了两个文件。它们应该是相同的，但实际上并不是。由`TimeSaver1`创建的文件是空的，这意味着它的实现存在问题。

由`TimeSaver2`生成的文件是有效的，但这是否意味着其实现是 100％正确的？未必，正如我们将在下一个示例中看到的那样。

# 还有更多...

有关`[[nodiscard]]`属性的更多信息可以在其参考页面上找到（[`en.cppreference.com/w/cpp/language/attributes/nodiscard`](https://en.cppreference.com/w/cpp/language/attributes/nodiscard)）。从 C++20 开始，`nodiscard`属性可以包括一个字符串文字，解释为什么不应丢弃该值；例如，`[[nodiscard("检查写入错误")]]`。

重要的是要理解，遵守安全准则确实可以使您的代码更安全，但并不保证它。在我们的`TimeSaver2`实现中，我们使用`time`返回的值，但我们没有检查它是否有效。相反，我们无条件地写入输出文件。同样，如果`write`返回非零数字，它仍然可以向文件写入比请求的数据少。即使您的代码形式上符合指南，它可能仍然存在相关问题。

# 使用静态代码分析器

所有安全准则都被定义为源代码或应用程序设计的具体要求的广泛集合。许多这些要求可以通过使用静态代码分析器自动检查。

**静态代码分析器**是一种可以分析源代码并在检测到违反代码质量要求的代码模式时警告开发人员的工具。在错误检测和预防方面，它们非常有效。由于它们可以在代码构建之前运行，因此很多错误都可以在开发的最早阶段修复，而不需要耗时的测试和调试过程。

除了错误检测和预防，静态代码分析器还用于证明代码在认证过程中符合目标要求和指南。

在这个示例中，我们将学习如何在我们的应用程序中使用静态代码分析器。

# 如何做...

我们将创建一个简单的程序，并运行其中一个许多可用的开源代码分析器，以检查潜在问题。按照以下步骤进行：

1.  转到我们之前创建的`~/test/returns`目录。

1.  从存储库安装`cppcheck`工具。确保您处于`root`帐户下，而不是`user`：

```cpp
# apt-get install cppcheck
```

1.  再次切换到`user`帐户：

```cpp
# su - user
$
```

1.  对`returns.cpp`文件运行`cppcheck`：

```cpp
$ cppcheck --std=posix --enable=warning returns.cpp
```

1.  分析它的输出。

# 它是如何工作的...

代码分析器可以解析我们应用程序的源代码，并根据多种代表不良编码实践的模式进行测试。

存在许多代码分析器，从开源和免费到昂贵的企业级商业产品。

在*使用所有函数的返回值*示例中提到的**MISRA**编码标准是商业标准。这意味着您需要购买许可证才能使用它，并且需要购买一个经过认证的代码分析器，以便测试代码是否符合 MISRA 标准。

出于学习目的，我们将使用一个名为`cppcheck`的开源代码分析器。它被广泛使用，并已经包含在 Ubuntu 存储库中。我们可以像安装其他 Ubuntu 软件包一样安装它：

```cpp
# apt-get install cppcheck $ cppcheck --std=posix --enable=warning returns.cpp
```

现在，我们将源文件名作为参数传递。检查很快，生成以下报告：

！[](img/659c3b78-ca64-474f-8917-0345f48808e4.png)

正如我们所看到的，它在我们的代码中检测到了两个问题，甚至在我们尝试构建之前。第一个问题出现在我们更安全、增强的`TimeSaver2`类中！为了使其符合 A0-1-2 要求，我们需要检查`close`返回的状态代码，并在发生错误时抛出异常。然而，我们在析构函数中执行此操作，违反了 C++错误处理机制。

代码分析器检测到的第二个问题是资源泄漏。这解释了为什么`TimeSaver1`会生成空文件。当打开文件时，我们意外地将文件描述符分配给局部变量，而不是实例变量，即`fd`：

```cpp
int fd = open(name, O_RDWR|O_CREAT|O_TRUNC, 0600);
```

现在，我们可以修复它们并重新运行`cppcheck`，以确保问题已经消失，并且没有引入新问题。在开发工作流程中使用代码分析器可以使您的代码更安全，性能更快，因为您可以在开发周期的早期阶段检测和预防问题。

# 还有更多...

尽管`cppcheck`是一个开源工具，但它支持多种 MISRA 检查。这并不意味着它是一个用于验证符合 MISRA 指南的认证工具，但它可以让您了解您的代码与 MISRA 要求的接近程度，以及可能需要多少努力使其符合要求。

MISRA 检查是作为一个附加组件实现的；您可以根据`cppcheck`的 GitHub 存储库的附加组件部分中的说明来运行它（[`github.com/danmar/cppcheck/tree/master/addons`](https://github.com/danmar/cppcheck/tree/master/addons)）。

# 使用前置条件和后置条件

在上一个示例中，我们学习了如何使用静态代码分析器来防止在开发的早期阶段出现编码错误。另一个防止错误的强大工具是**按合同编程**。

按合同编程是一种实践，开发人员在其中明确定义函数或模块的输入值、结果和中间状态的合同或期望。虽然中间状态取决于实现，但输入和输出值的合同可以作为公共接口的一部分进行定义。这些期望分别称为**前置条件**和**后置条件**，有助于避免由模糊定义的接口引起的编程错误。

在这个示例中，我们将学习如何在我们的 C++代码中定义前置条件和后置条件。

# 如何做...

为了测试前置条件和后置条件的工作原理，我们将部分重用我们在上一个示例中使用的**`TimeSaver1`**类的代码。按照以下步骤进行：

1.  在您的工作目录中，即`〜/test`，创建一个名为`assert`的子目录。

1.  使用您喜欢的文本编辑器在`assert`子目录中创建一个名为`assert.cpp`的文件。

1.  将`TimeSaver1`类的修改版本添加到`assert.cpp`文件中：

```cpp
#include <cassert>
#include <system_error>

#include <unistd.h>
#include <sys/fcntl.h>
#include <time.h>

class TimeSaver1 {
  int fd = -1;

public:
  TimeSaver1(const char* name) {
    assert(name != nullptr);
    assert(name[0] != '\0');

    int fd = open(name, O_RDWR|O_CREAT|O_TRUNC, 0600);
    if (fd < 0) {
      throw std::system_error(errno,
                              std::system_category(),
                              "Failed to open file");
    }
    assert(this->fd >= 0);
  }

  ~TimeSaver1() {
    assert(this->fd >= 0);
    close(fd);
  }
};
```

1.  接下来是一个简单的`main`函数：

```cpp
int main() {
  TimeSaver1 ts1("");
  return 0;
}
```

1.  将构建规则放入`CMakeLists.txt`文件中：

```cpp
cmake_minimum_required(VERSION 3.5.1)
project(assert)
add_executable(assert assert.cpp)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

SET(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++)
```

1.  现在您可以构建和运行应用程序。

# 它是如何工作的...

在这里，我们重用了上一个示例中`TimeSaver1`类的一些代码。为简单起见，我们删除了`Update`方法，只留下了它的构造函数和析构函数。

我们故意保留了在上一个示例中由静态代码分析器发现的相同错误，以检查前置条件和后置条件检查是否可以防止这类问题。

我们的构造函数接受一个文件名作为参数。对于文件名，我们没有特定的限制，除了它应该是有效的。两个明显无效的文件名如下：

+   一个空指针作为名称

+   一个空的名称

我们将这些规则作为前置条件使用`assert`宏：

```cpp
assert(name != nullptr);
assert(name[0] != '\0');
```

要使用这个宏，我们需要包含一个头文件，即`csassert`：

```cpp
#include <cassert>
```

接下来，我们使用文件名打开文件并将其存储在`fd`变量中。我们将其分配给局部变量`fd`，而不是实例变量`fd`。这是我们想要检测到的一个编码错误：

```cpp
int fd = open(name, O_RDWR|O_CREAT|O_TRUNC, 0600);
```

最后，我们在构造函数中放置后置条件。在我们的情况下，唯一的后置条件是实例变量`fd`应该是有效的：

```cpp
assert(this->fd >= 0);
```

注意我们用 this 作为前缀以消除它与局部变量的歧义。同样，我们在析构函数中添加了一个前置条件：

```cpp
assert(this->fd >= 0);
```

在这里我们不添加任何后置条件，因为在析构函数返回后，实例就不再有效了。

现在，让我们测试我们的代码。在`main`函数中，我们创建了一个`TimeSaver1`的实例，将一个空的文件名作为参数传递：

```cpp
TimeSaver1 ts1("");
```

在构建和运行程序之后，我们将看到以下输出：

![](img/bcd001f1-c8c9-4e3f-bd48-a4dbc27177be.png)

构造函数中的前置条件检查已经检测到了合同的违反并终止了应用程序。让我们将文件名更改为有效的文件名：

```cpp
TimeSaver1 ts1("timestamp.bin");
```

我们再次构建和运行应用程序，得到了不同的输出：

![](img/2a162765-e45c-4207-a02c-fe63f35de7c1.png)

现在，所有的前置条件都已经满足，但我们违反了后置条件，因为我们没有更新实例变量`fd`。在第 16 行删除`fd`前的类型定义，如下所示：

```cpp
fd = open(name, O_RDWR|O_CREAT|O_TRUNC, 0600);
```

重新构建并再次运行程序会产生空输出：

![](img/8a57bd09-8c9e-4004-91b6-1a39c806c0e2.png)

这表明输入参数和结果的所有期望都已经满足。即使以基本形式，使用合同编程也帮助我们防止了两个编码问题。这就是为什么这种技术在软件开发的所有领域以及特别是在安全关键系统中被广泛使用的原因。

# 还有更多...

对于 C++20 标准，预计会添加更详细的合同编程支持。然而，它已经推迟到了以后的标准。提案的描述可以在论文*A Contract Design* ([`www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0380r1.pdf`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0380r1.pdf))中找到，作者是 G. Dos Reis, J. D. Garcia, J. Lakos, A. Meredith, N. Myers, B. Stroustrup。

# 探索代码正确性的形式验证

静态代码分析器和合同编程方法有助于开发人员显著减少其代码中的编码错误数量。然而，在安全关键软件开发中，这还不够。重要的是正式证明软件组件的设计是正确的。

有一些相当复杂的方法来做到这一点，还有一些工具可以自动化这个过程。在这个示例中，我们将探索一种名为 CPAchecker 的正式软件验证工具之一 ([`cpachecker.sosy-lab.org/index.php`](https://cpachecker.sosy-lab.org/index.php))。

# 如何做...

我们将下载并安装`CPAcheck`到我们的构建环境中，然后对一个示例程序运行它。按照以下步骤进行：

1.  用包括您的构建环境在内的终端打开。

1.  确保您有 root 权限。如果没有，按*Ctrl* + *D*退出*user*会话返回到*root*会话。

1.  安装 Java 运行时：

```cpp
# apt-get install openjdk-11-jre
```

1.  切换到用户会话并切换到`/mnt`目录：

```cpp
# su - user
$ cd /mnt
```

1.  下载并解压`CPACheck`存档，如下所示：

```cpp
$ wget -O - https://cpachecker.sosy-lab.org/CPAchecker-1.9-unix.tar.bz2 | tar xjf -
```

1.  切换到`CPAchecker-1.9-unix`目录：

```cpp
$ cd CPAchecker-1.9-unix
```

1.  对示例文件运行`CPAcheck`：

```cpp
./scripts/cpa.sh -default doc/examples/example.c 
```

1.  下载故意包含错误的示例文件：

```cpp
$ wget https://raw.githubusercontent.com/sosy-lab/cpachecker/trunk/doc/examples/example_bug.c
```

1.  对新示例运行检查器：

```cpp
./scripts/cpa.sh -default example_bug.c 
```

1.  切换到您的网络浏览器并打开由工具生成的`~/test/CPAchecker-1.9-unix/output/Report.html`报告文件。

# 它是如何工作的...

要运行`CPAcheck`，我们需要安装 Java 运行时。这在 Ubuntu 存储库中可用，我们使用`apt-get`来安装它。

下一步是下载`CPAcheck`本身。我们使用`wget`工具下载存档文件，并立即将其提供给`tar`实用程序进行提取。完成后，可以在`CPAchecker-1.9-unix`目录中找到该工具。

我们使用预打包的示例文件之一来检查工具的工作方式：

```cpp
./scripts/cpa.sh -default doc/examples/example.c
```

它生成了以下输出：

![](img/ff8fcef6-80fd-45a3-9eed-4785e0e00f6b.png)

我们可以看到，该工具没有发现这个文件中的任何问题。在`CPAcheck`存档中没有包含错误的类似文件，但我们可以从其网站上下载：

```cpp
$ wget https://raw.githubusercontent.com/sosy-lab/cpachecker/trunk/doc/examples/example_bug.c
```

我们再次运行该工具并获得以下输出：

![](img/ab5c77a6-4eb2-4ef5-8dea-d5ad44974a53.png)

现在，结果不同了：检测到了一个错误。我们可以打开工具生成的 HTML 报告进行进一步分析。除了日志和统计信息外，它还显示了流自动化图：

![](img/9fdfd67a-296b-404e-a6e3-ee5065fd6216.png)

正式验证方法和工具是复杂的，可以处理相对简单的应用程序，但它们保证了所有情况下应用程序逻辑的正确性。

# 还有更多...

您可以在其网站上找到有关 CPAchecker 的更多信息（[`cpachecker.sosy-lab.org/index.php`](https://cpachecker.sosy-lab.org/index.php)）。
