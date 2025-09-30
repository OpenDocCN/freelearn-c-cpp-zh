# 附录 2

# 构建速度优化

Clang 实现了多项功能，旨在提高大型项目的构建速度。其中最有趣的功能之一是预编译头文件和模块。它们可以被视为允许缓存 AST 的某些部分并重新用于不同编译调用的技术。缓存可以显著提高项目的构建速度，并且可以使用这些功能来加速不同的 Clang 工具执行。例如，预编译头文件被用作 Clangd 文档编辑的主要优化。

在本附录中，我们将涵盖两个主要主题

+   预编译头文件

+   模块

## 10.1 技术要求

本附录的源代码位于本书 GitHub 仓库的 `chapter10` 文件夹中：[`github.com/PacktPublishing/Clang-Compiler-Frontend-Packt/tree/main/chapter10`](https://github.com/PacktPublishing/Clang-Compiler-Frontend-Packt/tree/main/chapter10)。

## 10.2 预编译头文件

**预编译头文件 PCH** 是 Clang 的一项功能，旨在提高 Clang 前端性能。基本思路是为头文件创建一个 AST（抽象语法树），并在编译过程中重用此 AST，用于包含头文件的源文件。

生成预编译头文件很简单 [5]。假设您有以下头文件，`header.h`：

```cpp
1 #pragma once 

2  

3 void foo() { 

4 }
```

**图 10.1**：要编译为 PCH 的头文件

您可以使用以下命令为其生成 PCH：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1 -emit-pch        \
                                        -x c++-header header.h \
                                        -o header.pch
```

这里，我们使用 `-x c++-header` 选项指定头文件应被视为 C++ 头文件。输出文件将被命名为 `header.pch`。

仅生成预编译头文件是不够的；您需要开始使用它们。一个典型的 C++ 源文件，包含头文件可能看起来像这样：

```cpp
1 #include "header.h" 

2  

3 int main() { 

4   foo(); 

5   return 0; 

6 }
```

**图 10.2**：包含 header.h 的源文件

如您所见，头文件包含如下所示：

```cpp
1 #include "header.h"
```

**图 10.3**：包含 header.h

默认情况下，Clang 不会使用 PCH，您必须使用以下命令显式指定：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1 -emit-obj        \
                                       -include-pch header.pch \
                                       main.cpp -o main.o
```

这里，我们使用 `-include-pch` 指定包含的预编译头文件：`header.pch`。

您可以使用调试器检查此命令，并将给出以下输出：

```cpp
1$ lldb <...>/llvm-project/install/bin/clang -- -cc1 -emit-obj -include-pch header.pch main.cpp -o main.o 

2 ... 

3 (lldb) b clang::ASTReader::ReadAST 

4 ... 

5 (lldb) r 

6 ... 

7 -> 4431   llvm::TimeTraceScope scope("ReadAST", FileName); 

8    4432 

9    4433   llvm::SaveAndRestore SetCurImportLocRAII(CurrentImportLoc, ImportLoc); 

10    4434   llvm::SaveAndRestore<std::optional<ModuleKind>> SetCurModuleKindRAII( 

11 (lldb) p FileName 

12 (llvm::StringRef)  (Data = "header.pch", Length = 10)
```

**图 10.4**：在 clang::ASTReader::ReadAST 中加载预编译头文件

从这个例子中，您可以看到 Clang 从预编译头文件中读取 AST。需要注意的是，预编译头文件是在解析之前读取的，这使得 Clang 在解析主源文件之前能够获取头文件中的所有符号。这使得显式包含头文件变得不必要。因此，您可以从源文件中删除 `#include "header.h"` 指令并成功编译。

没有预编译头文件，您将遇到以下编译错误：

```cpp
main.cpp:4:3: error: use of undeclared identifier ’foo’
    4 |   foo();
      |   ^
1  error generated.
```

**图 10.5**：由于缺少包含而引发的编译错误

值得注意的是，只有第一个`--include-pch`选项将被处理；所有其他选项将被忽略。这反映了翻译单元只能有一个预编译头文件的事实。另一方面，一个预编译头文件可以包含另一个预编译头文件。这种功能被称为链式预编译头文件[3]，因为它创建了一个依赖链，其中一个预编译头文件依赖于另一个预编译头文件。

预编译头文件的使用不仅限于常规编译。正如我们在*图** 8.38**中看到的那样，Clangd 中的 AST 构建，预编译头文件在 Clangd 中作为包含头文件的序言缓存占位符，被积极用于性能优化。

预编译头文件是一种长期使用的技术，但它有一些限制。其中最重要的限制是只能有一个预编译头文件，这显著限制了 PCH 在实际项目中的使用。模块解决了与预编译头文件相关的一些问题。让我们来探讨这些问题。

## 10.3 Clang 模块

模块，或称为**预编译模块**（**PCMs**），可以被认为是预编译头文件演化的下一步。它们也代表了一种以二进制形式解析的抽象语法树（AST），但形成了一个有向无环图（DAG，树），这意味着一个模块可以包含多个其他模块。

与只能为每个编译单元引入一个预编译头文件的预编译头文件相比，这是一个重大的改进。

C++20 标准[21]引入了与模块相关的两个概念。第一个是普通模块，在[21]的第*10 节*中描述。另一个是所谓的头单元，主要在第*15.5 节*中描述。头单元可以被认为是普通头文件和模块之间的一个中间步骤，并允许使用`import`指令来导入普通头文件。

我们将关注 Clang 模块，这可以被认为是 C++标准中头单元的实现。使用 Clang 模块有两种不同的选项。第一个被称为**显式模块**。第二个被称为**隐式模块**。我们将探讨这两种情况，但将从我们想要使用模块的测试项目的描述开始。

#### 测试项目描述

对于模块的实验，我们将考虑一个包含两个头文件`header1.h`和`header2.h`的例子，分别定义了`void foo1()`和`void foo2()`函数，如下所示：

```cpp
1 #pragma once 

2  

3 void foo1() {}
```

**头文件：header1.h**

**```cpp
1 #pragma once 

2  

3 void foo2() {}
```

**头文件：header2.h**

****图 10.6**：用于测试的头文件

这些头文件将在以下源文件中使用：

```cpp
1 #include "header1.h" 

2 #include "header2.h" 

3  

4 int main() { 

5   foo1(); 

6   foo2(); 

7   return 0; 

8 }
```

**图 10.7**：源文件：main.cpp

我们将把我们的头文件组织成模块。Clang 使用一个包含逻辑结构的特殊文件，称为**模块映射文件**。让我们看看我们的测试项目中的文件看起来像什么。

#### 模块映射文件

我们项目的模块映射文件将被命名为`module.modulemap`，其内容如下：

```cpp
1 module header1 { 

2   header "header1.h" 

3   export * 

4 } 

5 module header2 { 

6   header "header2.h" 

7   export * 

8 }
```

**图 10.8**：模块映射文件：module.modulemap

如图 10.8 所示，我们定义了两个模块，**header1**和**header2**。

每个模块只包含一个头文件，并导出其所有符号。

现在我们已经收集了所有必要的部分，我们准备构建和使用模块。模块可以是显式构建或隐式构建。让我们从显式构建开始。

#### 显式模块

模块的结构由模块映射文件描述，如图 10.8 所示。我们每个模块只有一个头文件，但一个真实的模块可能包含多个头文件。因此，为了构建一个模块，我们必须指定模块的结构（模块映射文件）和我们想要构建的模块名称。例如，对于**header1**模块，我们可以使用以下构建命令：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1            \
        -emit-module -o header1.pcm                    \
        -fmodules module.modulemap -fmodule-name=header1 \
        -x c++-header -fno-implicit-modules
```

编译命令中有几个重要方面。第一个是**-cc1**选项，它表示我们只调用编译器前端。有关更多信息，请参阅*第 2.3 节*“Clang 驱动程序概述”。此外，我们指定要创建一个名为`header1.pcm`的构建工件（模块），使用以下选项：`-emit-module -o header1.pcm`。逻辑结构和要构建的所需模块在`module.modulemap`文件中指定，该文件必须使用`-fmodule-name=header1`选项作为编译参数指定。启用模块功能是通过使用`-fmodules`标志完成的，我们还使用`-x c++-header`选项指定我们的头文件是 C++头文件。为了显式禁用隐式模块，我们在命令中包含`-fno-implicit-modules`，因为隐式模块默认启用，但我们目前不想使用它们。

第二个模块（`header2`）有类似的编译命令：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1            \
        -emit-module -o header2.pcm                    \
        -fmodules module.modulemap -fmodule-name=header2 \
        -x c++-header -fno-implicit-modules
```

下一步是使用生成的模块编译`main.cpp`，可以按照以下方式完成：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1       \
       -emit-obj main.cpp                         \
       -fmodules -fmodule-map-file=module.modulemap \
       -fmodule-file=header1=header1.pcm          \
       -fmodule-file=header2=header2.pcm          \
       -o main.o -fno-implicit-modules
```

如我们所见，模块名称和构建工件（PCM 文件）都是通过使用`-fmodule-file`编译选项指定的。使用的格式，例如`header1=header1.pcm`，表示`header1.pcm`对应于`header1`模块。我们还使用`-fmodule-map-file`选项指定模块映射文件。值得注意的是，我们创建了两个构建工件：`header1.pcm`和`header2.pcm`，并将它们一起用于编译。这与预编译头文件的情况不同，因为如*第 10.2 节*所述，只允许一个预编译头文件，我们将在*图 10.9*中稍后研究隐式模块。

我们通过编译命令生成了一个目标文件`main.o`。该目标文件可以链接如下：

```cpp
$ <...>/llvm-project/install/bin/clang main.o -o main -lstdc++
```

让我们验证模块在编译期间是否被加载。这可以通过 LLDB 完成，如下所示：

```cpp
1$ lldb <...>/llvm-project/install/bin/clang -- -cc1 -emit-obj main.cpp -fmodules -fmodule-map-file=module.modulemap -fmodule-file=header1=header1.pcm -fmodule-file=header2=header2.pcm -o main.o -fno-implicit-modules 

2 ... 

3 (lldb) b clang::CompilerInstance::findOrCompileModuleAndReadAST 

4 ... 

5 (lldb) r 

6 ... 

7 Process 135446 stopped 

8 * thread #1, name = ’clang’, stop reason = breakpoint 1.1 

9     frame #0: ... findOrCompileModuleAndReadAST(..., ModuleName=(Data = "header1", Length = 7), ... 

10 ... 

11 (lldb) c 

12 Process 135446 stopped 

13 * thread #1, name = ’clang’, stop reason = breakpoint 1.1 

14     frame #0: ... findOrCompileModuleAndReadAST(..., ModuleName=(Data = "header2", Length = 7), .... 

15 ... 

16 (lldb) c 

17 Process 135446 resumed 

18 Process 135446 exited with status = 0 (0x00000000)
```

**图 10.9**：显式模块加载

我们在`clang::CompilerInstance::findOrCompileModuleAndReadAST`处设置了一个断点，如图 10.9 中的*第 3 行*所示。我们两次触发了断点：第一次是在名为`header1`的模块的*第 9 行*，然后是在名为`header2`的模块的*第 14 行*。

当使用显式模块时，必须在所有编译命令中明确定义构建工件并指定它们将被存储的路径，正如我们刚刚发现的。然而，所有必需的信息都存储在模块映射文件中（参见图 10.8）。编译器可以利用这些信息自动创建所有必要的构建工件。对于这个问题的答案是肯定的，并且这种功能由隐式模块提供。让我们来探索一下。

#### 隐式模块

如前所述，模块映射文件包含构建所有模块（`header1`和`header2`）以及用于依赖文件（`main.cpp`）构建所需的所有信息。因此，我们必须指定模块映射文件的路径以及构建工件将存储的文件夹。这可以通过以下方式完成：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1 \
      -emit-obj main.cpp                  \
      -fmodules                           \
      -fmodule-map-file=module.modulemap  \
      -fmodules-cache-path=./cache        \
      -o main.o
```

如我们所见，我们没有指定`-fno-implicit-modules`，并且我们也指定了构建工件路径为`-fmodules-cache-path=./cache`。如果我们检查路径，我们将能够看到创建的模块：

```cpp
$ tree ./cache
./cache
|-- 2AL78TH69W6HR
    |-- header1-R65CPR1VCRM1.pcm
    |-- header2-R65CPR1VCRM1.pcm
    |-- modules.idx
2  directories, 3 files
```

**图 10.10**：Clang 为隐式模块生成的缓存

Clang 将监控缓存文件夹（在我们的例子中是`./cache`），并删除长时间未使用的构建工件。如果它们的依赖项（例如，包含的标题）已更改，它还将重新构建模块。

模块是一个非常强大的工具，但就像每个强大的工具一样，它们可以引入非平凡的问题。让我们来探索由模块可能引起的最有趣的问题。

#### 一些与模块相关的问题

使用模块的代码可能会向你的程序引入一些非平凡的行为。考虑一个由两个标题组成的工程，如下所示：

```cpp
1 #pragma once 

2  

3 int h1 = 1;
```

**标题文件：header1.h**

**[PRE19**]

**标题文件：header2.h**

****图 10.11**：用于测试的标题文件

`main.cpp`中只包含了`header1.h`，如下所示

```cpp
1 #include "header1.h" 

2  

3 int main() { 

4   int h = h1 + h2; 

5   return 0; 

6 }
```

**图 10.12**：源文件：main.cpp

代码将无法编译：

```cpp
$ <...>/llvm-project/install/bin/clang  main.cpp -o main -lstdc++
main.cpp:4:16: error: use of undeclared identifier ’h2’
  int h = h1 + h2;
               ^
1  error generated.
```

**图 10.13**：由于缺少标题文件而引发的编译错误

错误很明显，因为我们没有包含包含`h2`变量定义的第二部分标题。

如果我们使用隐式模块，情况将不同。考虑以下`module.modulemap`文件：

```cpp
1 module h1 { 

2   header "header1.h" 

3   export * 

4   module h2 { 

5     header "header2.h" 

6     export * 

7   } 

8 }
```

**图 10.14**：引入隐式依赖的模块映射文件

此文件创建了两个模块，`h1`和`h2`。第二个模块包含在第一个模块中。

如果我们按照以下方式编译，编译将成功：

```cpp
$ <...>/llvm-project/install/bin/clang -cc1 \
        -emit-obj main.cpp                \
        -fmodules                         \
        -fmodule-map-file=module.modulemap\
        -fmodules-cache-path=./cache      \
        -o main.o
$ <...>/llvm-project/install/bin/clang main.o -o main -lstdc++
```

**图 10.15**: 成功编译一个缺少头文件但启用了隐式模块的文件

编译完成后没有出现任何错误，因为 modulemap 隐式地将`header2.h`添加到使用的模块（`h1`）。我们还使用`export *`指令导出了所有符号。因此，当 Clang 遇到`#include "header1.h"`时，它加载相应的`h1`模块，因此隐式地加载了在`h2`模块和`header2.h`头文件中定义的符号。

该示例说明了当在项目中使用模块时，可见作用域可能会泄露。当项目启用和禁用模块时构建，这可能导致项目构建出现意外的行为。

## 10.4 进一步阅读

+   Clang 模块: [`clang.llvm.org/docs/Modules.html`](https://clang.llvm.org/docs/Modules.html)

+   预编译头文件和模块内部结构: [`clang.llvm.org/docs/PCHInternals.html`](https://clang.llvm.org/docs/PCHInternals.html)
