# 6

# 高级代码分析

如前一章所述，Clang-Tidy 检查依赖于 AST 提供的高级匹配。然而，这种方法可能不足以检测更复杂的问题，例如生命周期问题（即，当对象或资源在已解除分配或超出作用域之后被访问或引用时，可能导致不可预测的行为或崩溃）。在本章中，我们将介绍基于 **控制流图**（**CFG**）的高级代码分析工具。Clang 静态分析器是此类工具的绝佳例子，Clang-Tidy 也集成了 CFG 的某些方面。我们将从典型用法示例开始，然后深入探讨实现细节。本章将以一个使用高级技术并扩展类复杂度概念到方法实现的定制检查结束。我们将定义圈复杂度并展示如何使用 Clang 提供的 CFG 库来计算它。在本章中，我们将探讨以下主题：

+   什么是静态分析

+   了解 CFG – 静态分析中使用的基本数据结构

+   如何在自定义 Clang-Tidy 检查中使用 CFG

+   Clang 提供了哪些分析工具以及它们的局限性

## 6.1 技术要求

本章的源代码位于本书 GitHub 存储库的 `chapter6` 文件夹中：[`github.com/PacktPublishing/Clang-Compiler-Frontend-Packt/tree/main/chapter6`](https://github.com/PacktPublishing/Clang-Compiler-Frontend-Packt/tree/main/chapter6)。

## 6.2 静态分析

静态分析是软件开发中的一种关键技术，它涉及在不实际运行程序的情况下检查代码。这种方法侧重于分析源代码或其编译版本，以检测各种问题，例如错误、漏洞和与编码标准的偏差。与需要执行程序的动态分析不同，静态分析允许在非运行时环境中检查代码。

更一般地说，静态分析旨在根据计算机程序的意义检查其特定的属性；也就是说，它可以被认为是语义分析的一部分（参见*图** 2.6**，解析器*）。例如，如果 𝒞 是所有 C/C++ 程序的集合，而 𝒫 是此类程序的一个属性，那么静态分析的目标是检查特定程序 P ∈𝒞 的属性，即回答 𝒫(P) 是否为真或假的问题。

我们在前一章中提到的 Clang-Tidy 检查（参见*节** 5.4**，自定义* *Clang-Tidy 检查*）是此类属性的一个很好的例子。实际上，它接受具有类定义的 C++ 代码，并根据方法数量决定该类是否复杂。

值得注意的是，并非所有程序的性质都可以进行检查。最明显的例子是著名的停机问题 [31]。

重要提示

停机问题可以表述如下：给定一个程序 P 和一个输入 I，确定当 P 在 I 上执行时，P 是停止运行还是无限期地继续运行。

形式上，问题是要决定，对于给定的程序 P 和输入 I，P(I)的计算最终是否会停止（停机）或永远不会终止（无限循环）。

阿兰·图灵证明了不存在一种通用的算法方法可以解决所有可能的程序-输入对的问题。这个结果意味着没有一种单一的算法可以正确地确定对于每一对（P，I），当 P 在 I 上运行时，P 是否会停止。

尽管并非所有程序的性质都能被证明，但在某些情况下是可以做到的。有相当数量的这种案例使得静态分析成为一个实用的工具。因此，我们可以使用这些工具在这些情况下系统地扫描代码，以确定代码的性质。这些工具擅长识别从简单的语法错误到更复杂的潜在错误的各种问题。静态分析的一个关键优势是它能够在开发周期的早期阶段捕捉到问题。这种早期检测不仅效率高，而且节省资源，因为它有助于在软件运行或部署之前识别和纠正问题。

静态分析在确保软件质量和合规性方面发挥着重要作用。它检查代码是否遵循规定的编码标准和指南，这在大型项目或对监管要求严格的行业中尤为重要。此外，它在揭示常见的安全漏洞方面非常有效，例如缓冲区溢出、SQL 注入漏洞和跨站脚本漏洞。

此外，静态分析通过确定冗余区域、不必要的复杂性和改进机会，有助于代码重构和优化。将此类工具集成到开发过程中，包括持续集成管道，是一种常见做法。这种集成允许对代码进行持续分析，每次提交或构建时都会进行，从而确保持续的质量保证。

我们在上章中创建的 Clang-Tidy 检查可以被视为静态分析程序的一个例子。在本章中，我们将考虑涉及数据结构（如 CFG）的更高级主题，我们将在下一节中看到。

## 6.3 CFG

**CFG**是编译设计和静态程序分析中的一个基本数据结构，它表示程序在执行过程中可能遍历的所有路径。

一个 CFG 由以下关键组件组成：

+   **节点**：对应于基本块，一个具有一个入口点和一个出口点的操作直线序列

+   **边**：表示从一个块到另一个块的控件流，包括条件和无条件分支

+   **起始和结束节点**：每个 CFG 都有一个唯一的入口节点和一个或多个出口节点

作为 CFG 的一个示例，考虑我们之前用作示例的两个整数最大值的函数；参见图 2.5：

```cpp
1 int max(int a, int b) { 

2   if (a > b) 

3     return a; 

4   return b; 

5 }
```

**图 6.1**: max.cpp 的 CFG 示例 C++代码

相应的 CFG 可以表示如下：

![图 6.2: max.cpp 的 CFG 示例](img/Figure6.2_B19722.png)

**图 6.2**: max.cpp 的 CFG 示例

如图 6.2 所示，该图直观地表示了`max`函数的 CFG（来自图 6.1），通过一系列连接的节点和有向边：

+   **入口节点**：在顶部，有一个“**entry**”节点，表示函数执行的起点。

+   **条件节点**：在入口节点下方，有一个标记为“**a** **> b**”的节点。此节点表示函数中的条件语句，其中比较*a*和*b*。

+   **真和假条件分支**：

    +   在真分支（左侧），有一个标记为“**返回** **a**”的节点，通过从“**a > b**”节点的边连接。这条边标记为“**true**”，表示如果*a*大于*b*，则流程流向此节点。

    +   在假分支（右侧），有一个标记为“**返回** **b**”的节点，通过从“**a > b**”节点的边连接。这条边标记为“**false**”，表示如果*a*不大于*b*，则流程流向此节点。

+   **出口节点**：在“**Return a**”和“**Return b**”节点下方，汇聚于一点，有一个“**exit**”节点。这表示函数的终止点，在返回*a*或*b*后，控制流退出函数。

此 CFG 有效地说明了`max`函数如何处理输入并基于比较决定返回哪个值。

CFG 表示也可以用来估计函数的复杂度。简而言之，更复杂的图像对应更复杂的系统。我们将使用一个称为循环复杂度的精确复杂度定义，或 M [28]，其计算方法如下：

| *M = E - N + 2P* |  |
| --- | --- |

其中：

+   E 是图中边的数量

+   N 是图中节点的数量

+   P 是连通分量的数量（对于单个 CFG，P 通常为 1）

对于前面讨论的`max`函数，CFG 可以分析如下：

+   **节点 (N)**: 有五个节点（入口，*a > b*，返回*a*，*b*，出口）

+   **边 (E)**: 有五条边（从入口到*a > b*，从*a > b*到返回*a*，从*a > b*到返回*b*，从返回*a*到出口，以及从返回*b*到出口）

+   **连通分量 (P)**: 由于它是一个单一函数，*P* = 1

将这些值代入公式，我们得到以下结果：

𝑀 = 5 − 5 + 2 × 1 = 2

因此，基于给定的 CFG，`max`函数的循环复杂度为 2。这表明代码中有两条线性独立的路径，对应于 if 语句的两个分支。

我们的下一步将是创建一个使用 CFG 来计算循环复杂度的 Clang-Tidy 检查。

## 6.4 自定义 CFG 检查

我们将使用在*第 5.4 节***中获得的关于自定义 Clang-Tidy 检查的知识来创建一个自定义 CFG 检查。如前所述，该检查将使用 Clang 的 CFG 来计算循环复杂度。如果计算出的复杂度超过阈值，则检查应发出警告。此阈值将作为配置参数设置，允许我们在测试期间更改它。让我们从创建项目骨架开始。

### 6.4.1 创建项目骨架

我们将使用`cyclomaticcomplexity`作为检查的名称，我们的项目骨架可以创建如下：

```cpp
$ ./clang-tools-extra/clang-tidy/add_new_check.py misc cyclomaticcomplexity
```

**图 6.3**：为 misc-cyclomaticcomplexity 检查创建骨架

运行结果将生成多个修改后的新文件。对我们来说，最重要的是位于`clang-tools-extra/clang-tidy/misc/`文件夹中的以下两个文件：

+   `misc/CyclomaticcomplexityCheck.h`：这是我们的检查的头文件

+   `misc/CyclomaticcomplexityCheck.cpp`：此文件将包含我们的检查实现

这些文件需要修改以达到检查所需的函数。

### 6.4.2 检查实现

对于头文件，我们旨在添加一个用于计算循环复杂度的私有函数。具体来说，需要插入以下代码：

```cpp
27 private: 

28   unsigned calculateCyclomaticComplexity(const CFG *cfg);
```

**图 6.4**：对 CyclomaticcomplexityCheck.h 的修改

在`.cpp`文件中需要更多的实质性修改。我们将从`registerMatchers`方法的实现开始，如下所示：

```cpp
17 void CyclomaticcomplexityCheck::registerMatchers(MatchFinder *Finder) { 

18   Finder->addMatcher(functionDecl().bind("func"), this); 

19 }
```

**图 6.5**：对 CyclomaticcomplexityCheck.cpp 的修改：registerMatchers 实现

根据代码，我们的检查将仅应用于函数声明，即`clang::FunctionDecl`。代码也可以扩展以支持其他 C++结构。

`check`方法的实现如图 6.6 所示。在*第 22-23 行*，我们对匹配的 AST 节点进行基本检查，在我们的例子中是`clang::FunctionDecl`。在*第 25-26 行*，我们使用`CFG::buildCFG`方法创建 CFG 对象。前两个参数指定了声明（`clang::Decl`）和声明的语句（`clang::Stmt`）。在*第 30 行*，我们使用阈值计算循环复杂度，该阈值可以作为我们检查的`"Threshold"`选项获得。这为测试不同的输入程序提供了灵活性。*第 31-34 行*包含了检查结果打印的实现。

```cpp
21 void CyclomaticcomplexityCheck::check(const MatchFinder::MatchResult &Result) { 

22   const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func"); 

23   if (!Func || !Func->hasBody()) return; 

24  

25   std::unique_ptr<CFG> cfg = 

26       CFG::buildCFG(Func, Func->getBody(), Result.Context, CFG::BuildOptions()); 

27   if (!cfg) return; 

28  

29   unsigned Threshold = Options.get("Threshold", 5); 

30   unsigned complexity = calculateCyclomaticComplexity(cfg.get()); 

31   if (complexity > Threshold) { 

32     diag(Func->getLocation(), "function %0 has high cyclomatic complexity (%1)") 

33         << Func << complexity; 

34   } 

35 }
```

**图 6.6**：对 CyclomaticcomplexityCheck.cpp 的修改：检查实现

`calculateCyclomaticComplexity`方法用于计算循环复杂度。它接受创建的`clang::CFG`对象作为输入参数。实现如下所示：

```cpp
37 unsigned CyclomaticcomplexityCheck::calculateCyclomaticComplexity( 

38     const CFG *cfg) { 

39   unsigned edges = 0; 

40   unsigned nodes = 0; 

41  

42   for (const auto *block : *cfg) { 

43     edges += block->succ_size(); 

44     ++nodes; 

45   } 

46  

47   return edges - nodes + 2;  // Simplified formula 

48 }
```

**图 6.7**：对 CyclomaticcomplexityCheck.cpp 的修改：calculateCyclomaticComplexity 实现

我们在第 42-45 行迭代所有 CFG 块。块的数量对应于节点数，在图 6.2 中用 N 表示。我们计算每个块的后续节点数之和，以计算边的数量，用 E 表示。我们假设对于我们的简化示例，连接组件的数量，用 P 表示，等于一个。

在实现检查后，是时候构建并在我们的示例上运行我们的新检查了；参见图 6.1。

### 6.4.3 构建和测试循环复杂度检查

我们将使用图 1.4 中指定的基本构建配置，并使用图 5.2 中的标准命令构建 Clang-Tidy：

```cpp
$ ninja install-clang-tidy
```

假设从图 1.4 的构建配置，此命令将 Clang-Tidy 二进制文件安装到`<...>/llvm-project/install/bin`文件夹中。

重要提示

如果你使用带有共享库的构建配置（将`BUILD_SHARED_LIBS`标志设置为`ON`），如图 1.12 所示，那么你可能需要使用`ninja install`安装和构建所有工件。

我们将在图 6.1 中显示的示例程序上运行我们的检查。正如我们之前计算的，测试的循环复杂度为 2，低于我们在`check`方法实现中指定的默认值 5，如图 6.6 所示。因此，我们需要将默认值重写为 1，以便在测试程序中看到警告。这可以通过使用我们之前用于`classchecker`检查测试的`-config`选项来完成，如图 5.20 所示。测试命令如下：

```cpp
1$ <...>/llvm-project/install/bin/clang-tidy                         \ 

2   -checks="-*,misc-cyclomaticcomplexity"                            \ 

3   -config="{CheckOptions:                                           \ 

4             [{key: misc-cyclomaticcomplexity.Threshold, value: ’1’}]}" \ 

5   max.cpp                                                          \ 

6   -- -std=c++17
```

**图 6.8**：在 max.cpp 示例上测试循环复杂度

图 6.8 中的*第 2 行*表明我们只想运行一个 Clang-Tidy 检查：`misc-cyclomaticcomplexity`。在*第 3-4 行*中，我们设置了所需的阈值。*第 5 行*指定了正在测试的文件名（在我们的例子中是`max.cpp`），而最后一行，*第 6 行*包含了我们程序的某些编译标志。

如果我们运行图 6.8 中的命令，将会得到以下输出：

```cpp
max.cpp:1:5: warning: function ’max’ has high cyclomatic complexity (2) ...
int max(int a, int b) {
    ^
```

**图 6.9**：在 max.cpp 示例上测试循环复杂度：输出

可能会提出以下问题：Clang 是如何构建 CFG 的？我们可以使用调试器来调查这个过程。

## 6.5 Clang 上的 CFG

CFG 是使用 Clang 工具进行高级静态分析的基本数据结构。Clang 从函数的 AST（抽象语法树）构建 CFG，识别基本块和控制流边。Clang 的 CFG 构建处理各种 C/C++结构，包括循环、条件语句、switch 情况以及如`setjmp/longjmp`和 C++异常等复杂结构。让我们使用图 6.1 中的示例来考虑这个过程。

### 6.5.1 通过示例进行 CFG 构建

我们在图 6.1 中的示例有五个节点，如图图 6.2 所示。让我们运行一个调试器来调查这个过程，如下所示：

```cpp
1$ lldb <...>/llvm-project/install/bin/clang-tidy --                   \ 

2   -checks="-*,misc-cyclomaticcomplexity"                              \ 

3   -config="{CheckOptions:                                             \ 

4              [{key: misc-cyclomaticcomplexity.Threshold, value: ’1’}]}" \ 

5   max.cpp                                                             \ 

6   -- -std=c++17 -Wno-all
```

**图 6.10**: 运行以调查 CFG 创建过程的调试器会话

我们使用了与图 6.8 中相同的命令，但将命令的第一行改为通过调试器运行检查。我们还改变了最后一行以抑制编译器的所有警告。

重要提示

高级静态分析是语义分析的一部分。例如，如果 Clang 检测到不可达的代码，将会打印警告，由`-Wunreachable-code`选项控制。检测器是 Clang 语义分析的一部分，并利用 CFGs（控制流图）以及 ASTs（抽象语法树）作为基本数据结构来检测此类问题。我们可以抑制这些警告，并因此通过指定特殊的`-Wno-all`命令行选项来禁用 Clang 中的 CFG 初始化，该选项抑制编译器生成的所有警告。

我们将在`CFGBuilder::createBlock`函数上设置断点，该函数创建 CFG 块。

```cpp
$ lldb <...>/llvm-project/install/bin/clang-tidy --                   \ 

  -checks="-*,misc-cyclomaticcomplexity"                              \ 

  -config="{CheckOptions:                                             \ 

             [{key: misc-cyclomaticcomplexity.Threshold, value: ’1’}]}" \ 

  max.cpp                                                             \ 

  -- -std=c++17 -Wno-all 

... 

(lldb) b CFGBuilder::createBlock 

Breakpoint 1: where = ...CFGBuilder::createBlock(bool) const ...
```

**图 6.11**: 运行调试器并设置 CFGBuilder::createBlock 的断点

如果我们运行调试器，我们将看到我们的示例函数被调用了五次；也就是说，为我们的`max`函数创建了五个 CFG 块：

```cpp
1(lldb) r 

2 ... 

3     frame #0: ...CFGBuilder::createBlock... 

4    1690 /// createBlock - Used to lazily create blocks that are connected 

5    1691 ///  to the current (global) successor. 

6    1692 CFGBlock *CFGBuilder::createBlock(bool add_successor) { 

7 -> 1693   CFGBlock *B = cfg->createBlock(); 

8    1694   if (add_successor && Succ) 

9    1695     addSuccessor(B, Succ); 

10    1696   return B; 

11  

12 (lldb) c 

13 ... 

14 (lldb) c 

15 ... 

16 (lldb) c 

17 ... 

18 (lldb) c 

19 ... 

20 (lldb) c 

21 ... 

221  warning generated. 

23 max.cpp:1:5: warning: function ’max’ has high cyclomatic complexity (2) [misc-cyclomaticcomplexity] 

24 int max(int a, int b) { 

25     ^ 

26 Process ... exited with status = 0 (0x00000000)
```

**图 6.12**: 创建 CFG 块，突出显示断点

如图 6.12 中所示的调试器会话可以被认为是 CFG 创建过程的入口点。现在，是时候深入探讨实现细节了。

### 6.5.2 CFG 构建实现细节

块是按相反的顺序创建的，如图图 6.13 所示。首先创建的是退出块，如图图 6.13 所示，*第 4 行*。然后，CFG 构建器遍历作为参数传递的`clang::Stmt`对象（*第 9 行*）。入口块最后创建，在*第 12 行*：

```cpp
1std::unique_ptr<CFG> CFGBuilder::buildCFG(const Decl *D, Stmt *Statement) { 

2   ... 

3   // Create an empty block that will serve as the exit block for the CFG. 

4   Succ = createBlock(); 

5   assert(Succ == &cfg->getExit()); 

6   Block = nullptr;  // the EXIT block is empty.  ... 

7   ... 

8   // Visit the statements and create the CFG. 

9   CFGBlock *B = Visit(Statement, ...); 

10   ... 

11   // Create an empty entry block that has no predecessors. 

12   cfg->setEntry(createBlock()); 

13   ... 

14   return std::move(cfg); 

15 }
```

**图 6.13**: 从 clang/lib/Analysis/CFG.cpp 中提取的简化 buildCFG 实现

访问者使用`clang::Stmt::getStmtClass`方法根据语句的类型实现一个临时的访问者，如下面的代码片段所示：

```cpp
1CFGBlock *CFGBuilder::Visit(Stmt * S, ...) { 

2   ... 

3   switch (S->getStmtClass()) { 

4     ... 

5     case Stmt::CompoundStmtClass: 

6       return VisitCompoundStmt(cast<CompoundStmt>(S), ...); 

7     ... 

8     case Stmt::IfStmtClass: 

9       return VisitIfStmt(cast<IfStmt>(S)); 

10     ... 

11     case Stmt::ReturnStmtClass: 

12     ... 

13       return VisitReturnStmt(S); 

14     ... 

15   } 

16 }
```

**图 6.14**: 状态访问者实现；用于我们示例的情况被突出显示，代码取自 clang/lib/Analysis/CFG.cpp

我们的例子包括两个返回语句和一个`if`语句，它们被组合成一个复合语句。访问者的相关部分在图 6.14 中显示。

在我们的例子中，传递的语句是一个复合语句；因此，图 6.14 中的第 6 行被激活。然后执行以下代码：

```cpp
1CFGBlock *CFGBuilder::VisitCompoundStmt(CompoundStmt *C, ...) { 

2   ... 

3   CFGBlock *LastBlock = Block; 

4  

5   for (Stmt *S : llvm::reverse(C->body())) { 

6    // If we hit a segment of code just containing ’;’ (NullStmts), we can 

7    // get a null block back.  In such cases, just use the LastBlock 

8    CFGBlock *newBlock = Visit(S, ...); 

9  

10    if (newBlock) 

11      LastBlock = newBlock; 

12  

13    if (badCFG) 

14      return nullptr; 

15    ... 

16   } 

17  

18   return LastBlock; 

19 }
```

**图 6.15**：复合语句访问者，代码来自 clang/lib/Analysis/CFG.cpp

在为我们的例子创建 CFG 时，访问了几个构造。第一个是`clang::IfStmt`。相关部分在以下图中显示：

```cpp
1CFGBlock *CFGBuilder::VisitIfStmt(IfStmt *I) { 

2   ... 

3   // Process the true branch. 

4   CFGBlock *ThenBlock; 

5   { 

6     Stmt *Then = I->getThen(); 

7     ... 

8     ThenBlock = Visit(Then, ...); 

9     ... 

10   } 

11  

12   // Specially handle "if (expr1 || ...)" and "if (expr1 && ...)" 

13   // ... 

14   if (Cond && Cond->isLogicalOp()) 

15     ... 

16   else { 

17     // Now create a new block containing the if statement. 

18     Block = createBlock(false); 

19     ... 

20   } 

21   ... 

22 }
```

**图 6.16**：`if`语句访问者，代码来自 clang/lib/Analysis/CFG.cpp

在第 18 行创建了一个特殊的`if`语句块。我们还访问了第 8 行的`then`条件。

`then`条件导致访问返回语句。相应的代码如下：

```cpp
1CFGBlock *CFGBuilder::VisitReturnStmt(Stmt *S) { 

2   // Create the new block. 

3   Block = createBlock(false); 

4   ... 

5   // Visit children 

6   if (ReturnStmt *RS = dyn_cast<ReturnStmt>(S)) { 

7     if (Expr *O = RS->getRetValue()) 

8       return Visit(O, ...); 

9     return Block; 

10   } 

11   ... 

12 }
```

**图 6.17**：返回语句访问者，代码来自 clang/lib/Analysis/CFG.cpp

对于我们的例子，它在第 3 行创建了一个块并访问了第 8 行的返回表达式。我们的返回表达式是一个简单的表达式，不需要创建新的块。

在图 6.13 到图 6.17 中展示的代码片段仅显示了块创建过程。为了简化，省略了一些重要部分。值得注意的是，构建过程还涉及以下内容：

+   边缘创建：一个典型的块可以有一个或多个后继者。每个块的节点（块）列表以及每个块的后继者（边）列表维护整个图结构，表示符号程序执行。

+   存储元信息：每个块存储与其相关的附加元信息。例如，每个块保留该块中语句的列表。

+   处理边缘情况：C++是一种复杂的语言，具有许多不同的语言结构，需要特殊处理。

CFG 是高级代码分析的基本数据结构。Clang 有几个使用 CFG 创建的工具。让我们简要地看看它们。

## 6.6 Clang 分析工具简要描述

如前所述，CFG 是 Clang 中其他分析工具的基础，其中一些是在 CFG 之上创建的。这些工具也使用高级数学来分析各种情况。最显著的工具如下 [32]：

+   LivenessAnalysis：确定计算值在覆盖之前是否会被使用，为每个语句和 CFGBlock 生成活动性集

+   未初始化变量：通过多次遍历识别未初始化变量的使用，包括对语句的初始分类和后续的变量使用计算

+   线程安全性分析：分析标记的函数和变量以确保线程安全性

Clang 中的 LivenessAnalysis 对于通过确定在某个点计算出的值在覆盖之前是否会被使用来优化代码至关重要。它为每个语句和 CFGBlock 生成活动集，指示变量或表达式的潜在未来使用。这种“可能”的后向分析通过将变量声明和赋值视为写入，将其他上下文视为读取，简化了读写分类，无论是否存在别名或字段使用。它在死代码消除和编译器优化（如高效的寄存器分配）中非常有价值，有助于释放内存资源并提高程序效率。尽管存在边缘案例和文档的挑战，但其直接的实现和缓存查询结果的能力使其成为提高软件性能和资源管理的重要工具。

重要注意事项

前向分析是编程中用来检查数据从程序开始到结束如何流动的方法。随着程序的运行，逐步跟踪数据路径使我们能够看到它的变化或去向。这种方法对于识别诸如设置不当的变量或跟踪程序中的数据流等问题至关重要。它与反向分析形成对比，反向分析从程序的末尾开始，向后工作。

Clang 中的未初始化变量分析旨在检测在初始化之前使用变量的情况，它作为一个前向的“必须”分析操作。它涉及多个遍历，包括对代码进行初始扫描以对语句进行分类，以及随后使用固定点算法通过 CFG 传播信息。它处理比 LivenessAnalysis 更复杂的场景，面临着诸如缺乏对记录字段和非可重用分析结果支持等挑战，这限制了它在某些情况下的效率。

Clang 中的线程安全性分析是一种前向分析，它专注于确保多线程代码中的正确同步。它为每个语句块中的每个语句计算被锁定互斥锁的集合，并利用注解来指示受保护的变量或函数。将 Clang 表达式转换为 TIL（类型中间语言）[32]，它有效地处理了 C++ 表达式和注解的复杂性。尽管它对 C++ 有强大的支持，并且对变量交互有深入的理解，但它面临着一些限制，例如缺乏对别名支持，这可能导致误报。

## 6.7 了解分析的限制

值得提及的是，使用 Clang 的 AST 和 CFG 可以进行一些分析的限制，其中最显著的如下 [2]：

+   Clang 的 AST 局限性：Clang 的 AST 不适合数据流分析和控制流推理，由于丢失了关键的语言信息，导致结果不准确且分析效率低下。分析的健全性也是一个考虑因素，某些分析（如可达性分析）的精确性如果足够精确，那么它们是有价值的，而不是总是保守的。

+   Clang 的 CFG 问题：尽管 Clang 的 CFG 旨在弥合 AST 和 LLVM IR 之间的差距，但它遇到了已知的问题，具有有限的跨程序能力，并且缺乏足够的测试覆盖率。

在[2]中提到的一个例子与 C++20 中引入的新特性 C++ coroutines 有关。该功能的一些方面是在 Clang 前端之外实现的，并且无法通过 Clang 的 AST 和 CFG 等工具看到。这种限制使得对这些功能的分析，尤其是生命周期分析，变得复杂。

尽管存在这些局限性，Clang 的 CFG 仍然是一个在编译器和编译器工具开发中广泛使用的强大工具。还有其他工具正在积极开发中 [27]，旨在弥合 Clang 的 CFG 能力上的差距。

## 6.8 摘要

在本章中，我们研究了 Clang 的 CFG，这是一种强大的数据结构，用于表示程序的符号执行。我们使用 CFG 创建了一个简单的 Clang-Tidy 检查，用于计算环路复杂度，这是一个用于估计代码复杂度的有用度量。此外，我们还探讨了 CFG 创建的细节及其基本内部结构的形成。我们讨论了一些使用 CFG 开发的工具，这些工具对于检测生命周期问题、线程安全和未初始化变量很有用。我们还简要描述了 CFG 的局限性以及其他工具如何解决这些局限性。

下一章将介绍重构工具。这些工具可以使用 Clang 编译器提供的 AST 执行复杂的代码修改。

## 6.9 未来阅读

+   Flemming Nielson、Hanne Riis Nielson 和 Chris Hankin，*程序分析原理*，Springer，2005 [29]

+   Xavier Rival 和 Kwangkeun Yi，*静态分析导论：抽象解释视角*，麻省理工学院出版社，2020 [30]

+   Kristóf Umann *Clang 中数据流分析的调查*: [`lists.llvm.org/pipermail/cfe-dev/2020-October/066937.html`](https://lists.llvm.org/pipermail/cfe-dev/2020-October/066937.html)

+   Bruno Cardoso Lopes 和 Nathan Lanza *基于 MLIR 的 Clang IR* *(CIR)*: [`discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319`](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319)
