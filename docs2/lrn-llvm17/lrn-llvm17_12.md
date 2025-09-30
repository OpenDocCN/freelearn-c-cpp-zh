

# 第九章：JIT 编译

LLVM 核心库包含一个名为 **ExecutionEngine** 的组件，该组件允许在内存中编译和执行 **中间表示**（**IR**）代码。使用此组件，我们可以构建 **即时**（**JIT**）编译器，这允许直接执行 IR 代码。即时编译器更像是一个解释器，因为不需要在辅助存储上存储目标代码。

在本章中，你将了解即时编译器的应用，以及 LLVM 即时编译器在原理上是如何工作的。你将探索 LLVM 动态编译器和解释器，并学习如何自己实现即时编译器工具。此外，你还将学习如何将即时编译器作为静态编译器的一部分使用，以及相关的挑战。

本章将涵盖以下主题：

+   了解 LLVM 的 JIT 实现和使用案例概述

+   使用 JIT 编译进行直接执行

+   从现有类实现自己的 JIT 编译器

+   从零开始实现自己的 JIT 编译器

到本章结束时，你将理解并知道如何开发一个即时编译器，无论是使用预配置的类还是定制版本以满足你的需求。

# 技术要求

你可以在[`github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter09`](https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter09) 找到本章使用的代码。

# LLVM 的整体 JIT 实现和使用案例

到目前为止，我们只看了 **预编译**（**AOT**）编译器。这些编译器编译整个应用程序。应用程序只能在编译完成后运行。如果编译是在应用程序的运行时执行的，那么编译器就是一个 JIT 编译器。JIT 编译器有一些有趣的用例：

+   **虚拟机的实现**：一种编程语言可以使用 AOT 编译器转换为字节码。在运行时，使用 JIT 编译器将字节码编译为机器代码。这种方法的优点是字节码是硬件无关的，而且由于 JIT 编译器的存在，与 AOT 编译器相比没有性能损失。Java 和 C# 目前使用这种模型，但这并不是一个新想法：1977 年的 USCD Pascal 编译器已经使用了类似的方法。

+   `lldb` LLVM 调试器使用这种方法在调试时评估源表达式。

+   **数据库查询**：数据库从数据库查询中创建一个执行计划。执行计划描述了对表和列的操作，当执行时，这些操作导致查询结果。可以使用即时编译器将执行计划转换为机器代码，从而加快查询的执行速度。

LLVM 的静态编译模型并不像人们想象的那样远离 JIT 模型。`llc` LLVM 静态编译器将 LLVM IR 编译成机器代码，并将结果保存为磁盘上的目标文件。如果目标文件不是存储在磁盘上而是在内存中，代码是否可执行？不是直接可执行，因为对全局函数和全局数据的引用使用重定位而不是绝对地址。从概念上讲，**重定位**描述了如何计算地址——例如，作为已知地址的偏移量。如果我们像链接器和动态加载器那样解析重定位到地址，那么我们可以执行目标代码。运行静态编译器将 IR 代码编译成内存中的目标文件，对内存中的目标文件执行链接步骤，然后运行代码，我们就得到了一个 JIT 编译器。LLVM 核心库中的 JIT 实现基于这个想法。

在 LLVM 的发展历史中，有几种 JIT 实现，具有不同的功能集。最新的 JIT API 是 **按需编译**（**ORC**）引擎。如果你对缩写词感兴趣，这是主要开发者的意图，在 **可执行和链接格式**（**ELF**）和 **调试标准**（**DWARF**）已经存在之后，再次发明一个基于托尔金的宇宙的缩写词。

ORC 引擎建立在并扩展了在内存中的目标文件上使用静态编译器和动态链接器的想法。该实现使用分层方法。两个基本级别是编译层和链接层。在这之上是一个提供懒编译支持的层。可以在懒编译层之上或之下堆叠一个转换层，允许开发者添加任意的转换或简单地通知某些事件。此外，这种分层方法的优势在于 JIT 引擎可以根据不同的需求进行定制。例如，高性能虚拟机可能会选择预先编译所有内容，并且不使用懒编译层。另一方面，其他虚拟机可能会强调启动时间和对用户的响应性，并借助懒编译层来实现这一点。

较旧的 MCJIT 引擎仍然可用，其 API 来自一个更早、已经删除的 JIT 引擎。随着时间的推移，这个 API 逐渐变得臃肿，并且缺乏 ORC API 的灵活性。目标是移除这个实现，因为 ORC 引擎现在提供了 MCJIT 引擎的所有功能，新的开发应该使用 ORC API。

在我们深入实现 JIT 编译器之前，下一节我们将探讨 `lli`，LLVM 解释器和动态编译器。

# 使用 JIT 编译进行直接执行

直接运行 LLVM IR 是想到即时编译器时的第一个想法。这正是 `lli` 工具、LLVM 解释器和动态编译器所做的事情。我们将在下一节中探讨 `lli` 工具。

## 探索 lli 工具

让我们用一个非常简单的例子来尝试 `lli` 工具。下面的 LLVM IR 可以存储在一个名为 `hello.ll` 的文件中，这相当于一个 C 的 hello world 应用程序。此文件声明了来自 C 库的 `printf()` 函数的原型。`hellostr` 常量包含要打印的消息。在 `main()` 函数内部，会生成对 `printf()` 函数的调用，并且这个函数包含一个将要打印的 `hellostr` 消息。应用程序总是返回 `0`。

完整的源代码如下：

```cpp

declare i32 @printf(ptr, ...)
@hellostr = private unnamed_addr constant [13 x i8] c"Hello world\0A\00"
define dso_local i32 @main(i32 %argc, ptr %argv) {
  %res = call i32 (ptr, ...) @printf(ptr @hellostr)
  ret i32 0
}
```

这个 LLVM IR 文件足够通用，适用于所有平台。我们可以直接使用以下命令使用 `lli` 工具执行 IR：

```cpp

$ lli hello.ll
Hello world
```

这里有趣的一点是如何找到 `printf()` 函数。IR 代码被编译成机器代码，并触发对 `printf` 符号的查找。这个符号在 IR 中找不到，因此当前进程会搜索它。`lli` 工具动态链接到 C 库，并在那里找到符号。

当然，`lli` 工具不会链接到你创建的库。为了启用这些函数的使用，`lli` 工具支持加载共享库和对象。以下 C 源代码仅打印一条友好的消息：

```cpp

#include <stdio.h>
void greetings() {
  puts("Hi!");
}
```

存储在 `greetings.c` 中，我们使用它来探索使用 `lli` 加载对象。以下命令将此源代码编译成一个共享库。`–fPIC` 选项指示 `clang` 生成位置无关代码，这对于共享库是必需的。此外，编译器使用 `–shared` 创建一个名为 `greetings.so` 的共享库：

```cpp

$ clang greetings.c -fPIC -shared -o greetings.so
```

我们还将文件编译成 `greetings.o` 对象文件：

```cpp

$ clang greetings.c -c -o greetings.o
```

现在我们有两个文件，即 `greetings.so` 共享库和 `greetings.o` 对象文件，我们将它们加载到 `lli` 工具中。

我们还需要一个调用 `greetings()` 函数的 LLVM IR 文件。为此，创建一个包含对函数的单个调用的 `main.ll` 文件：

```cpp

declare void @greetings(...)
define dso_local i32 @main(i32 %argc, i8** %argv) {
  call void (...) @greetings()
  ret i32 0
}
```

注意，在执行时，之前的 IR 会崩溃，因为 `lli` 无法定位到问候符号：

```cpp

$ lli main.ll
JIT session error: Symbols not found: [ _greetings ]
lli: Failed to materialize symbols: { (main, { _main }) }
```

`greetings()` 函数定义在一个外部文件中，为了修复崩溃，我们必须告诉 `lli` 工具需要加载哪些额外的文件。为了使用共享库，你必须使用 `–load` 选项，它接受共享库的路径作为参数：

```cpp

$ lli –load ./greetings.so main.ll
Hi!
```

如果包含共享库的目录不在动态加载器的搜索路径中，则指定共享库的路径很重要。如果省略，则库将无法找到。

或者，我们可以指示 `lli` 使用 `–extra-object` 加载对象文件：

```cpp

$ lli –extra-object greetings.o main.ll
Hi!
```

其他支持选项包括 `–extra-archive`，它加载一个存档，以及 `–extra-module`，它加载另一个位代码文件。这两个选项都需要文件路径作为参数。

你现在知道了如何使用 `lli` 工具直接执行 LLVM IR。在下一节中，我们将实现自己的 JIT 工具。

# 使用 LLJIT 实现自己的 JIT 编译器

`lli` 工具不过是围绕 LLVM API 的一个薄包装。在第一部分，我们了解到 ORC 引擎使用分层方法。`ExecutionSession` 类代表一个正在运行的 JIT 程序。除了其他项目外，这个类还持有诸如使用的 `JITDylib` 实例等信息。一个 `JITDylib` 实例是一个符号表，它将符号名称映射到地址。例如，这些可以是定义在 LLVM 立即编译代码文件中的符号，或者加载的共享库中的符号。

对于执行 LLVM 立即编译代码，我们不需要自己创建 JIT 栈，因为 `LLJIT` 类提供了这一功能。在从较老的 MCJIT 实现迁移时，您也可以使用这个类，因为这个类本质上提供了相同的功能。

为了说明 `LLJIT` 工具的功能，我们将创建一个包含 JIT 功能的交互式计算器应用程序。我们的 JIT 计算器的主要源代码将扩展自 *第二章*，《编译器结构》中的 `calc` 示例。

我们交互式即时编译计算器的核心思想如下：

1.  允许用户输入一个函数定义，例如 `def f(x) =` `x*2`。

1.  用户输入的函数随后将由 `LLJIT` 工具编译成一个函数——在这种情况下，是 `f` 函数。

1.  允许用户使用数值调用他们定义的函数：`f(3)`。

1.  使用提供的参数评估函数，并将结果打印到控制台：`6`。

在我们讨论将 JIT 功能集成到计算器源代码之前，有一些主要差异需要指出，与原始计算器示例相比：

+   首先，我们之前只输入和解析以 `with` 关键字开头的函数，而不是之前描述的 `def` 关键字。对于本章，我们只接受以 `def` 关键字开头的函数定义，并在我们的 `DefDecl` 中表示为特定的节点。`DefDecl` 类知道它所定义的参数及其名称，并且函数名也存储在这个类中。

+   其次，我们还需要我们的抽象语法树（AST）能够识别函数调用，以表示 `LLJIT` 工具所消耗或即时编译（JIT）的函数。每当用户输入一个函数名，后面跟着括号内的参数时，AST 会将这些识别为 `FuncCallFromDef` 节点。这个类本质上知道与 `DefDecl` 类相同的信息。

由于增加了这两个 AST 类，可以明显预期语义分析、解析器和代码生成类将相应地调整以处理我们 AST 中的变化。需要注意的是，还增加了一个新的数据结构，称为`JITtedFunctions`，这些类都了解这个数据结构。这个数据结构是一个映射，其中定义的函数名作为键，函数定义中存储的参数数量作为映射中的值。我们将在稍后看到这个数据结构如何在我们的 JIT 计算器中利用。

关于我们对`calc`示例所做的更改的更多细节，包含从`calc`和本节 JIT 实现的更改的完整源代码可以在`lljit`源目录中找到。

## 将 LLJIT 引擎集成到计算器中

首先，让我们讨论如何在交互式计算器中设置 JIT 引擎。与 JIT 引擎相关的所有实现都存在于`Calc.cpp`文件中，该文件有一个`main()`循环用于程序的执行：

1.  除了包括我们的代码生成、语义分析器和解析器实现的头文件外，我们还必须包含几个头文件。《LLJIT.h》头文件定义了`LLJIT`类和 ORC API 的核心类。接下来，需要`InitLLVM.h`头文件来进行工具的基本初始化，以及需要`TargetSelect.h`头文件来进行本地目标的初始化。最后，我们还包含了`<iostream>` C++头文件，以便允许用户在我们的计算器应用程序中输入：

    ```cpp

    #include "CodeGen.h"
    #include "Parser.h"
    #include "Sema.h"
    #include "llvm/ExecutionEngine/Orc/LLJIT.h"
    #include "llvm/Support/InitLLVM.h"
    #include "llvm/Support/TargetSelect.h"
    #include <iostream>
    ```

1.  接下来，我们将`llvm`和`llvm::orc`命名空间添加到当前作用域：

    ```cpp

    using namespace llvm;
    using namespace llvm::orc;
    ```

1.  我们将要创建的`LLJIT`实例中的许多调用都会返回一个错误类型，`Error`。`ExitOnError`类允许我们在记录到`stderr`并退出应用程序的同时丢弃由`LLJIT`实例返回的`Error`值。我们声明一个全局的`ExitOnError`变量如下：

    ```cpp

    ExitOnError ExitOnErr;
    ```

1.  然后，我们添加`main()`函数，该函数初始化工具和本地目标：

    ```cpp

    int main(int argc, const char **argv{
      InitLLVM X(argc, argv);
      InitializeNativeTarget();
      InitializeNativeTargetAsmPrinter();
      InitializeNativeTargetAsmParser();
    ```

1.  我们使用`LLJITBuilder`类创建一个`LLJIT`实例，并将其封装在之前声明的`ExitOnErr`变量中，以防出现错误。一个可能出错的原因是平台尚未支持 JIT 编译：

    ```cpp

    auto JIT = ExitOnErr(LLJITBuilder().create());
    ```

1.  接下来，我们声明我们的`JITtedFunctions`映射，该映射跟踪函数定义，正如我们之前所描述的：

    ```cpp

    StringMap<size_t> JITtedFunctions;
    ```

1.  为了方便等待用户输入的环境，我们添加了一个`while()`循环，并允许用户输入一个表达式，将用户输入的行保存到一个名为`calcExp`的字符串中：

    ```cpp

      while (true) {
        outs() << "JIT calc > ";
        std::string calcExp;
        std::getline(std::cin, calcExp);
    ```

1.  之后，初始化 LLVM 上下文类和新的 LLVM 模块。模块的数据布局也相应设置，我们还声明了一个代码生成器，该生成器将用于为用户在命令行上定义的函数生成 IR：

    ```cpp

        std::unique_ptr<LLVMContext> Ctx = std::make_unique<LLVMContext>();
        std::unique_ptr<Module> M = std::make_unique<Module>("JIT calc.expr", *Ctx);
        M->setDataLayout(JIT->getDataLayout());
        CodeGen CodeGenerator;
    ```

1.  我们必须解释用户输入的行，以确定用户是定义一个新函数还是调用他们之前定义并带有参数的函数。在接收用户输入的行时定义了一个`Lexer`类。我们将看到词法分析器主要关注两个主要情况：

    ```cpp

        Lexer Lex(calcExp);
        Token::TokenKind CalcTok = Lex.peek();
    ```

1.  词法分析器可以检查用户输入的第一个标记。如果用户正在定义一个新的函数（由`def`关键字或`Token::KW_def`标记表示），那么我们将解析它并检查其语义。如果解析器或语义分析器检测到用户定义的函数有任何问题，将相应地发出错误，计算器程序将停止。如果没有检测到解析器或语义分析器的错误，这意味着我们有一个有效的 AST 数据结构，即`DefDecl`：

    ```cpp

       if (CalcTok == Token::KW_def) {
          Parser Parser(Lex);
          AST *Tree = Parser.parse();
          if (!Tree || Parser.hasError()) {
            llvm::errs() << "Syntax errors occured\n";
            return 1;
          }
          Sema Semantic;
          if (Semantic.semantic(Tree, JITtedFunctions)) {
            llvm::errs() << "Semantic errors occured\n";
            return 1;
          }
    ```

1.  然后，我们可以将新构建的 AST 传递给我们的代码生成器，编译用户定义函数的中间表示（IR）。IR 生成的具体细节将在之后讨论，但这个编译为 IR 的函数需要知道模块和我们的`JITtedFunctions`映射。在生成 IR 之后，我们可以通过调用`addIRModule()`并将模块和上下文包装在`ThreadSafeModule`类中来将此信息添加到我们的`LLJIT`实例中，以防止这些信息被其他并发线程访问：

    ```cpp

          CodeGenerator.compileToIR(Tree, M.get(), JITtedFunctions);
          ExitOnErr(
              JIT->addIRModule(ThreadSafeModule(std::move(M),           std::move(Ctx))));
    ```

1.  相反，如果用户正在调用带有参数的函数，这由`Token::ident`标记表示，我们还需要在将输入转换为有效的 AST 之前解析和语义检查用户输入是否有效。这里的解析和检查与之前略有不同，因为它可能包括确保用户提供给函数调用的参数数量与函数最初定义的参数数量相匹配的检查：

    ```cpp

       } else if (CalcTok == Token::ident) {
          outs() << "Attempting to evaluate expression:\n";
          Parser Parser(Lex);
          AST *Tree = Parser.parse();
          if (!Tree || Parser.hasError()) {
            llvm::errs() << "Syntax errors occured\n";
            return 1;
          }
          Sema Semantic;
          if (Semantic.semantic(Tree, JITtedFunctions)) {
            llvm::errs() << "Semantic errors occured\n";
            return 1;
          }
    ```

1.  一旦为函数调用构建了一个有效的抽象语法树（AST），即`FuncCallFromDef`，我们就从 AST 中获取函数名称，然后代码生成器准备生成对之前添加到`LLJIT`实例中的函数的调用。在幕后发生的是，用户定义的函数被重新生成为一个 LLVM 调用，在一个将执行原始函数实际评估的单独函数中。这一步需要 AST、模块、函数调用名称以及我们的函数定义映射：

    ```cpp

          llvm::StringRef FuncCallName = Tree->getFnName();
          CodeGenerator.prepareCalculationCallFunc(Tree, M.get(),       FuncCallName, JITtedFunctions);
    ```

1.  在代码生成器完成重新生成原始函数和创建单独评估函数的工作后，我们必须将此信息添加到`LLJIT`实例中。我们创建一个`ResourceTracker`实例来跟踪分配给添加到`LLJIT`的函数的内存，以及另一个模块和上下文的`ThreadSafeModule`实例。然后，这两个实例被添加到 JIT 作为一个 IR 模块：

    ```cpp

          auto RT = JIT->getMainJITDylib().createResourceTracker();
          auto TSM = ThreadSafeModule(std::move(M), std::move(Ctx));
          ExitOnErr(JIT->addIRModule(RT, std::move(TSM)));
    ```

1.  然后，通过`lookup()`方法在我们的`LLJIT`实例中查询单独的评估函数，通过将我们的评估函数名称`calc_expr_func`提供给函数。如果查询成功，`calc_expr_func`函数的地址被转换为适当类型，这是一个不接受任何参数并返回单个整数的函数。一旦获得函数的地址，我们就调用该函数以生成用户定义函数的参数所提供的参数的结果，然后将结果打印到控制台：

    ```cpp

          auto CalcExprCall = ExitOnErr(JIT->lookup("calc_expr_func"));
          int (*UserFnCall)() = CalcExprCall.toPtr<int (*)()>();
          outs() << "User defined function evaluated to:       " << UserFnCall() << "\n";
    ```

1.  函数调用完成后，之前与我们的函数关联的内存随后通过`ResourceTracker`释放：

    ```cpp

    ExitOnErr(RT->remove());
    ```

## 代码生成更改以支持通过 LLJIT 进行 JIT 编译

现在，让我们简要地看一下我们在`CodeGen.cpp`中做出的某些更改，以支持我们的基于 JIT 的计算器：

1.  如前所述，代码生成类有两个重要方法：一个是将用户定义的函数编译成 LLVM IR 并将 IR 打印到控制台，另一个是准备计算评估函数`calc_expr_func`，它包含对原始用户定义函数的评估调用。第二个函数也将生成的 IR 打印给用户：

    ```cpp

    void CodeGen::compileToIR(AST *Tree, Module *M,
                        StringMap<size_t> &JITtedFunctions) {
      ToIRVisitor ToIR(M, JITtedFunctions);
      ToIR.run(Tree);
      M->print(outs(), nullptr);
    }
    void CodeGen::prepareCalculationCallFunc(AST *FuncCall,
               Module *M, llvm::StringRef FnName,
               StringMap<size_t> &JITtedFunctions) {
      ToIRVisitor ToIR(M, JITtedFunctions);
      ToIR.genFuncEvaluationCall(FuncCall);
      M->print(outs(), nullptr);
    }
    ```

1.  如前所述的源代码所示，这些代码生成函数定义了一个`ToIRVisitor`实例，它接受我们的模块和一个`JITtedFunctions`映射，在初始化时用于其构造函数：

    ```cpp

    class ToIRVisitor : public ASTVisitor {
      Module *M;
      IRBuilder<> Builder;
      StringMap<size_t> &JITtedFunctionsMap;
    . . .
    public:
      ToIRVisitor(Module *M,
                  StringMap<size_t> &JITtedFunctions)
          : M(M), Builder(M->getContext()),       JITtedFunctionsMap(JITtedFunctions) {
    ```

1.  最终，这些信息被用来生成 IR 或评估之前为 IR 生成的函数。当生成 IR 时，代码生成器期望看到一个`DefDecl`节点，它代表定义一个新函数。函数名称及其定义的参数数量存储在函数定义映射中：

    ```cpp

    virtual void visit(DefDecl &Node) override {
        llvm::StringRef FnName = Node.getFnName();
        llvm::SmallVector<llvm::StringRef, 8> FunctionVars =     Node.getVars();
        (JITtedFunctionsMap)[FnName] = FunctionVars.size();
    ```

1.  之后，通过`genUserDefinedFunction()`调用创建实际函数定义：

    ```cpp

        Function *DefFunc = genUserDefinedFunction(FnName);
    ```

1.  在`genUserDefinedFunction()`中，第一步是检查函数是否在模块中存在。如果不存在，我们确保函数原型存在于我们的映射数据结构中。然后，我们使用名称和参数数量来构建一个具有用户定义的参数数量的函数，并使该函数返回一个单一整数值：

    ```cpp

    Function *genUserDefinedFunction(llvm::StringRef Name) {
        if (Function *F = M->getFunction(Name))
          return F;
        Function *UserDefinedFunction = nullptr;
        auto FnNameToArgCount = JITtedFunctionsMap.find(Name);
        if (FnNameToArgCount != JITtedFunctionsMap.end()) {
          std::vector<Type *> IntArgs(FnNameToArgCount->second,       Int32Ty);
          FunctionType *FuncType = FunctionType::get(Int32Ty,       IntArgs, false);
          UserDefinedFunction =
              Function::Create(FuncType,           GlobalValue::ExternalLinkage, Name, M);
        }
        return UserDefinedFunction;
      }
    ```

1.  在生成用户定义的函数之后，创建一个新的基本块，并将我们的函数插入到基本块中。每个函数参数也与用户定义的名称相关联，因此我们也为所有函数参数设置了相应的名称，以及生成在函数内部操作的数学运算：

    ```cpp

        BasicBlock *BB = BasicBlock::Create(M->getContext(),     "entry", DefFunc);
        Builder.SetInsertPoint(BB);
        unsigned FIdx = 0;
        for (auto &FArg : DefFunc->args()) {
          nameMap[FunctionVars[FIdx]] = &FArg;
          FArg.setName(FunctionVars[FIdx++]);
        }
        Node.getExpr()->accept(*this);
      };
    ```

1.  当评估用户定义的函数时，在我们的示例中期望的 AST 被称为`FuncCallFromDef`节点。首先，我们定义评估函数并将其命名为`calc_expr_func`（接受零个参数并返回一个结果）：

    ```cpp

      virtual void visit(FuncCallFromDef &Node) override {
        llvm::StringRef CalcExprFunName = "calc_expr_func";
        FunctionType *CalcExprFunTy = FunctionType::get(Int32Ty, {},     false);
        Function *CalcExprFun = Function::Create(
            CalcExprFunTy, GlobalValue::ExternalLinkage,         CalcExprFunName, M);
    ```

1.  接下来，我们创建一个新的基本块以插入`calc_expr_func`：

    ```cpp

        BasicBlock *BB = BasicBlock::Create(M->getContext(),     "entry", CalcExprFun);
        Builder.SetInsertPoint(BB);
    ```

1.  与之前类似，用户定义的函数是通过`genUserDefinedFunction()`检索的，我们将函数调用的数值参数传递给刚刚重新生成的原始函数：

    ```cpp

        llvm::StringRef CalleeFnName = Node.getFnName();
        Function *CalleeFn = genUserDefinedFunction(CalleeFnName);
    ```

1.  一旦我们有了实际的`llvm::Function`实例，我们就利用`IRBuilder`创建对定义的函数的调用，并返回结果，以便在最终将结果打印给用户时可以访问：

    ```cpp

        auto CalleeFnVars = Node.getArgs();
        llvm::SmallVector<Value *> IntParams;
        for (unsigned i = 0, end = CalleeFnVars.size(); i != end;     ++i) {
          int ArgsToIntType;
          CalleeFnVars[i].getAsInteger(10, ArgsToIntType);
          Value *IntParam = ConstantInt::get(Int32Ty, ArgsToIntType,       true);
          IntParams.push_back(IntParam);
        }
        Builder.CreateRet(Builder.CreateCall(CalleeFn, IntParams,     "calc_expr_res"));
      };
    ```

## 构建基于 LLJIT 的计算器

最后，为了编译我们的 JIT 计算器源代码，我们还需要创建一个包含构建描述的`CMakeLists.txt`文件，并将其保存到`Calc.cpp`和我们的其他源文件旁边：

1.  我们将所需的最低 CMake 版本设置为 LLVM 所需的版本，并为项目命名：

    ```cpp

    cmake_minimum_required (VERSION 3.20.0)
    project ("jit")
    ```

1.  需要加载 LLVM 包，并将 LLVM 提供的 CMake 模块目录添加到搜索路径中。然后，我们包含`DetermineGCCCompatible`和`ChooseMSVCCRT`模块，这些模块检查编译器是否具有 GCC 兼容的命令行语法，并确保使用与 LLVM 相同的 C 运行时：

    ```cpp

    find_package(LLVM REQUIRED CONFIG)
    list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
    include(DetermineGCCCompatible)
    include(ChooseMSVCCRT)
    ```

1.  我们还需要添加来自 LLVM 的定义和`include`路径。使用的 LLVM 组件通过函数调用映射到库名称：

    ```cpp

    add_definitions(${LLVM_DEFINITIONS})
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    llvm_map_components_to_libnames(llvm_libs Core OrcJIT
                                              Support native)
    ```

1.  之后，如果确定编译器具有 GCC 兼容的命令行语法，我们还会检查是否启用了运行时类型信息和异常处理。如果没有启用，则相应地添加 C++标志以关闭这些功能：

    ```cpp

    if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
      if(NOT LLVM_ENABLE_RTTI)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
      endif()
      if(NOT LLVM_ENABLE_EH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
      endif()
    endif()
    ```

1.  最后，我们定义了可执行文件名称、要编译的源文件以及要链接的库：

    ```cpp

    add_executable (calc
      Calc.cpp CodeGen.cpp Lexer.cpp Parser.cpp Sema.cpp)
    target_link_libraries(calc PRIVATE ${llvm_libs})
    ```

上述步骤就是我们的基于 JIT 的交互式计算器工具所需的所有步骤。接下来，创建并切换到构建目录，然后运行以下命令以创建和编译应用程序：

```cpp

$ cmake –G Ninja <path to source directory>
$ ninja
```

这会编译`calc`工具。然后我们可以启动计算器，开始定义函数，并查看我们的计算器如何评估我们定义的函数。

以下示例调用显示了首先定义的函数的 IR，然后是创建的`calc_expr_func`函数，该函数用于生成对最初定义的函数的调用，以便使用传递给它的任何参数评估该函数：

```cpp

$ ./calc
JIT calc > def f(x) = x*2
define i32 @f(i32 %x) {
entry:
  %0 = mul nsw i32 %x, 2
  ret i32 %0
}
JIT calc > f(20)
Attempting to evaluate expression:
define i32 @calc_expr_func() {
entry:
  %calc_expr_res = call i32 @f(i32 20)
  ret i32 %calc_expr_res
}
declare i32 @f(i32)
User defined function evaluated to: 40
JIT calc > def g(x,y) = x*y+100
define i32 @g(i32 %x, i32 %y) {
entry:
  %0 = mul nsw i32 %x, %y
  %1 = add nsw i32 %0, 100
  ret i32 %1
}
JIT calc > g(8,9)
Attempting to evaluate expression:
define i32 @calc_expr_func() {
entry:
  %calc_expr_res = call i32 @g(i32 8, i32 9)
  ret i32 %calc_expr_res
}
declare i32 @g(i32, i32)
User defined function evaluated to: 172
```

就这样！我们刚刚创建了一个基于 JIT 的计算器应用程序！

由于我们的 JIT 计算器旨在作为一个简单的示例，说明如何将`LLJIT`集成到我们的项目中，因此值得注意的是存在一些限制：

+   此计算器不接受十进制值的负数

+   我们不能重新定义同一个函数超过一次

对于第二个限制，这是按设计进行的，因此由 ORC API 本身预期并强制执行：

```cpp

$ ./calc
JIT calc > def f(x) = x*2
define i32 @f(i32 %x) {
entry:
  %0 = mul nsw i32 %x, 2
  ret i32 %0
}
JIT calc > def f(x,y) = x+y
define i32 @f(i32 %x, i32 %y) {
entry:
  %0 = add nsw i32 %x, %y
  ret i32 %0
}
Duplicate definition of symbol '_f'
```

请记住，除了暴露当前进程或共享库中的符号之外，还有许多其他方法可以暴露名称。例如，`StaticLibraryDefinitionGenerator`类暴露了静态归档中找到的符号，并可用于`DynamicLibrarySearchGenerator`类。

此外，`LLJIT`类还有一个`addObjectFile()`方法来暴露对象文件的符号。如果现有的实现不符合您的需求，您也可以提供自己的`DefinitionGenerator`实现。

如我们所见，使用预定义的`LLJIT`类很方便，但它可能会限制我们的灵活性。在下一节中，我们将探讨如何使用 ORC API 提供的层来实现 JIT 编译器。

# 从头开始构建 JIT 编译器类

使用 ORC 的分层方法，构建针对特定需求的 JIT 编译器非常容易。没有一种适合所有情况的 JIT 编译器，本章的第一部分给出了一些示例。让我们看看如何从头开始设置 JIT 编译器。

ORC API 使用堆叠在一起的层。最低层是对象链接层，由`llvm::orc::RTDyldObjectLinkingLayer`类表示。它负责将内存中的对象链接起来，并将它们转换为可执行代码。这个任务所需的内存由`MemoryManager`接口的一个实例管理。有一个默认实现，但如果我们需要，也可以使用自定义版本。

在对象链接层之上是编译层，它负责创建内存中的对象文件。`llvm::orc::IRCompileLayer`类接受 IR 模块作为输入，并将其编译为对象文件。`IRCompileLayer`类是`IRLayer`类的子类，`IRLayer`是一个用于接受 LLVM IR 的层实现的通用类。

这两个层已经构成了 JIT 编译器的核心：它们添加一个 LLVM IR 模块作为输入，该模块在内存中编译和链接。为了添加额外的功能，我们可以在两个层之上添加更多的层。

例如，`CompileOnDemandLayer`类将模块分割，以便只编译请求的函数。这可以用于实现懒编译。此外，`CompileOnDemandLayer`类也是`IRLayer`类的子类。以非常通用的方式，`IRTransformLayer`类，也是`IRLayer`类的子类，允许我们对模块应用转换。

另一个重要的类是`ExecutionSession`类。这个类代表一个正在运行的 JIT 程序。本质上，这意味着该类管理`JITDylib`符号表，提供符号的查找功能，并跟踪使用的资源管理器。

JIT 编译器的通用配方如下：

1.  初始化`ExecutionSession`类的一个实例。

1.  初始化层，至少包括`RTDyldObjectLinkingLayer`类和`IRCompileLayer`类。

1.  创建第一个`JITDylib`符号表，通常使用`main`或类似名称。

JIT 编译器的一般用法也非常简单：

1.  将 IR 模块添加到符号表中。

1.  查找符号，触发相关函数的编译，以及可能整个模块的编译。

1.  执行函数。

在下一小节中，我们将按照通用配方实现一个 JIT 编译器类。

## 创建 JIT 编译器类

为了保持 JIT 编译器类的实现简单，所有内容都放置在`JIT.h`中，在一个可以创建的源目录`jit`内。然而，与使用`LLJIT`相比，类的初始化要复杂一些。由于处理可能的错误，我们需要一个工厂方法在调用构造函数之前预先创建一些对象。创建类的步骤如下：

1.  我们首先使用`JIT_H`预处理器定义来保护头文件，防止多次包含：

    ```cpp

    #ifndef JIT_H
    #define JIT_H
    ```

1.  首先，需要一些`include`文件。其中大部分提供与头文件同名的类。`Core.h`头文件提供了一些基本类，包括`ExecutionSession`类。此外，`ExecutionUtils.h`头文件提供了`DynamicLibrarySearchGenerator`类来搜索库中的符号。此外，`CompileUtils.h`头文件提供了`ConcurrentIRCompiler`类：

    ```cpp

    #include "llvm/Analysis/AliasAnalysis.h"
    #include "llvm/ExecutionEngine/JITSymbol.h"
    #include "llvm/ExecutionEngine/Orc/CompileUtils.h"
    #include "llvm/ExecutionEngine/Orc/Core.h"
    #include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
    #include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
    #include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
    #include
         "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
    #include "llvm/ExecutionEngine/Orc/Mangling.h"
    #include
        "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
    #include
            "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
    #include "llvm/ExecutionEngine/SectionMemoryManager.h"
    #include "llvm/Passes/PassBuilder.h"
    #include "llvm/Support/Error.h"
    ```

1.  声明一个新的类。我们的新类将被称为`JIT`：

    ```cpp

    class JIT {
    ```

1.  私有数据成员反映了 ORC 层和一些辅助类。`ExecutionSession`、`ObjectLinkingLayer`、`CompileLayer`、`OptIRLayer`和`MainJITDylib`实例代表正在运行的 JIT 程序、层和符号表，如前所述。此外，`TargetProcessControl`实例用于与 JIT 目标进程交互。这可以是同一个进程，同一台机器上的另一个进程，或者不同机器上的远程进程，可能具有不同的架构。`DataLayout`和`MangleAndInterner`类用于以正确的方式混淆符号名称。此外，符号名称被内部化，这意味着所有相同名称的地址相同。这意味着要检查两个符号名称是否相等，只需比较地址即可，这是一个非常快速的操作：

    ```cpp

      std::unique_ptr<llvm::orc::TargetProcessControl> TPC;
      std::unique_ptr<llvm::orc::ExecutionSession> ES;
      llvm::DataLayout DL;
      llvm::orc::MangleAndInterner Mangle;
      std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer>
          ObjectLinkingLayer;
      std::unique_ptr<llvm::orc::IRCompileLayer>
          CompileLayer;
      std::unique_ptr<llvm::orc::IRTransformLayer>
          OptIRLayer;
      llvm::orc::JITDylib &MainJITDylib;
    ```

1.  初始化被分为三个部分。在 C++中，构造函数不能返回错误。简单且推荐的方法是创建一个静态工厂方法，在构造对象之前进行错误处理。层的初始化更为复杂，因此我们也为它们引入了工厂方法。

    在`create()`工厂方法中，我们首先创建一个`SymbolStringPool`实例，该实例用于实现字符串国际化，并被多个类共享。为了控制当前进程，我们创建一个`SelfTargetProcessControl`实例。如果我们想针对不同的进程，则需要更改这个实例。

    接下来，我们构建一个`JITTargetMachineBuilder`实例，我们需要知道 JIT 进程的目标三元组。之后，我们查询目标机器构建器以获取数据布局。如果构建器无法根据提供的三元组实例化目标机器，则此步骤可能会失败——例如，因为对该目标的支持没有编译到 LLVM 库中：

    ```cpp

    public:
      static llvm::Expected<std::unique_ptr<JIT>> create() {
        auto SSP =
            std::make_shared<llvm::orc::SymbolStringPool>();
        auto TPC =
            llvm::orc::SelfTargetProcessControl::Create(SSP);
        if (!TPC)
          return TPC.takeError();
        llvm::orc::JITTargetMachineBuilder JTMB(
            (*TPC)->getTargetTriple());
        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL)
          return DL.takeError();
    ```

1.  在这一点上，我们已经处理了所有可能失败的调用。现在我们可以初始化`ExecutionSession`实例。最后，调用`JIT`类的构造函数，传入所有实例化的对象，并将结果返回给调用者：

    ```cpp

        auto ES =
            std::make_unique<llvm::orc::ExecutionSession>(
                std::move(SSP));
        return std::make_unique<JIT>(
            std::move(*TPC), std::move(ES), std::move(*DL),
            std::move(JTMB));
      }
    ```

1.  `JIT`类的构造函数将传入的参数移动到私有数据成员中。层对象通过调用具有`create`前缀的静态工厂名称来构建。此外，每个层工厂方法都需要对`ExecutionSession`实例的引用，这将层连接到正在运行的 JIT 会话。除了位于层堆栈底部的对象链接层之外，每个层都需要对前一个层的引用，说明了堆叠顺序：

    ```cpp

    JIT(std::unique_ptr<llvm::orc::ExecutorProcessControl>
              EPCtrl,
          std::unique_ptr<llvm::orc::ExecutionSession>
              ExeS,
          llvm::DataLayout DataL,
          llvm::orc::JITTargetMachineBuilder JTMB)
          : EPC(std::move(EPCtrl)), ES(std::move(ExeS)),
            DL(std::move(DataL)), Mangle(*ES, DL),
            ObjectLinkingLayer(std::move(
                createObjectLinkingLayer(*ES, JTMB))),
            CompileLayer(std::move(createCompileLayer(
                *ES, *ObjectLinkingLayer,
                std::move(JTMB)))),
            OptIRLayer(std::move(
                createOptIRLayer(*ES, *CompileLayer))),
            MainJITDylib(
                ES->createBareJITDylib("<main>")) {
    ```

1.  在构造函数的主体中，我们添加了一个生成器来搜索当前进程中的符号。`GetForCurrentProcess()`方法很特殊，因为返回值被包裹在一个`Expected<>`模板中，表示也可以返回一个`Error`对象。然而，由于我们知道不会发生错误，当前进程最终会运行！因此，我们使用`cantFail()`函数解包结果，如果确实发生了错误，则终止应用程序：

    ```cpp

        MainJITDylib.addGenerator(llvm::cantFail(
            llvm::orc::DynamicLibrarySearchGenerator::
                GetForCurrentProcess(DL.getGlobalPrefix())));
      }
    ```

1.  要创建一个对象链接层，我们需要提供一个内存管理器。在这里，我们坚持使用默认的`SectionMemoryManager`类，但如果需要，我们也可以提供不同的实现：

    ```cpp

      static std::unique_ptr<
          llvm::orc::RTDyldObjectLinkingLayer>
      createObjectLinkingLayer(
          llvm::orc::ExecutionSession &ES,
          llvm::orc::JITTargetMachineBuilder &JTMB) {
        auto GetMemoryManager = []() {
          return std::make_unique<
              llvm::SectionMemoryManager>();
        };
        auto OLLayer = std::make_unique<
            llvm::orc::RTDyldObjectLinkingLayer>(
            ES, GetMemoryManager);
    ```

1.  对于在 Windows 上使用的**通用对象文件格式**（**COFF**）对象文件格式，存在一个轻微的复杂性。此文件格式不允许将函数标记为导出。这随后导致对象链接层内部的检查失败：存储在符号中的标志与 IR 中的标志进行比较，由于缺少导出标记，导致不匹配。解决方案是为此文件格式覆盖标志。这完成了对象层的构建，并将对象返回给调用者：

    ```cpp

        if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
          OLLayer
             ->setOverrideObjectFlagsWithResponsibilityFlags(
                  true);
          OLLayer
             ->setAutoClaimResponsibilityForObjectSymbols(
                  true);
        }
        return OLLayer;
      }
    ```

1.  要初始化编译器层，需要一个`IRCompiler`实例。`IRCompiler`实例负责将 IR 模块编译成对象文件。如果我们的 JIT 编译器不使用线程，则可以使用`SimpleCompiler`类，该类使用给定的目标机器编译 IR 模块。`TargetMachine`类不是线程安全的，因此`SimpleCompiler`类也不是。为了支持多线程编译，我们使用`ConcurrentIRCompiler`类，为每个要编译的模块创建一个新的`TargetMachine`实例。这种方法解决了多线程的问题：

    ```cpp

      static std::unique_ptr<llvm::orc::IRCompileLayer>
      createCompileLayer(
          llvm::orc::ExecutionSession &ES,
          llvm::orc::RTDyldObjectLinkingLayer &OLLayer,
          llvm::orc::JITTargetMachineBuilder JTMB) {
        auto IRCompiler = std::make_unique<
            llvm::orc::ConcurrentIRCompiler>(
            std::move(JTMB));
        auto IRCLayer =
            std::make_unique<llvm::orc::IRCompileLayer>(
                ES, OLLayer, std::move(IRCompiler));
        return IRCLayer;
      }
    ```

1.  我们不是直接将 IR 模块编译成机器代码，而是安装一个先优化 IR 的层。这是一个故意的决策：我们将我们的 JIT 编译器转变为一个优化 JIT 编译器，它产生的代码更快，但生成代码所需的时间更长，这意味着对用户来说会有延迟。我们没有添加懒编译，所以当查找符号时，整个模块都会被编译。这可能会在用户看到代码执行之前增加相当长的时间。

注意

在所有情况下引入懒编译都不是一个合适的解决方案。懒编译是通过将每个函数移动到它自己的模块中实现的，当查找函数名时进行编译。这防止了诸如 *内联* 这样的跨程序优化，因为内联器需要访问被调用函数的体来内联它们。因此，用户会看到懒编译时的启动速度更快，但产生的代码并不像可能的那样优化。这些设计决策取决于预期的用途。在这里，我们决定要快速代码，接受较慢的启动时间。此外，这意味着优化层本质上是一个转换层。

`IRTransformLayer` 类将转换委托给一个函数——在我们的例子中，是 `optimizeModule` 函数：

```cpp

  static std::unique_ptr<llvm::orc::IRTransformLayer>
  createOptIRLayer(
      llvm::orc::ExecutionSession &ES,
      llvm::orc::IRCompileLayer &CompileLayer) {
    auto OptIRLayer =
        std::make_unique<llvm::orc::IRTransformLayer>(
            ES, CompileLayer,
            optimizeModule);
    return OptIRLayer;
  }
```

1.  `optimizeModule()` 函数是一个对 IR 模块进行转换的例子。该函数获取一个模块作为参数，并返回 IR 模块的转换版本。由于 JIT 编译器可能以多线程方式运行，IR 模块被包装在一个 `ThreadSafeModule` 实例中：

    ```cpp

      static llvm::Expected<llvm::orc::ThreadSafeModule>
      optimizeModule(
          llvm::orc::ThreadSafeModule TSM,
          const llvm::orc::MaterializationResponsibility
              &R) {
    ```

1.  为了优化 IR，我们回顾了在 *添加优化管道到您的编译器* 部分的 *第七章* 中的一些信息，*优化 IR*。我们需要一个 `PassBuilder` 实例来创建一个优化管道。首先，我们定义了一对分析管理器，并在之后在管道构建器中注册它们。之后，我们使用 `O2` 级别的默认优化管道填充一个 `ModulePassManager` 实例。这又是一个设计决策：`O2` 级别已经产生了快速的机器代码，但在 `O3` 级别会产生更快代码。接下来，我们在模块上运行管道，最后，将优化后的模块返回给调用者：

    ```cpp

        TSM.withModuleDo([](llvm::Module &M) {
          bool DebugPM = false;
          llvm::PassBuilder PB(DebugPM);
          llvm::LoopAnalysisManager LAM(DebugPM);
          llvm::FunctionAnalysisManager FAM(DebugPM);
          llvm::CGSCCAnalysisManager CGAM(DebugPM);
          llvm::ModuleAnalysisManager MAM(DebugPM);
          FAM.registerPass(
              [&] { return PB.buildDefaultAAPipeline(); });
          PB.registerModuleAnalyses(MAM);
          PB.registerCGSCCAnalyses(CGAM);
          PB.registerFunctionAnalyses(FAM);
          PB.registerLoopAnalyses(LAM);
          PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
          llvm::ModulePassManager MPM =
              PB.buildPerModuleDefaultPipeline(
                  llvm::PassBuilder::OptimizationLevel::O2,
                  DebugPM);
          MPM.run(M, MAM);
        });
        return TSM;
      }
    ```

1.  `JIT` 类的客户端需要一个方法来添加一个 IR 模块，我们通过 `addIRModule()` 函数提供这个功能。回想一下我们创建的层栈：我们必须将 IR 模块添加到顶层；否则，我们可能会意外地跳过一些层。这将是一个不易发现的编程错误：如果将 `OptIRLayer` 成员替换为 `CompileLayer` 成员，那么我们的 `JIT` 类仍然可以工作，但不再是一个优化 JIT，因为我们跳过了这个层。对于这个小实现来说，这不是一个问题，但在大型 JIT 优化中，我们会引入一个函数来返回顶层层：

    ```cpp

      llvm::Error addIRModule(
          llvm::orc::ThreadSafeModule TSM,
          llvm::orc::ResourceTrackerSP RT = nullptr) {
        if (!RT)
          RT = MainJITDylib.getDefaultResourceTracker();
        return OptIRLayer->add(RT, std::move(TSM));
      }
    ```

1.  同样，我们的 JIT 类的客户端需要一个查找符号的方法。我们将此委托给 `ExecutionSession` 实例，传递对主符号表的引用以及请求的符号的混淆和内部化名称：

    ```cpp

      llvm::Expected<llvm::orc::ExecutorSymbolDef>
      lookup(llvm::StringRef Name) {
        return ES->lookup({&MainJITDylib},
                          Mangle(Name.str()));
      }
    ```

如我们所见，此 JIT 类的初始化可能很棘手，因为它涉及 `JIT` 类的工厂方法和构造函数调用，以及每一层的工厂方法。尽管这种分布是由 C++ 的限制造成的，但代码本身是直接的。

接下来，我们将使用新的 JIT 编译器类来实现一个简单的命令行实用程序，该实用程序接受 LLVM IR 文件作为输入。

## 使用我们新的 JIT 编译器类

我们首先创建一个名为 `JIT.cpp` 的文件，与 `JIT.h` 文件位于同一目录下，并将以下内容添加到这个源文件中：

1.  首先，包含几个头文件。我们必须包含 `JIT.h` 以使用我们的新类，以及 `IRReader.h` 头文件，因为它定义了一个用于读取 LLVM IR 文件的功能。`CommandLine.h` 头文件允许我们以 LLVM 风格解析命令行选项。接下来，需要 `InitLLVM.h` 以进行工具的基本初始化。最后，需要 `TargetSelect.h` 以进行本地目标的初始化：

    ```cpp

    #include "JIT.h"
    #include "llvm/IRReader/IRReader.h"
    #include "llvm/Support/CommandLine.h"
    #include "llvm/Support/InitLLVM.h"
    #include "llvm/Support/TargetSelect.h"
    ```

1.  接下来，我们将 `llvm` 命名空间添加到当前作用域中：

    ```cpp

    using namespace llvm;
    ```

1.  我们的 JIT 工具期望命令行上恰好有一个输入文件，我们使用 `cl::opt<>` 类声明它：

    ```cpp

    static cl::opt<std::string>
        InputFile(cl::Positional, cl::Required,
                  cl::desc("<input-file>"));
    ```

1.  要读取 IR 文件，我们调用 `parseIRFile()` 函数。文件可以是文本 IR 表示或位代码文件。该函数返回创建的模块的指针。此外，错误处理略有不同，因为文本 IR 文件可以解析，这并不一定是语法正确的。最后，`SMDiagnostic` 实例在出现语法错误时持有错误信息。在发生错误的情况下，将打印错误信息，并退出应用程序：

    ```cpp

    std::unique_ptr<Module>
    loadModule(StringRef Filename, LLVMContext &Ctx,
               const char *ProgName) {
      SMDiagnostic Err;
      std::unique_ptr<Module> Mod =
          parseIRFile(Filename, Err, Ctx);
      if (!Mod.get()) {
        Err.print(ProgName, errs());
        exit(-1);
      }
      return Mod;
    }
    ```

1.  `jitmain()` 函数放置在 `loadModule()` 方法之后。此函数设置我们的 JIT 引擎并编译一个 LLVM IR 模块。该函数需要执行所需的 LLVM 模块和 IR。此模块还需要 LLVM 上下文类，因为上下文类包含重要的类型信息。目标是调用 `main()` 函数，因此我们还传递了常用的 `argc` 和 `argv` 参数：

    ```cpp

    Error jitmain(std::unique_ptr<Module> M,
                  std::unique_ptr<LLVMContext> Ctx,
                  int argc, char *argv[]) {
    ```

1.  接下来，我们创建我们之前构建的 JIT 类的实例。如果发生错误，则相应地返回错误信息：

    ```cpp

      auto JIT = JIT::create();
      if (!JIT)
        return JIT.takeError();
    ```

1.  然后，我们将模块添加到主 `JITDylib` 实例中，再次将模块和上下文包装在 `ThreadSafeModule` 实例中。如果发生错误，则返回错误信息：

    ```cpp

      if (auto Err = (*JIT)->addIRModule(
              orc::ThreadSafeModule(std::move(M),
                                    std::move(Ctx))))
        return Err;
    ```

1.  此后，我们查找 `main` 符号。此符号必须在命令行上给出的 IR 模块中。查找触发该 IR 模块的编译。如果 IR 模块内部引用了其他符号，则它们将使用之前步骤中添加的生成器进行解析。结果是 `ExecutorAddr` 类，它表示执行进程的地址：

    ```cpp

      llvm::orc::ExecutorAddr MainExecutorAddr = MainSym->getAddress();
      auto *Main = MainExecutorAddr.toPtr<int(int, char**)>();
    ```

1.  现在，我们可以在 IR 模块中调用 `main()` 函数，并传递函数期望的 `argc` 和 `argv` 参数。我们忽略返回值：

    ```cpp

      (void)Main(argc, argv);
    ```

1.  函数执行后，我们报告成功：

    ```cpp

      return Error::success();
    }
    ```

1.  在实现了一个 `jitmain()` 函数之后，我们添加一个 `main()` 函数，该函数初始化工具和本地目标，并解析命令行：

    ```cpp

    int main(int argc, char *argv[]) {
      InitLLVM X(argc, argv);
      InitializeNativeTarget();
      InitializeNativeTargetAsmPrinter();
      InitializeNativeTargetAsmParser();
      cl::ParseCommandLineOptions(argc, argv, "JIT\n");
    ```

1.  之后，初始化了 LLVM 上下文类，并加载了命令行上指定的 IR 模块：

    ```cpp

      auto Ctx = std::make_unique<LLVMContext>();
      std::unique_ptr<Module> M =
          loadModule(InputFile, *Ctx, argv[0]);
    ```

1.  在加载 IR 模块后，我们可以调用 `jitmain()` 函数。为了处理错误，我们使用 `ExitOnError` 实用类在遇到错误时打印错误消息并退出应用程序。我们还设置了一个带有应用程序名称的横幅，该横幅在错误消息之前打印：

    ```cpp

      ExitOnError ExitOnErr(std::string(argv[0]) + ": ");
      ExitOnErr(jitmain(std::move(M), std::move(Ctx),
                        argc, argv));
    ```

1.  如果控制流到达这一点，则表示 IR 已成功执行。我们返回 `0` 以指示成功：

    ```cpp

      return 0;
    }
    ```

现在，我们可以通过编译一个简单的示例来测试我们新实现的 JIT 编译器，该示例将 `Hello World!` 打印到控制台。在底层，新类使用固定的优化级别，因此对于足够大的模块，我们可以注意到启动和运行时的差异。

要构建我们的 JIT 编译器，我们可以遵循与在 *使用 LLJIT 实现自己的 JIT 编译器* 部分接近结尾时相同的 CMake 步骤，我们只需确保 `JIT.cpp` 源文件正在使用正确的库进行编译以进行链接：

```cpp

add_executable(JIT JIT.cpp)
include_directories(${CMAKE_SOURCE_DIR})
target_link_libraries(JIT ${llvm_libs})
```

然后，我们切换到 `build` 目录并编译应用程序：

```cpp

$ cmake –G Ninja <path to jit source directory>
$ ninja
```

我们的 `JIT` 工具现在可以使用了。可以像以下这样用 C 编写一个简单的 `Hello World!` 程序：

```cpp

$ cat main.c
#include <stdio.h>
int main(int argc, char** argv) {
  printf("Hello world!\n");
  return 0;
}
```

接下来，我们可以使用以下命令将 Hello World C 源代码编译成 LLVM IR：

```cpp

$ clang -S -emit-llvm main.c
```

记住 – 我们将 C 源代码编译成 LLVM IR，因为我们的 JIT 编译器接受 IR 文件作为输入。最后，我们可以使用以下方式调用我们的 JIT 编译器：

```cpp

$ JIT main.ll
Hello world!
```

# 摘要

在本章中，你学习了如何开发 JIT 编译器。你从了解 JIT 编译器的可能应用开始，并探索了 `lli`，LLVM 的动态编译器和解释器。使用预定义的 `LLJIT` 类，你构建了一个基于 JIT 的交互式计算器工具，并学习了查找符号和将 IR 模块添加到 `LLJIT` 中。为了能够利用 ORC API 的分层结构，你还实现了一个优化的 `JIT` 类。

在下一章中，你将学习如何利用 LLVM 工具进行调试。
