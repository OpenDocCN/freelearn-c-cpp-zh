# 第四章 准备优化

在本章中，我们将介绍以下内容：

+   不同的优化级别

+   编写自己的 LLVM 插件

+   使用 opt 工具运行自己的插件

+   在新插件中使用另一个插件

+   使用插件管理器注册插件

+   编写一个分析插件

+   编写别名分析插件

+   使用其他分析插件

# 简介

一旦源代码转换完成，输出将以 LLVM IR 形式呈现。这个 IR 作为将代码转换为汇编代码的通用平台，具体取决于后端。然而，在转换为汇编代码之前，IR 可以被优化以生成更有效的代码。IR 是 SSA 形式，其中每个对变量的新赋值都是一个新变量——这是 SSA 表示的经典案例。

在 LLVM 基础设施中，一个插件用于优化 LLVM IR。插件在 LLVM IR 上运行，处理 IR，分析它，识别优化机会，并修改 IR 以生成优化代码。命令行界面 **opt** 用于在 LLVM IR 上运行优化插件。

在接下来的章节中，将讨论各种优化技术。还将探讨如何编写和注册新的优化插件。

# 不同的优化级别

优化级别有多种，从 0 级开始，到 3 级结束（也有 `s` 用于空间优化）。随着优化级别的提高，代码的优化程度也越来越高。让我们尝试探索各种优化级别。

## 准备工作...

通过在 LLVM IR 上运行 opt 命令行界面，可以理解不同的优化级别。为此，可以使用 **Clang** 前端首先将一个示例 C 程序转换为 IR。

1.  打开一个 `example.c` 文件，并在其中编写以下代码：

    ```cpp
    $ vi  example.c
    int main(int argc, char **argv) {
      int i, j, k, t = 0;
      for(i = 0; i < 10; i++) {
        for(j = 0; j < 10; j++) {
          for(k = 0; k < 10; k++) {
            t++;
          }
        }
        for(j = 0; j < 10; j++) {
          t++;
        }
      }
      for(i = 0; i < 20; i++) {
        for(j = 0; j < 20; j++) {
          t++;
        }
        for(j = 0; j < 20; j++) {
          t++;
        }
      }
      return t;
    }
    ```

1.  现在可以使用 `clang` 命令将其转换为 LLVM IR，如下所示：

    ```cpp
    $ clang –S –O0 –emit-llvm example.c

    ```

    将生成一个新的文件，`example.ll`，其中包含 LLVM IR。此文件将用于演示可用的各种优化级别。

## 如何操作…

执行以下步骤：

1.  可以在生成的 IR 文件 `example.ll` 上运行 opt 命令行工具：

    ```cpp
    $ opt –O0 –S example.ll

    ```

    `–O0` 语法指定了最低的优化级别。

1.  同样，您可以运行其他优化级别：

    ```cpp
    $ opt –O1 –S example.ll
    $ opt –O2 –S example.ll
    $ opt –O3 –S example.ll

    ```

## 它是如何工作的…

opt 命令行界面接受 `example.ll` 文件作为输入，并运行每个优化级别中指定的插件系列。它可以在同一优化级别中重复某些插件。要查看每个优化级别中使用的插件，您必须添加 `--debug-pass=Structure` 命令行选项到之前的 opt 命令中。

## 相关内容

+   要了解更多关于可以与 opt 工具一起使用的其他选项，请参阅 [`llvm.org/docs/CommandGuide/opt.html`](http://llvm.org/docs/CommandGuide/opt.html)

# 编写自己的 LLVM 插件

所有 LLVM pass 都是`pass`类的子类，它们通过重写从`pass`继承的虚方法来实现功能。LLVM 对目标程序应用一系列分析和转换。pass 是 Pass LLVM 类的实例。

## 准备工作

让我们看看如何编写一个 pass。让我们把这个 pass 命名为`function block counter`；一旦完成，它将在运行时简单地显示函数的名称并计算该函数中的基本块数量。首先，需要为这个 pass 编写一个`Makefile`。按照以下步骤编写`Makefile`：

1.  在`llvm lib/Transform`文件夹中打开一个`Makefile`：

    ```cpp
    $ vi Makefile

    ```

1.  指定 LLVM 根文件夹的路径和库名称，并在`Makefile`中指定它，如下所示：

    ```cpp
    LEVEL = ../../..
    LIBRARYNAME = FuncBlockCount
    LOADABLE_MODULE = 1
    include $(LEVEL)/Makefile.common
    ```

这个`Makefile`指定当前目录中的所有`.cpp`文件都要编译并链接成一个共享对象。

## 如何做…

执行以下步骤：

1.  创建一个名为`FuncBlockCount.cpp`的新`.cpp`文件：

    ```cpp
    $ vi FuncBlockCount.cpp

    ```

1.  在这个文件中，包含一些来自 LLVM 的头文件：

    ```cpp
    #include "llvm/Pass.h"
    #include "llvm/IR/Function.h"
    #include "llvm/Support/raw_ostream.h"
    ```

1.  包含`llvm`命名空间以启用对 LLVM 函数的访问：

    ```cpp
    using namespace llvm;
    ```

1.  然后从匿名命名空间开始：

    ```cpp
    namespace {
    ```

1.  接下来声明这个 pass：

    ```cpp
    struct FuncBlockCount : public FunctionPass {
    ```

1.  然后声明 pass 标识符，LLVM 将使用它来识别 pass：

    ```cpp
    static char ID;
    FuncBlockCount() : FunctionPass(ID) {}
    ```

1.  这个步骤是编写一个 pass 过程中最重要的步骤之一——编写一个`run`函数。因为这个 pass 继承了`FunctionPass`并在函数上运行，所以定义了一个`runOnFunction`来在函数上运行：

    ```cpp
    bool runOnFunction(Function &F) override {
          errs() << "Function " << F.getName() << '\n';
          return false;
        }
      };
    }
    ```

    这个函数打印正在处理的函数的名称。

1.  下一步是初始化 pass ID：

    ```cpp
    char FuncBlockCount::ID = 0;
    ```

1.  最后，需要注册这个 pass，包括命令行参数和名称：

    ```cpp
    static RegisterPass<FuncBlockCount> X("funcblockcount", "Function Block Count", false, false);
    ```

    将所有内容组合起来，整个代码看起来像这样：

    ```cpp
    #include "llvm/Pass.h"
    #include "llvm/IR/Function.h"
    #include "llvm/Support/raw_ostream.h"
    using namespace llvm;
    namespace {
    struct FuncBlockCount : public FunctionPass {
      static char ID;
      FuncBlockCount() : FunctionPass(ID) {}
      bool runOnFunction(Function &F) override {
        errs() << "Function " << F.getName() << '\n';
        return false;
      }
               };
            }
           char FuncBlockCount::ID = 0;
           static RegisterPass<FuncBlockCount> X("funcblockcount", "Function Block Count", false, false);
    ```

## 工作原理

一个简单的`gmake`命令编译文件，因此会在 LLVM 根目录下生成一个新的文件`FuncBlockCount.so`。这个共享对象文件可以动态加载到 opt 工具中，以便在 LLVM IR 代码上运行。如何在下一节中加载和运行它将会演示。

## 参见

+   要了解更多关于如何从头开始构建 pass 的信息，请访问[`llvm.org/docs/WritingAnLLVMPass.html`](http://llvm.org/docs/WritingAnLLVMPass.html)

# 使用 opt 工具运行自己的 pass

在前面的配方中编写的 pass，即*编写自己的 LLVM pass*，已经准备好在 LLVM IR 上运行。这个 pass 需要动态加载，以便 opt 工具能够识别和执行它。

## 如何做…

执行以下步骤：

1.  在`sample.c`文件中编写 C 测试代码，我们将在下一步将其转换为`.ll`文件：

    ```cpp
    $ vi sample.c

    int foo(int n, int m) {
      int sum = 0;
      int c0;
      for (c0 = n; c0 > 0; c0--) {
        int c1 = m;
        for (; c1 > 0; c1--) {
          sum += c0 > c1 ? 1 : 0;
        }
      }
      return sum;
    }
    ```

1.  使用以下命令将 C 测试代码转换为 LLVM IR：

    ```cpp
    $ clang –O0 –S –emit-llvm sample.c –o sample.ll

    ```

    这将生成一个`sample.ll`文件。

1.  使用以下命令使用 opt 工具运行新的 pass：

    ```cpp
    $ opt  -load (path_to_.so_file)/FuncBlockCount.so  -funcblockcount sample.ll
    ```

    输出将类似于以下内容：

    ```cpp
    Function foo

    ```

## 它是如何工作的…

如前述代码所示，共享对象动态地加载到 opt 命令行工具中并运行 pass。它遍历函数并显示其名称。它不修改 IR。新 pass 的进一步增强将在下一菜谱中演示。

## 参见

+   要了解更多关于 Pass 类的各种类型的信息，请访问[`llvm.org/docs/WritingAnLLVMPass.html#pass-classes-and-requirements`](http://llvm.org/docs/WritingAnLLVMPass.html#pass-classes-and-requirements)

# 在新 pass 中使用另一个 pass

一个 pass 可能需要另一个 pass 来获取一些分析数据、启发式方法或任何此类信息以决定进一步的行动。该 pass 可能只需要一些分析，如内存依赖，或者它可能还需要修改后的 IR。你刚刚看到的新的 pass 只是简单地打印出函数的名称。让我们看看如何增强它以计算循环中的基本块数量，这同时也展示了如何使用其他 pass 的结果。

## 准备工作

在上一个菜谱中使用的代码保持不变。然而，为了增强它——如下一节所示——以便它能够计算 IR 中的基本块数量，需要进行一些修改。

## 如何做到这一点…

`getAnalysis`函数用于指定将使用哪个其他 pass：

1.  由于新的 pass 将计算基本块的数量，它需要循环信息。这通过使用`getAnalysis`循环函数来指定：

    ```cpp
     LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    ```

1.  这将调用`LoopInfo` pass 以获取有关循环的信息。遍历此对象提供了基本块信息：

    ```cpp
    unsigned num_Blocks = 0;
      Loop::block_iterator bb;
      for(bb = L->block_begin(); bb != L->block_end();++bb)
        num_Blocks++;
      errs() << "Loop level " << nest << " has " << num_Blocks
    << " blocks\n";
    ```

1.  这将遍历循环以计算其内部的基本块数量。然而，它只计算最外层循环中的基本块。要获取最内层循环的信息，递归调用`getSubLoops`函数将有所帮助。将逻辑放入单独的函数并递归调用它更有意义：

    ```cpp
    void countBlocksInLoop(Loop *L, unsigned nest) {
      unsigned num_Blocks = 0;
      Loop::block_iterator bb;
      for(bb = L->block_begin(); bb != L->block_end();++bb)
        num_Blocks++;
      errs() << "Loop level " << nest << " has " << num_Blocks
    << " blocks\n";
      std::vector<Loop*> subLoops = L->getSubLoops();
      Loop::iterator j, f;
      for (j = subLoops.begin(), f = subLoops.end(); j != f;
    ++j)
        countBlocksInLoop(*j, nest + 1);
    }

    virtual bool runOnFunction(Function &F) {
      LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      errs() << "Function " << F.getName() + "\n";
      for (Loop *L : *LI)
        countBlocksInLoop(L, 0);
      return false;
    }
    ```

## 它是如何工作的…

新修改的 pass 现在需要在样本程序上运行。按照以下步骤修改并运行样本程序：

1.  打开`sample.c`文件，并用以下程序替换其内容：

    ```cpp
    int main(int argc, char **argv) {
      int i, j, k, t = 0;
      for(i = 0; i < 10; i++) {
        for(j = 0; j < 10; j++) {
          for(k = 0; k < 10; k++) {
            t++;
          }
        }
        for(j = 0; j < 10; j++) {
          t++;
        }
      }
      for(i = 0; i < 20; i++) {
        for(j = 0; j < 20; j++) {
          t++;
        }
        for(j = 0; j < 20; j++) {
          t++;
        }
      }
      return t;
    }
    ```

1.  使用 Clang 将其转换为`.ll`文件：

    ```cpp
    $ clang –O0 –S –emit-llvm sample.c –o sample.ll

    ```

1.  在先前的样本程序上运行新的 pass：

    ```cpp
    $ opt  -load (path_to_.so_file)/FuncBlockCount.so  -funcblockcount sample.ll

    ```

    输出将类似于以下内容：

    ```cpp
    Function main
    Loop level 0 has 11 blocks
    Loop level 1 has 3 blocks
    Loop level 1 has 3 blocks
    Loop level 0 has 15 blocks
    Loop level 1 has 7 blocks
    Loop level 2 has 3 blocks
    Loop level 1 has 3 blocks

    ```

## 更多内容…

LLVM 的 pass manager 提供了一个调试 pass 选项，它给我们机会看到哪些 pass 与我们的分析和优化交互，如下所示：

```cpp
$ opt  -load (path_to_.so_file)/FuncBlockCount.so  -funcblockcount sample.ll –disable-output –debug-pass=Structure

```

# 使用 pass manager 注册一个 pass

到目前为止，一个新的 pass 是一个独立运行的动态对象。opt 工具由一系列这样的 pass 组成，这些 pass 已注册到 pass manager 和 LLVM 中。让我们看看如何将我们的 pass 注册到 Pass Manager 中。

## 准备工作

`PassManager`类接受一个 pass 列表，确保它们的先决条件设置正确，然后安排 pass 以高效运行。Pass Manager 执行两个主要任务以尝试减少一系列 pass 的执行时间：

+   共享分析结果，以尽可能避免重新计算分析结果

+   通过流水线传递的执行，将程序中的传递执行流水线化，以通过流水线传递来获得一系列传递更好的缓存和内存使用行为

## 如何做…

按照给定步骤使用 Pass Manager 注册传递：

1.  在`FuncBlockCount.cpp`文件中定义`DEBUG_TYPE`宏，指定调试名称：

    ```cpp
    #define DEBUG_TYPE "func-block-count"
    ```

1.  在`FuncBlockCount`结构中，指定`getAnalysisUsage`语法如下：

    ```cpp
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<LoopInfoWrapperPass>();
      }
    ```

1.  现在初始化新传递的宏：

    ```cpp
    INITIALIZE_PASS_BEGIN(FuncBlockCount, " funcblockcount ",
                         "Function Block Count", false, false)
    INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)

    INITIALIZE_PASS_END(FuncBlockCount, "funcblockcount",
                       "Function Block Count", false, false)

    Pass *llvm::createFuncBlockCountPass() { return new FuncBlockCount(); }
    ```

1.  在位于`include/llvm/`的`LinkAllPasses.h`文件中添加`createFuncBlockCount`传递函数：

    ```cpp
    (void) llvm:: createFuncBlockCountPass ();

    ```

1.  将声明添加到位于`include/llvm/Transforms`的`Scalar.h`文件中：

    ```cpp
    Pass * createFuncBlockCountPass ();

    ```

1.  还要修改传递的构造函数：

    ```cpp
    FuncBlockCount() : FunctionPass(ID) {initializeFuncBlockCount Pass (*PassRegistry::getPassRegistry());}
    ```

1.  在位于`lib/Transforms/Scalar/`的`Scalar.cpp`文件中，添加初始化传递条目：

    ```cpp
    initializeFuncBlockCountPass (Registry);
    ```

1.  将此初始化声明添加到位于`include/llvm/`的`InitializePasses.h`文件中：

    ```cpp
    void initializeFuncBlockCountPass (Registry);
    ```

1.  最后，将`FuncBlockCount.cpp`文件名添加到位于`lib/Transforms/Scalar/`的`CMakeLists.txt`文件中：

    ```cpp
    FuncBlockCount.cpp
    ```

## 它是如何工作的…

使用指定在第一章的`cmake`命令编译 LLVM，*LLVM 设计和使用*。Pass Manager 将包括此传递在 opt 命令行工具的传递管道中。此外，此传递可以从命令行独立运行：

```cpp
$ opt –funcblockcount sample.ll

```

## 参见

+   要了解如何在 Pass Manager 中简单步骤添加传递，请研究[`llvm.org/viewvc/llvm-project/llvm/trunk/lib/Transforms/Scalar/LoopInstSimplify.cpp`](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Transforms/Scalar/LoopInstSimplify.cpp)中的 LoopInstSimplify 传递

# 编写分析传递

分析传递提供了关于 IR 的高级信息，但实际上并不改变 IR。分析传递提供的结果可以被另一个分析传递使用来计算其结果。此外，一旦分析传递计算出结果，其结果可以被不同的传递多次使用，直到运行此传递的 IR 被更改。在本食谱中，我们将编写一个分析传递，用于计算并输出函数中使用的指令码数量。

## 准备工作

首先，我们编写将要运行传递的测试代码：

```cpp
$ cat testcode.c
int func(int a, int b){
  int sum = 0;
  int iter;
  for (iter = 0; iter < a; iter++) {
    int iter1;
    for (iter1 = 0; iter1 < b; iter1++) {
      sum += iter > iter1 ? 1 : 0;
    }
  }
  return sum;
}
```

将其转换为`.bc`文件，我们将将其用作分析传递的输入：

```cpp
$ clang -c -emit-llvm testcode.c -o testcode.bc

```

现在创建包含传递源代码的文件，位于`llvm_root_dir/lib/Transforms/opcodeCounter`。在这里，`opcodeCounter`是我们创建的目录，我们的传递源代码将驻留于此。

进行必要的`Makefile`更改，以便此传递可以编译。

## 如何做…

现在让我们开始编写我们的分析传递的源代码：

1.  包含必要的头文件并使用`llvm`命名空间：

    ```cpp
    #define DEBUG_TYPE "opcodeCounter"
    #include "llvm/Pass.h"
    #include "llvm/IR/Function.h"
    #include "llvm/Support/raw_ostream.h"
    #include <map>
    using namespace llvm;
    ```

1.  创建定义传递的结构：

    ```cpp
    namespace {
    struct CountOpcode: public FunctionPass {
    ```

1.  在结构中，创建必要的用于计算指令码数量和表示传递 ID 的数据结构：

    ```cpp
    std::map< std::string, int> opcodeCounter;
    static char ID;
    CountOpcode () : FunctionPass(ID) {}
    ```

1.  在前面的结构中，编写过程的实际实现代码，重载`runOnFunction`函数：

    ```cpp
    virtual bool runOnFunction (Function &F) {
     llvm::outs() << "Function " << F.getName () << '\n';
    for ( Function::iterator bb = F.begin(), e = F.end(); bb != e; ++bb) {
      for ( BasicBlock::iterator i = bb->begin(), e = bb->end(); i!= e; ++i) {
        if(opcodeCounter.find(i->getOpcodeName()) == opcodeCounter.end()) {
        opcodeCounter[i->getOpcodeName()] = 1;
        } else {
        opcodeCounter[i->getOpcodeName()] += 1;
        }
      }
    }

    std::map< std::string, int>::iterator i = opcodeCounter.begin();
    std::map< std::string, int>::iterator e = opcodeCounter.end();
    while (i != e) {
      llvm::outs()  << i->first << ": " << i->second << "\n";
      i++;
    }
    llvm::outs()  << "\n";
    opcodeCounter.clear();
    return false;
    }
    };
    }
    ```

1.  编写注册过程的代码：

    ```cpp
    char CountOpcode::ID = 0;
    static RegisterPass<CountOpcode> X("opcodeCounter", "Count number of opcode in a functions");

    ```

1.  使用`make`或`cmake`命令编译此过程。

1.  使用 opt 工具在测试代码上运行此过程，以获取函数中存在的操作码数量的信息：

    ```cpp
    $ opt -load path-to-build-folder/lib/LLVMCountopcodes.so -opcodeCounter -disable-output testcode.bc
    Function func
    add: 3
    alloca: 5
    br: 8
    icmp: 3
    load: 10
    ret: 1
    select: 1
    store: 8

    ```

## 它是如何工作的…

此分析过程在函数级别上工作，为程序中的每个函数运行一次。因此，在声明`CountOpcodes : public FunctionPass`结构时，我们继承了`FunctionPass`函数。

`opcodeCounter`函数记录函数中使用的每个操作码的数量。在下面的循环中，我们从所有函数中收集操作码：

```cpp
for (Function::iterator bb = F.begin(), e = F.end(); bb != e; ++bb) {
for (BasicBlock::iterator i = bb->begin(), e = bb->end(); i != e; ++i) {
```

第一个`for`循环遍历函数中存在的所有基本块，第二个`for`循环遍历基本块中存在的所有指令。

第一个`for`循环中的代码是实际收集操作码及其数量的代码。`for`循环下面的代码是为了打印结果。由于我们使用映射来存储结果，我们遍历它以打印函数中操作码名称及其数量的配对。

我们返回`false`，因为我们没有在测试代码中修改任何内容。代码的最后两行是为了将此过程注册为给定名称，以便 opt 工具可以使用此过程。

最后，在执行测试代码时，我们得到函数中使用的不同操作码及其数量。

# 编写别名分析过程

别名分析是一种技术，通过它我们可以知道两个指针是否指向同一位置——也就是说，是否可以通过多种方式访问同一位置。通过获取此分析的结果，您可以决定进一步的优化，例如公共子表达式消除。有不同方式和算法可以执行别名分析。在本配方中，我们不会处理这些算法，但我们将了解 LLVM 如何提供编写自己的别名分析过程的基础设施。在本配方中，我们将编写一个别名分析过程，以了解如何开始编写此类过程。我们不会使用任何特定算法，但在分析的每个情况下都会返回`MustAlias`响应。

## 准备工作

编写用于别名分析的测试代码。在这里，我们将使用之前配方中使用的`testcode.c`文件作为测试代码。

进行必要的`Makefile`更改，通过在`llvm/lib/Analysis/Analysis.cpp`、`llvm/include/llvm/InitializePasses.h`、`llvm/include/llvm/LinkAllPasses.h`和`llvm/include/llvm/Analysis/Passes.h`中添加过程条目来注册过程，并在`llvm_source_dir/lib/Analysis/`下创建一个名为`EverythingMustAlias.cpp`的文件，该文件将包含我们过程的源代码。

## 如何做到这一点...

执行以下步骤：

1.  包含必要的头文件并使用`llvm`命名空间：

    ```cpp
    #include "llvm/Pass.h"
    #include "llvm/Analysis/AliasAnalysis.h"
    #include "llvm/IR/DataLayout.h"
    #include "llvm/IR/LLVMContext.h"
    #include "llvm/IR/Module.h"
    using namespace llvm;
    ```

1.  通过继承 `ImmutablePass` 和 `AliasAnalysis` 类来为我们的传递创建一个结构：

    ```cpp
    namespace {
    struct EverythingMustAlias : public ImmutablePass, public AliasAnalysis {
    ```

1.  声明数据结构和构造函数：

    ```cpp
    static char ID;
    EverythingMustAlias() : ImmutablePass(ID) {}
    initializeEverythingMustAliasPass(*PassRegistry::getPassRegistry());}
    ```

1.  实现用于获取调整后的分析指针的 `getAdjustedAnalysisPointer` 函数：

    ```cpp
        void *getAdjustedAnalysisPointer(const void *ID) override {
          if (ID == &AliasAnalysis::ID)
            return (AliasAnalysis*)this;
          return this;
        }
    ```

1.  实现用于初始化传递的 `initializePass` 函数：

    ```cpp
    bool doInitialization(Module &M) override {
         DL = &M.getDataLayout();
          return true;
        }
    ```

1.  实现用于 `alias` 的函数：

    ```cpp
    void *getAdjustedAnalysisPointer(const void *ID) override {
          if (ID == &AliasAnalysis::ID)
            return (AliasAnalysis*)this;
          return this;
        }
    };
    }
    ```

1.  注册传递：

    ```cpp
    char EverythingMustAlias::ID = 0;
    INITIALIZE_AG_PASS(EverythingMustAlias, AliasAnalysis, "must-aa",
    "Everything Alias (always returns 'must' alias)", true, true, true)

    ImmutablePass *llvm::createEverythingMustAliasPass() { return new EverythingMustAlias(); }
    ```

1.  使用 `cmake` 或 `make` 命令编译传递：

1.  使用编译传递后形成的 `.so` 文件执行测试代码：

    ```cpp
    $ opt  -must-aa -aa-eval -disable-output testcode.bc
    ===== Alias Analysis Evaluator Report =====
     10 Total Alias Queries Performed
     0 no alias responses (0.0%)
     0 may alias responses (0.0%)
     0 partial alias responses (0.0%)
     10 must alias responses (100.0%)
     Alias Analysis Evaluator Pointer Alias Summary: 0%/0%/0%/100%
     Alias Analysis Mod/Ref Evaluator Summary: no mod/ref!

    ```

## 它是如何工作的…

`AliasAnalysis` 类提供了各种别名分析实现应支持的接口。它导出 `AliasResult` 和 `ModRefResult` 枚举，分别表示 `alias` 和 `modref` 查询的结果。

`alias` 方法用于检查两个内存对象是否指向同一位置。它接受两个内存对象作为输入，并返回适当的 `MustAlias`、`PartialAlias`、`MayAlias` 或 `NoAlias`。

`getModRefInfo` 方法返回有关指令执行是否可以读取或修改内存位置的信息。前面示例中的传递通过为每一对指针返回值 `MustAlias` 来工作，正如我们实现的那样。在这里，我们继承了 `ImmutablePasses` 类，它适合我们的传递，因为它是一个非常基础的传递。我们继承了 `AliasAnalysis` 传递，它为我们提供了实现接口。

当传递通过多重继承实现分析接口时，使用 `getAdjustedAnalysisPointer` 函数。如果需要，它应该覆盖此方法以调整指针以满足指定的传递信息。

`initializePass` 函数用于初始化包含 `InitializeAliasAnalysis` 方法的传递，该方法应包含实际的别名分析实现。

`getAnalysisUsage` 方法用于通过显式调用 `AliasAnalysis::getAnalysisUsage` 方法来声明对其他传递的任何依赖。

`alias` 方法用于确定两个内存对象是否相互别名。它接受两个内存对象作为输入，并返回适当的 `MustAlias`、`PartialAlias`、`MayAlias` 或 `NoAlias` 响应。

`alias` 方法之后的代码用于注册传递。最后，当我们使用此传递覆盖测试代码时，我们得到 10 个 `MustAlias` 响应（`100.0%`），正如我们在传递中实现的那样。

## 参见

要更详细地了解 LLVM 别名分析，请参阅 [`llvm.org/docs/AliasAnalysis.html`](http://llvm.org/docs/AliasAnalysis.html)。

# 使用其他分析传递

在这个菜谱中，我们将简要了解由 LLVM 提供的其他分析传递，这些传递可以用于获取关于基本块、函数、模块等的分析信息。我们将查看已经实现在内的传递，以及我们如何为我们的目的使用它们。我们不会查看所有传递，而只会查看其中的一些。

## 准备中…

在 `testcode1.c` 文件中编写测试代码，该文件将用于分析目的：

```cpp
$ cat testcode1.c
void func() {
int i;
char C[2];
char A[10];
for(i = 0; i != 10; ++i) {
  ((short*)C)[0] = A[i];
  C[1] = A[9-i];
}
}
```

使用以下命令行将 C 代码转换为位码格式：

```cpp
$ clang -c -emit-llvm testcode1.c -o testcode1.bc

```

## 如何做到这一点...

按照给出的步骤使用其他分析遍历：

1.  通过将 `–aa-eval` 作为命令行选项传递给 opt 工具来使用别名分析评估遍历：

    ```cpp
    $ opt -aa-eval -disable-output testcode1.bc
    ===== Alias Analysis Evaluator Report =====
    36 Total Alias Queries Performed
    0 no alias responses (0.0%)
    36 may alias responses (100.0%) 
    0 partial alias responses (0.0%)
    0 must alias responses (0.0%)
    Alias Analysis Evaluator Pointer Alias Summary: 0%/100%/0%/0%
    Alias Analysis Mod/Ref Evaluator Summary: no mod/ref!

    ```

1.  使用 `–print-dom-info` 命令行选项与 opt 一起打印支配树信息：

    ```cpp
    $ opt  -print-dom-info -disable-output testcode1.bc
    =============================--------------------------------
    Inorder Dominator Tree:
     [1] %0 {0,9}
     [2] %1 {1,8}
     [3] %4 {2,5}
     [4] %19 {3,4}
     [3] %22 {6,7}

    ```

1.  使用 `–count-aa` 命令行选项与 opt 一起计算一个遍历对另一个遍历发出的查询次数：

    ```cpp
    $ opt -count-aa -basicaa -licm -disable-output testcode1.bc
    No alias:    [4B] i32* %i, [1B] i8* %7
    No alias:    [4B] i32* %i, [2B] i16* %12
    No alias:    [1B] i8* %7, [2B] i16* %12
    No alias:    [4B] i32* %i, [1B] i8* %16
    Partial alias:    [1B] i8* %7, [1B] i8* %16
    No alias:    [2B] i16* %12, [1B] i8* %16
    Partial alias:    [1B] i8* %7, [1B] i8* %16
    No alias:    [4B] i32* %i, [1B] i8* %18
    No alias:    [1B] i8* %18, [1B] i8* %7
    No alias:    [1B] i8* %18, [1B] i8* %16
    Partial alias:    [2B] i16* %12, [1B] i8* %18
    Partial alias:    [2B] i16* %12, [1B] i8* %18

    ===== Alias Analysis Counter Report =====
     Analysis counted:
     12 Total Alias Queries Performed
     8 no alias responses (66%)
     0 may alias responses (0%)
     4 partial alias responses (33%)
     0 must alias responses (0%)
     Alias Analysis Counter Summary: 66%/0%/33%/0%

     0 Total Mod/Ref Queries Performed

    ```

1.  使用 `-print-alias-sets` 命令行选项在程序中打印别名集，带上 opt：

    ```cpp
    $ opt  -basicaa -print-alias-sets -disable-output testcode1.bc
    Alias Set Tracker: 3 alias sets for 5 pointer values.
     AliasSet[0x336b120, 1] must alias, Mod/Ref   Pointers: (i32* %i, 4)
     AliasSet[0x336b1c0, 2] may alias, Ref       Pointers: (i8* %7, 1), (i8* %16, 1)
     AliasSet[0x338b670, 2] may alias, Mod       Pointers: (i16* %12, 2), (i8* %18, 1)

    ```

## 它是如何工作的...

在第一种情况下，当我们使用 `-aa-eval` 选项时，opt 工具运行别名分析评估遍历，该遍历将分析输出到屏幕上。它遍历函数中所有指针对，并查询这两个指针是否是别名。

使用 `-print-dom-info` 选项，运行打印支配树的遍历，通过这个遍历可以获得关于支配树的信息。

在第三种情况下，我们执行 `opt -count-aa -basicaa –licm` 命令。`count-aa` 命令选项计算 `licm` 遍历对 `basicaa` 遍历发出的查询次数。这个信息是通过 opt 工具的计数别名分析遍历获得的。

要打印程序中的所有别名集，我们使用 `- print-alias-sets` 命令行选项。在这种情况下，它将打印出使用 `basicaa` 遍历分析后获得的别名集。

## 参见

参考以下链接了解此处未提及的更多遍历：[`llvm.org/docs/Passes.html#anal`](http://llvm.org/docs/Passes.html#anal)。
