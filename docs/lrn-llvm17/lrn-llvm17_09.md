

# 第七章：优化 IR

LLVM 使用一系列的 passes 来优化 IR。一个 pass 操作于 IR 的一个单元，例如一个函数或一个模块。操作可以是转换，以定义的方式改变 IR，或者分析，收集如依赖等信息。这个一系列的 passes 被称为**pass pipeline**。pass 管理器在 IR 上执行 pass pipeline，这是我们的编译器产生的。因此，你需要了解 pass 管理器做什么以及如何构建一个 pass pipeline。编程语言的语义可能需要开发新的 passes，我们必须将这些 passes 添加到管道中。

在本章中，你将学习以下内容：

+   如何利用 LLVM pass 管理器在 LLVM 中实现 passes

+   如何在 LLVM 项目中实现一个作为示例的 instrumentation pass，以及一个独立的插件

+   在使用 LLVM 工具中的 pprofiler pass 时，你将学习如何使用`opt`和`clang`与 pass 插件一起使用

+   在向你的编译器添加优化 pipeline 时，你将使用基于新 pass 管理器的优化 pipeline 扩展`tinylang`编译器。

到本章结束时，你将了解如何开发一个新的 pass，以及如何将其添加到 pass 管道中。你还将能够在你的编译器中设置 pass 管道。

# 技术要求

本章的源代码可在[`github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter07`](https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter07)找到。

# LLVM 的 Pass 管理器

LLVM 核心库优化编译器创建的 IR，并将其转换为对象代码。这个巨大的任务被分解为称为**passes**的单独步骤。这些 pass 需要按正确的顺序执行，这是 pass 管理器的目标。

为什么不直接硬编码 pass 的顺序呢？你的编译器的用户通常期望编译器提供不同级别的优化。开发者更倾向于在开发时优先考虑快速的编译速度，而不是优化。最终的应用程序应该尽可能快地运行，并且你的编译器应该能够执行复杂的优化，接受更长的编译时间。不同级别的优化意味着需要执行不同数量的优化 pass。因此，作为一个编译器编写者，你可能想利用你对源语言的知识来提供自己的 pass。例如，你可能想用内联 IR 或预计算的結果来替换已知的库函数。对于 C 语言，这样的 pass 是 LLVM 库的一部分，但对于其他语言，你可能需要自己提供。在引入自己的 pass 之后，你可能需要重新排序或添加一些 pass。例如，如果你知道你的 pass 操作会留下一些不可达的 IR 代码，那么你希望在你的 pass 之后额外运行死代码删除 pass。pass 管理器帮助组织这些需求。

流程通常根据其工作的范围进行分类：

+   一个 *模块流程* 以整个模块作为输入。此类流程在其给定的模块上执行其工作，并可用于此模块内的过程内操作。

+   一个 *调用图流程* 在调用图的 **强连通分量**（**SCCs**）上操作。它按自下而上的顺序遍历这些组件。

+   一个 *函数流程* 以单个函数作为输入，并且只在此函数上执行其工作。

+   一个 *循环流程* 在函数内的循环上工作。

除了 IR 代码之外，流程还可能需要、更新或使某些分析结果无效。执行了许多不同的分析，例如别名分析或构建支配树。如果流程需要此类分析，则可以从分析管理器请求它。如果信息已经计算，则将返回缓存的結果。否则，将计算信息。如果流程更改 IR 代码，则需要宣布哪些分析结果是保留的，以便在必要时使缓存的分析信息无效。

在底层，流程管理器确保以下内容：

+   分析结果在流程之间共享。这需要跟踪哪个流程需要哪种分析以及每个分析的状态。目标是避免不必要的分析预计算，并尽快释放分析结果占用的内存。

+   流程以管道方式执行。例如，如果应该按顺序执行多个函数流程，则流程管理器将每个这些函数流程运行在第一个函数上。然后，它将运行所有函数流程在第二个函数上，依此类推。这里的底层思想是改善缓存行为，因为编译器只对有限的数据集（一个 IR 函数）执行转换，然后转到下一个有限的数据集。

让我们实现一个新的 IR 转换流程，并探索如何将其添加到优化流程中。

# 实现一个新的流程

流程可以对 LLVM IR 执行任意复杂的转换。为了说明添加新流程的机制，我们添加了一个执行简单仪表化的流程。

为了调查程序的性能，了解函数被调用的频率以及它们运行的时间很有趣。收集这些数据的一种方法是在每个函数中插入计数器。这个过程被称为 `ppprofiler`。我们将开发新的流程，使其可以作为独立插件使用，或者作为插件添加到 LLVM 源树中。之后，我们将探讨随 LLVM 一起提供的流程如何集成到框架中。

## 将 ppprofiler 流程作为插件开发

在本节中，我们将探讨如何从 LLVM 树中创建一个新的作为插件的`pass`。新`pass`的目标是在函数的入口处插入对`__ppp_enter()`函数的调用，并在每个返回指令之前插入对`__ppp_exit()`函数的调用。仅传递当前函数的名称作为参数。这些函数的实现可以计算调用次数并测量经过的时间。我们将在本章末尾实现这个运行时支持。我们将检查如何开发这个`pass`。

我们将源代码存储在`PPProfiler.cpp`文件中。按照以下步骤操作：

1.  首先，让我们包含一些文件：

    ```cpp

    #include "llvm/ADT/Statistic.h"
    #include "llvm/IR/Function.h"
    #include "llvm/IR/PassManager.h"
    #include "llvm/Passes/PassBuilder.h"
    #include "llvm/Passes/PassPlugin.h"
    #include "llvm/Support/Debug.h"
    ```

1.  为了缩短源代码，我们将告诉编译器我们正在使用`llvm`命名空间：

    ```cpp

    using namespace llvm;
    ```

1.  LLVM 的内置调试基础设施要求我们定义一个调试类型，这是一个字符串。这个字符串稍后会在打印的统计信息中显示：

    ```cpp

    #define DEBUG_TYPE "ppprofiler"
    ```

1.  接下来，我们将定义一个带有`ALWAYS_ENABLED_STATISTIC`宏的计数器变量。第一个参数是计数器变量的名称，而第二个参数是将在统计中打印的文本：

    ```cpp

    ALWAYS_ENABLED_STATISTIC(
        NumOfFunc, "Number of instrumented functions.");
    ```

注意

可以使用两个宏来定义计数器变量。如果你使用`STATISTIC`宏，那么统计值仅在调试构建中收集，如果启用了断言，或者在 CMake 命令行上设置了`LLVM_FORCE_ENABLE_STATS`为`ON`。如果你使用`ALWAYS_ENABLED_STATISTIC`宏，那么统计值总是被收集。然而，使用`–stats`命令行选项打印统计信息仅适用于前一种方法。如果需要，你可以通过调用`llvm::PrintStatistics(llvm::raw_ostream)`函数来打印收集到的统计信息。

1.  接下来，我们必须在匿名命名空间中声明`pass`类。该类继承自`PassInfoMixin`模板。这个模板仅添加了一些样板代码，例如`name()`方法。它不用于确定`pass`的类型。当`pass`执行时，`run()`方法会被 LLVM 调用。我们还需要一个名为`instrument()`的辅助方法：

    ```cpp

    namespace {
    class PPProfilerIRPass
        : public llvm::PassInfoMixin<PPProfilerIRPass> {
    public:
      llvm::PreservedAnalyses
      run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
    private:
      void instrument(llvm::Function &F,
                      llvm::Function *EnterFn,
                      llvm::Function *ExitFn);
    };
    }
    ```

1.  现在，让我们定义如何对函数进行仪器化。除了要仪器化的函数外，还需要传递要调用的函数：

    ```cpp

    void PPProfilerIRPass::instrument(llvm::Function &F,
                                      Function *EnterFn,
                                      Function *ExitFn) {
    ```

1.  在函数内部，我们更新统计计数器：

    ```cpp

      ++NumOfFunc;
    ```

1.  为了方便插入 IR 代码，我们需要`IRBuilder`类的实例。我们将它设置到第一个基本块，即函数的入口块：

    ```cpp

      IRBuilder<> Builder(&*F.getEntryBlock().begin());
    ```

1.  现在我们有了构建器，我们可以插入一个全局常量，该常量包含我们希望进行仪器化的函数的名称：

    ```cpp

      GlobalVariable *FnName =
          Builder.CreateGlobalString(F.getName());
    ```

1.  接下来，我们将插入对`__ppp_enter()`函数的调用，并将名称作为参数传递：

    ```cpp

      Builder.CreateCall(EnterFn->getFunctionType(), EnterFn,
                         {FnName});
    ```

1.  要调用`__ppp_exit()`函数，我们必须定位所有返回指令。方便的是，由调用`SetInsertionPoint()`函数设置的插入点在传递的指令之前，因此我们只需在那个点插入调用即可：

    ```cpp

      for (BasicBlock &BB : F) {
        for (Instruction &Inst : BB) {
          if (Inst.getOpcode() == Instruction::Ret) {
            Builder.SetInsertPoint(&Inst);
            Builder.CreateCall(ExitFn->getFunctionType(),
                               ExitFn, {FnName});
          }
        }
      }
    }
    ```

1.  接下来，我们将实现 `run()` 方法。LLVM 通过模块传递我们的通过，以及一个分析管理器，如果需要，我们可以从中请求分析结果：

    ```cpp

    PreservedAnalyses
    PPProfilerIRPass::run(Module &M,
                          ModuleAnalysisManager &AM) {
    ```

1.  这里有一点小麻烦：如果包含 `__ppp_enter()` 和 `__ppp_exit()` 函数实现的运行时模块被仪器化，那么我们会遇到麻烦，因为我们创建了一个无限递归。为了避免这种情况，如果这些函数中的任何一个被定义，我们必须简单地什么也不做：

    ```cpp

      if (M.getFunction("__ppp_enter") ||
          M.getFunction("__ppp_exit")) {
        return PreservedAnalyses::all();
      }
    ```

1.  现在，我们准备声明函数。这里没有什么不寻常的：首先创建函数类型，然后是函数：

    ```cpp

      Type *VoidTy = Type::getVoidTy(M.getContext());
      PointerType *PtrTy =
          PointerType::getUnqual(M.getContext());
      FunctionType *EnterExitFty =
          FunctionType::get(VoidTy, {PtrTy}, false);
      Function *EnterFn = Function::Create(
          EnterExitFty, GlobalValue::ExternalLinkage,
          "__ppp_enter", M);
      Function *ExitFn = Function::Create(
          EnterExitFty, GlobalValue::ExternalLinkage,
          "__ppp_exit", M);
    ```

1.  现在我们需要做的就是遍历模块中的所有函数，并通过调用我们的 `instrument()` 方法来对找到的函数进行仪器化。当然，我们需要忽略函数声明，因为它们只是原型。也可能存在没有名称的函数，这不适合我们的方法。我们也会过滤掉这些函数：

    ```cpp

      for (auto &F : M.functions()) {
        if (!F.isDeclaration() && F.hasName())
          instrument(F, EnterFn, ExitFn);
      }
    ```

1.  最后，我们必须声明我们没有保留任何分析。这可能是过于悲观了，但通过这样做我们可以确保安全：

    ```cpp

      return PreservedAnalyses::none();
    }
    ```

    我们新通过的功能现在已实现。为了能够使用我们的通过，我们需要将其注册到 `PassBuilder` 对象中。这可以通过两种方式实现：静态或动态。如果插件是静态链接的，那么它需要提供一个名为 `get<Plugin-Name>PluginInfo()` 的函数。要使用动态链接，需要提供 `llvmGetPassPluginInfo()` 函数。在两种情况下，都会返回一个 `PassPluginLibraryInfo` 结构体的实例，该结构体提供有关插件的一些基本信息。最重要的是，这个结构体包含一个指向注册通过函数的指针。让我们将其添加到我们的源代码中。

1.  在 `RegisterCB()` 函数中，我们注册了一个 Lambda 函数，该函数在解析通过管道字符串时被调用。如果通过的名称是 `ppprofiler`，那么我们将我们的通过添加到模块通过管理器中。这些回调将在下一节中进一步说明：

    ```cpp

    void RegisterCB(PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
          [](StringRef Name, ModulePassManager &MPM,
             ArrayRef<PassBuilder::PipelineElement>) {
            if (Name == "ppprofiler") {
              MPM.addPass(PPProfilerIRPass());
              return true;
            }
            return false;
          });
    }
    ```

1.  当插件静态链接时，会调用 `getPPProfilerPluginInfo()` 函数。它返回有关插件的一些基本信息：

    ```cpp

    llvm::PassPluginLibraryInfo getPPProfilerPluginInfo() {
      return {LLVM_PLUGIN_API_VERSION, "PPProfiler", "v0.1",
              RegisterCB};
    }
    ```

1.  最后，如果插件是动态链接的，那么当插件被加载时将调用 `llvmGetPassPluginInfo()` 函数。然而，当将此代码静态链接到工具中时，可能会遇到链接器错误，因为该函数可能在多个源文件中定义。解决方案是使用宏来保护该函数：

    ```cpp

    #ifndef LLVM_PPPROFILER_LINK_INTO_TOOLS
    extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
    llvmGetPassPluginInfo() {
      return getPPProfilerPluginInfo();
    }
    #endif
    ```

这样，我们就实现了通过插件。在我们查看如何使用新插件之前，让我们看看如果我们想将通过插件添加到 LLVM 源树中，需要做哪些更改。

## 将通过添加到 LLVM 源树中

如果你计划使用预编译的 clang 等工具，将新转换作为插件实现是有用的。另一方面，如果你编写自己的编译器，那么将你的新转换直接添加到 LLVM 源树中可能有很好的理由。你可以以两种不同的方式这样做——作为插件和作为一个完全集成的转换。插件方法需要的更改较少。

### 利用 LLVM 源树内部的插件机制

执行对 LLVM IR 进行转换的转换函数的源代码位于 `llvm-project/llvm/lib/Transforms` 目录中。在此目录内，创建一个名为 `PPProfiler` 的新目录，并将源文件 `PPProfiler.cpp` 复制到其中。你不需要对源代码进行任何修改！

要将新插件集成到构建系统中，创建一个名为 `CMakeLists.txt` 的文件，并包含以下内容：

```cpp

add_llvm_pass_plugin(PPProfiler PPProfiler.cpp)
```

最后，在父目录中的 `CmakeLists.txt` 文件中，你需要通过添加以下行来包含新的源目录：

```cpp

add_subdirectory(PPProfiler)
```

你现在可以准备使用添加了 `PPProfiler` 的 LLVM 进行构建了。切换到 LLVM 的构建目录，并手动运行 Ninja：

```cpp

$ ninja install
```

CMake 会检测构建描述的变化并重新运行配置步骤。你将看到额外的行：

```cpp

-- Registering PPProfiler as a pass plugin (static build: OFF)
```

这表明插件已被检测到，并已作为共享库构建。在安装步骤之后，你将在 `<install directory>/lib` 目录中找到该共享库，`PPProfiler.so`。

到目前为止，与上一节中的转换插件相比，唯一的区别是共享库作为 LLVM 的一部分被安装。但你也可以将新的插件静态链接到 LLVM 工具。为此，你需要重新运行 CMake 配置，并在命令行上添加 `-DLLVM_PPPROFILER_LINK_INTO_TOOLS=ON` 选项。从 CMake 中查找此信息以确认更改后的构建选项：

```cpp

-- Registering PPProfiler as a pass plugin (static build: ON)
```

再次编译和安装 LLVM 后，以下内容发生了变化：

+   插件被编译到静态库 `libPPProfiler.a` 中，并且该库被安装到 `<install directory>/lib` 目录中。

+   LLVM 工具，如 **opt**，都与该库链接。

+   插件被注册为扩展。你可以检查 `<install directory>/include/llvm/Support/Extension.def` 文件现在是否包含以下行：

    ```cpp

    HANDLE_EXTENSION(PPProfiler)
    ```

此外，所有支持此扩展机制的工具都会获取新的转换。在 *创建优化管道* 部分中，你将学习如何在你的编译器中实现这一点。

这种方法效果很好，因为新的源文件位于一个单独的目录中，并且只更改了一个现有文件。这最大限度地减少了如果你尝试保持修改后的 LLVM 源树与主仓库同步时的合并冲突概率。

也有情况，将新的 pass 作为插件添加并不是最佳方式。LLVM 提供的 pass 使用不同的注册方式。如果您开发了一个新的 pass 并提议将其添加到 LLVM 中，并且 LLVM 社区接受了您的贡献，那么您将希望使用相同的注册机制。

### 完全集成 pass 到 pass 注册表中

要完全集成新的 pass 到 LLVM 中，插件的源代码需要稍微不同的结构。这样做的主要原因是因为 pass 类的构造函数是从 pass 注册表中调用的，这要求类接口被放入头文件中。

与之前一样，您必须将新的 pass 放入 LLVM 的`Transforms`组件中。通过创建`llvm-project/llvm/include/llvm/Transforms/PPProfiler/PPProfiler.h`头文件开始实现；该文件的内容是类定义；将其放入`llvm`命名空间。不需要其他更改：

```cpp

#ifndef LLVM_TRANSFORMS_PPPROFILER_PPPROFILER_H
#define LLVM_TRANSFORMS_PPPROFILER_PPPROFILER_H
#include "llvm/IR/PassManager.h"
namespace llvm {
class PPProfilerIRPass
    : public llvm::PassInfoMixin<PPProfilerIRPass> {
public:
  llvm::PreservedAnalyses
  run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
private:
  void instrument(llvm::Function &F,
                  llvm::Function *EnterFn,
                  llvm::Function *ExitFn);
};
} // namespace llvm
#endif
```

接下来，将 pass 插件的源文件`PPProfiler.cpp`复制到新目录`llvm-project/llvm/lib/Transforms/PPProfiler`中。此文件需要按以下方式更新：

1.  由于类定义现在在头文件中，您必须从该文件中移除类定义。在顶部，添加对头文件的`#include`指令：

    ```cpp

    #include "llvm/Transforms/PPProfiler/PPProfiler.h"
    ```

1.  必须删除`llvmGetPassPluginInfo()`函数，因为 pass 没有构建成它自己的共享库。

与之前一样，您还需要提供一个`CMakeLists.txt`文件用于构建。您必须声明新的 pass 作为一个新组件：

```cpp

add_llvm_component_library(LLVMPPProfiler
  PPProfiler.cpp
  LINK_COMPONENTS
  Core
  Support
)
```

然后，就像在上一节中一样，您需要通过在父目录的`CMakeLists.txt`文件中添加以下行来包含新的源目录：

```cpp

add_subdirectory(PPProfiler)
```

在 LLVM 内部，可用的 passes 被保存在`llvm/lib/Passes/PassRegistry.def`数据库文件中。您需要更新此文件。新的 pass 是一个模块 pass，因此我们需要在文件中搜索定义模块 passes 的部分，例如，通过搜索`MODULE_PASS`宏。在此部分中，添加以下行：

```cpp

MODULE_PASS("ppprofiler", PPProfilerIRPass())
```

此数据库文件用于`llvm/lib/Passes/PassBuilder.cpp`类。此文件需要包含您的新头文件：

```cpp

#include "llvm/Transforms/PPProfiler/PPProfiler.h"
```

这些都是基于新 pass 插件版本所需的所有源代码更改。

由于您创建了一个新的 LLVM 组件，因此还需要在`llvm/lib/Passes/CMakeLists.txt`文件中添加一个链接依赖项。在`LINK_COMPONENTS`关键字下，您需要添加一行，包含新组件的名称：

```cpp

  PPProfiler
```

Et voilà – 您现在可以构建和安装 LLVM。新的 pass，`ppprofiler`，现在对所有 LLVM 工具都可用。它已被编译进`libLLVMPPProfiler.a`库，并在构建系统中作为`PPProfiler`组件可用。

到目前为止，我们已经讨论了如何创建一个新的 pass。在下一节中，我们将探讨如何使用`ppprofiler` pass。

# 使用 LLVM 工具的 pprofiler pass

回想一下我们在 *Developing the ppprofiler pass as a plugin* 部分中从 LLVM 树中开发出来的 ppprofiler 传递函数。在这里，我们将学习如何使用这个传递函数与 LLVM 工具一起使用，如 `opt` 和 `clang`，因为它们可以加载插件。

让我们先看看 `opt`。

### 在 opt 中运行传递插件

要尝试新的插件，你需要一个包含 LLVM IR 的文件。最简单的方法是将一个 C 程序翻译过来，例如一个基本的“Hello World”风格程序：

```cpp

#include <stdio.h>
int main(int argc, char *argv[]) {
  puts("Hello");
  return 0;
}
```

使用 `clang` 编译此文件，`hello.c`：

```cpp

$ clang -S -emit-llvm -O1 hello.c
```

你将得到一个非常简单的 IR 文件，名为 `hello.ll`，其中包含以下代码：

```cpp

$ cat hello.ll
@.str = private unnamed_addr constant [6 x i8] c"Hello\00",
        align 1
define dso_local i32 @main(
          i32 noundef %0, ptr nocapture noundef readnone %1) {
  %3 = tail call i32 @puts(
                 ptr noundef nonnull dereferenceable(1) @.str)
  ret i32 0
}
```

这足以测试传递函数。

要运行此传递函数，你必须提供一些参数。首先，你需要告诉 `opt` 通过 `--load-pass-plugin` 选项加载共享库。要运行单个传递函数，你必须指定 `--passes` 选项。使用 `hello.ll` 文件作为输入，你可以运行以下命令：

```cpp

$ opt --load-pass-plugin=./PPProfile.so \
      --passes="ppprofiler" --stats hello.ll -o hello_inst.bc
```

如果启用了统计生成，你将看到以下输出：

```cpp

===--------------------------------------------------------===
                 ... Statistics Collected ...
===--------------------------------------------------------===
1 ppprofiler - Number of instrumented functions.
```

否则，你将被告知统计收集未启用：

```cpp

Statistics are disabled.  Build with asserts or with
-DLLVM_FORCE_ENABLE_STATS
```

位码文件 `hello_inst.bc` 是结果。你可以使用 `llvm-dis` 工具将此文件转换为可读的 IR。正如预期的那样，你会看到对 `__ppp_enter()` 和 `__ppp_exit()` 函数的调用以及一个用于函数名称的新常量：

```cpp

$ llvm-dis hello_inst.bc -o –
@.str = private unnamed_addr constant [6 x i8] c"Hello\00",
        align 1
@0 = private unnamed_addr constant [5 x i8] c"main\00",
     align 1
define dso_local i32 @main(i32 noundef %0,
                          ptr nocapture noundef readnone %1) {
  call void @__ppp_enter(ptr @0)
  %3 = tail call i32 @puts(
                 ptr noundef nonnull dereferenceable(1) @.str)
  call void @__ppp_exit(ptr @0)
  ret i32 0
}
```

这已经看起来不错了！如果我们可以将这个 IR 转换为可执行文件并运行它，那就更好了。为此，你需要为被调用的函数提供实现。

通常，对某个功能的运行时支持比将其添加到编译器本身更复杂。这种情况也是如此。当调用 `__ppp_enter()` 和 `__ppp_exit()` 函数时，你可以将其视为一个事件。为了稍后分析数据，有必要保存这些事件。你希望获取的基本数据是事件类型、函数名称及其地址，以及时间戳。没有技巧，这并不像看起来那么简单。让我们试试看。

创建一个名为 `runtime.c` 的文件，内容如下：

1.  你需要文件 I/O、标准函数和时间支持。这由以下包含提供：

    ```cpp

    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    ```

1.  对于文件，需要一个文件描述符。此外，当程序结束时，应该正确关闭该文件描述符：

    ```cpp

    static FILE *FileFD = NULL;
    static void cleanup() {
      if (FileFD == NULL) {
        fclose(FileFD);
        FileFD = NULL;
      }
    }
    ```

1.  为了简化运行时，只使用一个固定的输出名称。如果文件未打开，则打开文件并注册 `cleanup` 函数：

    ```cpp

    static void init() {
      if (FileFD == NULL) {
        FileFD = fopen("ppprofile.csv", "w");
        atexit(&cleanup);
      }
    }
    ```

1.  你可以使用 `clock_gettime()` 函数来获取时间戳。`CLOCK_PROCESS_CPUTIME_ID` 参数返回此进程消耗的时间。请注意，并非所有系统都支持此参数。如果需要，你可以使用其他时钟，例如 `CLOCK_REALTIME`：

    ```cpp

    typedef unsigned long long Time;
    static Time get_time() {
      struct timespec ts;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
      return 1000000000L * ts.tv_sec + ts.tv_nsec;
    }
    ```

1.  现在，定义 `__ppp_enter()` 函数很容易。只需确保文件已打开，获取时间戳，并写入事件：

    ```cpp

    void __ppp_enter(const char *FnName) {
      init();
      Time T = get_time();
      void *Frame = __builtin_frame_address(1);
      fprintf(FileFD,
              // "enter|name|clock|frame"
              „enter|%s|%llu|%p\n", FnName, T, Frame);
    }
    ```

1.  `__ppp_exit()` 函数仅在事件类型方面有所不同：

    ```cpp

    void __ppp_exit(const char *FnName) {
      init();
      Time T = get_time();
      void *Frame = __builtin_frame_address(1);
      fprintf(FileFD,
              // "exit|name|clock|frame"
              „exit|%s|%llu|%p\n", FnName, T, Frame);
    }
    ```

这就完成了对运行时支持的简单实现。在我们尝试之前，应该对实现进行一些说明，因为很明显，这里有几个问题部分。

首先，由于只有一个文件描述符，并且对其访问没有保护，因此实现不是线程安全的。尝试使用此运行时实现与多线程程序一起使用，很可能会导致输出文件中的数据混乱。

此外，我们忽略了检查与 I/O 相关的函数的返回值，这可能导致数据丢失。

但最重要的是，事件的戳记并不精确。调用函数已经增加了开销，但在该函数中执行 I/O 操作使其变得更糟。原则上，你可以匹配函数的进入和退出事件并计算函数的运行时间。然而，这个值本身是有缺陷的，因为它可能包括 I/O 所需的时间。总之，不要相信这里记录的时间。

尽管存在所有这些缺陷，这个小运行时文件仍允许我们生成一些输出。将带有运行时代码的文件的 bitcode 与编译器文件一起编译，并运行生成的可执行文件：

```cpp

$ clang hello_inst.bc runtime.c
$ ./a.out
```

这在包含以下内容的目录中生成一个名为`ppprofile.csv`的新文件：

```cpp

$ cat ppprofile.csv
enter|main|3300868|0x1
exit|main|3760638|0x1
```

太棒了——新的 pass 和运行时似乎都工作得很好！

指定 pass pipeline

使用`–-passes`选项，你不仅可以命名单个 pass，还可以描述整个 pipeline。例如，优化级别 2 的默认 pipeline 命名为`default<O2>`。你可以使用`–-passes="ppprofile,default<O2>"`参数在默认 pipeline 之前运行`ppprofile` pass。请注意，此类 pipeline 描述中的 pass 名称必须是同一类型。

现在，让我们转向使用新的 pass 与`clang`。

### 将新的 pass 插入到 clang 中

在上一节中，你学习了如何使用`opt`运行单个 pass。如果你需要调试一个 pass，这很有用，但对于真正的编译器，步骤不应该那么复杂。

为了达到最佳效果，编译器需要按照一定的顺序运行优化通行。LLVM 通行管理器为通行执行提供了默认顺序。这也被称为 `opt`，您可以使用 `–passes` 选项指定不同的通行管道。这很灵活，但对于用户来说也很复杂。实际上，大多数时候，您只想在非常具体的位置添加新的通行，例如在运行优化通行之前或循环优化过程结束时。这些位置被称为 `PassBuilder` 类允许您在扩展点注册通行。例如，您可以通过调用 `registerPipelineStartEPCallback()` 方法将通行添加到优化管道的开始处。这正是我们需要的 `ppprofiler` 通行的地方。在优化过程中，函数可能会被内联，而通行可能会错过这些内联函数。相反，在优化通行之前运行通行可以保证所有函数都被仪器化。

要使用这种方法，您需要扩展通行插件中的 `RegisterCB()` 函数。将以下代码添加到函数中：

```cpp

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &PM, OptimizationLevel Level) {
        PM.addPass(PPProfilerIRPass());
      });
```

当通行管理器填充默认的通行管道时，它会调用所有扩展点的回调。我们只需在这里添加新的通行。

要将插件加载到 `clang` 中，您可以使用 `-fpass-plugin` 选项。现在创建 `hello.c` 文件的仪器化可执行文件几乎变得微不足道：

```cpp

$ clang -fpass-plugin=./PPProfiler.so hello.c runtime.c
```

请运行可执行文件并验证运行是否创建了 `ppprofiler.csv` 文件。

注意

由于通行检查特殊函数尚未在模块中声明，因此 `runtime.c` 文件没有被仪器化。

这已经看起来更好了，但它是否适用于更大的程序？让我们假设您想为 *第五章* 构建 `tinylang` 编译器的仪器化二进制文件。您将如何做？

您可以在 CMake 命令行上传递编译器和链接器标志，这正是我们所需要的。C++ 编译器的标志在 `CMAKE_CXX_FLAGS` 变量中给出。因此，在 CMake 命令行上指定以下内容会将新的通行添加到所有编译器运行中：

```cpp

-DCMAKE_CXX_FLAGS="-fpass-plugin=<PluginPath>/PPProfiler.so"
```

请将 `<PluginPath>` 替换为共享库的绝对路径。

类似地，指定以下内容会将 `runtime.o` 文件添加到每个链接调用中。再次提醒，请将 `<RuntimePath>` 替换为 `runtime.c` 编译版本的绝对路径：

```cpp

-DCMAKE_EXE_LINKER_FLAGS="<RuntimePath>/runtime.o"
```

当然，这需要使用 `clang` 作为构建编译器。确保 `clang` 作为构建编译器使用的最快方法是相应地设置 `CC` 和 `CXX` 环境变量：

```cpp

export CC=clang
export CXX=clang++
```

使用这些附加选项，*第五章* 中的 CMake 配置应正常运行。

在构建 `tinylang` 可执行文件后，您可以使用示例 `Gcd.mod` 文件运行它。这次 `ppprofile.csv` 文件也将被写入，这次有超过 44,000 行！

当然，拥有这样的数据集会引发一个问题：你是否能从中获得有用的信息。例如，获取最常调用的 10 个函数的列表，包括函数的调用次数和在该函数中花费的时间，将是有用的信息。幸运的是，在 Unix 系统中，你有一些工具可以帮助你。让我们构建一个简短的管道，匹配进入事件和退出事件，计算函数，并显示前 10 个函数。`awk` Unix 工具帮助完成这些步骤中的大多数。

为了匹配进入事件和退出事件，进入事件必须存储在 `record` 关联映射中。当匹配到退出事件时，会查找存储的进入事件，并写入新的记录。发出的行包含进入事件的戳记，退出事件的戳记，以及两者之间的差异。我们必须将此放入 `join.awk` 文件中：

```cpp

BEGIN { FS = "|"; OFS = "|" }
/enter/ { record[$2] = $0 }
/exit/ { split(record[$2],val,"|")
         print val[2], val[3], $3, $3-val[3], val[4] }
```

为了计算函数调用和执行，使用了两个关联映射，`count` 和 `sum`。在 `count` 中，计算函数调用，而在 `sum` 中，添加执行时间。最后，映射被导出。你可以将此放入 `avg.awk` 文件中：

```cpp

BEGIN { FS = "|"; count[""] = 0; sum[""] = 0 }
{ count[$1]++; sum[$1] += $4 }
END { for (i in count) {
        if (i != "") {
          print count[i], sum[i], sum[i]/count[i], I }
} }
```

运行这两个脚本后，结果可以按降序排序，然后可以从文件中取出前 10 行。然而，我们仍然可以改进函数名，`__ppp_enter()` 和 `__ppp_exit()`，它们是混淆的，因此难以阅读。使用 `llvm-cxxfilt` 工具，可以取消混淆。以下是一个 `demangle.awk` 脚本：

```cpp

{ cmd = "llvm-cxxfilt " $4
  (cmd) | getline name
  close(cmd); $4 = name; print }
```

要获取前 10 个函数调用，你可以运行以下命令：

```cpp

$ cat ppprofile.csv | awk -f join.awk | awk -f avg.awk |\
  sort -nr | head -15 | awk -f demangle.awk
```

这里是输出的一些示例行：

```cpp

446 1545581 3465.43 charinfo::isASCII(char)
409 826261 2020.2 llvm::StringRef::StringRef()
382 899471 2354.64
           tinylang::Token::is(tinylang::tok::TokenKind) const
171 1561532 9131.77 charinfo::isIdentifierHead(char)
```

第一个数字是函数的调用次数，第二个是累计执行时间，第三个数字是平均执行时间。正如之前所解释的，尽管调用次数应该是准确的，但不要相信时间值。

到目前为止，我们已经实现了一个新的仪器传递，要么作为插件，要么作为 LLVM 的补充，并在一些实际场景中使用了它。在下一节中，我们将探讨如何在我们的编译器中设置优化管道。

# 向你的编译器添加优化管道

在前几章中我们开发的 `tinylang` 编译器对 IR 代码不进行优化。在接下来的几个小节中，我们将向编译器添加一个优化管道来实现这一点。

## 创建优化管道

`PassBuilder`类对于设置优化管道至关重要。这个类了解所有已注册的 pass，并可以从文本描述中构建 pass 管道。我们可以使用这个类从命令行上的描述创建 pass 管道，或者使用基于请求的优化级别的默认管道。我们还支持使用 pass 插件，例如我们在上一节中讨论的`ppprofiler` pass 插件。有了这个，我们可以模拟**opt**工具的部分功能，并且也可以为命令行选项使用类似的名字。

`PassBuilder`类填充了一个`ModulePassManager`类的实例，这是持有构建的 pass 管道并运行它的 pass 管理器。代码生成 pass 仍然使用旧的 pass 管理器。因此，我们必须保留旧的 pass 管理器用于此目的。

对于实现，我们将从我们的`tinylang`编译器扩展`tools/driver/Driver.cpp`文件：

1.  我们将使用新的类，因此我们将从添加新的包含文件开始。`llvm/Passes/PassBuilder.h`文件定义了`PassBuilder`类。`llvm/Passes/PassPlugin.h`文件是插件支持所必需的。最后，`llvm/Analysis/TargetTransformInfo.h`文件提供了一个将 IR 级转换与特定目标信息连接的 pass：

    ```cpp

    #include "llvm/Passes/PassBuilder.h"
    #include "llvm/Passes/PassPlugin.h"
    #include "llvm/Analysis/TargetTransformInfo.h"
    ```

1.  为了使用新 pass 管理器的某些功能，我们必须添加三个命令行选项，使用与`opt`工具相同的名称。`--passes`选项允许以文本形式指定 pass 管道，而`--load-pass-plugin`选项允许使用 pass 插件。如果提供了`--debug-pass-manager`选项，则 pass 管理器将打印出有关已执行 pass 的信息：

    ```cpp

    static cl::opt<bool>
        DebugPM("debug-pass-manager", cl::Hidden,
                cl::desc("Print PM debugging information"));
    static cl::opt<std::string> PassPipeline(
        "passes",
        cl::desc("A description of the pass pipeline"));
    static cl::list<std::string> PassPlugins(
        "load-pass-plugin",
        cl::desc("Load passes from plugin library"));
    ```

1.  用户通过优化级别影响 pass 管道的构建。`PassBuilder`类支持六个不同的优化级别：无优化、三个优化速度的级别和两个减少大小的级别。我们可以通过一个命令行选项捕获所有级别：

    ```cpp

    static cl::opt<signed char> OptLevel(
        cl::desc("Setting the optimization level:"),
        cl::ZeroOrMore,
        cl::values(
            clEnumValN(3, "O", "Equivalent to -O3"),
            clEnumValN(0, "O0", "Optimization level 0"),
            clEnumValN(1, "O1", "Optimization level 1"),
            clEnumValN(2, "O2", "Optimization level 2"),
            clEnumValN(3, "O3", "Optimization level 3"),
            clEnumValN(-1, "Os",
                       "Like -O2 with extra optimizations "
                       "for size"),
            clEnumValN(
                -2, "Oz",
                "Like -Os but reduces code size further")),
        cl::init(0));
    ```

1.  LLVM 的插件机制支持静态链接插件的插件注册表，该注册表在项目配置期间创建。为了使用此注册表，我们必须包含`llvm/Support/Extension.def`数据库文件以创建返回插件信息的函数的原型：

    ```cpp

    #define HANDLE_EXTENSION(Ext)                          \
      llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
    #include "llvm/Support/Extension.def"
    ```

1.  现在，我们必须用新版本替换现有的`emit()`函数。此外，我们必须在函数顶部声明所需的`PassBuilder`实例：

    ```cpp

    bool emit(StringRef Argv0, llvm::Module *M,
              llvm::TargetMachine *TM,
              StringRef InputFilename) {
      PassBuilder PB(TM);
    ```

1.  为了实现命令行上提供的 pass 插件的支持，我们必须遍历用户提供的插件库列表，并尝试加载插件。如果失败，我们将发出错误消息；否则，我们将注册 pass：

    ```cpp

      for (auto &PluginFN : PassPlugins) {
        auto PassPlugin = PassPlugin::Load(PluginFN);
        if (!PassPlugin) {
          WithColor::error(errs(), Argv0)
              << "Failed to load passes from '" << PluginFN
              << "'. Request ignored.\n";
          continue;
        }
        PassPlugin->registerPassBuilderCallbacks(PB);
      }
    ```

1.  与静态插件注册表中的信息以类似的方式使用，将那些插件注册到我们的`PassBuilder`实例：

    ```cpp

    #define HANDLE_EXTENSION(Ext)                          \
      get##Ext##PluginInfo().RegisterPassBuilderCallbacks( \
          PB);
    #include "llvm/Support/Extension.def"
    ```

1.  现在，我们需要声明不同分析管理器的变量。唯一的参数是调试标志：

    ```cpp

      LoopAnalysisManager LAM(DebugPM);
      FunctionAnalysisManager FAM(DebugPM);
      CGSCCAnalysisManager CGAM(DebugPM);
      ModuleAnalysisManager MAM(DebugPM);
    ```

1.  接下来，我们必须通过在 `PassBuilder` 实例上调用相应的 `register` 方法来填充分析管理器。通过这个调用，分析管理器被填充了默认的分析传递，并且也运行了注册回调。我们还必须确保函数分析管理器使用默认的别名分析管道，并且所有分析管理器都知道彼此：

    ```cpp

      FAM.registerPass(
          [&] { return PB.buildDefaultAAPipeline(); });
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    ```

1.  `MPM` 模块传递管理器持有我们构建的传递管道。实例使用调试标志初始化：

    ```cpp

      ModulePassManager MPM(DebugPM);
    ```

1.  现在，我们需要实现两种不同的方法来用传递管道填充模块传递管理器。如果用户在命令行上提供了传递管道——也就是说，他们使用了 `--passes` 选项——那么我们就使用这个作为传递管道：

    ```cpp

      if (!PassPipeline.empty()) {
        if (auto Err = PB.parsePassPipeline(
                MPM, PassPipeline)) {
          WithColor::error(errs(), Argv0)
              << toString(std::move(Err)) << "\n";
          return false;
        }
      }
    ```

1.  否则，我们使用选择的优化级别来确定构建传递管道。默认传递管道的名称是 `default`，它接受优化级别作为参数：

    ```cpp

      else {
        StringRef DefaultPass;
        switch (OptLevel) {
        case 0: DefaultPass = "default<O0>"; break;
        case 1: DefaultPass = "default<O1>"; break;
        case 2: DefaultPass = "default<O2>"; break;
        case 3: DefaultPass = "default<O3>"; break;
        case -1: DefaultPass = "default<Os>"; break;
        case -2: DefaultPass = "default<Oz>"; break;
        }
        if (auto Err = PB.parsePassPipeline(
                MPM, DefaultPass)) {
          WithColor::error(errs(), Argv0)
              << toString(std::move(Err)) << "\n";
          return false;
        }
      }
    ```

1.  这样，对 IR 代码运行转换的传递管道已经设置好了。在此步骤之后，我们需要一个打开的文件来写入结果。系统汇编器和 LLVM IR 输出是基于文本的，因此我们应该为它们设置 `OF_Text` 标志：

    ```cpp

      std::error_code EC;
      sys::fs::OpenFlags OpenFlags = sys::fs::OF_None;
      CodeGenFileType FileType = codegen::getFileType();
      if (FileType == CGFT_AssemblyFile)
        OpenFlags |= sys::fs::OF_Text;
      auto Out = std::make_unique<llvm::ToolOutputFile>(
          outputFilename(InputFilename), EC, OpenFlags);
      if (EC) {
        WithColor::error(errs(), Argv0)
            << EC.message() << '\n';
        return false;
      }
    ```

1.  对于代码生成过程，我们必须使用旧的传递管理器。我们只需声明 `CodeGenPM` 实例并添加传递，这样就可以在 IR 转换级别提供目标特定信息：

    ```cpp

      legacy::PassManager CodeGenPM;
      CodeGenPM.add(createTargetTransformInfoWrapperPass(
          TM->getTargetIRAnalysis()));
    ```

1.  要输出 LLVM IR，我们必须添加一个传递，该传递将 IR 打印到流中：

    ```cpp

      if (FileType == CGFT_AssemblyFile && EmitLLVM) {
        CodeGenPM.add(createPrintModulePass(Out->os()));
      }
    ```

1.  否则，我们必须让 `TargetMachine` 实例添加所需的代码生成传递，这些传递由我们传递的 `FileType` 值指导：

    ```cpp

      else {
        if (TM->addPassesToEmitFile(CodeGenPM, Out->os(),
                                    nullptr, FileType)) {
          WithColor::error()
              << "No support for file type\n";
          return false;
        }
      }
    ```

1.  在所有这些准备工作之后，我们现在可以执行传递了。首先，我们必须在 IR 模块上运行优化管道。接下来，运行代码生成传递。当然，在所有这些工作之后，我们希望保留输出文件：

    ```cpp

      MPM.run(*M, MAM);
      CodeGenPM.run(*M);
      Out->keep();
      return true;
    }
    ```

1.  这段代码很多，但过程很简单。当然，我们还需要更新 `tools/driver/CMakeLists.txt` 构建文件中的依赖项。除了添加目标组件之外，我们还必须添加来自 LLVM 的所有转换和代码生成组件。组件名称大致类似于源代码所在的目录名称。在配置过程中，组件名称被转换为链接库名称：

    ```cpp

    set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD}
      AggressiveInstCombine Analysis AsmParser
      BitWriter CodeGen Core Coroutines IPO IRReader
      InstCombine Instrumentation MC ObjCARCOpts Remarks
      ScalarOpts Support Target TransformUtils Vectorize
      Passes)
    ```

1.  我们的编译器驱动程序支持插件，我们必须宣布这种支持：

    ```cpp

    add_tinylang_tool(tinylang Driver.cpp SUPPORT_PLUGINS)
    ```

1.  如前所述，我们必须链接到我们自己的库：

    ```cpp

    target_link_libraries(tinylang
      PRIVATE tinylangBasic tinylangCodeGen
      tinylangLexer tinylangParser tinylangSema)
    ```

    这些是对源代码和构建系统的必要补充。

1.  要构建扩展编译器，你必须切换到你的 `build` 目录并输入以下命令：

    ```cpp

    $ ninja
    ```

构建系统文件的更改将被自动检测，并且在编译和链接我们的更改源之前将运行 `cmake`。如果您需要重新运行配置步骤，请按照 *第一章* 中 *安装 LLVM* 的 *编译 tinylang 应用程序* 部分的说明操作。

由于我们已经将`opt`工具的选项作为蓝图使用，你应该尝试使用选项来运行`tinylang`，以加载一个 pass 插件并运行该插件，就像我们在前面的章节中所做的那样。

根据当前实现，我们可以运行默认的 pass 管道，或者我们可以自己构建一个。后者非常灵活，但在几乎所有情况下，这都会过于冗余。默认管道对 C-like 语言运行得非常好。然而，缺少的是扩展 pass 管道的方法。我们将在下一节中看看如何实现这一点。

## 扩展 pass 管道

在前面的章节中，我们使用了`PassBuilder`类来创建一个 pass 管道，无论是从用户提供的描述还是预定义的名称。现在，让我们看看另一种自定义 pass 管道的方法：使用扩展点。

在构建 pass 管道的过程中，pass 构建器允许添加用户贡献的 pass。这些位置被称为**扩展点**。存在几个扩展点，如下所示：

+   允许我们在管道开始处添加 pass 的管道开始扩展点

+   允许我们在指令组合器 pass 的每个实例之后添加 pass 的 peephole 扩展点

还存在其他扩展点。要使用扩展点，你必须注册一个回调。在构建 pass 管道的过程中，你的回调将在定义的扩展点处运行，并可以向给定的 pass 管理器添加 pass。

要为管道开始扩展点注册一个回调，你必须调用`PassBuilder`类的`registerPipelineStartEPCallback()`方法。例如，要将我们的`PPProfiler` pass 添加到管道的开始处，你需要将 pass 适配为模块 pass，通过调用`createModuleToFunctionPassAdaptor()`模板函数，然后将 pass 添加到模块 pass 管理器：

```cpp

PB.registerPipelineStartEPCallback(
    [](ModulePassManager &MPM) {
        MPM.addPass(PPProfilerIRPass());
    });
```

你可以在创建管道之前，在任何位置添加此代码片段，即在调用`parsePassPipeline()`方法之前。

对我们在上一节中所做的事情的一个非常自然的扩展是让用户在命令行上传递一个扩展点的管道描述。`opt`工具也允许这样做。让我们为管道开始扩展点添加以下代码到`tools/driver/Driver.cpp`文件：

1.  首先，我们必须为用户提供一个新的命令行来指定管道描述。再次，我们从`opt`工具中获取选项名称：

    ```cpp

    static cl::opt<std::string> PipelineStartEPPipeline(
        "passes-ep-pipeline-start",
        cl::desc("Pipeline start extension point));
    ```

1.  使用 Lambda 函数作为回调是最方便的方法。为了解析管道描述，我们必须调用`PassBuilder`实例的`parsePassPipeline()`方法。将 pass 添加到`PM` pass 管理器，并将其作为参数传递给 Lambda 函数。如果发生错误，我们只打印错误消息而不会停止应用程序。你可以在调用`crossRegisterProxies()`方法之后添加此代码片段：

    ```cpp

      PB.registerPipelineStartEPCallback(
          &PB, Argv0 {
            if (auto Err = PB.parsePassPipeline(
                    PM, PipelineStartEPPipeline)) {
              WithColor::error(errs(), Argv0)
                  << "Could not parse pipeline "
                  << PipelineStartEPPipeline.ArgStr << ": "
                  << toString(std::move(Err)) << "\n";
            }
          });
    ```

小贴士

为了允许用户在每一个扩展点添加 passes，您需要为每个扩展点添加前面的代码片段。

1.  现在是尝试不同的 `pass manager` 选项的好时机。使用 `--debug-pass-manager` 选项，您可以跟踪执行顺序中哪些 passes 被执行。您还可以在每个 pass 之前或之后打印 IR，这可以通过 `--print-before-all` 和 `--print-after-all` 选项来实现。如果您创建了您自己的 pass 管道，那么您可以在感兴趣的位置插入 `print` pass。例如，尝试 `--passes="print,inline,print"` 选项。此外，为了确定哪个 pass 改变了 IR 代码，您可以使用 `--print-changed` 选项，该选项仅在 IR 代码与上一个 pass 的结果相比有变化时打印 IR 代码。大大减少的输出使得跟踪 IR 转换变得容易得多。

    `PassBuilder` 类有一个嵌套的 `OptimizationLevel` 类来表示六个不同的优化级别。我们不仅可以将 `"default<O?>"` 管道描述作为 `parsePassPipeline()` 方法的参数，还可以调用 `buildPerModuleDefaultPipeline()` 方法，该方法为请求级别构建默认的优化管道——除了级别 `O0`。这个优化级别意味着不执行任何优化。

    因此，没有 passes 被添加到 pass manager 中。如果我们仍然想运行某个 pass，那么我们可以手动将其添加到 pass manager 中。在这个级别上运行的一个简单 pass 是 `AlwaysInliner` pass，它将带有 `always_inline` 属性的函数内联到调用者中。在将优化级别的命令行选项值转换为 `OptimizationLevel` 类的相应成员之后，我们可以这样实现：

    ```cpp

        PassBuilder::OptimizationLevel Olevel = …;
        if (OLevel == PassBuilder::OptimizationLevel::O0)
          MPM.addPass(AlwaysInlinerPass());
        else
          MPM = PB.buildPerModuleDefaultPipeline(OLevel, DebugPM);
    ```

    当然，以这种方式可以向 pass manager 添加多个 passes。`PassBuilder` 在构建 pass 管道时也会使用 `addPass()` 方法。

运行扩展点回调

由于优化级别 `O0` 的 pass 管道没有被填充，因此注册的扩展点没有被调用。如果您使用扩展点注册应在 `O0` 级别运行的 passes，这会存在问题。您可以通过调用 `runRegisteredEPCallbacks()` 方法来运行注册的扩展点回调，这将导致只包含通过扩展点注册的 passes 的 pass manager。

通过将优化管道添加到 `tinylang`，您创建了一个类似于 `clang` 的优化编译器。LLVM 社区在每个版本中都致力于改进优化和优化管道。因此，默认管道很少不被使用。通常，新 passes 被添加来实现编程语言的某些语义。

# 摘要

在本章中，你学习了如何为 LLVM 创建一个新的 Pass。你使用 Pass 管道描述和扩展点运行了该 Pass。你通过构建和执行类似于`clang`的 Pass 管道，扩展了你的编译器，将`tinylang`转换成了一个优化编译器。Pass 管道允许在扩展点添加 Pass，你学习了如何在这些点上注册 Pass。这允许你通过你开发的 Pass 或现有的 Pass 来扩展优化管道。

在下一章中，你将学习`clang`的基础知识，以显著减少手动编程。
