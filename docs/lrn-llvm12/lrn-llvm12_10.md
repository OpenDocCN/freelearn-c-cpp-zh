# 第八章：优化 IR

LLVM 使用一系列 Passes 来优化**中间表示**（**IR**）。Pass 对 IR 单元执行操作，可以是函数或模块。操作可以是转换，以定义的方式更改 IR，也可以是分析，收集依赖关系等信息。一系列 Passes 称为**Pass 管道**。Pass 管理器在我们的编译器生成的 IR 上执行 Pass 管道。因此，我们需要了解 Pass 管理器的作用以及如何构建 Pass 管道。编程语言的语义可能需要开发新的 Passes，并且我们必须将这些 Passes 添加到管道中。

在本章中，我们将涵盖以下主题：

+   介绍 LLVM Pass 管理器

+   使用新 Pass 管理器实现 Pass

+   为旧 Pass 管理器使用 Pass

+   向您的编译器添加优化管道

在本章结束时，您将了解如何开发新的 Pass 以及如何将其添加到 Pass 管道中。您还将获得设置自己编译器中 Pass 管道所需的知识。

# 技术要求

本章的源代码可在[`github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter08`](https://github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter08)找到

您可以在[`bit.ly/3nllhED`](https://bit.ly/3nllhED)找到代码的实际应用视频

# 介绍 LLVM Pass 管理器

LLVM 核心库优化编译器创建的 IR 并将其转换为目标代码。这项巨大的任务被分解为称为**Passes**的单独步骤。这些 Passes 需要按正确的顺序执行，这是 Pass 管理器的目标。

但是为什么不硬编码 Passes 的顺序呢？嗯，您的编译器的用户通常期望您的编译器提供不同级别的优化。开发人员更喜欢在开发时间内更快的编译速度而不是优化。最终应用程序应尽可能快地运行，您的编译器应能够执行复杂的优化，接受更长的编译时间。不同级别的优化意味着需要执行不同数量的优化 Passes。作为编译器编写者，您可能希望提供自己的 Passes，以利用您对源语言的了解。例如，您可能希望用内联 IR 或者可能的话用该函数的计算结果替换已知的库函数。对于 C，这样的 Pass 是 LLVM 核心库的一部分，但对于其他语言，您需要自己提供。并且引入自己的 Passes，您可能需要重新排序或添加一些 Passes。例如，如果您知道您的 Pass 的操作使一些 IR 代码不可达，则还应在您自己的 Pass 之后运行死代码删除 Pass。Pass 管理器帮助您组织这些要求。

Pass 通常根据其工作范围进行分类：

+   *函数 Pass*接受单个函数作为输入，并仅对该函数执行其工作。

+   *模块 Pass*接受整个模块作为输入。这样的 Pass 在给定模块上执行其工作，并且可以用于模块内的程序内操作。

+   *调用图* Pass 按自底向上的顺序遍历调用图的函数。

除了 IR 代码之外，Pass 还可能消耗、产生或使一些分析结果无效。进行了许多不同的分析；例如，别名分析或支配树的构建。支配树有助于将不变的代码移出循环，因此只有在支配树创建后才能运行执行此类转换的 Pass。另一个 Pass 可能执行一个转换，这可能会使现有的支配树无效。

在幕后，Pass 管理器确保以下内容：

+   分析结果在 Passes 之间共享。这要求您跟踪哪个 Pass 需要哪个分析，以及每个分析的状态。目标是避免不必要的分析重新计算，并尽快释放分析结果所占用的内存。

+   Pass 以管道方式执行。例如，如果应该按顺序执行多个函数 Pass，那么 Pass 管理器将在第一个函数上运行每个函数 Pass。然后它将在第二个函数上运行所有函数 Pass，依此类推。这里的基本思想是改善缓存行为，因为编译器仅对有限的数据集（即一个 IR 函数）执行转换，然后转移到下一个有限的数据集。

LLVM 中有两个 Pass 管理器，如下：

+   旧的（或传统的）Pass 管理器

+   新的 Pass 管理器

未来属于新的 Pass 管理器，但过渡尚未完成。一些关键的 Pass，如目标代码发射，尚未迁移到新的 Pass 管理器，因此了解两个 Pass 管理器非常重要。

旧的 Pass 管理器需要一个 Pass 从一个基类继承，例如，从`llvm::FunctionPass`类继承一个函数 Pass。相比之下，新的 Pass 管理器依赖于基于概念的方法，只需要从特殊的`llvm::PassInfo<>` mixin 类继承。旧的 Pass 管理器中 Passes 之间的依赖关系没有明确表达。在新的 Pass 管理器中，需要明确编码。新的 Pass 管理器还采用了不同的分析处理方法，并允许通过命令行上的文本表示来指定优化管道。一些 LLVM 用户报告说，仅通过从旧的 Pass 管理器切换到新的 Pass 管理器，编译时间就可以减少高达 10%，这是使用新的 Pass 管理器的非常有说服力的论点。

首先，我们将为新的 Pass 管理器实现一个 Pass，并探索如何将其添加到优化管道中。稍后，我们将看看如何在旧的 Pass 管理器中使用 Pass。

# 使用新的 Pass 管理器实现 Pass

Pass 可以对 LLVM IR 执行任意复杂的转换。为了说明添加新 Pass 的机制，我们的新 Pass 只计算 IR 指令和基本块的数量。我们将 Pass 命名为`countir`。将 Pass 添加到 LLVM 源树或作为独立的 Pass 略有不同，因此我们将在以下部分都进行。

## 将 Pass 添加到 LLVM 源树

让我们从将新 Pass 添加到 LLVM 源开始。如果我们以后想要在 LLVM 树中发布新的 Pass，这是正确的方法。

对 LLVM IR 执行转换的 Pass 的源代码位于`llvm-project/llvm/lib/Transforms`文件夹中，头文件位于`llvm-project/llvm/include/llvm/Transforms`文件夹中。由于 Pass 太多，它们被分类到适合它们的类别的子文件夹中。

对于我们的新 Pass，在两个位置都创建一个名为`CountIR`的新文件夹。首先，让我们实现`CountIR.h`头文件：

1.  像往常一样，我们需要确保文件可以被多次包含。此外，我们需要包含 Pass 管理器的定义：

```cpp
#ifndef LLVM_TRANSFORMS_COUNTIR_COUNTIR_H
#define LLVM_TRANSFORMS_COUNTIR_COUNTIR_H
#include "llvm/IR/PassManager.h"
```

1.  因为我们在 LLVM 源代码中，所以我们将新的`CountIR`类放入`llvm`命名空间中。该类继承自`PassInfoMixin`模板。该模板仅添加了一些样板代码，例如`name()`方法。它不用于确定 Pass 的类型。

```cpp
namespace llvm {
class CountIRPass : public PassInfoMixin<CountIRPass> {
```

1.  在运行时，将调用任务的`run()`方法。`run()`方法的签名确定 Pass 的类型。这里，第一个参数是对`Function`类型的引用，因此这是一个函数 Pass：

```cpp
public:
  PreservedAnalyses run(Function &F,
                        FunctionAnalysisManager &AM);
```

1.  最后，我们需要关闭类、命名空间和头文件保护：

```cpp
};
} // namespace llvm
#endif
```

当然，我们的新 Pass 的定义是如此简单，因为我们只执行了一个微不足道的任务。

让我们继续在`CountIIR.cpp`文件中实现 Pass。LLVM 支持在调试模式下收集有关 Pass 的统计信息。对于我们的 Pass，我们将利用这个基础设施。

1.  我们首先包含我们自己的头文件和所需的 LLVM 头文件：

```cpp
#include "llvm/Transforms/CountIR/CountIR.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
```

1.  为了缩短源代码，我们告诉编译器我们正在使用`llvm`命名空间：

```cpp
using namespace llvm;
```

1.  LLVM 的内置调试基础设施要求我们定义一个调试类型，即一个字符串。这个字符串稍后将显示在打印的统计信息中：

```cpp
#define DEBUG_TYPE "countir"
```

1.  我们使用`STATISTIC`宏定义了两个计数器变量。第一个参数是计数器变量的名称，第二个参数是将在统计中打印的文本：

```cpp
STATISTIC(NumOfInst, "Number of instructions.");
STATISTIC(NumOfBB, "Number of basic blocks.");
```

1.  在`run()`方法中，我们循环遍历函数的所有基本块，并递增相应的计数器。我们对基本块的所有指令也是一样的。为了防止编译器警告我们关于未使用的变量，我们插入了对`I`变量的无操作使用。因为我们只计数而不改变 IR，我们告诉调用者我们已经保留了所有现有的分析：

```cpp
PreservedAnalyses
CountIRPass::run(Function &F,
                 FunctionAnalysisManager &AM) {
  for (BasicBlock &BB : F) {
    ++NumOfBB;
    for (Instruction &I : BB) {
      (void)I;
      ++NumOfInst;
    }
  }
  return PreservedAnalyses::all();
}
```

到目前为止，我们已经实现了新 Pass 的功能。我们稍后将重用这个实现来进行一个树外的 Pass。对于 LLVM 树内的解决方案，我们必须更改 LLVM 中的几个文件来宣布新 Pass 的存在：

1.  首先，我们需要在源文件夹中添加一个`CMakeLists.txt`文件。这个文件包含了一个新的 LLVM 库名`LLVMCountIR`的构建指令。新库需要链接 LLVM 的`Support`组件，因为我们使用了调试和统计基础设施，以及 LLVM 的`Core`组件，其中包含了 LLVM IR 的定义：

```cpp
add_llvm_component_library(LLVMCountIR
  CountIR.cpp
  LINK_COMPONENTS Core Support )
```

1.  为了使这个新库成为构建的一部分，我们需要将该文件夹添加到父文件夹的`CMakeLists.txt`文件中，即`llvm-project/llvm/lib/Transforms/CMakeList.txt`文件。然后，添加以下行：

```cpp
add_subdirectory(CountIR)
```

1.  `PassBuilder`类需要知道我们的新 Pass。为此，我们在`llvm-project/llvm/lib/Passes/PassBuilder.cpp`文件的`include`部分添加以下行：

```cpp
#include "llvm/Transforms/CountIR/CountIR.h"
```

1.  作为最后一步，我们需要更新 Pass 注册表，这在`llvm-project/llvm/lib/Passes/PassRegistry.def`文件中。查找定义函数 Pass 的部分，例如通过搜索`FUNCTION_PASS`宏。在这个部分中，添加以下行：

```cpp
FUNCTION_PASS("countir", CountIRPass())
```

1.  我们现在已经做出了所有必要的更改。按照*第一章*中的构建说明，*使用 CMake 构建*部分，重新编译 LLVM。要测试新的 Pass，我们将以下 IR 代码存储在我们的`build`文件夹中的`demo.ll`文件中。代码有两个函数，总共三条指令和两个基本块：

```cpp
define internal i32 @func() {
  ret i32 0
}
define dso_local i32 @main() {
  %1 = call i32 @func()
  ret i32 %1
}
```

1.  我们可以使用`opt`实用程序来使用新的 Pass。要运行新的 Pass，我们将利用`--passes="countir"`选项。要获得统计输出，我们需要添加`--stats`选项。因为我们不需要生成的位码，我们还指定了`--disable-output`选项：

```cpp
$ bin/opt --disable-output --passes="countir" –-stats demo.ll
===--------------------------------------------------------===
                   ... Statistics Collected ...
===--------------------------------------------------------===
2 countir - Number of basic blocks.
3 countir - Number of instructions. 
```

1.  我们运行我们的新 Pass，输出符合我们的期望。我们已经成功扩展了 LLVM！

运行单个 Pass 有助于调试。使用`--passes`选项，您不仅可以命名单个 Pass，还可以描述整个流水线。例如，优化级别 2 的默认流水线被命名为`default<O2>`。您可以在默认流水线之前使用`--passes="module(countir),default<O2>"`参数运行`countir` Pass。这样的流水线描述中的 Pass 名称必须是相同类型的。默认流水线是一个模块 Pass，我们的`countir` Pass 是一个函数 Pass。要从这两者创建一个模块流水线，首先我们必须创建一个包含`countir` Pass 的模块 Pass。这是通过`module(countir)`来完成的。您可以通过以逗号分隔的列表指定更多的函数 Passes 添加到这个模块 Pass 中。同样，模块 Passes 也可以组合。为了研究这一点的影响，您可以使用`inline`和`countir` Passes：以不同的顺序运行它们，或者作为模块 Pass，将给出不同的统计输出。

将新的 Pass 添加到 LLVM 源代码树中是有意义的，如果您计划将您的 Pass 作为 LLVM 的一部分发布。如果您不打算这样做，或者希望独立于 LLVM 分发您的 Pass，那么您可以创建一个 Pass 插件。在下一节中，我们将查看如何执行这些步骤。

## 作为插件添加新的 Pass

为了将新的 Pass 作为插件提供，我们将创建一个使用 LLVM 的新项目：

1.  让我们从在我们的源文件夹中创建一个名为`countirpass`的新文件夹开始。该文件夹将具有以下结构和文件：

```cpp
|-- CMakeLists.txt
|-- include
|   `-- CountIR.h
|-- lib
    |-- CMakeLists.txt
    `-- CountIR.cpp
```

1.  请注意，我们已经重用了上一节的功能，只是做了一些小的调整。`CountIR.h`头文件现在位于不同的位置，所以我们改变了用作守卫的符号的名称。我们也不再使用`llvm`命名空间，因为我们现在不在 LLVM 源代码之内。由于这个改变，头文件变成了以下内容：

```cpp
#ifndef COUNTIR_H
#define COUNTIR_H
#include "llvm/IR/PassManager.h"
class CountIRPass
    : public llvm::PassInfoMixin<CountIRPass> {
public:
  llvm::PreservedAnalyses
  run(llvm::Function &F,
      llvm::FunctionAnalysisManager &AM);
};
#endif
```

1.  我们可以从上一节复制`CountIR.cpp`实现文件。这里也需要做一些小的改动。因为我们的头文件路径已经改变，所以我们需要用以下内容替换`include`指令：

```cpp
#include "CountIR.h"
```

1.  我们还需要在 Pass builder 中注册新的 Pass。这是在插件加载时发生的。Pass 插件管理器调用特殊函数`llvmGetPassPluginInfo()`，进行注册。对于这个实现，我们需要两个额外的`include`文件：

```cpp
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
```

用户可以使用`--passes`选项在命令行上指定要运行的 Passes。`PassBuilder`类从字符串中提取 Pass 名称。为了创建命名 Pass 的实例，`PassBuilder`类维护一个回调函数列表。基本上，回调函数会以 Pass 名称和 Pass 管理器作为参数进行调用。如果回调函数知道 Pass 名称，那么它会将这个 Pass 的实例添加到 Pass 管理器中。对于我们的 Pass，我们需要提供这样一个回调函数：

```cpp
bool PipelineParsingCB(
    StringRef Name, FunctionPassManager &FPM,
    ArrayRef<PassBuilder::PipelineElement>) {
  if (Name == "countir") {
    FPM.addPass(CountIRPass());
    return true;
  }
  return false;
}
```

1.  当然，我们需要将这个函数注册为`PassBuilder`实例。插件加载后，将为此目的调用注册回调。我们的注册函数如下：

```cpp
void RegisterCB(PassBuilder &PB) {
  PB.registerPipelineParsingCallback(PipelineParsingCB);
}
```

1.  最后，每个插件都需要提供上述`llvmGetPassPluginInfo()`函数。这个函数返回一个结构，包含四个元素：我们的插件使用的 LLVM 插件 API 版本、名称、插件的版本号和注册回调。插件 API 要求函数使用`extern "C"`约定。这是为了避免 C++名称混淆的问题。这个函数非常简单：

```cpp
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CountIR", "v0.1",
          RegisterCB};
}
```

为每个回调实现一个单独的函数有助于我们理解正在发生的事情。如果您的插件提供了多个 Passes，那么您可以扩展`RegisterCB`回调函数以注册所有 Passes。通常，您可以找到一个非常紧凑的方法。以下的`llvmGetPassPluginInfo()`函数将`PipelineParsingCB()`、`RegisterCB()`和之前的`llvmGetPassPluginInfo()`合并为一个函数。它通过使用 lambda 函数来实现：

```cpp
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CountIR", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager                        &FPM,
                ArrayRef<PassBuilder::PipelineElement>)  
                {
                  if (Name == "countir") {
                    FPM.addPass(CountIRPass());
                    return true;
                  }
                  return false;
                });
          }};
}
```

1.  现在，我们只需要添加构建文件。`lib/CMakeLists.txt`文件只包含一个命令来编译源文件。LLVM 特定的命令`add_llvm_library()`确保使用用于构建 LLVM 的相同编译器标志：

```cpp
add_llvm_library(CountIR MODULE CountIR.cpp)
```

顶层的`CMakeLists.txt`文件更加复杂。

1.  像往常一样，我们设置所需的 CMake 版本和项目名称。此外，我们将`LLVM_EXPORTED_SYMBOL_FILE`变量设置为`ON`。这对于使插件在 Windows 上工作是必要的：

```cpp
cmake_minimum_required(VERSION 3.4.3)
project(countirpass)
set(LLVM_EXPORTED_SYMBOL_FILE ON)
```

1.  接下来，我们寻找 LLVM 安装。我们还将在控制台上打印有关找到的版本的信息：

```cpp
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
```

1.  现在，我们可以将 LLVM 的`cmake`文件夹添加到搜索路径中。我们包括 LLVM 特定的文件`ChooseMSVCCRT`和`AddLLVM`，它们提供了额外的命令：

```cpp
list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
include(ChooseMSVCCRT)
include(AddLLVM)
```

1.  编译器需要了解所需的定义和 LLVM 路径：

```cpp
include_directories("${LLVM_INCLUDE_DIR}")
add_definitions("${LLVM_DEFINITIONS}")
link_directories("${LLVM_LIBRARY_DIR}")
```

1.  最后，我们添加自己的包含和源文件夹：

```cpp
include_directories(BEFORE include)
add_subdirectory(lib)
```

1.  在实现了所有必需的文件之后，我们现在可以在`countirpass`文件夹旁边创建`build`文件夹。首先，切换到构建目录并创建构建文件：

```cpp
$ cmake –G Ninja ../countirpass
```

1.  然后，您可以编译插件，如下所示：

```cpp
$ ninja
```

1.  您可以使用`opt`实用程序使用插件，`opt`实用程序会生成输入文件的优化版本。要使用插件，您需要指定一个额外的参数来加载插件：

```cpp
$ opt --load-pass-plugin=lib/CountIR.so --passes="countir"\
  --disable-output –-stats demo.ll
```

输出与以前版本相同。恭喜，Pass 插件有效！

到目前为止，我们只为新 Pass 管理器创建了一个 Pass。在下一节中，我们还将扩展旧 Pass 管理器的 Pass。

# 调整 Pass 以与旧 Pass 管理器一起使用

未来属于新 Pass 管理器，为旧 Pass 管理器专门开发新 Pass 是没有意义的。然而，在进行过渡阶段期间，如果一个 Pass 可以与两个 Pass 管理器一起工作，那将是有用的，因为 LLVM 中的大多数 Pass 已经这样做了。

旧 Pass 管理器需要一个从特定基类派生的 Pass。例如，函数 Pass 必须从`FunctionPass`基类派生。还有更多的不同之处。Pass 管理器运行的方法被命名为`runOnFunction()`，还必须提供 Pass 的`ID`。我们在这里遵循的策略是创建一个单独的类，可以与旧 Pass 管理器一起使用，并以一种可以与两个 Pass 管理器一起使用的方式重构源代码。

我们将 Pass 插件用作基础。在`include/CountIR.h`头文件中，我们添加一个新的类定义，如下所示：

1.  新类需要从`FunctionPass`类派生，因此我们包含一个额外的头文件来获取类定义：

```cpp
#include "llvm/Pass.h"
```

1.  我们将新类命名为`CountIRLegacyPass`。该类需要内部 LLVM 机制的 ID，并用其初始化父类：

```cpp
class CountIRLegacyPass : public llvm::FunctionPass {
public:
  static char ID;
  CountIRLegacyPass() : llvm::FunctionPass(ID) {}
```

1.  为了实现 Pass 功能，必须重写两个函数。`runOnFunction()`方法用于每个 LLVM IR 函数，并实现我们的计数功能。`getAnalysisUsage()`方法用于宣布所有分析结果都已保存：

```cpp
  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const     override;
};
```

1.  现在头文件的更改已经完成，我们可以增强`lib/CountIR.cpp`文件中的实现。为了重用计数功能，我们将源代码移入一个新的函数：

```cpp
void runCounting(Function &F) {
  for (BasicBlock &BB : F) {
    ++NumOfBB;
    for (Instruction &I : BB) {
      (void)I;
      ++NumOfInst;
    }
  }
}
```

1.  新 Pass 管理器的方法需要更新，以便使用新功能：

```cpp
PreservedAnalyses
CountIRPass::run(Function &F, FunctionAnalysisManager &AM) {
  runCounting(F);
  return PreservedAnalyses::all();
}
```

1.  以同样的方式，我们实现了旧 Pass 管理器的方法。通过返回`false`值，我们表明 IR 没有发生变化：

```cpp
bool CountIRLegacyPass::runOnFunction(Function &F) {
  runCounting(F);
  return false;
}
```

1.  为了保留现有的分析结果，必须以以下方式实现`getAnalysisUsage()`方法。这类似于新 Pass 管理器中`PreservedAnalyses::all()`的返回值。如果不实现此方法，则默认情况下会丢弃所有分析结果：

```cpp
void CountIRLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
}
```

1.  `ID`字段可以用任意值初始化，因为 LLVM 使用字段的地址。通常值为`0`，所以我们也使用它：

```cpp
char CountIRLegacyPass::ID = 0;
```

1.  现在只缺少 Pass 注册。要注册新 Pass，我们需要提供`RegisterPass<>`模板的静态实例。第一个参数是调用新 Pass 的命令行选项的名称。第二个参数是 Pass 的名称，用于在调用`-help`选项时向用户提供信息等：

```cpp
static RegisterPass<CountIRLegacyPass>
    X("countir", "CountIR Pass");
```

1.  这些变化足以让我们在旧 Pass 管理器和新 Pass 管理器下调用我们的新 Pass。为了测试这个添加，切换回`build`文件夹并编译 Pass：

```cpp
$ ninja
```

1.  为了在旧 Pass 管理器中加载插件，我们需要使用`--load`选项。我们的新 Pass 是使用`--countir`选项调用的：

```cpp
$ opt --load lib/CountIR.so --countir –-stats\
  --disable-output demo.ll
```

提示

请还要检查，在上一节的命令行中，使用新 Pass 管理器调用我们的 Pass 是否仍然正常工作！

能够使用 LLVM 提供的工具运行我们的新 Pass 是很好的，但最终，我们希望在我们的编译器内运行它。在下一节中，我们将探讨如何设置优化流水线以及如何自定义它。

# 向您的编译器添加优化流水线

我们的`tinylang`编译器，在前几章中开发，对创建的 IR 代码不进行任何优化。在接下来的章节中，我们将向编译器添加一个优化流水线，以实现这一点。

## 使用新 Pass 管理器创建优化流水线

优化流水线设置的核心是`PassBuilder`类。这个类知道所有注册的 Pass，并可以根据文本描述构建 Pass 流水线。我们使用这个类来从命令行给出的描述创建 Pass 流水线，或者使用基于请求的优化级别的默认流水线。我们还支持使用 Pass 插件，例如我们在上一节中讨论的`countir` Pass 插件。通过这样做，我们模仿了`opt`工具的部分功能，并且还使用了类似的命令行选项名称。

`PassBuilder`类填充了一个`ModulePassManager`类的实例，这是用于保存构建的 Pass 流水线并实际运行它的 Pass 管理器。代码生成 Pass 仍然使用旧 Pass 管理器；因此，我们必须保留旧 Pass 管理器以实现这一目的。

对于实现，我们扩展了我们的`tinylang`编译器中的`tools/driver/Driver.cpp`文件：

1.  我们使用新的类，因此我们首先添加新的`include`文件。`llvm/Passes/PassBuilder.h`文件提供了`PassBuilder`类的定义。`llvm/Passes/PassPlugin.h`文件是插件支持所需的。最后，`llvm/Analysis/TargetTransformInfo.h`文件提供了一个将 IR 级别转换与特定目标信息连接起来的 Pass：

```cpp
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Analysis/TargetTransformInfo.h"
```

1.  为了使用新 Pass 管理器的某些功能，我们添加了三个命令行选项，使用与`opt`工具相同的名称。`--passes`选项允许 Pass 流水线的文本规范，`--load-pass-plugin`选项允许使用 Pass 插件。如果给出`--debug-pass-manager`选项，则 Pass 管理器会打印有关执行的 Pass 的信息：

```cpp
static cl::opt<bool>
    DebugPM("debug-pass-manager", cl::Hidden,
            cl::desc("Print PM debugging 
                     information"));
static cl::opt<std::string> PassPipeline(
    "passes",
    cl::desc("A description of the pass pipeline"));
static cl::list<std::string> PassPlugins(
    "load-pass-plugin",
    cl::desc("Load passes from plugin library"));
```

1.  用户通过优化级别影响 Pass 流水线的构建。`PassBuilder`类支持六个不同的优化级别：一个无优化级别，三个用于优化速度的级别，以及两个用于减小大小的级别。我们在一个命令行选项中捕获所有这些级别：

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
                   "Like -O2 with extra 
                    optimizations "
                   "for size"),
        clEnumValN(
            -2, "Oz",
            "Like -Os but reduces code size further")),
    cl::init(0));
```

1.  LLVM 的插件机制支持静态插件注册表，在项目配置期间创建。为了利用这个注册表，我们包括`llvm/Support/Extension.def`数据库文件来创建返回插件信息的函数的原型：

```cpp
#define HANDLE_EXTENSION(Ext)                          \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"
```

1.  我们用新版本替换现有的`emit()`函数。我们在函数顶部声明所需的`PassBuilder`实例：

```cpp
bool emit(StringRef Argv0, llvm::Module *M,
          llvm::TargetMachine *TM,
          StringRef InputFilename) {
  PassBuilder PB(TM);
```

1.  为了实现对命令行上给出的 Pass 插件的支持，我们循环遍历用户给出的插件库列表，并尝试加载插件。如果失败，我们会发出错误消息；否则，我们注册 Passes：

```cpp
  for (auto &PluginFN : PassPlugins) {
    auto PassPlugin = PassPlugin::Load(PluginFN);
    if (!PassPlugin) {
      WithColor::error(errs(), Argv0)
          << "Failed to load passes from '" 
          << PluginFN
          << "'. Request ignored.\n";
      continue;
    }
    PassPlugin->registerPassBuilderCallbacks(PB);
  }
```

1.  静态插件注册表中的信息类似地用于向我们的`PassBuilder`实例注册这些插件：

```cpp
#define HANDLE_EXTENSION(Ext)                          \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks( \
      PB);
#include "llvm/Support/Extension.def"
```

1.  我们需要声明不同分析管理器的变量。唯一的参数是调试标志：

```cpp
  LoopAnalysisManager LAM(DebugPM);
  FunctionAnalysisManager FAM(DebugPM);
  CGSCCAnalysisManager CGAM(DebugPM);
  ModuleAnalysisManager MAM(DebugPM);
```

1.  接下来，我们通过在`PassBuilder`实例上调用相应的`register`方法来填充分析管理器。通过这个调用，分析管理器填充了默认的分析 Passes，并且还运行注册回调。我们还确保函数分析管理器使用默认的别名分析管道，并且所有分析管理器都知道彼此：

```cpp
  FAM.registerPass(
      [&] { return PB.buildDefaultAAPipeline(); });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
```

1.  `MPM`模块 Pass 管理器保存我们构建的 Pass 管道。该实例使用调试标志进行初始化：

```cpp
  ModulePassManager MPM(DebugPM);
```

1.  我们实现了两种不同的方法来填充模块 Pass 管理器与 Pass 管道。如果用户在命令行上提供了 Pass 管道，也就是说，他们使用了`--passes`选项，那么我们将使用这个作为 Pass 管道：

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

1.  否则，我们使用选择的优化级别来确定要构建的 Pass 管道。默认 Pass 管道的名称是`default`，它将优化级别作为参数：

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

1.  现在设置了在 IR 代码上运行转换的 Pass 管道。我们需要打开一个文件来写入结果。系统汇编器和 LLVM IR 输出都是基于文本的，因此我们应该为它们都设置`OF_Text`标志：

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

1.  对于代码生成，我们必须使用旧的 Pass 管理器。我们只需声明`CodeGenPM`实例并添加使目标特定信息在 IR 转换级别可用的 Pass：

```cpp
  legacy::PassManager CodeGenPM;
  CodeGenPM.add(createTargetTransformInfoWrapperPass(
      TM->getTargetIRAnalysis()));
```

1.  为了输出 LLVM IR，我们添加了一个只打印 IR 到流中的 Pass：

```cpp
  if (FileType == CGFT_AssemblyFile && EmitLLVM) {
    CodeGenPM.add(createPrintModulePass(Out->os()));
  }
```

1.  否则，我们让`TargetMachine`实例添加所需的代码生成 Passes，由我们作为参数传递的`FileType`值指导：

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

1.  经过所有这些准备，我们现在准备执行 Passes。首先，我们在 IR 模块上运行优化管道。接下来，运行代码生成 Passes。当然，在所有这些工作之后，我们希望保留输出文件：

```cpp
  MPM.run(*M, MAM);
  CodeGenPM.run(*M);
  Out->keep();
  return true;
}
```

1.  这是很多代码，但很简单。当然，我们还必须更新`tools/driver/CMakeLists.txt`构建文件中的依赖项。除了添加目标组件外，我们还从 LLVM 中添加所有转换和代码生成组件。名称大致类似于源代码所在的目录名称。在配置过程中，组件名称将被转换为链接库名称：

```cpp
set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD}
  AggressiveInstCombine Analysis AsmParser
  BitWriter CodeGen Core Coroutines IPO IRReader
  InstCombine Instrumentation MC ObjCARCOpts Remarks
  ScalarOpts Support Target TransformUtils Vectorize
  Passes)
```

1.  我们的编译器驱动程序支持插件，并宣布以下支持：

```cpp
add_tinylang_tool(tinylang Driver.cpp SUPPORT_PLUGINS)
```

1.  与以前一样，我们必须链接到我们自己的库：

```cpp
target_link_libraries(tinylang
  PRIVATE tinylangBasic tinylangCodeGen
  tinylangLexer tinylangParser tinylangSema)
```

这些是源代码和构建系统的必要补充。

1.  要构建扩展的编译器，请进入您的`build`目录并输入以下内容：

```cpp
$ ninja
```

构建系统的文件更改会自动检测到，并且在编译和链接我们更改的源代码之前运行`cmake`。如果您需要重新运行配置步骤，请按照*第二章*中的说明，*LLVM 源代码漫游*，*编译 tinylang 应用程序*部分中的说明进行操作。

由于我们已经使用`opt`工具的选项作为蓝图，您应该尝试使用加载 Pass 插件并运行 Pass 的选项来运行`tinylang`，就像我们在前面的部分中所做的那样。

通过当前的实现，我们可以运行默认的 Pass 管道或自己构建一个。后者非常灵活，但在几乎所有情况下都是过度的。默认管道非常适用于类似 C 的语言。缺少的是扩展 Pass 管道的方法。在下一节中，我们将解释如何实现这一点。

## 扩展 Pass 管道

在上一节中，我们使用`PassBuilder`类从用户提供的描述或预定义名称创建 Pass 管道。现在，我们将看另一种自定义 Pass 管道的方法：使用**扩展点**。

在构建 Pass 管道期间，Pass 构建器允许您添加用户贡献的 Passes。这些地方被称为扩展点。存在许多扩展点，例如以下：

+   管道开始扩展点允许您在管道开始时添加 Passes。

+   窥孔扩展点允许您在指令组合器 Pass 的每个实例之后添加 Passes。

还存在其他扩展点。要使用扩展点，您需要注册一个回调。在构建 Pass 管道期间，您的回调在定义的扩展点处运行，并可以向给定的 Pass 管理器添加 Pass。

要为管道开始扩展点注册回调，您需要调用`PassBuilder`类的`registerPipelineStartEPCallback()`方法。例如，要将我们的`CountIRPass` Pass 添加到管道的开头，您需要将 Pass 调整为使用`createModuleToFunctionPassAdaptor()`模板函数作为模块 Pass，并将 Pass 添加到模块 Pass 管理器中：

```cpp
PB.registerPipelineStartEPCallback(
    [](ModulePassManager &MPM) {
        MPM.addPass(
             createModuleToFunctionPassAdaptor(
                 CountIRPass());
    });
```

您可以在创建管道之前的任何时间点将此片段添加到 Pass 管道设置代码中，也就是在调用`parsePassPipeline()`方法之前。

在上一节所做的工作的自然扩展是让用户通过命令行传递管道描述。`opt`工具也允许这样做。让我们为管道开始扩展点做这个。首先，我们将以下代码添加到`tools/driver/Driver.cpp`文件中：

1.  我们为用户添加了一个新的命令行，用于指定管道描述。同样，我们从`opt`工具中获取选项名称：

```cpp
static cl::opt<std::string> PipelineStartEPPipeline(
    "passes-ep-pipeline-start",
    cl::desc("Pipeline start extension point));
```

1.  使用 lambda 函数作为回调是最方便的方式。为了解析管道描述，我们调用`PassBuilder`实例的`parsePassPipeline()`方法。Passes 被添加到`PM` Pass 管理器，并作为参数传递给 lambda 函数。如果出现错误，我们会打印错误消息而不会停止应用程序。您可以在调用`crossRegisterProxies()`方法之后添加此片段：

```cpp
  PB.registerPipelineStartEPCallback(
      &PB, Argv0 {
        if (auto Err = PB.parsePassPipeline(
                PM, PipelineStartEPPipeline)) {
          WithColor::error(errs(), Argv0)
              << "Could not parse pipeline "
              << PipelineStartEPPipeline.ArgStr 
              << ": "
              << toString(std::move(Err)) << "\n";
        }
      });
```

提示

为了允许用户在每个扩展点添加 Passes，您需要为每个扩展点添加前面的代码片段。

1.  现在是尝试不同`pass manager`选项的好时机。使用`--debug-pass-manager`选项，您可以跟踪执行 Passes 的顺序。您可以使用`--print-before-all`和`--print-after-all`选项在每次调用 Pass 之前或之后打印 IR。如果您创建自己的 Pass 管道，那么您可以在感兴趣的点插入`print` Pass。例如，尝试`--passes="print,inline,print"`选项。您还可以使用`print` Pass 来探索各种扩展点。

```cpp
    PassBuilder::OptimizationLevel Olevel = …;
    if (OLevel == PassBuilder::OptimizationLevel::O0)
      MPM.addPass(AlwaysInlinerPass());
    else
      MPM = PB.buildPerModuleDefaultPipeline(OLevel,           DebugPM);
```

当然，也可以以这种方式向 Pass 管理器添加多个 Pass。`PassBuilder`类在构建 Pass 管道期间还使用`addPass()`方法。

LLVM 12 中的新功能-运行扩展点回调

因为 Pass 管道在优化级别`O0`下没有填充，所以注册的扩展点不会被调用。如果您使用扩展点来注册应该在`O0`级别运行的 Passes，这将是有问题的。在 LLVM 12 中，可以调用新的`runRegisteredEPCallbacks()`方法来运行已注册的扩展点回调，从而使 Pass 管理器仅填充通过扩展点注册的 Passes。

通过将优化管道添加到`tinylang`中，您可以创建一个类似 clang 的优化编译器。LLVM 社区致力于在每个发布版本中改进优化和优化管道。因此，默认情况下很少不使用默认管道。通常情况下，会添加新的 Passes 来实现编程语言的某些语义。

# 总结

在本章中，您学会了如何为 LLVM 创建新的 Pass。您使用 Pass 管道描述和扩展点运行了 Pass。您通过构建和执行类似 clang 的 Pass 管道来扩展了您的编译器，将`tinylang`变成了一个优化编译器。Pass 管道允许您在扩展点添加 Passes，并且您学会了如何在这些点注册 Passes。这使您能够使用自己开发的 Passes 或现有 Passes 扩展优化管道。

在下一章中，我们将探讨 LLVM 如何从优化的 IR 生成机器指令。
