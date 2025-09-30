# 第九章：*第九章*：使用 PassManager 和 AnalysisManager

在本书的前一节*前端开发*中，我们开始介绍了 Clang 的内部结构，它是 LLVM 为 C 系列编程语言提供的官方前端。我们探讨了各种项目，涉及技能和知识，这些可以帮助你处理与源代码紧密相关的问题。

在本书的这一部分，我们将使用**LLVM IR** – 一种针对编译优化和代码生成的目标无关的**中间表示**（**IR**）。与 Clang 的**抽象语法树**（**AST**）相比，LLVM IR 通过封装额外的执行细节提供了不同层次的抽象，从而能够实现更强大的程序分析和转换。除了 LLVM IR 的设计，围绕这种 IR 格式还有一个成熟的生态系统，它提供了无数的资源，如库、工具和算法实现。我们将涵盖 LLVM IR 的多个主题，包括最常见的 LLVM Pass 开发、使用和编写程序分析，以及与 LLVM IR API 一起工作的最佳实践和技巧。此外，我们还将回顾更高级的技能，如**程序引导优化**（**PGO**）和 sanitizer 开发。

在本章中，我们将讨论编写用于新**PassManager**的转换**Pass**和程序分析。LLVM Pass 是整个项目中最为基础和关键的概念之一。它允许开发者将程序处理逻辑封装成一个模块化的单元，该单元可以通过**PassManager**根据情况自由组合其他 Pass。在 Pass 基础设施的设计方面，LLVM 实际上对 PassManager 和 AnalysisManager 都进行了彻底的改造，以提高它们的运行时性能和优化质量。新的 PassManager 为其封装的 Pass 使用了相当不同的接口。然而，这个新接口与旧接口不兼容，这意味着你无法在新 PassManager 中运行旧 Pass，反之亦然。更糟糕的是，网上关于这个新接口的学习资源并不多，尽管现在它们在 LLVM 和 Clang 中默认启用。本章的内容将填补这一空白，并提供关于 LLVM 中这一关键子系统的最新指南。

在本章中，我们将涵盖以下主题：

+   为新的 PassManager 编写 LLVM Pass

+   使用新的 AnalysisManager

+   在新的 PassManager 中学习 instrumentations

通过本章学习到的知识，你应该能够编写一个 LLVM Pass，使用新的 Pass 基础设施，来转换甚至优化你的输入代码。你也可以通过利用 LLVM 程序分析框架提供的分析数据来进一步提高你 Pass 的质量。

# 技术要求

在本章中，我们将主要使用一个名为`opt`的命令行工具来测试我们的 Pass。您可以使用以下命令构建它：

```cpp
$ ninja opt
```

本章的代码示例可以在[`github.com/PacktPublishing/LLVM-Techniques-Tips-and-Best-Practices-Clang-and-Middle-End-Libraries/tree/main/Chapter09`](https://github.com/PacktPublishing/LLVM-Techniques-Tips-and-Best-Practices-Clang-and-Middle-End-Libraries/tree/main/Chapter09)找到。

# 为新的 PassManager 编写 LLVM Pass

**LLVM 中的 Pass**是执行针对 LLVM IR 的某些操作所需的基本单元。它类似于工厂中的一个单一生产步骤，其中需要处理的产品是 LLVM IR，而工厂工人是 Pass。同样，一个正常的工厂通常有多个制造步骤，LLVM 也由多个按顺序执行的 Pass 组成，称为**Pass 流水线**。*图 9.1*显示了 Pass 流水线的一个示例：

![图 9.1 – LLVM Pass 流水线和其中间结果的示例

![img/Figure_9.1_B14590.jpg]

图 9.1 – LLVM Pass 流水线和其中间结果的示例

在前面的图中，多个 Pass 按直线排列。`foo`函数的 LLVM IR 被一个 Pass 接着另一个 Pass 处理。`foo`和将一个乘以 2 的算术乘法(`mul`)替换为左移(`shl`)1，这在大多数硬件架构中被认为比乘法更容易。此外，此图还说明了**代码生成**步骤被建模为 Pass。在 LLVM 中，代码生成将目标无关的 LLVM IR 转换为特定硬件架构的汇编代码（例如，*图 9.1*中的**x86_64**）。每个详细过程，如寄存器分配、指令选择或指令调度，都被封装到一个单独的 Pass 中，并按一定顺序执行。

代码生成 Pass

代码生成 Pass 具有与正常 LLVM IR Pass 不同的 API。此外，在代码生成阶段，LLVM IR 实际上被转换为另一种类型的 IR，称为**机器 IR**（**MIR**）。然而，在本章中，我们只将涵盖 LLVM IR 及其 Pass。

这个 Pass 流水线在概念上由一个名为**PassManager**的基础设施管理。PassManager 拥有计划 – 例如，它们的执行顺序 – 来运行这些 Pass。传统上，我们实际上使用*Pass 流水线*和*PassManager*这两个术语互换，因为它们几乎有相同的任务。在*新 PassManager 中的学习工具*部分，我们将更详细地介绍流水线本身，并讨论如何自定义这些封装 Pass 的执行顺序。

现代编译器中的代码转换可能很复杂。正因为如此，多个转换 Pass 可能需要相同的一组程序信息，这在 LLVM 中被称为**分析**，以便完成它们的工作。此外，为了达到最大效率，LLVM 还会**缓存**这些分析数据，以便在可能的情况下重用。然而，由于转换 Pass 可能会更改 IR，一些之前收集的缓存分析数据在运行该 Pass 后可能会过时。为了解决这些挑战，除了 PassManager 之外，LLVM 还创建了**AnalysisManager**来管理与程序分析相关的所有内容。我们将在*使用新的 AnalysisManager*部分中深入了解 AnalysisManager。

如本章引言中所述，LLVM 对其 Pass 和 PassManager（以及 AnalysisManager）基础设施进行了一系列的重大改造。新的基础设施运行速度更快，生成的结果质量更好。尽管如此，新的 Pass 与旧的一个有很多不同之处；我们将在途中简要解释这些差异。然而，除了这一点之外，我们将在本章的其余部分默认只讨论新的 Pass 基础设施。

在本节中，我们将向您展示如何为新的 PassManager 开发一个简单的 Pass。像往常一样，我们将从描述我们即将使用的示例项目开始。然后，我们将向您展示使用`opt`实用程序创建一个 Pass 的步骤，该 Pass 可以从插件动态加载到之前提到的 Pass 管道中。

## 项目概述

在本节中，我们使用的示例项目被称为`noalias`属性，将其应用于所有具有指针类型的函数参数。实际上，它向 C 代码中的函数参数添加了`restrict`关键字。首先，让我们解释一下`restrict`关键字的作用。

C 和 C++中的`restrict`关键字

`restrict`关键字是在 C99 中引入的。然而，它在 C++中没有对应的关键字。不过，主流编译器如 Clang、GCC 和 MSVS 都在 C++中支持相同的功能。例如，在 Clang 和 GCC 中，您可以在 C++代码中使用`__restrict__`或`__restrict`，它具有与 C 中`restrict`相同的效果。

`restrict`关键字也可以与 C 中的指针类型变量一起使用。在最常见的案例中，它与指针类型函数参数一起使用。以下是一个示例：

```cpp
int foo(int* restrict x, int* restrict y) {
  *x = *y + 1;
  return *y;
}
```

实际上，这个额外的属性告诉编译器，参数 `x` 永远不会指向与参数 `y` 相同的内存区域。换句话说，程序员可以使用这个关键字来*说服*编译器他们永远不会调用 `foo` 函数，如下所示：

```cpp
…
// Programmers will NEVER write the following code
int main() {
  int V = 1;
  return foo(&V, &V);
}
```

这背后的原因是，如果编译器知道两个指针——在这种情况下，两个指针参数——永远不会指向相同的内存区域，它可以进行更**激进**的优化。为了给您一个更具体的理解，如果您比较带有和没有 `restrict` 关键字的 `foo` 函数的汇编代码，后者版本在 x86_64 上执行需要五条指令：

```cpp
foo:                                    
     mov   eax, dword ptr [rsi]
     add   eax, 1
     mov   dword ptr [rdi], eax
     mov   eax, dword ptr [rsi]
     ret
```

添加了 `restrict` 关键字的版本仅需要四条指令：

```cpp
foo:                                    
     mov   eax, dword ptr [rsi]
     lea   ecx, [rax + 1]
     mov   dword ptr [rdi], ecx
     ret
```

虽然这里的差异看起来很微妙，但在没有 `restrict` 的版本中，编译器需要插入一个额外的内存加载操作来确保最后一个参数 `*y`（在原始 C 代码中）总是读取最新的值。这种额外的开销可能会在更复杂的代码库中逐渐累积，并最终成为性能瓶颈。

现在，您已经了解了 `restrict` 的工作原理及其在确保良好性能方面的重要性。在 LLVM IR 中，也有一个相应的指令来模拟 `restrict` 关键字：`noalias` 属性。如果程序员在原始源代码中给出了如 `restrict` 之类的提示，则此属性附加到指针函数参数上。例如，带有 `restrict` 关键字的 `foo` 函数可以转换为以下 LLVM IR：

```cpp
define i32 @foo(i32* noalias %0, i32* noalias %1) {
  %3 = load i32, i32* %1
  %4 = add i32 %3, 1
  store i32 %4, i32* %0
  ret i32 %3
}
```

此外，我们还可以在 C 代码中生成 `foo` 函数的 LLVM IR 代码，而不使用 `restrict`，如下所示：

```cpp
define i32 @foo(i32* %0, i32* %1) {
  %3 = load i32, i32* %1
  %4 = add i32 %3, 1
  store i32 %4, i32* %0
  %5 = load i32, i32* %1
  ret i32 %5
}
```

在这里，您会发现有一个额外的内存加载（如前一个片段中突出显示的指令所示），这与之前汇编示例中发生的情况类似。也就是说，LLVM 无法执行更激进的优化来删除该内存加载，因为它不确定这些指针是否重叠。

在本节中，我们将编写一个 Pass，将 `noalias` 属性添加到函数的每个指针参数。该 Pass 将作为插件构建，一旦加载到 `opt` 中，用户可以使用 `--passes` 参数显式触发 `StrictOpt`，如下所示：

```cpp
$ opt --load-pass-plugin=StrictOpt.so \
      --passes="function(strict-opt)" \
      -S -o – test.ll
```

或者，如果优化级别大于或等于 `-O3`，我们可以在其他优化之前运行 `StrictOpt`。以下是一个示例：

```cpp
$ opt -O3 --enable-new-pm \
      --load-pass-plugin=StrictOpt.so \
      -S -o – test.ll
```

我们将很快向您展示如何在这两种模式之间切换。

仅用于演示的 Pass

注意，`StrictOpt` 仅仅是一个仅用于演示的 Pass，并且将 `noalias` 添加到每个指针函数参数绝对不是您在现实世界用例中应该做的事情。这是因为这可能会破坏目标程序的**正确性**。

在下一节中，我们将向您展示创建此 Pass 的详细步骤。

## 编写 StrictOpt Pass

以下说明将引导您完成开发核心 Pass 逻辑的过程，然后再介绍如何动态地将 `StrictOpt` 注册到 Pass 管道中：

1.  这次我们只有两个源文件：`StrictOpt.h` 和 `StrictOpt.cpp`。在前者文件中，我们放置了 `StrictOpt` Pass 的框架：

    ```cpp
    #include "llvm/IR/PassManager.h"
    struct StrictOpt : public Function IR unit. The run method is the primary entry point for this Pass, which we are going to fill in later. It takes two arguments: a Function class that we will work on and a FunctionAnalysisManager class that can give you analysis data. It returns a PreservedAnalyses instance, which tells PassManager (and AnalysisManager) what analysis data was *invalidated* by this Pass.If you have prior experience in writing LLVM Pass for the *legacy* PassManager, you might find several differences between the legacy Pass and the new Pass:a) The Pass class no longer derives from one of the `FunctionPass`, `ModulePass`, or `LoopPass`. Instead, the Passes running on different IR units are all deriving from `PassInfoMixin<YOUR_PASS>`. In fact, deriving from `PassInfoMixin` is *not* even a requirement for a functional Pass anymore – we will leave this as an exercise for you.b) Instead of *overriding* methods, such as `runOnFunction` or `runOnModule`, you will define a normal class member method, `run` (be aware that `run` does *not* have an `override` keyword that follows), which operates on the desired IR unit.Overall, the new Pass has a cleaner interface compared to the legacy one. This difference also allows the new PassManager to have less overhead runtime.
    ```

1.  为了实现上一步骤中的框架，我们正在前往 `StrictOpt.cpp` 文件。在这个文件中，首先，我们创建以下方法定义：

    ```cpp
    #include "StrictOpt.h"
    using namespace llvm;
    PreservedAnalyses StrictOpt::run(Function &F,
                              FunctionAnalysisManager &FAM) {
      return PreservedAnalyses::all(); // Just a placeholder
    }
    ```

    返回的 `PreservedAnalyses::all()` 实例只是一个占位符，稍后将被移除。

1.  现在，我们最终正在创建代码来向指针函数参数添加 `noalias` 属性。逻辑很简单：对于 `Function` 类中的每个 `Argument` 实例，如果它满足条件，则附加 `noalias`：

    ```cpp
    // Inside StrictOpt::run…
    bool Modified = false;
    for (auto &Arg : F.args() method of the Function class will return a range of Argument instances representing all of the formal parameters. We check each of their types to make sure there isn't an existing noalias attribute (which is represented by the Attribute::NoAlias enum). If everything looks good, we use addAttr to attach noalias. Here, the `Modified` flag here records whether any of the arguments were modified in this function. We will use this flag shortly.
    ```

1.  由于转换 Pass 可能会改变程序的 IR，某些分析数据在转换后可能会过时。因此，在编写 Pass 时，我们需要返回一个 `PreservedAnalyses` 实例来显示哪些分析受到了影响，并且应该进行重新计算。虽然 LLVM 中有大量的分析可用，我们不需要逐一列举它们。相反，有一些方便的实用函数可以创建代表 *所有分析* 或 *没有分析* 的 `PreservedAnalyses` 实例，这样我们只需要从其中减去或添加（未）受影响的分析即可。以下是我们在 `StrictOpt` 中所做的工作：

    ```cpp
    #include "llvm/Analysis/AliasAnalysis.h"
    …
    // Inside StrictOpt::run…
    auto PA = PreservedAnalyses instance, PA, which represents *all analyses*. Then, if the Function class we are working on here has been modified, we *discard* the AAManager analysis via the abandon method. AAManager represents the noalias attribute we are discussing here has strong relations with this analysis since they're working on a nearly identical problem. Therefore, if any new noalias attribute was generated, all the cached alias analysis data would be outdated. This is why we invalidate it using abandon.Note that you can always return a `PreservedAnalyses::none()` instance, which tells AnalysisManager to mark *every* analysis as outdated if you are not sure what analyses have been affected. This comes at a cost, of course, since AnalysisManager then needs to spend extra effort to recalculate the analyses that might contain expensive computations.
    ```

1.  `StrictOpt` 的核心逻辑基本上已经完成。现在，我们将向您展示如何动态地将 Pass 注册到管道中。在 `StrictOpt.cpp` 中，我们创建了一个特殊的全局函数，称为 `llvmGetPassPluginInfo`，其轮廓如下：

    ```cpp
    extern "C" ::llvm::PassPluginLibraryInfo instance, which contains various piecesLLVM_PLUGIN_API_VERSION) and the Pass name (StrictOpt). One of its most important fields is a lambda function that takes a single PassBuilder& argument. In that particular function, we are going to insert our StrictOpt into a proper position within the Pass pipeline.`PassBuilder`, as its name suggests, is an entity LLVM that is used to build the Pass pipeline. In addition to its primary job, which involves configuring the pipeline according to the optimization level, it also allows developers to insert Passes into some of the places in the pipeline. Furthermore, to increase its flexibility, `PassBuilder` allows you to specify a *textual* description of the pipeline you want to run by using the `--passes` argument on `opt`, as we have seen previously. For instance, the following command will run `InstCombine`, `PromoteMemToReg`, and `SROA` (`opt` will run our Pass if `strict-opt` appears in the `--passes` argument, as follows:

    ```

    $ opt registerPipelineParsingCallback 方法在 PassBuilder 中：

    ```cpp
    …
    [](PassBuilder &PB) {
      using PipelineElement = typename PassBuilder::PipelineElement;
      PB.registerPipelineParsingCallback method takes another lambda callback as the argument. This callback is invoked whenever PassBuilder encounters an unrecognized Pass name while parsing the textual pipeline representation. Therefore, in our implementation, we simply insert our StrictOpt pass into the pipeline via FunctionPassManager::addPass when the unrecognized Pass name, that is, the Name parameter, is strict-opt.
    ```

    ```cpp

    ```

1.  或者，我们还想在 Pass 管道开始时触发我们的 `StrictOpt`，而不使用文本管道描述，正如我们在 *项目概述* 部分中描述的那样。这意味着在将 Pass 加载到 `opt` 中使用以下命令后，Pass 将在其他的 Pass 之前运行：

    ```cpp
    $ opt -O2 --enable-new-pm \
          --enable-new-pm flag in the preceding command forced opt to use the new PassManager since it's still using the legacy one by default. We haven't used this flag before because --passes implicitly enables the new PassManager under the hood.)To do this, instead of using `PassBuilder::registerPipelineParsingCallback` to register a custom (pipeline) parser callback, we are going to use `registerPipelineStartEPCallback` to handle this. Here is the alternative version of the code snippet from the previous step:

    ```

    …

    [](PassBuilder &PB) {

    using OptimizationLevel

    = typename PassBuilder::OptimizationLevel;

    PB.registerPipelineStartEPCallback(

    [](ModulePassManager &MPM, OptimizationLevel OL) {

    if (OL.getSpeedupLevel() >= 2) {

    MPM.addPass(

    createModuleToFunctionPassAdaptor(StrictOpt()));

    }

    });

    }

    ```cpp

    ```

在前面的代码片段中有几个值得注意的点：

+   我们在这里使用的 `registerPipelineStartEPCallback` 方法注册了一个回调，该回调可以自定义 Pass 管道中的某些位置，称为 **扩展点**（**EPs**）。我们在这里将要定制的 EP 是管道中最早的位置之一。

+   与我们在 `registerPipelineParsingCallback` 中看到的 lambda 回调相比，`registerPipelineStartEPCallback` 的 lambda 回调只提供 `ModulePassManager`，而不是 `FunctionPassManager`，以插入我们的 `StrictOpt` Pass，这是一个函数 Pass。我们使用 `ModuleToFunctionPassAdapter` 来解决这个问题。

    `ModuleToFunctionPassAdapter` 是一个模块 Pass，可以在模块的封装函数上运行给定的函数 Pass。它适用于仅在 `ModulePassManager` 可用的上下文中运行函数 Pass，例如在这个场景中。前面代码中突出显示的 `createModuleToFunctionPassAdaptor` 函数用于从一个特定的函数 Pass 创建一个新的 `ModuleToFunctionPassAdapter` 实例。

+   最后，在这个版本中，我们只有在优化级别大于或等于 `-O2` 时才启用 `StrictOpt`。因此，我们利用传递给 lambda 回调的 `OptimizationLevel` 参数来决定是否将 `StrictOpt` 插入到管道中。

    通过这些 Pass 注册步骤，我们还学习了如何触发我们的 `StrictOpt`，而无需显式指定文本 Pass 管道。

总结来说，在本节中，我们学习了 LLVM Pass 和 Pass 管道的要点。通过 `StrictOpt` 项目，我们学习了如何开发一个 Pass——它也被封装为插件——用于新的 PassManager，以及如何以两种不同的方式在 `opt` 中动态注册它：首先，通过通过文本描述显式触发 Pass，其次，在管道中的某个时间点（EP）运行它。我们还学习了如何根据 Pass 中所做的更改使分析无效。这些技能可以帮助您开发高质量和现代的 LLVM Pass，以最大灵活性以可组合的方式处理 IR。在下一节中，我们将深入了解 LLVM 的程序分析基础设施。这大大提高了普通 LLVM 转换 Pass 的能力。

# 使用新的 AnalysisManager

现代编译器的优化可能很复杂。它们通常需要从目标程序中获取大量信息，以便做出正确的决策和最优的转换。例如，在 *编写用于新 PassManager 的 LLVM Pass* 部分中，LLVM 使用了 `noalias` 属性来计算内存别名信息，这些信息最终可能被用来删除冗余的内存加载。

其中一些信息——在 LLVM 中称为 **analysis**——评估成本很高。此外，单个分析也可能依赖于其他分析。因此，LLVM 创建了一个 **AnalysisManager** 组件来处理与 LLVM 程序分析相关的所有任务。在本节中，我们将向您展示如何在自己的 Pass 中使用 AnalysisManager，以便编写更强大和复杂的程序转换或分析。我们还将使用一个示例项目，**HaltAnalyzer**，来驱动本教程。下一节将在详细介绍开发步骤之前，为您提供 HaltAnalyzer 的概述。

## 项目概述

HaltAnalyzer 是在一个场景中设置的，其中目标程序使用一个特殊函数`my_halt`，当它被调用时终止程序执行。`my_halt`函数类似于`std::terminate`函数，或者当其健全性检查失败时的`assert`函数。

HaltAnalyzer 的任务是分析程序，以找到由于`my_halt`函数而*保证无法到达*的基本块。更具体地说，让我们以下面的 C 代码为例：

```cpp
int foo(int x, int y) {
  if (x < 43) {
    my_halt();
    if (y > 45)
      return x + 1;
    else {
      bar();
      return x;
    }
  } else {
    return y;
  }
}
```

因为`my_halt`在`if (x < 43)`语句的真块开始时被调用，所以前面代码片段中高亮显示的代码永远不会被执行（即`my_halt`在到达这些行之前就停止了所有程序执行）。

HaltAnalyzer 应该识别这些基本块，并向`stderr`打印出警告信息。就像上一节中的示例项目一样，HaltAnalyzer 也是一个封装在插件中的函数 Pass。因此，如果我们使用前面的代码片段作为 HaltAnalyzer Pass 的输入，它应该打印出以下信息：

```cpp
$ opt --enable-new-pm --load-pass-plugin ./HaltAnalyzer.so \
      --disable-output ./test.ll
[WARNING] Unreachable BB: label %if.else
[WARNING] Unreachable BB: label %if.then2
$
```

`%if.else`和`%if.then2`字符串只是`if (y > 45)`语句中基本块的名称（你可能会看到不同的名称）。另一个值得注意的事情是`--disable-output`命令行标志。默认情况下，`opt`实用程序会打印出 LLVM IR 的二进制形式（即 LLVM 位码），除非用户通过`-o`标志将输出重定向到其他地方。使用上述标志只是为了告诉`opt`不要这样做，因为我们这次对 LLVM IR 的最终内容不感兴趣（因为我们不会对其进行修改）。

虽然 HaltAnalyzer 的算法看起来相当简单，但从零开始编写它可能是个头疼的问题。这就是为什么我们正在利用 LLVM 提供的一项分析：**支配树（DT）**。**控制流图（CFG）支配**的概念在大多数入门级编译器课程中都有讲解，所以我们在这里就不深入解释了。简单来说，如果我们说一个基本块*支配*另一个块，那么到达后者的每个执行流程都保证首先经过前者。DT 是 LLVM 中最重要且最常用的分析之一；大多数与控制流相关的转换都离不开它。

将这个想法应用到 HaltAnalyzer 中，我们只是在寻找所有被包含`my_halt`函数调用的基本块支配的基本块（我们在警告信息中排除了包含`my_halt`调用站点的基本块）。在下一节中，我们将向您展示如何编写 HaltAnalyzer 的详细说明。

## 编写 HaltAnalyzer Pass

在这个项目中，我们只创建一个源文件，`HaltAnalyzer.cpp`。大部分基础设施，包括`CMakeListst.txt`，都可以从上一节中的`StrictOpt`项目重用：

1.  在`HaltAnalyzer.cpp`内部，首先，我们创建以下 Pass 框架：

    ```cpp
    class HaltAnalyzer : public PassInfoMixin<HaltAnalyzer> {
      static constexpr const char* HaltFuncName = "my_halt";
      // All the call sites to "my_halt"
      SmallVector<Instruction*, 2> run method that we saw in the previous section, we are creating an additional method, findHaltCalls, which will collect all of the Instruction calls to my_halt in the current function and store them inside the Calls vector.
    ```

1.  让我们先实现`findHaltCalls`：

    ```cpp
    void HaltAnalyzer::findHaltCalls(Function &F) {
      Calls.clear();
      for (auto &I : llvm::instructions to iterate through every Instruction call in the current function and check them one by one. If the Instruction call is a CallInst – representing a typical function call site – and the callee name is my_halt, we will push it into the Calls vector for later use.Function name manglingBe aware that when a line of C++ code is compiled into LLVM IR or native code, the name of any symbol – including the function name – will be different from what you saw in the original source code. For example, a simple function that has the name of *foo* and takes no argument might have *_Z3foov* as its name in LLVM IR. We call such a transformation in C++ **name mangling**. Different platforms also adopt different name mangling schemes. For example, in Visual Studio, the same function name becomes *?foo@@YAHH@Z* in LLVM IR.
    ```

1.  现在，让我们回到`HaltAnalyzer::run`方法。我们将做两件事。我们将通过`findHaltCalls`收集对`my_halt`的调用位置，这是我们刚刚编写的，然后检索 DT 分析数据：

    ```cpp
    #include "llvm/IR/Dominators.h"
    …
    PreservedAnalyses
    HaltAnalyzer::run(Function &F, FunctionAnalysisManager type argument to retrieve specific analysis data (in this case, DominatorTree) for a specific Function class.Although, so far, we have (kind of) used the words *analysis* and *analysis data* interchangeably, in a real LLVM implementation, they are actually two different entities. Take the DT that we are using here as an example:a) `Function`. In other words, it is the one that *performs* the analysis.b) `DominatorTreeAnalysis`. This is just static data that will be cached by AnalysisManager until it is invalidated.Furthermore, LLVM asks every analysis to clarify its affiliated result type via the `Result` member type. For example, `DominatorTreeAnalysis::Result` is equal to `DominatorTree`.To make this even more formal, to associate the analysis data of an analysis class, `T`, with a `Function` variable, `F`, we can use the following snippet:

    ```

    // `FAM`是 FunctionAnalysisManager

    typename T::Result &Data = FAM.getResult<T>(F);

    ```cpp

    ```

1.  在我们检索到`DominatorTree`之后，是时候找到我们之前收集的所有由`Instruction`调用位置支配的基本块了：

    ```cpp
    PreservedAnalyses
    HaltAnalyzer::run(Function &F, FunctionAnalysisManager &FAM) {
      …
      SmallVector<BasicBlock*, 4> DomBBs;
      for (auto *I : Calls) {
        auto *BB = I->getParent();
        DomBBs.clear();
        DT.DominatorTree::getDescendants method, we can retrieve all of the basic blocks dominated by a my_halt call site. Note that the results from getDescendants will also contain the block you put into the query (in this case, the block containing the my_halt call sites), so we need to exclude it before printing the basic block name using the BasicBlock::printAsOperand method.With the ending of the returning `PreservedAnalyses::all()`, which tells AnalysisManager that this Pass does not invalidate any analysis since we don't modify the IR at all, we will wrap up the `HaltAnalyzer::run` method here.
    ```

1.  最后，我们需要将我们的 HaltAnalyzer Pass 动态地插入到 Pass pipeline 中。我们使用与上一节相同的方法，通过实现`llvmGetPassPluginInfo`函数并使用`PassBuilder`将我们的 Pass 放置在 pipeline 中的某个 EP（扩展点）：

    ```cpp
    extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
    llvmGetPassPluginInfo() {
      return {
        LLVM_PLUGIN_API_VERSION, "HaltAnalyzer", "v0.1",
        [](PassBuilder &PB) {
          using OptimizationLevel
            = typename PassBuilder::OptimizationLevel;
          PB.StrictOpt in the previous section, we are using registerOptimizerLastEPCallback to insert HaltAnalyzer *after* all of the other optimization Passes. The rationale behind this is that some optimizations might move basic blocks around, so prompting warnings too early might not be very useful. Nevertheless, we are still leveraging ModuletoFunctionPassAdaptor to wrap around our Pass; this is because registerOptimizerLastEPCallback only provides ModulePassManager for us to add our Pass, which is a function Pass.
    ```

这些都是实现我们的 HaltAnalyzer 所必需的步骤。现在你已经学会了如何使用 LLVM 的程序分析基础设施在 LLVM Pass 中获取有关目标程序更多信息。这些技能可以让你在开发 Pass 时对 IR 有更深入的了解。此外，这个基础设施允许你重用 LLVM 提供的优质、现成的程序分析算法，而不是自己重新造轮子。要浏览 LLVM 提供的所有分析，源树中的`llvm/include/llvm/Analysis`文件夹是一个很好的起点。这个文件夹中的大多数头文件都是独立的分析数据文件，你可以使用它们。

在本章的最后部分，我们将向您展示一些有用的诊断技术，这些技术对于调试 LLVM Pass 非常有用。

# 在新的 PassManager 中学习 instrumentations

LLVM 中的 PassManager 和 AnalysisManager 是复杂的软件组件。它们管理着数百个 Pass 和分析之间的交互，当我们试图诊断由它们引起的问题时，这可能是一个挑战。此外，编译器工程师修复编译器中的崩溃或**Miscompilation** bugs 是非常常见的。在这些情况下，有用的 instrumentation 工具可以为 Pass 和 Pass pipeline 提供洞察力，从而大大提高修复这些问题的效率。幸运的是，LLVM 已经提供了许多这样的工具。

Miscompilation

**Miscompilation** bugs usually refer to logical issues in the **compiled program**, which were introduced by compilers. For example, an overly aggressive compiler optimization removes certain loops that shouldn't be removed, causing the compiled software to malfunction, or mistakenly reorder memory barriers and create *race conditions* in the generated code.

我们将在接下来的每个部分中一次介绍一个工具。以下是它们的列表：

+   打印 Pass pipeline 详细信息

+   在每个 Pass 之后打印 IR 的变化

+   分割 Pass pipeline

这些工具可以在 `opt` 的命令行界面中交互。实际上，你还可以创建 *自己的* 仪器工具（甚至不需要更改 LLVM 源树！）；我们将把这个留给你作为练习。

## 打印 Pass 管道详细信息

当使用 `clang`（或 `opt`）时，我们熟悉许多不同的 `-O1`、`-O2` 或 `-Oz` 标志。每个优化级别都在运行 *不同集合的 Pass* 并以 *不同顺序* 安排它们。在某些情况下，这可能会极大地影响生成的代码，从性能或正确性方面来看。因此，有时了解这些配置对于获得我们将要处理的问题的清晰理解至关重要。

要打印出 `opt` 中所有 Pass 及其当前运行顺序，我们可以使用 `--debug-pass-manager` 标志。例如，给定以下 C 代码，`test.c`，我们将看到以下内容：

```cpp
int bar(int x) {
  int y = x;
  return y * 4;
}
int foo(int z) {
  return z + z * 2;
}
```

我们首先使用以下命令为其生成 IR：

```cpp
$ clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S test.c
```

`-disable-O0-optnone` 标志

默认情况下，`clang` 将在 `-O0` 优化级别下将特殊属性 `optnone` 附接到每个函数上。此属性将防止对附加函数进行任何进一步优化。在这里，`-disable-O0-optnone`（前端）标志阻止 `clang` 附上此属性。

然后，我们使用以下命令来打印出在 `-O2` 优化级别下运行的所有 Pass：

```cpp
$ opt -O2 --disable-output --debug-pass-manager test.ll
Starting llvm::Module pass manager run.
…
Running pass: Annotation2MetadataPass on ./test.ll
Running pass: ForceFunctionAttrsPass on ./test.ll
…
Starting llvm::Function pass manager run.
Running pass: SimplifyCFGPass on bar
Running pass: SROA on bar
Running analysis: DominatorTreeAnalysis on bar
Running pass: EarlyCSEPass on bar
…
Finished llvm::Function pass manager run.
…
Starting llvm::Function pass manager run.
Running pass: SimplifyCFGPass on foo
…
Finished llvm::Function pass manager run.
Invalidating analysis: VerifierAnalysis on ./test.ll
…
$
```

上述命令行输出告诉我们 `opt` 首先运行一组 *模块级* 优化；这些 Pass 的顺序（例如，`Annotation2MetadataPass` 和 `ForceFunctionAttrsPass`）也被列出。之后，对 `bar` 函数（例如，`SROA`）执行一系列 *函数级* 优化，然后再对这些优化应用于 `foo` 函数。此外，它还显示了管道中使用的分析（例如，`DominatorTreeAnalysis`），并提示我们有关它们因某个 Pass 而失效的消息。

总结来说，`--debug-pass-manager` 是一个有用的工具，可以窥探在特定优化级别下 Pass 管道运行的 Pass 及其顺序。了解这些信息可以帮助你获得 Pass 和分析如何与输入 IR 交互的整体图景。

## 打印每个 Pass 后的 IR 变化

要了解特定转换 Pass 对目标程序的影响，最直接的方法之一是比较该 Pass 处理前后的 IR。更具体地说，在大多数情况下，我们感兴趣的是特定转换 Pass 所做的 *更改*。例如，如果 LLVM 错误地删除了它不应该删除的循环，我们想知道是哪个 Pass 做了这件事，以及 Pass 管道中删除发生的时间。

通过使用`--print-changed`标志（以及我们将很快介绍的某些其他支持的标志）与`opt`结合，我们可以在每次 Pass 修改 IR 的情况下打印出 IR。使用上一段中的`test.c`（及其 IR 文件`test.ll`）示例代码，我们可以使用以下命令来打印变化，如果有任何变化的话：

```cpp
$ opt -O2 --disable-output --print-changed ./test.ll
*** IR Dump At Start: ***
...
define dso_local i32 @bar(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  %y = alloca i32, align 4
  …
  %1 = load i32, i32* %y, align 4
  %mul = mul nsw i32 %1, 4
  ret i32 %mul
}
...
*** IR Dump After VerifierPass (module) omitted because no change ***
…
...
*** IR Dump After SROA *** (function: bar)
; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 %x) #0 {
entry:
  %mul = mul nsw i32 %x, 4
  ret i32 %mul
}
...
$
```

在这里，我们只展示了少量输出。然而，在代码片段的高亮部分，我们可以看到这个工具将首先打印出原始 IR（`IR Dump At Start`），然后显示每个 Pass 处理后的 IR。例如，前面的代码片段显示，经过 SROA Pass 后，`bar`函数变得短得多。如果一个 Pass 根本未修改 IR，它将省略 IR 转储以减少噪声。

有时候，我们只对特定函数集上的变化感兴趣，比如在这个例子中的`foo`函数。而不是打印整个模块的*变更日志*，我们可以添加`--filter-print-funcs=<function names>`标志来仅打印函数子集的 IR 变化。例如，要仅打印`foo`函数的 IR 变化，可以使用以下命令：

```cpp
$ opt -O2 --disable-output \
          --print-changed --filter-print-funcs=foo ./test.ll
```

就像`--filter-print-funcs`一样，有时候我们只想看到特定 Pass 集所做的变化，比如 SROA 和`InstCombine` Pass。在这种情况下，我们可以添加`--filter-passes=<Pass names>`标志。例如，要仅查看与 SROA 和`InstCombine`相关的内容，可以使用以下命令：

```cpp
$ opt -O2 --disable-output \
          --print-changed \
          --filter-passes=SROA,InstCombinePass ./test.ll
```

现在你已经学会了如何打印管道中所有 Pass 的 IR 差异，并使用额外的过滤器进一步关注特定的函数或 Pass。换句话说，这个工具可以帮助你轻松观察 Pass 管道中变化的*进展*，并快速找到你可能感兴趣的任何痕迹。在下一节中，我们将学习如何通过*二分法*Pass 管道来调试代码优化中提出的问题。

## 二分 Pass 管道

在前几节中，我们介绍了`--print-changed`标志，该标志在整个 Pass 管道中打印出*IR 变更日志*。我们还提到，调用我们感兴趣的变化是有用的；例如，一个导致误编译错误的无效代码转换。或者，我们也可以在`opt`中使用`--opt-bisect-limit=<N>`标志，通过*禁用*除了前 N 个之外的所有 Pass 来二分 Pass 管道。以下命令展示了这个示例：

```cpp
$ opt -O2 --opt-bisect-limit=5 -S -o – test.ll
BISECT: running pass (1) Annotation2MetadataPass on module (./test.ll)
BISECT: running pass (2) ForceFunctionAttrsPass on module (./test.ll)
BISECT: running pass (3) InferFunctionAttrsPass on module (./test.ll)
BISECT: running pass (4) SimplifyCFGPass on function (bar)
BISECT: running pass (5) SROA on function (bar)
BISECT: NOT running pass (6) EarlyCSEPass on function (bar)
BISECT: NOT running pass (7) LowerExpectIntrinsicPass on function (bar)
BISECT: NOT running pass (8) SimplifyCFGPass on function (foo)
BISECT: NOT running pass (9) SROA on function (foo)
BISECT: NOT running pass (10) EarlyCSEPass on function (foo)
...
define dso_local i32 @bar(i32 %x) #0 {
entry:
  %mul = mul nsw i32 %x, 4
  ret i32 %mul
}
define dso_local i32 @foo(i32 %y) #0 {
entry:
  %y.addr = alloca i32, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %y.addr, align 4
  %1 = load i32, i32* %y.addr, align 4
  %mul = mul nsw i32 %1, 2
  %add = add nsw i32 %0, %mul
  ret i32 %add
}
$
```

（请注意，这与前几节中显示的示例不同；前面的命令已打印出`--opt-bisect-limit`和最终文本 IR 的消息。）

由于我们实现了`--opt-bisect-limit=5`标志，Pass 管道只运行了前五个 Pass。正如诊断消息所示，SROA 应用于`bar`函数，但没有应用于`foo`函数，导致`foo`函数的最终 IR 不太优化。

通过更改 `--opt-bisect-limit` 后面的数字，我们可以调整截止点，直到出现某些代码更改或触发某个特定错误（例如，崩溃）。这特别有用，可以作为 *早期过滤步骤* 来缩小原始问题在管道中 Pass 的范围。此外，由于它使用数值作为参数，这个特性非常适合自动化环境，例如自动崩溃报告工具或性能回归跟踪工具。

在本节中，我们介绍了 `opt` 中的一些有用的仪器工具，用于调试和诊断 Pass 管道。这些工具可以大大提高您在修复问题时的生产力，例如编译器崩溃、性能回归（在目标程序上）和误编译错误。

# 摘要

在本章中，我们学习了如何为新的 PassManager 编写 LLVM Pass，以及如何通过 AnalysisManager 在 Pass 中使用程序分析数据。我们还学习了如何利用各种仪器工具来改善与 Pass 管道一起工作的开发体验。通过本章获得的知识，您现在可以编写一个处理 LLVM IR 的 Pass，这可以用来转换甚至优化程序。

这些主题是在开始任何 IR 级别的转换或分析任务之前需要学习的最基本和最重要的技能之一。如果您一直在使用传统的 PassManager，这些技能也可以帮助您将代码迁移到新的 PassManager 系统，该系统现在已被默认启用。

在下一章中，我们将向您展示在使用 LLVM IR 的 API 时，您应该知道的各项技巧和最佳实践。

# 问题

1.  在“为新的 PassManager 编写 LLVM Pass”部分的 `StrictOpt` 示例中，您如何在不需要派生 `PassInfoMixin` 类的情况下编写一个 Pass？

1.  您如何为新的 PassManager 开发自定义的仪器？此外，您如何在不修改 LLVM 源树的情况下做到这一点？（提示：想想我们在本章中学到的 Pass 插件。）
