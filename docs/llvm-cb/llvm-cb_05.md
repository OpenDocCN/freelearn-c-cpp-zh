# 第五章：实现优化

在本章中，我们将涵盖以下食谱：

+   编写死代码消除传递

+   编写内联转换传递

+   编写内存优化传递

+   结合 LLVM IR

+   转换和优化循环

+   重新关联表达式

+   向量化 IR

+   其他优化传递

# 简介

在上一章中，我们看到了如何在 LLVM 中编写传递。我们还通过别名分析示例演示了编写几个分析传递。这些传递只是读取源代码并提供了有关它的信息。在本章中，我们将进一步编写转换传递，这些传递实际上会更改源代码，试图优化代码以实现更快的执行。在前两个食谱中，我们将向您展示如何编写转换传递以及它是如何更改代码的。之后，我们将看到我们如何可以在传递的代码中进行更改，以调整传递的行为。

# 编写死代码消除传递

在本食谱中，你将学习如何从程序中消除死代码。通过消除死代码，我们指的是移除对源程序执行输出的结果没有任何影响的代码。这样做的主要原因包括减少程序大小，从而提高代码质量并使代码更容易调试；以及提高程序的运行时间，因为不必要的代码被阻止执行。在本食谱中，我们将向您展示一种死代码消除的变体，称为激进死代码消除，它假设每一段代码都是死代码，直到证明其不是为止。我们将看到如何自己实现这个传递，以及我们需要对传递进行哪些修改，以便它可以在 LLVM 主干的`lib/Transforms/Scalar`文件夹中的其他传递一样运行。

## 准备工作

为了展示死代码消除的实现，我们需要一段测试代码，我们将在这个代码上运行激进死代码消除传递：

```cpp
$ cat testcode.ll
declare i32 @strlen(i8*) readonly nounwind
define void @test() {
 call i32 @strlen( i8* null )
 ret void
}

```

在这段测试代码中，我们可以看到在`test`函数中调用了`strlen`函数，但返回值没有被使用。因此，这应该被我们的传递视为死代码。

在文件中，包含位于`/llvm/`的`InitializePasses.h`文件；并在`llvm`命名空间中，添加我们即将编写的传递的条目：

```cpp
namespace llvm {
…
…
void initializeMYADCEPass(PassRegistry&);    // Add this line
```

在`scalar.h`文件中，位于`include/llvm-c/scalar.h/Transform/`，添加传递的条目：

```cpp
void LLVMAddMYAggressiveDCEPass(LLVMPassManagerRef PM);
```

在`include/llvm/Transform/scalar.h`文件中，在`llvm`命名空间中添加传递的条目：

```cpp
FunctionPass *createMYAggressiveDCEPass();
```

在`lib/Transforms/Scalar/scalar.cpp`文件中，在两个地方添加传递的条目。在`void llvm::initializeScalarOpts(PassRegistry &Registry)`函数中，添加以下代码：

```cpp
initializeMergedLoadStoreMotionPass(Registry);  // already present in the file
initializeMYADCEPass(Registry);    // add this line
initializeNaryReassociatePass(Registry);  // already present in the file
…
…
void LLVMAddMemCpyOptPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMemCpyOptPass());
}

// add the following three lines
void LLVMAddMYAggressiveDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMYAggressiveDCEPass());
}

void LLVMAddPartiallyInlineLibCallsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPartiallyInlineLibCallsPass());
}
…
```

## 如何做到这一点…

我们现在将编写传递的代码：

1.  包含必要的头文件：

    ```cpp
    #include "llvm/Transforms/Scalar.h"
    #include "llvm/ADT/DepthFirstIterator.h"
    #include "llvm/ADT/SmallPtrSet.h"
    #include "llvm/ADT/SmallVector.h"
    #include "llvm/ADT/Statistic.h"
    #include "llvm/IR/BasicBlock.h"
    #include "llvm/IR/CFG.h"
    #include "llvm/IR/InstIterator.h"
    #include "llvm/IR/Instructions.h"
    #include "llvm/IR/IntrinsicInst.h"
    #include "llvm/Pass.h"
    using namespace llvm;
    ```

1.  声明我们的传递结构：

    ```cpp
    namespace {
    struct MYADCE : public FunctionPass {
      static char ID; // Pass identification, replacement for typeid
      MYADCE() : FunctionPass(ID) {
        initializeMYADCEPass(*PassRegistry::getPassRegistry());
      }

      bool runOnFunction(Function& F) override;

      void getAnalysisUsage(AnalysisUsage& AU) const override {
        AU.setPreservesCFG();
      }
    };
    }
    ```

1.  初始化传递及其 ID：

    ```cpp
    char MYADCE::ID = 0;
    INITIALIZE_PASS(MYADCE, "myadce", "My Aggressive Dead Code Elimination", false, false)
    ```

1.  在`runOnFunction`函数中实现实际的传递：

    ```cpp
    bool MYADCE::runOnFunction(Function& F) {
      if (skipOptnoneFunction(F))
        return false;

      SmallPtrSet<Instruction*, 128> Alive;
      SmallVector<Instruction*, 128> Worklist;

      // Collect the set of "root" instructions that are known live.
      for (Instruction &I : inst_range(F)) {
        if (isa<TerminatorInst>(I) || isa<DbgInfoIntrinsic>(I) || isa<LandingPadInst>(I) || I.mayHaveSideEffects()) {
          Alive.insert(&I);
          Worklist.push_back(&I);
        }
      }

      // Propagate liveness backwards to operands.
      while (!Worklist.empty()) {
        Instruction *Curr = Worklist.pop_back_val();
        for (Use &OI : Curr->operands()) {
          if (Instruction *Inst = dyn_cast<Instruction>(OI))
            if (Alive.insert(Inst).second)
              Worklist.push_back(Inst);
        }
      }

    // the instructions which are not in live set are considered dead in this pass. The instructions which do not effect the control flow, return value and do not have any side effects are hence deleted.
      for (Instruction &I : inst_range(F)) {
        if (!Alive.count(&I)) {
          Worklist.push_back(&I);
          I.dropAllReferences();
        }
      }

      for (Instruction *&I : Worklist) {
        I->eraseFromParent();
      }

      return !Worklist.empty();
    }
    }

    FunctionPass *llvm::createMYAggressiveDCEPass() {
      return new MYADCE();
    }
    ```

1.  在编译`testcode.ll`文件后运行前面的 pass，该文件可以在本教程的*准备工作*部分找到：

    ```cpp
    $ opt -myadce -S testcode.ll

    ; ModuleID = 'testcode.ll'

    ; Function Attrs: nounwind readonly
    declare i32 @strlen(i8*) #0

    define void @test() {
     ret void
    }

    ```

## 它是如何工作的...

此 pass 通过首先在`runOnFunction`函数的第一个`for`循环中收集所有活跃的根指令列表来工作。

使用这些信息，我们在`while` `(!Worklist.empty())`循环中向后传播活跃性到操作数。

在下一个`for`循环中，我们移除不活跃的指令，即死代码。同时，我们检查是否对这些值有任何引用。如果有，我们将丢弃所有这样的引用，它们也是死代码。

在测试代码上运行此 pass 后，我们看到死代码；`strlen`函数的调用被移除。

注意，代码已被添加到 LLVM 主分支修订号 234045 中。因此，当您实际尝试实现它时，一些定义可能会更新。在这种情况下，相应地修改代码。

## 参见

对于其他各种死代码消除方法，您可以参考`llvm/lib/Transforms/Scalar`文件夹，其中包含其他类型 DCEs 的代码。

# 编写内联转换 pass

正如我们所知，内联意味着在调用点展开被调用函数的函数体，因为它可能通过代码的更快执行而变得有用。编译器决定是否内联一个函数。在本教程中，您将学习如何编写一个简单的函数内联 pass，该 pass 利用 LLVM 的内联实现。我们将编写一个处理带有`alwaysinline`属性的函数的 pass。

## 准备工作

让我们编写一个测试代码，我们将在这个测试代码上运行我们的 pass。在`lib/Transforms/IPO/IPO.cpp`和`include/llvm/InitializePasses.h`文件、`include/llvm/Transforms/IPO.h`文件以及`/include/llvm-c/Transforms/IPO.h`文件中做出必要的更改，以包含以下 pass。还需要对`makefile`进行必要的更改以包含此 pass：

```cpp
$ cat testcode.c
define i32 @inner1() alwaysinline {
 ret i32 1
}
define i32 @outer1() {
 %r = call i32 @inner1()
 ret i32 %r
}

```

## 如何做这件事...

我们现在将编写 pass 的代码：

1.  包含必要的头文件：

    ```cpp
    #include "llvm/Transforms/IPO.h"
    #include "llvm/ADT/SmallPtrSet.h"
    #include "llvm/Analysis/AliasAnalysis.h"
    #include "llvm/Analysis/AssumptionCache.h"
    #include "llvm/Analysis/CallGraph.h"
    #include "llvm/Analysis/InlineCost.h"
    #include "llvm/IR/CallSite.h"
    #include "llvm/IR/CallingConv.h"
    #include "llvm/IR/DataLayout.h"
    #include "llvm/IR/Instructions.h"
    #include "llvm/IR/IntrinsicInst.h"
    #include "llvm/IR/Module.h"
    #include "llvm/IR/Type.h"
    #include "llvm/Transforms/IPO/InlinerPass.h"

    using namespace llvm;

    ```

1.  描述我们的 pass 的类：

    ```cpp
    namespace {

    class MyInliner : public Inliner {
     InlineCostAnalysis *ICA;

    public:
     MyInliner() : Inliner(ID, -2000000000,
    /*InsertLifetime*/ true),
     ICA(nullptr) {
     initializeMyInlinerPass(*PassRegistry::getPassRegistry());
     }

     MyInliner(bool InsertLifetime)
     : Inliner(ID, -2000000000, InsertLifetime), ICA(nullptr) {
     initializeMyInlinerPass(*PassRegistry::getPassRegistry());
     }

     static char ID;

     InlineCost getInlineCost(CallSite CS) override;

     void getAnalysisUsage(AnalysisUsage &AU) const override;
     bool runOnSCC(CallGraphSCC &SCC) override;

     using llvm::Pass::doFinalization;
     bool doFinalization(CallGraph &CG) override {
     return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/ true);
     }
    };
    }

    ```

1.  初始化 pass 并添加依赖项：

    ```cpp
    char MyInliner::ID = 0;
    INITIALIZE_PASS_BEGIN(MyInliner, "my-inline",
     "Inliner for always_inline functions", false, false)
    INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
    INITIALIZE_PASS_DEPENDENCY(AssumptionTracker)
    INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
    INITIALIZE_PASS_DEPENDENCY(InlineCostAnalysis)
    INITIALIZE_PASS_END(MyInliner, "my-inline",
     "Inliner for always_inline functions", false, false)

    Pass *llvm::createMyInlinerPass() { return new MyInliner(); }

    Pass *llvm::createMynlinerPass(bool InsertLifetime) {
     return new MyInliner(InsertLifetime);
    }

    ```

1.  实现获取内联成本的函数：

    ```cpp
    InlineCost MyInliner::getInlineCost(CallSite CS) {
     Function *Callee = CS.getCalledFunction();
    if (Callee && !Callee->isDeclaration() &&
     CS.hasFnAttr(Attribute::AlwaysInline) &&
     ICA->isInlineViable(*Callee))
     return InlineCost::getAlways();

     return InlineCost::getNever();
    }

    ```

1.  编写其他辅助方法：

    ```cpp
    bool MyInliner::runOnSCC(CallGraphSCC &SCC) {
     ICA = &getAnalysis<InlineCostAnalysis>();
     return Inliner::runOnSCC(SCC);
    }

    void MyInliner::getAnalysisUsage(AnalysisUsage &AU) const {
     AU.addRequired<InlineCostAnalysis>();
     Inliner::getAnalysisUsage(AU);
    }

    ```

1.  编译通过。编译完成后，在先前的测试用例上运行它：

    ```cpp
    $ opt -inline-threshold=0 -always-inline -S test.ll

    ; ModuleID = 'test.ll'

    ; Function Attrs: alwaysinline
    define i32 @inner1() #0 {
     ret i32 1
    }
    define i32 @outer1() {
     ret i32 1
    }

    ```

## 它是如何工作的...

我们编写的这个 pass 将适用于具有`alwaysinline`属性的函数。这个 pass 将始终内联这些函数。

这里工作的主函数是`InlineCost` `getInlineCost(CallSite` `CS`)`。这是一个位于`inliner.cpp`文件中的函数，需要在这里重写。因此，基于这里计算的内联成本，我们决定是否内联一个函数。内联过程的实际实现，即内联过程是如何工作的，可以在`inliner.cpp`文件中找到。

在这种情况下，我们返回 `InlineCost::getAlways()`；对于带有 `alwaysinline` 属性的函数。对于其他函数，我们返回 `InlineCost::getNever()`。这样，我们可以实现这个简单情况的内联。如果你想要深入了解并尝试其他内联变体——以及学习如何做出内联决策——你可以查看 `inlining.cpp` 文件。

当这个传递在测试代码上运行时，我们看到 `inner1` 函数的调用被其实际函数体所替换。

# 编写内存优化传递

在这个配方中，我们将简要讨论一个处理内存优化的转换传递。

## 准备工作

对于这个配方，你需要安装 opt 工具。

## 如何实现它…

1.  编写我们将运行 `memcpy` 优化传递的测试代码：

    ```cpp
    $ cat memcopytest.ll
    @cst = internal constant [3 x i32] [i32 -1, i32 -1, i32 -1], align 4

    declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
    declare void @foo(i32*) nounwind

    define void @test1() nounwind {
     %arr = alloca [3 x i32], align 4
     %arr_i8 = bitcast [3 x i32]* %arr to i8*
     call void @llvm.memcpy.p0i8.p0i8.i64(i8* %arr_i8, i8* bitcast ([3 x i32]* @cst to i8*), i64 12, i32 4, i1 false)
     %arraydecay = getelementptr inbounds [3 x i32], [3 x i32]* %arr, i64 0, i64 0
     call void @foo(i32* %arraydecay) nounwind
     ret void
    }

    ```

1.  在前面的测试用例上运行 `memcpyopt` 传递：

    ```cpp
    $ opt -memcpyopt -S memcopytest.ll
    ; ModuleID = ' memcopytest.ll'

    @cst = internal constant [3 x i32] [i32 -1, i32 -1, i32 -1], align 4

    ; Function Attrs: nounwind
    declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #0

    ; Function Attrs: nounwind
    declare void @foo(i32*) #0

    ; Function Attrs: nounwind
    define void @test1() #0 {
     %arr = alloca [3 x i32], align 4
     %arr_i8 = bitcast [3 x i32]* %arr to i8*
     call void @llvm.memset.p0i8.i64(i8* %arr_i8, i8 -1, i64 12, i32 4, i1 false)
     %arraydecay = getelementptr inbounds [3 x i32]* %arr, i64 0, i64 0
     call void @foo(i32* %arraydecay) #0
     ret void
    }

    ; Function Attrs: nounwind
    declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #0

    attributes #0 = { nounwind }

    ```

## 它是如何工作的…

`Memcpyopt` 传递处理尽可能消除 `memcpy` 调用，或将它们转换为其他调用。

考虑这个 `memcpy` 调用：

`call void @llvm.memcpy.p0i8.p0i8.i64(i8* %arr_i8, i8* bitcast ([3 x i32]* @cst to i8*), i64 12, i32 4, i1 false)`。

在前面的测试用例中，这个传递将其转换为 `memset` 调用：

`call void @llvm.memset.p0i8.i64(i8* %arr_i8, i8 -1, i64 12, i32 4, i1 false)`。

如果我们查看传递的源代码，我们会意识到这种转换是由 `llvm/lib/Transforms/Scalar` 文件夹中的 `MemCpyOptimizer.cpp` 文件中的 `tryMergingIntoMemset` 函数引起的。

`tryMergingIntoMemset` 函数在扫描指令时会寻找一些其他模式来折叠。它寻找相邻内存中的存储，并在看到连续的存储时，尝试将它们合并到 `memset` 中。

`processMemSet` 函数会检查这个 `memset` 附近的任何其他相邻的 `memset`，这有助于我们扩展 `memset` 调用来创建一个更大的存储。

## 参见

要查看各种内存优化传递的详细信息，请访问 [`llvm.org/docs/Passes.html#memcpyopt-memcpy-optimization`](http://llvm.org/docs/Passes.html#memcpyopt-memcpy-optimization)。

# 结合 LLVM IR

在这个配方中，你将了解 LLVM 中的指令组合。通过指令组合，我们指的是用更有效的指令替换一系列指令，这些指令在更少的机器周期内产生相同的结果。在这个配方中，我们将看到我们如何修改 LLVM 代码以组合某些指令。

## 入门

为了测试我们的实现，我们将编写测试代码，我们将使用它来验证我们的实现是否正确地组合了指令：

```cpp
define i32 @test19(i32 %x, i32 %y, i32 %z) {
 %xor1 = xor i32 %y, %z
 %or = or i32 %x, %xor1
 %xor2 = xor i32 %x, %z
 %xor3 = xor i32 %xor2, %y
 %res = xor i32 %or, %xor3
 ret i32 %res
}
```

## 如何实现它…

1.  打开 `lib/Transforms/InstCombine/InstCombineAndOrXor.cpp` 文件。

1.  在 `InstCombiner::visitXor(BinaryOperator &I)` 函数中，进入 `if` 条件——`if `(Op0I && Op1I)`——并添加以下内容：

    ```cpp
    if (match(Op0I, m_Or(m_Xor(m_Value(B), m_Value(C)), m_Value(A))) &&
            match(Op1I, m_Xor( m_Xor(m_Specific(A), m_Specific(C)), m_Specific(B)))) {
          return BinaryOperator::CreateAnd(A, Builder->CreateXor(B,C)); }
    ```

1.  现在重新构建 LLVM，以便 Opt 工具可以使用新功能并按这种方式运行测试用例：

    ```cpp
    Opt –instcombine –S testcode.ll
    define i32 @test19(i32 %x, i32 %y, i32 %z) {
    %1 = xor i32 %y, %z
     %res = and i32 %1, %x
     ret i32 %res
    }

    ```

## 它是如何工作的…

在这个配方中，我们在指令组合文件中添加了代码，该代码处理涉及 AND、OR 和 XOR 运算符的转换。

我们添加了匹配`(A` `|` `(B` `^` `C))` `^` `((A` `^` `C)` `^` `B)`形式的代码，并将其简化为`A` `&` `(B` `^` `C)`。`if (match(Op0I, m_Or(m_Xor(m_Value(B), m_Value(C)), m_Value(A))) && match(Op1I, m_Xor( m_Xor(m_Specific(A), m_Specific(C)), m_Specific(B))))`这一行寻找与段落开头所示模式相似的图案。

`return` `BinaryOperator::CreateAnd(A,` `Builder->CreateXor(B,C));`这一行在构建新指令后返回简化后的值，替换了之前匹配的代码。

当我们在测试代码上运行`instcombine`优化步骤时，我们得到减少后的结果。你可以看到操作数从五个减少到两个。

## 参考以下内容

+   指令组合的主题非常广泛，有大量的可能性。与指令组合功能相似的是指令简化功能，我们在其中简化复杂的指令，但不一定减少指令的数量，就像指令组合那样。要深入了解这一点，请查看`lib/Transforms/InstCombine`文件夹中的代码。

# 转换和优化循环

在这个配方中，我们将了解如何转换和优化循环以获得更短的执行时间。我们将主要关注**循环不变代码移动**（**LICM**）优化技术，并了解其工作原理和代码转换方式。我们还将查看一个相对简单的技术，称为**循环删除**，其中我们消除那些具有非无限、可计算的循环次数且对函数返回值没有副作用的无用循环。

## 准备工作

你必须为这个配方构建 opt 工具。

## 如何做…

1.  为 LICM（Loop-Invariant Code Motion）优化步骤编写测试用例：

    ```cpp
    $ cat testlicm.ll
    define void @testfunc(i32 %i) {
    ; <label>:0
     br label %Loop
    Loop:        ; preds = %Loop, %0
     %j = phi i32 [ 0, %0 ], [ %Next, %Loop ]        ; <i32> [#uses=1]
     %i2 = mul i32 %i, 17        ; <i32> [#uses=1]
     %Next = add i32 %j, %i2        ; <i32> [#uses=2]
     %cond = icmp eq i32 %Next, 0        ; <i1> [#uses=1]
     br i1 %cond, label %Out, label %Loop
    Out:        ; preds = %Loop
     ret void
    }

    ```

1.  在以下测试代码上执行 LICM 优化步骤：

    ```cpp
    $ opt licmtest.ll -licm -S
    ; ModuleID = 'licmtest.ll'

    define void @testfunc(i32 %i) {
     %i2 = mul i32 %i, 17
     br label %Loop

    Loop:                                             ; preds = %Loop, %0
     %j = phi i32 [ 0, %0 ], [ %Next, %Loop ]
     %Next = add i32 %j, %i2
     %cond = icmp eq i32 %Next, 0
     br i1 %cond, label %Out, label %Loop

    Out:                                              ; preds = %Loop
     ret void
    }

    ```

1.  为循环删除优化步骤编写测试代码：

    ```cpp
    $ cat deletetest.ll
    define void @foo(i64 %n, i64 %m) nounwind {
    entry:
     br label %bb

    bb:
     %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb2 ]
     %t0 = add i64 %x.0, 1
     %t1 = icmp slt i64 %x.0, %n
     br i1 %t1, label %bb2, label %return
    bb2:
     %t2 = icmp slt i64 %x.0, %m
     br i1 %t1, label %bb, label %return

    return:
     ret void
    }

    ```

1.  最后，在测试代码上运行循环删除优化步骤：

    ```cpp
    $ opt deletetest.ll -loop-deletion -S
    ; ModuleID = "deletetest.ll'

    ; Function Attrs: nounwind
    define void @foo(i64 %n, i64 %m) #0 {
    entry:
     br label %return

    return:                                           ; preds = %entry
     ret void
    }

    attributes #0 = { nounwind }

    ```

## 它是如何工作的…

LICM 优化步骤执行循环不变代码移动；它试图将循环中未修改的代码移出循环。它可以移动到循环前头的代码块之上，或者从退出块之后退出循环。

在前面显示的示例中，我们看到了代码的`%i2` `=` `mul` `i32` `%i,` `17`部分被移动到循环之上，因为它在那个示例中显示的循环块内没有被修改。

循环删除优化步骤会寻找那些具有非无限循环次数且不影响函数返回值的循环。

在测试代码中，我们看到了具有循环部分的两个基本块`bb:`和`bb2:`都被删除了。我们还看到了`foo`函数直接跳转到返回语句。

还有许多其他优化循环的技术，例如`loop-rotate`、`loop-unswitch`和`loop-unroll`，你可以亲自尝试。然后你会看到它们如何影响代码。

# 重新关联表达式

在本食谱中，你将了解重新关联表达式及其在优化中的帮助。

## 准备就绪

该 opt 工具需要安装才能使本食谱生效。

## 如何做到这一点…

1.  为简单的重新关联转换编写测试用例：

    ```cpp
    $ cat testreassociate.ll
    define i32 @test(i32 %b, i32 %a) {
     %tmp.1 = add i32 %a, 1234
     %tmp.2 = add i32 %b, %tmp.1
     %tmp.4 = xor i32 %a, -1
     ; (b+(a+1234))+~a -> b+1233
     %tmp.5 = add i32 %tmp.2, %tmp.4
     ret i32 %tmp.5
    }

    ```

1.  在这个测试用例上运行重新关联传递，以查看代码是如何修改的：

    ```cpp
    $ opt testreassociate.ll  –reassociate –die –S
    define i32 @test(i32 %b, i32 %a) {
    %tmp.5 = add i32 %b, 1233
    ret i32 %tmp.5
    }

    ```

## 它是如何工作的 …

通过重新关联，我们是指应用代数性质，如结合性、交换性和分配性，来重新排列一个表达式，以便启用其他优化，例如常量折叠、LICM 等。

在前面的例子中，我们使用了逆性质来消除像 `"X` `+` `~X"` `->` `"-1"` 这样的模式，通过重新关联来实现。

测试用例的前三行给出了形式为 `(b+(a+1234))+~a` 的表达式。在这个表达式中，使用重新关联传递，我们将 `a+~a` 转换为 `-1`。因此，在结果中，我们得到最终的返回值是 `b+1234-1` `=` `b+1233`。

处理这种转换的代码位于 `lib/Transforms/Scalar` 下的 `Reassociate.cpp` 文件中。

如果你查看此文件，特别是代码段，你可以看到它检查操作数列表中是否有 `a` 和 `~a`：

```cpp
if (!BinaryOperator::isNeg(TheOp) && !BinaryOperator::isNot(TheOp))
      continue;

    Value *X = nullptr;
    …
    …
    else if (BinaryOperator::isNot(TheOp))
      X = BinaryOperator::getNotArgument(TheOp);

unsigned FoundX = FindInOperandList(Ops, i, X);
```

以下代码负责在表达式中遇到此类值时处理和插入 `-1` 值：

```cpp
if (BinaryOperator::isNot(TheOp)) {
      Value *V = Constant::getAllOnesValue(X->getType());
      Ops.insert(Ops.end(), ValueEntry(getRank(V), V));
      e += 1;
    }
```

# 向量化 IR

**向量化**是编译器的重要优化，我们可以向量化代码以一次执行多个数据集上的指令。如果后端架构支持向量寄存器，可以加载大量数据到这些向量寄存器中，并在寄存器上执行特殊的向量指令。

在 LLVM 中有两种类型的向量化——**超词并行性**（**SLP**）和**循环向量化**。循环向量化处理循环中的向量化机会，而 SLP 向量化处理基本块中的直线代码的向量化。在本食谱中，我们将看到直线代码是如何向量化。

## 准备就绪

SLP 向量化构建 IR 表达式的自下而上的树，并广泛比较树的节点，以查看它们是否相似，从而可以组合成向量。需要修改的文件是 `lib/Transform/Vectorize/SLPVectorizer.cpp`。

我们将尝试向量化一段直线代码，例如 `return` `a[0]` `+` `a[1]` `+` `a[2]` `+` `a[3]`。

前一类代码的表达式树将是一个有点单边的树。我们将运行 DFS 来存储操作数和运算符。

前一类表达式的 IR 将看起来像这样：

```cpp
define i32 @hadd(i32* %a) {
entry:
    %0 = load i32* %a, align 4
    %arrayidx1 = getelementptr inbounds i32* %a, i32 1
    %1 = load i32* %arrayidx1, align 4
    %add = add nsw i32 %0, %1
    %arrayidx2 = getelementptr inbounds i32* %a, i32 2
    %2 = load i32* %arrayidx2, align 4
    %add3 = add nsw i32 %add, %2
    %arrayidx4 = getelementptr inbounds i32* %a, i32 3
    %3 = load i32* %arrayidx4, align 4
    %add5 = add nsw i32 %add3, %3
    ret i32 %add5
}
```

向量化模型遵循三个步骤：

1.  检查是否可以向量化。

1.  计算向量化代码相对于标量代码的盈利能力。

1.  如果满足这两个条件，则向量化代码：

## 如何做到它...

1.  打开 `SLPVectorizer.cpp` 文件。需要实现一个新函数，用于对 *准备就绪* 部分中显示的 IR 表达式的表达式树进行 DFS 遍历：

    ```cpp
    bool matchFlatReduction(PHINode *Phi, BinaryOperator *B, const DataLayout *DL) {

      if (!B)
        return false;

      if (B->getType()->isVectorTy() ||
        !B->getType()->isIntegerTy())
        return false;

    ReductionOpcode = B->getOpcode();
    ReducedValueOpcode = 0;
    ReduxWidth = MinVecRegSize / DL->getTypeAllocSizeInBits(B->getType());
    ReductionRoot = B;
    ReductionPHI = Phi;

    if (ReduxWidth < 4)
      return false;
    if (ReductionOpcode != Instruction::Add)
      return false;

    SmallVector<BinaryOperator *, 32> Stack;
    ReductionOps.push_back(B);
    ReductionOpcode = B->getOpcode();
    Stack.push_back(B);

    // Traversal of the tree.
    while (!Stack.empty()) {
      BinaryOperator *Bin = Stack.back();
      if (Bin->getParent() != B->getParent())
        return false;
      Value *Op0 = Bin->getOperand(0);
      Value *Op1 = Bin->getOperand(1);
      if (!Op0->hasOneUse() || !Op1->hasOneUse())
        return false;
      BinaryOperator *Op0Bin = dyn_cast<BinaryOperator>(Op0); BinaryOperator *Op1Bin = dyn_cast<BinaryOperator>(Op1); Stack.pop_back();

      // Do not handle case where both the operands are binary
    //operators
      if (Op0Bin && Op1Bin)
        return false;
      // Both the operands are not binary operator.
      if (!Op0Bin && !Op1Bin) {
        ReducedVals.push_back(Op1);
        ReducedVals.push_back(Op0);

        ReductionOps.push_back(Bin);
        continue;
    }

    // One of the Operand is binary operand, push that into stack
    // for further processing. Push the other non-binary operand //into ReducedVals.
      if (Op0Bin) {
        if (Op0Bin->getOpcode() != ReductionOpcode)
          return false;
        Stack.push_back(Op0Bin);
        ReducedVals.push_back(Op1);

        ReductionOps.push_back(Op0Bin);
      }

      if (Op1Bin) {

        if (Op1Bin->getOpcode() != ReductionOpcode)
          return false;
        Stack.push_back(Op1Bin);
        ReducedVals.push_back(Op0);
        ReductionOps.push_back(Op1Bin);
      }
    }
    SmallVector<Value *, 16> Temp;
    // Reverse the loads from a[3], a[2], a[1], a[0]

    // to a[0], a[1], a[2], a[3] for checking incremental
    // consecutiveness further ahead.
    while (!ReducedVals.empty())
      Temp.push_back(ReducedVals.pop_back_val());
    ReducedVals.clear();
    for (unsigned i = 0, e = Temp.size(); i < e; ++i)
      ReducedVals.push_back(Temp[i]);
      return true;
    }
    ```

1.  计算结果向量化 IR 的成本，并得出是否进行向量化是有利可图的结论。在 `SLPVectorizer.cpp` 文件中，向 `getReductionCost()` 函数中添加以下行：

    ```cpp
    int HAddCost = INT_MAX;
    // If horizontal addition pattern is identified, calculate cost.

    // Such horizontal additions can be modeled into combination of

    // shuffle sub-vectors and vector adds and one single extract element

    // from last resultant vector.

    // e.g. a[0]+a[1]+a[2]+a[3] can be modeled as // %1 = load <4 x> %0
    // %2 = shuffle %1 <2, 3, undef, undef>
    // %3 = add <4 x> %1, %2
    // %4 = shuffle %3 <1, undef, undef, undef>

    // %5 = add <4 x> %3, %4

    // %6 = extractelement %5 <0>
    if (IsHAdd) {
      unsigned VecElem = VecTy->getVectorNumElements();
      unsigned NumRedxLevel = Log2_32(VecElem);
      HAddCost = NumRedxLevel *
       (TTI->getArithmeticInstrCost(ReductionOpcode, VecTy) + TTI->getShuffleCost(TargetTransformInfo::SK_ExtractSubvector, VecTy, VecElem / 2, VecTy)) + TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, 0);
      }
    ```

1.  在同一函数中，在计算 `PairwiseRdxCost` 和 `SplittingRdxCost` 之后，将它们与 `HAddCost` 进行比较：

    ```cpp
    VecReduxCost = HAddCost < VecReduxCost ? HAddCost : VecReduxCost;
    ```

1.  在 `vectorizeChainsInBlock()` 函数中，调用您刚刚定义的 `matchFlatReduction()` 函数：

    ```cpp
    // Try to vectorize horizontal reductions feeding into a return.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(it))

    if (RI->getNumOperands() != 0)
    if (BinaryOperator *BinOp =
       dyn_cast<BinaryOperator>(RI->getOperand(0))) {

      DEBUG(dbgs() << "SLP: Found a return to vectorize.\n");

      HorizontalReduction HorRdx;
      IsReturn = true;

      if ((HorRdx.matchFlatReduction(nullptr, BinOp, DL) && HorRdx.tryToReduce(R, TTI)) || tryToVectorizePair(BinOp->getOperand(0), BinOp->getOperand(1), R)) {
      Changed = true;

      it = BB->begin();
      e = BB->end();
      continue;

    }
    }
    ```

1.  定义两个全局标志以跟踪水平减少，该减少输入到返回中：

    ```cpp
    static bool IsReturn = false;
    static bool IsHAdd = false;
    ```

1.  如果小树能够返回输入，则允许对它们进行向量化。将以下行添加到 `isFullyVectorizableTinyTree()` 函数中：

    ```cpp
    if (VectorizableTree.size() == 1 && IsReturn && IsHAdd)return true;
    ```

## 它是如何工作的……

保存包含上述代码的文件后，编译 LLVM 项目，并在示例 IR 上运行 opt 工具，如下所示：

1.  打开 `example.ll` 文件，并将以下 IR 粘贴到其中：

    ```cpp
    define i32 @hadd(i32* %a) {
    entry:
     %0 = load i32* %a, align 4
     %arrayidx1 = getelementptr inbounds i32* %a, i32 1
     %1 = load i32* %arrayidx1, align 4
     %add = add nsw i32 %0, %1
     %arrayidx2 = getelementptr inbounds i32* %a, i32 2
     %2 = load i32* %arrayidx2, align 4
     %add3 = add nsw i32 %add, %2
     %arrayidx4 = getelementptr inbounds i32* %a, i32 3
     %3 = load i32* %arrayidx4, align 4
     %add5 = add nsw i32 %add3, %3
     ret i32 %add5
    }

    ```

1.  在 `example.ll` 上运行 opt 工具：

    ```cpp
    $ opt -basicaa -slp-vectorizer -mtriple=aarch64-unknown-linux-gnu -mcpu=cortex-a57

    ```

    输出将是向量化的代码，如下所示：

    ```cpp
    define i32 @hadd(i32* %a) {

    entry:

    %0 = bitcast i32* %a to <4 x i32>*
    %1 = load <4 x i32>* %0, align 4 %rdx.shuf = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>

    %bin.rdx = add <4 x i32> %1,

    %rdx.shuf %rdx.shuf1 = shufflevector <4 x i32>

    %bin.rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef> %bin.rdx2 = add <4 x i32> %bin.rdx, %rdx.shuf1

    %2 = extractelement <4 x i32> %bin.rdx2, i32 0

    ret i32 %2

    }

    ```

观察到代码被向量化了。`matchFlatReduction()` 函数对表达式进行深度优先遍历，并将所有加载存储在 `ReducedVals` 中，而加法存储在 `ReductionOps` 中。之后，在 `HAddCost` 中计算水平向量化的成本，并与标量成本进行比较。结果证明这是有利的。因此，它将表达式向量化。这由 `tryToReduce()` 函数处理，该函数已经实现。

## 参考以下内容……

+   对于详细的向量化概念，请参阅 Ira Rosen、Dorit Nuzman 和 Ayal Zaks 撰写的论文 *GCC 中的循环感知 SLP*。

# 其他优化过程

在这个菜谱中，我们将查看一些更多的转换过程，这些过程更像是工具过程。我们将查看 `strip-debug-symbols` 过程和 `prune-eh` 过程。

## 准备工作……

必须安装 opt 工具。

## 如何操作……

1.  编写一个测试用例来检查 strip-debug 过程，该过程从测试代码中删除调试符号：

    ```cpp
    $ cat teststripdebug.ll
    @x = common global i32 0                          ; <i32*> [#uses=0]

    define void @foo() nounwind readnone optsize ssp {
    entry:
     tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !5, metadata !{}), !dbg !10
     ret void, !dbg !11
    }

    declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

    !llvm.dbg.cu = !{!2}
    !llvm.module.flags = !{!13}
    !llvm.dbg.sp = !{!0}
    !llvm.dbg.lv.foo = !{!5}
    !llvm.dbg.gv = !{!8}

    !0 = !MDSubprogram(name: "foo", linkageName: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !12, scope: !1, type: !3, function: void ()* @foo)
    !1 = !MDFile(filename: "b.c", directory: "/tmp")
    !2 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: 0, file: !12, enums: !4, retainedTypes: !4)
    !3 = !MDSubroutineType(types: !4)
    !4 = !{null}
    !5 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "y", line: 3, scope: !6, file: !1, type: !7)
    !6 = distinct !MDLexicalBlock(line: 2, column: 0, file: !12, scope: !0)
    !7 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
    !8 = !MDGlobalVariable(name: "x", line: 1, isLocal: false, isDefinition: true, scope: !1, file: !1, type: !7, variable: i32* @x)
    !9 = !{i32 0}
    !10 = !MDLocation(line: 3, scope: !6)
    !11 = !MDLocation(line: 4, scope: !6)
    !12 = !MDFile(filename: "b.c", directory: "/tmp")
    !13 = !{i32 1, !"Debug Info Version", i32 3}

    ```

1.  通过将 `–strip-debug` 命令行选项传递给 `opt` 工具来运行 `strip-debug-symbols` 过程：

    ```cpp
    $ opt -strip-debug teststripdebug.ll  -S
    ; ModuleID = ' teststripdebug.ll'

    @x = common global i32 0

    ; Function Attrs: nounwind optsize readnone ssp
    define void @foo() #0 {
    entry:
     ret void
    }

    attributes #0 = { nounwind optsize readnone ssp }

    !llvm.module.flags = !{!0}

    !0 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}

    ```

1.  编写一个测试用例来检查 `prune-eh` 过程：

    ```cpp
    $ cat simpletest.ll
    declare void @nounwind() nounwind

    define internal void @foo() {
     call void @nounwind()
     ret void
    }

    define i32 @caller() {
     invoke void @foo( )
     to label %Normal unwind label %Except

    Normal:        ; preds = %0
     ret i32 0

    Except:        ; preds = %0
     landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
     catch i8* null
     ret i32 1
    }
    declare i32 @__gxx_personality_v0(...)

    ```

1.  通过将 `–prune-eh` 命令行选项传递给 opt 工具来运行该过程，以删除未使用的异常信息：

    ```cpp
    $ opt -prune-eh -S simpletest.ll
    ; ModuleID = 'simpletest.ll'

    ; Function Attrs: nounwind
    declare void @nounwind() #0

    ; Function Attrs: nounwind
    define internal void @foo() #0 {
     call void @nounwind()
     ret void
    }
    ; Function Attrs: nounwind
    define i32 @caller() #0 {
     call void @foo()
     br label %Normal

    Normal:                                           ; preds = %0
     ret i32 0
    }

    declare i32 @__gxx_personality_v0(...)

    attributes #0 = { nounwind }

    ```

## 它是如何工作的……

在第一种情况下，当我们运行 `strip-debug` 过程时，它会从代码中删除调试信息，我们可以得到紧凑的代码。此过程仅在寻找紧凑代码时必须使用，因为它可以删除虚拟寄存器的名称以及内部全局变量和函数的符号，从而使源代码更难以阅读，并使代码逆向工程变得困难。

处理此转换的代码部分位于 `llvm/lib/Transforms/IPO/StripSymbols.cpp` 文件中，其中 `StripDeadDebugInfo::runOnModule` 函数负责删除调试信息。

第二个测试是使用`prune-eh`过程来移除未使用的异常信息，该过程实现了一个跨过程过程。它会遍历调用图，只有当被调用者不能抛出异常时，才将调用指令转换为调用指令，并且如果函数不能抛出异常，则将其标记为`nounwind`。

## 参见

+   参考以下链接了解其他转换过程：[`llvm.org/docs/Passes.html#transform-passes`](http://llvm.org/docs/Passes.html#transform-passes)
