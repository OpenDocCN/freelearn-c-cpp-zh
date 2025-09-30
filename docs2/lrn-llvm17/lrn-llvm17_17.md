

# 第十三章：超出指令选择

现在我们已经在前几章学习了使用 SelectionDAG 和 GlobalISel 基于 LLVM 的框架进行指令选择，我们可以探索指令选择之外的其它有趣概念。本章包含了一些对于高度优化编译器来说可能很有趣的后端之外的高级主题。例如，一些遍历操作会超出指令选择，并且可以对各种指令执行不同的优化，这意味着开发者有足够的自由在这个编译器的这个阶段引入他们自己的遍历操作来执行有意义的特定目标任务。

最终，在本章中，我们将深入研究以下概念：

+   将新的机器函数遍历操作添加到 LLVM 中

+   将新的目标集成到 clang 前端

+   如何针对不同的 CPU 架构

# 将新的机器函数遍历操作添加到 LLVM 中

在本节中，我们将探讨如何在 LLVM 中实现一个新的机器函数遍历操作，该操作在指令选择之后运行。具体来说，将创建一个 `MachineFunctionPass` 类，它是 LLVM 中原始 `FunctionPass` 类的一个子集，可以通过 `opt` 运行。这个类通过 `llc` 适配原始基础设施，允许实现操作在 `MachineFunction` 表示形式上运行的遍历操作。

需要注意的是，后端中遍历的实现使用了旧遍历管理器的接口，而不是新遍历管理器。这是因为 LLVM 目前在后端没有完整的可工作的新遍历管理器实现。因此，本章将遵循在旧遍历管理器管道中添加新遍历的方法。

在实际实现方面，例如函数遍历操作，机器函数遍历操作一次优化一个（机器）函数，但不是覆盖 `runOnFunction()` 方法，而是覆盖 `runOnMachineFunction()` 方法。本节将要实现的机器函数遍历操作是一个检查除零发生的遍历操作，具体来说，是在后端中插入陷阱代码。这种类型的遍历操作对于 M88k 目标很重要，因为 MC88100 硬件在检测除零情况上存在限制。

从上一章的后端实现继续，让我们看看后端机器函数遍历操作是如何实现的！

## 实现 M88k 目标的顶层接口

首先，在 `llvm/lib/Target/M88k/M88k.h` 中，让我们在 `llvm` 命名空间声明内添加两个原型，这些原型将在以后使用：

1.  将要实现的机器函数遍历操作将被命名为 `M88kDivInstrPass`。我们将添加一个函数声明来初始化这个遍历操作，并接收遍历注册表，这是一个管理所有遍历注册和初始化的类：

    ```cpp

    void initializeM88kDivInstrPass(PassRegistry &);
    ```

1.  接下来，声明实际创建 `M88kDivInstr` 遍历的函数，其参数为 M88k 目标机信息：

    ```cpp

    FunctionPass *createM88kDivInstr(const M88kTargetMachine &);
    ```

## 添加机器函数遍历的 TargetMachine 实现

接下来，我们将分析在 `llvm/lib/Target/M88k/M88kTargetMachine.cpp` 中需要进行的某些更改：

1.  在 LLVM 中，通常会给用户提供切换遍历开/关的选项。因此，让我们为我们的机器函数遍历提供相同的灵活性。我们首先声明一个名为 `m88k-no-check-zero-division` 的命令行选项，并将其初始化为 `false`，这意味着除非用户明确将其关闭，否则总会进行零除检查。我们将在 `llvm` 命名空间声明下添加此选项，并且它是 `llc` 的一个选项：

    ```cpp

    using namespace llvm;
    static cl::opt<bool>
        NoZeroDivCheck("m88k-no-check-zero-division", cl::Hidden,
                       cl::desc("M88k: Don't trap on integer division by zero."),
                       cl::init(false));
    ```

1.  还有一个惯例是创建一个正式的方法来返回命令行值，这样我们就可以查询它以确定是否运行遍历。我们的原始命令行选项将被 `noZeroDivCheck()` 方法包装起来，这样我们就可以在以后利用命令行结果：

    ```cpp

    M88kTargetMachine::~M88kTargetMachine() {}
    bool M88kTargetMachine::noZeroDivCheck() const { return NoZeroDivCheck; }
    ```

1.  接下来，在 `LLVMInitializeM88kTarget()` 中，我们将注册和初始化 M88k 目标和遍历的地方，插入对之前在 `llvm/lib/Target/M88k/M88k.h` 中声明的 `initializeM88kDivInstrPass()` 方法的调用：

    ```cpp

    extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeM88kTarget() {
       RegisterTargetMachine<M88kTargetMachine> X(getTheM88kTarget());
       auto &PR = *PassRegistry::getPassRegistry();
       initializeM88kDAGToDAGISelPass(PR);
       initializeM88kDivInstrPass(PR);
     }
    ```

1.  M88k 目标还需要重写 `addMachineSSAOptimization()` 方法，这是一个在指令处于 SSA 形式时添加优化指令的遍历的方法。本质上，我们的机器函数遍历被添加为一种机器 SSA 优化。该方法被声明为一个需要重写的方法。我们将在 `M88kTargetMachine.cpp` 的末尾添加完整的实现：

    ```cpp

       bool addInstSelector() override;
       void addPreEmitPass() override;
       void addMachineSSAOptimization() override;
    . . .
    void M88kPassConfig::addMachineSSAOptimization() {
       addPass(createM88kDivInstr(getTM<M88kTargetMachine>()));
       TargetPassConfig::addMachineSSAOptimization();
    }
    ```

1.  我们返回用于切换机器函数遍历开/关的命令行选项的方法（`noZeroDivCheck()` 方法）也声明在 `M88kTargetMachine.h` 中：

    ```cpp

       ~M88kTargetMachine() override;
       bool noZeroDivCheck() const;
    ```

## 开发机器函数遍历的细节

现在，M88k 目标机的实现已经完成，下一步将是开发机器函数遍历本身。实现包含在新文件 `llvm/lib/Target/M88k/M88kDivInstr.cpp` 中：

1.  为我们的机器函数遍历添加必要的头文件。这包括提供访问 M88k 目标信息的头文件，以及允许我们对机器函数和机器指令进行操作的头文件：

    ```cpp

    #include "M88k.h"
    #include "M88kInstrInfo.h"
    #include "M88kTargetMachine.h"
    #include "MCTargetDesc/M88kMCTargetDesc.h"
    #include "llvm/ADT/Statistic.h"
    #include "llvm/CodeGen/MachineFunction.h"
    #include "llvm/CodeGen/MachineFunctionPass.h"
    #include "llvm/CodeGen/MachineInstrBuilder.h"
    #include "llvm/CodeGen/MachineRegisterInfo.h"
    #include "llvm/IR/Instructions.h"
    #include "llvm/Support/Debug.h"
    ```

1.  之后，我们将添加一些代码来为我们的机器函数遍历做准备。首先是 `DEBUG_TYPE` 定义，命名为 `m88k-div-instr`，用于调试时的细粒度控制。具体来说，定义这个 `DEBUG_TYPE` 允许用户指定机器函数遍历的名称，并在启用调试信息时查看与遍历相关的任何调试信息：

    ```cpp

    #define DEBUG_TYPE "m88k-div-instr"
    ```

1.  我们还指定了正在使用`llvm`命名空间，并为我们的机器函数声明了一个`STATISTIC`值。这个统计值称为`InsertedChecks`，它跟踪编译器插入的除以零检查的数量。最后，声明了一个匿名命名空间来封装随后的机器函数传递实现：

    ```cpp

    using namespace llvm;
    STATISTIC(InsertedChecks, "Number of inserted checks for division by zero");
    namespace {
    ```

1.  如前所述，这个机器函数传递旨在检查除以零的情况，并插入会导致 CPU 陷阱的指令。这些指令需要条件码，因此我们定义了一个名为`CC0`的`enum`值，其中包含了适用于 M88k 目标的条件码及其编码：

    ```cpp

    enum class CC0 : unsigned {
      EQ0 = 0x2,
      NE0 = 0xd,
      GT0 = 0x1,
      LT0 = 0xc,
      GE0 = 0x3,
      LE0 = 0xe
    };
    ```

1.  让我们创建我们的机器函数传递的实际类，称为`M88kDivInstr`。首先，我们创建它作为一个继承并属于`MachineFunctionPass`类型的实例。接下来，我们声明了`M88kDivInstr`传递所需的各个必要实例。这包括我们将在稍后创建和详细说明的`M88kBuilder`，以及包含目标指令和寄存器信息的`M88kTargetMachine`。此外，我们在发出指令时还需要寄存器银行信息和机器寄存器信息。还添加了一个`AddZeroDivCheck`布尔值来表示之前的命令行选项，它打开或关闭我们的传递：

    ```cpp

    class M88kDivInstr : public MachineFunctionPass {
      friend class M88kBuilder;
      const M88kTargetMachine *TM;
      const TargetInstrInfo *TII;
      const TargetRegisterInfo *TRI;
      const RegisterBankInfo *RBI;
      MachineRegisterInfo *MRI;
      bool AddZeroDivCheck;
    ```

1.  对于`M88kDivInstr`类的公共变量和方法，我们声明了一个识别号，LLVM 将使用它来识别我们的传递，以及`M88kDivInstr`构造函数，它接受`M88kTargetMachine`。接下来，我们重写了`getRequiredProperties()`方法，它代表了`MachineFunction`在优化过程中可能拥有的属性，我们还重写了`runOnMachineFunction()`方法，这将是我们的传递在检查任何除以零时运行的主要方法之一。公开声明的第二个重要函数是`runOnMachineBasicBlock()`函数，它将在`runOnMachineFunction()`内部执行：

    ```cpp

    public:
      static char ID;
      M88kDivInstr(const M88kTargetMachine *TM = nullptr);
      MachineFunctionProperties getRequiredProperties() const override;
      bool runOnMachineFunction(MachineFunction &MF) override;
      bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    ```

1.  最后，最后一部分是声明私有方法和关闭类。在`M88kDivInstr`类中，我们声明的唯一私有方法是`addZeroDivCheck()`方法，它会在任何除法指令之后插入除以零的检查。正如我们稍后将会看到的，`MachineInstr`需要在 M88k 目标上指向特定的除法指令：

    ```cpp

    private:
      void addZeroDivCheck(MachineBasicBlock &MBB, MachineInstr *DivInst);
    };
    ```

1.  接下来创建了一个`M88kBuilder`类，这是一个专门化的构建实例，用于创建 M88k 特定的指令。这个类保持了一个`MachineBasicBlock`实例（以及相应的迭代器）和`DebugLoc`，以跟踪这个构建类的调试位置。其他必要的实例包括目标指令信息、目标寄存器信息和 M88k 目标寄存器银行信息：

    ```cpp

    class M88kBuilder {
      MachineBasicBlock *MBB;
      MachineBasicBlock::iterator I;
      const DebugLoc &DL;
      const TargetInstrInfo &TII;
      const TargetRegisterInfo &TRI;
      const RegisterBankInfo &RBI;
    ```

1.  对于 `M88kBuilder` 类的公共方法，我们必须实现这个构建器的构造函数。在初始化时，我们的专用构建器需要一个 `M88kDivInstr` 传递的实例来初始化目标指令、寄存器信息以及寄存器银行信息，以及 `MachineBasicBlock` 和一个调试位置：

    ```cpp

    public:
      M88kBuilder(M88kDivInstr &Pass, MachineBasicBlock *MBB, const DebugLoc &DL)
          : MBB(MBB), I(MBB->end()), DL(DL), TII(*Pass.TII), TRI(*Pass.TRI),
            RBI(*Pass.RBI) {}
    ```

1.  接下来，创建了一个在 M88k 构建器内部设置 `MachineBasicBlock` 的方法，并且相应地设置了 `MachineBasicBlock` 迭代器：

    ```cpp

      void setMBB(MachineBasicBlock *NewMBB) {
        MBB = NewMBB;
        I = MBB->end();
      }
    ```

1.  接下来需要实现 `constrainInst()` 函数，它是在处理 `MachineInstr` 实例时需要的。对于一个给定的 `MachineInstr`，我们检查 `MachineInstr` 实例的操作数的寄存器类是否可以通过现有的函数 `constrainSelectedInstRegOperands()` 进行约束。如图所示，这个机器函数传递要求机器指令的寄存器操作数可以约束：

    ```cpp

      void constrainInst(MachineInstr *MI) {
        if (!constrainSelectedInstRegOperands(*MI, TII, TRI, RBI))
          llvm_unreachable("Could not constrain register operands");
      }
    ```

1.  这个传递插入的指令之一是一个 `BCND` 指令，它在 `M88kInstrInfo.td` 中定义，是 M88k 目标上的条件分支。为了创建这个指令，我们需要一个条件码，即 `CC0` 枚举，这些枚举在 `M88kDivInstr.cpp` 的开头实现——即一个寄存器和 `MachineBasicBlock`。创建 `BCND` 指令后，简单地返回，并在检查新创建的指令是否可以约束之后返回。此外，这完成了 `M88kBuilder` 类的类实现并完成了之前声明的匿名命名空间：

    ```cpp

      MachineInstr *bcnd(CC0 Cc, Register Reg, MachineBasicBlock *TargetMBB) {
        MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::BCND))
                               .addImm(static_cast<int64_t>(Cc))
                               .addReg(Reg)
                               .addMBB(TargetMBB);
        constrainInst(MI);
        return MI;
      }
    ```

1.  对于机器函数传递，我们还需要一个陷阱指令，这是一个 `TRAP503` 指令。这个指令需要一个寄存器，如果寄存器的 0 位没有被设置，则会引发一个带有向量 503 的陷阱，这将在零除之后发生。在创建 `TRAP503` 指令后，在返回之前检查 `TRAP503` 的约束。此外，这也完成了 `M88kBuilder` 类的实现和之前声明的匿名命名空间：

    ```cpp

      MachineInstr *trap503(Register Reg) {
        MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::TRAP503)).addReg(Reg);
        constrainInst(MI);
        return MI;
      }
    };
    } // end anonymous namespace
    ```

1.  现在我们可以开始实现机器函数传递中执行实际检查的函数了。首先，让我们探索一下 `addZeroDivCheck()` 函数是如何实现的。这个函数简单地在一个当前机器指令（预期指向 `DIVSrr` 或 `DIVUrr`）之间插入一个除以零的检查；这些是分别表示有符号和无符号除法的助记符。插入 `BCND` 和 `TRAP503` 指令，并将 `InsertedChecks` 统计量增加以指示两个指令的添加：

    ```cpp

    void M88kDivInstr::addZeroDivCheck(MachineBasicBlock &MBB,
                                       MachineInstr *DivInst) {
     assert(DivInst->getOpcode() == M88k::DIVSrr ||
             DivInst->getOpcode() == M88k::DIVUrr && "Unexpected          opcode");
      MachineBasicBlock *TailBB = MBB.splitAt(*DivInst);
      M88kBuilder B(*this, &MBB, DivInst->getDebugLoc());
      B.bcnd(CC0::NE0, DivInst->getOperand(2).getReg(), TailBB);
      B.trap503(DivInst->getOperand(2).getReg());
      ++InsertedChecks;
    }
    ```

1.  `runOnMachineFunction()` 函数接下来被实现，并且是创建 LLVM 中的一种函数传递类型时需要重写的重要函数之一。这个函数返回 true 或 false，取决于在机器函数传递期间是否进行了任何更改。此外，对于给定的机器函数，我们收集所有相关的 M88k 子目标信息，包括目标指令、目标寄存器、寄存器银行和机器寄存器信息。是否启用或禁用 `M88kDivInstr` 机器函数传递的详细信息也被查询并存储在 `AddZeroDivCheck` 变量中。此外，对机器函数中的所有机器基本块进行分析，以检查除以零的情况。执行机器基本块分析的功能是 `runOnMachineBasicBlock()`；我们将在接下来实现这个功能。最后，如果机器函数已更改，这通过返回的 `Changed` 变量来指示：

    ```cpp

    bool M88kDivInstr::runOnMachineFunction(MachineFunction &MF) {
      const M88kSubtarget &Subtarget =   MF.getSubtarget<M88kSubtarget>();
      TII = Subtarget.getInstrInfo();
      TRI = Subtarget.getRegisterInfo();
      RBI = Subtarget.getRegBankInfo();
      MRI = &MF.getRegInfo();
      AddZeroDivCheck = !TM->noZeroDivCheck();
      bool Changed = false;
      for (MachineBasicBlock &MBB : reverse(MF))
        Changed |= runOnMachineBasicBlock(MBB);
      return Changed;
    }
    ```

1.  对于 `runOnMachineBasicBlock()` 函数，也返回一个 `Changed` 布尔标志，以指示机器基本块是否已更改；然而，它最初被设置为 `false`。此外，在机器基本块内，我们需要分析所有机器指令并检查指令是否是 `DIVUrr` 或 `DIVSrr` 操作码。除了检查操作码是否是除法指令外，我们还需要检查用户是否已启用或禁用我们的机器函数传递。如果所有这些条件都满足，将通过之前实现的 `addZeroDivCheck()` 函数相应地添加带有条件分支和陷阱指令的除以零检查。

    ```cpp

    bool M88kDivInstr::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
      bool Changed = false;
      for (MachineBasicBlock::reverse_instr_iterator I =   MBB.instr_rbegin();
           I != MBB.instr_rend(); ++I) {
        unsigned Opc = I->getOpcode();
        if ((Opc == M88k::DIVUrr || Opc == M88k::DIVSrr) &&     AddZeroDivCheck) {
          addZeroDivCheck(MBB, &*I);
          Changed = true;
        }
      }
      return Changed;
    }
    ```

1.  之后，我们需要实现构造函数以初始化我们的函数传递并设置适当的机器函数属性。这可以通过在 `M88kDivInstr` 类的构造函数中调用 `initializeM88kDivInstrPass()` 函数并设置机器函数属性以指示我们的传递需要机器函数处于 SSA 形式来实现：

    ```cpp

    M88kDivInstr::M88kDivInstr(const M88kTargetMachine *TM)
        : MachineFunctionPass(ID), TM(TM) {
      initializeM88kDivInstrPass(*PassRegistry::getPassRegistry());
    }
    MachineFunctionProperties M88kDivInstr::getRequiredProperties() const {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::IsSSA);
    }
    ```

1.  下一步是初始化我们的机器函数传递的 ID，并使用我们的机器函数传递的详细信息实例化 `INITIALIZE_PASS` 宏。这需要传递实例、命名信息以及两个布尔参数，指示传递是否仅检查 CFG 以及传递是否是分析传递。由于 `M88kDivInstr` 不执行这些操作，因此将两个 `false` 参数指定给传递初始化宏：

    ```cpp

    char M88kDivInstr::ID = 0;
    INITIALIZE_PASS(M88kDivInstr, DEBUG_TYPE, "Handle div instructions", false, false)
    ```

1.  最后，`createM88kDivInstr()` 函数创建 `M88kDivInstr` 传递的新实例，并带有 `M88kTargetMachine` 实例。这被封装在 `llvm` 命名空间中，并在完成此函数后结束命名空间：

    ```cpp

    namespace llvm {
    FunctionPass *createM88kDivInstr(const M88kTargetMachine &TM) {
      return new M88kDivInstr(&TM);
    }
    } // end namespace llvm
    ```

## 构建新实现的机器函数传递

我们几乎完成了我们新的机器函数传递的实现！现在，我们需要确保 CMake 意识到 `M88kDivinstr.cpp` 中的新机器函数传递。然后，此文件被添加到 `llvm/lib/Target/M88k/CMakeLists.txt`：

```cpp

add_llvm_target(M88kCodeGen
   M88kAsmPrinter.cpp
   M88kDivInstr.cpp
   M88kFrameLowering.cpp
   M88kInstrInfo.cpp
   M88kISelDAGToDAG.cpp
```

最后一步是使用以下命令构建带有我们新的机器函数传递实现的 LLVM。我们需要 `-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=M88k` CMake 选项来构建 M88k 目标：

```cpp

$ cmake -G Ninja ../llvm-project/llvm -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=M88k -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="llvm"
$ ninja
```

通过这样，我们已经实现了机器函数传递，但不是很有趣吗？我们可以通过通过 `llc` 传递 LLVM IR 来演示此传递的结果。

## 使用 llc 运行机器函数传递的快照

我们有以下 IR，其中包含除以零的操作：

```cpp

$ cat m88k-divzero.ll
target datalayout = "E-m:e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-a:8:16-n32"
target triple = "m88k-unknown-openbsd"
@dividend = dso_local global i32 5, align 4
define dso_local i32 @testDivZero() #0 {
  %1 = load i32, ptr @dividend, align 4
  %2 = sdiv i32 %1, 0
  ret i32 %2
}
```

让我们将其输入到 llc 中：

```cpp

$ llc m88k-divzero.ll
```

通过这样做，我们会看到，在生成的汇编中，默认情况下，除以零检查（由 `bcnd.n` (`BCND`) 和 `tb0` (`TRAP503`) 表示）是由我们新的机器函数传递插入的：

```cpp

| %bb.1:
     subu %r2, %r0, %r2
     bcnd.n ne0, %r0, .LBB0_2
     divu %r2, %r2, 0
     tb0 0, %r3, 503
. . .
.LBB0_3:
     bcnd.n ne0, %r0, .LBB0_4
     divu %r2, %r2, 0
     tb0 0, %r3, 503
```

然而，让我们看看当我们指定 `--m88k-no-check-zero-division` 给 `llc` 时会发生什么：

```cpp

$ llc m88k-divzero.ll –m88k-no-check-zero-division
```

此选项通知后端 `llc` 不要运行检查除以零的传递。生成的汇编将不包含任何 `BCND` 或 `TRAP503` 指令。以下是一个示例：

```cpp

| %bb.1:
     subu %r2, %r0, %r2
     divu %r2, %r2, 0
     jmp.n %r1
     subu %r2, %r0, %r2
```

如我们所见，实现机器函数传递需要几个步骤，但这些程序可以作为你实现任何适合你需求的机器函数传递的指南。由于我们已经在本节中广泛探讨了后端，让我们转换方向，看看我们如何让前端了解 M88k 目标。

# 将新目标集成到 clang 前端

在前面的章节中，我们在 LLVM 中开发了 M88k 目标的后端实现。为了完成 M88k 目标的编译器实现，我们将研究通过添加 clang 的 M88k 目标实现来将我们的新目标连接到前端。

## 在 clang 中实现驱动集成

让我们从将驱动集成添加到 M88k 的 clang 开始：

1.  我们将要做的第一个更改是在 `clang/include/clang/Basic/TargetInfo.h` 文件内部。`BuiltinVaListKind` 枚举列出了每个目标的不同类型的 `__builtin_va_list`，这用于变长函数支持，因此为 M88k 添加了一个相应的类型：

    ```cpp

    enum BuiltinVaListKind {
    . . .
        // typedef struct __va_list_tag {
        //    int __va_arg;
        //    int *__va_stk;
        //    int *__va_reg;
        //} va_list;
        M88kBuiltinVaList
      };
    ```

1.  接下来，我们必须添加一个新的头文件，`clang/lib/Basic/Targets/M88k.h`。此文件是前端 M88k 目标功能支持的头部文件。第一步是定义一个新的宏，以防止多次包含相同的头文件、类型、变量等。我们还需要包含实现所需的各个头文件：

    ```cpp

    #ifndef LLVM_CLANG_LIB_BASIC_TARGETS_M88K_H
    #define LLVM_CLANG_LIB_BASIC_TARGETS_M88K_H
    #include "OSTargets.h"
    #include "clang/Basic/TargetInfo.h"
    #include "clang/Basic/TargetOptions.h"
    #include "llvm/Support/Compiler.h"
    #include "llvm/TargetParser/Triple.h"
    ```

1.  我们将要声明的函数将被添加到 `clang` 和 `targets` 命名空间中，就像 `llvm-project` 内的其他目标一样：

    ```cpp

    namespace clang {
    namespace targets {
    ```

1.  现在让我们声明实际的`M88kTargetInfo`类，并让它扩展原始的`TargetInfo`类。这个类被标记为`LLVM_LIBRARY_VISIBILITY`，因为如果这个类链接到共享库，这个属性允许`M88kTargetInfo`类仅在库内部可见，外部不可访问：

    ```cpp

    class LLVM_LIBRARY_VISIBILITY M88kTargetInfo: public TargetInfo {
    ```

1.  此外，我们必须声明两个变量——一个字符数组来表示寄存器名称，以及一个`enum`值，包含 M88k 目标中可选择的 CPU 类型。我们设置的默认 CPU 是`CK_Unknown` CPU。稍后，我们将看到这可以被用户选项覆盖：

    ```cpp

      static const char *const GCCRegNames[];
      enum CPUKind { CK_Unknown, CK_88000, CK_88100, CK_88110 } CPU = CK_Unknown;
    ```

1.  然后，我们开始声明在我们的类实现中需要的公共方法。除了我们类的构造函数外，我们还定义了各种 getter 方法。这包括获取特定目标`#define`值的函数，获取目标支持的内置函数列表的函数，返回 GCC 寄存器名称及其别名的函数，以及最终返回我们之前添加到`clang/include/clang/Basic/TargetInfo.h`中的 M88k `BuiltinVaListKind`的函数：

    ```cpp

    public:
      M88kTargetInfo(const llvm::Triple &Triple, const TargetOptions &);
      void getTargetDefines(const LangOptions &Opts,
                            MacroBuilder &Builder) const override;
      ArrayRef<Builtin::Info> getTargetBuiltins() const override;
      ArrayRef<const char *> getGCCRegNames() const override;
      ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
      BuiltinVaListKind getBuiltinVaListKind() const override {
        return TargetInfo::M88kBuiltinVaList;
      }
    ```

1.  在 getter 方法之后，我们还必须定义执行对 M88k 目标进行各种检查的方法。第一个方法检查 M88k 目标是否具有特定的目标特性，以字符串的形式提供。其次，我们添加了一个函数来验证内联汇编时使用的约束条件。最后，我们有一个函数检查特定的 CPU 是否适用于 M88k 目标，也以字符串的形式提供：

    ```cpp

      bool hasFeature(StringRef Feature) const override;
      bool validateAsmConstraint(const char *&Name,
                                 TargetInfo::ConstraintInfo &info)                              const override;
      bool isValidCPUName(StringRef Name) const override;
    ```

1.  接下来，让我们声明`M88kTargetInfo`类的 setter 方法。第一个方法简单地设置我们想要针对的特定 M88k CPU，而第二个方法设置一个向量，包含所有有效的支持 M88k 的 CPU：

    ```cpp

      bool setCPU(const std::string &Name) override;
      void fillValidCPUList(SmallVectorImpl<StringRef> &Values)   const override;
    };
    ```

1.  为了完成驱动程序的头部实现，让我们总结一下我们在开始时添加的命名空间和宏定义：

    ```cpp

    } // namespace targets
    } // namespace clang
    #endif // LLVM_CLANG_LIB_BASIC_TARGETS_M88K_H
    ```

1.  现在我们已经完成了`clang/lib/Basic/Targets`中的 M88k 头文件，我们必须在`clang/lib/Basic/Targets/M88k.cpp`中添加相应的`TargetInfo` C++实现。我们将首先包含所需的头文件，特别是我们刚刚创建的新`M88k.h`头文件：

    ```cpp

    #include "M88k.h"
    #include "clang/Basic/Builtins.h"
    #include "clang/Basic/Diagnostic.h"
    #include "clang/Basic/TargetBuiltins.h"
    #include "llvm/ADT/StringExtras.h"
    #include "llvm/ADT/StringRef.h"
    #include "llvm/ADT/StringSwitch.h"
    #include "llvm/TargetParser/TargetParser.h"
    #include <cstring>
    ```

1.  就像我们在标题中之前做的那样，我们从`clang`和`targets`命名空间开始，然后也开始实现`M88kTargetInfo`类的构造函数：

    ```cpp

    namespace clang {
    namespace targets {
    M88kTargetInfo::M88kTargetInfo(const llvm::Triple &Triple,
                                   const TargetOptions &)
        : TargetInfo(Triple) {
    ```

1.  在构造函数中，我们为 M88k 目标设置数据布局字符串。正如你可能之前看到的，这个数据布局字符串出现在生成的 LLVM IR 文件顶部。数据布局字符串每个部分的解释在这里描述：

    ```cpp

      std::string Layout = "";
      Layout += "E"; // M68k is Big Endian
      Layout += "-m:e";
      Layout += "-p:32:32:32"; // Pointers are 32 bit.
      // All scalar types are naturally aligned.
      Layout += "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64";
      // Floats and doubles are also naturally aligned.
      Layout += "-f32:32:32-f64:64:64";
      // We prefer 16 bits of aligned for all globals; see above.
      Layout += "-a:8:16";
      Layout += "-n32"; // Integer registers are 32bits.
      resetDataLayout(Layout);
    ```

1.  `M88kTargetInfo`类的构造函数通过设置各种变量类型为`signed long long`、`unsigned long`或`signed int`来结束：

    ```cpp

      IntMaxType = SignedLongLong;
      Int64Type = SignedLongLong;
      SizeType = UnsignedLong;
      PtrDiffType = SignedInt;
      IntPtrType = SignedInt;
    }
    ```

1.  之后，实现了设置目标 CPU 的函数。这个函数接受一个字符串，并将 CPU 设置为用户在`llvm::StringSwitch`中提供的特定 CPU 字符串，这实际上只是一个常规的 switch 语句，但专门用于 LLVM 中的字符串。我们可以看到，在 M88k 目标上有三种支持的 CPU 类型，还有一个`CK_Unknown`类型，用于如果提供的字符串与任何预期的类型都不匹配：

    ```cpp

    bool M88kTargetInfo::setCPU(const std::string &Name) {
      StringRef N = Name;
      CPU = llvm::StringSwitch<CPUKind>(N)
                .Case("generic", CK_88000)
                .Case("mc88000", CK_88000)
                .Case("mc88100", CK_88100)
                .Case("mc88110", CK_88110)
                .Default(CK_Unknown);
      return CPU != CK_Unknown;
    }
    ```

1.  之前已经提到，在 M88k 目标上支持并有效的 CPU 类型有三种：`mc88000`、`mc88100`和`mc88110`，其中`generic`类型简单地就是`mc88000` CPU。我们必须实现以下函数来在 clang 中强制执行这些有效的 CPU：首先，我们必须声明一个字符串数组`ValidCPUNames[]`，以表示 M88k 上的有效 CPU 名称。其次，`fillValidCPUList()`方法将有效 CPU 名称数组填充到一个向量中。然后，这个向量在`isValidCPUName()`方法中使用，以检查提供的特定 CPU 名称是否确实适用于我们的 M88k 目标：

    ```cpp

    static constexpr llvm::StringLiteral ValidCPUNames[] = {
        {"generic"}, {"mc88000"}, {"mc88100"}, {"mc88110"}};
    void M88kTargetInfo::fillValidCPUList(
        SmallVectorImpl<StringRef> &Values) const {
      Values.append(std::begin(ValidCPUNames),   std::end(ValidCPUNames));
    }
    bool M88kTargetInfo::isValidCPUName(StringRef Name) const {
      return llvm::is_contained(ValidCPUNames, Name);
    }
    ```

1.  接下来，实现`getTargetDefines()`方法。这个函数定义了前端必需的宏，例如有效 CPU 类型。除了`__m88k__`和`__m88k`宏之外，我们还必须为有效 CPU 定义相应的 CPU 宏：

    ```cpp

    void M88kTargetInfo::getTargetDefines(const LangOptions &Opts,
                                          MacroBuilder &Builder) const {
      using llvm::Twine;
      Builder.defineMacro("__m88k__");
      Builder.defineMacro("__m88k");
      switch (CPU) { // For sub-architecture
      case CK_88000:
        Builder.defineMacro("__mc88000__");
        break;
      case CK_88100:
        Builder.defineMacro("__mc88100__");
        break;
      case CK_88110:
        Builder.defineMacro("__mc88110__");
        break;
      default:
        break;
      }
    }
    ```

1.  接下来的几个函数是存根函数，但它们对于前端的基本支持是必需的。这包括从目标获取内置函数的函数以及查询目标是否支持特定功能的函数。目前，我们将它们留空实现，并为这些函数设置默认返回值，以便以后实现：

    ```cpp

    ArrayRef<Builtin::Info> M88kTargetInfo::getTargetBuiltins() const {
      return std::nullopt;
    }
    bool M88kTargetInfo::hasFeature(StringRef Feature) const {
      return Feature == "M88000";
    }
    ```

1.  在这些函数之后，我们将为 M88k 上的寄存器名称添加一个实现。通常，支持的寄存器名称列表及其用途可以在感兴趣的具体平台的 ABI 中找到。在这个实现中，我们将实现 0-31 号的主要通用寄存器，并创建一个数组来存储这些信息。至于寄存器别名，请注意，我们目前实现的寄存器没有别名：

    ```cpp

    const char *const M88kTargetInfo::GCCRegNames[] = {
        "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  
        "r8",  "r9",  "r10",  "r11", "r12", "r13", "r14", "r15", 
        "r16", "r17", "r18",  "r19", "r20", "r21",  "r22", "r23", 
        "r24", "r25", "r26", "r27", "r28", "r29",  "r39", "r31"};
    ArrayRef<const char *> M88kTargetInfo::getGCCRegNames() const {
      return llvm::makeArrayRef(GCCRegNames);
    }
    ArrayRef<TargetInfo::GCCRegAlias> M88kTargetInfo::getGCCRegAliases() const {
      return std::nullopt; // No aliases.
    }
    ```

1.  我们将实现的最后一个函数是验证目标内联汇编约束的函数。这个函数简单地接受一个字符，它代表内联汇编约束，并相应地处理这个约束。实现了一些内联汇编寄存器约束，例如地址、数据和浮点寄存器，以及一些常数的约束：

    ```cpp

    bool M88kTargetInfo::validateAsmConstraint(
        const char *&Name, TargetInfo::ConstraintInfo &info) const {
      switch (*Name) {
      case 'a': // address register
      case 'd': // data register
      case 'f': // floating point register
        info.setAllowsRegister();
        return true;
      case 'K': // the constant 1
      case 'L': // constant -1²⁰ .. 1¹⁹
      case 'M': // constant 1-4:
        return true;
      }
      return false;
    }
    ```

1.  我们通过关闭文件开始时启动的`clang`和`targets`命名空间来结束文件：

    ```cpp

    } // namespace targets
    } // namespace clang
    ```

在完成`clang/lib/Basic/Targets/M88k.cpp`的实现后，需要在`clang/include/clang/Driver/Options.td`中添加 M88k 功能组和有效 CPU 类型的实现：

回想一下，我们之前为我们的 M88k 目标定义了三种有效的 CPU 类型：`mc88000`、`mc88100` 和 `mc88110`。这些 CPU 类型也需要在 `Options.td` 中定义，因为该文件是定义所有将被 clang 接受的选项和标志的中心位置：

1.  首先，我们必须添加 `m_m88k_Features_Group`，它代表一组将可用于 M88k 目标的特性：

    ```cpp

    def m_m88k_Features_Group: OptionGroup<"<m88k features group>">,
                               Group<m_Group>, DocName<"M88k">;
    ```

1.  然后，我们必须在 M88k 特性组中定义三种有效的 M88k CPU 类型作为一个特性：

    ```cpp

    def m88000 : Flag<["-"], "m88000">, Group<m_m88k_Features_Group>;
    def m88100 : Flag<["-"], "m88100">, Group<m_m88k_Features_Group>;
    def m88110 : Flag<["-"], "m88110">, Group<m_m88k_Features_Group>;
    ```

这样，我们就实现了将 M88k 目标与 clang 连接的驱动程序集成部分。

## 在 clang 中实现 M88k 的 ABI 支持

现在，我们需要在 clang 的前端添加 ABI 支持，这允许我们从前端生成针对 M88k 目标的特定代码：

1.  让我们从添加以下 `clang/lib/CodeGen/TargetInfo.h` 开始。这是一个原型，用于为 M88k 目标创建代码生成信息：

    ```cpp

    std::unique_ptr<TargetCodeGenInfo> createM88kTargetCodeGenInfo(CodeGenModule &CGM);
    ```

1.  我们还需要将以下代码添加到 `clang/lib/Basic/Targets.cpp` 中，这将帮助 clang 学习 M88k 可接受的目标三元组。正如我们所见，对于 M88k 目标，可接受的操作系统是 OpenBSD。这意味着 clang 接受 `m88k-openbsd` 作为目标三元组：

    ```cpp

     #include "Targets/M88k.h"
     #include "Targets/MSP430.h"
    . . .
       case llvm::Triple::m88k:
         switch (os) {
         case llvm::Triple::OpenBSD:
           return std::make_unique<OpenBSDTargetInfo<M88kTargetInfo>>(Triple, Opts);
         default:
           return std::make_unique<M88kTargetInfo>(Triple, Opts);
         }
       case llvm::Triple::le32:
    . . .
    ```

    现在，我们需要创建一个名为 `clang/lib/CodeGen/Targets/M88k.cpp` 的文件，这样我们就可以继续为 M88k 进行代码生成信息和 ABI 实现了。

1.  在 `clang/lib/CodeGen/Targets/M88k.cpp` 中，我们必须添加以下必要的头文件，其中之一是我们刚刚修改的 `TargetInfo.h` 头文件。然后，我们必须指定我们正在使用 `clang` 和 `clang::codegen` 命名空间：

    ```cpp

    #include "ABIInfoImpl.h"
    #include "TargetInfo.h"
    using namespace clang;
    using namespace clang::CodeGen;
    ```

1.  然后，我们必须声明一个新的匿名命名空间，并将我们的 `M88kABIInfo` 放入其中。`M88kABIInfo` 从 clang 的现有 `ABIInfo` 继承，并在其中包含 `DefaultABIInfo`。对于我们的目标，我们严重依赖现有的 `ABIInfo` 和 `DefaultABIInfo`，这显著简化了 `M88kABIInfo` 类：

    ```cpp

    namespace {
    class M88kABIInfo final : public ABIInfo {
      DefaultABIInfo defaultInfo;
    ```

1.  此外，除了添加 `M88kABIInfo` 类的构造函数之外，还添加了一些方法。`computeInfo()` 实现了默认的 `clang::CodeGen::ABIInfo` 类。还有一个 `EmitVAArg()` 函数，它生成从传入的指针中检索参数的代码；稍后更新。这主要用于变长函数支持：

    ```cpp

    public:
      explicit M88kABIInfo(CodeGen::CodeGenTypes &CGT)
          : ABIInfo(CGT), defaultInfo(CGT) {}
      void computeInfo(CodeGen::CGFunctionInfo &FI) const override {}
      CodeGen::Address EmitVAArg(CodeGen::CodeGenFunction &CGF,
                                 CodeGen::Address VAListAddr,
                                 QualType Ty) const override {
        return VAListAddr;
      }
    };
    ```

1.  接下来，我们添加 `M88kTargetCodeGenInfo` 类的构造函数，它扩展了原始的 `TargetCodeGenInfo`。之后，我们必须关闭最初创建的匿名命名空间：

    ```cpp

    class M88kTargetCodeGenInfo final : public TargetCodeGenInfo {
    public:
      explicit M88kTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
          : TargetCodeGenInfo(std::make_unique<DefaultABIInfo>(CGT)) {} };
    }
    ```

1.  最后，我们必须添加实现来创建实际的 `M88kTargetCodeGenInfo` 类，并将其作为 `std::unique_ptr` 使用，它接受一个生成 LLVM IR 代码的 `CodeGenModule`。这直接对应于最初添加到 `TargetInfo.h` 中的内容：

    ```cpp

    std::unique_ptr<TargetCodeGenInfo>
    CodeGen::createM88kTargetCodeGenInfo(CodeGenModule &CGM) {
      return std::make_unique<M88kTargetCodeGenInfo>(CGM.getTypes());
    }
    ```

这就完成了前端对 M88k 的 ABI 支持。

## 在 clang 中实现 M88k 的工具链支持

在 clang 中 M88k 目标集成的最后部分是实现针对我们的目标的工具链支持。像之前一样，我们需要为工具链支持创建一个头文件。我们称这个头文件为 `clang/lib/Driver/ToolChains/Arch/M88k.h`：

1.  首先，我们必须定义 `LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H` 以防止以后多次包含，并添加任何必要的头文件以供以后使用。在此之后，我们必须声明 `clang`、`driver`、`tools` 和 `m88k` 命名空间，每个嵌套在另一个内部：

    ```cpp

    #ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H
    #define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H
    #include "clang/Driver/Driver.h"
    #include "llvm/ADT/StringRef.h"
    #include "llvm/Option/Option.h"
    #include <string>
    #include <vector>
    namespace clang {
    namespace driver {
    namespace tools {
    namespace m88k {
    ```

1.  接下来，我们必须声明一个 `enum` 值来描述浮点 ABI，这是用于软浮点和硬浮点的。这意味着浮点计算可以由浮点硬件本身完成，这很快，或者通过软件仿真，这会慢一些：

    ```cpp

    enum class FloatABI { Invalid, Soft, Hard, };
    ```

1.  在此之后，我们必须添加定义以通过驱动程序获取浮点 ABI，并通过 clang 的 `-mcpu=` 和 `-mtune=` 选项获取 CPU。我们还必须声明一个从驱动程序检索目标功能的函数：

    ```cpp

    FloatABI getM88kFloatABI(const Driver &D, const llvm::opt::ArgList &Args);
    StringRef getM88kTargetCPU(const llvm::opt::ArgList &Args);
    StringRef getM88kTuneCPU(const llvm::opt::ArgList &Args);
    void getM88kTargetFeatures(const Driver &D, const llvm::Triple &Triple, const llvm::opt::ArgList &Args, std::vector<llvm::StringRef> &Features);
    ```

1.  最后，我们通过结束命名空间和我们最初定义的宏来结束头文件：

    ```cpp

    } // end namespace m88k
    } // end namespace tools
    } // end namespace driver
    } // end namespace clang
    #endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H
    ```

我们将要实现的最后一个文件是工具链支持的 C++ 实现，位于 `clang/lib/Driver/ToolChains/Arch/M88k.cpp`：

1.  我们将再次从包括我们稍后将要使用的必要头文件和命名空间开始实现。我们还必须包括我们之前创建的 `M88k.h` 头文件：

    ```cpp

    #include "M88k.h"
    #include "ToolChains/CommonArgs.h"
    #include "clang/Driver/Driver.h"
    #include "clang/Driver/DriverDiagnostic.h"
    #include "clang/Driver/Options.h"
    #include "llvm/ADT/SmallVector.h"
    #include "llvm/ADT/StringSwitch.h"
    #include "llvm/Option/ArgList.h"
    #include "llvm/Support/Host.h"
    #include "llvm/Support/Regex.h"
    #include <sstream>
    using namespace clang::driver;
    using namespace clang::driver::tools;
    using namespace clang;
    using namespace llvm::opt;
    ```

1.  接下来实现的是 `normalizeCPU()` 函数，该函数将 CPU 名称处理为 clang 中的 `-mcpu=` 选项。正如我们所见，每个 CPU 名称都有几个可接受的变体。此外，当用户指定 `-mcpu=native` 时，它允许他们为当前主机的 CPU 类型进行编译：

    ```cpp

    static StringRef normalizeCPU(StringRef CPUName) {
      if (CPUName == "native") {
        StringRef CPU = std::string(llvm::sys::getHostCPUName());
        if (!CPU.empty() && CPU != "generic")
          return CPU;
      }
      return llvm::StringSwitch<StringRef>(CPUName)
          .Cases("mc88000", "m88000", "88000", "generic", "mc88000")
          .Cases("mc88100", "m88100", "88100", "mc88100")
          .Cases("mc88110", "m88110", "88110", "mc88110")
          .Default(CPUName);
    }
    ```

1.  接下来，我们必须实现 `getM88kTargetCPU()` 函数，其中，给定我们在 `clang/include/clang/Driver/Options.td` 中之前实现的 clang CPU 名称，我们获取我们针对的 M88k CPU 的相应 LLVM 名称：

    ```cpp

    StringRef m88k::getM88kTargetCPU(const ArgList &Args) {
      Arg *A = Args.getLastArg(options::OPT_m88000, options::OPT_m88100, options::OPT_m88110, options::OPT_mcpu_EQ);
      if (!A)
        return StringRef();
      switch (A->getOption().getID()) {
      case options::OPT_m88000:
        return "mc88000";
      case options::OPT_m88100:
        return "mc88100";
      case options::OPT_m88110:
        return "mc88110";
      case options::OPT_mcpu_EQ:
        return normalizeCPU(A->getValue());
      default:
        llvm_unreachable("Impossible option ID");
      }
    }
    ```

1.  在之后实现的是 `getM88kTuneCPU()` 函数。这是 clang `-mtune=` 选项的行为，它将指令调度模型更改为使用给定 CPU 的数据来针对 M88k。我们简单地针对我们当前正在针对的任何 CPU 进行调整：

    ```cpp

    StringRef m88k::getM88kTuneCPU(const ArgList &Args) {
      if (const Arg *A = Args.getLastArg(options::OPT_mtune_EQ))
        return normalizeCPU(A->getValue());
      return StringRef();
    }
    ```

1.  我们还将实现 `getM88kFloatABI()` 方法，该方法获取浮点 ABI。最初，我们将 ABI 设置为 `m88k::FloatABI::Invalid` 作为默认值。接下来，我们必须检查命令行是否传递了任何 `-msoft-float` 或 `-mhard-float` 选项。如果指定了 `-msoft-float`，则相应地将 ABI 设置为 `m88k::FloatABI::Soft`。同样，当指定 `-mhard-float` 时，我们将 `m88k::FloatABI::Hard` 设置为 clang。最后，如果没有指定这些选项中的任何一个，我们将选择当前平台上的默认值，对于 M88k 来说这将是一个硬浮点值：

    ```cpp

    m88k::FloatABI m88k::getM88kFloatABI(const Driver &D, const ArgList &Args) {
      m88k::FloatABI ABI = m88k::FloatABI::Invalid;
      if (Arg *A =
              Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float)) {
        if (A->getOption().matches(options::OPT_msoft_float))
          ABI = m88k::FloatABI::Soft;
        else if (A->getOption().matches(options::OPT_mhard_float))
          ABI = m88k::FloatABI::Hard;
      }
      if (ABI == m88k::FloatABI::Invalid)
        ABI = m88k::FloatABI::Hard;
      return ABI;
    }
    ```

1.  我们接下来将添加`getM88kTargetFeatures()`的实现。这个函数的重要部分是作为参数传递的`Features`向量。正如我们所见，唯一处理的目标特性是浮点 ABI。从驱动程序及其传递的参数中，我们将从之前步骤中实现的浮点 ABI 中获取适当的浮点 ABI。请注意，我们还将`-hard-float`目标特性添加到`Features`向量中，以支持软浮点 ABI，这意味着目前 M88k 只支持硬浮点：

    ```cpp

    void m88k::getM88kTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                     const ArgList &Args,
                                     std::vector<StringRef> &Features) {
      m88k::FloatABI FloatABI = m88k::getM88kFloatABI(D, Args);
      if (FloatABI == m88k::FloatABI::Soft)
        Features.push_back("-hard-float");
    }
    ```

## 构建具有 clang 集成的 M88k 目标

我们几乎完成了将 M88k 集成到 clang 中的实现。最后一步是将我们添加的新 clang 文件添加到相应的`CMakeLists.txt`文件中，这样我们就可以使用我们的 M88k 目标实现来构建 clang 项目：

1.  首先，将`Targets/M88k.cpp`行添加到`clang/lib/Basic/CMakeLists.txt`。

1.  接下来，将`Targets/M88k.cpp`添加到`clang/lib/CodeGen/CMakeLists.txt`。

1.  最后，将`ToolChains/Arch/M88k.cpp`添加到`clang/lib/Driver/CMakeLists.txt`。

就这样！这标志着我们为 M88k 目标工具链支持的工具链实现完成，这也意味着我们已经完成了 M88k 对 clang 的集成！

我们需要做的最后一步是用 M88k 目标构建 clang。以下命令将构建 clang 和 LLVM 项目。对于 clang，请注意 M88k 目标。在这里，必须添加与上一节相同的 CMake 选项`-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=M88k`：

```cpp

$ cmake -G Ninja ../llvm-project/llvm -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=M88k -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;llvm"
$ ninja
```

现在我们应该有一个可以识别 M88k 目标的 clang 版本！我们可以通过检查 clang 支持的目标列表来确认这一点，通过`--print-targets`选项：

```cpp

$ clang --print-targets | grep M88k
    m88k        - M88k
```

在本节中，我们深入探讨了将新的后端目标集成到 clang 中并使其被识别的技术细节。在下一节中，我们将探讨交叉编译的概念，我们将详细说明从当前主机针对不同 CPU 架构的流程。

# 针对不同的 CPU 架构

今天，尽管资源有限，许多小型计算机，如树莓派（Raspberry Pi），仍在使用中。在这样的计算机上运行编译器通常是不可能的，或者需要花费太多时间。因此，编译器的一个常见要求是为不同的 CPU 架构生成代码。为主机编译不同目标可执行文件的全过程被称为交叉编译。

在交叉编译中，涉及两个系统：主机系统和目标系统。编译器在主机系统上运行，为目标系统生成代码。为了表示系统，使用所谓的三元组。这是一个配置字符串，通常由 CPU 架构、供应商和操作系统组成。此外，通常还会将有关环境的附加信息添加到配置字符串中。例如，`x86_64-pc-win32` 三元组用于在 64 位 X86 CPU 上运行的 Windows 系统。CPU 架构是 `x86_64`，`pc` 是一个通用的供应商，`win32` 是操作系统，所有这些部分都由连字符连接。在 ARMv8 CPU 上运行的 Linux 系统使用 `aarch64-unknown-linux-gnu` 作为三元组，其中 `aarch64` 是 CPU 架构。此外，操作系统是 `linux`，运行 `gnu` 环境。基于 Linux 的系统没有真正的供应商，因此这部分是 `unknown`。此外，对于特定目的而言不明确或不重要的部分通常会被省略：`aarch64-linux-gnu` 三元组描述了相同的 Linux 系统。

假设你的开发机器运行的是基于 X86 64 位 CPU 的 Linux 系统，并且你想要交叉编译到运行 Linux 的 ARMv8 CPU 系统上。主机三元组是 `x86_64-linux-gnu`，目标三元组是 `aarch64-linux-gnu`。不同的系统有不同的特性。因此，你的应用程序必须以可移植的方式编写；否则，可能会出现复杂情况。一些常见的问题如下：

+   **字节序（Endianness）**：多字节值在内存中存储的顺序可能不同。

+   `int` 类型可能不足以容纳指针。

+   `long double` 类型可以使用 64 位（ARM）、80 位（X86）或 128 位（ARMv8）。PowerPC 系统可能使用双双精度算术来表示 `long double`，通过组合两个 64 位的 `double` 值来提供更高的精度。

如果你没有注意这些要点，那么你的应用程序在目标平台上可能会表现出意外的行为或崩溃，即使它在主机系统上运行得很好。LLVM 库在不同的平台上进行了测试，并且还包含对上述问题的可移植解决方案。

对于交叉编译，需要以下工具：

+   能够为目标生成代码的编译器

+   能够为目标生成二进制文件的可链接器

+   目标系统的头文件和库

幸运的是，Ubuntu 和 Debian 发行版有支持交叉编译的软件包。我们在以下设置中利用了这一点。`gcc` 和 `g++` 编译器、链接器 `ld` 以及库都作为预编译的二进制文件提供，这些二进制文件生成 ARMv8 代码和可执行文件。以下命令安装了所有这些软件包：

```cpp

$ sudo apt –y install gcc-12-aarch64-linux-gnu \
  g++-12-aarch64-linux-gnu binutils-aarch64-linux-gnu \
  libstdc++-12-dev-arm64-cross
```

新文件安装于 `/usr/aarch64-linux-gnu` 目录下。此目录是目标系统的（逻辑）根目录。它包含通常的 `bin`、`lib` 和 `include` 目录。交叉编译器（`aarch64-linux-gnu-gcc-8` 和 `aarch64-linux-gnu-g++-8`）了解此目录。

在其他系统上交叉编译

一些发行版，如 Fedora，只为裸机目标（如 Linux 内核）提供交叉编译支持，但未提供用户空间应用程序所需的头文件和库文件。在这种情况下，你可以简单地从目标系统复制缺少的文件。

如果你的发行版没有包含所需的工具链，则可以从源代码构建它。对于编译器，你可以使用 clang 或 gcc/g++。gcc 和 g++ 编译器必须配置为为目标系统生成代码，而 binutils 工具需要处理目标系统的文件。此外，C 和 C++ 库需要使用此工具链进行编译。步骤因操作系统、主机和目标架构而异。在网上，如果你搜索 `gcc` `cross-compile <架构>`，你可以找到说明。

准备就绪后，你几乎可以开始交叉编译示例应用程序（包括 LLVM 库）了，除了一个小细节。LLVM 使用 *第一章* 中的 `llvm-tblgen` 或你可以只编译此工具。假设你在这个包含本书 GitHub 仓库克隆的目录中，输入以下命令：

```cpp

$ mkdir build-host
$ cd build-host
$ cmake -G Ninja \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  ../llvm-project/llvm
$ ninja llvm-tblgen
$ cd ..
```

这些步骤现在应该很熟悉了。创建并进入一个构建目录。`cmake` 命令只为 X86 目标创建 LLVM 的构建文件。为了节省空间和时间，执行了发布构建，但启用了断言以捕获可能的错误。只有 `llvm-tblgen` 工具使用 `ninja` 进行编译。

使用 `llvm-tblgen` 工具后，你现在可以开始交叉编译过程。CMake 命令行非常长，所以你可能想将命令存储在脚本文件中。与之前的构建相比，差异在于必须提供更多信息：

```cpp

$ mkdir build-target
$ cd build-target
$ cmake -G Ninja \
  -DCMAKE_CROSSCOMPILING=True \
  -DLLVM_TABLEGEN=../build-host/bin/llvm-tblgen \
  -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu \
  -DLLVM_TARGET_ARCH=AArch64 \
  -DLLVM_TARGETS_TO_BUILD=AArch64 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_PROJECTS=tinylang \
  -DLLVM_EXTERNAL_TINYLANG_SOURCE_DIR=../tinylang \
  -DCMAKE_INSTALL_PREFIX=../target-tinylang \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc-12 \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++-12 \
  ../llvm-project/llvm
$ ninja
```

再次，在运行 CMake 命令之前，创建一个构建目录并进入它。其中一些这些 CMake 参数之前未使用过，需要一些解释：

+   将 `CMAKE_CROSSCOMPILING` 设置为 `ON` 告诉 CMake 我们正在进行交叉编译。

+   `LLVM_TABLEGEN` 指定了要使用的 `llvm-tblgen` 工具的路径。这是之前构建中使用的版本。

+   `LLVM_DEFAULT_TARGET_TRIPLE` 是目标架构的三元组。

+   `LLVM_TARGET_ARCH` 用于 **JIT** 代码生成。它默认为主机架构。对于交叉编译，这必须设置为目标架构。

+   `LLVM_TARGETS_TO_BUILD` 是 LLVM 应该包含代码生成器的目标列表。该列表至少应包括目标架构。

+   `CMAKE_C_COMPILER` 和 `CMAKE_CXX_COMPILER` 分别指定用于构建的 C 和 C++ 编译器。交叉编译器的二进制文件以目标三元组为前缀，并且 CMake 不会自动找到它们。

使用其他参数，请求启用断言的发布构建，并且我们的 tinylang 应用程序作为 LLVM 的一部分构建。一旦编译过程完成，`file` 命令可以证明我们已经为 ARMv8 创建了一个二进制文件。具体来说，我们可以运行 `$ file bin/tinylang` 并检查输出是否显示 `ELF 64-bit object for the ARM` `aarch64 architecture`。

使用 clang 进行交叉编译

由于 LLVM 为不同的架构生成代码，使用 clang 进行交叉编译似乎是显而易见的。这里的障碍是 LLVM 并不提供所有必需的部分——例如，C 库缺失。因此，您必须混合使用 LLVM 和 GNU 工具，结果您需要告诉 CMake 更多关于您使用环境的细节。至少，您需要为 clang 和 clang++ 指定以下选项：`--target=<target-triple>`（启用针对不同目标的代码生成），`--sysroot=<path>`（目标根目录的路径），`I`（头文件的搜索路径），和 `--L`（库的搜索路径）。在 CMake 运行期间，一个小型应用程序被编译，如果您的设置有问题，CMake 会抱怨。这一步足以检查您是否有正常工作的环境。常见问题包括选择错误的头文件或由于库名称不同或搜索路径错误导致的链接失败。

跨平台编译令人惊讶地复杂。通过本节中的说明，您将能够为所选的目标架构交叉编译您的应用程序。

# 摘要

在本章中，您学习了创建运行在指令选择之外的传递，特别是探索了后端中机器函数传递背后的创建！您还发现了如何将一个新的实验性目标添加到 clang 中，以及一些所需的驱动程序、ABI 和工具链更改。最后，在考虑编译器构建的最高准则时，您学习了如何为另一个目标架构交叉编译您的应用程序。

现在，我们已经到达了《学习 LLVM 17》的尾声，您已经具备了在项目中以创新方式使用 LLVM 的知识，并探索了许多有趣的主题。LLVM 生态系统非常活跃，新功能不断添加，因此请务必关注其发展！

作为编译器开发者，我们很高兴能撰写关于 LLVM 的文章，并在过程中发现了一些新功能。享受使用 LLVM 的乐趣吧！
