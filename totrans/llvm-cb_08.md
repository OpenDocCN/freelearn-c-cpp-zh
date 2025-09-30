# 第八章：编写 LLVM 后端

在本章中，我们将介绍以下内容：

+   定义寄存器和寄存器集

+   定义调用约定

+   定义指令集

+   实现帧降低

+   打印指令

+   选择指令

+   添加指令编码

+   支持子目标

+   降低到多个指令

+   注册目标

# 简介

编译器的最终目标是生成目标代码，或者可以转换为对象代码并在实际硬件上执行的汇编代码。为了生成汇编代码，编译器需要了解目标机器架构的各个方面——寄存器、指令集、调用约定、流水线等。在这个阶段还可以进行许多优化。

LLVM 有自己定义目标机器的方式。它使用 `tablegen` 来指定目标寄存器、指令、调用约定等。`tablegen` 函数简化了我们以编程方式描述大量架构属性的方式。

LLVM 的后端具有流水线结构，指令会经过类似这样的阶段；从 LLVM IR 到 `SelectionDAG`，然后到 `MachineDAG`，接着到 `MachineInstr`，最后到 `MCInst`。

IR 转换为 SelectionDAG（**DAG** 代表 **有向无环图**）。然后进行 SelectionDAG 合法化，将非法指令映射到目标机器允许的合法操作上。在此阶段之后，SelectionDAG 转换为 MachineDAG，这基本上是后端支持的指令选择。

CPU 执行一系列线性指令。调度步骤的目标是通过为操作分配顺序来线性化 DAG。LLVM 的代码生成器采用巧妙的启发式方法（如寄存器压力降低）来尝试生成一个能够产生更快速代码的调度。寄存器分配策略在生成更好的 LLVM 代码中也起着重要作用。

本章介绍了如何从头开始构建一个 LLVM 玩具后端。到本章结束时，我们将能够为示例玩具后端生成汇编代码。

## 示例后端

本章考虑的示例后端是一个简单的 RISC 类型架构，有少量寄存器（例如 r0-r3），一个栈指针（sp）和一个链接寄存器（lr），用于存储返回地址。

这个玩具后端的调用约定类似于 ARM 架构——传递给函数的参数将存储在寄存器集 r0-r1 中，返回值将存储在 r0 中。

# 定义寄存器和寄存器集

这个菜谱展示了如何在 `.td` 文件中定义寄存器和寄存器集。`tablegen` 函数将这个 `.td` 文件转换为 `.inc` 文件，这些文件将成为我们 `.cpp` 文件中的 `#include` 声明，并引用寄存器。

## 准备工作

我们已经定义了我们的玩具目标机器具有四个寄存器（r0-r3）、一个堆栈寄存器（sp）和一个链接寄存器（lr）。这些可以在`TOYRegisterInfo.td`文件中指定。`tablegen`函数提供了`Register`类，可以扩展以指定寄存器。

## 如何做到这一点…

要使用目标描述文件定义后端架构，请按照以下步骤进行。

1.  在`lib/Target`目录下创建一个名为`TOY`的新文件夹：

    ```cpp
    $ mkdir llvm_root_directory/lib/Target/TOY

    ```

1.  在新创建的`TOY`文件夹中创建一个名为`TOYRegisterInfo.td`的新文件：

    ```cpp
    $ cd llvm_root_directory/lib/Target/TOY
    $ vi TOYRegisterInfo.td

    ```

1.  定义硬件编码、命名空间、寄存器和寄存器类：

    ```cpp
    class TOYReg<bits<16> Enc, string n> : Register<n> {
      let HWEncoding = Enc;
      let Namespace = "TOY";
    }

    foreach i = 0-3 in {
        def R#i : R<i, "r"#i >;
    }

    def SP  : TOYReg<13, "sp">;
    def LR  : TOYReg<14, "lr">;

    def GRRegs : RegisterClass<"TOY", [i32], 32,
      (add R0, R1, R2, R3, SP)>;
    ```

## 它是如何工作的…

`tablegen`函数处理这个`.td`文件以生成`.inc`文件，该文件通常为这些寄存器生成枚举。这些枚举可以在`.cpp`文件中使用，其中寄存器可以引用为`TOY::R0`。这些`.inc`文件将在我们构建 LLVM 项目时生成。

## 参见

+   要获取更多关于如何为更高级的架构（如 ARM）定义寄存器的详细信息，请参考 LLVM 源代码中的`lib/Target/ARM/ARMRegisterInfo.td`文件。

# 定义调用约定

调用约定指定了值是如何在函数调用之间传递的。我们的 TOY 架构指定了两个参数将通过两个寄存器（r0 和 r1）传递，其余的将通过堆栈传递。这个配方展示了如何定义调用约定，该约定将通过函数指针在`ISelLowering`（第六章中讨论的指令选择降低阶段，*目标无关代码生成器*）中使用。

调用约定将在`TOYCallingConv.td`文件中定义，该文件将主要包含两个部分——一个用于定义返回值约定，另一个用于定义参数传递约定。返回值约定指定了返回值将驻留在何处以及哪些寄存器中。参数传递约定将指定传递的参数将驻留在何处以及哪些寄存器中。在定义玩具架构的调用约定时，将继承`CallingConv`类。

## 如何做到这一点…

要实现调用约定，请按照以下步骤进行：

1.  在`lib/Target/TOY`文件夹中创建一个名为`TOYCallingConv.td`的新文件：

    ```cpp
    $ vi TOYCallingConv.td

    ```

1.  在该文件中，定义返回值约定，如下所示：

    ```cpp
    def RetCC_TOY : CallingConv<[
     CCIfType<[i32], CCAssignToReg<[R0]>>,
     CCIfType<[i32], CCAssignToStack<4, 4>>
    ]>;

    ```

1.  此外，定义参数传递约定，如下所示：

    ```cpp
    def CC_TOY : CallingConv<[
     CCIfType<[i8, i16], CCPromoteToType<i32>>,
     CCIfType<[i32], CCAssignToReg<[R0, R1]>>,
     CCIfType<[i32], CCAssignToStack<4, 4>>
    ]>;

    ```

1.  定义调用者保存的寄存器集：

    ```cpp
    def CC_Save : CalleeSavedRegs<(add R2, R3)>;

    ```

## 它是如何工作的…

在你刚才阅读的`.td`文件中，已经指定了 32 位整型的返回值存储在 r0 寄存器中。每当向函数传递参数时，前两个参数将存储在 r0 和 r1 寄存器中。还指定了每当遇到任何数据类型，如 8 位或 16 位的整数时，它将被提升为 32 位整数类型。

`tablegen`函数生成一个`TOYCallingConv.inc`文件，该文件将在`TOYISelLowering.cpp`文件中引用。用于定义参数处理的两个目标`hook`函数是`LowerFormalArguments()`和`LowerReturn()`。

## 参见

+   要查看高级架构（如 ARM）的详细实现，请查看`lib/Target/ARM/ARMCallingConv.td`文件

# 定义指令集

架构的指令集根据架构中存在的各种特征而变化。本食谱演示了如何为目标架构定义指令集。

指令目标描述文件中定义了三个内容：操作数、汇编字符串和指令模式。规范包含定义或输出的列表，以及使用或输入的列表。可以有不同类型的操作数类，例如`Register`类，以及立即数和更复杂的`register + imm`操作数。

在这里，演示了一个简单的加法指令定义，它接受两个寄存器作为操作数。

## 如何操作...

要使用目标描述文件定义指令集，请按照以下步骤进行。

1.  在`lib/Target/TOY`文件夹中创建一个名为`TOYInstrInfo.td`的新文件：

    ```cpp
    $ vi TOYInstrInfo.td

    ```

1.  指定两个寄存器操作数之间`add`指令的操作数、汇编字符串和指令模式：

    ```cpp
    def ADDrr : InstTOY<(outs GRRegs:$dst),
                        (ins GRRegs:$src1, GRRegs:$src2),
                         "add $dst, $src1,z$src2",
    [(set i32:$dst, (add i32:$src1, i32:$src2))]>;
    ```

## 它是如何工作的…

`add`寄存器到寄存器指令指定`$dst`作为结果操作数，它属于`General Register`类型类；输入`$src1`和`$src2`作为两个输入操作数，它们也属于`General Register`类型类；指令汇编字符串为 32 位整型的`"add $dst, $src1, $src2"`。

因此，将生成两个寄存器之间`add`操作的汇编代码，如下所示：

```cpp
add r0, r0, r1
```

上述代码指示将 r0 和 r1 寄存器的内容相加，并将结果存储在 r0 寄存器中。

## 参见

+   许多指令将具有相同类型的指令模式——例如`add`、`sub`等 ALU 指令。在这种情况下，可以使用多类来定义公共属性。有关高级架构（如 ARM）的各种指令集的更详细信息，请参阅`lib/Target/ARM/ARMInstrInfo.td`文件

# 实现帧降低

本食谱讨论了目标架构的帧降低。帧降低涉及函数调用的前缀和后缀的生成。

## 准备工作

### 注意

需要定义两个用于帧降低的函数，即`TOYFrameLowering::emitPrologue()`和`TOYFrameLowering::emitEpilogue()`。

## 如何操作…

以下函数定义在`lib/Target/TOY`文件夹中的`TOYFrameLowering.cpp`文件中：

1.  `emitPrologue`函数可以定义如下：

    ```cpp
    void TOYFrameLowering::emitPrologue(MachineFunction &MF) const {
      const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
      MachineBasicBlock &MBB = MF.front();
      MachineBasicBlock::iterator MBBI = MBB.begin();
      DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
      uint64_t StackSize = computeStackSize(MF);
      if (!StackSize) {
        return;
      }
      unsigned StackReg = TOY::SP;
      unsigned OffsetReg = materializeOffset(MF, MBB, MBBI, (unsigned)StackSize);
      if (OffsetReg) {
        BuildMI(MBB, MBBI, dl, TII.get(TOY::SUBrr), StackReg)
            .addReg(StackReg)
            .addReg(OffsetReg)
            .setMIFlag(MachineInstr::FrameSetup);
      } else {
        BuildMI(MBB, MBBI, dl, TII.get(TOY::SUBri), StackReg)
            .addReg(StackReg)
            .addImm(StackSize)
            .setMIFlag(MachineInstr::FrameSetup);
      }
    }
    ```

1.  `emitEpilogue`函数可以定义如下：

    ```cpp
    void TOYFrameLowering::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {

      const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
    MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
      DebugLoc dl = MBBI->getDebugLoc();
      uint64_t StackSize = computeStackSize(MF);
      if (!StackSize) {
        return;
      }
      unsigned StackReg = TOY::SP;
      unsigned OffsetReg = materializeOffset(MF, MBB, MBBI, (unsigned)StackSize);
      if (OffsetReg) {
        BuildMI(MBB, MBBI, dl, TII.get(TOY::ADDrr), StackReg)
            .addReg(StackReg)
            .addReg(OffsetReg)
            .setMIFlag(MachineInstr::FrameSetup);
      } else {
        BuildMI(MBB, MBBI, dl, TII.get(TOY::ADDri), StackReg)
            .addReg(StackReg)
            .addImm(StackSize)
            .setMIFlag(MachineInstr::FrameSetup);
      }
    }
    ```

1.  以下是一些用于确定`ADD`栈操作偏移量的辅助函数：

    ```cpp
    static unsigned materializeOffset(MachineFunction &MF, MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned Offset) {
      const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
      DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
      const uint64_t MaxSubImm = 0xfff;
      if (Offset <= MaxSubImm) {
        return 0;
      } else {
        unsigned OffsetReg = TOY::R2;
        unsigned OffsetLo = (unsigned)(Offset & 0xffff);
        unsigned OffsetHi = (unsigned)((Offset & 0xffff0000) >> 16);
        BuildMI(MBB, MBBI, dl, TII.get(TOY::MOVLOi16), OffsetReg)
            .addImm(OffsetLo)
            .setMIFlag(MachineInstr::FrameSetup);
        if (OffsetHi) {
          BuildMI(MBB, MBBI, dl, TII.get(TOY::MOVHIi16), OffsetReg)
              .addReg(OffsetReg)
              .addImm(OffsetHi)
              .setMIFlag(MachineInstr::FrameSetup);
        }
        return OffsetReg;
      }
    }
    ```

1.  以下是一些用于计算栈大小的辅助函数：

    ```cpp
    uint64_t TOYFrameLowering::computeStackSize(MachineFunction &MF) const {
      MachineFrameInfo *MFI = MF.getFrameInfo();
      uint64_t StackSize = MFI->getStackSize();
      unsigned StackAlign = getStackAlignment();
      if (StackAlign > 0) {
        StackSize = RoundUpToAlignment(StackSize, StackAlign);
      }
      return StackSize;
    }
    ```

## 它是如何工作的…

`emitPrologue` 函数首先计算栈大小以确定是否需要使用前导代码。然后通过计算偏移量来调整栈指针。对于后导代码，它首先检查是否需要后导代码。然后恢复栈指针到函数开始时的状态。

例如，考虑以下输入 IR：

```cpp
%p = alloca i32, align 4
store i32 2, i32* %p
%b = load i32* %p, align 4
%c = add nsw i32 %a, %b
```

生成的 TOY 汇编将看起来像这样：

```cpp
sub sp, sp, #4 ; prologue
movw r1, #2
str r1, [sp]
add r0, r0, #2
add sp, sp, #4 ; epilogue
```

## 参见

+   对于高级架构框架降低，例如在 ARM 中，请参考 `lib/Target/ARM/ARMFrameLowering.cpp` 文件。

# 打印指令

打印汇编指令是生成目标代码的重要步骤。定义了各种类，作为流式传输的网关。指令字符串由之前定义的 `.td` 文件提供。

## 准备工作

打印指令的第一步是在 `.td` 文件中定义指令字符串，这在 *定义指令集* 菜谱中已完成。

## 如何操作…

执行以下步骤：

1.  在 `TOY` 文件夹内创建一个名为 `InstPrinter` 的新文件夹：

    ```cpp
    $ cd lib/Target/TOY
    $ mkdir InstPrinter

    ```

1.  在一个新文件中，称为 `TOYInstrFormats.td`，定义 `AsmString` 变量：

    ```cpp
    class InstTOY<dag outs, dag ins, string asmstr, list<dag> pattern>
        : Instruction {
      field bits<32> Inst;
      let Namespace = "TOY";
      dag OutOperandList = outs;
      dag InOperandList = ins;
      let AsmString   = asmstr;
      let Pattern = pattern;
      let Size = 4;
    }
    ```

1.  创建一个名为 `TOYInstPrinter.cpp` 的新文件，并定义 `printOperand` 函数，如下所示：

    ```cpp
    void TOYInstPrinter::printOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) {
      const MCOperand &Op = MI->getOperand(OpNo);
      if (Op.isReg()) {
        printRegName(O, Op.getReg());
        return;
      }

      if (Op.isImm()) {
        O << "#" << Op.getImm();
        return;
      }
      assert(Op.isExpr() && "unknown operand kind in printOperand");
      printExpr(Op.getExpr(), O);
    }
    ```

1.  此外，定义一个打印寄存器名称的函数：

    ```cpp
    void TOYInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
      OS << StringRef(getRegisterName(RegNo)).lower();
    }
    ```

1.  定义一个打印指令的函数：

    ```cpp
    void TOYInstPrinter::printInst(const MCInst *MI, raw_ostream &O,StringRef Annot) {
      printInstruction(MI, O);
      printAnnotation(O, Annot);
    }
    ```

1.  它还要求指定 `MCASMinfo` 以打印指令。这可以通过定义 `TOYMCAsmInfo.h` 和 `TOYMCAsmInfo.cpp` 文件来完成。

    `TOYMCAsmInfo.h` 文件可以定义如下：

    ```cpp
    #ifndef TOYTARGETASMINFO_H
    #define TOYTARGETASMINFO_H

    #include "llvm/MC/MCAsmInfoELF.h"

    namespace llvm {
    class StringRef;
    class Target;

    class TOYMCAsmInfo : public MCAsmInfoELF {
      virtual void anchor();

    public:
      explicit TOYMCAsmInfo(StringRef TT);
    };

    } // namespace llvm
    #endif
    ```

    `TOYMCAsmInfo.cpp` 文件可以定义如下：

    ```cpp
    #include "TOYMCAsmInfo.h"
    #include "llvm/ADT/StringRef.h"
    using namespace llvm;

    void TOYMCAsmInfo::anchor() {}

    TOYMCAsmInfo::TOYMCAsmInfo(StringRef TT) {
      SupportsDebugInformation = true;
      Data16bitsDirective = "\t.short\t";
      Data32bitsDirective = "\t.long\t";
      Data64bitsDirective = 0;
      ZeroDirective = "\t.space\t";
      CommentString = "#";

      AscizDirective = ".asciiz";

      HiddenVisibilityAttr = MCSA_Invalid;
      HiddenDeclarationVisibilityAttr = MCSA_Invalid;
      ProtectedVisibilityAttr = MCSA_Invalid;
    }
    ```

1.  定义指令打印器的 `LLVMBuild.txt` 文件：

    ```cpp
    [component_0]
    type = Library
    name = TOYAsmPrinter
    parent = TOY
    required_libraries = MC Support
    add_to_library_groups = TOY
    ```

1.  定义 `CMakeLists.txt`：

    ```cpp
    add_llvm_library(LLVMTOYAsmPrinter
      TOYInstPrinter.cpp
      )
    ```

## 工作原理…

当最终编译发生时，**llc** 工具——一个静态编译器——将生成 TOY 架构的汇编代码。

例如，以下 IR 当提供给 llc 工具时，将生成如下汇编：

```cpp
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32- i16:16:32-i64:32-f64:32-a:0:32-n32"
target triple = "toy"
define i32 @foo(i32 %a, i32 %b) {
   %c = add nsw i32 %a, %b
   ret i32 %c
}

$ llc foo.ll
.text
.file "foo.ll"
.globl foo
.type foo,@function
foo:     # @foo
# BB#0:   # %entry
add r0, r0, r1
b lr
.Ltmp0:
.size foo, .Ltmp0-foo
```

# 选择指令

DAG 中的 IR 指令需要降低到特定目标的指令。SDAG 节点包含 IR，需要映射到机器特定的 DAG 节点。选择阶段的输出已准备好进行调度。

## 准备工作

1.  为了选择特定机器的指令，需要定义一个单独的类，`TOYDAGToDAGISel`。要编译包含此类定义的文件，请将文件名添加到 `TOY` 文件夹中的 `CMakeLists.txt` 文件中：

    ```cpp
    $ vi CMakeLists .txt
    add_llvm_target(...
    ...
    TOYISelDAGToDAG.cpp
    ...
    )
    ```

1.  需要在 `TOYTargetMachine.h` 和 `TOYTargetMachine.cpp` 文件中添加一个遍历入口：

    ```cpp
    $ vi TOYTargetMachine.h
    const TOYInstrInfo *getInstrInfo() const override {
    return getSubtargetImpl()->getInstrInfo();
    }
    ```

1.  `TOYTargetMachine.cpp` 中的以下代码将在指令选择阶段创建一个遍历：

    ```cpp
    class TOYPassConfig : public TargetPassConfig {
    public:
    ...
    virtual bool addInstSelector();
    };
    ...
    bool TOYPassConfig::addInstSelector() {
    addPass(createTOYISelDag(getTOYTargetMachine()));
    return false;
    }
    ```

## 如何操作…

要定义一个指令选择函数，请按照以下步骤进行：

1.  创建一个名为 `TOYISelDAGToDAG.cpp` 的文件：

    ```cpp
    $ vi TOYISelDAGToDAG.cpp

    ```

1.  包含以下文件：

    ```cpp
    #include "TOY.h"
    #include "TOYTargetMachine.h"
    #include "llvm/CodeGen/SelectionDAGISel.h"
    #include "llvm/Support/Compiler.h"
    #include "llvm/Support/Debug.h"
    #include "TOYInstrInfo.h"

    ```

1.  定义一个名为 `TOYDAGToDAGISel` 的新类，如下所示，它将继承自 `SelectionDAGISel` 类：

    ```cpp
    class TOYDAGToDAGISel : public SelectionDAGISel {
      const TOYSubtarget &Subtarget;

    public:
      explicit TOYDAGToDAGISel(TOYTargetMachine &TM, CodeGenOpt::Level OptLevel)
    : SelectionDAGISel(TM, OptLevel),   Subtarget(*TM.getSubtargetImpl()) {}
    };
    ```

1.  在这个类中需要定义的最重要函数是`Select()`，它将返回一个针对机器指令的特定`SDNode`对象：

    在类中声明它：

    ```cpp
    SDNode *Select(SDNode *N);

    ```

    进一步定义如下：

    ```cpp
    SDNode *TOYDAGToDAGISel::Select(SDNode *N) {
     return SelectCode(N);
    }

    ```

1.  另一个重要的功能是用于定义地址选择函数，该函数将计算加载和存储操作的基础地址和偏移量。

    如下所示声明它：

    ```cpp
        bool SelectAddr(SDValue Addr, SDValue &Base, SDValue &Offset);
    ```

    如此定义它：

    ```cpp
    bool TOYDAGToDAGISel::SelectAddr(SDValue Addr, SDValue &Base, SDValue &Offset) {
      if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
            Base = CurDAG->getTargetFrameIndex(FIN->getIndex(),
                                           getTargetLowering()- >getPointerTy());
            Offset = CurDAG->getTargetConstant(0, MVT::i32);
            return true;
        }
      if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
            Addr.getOpcode() == ISD::TargetGlobalAddress ||
            Addr.getOpcode() == ISD::TargetGlobalTLSAddress) {
            return false; // direct calls.
      }

        Base = Addr;
        Offset = CurDAG->getTargetConstant(0, MVT::i32);
        return true;
    }
    ```

1.  `createTOYISelDag`转换将合法化的 DAG 转换为特定于玩具的 DAG，以便在同一文件中进行指令调度：

    ```cpp
    FunctionPass *llvm::createTOYISelDag(TOYTargetMachine &TM, CodeGenOpt::Level OptLevel) {
    return new TOYDAGToDAGISel(TM, OptLevel);
    }
    ```

## 它是如何工作的…

`TOYISelDAGToDAG.cpp`中的`TOYDAGToDAGISel::Select()`函数用于选择 OP 代码 DAG 节点，而`TOYDAGToDAGISel::SelectAddr()`用于选择具有`addr`类型的 DATA DAG 节点。请注意，如果地址是全局或外部的，我们返回地址为 false，因为它的地址是在全局上下文中计算的。

## 参见

+   关于选择复杂架构（如 ARM）的机器指令 DAG 的详细信息，请查看 LLVM 源代码中的`lib/Target/ARM/ARMISelDAGToDAG.cpp`文件。

# 添加指令编码

如果指令需要根据它们相对于位字段的编码进行特定化，这可以通过在定义指令时在`.td`文件中指定位字段来实现。

## 如何做到这一点…

在定义指令时包含指令编码，请按照以下步骤进行：

1.  将用于注册`add`指令的寄存器操作数将有一些定义的指令编码。指令的大小为 32 位，其编码如下：

    ```cpp
    bits 0 to 3 -> src2, second register operand
    bits 4 to 11 -> all zeros
    bits 12 to 15 -> dst, for destination register
    bits 16 to 19 -> src1, first register operand
    bit 20 -> zero
    bit 21 to 24 -> for opcode
    bit 25 to 27 -> all zeros
    bit 28 to 31 -> 1110
    ```

    这可以通过在`.td`文件中指定前导位模式来实现。

1.  在`TOYInstrFormats.td`文件中，定义一个新变量，称为`Inst`：

    ```cpp
    class InstTOY<dag outs, dag ins, string asmstr, list<dag> pattern>
          : Instruction {
      field bits<32> Inst;

      let Namespace = "TOY";
        …
       …
        let AsmString   = asmstr;
       …
     …
     }
    ```

1.  在`TOYInstrInfo.td`文件中，定义一个指令编码：

    ```cpp
    def ADDrr : InstTOY<(outs GRRegs:$dst),(ins GRRegs:$src1, GRRegs:$src2) ... > {
    bits<4> src1;
    bits<4> src2;
    bits<4> dst;
    let Inst{31-25} = 0b1100000;
    let Inst{24-21} = 0b1100; // Opcode
    let Inst{20} = 0b0;
    let Inst{19-16} = src1; // Operand 1
    let Inst{15-12} = dst; // Destination
    let Inst{11-4} = 0b00000000;
    let Inst{3-0} = src2;
    }
    ```

1.  在`TOY/MCTargetDesc`文件夹中，在`TOYMCCodeEmitter.cpp`文件中，如果机器指令操作数是寄存器，将调用编码函数。

    ```cpp
    unsigned TOYMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                                 const MCOperand &MO,
                                                 SmallVectorImpl<MCFixup> &Fixups,
                                                 const MCSubtargetInfo &STI) const {
        if (MO.isReg()) {
          return CTX.getRegisterInfo()- >getEncodingValue(MO.getReg());
      }
    ```

1.  此外，在同一文件中，指定了一个用于编码指令的函数：

    ```cpp
    void TOYMCCodeEmitter::EncodeInstruction(const MCInst &MI, raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
          const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
          if (Desc.getSize() != 4) {
            llvm_unreachable("Unexpected instruction size!");
      }

          const uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);

      EmitConstant(Binary, Desc.getSize(), OS);
     ++MCNumEmitted;
    }
    ```

## 它是如何工作的…

在`.td`文件中，已经指定了指令的编码——操作数、目的、标志条件和指令操作码的位。机器代码生成器通过从`.td`文件生成`.inc`文件并通过函数调用获取这些编码。它将这些指令编码并发出相同的指令打印。

## 参见

+   对于如 ARM 这样的复杂架构，请查看 LLVM 主分支`lib/Target/ARM`目录下的`ARMInstrInfo.td`和`ARMInstrInfo.td`文件。

# 支持子目标

目标可能有一个子目标——通常是一个带有指令的变体——处理操作数的方式，以及其他方式。这种子目标特性可以在 LLVM 后端得到支持。子目标可能包含一些额外的指令、寄存器、调度模型等。ARM 有如 NEON 和 THUMB 这样的子目标，而 x86 有如 SSE、AVX 这样的子目标特性。对于子目标特性，指令集是不同的，例如，ARM 的 NEON 和 SSE/AVX 支持向量指令的子目标特性。SSE 和 AVX 也支持向量指令集，但它们的指令各不相同。

## 如何做...

这个示例将演示如何在后端添加对支持子目标特性的支持。必须定义一个新的类，该类将继承`TargetSubtargetInfo`类：

1.  创建一个名为`TOYSubtarget.h`的新文件：

    ```cpp
    $ vi TOYSubtarget.h

    ```

1.  包含以下文件：

    ```cpp
    #include "TOY.h"
    #include "TOYFrameLowering.h"
    #include "TOYISelLowering.h"
    #include "TOYInstrInfo.h"
    #include "TOYSelectionDAGInfo.h"
    #include "TOYSubtarget.h"
    #include "llvm/Target/TargetMachine.h"
    #include "llvm/Target/TargetSubtargetInfo.h"
    #include "TOYGenSubtargetInfo.inc"

    ```

1.  定义一个名为`TOYSubtarget`的新类，其中包含有关数据布局、目标降低、目标选择 DAG、目标帧降低等信息的一些私有成员：

    ```cpp
    class TOYSubtarget : public TOYGenSubtargetInfo {
      virtual void anchor();

    private:
      const DataLayout DL;       // Calculates type size & alignment.
      TOYInstrInfo InstrInfo;
      TOYTargetLowering TLInfo;
      TOYSelectionDAGInfo TSInfo;
      TOYFrameLowering FrameLowering;
      InstrItineraryData InstrItins;
    ```

1.  声明其构造函数：

    ```cpp
    TOYSubtarget(const std::string &TT, const std::string &CPU, const std::string &FS, TOYTargetMachine &TM);
    ```

    这个构造函数初始化数据成员以匹配指定的三元组。

1.  定义一些辅助函数以返回类特定的数据：

    ```cpp
    const InstrItineraryData *getInstrItineraryData() const override {
      return &InstrItins;
    }

    const TOYInstrInfo *getInstrInfo() const override { return &InstrInfo; }

    const TOYRegisterInfo *getRegisterInfo() const override {
      return &InstrInfo.getRegisterInfo();
    }

    const TOYTargetLowering *getTargetLowering() const override {
      return &TLInfo;
    }

    const TOYFrameLowering *getFrameLowering() const override {
      return &FrameLowering;
    }

    const TOYSelectionDAGInfo *getSelectionDAGInfo() const override {
      return &TSInfo;
    }

    const DataLayout *getDataLayout() const override { return &DL; }

    void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

    TO LC,

    Please maintain the representation of the above code EXACTLY as seen above.
    ```

1.  创建一个名为`TOYSubtarget.cpp`的新文件，并按如下方式定义构造函数：

    ```cpp
    TOYSubtarget::TOYSubtarget(const std::string &TT, const std::string &CPU, const std::string &FS, TOYTargetMachine &TM)
          DL("e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32- f64:32-a:0:32-n32"),
          InstrInfo(), TLInfo(TM), TSInfo(DL), FrameLowering() {}
    ```

子目标有自己的数据布局定义，以及其他信息，如帧降低、指令信息、子目标信息等。

## 参见

+   要深入了解子目标实现细节，请参考 LLVM 源代码中的`lib/Target/ARM/ARMSubtarget.cpp`文件

# 将代码降低到多个指令

让我们以实现一个 32 位立即数加载的高/低对为例，其中 MOVW 表示移动一个 16 位低立即数和一个清除的 16 位高位，而 MOVT 表示移动一个 16 位高立即数。

## 如何做...

实现这种多指令降低可能有多种方式。我们可以通过使用伪指令或在选择 DAG 到 DAG 阶段来完成。

1.  要在不使用伪指令的情况下完成，定义一些约束。这两条指令必须按顺序执行。MOVW 清除了高 16 位。它的输出通过 MOVT 读取以填充高 16 位。这可以通过在 tablegen 中指定约束来实现：

    ```cpp
    def MOVLOi16 : MOV<0b1000, "movw", (ins i32imm:$imm),
                      [(set i32:$dst, i32imm_lo:$imm)]>;
    def MOVHIi16 : MOV<0b1010, "movt", (ins GRRegs:$src1, i32imm:$imm),
                      [/* No Pattern */]>;
    ```

    第二种方式是在`.td`文件中定义伪指令：

    ```cpp
    def MOVi32 : InstTOY<(outs GRRegs:$dst), (ins i32imm:$src), "", [(set i32:$dst, (movei32 imm:$src))]> {
      let isPseudo = 1;
    }
    ```

1.  然后伪指令通过`TOYInstrInfo.cpp`文件中的目标函数降低：

    ```cpp
    bool TOYInstrInfo::expandPostRAPseudo(MachineBasicBlock::iterato r MI) const {
      if (MI->getOpcode() == TOY::MOVi32){
        DebugLoc DL = MI->getDebugLoc();
        MachineBasicBlock &MBB = *MI->getParent();

        const unsigned DstReg = MI->getOperand(0).getReg();
        const bool DstIsDead = MI->getOperand(0).isDead();

        const MachineOperand &MO = MI->getOperand(1);

        auto LO16 = BuildMI(MBB, MI, DL, get(TOY::MOVLOi16), DstReg);
        auto HI16 = BuildMI(MBB, MI, DL, get(TOY::MOVHIi16))
                        .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
                        .addReg(DstReg);

      MBB.erase(MI);
        return true;
      }
    }
    ```

1.  编译整个 LLVM 项目：

    例如，一个包含 IR 的`ex.ll`文件看起来像这样：

    ```cpp
    define i32 @foo(i32 %a) #0 {
    %b = add nsw i32 %a, 65537 ; 0x00010001
    ret i32 %b
    }
    ```

    生成的汇编代码将如下所示：

    ```cpp
    movw r1, #1
    movt r1, #1
    add r0, r0, r1
    b lr
    ```

## 它是如何工作的...

第一条指令，`movw`，将移动低 16 位中的 1 并清除高 16 位。因此，第一条指令将在 r1 中写入`0x00000001`。在下一条指令`movt`中，将写入高 16 位。因此，在 r1 中，将写入`0x0001XXXX`，而不会影响低位。最后，r1 寄存器中将包含`0x00010001`。每当遇到`.td`文件中指定的伪指令时，其展开函数将被调用以指定伪指令将展开成什么。

在前面的例子中，`mov32`立即数将由两条指令实现：`movw`（低 16 位）和`movt`（高 16 位）。它在`.td`文件中被标记为伪指令。当需要发出此伪指令时，其展开函数被调用，它构建两个机器指令：`MOVLOi16`和`MOVHIi16`。这些映射到目标架构的`movw`和`movt`指令。

## 参考也

+   要深入了解实现这种多指令降低的实现，请查看 LLVM 源代码中的 ARM 目标实现，在`lib/Target/ARM/ARMInstrInfo.td`文件中。

# 注册目标

对于在 TOY 目标架构中运行 llc 工具，它必须与 llc 工具注册。这个配方演示了需要修改哪些配置文件来注册一个目标。构建文件在这个配方中被修改。

## 如何做到这一点…

要使用静态编译器注册目标，请按照以下步骤操作：

1.  首先，将 TOY 后端条目添加到`llvm_root_dir/CMakeLists.txt`：

    ```cpp
    set(LLVM_ALL_TARGETS
      AArch64
      ARM
      …
      …
      TOY
      )
    ```

1.  然后将 toy 条目添加到`llvm_root_dir/include/llvm/ADT/Triple.h`：

    ```cpp
    class Triple {
    public:
      enum ArchType {
        UnknownArch,

        arm,        // ARM (little endian): arm, armv.*, xscale
        armeb,      // ARM (big endian): armeb
        aarch64,    // AArch64 (little endian): aarch64
        …
       …

    toy     // TOY: toy
    };
    ```

1.  将 toy 条目添加到`llvm_root_dir/include/llvm/ MC/MCExpr.h`：

    ```cpp
    class MCSymbolRefExpr : public MCExpr {
    public:
    enum VariantKind {
    ...
    VK_TOY_LO,
    VK_TOY_HI,
    };
    ```

1.  将 toy 条目添加到`llvm_root_dir/include/llvm/ Support/ELF.h`：

    ```cpp
    enum {
      EM_NONE          = 0, // No machine
      EM_M32           = 1, // AT&T WE 32100
      …
      …
      EM_TOY           = 220 // whatever is the next number
    };
    ```

1.  然后，将 toy 条目添加到`lib/MC/MCExpr.cpp`：

    ```cpp
    StringRef MCSymbolRefExpr::getVariantKindName(VariantKind Kind) {
    switch (Kind) {

      …
      …
      case VK_TOY_LO: return "TOY_LO";
      case VK_TOY_HI: return "TOY_HI";
      }
    …
    }
    ```

1.  接下来，将 toy 条目添加到`lib/Support/Triple.cpp`：

    ```cpp
    const char *Triple::getArchTypeName(ArchType Kind) {
      switch (Kind) {
     …
     …
     case toy:         return "toy";

    }

    const char *Triple::getArchTypePrefix(ArchType Kind) {
      switch (Kind) {
     …
     …
    case toy:         return "toy";
      }
    }

    Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
    …
    …
        .Case("toy", toy)
    …
    }

    static Triple::ArchType parseArch(StringRef ArchName) {
    …
    …
        .Case("toy", Triple::toy)
    …
    }

    static unsigned getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
    …
    …
    case llvm::Triple::toy:
        return 32;

    …
    …
    }

    Triple Triple::get32BitArchVariant() const {
    …
    …
    case Triple::toy:
        // Already 32-bit.
        break;
    …
    }

    Triple Triple::get64BitArchVariant() const {
    …
    …
    case Triple::toy:
        T.setArch(UnknownArch);
        break;

    …
    …
    }
    ```

1.  将 toy 目录条目添加到`lib/Target/LLVMBuild.txt`：

    ```cpp
    [common]
    subdirectories = ARM AArch64 CppBackend Hexagon MSP430 … … TOY
    ```

1.  在`lib/Target/TOY`文件夹中创建一个名为`TOY.h`的新文件：

    ```cpp
    #ifndef TARGET_TOY_H
    #define TARGET_TOY_H

    #include "MCTargetDesc/TOYMCTargetDesc.h"
    #include "llvm/Target/TargetMachine.h"

    namespace llvm {
    class TargetMachine;
    class TOYTargetMachine;

    FunctionPass *createTOYISelDag(TOYTargetMachine &TM,
                                   CodeGenOpt::Level OptLevel);
    } // end namespace llvm;

    #endif
    ```

1.  在`lib/Target/TOY`文件夹中创建一个名为`TargetInfo`的新文件夹。在该文件夹内，创建一个名为`TOYTargetInfo.cpp`的新文件，如下所示：

    ```cpp
    #include "TOY.h"
    #include "llvm/IR/Module.h"
    #include "llvm/Support/TargetRegistry.h"
    using namespace llvm;

    Target llvm::TheTOYTarget;

    extern "C" void LLVMInitializeTOYTargetInfo() {
      RegisterTarget<Triple::toy> X(TheTOYTarget, "toy", "TOY");
    }
    ```

1.  在同一文件夹中，创建`CMakeLists.txt`文件：

    ```cpp
    add_llvm_library(LLVMTOYInfo
      TOYTargetInfo.cpp
      )
    ```

1.  创建一个`LLVMBuild.txt`文件：

    ```cpp
    [component_0]
    type = Library
    name = TOYInfo
    parent = TOY
    required_libraries = Support
    add_to_library_groups = TOY
    ```

1.  在`lib/Target/TOY`文件夹中，创建一个名为`TOYTargetMachine.cpp`的文件：

    ```cpp
    #include "TOYTargetMachine.h"
    #include "TOY.h"
    #include "TOYFrameLowering.h"
    #include "TOYInstrInfo.h"
    #include TOYISelLowering.h"
    #include "TOYSelectionDAGInfo.h"
    #include "llvm/CodeGen/Passes.h"
    #include "llvm/IR/Module.h"
    #include "llvm/PassManager.h"
    #include "llvm/Support/TargetRegistry.h"
    using namespace llvm;

    TOYTargetMachine::TOYTargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS, const TargetOptions &Options,
    Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
        : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
          Subtarget(TT, CPU, FS, *this) {
      initAsmInfo();
    }

    namespace {
    class TOYPassConfig : public TargetPassConfig {
    public:
      TOYPassConfig(TOYTargetMachine *TM, PassManagerBase &PM)
          : TargetPassConfig(TM, PM) {}

      TOYTargetMachine &getTOYTargetMachine() const {
        return getTM<TOYTargetMachine>();
      }

      virtual bool addPreISel();
      virtual bool addInstSelector();
      virtual bool addPreEmitPass();
    };
    } // namespace

    TargetPassConfig *TOYTargetMachine::createPassConfig(PassManagerBase &PM) {
      return new TOYPassConfig(this, PM);
    }

    bool TOYPassConfig::addPreISel() { return false; }

    bool TOYPassConfig::addInstSelector() {
      addPass(createTOYISelDag(getTOYTargetMachine(), getOptLevel()));
      return false;
    }

    bool TOYPassConfig::addPreEmitPass() { return false; }

    // Force static initialization.
    extern "C" void LLVMInitializeTOYTarget() {
      RegisterTargetMachine<TOYTargetMachine> X(TheTOYTarget);
    }

    void TOYTargetMachine::addAnalysisPasses(PassManagerBase &PM) {}
    ```

1.  创建一个名为`MCTargetDesc`的新文件夹和一个名为`TOYMCTargetDesc.h`的新文件：

    ```cpp
    #ifndef TOYMCTARGETDESC_H
    #define TOYMCTARGETDESC_H

    #include "llvm/Support/DataTypes.h"

    namespace llvm {
    class Target;
    class MCInstrInfo;
    class MCRegisterInfo;
    class MCSubtargetInfo;
    class MCContext;
    class MCCodeEmitter;
    class MCAsmInfo;
    class MCCodeGenInfo;
    class MCInstPrinter;
    class MCObjectWriter;
    class MCAsmBackend;

    class StringRef;
    class raw_ostream;

    extern Target TheTOYTarget;

    MCCodeEmitter *createTOYMCCodeEmitter(const MCInstrInfo &MCII, const MCRegisterInfo &MRI, const MCSubtargetInfo &STI, MCContext &Ctx);

    MCAsmBackend *createTOYAsmBackend(const Target &T, const MCRegisterInfo &MRI, StringRef TT, StringRef   CPU);

    MCObjectWriter *createTOYELFObjectWriter(raw_ostream &OS, uint8_t OSABI);

    } // End llvm namespace

    #define GET_REGINFO_ENUM
    #include "TOYGenRegisterInfo.inc"

    #define GET_INSTRINFO_ENUM
    #include "TOYGenInstrInfo.inc"

    #define GET_SUBTARGETINFO_ENUM
    #include "TOYGenSubtargetInfo.inc"

    #endif
    ```

1.  在同一文件夹中创建一个名为`TOYMCTargetDesc.cpp`的文件：

    ```cpp
    #include "TOYMCTargetDesc.h"
    #include "InstPrinter/TOYInstPrinter.h"
    #include "TOYMCAsmInfo.h"
    #include "llvm/MC/MCCodeGenInfo.h"
    #include "llvm/MC/MCInstrInfo.h"
    #include "llvm/MC/MCRegisterInfo.h"
    #include "llvm/MC/MCSubtargetInfo.h"
    #include "llvm/MC/MCStreamer.h"
    #include "llvm/Support/ErrorHandling.h"
    #include "llvm/Support/FormattedStream.h"
    #include "llvm/Support/TargetRegistry.h"

    #define GET_INSTRINFO_MC_DESC
    #include "TOYGenInstrInfo.inc"

    #define GET_SUBTARGETINFO_MC_DESC
    #include "TOYGenSubtargetInfo.inc"

    #define GET_REGINFO_MC_DESC
    #include "TOYGenRegisterInfo.inc"

    using namespace llvm;

    static MCInstrInfo *createTOYMCInstrInfo() {
      MCInstrInfo *X = new MCInstrInfo();
      InitTOYMCInstrInfo(X);
      return X;
    }

    static MCRegisterInfo *createTOYMCRegisterInfo(StringRef TT) {
      MCRegisterInfo *X = new MCRegisterInfo();
      InitTOYMCRegisterInfo(X, TOY::LR);
      return X;
    }

    static MCSubtargetInfo *createTOYMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                     StringRef FS) {
      MCSubtargetInfo *X = new MCSubtargetInfo();
      InitTOYMCSubtargetInfo(X, TT, CPU, FS);
      return X;
    }

    static MCAsmInfo *createTOYMCAsmInfo(const MCRegisterInfo &MRI, StringRef TT) {
      MCAsmInfo *MAI = new TOYMCAsmInfo(TT);
      return MAI;
    }

    static MCCodeGenInfo *createTOYMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                 CodeModel::Model CM,
                                                 CodeGenOpt::Level OL) {
      MCCodeGenInfo *X = new MCCodeGenInfo();
      if (RM == Reloc::Default) {
        RM = Reloc::Static;
      }
      if (CM == CodeModel::Default) {
        CM = CodeModel::Small;
      }
      if (CM != CodeModel::Small && CM != CodeModel::Large) {
        report_fatal_error("Target only supports CodeModel Small or Large");
      }

      X->InitMCCodeGenInfo(RM, CM, OL);
      return X;
    }

    static MCInstPrinter *
    createTOYMCInstPrinter(const Target &T, unsigned SyntaxVariant,
                           const MCAsmInfo &MAI, const MCInstrInfo &MII,
                           const MCRegisterInfo &MRI, const MCSubtargetInfo &STI) {
      return new TOYInstPrinter(MAI, MII, MRI);
    }

    static MCStreamer *
    createMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS, bool isVerboseAsm, bool useDwarfDirectory,MCInstPrinter *InstPrint, MCCodeEmitter *CE,MCAsmBackend *TAB, bool ShowInst) {
      return createAsmStreamer(Ctx, OS, isVerboseAsm,   useDwarfDirectory, InstPrint,  CE,   TAB, ShowInst);
    }

    static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                        MCContext &Ctx, MCAsmBackend &MAB,
                                        raw_ostream &OS,
                                        MCCodeEmitter *Emitter,
                                        const MCSubtargetInfo &STI,
                                        bool RelaxAll,
                                        bool NoExecStack) {
      return createELFStreamer(Ctx, MAB, OS, Emitter, false, NoExecStack);
    }

    // Force static initialization.
    extern "C" void LLVMInitializeTOYTargetMC() {
      // Register the MC asm info.
      RegisterMCAsmInfoFn X(TheTOYTarget, createTOYMCAsmInfo);

      // Register the MC codegen info.
      TargetRegistry::RegisterMCCodeGenInfo(TheTOYTarget, createTOYMCCodeGenInfo);

      // Register the MC instruction info.
      TargetRegistry::RegisterMCInstrInfo(TheTOYTarget, createTOYMCInstrInfo);

      // Register the MC register info.
      TargetRegistry::RegisterMCRegInfo(TheTOYTarget, createTOYMCRegisterInfo);

      // Register the MC subtarget info.
      TargetRegistry::RegisterMCSubtargetInfo(TheTOYTarget,
                                              createTOYMCSubtargetInfo);

      // Register the MCInstPrinter
      TargetRegistry::RegisterMCInstPrinter(TheTOYTarget, createTOYMCInstPrinter);

      // Register the ASM Backend.   TargetRegistry::RegisterMCAsmBackend(TheTOYTarget, createTOYAsmBackend);

      // Register the assembly streamer.
      TargetRegistry::RegisterAsmStreamer(TheTOYTarget, createMCAsmStreamer);

      // Register the object streamer.
      TargetRegistry::RegisterMCObjectStreamer(TheTOYTarget, createMCStreamer);

      // Register the MCCodeEmitter
      TargetRegistry::RegisterMCCodeEmitter(TheTOYTarget, createTOYMCCodeEmitter);
    }
    ```

1.  在同一文件夹中，创建一个`LLVMBuild.txt`文件：

    ```cpp
    [component_0]
    type = Library
    name = TOYDesc
    parent = TOY
    required_libraries = MC Support TOYAsmPrinter TOYInfo
    add_to_library_groups = TOY
    ```

1.  创建一个`CMakeLists.txt`文件：

    ```cpp
    add_llvm_library(LLVMTOYDesc
      TOYMCTargetDesc.cpp)
    ```

## 它是如何工作的…

按照以下方式构建整个 LLVM 项目：

```cpp
$ cmake llvm_src_dir –DCMAKE_BUILD_TYPE=Release – DLLVM_TARGETS_TO_BUILD="TOY"
$ make

```

在这里，我们指定我们正在为 toy 目标构建 LLVM 编译器。构建完成后，检查是否可以通过`llc`命令看到 TOY 目标：

```cpp
$ llc –version
…
…
Registered Targets :
toy – TOY

```

## 参考也

+   对于关于涉及流水线和调度的复杂目标的更详细描述，请参考陈中舒和 Anoushe Jamshidi 所著的*教程：为 Cpu0 架构创建 LLVM 后端*中的章节。
