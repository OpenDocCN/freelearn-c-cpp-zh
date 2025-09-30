# 第七章 生成目标架构代码

编译器生成的代码最终必须在目标机器上执行。LLVM IR 的抽象形式有助于为各种架构生成代码。目标机器可以是任何东西——CPU、GPU、DSP 等。目标机器有一些定义性的方面，如寄存器集、指令集、函数的调用约定和指令流水线。这些方面或属性是通过**tablegen**工具生成的，以便在编写机器代码生成程序时易于使用。

LLVM 后端有一个流水线结构，其中指令通过阶段从 LLVM IR 到**SelectionDAG**，然后到**MachineDAG**，然后到**MachineInstr**，最后到**MCInst**。IR 被转换为 SelectionDAG。然后 SelectionDAG 经过合法化和优化。在此阶段之后，DAG 节点被映射到目标指令（指令选择）。然后 DAG 经过指令调度，生成指令的线性序列。虚拟寄存器随后被分配到目标机器寄存器，这涉及到最优化的寄存器分配以最小化内存溢出。

本章描述了如何表示目标架构。它还描述了如何生成汇编代码。

本章讨论的主题如下：

+   定义寄存器和寄存器集

+   定义调用约定

+   定义指令集

+   实现帧降低

+   选择指令

+   打印指令

+   注册目标

# 示例后端

为了理解目标代码生成，我们定义了一个简单的 RISC 型架构 TOY 机器，具有最少的寄存器，例如`r0-r3`，一个栈指针`SP`，一个链接寄存器`LR`（用于存储返回地址）；以及一个`CPSR`——当前状态程序寄存器。这个玩具后端的调用约定类似于 ARM thumb-like 架构——传递给函数的参数将存储在寄存器集`r0-r1`中，返回值将存储在`r0`中。

## 定义寄存器和寄存器集

寄存器集是通过 tablegen 工具定义的。Tablegen 有助于维护大量特定领域的信息记录。它提取了这些记录的共同特征。这有助于减少描述中的重复，并形成表示领域信息的一种结构化方式。请访问[`llvm.org/docs/TableGen/`](http://llvm.org/docs/TableGen/)以详细了解 tablegen。`TableGen`文件由`TableGen 二进制：llvm-tblgen`解释。

我们在前一段落中描述了我们的示例后端，它有四个寄存器（`r0-r3`），一个栈寄存器（`SP`）和一个链接寄存器（`LR`）。这些可以在`TOYRegisterInfo.td`文件中指定。`tablegen`函数提供了`Register`类，可以扩展以指定寄存器。创建一个名为`TOYRegisterInfo.td`的新文件。

寄存器可以通过扩展 `Register` 类来定义。

```cpp
class TOYReg<bits<16> Enc, string n> : Register<n> {
let HWEncoding = Enc;
let Namespace = "TOY";
}
```

寄存器 `r0-r3` 属于通用 `Register` 类。这可以通过扩展 `RegisterClass` 来指定。

```cpp
foreach i = 0-3 in {
def R#i : R<i, "r"#i >;
}

def GRRegs : RegisterClass<"TOY", [i32], 32,
(add R0, R1, R2, R3, SP)>;
```

剩余的，寄存器 `SP`、`LR` 和 `CPSR` 可以如下定义：

```cpp
def SP : TOYReg<13, "sp">;
def LR : TOYReg<14, "lr">;
def CPSR  : TOYReg<16, "cpsr">;
```

当所有这些放在一起时，`TOYRegisterInfo.td` 看起来如下所示：

```cpp
class TOYReg<bits<16> Enc, string n> : Register<n> {
let HWEncoding = Enc;
let Namespace = "TOY";
}

foreach i = 0-3 in {
def R#i : R<i, "r"#i >;
}

def SP : TOYReg<13, "sp">;
def LR : TOYReg<14, "lr">;
def GRRegs : RegisterClass<"TOY", [i32], 32,
(add R0, R1, R2, R3, SP)>;
```

我们可以将此文件放在名为 `TOY` 的新文件夹中，该文件夹位于名为 `Target` 的父文件夹中，位于 llvm 的根目录下，即 `llvm_root_directory/lib/Target/TOY/ TOYRegisterInfo.td`。

表生成工具 `llvm-tablegen` 处理这个 `.td` 文件以生成 `.inc` 文件，该文件通常为这些寄存器生成枚举。这些枚举可以在 `.cpp` 文件中使用，其中寄存器可以引用为 `TOY::R0`。

## 定义调用约定

调用约定指定了值如何传递到和从函数调用返回。我们的 `TOY` 架构指定两个参数通过两个寄存器 `r0` 和 `r1` 传递，其余的传递到栈上。定义的调用约定随后通过引用函数指针在指令选择阶段使用。

在定义调用约定时，我们必须表示两个部分——一个用于定义约定返回值，另一个用于定义参数传递的约定。父类 `CallingConv` 被继承以定义调用约定。

在我们的 `TOY` 架构中，返回值存储在 `r0` 寄存器中。如果有更多参数，整数值将存储在大小为 4 字节且 4 字节对齐的栈槽中。这可以在 `TOYCallingConv.td` 中如下声明：

```cpp
def RetCC_TOY : CallingConv<[
CCIfType<[i32], CCAssignToReg<[R0]>>,
CCIfType<[i32], CCAssignToStack<4, 4>>
]>;
```

参数传递约定可以定义为以下内容：

```cpp
def CC_TOY : CallingConv<[
CCIfType<[i8, i16], CCPromoteToType<i32>>,
CCIfType<[i32], CCAssignToReg<[R0, R1]>>,
CCIfType<[i32], CCAssignToStack<4, 4>>
]>;
```

前面的声明说明了以下三个内容：

+   如果参数的数据类型是 `i8` 或 `i16`，它将被提升为 `i32`

+   前两个参数将存储在寄存器 `r0` 和 `r1` 中

+   如果有更多参数，它们将存储在 `Stack`

我们还定义了调用者保留寄存器，因为调用者保留寄存器用于存储应在调用之间保留的长生存期值。

```cpp
def CC_Save : CalleeSavedRegs<(add R2, R3)>;
```

在构建项目后，`llvm-tablegen` 工具生成一个 `TOYCallingConv.inc` 文件，该文件将在 `TOYISelLowering.cpp` 文件中的指令选择阶段被包含。

## 定义指令集

架构具有丰富的指令集来表示目标机器支持的各项操作。在表示指令时，通常需要在目标描述文件中定义以下三个内容：

+   操作数

+   汇编字符串

+   指令模式

规范包含一个定义或输出的列表，以及一个使用或输入的列表。可以有不同类型的操作数类，例如 `Register` 类，以及立即数和更复杂的 `register+imm` 操作数。

例如，我们可以在 `TOYInstrInfo.td` 中如下定义我们的玩具机器的寄存器到寄存器的加法：

```cpp
def ADDrr : InstTOY<(outs GRRegs:$dst),
(ins GRRegs:$src1, GRRegs:$src2),
"add $dst, $src1,z$src2",
[(set i32:$dst, (add i32:$src1, i32:$src2))]>;
```

在上述声明中，`'ins'` 有两个属于通用寄存器类的寄存器 `$src1` 和 `$src2`，它们持有两个操作数。操作的结果将被放入 `'outs'`，这是一个属于通用 `Register` 类的 `$dst` 寄存器。汇编字符串是 "`add $dst, $src1,z$src2`"。`$src1`、`$src2` 和 `$dst` 的值将在寄存器分配时确定。因此，将生成两个寄存器之间 `add` 操作的汇编，如下所示：

```cpp
add r0, r0, r1
```

我们在上面看到，一个简单的指令可以使用 tablegen 来表示。类似于 `add register to register` 指令，可以定义一个 `subtract register from a register` 指令。我们留给读者去尝试。更详细地表示复杂指令可以从项目代码中的 ARM 或 X86 架构规范中找到。

# 实现帧降低

帧降低涉及发出函数的前置和后置代码。前置代码发生在函数的开始处，它设置了被调用函数的栈帧。后置代码在函数的最后执行，它恢复调用（父）函数的栈帧。

在程序执行过程中，"`栈`" 扮演着几个角色，如下所述：

+   在调用函数时跟踪返回地址

+   在函数调用上下文中存储局部变量

+   从调用者传递参数给被调用者。

因此，在实现帧降低时，需要定义两个主要功能 - `emitPrologue()` 和 `emitEpilogue()`。

`emitPrologue()` 函数可以定义为以下内容：

```cpp
void TOYFrameLowering::emitPrologue(MachineFunction &MF) const {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();

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

上面的函数遍历 **Machine Basic Block**。它为函数计算栈大小，计算栈大小的偏移量，并发出使用栈寄存器设置帧的指令。

同样，`emitEpilogue()` 函数可以定义为以下内容：

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

前面的函数还计算栈大小，遍历机器基本块，并在函数返回时设置函数帧。请注意，这里的栈是递减的。

`emitPrologue()` 函数首先计算栈大小以确定是否需要前置代码。然后它通过计算偏移量来调整栈指针。对于 `emitEpilogue()`，它首先检查是否需要后置代码。然后它将栈指针恢复到函数开始时的状态。

例如，考虑这个输入 IR：

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

# 降低指令

在本章中，我们将看到三个方面的实现 - 函数调用约定、形式参数调用约定和返回值调用约定。我们创建一个文件 `TOYISelLowering.cpp`，并在其中实现指令降低。

首先，让我们看看如何实现调用约定。

```cpp
SDValue TOYTar-getLoweing::LowerCall(TargetLowering::CallLoweringInfo &CLI, SmallVectorImpl<SDValue> &InVals)
 const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &Loc = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  CallingConv::ID CallConv = CLI.CallConv;
  const bool isVarArg = CLI.IsVarArg;

  CLI.IsTailCall = false;

  if (isVarArg) {
    llvm_unreachable("Unimplemented");
  }

  // Analyze operands of the call, assigning locations to each
  // operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CC_TOY);

  // Get the size of the outgoing arguments stack space
  // requirement.
  const unsigned NumBytes = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getIntPtrConstant(NumBytes, Loc, true), Loc);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];

    // We only handle fully promoted arguments.
    assert(VA.getLocInfo() == CCValAssign::Full && "Unhandled loc 
    info");

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      continue;
    }

    assert(VA.isMemLoc() &&
           "Only support passing arguments through registers or 
           via the stack");

    SDValue StackPtr = DAG.getRegister(TOY::SP, MVT::i32);
    SDValue PtrOff = DAG.getIntPtrConstant(VA.getLocMemOffset(), 
    Loc);
    PtrOff = DAG.getNode(ISD::ADD, Loc, MVT::i32, StackPtr, 
    PtrOff);
    MemOpChains.push_back(DAG.getStore(Chain, Loc, Arg, PtrOff,
                                       MachinePointerInfo(), false, false, 0));
  }

  // Emit all stores, make sure they occur before the call.
  if (!MemOpChains.empty()) {
    Chain = DAG.getNode(ISD::TokenFactor, Loc, MVT::Other, MemOpChains);
  }

  // Build a sequence of copy-to-reg nodes chained together with
  // token chain
  // and flag operands which copy the outgoing args into the
  // appropriate regs.
  SDValue InFlag;
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, Loc, Reg.first, Reg.second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // We only support calling global addresses.
  GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee);
  assert(G && "We only support the calling of global address-es");

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  Callee = DAG.getGlobalAddress(G->getGlobal(), Loc, PtrVT, 0);

  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they
  // are known live into the call.
  for (auto &Reg : RegsToPass) {
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));
  }

  // Add a register mask operand representing the call-preserved
  // registers.
  const uint32_t *Mask;
  const TargetRegisterInfo *TRI = DAG.getSubtarget().getRegisterInfo();
  Mask = TRI->getCallPreservedMask(DAG.getMachineFunction(), CallConv);

  assert(Mask && "Missing call preserved mask for calling 
  convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode()) {
    Ops.push_back(InFlag);
  }

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // Returns a chain and a flag for retval copy to use.
  Chain = DAG.getNode(TOYISD::CALL, Loc, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, Loc, true),
                             DAG.getIntPtrConstant(0, Loc, true), InFlag, Loc);
  if (!Ins.empty()) {
    InFlag = Chain.getValue(1);
  }

  // Handle result values, copying them out of physregs into vregs 
  // that we return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, 
                         Loc, DAG, InVals);
}
```

在上述函数中，我们首先分析了调用的操作数，为每个操作数分配位置，并计算了参数栈空间的大小。然后，我们扫描`register/memloc`分配，并插入`copies`和`loads`。对于我们的示例目标，我们支持通过寄存器或通过栈传递参数（记住上一节中定义的调用约定）。然后，我们发出所有存储操作，确保它们在调用之前发生。我们构建一系列`copy-to-reg`节点，将输出参数复制到适当的寄存器中。然后，我们添加一个表示调用保留寄存器的寄存器掩码操作数。我们返回一个链和标志，用于返回值复制，并最终处理结果值，将它们从`physregs`复制到我们返回的`vregs`中。

我们现在将查看正式参数调用约定的实现。

```cpp
SDValue TOYTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc dl, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  assert(!isVarArg && "VarArg not supported");

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CC_TOY);

  for (auto &VA : ArgLocs) {
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      EVT RegVT = VA.getLocVT();
      assert(RegVT.getSimpleVT().SimpleTy == MVT::i32 &&
             "Only support MVT::i32 register passing");
      const unsigned VReg =
          RegInfo.createVirtualRegister(&TOY::GRRegsRegClass);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);
      SDValue ArgIn = DAG.getCopyFromReg(Chain, dl, VReg, RegVT);

      InVals.push_back(ArgIn);
      continue;
    }

    assert(VA.isMemLoc() &&
           "Can only pass arguments as either registers or via the 
           stack");

    const unsigned Offset = VA.getLocMemOffset();

    const int FI = MF.getFrameInfo()->CreateFixedObject(4, Offset, 
    true);
    EVT PtrTy = getPointerTy(DAG.getDataLayout());
    SDValue FIPtr = DAG.getFrameIndex(FI, PtrTy);

    assert(VA.getValVT() == MVT::i32 &&
           "Only support passing arguments as i32");
    SDValue Load = DAG.getLoad(VA.getValVT(), dl, Chain, FIPtr,
                               MachinePointerInfo(), false, false, false, 0);

    InVals.push_back(Load);
  }
  return Chain;
}
```

在上述正式参数调用约定的实现中，我们为所有传入的参数分配了位置。我们只处理通过寄存器或栈传递的参数。我们现在将查看返回值调用约定的实现。

```cpp
bool TOYTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, RVLocs, Context);
  if (!CCInfo.CheckReturn(Outs, RetCC_TOY)) {
    return false;
  }
  if (CCInfo.getNextStackOffset() != 0 && isVarArg) {
    return false;
  }
  return true;
}

SDValue
TOYTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg, const SmallVec torImpl<ISD::OutputArg> & Outs, const SmallVectorImpl<SDValue> const SmallVec torImpl<ISD::OutputArg> & Outs,
  if (isVarArg) {
    report_fatal_error("VarArg not supported");
  }

  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeReturn(Outs, RetCC_TOY);

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), OutVals[i], Flag);

    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode()) {
    RetOps.push_back(Flag);
  }

  return DAG.getNode(TOYISD::RET_FLAG, dl, MVT::Other, RetOps);
}
```

我们首先检查是否可以降低返回值。然后收集有关寄存器和栈槽位的信息。我们将结果值复制到输出寄存器中，并最终返回一个表示返回值的 DAG 节点。

# 打印指令

打印汇编指令是生成目标代码的重要步骤。定义了各种类，它们作为流式传输的网关。

首先，我们在`TOYInstrFormats.td`文件中初始化指令类，分配操作数、汇编字符串、模式、输出变量等：

```cpp
class InstTOY<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  field bits<32> Inst;
  let Namespace = "TOY";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let Size = 4;
}
```

然后，我们在`TOYInstPrinter.cpp`中定义了打印操作数的函数。

```cpp
void TOYInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
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

此函数简单地打印操作数、寄存器或立即值，视情况而定。

我们还在同一文件中定义了一个打印寄存器名称的函数：

```cpp
void TOYInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << StringRef(getRegisterName(RegNo)).lower();
}
```

接下来，我们定义了一个打印指令的函数：

```cpp
void TOYInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot) {
  printInstruction(MI, O);
  printAnnotation(O, Annot);
}
```

接下来，我们如下声明和定义汇编信息：

我们创建一个`TOYMCAsmInfo.h`并声明一个`ASMInfo`类：

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

构造函数可以在`TOYMCAsmInfo.cpp`中定义如下：

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

对于编译，我们如下定义`LLVMBuild.txt`:

```cpp
[component_0]
type = Library
name = TOYAsmPrinter
parent = TOY
required_libraries = MC Support
add_to_library_groups = TOY
```

此外，我们定义了`CMakeLists.txt`文件如下：

```cpp
add_llvm_library(LLVMTOYAsmPrinter
TOYInstPrinter.cpp
)
```

当最终编译发生时，`llc`工具（一个静态编译器）将生成`TOY`架构的汇编代码（在将`TOY`架构注册到`llc`工具之后）。

要将我们的`TOY`目标注册到静态编译器`llc`，请遵循以下步骤：

1.  首先，将`TOY`后端条目添加到`llvm_root_dir/CMakeLists.txt`:

    ```cpp
    set(LLVM_ALL_TARGETS
    AArch64
    ARM
    …
    …
    TOY
    )
    ```

1.  然后，将`toy`条目添加到`llvm_root_dir/include/llvm/ADT/Triple.h`:

    ```cpp
    class Triple {
    public:
    enum ArchType {
    UnknownArch,
    arm, // ARM (little endian): arm, armv.*, xscale
    armeb, // ARM (big endian): armeb
    aarch64, // AArch64 (little endian): aarch64
    …
    …
    toy // TOY: toy
    };
    ```

1.  将`toy`条目添加到`llvm_root_dir/include/llvm/MC/MCExpr.h`:

    ```cpp
    class MCSymbolRefExpr : public MCExpr {
    public:
    enum VariantKind {
    ...
    VK_TOY_LO,
    VK_TOY_HI,
    };
    ```

1.  将`toy`条目添加到`llvm_root_dir/include/llvm/Support/ELF.h`:

    ```cpp
    enum {
    EM_NONE = 0, // No machine
    EM_M32 = 1, // AT&T WE 32100
    …
    …
    EM_TOY = 220 // whatever is the next number
    };
    ```

1.  然后，将`toy`条目添加到`lib/MC/MCExpr.cpp`:

    ```cpp
    StringRef MCSymbolRefExpr::getVariantKindName(VariantKind
    Kind) {
    switch (Kind) {
    …
    …
    case VK_TOY_LO: return "TOY_LO";
    case VK_TOY_HI: return "TOY_HI";
    }
    …
    }
    ```

1.  接下来，将`toy`条目添加到`lib/Support/Triple.cpp`:

    ```cpp
    const char *Triple::getArchTypeName(ArchType Kind) {
    switch (Kind) {
    …
    …
    case toy: return "toy";
    }
    const char *Triple::getArchTypePrefix(ArchType Kind) {
    switch (Kind) {
    …
    …
    case toy: return "toy";
    }
    }
    Triple::ArchType Triple::getArchTypeForLLVMName(StringRef
    Name) {
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
    static unsigned
    getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
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

1.  将`toy`目录条目添加到`lib/Target/LLVMBuild.txt`:

    ```cpp
    [common]
    subdirectories = ARM AArch64 CppBackend Hexagon MSP430 … …
    TOY
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

1.  在同一文件夹中创建`CMakeLists.txt`文件：

    ```cpp
    add_llvm_library(LLVMTOYInfo TOYTargetInfo.cpp)
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

1.  在`lib/Target/TOY`文件夹中创建一个名为`TOYTargetMachine.cpp`的文件：

    ```cpp
    #include "TOYTargetMachine.h"
    #include "TOY.h"
    #include "TOYFrameLowering.h"
    #include "TOYInstrInfo.h"
    #include "TOYISelLowering.h "
    #include "TOYSelectionDAGInfo.h"
    #include "llvm/CodeGen/Passes.h"
    #include "llvm/IR/Module.h"
    #include "llvm/PassManager.h"
    #include "llvm/Support/TargetRegistry.h"
    using namespace llvm;

    TOYTargetMachine::TOYTargetMachine(const Target &T, StringRef TT, StringRef CPU, StringRef FS, const 
    TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM, CodeGenOpt::Level OL)
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

    TargetPassConfig *TOYTargetMachine::createPassConfig
    (PassManagerBase &PM) {
      return new TOYPassConfig(this, PM);
    }

    bool TOYPassConfig::addPreISel() { return false; }

    bool TOYPassConfig::addInstSelector() {
      addPass(createTOYISelDag(getTOYTargetMachine(), 
    getOptLevel()));
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

    MCAsmBackend *createTOYAsmBackend(const Target &T, const MCRegisterInfo &MRI, StringRef TT, StringRef CPU);

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

1.  在同一文件夹中再创建一个名为`TOYMCTargetDesc.cpp`的文件：

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

    static MCSubtargetInfo *createTOYMCSubtargetInfo(StringRef TT, StringRef CPU, StringRef FS) {
      MCSubtargetInfo *X = new MCSubtargetInfo();
      InitTOYMCSubtargetInfo(X, TT, CPU, FS);
      return X;
    }

    static MCAsmInfo *createTOYMCAsmInfo(const MCRegisterInfo &MRI, StringRef TT) {
      MCAsmInfo *MAI = new TOYMCAsmInfo(TT);
      return MAI;
    }
    static MCCodeGenInfo *createTOYMCCodeGenInfo(StringRef TT, Reloc::Model RM, CodeModel::Model CM, CodeGenOpt::Level OL)
     {
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
                           const MCAsmInfo &MAI, const MCInstrInfo & MII, const MCRegisterInfo &MRI, const MCSubtargetInfo &STI) {
      return new TOYInstPrinter(MAI, MII, MRI);
    }

    static MCStreamer *
    createMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                        bool isVerboseAsm, bool useDwarfDirectory,
                        MCInstPrinter *InstPrint, MCCodeEmitter *CE,
                        MCAsmBackend *TAB, bool ShowInst) {
      return createAsmStreamer(Ctx, OS, isVerboseAsm, useD - warfDirectory, InstPrint, CE, TAB, ShowInst);
    }

    static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
    MCContext &Ctx, MCAsmBackend &MAB, raw_ostream &OS,
    MCCodeEmitter *Emitter, const MCSubtargetInfo &STI,
    bool RelaxAll, bool NoExecStack) {
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
                                              createTOYMCSub targetInfo);
      // Register the MCInstPrinter
      TargetRegistry::RegisterMCInstPrinter(TheTOYTarget, createTOYMCInstPrinter);
      // Register the ASM Backend.
      TargetRegistry::RegisterMCAsmBackend(TheTOYTarget, createTOYAsmBackend);
      // Register the assembly streamer.
      TargetRegistry::RegisterAsmStreamer(TheTOYTarget, createMCAsmStreamer);
      // Register the object streamer.
      TargetRegistry::RegisterMCObjectStreamer(TheTOYTarget, createMCStreamer);
      // Register the MCCodeEmitter
      TargetRegistry::RegisterMCCodeEmitter(TheTOYTarget, createTOYMCCodeEmitter);
    }
    ```

1.  在同一文件夹中创建一个`LLVMBuild.txt`文件：

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

    按如下方式构建整个 LLVM 项目：

    ```cpp
    $ cmake llvm_src_dir –DCMAKE_BUILD_TYPE=Release –
    DLLVM_TARGETS_TO_BUILD="TOY"
    $ make

    Here, we have specified that we are building the LLVM compiler for the toy target. After the build completes, check whether the TOY target appears with the llc command:
    $ llc –version
    …
    …
    Registered Targets :
    toy – TOY
    ```

以下 IR，当提供给`llc`工具时，将生成如下的汇编：

```cpp
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32- i16:16:32-i64:32-f64:32-a:0:32-n32"
target triple = "toy"
define i32 @foo(i32 %a, i32 %b){
  %c = add nsw i32 %a, %b
  ret i32 %c
}

$ llc foo.ll

.text
.file "foo.ll"
.globl foo
.type foo,@function
foo: # @foo
# BB#0: # %entry
add r0, r0, r1
b lr
.Ltmp0:
.size foo, .Ltmp0-foo
```

要查看如何使用`llc`注册目标的详细信息，您可以访问[`llvm.org/docs/WritingAnLLVMBackend.html#target-registration`](http://llvm.org/docs/WritingAnLLVMBackend.html#target-registration)和[`jonathan2251.github.io/lbd/llvmstructure.html#target-registration`](http://jonathan2251.github.io/lbd/llvmstructure.html#target-registration)由陈中舒和 Anoushe Jamshidi 编写。

# 摘要

在本章中，我们简要讨论了如何在 LLVM 中表示目标架构机器。我们看到了使用 tablegen 组织数据（如寄存器集、指令集、调用约定等）的便捷性，对于给定的目标。然后`llvm-tablegen`将这些目标描述`.td`文件转换为枚举，这些枚举可以在程序逻辑（如帧降低、指令选择、指令打印等）中使用。更详细和复杂的架构，如 ARM 和 X86，可以提供对目标详细描述的见解。

在第一章中，我们尝试了一个基本练习，以熟悉 LLVM 基础设施提供的各种工具。在随后的章节中，即第二章，*构建 LLVM IR*和第三章，*高级 LLVM IR*中，我们使用了 LLVM 提供的 API 来生成 IR。读者可以在他们的前端使用这些 API 将他们的语言转换为 LLVM IR。在第五章，*高级 IR 块变换*中，我们习惯了 IR 优化的 Pass Pipeline，并经历了一些示例。在第六章，*IR 到选择 DAG 阶段*中，读者熟悉了将 IR 转换为选择 DAG 的过程，这是生成机器代码的一个步骤。在本章的最后，我们看到了如何使用 tablegen 表示示例架构并用于生成代码。

阅读完这本书后，我们希望读者能够熟悉 LLVM 基础设施，并准备好深入探索 LLVM，为自己的定制架构或定制语言创建编译器。编译愉快！
