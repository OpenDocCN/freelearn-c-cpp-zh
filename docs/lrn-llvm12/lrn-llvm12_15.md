# 第十二章：创建您自己的后端

LLVM 具有非常灵活的架构。您也可以向其添加新的目标后端。后端的核心是目标描述，其中大部分代码都是由此生成的。但是，目前还无法生成完整的后端，并且实现调用约定需要手动编写代码。在本章中，我们将学习如何为历史 CPU 添加支持。

在本章中，我们将涵盖以下内容：

+   为新后端做准备，让您了解 M88k CPU 架构，并指导您在何处找到所需的信息。

+   将新架构添加到 Triple 类将教会您如何使 LLVM 意识到新的 CPU 架构。

+   在 LLVM 中扩展 ELF 文件格式定义，您将为处理 ELD 对象文件的库和工具添加对 M88k 特定重定位的支持。

+   在创建目标描述中，您将使用 TableGen 语言开发目标描述的所有部分。

+   在实现 DAG 指令选择类中，您将创建所需的指令选择的传递和支持类。

+   生成汇编指令教会您如何实现汇编打印机，负责生成文本汇编程序。

+   在发出机器代码中，您将了解必须提供哪些额外的类来使**机器代码**（**MC**）层能够向目标文件写入代码。

+   在添加反汇编支持中，您将学习如何实现反汇编器的支持。

+   在将所有内容组合在一起中，您将把新后端的源代码集成到构建系统中。

通过本章的学习，您将了解如何开发一个新的完整后端。您将了解后端由不同部分组成，从而更深入地了解 LLVM 架构。

# 技术要求

该章节的代码文件可在[`github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter12`](https://github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter12)找到

您可以在[`bit.ly/3nllhED`](https://bit.ly/3nllhED)找到代码演示视频

# 为新后端做准备

无论是出于商业需要支持新的 CPU，还是仅仅是为了为一些旧的架构添加支持而进行的爱好项目，向 LLVM 添加新的后端都是一项重大任务。以下各节概述了您需要开发新后端的内容。我们将为 20 世纪 80 年代的 RISC 架构 Motorola M88k 添加一个后端。

参考资料

您可以在维基百科上阅读更多关于该架构的信息：[`en.wikipedia.org/wiki/Motorola_88000`](https://en.wikipedia.org/wiki/Motorola_88000)。有关该架构的重要信息仍然可以在互联网上找到。您可以在[`www.bitsavers.org/components/motorola/88000/`](http://www.bitsavers.org/components/motorola/88000/)找到包含指令集和时序信息的 CPU 手册，以及包含 ELF 格式和调用约定定义的 System V ABI M88k 处理器补充的信息。

OpenBSD，可在[`www.openbsd.org/`](https://www.openbsd.org/)找到，仍然支持 LUNA-88k 系统。在 OpenBSD 系统上，很容易为 M88k 创建 GCC 交叉编译器。并且有一个名为 GXemul 的仿真器，可运行 M88k 架构的某些 OpenBSD 版本，该仿真器可在 http://gavare.se/gxemul/找到。

总的来说，M88k 架构已经停产很久了，但我们找到了足够的信息和工具，使其成为向 LLVM 添加后端的有趣目标。我们将从一个非常基本的任务开始，并扩展`Triple`类。

# 将新架构添加到 Triple 类

`Triple`类的一个实例代表 LLVM 正在为其生成代码的目标平台。为了支持新的架构，第一步是扩展`Triple`类。在`llvm/include/llvm/ADT/Triple.h`文件中，您需要向`ArchType`枚举添加一个成员和一个新的谓词：

```cpp
class Triple {
public:
  enum ArchType {
  // Many more members
    m88k,           // M88000 (big endian): m88k
  };
  /// Tests whether the target is M88k.
  bool isM88k() const {
    return getArch() == Triple::m88k;
  }
// Many more methods
};
```

在`llvm/lib/Support/Triple.cpp`文件中，有许多使用`ArchType`枚举的方法。您需要扩展所有这些方法；例如，在`getArchTypeName()`方法中，您需要添加一个新的 case 语句：

```cpp
  switch (Kind) {
// Many more cases
  case m88k:           return "m88k";
  }
```

在大多数情况下，如果您忘记处理一个函数中的新的`m88k`枚举成员，编译器会警告您。接下来，我们将扩展**可执行和可链接格式**（**ELF**）的定义。

# 扩展 LLVM 中的 ELF 文件格式定义

ELF 文件格式是 LLVM 支持读取和写入的二进制对象文件格式之一。ELF 本身为许多 CPU 架构定义了规范，也有 M88k 架构的定义。我们需要做的就是添加重定位的定义和一些标志。重定位在《第四章》，《对象文件》，《System V ABI M88k Processor》补充书中给出：

1.  我们需要在`llvm/include/llvm/BinaryFormat/ELFRelocs/M88k.def`文件中输入以下内容：

```cpp
#ifndef ELF_RELOC
#error "ELF_RELOC must be defined"
#endif
ELF_RELOC(R_88K_NONE, 0)
ELF_RELOC(R_88K_COPY, 1)
// Many more…
```

1.  我们还需要向`llvm/include/llvm/BinaryFormat/ELF.h`文件添加一些标志，并包括重定位的定义：

```cpp
// M88k Specific e_flags
enum : unsigned {
  EF_88K_NABI = 0x80000000,   // Not ABI compliant
  EF_88K_M88110 = 0x00000004  // File uses 88110-
                              // specific 
                              // features
};
// M88k relocations.
enum {
#include "ELFRelocs/M88k.def"
};
```

代码可以添加到文件的任何位置，但最好保持排序顺序，并在 MIPS 架构的代码之前插入。

1.  我们还需要扩展一些其他方法。在`llvm/include/llvm/Object/ELFObjectFile.h`文件中有一些在枚举成员和字符串之间进行转换的方法。例如，我们必须向`getFileFormatName()`方法添加一个新的 case 语句：

```cpp
  switch (EF.getHeader()->e_ident[ELF::EI_CLASS]) {
// Many more cases
    case ELF::EM_88K:
      return "elf32-m88k";
  }
```

1.  同样地，我们扩展`getArch()`方法。

1.  最后，在`llvm/lib/Object/ELF.cpp`文件中使用重定位定义，在`getELFRelocationTypeName()`方法中：

```cpp
  switch (Machine) {
// Many more cases
  case ELF::EM_88K:
    switch (Type) {
#include "llvm/BinaryFormat/ELFRelocs/M88k.def"
    default:
      break;
    }
    break;
  }
```

1.  要完成支持，您还可以在`llvm/lib/ObjectYAML/ELFYAML.cpp`文件中添加重定位，在映射`ELFYAML::ELF_REL`枚举的方法中。

1.  在这一点上，我们已经完成了对 ELF 文件格式中 m88k 架构的支持。您可以使用`llvm-readobj`工具检查一个 ELF 目标文件，例如，在 OpenBSD 上由交叉编译器创建的。同样地，您可以使用`yaml2obj`工具为 m88k 架构创建一个 ELF 目标文件。

为对象文件格式添加支持是否是强制性的？

将对架构的支持集成到 ELF 文件格式实现中只需要几行代码。如果您为其创建 LLVM 后端的架构使用 ELF 格式，那么您应该采用这种方式。另一方面，为全新的二进制文件格式添加支持本身就是一项复杂的任务。在这种情况下，一个可能的方法是只输出汇编文件，并使用外部汇编器创建目标文件。

有了这些添加，ELF 文件格式的实现现在支持 M88k 架构。在下一节中，我们将为 M88k 架构创建目标描述，描述架构的指令、寄存器、调用约定和其他细节。

# 创建目标描述

**目标描述**是后端实现的核心。在理想的情况下，我们可以从目标描述生成整个后端。这个目标尚未实现，因此我们需要稍后扩展生成的代码。让我们从顶层文件开始剖析目标描述。

## 实现目标描述的顶层文件

我们将我们新后端的文件放入`llvm/lib/Target/M88k`目录。目标描述在`M88k.td`文件中：

1.  在这个文件中，我们首先需要包括 LLVM 预定义的基本目标描述类，然后是我们将在下一节中创建的文件：

```cpp
include "llvm/Target/Target.td"
include "M88kRegisterInfo.td"
include "M88kCallingConv.td"
include "M88kSchedule.td"
include "M88kInstrFormats.td"
include "M88kInstrInfo.td"
```

1.  接下来，我们还定义了支持的处理器。除其他事项外，这还转换为`-mcpu=`选项的参数：

```cpp
def : ProcessorModel<"mc88110", M88kSchedModel, []>;
```

1.  所有这些定义都完成后，我们现在可以将我们的目标组合起来。我们定义这些子类，以防需要修改一些默认值。`M88kInstrInfo`类包含有关指令的所有信息：

```cpp
def M88kInstrInfo : InstrInfo;
```

1.  我们为`.s`汇编文件定义了一个解析器，并且我们还声明寄存器名称始终以`%`为前缀：

```cpp
def M88kAsmParser : AsmParser;
def M88kAsmParserVariant : AsmParserVariant {
  let RegisterPrefix = "%";
}
```

1.  接下来，我们为汇编写入器定义一个类，负责编写`.s`汇编文件：

```cpp
def M88kAsmWriter : AsmWriter;
```

1.  最后，所有这些记录都被放在一起来定义目标：

```cpp
def M88k : Target {
  let InstructionSet = M88kInstrInfo;
  let AssemblyParsers  = [M88kAsmParser];
  let AssemblyParserVariants = [M88kAsmParserVariant];
  let AssemblyWriters = [M88kAsmWriter];
  let AllowRegisterRenaming = 1;
}
```

现在顶层文件已经实现，我们创建包含的文件，从下一节开始定义寄存器定义。

## 添加寄存器定义

CPU 架构通常定义一组寄存器。这些寄存器的特性可以有很大的变化。一些架构允许访问子寄存器。例如，x86 架构具有特殊的寄存器名称，用于仅访问寄存器值的一部分。其他架构则不实现这一点。除了通用寄存器、浮点寄存器和矢量寄存器外，架构还可以定义特殊寄存器，例如用于状态代码或浮点运算配置。您需要为 LLVM 定义所有这些信息。

M88k 架构定义了通用寄存器、浮点寄存器和控制寄存器。为了使示例简单，我们只定义通用寄存器。我们首先定义寄存器的超类。寄存器的编码仅使用`5`位，因此我们限制了保存编码的字段。我们还定义，所有生成的 C++代码应该驻留在`M88k`命名空间中：

```cpp
class M88kReg<bits<5> Enc, string n> : Register<n> {
  let HWEncoding{15-5} = 0;
  let HWEncoding{4-0} = Enc;
  let Namespace = "M88k";
}
```

`M88kReg`类用于所有寄存器类型。我们为通用寄存器定义了一个特殊的类：

```cpp
class GRi<bits<5> Enc, string n> : M88kReg<Enc, n>;
```

现在我们可以定义所有 32 个通用寄存器：

```cpp
foreach I = 0-31 in {
  def R#I : GRi<I, "r"#I>;
}
```

单个寄存器需要分组在寄存器类中。寄存器的序列顺序还定义了寄存器分配器中的分配顺序。在这里，我们只是添加所有寄存器：

```cpp
def GPR : RegisterClass<"M88k", [i32], 32,
                            (add (sequence "R%u", 0, 31))>;
```

最后，我们需要基于寄存器类定义一个操作数。该操作数用于选择 DAG 节点以匹配寄存器，并且还可以扩展以表示打印和匹配汇编代码中的方法名称：

```cpp
def GPROpnd : RegisterOperand<GPR>;
```

这完成了我们对寄存器的定义。在下一节中，我们将使用这些定义来定义调用约定。

## 定义调用约定

**调用约定**定义了如何传递参数给函数。通常，第一个参数是通过寄存器传递的，其余的参数是通过堆栈传递的。还必须制定关于如何传递聚合和如何从函数返回值的规则。根据这里给出的定义，生成了分析器类，稍后在调用降级期间使用。

您可以在*第三章*中阅读 M88k 架构使用的调用约定，*低级系统信息*，*System V ABI M88k 处理器*补充书。让我们将其翻译成 TableGen 语法：

1.  我们为调用约定定义一个记录：

```cpp
def CC_M88k : CallingConv<[
```

1.  M88k 架构只有 32 位寄存器，因此需要将较小数据类型的值提升为 32 位：

```cpp
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,
```

1.  调用约定规定，对于聚合返回值，内存的指针将传递到`r12`寄存器中：

```cpp
  CCIfSRet<CCIfType<[i32], CCAssignToReg<[R12]>>>,
```

1.  寄存器`r2`到`r9`用于传递参数：

```cpp
  CCIfType<[i32,i64,f32,f64],
          CCAssignToReg<[R2, R3, R4, R5, R6, R7, R8, 
            R9]>>,
```

1.  每个额外的参数都以 4 字节对齐的插槽传递到堆栈上：

```cpp
  CCAssignToStack<4, 4>
]>;
```

1.  另一个记录定义了如何将结果传递给调用函数。32 位值在`r2`寄存器中传递，64 位值使用`r2`和`r3`寄存器：

```cpp
def RetCC_M88k : CallingConv<[
  CCIfType<[i32,f32], CCAssignToReg<[R2]>>,
  CCIfType<[i64,f64], CCAssignToReg<[R2, R3]>>
]>;
```

1.  最后，调用约定还说明了由被调用函数保留的寄存器：

```cpp
def CSR_M88k :
         CalleeSavedRegs<(add (sequence "R%d", 14, 
           25), R30)>;
```

如果需要，您还可以定义多个调用约定。在下一节中，我们将简要介绍调度模型。

## 创建调度模型

调度模型被代码生成用来以最佳方式排序指令。定义调度模型可以提高生成代码的性能，但对于代码生成并不是必需的。因此，我们只为模型定义一个占位符。我们添加的信息是 CPU 最多可以同时发出两条指令，并且它是一个顺序 CPU：

```cpp
def M88kSchedModel : SchedMachineModel {
  let IssueWidth = 2;
  let MicroOpBufferSize = 0;
  let CompleteModel = 0;
  let NoModel = 1;
}
```

您可以在 YouTube 上的[`www.youtube.com/watch?v=brpomKUynEA`](https://www.youtube.com/watch?v=brpomKUynEA)上找到有关如何创建完整调度模型的教程*编写优秀的调度程序*。

接下来，我们将定义指令格式和指令。

## 定义指令格式和指令信息

我们已经在*第九章**，指令选择*中查看了指令格式和指令信息，在*支持新机器指令*部分。为了定义 M88k 架构的指令，我们遵循相同的方法。首先，我们为指令记录定义一个基类。这个类最重要的字段是`Inst`字段，它保存了指令的编码。这个类的大多数其他字段定义只是为`Instruction`超类中定义的字段赋值：

```cpp
class InstM88k<dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin = 
           NoItinerary>
   : Instruction {
  field bits<32> Inst;
  field bits<32> SoftFail = 0; 
  let Namespace = "M88k";
  let Size = 4;
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
  let DecoderNamespace = "M88k";
  let Itinerary = itin;
}
```

这个基类用于所有指令格式，因此也用于`F_JMP`格式。您可以使用处理器的用户手册中的编码。该类有两个参数，必须是编码的一部分。`func`参数定义了编码的第 11 到 15 位，这些位定义了指令是带有或不带有保存返回地址的跳转指令。`next`参数是一个位，定义了下一条指令是否无条件执行。这类似于 MIPS 架构的延迟槽。

该类还定义了`rs2`字段，其中保存了保存目标地址的寄存器的编码。其他参数包括 DAG 输入和输出操作数，文本汇编器字符串，用于选择此指令的 DAG 模式，以及调度器模型的行程类：

```cpp
class F_JMP<bits<5> func, bits<1> next,
            dag outs, dag ins, string asmstr,
            list<dag> pattern,
            InstrItinClass itin = NoItinerary>
   : InstM88k<outs, ins, asmstr, pattern, itin> {
  bits<5> rs2;
  let Inst{31-26} = 0b111101;
  let Inst{25-16} = 0b0000000000;
  let Inst{15-11} = func;
  let Inst{10}    = next;
  let Inst{9-5}   = 0b00000;
  let Inst{4-0}   = rs2;
}
```

有了这个，我们最终可以定义指令了。跳转指令是基本块中的最后一条指令，因此我们需要设置`isTerminator`标志。因为控制流不能通过此指令，我们还必须设置`isBarrier`标志。我们从处理器的用户手册中获取`func`和`next`参数的值。

输入 DAG 操作数是一个通用寄存器，并指的是前一个寄存器信息中的操作数。编码存储在`rs2`字段中，来自前一个类定义。输出操作数为空。汇编字符串给出了指令的文本语法，也指的是寄存器操作数。DAG 模式使用预定义的`brind`操作符。如果 DAG 包含一个以寄存器中保存的目标地址为目标的间接跳转节点，则选择此指令：

```cpp
let isTerminator = 1, isBarrier = 1 in
  def JMP : F_JMP<0b11000, 0, (outs), (ins GPROpnd:$rs2),
                  "jmp $rs2", [(brind GPROpnd:$rs2)]>;
```

我们需要以这种方式为所有指令定义记录。

在这个文件中，我们还实现了指令选择的其他必要模式。一个典型的应用是常量合成。M88k 架构有 32 位宽的寄存器，但是带有立即数操作数的指令只支持 16 位宽的常量。因此，诸如寄存器和 32 位常量之间的按位`and`等操作必须分成两条使用 16 位常量的指令。

幸运的是，`and`指令中的一个标志定义了操作是应用于寄存器的下半部分还是上半部分。使用 LO16 和 HI16 操作符来提取常量的下半部分或上半部分，我们可以为寄存器和 32 位宽常量之间的`and`操作制定一个 DAG 模式：

```cpp
def : Pat<(and GPR:$rs1, uimm32:$imm),
          (ANDri (ANDriu GPR:$rs1, (HI16 i32:$imm)),
                                   (LO16 i32:$imm))>;
```

`ANDri`操作符是将常量应用于寄存器的低半部分的`and`指令，而`ANDriu`操作符使用寄存器的上半部分。当然，在模式中使用这些名称之前，我们必须像定义`jmp`指令一样定义指令。此模式解决了使用 32 位常量进行`and`操作的问题，在指令选择期间为其生成两条机器指令。

并非所有操作都可以由预定义的 DAG 节点表示。例如，M88k 架构定义了位字段操作，可以看作是普通`and`/`or`操作的泛化。对于这样的操作，可以引入新的节点类型，例如`set`指令：

```cpp
def m88k_set : SDNode<"M88kISD::SET", SDTIntBinOp>;
```

这定义了`SDNode`类的新记录。第一个参数是表示新操作的 C++枚举成员。第二个参数是所谓的类型配置文件，定义了参数的类型和数量以及结果类型。预定义的`SDTIntBinOp`类定义了两个整数参数和一个整数结果类型，这对于此操作是合适的。您可以在`llvm/include/llvm/Target/TargetSelectionDAG.td`文件中查找预定义的类。如果没有合适的预定义类型配置文件，那么您可以定义一个新的。

对于调用函数，LLVM 需要某些不能预定义的定义，因为它们不完全是与目标无关的。例如，对于返回，我们需要指定`retflag`记录：

```cpp
def retflag : SDNode<"M88kISD::RET_FLAG", SDTNone,
                 [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
```

将此与`m88k_set`记录进行比较，这也为 DAG 节点定义了一些标志：链和粘合序列被使用，并且操作符可以接受可变数量的参数。

逐步实现指令

现代 CPU 很容易有成千上万条指令。一次不要实现所有指令是有意义的。相反，您应该首先集中在基本指令上，例如逻辑操作和调用和返回指令。这足以使基本的后端工作。然后，您可以添加更多的指令定义和模式。

这完成了我们对目标描述的实现。从目标描述中，使用`llvm-tblgen`工具自动生成了大量代码。为了完成指令选择和后端的其他部分，我们仍然需要使用生成的代码开发 C++源代码。在下一节中，我们将实现 DAG 指令选择。

# 实现 DAG 指令选择类

DAG 指令选择器的大部分是由`llvm-tblgen`工具生成的。我们仍然需要使用生成的代码创建类，并将所有内容放在一起。让我们从初始化过程的一部分开始。

## 初始化目标机器

每个后端都必须提供至少一个`TargetMachine`类，通常是`LLVMTargetMachine`类的子类。`M88kTargetMachine`类包含了代码生成所需的许多细节，并且还充当其他后端类的工厂，尤其是`Subtarget`类和`TargetPassConfig`类。`Subtarget`类保存了代码生成的配置，例如启用了哪些特性。`TargetPassConfig`类配置了后端的机器传递。我们的`M88kTargetMachine`类的声明在`M88ktargetMachine.h`文件中，如下所示：

```cpp
class M88kTargetMachine : public LLVMTargetMachine {
public:
  M88kTargetMachine(/* parameters */);
  ~M88kTargetMachine() override;
  const M88kSubtarget *getSubtargetImpl(const Function &)
                                        const override;
  const M88kSubtarget *getSubtargetImpl() const = delete;
  TargetPassConfig *createPassConfig(PassManagerBase &PM)
                                                     override;
};
```

请注意，每个函数可能有不同的子目标。

`M88kTargetMachine.cpp`文件中的实现是直接的。最有趣的是为此后端设置机器传递。这创建了与选择 DAG（如果需要，还有全局指令选择）的连接。类中创建的传递后来被添加到传递管道中，以从 IR 生成目标文件或汇编程序：

```cpp
namespace {
class M88kPassConfig : public TargetPassConfig {
public:
  M88kPassConfig(M88kTargetMachine &TM, PassManagerBase 
    &PM)
      : TargetPassConfig(TM, PM) {}
  M88kTargetMachine &getM88kTargetMachine() const {
    return getTM<M88kTargetMachine>();
  }
  bool addInstSelector() override {
    addPass(createM88kISelDag(getM88kTargetMachine(), 
                              getOptLevel()));
    return false;
  }
};
} // namespace
TargetPassConfig *M88kTargetMachine::createPassConfig(
    PassManagerBase &PM) {
  return new M88kPassConfig(*this, PM);
}
```

`SubTarget`实现从`M88kTargetMachine`类返回，可以访问其他重要的类。`M88kInstrInfo`类返回有关指令的信息，包括寄存器。`M88kTargetLowering`类提供了与调用相关指令的降低，并允许添加自定义的 DAG 规则。大部分类是由`llvm-tblgen`工具生成的，我们需要包含生成的头文件。

`M88kSubTarget.h`文件中的定义如下：

```cpp
#define GET_SUBTARGETINFO_HEADER
#include "M88kGenSubtargetInfo.inc"
namespace llvm {
class M88kSubtarget : public M88kGenSubtargetInfo {
  Triple TargetTriple;
  virtual void anchor();
  M88kInstrInfo InstrInfo;
  M88kTargetLowering TLInfo;
  M88kFrameLowering FrameLowering;
public:
  M88kSubtarget(const Triple &TT, const std::string &CPU,
                const std::string &FS,
                const TargetMachine &TM);
  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);
  const TargetFrameLowering *getFrameLowering() const 
    override
  { return &FrameLowering; }
  const M88kInstrInfo *getInstrInfo() const override
  { return &InstrInfo; }
  const M88kRegisterInfo *getRegisterInfo() const override
  { return &InstrInfo.getRegisterInfo(); }
  const M88kTargetLowering *getTargetLowering() const 
    override 
  { return &TLInfo; }
};
} // end namespace llvm
```

接下来，我们实现选择 DAG。

## 添加选择 DAG 实现

选择 DAG 在同名文件中的`M88kDAGtoDAGIsel`类中实现。在这里，我们受益于已经创建了目标机器描述：大部分功能都是从这个描述中生成的。在最初的实现中，我们只需要重写`Select()`函数并将其转发到生成的`SelectCode`函数。还可以为特定情况重写更多函数，例如，如果我们需要扩展 DAG 的预处理，或者如果我们需要添加特殊的内联汇编约束。

因为这个类是一个机器函数传递，我们还为传递提供了一个名称。主要的实现部分来自生成的文件，我们在类的中间包含了这个文件：

```cpp
class M88kDAGToDAGISel : public SelectionDAGISel {
  const M88kSubtarget *Subtarget;
public:
  M88kDAGToDAGISel(M88kTargetMachine &TM,
                   CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(TM, OptLevel) {}
  StringRef getPassName() const override {
    return "M88k DAG->DAG Pattern Instruction Selection";
  }
#include "M88kGenDAGISel.inc"
  void Select(SDNode *Node) override {
    SelectCode(Node);
  }
};
```

我们还在这个文件中添加了创建传递的工厂函数：

```cpp
FunctionPass *llvm::createM88kISelDag(M88kTargetMachine &TM,
                                 CodeGenOpt::Level                 
                                   OptLevel) {
  return new M88kDAGToDAGISel(TM, OptLevel);
}
```

现在我们可以实现目标特定的操作，这些操作无法在目标描述中表达。

## 支持目标特定操作

让我们转向`M88kTargetLowering`类，在`M88kISelLowering.h`文件中定义。这个类配置指令 DAG 选择过程，并增强了目标特定操作的降低。

在目标描述中，我们定义了新的 DAG 节点。与新类型一起使用的枚举也在这个文件中定义，继续使用上一个预定义数字的编号：

```cpp
namespace M88kISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  RET_FLAG,
  SET,
};
} // end namespace M88kISD
```

该类需要为函数调用提供所需的降低方法。为了保持简单，我们只关注返回值。该类还可以为需要自定义处理的操作定义`LowerOperation()`挂钩方法。我们还可以启用自定义 DAG 组合方法，为此我们定义`PerformDAGCombine()`方法：

```cpp
class M88kTargetLowering : public TargetLowering {
  const M88kSubtarget &Subtarget;
public:
  explicit M88kTargetLowering(const TargetMachine &TM,
                              const M88kSubtarget &STI);
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const 
                                                     override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) 
                                               const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv,
          bool IsVarArg,
          const SmallVectorImpl<ISD::OutputArg> &Outs,
          const SmallVectorImpl<SDValue> &OutVals,
          const SDLoc &DL,
          SelectionDAG &DAG) const override;
};
```

该类的实现在`M88kISelLowering.cpp`文件中。首先，我们看一下如何降低返回值：

1.  需要调用约定的生成函数，因此我们包含了生成的文件：

```cpp
#include "M88kGenCallingConv.inc"
```

1.  `LowerReturn()`方法有很多参数，所有这些参数都是由`TargetLowering`超类定义的。最重要的是`Outs`向量，它保存了返回参数的描述，以及`OutVals`向量，它保存了返回值的 DAG 节点：

```cpp
SDValue M88kTargetLowering::LowerReturn(SDValue Chain,
            CallingConv::ID CallConv,
            bool IsVarArg,
            const SmallVectorImpl<ISD::OutputArg> 
              &Outs,
            const SmallVectorImpl<SDValue> &OutVals,
            const SDLoc &DL, SelectionDAG &DAG) const {
```

1.  我们使用`CCState`类来分析返回参数，并传递一个对生成的`RetCC_M88k`函数的引用。结果，我们已经对所有的返回参数进行了分类：

```cpp
  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs,
                                      *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_M88k);
```

1.  如果是`void`函数，则无需操作并返回。请注意，返回节点的类型是`RET_FLAG`。我们在目标描述中定义了这个新的`ret_flag`节点：

```cpp
  if (RetLocs.empty())
    return DAG.getNode(M88kISD::RET_FLAG, DL,
                       MVT::Other, Chain);
```

1.  否则，我们需要循环遍历返回参数。对于每个返回参数，我们都有一个`CCValAssign`类的实例，告诉我们如何处理参数：

```cpp
  SDValue Glue;
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; 
       ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];
```

1.  值可能需要提升。如果需要，我们添加一个带有所需扩展操作的 DAG 节点：

```cpp
    switch (VA.getLocInfo()) {
    case CCValAssign::SExt:
      RetValue = DAG.getNode(ISD::SIGN_EXTEND, DL,
                             VA.getLocVT(), RetValue);
      break;
    case CCValAssign::ZExt:
      RetValue = DAG.getNode(ISD::ZERO_EXTEND, DL, 
                             VA.getLocVT(), RetValue);
      break;
    case CCValAssign::AExt:
      RetValue = DAG.getNode(ISD::ANY_EXTEND, DL,  
                             VA.getLocVT(), RetValue);
      break;
    case CCValAssign::Full:
      break;
    default:
      llvm_unreachable("Unhandled VA.getLocInfo()");
    }
```

1.  当值具有正确的类型时，我们将该值复制到寄存器中返回，并将复制的链和粘合在一起。这完成了循环：

```cpp
    Register Reg = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RetValue, 
                             Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, 
                                     VA.getLocVT()));
  }
```

1.  最后，我们需要更新链和粘合：

```cpp
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);
```

1.  然后我们将返回`ret_flag`节点，连接降低的结果：

```cpp
  return DAG.getNode(M88kISD::RET_FLAG, DL, 
    MVT::Other, 
                     RetOps);
}
```

为了能够调用函数，我们必须实现`LowerFormalArguments()`和`LowerCall()`方法。这两种方法都遵循类似的方法，因此这里不再显示。

## 配置目标降低

必须始终实现降低函数调用和参数的方法，因为它们始终是与目标相关的。其他操作可能在目标架构中有或没有支持。为了使降低过程意识到这一点，我们在`M88kTargetLowering`类的构造函数中设置了配置：

1.  构造函数以`TargetMachine`和`M88kSubtarget`实例作为参数，并用它们初始化相应的字段：

```cpp
M88kTargetLowering::M88kTargetLowering(
       const TargetMachine &TM, const M88kSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
```

1.  首先添加所有寄存器类。我们只定义了通用寄存器，因此这只是一个简单的调用：

```cpp
  addRegisterClass(MVT::i32, &M88k::GPRRegClass);
```

1.  在添加所有寄存器类之后，我们计算寄存器的派生属性。例如，由于寄存器宽度为 32 位，因此此函数将 64 位数据类型标记为需要两个寄存器：

```cpp
  computeRegisterProperties(Subtarget.getRegisterInfo());
```

1.  我们还需要告诉哪个寄存器用于堆栈指针。在 M88k 架构中，使用`r31`寄存器：

```cpp
  setStackPointerRegisterToSaveRestore(M88k::R31);
```

1.  我们还需要定义`boolean`值的表示方式。基本上，我们在这里说使用值 0 和 1。其他可能的选项是仅查看值的第 0 位，忽略所有其他位，并将值的所有位设置为 0 或 1：

```cpp
  setBooleanContents(ZeroOrOneBooleanContent);
```

1.  对于每个需要特殊处理的操作，我们必须调用`setOperationAction()`方法。该方法以操作、值类型和要执行的操作作为输入。如果操作有效，则使用`Legal`操作值。如果类型应该提升，则使用`Promote`操作值，如果操作应该导致库调用，则使用`LibCall`操作值。

如果给出`Expand`操作值，则指令选择首先尝试将此操作扩展为其他操作。如果这不可能，则使用库调用。最后，如果使用`Custom`操作值，我们可以实现自己的操作。在这种情况下，将为具有此操作的节点调用`LowerOperation()`方法。例如，我们将`CTTZ`计数尾随零操作设置为`Expand`操作。此操作将被一系列原始位操作替换：

```cpp
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
```

1.  M88k 架构具有位字段操作，对于该操作，很难在目标描述中定义模式。在这里，我们告诉指令选择器，我们希望在`or` DAG 节点上执行额外的匹配：

```cpp
  setTargetDAGCombine(ISD::OR);
}
```

根据目标架构，设置构造函数中的配置可能会更长。我们只定义了最低限度，例如忽略了浮点运算。

我们已经标记了`or`操作以执行自定义组合。因此，指令选择器在调用生成的指令选择之前调用`PerformDAGCombine()`方法。此函数在指令选择的各个阶段调用，但通常，我们只在操作被合法化后执行匹配。通用实现是查看操作并跳转到处理匹配的函数。

```cpp
SDValue M88kTargetLowering::PerformDAGCombine(SDNode *N,
                                 DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::OR:
    return performORCombine(N, DCI);
  }
  return SDValue();
}
```

在`performORCombine()`方法中，我们尝试检查是否可以为`or`操作生成`set`指令。`set`指令将一系列连续的位设置为 1，从指定的位偏移开始。这是`or`操作的特殊情况，第二个操作数是匹配此格式的常量。由于 M88k 架构的`or`指令仅适用于 16 位常量，因此此匹配是有益的，否则，我们将不得不合成常量，导致两个`or`指令。此方法使用`isShiftedMask()`辅助函数来确定常量值是否具有所需的形式。

如果第二个操作数是所需形式的常量，则此函数返回表示`set`指令的节点。否则，返回值`SDValue()`表示未找到匹配模式，应调用生成的 DAG 模式匹配器：

```cpp
SDValue performORCombine(SDNode *N, 
    TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  uint64_t Width, Offset;
  ConstantSDNode *Mask =
                   dyn_cast<ConstantSDNode>(N->getOperand(
                     1));
  if (!Mask ||
      !isShiftedMask(Mask->getZExtValue(), Width, Offset))
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  return DAG.getNode(M88kISD::SET, DL, ValTy, 
          N->getOperand(0),
          DAG.getConstant(Width << 5 | Offset, DL, 
            MVT::i32));
}
```

为了完成整个降低过程的实现，我们需要实现`M88kFrameLowering`类。这个类负责处理堆栈帧。这包括生成序言和结语代码，处理寄存器溢出等。对于第一个实现，您可以只提供空函数。显然，为了完整的功能，这个类必须被实现。

这完成了我们对指令选择的实现。接下来，我们将看看最终的指令是如何发出的。

# 生成汇编指令

在前几节中实现的指令选择将 IR 指令降低为`MachineInstr`实例。这已经是指令的更低表示，但还不是机器码本身。后端管道中的最后一步是发出指令，可以是汇编文本，也可以是目标文件。`M88kAsmPrinter`机器传递负责这项任务。

基本上，这个传递将`MachineInstr`实例降低到`MCInst`实例，然后发出到一个流器。`MCInst`类表示真正的机器码指令。这种额外的降低是必需的，因为`MachineInstr`类仍然没有所有必需的细节。

对于第一种方法，我们可以将我们的实现限制在重写`emitInstruction()`方法。您需要重写更多的方法来支持几种操作数类型，主要是为了发出正确的重定位。这个类还负责处理内联汇编器，如果需要的话，您也需要实现它。

因为`M88kAsmPrinter`类再次是一个机器函数传递，我们还需要重写`getPassName()`方法。该类的声明如下：

```cpp
class M88kAsmPrinter : public AsmPrinter {
public:
  explicit M88kAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> 
                           Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}
  StringRef getPassName() const override
  { return "M88k Assembly Printer"; }
  void emitInstruction(const MachineInstr *MI) override;
};
```

基本上，在`emitInstruction()`方法中我们必须处理两种不同的情况。`MachineInstr`实例仍然可以有操作数，这些操作数不是真正的机器指令。例如，对于返回`ret_flag`节点，具有`RET`操作码值。在 M88k 架构上，没有`return`指令。相反，会跳转到存储在`r1`寄存器中的地址。因此，当检测到`RET`操作码时，我们需要构造分支指令。在默认情况下，降低只需要`MachineInstr`实例的信息，我们将这个任务委托给`M88kMCInstLower`类：

```cpp
void M88kAsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst LoweredMI;
  switch (MI->getOpcode()) {
  case M88k::RET:
    LoweredMI = MCInstBuilder(M88k::JMP).addReg(M88k::R1);
    break;
  default:
    M88kMCInstLower Lower(MF->getContext(), *this);
    Lower.lower(MI, LoweredMI);
    break;
  }
  EmitToStreamer(*OutStreamer, LoweredMI);
}
```

`M88kMCInstLower`类没有预定义的超类。它的主要目的是处理各种操作数类型。由于目前我们只有一组非常有限的支持的操作数类型，我们可以将这个类简化为只有一个方法。`lower()`方法设置`MCInst`实例的操作码和操作数。只处理寄存器和立即操作数；其他操作数类型被忽略。对于完整的实现，我们还需要处理内存地址。

```cpp
void M88kMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) 
  {
    const MachineOperand &MO = MI->getOperand(I);
    switch (MO.getType()) {
    case MachineOperand::MO_Register:
      if (MO.isImplicit())
        break;
      OutMI.addOperand(MCOperand::createReg(MO.getReg()));
      break;
    case MachineOperand::MO_Immediate:
      OutMI.addOperand(MCOperand::createImm(MO.getImm()));
      break;
    default:
      break;
    }
  }
}
```

汇编打印机需要一个工厂方法，在初始化期间调用，例如从`InitializeAllAsmPrinters()`方法：

```cpp
extern "C" LLVM_EXTERNAL_VISIBILITY void 
LLVMInitializeM88kAsmPrinter() {
  RegisterAsmPrinter<M88kAsmPrinter> X(getTheM88kTarget());
}
```

最后，将指令降低到真实的机器码指令后，我们还没有完成。我们需要在 MC 层实现各种小的部分，我们将在下一节中讨论。

# 发出机器码

MC 层负责以文本或二进制形式发出机器码。大部分功能要么在各种 MC 类中实现并且只需要配置，要么从目标描述生成实现。

MC 层的初始化在`MCTargetDesc/M88kMCTargetDesc.cpp`文件中进行。以下类已在`TargetRegistry`单例中注册：

+   `M88kMCAsmInfo`：这个类提供基本信息，比如代码指针的大小，堆栈增长的方向，注释符号，或者汇编指令的名称。

+   `M88MCInstrInfo`：这个类保存有关指令的信息，例如指令的名称。

+   `M88kRegInfo`：此类提供有关寄存器的信息，例如寄存器的名称或哪个寄存器是堆栈指针。

+   `M88kSubtargetInfo`：此类保存调度模型的数据和解析和设置 CPU 特性的方法。

+   `M88kMCAsmBackend`：此类提供了获取与目标相关的修正数据的辅助方法。它还包含了用于对象编写器类的工厂方法。

+   `M88kMCInstPrinter`：此类包含一些辅助方法，用于以文本形式打印指令和操作数。如果操作数在目标描述中定义了自定义打印方法，则必须在此类中实现。

+   `M88kMCCodeEmitter`：此类将指令的编码写入流中。

根据后端实现的范围，我们不需要注册和实现所有这些类。如果不支持文本汇编器输出，则可以省略注册`MCInstPrinter`子类。如果不支持编写目标文件，则可以省略`MCAsmBackend`和`MCCodeEmitter`子类。

我们首先包含生成的部分，并为其提供工厂方法：

```cpp
#define GET_INSTRINFO_MC_DESC
#include "M88kGenInstrInfo.inc"
#define GET_SUBTARGETINFO_MC_DESC
#include "M88kGenSubtargetInfo.inc"
#define GET_REGINFO_MC_DESC
#include "M88kGenRegisterInfo.inc"
static MCInstrInfo *createM88kMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitM88kMCInstrInfo(X);
  return X;
}
static MCRegisterInfo *createM88kMCRegisterInfo(
                                           const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitM88kMCRegisterInfo(X, M88k::R1);
  return X;
}
static MCSubtargetInfo *createM88kMCSubtargetInfo(
              const Triple &TT, StringRef CPU, StringRef 
                FS) {
  return createM88kMCSubtargetInfoImpl(TT, CPU, FS);
}
```

我们还为其他文件中实现的类提供了一些工厂方法：

```cpp
static MCAsmInfo *createM88kMCAsmInfo(
                  const MCRegisterInfo &MRI, const Triple &TT,
                  const MCTargetOptions &Options) {
  return new M88kMCAsmInfo(TT);
}
static MCInstPrinter *createM88kMCInstPrinter(
                 const Triple &T, unsigned SyntaxVariant,
                 const MCAsmInfo &MAI, const MCInstrInfo &MII,
                 const MCRegisterInfo &MRI) {
  return new M88kInstPrinter(MAI, MII, MRI);
}
```

要初始化 MC 层，我们只需要使用`TargetRegistry`单例注册所有工厂方法：

```cpp
extern "C" LLVM_EXTERNAL_VISIBILITY
void LLVMInitializeM88kTargetMC() {
  TargetRegistry::RegisterMCAsmInfo(getTheM88kTarget(), 
                                         createM88kMCAsmInfo);
  TargetRegistry::RegisterMCCodeEmitter(getTheM88kTarget(),

                                     createM88kMCCodeEmitter);
  TargetRegistry::RegisterMCInstrInfo(getTheM88kTarget(),
                                       createM88kMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(getTheM88kTarget(),
                                    createM88kMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(getTheM88kTarget(),
                                   createM88kMCSubtargetInfo);
  TargetRegistry::RegisterMCAsmBackend(getTheM88kTarget(),
                                      createM88kMCAsmBackend);
  TargetRegistry::RegisterMCInstPrinter(getTheM88kTarget(),
                                     createM88kMCInstPrinter);
}
```

此外，在`MCTargetDesc/M88kTargetDesc.h`头文件中，我们还需要包含生成源的头部部分，以便其他人也可以使用：

```cpp
#define GET_REGINFO_ENUM
#include "M88kGenRegisterInfo.inc"
#define GET_INSTRINFO_ENUM
#include "M88kGenInstrInfo.inc"
#define GET_SUBTARGETINFO_ENUM
#include "M88kGenSubtargetInfo.inc"
```

我们将注册类的源文件都放在`MCTargetDesc`目录中。对于第一个实现，只需为这些类提供存根即可。例如，只要目标描述中没有添加对内存地址的支持，就不会生成修正。`M88kMCAsmInfo`类可以非常快速地实现，因为我们只需要在构造函数中设置一些属性：

```cpp
M88kMCAsmInfo::M88kMCAsmInfo(const Triple &TT) {
  CodePointerSize = 4;
  IsLittleEndian = false;
  MinInstAlignment = 4;
  CommentString = "#";
}
```

在为 MC 层实现了支持类之后，我们现在能够将机器码输出到文件中。

在下一节中，我们实现了用于反汇编的类，这是相反的操作：将目标文件转换回汇编器文本。

# 添加反汇编支持

目标描述中指令的定义允许构建解码器表，用于将目标文件反汇编为文本汇编器。解码器表和解码器函数由`llvm-tblgen`工具生成。除了生成的代码，我们只需要提供注册和初始化`M88kDisassembler`类以及一些辅助函数来解码寄存器和操作数的代码。我们将实现放在`Disassembler/M88kDisassembler.cpp`文件中。

`M88kDisassembler`类的`getInstruction()`方法执行解码工作。它以字节数组作为输入，并将下一条指令解码为`MCInst`类的实例。类声明如下：

```cpp
using DecodeStatus = MCDisassembler::DecodeStatus;
namespace {
class M88kDisassembler : public MCDisassembler {
public:
  M88kDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}
  ~M88kDisassembler() override = default;
  DecodeStatus getInstruction(MCInst &instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, 
                              uint64_t Address,
                              raw_ostream &CStream) const 
                                                     override;
};
}
```

生成的类未经限定地引用`DecodeStatus`枚举，因此我们必须使此名称可见。

要初始化反汇编器，我们定义一个简单的工厂函数来实例化一个新对象：

```cpp
static MCDisassembler *
createM88kDisassembler(const Target &T,
                       const MCSubtargetInfo &STI,
                       MCContext &Ctx) {
  return new M88kDisassembler(STI, Ctx);
}
```

在`LLVMInitializeM88kDisassembler()`函数中，我们在目标注册表中注册工厂函数：

```cpp
extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeM88kDisassembler() {
  TargetRegistry::RegisterMCDisassembler(
      getTheM88kTarget(), createM88kDisassembler);
}
```

当 LLVM 核心库初始化时，此函数将从`InitializeAllDisassemblers()`函数或`InitializeNativeTargetDisassembler()`函数中调用。

生成的解码器函数需要辅助函数来解码寄存器和操作数。原因是这些元素的编码通常涉及目标描述中未表达的特殊情况。例如，两条指令之间的距离总是偶数，因此可以忽略最低位，因为它总是零。

要解码寄存器，必须定义`DecodeGPRRegisterClass()`函数。32 个寄存器用 0 到 31 之间的数字进行编码，我们可以使用静态的`GPRDecoderTable`表来在编码和生成的寄存器枚举之间进行映射：

```cpp
static const uint16_t GPRDecoderTable[] = {
    M88k::R0,  M88k::R1,  M88k::R2,  M88k::R3,
    M88k::R4,  M88k::R5,  M88k::R6,  M88k::R7,
    M88k::R8,  M88k::R9,  M88k::R10, M88k::R11,
    M88k::R12, M88k::R13, M88k::R14, M88k::R15,
    M88k::R16, M88k::R17, M88k::R18, M88k::R19,
    M88k::R20, M88k::R21, M88k::R22, M88k::R23,
    M88k::R24, M88k::R25, M88k::R26, M88k::R27,
    M88k::R28, M88k::R29, M88k::R30, M88k::R31,
};
static DecodeStatus
DecodeGPRRegisterClass(MCInst &Inst, uint64_t RegNo,
                       uint64_t Address,
                       const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;
  unsigned Register = GPRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}
```

所有其他所需的解码器函数都遵循与`DecodeGPRRegisterClass()`函数相同的模式：

1.  检查要解码的值是否符合所需的大小限制。如果不符合，则返回`MCDisassembler::Fail`值。

1.  解码值并将其添加到`MCInst`实例中。

1.  返回`MCDisassembler::Success`以指示成功。

然后，我们可以包含生成的解码器表和函数：

`#include "M88kGenDisassemblerTables.inc"`

最后，我们可以定义`getInstruction()`方法。该方法有两个结果值，解码指令和指令的大小。如果字节数组太小，则大小必须设置为`0`。这很重要，因为大小参数被调用者用来将指针推进到下一个内存位置，即使解码失败也是如此。

对于 M88k 架构，该方法很简单，因为所有指令都是 4 个字节长。因此，在从数组中提取 4 个字节后，可以调用生成的解码器函数：

```cpp
DecodeStatus M88kDisassembler::getInstruction(
    MCInst &MI, uint64_t &Size, ArrayRef<uint8_t> Bytes,
    uint64_t Address, raw_ostream &CS) const {
  if (Bytes.size() < 4) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 4;
  uint32_t Inst = 0;
  for (uint32_t I = 0; I < Size; ++I)
    Inst = (Inst << 8) | Bytes[I];
  return decodeInstruction(DecoderTableM88k32, MI, Inst,
                           Address, this, STI);
}
```

这完成了反汇编器的实现。

在实现了所有类之后，我们只需要设置构建系统以选择新的目标后端，这将在下一节中添加。

# 将所有部分组合在一起

我们的新目标位于`llvm/lib/Target/M88k`目录中，需要集成到构建系统中。为了方便开发，我们将其添加为`llvm/CMakeLists.txt`文件中的实验性目标。我们用我们的目标名称替换现有的空字符串：

```cpp
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "M88k"  … )
```

我们还需要提供一个`llvm/lib/Target/M88k/CMakeLists.txt`文件来构建我们的目标。除了列出目标的 C++文件外，它还定义了从目标描述生成源代码。

从目标描述生成所有类型的源

`llvm-tblgen`工具的不同运行会生成不同部分的 C++代码。然而，我建议将所有部分的生成都添加到`CMakeLists.txt`文件中。原因是这样做提供了更好的检查。例如，如果您在指令编码中出错，那么只有在反汇编器代码生成期间才会捕获到错误。因此，即使您不打算支持反汇编器，生成其源代码仍然是值得的。

文件如下所示：

1.  首先，我们定义一个名为`M88k`的新的 LLVM 组件：

```cpp
add_llvm_component_group(M88k)
```

1.  接下来，我们命名目标描述文件，添加语句以使用 TableGen 生成各种源代码片段，并为其定义一个公共目标：

```cpp
set(LLVM_TARGET_DEFINITIONS M88k.tdtablegen(LLVM M88kGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM M88kGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM M88kGenCallingConv.inc -gen-callingconv)
tablegen(LLVM M88kGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM M88kGenDisassemblerTables.inc 
                                       -gen-disassembler)
tablegen(LLVM M88kGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM M88kGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM M88kGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM M88kGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(M88kCommonTableGen)
```

1.  我们必须列出新组件由哪些源文件组成：

```cpp
add_llvm_target(M88kCodeGen
  M88kAsmPrinter.cpp M88kFrameLowering.cpp
  M88kISelDAGToDAG.cpp M88kISelLowering.cpp
  M88kRegisterInfo.cpp M88kSubtarget.cpp
  M88kTargetMachine.cpp )
```

1.  最后，我们在构建中包含了 MC 和反汇编器类的目录：

```cpp
add_subdirectory(MCTargetDesc)
add_subdirectory(Disassembler)
```

现在我们准备用新的后端目标编译 LLVM。在构建目录中，我们可以简单地运行这个命令：

```cpp
$ ninja
```

这将检测到更改的`CmakeLists.txt`文件，再次运行配置步骤，并编译新的后端。要检查一切是否顺利，您可以运行这个命令：

```cpp
$ bin/llc –version
```

输出应该在`Registered Target`部分包含以下行：

```cpp
    m88k       - M88k
```

万岁！我们完成了后端的实现。让我们试试。LLVM IR 中的以下`f1`函数执行函数的两个参数之间的按位 AND 操作，并返回结果。将其保存在`example.ll`文件中：

```cpp
target triple = "m88k-openbsd"
define i32 @f1(i32 %a, i32 %b) {
  %res = and i32 %a, %b
  ret i32 %res
}
```

运行`llc`工具如下以在控制台上查看生成的汇编文本：

```cpp
$ llc < example.ll
        .text
        .file   "<stdin>"
        .globl  f1                              # -- Begin function f1
        .align  3
        .type   f1,@function
f1:                                     # @f1
        .cfi_startproc
# %bb.0:
        and %r2, %r2, %r3
        jmp %r1
.Lfunc_end0:
        .size   f1, .Lfunc_end0-f1
        .cfi_endproc
                                        # -- End function
        .section        ".note.GNU-stack","",@progbits
```

输出符合有效的 GNU 语法。对于`f1`函数，生成了`and`和`jmp`指令。参数传递在`%r2`和`%r3`寄存器中，这些寄存器在`and`指令中被使用。结果存储在`%r2`寄存器中，这也是返回 32 位值的寄存器。函数的返回通过跳转到`%r1`寄存器中保存的地址来实现，这也符合 ABI。一切看起来都很不错！

通过本章学到的知识，你现在可以实现自己的 LLVM 后端。对于许多相对简单的 CPU，比如数字信号处理器（DSP），你不需要实现更多的内容。当然，M88k CPU 架构的实现还不支持所有的特性，例如浮点寄存器。然而，你现在已经了解了 LLVM 后端开发中应用的所有重要概念，有了这些，你将能够添加任何缺失的部分！

# 总结

在本章中，你学会了如何为 LLVM 开发一个新的后端目标。你首先收集了所需的文档，并通过增强`Triple`类使 LLVM 意识到了新的架构。文档还包括 ELF 文件格式的重定位定义，你还为 LLVM 添加了对此的支持。

你了解了目标描述包含的不同部分，并使用从中生成的 C++源代码，学会了如何实现指令选择。为了输出生成的代码，你开发了一个汇编打印程序，并学会了需要哪些支持类来写入目标文件。你还学会了如何添加反汇编支持，用于将目标文件转换回汇编文本。最后，你扩展了构建系统，将新的目标包含在构建中。

现在你已经具备了在自己的项目中以创造性方式使用 LLVM 所需的一切。LLVM 生态系统非常活跃，不断添加新特性，所以一定要关注所有的发展！

作为一个编译器开发者，能够写关于 LLVM 并在过程中发现一些新特性对我来说是一种乐趣。享受 LLVM 的乐趣吧！
