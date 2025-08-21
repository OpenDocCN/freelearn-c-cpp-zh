# 第九章：指令选择

到目前为止使用的 LLVM IR 仍然需要转换为机器指令。这称为**指令选择**，通常缩写为**ISel**。指令选择是目标后端的重要部分，LLVM 有三种不同的选择指令的方法：选择 DAG，快速指令选择和全局指令选择。

在本章中，您将学习以下主题：

+   了解 LLVM 目标后端结构，介绍了目标后端执行的任务，并检查了要运行的机器传递。

+   使用**机器 IR**（**MIR**）来测试和调试后端，这有助于在指定的传递后输出 MIR 并在 MIR 文件上运行传递。

+   指令选择的工作方式，您将了解 LLVM 执行指令选择的不同方式。

+   支持新的机器指令，其中您添加一个新的机器指令并使其可用于指令选择。

通过本章结束时，您将了解目标后端的结构以及指令选择的工作方式。您还将获得向汇编程序和指令选择中添加当前不受支持的机器指令的知识，以及如何测试您的添加。

# 技术要求

要查看图形可视化，您必须安装**Graphviz**软件，可从[`graphviz.org/`](https://graphviz.org/)下载。源代码可在[`gitlab.com/graphviz/graphviz/`](http://gitlab.com/graphviz/graphviz/)上找到。

本章的源代码可在[`github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter09`](https://github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter09)上找到

您可以在[`bit.ly/3nllhED`](https://bit.ly/3nllhED)上找到代码演示视频

# 了解 LLVM 目标后端结构

在优化了 LLVM IR 之后，选择的 LLVM 目标用于从中生成机器代码。在目标后端中执行以下任务，包括：

1.  用于指令选择的**有向无环图**（**DAG**），通常称为**SelectionDAG**，被构建。

1.  选择与 IR 代码对应的机器指令。

1.  选择的机器指令按最佳顺序排列。

1.  虚拟寄存器被机器寄存器替换。

1.  向函数添加序言和尾声代码。

1.  基本块按最佳顺序排列。

1.  运行特定于目标的传递。

1.  发出目标代码或汇编。

所有这些步骤都被实现为机器函数传递，派生自`MachineFunctionPass`类。这是`FunctionPass`类的子类，是旧的 pass 管理器使用的基类之一。截至 LLVM 12，将机器函数传递转换为新的 pass 管理器仍然是一个正在进行中的工作。

在所有这些步骤中，LLVM 指令都会经历转换。在代码级别，LLVM IR 指令由`Instruction`类的实例表示。在指令选择阶段，它被转换为`MachineInstr`实例。这是一个更接近实际机器级别的表示。它已经包含了对目标有效的指令，但仍然在虚拟寄存器上运行（直到寄存器分配），并且还可以包含某些伪指令。指令选择后的传递会对此进行细化，最终创建一个`MCInstr`实例，这是真实机器指令的表示。`MCInstr`实例可以写入对象文件或打印为汇编代码。

要探索后端传递，您可以创建一个包含以下内容的小型 IR 文件：

```cpp
define i16 @sum(i16 %a, i16 %b) {
  %res = add i16 %a, 3
  ret i16 %res
}
```

将此代码保存为`sum.ll`。使用 LLVM 静态编译器`llc`为 MIPS 架构编译它。这个工具将 LLVM IR 编译成汇编文本或目标文件。可以使用`–mtriple`选项在命令行上覆盖目标平台的编译。使用`–debug-pass=Structure`选项调用`llc`工具：

```cpp
$ llc -mtriple=mips-linux-gnu -debug-pass=Structure < sum.ll
```

除了生成的汇编代码，你还会看到一个要运行的机器 pass 的长列表。其中，`MIPS DAG->DAG Pattern Instruction Selection` pass 执行指令选择，`Mips Delay Slot Filler`是一个特定于目标的 pass，而在清理之前的最后一个 pass，`Mips Assembly Printer`，负责打印汇编代码。在所有这些 pass 中，指令选择 pass 是最有趣的，我们将在下一节详细讨论。

# 使用 MIR 测试和调试后端

你在前面的部分看到目标后端运行了许多 pass。然而，这些 pass 中的大多数并不是在 LLVM IR 上运行的，而是在 MIR 上运行的。这是指令的一个与目标相关的表示，因此比 LLVM IR 更低级。它仍然可以包含对虚拟寄存器的引用，因此它还不是目标 CPU 的纯指令。

要查看 IR 级别的优化，例如，可以告诉`llc`在每个 pass 之后转储 IR。这在后端的机器 pass 中不起作用，因为它们不在 IR 上工作。相反，MIR 起到了类似的作用。

MIR 是当前模块中机器指令当前状态的文本表示。它利用了 YAML 格式，允许序列化和反序列化。基本思想是你可以在某个点停止 pass 管道并以 YAML 格式检查状态。你也可以修改 YAML 文件，或者创建你自己的文件，并传递它，并检查结果。这样可以方便地进行调试和测试。

让我们来看看 MIR。使用`llc`工具和`--stop-after=finalize-isel`选项以及之前使用的测试输入文件运行：

```cpp
$ llc -mtriple=mips-linux-gnu \
        -stop-after=finalize-isel < sum.ll
```

这指示`llc`在指令选择完成后转储 MIR。缩短的输出看起来像这样：

```cpp
---
name:                 sum
body:                  |
  bb.0 (%ir-block.0):
     liveins: $a0, $a1
     %1:gpr32 = COPY $a1
     %0:gpr32 = COPY $a0
     %2:gpr32 = ADDu %0, %1
     $v0 = COPY %2
     RetRA implicit $v0
... 
```

有几个属性你立即注意到。首先，有一些虚拟寄存器，比如`%0`和实际的机器寄存器，比如`$a0`。这是由 ABI 降级引起的。为了在不同的编译器和语言之间具有可移植性，函数遵循调用约定的一部分，这是`$a0`的一部分。因为 MIR 输出是在指令选择之后但是在寄存器分配之前生成的，所以你仍然可以看到虚拟寄存器的使用。

在 LLVM IR 中的`add`指令，MIR 文件中使用的是机器指令`ADDu`。你还可以看到虚拟寄存器有一个寄存器调用附加，这种情况下是`gpr32`。在 MIPS 架构上没有 16 位寄存器，因此必须使用 32 位寄存器。

`bb.0`标签指的是第一个基本块，标签后面的缩进内容是基本块的一部分。第一条语句指定了进入基本块时活跃的寄存器。之后是指令。在这种情况下，只有`$a0`和`$a1`，两个参数，在进入时是活跃的。

MIR 文件中还有很多其他细节。你可以在 LLVM MIR 文档中阅读有关它们的内容[`llvm.org/docs/MIRLangRef.html`](https://llvm.org/docs/MIRLangRef.html)。

你遇到的一个问题是如何找出一个 pass 的名称，特别是如果你只需要检查该 pass 之后的输出而不是积极地在其上工作。当使用`-debug-pass=Structure`选项与`llc`一起时，激活 pass 的选项被打印在顶部。例如，如果你想在`Mips Delay Slot Filler` pass 之前停止，那么你需要查看打印出的列表，并希望找到`-mips-delay-slot-filler`选项，这也会给出 pass 的名称。

MIR 文件格式的主要应用是帮助测试目标后端中的机器传递。使用`llc`和`--stop-after`选项，您可以在指定的传递之后获得 MIR。通常，您将使用这个作为您打算测试用例的基础。您首先注意到的是 MIR 输出非常冗长。例如，许多字段是空的。为了减少这种混乱，您可以在`llc`命令行中添加`-simplify-mir`选项。

您可以根据需要保存和更改 MIR 以进行测试。`llc`工具可以运行单个传递，这非常适合使用 MIR 文件进行测试。假设您想要测试`MIPS Delay Slot Filler`传递。延迟槽是 RISC 架构（如 MIPS 或 SPARC）的一个特殊属性：跳转后的下一条指令总是被执行。因此，编译器必须确保每次跳转后都有一个合适的指令，这个传递就是执行这个任务的。

我们在运行传递之前生成 MIR：

```cpp
$ llc -mtriple=mips-linux-gnu \
        -stop-before=mips-delay-slot-filler -simplify-mir \
        < sum.ll  >delay.mir
```

输出要小得多，因为我们使用了`-simplify-mir`选项。函数的主体现在是以下内容：

```cpp
body:                  |
  bb.0 (%ir-block.0):
     liveins: $a0, $a1
     renamable $v0 = ADDu killed renamable $a0,
                             killed renamable $a1
     PseudoReturn undef $ra, implicit $v0
```

最重要的是，您将看到`ADDu`指令，后面是返回的伪指令。

使用`delay.ll`文件作为输入，我们现在运行延迟槽填充器传递：

```cpp
$ llc -mtriple=mips-linux-gnu \
        -run-pass=mips-delay-slot-filler -o - delay.mir
```

现在将输出中的函数与之前的函数进行比较：

```cpp
body:                  |
  bb.0 (%ir-block.0):
     PseudoReturn undef $ra, implicit $v0 {
        renamable $v0 = ADDu killed renamable $a0,
                                killed renamable $a1
     }
```

您会看到`ADDu`和返回的伪指令的顺序已经改变，`ADDu`指令现在嵌套在返回内部：传递将`ADDu`指令标识为适合延迟槽的指令。

如果延迟槽的概念对您来说是新的，您还会想要查看生成的汇编代码，您可以使用`llc`轻松生成：

```cpp
$ llc -mtriple=mips-linux-gnu < sum.ll
```

输出包含很多细节，但是通过基本块的`bb.0`名称的帮助，您可以轻松地定位生成的汇编代码：

```cpp
# %bb.0:
           jr        $ra
           addu     $2, $4, $5
```

确实，指令的顺序改变了！

掌握了这些知识，我们来看一下目标后端的核心，并检查 LLVM 中如何执行机器指令选择。

# 指令选择的工作原理

LLVM 后端的任务是从 LLVM IR 创建机器指令。这个过程称为**指令选择**或**降低**。受到尽可能自动化这项任务的想法的启发，LLVM 开发人员发明了 TableGen 语言来捕获目标描述的所有细节。我们首先看一下这种语言，然后再深入研究指令选择算法。

## 在 TableGen 语言中指定目标描述

机器指令有很多属性：汇编器和反汇编器使用的助记符、在内存中表示指令的位模式、输入和输出操作数等。LLVM 开发人员决定将所有这些信息都捕获在一个地方，即`.td`后缀。

原则上，TableGen 语言非常简单。您所能做的就是定义记录。`Register`类定义了寄存器的共同属性，您可以为寄存器`R0`定义一个具体的记录：

```cpp
class Register {
  string name;
}
def R0 : Register {
  let name = "R0";
  string altName = "$0";
}
```

您可以使用`let`关键字来覆盖一个值。

TableGen 语言有很多语法糖，使处理记录变得更容易。例如，一个类可以有一个模板参数：

```cpp
class Register<string n> {
  string name = n;
}
def R0 : Register<"R0"> {
  string altName = "$0";
}
```

TableGen 语言是静态类型的，您必须指定每个值的类型。一些支持的类型如下：

+   `位`：一个单独的位

+   `int`：64 位整数值

+   `bits<n>`：由*n*位组成的整数类型

+   `string`：一个字符字符串

+   `list<t>`：类型为`t`的元素列表

+   `dag`：**有向无环图**（**DAG**；指令选择使用）

类的名称也可以用作类型。例如，`list<Register>`指定了`Register`类的元素列表。

该语言允许使用`include`关键字包含其他文件。对于条件编译，支持预处理指令`#define`、`#ifdef`和`#ifndef`。

LLVM 中的 TableGen 库可以解析用 TableGen 语言编写的文件，并创建记录的内存表示。您可以使用这个库来创建自己的生成器。

LLVM 自带了一个名为`llvm-tblgen`的生成器工具和一些`.td`文件。后端的目标描述首先包括`llvm/Target/Target.td`文件。该文件定义了诸如`Register`、`Target`或`Processor`之类的类。`llvm-tblgen`工具了解这些类，并从定义的记录生成 C++代码。

让我们以 MIPS 后端为例来看一下。目标描述在`llvm/lib/Target/Mips`文件夹中的`Mips.td`文件中。该文件包括了最初提到的`Target.td`文件。它还定义了目标特性，例如：

```cpp
def FeatureMips64r2
  : SubtargetFeature<"mips64r2", "MipsArchVersion", 
                     "Mips64r2", "Mips64r2 ISA Support",
                     [FeatureMips64, FeatureMips32r2]>;
```

这些特性后来被用来定义 CPU 模型，例如：

```cpp
def : Proc<"mips64r2", [FeatureMips64r2]>;
```

其他定义寄存器、指令、调度模型等的文件也包括在内。

`llvm-tblgen`工具可以显示由目标描述定义的记录。如果你在`build`目录中，那么以下命令将在控制台上打印记录：

```cpp
$ bin/llvm-tblgen \
  -I../llvm-project/llvm/lib/Target/Mips/ \
  -I../llvm-project/llvm/include \
  ../llvm-project/llvm/lib/Target/Mips/Mips.td
```

与 Clang 一样，`-I`选项会在包含文件时添加一个目录进行搜索。查看记录对于调试很有帮助。该工具的真正目的是从记录生成 C++代码。例如，使用`-gen-subtarget`选项，将向控制台发出解析`llc`的`-mcpu=`和`-mtarget=`选项所需的数据：

```cpp
$ bin/llvm-tblgen \
  -I../llvm-project/llvm/lib/Target/Mips/ \
  -I../llvm-project/llvm/include \
  ../llvm-project/llvm/lib/Target/Mips/Mips.td \
  -gen-subtarget
```

将该命令生成的代码保存到一个文件中，并探索特性和 CPU 在生成的代码中的使用方式！

指令的编码通常遵循一些模式。因此，指令的定义被分成了定义位编码和指令具体定义的类。MIPS 指令的编码在文件`llvm/Target/Mips/MipsInstrFormats.td`中。让我们来看一下`ADD_FM`格式的定义：

```cpp
class ADD_FM<bits<6> op, bits<6> funct> : StdArch {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<32> Inst;
  let Inst{31-26} = op;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = funct;
}
```

在记录主体中，定义了几个新的位字段：`rd`、`rs`等。它们用于覆盖`Inst`字段的部分内容，该字段保存指令的位模式。`rd`、`rs`和`rt`位字段编码了指令操作的寄存器，而`op`和`funct`参数表示操作码和函数编号。`StdArch`超类只添加了一个字段，说明该格式遵循标准编码。

MIPS 目标中的大多数指令编码不涉及 DAG 节点，也不指定汇编助记符。为此定义了一个单独的类。MIPS 架构中的一条指令是`nor`指令，它计算第一个和第二个输入寄存器的按位或，反转结果的位，并将结果赋给输出寄存器。这条指令有几个变体，以下的`LogicNOR`类有助于避免多次重复相同的定义：

```cpp
class LogicNOR<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rs, RO:$rt),
            !strconcat(opstr, "\t$rd, $rs, $rt"),
            [(set RO:$rd, (not (or RO:$rs, RO:$rt)))],
            II_NOR, FrmR, opstr> {
  let isCommutable = 1;
}
```

哇，记录这个简单的概念现在看起来很复杂。让我们剖析一下这个定义。这个类派生自`InstSE`类，这个类总是用于具有标准编码的指令。如果你继续跟踪超类层次结构，你会看到这个类派生自`Instruction`类，这是一个预定义的类，表示目标的指令。`(outs RO:$rd)`参数将最终指令的结果定义为 DAG 节点。`RO`部分是指`LogicNOR`类的同名参数，表示寄存器操作数。`$rd`是要使用的寄存器。这是稍后将放入指令编码中的值，在`rd`字段中。第二个参数定义了指令将操作的值。总之，这个类是用于操作三个寄存器的指令。`!strconcat(opstr, "\t$rd, $rs, $rt")`参数组装了指令的文本表示。`!strconcat`操作符是 TableGen 中预定义的功能，用于连接两个字符串。你可以在 TableGen 程序员指南中查找所有预定义的操作符：[`llvm.org/docs/TableGen/ProgRef.html`](https://llvm.org/docs/TableGen/ProgRef.html)。

它遵循一个模式定义，类似于`nor`指令的文本描述，并描述了这个指令的计算。模式的第一个元素是操作，后面是一个逗号分隔的操作数列表。操作数指的是 DAG 参数中的寄存器名称，并且还指定了 LLVM IR 值类型。LLVM 有一组预定义的操作符，比如`add`和`and`，可以在模式中使用。这些操作符属于`SDNode`类，也可以用作参数。你可以在文件`llvm/Target/TargetSelectionDAG.td`中查找预定义的操作符。

`II_NOR`参数指定了调度模型中使用的行程类别，`FrmR`参数是一个定义的值，用于识别此指令格式。最后，`opstr`助记符被传递给超类。这个类的主体非常简单：它只是指定`nor`操作是可交换的，这意味着操作数的顺序可以交换。

最后，这个类用于定义一个指令的记录，例如，用于 64 位模式下的`nor`指令：

```cpp
def NOR64 : LogicNOR<"nor", GPR64Opnd>, ADD_FM<0, 0x27>,                                    
                              GPR_64;
```

这是最终的定义，可以从`def`关键字中识别出来。它使用`LogicNOR`类来定义 DAG 操作数和模式，使用`ADD_FM`类来指定二进制指令编码。额外的`GPR_64`谓词确保只有在 64 位寄存器可用时才使用这个指令。

开发人员努力避免多次重复定义，一个经常使用的方法是使用`multiclass`类。`multiclass`类可以一次定义多个记录。

例如，MIPS CPU 的浮点单元可以执行单精度或双精度浮点值的加法。这两个指令的定义非常相似，因此定义了一个`multiclass`类，一次创建两个指令：

```cpp
multiclass ADDS_M<…> {
  def _D32 : ADDS_FT<…>, FGR_32;
  def _D64 : ADDS_FT<…>, FGR_64;
}
```

`ADDS_FT`类定义了指令格式，类似于`LogicNOR`类。`FGR_32`和`FGR_64`谓词用于在编译时决定可以使用哪个指令。重要的部分是定义了`_D32`和`_D64`记录。这些是记录的模板。然后使用`defm`关键字定义指令记录：

```cpp
defm FADD : ADDS_M<…>;
```

这一次同时定义了多类中的两个记录，并为它们分配了名称`FADD_D32`和`FADD_D64`。这是避免代码重复的一种非常强大的方式，它经常在目标描述中使用，但结合其他 TableGen 功能，可能会导致非常晦涩的定义。

有了目标描述组织的知识，我们现在可以在下一节中探索指令选择。

## 使用选择 DAG 进行指令选择

LLVM 将 IR 转换为机器指令的标准方式是通过 DAG。使用目标描述中提供的模式匹配和自定义代码，IR 指令被转换为机器指令。这种方法并不像听起来那么简单：IR 大多是与目标无关的，并且可能包含目标不支持的数据类型。例如，代表单个位的`i1`类型在大多数目标上都不是有效的类型。

selectionDAG 由`SDNode`类型的节点组成，在文件`llvm/CodeGen/SelectionDAGNodes.h`中定义。节点表示的操作称为`OpCode`，目标独立代码在文件`llvm/CodeGen/ISDOpcodes.h`中定义。除了操作，节点还存储操作数和它产生的值。

节点的值和操作数形成数据流依赖关系。控制流依赖由链边表示，具有特殊类型`MVT::Other`。这使得可以保持具有副作用的指令的顺序，例如，加载指令。

使用选择 DAG 进行指令选择的步骤如下：

1.  如何跟踪指令选择过程

1.  DAG 被优化了。

1.  DAG 中的类型被合法化了。

1.  指令被选择了。

1.  DAG 中的操作被合法化了。

1.  DAG 被优化了。

1.  指令被排序了。

1.  就像上一节的 MIR 输出中一样，您在这里看到`CopyFromReg`指令，它们将 ABI 使用的寄存器的内容传输到虚拟节点。由于示例使用 16 位值，但 MIPS 架构仅对 32 位值有本机支持，因此需要`truncate`节点。`add`操作是在 16 位虚拟寄存器上执行的，并且结果被扩展并返回给调用者。对于上述每个步骤，都会打印这样的部分。

让我们看看如何跟踪每个步骤对选择 DAG 的更改。

### ![图 9.1 - 为 sum.ll 文件构建的选择 DAG

您可以以两种不同的方式看到指令选择的工作。如果将`-debug-only=isel`选项传递给`llc`工具，则每个步骤的结果将以文本格式打印出来。如果您需要调查为什么选择了机器指令，这将是一个很大的帮助。例如，运行以下命令以查看“Understanding the LLVM target backend structure”部分的`sum.ll`文件的输出：

```cpp
$ llc -mtriple=mips-linux-gnu -debug-only=isel < sum.ll
```

这打印了大量信息。在输出顶部，您可以看到输入的初始创建的 DAG 的描述：

```cpp
Initial selection DAG: %bb.0 'sum:'
SelectionDAG has 12 nodes:
  t0: ch = EntryToken
              t2: i32,ch = CopyFromReg t0, Register:i32 %0
           t5: i16 = truncate t2
              t4: i32,ch = CopyFromReg t0, Register:i32 %1
           t6: i16 = truncate t4
        t7: i16 = add t5, t6
     t8: i32 = any_extend t7
  t10: ch,glue = CopyToReg t0, Register:i32 $v0, t8
  t11: ch = MipsISD::Ret t10, Register:i32 $v0, t10:1 
```

DAG 被构建了。

LLVM 还可以借助*Graphviz*软件生成选择 DAG 的可视化。如果将`–view-dag-combine1-dags`选项传递给`llc`工具，则会打开一个窗口显示构建的 DAG。例如，使用前面的小文件运行`llc`：

```cpp
$ llc -mtriple=mips-linux-gnu  –view-dag-combine1-dags sum.ll
```

DAG 被优化了。

在 Windows PC 上运行，您将看到 DAG：

在 Windows PC 上运行，您将看到 DAG：

图 9.1 - 为 sum.ll 文件构建的选择 DAG

确保文本表示和此图包含相同的信息。`EntryToken`是 DAG 的起点，`GraphRoot`是最终节点。控制流的链用蓝色虚线箭头标记。黑色箭头表示数据流。红色箭头将节点粘合在一起，防止重新排序。即使对于中等大小的函数，图可能会变得非常大。它不包含比带有`-debug-only=isel`选项的文本输出更多或其他信息，只是呈现更加舒适。您还可以在其他时间生成图，例如：

+   将`--view-legalize-types-dags`选项添加到类型合法化之前查看 DAG。

+   添加`–view-isel-dags`选项以查看选择指令。

您可以使用`--help-hidden`选项查看查看 DAG 的所有可用选项。由于 DAG 可能变得庞大和混乱，您可以使用`-filter-view-dags`选项将渲染限制为一个基本块。

### 检查指令选择

了解如何可视化 DAG 后，我们现在可以深入了解细节。选择 DAG 是从 IR 构建的。对于 IR 中的每个函数，`SelectionDAGBuilder`类通过`SelectionDAGBuilder`类填充`SelectionDAG`类的实例。在此步骤中没有进行特殊优化。尽管如此，目标需要提供一些函数来降低调用、参数处理、返回跳转等。为此，目标必须实现`TargetLowering`接口。在目标的文件夹中，源代码通常在`XXXISelLowering.h`和`XXXISelLowering.cpp`文件中。`TargetLowering`接口的实现提供了指令过程所需的所有信息，例如目标上支持的数据类型和操作。

优化步骤会运行多次。优化器执行简单的优化，例如识别支持这些操作的目标上的旋转。这里的原理是产生一个清理过的 DAG，从而简化其他步骤。

在类型合法化步骤中，目标不支持的类型将被替换为支持的类型。例如，如果目标本机只支持 32 位宽整数，则较小的值必须通过符号或零扩展转换为 32 位。这称为`TargetLowering`接口。类型合法化后，选择 DAG 对`sum.ll`文件具有以下文本表示：

```cpp
Optimized type-legalized selection DAG: %bb.0 'sum:'
SelectionDAG has 9 nodes:
  t0: ch = EntryToken
        t2: i32,ch = CopyFromReg t0, Register:i32 %0
        t4: i32,ch = CopyFromReg t0, Register:i32 %1
     t12: i32 = add t2, t4
  t10: ch,glue = CopyToReg t0, Register:i32 $v0, t12
  t11: ch = MipsISD::Ret t10, Register:i32 $v0, t10:1
```

如果将此与最初构建的 DAG 进行比较，那么这里只使用了 32 位寄存器。16 位值被提升，因为本机只支持 32 位值。

操作合法化类似于类型合法化。这一步是必要的，因为并非所有操作都可能被目标支持，或者即使目标本机支持某种类型，也可能并非所有操作都有效。例如，并非所有目标都有用于人口统计的本机指令。在这种情况下，该操作将被一系列操作替换以实现功能。如果类型不适合操作，则可以将类型提升为更大的类型。后端作者还可以提供自定义代码。如果合法化操作设置为`Custom`，则将为这些操作调用`TargetLowering`类中的`LowerOperation()`方法。该方法必须创建操作的合法版本。在`sum.ll`示例中，`add`操作已经是合法的，因为平台支持两个 23 位寄存器的加法，而且没有改变。

在类型和操作被合法化之后，指令选择就会发生。选择的大部分部分是自动化的。请记住前一节中，您在指令描述中提供了一个模式。从这些描述中，`llvm-tblgen`工具生成了一个模式匹配器。基本上，模式匹配器试图找到与当前 DAG 节点匹配的模式。然后选择与该模式相关联的指令。模式匹配器被实现为字节码解释器。解释器的可用代码在`llvm/CodeGen/SelectionDAGISel.h`头文件中定义。`XXXISelDAGToDAG`类实现了目标的指令选择。对于每个 DAG 节点，都会调用`Select()`方法。默认情况下会调用生成的匹配器，但您也可以为它未处理的情况添加代码。

值得注意的是，选择 DAG 节点与所选指令之间没有一对一的关系。DAG 节点可以扩展为多条指令，而多个 DAG 节点可以合并为单条指令。前者的一个例子是合成立即值。特别是在 RISC 架构上，立即值的位长度受限。32 位目标可能仅支持 16 位长度的嵌入式立即值。要执行需要 32 位常量值的操作，通常会将其拆分为两个 16 位值，然后生成使用这两个 16 位值的两个或更多指令。在 MIPS 目标中，您会发现这方面的模式。位域指令是后一种情况的常见例子：`and`，`or`和`shift` DAG 节点的组合通常可以匹配到特殊的位域指令，从而只需一条指令即可处理两个或更多 DAG 节点。

通常，您可以在目标描述中指定一个模式，以组合两个或多个 DAG 节点。对于更复杂的情况，这些情况不容易用模式处理，您可以标记顶部节点的操作，需要特殊的 DAG 组合处理。对于这些节点，在`XXXISelLowering`类中调用`PerformDAGCombine（）`方法。然后，您可以检查任意复杂的模式，如果找到匹配，那么您可以返回表示组合 DAG 节点的操作。在运行 DAG 节点的生成匹配器之前调用此方法。

您可以在`sum.ll`文件的打印输出中跟踪指令选择过程。对于`add`操作，您会在那里找到以下行：

```cpp
ISEL: Starting selection on root node: t12: i32 = add t2, t4
ISEL: Starting pattern match
  Initial Opcode index to 27835
  …
  Morphed node: t12: i32 = ADDu t2, t4
ISEL: Match complete!
```

索引号指向生成匹配器的数组。起始索引为`27835`（一个可以在发布版本之间更改的任意值），经过一些步骤后，选择了`ADDu`指令。

遵循模式匹配

如果遇到模式问题，您还可以通过阅读生成的字节码来追踪匹配过程。您可以在`build`目录中的`lib/Target/XXX/XXXGenDAGIsel.inc`文件中找到源代码。您可以在文本编辑器中打开文件，并在先前的输出中搜索索引。每行都以索引号为前缀，因此您可以轻松找到数组中的正确位置。使用的谓词也会以注释的形式打印出来，因此它们可以帮助您理解为什么某个特定的模式未被选择。

### 将 DAG 转换为指令序列

在指令选择之后，代码仍然是一个图。这种数据结构需要被展平，这意味着指令必须按顺序排列。图包含数据和控制流依赖关系，但总是有几种可能的方式来安排指令，以满足这些依赖关系。我们希望的是一种最大程度利用硬件的顺序。现代硬件可以并行发出多条指令，但总是有限制。这种限制的一个简单例子是一个指令需要另一个指令的结果。在这种情况下，硬件可能无法发出两条指令，而是按顺序执行指令。

您可以向目标描述添加调度模型，描述可用的单元及其属性。例如，如果 CPU 有两个整数算术单元，那么这些信息就被捕捉在模型中。对于每个指令，有必要知道模型的哪个部分被使用。有不同的方法来做到这一点。较新的、推荐的方法是使用所谓的机器指令调度器来定义调度模型。为此，您需要为目标描述中的每个子目标定义一个`SchedMachineModel`记录。基本上，模型由指令和处理器资源的输入和输出操作数的定义组成。然后，这两个定义与延迟值一起关联。您可以在`llvm/Target/TargetSched.td`文件中查找此模型的预定义类型。查看 Lanai 目标以获取一个非常简单的模型，并在 SystemZ 目标中获取一个复杂的调度模型。

还有一个基于所谓行程的较旧模型。使用这个模型，您将处理器单元定义为`FuncUnit`记录。使用这样一个单元的步骤被定义为`InstrStage`记录。每个指令都与一个行程类相关联。对于每个行程类，定义了使用的处理器流水线由`InstrStage`记录组成，以及执行所需的处理器周期数。您可以在`llvm/Target/TargetItinerary.td`文件中找到行程模型的预定义类型。

一些目标同时使用这两种模型。一个原因是由于开发历史。基于行程的模型是最早添加到 LLVM 中的，目标开始使用这个模型。当新的机器指令调度器在 5 年多以后添加时，没有人关心足够迁移已经存在的模型。另一个原因是，使用行程模型不仅可以对使用多个处理器单元的指令进行建模，还可以指定在哪些周期使用这些单元。然而，这种细节级别很少需要，如果需要，那么可以参考机器指令调度器模型来定义行程，基本上将这些信息也引入到新模型中。

如果存在，调度模型用于以最佳方式排序指令。在这一步之后，DAG 不再需要，并被销毁。

使用选择 DAG 进行指令选择几乎可以得到最佳结果，但在运行时和内存使用方面会付出代价。因此，开发了替代方法，我们将在下一节中进行讨论。在下一节中，我们将看一下快速指令选择方法。

## 快速指令选择 - FastISel

使用选择 DAG 进行指令选择会消耗编译时间。如果您正在开发一个应用程序，那么编译器的运行时很重要。您也不太关心生成的代码，因为更重要的是发出完整的调试信息。因此，LLVM 开发人员决定实现一个特殊的指令选择器，它具有快速的运行时，但生成的代码不太优化，并且仅用于`-O0`优化级别。这个组件称为快速指令选择，简称**FastIsel**。

实现在`XXXFastISel`类中。并非每个目标都支持这种指令选择方法，如果是这种情况，选择 DAG 方法也用于`-O0`。实现很简单：从`FastISel`类派生一个特定于目标的类，并实现一些方法。TableGen 工具从目标描述中生成了大部分所需的代码。然而，需要一些工作来实现这个指令选择器。一个根本原因是你需要正确地获取调用约定，这通常是复杂的。

MIPS 目标具有快速指令选择的实现。您可以通过向`llc`工具传递`-fast-isel`选项来启用快速指令选择。使用第一节中的`sum.ll`示例文件，调用如下：

```cpp
$ llc -mtriple=mips-linux-gnu -fast-isel –O0 sum.ll
```

快速指令选择运行非常快，但它是一条完全不同的代码路径。一些 LLVM 开发人员决定寻找一个既能快速运行又能产生良好代码的解决方案，目标是在未来替换选择`dag`和快速指令选择器。我们将在下一节讨论这种方法。

## 新的全局指令选择 - GlobalISel

使用选择 dag，我们可以生成相当不错的机器代码。缺点是它是一个非常复杂的软件。这意味着它很难开发、测试和维护。快速指令选择工作迅速，复杂性较低，但不能产生良好的代码。除了由 TableGen 生成的代码外，这两种方法几乎没有共享代码。

我们能否兼得两全？一种指令选择算法，既快速，易于实现，又能产生良好的代码？这就是向 LLVM 框架添加另一种指令选择算法 - 全局指令选择的动机。短期目标是首先替换 FastISel，长期目标是替换选择 dag。

全局指令选择采用的方法是建立在现有基础设施之上。整个任务被分解为一系列机器函数传递。另一个主要的设计决定是不引入另一种中间表示，而是使用现有的`MachineInstr`类。但是，会添加新的通用操作码。

当前的步骤顺序如下：

1.  `IRTranslator` pass 使用通用操作码构建初始机器指令。

1.  `Legalizer` pass 在一步中使类型和操作合法化。这与选择 dag 不同，后者需要两个不同的步骤。真实的 CPU 架构有时很奇怪，可能只支持某种数据类型的一条指令。选择 dag 处理这种情况不好，但在全局指令选择的组合步骤中很容易处理。

1.  生成的机器指令仍然在虚拟寄存器上操作。在`RegBankSelect` pass 中，选择了一个寄存器组。寄存器组代表 CPU 上的寄存器类型，例如通用寄存器。这比目标描述中的寄存器定义更粗粒度。重要的是它将类型信息与指令关联起来。类型信息基于目标中可用的类型，因此这已经低于 LLVM IR 中的通用类型。

1.  此时，已知类型和操作对于目标是合法的，并且每条指令都与类型信息相关联。接下来的`InstructionSelect` pass 可以轻松地用机器指令替换通用指令。

全局指令选择后，通常会运行后端传递，如指令调度、寄存器分配和基本块放置。

全局指令选择已编译到 LLVM 中，但默认情况下未启用。如果要使用它，需要给`llc`传递`-global-isel`选项，或者给`clang`传递`-mllvm global-isel`选项。您可以控制全局指令选择无法处理 IR 构造时的处理方式。当您给`llc`传递`-global-isel-abort=0`选项时，选择 dag 将作为后备。使用`=1`时，应用程序将终止。为了防止这种情况，您可以给`llc`传递`-global-isel-abort=0`选项。使用`=2`时，选择 dag 将作为后备，并打印诊断消息以通知您有关问题。

要将全局指令选择添加到目标，您只需要重写目标的`TargetPassConfig`类中的相应函数。这个类由`XXXTargetMachine`类实例化，并且实现通常可以在同一个文件中找到。例如，您可以重写`addIRTranslator()`方法，将`IRTranslator` pass 添加到目标的机器 pass 中。

开发主要发生在 AArch64 目标上，目前该目标对全局指令选择有最好的支持。许多其他目标，包括 x86 和 Power，也已经添加了对全局指令选择的支持。一个挑战是从表描述中生成的代码并不多，所以仍然有一定量的手动编码需要完成。另一个挑战是目前不支持大端目标，因此纯大端目标（如 SystemZ）目前无法使用全局指令选择。这两个问题肯定会随着时间的推移得到改善。

Mips 目标具有全局指令选择的实现，但有一个限制，即它只能用于小端目标。您可以通过向`llc`工具传递`–global-isel`选项来启用全局指令选择。使用第一节的`sum.ll`示例文件，调用如下：

```cpp
$ llc -mtriple=mipsel-linux-gnu -global-isel sum.ll
```

请注意，目标`mipsel-linux-gnu`是小端目标。使用大端`mips-linux-gnu`目标会导致错误消息。

全局指令选择器比选择 DAG 快得多，并且已经产生了比快速指令选择更高的代码质量。

# 支持新的机器指令

您的目标 CPU 可能具有 LLVM 尚不支持的机器指令。例如，使用 MIPS 架构的制造商经常向核心 MIPS 指令集添加特殊指令。RISC-V 指令集的规范明确允许制造商添加新指令。或者您正在添加一个全新的后端，那么您必须添加 CPU 的指令。在下一节中，我们将为 LLVM 后端的单个新机器指令添加汇编器支持。

## 添加汇编和代码生成的新指令

新的机器指令通常与特定的 CPU 特性相关联。然后，只有在用户使用`--mattr=`选项选择了该特性时，新指令才会被识别。

例如，我们将在 MIPS 后端添加一个新的机器指令。这个虚构的新机器指令首先将两个输入寄存器`$2`和`$3`的值平方，然后将两个平方的和赋给输出寄存器`$1`：

```cpp
sqsumu $1, $2, $3
```

指令的名称是`sqsumu`，源自平方和求和操作。名称中的最后一个`u`表示该指令适用于无符号整数。

我们首先要添加的 CPU 特性称为`sqsum`。这将允许我们使用`--mattr=+sqsum`选项调用`llc`来启用对新指令的识别。

我们将添加的大部分代码位于`llvm/lib/Target/Mips`文件夹中。顶层文件是`Mips.td`。查看该文件，并找到定义各种特性的部分。在这里，您添加我们新特性的定义：

```cpp
def FeatureSQSum
     : SubtargetFeature<"sqsum", "HasSQSum", "true",
                                 "Use square-sum instruction">;
```

`SubtargetFeature`类有四个模板参数。第一个`sqsum`是特性的名称，用于命令行。第二个参数`HasSQSum`是`Subtarget`类中表示此特性的属性的名称。接下来的参数是特性的默认值和描述，用于在命令行上提供帮助。TableGen 会为`MipsSubtarget`类生成基类，该类在`MipsSubtarget.h`文件中定义。在这个文件中，我们在类的私有部分添加新属性，其中定义了所有其他属性：

```cpp
  // Has square-sum instruction.
  bool HasSQSum = false;
```

在公共部分，我们还添加了一个方法来检索属性的值。我们需要这个方法来进行下一个添加：

```cpp
  bool hasSQSum() const { return HasSQSum; }
```

有了这些添加，我们已经能够在命令行上设置`sqsum`功能，尽管没有效果。

为了将新指令与`sqsum`功能关联起来，我们需要定义一个谓词，指示是否选择了该功能。我们将其添加到`MipsInstrInfo.td`文件中，可以是在定义所有其他谓词的部分，也可以简单地添加到末尾：

```cpp
def HasSQSum : Predicate<"Subtarget->hasSQSum()">,
                     AssemblerPredicate<(all_of FeatureSQSum)>;
```

该谓词使用先前定义的`hasSQSum()`方法。此外，`AssemblerPredicate`模板指定了在为汇编器生成源代码时使用的条件。我们只需引用先前定义的功能。

我们还需要更新调度模型。MIPS 目标使用行程表和机器指令调度器。对于行程表模型，在`MipsSchedule.td`文件中为每条指令定义了一个`InstrItinClass`记录。只需在此文件的所有行程表都被定义的部分添加以下行：

```cpp
def II_SQSUMU : InstrItinClass;
```

我们还需要提供有关指令成本的详细信息。通常，您可以在 CPU 的文档中找到这些信息。对于我们的指令，我们乐观地假设它只需要在 ALU 中一个周期。这些信息被添加到同一文件中的`MipsGenericItineraries`定义中：

```cpp
InstrItinData<II_SQSUMU, [InstrStage<1, [ALU]>]>
```

有了这个，基于行程表的调度模型的更新就完成了。MIPS 目标还在`MipsScheduleGeneric.td`文件中定义了一个基于机器指令调度器模型的通用调度模型。因为这是一个涵盖所有指令的完整模型，我们还需要添加我们的指令。由于它是基于乘法的，我们只需扩展`MULT`和`MULTu`指令的现有定义：

```cpp
def : InstRW<[GenericWriteMul], (instrs MULT, MULTu, SQSUMu)>;
```

MIPS 目标还在`MipsScheduleP5600.td`文件中为 P5600 CPU 定义了一个调度模型。显然，我们的新指令在这个目标上不受支持，所以我们将其添加到不支持的功能列表中：

```cpp
list<Predicate> UnsupportedFeatures = [HasSQSum, HasMips3, … 
```

现在我们准备在`Mips64InstrInfo.td`文件的末尾添加新指令。TableGen 定义总是简洁的，因此我们对其进行分解。该定义使用 MIPS 目标描述中的一些预定义类。我们的新指令是一个算术指令，并且按设计，它适用于`ArithLogicR`类。第一个参数`"sqsumu"`指定了指令的汇编助记符。下一个参数`GPR64Opnd`表示指令使用 64 位寄存器作为操作数，接下来的`1`参数表示操作数是可交换的。最后，为指令给出了一个行程表。`ADD_FM`类用于指定指令的二进制编码。对于真实的指令，必须根据文档选择参数。然后是`ISA_MIPS64`谓词，指示指令适用于哪个指令集。最后，我们的`SQSUM`谓词表示只有在启用我们的功能时指令才有效。完整的定义如下：

```cpp
def SQSUMu  : ArithLogicR<"sqsumu", GPR64Opnd, 1, II_SQSUMU>,
                  ADD_FM<0x1c, 0x28>, ISA_MIPS64, SQSUM
```

如果您只想支持新指令，那么这个定义就足够了。在这种情况下，请确保用 `;` 结束定义。通过添加选择 DAG 模式，您可以使指令可用于代码生成器。该指令使用两个操作寄存器 `$rs` 和 `$rt`，以及目标寄存器 `$rd`，这三个寄存器都由 `ADD_FM` 二进制格式类定义。理论上，要匹配的模式很简单：使用 `mul` 乘法运算符对每个寄存器的值进行平方，然后使用 `add` 运算符将两个乘积相加，并赋值给目标寄存器 `$rd`。模式变得有点复杂，因为在 MIPS 指令集中，乘法的结果存储在一个特殊的寄存器对中。为了可用，结果必须移动到通用寄存器中。在操作合法化期间，通用的 `mul` 运算符被替换为 MIPS 特定的 `MipsMult` 操作进行乘法，以及 `MipsMFLO` 操作将结果的低位部分移动到通用寄存器中。在编写模式时，我们必须考虑到这一点，模式如下所示：

```cpp
{
  let Pattern = [(set GPR64Opnd:$rd,
                              (add (MipsMFLO (MipsMult   
                                GPR64Opnd:$rs, 

                                GPR64Opnd:$rs)),
                                      (MipsMFLO (MipsMult 
                                        GPR64Opnd:$rt, 

                                        GPR64Opnd:$rt)))
                                )];
}
```

如*使用选择 DAG 进行指令选择*部分所述，如果此模式与当前 DAG 节点匹配，则会选择我们的新指令。由于 `SQSUM` 谓词，只有在激活 `sqsum` 功能时才会发生这种情况。让我们用一个测试来检查一下！

## 测试新指令

如果您扩展了 LLVM，那么最好的做法是使用自动化测试来验证。特别是如果您想将您的扩展贡献给 LLVM 项目，那么就需要良好的测试。

在上一节中添加了一个新的机器指令后，我们必须检查两个不同的方面：

+   首先，我们必须验证指令编码是否正确。

+   其次，我们必须确保代码生成按预期工作。

LLVM 项目使用 `llvm-mc` 工具。除了其他任务，此工具可以显示指令的编码。为了进行临时检查，您可以运行以下命令来显示指令的编码：

```cpp
$ echo "sqsumu \$1,\$2,\$3" | \
  llvm-mc --triple=mips64-linux-gnu -mattr=+sqsum \
              --show-encoding
```

这已经显示了部分输入和在自动化测试用例中运行的命令。为了验证结果，您可以使用 `FileCheck` 工具。`llvm-mc` 的输出被传送到这个工具中。此外，`FileCheck` 会读取测试用例文件。测试用例文件包含了以 `CHECK:` 关键字标记的行，之后是预期的输出。`FileCheck` 会尝试将这些行与传送到它的数据进行匹配。如果没有找到匹配项，则会显示错误。将以下内容的 `sqsumu.s` 测试用例文件放入 `llvm/test/MC/Mips` 目录中：

```cpp
# RUN: llvm-mc %s -triple=mips64-linux-gnu -mattr=+sqsum \
# RUN:  --show-encoding | FileCheck %s
# CHECK: sqsumu  $1, $2, $3 # encoding: [0x70,0x43,0x08,0x28]
     sqsumu $1, $2, $3
```

如果您在 `llvm/test/Mips/MC` 文件夹中，可以使用以下命令运行测试，最后会报告成功：

```cpp
$ llvm-lit sqsumu.s
-- Testing: 1 tests, 1 workers --
PASS: LLVM :: MC/Mips/sqsumu.s (1 of 1)
Testing Time: 0.11s
  Passed: 1
```

LIT 工具解释 `RUN:` 行，将 `%s` 替换为当前的文件名。`FileCheck` 工具读取文件，解析 `CHECK:` 行，并尝试匹配来自管道的输入。这是一种非常有效的测试方法。

如果您在 `build` 目录中，可以使用以下命令调用 LLVM 测试：

```cpp
$ ninja check-llvm
```

您还可以运行一个文件夹中包含的测试，只需添加以破折号分隔的文件夹名称。要运行 `llvm/test/Mips/MC` 文件夹中的测试，可以输入以下命令：

```cpp
$ ninja check-llvm-mips-mc
```

要为代码生成构建一个测试用例，您可以遵循相同的策略。以下的 `sqsum.ll` 文件包含了用于计算斜边平方的 LLVM IR 代码：

```cpp
define i64 @hyposquare(i64 %a, i64 %b) {
  %asq = mul i64 %a, %a
  %bsq = mul i64 %b, %b
  %res = add i64 %asq, %bsq
  ret i64 %res
}
```

要查看生成的汇编代码，您可以使用 `llc` 工具：

```cpp
$ llc –mtriple=mips64-linux-gnu –mattr=+sqsum < sqsum.ll
```

确保您在输出中看到我们的新 `sqsum` 指令。还请检查，如果删除 `–mattr=+sqsum` 选项，则不会生成该指令。

掌握了这些知识，您可以构建测试用例。这次，我们使用两个`RUN：`行：一个用于检查我们是否生成了新指令，另一个用于检查是否没有生成。我们可以在一个测试用例文件中执行这两个操作，因为我们可以告诉`FileCheck`工具查找的标签与`CHECK：`不同。将以下内容的测试用例文件`sqsum.ll`放入`llvm/test/CodeGen/Mips`文件夹中：

```cpp
; RUN: llc -mtriple=mips64-linux-gnu -mattr=+sqsum < %s |\
; RUN:  FileCheck -check-prefix=SQSUM %s
; RUN: llc -mtriple=mips64-linux-gnu < %s |\
; RUN:  FileCheck --check-prefix=NOSQSUM %s
define i64 @hyposquare(i64 %a, i64 %b) {
; SQSUM-LABEL: hyposquare:
; SQSUM: sqsumu $2, $4, $5
; NOSQSUM-LABEL: hyposquare:
; NOSQSUM: dmult $5, $5
; NOSQSUM: mflo $1
; NOSQSUM: dmult $4, $4
; NOSQSUM: mflo $2
; NOSQSUM: addu $2, $2, $1
  %asq = mul i64 %a, %a
  %bsq = mul i64 %b, %b
  %res = add i64 %asq, %bsq
  ret i64 %res
}
```

与其他测试一样，您可以使用以下命令在文件夹中单独运行测试：

```cpp
$ llvm-lit squm.ll
```

或者，您可以使用以下命令从构建目录运行它：

```cpp
$ ninja check-llvm-mips-codegen
```

通过这些步骤，您增强了 LLVM 汇编器的功能，使其支持新指令，启用了指令选择以使用这个新指令，并验证了编码是否正确，代码生成是否按预期工作。

# 总结

在本章中，您学习了 LLVM 目标的后端结构。您使用 MIR 来检查通过后的状态，并使用机器 IR 来运行单个通过。有了这些知识，您可以调查后端通过中的问题。

您学习了 LLVM 中如何使用选择 DAG 来实现指令选择，并且还介绍了使用 FastISel 和 GlobalISel 进行指令选择的替代方法，这有助于决定如果您的平台提供所有这些算法，则选择哪种算法。

您扩展了 LLVM 以支持汇编器中的新机器指令和指令选择，帮助您添加对当前不支持的 CPU 功能的支持。为了验证扩展，您为其开发了自动化测试用例。

在下一章中，我们将研究 LLVM 的另一个独特特性：一步生成和执行代码，也称为**即时**（**JIT**）编译。
