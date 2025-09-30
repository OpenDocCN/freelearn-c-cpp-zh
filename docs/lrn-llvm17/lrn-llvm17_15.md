

# 第十一章：目标描述

LLVM 具有非常灵活的架构。您也可以向其中添加新的目标后端。后端的核心是目标描述，大部分代码都是从那里生成的。在本章中，您将学习如何添加对历史 CPU 的支持。

在本章中，您将涵盖以下内容：

+   *为新后端做准备* 介绍了 M88k CPU 架构，并展示了如何找到所需信息

+   *将新架构添加到 Triple 类中* 教您如何让 LLVM 识别新的 CPU 架构

+   *在 LLVM 中扩展 ELF 文件格式定义* 展示了如何向处理 ELF 对象文件的库和工具添加对 M88k 特定重定位的支持

+   *创建目标描述* 将您对 TableGen 语言的了解应用于在目标描述中建模寄存器文件和指令

+   *将 M88k 后端添加到 LLVM* 解释了为 LLVM 后端所需的最低基础设施

+   *实现汇编器解析器* 展示了如何开发汇编器

+   *创建反汇编器* 教您如何创建反汇编器

到本章结束时，您将了解如何将新后端添加到 LLVM。您将获得开发目标描述中的寄存器文件定义和指令定义的知识，并且您将知道如何从该描述中创建汇编器和反汇编器。

# 为新后端做准备

无论是否需要商业上支持新的 CPU，还是仅仅作为一个爱好项目来添加对某些旧架构的支持，将新后端添加到 LLVM 都是一项重大任务。本节和接下来的两章概述了您需要为新后端开发的内容。我们将添加一个用于摩托罗拉 M88k 架构的后端，这是一个 80 年代的 RISC 架构。

参考文献

您可以在维基百科上了解更多关于这种摩托罗拉架构的信息：[`en.wikipedia.org/wiki/Motorola_88000`](https://en.wikipedia.org/wiki/Motorola_88000)。关于该架构的最重要信息仍然可以在互联网上找到。您可以在 [`www.bitsavers.org/components/motorola/88000/`](http://www.bitsavers.org/components/motorola/88000/) 找到 CPU 手册以及指令集和时序信息，在 [`archive.org/details/bitsavers_attunixSysa0138776555SystemVRelease488000ABI1990_8011463`](https://archive.org/details/bitsavers_attunixSysa0138776555SystemVRelease488000ABI1990_8011463) 可以找到 System V ABI M88k 处理器补充，其中包含了 ELF 格式和调用约定的定义。

OpenBSD，可在 [`www.openbsd.org/`](https://www.openbsd.org/) 获取，仍然支持 LUNA-88k 系统。在 OpenBSD 系统上，创建 M88k 的 GCC 跨编译器很容易。而且，使用可在 [`gavare.se/gxemul/`](http://gavare.se/gxemul/) 获取的 GXemul，我们可以得到一个能够运行某些 OpenBSD 版本的 M88k 架构的仿真器。

M88k 架构已经很久没有生产了，但我们找到了足够的信息和工具，使其成为添加 LLVM 后端的一个有趣目标。我们将从扩展 `Triple` 类的非常基础的任务开始。

# 将新的架构添加到 Triple 类

`Triple` 类的实例表示 LLVM 为其生成代码的目标平台。为了支持新的架构，第一个任务是扩展 `Triple` 类。在 `llvm/include/llvm/TargetParser/Triple.h` 文件中，向 `ArchType` 枚举添加一个成员以及一个新的谓词：

```cpp

class Triple {
public:
  enum ArchType {
      // Many more members
      m88k,      // M88000 (big endian): m88k
  };
  /// Tests whether the target is M88k.
  bool isM88k() const {
      return getArch() == Triple::m88k;
  }
// Many more methods
};
```

在 `llvm/lib/TargetParser/Triple.cpp` 文件内部，有许多使用 `ArchType` 枚举的方法。你需要扩展它们所有；例如，在 `getArchTypeName()` 方法中，你需要添加一个新的 `case` 语句，如下所示：

```cpp

 switch (Kind) {
     // Many more cases
     case m88k:           return "m88k";
  }
```

大多数情况下，编译器会警告你如果在某个函数中忘记处理新的 `m88k` 枚举成员。接下来，我们将扩展 **可执行和链接** **格式** (**ELF**)。

# 扩展 LLVM 中的 ELF 文件格式定义

ELF 文件格式是 LLVM 支持的几种二进制对象文件格式之一。ELF 本身是为许多 CPU 架构定义的，也有 M88k 架构的定义。我们只需要添加重定位的定义和一些标志。重定位在 *系统 V ABI M88k 处理器* 补充书的 *IR 代码生成基础* 章节中给出（见章节开头 *为新的后端设置舞台* 部分的链接）：*第四章*

1.  我们需要在 `llvm/include/llvm/BinaryFormat/ELFRelocs/M88k.def` 文件中输入以下代码：

    ```cpp

    #ifndef ELF_RELOC
    #error "ELF_RELOC must be defined"
    #endif
    ELF_RELOC(R_88K_NONE, 0)
    ELF_RELOC(R_88K_COPY, 1)
    // Many more…
    ```

1.  我们还在 `llvm/include/llvm/BinaryFormat/ELF.h` 文件中添加了以下标志，以及重定位的定义：

    ```cpp

    // M88k Specific e_flags
    enum : unsigned {
        EF_88K_NABI = 0x80000000,   // Not ABI compliant
        EF_88K_M88110 = 0x00000004  // File uses 88110-specific features
    };
    // M88k relocations.
    enum {
        #include "ELFRelocs/M88k.def"
    };
    ```

    代码可以添加到文件的任何位置，但最好是保持文件结构，并在 MIPS 架构的代码之前插入它。

1.  我们还需要扩展一些其他方法。在 `llvm/include/llvm/Object/ELFObjectFile.h` 文件中，有一些方法在枚举成员和字符串之间进行转换。例如，我们必须向 `getFileFormatName()` 方法添加一个新的 `case` 语句：

    ```cpp

      switch (EF.getHeader()->e_ident[ELF::EI_CLASS]) {
    // Many more cases
        case ELF::EM_88K:
          return "elf32-m88k";
      }
    ```

1.  同样，我们扩展 `getArch()` 方法：

    ```cpp

      switch (EF.getHeader().e_machine) {
    // Many more cases
      case ELF::EM_88K:
        return Triple::m88k;
    ```

1.  最后，我们在 `llvm/lib/Object/ELF.cpp` 文件中的 `getELFRelocationTypeName()` 方法使用重定位定义：

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

1.  为了完成支持，你也可以扩展 `llvm/lib/ObjectYAML/ELFYAML.cpp` 文件。此文件由 `yaml2obj` 和 `obj2yaml` 工具使用，它们根据 YAML 描述创建 ELF 文件，反之亦然。第一个添加需要在 `ScalarEnumerationTraits<ELFYAML::ELF_EM>::enumeration()` 方法中完成，该方法列出 ELF 架构的所有值：

    ```cpp

      ECase(EM_88K);
    ```

1.  同样，在 `ScalarEnumerationTraits<ELFYAML::ELF_REL>::enumeration()` 方法中，你需要再次包含重定位的定义：

    ```cpp

      case ELF::EM_88K:
    #include "llvm/BinaryFormat/ELFRelocs/M88k.def"
        break;
    ```

在这个阶段，我们已经完成了对 ELF 文件格式中 m88k 架构的支持。您可以使用 `llvm-readobj` 工具来检查 ELF 对象文件，例如，由 OpenBSD 上的交叉编译器创建的文件。同样，您可以使用 `yaml2obj` 工具为 m88k 架构创建 ELF 对象文件。

添加对对象文件格式的支持是强制性的吗？

将对架构的支持集成到 ELF 文件格式实现中只需要几行代码。如果您正在创建的 LLVM 后端使用的架构是 ELF 格式，那么您应该选择这条路径。另一方面，添加对完全新的二进制文件格式的支持是一个复杂的过程。如果这是必需的，那么常用的方法是将汇编文件输出，并使用外部汇编器创建对象文件。

通过这些添加，LLVM 对 ELF 文件格式的实现现在支持了 M88k 架构。在下一节中，我们将为 M88k 架构创建目标描述，它描述了指令、寄存器以及许多关于架构的细节。

# 创建目标描述

`llvm/include/llvm/Target/Target.td` 文件，可以在 [`github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/Target.td`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/Target.td) 上找到。这个文件注释很多，是关于定义使用的有用信息来源。

在理想的世界里，我们会从目标描述中生成整个后端。这个目标尚未实现，因此，我们以后还需要扩展生成的代码。由于文件大小，目标描述被拆分成了几个文件。顶层文件将是 `M88k.td`，位于 `llvm/lib/Target/M88k` 目录中，该目录还包括其他文件。让我们看看一些文件，从寄存器定义开始。

## 添加寄存器定义

CPU 架构通常定义一组寄存器。这些寄存器的特性可能不同。一些架构允许访问子寄存器。例如，x86 架构有特殊的寄存器名称来访问寄存器值的一部分。其他架构不实现这一点。除了通用、浮点数和向量寄存器外，架构还可能有用于状态代码或浮点操作配置的特殊寄存器。我们需要为 LLVM 定义所有这些信息。寄存器定义存储在 `M88kRegisterInfo.td` 文件中，也可以在 `llvm/lib/Target/M88k` 目录中找到。

M88k 架构定义了通用寄存器、用于浮点操作的扩展寄存器和控制寄存器。为了使示例保持简洁，我们只定义通用寄存器。我们首先定义寄存器的超类。寄存器有一个名称和一个编码。名称用于指令的文本表示。同样，编码用作指令二进制表示的一部分。该架构定义了 32 个寄存器，因此寄存器的编码使用 5 位，所以我们限制了包含编码的字段。我们还定义了所有生成的 C++代码都应该位于`M88k`命名空间中：

```cpp

class M88kReg<bits<5> Enc, string n> : Register<n> {
  let HWEncoding{15-5} = 0;
  let HWEncoding{4-0} = Enc;
  let Namespace = "M88k";
}
```

接下来，我们可以定义所有 32 个通用寄存器。`r0`寄存器是特殊的，因为它在读取时总是返回常数`0`，因此我们将该寄存器的`isConstant`标志设置为`true`：

```cpp

foreach I = 0-31 in {
  let isConstant = !eq(I, 0) in
    def R#I : M88kReg<I, "r"#I>;
}
```

对于寄存器分配器，单个寄存器需要被分组到寄存器类中。寄存器的顺序定义了分配顺序。寄存器分配器还需要其他关于寄存器的信息，例如，例如，可以存储在寄存器中的值类型、寄存器的位溢出大小以及内存中所需的对齐方式。我们不是直接使用`RegisterClass`基类，而是创建了一个新的`M88kRegisterClass`类。这允许我们根据需要更改参数列表。它还避免了重复使用生成代码中使用的 C++命名空间名称，这是`RegisterClass`类的第一个参数：

```cpp

class M88kRegisterClass<list<ValueType> types, int size,
                        int alignment, dag regList,
                        int copycost = 1>
  : RegisterClass<"M88k", types, alignment, regList> {
      let Size = size;
      let CopyCost = copycost;
}
```

此外，我们定义了一个用于寄存器操作数的类。操作数描述了指令的输入和输出。它们在指令的汇编和反汇编过程中以及指令选择阶段使用的模式中都被使用。使用我们自己的类，我们可以给用于解码寄存器操作数的生成函数一个符合 LLVM 编码指南的名称：

```cpp

class M88kRegisterOperand<RegisterClass RC>
    : RegisterOperand<RC> {
  let DecoderMethod = "decode"#RC#"RegisterClass";
}
```

基于这些定义，我们现在定义通用寄存器。请注意，m88k 架构的通用寄存器是 32 位宽的，可以存储整数和浮点值。为了避免编写所有寄存器名称，我们使用`sequence`生成器，它根据模板字符串生成字符串列表：

```cpp

def GPR : M88kRegisterClass<[i32, f32], 32, 32,
                            (add (sequence "R%u", 0, 31))>;
```

同样，我们定义了寄存器操作数。`r0`寄存器是特殊的，因为它包含常数`0`。这个事实可以被全局指令选择框架使用，因此我们将此信息附加到寄存器操作数上：

```cpp

def GPROpnd : M88kRegisterOperand<GPR> {
  let GIZeroRegister = R0;
}
```

m88k 架构有一个扩展，它定义了一个仅用于浮点值的扩展寄存器文件。您将按照与通用寄存器相同的方式定义这些寄存器。

通用寄存器也成对使用，主要用于 64 位浮点运算，我们需要对它们进行建模。我们使用`sub_hi`和`sub_lo`子寄存器索引来描述高 32 位和低 32 位。我们还需要设置生成的代码的 C++命名空间：

```cpp

let Namespace = "M88k" in {
  def sub_hi : SubRegIndex<32, 0>;
  def sub_lo : SubRegIndex<32, 32>;
}
```

然后使用`RegisterTuples`类定义寄存器对。该类将子寄存器索引列表作为第一个参数，将寄存器列表作为第二个参数。我们只需要偶数/奇数对，我们通过序列的可选第四个参数（生成序列时使用的步长）来实现这一点：

```cpp

def GRPair : RegisterTuples<[sub_hi, sub_lo],
                          [(add (sequence "R%u", 0, 30, 2)),
                           (add (sequence "R%u", 1, 31, 2))]>;
```

要使用寄存器对，我们定义了一个寄存器类和一个寄存器操作数：

```cpp

def GPR64 : M88kRegisterClass<[i64, f64], 64, 32,
                              (add GRPair), /*copycost=*/ 2>;
def GPR64Opnd : M88kRegisterOperand<GPR64>;
```

请注意，我们将`copycost`参数设置为`2`，因为我们需要两个指令而不是一个来复制寄存器对到另一个寄存器对。

这完成了我们对寄存器的定义。在下一节中，我们将定义指令格式。

## 定义指令格式和指令信息

指令使用 TableGen 的`Instruction`类定义。定义一个指令是一个复杂任务，因为我们必须考虑许多细节。指令有一个文本表示，用于汇编器和反汇编器。它有一个名称，例如`and`，并且它可能有操作数。汇编器将文本表示转换为二进制格式，因此我们必须定义该格式的布局。为了指令选择，我们需要将一个模式附加到指令上。为了管理这种复杂性，我们定义了一个类层次结构。基类将描述各种指令格式，并存储在`M88kIntrFormats.td`文件中。指令本身和其他用于指令选择的定义存储在`M88kInstrInfo.td`文件中。

让我们从定义一个名为`M88kInst`的 m88k 架构指令类开始。我们从这个预定义的`Instruction`类中派生这个类。我们的新类有几个参数。`outs`和`ins`参数描述了输出和输入操作数，作为一个使用特殊`dag`类型的列表。指令的文本表示被分为`asm`参数中给出的助记符和操作数。最后，`pattern`参数可以存储用于指令选择的模式。

我们还需要定义两个新字段：

+   `Inst`字段用于存储指令的位模式。由于指令的大小取决于平台，因此该字段不能预先定义。m88k 架构的所有指令都是 32 位宽，因此该字段具有`bits<32>`类型。

+   另一个字段称为`SoftFail`，其类型与`Inst`相同。它包含一个位掩码，用于与指令一起使用，其实际编码可以与`Inst`字段中的位不同，但仍有效。唯一需要这个的平台是 ARM，因此我们可以简单地将此字段设置为`0`。

其他字段在超类中定义，我们只需设置值。TableGen 语言中可以进行简单的计算，我们使用它来创建 `AsmString` 字段的值，该字段持有完整的汇编表示。如果 `operands` 操作数字符串为空，则 `AsmString` 字段将只包含 `asm` 参数的值；否则，它将是两个字符串的连接，它们之间有一个空格：

```cpp

class InstM88k<dag outs, dag ins, string asm, string operands,
               list<dag> pattern = []>
  : Instruction {
  bits<32> Inst;
  bits<32> SoftFail = 0;
  let Namespace = "M88k";
  let Size = 4;
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = !if(!eq(operands, ""), asm,
                      !strconcat(asm, " ", operands));
  let Pattern = pattern;
  let DecoderNamespace = "M88k";
}
```

对于指令编码，制造商通常将指令分组，同一组中的指令具有相似的编码。我们可以使用这些组来系统地创建定义指令格式的类。例如，m88k 架构的所有逻辑操作将目标寄存器编码在位 21 到 25，第一个源寄存器编码在位 16 到 20。请注意这里的实现模式：我们声明 `rd` 和 `rs1` 字段用于值，并将这些值分配给 `Inst` 字段中正确的位位置，这是我们之前在超类中定义的：

```cpp

class F_L<dag outs, dag ins, string asm, string operands,
          list<dag> pattern = []>
   : InstM88k<outs, ins, asm, operands, pattern> {
  bits<5>  rd;
  bits<5>  rs1;
  let Inst{25-21} = rd;
  let Inst{20-16} = rs1;
}
```

基于这种格式的逻辑操作有几组。其中之一是使用三个寄存器的指令组，在手册中称为 **三地址模式**：

```cpp

class F_LR<bits<5> func, bits<1> comp, string asm,
           list<dag> pattern = []>
   : F_L<(outs GPROpnd:$rd), (ins GPROpnd:$rs1, GPROpnd:$rs2),
         !if(comp, !strconcat(asm, ".c"), asm),
         "$rd, $rs1, $rs2", pattern> {
  bits<5>  rs2;
  let Inst{31-26} = 0b111101;
  let Inst{15-11} = func;
  let Inst{10}    = comp;
  let Inst{9-5}   = 0b00000;
  let Inst{4-0}   = rs2;
}
```

让我们更详细地检查这个类提供的功能。`func` 参数指定操作。作为一个特殊功能，第二个操作数可以在操作之前取补码，这通过设置 `1` 来指示。助记符在 `asm` 参数中给出，并且可以传递一个指令选择模式。

通过初始化超类，我们可以提供更多信息。`and` 指令的完整汇编文本模板是 `and $rd, $rs1, $rs2`。这个操作数字符串对于这个组的所有指令都是固定的，因此我们可以在这里定义它。助记符由这个类的用户给出，但我们可以在这里附加 `.c` 后缀，表示第二个操作数应该首先取补码。最后，我们可以定义输出和输入操作数。这些操作数表示为 `(``outs GPROpnd:$rd)`。

`outs` 操作符表示这个有向图（dag）为输出操作数列表。唯一的参数 `GPROpnd:$rd` 包含一个类型和一个名称。它连接了我们之前已经看到的一些部分。类型是 `GPROnd`，这是我们在上一节中定义的寄存器操作数的名称。名称 `$rd` 指的是目标寄存器。我们之前在操作数字符串中使用过这个名称，也在 `F_L` 超类中作为字段名称。输入操作数以类似的方式定义。类的其余部分初始化 `Inst` 字段的其余位。请花点时间检查一下，现在是否确实已经分配了所有 32 位。

我们将最终的指令定义放在`M88kInstrInfo.td`文件中。由于我们为每个逻辑指令有两个变体，我们使用多类同时定义这两个指令。我们还在这里定义了指令选择的模式，作为一个有向无环图。模式中的操作是`set`，第一个参数是目标寄存器。第二个参数是一个嵌套图，这是实际的模式。再次强调，操作名称是第一个`OpNode`元素。LLVM 有许多预定义的操作，你可以在`llvm/include/llvm/Target/TargetSelectionDAG.td`文件中找到它们（[`github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/TargetSelectionDAG.td)）。例如，有`and`操作，表示位与操作。参数是两个源寄存器，`$rs1`和`$rs2`。你可以大致这样阅读这个模式：如果指令选择的输入包含使用两个寄存器的 OpNode 操作，则将这个操作的结果分配给`$rd`寄存器并生成这个指令。利用图结构，你可以定义更复杂的模式。例如，第二个模式使用`not`操作数将补码集成到模式中。

需要指出的小细节是逻辑运算具有交换律。这有助于指令选择，因此我们将这些指令的`isCommutable`标志设置为`1`：

```cpp

multiclass Logic<bits<5> Fun, string OpcStr, SDNode OpNode> {
  let isCommutable = 1 in
    def rr : F_LR<Fun, /*comp=*/0b0, OpcStr,
                  [(set i32:$rd,
                   (OpNode GPROpnd:$rs1, GPROpnd:$rs2))]>;
  def rrc : F_LR<Fun, /*comp=*/0b1, OpcStr,
                 [(set i32:$rd,
                 (OpNode GPROpnd:$rs1, (not GPROpnd:$rs2)))]>;
}
```

最后，我们定义了指令的记录：

```cpp

defm AND : Logic<0b01000, "and", and>;
defm XOR : Logic<0b01010, "xor", xor>;
defm OR  : Logic<0b01011, "or", or>;
```

第一个参数是功能的位模式，第二个参数是助记符，第三个参数是模式中使用的 dag 操作。

要完全理解类层次结构，请重新查看类定义。指导设计原则是避免信息重复。例如，`0b01000`功能位模式只使用了一次。如果没有`Logic`多类，你需要输入这个位模式两次，并多次重复模式，这容易出错。

请注意，为指令建立命名方案是很好的做法。例如，`and`指令的记录命名为`ANDrr`，而带有补码寄存器的变体命名为`ANDrrc`。这些名称最终会出现在生成的 C++源代码中，使用命名方案有助于理解名称指的是哪个汇编指令。

到目前为止，我们已经对 m88k 架构的寄存器文件进行了建模，并定义了一些指令。在下一节中，我们将创建顶层文件。

## 为目标描述创建顶层文件

到目前为止，我们已经创建了`M88kRegisterInfo.td`、`M88kInstrFormats.td`和`M88kInstrInfo.td`文件。目标描述是一个单独的文件，称为`M88k.td`。该文件首先包含 LLVM 定义，然后是我们已实现的文件：

```cpp

include "llvm/Target/Target.td"
include "M88kRegisterInfo.td"
include "M88kInstrFormats.td"
include "M88kInstrInfo.td"
```

我们将在添加更多后端功能时扩展这个 `include` 部分。

顶层文件还定义了一些全局实例。第一个名为 `M88kInstrInfo` 的记录包含了所有指令的信息：

```cpp

def M88kInstrInfo : InstrInfo;
```

我们将汇编器类命名为 `M88kAsmParser`。为了使 TableGen 能够识别硬编码的寄存器，我们指定寄存器名称以百分号开头，并且需要定义一个汇编器解析器变体来指定这一点：

```cpp

def M88kAsmParser : AsmParser;
def M88kAsmParserVariant : AsmParserVariant {
  let RegisterPrefix = "%";
}
```

最后，我们需要定义目标：

```cpp

def M88k : Target {
  let InstructionSet = M88kInstrInfo;
  let AssemblyParsers  = [M88kAsmParser];
  let AssemblyParserVariants = [M88kAsmParserVariant];
}
```

现在我们已经定义了足够的目标信息，可以编写第一个实用工具。在下一节中，我们将添加 M88k 后端到 LLVM。

# 将 M88k 后端添加到 LLVM

我们尚未讨论目标描述文件放置的位置。LLVM 中的每个后端都在 `llvm/lib/Target` 下的一个子目录中。我们在这里创建 `M88k` 目录，并将目标描述文件复制到其中。

当然，仅仅添加 TableGen 文件是不够的。LLVM 使用一个注册表来查找目标实现的实例，并期望某些全局函数注册这些实例。由于某些部分是生成的，我们已提供实现。

关于每个目标的所有信息，如目标三元组、目标机器的工厂函数、汇编器、反汇编器等，都存储在 `Target` 类的一个实例中。每个目标都持有该类的静态实例，并且该实例在中央注册表中注册：

1.  实现在我们目标中的 `TargetInfo` 子目录下的 `M88kTargetInfo.cpp` 文件中。`Target` 类的单个实例被保留在 `getTheM88kTarget()` 函数中：

    ```cpp

    using namespace llvm;
    Target &llvm::getTheM88kTarget() {
      static Target TheM88kTarget;
      return TheM88kTarget;
    }
    ```

1.  LLVM 要求每个目标提供一个 `LLVMInitialize<Target Name>TargetInfo()` 函数来注册目标实例。该函数必须具有 C 链接，因为它也用于 LLVM C API：

    ```cpp

    extern "C" LLVM_EXTERNAL_VISIBILITY void
    LLVMInitializeM88kTargetInfo() {
        RegisterTarget<Triple::m88k, /*HasJIT=*/false> X(
          getTheM88kTarget(), "m88k", "M88k", "M88k");
    }
    ```

1.  我们还需要在同一个目录中创建一个 `M88kTargetInfo.h` 头文件，它只包含一个声明：

    ```cpp

    namespace llvm {
    class Target;
    Target &getTheM88kTarget();
    }
    ```

1.  最后，我们添加一个 `CMakeLists.txt` 文件用于构建：

    ```cpp

    add_llvm_component_library(LLVMM88kInfo
      M88kTargetInfo.cpp
      LINK_COMPONENTS  Support
      ADD_TO_COMPONENT M88k)
    ```

接下来，我们在目标实例中部分填充了在**机器代码**（**MC**）级别使用的相关信息。让我们开始吧：

1.  实现在我们目标中的 `MCTargetDesc` 子目录下的 `M88kMCTargetDesc.cpp` 文件中。TableGen 将我们在上一节中创建的目标描述转换为 C++ 源代码片段。在这里，我们包含了寄存器信息、指令信息和子目标信息的部分：

    ```cpp

    using namespace llvm;
    #define GET_INSTRINFO_MC_DESC
    #include "M88kGenInstrInfo.inc"
    #define GET_SUBTARGETINFO_MC_DESC
    #include "M88kGenSubtargetInfo.inc"
    #define GET_REGINFO_MC_DESC
    #include "M88kGenRegisterInfo.inc"
    ```

1.  目标注册表期望为这里的每个类提供一个工厂方法。让我们从指令信息开始。我们分配一个 `MCInstrInfo` 类的实例，并调用生成的 `InitM88kMCInstrInfo()` 函数来填充对象：

    ```cpp

    static MCInstrInfo *createM88kMCInstrInfo() {
        MCInstrInfo *X = new MCInstrInfo();
        InitM88kMCInstrInfo(X);
        return X;
    }
    ```

1.  接下来，我们分配一个 `MCRegisterInfo` 类的对象，并调用一个生成的函数来填充它。额外的 `M88k::R1` 参数值告诉 LLVM，`r1` 寄存器持有返回地址：

    ```cpp

    static MCRegisterInfo *
    createM88kMCRegisterInfo(const Triple &TT) {
        MCRegisterInfo *X = new MCRegisterInfo();
        InitM88kMCRegisterInfo(X, M88k::R1);
        return X;
    }
    ```

1.  最后，我们需要一个子目标信息的工厂方法。该方法接受一个目标三元组、一个 CPU 名称和一个特性字符串作为参数，并将它们转发到生成的函数：

    ```cpp

    static MCSubtargetInfo *
    createM88kMCSubtargetInfo(const Triple &TT,
                              StringRef CPU, StringRef FS) {
      return createM88kMCSubtargetInfoImpl(TT, CPU,
                                           /*TuneCPU*/ CPU,
                                           FS);
    }
    ```

1.  定义了工厂方法之后，我们现在可以注册它们。类似于目标注册，LLVM 预期一个全局函数名为 `LLVMInitialize<Target Name>TargetMC()`：

    ```cpp

    extern "C" LLVM_EXTERNAL_VISIBILITY void
    LLVMInitializeM88kTargetMC() {
      TargetRegistry::RegisterMCInstrInfo(
          getTheM88kTarget(), createM88kMCInstrInfo);
      TargetRegistry::RegisterMCRegInfo(
          getTheM88kTarget(), createM88kMCRegisterInfo);
      TargetRegistry::RegisterMCSubtargetInfo(
          getTheM88kTarget(), createM88kMCSubtargetInfo);
    }
    ```

1.  `M88kMCTargetDesc.h` 头文件只是使一些生成的代码可用：

    ```cpp

    #define GET_REGINFO_ENUM
    #include "M88kGenRegisterInfo.inc"
    #define GET_INSTRINFO_ENUM
    #include "M88kGenInstrInfo.inc"
    #define GET_SUBTARGETINFO_ENUM
    #include "M88kGenSubtargetInfo.inc"
    ```

实现几乎完成。为了防止链接器错误，我们需要提供另一个函数，该函数注册一个 `TargetMachine` 类对象的工厂方法。这个类对于代码生成是必需的，我们在 *第十二章* *指令选择* 中实现它，接下来。在这里，我们只是在 `M88kTargetMachine.cpp` 文件中定义了一个空函数：

```cpp

#include "TargetInfo/M88kTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeM88kTarget() {
  // TODO Register the target machine. See chapter 12.
}
```

这标志着我们第一次实现的结束。然而，LLVM 还不知道我们的新后端。要集成它，打开 `llvm/CMakeLists.txt` 文件，找到定义所有实验性目标的章节，并将 M88k 目标添加到列表中：

```cpp

set(LLVM_ALL_EXPERIMENTAL_TARGETS ARC … M88k  …)
```

假设我们的新后端源代码在目录中，你可以通过输入以下内容来配置构建：

```cpp

$ mkdir build
$ cd build
$ cmake -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=M88k \
  ../llvm-m88k/llvm
…
-- Targeting M88k
…
```

在构建 LLVM 之后，你可以验证工具已经知道我们的新目标：

```cpp

$ bin/llc –version
LLVM (http://llvm.org/):
  LLVM version 17.0.2
  Registered Targets:
    m88k     - M88k
```

到达这个阶段的过程很艰难，所以花点时间庆祝一下吧！

修复可能的编译错误

在 LLVM 17.0.2 中存在一个小疏忽，导致编译错误。在代码的一个地方，子目标信息的 TableGen 发射器使用了已删除的值 `llvm::None` 而不是 `std::nullopt`，导致在编译 `M88kMCTargetDesc.cpp` 时出错。修复此问题的最简单方法是 cherry-pick 从 LLVM 18 开发分支的修复：`git cherry-pick -x a587f429`。

在下一节中，我们实现汇编器解析器，这将给我们第一个工作的 LLVM 工具。

# 实现汇编器解析器

汇编器解析器很容易实现，因为 LLVM 为此提供了一个框架，大部分代码都是从目标描述中生成的。

我们类中的 `ParseInstruction()` 方法在框架检测到需要解析指令时被调用。该方法通过提供的词法分析器解析输入，并构建一个所谓的操作数向量。操作数可以是一个指令助记符、寄存器名称或立即数，或者它可以是针对目标特定的类别。例如，从 `jmp %r2` 输入中构建了两个操作数：一个用于助记符的标记操作数和一个寄存器操作数。

然后，一个生成的匹配器尝试将操作数向量与指令进行匹配。如果找到匹配项，则创建 `MCInst` 类的一个实例，该实例包含解析后的指令。否则，会发出错误消息。这种方法的优势是它自动从目标描述中推导出匹配器，而无需处理所有语法上的怪癖。

然而，我们需要添加几个额外的支持类来使汇编解析器工作。这些额外的类都存储在`MCTargetDesc`目录中。

### 实现 M88k 目标的 MCAsmInfo 支持类

在本节中，我们探讨实现汇编解析器配置的第一个必需类：`MCAsmInfo`类：

1.  我们需要为汇编解析器设置一些定制参数。`MCAsmInfo`基类([`github.com/llvm/llvm-project/blob/main/llvm/include/llvm/MC/MCAsmInfo.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/MC/MCAsmInfo.h))包含了通用参数。此外，为每个支持的文件格式创建了一个子类；例如，`MCAsmInfoELF`类([`github.com/llvm/llvm-project/blob/main/llvm/include/llvm/MC/MCAsmInfoELF.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/MC/MCAsmInfoELF.h))。这样做的原因是，使用相同文件格式的系统汇编器具有共同的特征，因为它们必须支持类似的功能。我们的目标操作系统是 OpenBSD，它使用 ELF 文件格式，因此我们从`MCAsmInfoELF`类派生出自定义的`M88kMCAsmInfo`类。`M88kMCAsmInfo.h`文件中的声明如下：

    ```cpp

    namespace llvm {
    class Triple;
    class M88kMCAsmInfo : public MCAsmInfoELF {
    public:
      explicit M88kMCAsmInfo(const Triple &TT);
    };
    ```

1.  `M88kMCAsmInfo.cpp`文件中的实现仅设置了一些默认值。目前有两个关键设置：使用大端模式以及使用`|`符号作为注释。其他设置用于后续的代码生成：

    ```cpp

    using namespace llvm;
    M88kMCAsmInfo::M88kMCAsmInfo(const Triple &TT) {
      IsLittleEndian = false;
      UseDotAlignForAlignment = true;
      MinInstAlignment = 4;
      CommentString = "|"; // # as comment delimiter is only
                           // allowed at first column
      ZeroDirective = "\t.space\t";
      Data64bitsDirective = "\t.quad\t";
      UsesELFSectionDirectiveForBSS = true;
      SupportsDebugInformation = false;
      ExceptionsType = ExceptionHandling::SjLj;
    }
    ```

现在我们已经完成了`MCAsmInfo`类的实现。接下来我们将学习实现下一个类，这个类帮助我们创建 LLVM 中指令的二进制表示。

### 实现 M88k 目标的 MCCodeEmitter 支持类

在 LLVM 内部，指令通过`MCInst`类的实例来表示。指令可以被输出为汇编文本或二进制形式到目标文件中。`M88kMCCodeEmitter`类创建指令的二进制表示，而`M88kInstPrinter`类则输出其文本表示。

首先，我们将实现`M88kMCCodeEmitter`类，该类存储在`M88kMCCodeEmitter.cpp`文件中：

1.  大多数类都是由 TableGen 生成的。因此，我们只需要添加一些样板代码。请注意，没有相应的头文件；工厂函数的原型将被添加到`M88kMCTargetDesc.h`文件中。它从设置输出指令数量的统计计数器开始：

    ```cpp

    using namespace llvm;
    #define DEBUG_TYPE "mccodeemitter"
    STATISTIC(MCNumEmitted,
              "Number of MC instructions emitted");
    ```

1.  `M88kMCCodeEmitter`类位于匿名命名空间中。我们只需要实现基类中声明的`encodeInstruction()`方法以及`getMachineOpValue()`辅助方法。其他`getBinaryCodeForInstr()`方法由 TableGen 从目标描述中生成：

    ```cpp

    namespace {
    class M88kMCCodeEmitter : public MCCodeEmitter {
      const MCInstrInfo &MCII;
      MCContext &Ctx;
    public:
      M88kMCCodeEmitter(const MCInstrInfo &MCII,
                        MCContext &Ctx)
          : MCII(MCII), Ctx(Ctx) {}
      ~M88kMCCodeEmitter() override = default;
      void encodeInstruction(
          const MCInst &MI, raw_ostream &OS,
          SmallVectorImpl<MCFixup> &Fixups,
          const MCSubtargetInfo &STI) const override;
      uint64_t getBinaryCodeForInstr(
          const MCInst &MI,
          SmallVectorImpl<MCFixup> &Fixups,
          const MCSubtargetInfo &STI) const;
      unsigned
      getMachineOpValue(const MCInst &MI,
                        const MCOperand &MO,
                        SmallVectorImpl<MCFixup> &Fixups,
                        const MCSubtargetInfo &STI) const;
    };
    } // end anonymous namespace
    ```

1.  `encodeInstruction()` 方法仅查找指令的二进制表示，增加统计计数器，并以大端格式写入字节。记住，指令具有固定的 4 字节大小，因此我们在端序流上使用 `uint32_t` 类型：

    ```cpp

    void M88kMCCodeEmitter::encodeInstruction(
        const MCInst &MI, raw_ostream &OS,
        SmallVectorImpl<MCFixup> &Fixups,
        const MCSubtargetInfo &STI) const {
      uint64_t Bits =
          getBinaryCodeForInstr(MI, Fixups, STI);
      ++MCNumEmitted;
      support::endian::write<uint32_t>(OS, Bits,
                                       support::big);
    }
    ```

1.  `getMachineOpValue()` 方法的任务是返回操作数的二进制表示。在目标描述中，我们定义了寄存器在指令中存储的位范围。在这里，我们计算存储在这些位置的值。该方法由生成的代码调用。我们只支持两种情况。对于寄存器，返回我们在目标描述中定义的寄存器编码。对于立即数，返回立即数值：

    ```cpp

    unsigned M88kMCCodeEmitter::getMachineOpValue(
        const MCInst &MI, const MCOperand &MO,
        SmallVectorImpl<MCFixup> &Fixups,
        const MCSubtargetInfo &STI) const {
      if (MO.isReg())
        return Ctx.getRegisterInfo()->getEncodingValue(
            MO.getReg());
      if (MO.isImm())
        return static_cast<uint64_t>(MO.getImm());
      return 0;
    }
    ```

1.  最后，我们包含生成的文件并为该类创建一个工厂方法：

    ```cpp

    #include "M88kGenMCCodeEmitter.inc"
    MCCodeEmitter *
    llvm::createM88kMCCodeEmitter(const MCInstrInfo &MCII,
                                  MCContext &Ctx) {
      return new M88kMCCodeEmitter(MCII, Ctx);
    }
    ```

### 为 M88k 目标实现指令打印支持类

`M88kInstPrinter` 类的结构与 `M88kMCCodeEmitter` 类类似。如前所述，`InstPrinter` 类负责输出 LLVM 指令的文本表示。类的大部分是由 TableGen 生成的，但我们必须添加打印操作数的支持。类在 `M88kInstPrinter.h` 头文件中声明。实现位于 `M88kInstPrinter.cpp` 文件中：

1.  让我们从头文件开始。在包含所需的头文件并声明 `llvm` 命名空间之后，声明了两个前向引用以减少所需的包含数量：

    ```cpp

    namespace llvm {
    class MCAsmInfo;
    class MCOperand;
    ```

1.  除了构造函数之外，我们只需要实现 `printOperand()` 和 `printInst()` 方法。其他方法由 TableGen 生成：

    ```cpp

    class M88kInstPrinter : public MCInstPrinter {
    public:
      M88kInstPrinter(const MCAsmInfo &MAI,
                      const MCInstrInfo &MII,
                      const MCRegisterInfo &MRI)
          : MCInstPrinter(MAI, MII, MRI) {}
      std::pair<const char *, uint64_t>
      getMnemonic(const MCInst *MI) override;
      void printInstruction(const MCInst *MI,
                            uint64_t Address,
                            const MCSubtargetInfo &STI,
                            raw_ostream &O);
      static const char *getRegisterName(MCRegister RegNo);
      void printOperand(const MCInst *MI, int OpNum,
                        const MCSubtargetInfo &STI,
                        raw_ostream &O);
      void printInst(const MCInst *MI, uint64_t Address,
                     StringRef Annot,
                     const MCSubtargetInfo &STI,
                     raw_ostream &O) override;
    };
    } // end namespace llvm
    ```

1.  实现位于 `M88kInstPrint.cpp` 文件中。在包含所需的头文件并使用 `llvm` 命名空间之后，包含生成了 C++ 片段的文件：

    ```cpp

    using namespace llvm;
    #define DEBUG_TYPE "asm-printer"
    #include "M88kGenAsmWriter.inc"
    ```

1.  `printOperand()` 方法检查操作数的类型，并输出一个寄存器名称或一个立即数。寄存器名称是通过 `getRegisterName()` 生成的函数查找的：

    ```cpp

    void M88kInstPrinter::printOperand(
        const MCInst *MI, int OpNum,
        const MCSubtargetInfo &STI, raw_ostream &O) {
      const MCOperand &MO = MI->getOperand(OpNum);
      if (MO.isReg()) {
        if (!MO.getReg())
          O << '0';
        else
          O << '%' << getRegisterName(MO.getReg());
      } else if (MO.isImm())
        O << MO.getImm();
      else
        llvm_unreachable("Invalid operand");
    }
    ```

1.  `printInst()` 方法仅调用生成的 `printInstruction()` 方法来打印指令，然后调用 `printAnnotation()` 方法来打印可能的注释：

    ```cpp

    void M88kInstPrinter::printInst(
        const MCInst *MI, uint64_t Address, StringRef Annot,
        const MCSubtargetInfo &STI, raw_ostream &O) {
      printInstruction(MI, Address, STI, O);
      printAnnotation(O, Annot);
    }
    ```

### 实现 M88k 特定的目标描述

在 `M88kMCTargetDesc.cpp` 文件中，我们需要做一些添加：

1.  首先，我们需要为 `MCInstPrinter` 类和 `MCAsmInfo` 类创建一个新的工厂方法：

    ```cpp

    static MCInstPrinter *createM88kMCInstPrinter(
        const Triple &T, unsigned SyntaxVariant,
        const MCAsmInfo &MAI, const MCInstrInfo &MII,
        const MCRegisterInfo &MRI) {
      return new M88kInstPrinter(MAI, MII, MRI);
    }
    static MCAsmInfo *
    createM88kMCAsmInfo(const MCRegisterInfo &MRI,
                        const Triple &TT,
                        const MCTargetOptions &Options) {
      return new M88kMCAsmInfo(TT);
    }
    ```

1.  最后，在 `LLVMInitializeM88kTargetMC()` 函数中，我们需要添加工厂方法的注册：

    ```cpp

    extern "C" LLVM_EXTERNAL_VISIBILITY void
    LLVMInitializeM88kTargetMC() {
      // …
      TargetRegistry::RegisterMCAsmInfo(
          getTheM88kTarget(), createM88kMCAsmInfo);
      TargetRegistry::RegisterMCCodeEmitter(
          getTheM88kTarget(), createM88kMCCodeEmitter);
      TargetRegistry::RegisterMCInstPrinter(
          getTheM88kTarget(), createM88kMCInstPrinter);
    }
    ```

现在我们已经实现了所有必需的支持类，我们最终可以添加汇编解析器。

### 创建 M88k 汇编解析器类

在`AsmParser`目录中只有一个`M88kAsmParser.cpp`实现文件。`M88kOperand`类表示一个解析后的操作数，并由生成的源代码和我们的汇编器解析器实现中的`M88kAssembler`类使用。这两个类都在匿名命名空间中，只有工厂方法是全局可见的。让我们首先看看`M88kOperand`类：

1.  操作数可以是标记、寄存器或立即数。我们定义了`OperandKind`枚举来区分这些情况。当前类型存储在`Kind`成员中。我们还存储操作数的起始和结束位置，这对于打印错误信息是必需的：

    ```cpp

    class M88kOperand : public MCParsedAsmOperand {
      enum OperandKind { OpKind_Token, OpKind_Reg,
                         OpKind_Imm };
      OperandKind Kind;
      SMLoc StartLoc, EndLoc;
    ```

1.  为了存储值，我们定义了一个联合。标记存储为`StringRef`，寄存器通过其编号来标识。立即数由`MCExpr`类表示：

    ```cpp

       union {
        StringRef Token;
        unsigned RegNo;
        const MCExpr *Imm;
      };
    ```

1.  构造函数初始化所有字段，除了联合。此外，我们定义了返回起始和结束位置值的方法：

    ```cpp

    public:
      M88kOperand(OperandKind Kind, SMLoc StartLoc,
                  SMLoc EndLoc)
          : Kind(Kind), StartLoc(StartLoc), EndLoc(EndLoc) {}
      SMLoc getStartLoc() const override { return StartLoc; }
      SMLoc getEndLoc() const override { return EndLoc; }
    ```

1.  对于每种操作数类型，我们必须定义四个方法。对于寄存器，方法包括`isReg()`来检查操作数是否为寄存器，`getReg()`来返回值，`createReg()`来创建寄存器操作数，以及`addRegOperands()`来将操作数添加到指令中。后一个函数在构建指令时由生成的源代码调用。标记和立即数的方法类似：

    ```cpp

      bool isReg() const override {
        return Kind == OpKind_Reg;
      }
      unsigned getReg() const override { return RegNo; }
      static std::unique_ptr<M88kOperand>
      createReg(unsigned Num, SMLoc StartLoc,
                SMLoc EndLoc) {
        auto Op = std::make_unique<M88kOperand>(
            OpKind_Reg, StartLoc, EndLoc);
        Op->RegNo = Num;
        return Op;
      }
      void addRegOperands(MCInst &Inst, unsigned N) const {
        assert(N == 1 && "Invalid number of operands");
        Inst.addOperand(MCOperand::createReg(getReg()));
      }
    ```

1.  最后，超类定义了一个抽象的`print()`虚方法，我们需要实现它。这仅用于调试目的：

    ```cpp

      void print(raw_ostream &OS) const override {
        switch (Kind) {
        case OpKind_Imm:
          OS << "Imm: " << getImm() << "\n"; break;
        case OpKind_Token:
          OS << "Token: " << getToken() << "\n"; break;
        case OpKind_Reg:
          OS << "Reg: "
             << M88kInstPrinter::getRegisterName(getReg())
             << „\n"; break;
        }
      }
    };
    ```

接下来，我们声明`M88kAsmParser`类。在声明之后，匿名命名空间将结束：

1.  在类的开头，我们包含生成的片段：

    ```cpp

    class M88kAsmParser : public MCTargetAsmParser {
    #define GET_ASSEMBLER_HEADER
    #include "M88kGenAsmMatcher.inc"
    ```

1.  接下来，我们定义所需的字段。我们需要对实际解析器的引用，它属于`MCAsmParser`类，以及一个对子目标信息的引用：

    ```cpp

      MCAsmParser &Parser;
      const MCSubtargetInfo &SubtargetInfo;
    ```

1.  为了实现汇编器，我们覆盖了`MCTargetAsmParser`超类中定义的一些方法。`MatchAndEmitInstruction()`方法尝试匹配一个指令并发出由`MCInst`类实例表示的指令。解析指令是在`ParseInstruction()`方法中完成的，而`parseRegister()`和`tryParseRegister()`方法负责解析寄存器。其他方法内部需要：

    ```cpp

      bool
      ParseInstruction(ParseInstructionInfo &Info,
                       StringRef Name, SMLoc NameLoc,
                       OperandVector &Operands) override;
      bool parseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                         SMLoc &EndLoc) override;
      OperandMatchResultTy
      tryParseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                       SMLoc &EndLoc) override;
      bool parseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                         SMLoc &EndLoc,
                         bool RestoreOnFailure);
      bool parseOperand(OperandVector &Operands,
                        StringRef Mnemonic);
      bool MatchAndEmitInstruction(
          SMLoc IdLoc, unsigned &Opcode,
          OperandVector &Operands, MCStreamer &Out,
          uint64_t &ErrorInfo,
          bool MatchingInlineAsm) override;
    ```

1.  构造函数是内联定义的。它主要初始化所有字段。这完成了类的声明，之后匿名命名空间结束：

    ```cpp

    public:
      M88kAsmParser(const MCSubtargetInfo &STI,
                    MCAsmParser &Parser,
                    const MCInstrInfo &MII,
                    const MCTargetOptions &Options)
          : MCTargetAsmParser(Options, STI, MII),
            Parser(Parser), SubtargetInfo(STI) {
        setAvailableFeatures(ComputeAvailableFeatures(
            SubtargetInfo.getFeatureBits()));
      }
    };
    ```

1.  现在我们包含汇编器生成的部分：

    ```cpp

    #define GET_REGISTER_MATCHER
    #define GET_MATCHER_IMPLEMENTATION
    #include "M88kGenAsmMatcher.inc"
    ```

1.  当期望指令时，会调用`ParseInstruction()`方法。它必须能够解析指令的所有语法形式。目前，我们只有接受三个操作数的指令，这些操作数由逗号分隔，因此解析很简单。请注意，在出现错误的情况下，返回值是`true`！

    ```cpp

    bool M88kAsmParser::ParseInstruction(
        ParseInstructionInfo &Info, StringRef Name,
        SMLoc NameLoc, OperandVector &Operands) {
      Operands.push_back(
          M88kOperand::createToken(Name, NameLoc));
      if (getLexer().isNot(AsmToken::EndOfStatement)) {
        if (parseOperand(Operands, Name)) {
          return Error(getLexer().getLoc(),
                       "expected operand");
        }
        while (getLexer().is(AsmToken::Comma)) {
          Parser.Lex();
          if (parseOperand(Operands, Name)) {
            return Error(getLexer().getLoc(),
                         "expected operand");
          }
        }
        if (getLexer().isNot(AsmToken::EndOfStatement))
          return Error(getLexer().getLoc(),
                       "unexpected token in argument list");
      }
      Parser.Lex();
      return false;
    }
    ```

1.  操作数可以是寄存器或立即数。我们稍微泛化一下，解析一个表达式而不是仅仅一个整数。这有助于以后添加地址模式。如果解析成功，解析的操作数将被添加到 `Operands` 列表中：

    ```cpp

    bool M88kAsmParser::parseOperand(
        OperandVector &Operands, StringRef Mnemonic) {
      if (Parser.getTok().is(AsmToken::Percent)) {
        MCRegister RegNo;
        SMLoc StartLoc, EndLoc;
        if (parseRegister(RegNo, StartLoc, EndLoc,
                          /*RestoreOnFailure=*/false))
          return true;
        Operands.push_back(M88kOperand::createReg(
            RegNo, StartLoc, EndLoc));
        return false;
      }
      if (Parser.getTok().is(AsmToken::Integer)) {
        SMLoc StartLoc = Parser.getTok().getLoc();
        const MCExpr *Expr;
        if (Parser.parseExpression(Expr))
          return true;
        SMLoc EndLoc = Parser.getTok().getLoc();
        Operands.push_back(
            M88kOperand::createImm(Expr, StartLoc, EndLoc));
        return false;
      }
      return true;
    }
    ```

1.  `parseRegister()` 方法尝试解析一个寄存器。首先，它检查是否存在百分号 `%`。如果其后跟一个与寄存器名称匹配的标识符，那么我们成功解析了一个寄存器，并在 `RegNo` 参数中返回寄存器编号。然而，如果我们无法识别寄存器，那么如果 `RestoreOnFailure` 参数为 `true`，我们可能需要撤销词法分析：

    ```cpp

    bool M88kAsmParser::parseRegister(
        MCRegister &RegNo, SMLoc &StartLoc, SMLoc &EndLoc,
        bool RestoreOnFailure) {
      StartLoc = Parser.getTok().getLoc();
      if (Parser.getTok().isNot(AsmToken::Percent))
        return true;
      const AsmToken &PercentTok = Parser.getTok();
      Parser.Lex();
      if (Parser.getTok().isNot(AsmToken::Identifier) ||
          (RegNo = MatchRegisterName(
               Parser.getTok().getIdentifier())) == 0) {
        if (RestoreOnFailure)
          Parser.getLexer().UnLex(PercentTok);
        return Error(StartLoc, "invalid register");
      }
      Parser.Lex();
      EndLoc = Parser.getTok().getLoc();
      return false;
    }
    ```

1.  覆盖的 `parseRegister()` 和 `tryparseRegister()` 方法只是对先前定义的方法的包装。后者方法还将布尔返回值转换为 `OperandMatchResultTy` 枚举的枚举成员：

    ```cpp

    bool M88kAsmParser::parseRegister(MCRegister &RegNo,
                                      SMLoc &StartLoc,
                                      SMLoc &EndLoc) {
      return parseRegister(RegNo, StartLoc, EndLoc,
                           /*RestoreOnFailure=*/false);
    }
    OperandMatchResultTy M88kAsmParser::tryParseRegister(
        MCRegister &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
      bool Result =
          parseRegister(RegNo, StartLoc, EndLoc,
                        /*RestoreOnFailure=*/true);
      bool PendingErrors = getParser().hasPendingError();
      getParser().clearPendingErrors();
      if (PendingErrors)
        return MatchOperand_ParseFail;
      if (Result)
        return MatchOperand_NoMatch;
      return MatchOperand_Success;
    }
    ```

1.  最后，`MatchAndEmitInstruction()` 方法驱动解析。该方法的大部分内容都是用于发出错误信息。为了识别指令，调用生成的 `MatchInstructionImpl()` 方法：

    ```cpp

    bool M88kAsmParser::MatchAndEmitInstruction(
        SMLoc IdLoc, unsigned &Opcode,
        OperandVector &Operands, MCStreamer &Out,
        uint64_t &ErrorInfo, bool MatchingInlineAsm) {
      MCInst Inst;
      SMLoc ErrorLoc;
      switch (MatchInstructionImpl(
          Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
      case Match_Success:
        Out.emitInstruction(Inst, SubtargetInfo);
        Opcode = Inst.getOpcode();
        return false;
      case Match_MissingFeature:
        return Error(IdLoc, "Instruction use requires "
                            "option to be enabled");
      case Match_MnemonicFail:
        return Error(IdLoc,
                     "Unrecognized instruction mnemonic");
      case Match_InvalidOperand: {
        ErrorLoc = IdLoc;
        if (ErrorInfo != ~0U) {
          if (ErrorInfo >= Operands.size())
            return Error(
                IdLoc, "Too few operands for instruction");
          ErrorLoc = ((M88kOperand &)*Operands[ErrorInfo])
                         .getStartLoc();
          if (ErrorLoc == SMLoc())
            ErrorLoc = IdLoc;
        }
        return Error(ErrorLoc,
                     "Invalid operand for instruction");
      }
      default:
        break;
      }
      llvm_unreachable("Unknown match type detected!");
    }
    ```

1.  并且像一些其他类一样，汇编器解析器有自己的工厂方法：

    ```cpp

    extern "C" LLVM_EXTERNAL_VISIBILITY void
    LLVMInitializeM88kAsmParser() {
      RegisterMCAsmParser<M88kAsmParser> X(
          getTheM88kTarget());
    }
    ```

这完成了汇编器解析器的实现。在构建 LLVM 之后，我们可以使用 **llvm-mc** 机器代码游乐场工具来汇编一条指令：

```cpp

$ echo 'and %r1,%r2,%r3' | \
  bin/llvm-mc --triple m88k-openbsd --show-encoding
        .text
        and %r1, %r2, %r3  | encoding: [0xf4,0x22,0x40,0x03]
```

注意使用垂直线 `|` 作为注释符号。这是我们配置在 `M88kMCAsmInfo` 类中的值。

调试汇编器匹配器

要调试汇编器匹配器，你指定 `--debug-only=asm-matcher` 命令行选项。这有助于理解为什么解析的指令无法匹配目标描述中定义的指令。

在下一节中，我们将向 llvm-mc 工具添加反汇编器功能。

# 创建反汇编器

实现反汇编器是可选的。然而，实现不需要太多的努力，并且生成反汇编器表可能会捕获其他生成器未检查的编码错误。反汇编器位于 `Disassembler` 子目录中的 `M88kDisassembler.cpp` 文件中：

1.  我们开始实现的过程是定义一个调试类型和 `DecodeStatus` 类型。这两个都是生成代码所必需的：

    ```cpp

    using namespace llvm;
    #define DEBUG_TYPE "m88k-disassembler"
    using DecodeStatus = MCDisassembler::DecodeStatus;
    ```

1.  `M88kDisassmbler` 类位于一个匿名命名空间中。我们只需要实现 `getInstruction()` 方法：

    ```cpp

    namespace {
    class M88kDisassembler : public MCDisassembler {
    public:
      M88kDisassembler(const MCSubtargetInfo &STI,
                       MCContext &Ctx)
          : MCDisassembler(STI, Ctx) {}
      ~M88kDisassembler() override = default;
      DecodeStatus
      getInstruction(MCInst &instr, uint64_t &Size,
                     ArrayRef<uint8_t> Bytes,
                     uint64_t Address,
                     raw_ostream &CStream) const override;
    };
    } // end anonymous namespace
    ```

1.  我们还需要提供一个工厂方法，它将被注册在目标注册表中：

    ```cpp

    static MCDisassembler *
    createM88kDisassembler(const Target &T,
                           const MCSubtargetInfo &STI,
                           MCContext &Ctx) {
        return new M88kDisassembler(STI, Ctx);
    }
    extern "C" LLVM_EXTERNAL_VISIBILITY void
    LLVMInitializeM88kDisassembler() {
      TargetRegistry::RegisterMCDisassembler(
          getTheM88kTarget(), createM88kDisassembler);
    }
    ```

1.  `decodeGPRRegisterClass()` 函数将寄存器编号转换为 TableGen 生成的寄存器枚举成员。这是 `M88kInstPrinter:: getMachineOpValue()` 方法的逆操作。注意我们在 `M88kRegisterOperand` 类的 `DecoderMethod` 字段中指定了这个函数的名称：

    ```cpp

    static const uint16_t GPRDecoderTable[] = {
        M88k::R0,  M88k::R1,  M88k::R2,  M88k::R3,
        // …
    };
    static DecodeStatus
    decodeGPRRegisterClass(MCInst &Inst, uint64_t RegNo,
                           uint64_t Address,
                           const void *Decoder) {
      if (RegNo > 31)
        return MCDisassembler::Fail;
      unsigned Register = GPRDecoderTable[RegNo];
      Inst.addOperand(MCOperand::createReg(Register));
      return MCDisassembler::Success;
    }
    ```

1.  然后我们包含生成的反汇编器表：

    ```cpp

    #include "M88kGenDisassemblerTables.inc"
    ```

1.  最后，我们解码指令。为此，我们需要从 `Bytes` 数组的下一个四个字节开始，从这些字节中创建指令编码，并调用生成的 `decodeInstruction()` 函数：

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
      if (decodeInstruction(DecoderTableM88k32, MI, Inst,
                            Address, this, STI) !=
          MCDisassembler::Success) {
          return MCDisassembler::Fail;
      }
      return MCDisassembler::Success;
    }
    ```

对于反汇编器来说，以上就是需要完成的所有工作。在编译 LLVM 之后，你可以使用 `llvm-mc` 工具再次测试其功能：

```cpp

$ echo "0xf4,0x22,0x40,0x03" | \
  bin/llvm-mc --triple m88k-openbsd –disassemble
        .text
        and %r1, %r2, %r3
```

此外，我们现在可以使用 `llvm-objdump` 工具来反汇编 ELF 文件。然而，为了使其真正有用，我们需要将所有指令添加到目标描述中。

# 摘要

在本章中，你学习了如何创建一个 LLVM 目标描述，并且开发了一个简单的后端目标，该目标支持为 LLVM 指令进行汇编和反汇编。你首先收集了所需的文档，并通过增强 `Triple` 类使 LLVM 意识到新的架构。文档还包括 ELF 文件格式的重定位定义，并且你为 LLVM 添加了对这些定义的支持。

然后，你学习了目标描述中的寄存器定义和指令定义，并使用生成的 C++ 源代码实现了指令汇编器和反汇编器。

在下一章中，我们将向后端添加代码生成功能。
