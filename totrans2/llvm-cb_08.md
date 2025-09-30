# Chapter 8. Writing an LLVM Backend

In this chapter, we will cover the following recipes:

*   Defining registers and register sets
*   Defining the calling convention
*   Defining the instruction set
*   Implementing frame lowering
*   Printing an instruction
*   Selecting an instruction
*   Adding instruction encoding
*   Supporting a subtarget
*   Lowering to multiple instructions
*   Registering a target

# Introduction

The ultimate goal of a compiler is to produce a target code, or an assembly code that can be converted into object code and executed on the actual hardware. To generate the assembly code, the compiler needs to know the various aspects of the architecture of the target machine—the registers, instruction set, calling convention, pipeline, and so on. There are lots of optimizations that can be done in this phase as well.

LLVM has its own way of defining the target machine. It uses `tablegen` to specify the target registers, instructions, calling convention, and so on. The `tablegen` function eases the way we describe a large set of architecture properties in a programmatic way.

LLVM has a pipeline structure for the backend, where instructions travel through phases like this; from the LLVM IR to `SelectionDAG`, then to `MachineDAG`, then to `MachineInstr`, and finally to `MCInst`.

The IR is converted into SelectionDAG (**DAG** stands for **Directed Acyclic Graph**). Then SelectionDAG legalization occurs where illegal instructions are mapped on the legal operations permitted by the target machine. After this stage, SelectionDAG is converted to MachineDAG, which is basically an instruction selection supported by the backend.

CPUs execute a linear sequence of instructions. The goal of the scheduling step is to linearize the DAG by assigning an order to its operations. LLVM's code generator employs clever heuristics (such as register pressure reduction) to try and produce a schedule that will result in faster code. Register allocation policies also play an important role in producing better LLVM code.

This chapter describes how to build an LLVM toy backend from scratch. By the end of this chapter, we will be able to generate assembly code for a sample toy backend.

## A sample backend

The sample backend considered in this chapter is a simple RISC-type architecture, with a few registers (say r0-r3), a stack pointer (sp), and a link register (lr), for storing the return address.

The calling convention of this toy backend is similar to the ARM architecture—arguments passed to the function will be stored in register sets r0-r1, and the return value will be stored in r0.

# Defining registers and registers sets

This recipe shows you how to define registers and register sets in `.td` files. The `tablegen` function will convert this `.td` file into `.inc` files, which will be the `#include` declarative in our `.cpp` files and refer to registers.

## Getting ready

We have defined our toy target machine to have four registers (r0-r3), a stack register (sp), and a link register (lr). These can be specified in the `TOYRegisterInfo.td` file. The `tablegen` function provides the `Register` class, which can be extended to specify the registers.

## How to do it…

To define the backend architecture using target descriptor files, proceed with the following steps.

1.  Create a new folder in `lib/Target` named `TOY`:

    [PRE0]

2.  Create a new `TOYRegisterInfo.td file` in the new `TOY` folder:

    [PRE1]

3.  Define the hardware encoding, namespace, registers, and the register class:

    [PRE2]

## How it works…

The `tablegen` function processes this `.td` file to generate the `.inc` file, which generally has enums generated for these registers. These enums can be used in the`.cpp` files, in which the registers can be referenced as `TOY::R0`. These `.inc` files will be generated when we build the LLVM project.

## See also

*   To get more details about how registers are defined for more advanced architecture, such as ARM, refer to the `lib/Target/ARM/ARMRegisterInfo.td` file in the source code of LLVM.

# Defining the calling convention

The calling convention specifies how values are passed to and from a function call. Our TOY architecture specifies that two arguments are passed in two registers, r0 and r1, while the remaining ones are passed to the stack. This recipe shows you how to define the calling convention, which will be used in `ISelLowering` (the instruction selection lowering phase discussed in [Chapter 6](part0065.xhtml#aid-1TVKI1 "Chapter 6. Target-independent Code Generator"), *Target Independent Code Generator*) via function pointers.

The calling convention will be defined in the `TOYCallingConv.td` file, which will have primarily two sections—one for defining the return value convention, and the other for defining the argument passing convention. The return value convention specifies how the return values will reside and in which registers. The argument passing convention will specify how the arguments passed will reside and in which registers. The `CallingConv` class is inherited while defining the calling convention of the toy architecture.

## How to do it…

To implement the calling convention, proceed with the following steps:

1.  Create a new `TOYCallingConv.td` file in the `lib/Target/TOY` folder:

    [PRE3]

2.  In that file, define the return value convention, as follows:

    [PRE4]

3.  Also, define the argument passing convention, like this:

    [PRE5]

4.  Define the callee saved register set:

    [PRE6]

## How it works…

In the `.td` file you just read about, it has been specified that the return values of the integer type of 32 bits are stored in the r0 register. Whenever arguments are passed to a function, the first two arguments will be stored in the r0 and r1 registers. It is also specified that whenever any data type, such as an integer of 8 bits or 16 bits, will be encountered, it will be promoted to the 32-bit integer type.

The `tablegen` function generates a `TOYCallingConv.inc` file, which will be referred to in the `TOYISelLowering.cpp` file. The two target `hook` functions used to define argument handling are `LowerFormalArguments()` and `LowerReturn()`.

## See also

*   To see a detailed implementation of advanced architectures, such as ARM, look into the `lib/Target/ARM/ARMCallingConv.td` file

# Defining the instruction set

The instruction set of an architecture varies according to various features present in the architecture. This recipe demonstrates how instruction sets are defined for target architecture.

Three things are defined in the instruction target description file: operands, the assembly string and the instruction pattern. The specification contains a list of definitions or outputs, and a list of uses or inputs. There can be different operand classes, such as the `Register` class, and the immediate and more complex `register + imm` operands.

Here, a simple add instruction definition that takes two registers as operands is demonstrated.

## How to do it…

To define an instruction set using target descriptor files, proceed with the following steps.

1.  Create a new file called `TOYInstrInfo.td` in the `lib/Target/TOY` folder:

    [PRE7]

2.  Specify the operands, assembly string, and instruction pattern for the `add` instruction between two register operands:

    [PRE8]

## How it works…

The `add` register to the register instruction specifies `$dst` as the result operand, which belongs to the `General Register` type class; inputs `$src1` and `$src2` as two input operands, which also belong to the `General Register` type class; and the instruction assembly string as `"add $dst, $src1, $src2"` of the 32-bit integer type.

So, an assembly will be generated for `add` between two registers, like this:

[PRE9]

The preceding code indicates to add the r0 and r1 register contents and store the result in the r0 register.

## See also

*   Many instructions will have the same type of instruction pattern—ALU instructions such as `add`, `sub`, and so on. In cases such as this multiclass can be used to define the common properties. For more detailed information about the various types of instruction sets for advanced architecture, such as ARM, refer to the `lib/Target/ARM/ARMInstrInfo.td file`

# Implementing frame lowering

This recipe talks about frame lowering for target architecture. Frame lowering involves emitting the prologue and epilogue of the function call.

## Getting ready

### Note

Two functions need to be defined for frame lowering, namely `TOYFrameLowering::emitPrologue()` and `TOYFrameLowering::emitEpilogue()`.

## How to do it…

The following functions are defined in the `TOYFrameLowering.cpp` file in the `lib/Target/TOY` folder:

1.  The `emitPrologue` function can be defined as follows:

    [PRE10]

2.  The `emitEpilogue` function can be defined like this:

    [PRE11]

3.  Here are some helper functions used to determine the offset for the `ADD` stack operation:

    [PRE12]

4.  The following are some more helper functions used to compute the stack size:

    [PRE13]

## How it works…

The `emitPrologue` function first computes the stack size to determine whether the prologue is required at all. Then it adjusts the stack pointer by calculating the offset. For the epilogue, it first checks whether the epilogue is required or not. Then it restores the stack pointer to what it was at the beginning of the function.

For example, consider this input IR:

[PRE14]

The TOY assembly generated will look like this:

[PRE15]

## See also

*   For advanced architecture frame lowering, such as in ARM, refer to the `lib/Target/ARM/ARMFrameLowering.cpp` file.

# Printing an instruction

Printing an assembly instruction is an important step in generating target code. Various classes are defined that work as a gateway to the streamers. The instruction string is provided by the `.td` file defined earlier.

## Getting ready

The first and foremost step for printing instructions is to define the instruction string in the `.td` file, which was done in the *Defining the instruction set* recipe.

## How to do it…

Perform the following steps:

1.  Create a new folder called `InstPrinter` inside the `TOY` folder:

    [PRE16]

2.  In a new file, called `TOYInstrFormats.td`, define the `AsmString` variable:

    [PRE17]

3.  Create a new file called `TOYInstPrinter.cpp`, and define the `printOperand` function, as follows:

    [PRE18]

4.  Also, define a function to print the register names:

    [PRE19]

5.  Define a function to print the instruction:

    [PRE20]

6.  It also requires `MCASMinfo` to be specified to print the instruction. This can be done by defining the `TOYMCAsmInfo.h` and `TOYMCAsmInfo.cpp` files.

    The `TOYMCAsmInfo.h` file can be defined as follows:

    [PRE21]

    The `TOYMCAsmInfo.cpp` file can be defined like this:

    [PRE22]

7.  Define the `LLVMBuild.txt` file for the instruction printer:

    [PRE23]

8.  Define `CMakeLists.txt`:

    [PRE24]

## How it works…

When the final compilation takes place, the **llc** tool—a static compiler—will generate the assembly of the TOY architecture.

For example, the following IR, when given to the llc tool, will generate an assembly as shown:

[PRE25]

# Selecting an instruction

An IR instruction in DAG needs to be lowered to a target-specific instruction. The SDAG node contains IR, which needs to be mapped on machine-specific DAG nodes. The outcome of the selection phase is ready for scheduling.

## Getting ready

1.  For selecting a machine-specific instruction, a separate class, `TOYDAGToDAGISel`, needs to be defined. To compile the file containing this class definition, add the filename to the `CMakeLists.txt` file in the `TOY` folder:

    [PRE26]

2.  A pass entry needs to be added in the `TOYTargetMachine.h` and `TOYTargetMachine.cpp` files:

    [PRE27]

3.  The following code in `TOYTargetMachine.cpp` will create a pass in the instruction selection stage:

    [PRE28]

## How to do it…

To define an instruction selection function, proceed with the following steps:

1.  Create a file called `TOYISelDAGToDAG.cpp`:

    [PRE29]

2.  Include the following files:

    [PRE30]

3.  Define a new class called `TOYDAGToDAGISel` as follows, which will inherit from the `SelectionDAGISel` class:

    [PRE31]

4.  The most important function to define in this class is `Select()`, which will return an `SDNode` object specific to the machine instruction:

    Declare it in the class:

    [PRE32]

    Define it further as follows:

    [PRE33]

5.  Another important function is used to define the address selection function, which will calculate the base and offset of the address for load and store operations.

    Declare it as shown here:

    [PRE34]

    Define it further, like this:

    [PRE35]

6.  The `createTOYISelDag` pass converts a legalized DAG into a toy-specific DAG, ready for instruction scheduling in the same file:

    [PRE36]

## How it works…

The `TOYDAGToDAGISel::Select()` function of `TOYISelDAGToDAG.cpp` is used for the selection of the OP code DAG node, while `TOYDAGToDAGISel::SelectAddr()` is used for the selection of the DATA DAG node with the `addr` type. Note that if the address is global or external, we return false for the address, since its address is calculated in the global context.

## See also

*   For details on the selection of DAG for machine instructions of complex architectures, such as ARM, look into the `lib/Target/ARM/ARMISelDAGToDAG.cpp` file in the LLVM source code.

# Adding instruction encoding

If the instructions need to be specific for how they are encoded with respect to bit fields, this can be done by specifying the bit field in the `.td` file when defining an instruction.

## How to do it…

To include instruction encoding while defining instructions, proceed with the following steps:

1.  A register operand that will be used to register the `add` instruction will have some defined encoding for its instruction. The size of the instruction is 32 bits, and the encoding for it is as follows:

    [PRE37]

    This can be achieved by specifying the preceding bit pattern in the `.td` files

2.  In the `TOYInstrFormats.td` file, define a new variable, called `Inst`:

    [PRE38]

3.  In the `TOYInstrInfo.td` file, define an instruction encoding:

    [PRE39]

4.  In the `TOY/MCTargetDesc` folder, in the `TOYMCCodeEmitter.cpp` file, the encoding function will be called if the machine instruction operand is a register:

    [PRE40]

5.  Also, in the same file, a function used to encode the instruction is specified:

    [PRE41]

## How it works…

In the `.td` files, the encoding of an instruction has been specified—the bits for the operands, the destination, flag conditions, and opcode of the instruction. The machine code emitter gets these encodings from the `.inc` file generated by `tablegen` from the `.td` files through function calls. It encodes these instructions and emits the same for instruction printing.

## See also

*   For complex architecture such as ARM, see the `ARMInstrInfo.td` and `ARMInstrInfo.td` files in the `lib/Target/ARM` directory of the LLVM trunk

# Supporting a subtarget

A target may have a subtarget—typically, a variant with instructions—way of handling operands, among others. This subtarget feature can be supported in the LLVM backend. A subtarget may contain some additional instructions, registers, scheduling models, and so on. ARM has subtargets such as NEON and THUMB, while x86 has subtarget features such as SSE, AVX, and so on. The instruction set differs for the subtarget feature, for example, NEON for ARM and SSE/AVX for subtarget features that support vector instructions. SSE and AVX also support the vector instruction set, but their instructions differ from each other.

## How to do it…

This recipe will demonstrate how to add a support subtarget feature in the backend. A new class that will inherit the `TargetSubtargetInfo` class has to be defined:

1.  Create a new file called `TOYSubtarget.h`:

    [PRE42]

2.  Include the following files:

    [PRE43]

3.  Define a new class, called `TOYSubtarget`, with some private members that have information on the data layout, target lowering, target selection DAG, target frame lowering, and so on:

    [PRE44]

4.  Declare its constructor:

    [PRE45]

    This constructor initializes the data members to match that of the specified triplet.

5.  Define some helper functions to return the class-specific data:

    [PRE46]

6.  Create a new file called `TOYSubtarget.cpp`, and define the constructor as follows:

    [PRE47]

The subtarget has its own data layout defined, with other information such as frame lowering, instruction information, subtarget information, and so on.

## See also

*   To dive into the details of subtarget implementation, refer to the `lib/Target/ARM/ARMSubtarget.cpp` file in the LLVM source code

# Lowering to multiple instructions

Let's take an example of implementing a 32-bit immediate load with high/low pairs, where MOVW implies moving a 16-bit low immediate and a clear 16 high bit, and MOVT implies moving a 16-bit high immediate.

## How to do it…

There can be various ways to implement this multiple instruction lowering. We can do this by using pseudo-instructions or in the selection DAG-to-DAG phase.

1.  To do it without pseudo-instructions, define some constraints. The two instructions must be ordered. MOVW clears the high 16 bits. Its output is read by MOVT to fill the high 16 bits. This can be done by specifying the constraints in tablegen:

    [PRE48]

    The second way is to define a pseudo-instruction in the `.td` file:

    [PRE49]

2.  The pseudo-instruction is then lowered by a target function in the `TOYInstrInfo.cpp` file:

    [PRE50]

3.  Compile the entire LLVM project:

    For example, an `ex.ll` file with IR will look like this:

    [PRE51]

    The assembly generated will look like this:

    [PRE52]

## How it works…

The first instruction, `movw`, will move 1 in the lower 16 bits and clear the high 16 bits. So in r1, `0x00000001` will be written by the first instruction. In the next instruction, `movt`, the higher 16 bits will be written. So in r1, `0x0001XXXX` will be written, without disturbing the lower bits. Finally, the r1 register will have `0x00010001` in it. Whenever a pseudo-instruction is encountered as specified in the `.td` file, its expand function is called to specify what the pseudo-instruction will expand to.

In the preceding case, the `mov32` immediate was to be implemented by two instructions: `movw` (the lower 16 bits) and `movt` (the higher 16 bits). It was marked as a pseudo-instruction in the `.td` file. When this pseudo-instruction needs to be emitted, its expand function is called, which builds two machine instructions: `MOVLOi16` and `MOVHIi16`. These map to the `movw` and `movt` instructions of the target architecture.

## See also

*   To dive deep into implementing such lowering of multiple instructions, look at the ARM target implementation in the LLVM source code in the `lib/Target/ARM/ARMInstrInfo.td` file.

# Registering a target

For running the llc tool in the TOY target architecture, it has to be registered with the llc tool. This recipe demonstrates which configuration files need to be modified to register a target. The build files are modified in this recipe.

## How to do it…

To register a target with a static compiler, follow these steps:

1.  First, add the entry of the TOY backend to `llvm_root_dir/CMakeLists.txt`:

    [PRE53]

2.  Then add the toy entry to `llvm_root_dir/include/llvm/ADT/Triple.h`:

    [PRE54]

3.  Add the toy entry to `llvm_root_dir/include/llvm/ MC/MCExpr.h`:

    [PRE55]

4.  Add the toy entry to `llvm_root_dir/include/llvm/ Support/ELF.h`:

    [PRE56]

5.  Then, add the toy entry to `lib/MC/MCExpr.cpp`:

    [PRE57]

6.  Next, add the toy entry to `lib/Support/Triple.cpp`:

    [PRE58]

7.  Add the toy directory entry to `lib/Target/LLVMBuild.txt`:

    [PRE59]

8.  Create a new file called `TOY.h` in the `lib/Target/TOY` folder:

    [PRE60]

9.  Create a new folder called `TargetInfo` in the `lib/Target/TOY` folder. Inside that folder, create a new file called `TOYTargetInfo.cpp`, as follows:

    [PRE61]

10.  In the same folder, create the `CMakeLists.txt` file:

    [PRE62]

11.  Create an `LLVMBuild.txt` file:

    [PRE63]

12.  In the `lib/Target/TOY` folder, create a file called `TOYTargetMachine.cpp`:

    [PRE64]

13.  Create a new folder called `MCTargetDesc` and a new file called `TOYMCTargetDesc.h`:

    [PRE65]

14.  Create one more file, called `TOYMCTargetDesc.cpp`, in the same folder:

    [PRE66]

15.  In the same folder, create an `LLVMBuild.txt` file:

    [PRE67]

16.  Create a `CMakeLists.txt` file:

    [PRE68]

## How it works…

Build the enitre LLVM project, as follows:

[PRE69]

Here, we have specified that we are building the LLVM compiler for the toy target. After the build completes, check whether the TOY target appears with the `llc` command:

[PRE70]

## See also

*   For a more detailed description about complex targets that involve pipelining and scheduling, follow the chapters in *Tutorial: Creating an LLVM Backend for the Cpu0 Architecture* by Chen Chung-Shu and Anoushe Jamshidi