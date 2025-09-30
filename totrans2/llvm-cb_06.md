# Chapter 6. Target-independent Code Generator

In this chapter, we will cover the following recipes:

*   The life of an LLVM IR instruction
*   Visualizing the LLVM IR CFG using GraphViz
*   Describing the target using TableGen
*   Defining an instruction set
*   Adding a machine code descriptor
*   Implementing the MachineInstrBuilder class
*   Implementing the MachineBasicBlock class
*   Implementing the MachineFunction class
*   Writing an instruction selector
*   Legalizing SelectionDAG
*   Optimizing SelectionDAG
*   Selecting instructions from the DAG
*   Scheduling instructions in SelectionDAG

# Introduction

After optimizing the LLVM IR, it needs to be converted into machine instructions for execution. The machine-independent code generator interface gives an abstract layer that helps convert IR into machine instructions. In this phase, the IR is converted into SelectionDAG (**DAG** stands for **Directed Acyclic Graph**). Various phases work on the nodes of SelectionDAG. This chapter describes the important phases in target-independent code generation.

# The life of an LLVM IR instruction

In previous chapters, we saw how high-level language instructions, statements, logical blocks, function calls, loops, and so on get transformed into the LLVM IR. Various optimization passes then process the IR to make it more optimal. The IR generated is in the SSA form and, in abstract format, almost independent of any high- or low-level language constraints, which facilitates optimization passes running on it. There might be some optimizations that are target-specific and take place later, when the IR gets converted into machine instructions.

After we get an optimal LLVM IR, the next phase is to convert it into target-machine-specific instructions. LLVM uses the SelectionDAG approach to convert the IR into machine instructions. The Linear IR is converted into SelectionDAG, a DAG that represents instructions as nodes. The SDAG then goes through various phases:

*   The SelectionDAG is created out of LLVM IR
*   Legalizing SDAG nodes
*   DAG combine optimization
*   Instruction selection from the target instruction
*   Scheduling and emitting a machine instruction
*   Register allocation—SSA destruction, register assignment, and register spilling
*   Emitting code

All the preceding stages are modularized in LLVM.

## C Code to LLVM IR

The first step is to convert the front end language example to LLVM IR. Let's take an example:

[PRE0]

Its LLVM IR will be as follows:

[PRE1]

## IR optimization

The IR then goes through various optimization passes, as described in previous chapters. The IR, in the transformation phase, goes through the `InstCombiner::visitSDiv()` function in the `InstCombine` pass. In that function, it also goes through the `SimplifySDivInst()` function and tries to check whether an opportunity exists to further simplify the instruction.

## LLVM IR to SelectionDAG

After the IR transformations and optimizations are over, the LLVM IR instruction passes through a **Selection DAG node** incarnation. Selection DAG nodes are created by the `SelectionDAGBuilder` class. The `SelectionDAGBuilder::visit()` function call from the `SelectionDAGISel` class visits each IR instruction for creating an `SDAGNode` node. The method that handles an `SDiv` instruction is `SelectionDAGBuilder::visitSDiv`. It requests a new `SDNode` node from the DAG with the`ISD::SDIV` opcode, which then becomes a node in the DAG.

## SelectionDAG legalization

The `SelectionDAG` node created may not be supported by the target architecture. In the initial phase of Selection DAG, these unsupported nodes are called *illegal*. Before the `SelectionDAG` machinery actually emits machine instructions from the DAG nodes, these undergo a few other transformations, legalization being one of the important phases.

The legalization of `SDNode` involves type and operation legalization. The target-specific information is conveyed to the target-independent algorithms via an interface called `TargetLowering`. This interface is implemented by the target and, describes how LLVM IR instructions should be lowered to legal `SelectionDAG` operations. For instance, x86 lowering is implemented in the `X86TargetLowering` interface. The `setOperationAction()` function specifies whether the ISD node needs to be expanded or customized by operation legalization. When `SelectionDAGLegalize::LegalizeOp` sees the expand flag, it replaces the `SDNode` node with the parameter specified in the `setOperationAction()` call.

## Conversion from target-independent DAG to machine DAG

Now that we have legalized the instruction, `SDNode` should be converted to `MachineSDNode`. The machine instructions are described in a generic table-based fashion in the target description `.td` files. Using `tablegen`, these files are then converted into `.inc` files that have registers/instructions as enums to refer to in the C++ code. Instructions can be selected by an automated selector, `SelectCode`, or they can be handled specifically by writing a customized `Select` function in the `SelectionDAGISel` class. The DAG node created at this step is a `MachineSDNode` node, a subclass of `SDNode` that holds the information required to construct an actual machine instruction but is still in the DAG node form.

## Scheduling instructions

A machine executes a linear set of instructions. So far, we have had machine instructions that are still in the DAG form. To convert a DAG into a linear set of instructions, a topological sort of the DAG can yield the instructions in linear order. However, the linear set of instructions generated might not result in the most optimized code, and may cause execution delays due to dependencies among instructions, register pressure, and pipeline stalling issues. Therein comes the concept of scheduling instructions. Since each target has its own set of registers and customized pipelining of the instructions, each target has its own hook for scheduling and calculating heuristics to produce optimized, faster code. After calculating the best possible way to arrange instructions, the scheduler emits the machine instructions in the machine basic block, and finally destroys the DAG.

## Register allocation

The registers allocated are virtual registers after the machine instructions are emitted. Practically, an infinite number of virtual registers can be allocated, but the actual target has a limited number of registers. These limited registers need to be allocated efficiently. If this is not done, some registers have to be spilled onto the memory, and this may result in redundant load/store operations. This will also result in wastage of CPU cycles, slowing down the execution as well as increasing the memory footprint.

There are various register allocation algorithms. An important analysis is done when allocating registers—liveness of variables and live interval analysis. If two variables live in the same interval (that is, if there exists an interval interference), then they cannot be allocated the same register. An interference graph is created by analyzing liveness, and a graph coloring algorithm can be used to allocate the registers. This algorithm, however, takes quadratic time to run. Hence, it may result in longer compilation time.

LLVM employs a greedy approach for register allocation, where variables that have large live ranges are allocated registers first. Small ranges fit into the gaps of registers available, resulting in less spill weight. Spilling is a load-store operation that occurs because no registers are available to be allocated. Spill weight is the cost of operations involved in the spilling. Sometimes, live range splitting also takes place to accommodate variables into the registers.

Note that the instructions are in the SSA form before register allocation. Now, the SSA form cannot exist in the real world because of the limited number of registers available. In some types of architecture, some instructions require fixed registers.

## Code emission

Now that the original high-level code has been translated into machine instructions, the next step is to emit the code. LLVM does this in two ways; the first is JIT, which directly emits the code to the memory. The second way is by using the MC framework to emit assembly and object files for all backend targets.The `LLVMTargetMachine::addPassesToEmitFile` function is responsible for defining the sequence of actions required to emit an object file. The actual MI-to-MCInst translation is done in the `EmitInstruction` function of the `AsmPrinter` interface. The static compiler tool, llc, generates assembly instructions for a target. Object file (or assembly code) emission is done by implementing the `MCStreamer` interface.

# Visualizing LLVM IR CFG using GraphViz

The LLVM IR control flow graph can be visualized using the **GraphViz** tool. It gives a visual depiction of the nodes formed and how the code flow follows in the IR generated. Since the important data structures in LLVM are graphs, this can be a very useful way to understand the IR flow when writing a custom pass or studying the behavior of the IR pattern.

## Getting ready

1.  To install `graphviz` on Ubuntu, first add its `ppa` repository:

    [PRE2]

2.  Update the package repository:

    [PRE3]

3.  Install `graphviz`:

    [PRE4]

    ### Note

    If you get the `graphviz : Depends: libgraphviz4 (>= 2.18) but it is not going to be installed` error, run the following commands:

    [PRE5]

    Then install `graphviz` again with the following command:

    [PRE6]

## How to do it…

1.  Once the IR has been converted to DAG, it can be viewed in different phases. Create a test.ll file with the following code:

    [PRE7]

2.  To display the DAG after it is built, before the first optimization pass, enter the following command:

    [PRE8]

    The following diagram shows the DAG before the first optimization pass:

    ![How to do it…](img/image00255.jpeg)
3.  To display the DAG before legalization, run this command:

    [PRE9]

    Here is a diagram that shows the DAG before the legalization phase:

    ![How to do it…](img/image00256.jpeg)
4.  To display the DAG before the second optimization pass, run the following command:

    [PRE10]

    The following diagram shows the DAG before the second optimization pass:

    ![How to do it…](img/image00257.jpeg)
5.  To display the DAG before the selection phase, enter this command:

    [PRE11]

    Here is a diagram that shows the DAG before the selection phase:

    ![How to do it…](img/image00258.jpeg)
6.  To display the DAG before scheduling, run the following command:

    [PRE12]

    The following diagram shows the DAG before the scheduling phase:

    ![How to do it…](img/image00259.jpeg)
7.  To display the scheduler's dependency graph, run this command:

    [PRE13]

    This diagram shows the scheduler's dependency graph:

    ![How to do it…](img/image00260.jpeg)

Notice the difference in the DAG before and after the legalize phase. The `sdiv` node has been converted into an `sdivrem` node. The x86 target doesn't support the `sdiv` node but supports the `sdivrem` instruction. In a way, the `sdiv` instruction is illegal for the x86 target. The legalize phase converted it into an `sdivrem` instruction, which is supported by the x86 target.

Also note the difference in the DAG before and after the instruction selection (ISel) phase. Target-machine-independent instructions such as `Load` are converted into the `MOV32rm` machine code (which means, move 32-bit data from the memory to the register). The ISel phase is an important phase that will be described in later recipes.

Observe the scheduling units for the DAG. Each unit is linked to other units, which shows the dependency between them. This dependency information is very important for deciding scheduling algorithms. In the preceding case, scheduling unit 0 (SU0) is dependent on scheduling unit 1 (SU1). So, the instructions in SU0 cannot be scheduled before the instructions in SU1\. SU1 is dependent on SU2, and so is SU2 on SU3.

## See also

*   For more details on how to view graphs in debug mode, go to [http://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code](http://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code)

# Describing targets using TableGen

The target architecture can be described in terms of the registers present, the instruction set, and so on. Describing each of them manually is a tedious task. `TableGen` is a tool for backend developers that describes their target machine with a declarative language—`*.td`. The `*.td` files will be converted to enums, DAG-pattern matching functions, instruction encoding/decoding functions, and so on, which can then be used in other C++ files for coding.

To define registers and the register set in the target description's `.td` files, `tablegen` will convert the intended `.td` file into `.inc` files, which will be `#include` syntax in our `.cpp` files referring to the registers.

## Getting ready

Let's assume that the sample target machine has four registers, `r0-r3`; a stack register, `sp`; and a link register, `lr`. These can be specified in the `SAMPLERegisterInfo.td` file. `TableGen` provides the `Register` class, which can be extended to specify registers.

## How to do it

1.  Create a new folder in `lib/Target` named `SAMPLE`:

    [PRE14]

2.  Create a new file called `SAMPLERegisterInfo.td` in the new `SAMPLE` folder:

    [PRE15]

3.  Define the hardware encoding, namespace, registers, and register class:

    [PRE16]

## How it works

`TableGen` processes this `.td` file to generate the `.inc` files, which have registers represented in the form of enums that can be used in the `.cpp` files. These `.inc` files will be generated when we build the LLVM project.

## See also

*   To get more details on how registers are defined for more advanced architecture, such as the x86, refer to the `X86RegisterInfo.td` file located at `llvm_source_code/lib/Target/X86/`

# Defining an instruction set

The instruction set of an architecture varies according to various features present in the architecture. This recipe demonstrates how instruction sets are defined for the target architecture.

## Getting ready

Three things are defined in the instruction target description file: operands, an assembly string, and an instruction pattern. The specification contains a list of definitions or outputs and a list of uses or inputs. There can be different operand classes such as the register class, and immediate or more complex `register + imm` operands.

Here, a simple add instruction definition is demonstrated. It takes two registers for the input and one register for the output.

## How to do it…

1.  Create a new file called `SAMPLEInstrInfo.td` in the `lib/Target/SAMPLE` folder:

    [PRE17]

2.  Specify the operands, assembly string, and instruction pattern for the add instruction between two register operands:

    [PRE18]

## How it works…

The `add` register instruction specifies `$dst` as the resultant operand, which belongs to the general register type class; the `$src1` and `$src2` inputs as two input operands, which also belong to the general register class; and the instruction assembly string as `add $dst, $src1, $src2`, which is of the 32-bit integer type.

So, an assembly will be generated for add between two registers, like this:

[PRE19]

This tells us to add the `r0` and `r1` registers' content and store the result in the `r0` register.

## See also

*   For more detailed information on various types of instruction sets for advanced architecture, such as the x86, refer to the `X86InstrInfo.td` file located at `lib/Target/X86/`
*   Detailed information of how target-specific things are defined will be covered in [Chapter 8](part0087.xhtml#aid-2IV0U1 "Chapter 8. Writing an LLVM Backend"), *Writing an LLVM Backend*. Some concepts might get repetitive, as the preceding recipes were described in brief to get a glimpse of the target architecture description and get a foretaste of the upcoming recipes

# Adding a machine code descriptor

The LLVM IR has functions, which have basic blocks. Basic blocks in turn have instructions. The next logical step is to convert those IR abstract blocks into machine-specific blocks. LLVM code is translated into a machine-specific representation formed from the `MachineFunction`, `MachineBasicBlock`, and `MachineInstr` instances. This representation contains instructions in their most abstract form—that is, having an opcode and a series of operands.

## How it's done…

Now the LLVM IR instruction has to be represented in the machine instruction. Machine instructions are instances of the `MachineInstr` class. This class is an extremely abstract way of representing machine instructions. In particular, it only keeps track of an opcode number and a set of operands. The opcode number is a simple unsigned integer that has a meaning only for a specific backend.

Let's look at some important functions defined in the `MachineInstr.cpp` file:

The `MachineInstr` constructor:

[PRE20]

This constructor creates an object of `MachineInstr` class and adds the implicit operands. It reserves space for the number of operands specified by the `MCInstrDesc` class.

One of the important functions is `addOperand`. It adds the specified operand to the instruction. If it is an implicit operand, it is added at the end of the operand list. If it is an explicit operand, it is added at the end of the explicit operand list, as shown here:

[PRE21]

The target architecture has some memory operands as well. To add those memory operands, a function called `addMemOperands()` is defined:

[PRE22]

The `setMemRefs()` function is the primary method for setting up a `MachineInstr` `MemRefs` list.

## How it works…

The `MachineInstr` class has an MCID member, with the `MCInstrDesc` type for describing the instruction, a `uint8_t` flags member, a memory reference member (`mmo_iterator` `MemRefs`), and a vector member of the `std::vector<MachineOperand`> operands. In terms of methods, the `MachineInstr` class provides the following:

*   A basic set of `get**` and `set**` functions for information queries, for example, `getOpcode()`, `getNumOperands()`, and so on
*   Bundle-related operations, for example, `isInsideBundle()`
*   Checking whether the instruction has certain properties, for example, `isVariadic()`, `isReturn()`, `isCall()`, and so on
*   Machine instruction manipulation, for example, `eraseFromParent()`
*   Register-related operations, such as `ubstituteRegister()`, `addRegisterKilled()`, and so on
*   Machine-instruction-creating methods, for example, `addOperand()`, `setDesc()`, and so on

Note that, although the `MachineInstr` class provides machine-instruction-creating methods, a dedicated function called `BuildMI()`, based on the `MachineInstrBuilder` class, is more convenient.

# Implementing the MachineInstrBuilder class

The `MachineInstrBuilder` class exposes a function called `BuildMI()`. This function is used to build machine instructions.

## How to do it…

Machine instructions are created by using the `BuildMI` functions, located in the `include/llvm/CodeGen/MachineInstrBuilder.h` file. The `BuildMI` functions make it easy to build arbitrary machine instructions.

For example, you can use `BuildMI` in code snippets for the following purposes:

1.  To create a `DestReg = mov 42` (rendered in the x86 assembly as `mov DestReg, 42`) instruction:

    [PRE23]

2.  To create the same instruction, but insert it at the end of a basic block:

    [PRE24]

3.  To create the same instruction, but insert it before a specified iterator point:

    [PRE25]

4.  To create a self-looping branch instruction:

    [PRE26]

## How it works…

The `BuildMI()` function is required for specifying the number of operands that the machine instruction will take, which facilitates efficient memory allocation. It is also required to specify whether operands use values or definitions.

# Implementing the MachineBasicBlock class

Similar to basic blocks in the LLVM IR, a `MachineBasicBlock` class has a set of machine instructions in sequential order. Mostly, a `MachineBasicBlock` class maps to a single LLVM IR basic block. However, there can be cases where multiple `MachineBasicBlocks` classes map to a single LLVM IR basic block. The `MachineBasicBlock` class has a method, called `getBasicBlock()`, that returns the IR basic block to which it is mapping.

## How to do it…

The following steps show how machine basic blocks are added:

1.  The `getBasicBlock` method will return only the current basic block:

    [PRE27]

2.  The basic blocks have successor as well as predecessor basic blocks. To keep track of those, vectors are defined as follows:

    [PRE28]

3.  An `insert` function should be added to insert a machine instruction into the basic block:

    [PRE29]

4.  A function called `SplitCriticalEdge()` splits the critical edges from this block to the given successor block, and returns the newly created block, or null if splitting is not possible. This function updates the `LiveVariables`, `MachineDominatorTree`, and `MachineLoopInfo` classes:

    [PRE30]

### Note

The full implementation of the preceding code is shown in the `MachineBasicBlock.cpp` file located at `lib/CodeGen/`.

## How it works…

As listed previously, several representative functions of different categories form the interface definition of the `MachineBasicBlock` class. The `MachineBasicBlock` class keeps a list of machine instructions such as `typedef ilist<MachineInstr>` instructions, instructions `Insts`, and the original LLVM BB (basic block). It also provides methods for purposes such as these:

*   BB information querying (for example, `getBasicBlock()` and `setHasAddressTaken()`)
*   BB-level manipulation (for example, `moveBefore()`, `moveAfter()`, and `addSuccessor()`)
*   Instruction-level manipulation (for example, `push_back()`, `insertAfter()`, and so on)

## See also

*   To see a detailed implementation of the `MachineBasicBlock class`, go through the `MachineBasicBlock.cpp` file located at `lib/CodeGen/`

# Implementing the MachineFunction class

Similar to the LLVM IR `FunctionBlock` class, a `MachineFunction` class contains a series of `MachineBasicBlocks` classes. These `MachineFunction` classes map to LLVM IR functions that are given as input to the instruction selector. In addition to a list of basic blocks, the `MachineFunction` class contains the `MachineConstantPool`, `MachineFrameInfo`, `MachineFunctionInfo`, and `MachineRegisterInfo` classes.

## How to do it…

Many functions are defined in the `MachineFunction` class, which does specific tasks. There are also many class member objects that keep information, such as the following:

*   `RegInfo` keeps information about each register that is in use in the function:

    [PRE31]

*   `MachineFrameInfo` keeps track of objects allocated on the stack:

    [PRE32]

*   `ConstantPool` keeps track of constants that have been spilled to the memory:

    [PRE33]

*   `JumpTableInfo` keeps track of jump tables for switch instructions:

    [PRE34]

*   The list of machine basic blocks in the function:

    [PRE35]

*   The `getFunction` function returns the LLVM function that the current machine code represents:

    [PRE36]

*   `CreateMachineInstr` allocates a new `MachineInstr` class:

    [PRE37]

## How it works…

The `MachineFunction` class primarily contains a list of `MachineBasicBlock` objects (`typedef ilist<MachineBasicBlock> BasicBlockListType; BasicBlockListType BasicBlocks;`), and defines various methods for retrieving information about the machine function and manipulating the objects in the basic blocks member. A very important point to note is that the `MachineFunction` class maintains the **control flow graph** (**CFG**) of all basic blocks in a function. Control flow information in CFG is crucial for many optimizations and analyses. So, it is important to know how the `MachineFunction` objects and the corresponding CFGs are constructed.

## See also

*   A detailed implementation of the `MachineFunction` class can be found in the `MachineFunction.cpp` file located at `lib/Codegen/`

# Writing an instruction selector

LLVM uses the `SelectionDAG` representation to represent the LLVM IR in a low-level data-dependence DAG for instruction selection. Various simplifications and target-specific optimizations can be applied to the `SelectionDAG` representation. This representation is target-independent. It is a significant, simple, and powerful representation used to implement IR lowering to target instructions.

## How to do it…

The following code shows a brief skeleton of the `SelectionDAG` class, its data members, and various methods used to set/retrieve useful information from this class. The `SelectionDAG` class is defined as follows:

[PRE38]

## How it works…

From the preceding code, it can be seen that the `SelectionDAG` class provides lots of target-independent methods to create `SDNode` of various kinds, and retrieves/computes useful information from the nodes in the `SelectionDAG` graph. There are also update and replace methods provided in the `SelectionDAG` class. Most of these methods are defined in the `SelectionDAG.cpp` file. Note that the `SelectionDAG` graph and its node type, `SDNode`, are designed in a way that is capable of storing both target-independent and target-specific information. For example, the `isTargetOpcode()` and `isMachineOpcode()` methods in the `SDNode` class can be used to determine whether an opcode is a target opcode or a machine opcode (target-independent). This is because the same class type, `NodeType`, is used to represent both the opcode of a real target and the opcode of a machine instruction, but with separate ranges.

# Legalizing SelectionDAG

A `SelectionDAG` representation is a target-independent representation of instructions and operands. However, a target may not always support the instruction or data type represented by `SelectionDAG`. In that sense, the initial `SelectionDAG` graph constructed can be called illegal. The DAG legalize phase converts the illegal DAG into a legal DAG supported by the target architecture.

A DAG legalize phase can follow two ways to convert unsupported data types into supported data types—by promoting smaller data types to larger data types, or by truncating larger data types into smaller ones. For example, suppose that a type of target architecture supports only i32 data types. In that case, smaller data types such as i8 and i16 need to be promoted to the i32 type. A larger data type, such as i64, can be expanded to give two i32 data types. The `Sign` and `Zero` extensions can be added so that the result remains consistent in the process of promoting or expanding data types.

Similarly, vector types can be legalized to supported vector types by either splitting the vector into smaller sized vectors (by extracting the elements from the vector), or by widening smaller vector types to larger, supported vector types. If vectors are not supported in the target architecture, then every element of the vector in the IR needs to be extracted in the scalar form.

The legalize phase can also instruct the kind of classes of registers supported for given data.

## How to do it…

The `SelectionDAGLegalize` class consists of various data members, tracking data structures to keep a track of legalized nodes, and various methods that are used to operate on nodes to legalize them. A sample snapshot of the legalize phase code from the LLVM trunk shows the basic skeleton of implementation of the legalize phase, as follows:

[PRE39]

## How it works…

Many function members of the `SelectionDAGLegalize` class, such as `LegalizeOp`, rely on target-specific information provided by the `const TargetLowering &TLI` member (other function members may also depend on the `const TargetMachine &TM` member) in the `SelectionDAGLegalize` class. Let's take an example to demonstrate how legalization works.

There are two types of legalization: type legalization and instruction legalization. Let's first see how type legalization works. Create a `test.ll` file using the following commands:

[PRE40]

The data type in this case is i64\. For the x86 target, which supports only the 32-bit data type, the data type you just saw is illegal. To run the preceding code, the data type has to be converted to i32\. This is done by the DAG Legalization phase.

To view the DAG before type legalization, run the following command line:

[PRE41]

The following figure shows the DAG before type legalization:

![How it works…](img/image00261.jpeg)

To see DAG after type legalization, enter the following command line:

[PRE42]

The following figure shows the DAG after type legalization:

![How it works…](img/image00262.jpeg)

On observing the DAG nodes carefully, you can see that every operation before legalization had the i64 type. This was because the IR had the data type i64—one-to-one mapping from the IR instruction to the DAG nodes. However, the target x86 machine supports only the i32 type (32-bit integer type). The DAG legalize phase converts unsupported i64 types to supported i32 types. This operation is called expanding—splitting larger types into smaller types. For example, in a target accepting only i32 values, all i64 values are broken down to pairs of i32 values. So, after legalization, you can see that all the operations now have i32 as the data type.

Let's see how instructions are legalized; create a `test.ll` file using the following commands:

[PRE43]

To view the DAG before legalization, enter the following command:

[PRE44]

The following figure shows the DAG before legalization:

![How it works…](img/image00263.jpeg)

To view the DAG after legalization, enter the following command:

[PRE45]

The following figure shows the DAG after the legalization phase:

![How it works…](img/image00264.jpeg)

The DAG, before instruction legalization, consists of `sdiv` instructions. Now, the x86 target does not support the `sdiv` instruction, hence it is illegal for the target. It does, however, support the `sdivrem` instruction. So, the legalization phase involves conversion of the `sdiv` instruction to the `sdivrem` instruction, as visible in the preceding two DAGs.

# Optimizing SelectionDAG

A `SelectionDAG` representation shows data and instructions in the form of nodes. Similar to the `InstCombine` pass in the LLVM IR, these nodes can be combined and optimized to form a minimized `SelectionDAG`. But, it's not just a `DAGCombine` operation that optimizes the SelectionDAG. A `DAGLegalize` phase may generate some unnecessary DAG nodes, which are cleaned up by subsequent runs of the DAG optimization pass. This finally represents the `SelectionDAG` in a more simple and elegant way.

## How to do it…

There are lots and lots of function members (most of them are named like this: `visit**()`) provided in the `DAGCombiner` class to perform optimizations by folding, reordering, combining, and modifying `SDNode` nodes. Note that, from the `DAGCombiner` constructor, we can guess that some optimizations require alias analysis information:

[PRE46]

## How it works…

As seen in the preceding code, some `DAGCombine` passes search for a pattern and then fold the patterns into a single DAG. This basically reduces the number of DAGs, while lowering DAGs. The result is an optimized `SelectionDAG` class.

## See also

*   For a more detailed implementation of the optimized `SelectionDAG` class, see the `DAGCombiner.cpp` file located at `lib/CodeGen/SelectionDAG/`

# Selecting instruction from the DAG

After legalization and DAG combination, the `SelectionDAG` representation is in the optimized phase. However, the instructions represented are still target-independent and need to be mapped on target-specific instructions. The instruction selection phase takes the target-independent DAG nodes as the input, matches patterns in them, and gives the output DAG nodes, which are target-specific.

The `TableGen` DAG instruction selector generator reads the instruction patterns from the `.td` file, and automatically builds parts of the pattern matching code.

## How to do it…

`SelectionDAGISel` is the common base class used for pattern-matching instruction selectors that are based on `SelectionDAG`. It inherits the `MachineFunctionPass` class. It has various functions used to determine the legality and profitability of operations such as folding. The basic skeleton of this class is as follows:

[PRE47]

## How it works…

The instruction selection phase involves converting target-independent instructions to target-specific instructions. The `TableGen` class helps select target-specific instructions. This phase basically matches target-independent input nodes, which gives an output consisting of target-supported nodes.

The `CodeGenAndEmitDAG()` function calls the `DoInstructionSelection()` function, which visits each DAG node and calls the `Select()` function for each node, like this:

[PRE48]

The `Select()` function is an abstract method implemented by the targets. The x86 target implements it in the `X86DAGToDAGISel::Select()` function. The `X86DAGToDAGISel::Select()` function intercepts some nodes for manual matching, but delegates the bulk of the work to the `X86DAGToDAGISel::SelectCode()` function.

The `X86DAGToDAGISel::SelectCod`e function is autogenerated by `TableGen`. It contains the matcher table, followed by a call to the generic `SelectionDAGISel::SelectCodeCommon()` function, passing it the table.

For example:

[PRE49]

To see the DAG before instruction selection, enter the following command line:

[PRE50]

The following figure shows the DAG before the instruction selection:

![How it works…](img/image00265.jpeg)

To see how DAG looks like after the instruction selection, enter the following command:

[PRE51]

The following figure shows the DAG after the instruction selection:

![How it works…](img/image00266.jpeg)

As seen, the `Load` operation is converted into the `MOV32rm` machine code by the instruction selection phase.

## See also

*   To see the detailed implementation of the instruction selection, take a look at the `SelectionDAGISel.cpp` file located at `lib/CodeGen/SelectionDAG/`

# Scheduling instructions in SelectionDAG

So far, we have had `SelectionDAG` nodes consisting of target-supported instructions and operands. However, the code is still in DAG representation. The target architecture executes instructions in sequential form. So, the next logical step is to schedule the `SelectionDAG` nodes.

A scheduler assigns the order of execution of instructions from the DAG. In this process, it takes into account various heuristics, such as register pressure, to optimize the execution order of instructions and to minimize latencies in instruction execution. After assigning the order of execution to the DAG nodes, the nodes are converted into a list of `MachineInstrs` and the `SelectionDAG` nodes are destroyed.

## How to do it…

There are several basic structures that are defined in the `ScheduleDAG.h` file and implemented in the `ScheduleDAG.cpp` file. The `ScheduleDAG` class is a base class for other schedulers to inherit, and it provides only graph-related manipulation operations such as an iterator, DFS, topological sorting, functions for moving nodes around, and so on:

[PRE52]

## How it works…

The scheduling algorithm implements the scheduling of instructions in the `SelectionDAG` class, which involves a variety of algorithms such as topological sorting, depth-first searching, manipulating functions, moving nodes, and iterating over a list of instructions. It takes into account various heuristics, such as register pressure, spilling cost, live interval analysis, and so on to determine the best possible scheduling of instructions.

## See also

*   For a detailed implementation of scheduling instructions, see the `ScheduleDAGSDNodes.cpp`, `ScheduleDAGSDNodes.h`, `ScheduleDAGRRList.cpp`, `ScheduleDAGFast.cpp`, and `ScheduleDAGVLIW.cpp` files located in the `lib/CodeGen/SelectionDAG` folder