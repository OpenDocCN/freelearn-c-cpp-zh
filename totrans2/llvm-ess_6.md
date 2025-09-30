# Chapter 6. IR to Selection DAG phase

Until the previous chapter, we saw how a frontend language can be converted to LLVM IR. We also saw how IR can be transformed into more optimized code. After a series of analysis and transformation passes, the final IR is the most optimized machine independent code. However, the IR is still an abstract representation of the actual machine code. The compiler has to generate target architecture code for execution.

LLVM uses DAG—a directed acyclic graph representation for code generation. The idea is to convert IR into a `SelectionDAG` and then go over a series of phases—DAG combine, legalization, instruction selection, instruction scheduling, etc—to finally allocate registers and emit machine code. Note that register allocation and instruction scheduling take place in an intertwined manner.

We are going to cover following topics in this chapter:

*   Converting IR to selectionDAG
*   Legalizing selectionDAG
*   Optimizing selectionDAG
*   Instruction selection
*   Scheduling and emitting machine instructions
*   Register allocation
*   Code emission

# Converting IR to selectionDAG

An IR instruction can be represented by an SDAG node. The whole set of instructions thus forms an interconnected directed acyclic graph, with each node corresponding to an IR instruction.

For example, consider the following LLVM IR:

[PRE0]

LLVM provides a `SelectionDAGBuilder` interface to create DAG nodes corresponding to IR instructions. Consider the binary operation:

[PRE1]

The following function is called when the given IR is encountered:

[PRE2]

Depending on the opcode—which is `Add` here—the corresponding visit function is invoked. In this case, `visitAdd()` is invoked, which further invokes the `visitBinary()` function. The `visitBinary()` function is as follows:

[PRE3]

This function takes two operands of the binary operator from IR and stores them into `SDValue` type. Then it invokes the `DAG.getNode()` function with opcode of the binary operator. This results in formation of a DAG node, which somewhat looks like the following:

![Converting IR to selectionDAG](img/00009.jpeg)

The operands `0` and `1` are load DAG nodes.

Consider the IR:

[PRE4]

On encountering the `sdiv` instruction, the function `visitSDiv()` is invoked.

[PRE5]

Similar to `visitBinary()`, this function also stores the two operands into `SDValue` gets a DAG node with `ISD::SDIV` as its operator. The node looks like the following:

![Converting IR to selectionDAG](img/00010.jpeg)

In our IR, the operand 0 is `%add`. Operand `1` is `%c`, which is passed as an argument to the function, which transforms to a load node when converting IR to `SelectionDAG`. For implementation of Load DAG node, go through the `visitLoad()` function in the `lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp` file.

After visiting all the IR instructions mentioned earlier, finally the IR is converted to `SelectionDAG` as follows:

![Converting IR to selectionDAG](img/00011.jpeg)

In the preceding diagram, note the following:

*   Black arrows mean data flow dependency
*   Red arrows mean glue dependency
*   Blue dashed arrows mean chain dependency

Glue prevents the two nodes from being broken up during scheduling. Chain dependencies prevent nodes with side effects. A data dependency indicates when an instruction depends on the result of a previous instruction.

# Legalizing SelectionDAG

In the preceding topic, we saw how an IR is converted to `SelectionDAG`. The whole process didn't involve any knowledge of target architecture for which we are trying to generate code. A DAG node might be illegal for the given target architecture. For example, the X86 architecture doesn't support the `sdiv` instruction. Instead, it supports `sdivrem` instruction. This target specific information is conveyed to the `SelectionDAG` phase by the `TargetLowering` interface. Targets implement this interface to describe how LLVM IR instructions should be lowered to legal `SelectionDAG` operations.

In our IR case, we need to 'expand' the `sdiv` instruction to `'sdivrem'` instruction. In the function void `SelectionDAGLegalize::LegalizeOp(SDNode *Node)`, the `TargetLowering::Expand` case is encountered, which invokes the `ExpandNode()` function call on that particular node.

[PRE6]

This function expands SDIV into the SDIVREM node:

[PRE7]

Finally, after legalization, the node becomes `ISD::SDIVREM`:

![Legalizing SelectionDAG](img/00012.jpeg)

Thus the above instruction has been '`legalized`' mapping to the instruction supported on the target architecture. What we saw above was an example of expand legalization. There are two other types of legalization—promotion and custom. A promotion promotes one type to a larger type. A custom legalization involves target-specific hook (maybe a custom operation—majorly seen with IR intrinsic). We leave it to the readers to explore these more in the `CodeGen` phase.

# Optimizing SelectionDAG

After converting the IR into `SelectionDAG`, many opportunities may arise to optimize the DAG itself. These optimization takes place in the `DAGCombiner` phase. These opportunities may arise due to set of architecture specific instructions.

Let's take an example:

[PRE8]

The preceding example in IR looks like the following:

[PRE9]

The example is basically extracting single element from a vector of `<4xi32>` and adding each element of the vector to give a scalar result.

Advanced architectures such as ARM has one single instruction to do the preceding operation—adding across single vector. The SDAG needs to be combined into a single DAG node by identifying the preceding pattern in `SelectionDAG`.

This can be done while selecting instruction in `AArch64DAGToDAGISel`.

[PRE10]

We define the `SelectADDV()` function as follows:

[PRE11]

Note that we have defined a helper function `checkVectorElemAdd()` earlier to check the chain of add selection DAG nodes.

[PRE12]

Let's see how this affects the code generation:

[PRE13]

Before the preceding code, the final code generated will be as follows:

[PRE14]

Clearly, the preceding code is a scalar code. After adding the preceding patch and compiling, the code generated will be as follows:

[PRE15]

# Instruction Selection

The `SelectionDAG` at this phase is optimized and legalized. However, the instructions are still not in machine code form. These instructions need to be mapped to architecture-specific instructions in the `SelectionDAG` itself. The `TableGen` class helps select target-specific instructions.

The `CodeGenAndEmitDAG()` function calls the `DoInstructionSelection()` function that visits each DAG node and calls the Select() function for each node. The `Select()` function is the main hook targets implement to select a node. The `Select()` function is a virtual method to be implemented by the targets.

For consideration, assume our target architecture is X86\. The `X86DAGToDAGISel::Select()` function intercepts some nodes for manual matching, but delegates the bulk of the work to the `X86DAGToDAGISel::SelectCode()` function. The `X86DAGToDAGISel::SelectCode()` function is auto generated by `TableGen`. It contains the matcher table, followed by a call to the generic `SelectionDAGISel::SelectCodeCommon()` function, passing it the table.

[PRE16]

For example, consider the following:

[PRE17]

Before instruction selection, the SDAG looks like the following:

[PRE18]

![Instruction Selection](img/00013.jpeg)

After Instruction Selection, SDAG looks like the following:

[PRE19]

![Instruction Selection](img/00014.jpeg)

# Scheduling and emitting machine instructions

Until now, we have been performing the operations on DAG. Now, for the machine to execute, we need to convert the DAGs into instruction that the machine can execute. One step towards it is emitting the list of instructions into `MachineBasicBlock`. This is done by the `Scheduler`, whose goal is to linearize the DAGs. The scheduling is dependent on the target architecture, as certain Targets will have target specific hooks which can affect the scheduling.

The class `InstrEmitter::EmitMachineNode` takes `SDNode *Node` as one of the input parameters for which it will be emitting machine instructions of the class `MachineInstr`. These instructions are emitted into a `MachineBasicBlock`.

The function calls `EmitSubregNode`, `EmitCopyToRegClassNode` and `EmitRegSequence` for the handling of `subreg` insert/extract, `COPY_TO_REGCLASS`, and `REG_SEQUENCE` respectively.

The call `MachineInstrBuilder` `MIB = BuildMI(*MF, Node->getDebugLoc(), II);` is used to build the Machine Instruction. The `CreateVirtualRegisters` function is called to add result register values created by this instruction.

The `for` loop emits the operands of the instruction :

[PRE20]

It inserts the instruction into its position in the `MachineBasicBlock`.

The following code marks unused registers as dead:

[PRE21]

As we had discussed earlier that the target specific hooks affect the scheduling, the code for that in this function is as follows:

[PRE22]

The `AdjustInstrPostInstrSelection` is a virtual function implemented by Targets.

Let's take an example to see the machine instructions generated in this step. To do this, we need to pass the command-line option `-print-machineinstrs` to the `llc` tool. Let's take the same testcode used earlier:

[PRE23]

Now, invoke the llc command and pass the `–print-machineinstrs` to it. Pass `test.ll` as the input file and store the output in the outfile:

[PRE24]

The `outfile` is large, containing many other phases of code generation apart from scheduling. We need to look into the section after "`# After Instruction Selection:`" in the output file, which is as follows:

[PRE25]

We can see in the output that certain places being taken by physical registers and some by virtual registers. We can also see the machine instruction `IDIV32r` in the output. In the next section, we will see how physical registers are assigned to these virtual registers present in the code.

# Register allocation

The next step of the code generator is register allocation. As we saw in the previous example, some of the registers being used were virtual registers. Register allocation is the task of assigning physical registers to these virtual registers. In LLVM, the virtual registers can be infinite in number, but the numbers of physical registers are limited depending on the target. So, by register allocation, we aim at maximizing the number of physical registers being assigned to virtual registers. We must note that the physical registers are limited in number, so it is not always possible that all the virtual registers can be assigned a physical register. If there is no physical register available at some point and we need a physical register for a variable, we might move a variable that is present in physical register to main memory and thus assign the freed register to the variable we want. This process of moving a variable from physical register to memory is called **spilling**. There are various algorithms to calculate which variable should be spilled from register to memory.

Another important role that the register allocator plays is SSA form deconstruction. The phi instructions present in the machine instruction till now need to be replaced with a regular instruction. The traditional way of doing so is to replace it with a copy instruction.

It must be noted that some of the machine fragments have already registers assigned to them. This is due to target requirements where it wants certain registers fixed to certain operations. Apart from these fixed registers, the register allocator takes care of the rest of the non-fixed registers.

Register allocation for mapping virtual registers to physical registers can be done in the following two ways:

*   **Direct Mapping**: It makes use of the `TargetRegisterInfo` class and the `MachineOperand` class. The developer in this case needs to provide the location where load and store instructions are to be inserted to get values from the memory and store values in the memory.
*   **Indirect Mapping**: In this, the `VirtRegMap` class takes care of inserting loads and stores. It also gets value from memory and stores value to memory. We need to use the `VirtRegMap::assignVirt2Phys(vreg, preg)` function for mapping virtual register to physical register.

LLVM has four register allocation techniques. We will briefly look what they are without going into the details of the algorithm. The four allocators are as follows:

*   **Basic Register Allocator**: The most basic register allocation technique of all the techniques. It can serve as a starter for implementing other register allocation techniques. The algorithm makes use of spill weight for prioritizing the virtual registers. The virtual register with the least weight gets the register allocated to it. When no physical register is available, the virtual register is spilled to memory.
*   **Fast Register Allocator**: This allocation is done at basic block level at a time and attempts to reuse values in registers by keeping them in registers for longer period of time.
*   **PBQP Register Allocator**: As mentioned in the source code file for this register allocation(`llvm/lib/CodeGen/RegAllocPBQP.cpp`), this allocator works by representing the register allocator as a PBQP problem and then solving it using PBQP solver.
*   **Greedy Register Allocator**: This is one of the efficient allocator of LLVM and works across the functions. Its allocation is done using live range splitting and minimizing spill costs.

Let's take an example to see the register allocation for the previous testcode `test.ll` and see how vregs are replaced with actual registers. Let's take the greedy allocator for allocation. You can choose any other allocator as well. The target machine used is x86-64 machine.

[PRE26]

We can see all the vregs present are gone now and have been replaced by actual registers. The machine used here was x86-64\. You can try out register allocation with `pbqp` allocator and see the difference in allocation. The `leal (%rdi,%rsi), %eax` instruction will be replaced with the following instructions:

[PRE27]

# Code Emission

We started from LLVM IR in the first section and converted it to `SelectioDAG` and then to `MachineInstr`. Now, we need to emit this code. Currently, we have LLVM JIT and MC to do so. LLVM JIT is the traditional way of generating the object code for a target on the go directly in the memory. What we are more interested in is the LLVM MC layer.

The MC layer is responsible for generation of assembly file/object file from the `MachineInstr` passed on to it from the previous step. In the MC Layer, the instructions are represented as `MCInst`, which are lightweight, as in they don't carry much information about the program as `MachineInstr`.

The code emission starts with the `AsmPrinter` class, which is overloaded by the target specific `AsmPrinter` class. This class deals with general lowering process by converting the `MachineFunction` functions into MC label constructs by making use of the target specific `MCInstLowering` interface(for x86 it is `X86MCInstLower` class in the `lib/Target/x86/X86MCInstLower.cpp` file).

Now, we have `MCInst` instructions that are passed to `MCStreamer` class for further step of generating either the assembly file or object code. Depending on the choice `MCStreamer` makes use of its subclass `MCAsmStreamer` to generate assembly code and `MCObjectStreamer` to generate the object code.

The target specific `MCInstPrinter` is called by `MCAsmStreamer` to print the assembly instructions. To generate the binary code, the LLVM object code assembler is called by `MCObjectStreamer`. The assembler in turn calls the `MCCodeEmitter::EncodeInstruction()` to generate the binary instructions.

We must note that the MC Layer is one of the big difference between LLVM and GCC. GCC always outputs assembly and then needs an external assembler to transform this assembly into object files, whereas for LLVM using its own assembler we can easily print the instructions in binary and by putting some wraps around them can generate the object file directly. This not only guarantees that the output emitted in text or binary forms will be same but also saves time over GCC by removing the calls to external processes.

Now, let's take an example to look at the MC Instruction corresponding to assembly using the `llc` tool. We make use of the same testcode `test.ll` file used earlier in the chapter.

To view the MC Instructions, we need to pass the command-line option `–asm-show-inst` option to `llc`. It will show the MC instructions as assembly file comments.

[PRE28]

We see the `MCInst` and `MCOperands` in the assembly comments. We can also view the binary encoding in assembly comments by passing the option `–show-mc-encoding` to `llc`.

[PRE29]

# Summary

In this chapter, we saw how LLVM IR is converted to `SelectionDAG`. The SDAG then goes through variety of transformation. The instructions are legalized, so are the data types. `SelectionDAG` also goes through the optimization phase where DAG nodes are combined to result in optimal nodes, which may be target-spacific. After DAG combine, it goes through instruction selection phase, where target architecture instructions are mapped to DAG nodes. After this, the DAGs are ordered in a linear order to facilitate execution by CPU, these DAGs are converted to `MachineInstr` and DAGs are destroyed. Assigning of physical register takes place in the next step to all the virtual registers present in the code. After this, the MC layer comes into picture and deals with the generation of Object and Assembly Code. Going ahead in the next chapter, we will see how to define a target; the various aspects of how a target is represented in LLVM by making use of Table Descriptor files and `TableGen`.