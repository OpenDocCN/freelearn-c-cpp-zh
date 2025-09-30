# Chapter 7. Optimizing the Machine Code

In this chapter, we will cover the following recipes:

*   Eliminating common subexpressions from machine code
*   Analyzing live intervals
*   Allocating registers
*   Inserting the prologue-epilogue code
*   Code emission
*   Tail call optimization
*   Sibling call optimization

# Introduction

The machine code generated so far is yet to be assigned real target architecture registers. The registers seen so far have been virtual registers, which are infinite in number. The machine code generated is in the SSA form. However, the target registers are limited in number. Hence, register allocation algorithms require a lot of heuristic calculations to allocate registers in an optimal way.

But, before register allocation, there exists opportunities for code optimization. The machine code being in the SSA form also makes it easy to apply optimizing algorithms. The algorithms for some optimizing techniques, such as machine dead code elimination and machine common subexpression elimination, are almost the same as in the LLVM IR. The difference lies in the constraints to be checked.

Here, one of the machine code optimization techniques implemented in the LLVM trunk code repository—machine CSE— will be discussed so that you can understand how algorithms are implemented for machine code.

# Eliminating common subexpression from machine code

The aim of the CSE algorithm is to eliminate common subexpressions to make machine code compact and remove unnecessary, duplicate code. Let's look at the code in the LLVM trunk to understand how it is implemented. The detailed code is in the `lib/CodeGen/MachineCSE.cpp` file.

## How to do it…

1.  The `MachineCSE` class runs on a machine function, and hence it should inherit the `MachineFunctionPass` class. It has various members, such as `TargetInstructionInfo`, which is used to get information about the target instruction (used in performing CSE); `TargetRegisterInfo`, which is used to get information about the target register (whether it belongs to a reserved register class, or to more such similar classes; and `MachineDominatorTree`, which is used to get information about the dominator tree for the machine block:

    [PRE0]

2.  The constructor for this class is defined as follows, which initializes the pass:

    [PRE1]

3.  The `getAnalysisUsage()` function determines which passes will run before this pass to get statistics that can be used in this pass:

    [PRE2]

4.  Declare some helper functions in this pass to check for simple copy propagation and trivially dead definitions, check for the liveness of physical registers and their definition uses, and so on:

    [PRE3]

5.  Some more helper functions help to determine the legality and profitability of the expression being a CSE candidate:

    [PRE4]

Let's look at the actual implementation of a CSE function:

1.  The `runOnMachineFunction()` function is called first as the pass runs:

    [PRE5]

2.  The `PerformCSE()` function is called next. It takes the root node of the `DomTree`, performs a DFS walk on the `DomTree` (starting from the root node), and populates a work list consisting of the nodes of the `DomTree`. After the DFS traverses through the `DomTree`, it processes the `MachineBasicBlock` class corresponding to each node in the work list:

    [PRE6]

3.  The next important function is the `ProcessBlock()` function, which acts on the machine basic block. The instructions in the `MachineBasicBlock` class are iterated and checked for legality and profitability if they can be a CSE candidate:

    [PRE7]

4.  Let's also look into the legality and profitability functions to determine the CSE candidates:

    [PRE8]

5.  The profitability function is written as follows:

    [PRE9]

## How it works…

The `MachineCSE` pass runs on a machine function. It gets the `DomTree` information and then traverses the `DomTree` in the DFS way, creating a work list of nodes that are essentially `MachineBasicBlocks`. It then processes each block for CSE. In each block, it iterates through all the instructions and checks whether any instruction is a candidate for CSE. Then it checks whether it is profitable to eliminate the identified expression. Once it has found that the identified CSE is profitable to eliminate, it eliminates the `MachineInstruction` class from the `MachineBasicBlock` class. It also performs a simple copy propagation of the machine instruction. In some cases, the `MachineInstruction` may not be a candidate for CSE in its initial run, but may become one after copy propagation.

## See more

To see more machine code optimization in the SSA form, look into the implementation of the machine dead code elimination pass in the `lib/CodeGen/DeadMachineInstructionElim.cpp file`.

# Analyzing live intervals

Further on in this chapter, we will be looking into register allocation. Before we head to that, however, you must understand the concepts of **live variable** and **live interval**. By live intervals, we mean the range in which a variable is live, that is, from the point where a variable is defined to its last use. For this, we need to calculate the set of registers that are immediately dead after the instruction (the last use of a variable), and the set of registers that are used by the instruction but not after the instruction. We calculate live variable information for each virtual register and physical register in the function. Using SSA to sparsely compute the lifetime information for the virtual registers enables us to only track the physical registers within a block. Before register allocation, LLVM assumes that physical registers are live only within a single basic block. This enables it to perform a single, local analysis to resolve physical register lifetimes within each basic block. After performing the live variable analysis, we have the information required for performing live interval analysis and building live intervals. For this, we start numbering the basic block and machine instructions. After that live-in values, typically arguments in registers are handled. Live intervals for virtual registers are computed for some ordering of the machine instructions (*1*, *N*). A live interval is an interval (*i*, *j*) for which a variable is live, where *1 >= i >= j > N*.

In this recipe, we will take a sample program and see how we can list down the live intervals for that program. We will look at how LLVM works to calculate these intervals.

## Getting ready

To get started, we need a piece of test code on which we will be performing live interval analysis. For simplicity, we will use C code and then convert it into LLVM IR:

1.  Write a test program with an `if` - `else` block:

    [PRE10]

2.  Use Clang to convert the C code into IR, and then view the generated IR using the `cat` command:

    [PRE11]

## How to do it…

1.  To list the live intervals, we will need to modify the code of the `LiveIntervalAnalysis.cpp` file by adding code to print the live intervals. We will add the following lines (marked with a `+` symbol before each added line):

    [PRE12]

2.  Build LLVM after modifying the preceding source file, and install it on the path.
3.  Now compile the test code in the IR form using the `llc` command. You will get the live intervals:

    [PRE13]

## How it works…

In the preceding example, we saw how live intervals are associated with each virtual register. The program points at the beginning and the end of live intervals are marked in square brackets. The process of generating these live intervals starts from the `LiveVariables::runOnMachineFunction(MachineFunction` `&mf)` function in the `lib/CodeGen/LiveVariables.cpp file`, where it assigns the definition and usage of the registers using the `HandleVirtRegUse` and `HandleVirtRegDef` functions. It gets the `VarInfo` object for the given virtual register using the `getVarInfo` function.

The `LiveInterval` and `LiveRange` classes are defined in `LiveInterval.cpp`. The functions in this file takes the information on the liveliness of each variable and then checks whether they overlap or not.

In the `LiveIntervalAnalysis.cpp` file, we have the implementation of the live interval analysis pass, which scans the basic blocks (ordered in a linear fashion) in depth-first order, and creates a live interval for each virtual and physical register. This analysis is used by the register allocators, which will be discussed in next recipe.

## See also

*   If you want to see in detail how the virtual registers for different basic blocks get generated, and see the lifetime of these virtual registers, use the `–debug-only=regalloc` command-line option with the `llc` tool when compiling the test case. You need a debug build of the LLVM for this.
*   To get more detail on live intervals, go through these code files:

    *   `Lib/CodeGen/ LiveInterval.cpp`
    *   `Lib/CodeGen/ LiveIntervalAnalysis.cpp`
    *   `Lib/CodeGen/ LiveVariables.cpp`

# Allocating registers

Register allocation is the task of assigning physical registers to virtual registers. Virtual registers can be infinite, but the physical registers for a machine are limited. So, register allocation is aimed at maximizing the number of physical registers getting assigned to virtual registers. In this recipe, we will see how registers are represented in LLVM, how can we tinker with the register information, the steps taking place, and built-in register allocators.

## Getting ready

You need to build and install LLVM.

## How to do it…

1.  To see how registers are represented in LLVM, open the `build-folder/lib/Target/X86/X86GenRegisterInfo.inc` file and check out the first few lines, which show that registers are represented as integers:

    [PRE14]

2.  For architectures that have registers that share the same physical location, check out the `RegisterInfo.td` file of that architecture for alias information. Let's check out the `lib/Target/X86/X86RegisterInfo.td` file. By looking at the following code snippet, we see how the `EAX`, `AX`, and `AL` registers are aliased (we only specify the smallest register alias):

    [PRE15]

3.  To change the number of physical registers available, go to the `TargetRegisterInfo.td` file and manually comment out some of the registers, which are the last parameters of the `RegisterClass`. Open the `X86RegisterInfo.cpp` file and remove the registers `AH`, `CH`, and `DH`:

    [PRE16]

4.  When you build LLVM, the `.inc` file in the first step will have been changed and will not contain the `AH`, `CH`, and DH registers.
5.  Use the test case from the previous recipe, *Analyzing live intervals*, in which we performed live interval analysis, and run the register allocation techniques provided by LLVM, namely `fast`, `basic`, `greedy`, and `pbqp`. Let's run two of them here and compare the results:

    [PRE17]

    Next, create the `intervalregbasic.s` file as shown:

    [PRE18]

    Next, run the following command to compare the two files:

    [PRE19]

    Create the `intervalregbqp.s` file:

    [PRE20]

6.  Now, use a `diff` tool and compare the two assemblies side by side.

## How it works…

The mapping of virtual registers on physical registers can be done in two ways:

*   **Direct Mapping**: By making use of the `TargetRegisterInfo` and `MachineOperand` classes. This depends on the developer, who needs to provide the location where load and store instructions should be inserted in order to get and store values in the memory.
*   **Indirect Mapping**: This depends on the `VirtRegMap` class to insert loads and stores, and to get and set values from the memory. Use the `VirtRegMap::assignVirt2Phys(vreg, preg)` function to map a virtual register on a physical one.

Another important role that the register allocator plays is in SSA form deconstruction. As traditional instruction sets do not support the `phi` instruction, we must replace it with other instructions to generate the machine code. The traditional way was to replace the `phi` instruction with the `copy` instruction.

After this stage, we do the actual mapping on the physical registers. We have four implementations of register allocation in LLVM, which have their algorithms for mapping the virtual registers on the physical registers. It is not possible to cover in detail any of those algorithms here. If you want to try and understand them, refer to the next section.

## See also

*   To learn more about the algorithms used in LLVM, look through the source codes located at `lib/CodeGen/`:

    *   `lib/CodeGen/RegAllocBasic.cpp`
    *   `lib/CodeGen/ RegAllocFast.cpp`
    *   `lib/CodeGen/ RegAllocGreedy.cpp`
    *   `lib/CodeGen/ RegAllocPBQP.cpp`

# Inserting the prologue-epilogue code

Inserting the prologue-epilogue code involves stack unwinding, finalizing the function layout, saving callee-saved registers and emitting the prologue and epilogue code. It also replaces abstract frame indexes with appropriate references. This pass runs after the register allocation phase.

## How to do it…

The skeleton and the important functions defined in the `PrologueEpilogueInserter` class are as follows:

*   The prologue epilogue inserter pass runs on a machine function, hence it inherits the `MachineFunctionPass` class. Its constructor initializes the pass:

    [PRE21]

*   There are various helper functions defined in this class that help insert the prologue and epilogue code:

    [PRE22]

*   The main function, `insertPrologEpilogCode()`, does the task of inserting the prologue and epilogue code:

    [PRE23]

*   The first function to execute in this pass is the `runOnFunction()` function. The comments in the code show the various operations carried out, such as calculating the call frame size, adjusting the stack variables, inserting the spill code for the callee-saved register for modified registers, calculating the actual frame offset, inserting the prologue and epilogue code for the function, replacing the abstract frame index with the actual offsets, and so on:

    [PRE24]

*   The main function that inserts prologue-epilogue code is the `insertPrologEpilogCode()` function. This function first takes the `TargetFrameLowering` object and then emits a prologue code for that function corresponding to that target. After that, for each basic block in that function, it checks whether there is a return statement. If there is a return statement, then it emits an epilogue code for that function:

    [PRE25]

## How it works…

The preceding code invokes the `emitEpilogue()` and the `emitPrologue()` functions in the `TargetFrameLowering` class, which will be discussed in the target-specific frame lowering recipes in later chapters.

# Code emission

The code emission phase lowers the code from code generator abstractions (such as `MachineFunction` class, `MachineInstr` class, and so on) to machine code layer abstractions (`MCInst` class, `MCStreamer` class, and so on). The important classes in this phase are the target-independent `AsmPrinter` class, target-specific subclasses of `AsmPrinter`, and the `TargetLoweringObjectFile` class.

The MC layer is responsible for emitting object files, which consist of labels, directives, and instructions; while the `CodeGen` layer consists of `MachineFunctions`, `MachineBasicBlock` and `MachineInstructions`. A key class used at this point in time is the `MCStreamer` class, which consists of assembler APIs. The `MCStreamer` class has functions such as `EmitLabel`, `EmitSymbolAttribute`, `SwitchSection`, and so on, which directly correspond to the aforementioned assembly-level directives.

There are four important things that need to be implemented for the target in order to emit code:

*   Define a subclass of the `AsmPrinter` class for the target. This class implements the general lowering process, converting the `MachineFunctions` functions into MC label constructs. The `AsmPrinter` base class methods and routines help implement a target-specific `AsmPrinter` class. The `TargetLoweringObjectFile` class implements much of the common logic for the `ELF`, `COFF`, or `MachO` targets.
*   Implement an instruction printer for the target. The instruction printer takes an `MCInst` class and renders it into a `raw_ostream` class as text. Most of this is automatically generated from the `.td` file (when you specify something like add `$dst`, `$src1`, `$src2` in the instructions), but you need to implement routines to print operands.
*   Implement code that lowers a `MachineInstr` class to an `MCInst` `class`, usually implemented in `<target>MCInstLower.cpp`. This lowering process is often target-specific, and is responsible for turning jump table entries, constant pool indices, global variable addresses, and so on into `MCLabels`, as appropriate. The instruction printer or the encoder takes the `MCInsts` that are generated.
*   Implement a subclass of `MCCodeEmitter` that lowers `MCInsts` to machine code bytes and relocations. This is important if you want to support direct `.o` file emission, or want to implement an assembler for your target.

## How to do it…

Let's visit some important functions in the `AsmPrinter` base class in the `lib/CodeGen/AsmPrinter/AsmPrinter.cpp` file:

*   `EmitLinkage()`: This emits the linkage of the given variables or functions:

    [PRE26]

*   `EmitGlobalVariable()`: This emits the specified global variable to the `.s` file:

    [PRE27]

*   `EmitFunctionHeader()`: This emits the header of the current function:

    [PRE28]

*   `EmitFunctionBody()`: This method emits the body and trailer of a function:

    [PRE29]

*   `EmitJumpTableInfo()`: This prints assembly representations of the jump tables used by the current function to the current output stream:

    [PRE30]

*   `EmitJumpTableEntry()`: This emits a jump table entry for the specified `MachineBasicBlock` class to the current stream:

    [PRE31]

*   Emit integer types of 8, 16, or 32 bit size:

    [PRE32]

For detailed implementation on code emission, see the `lib/CodeGen/AsmPrinter/AsmPrinter.cpp` file. One important thing to note is that this class uses the `OutStreamer` class object to output assembly instructions. The details of target-specific code emission will be covered in later chapters.

# Tail call optimization

In this recipe, we will see how **tail call optimization** is done in LLVM. Tail call optimization is a technique where the callee reuses the stack of the caller instead of adding a new stack frame to the call stack, hence saving stack space and the number of returns when dealing with mutually recursive functions.

## Getting ready

We need to make sure of the following:

*   The `llc` tool must be installed in `$PATH`
*   The `tailcallopt` option must be enabled
*   The test code must have a tail call

## How to do it…

1.  Write the test code for checking tail call optimization:

    [PRE33]

2.  Run the `llc` tool with the `–tailcallopt` option on the test code to generate the assembly file with the tailcall-optimized code:

    [PRE34]

3.  Display the output generated:

    [PRE35]

4.  Using the `llc` tool, generate the assembly again but without using the `-tailcallopt` option:

    [PRE36]

5.  Display the output using the `cat` command:

    [PRE37]

    Compare the two assemblies using a diff tool. We used the meld tool here:

    ![How to do it…](img/image00267.jpeg)

## How it works…

The tail call optimization is a compiler optimization technique, which a compiler can use to make a call to a function and take up no additional stack space; we don't need to create a new stack frame for this function call. This happens if the last instruction executed in a function is a call to another function. A point to note is that the caller function now does not need the stack space; it simply calls a function (another function or itself) and returns whatever value the called function would have returned. This optimization can make recursive calls take up constant and limited space. In this optimization, the code might not always be in the form for which a tail call is possible. It tries and modifies the source to see whether a tail call is possible or not.

In the preceding test case, we see that a push-and-pop instruction is added due to tail call optimization. In LLVM, the tail call optimization is handled by the architecture-specific `ISelLowering.cpp` file; for x86, it is the `X86ISelLowering.cpp` file:

[PRE38]

The preceding code is used to call the `IsEligibleForTailCallOptimization()` function when the `tailcallopt` flag is passed. The `IsEligibleForTailCallOptimization()` function decides whether or not the piece of code is eligible for tail call optimization. If it is, then the code generator will make the necessary changes.

# Sibling call optimisation

In this recipe, we will see how **sibling call optimization** works in LLVM. Sibling call optimization can be looked at as an optimized tail call, the only constraint being that the functions should share a similar function signature, that is, matching return types and matching function arguments.

## Getting ready

Write a test case for sibling call optimization, making sure that the caller and callee have the same calling conventions (in either C or **fastcc**), and that the call in the tail position is a tail call:

[PRE39]

## How to do it…

1.  Run the `llc` tool to generate the assembly:

    [PRE40]

2.  View the generated assembly using the `cat` command:

    [PRE41]

## How it works…

Sibling call optimization is a restricted version of tail call optimization that can be performed on tail calls without passing the `tailcallopt` option. Sibling call optimization works in a similar way to tail call optimization, except that the sibling calls are automatically detected and do not need any ABI changes. The similarity needed in the function signatures is because when the caller function (which calls a tail recursive function) tries to clean up the callee's argument, after the callee has done its work, this may lead to memory leak if the callee exceeds the argument space to perform a sibling call to a function requiring more stack space for arguments.