# Chapter 5. Implementing Optimizations

In this chapter, we will cover the following recipes:

*   Writing a dead code elimination pass
*   Writing an inlining transformation pass
*   Writing a pass for memory optimization
*   Combining LLVM IR
*   Transforming and optimizing loops
*   Reassociating expressions
*   Vectorizing IR
*   Other optimization passes

# Introduction

In the previous chapter, we saw how to write a pass in LLVM. We also demonstrated writing a few analysis passes with an example of alias analysis. Those passes just read the source code and gave us information about it. In this chapter, we will go further and write transformation passes that will actually change the source code, trying to optimize it for the faster execution of code. In the first two recipes, we will show you how a transformation pass is written and how it changes the code. After that, we will see how we can make changes in the code of passes to tinker with the behavior of the passes.

# Writing a dead code elimination pass

In this recipe, you will learn how to eliminate dead code from the program. By dead code elimination, we mean removing the code that has no effect whatsoever on the results that the source program outputs on executing. The main reasons to do so are reduction of the program size, which makes the code quality good and makes it easier to debug the code later on; and improving the run time of the program, as the unnecessary code is prevented from being executed. In this recipe, we will show you a variant of dead code elimination, called aggressive dead code elimination, that assumes every piece of code to be dead until proven otherwise. We will see how to implement this pass ourselves, and what modifications we need to make so that the pass can run just like other passes in the `lib/Transforms/Scalar` folder of the LLVM trunk.

## Getting ready

To show the implementation of dead code elimination, we will need a piece of test code, on which we will run the aggressive dead code elimination pass:

[PRE0]

In this test code, we can see that a call to the `strlen` function is made in the `test` function, but the return value is not used. So, this should be treated as dead code by our pass.

In the file, include the `InitializePasses.h` file, located at `/llvm/`; and in the `llvm` namespace, add an entry for the pass that we are going to write:

[PRE1]

In the `scalar.h` file, located at `include/llvm-c/scalar.h/Transform/`, add the entry for the pass:

[PRE2]

In the `include/llvm/Transform/scalar.h` file, add the entry for the pass in the `llvm` namespace:

[PRE3]

In the `lib/Transforms/Scalar/scalar.cpp` file, add the entry for the pass in two places. In the `void` `llvm::initializeScalarOpts(PassRegistry` `&Registry)` function, add the following code:

[PRE4]

## How to do it…

We will now write the code for the pass:

1.  Include the necessary header files:

    [PRE5]

2.  Declare the structure of our pass:

    [PRE6]

3.  Initialize the pass and its ID:

    [PRE7]

4.  Implement the actual pass in the `runOnFunction` function:

    [PRE8]

5.  Run the preceding pass after compiling the `testcode.ll` file, which can be found in the *Getting ready* section of this recipe:

    [PRE9]

## How it works…

The pass works by first collecting a list of all the root instructions that are live in the first `for` loop of the `runOnFunction` function.

Using this information, we move backwards, propagating liveness to the operands in the `while` `(!Worklist.empty())` loop.

In the next `for` loop, we remove the instructions that are not live, that is, dead. Also, we check whether any reference was made to these values. If so, we drop all such references, which are also dead.

On running the the pass on the test code, we see the dead code; the call to the `strlen` function is removed.

Note that the code has been added to the LLVM trunk revision number 234045\. So, when you are actually trying to implement it, some definitions might be updated. In this case, modify the code accordingly.

## See also

For various other kinds of dead code elimination method, you can refer to the `llvm/lib/Transfroms/Scalar` folder, where the code for other kinds of DCEs is present.

# Writing an inlining transformation pass

As we know, by inlining we mean expanding the function body of the function called at the call site, as it may prove useful through faster execution of code. The compiler takes the decision whether to inline a function or not. In this recipe, you will learn to how to write a simple function-inlining pass that makes use of the implementation in LLVM for inlining. We will write a pass that will handle the functions marked with the `alwaysinline` attribute.

## Getting ready

Let's write a test code that we will run our pass on. Make the necessary changes in the `lib/Transforms/IPO/IPO.cpp` and `include/llvm/InitializePasses.h` files, the `include/llvm/Transforms/IPO.h` file, and the `/include/llvm-c/Transforms/IPO.h` file to include the following pass. Also make the necessary `makefile` changes to include his pass:

[PRE10]

## How to do it…

We will now write the code for the pass:

1.  Include the necessary header files:

    [PRE11]

2.  Describe the class for our pass:

    [PRE12]

3.  Initialize the pass and add the dependencies:

    [PRE13]

4.  Implement the function to get the inlining cost:

    [PRE14]

5.  Write the other helper methods:

    [PRE15]

6.  Compile the pass. After compiling, run it on the preceding test case:

    [PRE16]

## How it works...

This pass that we have written will work for the functions with the `alwaysinline` attribute. The pass will always inline such functions.

The main function at work here is `InlineCost` `getInlineCost(CallSite` `CS)`. This is a function in the `inliner.cpp` file, which needs to be overridden here. So, on the basis of the inlining cost calculated here, we decide whether or not to inline a function. The actual implementation, on how the inlining process works, is in the `inliner.cpp` file.

In this case, we return `InlineCost::getAlways()`; for the functions marked with the `alwaysinline` attribute. For the others, we return `InlineCost::getNever()`. In this way, we can implement inlining for this simple case. If you want to dig deeper and try other variations of inlining—and learn how to make decisions about inlining—you can check out the `inlining.cpp` file.

When this pass is run over the test code, we see that the call of the `inner1` function is replaced by its actual function body.

# Writing a pass for memory optimization

In this recipe, we will briefly discuss a transformation pass that deals with memory optimization.

## Getting ready

For this recipe, you will need the opt tool installed.

## How to do it…

1.  Write the test code on which we will run the `memcpy` optimization pass:

    [PRE17]

2.  Run the `memcpyopt` pass on the preceding test case:

    [PRE18]

## How it works…

The `Memcpyopt` pass deals with eliminating the `memcpy` calls wherever possible, or transforms them into other calls.

Consider this `memcpy` call:

`call void @llvm.memcpy.p0i8.p0i8.i64(i8* %arr_i8, i8* bitcast ([3 x i32]* @cst to i8*), i64 12, i32 4, i1 false)`.

In the preceding test case, this pass converts it into a `memset` call:

`call void @llvm.memset.p0i8.i64(i8* %arr_i8, i8 -1, i64 12, i32 4, i1 false)`

If we look into the source code of the pass, we realize that this transformation is brought about by the `tryMergingIntoMemset` function in the `MemCpyOptimizer.cpp` file in `llvm/lib/Transforms/Scalar`.

The `tryMergingIntoMemset` function looks for some other pattern to fold away when scanning forward over instructions. It looks for stores in the neighboring memory and, on seeing consecutive ones, it attempts to merge them together into `memset`.

The `processMemSet` function looks out for any other neighboring `memset` to this `memset`, which helps us widen out the `memset` call to create a single larger store.

## See also

To see the details of the various types of memory optimization passes, go to [http://llvm.org/docs/Passes.html#memcpyopt-memcpy-optimization](http://llvm.org/docs/Passes.html#memcpyopt-memcpy-optimization).

# Combining LLVM IR

In this recipe, you will learn about instruction combining in LLVM. By instruction combining, we mean replacing a sequence of instructions with more efficient instructions that produce the same result in fewer machine cycles. In this recipe, we will see how we can make modifications in the LLVM code to combine certain instructions.

## Getting started

To test our implementation, we will write test code that we will use to verify that our implementation is working properly to combine instructions:

[PRE19]

## How to do it…

1.  Open the `lib/Transforms/InstCombine/InstCombineAndOrXor.cpp` file.
2.  In the `InstCombiner::visitXor(BinaryOperator` `&I)` function, go to the `if` condition—`if` `(Op0I` `&&` `Op1I)`—and add this:

    [PRE20]

3.  Now build LLVM again so that the Opt tool can use the new functionality and run the test case in this way:

    [PRE21]

## How it works…

In this recipe, we added code to the instruction combining file, which handles transformations involving the AND, OR, and XOR operators.

We added code for matching the pattern of the `(A` `|` `(B` `^` `C))` `^` `((A` `^` `C)` `^` `B)` form, and reduced it to `A` `&` `(B` `^` `C)`. The `if (match(Op0I, m_Or(m_Xor(m_Value(B), m_Value(C)), m_Value(A))) && match(Op1I, m_Xor( m_Xor(m_Specific(A), m_Specific(C)), m_Specific(B))))` line looks out for the pattern similar to the one shown at the start of this paragraph.

The `return` `BinaryOperator::CreateAnd(A,` `Builder->CreateXor(B,C));` line returns the reduced value after building a new instruction, replacing the previous matched code.

When we run the `instcombine` pass over the test code, we get the reduced result. You can see the number of operations is reduced from five to two.

## See also

*   The topic of instruction combining is very wide, and there are loads and loads of possibilities. Similar to the instruction combining function is the instruction simplify function, where we simplify complicated instructions but don't necessarily reduce the number of instructions, as is the case with instruction combining. To look more deeply into this, go through the code in the `lib/Transforms/InstCombine` folder

# Transforming and optimizing loops

In this recipe, we will see how we can transform and optimize loops to get shorter execution times. We will mainly be looking into the **Loop-Invariant Code Motion** (**LICM**) optimization technique, and see how it works and how it transforms the code. We will also look at a relatively simpler technique called **loop deletion**, where we eliminate loops with non-infinite, computable trip counts that have no side effects on a function's return value.

## Getting ready

You must have the opt tool built for this recipe.

## How to do it…

1.  Write the test cases for the LICM pass:

    [PRE22]

2.  Execute the LICM pass on this test code:

    [PRE23]

3.  Write the test code for the loop deletion pass:

    [PRE24]

4.  Finally, run the loop deletion pass over the test code:

    [PRE25]

## How it works…

The LICM pass performs loop-invariant code motion; it tries to move the code that is not modified in the loop out of the loop. It can go either above the loop in the pre-header block, or after the loop exits from the exit block.

In the example shown earlier, we saw the `%i2` `=` `mul` `i32` `%i,` `17` part of the code being moved above the loop, as it is not getting modified within the loop block shown in that example.

The loop deletion pass looks out for loops with non-infinite trip counts that have no effect on the return value of the function.

In the test code, we saw how both the basic blocks `bb:` and `bb2:`, which have the loop part, get deleted. We also saw how the `foo` function directly branches to the return statement.

There are many other techniques for optimizing loops, such as `loop-rotate`, `loop-unswitch`, and `loop-unroll`, which you can try yourself. You will then see how they affect the code.

# Reassociating expressions

In this recipe, you will learn about reassociating expressions and how it helps in optimization.

## Getting Ready

The opt tool should be installed for this recipe to work.

## How to do it…

1.  Write the test case for a simple reassociate transformation:

    [PRE26]

2.  Run the reassociate pass on this test case to see how the code is modified:

    [PRE27]

## How it works …

By reassociation, we mean applying algebraic properties such as associativity, commutativity, and distributivity to rearrange an expression to enable other optimizations, such as constant folding, LICM, and so on.

In the preceding example, we used the inverse property to eliminate patterns such as `"X` `+` `~X"` `->` `"-1"` using reassociation.

The first three lines of the test case give us the expression of the form `(b+(a+1234))+~a`. In this expression, using the reassociate pass, we transform `a+~a` `to` `-1`. Hence, in the result, we get the final return value as `b+1234-1` `=` `b+1233`.

The code that handles this transformation is in the `Reassociate.cpp` file, located under `lib/Transforms/Scalar`.

If you look into this file, specifically the code segment, you can see that it checks whether there are `a` and `~a` in the operand list:

[PRE28]

The following code is responsible for handling and inserting the `-1` value when it gets such values in the expression:

[PRE29]

# Vectorizing IR

**Vectorization** is an important optimization for compilers where we can vectorize code to execute an instruction on multiple datasets in one go. If the backend architecture supports vector registers, a broad range of data can be loaded into those vector registers, and special vector instructions can be executed on the registers.

There are two types of vectorization in LLVM—**Superword-Level Parallelism** (**SLP**) and **loop vectorization**. Loop vectorization deals with vectorization opportunities in a loop, while SLP vectorization deals with vectorizing straight-line code in a basic block. In this recipe, we will see how straight-line code is vectorized.

## Getting ready

SLP vectorization constructs a bottom-up tree of the IR expression, and broadly compares the nodes of the tree to see whether they are similar and hence can be combined to form vectors. The file to be modified is `lib/Transform/Vectorize/SLPVectorizer.cpp`.

We will try to vectorize a piece of straight-line code, such as `return` `a[0]` `+` `a[1]` `+` `a[2]` `+` `a[3]`.

The expression tree for the preceding type of code will be a somewhat one-sided tree. We will run a DFS to store the operands and the operators.

The IR for the preceding kind of expression will look like this:

[PRE30]

The vectorization model follows three steps:

1.  Checking whether it's legal to vectorize.
2.  Calculating the profitability of the vectorized code over the scalarized code.
3.  Vectorizing the code if these two conditions are satisfied.

## How to do it...

1.  Open the `SLPVectorizer.cpp` file. A new function needs to be implemented for DFS traversal of the expression tree for the IR shown in the *Getting ready* section:

    [PRE31]

2.  Calculate the cost of the resultant vectorized IR and conclude whether it is profitable to vectorize. In the `SLPVectorizer.cpp` file, add the following lines to the `getReductionCost()` function:

    [PRE32]

3.  In the same function, after calculating `PairwiseRdxCost` and `SplittingRdxCost`, compare them with `HAddCost`:

    [PRE33]

4.  In the `vectorizeChainsInBlock()` function, call the `matchFlatReduction()` function you just defined:

    [PRE34]

5.  Define two global flags to keep a track of horizontal reduction, which feeds into a return:

    [PRE35]

6.  Allow the vectorization of small trees if they feed into a return. Add the following line to the `isFullyVectorizableTinyTree()` function:

    [PRE36]

## How it works…

Compile the LLVM project after saving the file containing the preceding code, and run the opt tool on the example IR, as follows:

1.  Open the `example.ll` file and paste the following IR in it:

    [PRE37]

2.  Run the opt tool on `example.ll`:

    [PRE38]

    The output will be vectorized code, like the following:

    [PRE39]

As observed, the code gets vectorized. The `matchFlatReduction()` function performs a DFS traversal of the expression and stores all the loads in `ReducedVals`, while adds are stored in `ReductionOps`. After this, the cost of horizontal vectorization is calculated in `HAddCost` and compared with scalar cost. It turns out to be profitable. Hence, it vectorizes the expression. This is handled in the `tryToReduce()` function, which is already implemented.

## See also…

*   For detailed vectorization concepts, refer to the paper *Loop-Aware SLP in GCC* by Ira Rosen, Dorit Nuzman, and Ayal Zaks

# Other optimization passes

In this recipe, we will look at some more transformational passes, which are more like of utility passes. We will look at the `strip-debug-symbols` pass and the `prune-eh` pass.

## Getting ready…

The opt tool must be installed.

## How to do it…

1.  Write a test case for checking the strip-debug pass, which strips off the debug symbols from the test code:

    [PRE40]

2.  Run the `strip-debug-symbols` pass by passing the `–strip-debug` command-line option to the `opt` tool:

    [PRE41]

3.  Write a test case for checking the `prune-eh` pass:

    [PRE42]

4.  Run the pass to remove unused exception information by passing the `–prune-eh` command-line option to the opt tool:

    [PRE43]

## How it works…

In the first case, where we are running the `strip-debug` pass, it removes the debug information from the code, and we can get compact code. This pass must be used only when we are looking for compact code, as it can delete the names of virtual registers and the symbols for internal global variables and functions, thus making the source code less readable and making it difficult to reverse engineer the code.

The part of code that handles this transformation is located in the `llvm/lib/Transforms/IPO/StripSymbols.cpp` file, where the `StripDeadDebugInfo::runOnModule` function is responsible for stripping the debug information.

The second test is for removing unused exception information using the `prune-eh` pass, which implements an interprocedural pass. This walks the call-graph, turning invoke instructions into call instructions only if the callee cannot throw an exception, and marking functions as `nounwind` if they cannot throw the exceptions.

## See also

*   Refer to [http://llvm.org/docs/Passes.html#transform-passes](http://llvm.org/docs/Passes.html#transform-passes) for other transformation passes