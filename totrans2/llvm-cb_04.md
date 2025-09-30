# Chapter 4. Preparing Optimizations

In this chapter, we will cover the following recipes:

*   Various levels of optimization
*   Writing your own LLVM pass
*   Running your own pass with the opt tool
*   Using another pass in a new pass
*   Registering a pass with pass manager
*   Writing an analysis pass
*   Writing an alias analysis pass
*   Using other analysis passes

# Introduction

Once the source code transformation completes, the output is in the LLVM IR form. This IR serves as a common platform for converting into assembly code, depending on the backend. However, before converting into an assembly code, the IR can be optimized to produce more effective code. The IR is in the SSA form, where every new assignment to a variable is a new variable itself—a classic case of an SSA representation.

In the LLVM infrastructure, a pass serves the purpose of optimizing LLVM IR. A pass runs over the LLVM IR, processes the IR, analyzes it, identifies the optimization opportunities, and modifies the IR to produce optimized code. The command-line interface **opt** is used to run optimization passes on LLVM IR.

In the upcoming chapters, various optimization techniques will be discussed. Also, how to write and register a new optimization pass will be explored.

# Various levels of optimization

There are various levels of optimization, starting at 0 and going up to 3 (there is also `s` for space optimization). The code gets more and more optimized as the optimization level increases. Let's try to explore the various optimization levels.

## Getting ready...

Various optimization levels can be understood by running the opt command-line interface on LLVM IR. For this, an example C program can first be converted to IR using the **Clang** frontend.

1.  Open an `example.c` file and write the following code in it:

    [PRE0]

2.  Now convert this into LLVM IR using the `clang` command, as shown here:

    [PRE1]

    A new file, `example.ll`, will be generated, containing LLVM IR. This file will be used to demonstrate the various optimization levels available.

## How to do it…

Do the following steps:

1.  The opt command-line tool can be run on the IR-generated `example.ll` file:

    [PRE2]

    The `–O0` syntax specifies the least optimization level.

2.  Similarly, you can run other optimization levels:

    [PRE3]

## How it works…

The opt command-line interface takes the `example.ll` file as the input and runs the series of passes specified in each optimization level. It can repeat some passes in the same optimization level. To see which passes are being used in each optimization level, you have to add the `--debug-pass=Structure` command-line option with the previous opt commands.

## See Also

*   To know more on various other options that can be used with the opt tool, refer to [http://llvm.org/docs/CommandGuide/opt.html](http://llvm.org/docs/CommandGuide/opt.html)

# Writing your own LLVM pass

All LLVM passes are subclasses of the `pass` class, and they implement functionality by overriding the virtual methods inherited from `pass`. LLVM applies a chain of analyses and transformations on the target program. A pass is an instance of the Pass LLVM class.

## Getting ready

Let's see how to write a pass. Let's name the pass `function block counter`; once done, it will simply display the name of the function and count the basic blocks in that function when run. First, a `Makefile` needs to be written for the pass. Follow the given steps to write a `Makefile`:

1.  Open a `Makefile` in the `llvm lib/Transform` folder:

    [PRE4]

2.  Specify the path to the LLVM root folder and the library name, and make this pass a loadable module by specifying it in `Makefile`, as follows:

    [PRE5]

This `Makefile` specifies that all the `.cpp` files in the current directory are to be compiled and linked together in a shared object.

## How to do it…

Do the following steps:

1.  Create a new `.cpp` file called `FuncBlockCount.cpp`:

    [PRE6]

2.  In this file, include some header files from LLVM:

    [PRE7]

3.  Include the `llvm` namespace to enable access to LLVM functions:

    [PRE8]

4.  Then start with an anonymous namespace:

    [PRE9]

5.  Next declare the pass:

    [PRE10]

6.  Then declare the pass identifier, which will be used by LLVM to identify the pass:

    [PRE11]

7.  This step is one of the most important steps in writing a pass—writing a `run` function. Since this pass inherits `FunctionPass` and runs on a function, a `runOnFunction` is defined to be run on a function:

    [PRE12]

    This function prints the name of the function that is being processed.

8.  The next step is to initialize the pass ID:

    [PRE13]

9.  Finally, the pass needs to be registered, with a command-line argument and a name:

    [PRE14]

    Putting everything together, the entire code looks like this:

    [PRE15]

## How it works

A simple `gmake` command compiles the file, so a new file `FuncBlockCount.so` is generated at the LLVM root directory. This shared object file can be dynamically loaded to the opt tool to run it on a piece of LLVM IR code. How to load and run it will be demonstrated in the next section.

## See also

*   To know more on how a pass can be built from scratch, visit [http://llvm.org/docs/WritingAnLLVMPass.html](http://llvm.org/docs/WritingAnLLVMPass.html)

# Running your own pass with the opt tool

The pass written in the previous recipe, *Writing your own LLVM pass*, is ready to be run on the LLVM IR. This pass needs to be loaded dynamically for the opt tool to recognize and execute it.

## How to do it…

Do the following steps:

1.  Write the C test code in the `sample.c` file, which we will convert into an `.ll` file in the next step:

    [PRE16]

2.  Convert the C test code into LLVM IR using the following command:

    [PRE17]

    This will generate a `sample.ll` file.

3.  Run the new pass with the opt tool, as follows:

    [PRE18]

    The output will look something like this:

    [PRE19]

## How it works…

As seen in the preceding code, the shared object loads dynamically into the opt command-line tool and runs the pass. It goes over the function and displays its name. It does not modify the IR. Further enhancement in the new pass is demonstrated in the next recipe.

## See also

*   To know more about the various types of the Pass class, visit [http://llvm.org/docs/WritingAnLLVMPass.html#pass-classes-and-requirements](http://llvm.org/docs/WritingAnLLVMPass.html#pass-classes-and-requirements)

# Using another pass in a new pass

A pass may require another pass to get some analysis data, heuristics, or any such information to decide on a further course of action. The pass may just require some analysis such as memory dependencies, or it may require the altered IR as well. The new pass that you just saw simply prints the name of the function. Let's see how to enhance it to count the basic blocks in a loop, which also demonstrates how to use other pass results.

## Getting ready

The code used in the previous recipe remains the same. Some modifications are required, however, to enhance it—as demonstrated in next section—so that it counts the number of basic blocks in the IR.

## How to do it…

The `getAnalysis` function is used to specify which other pass will be used:

1.  Since the new pass will be counting the number of basic blocks, it requires loop information. This is specified using the `getAnalysis` loop function:

    [PRE20]

2.  This will call the `LoopInfo` pass to get information on the loop. Iterating through this object gives the basic block information:

    [PRE21]

3.  This will go over the loop to count the basic blocks inside it. However, it counts only the basic blocks in the outermost loop. To get information on the innermost loop, recursive calling of the `getSubLoops` function will help. Putting the logic in a separate function and calling it recursively makes more sense:

    [PRE22]

## How it works…

The newly modified pass now needs to run on a sample program. Follow the given steps to modify and run the sample program:

1.  Open the `sample.c` file and replace its content with the following program:

    [PRE23]

2.  Convert it into a `.ll` file using Clang:

    [PRE24]

3.  Run the new pass on the previous sample program:

    [PRE25]

    The output will look something like this:

    [PRE26]

## There's more…

The LLVM's pass manager provides a debug pass option that gives us the chance to see which passes interact with our analyses and optimizations, as follows:

[PRE27]

# Registering a pass with pass manager

Until now, a new pass was a dynamic object that was run independently. The opt tool consists of a pipeline of such passes that are registered with the pass manager, and a part of LLVM. Let's see how to register our pass with the Pass Manager.

## Getting ready

The `PassManager` class takes a list of passes, ensures that their prerequisites are set up correctly, and then schedules the passes to run efficiently. The Pass Manager does two main tasks to try to reduce the execution time of a series of passes:

*   Shares the analysis results to avoid recomputing analysis results as much as possible
*   Pipelines the execution of passes to the program to get better cache and memory usage behavior out of a series of passes by pipelining the passes together

## How to do it…

Follow the given steps to register a pass with Pass Manager:

1.  Define a `DEBUG_TYPE` macro, specifying the debugging name in the `FuncBlockCount.cpp` file:

    [PRE28]

2.  In the `FuncBlockCount` struct, specify the `getAnalysisUsage` syntax as follows:

    [PRE29]

3.  Now initialize the macros for initialization of the new pass:

    [PRE30]

4.  Add the `createFuncBlockCount` Pass function in the `LinkAllPasses.h` file, located at `include/llvm/`:

    [PRE31]

5.  Add the declaration to the `Scalar.h` file, located at `include/llvm/Transforms`:

    [PRE32]

6.  Also modify the constructor of the pass:

    [PRE33]

7.  In the `Scalar.cpp file`, located at `lib/Transforms/Scalar/`, add the initialization pass entry:

    [PRE34]

8.  Add this initialization declaration to the `InitializePasses.h` file, which is located at `include/llvm/`:

    [PRE35]

9.  Finally, add the `FuncBlockCount.cpp` filename to the `CMakeLists.txt` file, located at `lib/Transforms/Scalar/`:

    [PRE36]

## How it works…

Compile the LLVM with the `cmake` command as specified in [Chapter 1](part0015.xhtml#aid-E9OE1 "Chapter 1. LLVM Design and Use"), *LLVM Design and Use*. The Pass Manager will include this pass in the pass pipeline of the opt command-line tool. Also, this pass can be run in isolation from the command line:

[PRE37]

## See Also

*   To know more about adding a pass in Pass Manager in simple steps, study the LoopInstSimplify pass at [http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Transforms/Scalar/LoopInstSimplify.cpp](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Transforms/Scalar/LoopInstSimplify.cpp)

# Writing an analysis pass

The analysis pass provides higher-level information about IR without actually changing the IR. The results that the analysis pass provides can be used by another analysis pass to compute its result. Also, once an analysis pass calculates the result, its result can be used several times by different passes until the IR on which this pass was run is changed. In this recipe, we will write an analysis pass that counts and outputs the number of opcodes used in a function.

## Getting ready

First of all, we write the test code on which we will be running our pass:

[PRE38]

Transform this into a `.bc` file, which we will use as the input to the analysis pass:

[PRE39]

Now create the file containing the pass source code in `llvm_root_dir/lib/Transforms/opcodeCounter`. Here, `opcodeCounter` is the directory we have created, and it is where our pass's source code will reside.

Make the necessary `Makefile` changes so that this pass can be compiled.

## How to do it…

Now let's start writing the source code for our analysis pass:

1.  Include the necessary header files and use the `llvm` namespace:

    [PRE40]

2.  Create the structure defining the pass:

    [PRE41]

3.  Within the structure, create the necessary data structures to count the number of opcodes and to denote the pass ID of the pass:

    [PRE42]

4.  Within the preceding structure, write the code for the actual implementation of the pass, overloading the `runOnFunction` function:

    [PRE43]

5.  Write the code for registering the pass:

    [PRE44]

6.  Compile the pass using the `make` or `cmake` command.
7.  Run the pass on the test code using the opt tool to get the information on the number of opcodes present in the function:

    [PRE45]

## How it works…

This analysis pass works on a function level, running once for each function in the program. Hence, we have inherited the `FunctionPass` function when declaring the `CountOpcodes : public FunctionPass` struct.

The `opcodeCounter` function keeps a count of every opcode that has been used in the function. In the following for loops, we collect the opcodes from all the functions:

[PRE46]

The first `for` loop iterates over all the basic blocks present in the function, and the second for loop iterates over all the instructions present in the basic block.

The code in the first `for` loop is the actual code that collects the opcodes and their numbers. The code below the `for` loops is meant for printing the results. As we have used a map to store the result, we iterate over it to print the pair of the opcode name and its number in the function.

We return `false` because we are not modifying anything in the test code. The last two lines of the code are meant for registering this pass with the given name so that the opt tool can use this pass.

Finally, on execution of the test code, we get the output as different opcodes used in the function and their numbers.

# Writing an alias analysis pass

Alias analysis is a technique by which we get to know whether two pointers point to the same location—that is, whether the same location can be accessed in more ways than one. By getting the results of this analysis, you can decide about further optimizations, such as common subexpression elimination. There are different ways and algorithms to perform alias analysis. In this recipe, we will not deal with these algorithms, but we will see how LLVM provides the infrastructure to write your own alias analysis pass. In this recipe, we will write an alias analysis pass to see how to get started with writing such a pass. We will not make use of any specific algorithm, but will return the `MustAlias` response in every case of the analysis.

## Getting ready

Write the test code that will be the input for alias analysis. Here, we will take the `testcode.c` file used in the previous recipe as the test code.

Make the necessary `Makefile` changes, make changes to register the pass by adding entries for the pass in `llvm/lib/Analysis/Analysis.cpp llvm/include/llvm/InitializePasses.h`, `llvm/include/llvm/LinkAllPasses.h`, `llvm/include/llvm/Analysis/Passes.h` and create a file in `llvm_source_dir/lib/Analysis/ named EverythingMustAlias.cpp` that will contain the source code for our pass.

## How to do it...

Do the following steps:

1.  Include the necessary header files and use the `llvm` namespace:

    [PRE47]

2.  Create a structure for our pass by inheriting the `ImmutablePass` and `AliasAnalysis` classes:

    [PRE48]

3.  Declare the data structures and constructor:

    [PRE49]

4.  Implement the `getAdjustedAnalysisPointer` function:

    [PRE50]

5.  Implement the `initializePass` function to initialize the pass:

    [PRE51]

6.  Implement the `alias` function:

    [PRE52]

7.  Register the pass:

    [PRE53]

8.  Compile the pass using the `cmake` or `make` command.
9.  Execute the test code using the `.so` file that is formed after compiling the pass:

    [PRE54]

## How it works…

The `AliasAnalysis` class gives the interface that the various alias analysis implementations should support. It exports the `AliasResult` and `ModRefResult` enums, representing the results of the `alias` and `modref` query respectively.

The `alias` method is used to check whether two memory objects are pointing to the same location or not. It takes two memory objects as the input and returns `MustAlias`, `PartialAlias`, `MayAlias`, or `NoAlias` as appropriate.

The `getModRefInfo` method returns the information on whether the execution of an instruction can read or modify a memory location. The pass in the preceding example works by returning the value `MustAlias` for every set of two pointers, as we have implemented it that way. Here, we have inherited the `ImmutablePasses` class, which suits our pass, as it is a very basic pass. We have inherited the `AliasAnalysis` pass, which provides the interface for our implementation.

The `getAdjustedAnalysisPointer` function is used when a pass implements an analysis interface through multiple inheritance. If needed, it should override this to adjust the pointer as required for the specified pass information.

The `initializePass` function is used to initialize the pass that contains the `InitializeAliasAnalysis` method, which should contain the actual implementation of the alias analysis.

The `getAnalysisUsage` method is used to declare any dependency on other passes by explicitly calling the `AliasAnalysis::getAnalysisUsage` method.

The `alias` method is used to determine whether two memory objects alias each other or not. It takes two memory objects as the input and returns the `MustAlias`, `PartialAlias`, `MayAlias`, or `NoAlias` responses as appropriate.

The code following the `alias` method is meant for registering the pass. Finally, when we use this pass over the test code, we get 10 `MustAlias` responses (`100.0%`) as the result, as implemented in our pass.

## See also

For a more detailed insight into LLVM alias analysis, refer to [http://llvm.org/docs/AliasAnalysis.html](http://llvm.org/docs/AliasAnalysis.html).

# Using other analysis passes

In this recipe, we will take a brief look into the other analysis passes that are provided by LLVM and can be used to get analysis information about a basic block, function, module, and so on. We will look into passes that have already been implemented in LLVM, and how we can use them for our purpose. We will not go through all the passes but take a look at only some of them.

## Getting ready…

Write the test code in the `testcode1.c` file, which will be used for analysis purposes:

[PRE55]

Convert the C code to bitcode format, using the following command line:

[PRE56]

## How to do it…

Follow the steps given to use other analysis passes:

1.  Use the alias analysis evaluator pass by passing `–aa-eval` as a command-line option to the opt tool:

    [PRE57]

2.  Print the dominator tree information using the `–print-dom-info` command-line option along with opt:

    [PRE58]

3.  Count the number of queries made by one pass to another using the `–count-aa` command-line option along with opt:

    [PRE59]

4.  Print the alias sets in a program using the `-print-alias-sets` command-line option with opt:

    [PRE60]

## How it works…

In the first case, where we use the `-aa-eval` option, the opt tool runs the alias analysis evaluator pass, which outputs the analysis on the screen. It iterates through all pairs of pointers in the function and queries whether the two are aliases of each other or not.

Using the `-print-dom-info` option, the pass for printing the dominator tree is run, through which information about the dominator tree can be obtained.

In the third case, we execute the `opt -count-aa -basicaa –licm` command. The `count-aa` command option counts the number of queries made by the `licm` pass to the `basicaa` pass. This information is obtained by the count alias analysis pass using the opt tool.

To print all the alias sets within a program, we use the `- print-alias-sets` command-line option. In this case, it prints the alias sets obtained after analyzing with the `basicaa` pass.

## See also

Refer to [http://llvm.org/docs/Passes.html#anal](http://llvm.org/docs/Passes.html#anal) to know about more passes not mentioned here.