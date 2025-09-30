# Chapter 3. Extending the Frontend and Adding JIT Support

In this chapter, we will cover the following recipes:

*   Handling decision making paradigms – if/then/else constructs
*   Generating code for loops
*   Handling user-defined operators – binary operators
*   Handling user-defined operators – unary operators
*   Adding JIT support

# Introduction

In the last chapter, the basics of the frontend component for a language were defined. This included defining tokens for different types of expressions, writing a lexer to tokenize a stream of input, chalking out a skeleton for the abstract syntax tree of various expressions, writing a parser, and generating code for the language. Also, how various optimizations can be hooked to the frontend was explained.

A language is more powerful and expressive when it has control flow and loops to decide the flow of a program. JIT support explores the possibility of compiling code on-the-fly. In this chapter, implementation of these more sophisticated programming paradigms will be discussed. This chapter deals with enhancements of a programming language that make it more meaningful and powerful to use. The recipes in this chapter demonstrate how to include those enhancements for a given language.

# Handling decision making paradigms – if/then/else constructs

In any programming language, executing a statement based on certain conditions gives a very powerful advantage to the language. The `if`/`then`/`else` constructs provide the capability to alter the control flow of a program, based on certain conditions. The condition is present in an `if` construct. If the condition is true, the expression following the `then` construct is executed. If it is `false`, the expression following the `else` construct is executed. This recipe demonstrates a basic infrastructure to parse and generate code for the `if`/`then`/`else` construct.

## Getting ready

The TOY language for `if`/`then`/`else` can be defined as:

[PRE0]

For checking a condition, a comparison operator is required. A simple *less than* operator, `<`, will serve the purpose. To handle `<`, precedence needs to be defined in the `init_precedence()` function, as shown here:

[PRE1]

Also, the `codegen()` function for binary expressions needs to be included for `<`:

[PRE2]

Now, the LLVM IR will generate a comparison instruction and a Boolean instruction as a result of the comparison, which will be used to determine where the control of the program will flow. It's time to handle the `if`/`then`/`else` paradigm now.

## How to do it...

Do the following steps:

1.  The lexer in the `toy.cpp` file has to be extended to handle the `if`/`then`/`else` constructs. This can be done by appending a token for this in the `enum` of tokens:

    [PRE3]

2.  The next step is to append the entries for these tokens in the `get_token()` function, where we match strings and return the appropriate tokens:

    [PRE4]

3.  Then we define an AST node in the `toy.cpp` file:

    [PRE5]

4.  The next step is to define the parsing logic for the `if`/`then`/`else` constructs:

    [PRE6]

    The parser logic is simple: first, the `if` token is searched for and the expression following it is parsed for the condition. After that, the `then` token is identified and the true condition expression is parsed. Then the `else` token is searched for and the false condition expression is parsed.

5.  Next we hook up the previously defined function with `Base_Parser()`:

    [PRE7]

6.  Now that the AST of `if`/`then`/`else` is filled with the expression by the parser, it's time to generate the LLVM IR for the conditional paradigm. Let's define the `Codegen()` function:

    [PRE8]

Now that we are ready with the code, let's compile and run it on a sample program containing the `if`/`then`/`else` constructs.

## How it works…

Do the following steps:

1.  Compile the `toy.cpp` file:

    [PRE9]

2.  Open an example file:

    [PRE10]

3.  Write the following `if`/`then`/`else` code in the example file:

    [PRE11]

4.  Compile the example file with the TOY compiler:

    [PRE12]

The LLVM IR generated for the `if`/`then`/`else` code will look like this:

[PRE13]

Here's what the output looks like:

![How it works…](img/image00254.jpeg)

The parser identifies the `if`/`then`/`else` constructs and the statements that are to be executed in true and false conditions, and stores them in the AST. The code generator then converts the AST into LLVM IR, where the condition statement is generated. IR is generated for true as well as false conditions. Depending on the state of the condition variable, the appropriate statement is executed at runtime.

## See also

*   For a detailed example on how an `if else` statement is handled in C++ by Clang, refer to [http://clang.llvm.org/doxygen/classclang_1_1IfStmt.html](http://clang.llvm.org/doxygen/classclang_1_1IfStmt.html).

# Generating code for loops

Loops make a language powerful enough to perform the same operation several times, with limited lines of code. Loops are present in almost every language. This recipe demonstrates how loops are handled in the TOY language.

## Getting ready

A loop typically has a start that initializes the induction variable, a step that indicates an increment or decrement in the induction variable, and an end condition for termination of the loop. The loop in our TOY language can be defined as follows:

[PRE14]

The start expression is the initialization of `i = 1`. The end condition for the loop is `i<n`. The first line of the code indicates `i` be incremented by `1`.

As long as the end condition is true, the loop iterates and, after each iteration, the induction variable, `i`, is incremented by 1\. An interesting thing called **PHI** node comes into the picture to decide which value the induction variable, `i`, will take. Remember that our IR is in the **single static assignment** (**SSA**) form. In a control flow graph, for a given variable, the values can come from two different blocks. To represent SSA in LLVM IR, the `phi` instruction is defined. Here is an example of `phi`:

[PRE15]

The preceding IR indicates that the value for `i` can come from two basic blocks: `%entry` and `%loop`. The value from the `%entry` block will be `1`, while the `%nextvar` variable will be from `%loop`. We will see the details after implementing the loop for our toy compiler.

## How to do it...

Like any other expression, loops are also handled by including states in lexer, defining the AST data structure to hold loop values, and defining the parser and the `Codegen()` function to generate the LLVM IR:

1.  The first step is to define tokens in the lexer in `toy.cpp` file:

    [PRE16]

2.  Then we include the logic in the lexer:

    [PRE17]

3.  The next step is to define the AST for the `for` loop:

    [PRE18]

4.  Then we define the parser logic for the loop:

    [PRE19]

5.  Next we define the `Codegen()` function to generate the LLVM IR:

    [PRE20]

## How it works...

Do the following steps:

1.  Compile the `toy.cpp` file:

    [PRE21]

2.  Open an example file:

    [PRE22]

3.  Write the following code for a `for` loop in the example file:

    [PRE23]

4.  Compile the example file with the TOY compiler:

    [PRE24]

5.  The LLVM IR for the preceding `for` loop code will be generated, as follows:

    [PRE25]

The parser you just saw identifies the loop, initialization of the induction variable, the termination condition, the step value for the induction variable, and the body of the loop. It then converts each of the blocks in LLVM IR, as seen previously.

As seen previously, a `phi` instruction gets two values for the variable `i` from two basic blocks: `%entry` and `%loop`. In the preceding case, the `%entry` block represents the value assigned to the induction variable at the start of the loop (this is `1`). The next updated value of `i` comes from the `%loop` block, which completes one iteration of the loop.

## See also

*   To get a detailed overview of how loops are handled for C++ in Clang, visit [http://llvm.org/viewvc/llvm-project/cfe/trunk/lib/Parse/ParseExprCXX.cpp](http://llvm.org/viewvc/llvm-project/cfe/trunk/lib/Parse/ParseExprCXX.cpp)

# Handling user-defined operators – binary operators

User-defined operators are similar to the C++ concept of operator overloading, where a default definition of an operator is altered to operate on a wide variety of objects. Typically, operators are unary or binary operators. Implementing binary operator overloading is easier with the existing infrastructure. Unary operators need some additional code to handle. First, binary operator overloading will be defined, and then unary operator overloading will be looked into.

## Getting ready

The first part is to define a binary operator for overloading. The logical OR operator (`|`) is a good example to start with. The `|` operator in our TOY language can be used as follows:

[PRE26]

As seen in the preceding code, if any of the values of the LHS or RHS are not equal to 0, then we return `1`. If both the LHS and RHS are null, then we return `0`.

## How to do it...

Do the following steps:

1.  The first step, as usual, is to append the `enum` states for the binary operator and return the enum states on encountering the `binary` keyword:

    [PRE27]

2.  The next step is to add an AST for the same. Note that it doesn't need a new AST to be defined. It can be handled with the function declaration AST. We just need to modify it by adding a flag to represent whether it's a binary operator. If it is, then determine its precedence:

    [PRE28]

3.  Once the modified AST is ready, the next step is to modify the parser of the function declaration:

    [PRE29]

4.  Then we modify the `Codegen()` function for the binary AST:

    [PRE30]

5.  Next we modify the function definition; it can be defined as:

    [PRE31]

## How it works...

Do the following steps:

1.  Compile the `toy.cpp` file:

    [PRE32]

2.  Open an example file:

    [PRE33]

3.  Write the following binary operator overloading code in the example file:

    [PRE34]

4.  Compile the example file with the TOY compiler:

    [PRE35]

The binary operator we just defined will be parsed. Its definition is also parsed. Whenever the `|` binary operator is encountered, the LHS and RHS are initialized and the definition body is executed, giving the appropriate result as per the definition. In the preceding example, if either the LHS or RHS is nonzero, then the result is `1`. If both the LHS and RHS are zero, then the result is `0`.

## See also

*   For detailed examples on handling other binary operators, refer to [http://llvm.org/docs/tutorial/LangImpl6.html](http://llvm.org/docs/tutorial/LangImpl6.html)

# Handling user-defined operators – unary operators

We saw in the previous recipe how binary operators can be handled. A language may also have some unary operator, operating on 1 operand. In this recipe, we will see how to handle unary operators.

## Getting ready

The first step is to define a unary operator in the TOY language. A simple unary NOT operator (`!`) can serve as a good example; let's see one definition:

[PRE36]

If the value `v` is equal to `1`, then `0` is returned. If the value is `0`, `1` is returned as the output.

## How to do it...

Do the following steps:

1.  The first step is to define the `enum` token for the unary operator in the `toy.cpp` file:

    [PRE37]

2.  Then we identify the unary string and return a unary token:

    [PRE38]

3.  Next, we define an AST for the unary operator:

    [PRE39]

4.  The AST is now ready. Let's define a parser for the unary operator:

    [PRE40]

5.  The next step is to call the `unary_parser()` function from the binary operator parser:

    [PRE41]

6.  Now let's call the `unary_parser()` function from the expression parser:

    [PRE42]

7.  Then we modify the function declaration parser:

    [PRE43]

8.  The final step is to define the `Codegen()` function for the unary operator:

    [PRE44]

## How it works...

Do the following steps:

1.  Compile the `toy.cpp` file:

    [PRE45]

2.  Open an example file:

    [PRE46]

3.  Write the following unary operator overloading code in the example file:

    [PRE47]

4.  Compile the example file with the TOY compiler:

    [PRE48]

    The output should be as shown:

    [PRE49]

The unary operator defined by the user will be parsed, and IR will be generated for it. In the case you just saw, if the unary operand is not zero then the result is `0`. If the operand is zero, then the result is `1`.

## See also

*   To learn more detailed implementations of unary operators, visit [http://llvm.org/docs/tutorial/LangImpl6.html](http://llvm.org/docs/tutorial/LangImpl6.html)

# Adding JIT support

A wide variety of tools can be applied to LLVM IR. For example, as demonstrated in [Chapter 1](part0015.xhtml#aid-E9OE1 "Chapter 1. LLVM Design and Use"), *LLVM Design and Use*, the IR can be dumped into bitcode or into an assembly. An optimization tool called opt can be run on IR. IR acts as the common platform—an abstract layer for all of these tools.

JIT support can also be added. It immediately evaluates the top-level expressions typed in. For example, `1 + 2;`, as soon as it is typed in, evaluates the code and prints the value as `3`.

## How to do it...

Do the following steps:

1.  Define a static global variable for the execution engine in the `toy.cpp` file:

    [PRE50]

2.  In the `toy.cpp` file's `main()` function, write the code for JIT:

    [PRE51]

3.  Modify the top-level expression parser in the `toy.cpp` file:

    [PRE52]

## How it works…

Do the following steps:

1.  Compile the `toy.cpp` program:

    [PRE53]

2.  Open an example file:

    [PRE54]

3.  Write the following TOY code in the example file:

    [PRE55]

4.  Finally, run the TOY compiler on the example file:

    [PRE56]

The LLVM JIT compiler matches the native platform ABI, casts the result pointer into a function pointer of that type, and calls it directly. There is no difference between JIT-compiled code and native machine code that is statically linked to the application.