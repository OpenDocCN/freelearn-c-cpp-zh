# Chapter 3. Advanced LLVM IR

LLVM provides a powerful intermediate representation for efficient compiler transformations and analysis, while providing a natural means to debug and visualize the transformations. The IR is so designed that it can be easily mapped to high level languages. LLVM IR provides typed information, which can be used for various optimizations.

In the last chapter, you learned how to create some simple LLVM instructions within a function and module. Starting from simple examples such as emitting binary operations, we constructed functions in a module and also created some complex programming paradigms such as if-else and loops. LLVM provides a rich set of instructions and intrinsics to emit a complex IR.

In this chapter, we will go through some more examples of LLVM IR which involve memory operations. Some advanced topics such as aggregate data types and operations on them will also be covered. The topics covered in this chapter are as follows:

*   Getting the address of an element
*   Reading from the memory
*   Writing into a memory location
*   Inserting a scalar into a vector
*   Extracting a scalar from a vector

# Memory access operations

Memory is an important component of almost all computing systems. Memory stores data, which needs to be read to perform operations on the computing system. Results of the operations are stored back in the memory.

The first step is to get the location of the desired element from the memory and store the address in which that particular element can be found. You will now learn how to calculate the address and perform load-store operations.

# Getting the address of an element

In LLVM, the `getelementptr` instruction is used to get the address of an element in an aggregate data structure. It only calculates the address and does not access the memory.

The first argument of the `getelementptr` instruction is a type used as the basis for calculating the address. The second argument is pointer or vector of pointers which act as base of the address - which in our array case will be `a`. The next arguments are the indices of the element to be accessed.

The Language reference ([http://llvm.org/docs/LangRef.html#getelementptr-instruction](http://llvm.org/docs/LangRef.html#getelementptr-instruction)) mentions important notes on `getelementptr` instruction as follows:

> The first index always indexes the pointer value given as the first argument, the second index indexes a value of the type pointed to (not necessarily the value directly pointed to, since the first index can be non-zero), etc. The first type indexed into must be a pointer value, subsequent types can be arrays, vectors, and structs. Note that subsequent types being indexed into can never be pointers, since that would require loading the pointer before continuing calculation.

This essentially implies two important things:

1.  Every pointer has an index, and the first index is always an array index. If it's a pointer to a structure, you have to use index 0 to mean (the first such structure), then the index of the element.
2.  The first type parameter helps GEP identify the sizes of the base structure and its elements, thus easily calculating the address. The resulting type (`%a1`) is not necessarily the same.

More elaborated explanation is provided at [http://llvm.org/docs/GetElementPtr.html](http://llvm.org/docs/GetElementPtr.html)

Let's assume that we have a pointer to a vector of two 32 bit integers `<2 x i32>* %a` and we want to access second integer from the vector. The address will be calculated as

[PRE0]

To emit this instruction, LLVM API can be used as follows:

First create an array type which will be passed as argument to the function.

[PRE1]

The whole code looks like:

[PRE2]

Compile the code:

[PRE3]

Output:

[PRE4]

# Reading from the memory

Now, since we have the address, we are ready to read the data from that address and assign the read value to a variable.

In LLVM the `load` instruction is used to read from a memory location. This simple instruction or combination of similar instructions may then be mapped to some of the sophisticated memory read instructions in low-level assembly.

A `load` instruction takes an argument, which is the memory address from which the data should be read. We obtained the address in the previous section by the `getelementptr` instruction in `a1`.

The `load` instruction looks like the following:

[PRE5]

This means that the `load` will take the data pointed by `a1` and save in `%val`.

To emit this we can use the API provided by LLVM in a function, as shown in the following code:

[PRE6]

Let's also return the loaded value:

[PRE7]

The whole code is as follows:

[PRE8]

Compile the following code:

[PRE9]

The following is the output:

[PRE10]

# Writing into a memory location

LLVM uses the `store` instruction to write into a memory location. There are two arguments to the `store` instruction: a value to store and an address at which to store it. The `store` instruction has no return value. Let's say that we want to write a data to the second element of the vector of two integers. The `store` instruction looks like `store i32 3, i32* %a1`. To emit the `store` instruction, we can use the following API provided by LLVM:

[PRE11]

For example, we will multiply the second element of the `<2 x i32>` vector by `16` and store it back at the same location.

Consider the following code:

[PRE12]

Compile the following code:

[PRE13]

The resulting output will be as follows:

[PRE14]

# Inserting a scalar into a vector

LLVM also provides the API to emit an instruction, which inserts a scalar into a vector type. Note that this vector is different from an array. A vector type is a simple derived type that represents a vector of elements. Vector types are used when multiple primitive data are operated in parallel using **single instruction multiple data** (**SIMD**). A vector type requires a size (number of elements) and an underlying primitive data type. For example, we have a vector `Vec` that has four integers of `i32` type `<4 x i32>`. Now, we want to insert the values 10, 20, 30, and 40 at 0, 1, 2, and 3 indexes of the vector.

The `insertelement` instruction takes three arguments. The first argument is a value of vector type. The second operand is a scalar value whose type must equal the element type of the first operand. The third operand is an index indicating the position at which to insert the value. The resultant value is a vector of the same type.

The `insertelement` instruction looks like the following:

[PRE15]

This can be further understood by keeping the following in mind:

*   `Vec` is of vector type `< 4 x i32 >`
*   `val0` is the value to be inserted
*   `idx` is the index at which the value is to be inserted in the vector

Consider the following code:

[PRE16]

Compile the following code:

[PRE17]

The resulting output is as follows:

[PRE18]

The vector `Vec` will have `<10, 20, 30, 40>` values.

# Extracting a scalar from a vector

An individual scalar element can be extracted from a vector. LLVM provides the `extractelement` instruction for the same. The first operand of an `extractelement` instruction is a value of vector type. The second operand is an index indicating the position from which to extract the element.

The `extractelement` instruction looks like the following:

[PRE19]

This can be further understood by keeping the following in mind:

*   `vec` is a vector
*   `idx` is the index at which the data to be extracted lies
*   `result` is of scalar type, which is `i32` here

Let's take an example where we want to add all the elements of a given vector and return an integer.

Consider the following code:

[PRE20]

Compile the following code:

[PRE21]

Output:

[PRE22]

# Summary

Memory operations form an important instruction for most of the target architecture. Some of the architectures have sophisticated instructions to move data in and out of the memory. Some even perform binary operations directly on the memory operands, while some of them load data from memory into registers and then perform operations on them (CISC vs RISC). Many load-store operations are also done by LLVM instrinsics. For examples, please refer to [http://llvm.org/docs/LangRef.html#masked-vector-load-and-store-intrinsics](http://llvm.org/docs/LangRef.html#masked-vector-load-and-store-intrinsics).

LLVM IR provides a common playfield for all the architectures. It provides elementary instructions for data operations on memory or on aggregate data types. The architectures, while lowering LLVM IR, may combine IR instructions to emit their specific instructions. In this chapter, we went through some advanced IR instructions and also looked into examples of them. For a detailed study, refer to [http://llvm.org/docs/LangRef.html](http://llvm.org/docs/LangRef.html), which provides the authoritative resource for LLVM IR instructions.

In the next chapter, you will study how LLVM IR can be optimized to reduce instructions and emit a clean code.