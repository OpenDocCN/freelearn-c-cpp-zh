# Chapter 7. Dynamic Memory Allocation

In the previous chapter, we talked about class definitions and how to devise your own custom class. We discussed how by devising our own custom classes, we can construct variables that represented entities within your game or program.

In this chapter, we will talk about dynamic memory allocations and how to create space in memory for groups of objects.

Assume that we have a simplified version of `class Player`, as before, with only a constructor and a destructor:

[PRE0]

We talked earlier about the *scope* of a variable in C++; to recap, the scope of a variable is the section of the program where that variable can be used. The scope of a variable is generally inside the block in which it was declared. A block is just any section of code contained between { and }. Here is a sample program that illustrates variable scope:

![Dynamic Memory Allocation](img/00071.jpeg)

In this sample program, the x variable has scope through all of main(). The y variable's scope is only inside the if block

We mentioned previously that in general variables are destroyed when they go out of scope. Let's test this idea out with instances of `class Player`:

[PRE1]

The output of this program is as follows:

[PRE2]

The destructor for the player object is called at the end of the player object's scope. Since the scope of a variable is the block within which it is defined in the three lines of code, the `Player` object would be destroyed immediately at the end of `main()`, when it goes out of scope.

# Dynamic memory allocation

Now, let's try allocating a `Player` object dynamically. What does that mean?

We use the `new` keyword to allocate it!

[PRE3]

The output of this program is as follows:

[PRE4]

The player does not die! How do we kill the player? We must explicitly call `delete` on the `player` pointer.

## The delete keyword

The `delete` operator invokes the destructor on the object being deleted, as shown in the following code:

[PRE5]

The output of the program is as follows:

[PRE6]

So, only "normal" (or "automatic" also called as non-pointer type) variable types get destroyed at the end of the block in which they were declared. Pointer types (variables declared with `*` and `new`) are not automatically destroyed even when they go out of scope.

What is the use of this? Dynamic allocations let you control when an object is created and destroyed. This will come in handy later.

## Memory leaks

So dynamically allocated objects created with `new` are not automatically deleted, unless you explicitly call `delete` on them. There is a risk here! It is called a *memory leak*. Memory leaks happen when an object allocated with `new` is not ever deleted. What can happen is that if a lot of objects in your program are allocated with `new` and then you stop using them, your computer will run out of memory eventually due to memory leakage.

Here is a ridiculous sample program to illustrate the problem:

[PRE7]

This program, if left to run long enough, will eventually gobble the computer's memory, as shown in the following screenshot:

![Memory leaks](img/00072.jpeg)

2 GB of RAM used for Player objects!

Note that no one ever intends to write a program with this type of problem in it! Memory leak problems happen accidentally. You must take care of your memory allocations and `delete` objects that are no longer in use.

# Regular arrays

An array in C++ can be declared as follows:

[PRE8]

The way this looks in memory is something like this:

![Regular arrays](img/00073.jpeg)

That is, inside the `array` variable are five slots or elements. Inside each of the slots is a regular `int` variable.

## The array syntax

So, how do you access one of the `int` values in the array? To access the individual elements of an array, we use square brackets, as shown in the following line of code:

[PRE9]

The preceding line of code would change the element at slot 0 of the array to a 10:

![The array syntax](img/00074.jpeg)

In general, to get to a particular slot of an array, you will write the following:

[PRE10]

Keep in mind that array slots are always indexed starting from 0\. To get into the first slot of the array, use `array[0]`. The second slot of the array is `array[1]` (not `array[2]`). The final slot of the array above is `array[4]` (not `array[5]`). The `array[5]` data type is out of bounds of the array! (There is no slot with index 5 in the preceding diagram. The highest index is 4.)

Don't go out of bounds of the array! It might work some times, but other times your program will crash with a **memory access violation** (accessing memory that doesn't belong to your program). In general, accessing memory that does not belong to your program is going to cause your app to crash, and if it doesn't do so immediately, there will be a hidden bug in your program that only causes problems once in a while. You must always be careful when indexing an array.

Arrays are built into C++, that is, you don't need to include anything special to have immediate use of arrays. You can have arrays of any type of data that you want, for example, arrays of `int`, `double`, `string`, and even your own custom object types (`Player`).

## Exercise

1.  Create an array of five strings and put inside it some names (made up or random, it doesn't matter).
2.  Create an array of doubles called `temps` with three elements and store the temperature for the last three days in it.

## Solutions

1.  The following is a sample program with an array of five strings:

    [PRE11]

2.  The following is just the array:

    [PRE12]

# C++ style dynamic size arrays (new[] and delete[])

It probably occurred to you that we won't always know the size of an array at the start of a program. We would need to allocate the array's size dynamically.

However, if you've tried it, you might have noticed that this doesn't work!

Let's try and use the `cin` command to take in an array size from the user. Let's ask the user how big he wants his array and try to create one for him of that size:

[PRE13]

We get the following error:

[PRE14]

The problem is that the compiler wants to allocate the size of the array. However, unless the variable size is marked `const`, the compiler will not be sure of its value at compile time. The C++ compiler cannot size the array at compile time, so it generates a compile time error.

To fix this, we have to allocate the array dynamically (on the "heap"):

[PRE15]

So the lessons here are as follows:

*   To allocate an array of some type (for example, `int`) dynamically, you must use new `int[numberOfElementsInArray]`.
*   Arrays allocated with `new[]` must be later deleted with `delete[]`, otherwise you'll get a memory leak! (that's `delete[]` with square brackets! Not regular delete).

# Dynamic C-style arrays

C-style arrays are a legacy topic, but they are still worth discussing since even though they are old, you might still see them used sometimes.

The way we declare a C-style array is as follows:

[PRE16]

The differences here are highlighted.

A C-style array is created using the `malloc()` function. The word malloc stands for "memory allocate". This function requires you to pass in the size of the array in bytes to create and not just the number of elements you want in the array. For this reason, we multiply the number of elements requested (size) by `sizeof` of the type inside the array. The size in bytes of a few typical C++ types is listed in the following table:

| C++ primitive type | sizeof (size in bytes) |
| --- | --- |
| `int` | 4 |
| `float` | 4 |
| `double` | 8 |
| `long long` | 8 |

Memory allocated with the `malloc()` function must later be released using `free()`.

# Summary

This chapter introduced you to C and C++ style arrays. In most of the UE4 code, you will use the UE4 editor built in collection classes (`TArray<T>`). However, you need familiarity with the basic C and C++ style arrays to be a very good C++ programmer.