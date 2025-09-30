# Chapter 04

# Process Memory Structure

In this chapter, we are going to talk about memory and its structure within a process. For a C programmer, memory management is always a crucial topic, and applying its best practices requires a basic knowledge about memory structure. In fact, this is not limited to C. In many programming languages such as C++ or Java, you need to have a fundamental understanding of memory and the way it works; otherwise, you face some serious issues that cannot be easily traced and fixed.

You might know that memory management is fully manual in C, and more than that, the programmer is the sole responsible person who allocates memory regions and deallocates them once they're no longer needed.

Memory management is different in high-level programming languages such as Java or C#, and it is done partly by the programmer and partly by the underlying language platform, such as **Java Virtual Machine** (**JVM**) in the case of using Java. In these languages, the programmer only issues memory allocations, but they are not responsible for the deallocations. A component called the *garbage collector* does the deallocation and frees up the allocated memory automatically.

Since there is no such garbage collector in C and C++, having some dedicated chapters for covering the concepts and issues regarding memory management is essential. That's why we have dedicated this chapter and the next to memory-related concepts, and these chapters together should give you a basic understanding of how memory works in C/C++.

Throughout this chapter:

*   We start by looking at the typical memory structure of a process. This will help us to discover the anatomy of a process and the way it interacts with the memory.
*   We discuss static and dynamic memory layouts.
*   We introduce the segments found in the aforementioned memory layouts. We see that some of them reside in the executable object file and the rest are created while the process is loading.
*   We introduce the probing tools and commands which can help us to detect the segments and see their content, both inside an object file and deep within a running process.

As part of this chapter, we get to know two segments called *Stack* and *Heap*. They are part of the dynamic memory layout of a process and all the allocations and deallocations happen in these segments. In the following chapter, we will discuss Stack and Heap segments in a greater detail because in fact, they are the segments that a programmer interacts with the most.

Let's start this chapter by talking about the *process memory layout*. This will give you an overall idea about how the memory of a running process is segmented, and what each segment is used for.

# Process memory layout

Whenever you run an executable file, the operating system creates a new process. A process is a live and running program that is loaded into the memory and has a unique **Process Identifier** (**PID**). The operating system is the sole responsible entity for spawning and loading new processes.

A process remains running until it either exits normally, or the process is given a signal, such as `SIGTERM`, `SIGINT`, or `SIGKILL`, which eventually makes it exit. The `SIGTERM` and `SIGINT` signals can be ignored, but `SIGKILL` will kill the process immediately and forcefully.

**Note**:

The signals mentioned in the preceding section are explained as follows:

`SIGTERM`: This is the termination signal. It allows the process to clean up.

`SIGINT`: This is the interrupt signal usually sent to the foreground process by pressing `Ctrl` + `C`.

`SIGKILL`: This is the kill signal and it closes the process forcefully without letting it clean up.

When creating a process, one of the first things that operating systems do is allocate a portion of memory dedicated to the process and then apply a predefined memory layout. This predefined memory layout is more or less the same in different operating systems, especially in Unix-like operating systems.

In this chapter, we're going to explore the structure of this memory layout, and a number of important and useful terms are introduced.

The memory layout of an ordinary process is divided into multiple parts. Each part is called a *segment*. Each segment is a region of memory which has a definite task and it is supposed to store a specific type of data. You can see the following list of segments being part of the memory layout of a running process:

*   Uninitialized data segment or **Block Started by Symbol** (**BSS**) segment
*   Data segment
*   Text segment or Code segment
*   Stack segment
*   Heap segment

In the following sections, we will study each of these segments individually, and we discuss the way they contribute to the execution of a program. In the next chapter, we will focus on Stack and Heap segments and we'll discuss them thoroughly. As part of our quest, let's introduce some tools that help us inspect the memory before going into the specifics of the above segments.

# Discovering memory structure

Unix-like operating systems provide a set of tools for inspecting the memory segments of a process. You learn in this section that some of these segments reside within the executable object file, and other segments are created dynamically at runtime, when the process is spawned.

As you should already know from the two previous chapters, an executable object file and a process are not the same thing, therefore it is expected to have different tools for inspecting each of them.

From the previous chapters, we know that an executable object file contains the machine instructions, and it is produced by the compiler. But a process is a running program spawned by executing an executable object file, consuming a region of the main memory, and the CPU is constantly fetching and executing its instructions.

A process is a living entity that is being executed inside the operating system while the executable object file is just a file containing a premade initial layout acting as a basis for spawning future processes. It is true that in the memory layout of a running process, some segments come directly from the base executable object file, and the rest are built dynamically at runtime while the process is being loaded. The former layout is called the **static memory layout**, and the latter is called the **dynamic memory layout**.

Static and dynamic memory layouts both have a predetermined set of segments. The content of the static memory layout is prewritten into the executable object file by the compiler, when compiling the source code. On the other hand, the content of the dynamic memory layout is written by the process instructions allocating memory for variables and arrays, and modifying them according to the program's logic.

With all that said, we can guess the content of the static memory layout either by just looking at the source code or the compiled object file. But this is not that easy regarding the dynamic memory layout as it cannot be determined without running the program. In addition, different runs of the same executable file can lead to different content in the dynamic memory layout. In other words, the dynamic content of a process is unique to that process and it should be investigated while the process is still running.

Let's begin with inspecting the static memory layout of a process.

# Probing static memory layout

The tools used for inspecting the static memory layout usually work on the object files. To get some initial insight, we'll start with an example, *example 4.1*, which is a minimal C program that doesn't have any variable or logic as part of it:

```cpp
int main(int argc, char** argv) {
  return 0;
}
```

Code Box 4-1 [ExtremeC_examples_chapter4_1.c]: A minimal C program

First, we need to compile the preceding program. We compile it in Linux using `gcc`:

```cpp
$ gcc ExtremeC_examples_chapter4_1.c -o ex4_1-linux.out
$
```

Shell Box 4-1: Compiling example 4.1 using gcc in Linux

After a successful compilation and having the final executable binary linked, we get an executable object file named `ex4_1-linux.out`. This file contains a predetermined static memory layout that is specific to the Linux operating system, and it will exist in all future processes spawned based on this executable file.

The `size` command is the first tool that we want to introduce. It can be used to print the static memory layout of an executable object file.

You can see the usage of the `size` command in order to see the various segments found as part of the static memory layout as follows:

```cpp
$ size ex4_1-linux.out
   text    data     bss     dec     hex   filename
   1099     544       8    1651     673   ex4_1-linux.out
$
```

Shell Box 4-2: Using the size command to see the static segments of ex4_1-linux.out

As you see, we have Text, Data, and BSS segments as part of the static layout. The shown sizes are in bytes.

Now, let's compile the same code, *example 4.1*, in a different operating system. We have chosen macOS and we are going to use the `clang` compiler:

```cpp
$ clang ExtremeC_examples_chapter4_1.c -o ex4_1-macos.out
$
```

Shell Box 4-3: Compiling example 4.1 using clang in macOS

Since macOS is a POSIX-compliant operating system just like Linux, and the `size` command is specified to be part of the POSIX utility programs, macOS should also have the `size` command. Therefore, we can use the same command to see the static memory segments of `ex4_1-macos.out`:

```cpp
$ size ex4_1-macos.out
__TEXT __DATA  __OBJC  others       dec         hex
4096   0       0       4294971392   4294975488  100002000
$ size -m ex4_1-macos.out
Segment __PAGEZERO: 4294967296
Segment __TEXT: 4096
    Section __text: 22
    Section __unwind_info: 72
    total 94
Segment __LINKEDIT: 4096
total 4294975488
$
```

Shell Box 4-4: Using the size command to see the static segments of ex4_1-macos.out

In the preceding shell box, we have run the `size` command twice; the second run gives us more details about the found memory segments. You might have noticed that we have Text and Data segments in macOS, just like Linux, but there is no BSS segment. Note that the BSS segment also exists in macOS, but it is not shown in the `size` output. Since the BSS segment contains uninitialized global variables, there is no need to allocate some bytes as part of the object file and it is enough to know how many bytes are required for storing those global variables.

In the preceding shell boxes, there is an interesting point to note. The size of the Text segment is 1,099 bytes in Linux while it is 4 KB in macOS. It can also be seen that the Data segment for a minimal C program has a non-zero size in Linux, but it is empty in macOS. It is apparent that the low-level memory details are different on various platforms.

Despite these little differences between Linux and macOS, we can see that both platforms have the Text, Data, and BSS segments as part of their static layout. From now on, we gradually explain what each of these segments are used for. In the upcoming sections, we'll discuss each segment separately and we give an example slightly different from *example 4.1* for each, in order to see how differently each segment responds to the minor changes in the code.

## BSS segment

We start with the BSS segment. **BSS** stands for **Block Started by Symbol**. Historically, the name was used to denote reserved regions for uninitialized words. Basically, that's the purpose that we use the BSS segment for; either uninitialized global variables or global variables set to zero.

Let's expand *example 4.1* by adding a few uninitialized global variables. You see that uninitialized global variables will contribute to the BSS segment. The following code box demonstrates *example 4.2*:

```cpp
int global_var1;
int global_var2;
int global_var3 = 0;
int main(int argc, char** argv) {
  return 0;
}
```

Code Box 4-2 [ExtremeC_examples_chapter4_2.c]: A minimal C program with a few global variables either uninitialized or set to zero

The integers `global_var1`, `global_var2`, and `global_var3` are global variables which are uninitialized. For observing the changes made to the resulting executable object file in Linux, in comparison to *example 4.1*, we again run the `size` command:

```cpp
$ gcc ExtremeC_examples_chapter4_2.c -o ex4_2-linux.out
$ size ex4_2-linux.out
   text    data     bss     dec     hex   filename
   1099     544      16    1659     67b   ex4_2-linux.out
$
```

Shell Box 4-5: Using the size command to see the static segments of ex4_2-linux.out

If you compare the preceding output with a similar output from *example 4.1*, you will notice that the size of the BSS segment has changed. In other words, declaring global variables that are *not* initialized or set to zero will add up to the BSS segment. These special global variables are part of the static layout and they become preallocated when a process is loading, and they never get deallocated until the process is alive. In other words, they have a static lifetime.

**Note**:

Because of design concerns, we usually prefer to use local variables in our algorithms. Having too many global variables can increase the binary size. In addition, keeping sensitive data in the global scope, it can introduce security concerns. Concurrency issues, especially data races, namespace pollution, unknown ownership, and having too many variables in the global scope, are some of the complications that global variables introduce.

Let's compile *example 4.2* in macOS and have a look at the output of the `size` command:

```cpp
$ clang ExtremeC_examples_chapter4_2.c -o ex4_2-macos.out
$ size ex4_2-macos.out 
__TEXT __DATA  __OBJC  others       dec         hex
4096   4096       0    4294971392   4294979584  100003000
$ size -m ex4_2-macos.out
Segment __PAGEZERO: 4294967296
Segment __TEXT: 4096
    Section __text: 22
    Section __unwind_info: 72
    total 94
Segment __DATA: 4096
    Section __common: 12
    total 12
Segment __LINKEDIT: 4096
total 4294979584
$
```

Shell Box 4-6: Using the size command to see the static segments of ex4_2-macos.out

And again, it is different from Linux. In Linux, we had preallocated 8 bytes for the BSS segment, when we had no global variables. In *example 4.2*, we added three new uninitialized global variables whose sizes sum up to 12 bytes, and the Linux C compiler expanded the BSS segment by 8 bytes. But in macOS, we still have no BSS segment as part of the `size`'s output, but the compiler has expanded the `data` segment from 0 bytes to 4KB, which is the default page size in macOS. This means that `clang` has allocated a new memory page for the `data` segment inside the layout. Again, this simply shows how much the details of the memory layout can be different in various platforms.

**Note**:

While allocating the memory, it doesn't matter how many bytes a program needs to allocate. The *allocator* always acquires memory in terms of *memory pages* until the total allocated size covers the program's need. More information about the Linux memory allocator can be found here: [https://www.kernel.org/doc/gorman/html/understand/understand009.html](https://www.kernel.org/doc/gorman/html/understand/understand009.html).

In *Shell Box 4-6*, we have a section named `__common`, inside the `_DATA` segment, which is 12 bytes, and it is in fact referring to the BSS segment that is not shown as BSS in the `size`'s output. It refers to 3 uninitialized global integer variables or 12 bytes (each integer being 4 bytes). It's worth taking note that uninitialized global variables are set to *zero* by default. There is no other value that could be imagined for uninitialized variables.

Let's now talk about the next segment in the static memory layout; the Data segment.

## Data segment

In order to show what type of variables are stored in the Data segment, we are going to declare more global variables, but this time we initialize them with non-zero values. The following example, *example 4.3*, expands *example 4.2* and adds two new initialized global variables:

```cpp
int global_var1;
int global_var2;
int global_var3 = 0;
double global_var4 = 4.5;
char global_var5 = 'A';
int main(int argc, char** argv) {
  return 0;
}
```

Code Box 4-3 [ExtremeC_examples_chapter4_3.c]: A minimal C program with both initialized and uninitialized global variables

The following shell box shows the output of the `size` command, in Linux, and for *example 4.3*:

```cpp
$ gcc ExtremeC_examples_chapter4_3.c -o ex4_3-linux.out
$ size ex4_3-linux.out
   text    data     bss     dec     hex filename
   1099     553      20    1672     688 ex4_3-linux.out
$
```

Shell Box 4-7: Using the size command to see the static segments of ex4_3-linux.out

We know that the Data segment is used to store the initialized global variables set to a non-zero value. If you compare the output of the `size` command for *examples 4.2* and *4.3*, you can easily see that the Data segment is increased by 9 bytes, which is the sum of the sizes of the two newly added global variables (one 8-byte `double` and one 1-byte `char`).

Let's look at the changes in macOS:

```cpp
$ clang ExtremeC_examples_chapter4_3.c -o ex4_3-macos.out
$ size ex4_3-macos.out 
__TEXT __DATA  __OBJC  others       dec         hex
4096   4096       0    4294971392   4294979584  100003000
$ size -m ex4_3-macos.out
Segment __PAGEZERO: 4294967296
Segment __TEXT: 4096
    Section __text: 22
    Section __unwind_info: 72
    total 94
Segment __DATA: 4096
    Section __data: 9
    Section __common: 12
    total 21
Segment __LINKEDIT: 4096
total 4294979584
$
```

Shell Box 4-8: Using the size command to see the static segments of ex4_3-macos.out

In the first run, we see no changes since the size of all global variables summed together is still way below 4KB. But in the second run, we see a new section as part of the `_DATA` segment; the `__data` section. The memory allocated for this section is 9 bytes, and it is in accordance with the size of the newly introduced initialized global variables. And still, we have 12 bytes for uninitialized global variables as we had in *example 4.2*, and in macOS.

On a further note, the `size` command only shows the size of the segments, but not their contents. There are other commands, specific to each operating system, that can be used to inspect the content of segments found in an object file. For instance, in Linux, you have `readelf` and `objdump` commands in order to see the content of *ELF* files. These tools can also be used to probe the static memory layout inside the object files. As part of two previous chapters we explored some of these commands.

Other than global variables, we can have some static variables declared inside a function. These variables retain their values while calling the same function multiple times. These variables can be stored either in the Data segment or the BSS segment depending on the platform and whether they are initialized or not. The following code box demonstrates how to declare some static variables within a function:

```cpp
void func() {
  static int i;
  static int j = 1;
  ...
}
```

Code Box 4-4: Declaration of two static variables, one initialized and the other one uninitialized

As you see in *Code Box 4-4*, the `i` and `j` variables are static. The `i` variable is uninitialized and the `j` variable is initialized with value `1`. It doesn't matter how many times you enter and leave the `func` function, these variables keep their most recent values.

To elaborate more on how this is done, at runtime, the `func` function has access to these variables located in either the Data segment or the BSS segment, which has a static lifetime. That's basically why these variables are called *static*. We know that the `j` variable is located in the Data segment simply because it has an initial value, and the `i` variable is supposed to be inside the BSS segment since it is not initialized.

Now, we want to introduce the second command to examine the content of the BSS segment. In Linux, the `objdump` command can be used to print out the content of memory segments found in an object file. This corresponding command in macOS is `gobjdump` which should be installed first.

As part of *example 4.4*, we try to examine the resulting executable object file to find the data written to the Data segment as some global variables. The following code box shows the code for *example 4.4*:

```cpp
int     x = 33;            // 0x00000021
int     y = 0x12153467;
char z[6] = "ABCDE";
int main(int argc, char**argv) {
  return 0;
}
```

Code Box 4-5 [ExtremeC_examples_chapter4_4.c]: Some initialized global variables which should be written to the Data segment

The preceding code is easy to follow. It just declares three global variables with some initial values. After compilation, we need to dump the content of the Data segment in order to find the written values.

The following commands will demonstrate how to compile and use `objdump` to see the content of the Data segment:

```cpp
$ gcc ExtremeC_examples_chapter4_4.c -o ex4_4.out
$ objdump -s -j .data ex4_4.out
a.out:     file format elf64-x86-64
Contents of section .data:
 601020 00000000 00000000 00000000 00000000  ...............
 601030 21000000 67341512 41424344 4500      !....4..ABCDE.
$
```

Shell Box 4-9: Using the objdump command to see the content of the Data segment

Let's explain how the preceding output, and especially the contents of the section `.data`, should be read. The first column on the left is the address column. The next four columns are the contents, and each of them is showing `4` bytes of data. So, in each row, we have the contents of 16 bytes. The last column on the right shows the ASCII representation of the same bytes shown in the middle columns. A dot character means that the character cannot be shown using alphanumerical characters. Note that the option `-s` tells `objdump` to show the full content of the chosen section and the option `-j .data` tells it to show the content of the section `.data`.

The first line is 16 bytes filled by zeros. There is no variable stored here, so nothing special for us. The second line shows the contents of the Data segment starting with the address `0x601030`. The first 4 bytes is the value stored in the `x` variable found in *example 4.4*. The next 4 bytes also contain the value for the `y` variable. The final 6 bytes are the characters inside the `z` array. The contents of `z` can be clearly seen in the last column.

If you pay enough attention to the content shown in *Shell Box 4-9*, you see that despite the fact that we write 33, in decimal base, as `0x00000021`, in hexadecimal base it is stored differently in the segment. It is stored as `0x21000000`. This is also true for the content of the `y` variable. We have written it as `0x12153467`, but it is stored differently as `0x67341512`. It seems that the order of bytes is reversed.

The effect explained is because of the *endianness* concept. Generally, we have two different types of endianness, *big-endian* and *little-endian*. The value `0x12153467` is the big-endian representation for the number `0x12153467`, as the biggest byte, `0x12`, comes first. But the value `0x67341512` is the little-endian representation for the number `0x12153467`, as the smallest byte, `0x67`, comes first.

No matter what the endianness is, we always read the correct value in C. Endianness is a property of the CPU and with a different CPU you may get a different byte order in your final object files. This is one of the reasons why you cannot run an executable object file on hardware with different endianness.

It would be interesting to see the same output on a macOS machine. The following shell box demonstrates how to use the `gobjdump` command in order to see the content of the Data segment:

```cpp
$ gcc ExtremeC_examples_chapter4_4.c -o ex4_4.out
$ gobjdump -s -j .data ex4_4.out
a.out:     file format mach-o-x86-64
Contents of section .data:
 100001000 21000000 67341512 41424344 4500      !...g4..ABCDE.
$
```

Shell Box 4-10: Using the gobjdump command in macOS to see the content of the Data segment

It should be read exactly like the Linux output found as part of *Shell Code 4-9*. As you see, in macOS, there are no 16-byte zero headers in the data segment. Endianness of the contents also shows that the binary has been compiled for a little-endian processor.

As a final note in this section, other tools like `readelf` in Linux and `dwarfdump` in macOS can be used in order to inspect the content of object files. The binary content of the object files can also be read using tools such as `hexdump`.

In the following section, we will discuss the Text segment and how it can be inspected using `objdump`.

## Text segment

As we know from *Chapter 2*, *Compilation and Linking*, the linker writes the resulting machine-level instructions into the final executable object file. Since the Text segment, or the Code segment, contains all the machine-level instructions of a program, it should be located in the executable object file, as part of its static memory layout. These instructions are fetched by the processor and get executed at runtime when the process is running.

To dive deeper, let's have a look at the Text segment of a real executable object file. For this purpose, we propose a new example. The following code box shows *example 4.5*, and as you see, it is just an empty `main` function:

```cpp
int main(int argc, char** argv) {
  return 0;
}
```

Code Box 4-6 [ExtremeC_examples_chapter4_5.c]: A minimal C program

We can use the `objdump` command to dump the various parts of the resulting executable object file. Note that the `objdump` command is only available in Linux, while other operating systems have their own set of commands to do the same.

The following shell box demonstrates using the `objdump` command to extract the content of various sections present in the executable object file resulting from *example 4.5*. Note that the output is shortened in order to only show the `main` function's corresponding section and its assembly instructions:

```cpp
$ gcc ExtremeC_examples_chapter4_5.c -o ex4_5.out
$ objdump -S ex4_5.out
ex4_5.out:     file format elf64-x86-64
Disassembly of section .init:
0000000000400390 <_init>:
... truncated.
.
.
Disassembly of section .plt:
00000000004003b0 <__libc_start_main@plt-0x10>:
... truncated
00000000004004d6 <main>:
  4004d6:   55                      push   %rbp
  4004d7:   48 89 e5                mov    %rsp,%rbp
  4004da:   b8 00 00 00 00          mov    $0x0,%eax
  4004df:   5d                      pop    %rbp
  4004e0:   c3                      retq
  4004e1:   66 2e 0f 1f 84 00 00    nopw   %cs:0x0(%rax,%rax,1)
  4004e8:   00 00 00
  4004eb:   0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)
00000000004004f0 <__libc_csu_init>:
... truncated
.
.
.
0000000000400564 <_fini>:
... truncated
$
```

Shell Box 4-11: Using objdump to show the content of the section corresponding to the main function

As you see in the preceding shell box, there are various sections containing machine-level instructions: the `.text`, `.init`, and `.plt` sections and some others, which all together allow a program to become loaded and running. All of these sections are part of the same Text segment found in the static memory layout, inside the executable object file.

Our C program, written for *example 4.5*, had only one function, the `main` function, but as you see, the final executable object file has a dozen other functions.

The preceding output, seen as part of *Shell Box 4-11*, shows that the `main` function is not the first function to be called in a C program and there are logics before and after `main` that should be executed. As explained in *Chapter 2*, *Compilation and Linking*, in Linux, these functions are usually borrowed from the `glibc` library, and they are put together by the linker to form the final executable object file.

In the following section, we start to probe the dynamic memory layout of a process.

# Probing dynamic memory layout

The dynamic memory layout is actually the runtime memory of a process, and it exists as long as the process is running. When you execute an executable object file, a program called *loader* takes care of the execution. It spawns a new process and it creates the initial memory layout which is supposed to be dynamic. To form this layout, the segments found in the static layout will be copied from the executable object file. More than that, two new segments will also be added to it. Only then can the process proceed and become running.

In short, we expect to have five segments in the memory layout of a running process. Three of these segments are directly copied from the static layout found in the executable object file. The two newly added segments are called Stack and Heap segments. These segments are dynamic, and they exist only when the process is running. This means that you cannot find any trace of them as part of the executable object file.

In this section, our ultimate goal is to probe the Stack and Heap segments and introduce tools and places in an operating system which can be used for this purpose. From time to time, we might refer to these segments as the process's dynamic memory layout, without considering the other three segments copied from the object file, but you should always remember that the dynamic memory of a process consists of all five segments together.

The Stack segment is the default memory region where we allocate variables from. It is a limited region in terms of size, and you cannot hold big objects in it. In contrast, the Heap segment is a bigger and adjustable region of memory which can be used to hold big objects and huge arrays. Working with the Heap segment requires its own API which we introduce as part of our discussion.

Remember, dynamic memory layout is different from *Dynamic Memory Allocation*. You should not mix these two concepts, since they are referring to two different things! As we progress, we'll learn more about different types of memory allocations, especially dynamic memory allocation.

The five segments found in the dynamic memory of a process are referring to parts of the main memory that are already *allocated*, *dedicated*, and *private* to a running process. These segments, excluding the Text segment, which is literally static and constant, are dynamic in a sense that their contents are always changing at runtime. That's due to the fact that these segments are constantly being modified by the algorithm that the process is executing.

Inspecting the dynamic memory layout of a process requires its own procedure. This implies that we need to have a running process before being able to probe its dynamic memory layout. This requires us to write examples which remain running for a fairly long time in order to keep their dynamic memory in place. Then, we can use our inspection tools to study their dynamic memory structure.

In the following section, we give an example on how to probe the structure of dynamic memory.

## Memory mappings

Let's start with a simple example. *Example 4.6* will be running for an indefinite amount of time. This way, we have a process that never dies, and in the meantime, we can probe its memory structure. And of course, we can *kill* it whenever we are done with the inspection. You can find the example in the following code box:

```cpp
#include <unistd.h> // Needed for sleep function
int main(int argc, char** argv) {
  // Infinite loop
  while (1) {
    sleep(1); // Sleep 1 second
  };
  return 0;
}
```

Code Box 4-6 [ExtremeC_examples_chapter4_6.c]: Example 4.6 used for probing dynamic memory layout

As you see, the code is just an infinite loop, which means that the process will run forever. So, we have enough time to inspect the process's memory. Let's first build it.

**Note**:

The `unistd.h` header is available only on Unix-like operating systems; to be more precise, in POSIX-compliant operating systems. This means that on Microsoft Windows, which is not POSIX-compliant, you have to include the `windows.h` header instead.

The following shell box shows how to compile the example in Linux:

```cpp
$ gcc ExtremeC_examples_chapter4_6.c -o ex4_6.out
$
```

Shell Box 4-12: Compiling example 4.6 in Linux

Then, we run it as follows. In order to use the same prompt for issuing further commands while the process is running, we should start the process in the background:

```cpp
$ ./ ex4_6.out &
[1] 402
$
```

Shell Box 4-13: Running example 4.6 in the background

The process is now running in the background. According to the output, the PID of the recently started process is 402, and we will use this PID to kill it in the future. The PID is different every time you run a program; therefore, you'll probably see a different PID on your computer. Note that whenever you run a process in the background, the shell prompt returns immediately, and you can issue further commands.

**Note**:

If you have the PID (Process ID) of a process, you can easily end it using the `kill` command. For example, if the PID is 402, the following command will work in Unix-like operating systems: `kill -9 402`.

The PID is the identifier we use to inspect the memory of a process. Usually, an operating system provides its own specific mechanism to query various properties of a process based on its PID. But here, we are only interested in the dynamic memory of a process and we'll use the available mechanism in Linux to find more about the dynamic memory structure of the above running process.

On a Linux machine, the information about a process can be found in files under the `/proc` directory. It uses a special filesystem called *procfs*. This filesystem is not an ordinary filesystem meant for keeping actual files, but it is more of a hierarchical interface to query about various properties of an individual process or the system as a whole.

**Note**:

procfs is not limited to Linux. It is usually part of Unix-like operating systems, but not all Unix-like operating systems use it. For example, FreeBSD uses this filesystem, but macOS doesn't.

Now, we are going to use procfs to see the memory structure of the running process. The memory of a process consists of a number of *memory mappings*. Each memory mapping represents a dedicated region of memory which is mapped to a specific file or segment as part of the process. Shortly, you'll see that both Stack and Heap segments have their own memory mappings in each process.

One of the things that you can use procfs for is to observe the current memory mappings of the process. Next, we are going to show this.

We know that the process is running with PID 402\. Using the `ls` command, we can see the contents of the `/proc/402` directory, shown as follows:

```cpp
$ ls -l /proc/402
total of 0
dr-xr-xr-x  2 root root 0 Jul 15 22:28 attr
-rw-r--r--  1 root root 0 Jul 15 22:28 autogroup
-r--------  1 root root 0 Jul 15 22:28 auxv
-r--r--r--  1 root root 0 Jul 15 22:28 cgroup
--w-------  1 root root 0 Jul 15 22:28 clear_refs
-r--r--r--  1 root root 0 Jul 15 22:28 cmdline
-rw-r--r--  1 root root 0 Jul 15 22:28 comm
-rw-r--r--  1 root root 0 Jul 15 22:28 coredump_filter
-r--r--r--  1 root root 0 Jul 15 22:28 cpuset
lrwxrwxrwx  1 root root 0 Jul 15 22:28 cwd -> /root/codes
-r--------  1 root root 0 Jul 15 22:28 environ
lrwxrwxrwx  1 root root 0 Jul 15 22:28 exe -> /root/codes/a.out
dr-x------  2 root root 0 Jul 15 22:28 fd
dr-x------  2 root root 0 Jul 15 22:28 fdinfo
-rw-r--r--  1 root root 0 Jul 15 22:28 gid_map
-r--------  1 root root 0 Jul 15 22:28 io
-r--r--r--  1 root root 0 Jul 15 22:28 limits
...
$
```

Shell Box 4-14: Listing the content of /proc/402

As you can see, there are many files and directories under the `/proc/402` directory. Each of these files and directories corresponds to a specific property of the process. For querying the memory mappings of the process, we have to see the contents of the file `maps` under the PID directory. We use the `cat` command to dump the contents of the `/proc/402/maps` file. It can be seen as follows:

```cpp
$ cat /proc/402/maps
00400000-00401000 r-xp 00000000 08:01 790655              .../extreme_c/4.6/ex4_6.out
00600000-00601000 r--p 00000000 08:01 790655              .../extreme_c/4.6/ex4_6.out
00601000-00602000 rw-p 00001000 08:01 790655              .../extreme_c/4.6/ex4_6.out
7f4ee16cb000-7f4ee188a000 r-xp 00000000 08:01 787362      /lib/x86_64-linux-gnu/libc-2.23.so
7f4ee188a000-7f4ee1a8a000 ---p 001bf000 08:01 787362      /lib/x86_64-linux-gnu/libc-2.23.so
7f4ee1a8a000-7f4ee1a8e000 r--p 001bf000 08:01 787362      /lib/x86_64-linux-gnu/libc-2.23.so
7f4ee1a8e000-7f4ee1a90000 rw-p 001c3000 08:01 787362      /lib/x86_64-linux-gnu/libc-2.23.so
7f4ee1a90000-7f4ee1a94000 rw-p 00000000 00:00 0
7f4ee1a94000-7f4ee1aba000 r-xp 00000000 08:01 787342      /lib/x86_64-linux-gnu/ld-2.23.so
7f4ee1cab000-7f4ee1cae000 rw-p 00000000 00:00 0
7f4ee1cb7000-7f4ee1cb9000 rw-p 00000000 00:00 0
7f4ee1cb9000-7f4ee1cba000 r--p 00025000 08:01 787342      /lib/x86_64-linux-gnu/ld-2.23.so
7f4ee1cba000-7f4ee1cbb000 rw-p 00026000 08:01 787342      /lib/x86_64-linux-gnu/ld-2.23.so
7f4ee1cbb000-7f4ee1cbc000 rw-p 00000000 00:00 0
7ffe94296000-7ffe942b7000 rw-p 00000000 00:00 0           [stack]
7ffe943a0000-7ffe943a2000 r--p 00000000 00:00 0           [vvar]
7ffe943a2000-7ffe943a4000 r-xp 00000000 00:00 0           [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0   [vsyscall]
$
```

Shell Box 4-15: Dumping the content of /proc/402/maps

As you see in *Shell Box 4-15*, the result consists of a number of rows. Each row represents a memory mapping that indicates a range of memory addresses (a region) that are allocated and mapped to a specific file or segment in the dynamic memory layout of the process. Each mapping has a number of fields separated by one or more spaces. Next, you can find the descriptions of these fields from left to right:

*   **Address range**: These are the start and end addresses of the mapped range. You can find a file path in front of them if the region is mapped to a file. This is a smart way to map the same loaded shared object file in various processes. We have talked about this as part of *Chapter 3*, *Object Files*.
*   **Permissions**: This indicates whether the content can be executed (`x`), read (`r`), or modified (`w`). The region can also be shared (`s`) by the other processes or be private (`p`) only to the owning process.
*   **Offset**: If the region is mapped to a file, this is the offset from the beginning of the file. It is usually 0 if the region is not mapped to a file.
*   **Device**: If the region is mapped to a file, this would be the device number (in the form of m:n), indicating a device that contains the mapped file. For example, this would be the device number of the hard disk that contains a shared object file.
*   **The inode**: If the region is mapped to a file, that file should reside on a filesystem. Then, this field would be the inode number of the file in that filesystem. An *inode* is an abstract concept within filesystems such as *ext4* which are mostly used in Unix-like operating systems. Each inode can represent both files and directories. Every inode has a number that is used to access its content.
*   **Pathname or description**: If the region is mapped to a file, this would be the path to that file. Otherwise, it would be left empty, or it would describe the purpose of the region. For example, `[stack]` indicates that the region is actually the Stack segment.

The `maps` file provides even more useful information regarding the dynamic memory layout of a process. We'll need a new example to properly demonstrate this.

## Stack segment

First, let's talk more about the Stack segment. The Stack is a crucial part of the dynamic memory in every process, and it exists in almost all architectures. You have seen it in the memory mappings described as `[stack]`.

Both Stack and Heap segments have dynamic contents which are constantly changing while the process is running. It is not easy to see the dynamic contents of these segments and most of the time you need a debugger such as `gdb` to go through the memory bytes and read them while a process is running.

As pointed out before, the Stack segment is usually limited in size, and it is not a good place to store big objects. If the Stack segment is full, the process cannot make any further function calls since the function call mechanism relies heavily on the functionality of the Stack segment.

If the Stack segment of a process becomes full, the process gets terminated by the operating system. *Stack overflow* is a famous error that happens when the Stack segment becomes full. We discuss the function call mechanism in future paragraphs.

As explained before, the Stack segment is a default memory region that variables are allocated from. Suppose that you've declared a variable inside a function, as follows:

```cpp
void func() {
  // The memory required for the following variable is
  // allocated from the stack segment.
  int a; 
  ... 
}
```

Code Box 4-7: Declaring a local variable which has its memory allocated from the Stack segment

In the preceding function, while declaring the variable, we have not mentioned anything to let the compiler know which segment the variable should be allocated from. Because of this, the compiler uses the Stack segment by default. The Stack segment is the first place that allocations are made from.

As its name implies, it is a *stack*. If you declare a local variable, it becomes allocated on top of the Stack segment. When you're leaving the scope of the declared local variable, the compiler has to pop the local variables first in order to bring up the local variables declared in the outer scope.

**Note**:

Stack, in its abstract form, is a **First In, Last Out** (**FILO**) or **Last In, First Out** (**LIFO**) data structure. Regardless of the implementation details, each entry is stored (pushed) on top of the stack, and it will be buried by further entries. One entry cannot be popped out without removing the above entries first.

Variables are not the only entities that are stored in the Stack segment. Whenever you make a function call, a new entry called a *stack frame* is placed on top of the Stack segment. Otherwise, you cannot return to the calling function or return the result back to the caller.

Having a healthy stacking mechanism is vital to have a working program. Since the size of the Stack is limited, it is a good practice to declare small variables in it. Also, the Stack shouldn't be filled by too many stack frames as a result of making infinite *recursive* calls or too many function calls.

From a different perspective, the Stack segment is a region used by you, as a programmer, to keep your data and declare the local variables used in your algorithms, and by the operating system, as the program runner, to keep the data needed for its internal mechanisms to execute your program successfully.

In this sense, you should be careful when working with this segment because misusing it or corrupting its data can interrupt the running process or even make it crash. The Heap segment is the memory segment that is only managed by the programmer. We will cover the Heap segment in the next section.

It is not easy to see the contents of the Stack segment from outside if we are only using the tools we've introduced for probing the static memory layout. This part of memory contains private data and can be sensitive. It is also private to the process, and other processes cannot read or modify it.

So, for sailing through the Stack memory, one has to attach something to a process and see the Stack segment through the eyes of that process. This can be done using a *debugger* program. A debugger attaches to a process and allows a programmer to control the target process and investigate its memory content. We will use this technique and examine the Stack memory in the following chapter. For now, we leave the Stack segment to discuss more about the Heap segment. We will get back to the Stack in the next chapter.

## Heap segment

The following example, *example 4.7*, shows how memory mappings can be used to find regions allocated for the Heap segment. It is quite similar to *example 4.6*, but it allocates a number of bytes from the Heap segment before entering the infinite loop.

Therefore, just like we did for *example 4.6*, we can go through the memory mappings of the running process and see which mapping refers to the Heap segment.

The following code box contains the code for *example 4.7*:

```cpp
#include <unistd.h> // Needed for sleep function
#include <stdlib.h> // Needed for malloc function
#include <stdio.h> // Needed for printf
int main(int argc, char** argv) {
  void* ptr = malloc(1024); // Allocate 1KB from heap
  printf("Address: %p\n", ptr);
  fflush(stdout); // To force the print
  // Infinite loop
  while (1) {
    sleep(1); // Sleep 1 second
  };
  return 0;
}
```

Code Box 4-8 [ExtremeC_examples_chapter4_7.c]: Example 4.7 used for probing the Heap segment

In the preceding code, we used the `malloc` function. It's the primary way to allocate extra memory from the Heap segment. It accepts the number of bytes that should be allocated, and it returns a generic pointer.

As a reminder, a generic pointer (or a void pointer) contains a memory address but it cannot be *dereferenced* and used directly. It should be cast to a specific pointer type before being used.

In *example 4.7*, we allocate 1024 bytes (or 1KB) before entering the loop. The program also prints the address of the pointer received from `malloc` before starting the loop. Let's compile the example and run it as we did for *example 4.7*:

```cpp
$ g++ ExtremeC_examples_chapter4_7.c -o ex4_7.out
$ ./ex4_7.out &
[1] 3451
Address: 0x19790010
$
```

Shell Box 4-16: Compiling and running example 4.7

Now, the process is running in the background, and it has obtained the PID 3451\.

Let's see what memory regions have been mapped for this process by looking at its `maps` file:

```cpp
$ cat /proc/3451/maps
00400000-00401000 r-xp 00000000 00:2f 176521             .../extreme_c/4.7/ex4_7.out
00600000-00601000 r--p 00000000 00:2f 176521             .../extreme_c/4.7/ex4_7.out
00601000-00602000 rw-p 00001000 00:2f 176521             .../extreme_c/4.7/ex4_7.out
01979000-0199a000 rw-p 00000000 00:00 0                  [heap]
7f7b32f12000-7f7b330d1000 r-xp 00000000 00:2f 30         /lib/x86_64-linux-gnu/libc-2.23.so
7f7b330d1000-7f7b332d1000 ---p 001bf000 00:2f 30         /lib/x86_64-linux-gnu/libc-2.23.so
7f7b332d1000-7f7b332d5000 r--p 001bf000 00:2f 30         /lib/x86_64-linux-gnu/libc-2.23.so
7f7b332d5000-7f7b332d7000 rw-p 001c3000 00:2f 30         /lib/x86_64-linux-gnu/libc-2.23.so
7f7b332d7000-7f7b332db000 rw-p 00000000 00:00 0 
7f7b332db000-7f7b33301000 r-xp 00000000 00:2f 27        /lib/x86_64-linux-gnu/ld-2.23.so
7f7b334f2000-7f7b334f5000 rw-p 00000000 00:00 0 
7f7b334fe000-7f7b33500000 rw-p 00000000 00:00 0 
7f7b33500000-7f7b33501000 r--p 00025000 00:2f 27         /lib/x86_64-linux-gnu/ld-2.23.so
7f7b33501000-7f7b33502000 rw-p 00026000 00:2f 27         /lib/x86_64-linux-gnu/ld-2.23.so
7f7b33502000-7f7b33503000 rw-p 00000000 00:00 0 
7ffdd63c2000-7ffdd63e3000 rw-p 00000000 00:00 0          [stack]
7ffdd63e7000-7ffdd63ea000 r--p 00000000 00:00 0          [vvar]
7ffdd63ea000-7ffdd63ec000 r-xp 00000000 00:00 0          [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0  [vsyscall]
$
```

Shell Box 4-17: Dumping the content of /proc/3451/maps

If you look at *Shell Box 4-17* carefully, you will see a new mapping which is highlighted, and it is being described by `[heap]`. This region has been added because of using the `malloc` function. If you calculate the size of the region, it is `0x21000` bytes or 132 KB. This means that to allocate only 1 KB in the code, a region of the size 132 KB has been allocated.

This is usually done in order to prevent further memory allocations when using `malloc` again in the future. That's simply because the memory allocation from the Heap segment is not cheap and it has both memory and time overheads.

If you go back to the code shown in *Code Box 4-8*, the address that the `ptr` pointer is pointing to is also interesting. The Heap's memory mapping, shown in *Shell Box 4-17*, is allocated from the address `0x01979000` to `0x0199a000`, and the address stored in `ptr` is `0x19790010`, which is obviously inside the Heap range, located at an offset of `16` bytes.

The Heap segment can grow to sizes far greater than 132 KB, even to tens of gigabytes, and usually it is used for permanent, global, and very big objects such as arrays and bit streams.

As pointed out before, allocation and deallocation within the heap segment require a program to call specific functions provided by the C standard. While you can have local variables on top of the Stack segment, and you can use them directly to interact with the memory, the Heap memory can be accessed only through pointers, and this is one of the reasons why knowing pointers and being able to work with them is crucial to every C programmer. Let's bring up *example 4.8*, which demonstrates how to use pointers to access the Heap space:

```cpp
#include <stdio.h>   // For printf function
#include <stdlib.h>  // For malloc and free function
void fill(char* ptr) {
  ptr[0] = 'H';
  ptr[1] = 'e';
  ptr[2] = 'l';
  ptr[3] = 'l';
  ptr[5] = 0;
}
int main(int argc, char** argv) {
  void* gptr = malloc(10 * sizeof(char));
  char* ptr = (char*)gptr;
  fill(ptr);
  printf("%s!\n", ptr);
  free(ptr);
  return 0;
}
```

Code Box 4-9 [ExtremeC_examples_chapter4_8.c]: Using pointers to interact with the Heap memory

The preceding program allocates 10 bytes from the Heap space using the `malloc` function. The `malloc` function receives the number of bytes that should be allocated and returns a generic pointer addressing the first byte of the allocated memory block.

For using the returned pointer, we have to cast it to a proper pointer type. Since we are going to use the allocated memory to store some characters, we choose to cast it to a `char` pointer. The casting is done before calling the `fill` function.

Note that the local pointer variables, `gptr` and `ptr`, are allocated from the Stack. These pointers need memory to store their values, and this memory comes from the Stack segment. But the address that they are pointing to is inside the Heap segment. This is the theme when working with Heap memories. You have local pointers which are allocated from the Stack segment, but they are actually pointing to a region allocated from the Heap segment. We show more of these in the following chapter.

Note that the `ptr` pointer inside the `fill` function is also allocated from the Stack but it is in a different scope, and it is different from the `ptr` pointer declared in the `main` function.

When it comes to Heap memory, the program, or actually the programmer, is responsible for memory allocation. The program is also responsible for deallocation of the memory when it is not needed. Having a piece of allocated Heap memory that is not *reachable* is considered a *memory leak*. By not being reachable, we mean that there is no pointer that can be used to address that region.

Memory leaks are fatal to programs because having an incremental memory leak will eventually use up the whole allowed memory space, and this can kill the process. That's why the program is calling the `free` function before returning from the `main` function. The call to the `free` function will deallocate the acquired Heap memory block, and the program shouldn't use those Heap addresses anymore.

More on Stack and Heap segments will come in the next chapter.

# Summary

Our initial goal in this chapter was to provide an overview of the memory structure of a process in a Unix-like operating system. As we have covered a lot in this chapter, take a minute to read through what we've been through, as you should now feel comfortable in understanding what we have accomplished:

*   We described the dynamic memory structure of a running process as well as the static memory structure of an executable object file.
*   We observed that the static memory layout is located inside the executable object file and it is broken into pieces which are called segments. We found out that the Text, Data, and BSS segments are part of the static memory layout.
*   We saw that the Text segment or Code segment is used to store the machine-level instructions meant to be executed when a new process is spawned out of the current executable object file.
*   We saw that the BSS segment is used to store global variables that are either uninitialized or set to zero.
*   We explained that the Data segment is used to store initialized global variables.
*   We used the `size` and `objdump` commands to probe the internals of object files. We can also use object file dumpers like `readelf` in order to find these segments inside an object file.
*   We probed the dynamic memory layout of a process. We saw that all segments are copied from the static memory layout into the dynamic memory of the process. However, there are two new segments in the dynamic memory layout; the Stack segment, and the Heap segment.
*   We explained that the Stack segment is the default memory region used for allocations.
*   We learned that the local variables are always allocated on top of the Stack region.
*   We also observed that the secret behind the function calls lies within the Stack segment and the way it works.
*   We saw that we have to use a specific API, or a set of functions, in order to allocate and deallocate Heap memory regions. This API is provided by the C standard library.
*   We discussed memory leakage and how it can happen regarding Heap memory regions.

The next chapter is about the Stack and Heap segments specifically. It will use the topics we have covered within this chapter, and it will add more to those foundations. More examples will be given, and new probing tools will be introduced; this will complete our discussion regarding memory management in C.