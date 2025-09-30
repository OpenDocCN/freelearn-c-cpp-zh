# Chapter 03

# Object Files

This chapter details the various products that a C/C++ project can have. Possible products include relocatable object files, executable object files, static libraries, and shared object files. However, relocatable object files are considered to be temporary products and they act as ingredients for making other types of products that are final.

It seems that today in C, it's crucial to have further discussion about the various types of object files and their internal structures. The majority of C books only talk about the C syntax and the language itself; but, in real-world you need more in-depth knowledge to be a successful C programmer.

When you are creating software, it is not just about the development and the programming language. In fact, it is about the whole process: writing the code, compilation, optimization, producing correct products, and further subsequent steps, in order to run and maintain those products on the target platforms.

You should be knowledgeable about these intermediate steps, to the extent that you are able to solve any issues you might encounter. This is even more serious regarding embedded development, as the hardware architectures and the instruction sets can be challenging and atypical.

This chapter is divided into the following sections:

1.  **Application binary interface**: Here, we are first going to talk about the **Application Binary Interface** (**ABI**) and its importance.
2.  **Object file formats**: In this section, we talk about various object file formats that exist today or they have become obsolete over the years. We also introduce ELF as the most used object file format in Unix-like systems.
3.  **Relocatable object files**: Here we discuss relocatable object files and the very first products of a C project. We take a look inside ELF relocatable object files to see what we can find there.
4.  **Executable object files**: As part of this section, we talk about the executable object files. We also explain how they are created from a number of relocatable object files. We discuss the differences between ELF relocatable and executable object files in terms of their internal structure.
5.  **Static library**: In this section, we talk about static libraries and how we can create them. We also demonstrate how to write a program and use already built static libraries.

1.  **Dynamic library**: Here we talk about shared object files. We demonstrate how to create them out of a number of relocatable object files and how to use them in a program. We also briefly talk about the internal structure of an ELF shared object file.

Our discussions in this chapter will be mostly themed around Unix-like systems, but we will discuss some differences in other operating systems like Microsoft Windows.

**Note**:

Before moving on to read this chapter, you need to be familiar with the basic ideas and steps required for building a C project. You need to know what a translation unit is and how linking is different from compilation. Please read the previous chapter before moving on with this one.

Let's begin the chapter by talking about ABI.

# Application binary interface (ABI)

As you may already know, every library or framework, regardless of the technologies or the programming language used, exposes a set of certain functionalities, which is known as its **Application Programming Interface** (**API**). If a library is supposed to be used by another code, then the consumer code should use the provided API. To be clear, nothing other than the API should be used in order to use a library because it is the public interface of the library and everything else is seen as a black box, hence cannot be used.

Now suppose after some time, the library's API undergoes some modifications. In order for the consumer code to continue using the newer versions of the library, the code must adapt itself to the new API; otherwise, it won't be able to use it anymore. The consumer code could stick to a certain version of the library (maybe an old one) and ignore the newer versions, but let's assume that there is a desire to upgrade to the latest version of the library.

To put it simply, an API is like a convention (or standard) accepted between two software components to serve or use each other. An ABI is pretty similar to API, but at a different level. While the API guarantees the compatibility of two software components to continue their functional cooperation, the ABI guarantees that two programs are compatible at the level of their machine-level instructions, together with their corresponding object files.

For instance, a program cannot use a dynamic or static library that has a different ABI. Perhaps worse than that, an executable file (which is, in fact, an object file) cannot be run on a system supporting a different ABI than the one that the executable file was built for. A number of vital and obvious system functionalities, such as *dynamic linking*, *loading an executable*, and *function calling convention*, should be done precisely according to an agreed upon ABI.

An ABI will typically cover the following things:

*   The instruction set of the target architecture, which includes the processor instructions, memory layout, endianness, registers, and so on.
*   Existing data types, their sizes, and the alignment policy.
*   The function calling convention describes how functions should be called. For example, subjects like the structure of the *stack frame* and the pushing order of the arguments are part of it.
*   Defining how *system calls* should be called in a Unix-like system.
*   Used *object file format*, which we will explain in the following section, for having *relocatable, executable*, and *shared object files*.
*   Regarding object files produced by a C++ compiler, the *name mangling*, *virtual table* layout, is part of the ABI.

The *System V ABI* is the most widely used ABI standard among Unix-like operating systems like Linux and the BSD systems. **Executable and Linking Format** (**ELF**) is the standard object file format used in the System V ABI.

**Note**:

The following link is the System V ABI for AMD 64-bit architectur[e: https://www.uclibc.org/docs/psABI-x86_64.](https://www.uclibc.org/docs/psABI-x86_64.pdf)pdf. You can go through the list of contents and see the areas it covers.

In the following section, we will discuss the object file formats, particularly ELF.

# Object file formats

As we explained in the previous chapter, *Chapter 2*, *Compilation and Linking*, on a platform, object files have their own specific format for storing machine-level instructions. Note that this is about the structure of object files and this is different from the fact that each architecture has its own instruction set. As we know from the previous discussion, these two variations are different parts of the ABI in a platform; the object file format and the architecture's instruction set.

In this section, we are going to have a brief look into some widely known object file formats. To start with, let's look at some object file formats used in various operating systems:

*   **ELF** used by Linux and many other Unix-like operating systems
*   **Mach-O** used in OS X (macOS and iOS) systems
*   **PE** used in Microsoft Windows

To give some history and context about the current and past object file formats, we can say that all object file formats that exist today are successors to the old `a.out` object file format. It was designed for early versions of Unix.

The term **a.out** stands for **assembler output**. Despite the fact that the file format is obsolete today, the name is still used as the default filename for the executable files produced by most linkers. You should remember seeing `a.out` in a number of examples in the first chapter of the book.

However, the `a.out` format was soon replaced by **COFF** or the **Common Object File Format**. COFF is the basis for ELF – the object format that we use in most Unix-like systems. Apple also replaced `a.out` with Mach-O as part of OS/X. Windows uses the **PE** or **Portable Execution** file format for its object files, which is based on COFF.

**Note**:

A deeper history of object file formats can be found here: [https://en.wikipedia.org/wiki/COFF#History](https://en.wikipedia.org/wiki/COFF#History). Knowing about the history of a specific topic will help you to get a better understanding of its evolution path and current and past characteristics.

As you can see, all of today's major object file formats are based on the historic object file format `a.out`, and then COFF, and in many ways share the same ancestry.

ELF is the standard object file format used in Linux and most Unix-like operating systems. In fact, ELF is the object file format used as part of the System V ABI, heavily employed in most Unix systems. Today, it is the most widely accepted object file format used by operating systems.

ELF is the standard binary file format for operating systems including, but not limited to:

*   Linux
*   FreeBSD
*   NetBSD
*   Solaris

This means that as long as the architecture beneath them remains the same, an ELF object file created for one of these operating systems can be run and used in others. ELF, like all other *file formats*, has a structure that we will describe briefly in the upcoming sections.

**Note**:

More information about ELF and its details can be f[ound here: https://www.uclibc.org/docs/psABI](https://www.uclibc.org/docs/psABI-x86_64.pdf)-x86_64.pdf. Note that this link refers to the System V ABI for AMD 64-bits (`amd64`) architecture.

You can also read the HTML version of the System V ABI here: [http://www.sco.com/developers/gabi/2003-12-17/ch4.intro.html](http://www.sco.com/developers/gabi/2003-12-17/ch4.intro.html).

In the upcoming sections, we are going to talk about the temporary and final products of a C project. We start with relocatable object files.

# Relocatable object files

In this section, we are going to talk about relocatable object files. As we explained in the previous chapter, these object files are the output of the assembly step in the C compilation pipeline. These files are considered to be temporary products of a C project, and they are the main ingredients to produce further and final products. For this reason, it would be useful to have a deeper look at them and see what we can find in a relocatable object file.

In a relocatable object file, we can find the following items regarding the compiled translation unit:

*   The machine-level instructions produced for the functions found in the translation unit (code).
*   The values of the initialized global variables declared in the translation unit (data).
*   The *symbol table* containing all the defined and reference symbols found in the translation unit.

These are the key items that can be found in any relocatable object file. Of course, the way that they are put together depends on the object file format, but using proper tools, you should be able to extract these items from a relocatable object file. We are going to do this for an ELF relocatable object file shortly.

But before delving into the example, let's talk about the reason why relocatable object files are named like this. In other words, what does the *relocatable* mean after all? The reason comes from the process that a linker performs in order to put some relocatable object files together and form a bigger object file – an executable object file or a shared object file.

We discuss what can be found in an executable file in the next section, but for now, we should know that the items we find in an executable object file are the sum of all the items found in all the constituent relocatable object files. Let's just talk about machine-level instructions.

The machine-level instructions found in one relocatable object file should be put next to the machine-level instructions coming from another relocatable object file. This means that the instructions should be easily *movable* or *relocatable*. For this to happen, the instructions have no addresses in a relocatable object file, and they obtain their addresses only after the linking step. This is the main reason why we call these object files relocatable. To elaborate more on this, we need to show it in a real example.

*Example 3.1* is about two source files, one containing the definitions of two functions, `max` and `max_3`, and the other source file containing the `main` function using the declared functions `max` and `max_3`. Next, you can see the content of the first source file:

```cpp
int max(int a, int b) {
  return a > b ? a : b;
}
int max_3(int a, int b, int c) {
  int temp = max(a, b);
  return c > temp ? c : temp;
}
```

Code Box 3-1 [ExtremeC_examples_chapter3_1_funcs.c]: A source file containing two function definitions

And the second source file looks like the following code box:

```cpp
int max(int, int);
int max_3(int, int, int);
int a = 5;
int b = 10;
int main(int argc, char** argv) {
  int m1 = max(a, b);
  int m2 = max_3(5, 8, -1);
  return 0;
}
```

Code Box 3-2 [ExtremeC_examples_chapter3_1.c]: The main function using the already declared functions. Definitions are put in a separate source file.

Let's produce the relocatable object files for the preceding source files. This way, we can investigate the content and that which we explained before. Note that, since we are compiling these sources on a Linux machine, we expect to see ELF object files as the result:

```cpp
$ gcc -c ExtremeC_examples_chapter3_1_funcs.c  -o funcs.o
$ gcc -c ExtremeC_examples_chapter3_1.c -o main.o
$
```

Shell Box 3-1: Compiling source files to their corresponding relocatable object files

Both `funcs.o` and `main.o` are relocatable ELF object files. In an ELF object file, the items described to be in a relocatable object file are put into a number of sections. In order to see the present sections in the preceding relocatable object files, we can use the `readelf` utility as follows:

```cpp
$ readelf -hSl funcs.o
[7/7]
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              REL (Relocatable file)
  Machine:                           Advanced Micro Devices X86-64
...
  Number of section headers:         12
  Section header string table index: 11
Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .text             PROGBITS         0000000000000000  00000040
       0000000000000045  0000000000000000  AX       0     0     1
...
  [ 3] .data             PROGBITS         0000000000000000  00000085
       0000000000000000  0000000000000000  WA       0     0     1
  [ 4] .bss              NOBITS           0000000000000000  00000085
       0000000000000000  0000000000000000  WA       0     0     1
...
  [ 9] .symtab           SYMTAB           0000000000000000  00000110
       00000000000000f0  0000000000000018          10     8     8
  [10] .strtab           STRTAB           0000000000000000  00000200
       0000000000000030  0000000000000000           0     0     1
  [11] .shstrtab         STRTAB           0000000000000000  00000278
       0000000000000059  0000000000000000           0     0     1
...
$
```

Shell Box 3-2: The ELF content of the funcs.o object file

As you can see in the preceding shell box, the relocatable object file has 11 sections. The sections in bold font are the sections that we have introduced as items existing in an object file. The `.text` section contains all the machine-level instructions for the translation unit. The `.data` and `.bss` sections contain the values for initialized global variables, and the number of bytes required for uninitialized global variables respectively. The `.symtab` section contains the symbol table.

Note that, the sections existing in both preceding object files are the same, but their content is different. Therefore, we don't show the sections for the other relocatable object file.

As we mentioned before, one of the sections in an ELF object file contains the symbol table. In the previous chapter, we had a thorough discussion about the symbol table and its entries. We described how it is being used by the linker to produce executable and shared object files. Here, we want to draw your attention to something about the symbol table that we didn't discuss in the previous chapter. This would be in accordance with our explanation on why relocatable object files are named in this manner.

Let's dump the symbol table for `funcs.o`. In the previous chapter, we used `objdump` but now, we are going to use `readelf` to do so:

```cpp
$ readelf -s funcs.o
Symbol table '.symtab' contains 10 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
...
     6: 0000000000000000     0 SECTION LOCAL  DEFAULT    7
     7: 0000000000000000     0 SECTION LOCAL  DEFAULT    5
     8: 0000000000000000    22 FUNC    GLOBAL DEFAULT    1 max
     9: 0000000000000016    47 FUNC    GLOBAL DEFAULT    1 max_3
$
```

Shell Box 3-3: The symbol table of the funcs.o object file

As you can see in the `Value` column, the address assigned to `max` is `0` and the address assigned to `max_3` is `22` (hexadecimal `16`). This means that the instructions related to these symbols are adjacent and their addresses start from 0\. These symbols, and their corresponding machine-level instructions, are ready to be relocated to other places in the final executable. Let's look at the symbol table of `main.o`:

```cpp
$ readelf -s main.o
Symbol table '.symtab' contains 14 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
...
     8: 0000000000000000     4 OBJECT  GLOBAL DEFAULT    3 a
     9: 0000000000000004     4 OBJECT  GLOBAL DEFAULT    3 b
    10: 0000000000000000    69 FUNC    GLOBAL DEFAULT    1 main
    11: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND _GLOBAL_OFFSET_TABLE_
    12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND max
    13: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND max_3
$
```

Shell Box 3-4: The symbol table of the main.o object file

As you can see, the symbols associated with global variables `a` and `b`, as well as the symbol for the `main` function are put at addresses that don't seem be the final addresses that they should be placed at. This is a sign of being a relocatable object file. As we have said before, the symbols in a relocatable object files don't have any final and absolute addresses and their addresses will be determined as part of the linking step.

In the following section, we continue to produce an executable file from the preceding relocatable object files. You will see that the symbol table is different.

# Executable Object Files

Now, it's time to talk about executable object files. You should know by now that executable object file is one of the final products of a C project. Like relocatable object files, they have the same items in the:; the machine-level instructions, the values for initialized global variables, and the symbol tabl;t however, the arrangement can be different. We can show this regarding the ELF executable object files since it would be easy to generate them and study their internal structure.

In order to produce an executable ELF object file, we continue with *example 3.1*. In the previous section, we generated relocatable object files for the two sources existing in the example, and in this section, we are going to link them to form an executable file.

The following commands do that for you, as explained in the previous chapter:

```cpp
$ gcc funcs.o main.o -o ex3_1.out
$ 
```

Shell Box 3-5: Linking previously built relocatable object files in example 3.1

In the previous section, we spoke about sections being present in an ELF object file. We should say that more sections exist in an ELF executable object file, but together with some segments. Every ELF executable object file, and as you will see later in this chapter, every ELF shared object file, has a number of *segments* in addition to sections. Each segment consists of a number of sections (zero or more), and the sections are put into segments based on their content.

For example, all sections containing machine-level instructions go into the same segment. You will see in *Chapter 4*, *Process Memory Structure*, that these segments nicely map to static *memory segments* found in the memory layout of a running process.

Let's look at the contents of an executable file and meet these segments. Similarly, to relocatable object files, we can use the same command to show the sectios, and the segments found in an executable ELF object file.

```cpp
$ readelf -hSl ex3_1.out
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file) 
  Machine:                           Advanced Micro Devices X86-64 
  Version:                           0x1
  Entry point address:               0x4f0
  Start of program headers:          64 (bytes into file)
  Start of section headers:          6576 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         9
  Size of section headers:           64 (bytes)
  Number of section headers:         28
  Section header string table index: 27
Section Headers: 
  [Nr] Name              Type             Address           Offset 
       Size              EntSize          Flags  Link  Info  Align 
  [ 0]                   NULL             0000000000000000  00000000 
      0000000000000000  0000000000000000           0     0     0 
  [ 1] .interp           PROGBITS         0000000000000238  00000238 
       000000000000001c  0000000000000000   A       0     0     1 
  [ 2] .note.ABI-tag     NOTE             0000000000000254  00000254 
       0000000000000020  0000000000000000   A       0     0     4 
  [ 3] .note.gnu.build-i NOTE             0000000000000274  00000274 
       0000000000000024  0000000000000000   A       0     0     4 
... 
  [26] .strtab           STRTAB           0000000000000000  00001678 
       0000000000000239  0000000000000000           0     0     1 
  [27] .shstrtab         STRTAB           0000000000000000  000018b1 
       00000000000000f9  0000000000000000           0     0     1 
Key to Flags: 
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info), 
  L (link order), O (extra OS processing required), G (group), T (TLS), 
  C (compressed), x (unknown), o (OS specific), E (exclude), 
  l (large), p (processor specific) 
Program Headers: 
  Type           Offset             VirtAddr           PhysAddr 
                 FileSiz            MemSiz              Flags  Align 
  PHDR           0x0000000000000040 0x0000000000000040 0x0000000000000040 
                 0x00000000000001f8 0x00000000000001f8  R      0x8 
  INTERP         0x0000000000000238 0x0000000000000238 0x0000000000000238 
                 0x000000000000001c 0x000000000000001c  R      0x1 
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2] 
... 
  GNU_EH_FRAME   0x0000000000000714 0x0000000000000714 0x0000000000000714 
                 0x000000000000004c 0x000000000000004c  R      0x4 
  GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000 
                 0x0000000000000000 0x0000000000000000  RW     0x10 
  GNU_RELRO      0x0000000000000df0 0x0000000000200df0 0x0000000000200df0 
                 0x0000000000000210 0x0000000000000210  R      0x1 
Section to Segment mapping: 
  Segment Sections... 
   00 
   01     .interp 
   02     .interp .note.ABI-tag .note.gnu.build-id .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .init .plt .plt.got .text .fini .rodata .eh_frame_hdr .eh_frame 
   03     .init_array .fini_array .dynamic .got .data .bss 
   04     .dynamic 
   05     .note.ABI-tag .note.gnu.build-id 
   06     .eh_frame_hdr 
   07 
   08     .init_array .fini_array .dynamic .got 
$
```

Shell Box 3-6: The ELF content of ex3_1.out executable object file

There are multiple notes about the above output:

*   We can see that the type of object file from the ELF point of vew, is a shared object file. In other words, in ELF, an executable object file is a shared object file that has some specific segments like `INTERP`. This segment (actually the `.interp` section which is referred to by this segment) is used by the loader program to load and execute the executable object file.
*   We have made four segments bold. The first one refers to the `INTERP` segment which is explained in the previous bullet point. The second one is the `TEXT` segment. It contains all the section having machine-level instructions. The third one is the `DATA` segment that contains all the values that should be used to initialize the global variables and other early structures. The fourth segment refers to the section that *dynamic linking* related information can be found. For instance, the shared object files that need to be loaded as part of the execution.
*   As you see, we've got more sections in comparison to a relocatable shared object, probably filled with data required to load and execute the object file.

As we explained in the previous section, the symbols found in the symbol table of a relocatable object file do not have any absolute and determined addresses. That's because the sections containing machine-level instructions are not linked yet.

In a deeper sense, linking a number of relocatable object files is actually to collect all similar sections from the given relocatable object files and put them together to form a bigger section, and finally put the resulting section into the output executable or the shared object file. Therefore, only after this step, the symbols can be finalized and obtain the addresses that are not going to change. In executable object files, the addresses are absolute, while in shared object files, the relative addresses are absolute. We will discuss this more in the section dedicated to dynamic libraries.

Let's look at the symbol table found in the executable file `ex3_1.out`. Note that the symbol table has many entries and that's why the output is not fully shown in the following shell box:

```cpp
$ readelf -s ex3_1.out
Symbol table '.dynsym' contains 6 entries: 
   Num:    Value          Size Type    Bind   Vis      Ndx Name 
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
... 
     5: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@GLIBC_2.2.5 (2) 
Symbol table '.symtab' contains 66 entries: 
   Num:    Value          Size Type    Bind   Vis      Ndx Name 
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
... 
    45: 0000000000201000     0 NOTYPE  WEAK   DEFAULT   22 data_start 
    46: 0000000000000610    47 FUNC    GLOBAL DEFAULT   13 max_3 
    47: 0000000000201014     4 OBJECT  GLOBAL DEFAULT   22 b 
    48: 0000000000201018     0 NOTYPE  GLOBAL DEFAULT   22 _edata 
    49: 0000000000000704     0 FUNC    GLOBAL DEFAULT   14 _fini 
    50: 00000000000005fa    22 FUNC    GLOBAL DEFAULT   13 max 
    51: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@@GLIBC_ 
... 
    64: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@@GLIBC_2.2 
    65: 00000000000004b8     0 FUNC    GLOBAL DEFAULT   10 _init 
$
```

Shell Box 3-7: The symbol tables found in the ex3_1.out executable object file

As you see in the preceding shell box, we have two different symbol tables in an executable object file. The first one, `.dynsym`, contains the symbols that should be resolved when loading the executable, but the second symbol table, `.symtab`, contains all the resolved symbols together with unresolved symbols brought from the dynamic symbol table. In other words, the symbol table contains the unresolved symbols from the dynamic table as well.

As you see, the resolved symbols in the symbol table have absolute corresponding addresses that they have obtained after the linking step. The addresses for `max` and `max_3` symbols are shown in bold font.

In this section, we took a brief look into the executable object file. In the next section, we are going to talk about static libraries.

# Static libraries

As we have explained before, a static library is one of the possible products of a C project. In this section, we are going to talk about static libraries and the way they are created and used. We will then continue this discussion by introducing dynamic libraries in the next section.

A static library is simply a Unix archive made from the relocatable object files. Such a library is usually linked together with other object files to form an executable object file.

Note that a static library itself is not considered as an object file, rather it is a container for them. In other words, static libraries are not ELF files in Linux systems, nor are they Mach-O files in macOS systems. They are simply archived files that have been created by the Unix `ar` utility.

When a linker is about to use a static library in the linking step, it first tries to extract the relocatable object files from it, then it starts to look up and resolve the undefined symbols that may be found in some of them.

Now, it's time to create a static library for a project with multiple source files. The first step is to create some relocatable object files. Once you have compiled all of the source files in a C/C++ project, you can use the Unix archiving tool, `ar`, to create the static library's archive file.

In Unix systems, static libraries are usually named according to an accepted and widely used convention. The name starts with `lib`, and it ends with the `.a` extension. This can be different for other operating systems; for instance, in Microsoft Windows, static libraries carry the `.lib` extension.

Suppose that, in an imaginary C project, you have the source files `aa.c`, `bb.c`, all the way up to `zz.c`. In order to produce the relocatable object files, you will need to compile the source files in a similar manner to how we use the commands next. Note that the compilation process has been thoroughly explained in the previous chapter:

```cpp
$ gcc -c aa.c -o aa.o
$ gcc -c bb.c -o bb.o
.
.
.
$ gcc -c zz.c -o zz.o
$
```

Shell Box 3-8: Compiling a number of sources to their corresponding relocatable object files

By running the preceding commands, we will get all the required relocatable object files. Note that this can take a considerable amount of time if the project is big and contains thousands of source files. Of course, having a powerful build machine, together with running the compilation jobs in parallel, can reduce the build time significantly.

When it comes to creating a static library file, we simply need to run the following command:

```cpp
$ ar crs libexample.a aa.o bb.o ... zz.o
$
```

Shell Box 3-9: The general recipe for making a static library out of a number of relocatable object files

As a result, `libexample.a` is created, which contains all of the preceding relocatable object files as a single archive. Explaining the `crs` option passed to `ar` would be out of the scope of this chapter, but in the following link, you can read [about its meaning: https://stackoverflow.com/questions/29714300/what-does-th](https://stackoverflow.com/questions/29714300/what-does-the-rcs-option-in-ar-do)e-rcs-option-in-ar-do.

**Note**:

The `ar` command does not necessarily create a *compressed* archive file. It is only used to put files together to form a single file that is an archive of all those files. The tool `ar` is general purpose, and you can use it to put any kind of files together and create your own archive out of them.

Now that we know how to create a static library, we are going to create a real one as part of *example 3.2*.

First, we are going to presume that *example 3.2* is a C project about geometry. The example consists of three source files and one header file. The purpose of the library is to define a selection of geometry related functions that can be used in other applications.

To do this, we need to create a static library file named `libgeometry.a` out of the three source files. By having the static library, we can use the header file and the static library file together in order to write another program that will use the geometry functions defined in the library.

The following code boxes are the contents of the source and header files. The first file, `ExtremeC_examples_chapter3_2_geometry.h`, contains all of the declarations that need to be exported from our geometry library. These declarations will be used by future applications that are going to use the library.

**Note**:

All the commands provided for creating object files are run and tested on Linux. Some modifications might be necessary if you're going to execute them on a different operating system.

We need to take note that future applications *must* be only dependent on the declarations and not the definitions at all. Therefore, firstly, let's look at the declarations of the geometry library:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_3_2_H
#define EXTREME_C_EXAMPLES_CHAPTER_3_2_H
#define PI 3.14159265359
typedef struct {
  double x;
  double y;
} cartesian_pos_2d_t;
typedef struct {
  double length;
  // in degrees
  double theta;
} polar_pos_2d_t;
typedef struct {
  double x;
  double y;
  double z;
} cartesian_pos_3d_t;
typedef struct {
  double length;
  // in degrees
  double theta;
  // in degrees
  double phi;
} polar_pos_3d_t;
double to_radian(double deg);
double to_degree(double rad);
double cos_deg(double deg);
double acos_deg(double deg);
double sin_deg(double deg);
double asin_deg(double deg);
cartesian_pos_2d_t convert_to_2d_cartesian_pos(
        const polar_pos_2d_t* polar_pos);
polar_pos_2d_t convert_to_2d_polar_pos(
        const cartesian_pos_2d_t* cartesian_pos);
cartesian_pos_3d_t convert_to_3d_cartesian_pos(
        const polar_pos_3d_t* polar_pos);
polar_pos_3d_t convert_to_3d_polar_pos(
        const cartesian_pos_3d_t* cartesian_pos);
#endif
```

Code Box 3-3 [ExtremeC_examples_chapter3_2_geometry.h]: The header file of example 3.2

The second file, which is a source file, contains the definitions of the trigonometry functions, the first six functions declared in the preceding header file:

```cpp
#include <math.h>
// We need to include the header file since
// we want to use the macro PI
#include "ExtremeC_examples_chapter3_2_geometry.h"
double to_radian(double deg) {
  return (PI * deg) / 180;
}
double to_degree(double rad) {
  return (180 * rad) / PI;
}
double cos_deg(double deg) {
  return cos(to_radian(deg));
}
double acos_deg(double deg) {
  return acos(to_radian(deg));
}
double sin_deg(double deg) {
  return sin(to_radian(deg));
}
double asin_deg(double deg) {
  return asin(to_radian(deg));
}
```

Code Box 3-4 [ExtremeC_examples_chapter3_2_trigon.c]: The source file containing the definitions of the trigonometry functions

Note that it is not necessary for sources to include the header file unless they are going to use a declaration like `PI` or `to_degree`, which is declared in the header file.

The third file, which is a source file again, contains the definitions of all 2D Geometry functions:

```cpp
#include <math.h>
// We need to include the header file since we want
// to use the types polar_pos_2d_t, cartesian_pos_2d_t,
// etc and the trigonometry functions implemented in
// another source file.
#include "ExtremeC_examples_chapter3_2_geometry.h"
cartesian_pos_2d_t convert_to_2d_cartesian_pos(
        const polar_pos_2d_t* polar_pos) {
  cartesian_pos_2d_t result;
  result.x = polar_pos->length * cos_deg(polar_pos->theta);
  result.y = polar_pos->length * sin_deg(polar_pos->theta);
  return result;
}
polar_pos_2d_t convert_to_2d_polar_pos(
        const cartesian_pos_2d_t* cartesian_pos) {
  polar_pos_2d_t result;
  result.length = sqrt(cartesian_pos->x * cartesian_pos->x +
    cartesian_pos->y * cartesian_pos->y);
  result.theta =
      to_degree(atan(cartesian_pos->y / cartesian_pos->x));
  return result;
}
```

Code Box 3-5 [ExtremeC_examples_chapter3_2_2d.c]: The source file containing the definitions of the 2D functions

And finally, the fourth file that contains the definitions of 3D Geometry functions:

```cpp
#include <math.h>
// We need to include the header file since we want to
// use the types polar_pos_3d_t, cartesian_pos_3d_t,
// etc and the trigonometry functions implemented in
// another source file.
#include "ExtremeC_examples_chapter3_2_geometry.h"
cartesian_pos_3d_t convert_to_3d_cartesian_pos(
        const polar_pos_3d_t* polar_pos) {
  cartesian_pos_3d_t result;
  result.x = polar_pos->length *
      sin_deg(polar_pos->theta) * cos_deg(polar_pos->phi);
  result.y = polar_pos->length *
      sin_deg(polar_pos->theta) * sin_deg(polar_pos->phi);
  result.z = polar_pos->length * cos_deg(polar_pos->theta);
  return result;
}
polar_pos_3d_t convert_to_3d_polar_pos(
        const cartesian_pos_3d_t* cartesian_pos) {
  polar_pos_3d_t result;
  result.length = sqrt(cartesian_pos->x * cartesian_pos->x +
    cartesian_pos->y * cartesian_pos->y +
    cartesian_pos->z * cartesian_pos->z);
  result.theta =
      to_degree(acos(cartesian_pos->z / result.length));
  result.phi =
      to_degree(atan(cartesian_pos->y / cartesian_pos->x));
  return result;
}
```

Code Box 3-6 [ExtremeC_examples_chapter3_2_3d.c]: The source file containing the definitions of the 3D functions

Now we'll create the static library file. To do this, firstly we need to compile the preceding sources to their corresponding relocatable object files. You need to note that we cannot link these object files to create an executable file as there is no `main` function in any of the preceding source files. Therefore, we can either keep them as relocatable object files or archive them to form a static library. We have another option to create a shared object file out of them, but we'll wait until the next section to look at this.

In this section, we have chosen to archive them in order to create a static library file. The following commands will do the compilation on a Linux system:

```cpp
$ gcc -c ExtremeC_examples_chapter3_2_trigon.c -o trigon.o
$ gcc -c ExtremeC_examples_chapter3_2_2d.c -o 2d.o
$ gcc -c ExtremeC_examples_chapter3_2_3d.c -o 3d.o
$
```

Shell Box 3-10: Compiling source files to their corresponding relocatable object files

When it comes to archiving these object files into a static library file, we need to run the following command:

```cpp
$ ar crs libgeometry.a trigon.o 2d.o 3d.o
$ mkdir -p /opt/geometry
$ mv libgeometry.a /opt/geometry
$
```

Shell Box 3-11: Creating the static library file out of the relocatable object files

As we can see, the file `libgeometry.a` has been created. As you see, we have moved the library file to the `/opt/geometry` directory to be easily locatable by any other program. Again, using the `ar` command, and via passing the `t` option, we can see the content of the archive file:

```cpp
$ ar t /opt/geometry/libgeometry.a
trigon.o
2d.o
3d.o
$
```

Shell Box 3-12: Listing the content of the static library file

As is clear from the preceding shell box, the static library file contains three relocatable object files as we intended. The next step is to use the static library file.

Now that we have created a static library for our geometry example, *example 3.2*, we are going to use it in a new application. When using a C library, we need to have access to the declarations that are exposed by the library together with its static library file. The declarations are considered as the *public interface* of the library, or more commonly, the API of the library.

We need declarations in the compile stage, when the compiler needs to know about the existence of types, function signatures, and so on. Header files serve this purpose. Other details such as type sizes and function addresses are needed at later stages; linking and loading.

As we said before, we usually find a C API (an API exposed by a C library) as a group of header files. Therefore, the header file from *example 3.2*, and the created static library file `libgeometry.a`, are enough for us to write a new program that uses our geometry library.

When it comes to using the static library, we need to write a new source file that includes the library's API and make use of its functions. We write the new code as a new example, *example 3.3*. The following code is the source that we have written for *example 3.3*:

```cpp
#include <stdio.h>
#include "ExtremeC_examples_chapter3_2_geometry.h"
int main(int argc, char** argv) {
  cartesian_pos_2d_t cartesian_pos;
  cartesian_pos.x = 100;
  cartesian_pos.y = 200;
  polar_pos_2d_t polar_pos =
      convert_to_2d_polar_pos(&cartesian_pos);
  printf("Polar Position: Length: %f, Theta: %f (deg)\n",
    polar_pos.length, polar_pos.theta);
  return 0;
}
```

Code Box 3-7 [ExtremeC_examples_chapter3_3.c]: The main function testing some of the geometry functions

As you can see, *example 3.3* has included the header file from *example 3.2*. It has done this because it needs the declarations of the functions that it is going to use.

We now need to compile the preceding source file to create its corresponding relocatable object file in a Linux system:

```cpp
$ gcc -c ExtremeC_examples_chapter3_3.c -o main.o
$
```

Shell Box 3-13: Compiling example 3.3

After we have done that, we need to link it with the static library that we created for *example 3.2*. In this case, we assume that the file `libgeometry.a` is located in the `/opt/geometry` directory, as we had in *Shell Box 3-11*. The following command will complete the build by performing the linking step and creating the executable object file, *ex3_3.out*:

```cpp
$ gcc main.o -L/opt/geometry -lgeometry -lm -o ex3_3.out
$
```

Shell Box 3-14: Linking with the static library created as part of example 3.2

To explain the preceding command, we are going to explain each passing option separately:

*   `-L/opt/geometry` tells `gcc` to consider the directory `/opt/geometry` as one of the various locations in which static and shared libraries could be found. There are well-known paths like `/usr/lib` or `/usr/local/lib` in which the linker searches for library files by default. If you do not specify the `-L` option, the linker only searches its default paths.
*   `-lgeometry` tells `gcc` to look for the file `libgeometry.a` or `libgeometry.so`. A file ending with `.so` is a shared object file, which we explain in the next section. Note the convention used. If you pass the option `-lxyz` for instance, the linker will search for the file `libxyz.a` or `libxyz.so` in the default and specified directories. If the file is not found, the linker stops and generates an error.
*   `-lm` tells `gcc` to look for another library named `libm.a` or `libm.so`. This library keeps the definitions of mathematical functions in *glibc*. We need it for the `cos`, `sin`, and `acos` functions. Note that we are building *example 3.3* on a Linux machine, which uses *glibc* as its default C library's implementation. In macOS and possibly some other Unix-like systems, you don't need to specify this option.
*   `-o ex3_3.out` tells `gcc` that the output executable file should be named `ex3_3.out`.

After running the preceding command, if everything goes smoothly, you will have an executable binary file that contains all the relocatable object files found in the static library `libgeometry.a` plus `main.o`.

Note that there will not be any dependency on the existence of the static library file after linking, as everything is *embedded* inside the executable file itself. In other words, the final executable file can be run on its own without needing the static library to be present.

However, executable files produced from the linkage of many static libraries usually have huge sizes. The more static libraries and the more relocatable object files inside them, the bigger the size of the final executable. Sometimes it can go up to several hundred megabytes or even a few gigabytes.

It is a trade-off between the size of the binary and the dependencies it might have. You can have a smaller binary, but by using shared libraries. It means that the final binary is not complete and cannot be run if the external shared libraries do not exist or cannot be found. We talk more about this in the upcoming sections.

In this section, we described what static libraries are and how they should be created and used. We also demonstrated how another program can use the exposed API and get linked to an existing static library. In the following section, we are going to talk about dynamic libraries and how to produce a shared object file (dynamic library) from sources in *example 3.2*, instead of using a static library.

# Dynamic libraries

Dynamic libraries, or shared libraries, are another way to produce libraries for reuse. As their name implies, unlike the static libraries, dynamic libraries are not part of the final executable itself. Instead, they should be loaded and brought in while loading a process for execution.

Since static libraries are part of the executable, the linker puts everything found in the given relocatable files into the final executable file. In other words, the linker detects the undefined symbols, and required definitions, and tries to find them in the given relocatable object files, then puts them all in the output executable file.

The final product is only produced when every undefined symbol is found. From a unique perspective, we detect all dependencies and resolve them at linking time. Regarding dynamic libraries, it is possible to have undefined symbols that are not resolved at linking time. These symbols are searched for when the executable product is about to be loaded and begin the execution.

In other words, a different kind of linking step is needed when you have undefined dynamic symbols. A *dynamic linker*, or simply the *loader*, usually does the linking while loading an executable file and preparing it to be run as a process.

Since the undefined dynamic symbols are not found in the executable file, they should be found somewhere else. These symbols should be loaded from shared object files. These files are sister files to static library files. While the static library files have a `.a` extension in their names, the shared object files carry the `.so` extension in most Unix-like systems. In macOS, they have the `.dylib` extension.

When loading a process and about to be launched, a shared object file will be loaded and mapped to a memory region accessible by the process. This procedure is done by a dynamic linker (or loader), which loads and executes an executable file.

Like we said in the section dedicated to executable object files, both ELF executable and shared object files have segments in their ELF structure. Each segment has zero or more sections in them. There are two main differences between an ELF executable object file and an ELF shared object file. Firstly, the symbols have relative absolute addresses that allow them to be loaded as part of many processes at the same time.

This means that while the address of each instruction is different in any process, the distance between two instructions remains fixed. In other words, the addresses are fixed relative to an offset. This is because the relocatable object files are *position independent*. We talk more about this in the last section of this chapter.

For instance, if two instructions are located at addresses 100 and 200 in a process, in another process they may be at 140 and 240, and in another one they could be at 323 and 423\. The related addresses are absolute, but the actual addresses can change. These two instructions will always be 100 addresses apart from each other.

The second difference is that some segments related to loading an ELF executable object file are not present in shared object files. This effectively means that shared object files cannot be executed.

Before giving more details on how a shared object is accessed from different processes, we need to show an example of how they are created and used. Therefore, we are going to create dynamic libraries for the same geometry library, *example 3.2*, that we worked on in the previous section.

In the previous section we created a static library for the geometry library. In this section, we want to compile the sources again in order to create a shared object file out of them. The following commands show you how to compile the three sources into their corresponding relocatable object files, with just one difference in comparison to what we did for *example 3.2*. In the following commands, note the `-fPIC` option that is passed to `gcc`:

```cpp
$ gcc -c ExtremeC_examples_chapter3_2_2d.c -fPIC -o 2d.o
$ gcc -c ExtremeC_examples_chapter3_2_3d.c -fPIC -o 3d.o
$ gcc -c ExtremeC_examples_chapter3_2_trigon.c -fPIC -o trigon.o
$
```

Shell Box 3-15: Compiling the sources of example 3.2 to corresponding position-independent relocatable object files

Looking at the commands, you can see that we have passed an extra option,`-fPIC`, to `gcc` while compiling the sources. This option is *mandatory* if you are going to create a shared object file out of some relocatable object files. **PIC** stands for **position independent code**. As we explained before, if a relocatable object file is position independent, it simply means that the instructions within it don't have fixed addresses. Instead, they have relative addresses; hence they can obtain different addresses in different processes. This is a requirement because of the way we use shared object files.

There is no guarantee that the loader program will load a shared object file at the same address in different processes. In fact, the loader creates memory mappings to the shared object files, and the address ranges for those mappings can be different. If the instruction addresses were absolute, we couldn't load the same shared object file in various processes, and in various memory regions, at the same time.

**Note**:

For more detailed information on how the dynamic loading of programs and shared object files works, you can see the following resources:

*   [https://software.intel.com/sites/default/files/m/a/1/e/dsohowto.pdf](https://software.intel.com/sites/default/files/m/a/1/e/dsohowto.pdf)
*   [https://www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html](https://www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html)

To create shared object files, you need to use the compiler, in this case, `gcc`, again. Unlike a static library file, which is a simple archive, a shared object file is an object file itself. Therefore, they should be created by the same linker program, for instance `ld`, that we used to produce the relocatable object files.

We know that, on most Unix-like systems, `ld` does that. However, it is strongly recommended not to use `ld` directly for linking object files for the reasons we explained in the previous chapter.

The following command shows how you should create a shared object file out of a number of relocatable object files that have been compiled using the `-fPIC` option:

```cpp
$ gcc -shared 2d.o 3d.o trigon.o -o libgeometry.so
$ mkdir -p /opt/geometry
$ mv libgeometry.so /opt/geometry
$
```

Shell Box 3-16: Creating a shared object file out of the relocatable object files

As you can see in the first command, we passed the `-shared` option, to ask `gcc` to create a shared object file out of the relocatable object files. The result is a shared object file named `libgeometry.so`. We have moved the shared object file to `/opt/geometry` to make it easily available to other programs willing to use it. The next step is to compile and link *example 3.3* again.

Previously, we compiled and linked *example 3.3* with the created static library file, `libgeometry.a`. Here, we are going to do the same, but instead, link it with `libgeometry.so`, a dynamic library.

While everything seems to be the same, especially the commands, they are in fact different. This time, we are going to link *example 3.3* with `libgeometry.so` instead of `libgeometry.a`, and more than that, the dynamic library won't get embedded into the final executable, instead it will load the library upon execution. While practicing this, make sure that you have removed the static library file, `libgeometry.a`, from `/opt/geometry` before linking *example 3.3* again:

```cpp
$ rm -fv /opt/geometry/libgeometry.a
$ gcc -c ExtremeC_examples_chapter3_3.c -o main.o
$ gcc main.o -L/opt/geometry-lgeometry -lm -o ex3_3.out
$
```

Shell Box 3-17: Linking example 3.3 against the built shared object file

As we explained before, the option `-lgeometry` tells the compiler to find and use a library, either static or shared, to link it with the rest of the object files. Since we have removed the static library file, the shared object file is picked up. If both the static library and shared object files exist for a defined library, then `gcc` prefers to pick the shared object file and link it with the program.

If you now try to run the executable file `ex3_3.out`, you will most probably face the following error:

```cpp
$ ./ex3_3.out
./ex3_3.out: error while loading shared libraries: libgeometry.so: cannot open shared object file: No such file or directory
$
```

Shell Box 3-18: Trying to run example 3.3

We haven't seen this error so far, because we were using static linkage and a static library. But now, by introducing dynamic libraries, if we are going to run a program that has *dynamic dependencies*, we should provide the required dynamic libraries to have it run. But what has happened and why we've received the error message?

The `ex3_3.out` executable file depends on `libgeometry.so`. That's because some of the definitions it needs can only be found inside that shared object file. We should note that this is not true for the static library `libgeometry.a`. An executable file linked with a static library can be run on its own as a standalone executable, since it has copied everything from the static library file, and therefore, doesn't rely on its existence anymore.

This is not true for the shared object files. We received the error because the program loader (dynamic linker) could not find `libgeometry.so` in its default search paths. Therefore, we need to add `/opt/geometry` to its search paths, so that it finds the `libgeometry.so` file there. To do this, we will update the environment variable `LD_LIBRARY_PATH` to point to the current directory.

The loader will check the value of this environment variable, and it will search the specified paths for the required shared libraries. Note that more than one path can be specified in this environment variable (using the separator colon `:`).

```cpp
$ export LD_LIBRARY_PATH=/opt/geometry 
$ ./ex3_3.out
Polar Position: Length: 223.606798, Theta: 63.434949 (deg)
$
```

Shell Box 3-19: Running example 3.3 by specifying LD_LIBRARY_PATH

This time, the program has successfully been run! This means that the program loader has found the shared object file and the dynamic linker has loaded the required symbols from it successfully.

Note that, in the preceding shell box, we used the `export` command to change the `LD_LIBRARY_PATH`. However, it is common to set the environment variable together with the execution command. You can see this in the following shell box. The result would be the same for both usages:

```cpp
$ LD_LIBRARY_PATH=/opt/geometry ./ex3_3.out
Polar Position: Length: 223.606798, Theta: 63.434949 (deg)
$
```

Shell Box 3-20: Running example 3.3 by specifying LD_LIBRARY_PATH as part of the same command

By linking an executable with several shared object files, as we did before, we tell the system that this executable file needs a number of shared libraries to be found and loaded at runtime. Therefore, before running the executable, the loader searches for those shared object files automatically, and the required symbols are mapped to the proper addresses that are accessible by the process. Only then can the processor begin the execution.

## Manual loading of shared libraries

Shared object files can also be loaded and used in a different way, in which they are not loaded *automatically* by the loader program (dynamic linker). Instead, the programmer will use a set of functions to load a shared object file *manually* before using some symbols (functions) that can be found inside that shared library. There are applications for this manual loading mechanism, and we'll talk about them once we've discussed the example we'll look at in this section.

*Example 3.4* demonstrates how to load a shared object file lazily, or manually, without having it in the linking step. This example borrows the same logic from *example 3.3*, but instead, it loads the shared object file `libgeometry.so` manually inside the program.

Before going through *example 3.4*, we need to produce `libgeometry.so` a bit differently in order to make *example 3.4* work. To do this, we have to use the following command in Linux:

```cpp
$ gcc -shared 2d.o 3d.o trigon.o -lm -o libgeometry.so
$
```

Shell Box 3-21: Linking the geometry shared object file against the standard math library

Looking at the preceding command, you can see a new option, `-lm`, which tells the linker to link the shared object file against the standard math library, `libm.so`. That is done because when we load `libgeometry.so` manually, its dependencies should, somehow, be loaded automatically. If they're not, then we will get errors about the symbols that are required by `libgeometry.so` itself, such as `cos` or `sqrt`. Note that we won't link the final executable file with the math standard library, and it will be resolved automatically by the loader when loading `libgeometry.so`.

Now that we have a linked shared object file, we can proceed to *example 3.4*:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "ExtremeC_examples_chapter3_2_geometry.h"
polar_pos_2d_t (*func_ptr)(cartesian_pos_2d_t*);
int main(int argc, char** argv) {
  void* handle = dlopen ("/opt/geometry/libgeometry.so", RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  func_ptr = dlsym(handle, "convert_to_2d_polar_pos");
  if (!func_ptr)  {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  cartesian_pos_2d_t cartesian_pos;
  cartesian_pos.x = 100;
  cartesian_pos.y = 200;
  polar_pos_2d_t polar_pos = func_ptr(&cartesian_pos);
  printf("Polar Position: Length: %f, Theta: %f (deg)\n",
    polar_pos.length, polar_pos.theta);
  return 0;
}
```

Code Box 3-8 [ExtremeC_examples_chapter3_4.c]: Example 3.4 loading the geometry shared object file manually

Looking at the preceding code, you can see how we have used the functions `dlopen` and `dlsym` to load the shared object file and then find the symbol `convert_to_2d_polar_pos` in it. The function `dlsym` returns a function pointer, which can be used to invoke the target function.

It is worth noting that the preceding code searches for the shared object file in `/opt/geometry`, and if there is no such object file, then an error message is shown. Note that in macOS, the shared object files end in the `.dylib` extension. Therefore, the preceding code should be modified in order to load the file with the correct extension.

The following command compiles the preceding code and runs the executable file:

```cpp
$ gcc ExtremeC_examples_chapter3_4.c -ldl -o ex3_4.out
$ ./ex3_4.out
Polar Position: Length: 223.606798, Theta: 63.434949 (deg)
$
```

Shell Box 3-22: Running example 3.4

As you can see, we did not link the program with the file `libgeometry.so`. We didn't do this because we want to instead load it manually when it is needed. This method is often referred to as the *lazy loading* of shared object files. Yet, despite the name, in certain scenarios, lazy loading the shared object files can be really useful.

One such case is when you have different shared object files for different implementations or versions of the same library. Lazy loading gives you increased freedom to load the desired shared objects according to your own logic and when it is needed, instead of having them automatically loaded at load time, where you have less control over them.

# Summary

This chapter mainly talked about various types of object files, as products of a C/C++ project after building. As part of this chapter, we covered the following points:

*   We discussed the API and ABI, along with their differences.
*   We went through various object file formats and looked at a brief history of them. They all share the same ancestor, but they have changed in their specific paths to become what they are today.
*   We talked about relocatable object files and their internal structure regarding ELF relocatable object files.
*   We discussed executable object files and the differences between them and relocatable object files. We also took a look at an ELF executable object file.
*   We showed static and dynamic symbol tables and how their content can be read using some command-line tools.
*   We discussed static linking and dynamic linking and how various symbol tables are looked up in order to produce the final binary or execute a program.
*   We discussed static library files and the fact that they are just archive files that contain a number of relocatable object files.
*   Shared object files (dynamic libraries) were discussed and we demonstrated how they can be made out of a number of relocatable object files.
*   We explained what position-independent code is and why the relocatable object files participating in the creation of a shared library must be position-independent.

In the following chapter, we will go through the memory structure of a process; another key topic in C/C++ programming. The various memory segments will be described as part of the next chapter and we'll see how we can write code that has no memory issues in it.