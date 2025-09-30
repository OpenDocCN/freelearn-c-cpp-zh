# Chapter 01

# Essential Features

*Extreme C* is a book that will provide you with both the fundamental and the advanced knowledge you need in order to develop and maintain real C applications. Generally, only knowing the syntax of a programming language is not enough to write successful programs with it – and this is of greater importance in C compared to most other languages. So, we're going to cover all of the concepts you need to write great software in C, from simple single process programs to more complex multiple process systems.

This first chapter is primarily concerned with particular features of C that you'll find extremely useful while you're writing C programs. These features are involved in situations you will encounter regularly while writing in C. Although there are a number of great books and tutorials on C programming that explain everything in detail and cover almost all aspects of C syntax, it would be useful to consider some key features here before we go deeper with C.

These features include preprocessor directives, variable pointers, function pointers, and structures. They are common, of course, in today's more modern programming languages, and it's easy to find their counterparts in Java, C#, Python, and so on. For example, *references* in Java can be considered as similar elements to variable pointers in C. These features and their related concepts are so fundamental, that without them, no piece of software could continue to work, even if it could get executed! Even a simple "hello world" program cannot work without loading a number of shared libraries that require the usage of *function pointers*!

So, whenever you see something like traffic lights, your car's central computer, the microwave oven in your kitchen, the operating system of your smartphone, or probably any other device that you generally don't think about, they all have pieces of software written in C.

Our lives today have been greatly impacted by the invention of the C programming language, and our world would be very different without C.

This chapter focuses on the essential features and machinery required to write expert C code and contains a handpicked collection of features for us to study in depth. We'll explore the following topics:

*   **Preprocessor directives, macros, and conditional compilation**: Preprocessing is one of those C features that you can't easily find in other programming languages. Preprocessing brings a lot of advantages, and we'll dive into some of its interesting applications, including *macros* and *conditional directives*.
*   **Variable pointers**: This section deep-dives into *variable pointers* and their uses. We'll also find some helpful insights by looking at some of the flaws that could be introduced by misusing variable pointers.
*   **Functions**: This section of the chapter is a deep dive into everything we know about functions, beyond just their syntax. In fact, the syntax is the easy part! In this section, we will look at functions as the building blocks for writing *procedural* code. This section also talks about the *function call mechanism* and how a function receives its arguments from the caller function.
*   **Function pointers**: Undoubtedly, *function pointers* are one of the most important features of C. A function pointer is a pointer that points to an existing function instead of a variable. The ability to store a pointer to an existing logic is profoundly important in algorithm design, and that's why we have a dedicated section on this topic. Function pointers appear in a vast range of applications ranging from loading dynamic libraries to *polymorphism*, and we'll be seeing plenty more of function pointers across the next couple of chapters.
*   **Structures**: C structures might have a simple syntax and convey a simple idea, but they are the main building blocks for writing *well-organized* and more *object-oriented* code. Their importance, together with function pointers, simply cannot be overstated! In the last section of this chapter, we'll revisit all the things that you need to know about structures in C and the tricks around them.

The essential features of C, and their surrounding concepts, play a key role in the Unix ecosystem, and they have led to C being an important and influential technology despite its old age and harsh syntax. We will talk more about the mutual influence of C and Unix upon one another in the upcoming chapters. For now, let's begin this first chapter by talking about preprocessor directives.

Before reading this chapter, bear in mind that you should be already familiar with C. Most of the examples are trivial in this chapter, but it is highly recommended that you know C syntax before moving on to other chapters. For your convenience, below is a list of topics that you should be familiar with before moving on with this book:

General knowledge of computer architecture — you should know about memory, CPU, peripheral devices and their characteristics, and how a program interacts with these elements in a computer system.

General knowledge of programming — you should know what an algorithm is, how its execution can be traced, what a source code is, what binary numbers are, and how their related arithmetic work.

Familiarity with using the Terminal and the basic shell commands in a Unix-like operating system such as Linux or macOS.

Intermediate knowledge about programming topics such as conditional statements, different kinds of loops, structures or classes in at least one programming language, pointers in C or C++, functions, and so on.

Basic knowledge about object-oriented programming — this is not mandatory because we will explain object-oriented programming in detail, but such knowledge will help you to have a better understanding while reading the chapters in the third part of the book; *Object Orientation*.

# Preprocessor directives

*Preprocessing* is a powerful feature in C. We'll cover it fully in *Chapter 2*, *Compilation and Linking*, but right now let's define preprocessing as something that allows you to engineer and modify your source code before submitting it to the compiler. This means that the C compilation pipeline has at least one step more in comparison to other languages. In other programming languages, the source code is directly sent to the compiler, but in C and C++, it should be preprocessed first.

This extra step has made C (and C++) a unique programming language in the sense that a C programmer can effectively change their source code before submitting it to the compiler. This feature is not present in most higher-level programming languages.

The purpose of preprocessing is to remove the preprocessing directives and substitute them with equivalent generated C code and prepare a final source that is ready to be submitted to the compiler.

The behavior of the C preprocessor can be controlled and influenced using a set of *directives*. C directives are lines of code starting with a `#` character in both header and source files. These lines are only meaningful to the C preprocessor and never to the C compiler. There are various directives in C, but some of them are very important especially the directives used for macro definition and the directives used for conditional compilation.

In the following section, we'll explain macros and give various examples demonstrating their various uses. We also analyze them further to find their advantages and disadvantages.

## Macros

There are many rumors regarding C macros. One says that they make your source code too complicated and less readable. Another says that you face issues while debugging applications if you have used macros in your code. You might have heard some of these statements yourself. But to what extent are they valid? Are macros evils that should be avoided? Or do they have some benefits that can be brought to your project?

The reality is that you'll find macros in any well-known C project. As proof, download a well-known C project such as Apache HTTP Server and do a grep for `#define`. You will see a list of files where macros are defined. For you as a C developer, there is no way to escape macros. Even if you don't use them yourself, you will likely see them in other people's code. Therefore, you need to learn what they are and how to deal with them.

The `grep` command refers to a standard shell utility program in Unix-like operating systems that searches for a pattern in a stream of characters. It can be used to search for a text or a pattern in the content of all files found in a given path.

Macros have a number of applications and you can see some of them as follows:

*   Defining a constant
*   Using as a function instead of writing a C function
*   Loop unrolling
*   Header guards
*   Code generation
*   Conditional compilation

While there are many more possible applications of macros, we'll focus on the above in the following sections.

### Defining a macro

Macros are defined using the `#define` directive. Each macro has a name and a possible list of parameters. It also has a *value* that gets substituted by its *name* in the preprocessing phase through a step called *macro expansion*. A macro can also be *undefined* with the `#undef` directive. Let's start with a simple example, *example 1.1*:

```cpp
#define ABC 5
int main(int argc, char** argv) {
  int x = 2;
  int y = ABC;
  int z = x + y;
  return 0;
}
```

Code Box 1-1 [ExtremeC_examples_chapter1_1.c]: Defining a macro

In the preceding code box, `ABC` is not a variable that holds an integer value nor an integer constant. In fact, it's a macro called `ABC` and its corresponding value is `5`. After the macro expansion phase, the resulting code that can be submitted to the C compiler looks similar to the one we see as follows:

```cpp
int main(int argc, char** argv) {
  int x = 2;
  int y = 5;
  int z = x + y;
  return 0;
}
```

Code Box 1-2: The generated code for the example 1.1 after macro expansion phase

The code in *Code Box 1-2* has a valid C syntax, and now the compiler can proceed and compile it. In the preceding example, the preprocessor did the macro expansion, and as a part of it, the preprocessor simply replaced the macro's name with its value. The preprocessor has also removed the comments on the beginning lines.

Let's now have a look at another example, *example 1.2*:

```cpp
#define ADD(a, b) a + b
int main(int argc, char** argv) {
  int x = 2;
  int y = 3;
  int z = ADD(x, y);
  return 0;
}
```

Code Box 1-3 [ExtremeC_examples_chapter1_2.c]: Defining a function-like macro

In the preceding code box, similar to *example 1.1*, `ADD` is not a function. It is just a *function-like macro* that accepts arguments. After preprocessing, the resulting code will be like this:

```cpp
int main(int argc, char** argv) {
  int x = 2;
  int y = 3
  int z = x + y;
  return 0;
}
```

Code Box 1-4: Example 1.2 after preprocessing and macro expansion

As you can see in the preceding code box, the expansion that has taken place is as follows. The argument `x` used as parameter `a` is replaced with all instances of `a` in the macro's value. This is the same for the parameter `b`, and its corresponding argument `y`. Then, the final substitution occurs, and we get `x + y` instead of `ADD(a, b)` in the preprocessed code.

Since function-like macros can accept input arguments, they can mimic C functions. In other words, instead of putting a frequently used logic into a C function, you can name that logic as a function-like macro and use that macro instead.

This way, the macro occurrences will be replaced by the frequently used logic, as part of the preprocessing phase, and there is no need to introduce a new C function. We will discuss this more and compare the two approaches.

Macros only exist before the compilation phase. This means that the compiler, theoretically, doesn't know anything about the macros. This is a very important point to remember if you are going to use macros instead of functions. The compiler knows everything about a function because it is part of the C grammar and it is parsed and being kept in the *parse tree*. But a macro is just a C preprocessor directive only known to the preprocessor itself.

Macros allow you to *generate* code before the compilation. In other programming languages such as Java, you need to use a *code generator* to fulfill this purpose. We will give examples regarding this application of macros.

Modern C compilers are aware of C preprocessor directives. Despite the common belief that they don't know anything about the preprocessing phase, they actually do. The modern C compilers know about the source before entering the preprocessing phase. Look at the following code:

```cpp
#include <stdio.h>
#define CODE \
printf("%d\n", i);
int main(int argc, char** argv) {
 CODE
 return 0;
}
```

Code Box 1-5 [example.c]: Macro definition which yields an undeclared identifier error

If you compile the above code using `clang` in macOS, the following would be the output:

```cpp
$ clang example.c
code.c:7:3: error: use of undeclared identifier 'i'
CODE
^
code.c:4:16: note: expanded from macro 'CODE'
printf("%d\n", i);
               ^
1 error generated.
$
```

Shell Box 1-1: The output of the compilation refers to the macro definition

As you see, the compiler has generated an error message that points exactly to the line in which the macro is defined.

As a side note, in most modern compilers, you can view the preprocessing result just before the compilation. For example, when using `gcc` or `clang`, you can use the `-E` option to dump the code after preprocessing. The following shell box demonstrates how to use the `-E` option. Note that the output is not fully shown:

```cpp
$ clang -E example.c
# 1 "sample.c"# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 361 "<built-in>" 3
...
# 412 "/Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/include/stdio.h" 2 3 4
# 2 "sample.c" 2
...
int main(int argc, char** argv) {
 printf("%d\n", i);
 return 0;
}
$
```

Shell Box 1-2: The code of example.c after preprocessing phase

Now we come to an important definition. A *translation unit* (or a *compilation unit*) is the preprocessed C code that is ready to be passed to the compiler.

In a translation unit, all directives are substituted with inclusions or macro expansions and a flat long piece of C code has been produced.

Now that you know more about macros, let's work on some more difficult examples. They will show you the power and danger of macros. In my opinion, extreme development deals with dangerous and delicate stuff in a skilled way, and this is exactly what C is about.

The next example is an interesting one. Just pay attention to how the macros are used in sequence to generate a loop:

```cpp
#include <stdio.h>
#define PRINT(a) printf("%d\n", a);
#define LOOP(v, s, e) for (int v = s; v <= e; v++) {
#define ENDLOOP }
int main(int argc, char** argv) {
 LOOP(counter, 1, 10)
 PRINT(counter)
 ENDLOOP
  return 0;
}
```

Code Box 1-6 [ExtremeC_examples_chapter1_3.c]: Using macros to generate a loop

As you see in the preceding code box, the code inside the `main` function is not a valid C code in any way! But after preprocessing, we get a correct C source code that compiles without any problem. Following is the preprocessed result:

```cpp
...
... content of stdio.h …
...
int main(int argc, char** argv) {
  for (int counter = 1; counter <= 10; counter++) {
    printf("%d\n", counter);
  }
  return 0;
}
```

Code Box 1-7: Example 1.3 after preprocessing phase

In *Code Box 1-6*, in the `main` function, we just used a different and not C-looking set of instructions to write our algorithm. Then after preprocessing, in *Code Box 1-7*, we got a fully functional and correct C program. This is an important application of macros; to define a new **domain specific language** (**DSL**) and write code using it.

DSLs are very useful in different parts of a project; for example, they are used heavily in testing frameworks such as Google Test framework (gtest) where a DSL is used to write assertions, expectations, and test scenarios.

We should note that we don't have any C directives in the final preprocessed code. This means that the `#include` directive in *Code Box 1-6* has been replaced by the contents of the file it was referring to. That is why you see the content of the `stdio.h` header file (which we replaced with ellipses) in *Code Box 1-7* before the `main` function.

Let's now look at the next example, *example 1.4*, which introduces two new operators regarding macro parameters; the `#` and `##` operators:

```cpp
#include <stdio.h>
#include <string.h>
#define CMD(NAME) \
 char NAME ## _cmd[256]  = ""; \
 strcpy(NAME ## _cmd, #NAME);
int main(int argc, char** argv) {
 CMD(copy)
 CMD(paste)
 CMD(cut)
  char cmd[256];
  scanf("%s", cmd);
  if (strcmp(cmd, copy_cmd) == 0) {
    // ...
  }
  if (strcmp(cmd, paste_cmd) == 0) {
    // ...
  }
  if (strcmp(cmd, cut_cmd) == 0) {
    // ...
  }
  return 0;
}
```

Code Box 1-8 [ExtremeC_examples_chapter1_4.c]: Using # and ## operators in a macro

While expanding the macro, the `#` operator turns the parameter into its string form surrounded by a pair of quotation marks. For example, in the preceding code, the `#` operator used before the `NAME` parameter turns it into `"copy"` in the preprocessed code.

The `##` operator has a different meaning. It just concatenates the parameters to other elements in the macro definition and usually forms variable names. Following is the final preprocessed source for *example 1.4*:

```cpp
...
... content of stdio.h ...
...
... content of string.h ...
...
int main(int argc, char** argv) {
  char copy_cmd[256] = ""; strcpy(copy_cmd, "copy");
  char paste_cmd[256] = ""; strcpy(paste_cmd, "paste");
  char cut_cmd[256] = ""; strcpy(cut_cmd, "cut");
  char cmd[256];
  scanf("%s", cmd);
  if (strcmp(cmd, copy_cmd) == 0) {
  }
  if (strcmp(cmd, paste_cmd) == 0) {
  }
  if (strcmp(cmd, cut_cmd) == 0) {
  }
  return 0;
}
```

Code Box 1-9: Example 1.4 after preprocessing phase

Comparing the source before and after preprocessing helps you to realize how `#` and `##` operators are applied to the macro arguments. Note that, in the final preprocessed code, all lines expanded from the same macro definition are on the same line.

It is a good practice to break long macros into multiple lines but do not forget to use \ (one backslash) to let the preprocessor know that the rest of the definition comes on the next line. Note that \ doesn't get substituted with a *newline* character. Instead, it is an indicator that the following line is the continuation of the same macro definition.

Now let's talk about a different type of macros. The next section is going to talk about *variadic macros* which can accept a variable number of arguments.

### Variadic macros

The next example, *example 1.5*, is dedicated to variadic macros, which can accept a variable number of input arguments. Sometimes the same variadic macro accepts 2 arguments, sometimes 4 arguments, and sometimes 7\. Variadic macros are very handy when you are not sure about the number of arguments in different usages of the same macro. A simple example is given as follows:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VERSION "2.3.4"
#define LOG_ERROR(format, ...) \
 fprintf(stderr, format, __VA_ARGS__)
int main(int argc, char** argv) {
  if (argc < 3) {
 LOG_ERROR("Invalid number of arguments for version %s\n.", VERSION);
    exit(1);
  }
  if (strcmp(argv[1], "-n") != 0) {
 LOG_ERROR("%s is a wrong param at index %d for version %s.", argv[1], 1, VERSION);
    exit(1);
  }
  // ...
  return 0;
}
```

Code Box 1-10 [ExtremeC_examples_chapter1_5.c]: Definition and usage of a variadic macro

In the preceding code box, you see a new identifier: `__VA_ARGS__`. It is an indicator that tells the preprocessor to replace it with all the remaining input arguments that are not assigned to any parameter yet.

In the preceding example, in the second usage of `LOG_ERROR`, according to the macro definition, the arguments `argv[1]`, `1`, and `VERSION` are those input arguments that are *not* assigned to any parameter. So, they are going to be used in place of `__VA_ARGS__` while expanding the macro.

As a side note, the function `fprintf` writes to a *file descriptor*. In *example 1.5*, the file descriptor is `stderr`, which is the *error stream* of the process. Also, note the ending semicolon after each `LOG_ERROR` usage. It is mandatory because the macro doesn't supply them as part of its definition and the programmer *must* add that semicolon to make the final preprocessed code syntactically correct.

The following code is the final output after passing through the C preprocessor:

```cpp
...
... content of stdio.h ...
...
... content of stdlib.h ...
...
... content of string.h ...
...
int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Invalid number of arguments for version %s\n.", "2.3.4");
    exit(1);
  }
  if (strcmp(argv[1], "-n") != 0) {
    fprintf(stderr, "%s is a wrong param at index %d for version %s.", argv[1], 1, "2.3.4");
    exit(1);
  }
  // ...
  return 0;
}
```

Code Box 1-11: Example 1.5 after preprocessing phase

The next example, *example 1.6*, is a progressive usage of variadic macros that tries to mimic a loop. There is a well-known example about this. Before having `foreach` in C++, the *boost* framework was (and still is) offering the `foreach` behavior using a number of macros.

In the following link, you can see how the `BOOST_FOREACH` macro is defined as the last thing in the header file: [https://www.boost.org/doc/libs/1_35_0/boost/foreach.hpp](https://www.boost.org/doc/libs/1_35_0/boost/foreach.hpp). It is used to iterate over a boost collection, and it is actually a function-like macro.

Our following example, *example 1.6*, is about a simple loop which is not comparable to boost's `foreach` at all, but yet, it is giving you an idea on how to use variadic macros for repeating a number of instructions:

```cpp
#include <stdio.h>
#define LOOP_3(X, ...) \
 printf("%s\n", #X);
#define LOOP_2(X, ...) \
 printf("%s\n", #X); \
 LOOP_3(__VA_ARGS__)
#define LOOP_1(X, ...) \
 printf("%s\n", #X); \
 LOOP_2(__VA_ARGS__)
#define LOOP(...) \
 LOOP_1(__VA_ARGS__)
int main(int argc, char** argv) {
 LOOP(copy paste cut)
 LOOP(copy, paste, cut)
 LOOP(copy, paste, cut, select)
  return 0;
}
```

Code Box 1-12 [ExtremeC_examples_chapter1_6.c]: Using variadic macros to mimic a loop

Before starting to explain the example, let's look at the final code after preprocessing. Then, the explanation of what happened will be easier:

```cpp
...
... content of stdio.h ...
...
int main(int argc, char** argv) {
  printf("%s\n", "copy paste cut"); printf("%s\n", ""); printf("%s\n", "");
  printf("%s\n", "copy"); printf("%s\n", "paste"); printf("%s\n", "cut");
  printf("%s\n", "copy"); printf("%s\n", "paste"); printf("%s\n", "cut");
  return 0;
}
```

Code Box 1-13: Example 1.6 after preprocessing phase

If you look at the preprocessed code carefully, you will see that the `LOOP` macro has been expanded to multiple `printf` instructions instead of looping instructions such as `for` or `while`. It is obvious why this is the case, and it's because of the fact that the preprocessor doesn't write smart C code for us. It is there to replace macros with the instructions given by us.

The only way to create a loop with a macro is just to put the iteration instructions one after another, and as some separate instructions. This means that a simple macro loop with 1000 iterations will be substituted with 1000 instructions in C and we won't have any actual C loop in the final code.

The preceding technique will lead to a large binary size which can be thought of as a disadvantage. But putting instructions one after another instead of putting them into a loop, which is known as *loop unrolling*, has its own applications, which require an acceptable level of performance in constrained and high-performance environments. According to what we explained so far, it seems that loop unrolling using macros is a trade-off between binary size and performance. We talk more about this in the upcoming section.

There is one more note about the preceding example. As you see, different usages of the `LOOP` macro in the `main` function have produced different results. In the first usage, we pass `copy paste cut` without any commas between the words. The preprocessor accepts it as a single input, so the simulated loop has only one iteration.

In the second usage, the input `copy, paste, cut` is passed with words separated by commas. Now, the preprocessor treats them as three different arguments; hence, the simulated loop has three iterations. This is clear from the following *Shell Box 1-3*.

In the third usage, we pass four values, `copy, paste, cut, select,` but only three of them are processed. As you see, the preprocessed code is exactly the same as the second usage. That's because of the fact that our looping macros are only capable of handling lists of up to three elements. Extra elements after the third are ignored.

Note that this doesn't produce compilation errors because nothing wrong has been generated as the final C code, but our macros are limited in the number of elements they can process:

```cpp
$ gcc ExtremeC_examples_chapter1_6.c
$ ./a.out
copy paste cut
copy
paste
cut
$
```

Shell Box 1-3: Compilation and output of example 1.6

### Advantages and disadvantages of macros

Let's start the discussion by talking a bit about software design. Defining macros and putting them together is an art, and at times an addictive one! You start building the expected preprocessed code in your mind even before having any macros defined and based on that you define your macros. Since it is an easy way to replicate the code and play with it, it can be overused. Overusing the macros may not be a big issue for you, but it might be for your teammates. But why is that?

Macros have an important characteristic. If you write something in macros, they will be replaced by other lines of code before the compilation phase, and finally, you'll have a flat long piece of code without any modularity at compile time. Of course, you have the modularity in your mind and probably in your macros, but it is not present in your final binaries. This is exactly where using macros can start to cause design issues.

Software design tries to package similar algorithms and concepts in several manageable and reusable *modules*, but macros try to make everything linear and flat. So, when you are using macros as some logical building blocks within your software design, the information regarding them can be lost after the preprocessing phase, as part of the final translation units. That's why the architects and designers use a rule of thumb about macros:

> If a macro can be written as a C function, then you should write a C function instead!

From the debugging perspective, again, it is said that macros are evil. A developer uses compilation errors to find the places where there exist *syntax errors* as part of their daily development tasks. They also use *logs* and possibly *compilation warnings* to detect a bug and fix it. The compilation errors and warnings both are beneficial to the bug analysis routine, and both of them are generated by the compilers.

Regarding the macros, and especially with old C compilers, the compiler didn't know anything about the macros and it was treating the compiling source (the translation unit) as a long, linear, flat piece of code. So, for a developer looking at the actual C code with macros and for the C compiler looking at the preprocessed code without macros, there were two different worlds. So, the developer could not understand easily what the compiler reported.

Hopefully, with the help of our modern C compilers, this issue is not that severe anymore. Nowadays, the famous C compilers such as `gcc` and `clang` know more about the preprocessing phase, and they try to keep, use, and report according to the source code that the developer sees. Otherwise, the problem with macros could be repeated with `#include` directives, simply because the main content of a translation unit is only known when all inclusions have happened. As a conclusion, we can say that the problem with debugging is less severe than the problem we explained in the previous paragraph about the software design.

If you remember, we brought up a discussion during the explanation of *example 1.6*. It was about a trade-off between the binary size and the performance of a program. A more general form of this trade-off is between having a single big binary and having multiple small binaries. Both of them are providing the same functionality, but the former can have a better performance.

The number of binaries used in a project, especially when the project is big, is more or less proportional to the degree of *modularization* and the design effort spent on it. As an example, a project having 60 libraries (shared or static) and one executable seems to be developed according to a software plan which is splitting dependencies into multiple libraries and using them in a single main executable.

In other words, when a project is being developed according to the software design principals and best practices, the number of binaries and their sizes are engineered in a careful way and usually will be comprised of multiple lightweight binaries with applicable minimum sizes, instead of having a single huge binary.

Software design tries to have each software component in a suitable position in a giant hierarchy instead of putting them in a linear order. And this is intrinsically against the performance even though its effect on the performance is tiny in most cases.

So, we can conclude that the discussion regarding *example 1.6* was about the trade-off between design and performance. When you need performance, sometimes you need to sacrifice the design and put things in a linear construction. For example, you could avoid loops and use *loop unrolling* instead.

From a different perspective, performance starts with choosing proper algorithms for the problems defined in the design phase. The next step is usually called *optimization* or *performance tuning*. In this phase, gaining performance is equivalent to letting the CPU just compute in a linear and sequential manner and not to force it to jump between different parts of the code. This can be done either by modifying the already used algorithms or by replacing them with some performant and usually more complex algorithms. This stage can come into conflict with the design philosophy. As we said before, design tries to put things in a hierarchy and make them non-linear, but the CPU expects things to be linear, already fetched and ready to be processed. So, this trade-off should be taken care of and balanced for each problem separately.

Let's explain the loop unrolling a bit more. This technique is mostly used in embedded development and especially in environments that suffer from limited processing power. The technique is to remove loops and make them linear to increase the performance and avoid the looping overhead while running iterations.

This is exactly what we did in *example 1.6*; we mimicked a loop with macros, which led to a linear set of instructions. In this sense, we can say that the macros can be used for performance tuning in embedded development and the environments in which a slight change in the way that the instructions are executed will cause a significant performance boost. More than that, macros can bring readability to the code and we can factor out repeated instructions.

Regarding the quote mentioned earlier that says that macros should be replaced by equivalent C functions, we know that the quote is there for the sake of design and it can be ignored in some contexts. In a context where improved performance is a key requirement, having a linear set of instructions that lead to better performance may be a necessity.

Code generation is another common application of macros. They can be used for introducing DSLs into a project. *Microsoft MFC*, *Qt*, *Linux Kernel*, and *wxWidgets* are a few projects out of thousands that are using macros to define their own DSLs. Most of them are C++ projects, but they are using this C feature to facilitate their APIs.

As a conclusion, C macros can have advantages if the impacts of their preprocessed form are investigated and known. If you're working on a project within a team, always share your decisions regarding the usage of the macros in the team and keep yourself aligned with the decisions made within the team.

## Conditional compilation

Conditional compilation is another unique feature of C. It allows you to have different preprocessed source code based on different conditions. Despite the meaning it implies, the compiler is not doing anything conditionally, but the preprocessed code that is passed to the compiler can be different based on some specified conditions. These conditions are evaluated by the preprocessor while preparing the preprocessed code. There are different directives contributing to the conditional compilation. You can see a list of them as follows:

*   `#ifdef`
*   `#ifndef`
*   `#else`
*   `#elif`
*   `#endif`

The following example, *example 1.7*, demonstrates a very basic usage of these directives:

```cpp
#define CONDITION
int main(int argc, char** argv) {
#ifdef CONDITION
  int i = 0;
  i++;
#endif
  int j= 0;
  return 0;
}
```

Code Box 1-14 [ExtremeC_examples_chapter1_7.c]: An example of conditional compilation

While preprocessing the preceding code, the preprocessor sees the `CONDITION` macro's definition and marks it as defined. Note that no value is proposed for the `CONDITION` macro and this is totally valid. Then, the preprocessor goes down further until it reaches the `#ifdef` statement. Since the `CONDITION` macro is already defined, all lines between `#ifdef` and `#endif` will be copied to the final source code.

You can see the preprocessed code in the following code box:

```cpp
int main(int argc, char** argv) {
  int i = 0;
  i++;
  int j= 0;
  return 0;
}
```

Code Box 1-15: Example 1.7 after preprocessing phase

If the macro was not defined, we wouldn't see any replacement for the `#if-#endif` directives. Therefore, the preprocessed code could be something like the following:

```cpp
int main(int argc, char** argv) {
  int j= 0;
  return 0;
}
```

Code Box 1-16: Example 1.7 after preprocessing phase assuming that CONDITION macro is not defined

Note the empty lines, in both code boxes *1-15* and *1-16*, which have been remained from the preprocessing phase, after replacing the `#ifdef-#endif` section with its evaluated value.

Macros can be defined using `-D` options passed to the compilation command. Regarding the preceding example, we can define the `CONDITION` macro as follows:

```cpp
$ gcc -DCONDITION -E main.c
```

This is a great feature because it allows you to have macros defined out of source files. This is especially helpful when having a single source code but compiling it for different architectures, for example, Linux or macOS, which have different default macro definitions and libraries.

One of the very common usages of `#ifndef` is to serve as a *header guard* statement. This statement protects a header file from being included twice in the preprocessing phase, and we can say that almost all C and C++ header files in nearly every project have this statement as their first instruction.

The following code, *example 1.8*, is an example on how to use a header guard statement. Suppose that this is the content of a header file and by chance, it could be included twice in a compilation unit. Note that *example 1.8* is just one header file and it is not supposed to be compiled:

```cpp
#ifndef EXAMPLE_1_8_H
#define EXAMPLE_1_8_H
void say_hello();
int read_age();
#endif
```

Code Box 1-17 [ExtremeC_examples_chapter1_8.h]: An example of a header guard

As you see, all variable and function declarations are put inside the `#ifndef` and `#endif` pair and they are protected against multiple inclusions by a macro. In the following paragraph, we explain how.

As the first inclusion happens, the `EXAMPLE_1_8_H` macro is not yet defined, so the preprocessor continues by entering the `#ifndef-#endif` block. The next statement defines the `EXAMPLE_1_8_H` macro, and the preprocessor copies everything to the preprocessed code until it reaches the `#endif` directive. As the second inclusion happens, the `EXAMPLE_1_8_H` macro is already defined, so the preprocessor skips all of the content inside the `#ifndef-#endif` section and moves to the next statement after `#endif`, if there is any.

It is a common practice that the whole content of a header file is put between the `#ifndef-#endif` pair, and nothing but comments are left outside.

As a final note in this section, instead of having a pair of `#ifndef-#endif` directives, one could use `#pragma once` in order to protect the header file from the *double inclusion* issue. The difference between conditional directives and the `#pragma once` directive is that the latter is not a C standard, despite the fact that it is supported by almost all C preprocessors. However, it is better to not to use it if *portability* of your code is a requirement.

The following code box contains a demonstration on how to use `#pragma once` in *example 1.8*, instead of `#ifndef-#endif` directives:

```cpp
#pragma once
void say_hello();
int read_age();
```

Code Box 1-18: Using #pragma once directive as part of example 1.8

Now, we close the topic of preprocessor directives while we have demonstrated some of their interesting characteristics and various applications. The next section is about variable pointers, which are another important feature of C.

# Variable pointers

The concept of a variable pointer, or for short pointer, is one of the most fundamental concepts in C. You can hardly find any direct sign of them in most high-level programming languages. In fact, they have been replaced by some twin concepts, for example, *references* in Java. It is worth mentioning that pointers are unique in the sense that the addresses they point to can be used directly by hardware, but this is not the case for the higher-level twin concepts like references.

Having a deep understanding about pointers and the way they work is crucial to become a skilled C programmer. They are one of the most fundamental concepts in memory management, and despite their simple syntax, they have the potential to lead to a disaster when used in a wrong way. We will cover memory management-related topics in *Chapter 4*, *Process Memory Structure*, and *Chapter 5*, *Stack and Heap*, but here in this chapter, we want to recap everything about pointers. If you feel confident about the basic terminology and the concepts surrounding the pointers, you can skip this section.

## Syntax

The idea behind any kind of pointer is very simple; it is just a simple variable that keeps a *memory address*. The first thing you may recall about them is the asterisk character, `*`, which is used for declaring a pointer in C. You can see it in *example 1.9*. The following code box demonstrates how to declare and use a variable pointer:

```cpp
int main(int argc, char** argv) {
  int var = 100;
  int* ptr = 0;
  ptr = &var;
  *ptr = 200;
  return 0;
}
```

Code Box 1-19 [ExtremeC_examples_chapter1_9.c]: Example on how to declare and use a pointer in C

The preceding example has everything you need to know about the pointer's syntax. The first line declares the `var` variable on top of the *Stack segment*. We will discuss the Stack segment in *Chapter 4*, *Process Memory Structure*. The second line declares the pointer `ptr` with an initial value of zero. A pointer which has the zero value is called a *null pointer*. As long as the `ptr` pointer retains its zero value, it is considered to be a null pointer. It is very important to *nullify* a pointer if you are *not* going to store a valid address upon declaration.

As you see in *Code Box 1-19*, no header file is included. Pointers are part of the C language, and you don't need to have anything included to be able to use them. Indeed, we can have C programs, which do not include any header file at all.

All of the following declarations are valid in C:

`int* ptr = 0;`

`int * ptr = 0;`

`int *ptr = 0;`

The third line in the `main` function introduces the `&` operator, which is called the *referencing operator*. It returns the address of the variable next to it. We need this operator to obtain the address of a variable. Otherwise, we cannot initialize pointers with valid addresses.

On the same line, the returned address is stored into the `ptr` pointer. Now, the `ptr` pointer is not a null pointer anymore. On the fourth line, we see another operator prior to the pointer, which is called the *dereferencing operator* and denoted by `*`. This operator allows you to have indirect access to the memory cell that the `ptr` pointer is pointing to. In other words, it allows you to read and modify the `var` variable through the pointer that is pointing to it. The fourth line is equivalent to the `var = 200;` statement.

A null pointer is not pointing to a valid memory address. Therefore, dereferencing a null pointer *must* be avoided because it is considered as an *undefined behavior*, which usually leads to a crash.

As a final note regarding the preceding example, we usually have the default macro `NULL` defined with value `0`, and it can be used to nullify pointers upon declaration. It is a good practice to use this macro instead of `0` directly because it makes it easier to distinguish between the variables and the pointers:

```cpp
char* ptr = NULL;
```

Code Box 1-20: Using the NULL macro to nullify a pointer

The pointers in C++ are exactly the same as in C. They need to be nullified by storing `0` or `NULL` in them, but C++11 has a new keyword for initializing the pointers. It is not a macro like `NULL` nor an integer like `0`. The keyword is `nullptr` and can be used to nullify the pointers or check whether they are null. The following example demonstrates how it is used in C++11:

```cpp
char* ptr = nullptr;
```

Code Box 1-21: Using nullptr to nullify a pointer in C++11

It is crucial to remember that pointers *must* be initialized upon declaration. If you don't want to store any valid memory address while declaring them, don't leave them uninitialized. Make it null by assigning `0` or `NULL!` Do this otherwise you may face a fatal bug!

In most modern compilers, an uninitialized pointer is always nullified. This means that the initial value is `0` for all uninitialized pointers. But this shouldn't be considered as an excuse to declare pointers without initializing them properly. Keep in mind that you are writing code for different architectures, old and new, and this may cause problems on legacy systems. In addition, you will get a list of errors and warnings for these kinds of uninitialized pointers in most *memory profilers*. Memory profilers will be explained thoroughly as part of *Chapter 4*, *Process Memory Structure*, and *Chapter 5*, *Stack and Heap*.

## Arithmetic on variable pointers

The simplest picture of memory is a very long one-dimensional array of bytes. With this picture in mind, if you're standing on one byte, you can only go back and forth in the array; there's no other possible movement. So, this would be the same for the pointers addressing different bytes in the memory. Incrementing the pointer makes the pointer go forward and decrementing it makes the pointer go backward. No other arithmetic operation is possible for the pointers.

Like we said previously, the arithmetic operations on a pointer are analogous to the movements in an array of bytes. We can use this figure to introduce a new concept: the *arithmetic step size*. We need to have this new concept because when you increment a pointer by 1, it might go forward more than 1 byte in the memory. Each pointer has an arithmetic step size, which means the number of bytes that the pointer will move if it is incremented or decremented by 1\. This arithmetic step size is determined by the C *data type* of the pointer.

In every platform, we have one single unit of memory and all pointers store the addresses inside that memory. So, all pointers should have equal sizes in terms of bytes. But this doesn't mean that all of them have equal arithmetic step sizes. As we mentioned earlier, the arithmetic step size of a pointer is determined by its C data type.

For example, an `int` pointer has the same size as a `char` pointer, but they have different arithmetic step sizes. `int*` usually has a 4-byte arithmetic step size and `char*` has a 1-byte arithmetic step size. Therefore, incrementing an integer pointer makes it move forward by 4 bytes in the memory (adds 4 bytes to the current address), and incrementing a character pointer makes it move forward by only 1 byte in the memory. The following example, *example 1.10*, demonstrates the arithmetic step sizes of two pointers with two different data types:

```cpp
#include <stdio.h>
int main(int argc, char** argv) {
  int var = 1;
  int* int_ptr = NULL; // nullify the pointer
  int_ptr = &var;
  char* char_ptr = NULL;
  char_ptr = (char*)&var;
  printf("Before arithmetic: int_ptr: %u, char_ptr: %u\n",
          (unsigned int)int_ptr, (unsigned int)char_ptr);
  int_ptr++;    // Arithmetic step is usually 4 bytes
  char_ptr++;   // Arithmetic step in 1 byte
  printf("After arithmetic: int_ptr: %u, char_ptr: %u\n",
          (unsigned int)int_ptr, (unsigned int)char_ptr);
  return 0;
}
```

Code Box 1-22 [ExtremeC_examples_chapter1_10.c]: Arithmetic step sizes of two pointers

The following shell box shows the output of *example 1.10*. Note that the printed addresses can be different for two successive runs on the same machine, and even from a platform to another, therefore you probably observe different addresses in your output:

```cpp
$ gcc ExtremeC_examples_chapter1_10.c
$ ./a.out
Before arithmetic: int_ptr: 3932338348, char_ptr: 3932338348
After arithmetic:  int_ptr: 3932338352, char_ptr: 3932338349
$
```

Shell Box 1-4: Output of example 1.10 after first run

It is clear from the comparison of the addresses before and after the arithmetic operations that the step size for the integer pointer is 4 bytes, and it is 1 byte for the character pointer. If you run the example again, the pointers probably refer to some other addresses, but their arithmetic step sizes remain the same:

```cpp
$ ./a.out
Before arithmetic: int_ptr: 4009638060, char_ptr: 4009638060
After arithmetic:  int_ptr: 4009638064, char_ptr: 4009638061
$
```

Shell Box 1-5: Output of example 1.10 after second run

Now that you know about the arithmetic step sizes, we can talk about a classic example of using pointer arithmetic to *iterate* over a region of memory. The examples 1.11 and 1.12 are about to print all the elements of an integer array. The trivial approach without using the pointers is brought in *example 1.11*, and the solution based on pointer arithmetic is given as part of *example 1.12*.

The following code box shows the code for *example 1.11*:

```cpp
#include <stdio.h>
#define SIZE 5
int main(int argc, char** argv) {
 int arr[SIZE];
 arr[0] = 9;
 arr[1] = 22;
 arr[2] = 30;
 arr[3] = 23;
 arr[4] = 18;
 for (int i = 0; i < SIZE; i++) {
   printf("%d\n", arr[i]);
 }
 return 0;
}
```

Code Box 1-23 [ExtremeC_examples_chapter1_11.c]: Iterating over an array without using pointer arithmetic

The code in *Code Box 1-23* should be familiar to you. It just uses a *loop counter* to refer to a specific index of the array and read its content. But if you want to use pointers instead of accessing the elements via the *indexer* syntax (an integer between `[` and `]`), it should be done differently. The following code box demonstrates how to use pointers to iterate over the array boundary:

```cpp
#include <stdio.h>
#define SIZE 5
int main(int argc, char** argv) {
 int arr[SIZE];
 arr[0] = 9;
 arr[1] = 22;
 arr[2] = 30;
 arr[3] = 23;
 arr[4] = 18;
 int* ptr = &arr[0];
 for (;;) {
   printf("%d\n", *ptr);
   if (ptr == &arr[SIZE - 1]) {
     break;
   }
   ptr++;
 }
 return 0;
}
```

Code Box 1-24 [ExtremeC_examples_chapter1_12.c]: Iterating over an array using pointer arithmetic

The second approach, demonstrated in *Code Box 1-24*, uses an infinite loop, which breaks when the address of the `ptr` pointer is the same as the last element of the array.

We know that arrays are adjacent variables inside the memory, so incrementing and decrementing a pointer which is pointing to an element effectively makes it move back and forth inside the array and eventually point to a different element.

As is clear from the preceding code, the `ptr` pointer has the data type `int*`. That's because of the fact that it must be able to point to any individual element of the array which is an integer of type `int`. Note that all the elements of an array are from the same type hence they have equal sizes. Therefore, incrementing the `ptr` pointer makes it point to the next element inside the array. As you see, before the `for` loop, `ptr` points to the first element of the array, and by further increments, it moves forward along the array's memory region. This is a very classic usage of pointer arithmetic.

Note that in C, an array is actually a pointer that points to its first element. So, in the example, the actual data type of `arr` is `int*`. Therefore, we could have written the line as follows:

```cpp
int* ptr = arr;
```

Instead of the line:

```cpp
int* ptr = &arr[0];
```

## Generic pointers

A pointer of type `void*` is said to be a generic pointer. It can point to any address like all other pointers, but we don't know its actual data type hence we don't know its arithmetic step size. Generic pointers are usually used to hold the content of other pointers, but they forget the actual data types of those pointers. Therefore, a generic pointer cannot be dereferenced, and one cannot do arithmetic on it because its underlying data type is unknown. The following example, *example 1.13*, shows us that dereferencing a generic pointer is not possible:

```cpp
#include <stdio.h>
int main(int argc, char** argv) {
 int var = 9;
 int* ptr = &var;
 void* gptr = ptr;
 printf("%d\n", *gptr);
 return 0;
}
```

Code Box 1-25 [ExtremeC_examples_chapter1_13.c]: Dereferencing a generic pointer generates a compilation error!

If you compile the preceding code using `gcc` in Linux, you will get the following error:

```cpp
$ gcc ExtremeC_examples_chapter1_13.c
In function 'main':warning: dereferencing 'void *' pointer
  printf("%d\n", *gptr);
                 ^~~~~
error: invalid use of void expression
  printf("%d\n", *gptr);
$
```

Shell Box 1-6: Compiling example 1.13 in Linux

And if you compile it using `clang` in macOS, the error message is different, but it refers to the same issue:

```cpp
$ clang ExtremeC_examples_chapter1_13.c
error: argument type 'void' is incomplete
  printf("%d\n", *gptr);
                 ^
1 error generated.
$
```

Shell Box 1-7: Compiling example 1.13 in macOS

As you see, both compilers don't accept dereferencing a generic pointer. In fact, it is meaningless to dereference a generic pointer! So, what are they good for? In fact, generic pointers are very handy to define *generic functions* that can accept a wide range of different pointers as their input arguments. The following example, *example 1.14*, tries to uncover the details regarding generic functions:

```cpp
#include <stdio.h>
void print_bytes(void* data, size_t length) {
  char delim = ' ';
  unsigned char* ptr = data;
  for (size_t i = 0; i < length; i++) {
    printf("%c 0x%x", delim, *ptr);
    delim = ',';
    ptr++;
  }
  printf("\n");
}
int main(int argc, char** argv) {
 int a = 9;
 double b = 18.9;
 print_bytes(&a, sizeof(int));
 print_bytes(&b, sizeof(double));
 return 0;
}
```

Code Box 1-26 [ExtremeC_examples_chapter1_14.c]: An example of a generic function

In the preceding code box, the `print_bytes` function receives an address as a `void*` pointer and an integer indicating the length. Using these arguments, the function prints all the bytes starting from the given address up to the given length. As you see, the function accepts a generic pointer, which allows the user to pass whatever pointer they want. Keep in mind that assignment to a *void pointer* (generic pointer) does *not* need an explicit cast. That is why we have passed the addresses of `a` and `b` without explicit casts.

Inside the `print_bytes` function, we have to use an `unsigned char` pointer in order to move inside the memory. Otherwise, we cannot do any arithmetic on the void pointer parameter, `data`, directly. As you may know, the step size of a `char*` or `unsigned char*` is one byte. So, it is the best pointer type for iterating over a range of memory addresses one byte at a time and processing all of those bytes one by one.

As a final note about this example, `size_t` is a standard and unsigned data type usually used for storing sizes in C.

`size_t` is defined in section 6.5.3.4 of the ISO/ICE 9899:TC3 standard. This ISO standard is the famous C99 specification revised in 2007\. This standard has been the basis for all C implementa[tions up until today. The link to ISO/ICE 9899:TC3 (2007)](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf) is [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf).

## Size of a pointer

If you search for the *size of a pointer in C* on Google, you may realize that you cannot find a definitive answer to that. There are many answers out there, and it is true that you cannot define a fixed size for a pointer in different architectures. The size of a pointer depends on the architecture rather than being a specific C concept. C doesn't worry too much about such hardware-related details, and it tries to provide a generic way of working with pointers and other programming concepts. That is why we know C as a standard. Only pointers and the arithmetic on them are important to C.

Architecture refers to the hardware used in a computer system. You will find more details in the upcoming chapter, *Compilation and Linking*.

You can always use the `sizeof` function to obtain the size of a pointer. It is enough to see the result of `sizeof(char*)` on your target architecture. As a rule of thumb, pointers are 4 bytes in 32-bit architectures and 8 bytes in 64-bit architectures, but you may find different sizes in other architectures. Keep in mind that the code you write *should not* be dependent on a specific value for the size of a pointer, and it should not make any assumptions about it. Otherwise, you will be in trouble while porting your code to other architectures.

## Dangling pointers

There are many known issues caused by misusing pointers. The issue of dangling pointers is a very famous one. A pointer usually points to an address to which there is a variable allocated. Reading or modifying an address where there is no variable registered is a big mistake and can result in a crash or a *segmentation fault* situation. Segmentation fault is a scary error that every C/C++ developer should have seen it at least once while working on code. This situation usually happens when you are misusing pointers. You are accessing places in memory that you are not allowed to. You had a variable there before, but it is deallocated by now.

Let's try to produce this situation as part of the following example, *example 1.15*:

```cpp
#include <stdio.h>
int* create_an_integer(int default_value) {
  int var = default_value;
  return &var;
}
int main() {
  int* ptr = NULL;
  ptr = create_an_integer(10);
  printf("%d\n", *ptr);
  return 0;
}
```

Code Box 1-27 [ExtremeC_examples_chapter1_15.c]: Producing a segmentation fault situation

In the preceding example, the `create_an_integer` function is used to create an integer. It declares an integer with a default value and returns its address to the caller. In the `main` function, the address of the created integer, `var`, is received and it gets stored in the `ptr` pointer. Then, the `ptr` pointer is dereferenced, and the value stored in the `var` variable gets printed.

But things are not that easy. When you want to compile this code using the `gcc` compiler on a Linux machine, it generates a warning as follows, but still successfully finishes the compilation, and you get the final executable:

```cpp
$ gcc ExtremeC_examples_chapter1_15.c
In function 'f':
warning: function returns address of local variable [-Wreturn-local-addr]
   return &var;
          ^~~~
$
```

Shell Box 1-8: Compiling example 1.15 in Linux

This is indeed an important warning message which can be easily missed and forgotten by the programmer. We'll talk more about this later as part of *Chapter 5*, *Stack and Heap*. Let's see what happens if we proceed and execute the resulting executable.

When you run *example 1.15*, you get a segmentation fault error, and the program crashes immediately:

```cpp
$ ./a.out
Segmentation fault (core dumped)
$
```

Shell Box 1-9: Segmentation fault when running example 1.15

So, what went wrong? The `ptr` pointer is dangling and points to an already deallocated portion of memory that was known to be the memory place of the variable, `var`.

The `var` variable is a local variable to the `create_an_integer` function, and it will be deallocated after leaving the function, but its address can be returned from the function. So, after copying the returned address into `ptr` as part of the `main` function, `ptr` becomes a dangling pointer pointing to an invalid address in memory. Now, dereferencing the pointer causes a serious problem and the program crashes.

If you look back at the warning generated by the compiler, it is clearly stating the problem.

It says that you are *returning the address of a local variable*, which will be deallocated after returning from the function. Smart compiler! If you take these warnings seriously, you won't face these scary bugs.

But what is the proper way to rewrite the example? Yes, using the *Heap memory*. We will cover heap memory fully in *Chapter 4*, *Process Memory Structure*, and *Chapter 5*, *Stack and Heap*, but, for now, we will rewrite the example using *Heap allocation*, and you will see how you can benefit from using *Heap* instead of the *Stack*.

*Example 1.16* below shows how to use Heap memory for allocating variables, and enabling the passing addresses between functions without facing any issues:

```cpp
#include <stdio.h>
#include <stdlib.h>
int* create_an_integer(int default_value) {
  int* var_ptr = (int*)malloc(sizeof(int));
  *var_ptr = default_value;
  return var_ptr;
}
int main() {
  int* ptr = NULL;
  ptr = create_an_integer(10);
  printf("%d\n", *ptr);
  free(ptr);
  return 0;
}
```

Code Box 1-28 [ExtremeC_examples_chapter1_16.c]: Rewriting example 1.15 using Heap memory

As you see in the preceding code box, we have included a new header file, `stdlib.h`, and we are using two new functions, `malloc` and `free`. The simple explanation is like this: the created integer variable inside the `create_an_integer` function is not a local variable anymore. Instead, it is a variable allocated from the Heap memory and its lifetime is not limited to the function declaring it. Therefore, it can be accessed in the caller (outer) function. The pointers pointing to this variable are not dangling anymore, and they can be dereferenced as long as the variable exists and is not freed. Eventually, the variable becomes deallocated by calling the `free` function as an end to its lifetime. Note that deallocating a Heap variable is mandatory when it is not needed anymore.

In this section, we went through all the essential discussions regarding variable pointers. In the upcoming section, we'll be talking about functions and their anatomy in C.

# Some details about functions

C is a *procedural* programming language. In C, functions act as procedures, and they are building blocks of a C program. So, it is important to know what they are, how they behave, and what is happening when you enter or leave a function. In general, functions (or procedures) are analogous to ordinary variables that store algorithms instead of values. By putting variables and functions together into a new type, we can store relevant values and algorithms under the same concept. This is what we do in *object-oriented programming*, and it will be covered in the third part of the book, *Object Orientation*. In this section, we want to explore functions and discuss their properties in C.

## Anatomy of a function

In this section, we want to recap everything about a C function in a single place. If you feel this is familiar to you, you can simply skip this section.

A function is a box of logic that has a name, a list of input parameters, and a list of output results. In C and many other programming languages that are influenced by C, functions return only one value. In object-oriented languages such as C++ and Java, functions (which are usually called *methods*) can also throw an *exception*, which is not the case for C. Functions are invoked by a function call, which is simply using the name of the function to execute its logic. A correct function call should pass all required arguments to the function and wait for its execution. Note that functions are always *blocking* in C. This means that the caller has to wait for the called function to finish and only then can it collect the returned result.

Opposite to a blocking function, we can have a *non-blocking* function. When calling a non-blocking function, the caller doesn't wait for the function to finish and it can continue its execution. In this scheme, there is usually a *callback* mechanism which is triggered when the called (or callee) function is finished. A non-blocking function can also be referred to as an *asynchronous function* or simply an *async function*. Since we don't have async functions in C, we need to implement them using multithreading solutions. We will explain these concepts in more detail in the fifth part of the book, *Concurrency*.

It is interesting to add that nowadays, there is a growing interest in using non-blocking functions over blocking functions. It is usually referred to as *event-oriented programming*. Non-blocking functions are centric in this programming approach, and most of the written functions are non-blocking.

In event-oriented programming, actual function calls happen inside an *event loop*, and proper callbacks are triggered upon the occurrence of an event. Frameworks such as `libuv` and `libev` promote this way of coding, and they allow you to design your software around one or several event loops.

## Importance in design

Functions are fundamental building blocks of procedural programming. Since their official support in programming languages, they have had a huge impact on the way we write code. Using functions, we can store logic in semi-variable entities and summon them whenever and wherever they are needed. Using them, we can write a specific logic only once and use it multiple times in various places.

In addition, functions allow us to hide a piece of logic from other existing logic. In other words, they introduce a level of abstraction between various logical components. To give an example, suppose that you have a function, `avg`, which calculates the average of an input array. And you have another function, `main`, which calls the function, `avg`. We say that the logic inside the `avg` function is hidden from the logic inside the `main` function.

Therefore, if you want to change the logic inside `avg`, you don't need to change the logic inside the `main` function. That's because the `main` function only depends on the name and the availability of the `avg` function. This is a great achievement, at least for those years when we had to use punched cards to write and execute programs!

We are still using this feature in designing libraries written in C or even higher-level programming languages such as C++ and Java.

## Stack management

If you look at the memory layout of a process running in a Unix-like operating system, you notice that all of the processes share a similar layout. We will discuss this layout in more detail in *Chapter 4*, *Process Memory Structure*, but for now, we want to introduce one of its *segments*; the Stack segment. The Stack segment is the default memory location where all local variables, arrays, and structures are allocated from. So, when you declare a local variable in a function, it is being allocated from the Stack segment. This allocation always happens on top of the Stack segment.

Notice the term *stack* in the name of the segment. It means that this segment behaves like a stack. The variables and arrays are always allocated on top of it, and those at the top are the first variables to get removed. Remember this analogy with the stack concept. We will return to this in the next paragraph.

The Stack segment is also used for function calls. When you call a function, a *stack frame* containing the return address and all of the passing arguments is put on top of the Stack segment, and only then is the function logic executed. When returning from the function, the stack frame is popped out, and the instruction addressed by the return address gets executed, which should usually continue the caller function.

All local variables declared in the function body are put on top of the Stack segment. So, when leaving the function, all Stack variables become freed. That is why we call them *local variables* and that is why a function cannot access the variables in another function. This mechanism also explains why local variables are not defined before entering a function and after leaving it.

Understanding the Stack segment and the way it works is crucial to writing correct and meaningful code. It also prevents common memory bugs from occurring. It is also a reminder that you cannot create any variable on the Stack with any size you like. The Stack is a limited portion of memory, and you could fill it up and potentially receive a *stack overflow* error. This usually happens when we have too many function calls consuming up all the Stack segment by their stack frames. This is very common when dealing with recursive functions, when a function calls itself without any break condition or limit.

## Pass-by-value versus pass-by-reference

In most computer programming books, there is a section dedicated to pass-by-value and pass-by-reference regarding the arguments passed to a function. Fortunately, or unfortunately, we have only pass-by-value in C.

There is no reference in C, so there is no pass-by-reference either. Everything is copied into the function's local variables, and you cannot read or modify them after leaving a function.

Despite the many examples that seem to demonstrate pass-by-reference function calls, I should say that passing by reference is an illusion in C. In the rest of this section, we want to uncover this illusion and convince you that those examples are also pass-by-value. The following example will demonstrate this:

```cpp
#include <stdio.h>
void func(int a) {
  a = 5;
}
int main(int argc, char** argv) {
  int x = 3;
  printf("Before function call: %d\n", x);
  func(x);
  printf("After function call: %d\n", x);
  return 0;
}
```

Code Box 1-29 [ExtremeC_examples_chapter1_17.c]: An example of a pass-by-value function call

It is easy to predict the output. Nothing changes about the `x` variable because it is passed by value. The following shell box shows the output of *example 1.17* and confirms our prediction:

```cpp
$ gcc ExtremeC_examples_chapter1_17.c
$ ./a.out
Before function call: 3
After function call: 3
$
```

Shell Box 1-10: Output of example 1.17

The following example, *example 1.18*, demonstrates that passing by reference doesn't exist in C:

```cpp
#include <stdio.h>
void func(int* a) {
  int b = 9;
  *a = 5;
  a = &b;
}
int main(int argc, char** argv) {
  int x = 3;
  int* xptr = &x;
  printf("Value before call: %d\n", x);
  printf("Pointer before function call: %p\n", (void*)xptr);
  func(xptr);
  printf("Value after call: %d\n", x);
  printf("Pointer after function call: %p\n", (void*)xptr);
  return 0;
}
```

Code Box 1-30 [ExtremeC_examples_chapter1_18.c]: An example of pass-by-pointer function call which differs from pass-by-reference

And this is the output:

```cpp
$ gcc ExtremeC_examples_chapter1_18.c
$ ./a.out
The value before the call: 3
Pointer before function call: 0x7ffee99a88ec
The value after the call: 5
Pointer after function call: 0x7ffee99a88ec
$
```

Shell Box 1-11: Output of example 1.18

As you see, the value of the pointer is not changed after the function call. This means that the pointer is passed as a pass-by-value argument. Dereferencing the pointer inside the `func` function has allowed accessing the variable where the pointer is pointing to. But you see that changing the value of the pointer parameter inside the function doesn't change its counterpart argument in the caller function. During a function call in C, all arguments are passed by value and dereferencing the pointers allows the modification of the caller function's variables.

It is worth adding that the above example demonstrates a pass-by-pointer example in which we pass pointers to variables instead of passing them directly. It is usually recommended to use pointers as arguments instead of passing big objects to a function but why? It is easy to guess. Copying 8 bytes of a pointer argument is much more efficient than copying hundreds of bytes of a big object.

Surprisingly, passing the pointer is not efficient in the above example! That's because of the fact that the `int` type is 4 bytes and copying it is more efficient than copying 8 bytes of its pointer. But this is not the case regarding structures and arrays. Since copying structures and arrays is done byte-wise, and all of the bytes in them should be copied one by one, it is usually better to pass pointers instead.

Now that we've covered some details regarding the functions in C, let's talk about function pointers.

# Function pointers

Having function pointers is another super feature of the C programming language. The two previous sections were about variable pointers and functions, and this section is going to combine them and talk about a more interesting topic: pointers to functions.

They have many applications, but splitting a large binary into smaller binaries and loading them again in another small executable is one of the most important applications. This has led to *modularization* and software design. Function pointers are building blocks for the implementation of polymorphism in C++ and allow us to extend our existing logic. In this section, we are going to cover them and prepare you for more advanced topics we'll cover over the coming chapters.

Like a variable pointer addressing a variable, a function pointer addresses a function and allows you to call that function indirectly. The following example, *example 1.19*, can be a good starter for this topic:

```cpp
#include <stdio.h>
int sum(int a, int b) {
  return a + b;
}
int subtract(int a, int b) {
  return a - b;
}
int main() {
  int (*func_ptr)(int, int);
  func_ptr = NULL;
  func_ptr = &sum;
  int result = func_ptr(5, 4);
  printf("Sum: %d\n", result);
  func_ptr = &subtract;
  result = func_ptr(5, 4);
  printf("Subtract: %d\n", result);
  return 0;
}
```

Code Box 1-31 [ExtremeC_examples_chapter1_19.c]: Using a single function pointer to call different functions

In the preceding code box, `func_ptr` is a function pointer. It can only point to a specific class of functions that match its signature. The signature limits the pointer to only point to functions that accept two integer arguments and return an integer result.

As you see, we have defined two functions called `sum` and `subtract` matching the `func_ptr` pointer's signature. The preceding example uses the `func_ptr` function pointer to point to the `sum` and `subtract` functions separately, then call them with the same arguments and compare the results. This is the output of the example:

```cpp
$ gcc ExtremeC_examples_chapter1_19.c
$ ./a.out
Sum: 9
Subtract: 1
$
```

Shell Box 1-12: Output of example 1.19

As you see in *example 1.19*, we can call different functions for the same list of arguments using a single function pointer, and this is an important feature. If you are familiar with object-oriented programming, the first thing that comes to mind is *polymorphism* and *virtual functions*. In fact, this is the only way to support polymorphism in C and mimic the C++ virtual functions. We will cover OOP as part of the third part of the book, *Object* *Orientation*.

Like variable pointers, it is important to initialize function pointers properly. For those function pointers which are not going to be initialized immediately upon declaration, it is mandatory to make them null. The nullification of function pointers is demonstrated in the preceding example, and it is pretty similar to variable pointers.

It is usually advised to define a new *type alias* for function pointers. The following example, *example 1.20*, demonstrates the way it should be done:

```cpp
#include <stdio.h>
typedef int bool_t;
typedef bool_t (*less_than_func_t)(int, int);
bool_t less_than(int a, int b) {
  return a < b ? 1 : 0;
}
bool_t less_than_modular(int a, int b) {
  return (a % 5) < (b % 5) ? 1 : 0;
}
int main(int argc, char** argv) {
  less_than_func_t func_ptr = NULL;
  func_ptr = &less_than;
  bool_t result = func_ptr(3, 7);
  printf("%d\n", result);
  func_ptr = &less_than_modular;
  result = func_ptr(3, 7);
  printf("%d\n", result);
  return 0;
}
```

Code Box 1-32 [ExtremeC_examples_chapter1_20.c]: Using a single function pointer to call different functions

The `typedef` keyword allows you to define an alias for an already defined type. There are two new type aliases in the preceding example: `bool_t`, which is an alias for the `int` type, and the `less_than_func_t` type, which is an alias type for the function pointer type, `bool_t (*)(int, int)`. These aliases add readability to the code and let you choose a shorter name for a long and complex type. In C, the name of a new type usually ends with `_t` by convention, and you can find this convention in many other standard type aliases such as `size_t` and `time_t`.

# Structures

From the design perspective, structures are one of the most fundamental concepts in C. Nowadays, they are not unique to C, and you can find their twin concepts nearly in every modern programming language.

But we should discuss them in the history of computation when there were no other programming languages offering such a concept. Among many efforts to move away from machine-level programming languages, introducing structures was a great step toward having *encapsulation* in a programming language. For thousands of years, the way we think hasn't changed a lot, and encapsulation has been a centric means for our logical reasoning.

But it was just after C that we finally had some tool, in this case, a programming language, which was able to understand the way we think and could store and process the building blocks of our reasoning. Finally, we got a language that resembles our thoughts and ideas, and all of this happened when we got structures. C structures weren't perfect in comparison to the encapsulation mechanisms found in modern languages, but they were enough for us to build a platform upon which to create our finest tools.

## Why structures?

You know that every programming language has some **Primitive Data Types** (**PDTs**). Using these PDTs, you can design your data structures and write your algorithms around them. These PDTs are part of the programming language, and they cannot be changed or removed. As an example, you cannot have C without the primitive types, `int` and `double`.

Structures come into play when you need to have your own defined data types, and the data types in the language are not enough. **User-Defined Types** (**UDTs**) are those types which are created by the user and they are not part of the language.

Note that UDTs are different from the types you could define using `typedef`. The keyword `typedef` doesn't really create a new type, but rather it defines an alias or synonym for an already defined type. But structures allow you to introduce totally new UDTs into your program.

Structures have twin concepts in other programming languages, for example, classes in C++ and Java or packages in Perl. They are considered to be the *type-makers* in these languages.

## Why user-defined types?

So, why do we need to create new types in a program? The answer to this question reveals the principles behind software design and the methods we use for our daily software development. We create new types because we do it every day using our brains in a routine analysis.

We don't look at our surroundings as integers, doubles, or characters. We have learned to group related attributes under the same object. We will discuss more the way we analyze our surroundings in *Chapter 6*, *OOP and Encapsulation*. But as an answer to our starting question, we need new types because we use them to analyze our problems at a higher level of logic, close enough to our human logic.

Here, you need to become familiar with the term *business logic*. Business logic is a set of all entities and regulations found in a business. For example, in the business logic of a banking system, you face concepts such as client, account, balance, money, cash, payment, and many more, which are there to make operations such as money withdrawal possible and meaningful.

Suppose that you had to explain some banking logic in pure integers, floats, or characters. It is almost impossible. If it is possible for programmers, it is almost meaningless to *business analysts*. In a real software development environment that has a well-defined business logic, programmers and business analysts cooperate closely. Therefore, they need to have a shared set of terminology, glossary, types, operations, regulations, logic, and so on.

Today, a programming language that does not support new types in its *type system* can be considered as a dead language. Maybe that's why most people see C as a dead programming language, mainly because they cannot easily define their new types in C, and they prefer to move to a higher-level language such as C++ or Java. Yes, it's not that easy to create a nice type system in C, but everything you need is present there.

Even today, there can be many reasons behind choosing C as the project's main language and accepting the efforts of creating and maintaining a nice type system in a C project and even today many companies do that.

Despite the fact that we need new types in our daily software analysis, CPUs do not understand these new types. CPUs try to stick to the PDTs and fast calculations because they are designed to do that. So, if you have a program written in your high-level language, it should be translated to CPU level instructions, and this may cost you more time and resources.

In this sense, fortunately, C is not very far away from the CPU-level logic, and it has a type system which can be easily translated. You may have heard that C is a low-level or hardware-level programming language. This is one of the reasons why some companies and organizations try to write and maintain their core frameworks in C, even today.

## What do structures do?

Structures encapsulate related values under a single unified type. As an early example, we can group `red`, `green`, and `blue` variables under a new single data type called `color_t`. The new type, `color_t`, can represent an RGB color in various programs like an image editing application. We can define the corresponding C structure as follows:

```cpp
struct color_t {
  int red;
  int green;
  int blue;
};
```

Code Box 1-33: A structure in C representing an RGB color

As we said before, structures do encapsulation. Encapsulation is one of the most fundamental concepts in software design. It is about grouping and encapsulating related fields under a new type. Then, we can use this new type to define the required variables. We will describe encapsulation thoroughly in *Chapter 6*, *OOP and Encapsulation*, while talking about object-oriented design.

Note that we use an `_t` suffix for naming new data types.

## Memory layout

It is usually important to C programmers to know exactly the memory layout of a structure variable. Having a bad layout in memory could cause performance degradations in certain architectures. Don't forget that we code to produce the instructions for the CPU. The values are stored in the memory, and the CPU should be able to read and write them fast enough. Knowing the memory layout helps a developer to understand the way the CPU works and to adjust their code to gain a better result.

The following example, *example 1.21*, defines a new structure type, `sample_t`, and declares one structure variable, `var`. Then, it populates its fields with some values and prints the size and the actual bytes of the variable in the memory. This way, we can observe the memory layout of the variable:

```cpp
#include <stdio.h>
struct sample_t {
  char first;
  char second;
  char third;
  short fourth;
};
void print_size(struct sample_t* var) {
  printf("Size: %lu bytes\n", sizeof(*var));
}
void print_bytes(struct sample_t* var) {
  unsigned char* ptr = (unsigned char*)var;
  for (int i = 0; i < sizeof(*var); i++, ptr++) {
    printf("%d ", (unsigned int)*ptr);
  }
  printf("\n");
}
int main(int argc, char** argv) {
  struct sample_t var;
  var.first = 'A';
  var.second = 'B';
  var.third = 'C';
  var.fourth = 765;
  print_size(&var);
  print_bytes(&var);
  return 0;
}
```

Code Box 1-34 [ExtremeC_examples_chapter1_21.c]: Printing the number of the bytes allocated for a structure variable

The thirst to know the exact memory layout of everything is a bit C/C++ specific, and vanishes as the programming languages become high level. For example, in Java and Python, the programmers tend to know less about the very low-level memory management details, and on the other hand, these languages don't provide many details about the memory.

As you see in *Code Box 1-34*, in C, you have to use the `struct` keyword before declaring a structure variable. Therefore, in the preceding example we have `struct sample_t var`, which shows how you should use the keyword before the structure type in the declaration clause. It is trivial to mention that you need to use a `.` (dot) to access the fields of a structure variable. If it is a structure pointer, you need to use `->` (arrow) to access its fields.

In order to prevent typing a lot of `struct`s throughout the code, while defining a new structure type and while declaring a new structure variable, we could use `typedef` to define a new alias type for the structure. Following is an example:

```cpp
typedef struct {
  char first;
  char second;
  char third;
  short fourth;
} sample_t;
```

Now, you can declare the variable without using the keyword `struct`:

```cpp
sample_t var;
```

The following is the output of the preceding example after being compiled and executed on a macOS machine. Note that the numbers generated may vary depending upon the host system:

```cpp
$ clang ExtremeC_examples_chapter1_21.c
$ ./a.out
Size: 6 bytes
65 66 67 0 253 2
$
```

Shell Box 1-13: Output of example 1.21

As you see in the preceding shell box, `sizeof(sample_t)` has returned 6 bytes. The memory layout of a structure variable is very similar to an array. In an array, all elements are adjacent to each other in the memory, and this is the same for a structure variable and its field. The difference is that, in an array, all elements have the same type and therefore the same size, but this is not the case regarding a structure variable. Each field can have a different type, and hence, it can have a different size. Unlike an array, the memory size of which is easily calculated, the size of a structure variable in the memory depends on a few factors and cannot be easily determined.

At first, it seems to be easy to guess the size of a structure variable. For the structure in the preceding example, it has four fields, three `char` fields, and one `short` field. With a simple calculation, if we suppose that `sizeof(char)` is 1 byte and `sizeof(short)` is 2 bytes, each variable of the type `sample_t` should have 5 bytes in its memory layout. But when we look at the output, we see that `sizeof(sample_t)` is 6 bytes. 1 byte more! Why do we have this extra byte? Again, while looking at the bytes in the memory layout of the structure variable, `var`, we can see that it is a bit different from our expectation which is `65 66 67 253 2`.

For making this clearer and explaining why the size of the structure variable is not 5 bytes, we need to introduce the *memory alignment* concept. The CPU always does all the computations. Besides that, it needs to load values from memory before being able to compute anything and needs to store the results back again in the memory after a computation. Computation is super-fast inside the CPU, but the memory access is very slow in comparison. It is important to know how the CPU interacts with the memory because then we can use the knowledge to boost a program or debug an issue.

The CPU usually reads a specific number of bytes in each memory access. This number of bytes is usually called a *word*. So, the memory is split into words and a word is an atomic unit used by the CPU to read from and write to the memory. The actual number of bytes in a word is an architecture-dependent factor. For example, in most 64-bit machines, the word size is 32 bits or 4 bytes. Regarding the memory alignment, we say that a variable is aligned in the memory if its starting byte is at the beginning of a word. This way, the CPU can load its value in an optimized number of memory accesses.

Regarding the previous example, *example 1.21*, the first 3 fields, `first`, `second`, and `third`, are 1 byte each, and they reside in the first word of the structure's layout, and they all can be read by just one memory access. About the fourth field, `fourth` occupies 2 bytes. If we forget about the memory alignment, its first byte will be the last byte of the first word, which makes it unaligned.

If this was the case, the CPU would be required to make two memory accesses together with shifting some bits in order to retrieve the value of the field. That is why we see an extra zero after byte `67`. The zero byte has been added in order to complete the current word and let the fourth field start in the next word. Here, we say that the first word is padded by one zero byte. The compiler uses the *padding* technique to align values in the memory. Padding is the extra bytes added to match the alignment.

It is possible to turn off the alignment. In C terminology, we use a more specific term for aligned structures. We say that the structure is not packed. *Packed structures* are not aligned and using them may lead to binary incompatibilities and performance degradation. You can easily define a structure that is packed. We will do it in the next example, *example 1.22*, which is pretty similar to the previous example, *example 1.21*. The `sample_t` structure is packed in this example. The following code box shows *example 1.22*. Note that the similar code are replaced by ellipses:

```cpp
#include <stdio.h>
struct __attribute__((__packed__)) sample_t {
  char first;
  char second;
  char third;
  short fourth;
} ;
void print_size(struct sample_t* var) {
  // ...
}
void print_bytes(struct sample_t* var) {
  // ...
}
int main(int argc, char** argv) {
  // ...
}
```

Code Box 1-35 [ExtremeC_examples_chapter1_22.c]: Declaring a packed structure

In the following shell box, the preceding code is compiled using `clang` and run on macOS:

```cpp
$ clang ExtremeC_examples_chapter1_22.c
$ ./a.out
Size: 5 bytes
65 66 67 253 2
$
```

Shell Box 1-14: Output of example 1.22

As you see in *Shell Box 1-14*, the printed size is exactly what we were expecting as part of *example 1.21*. The final layout is also matched with our expectation. Packed structures are usually used in memory-constrained environments, but they can have a huge negative impact on the performance on most architectures. Only new CPUs can handle reading an unaligned value from multiple words without enforcing extra cost. Note that memory alignment is enabled by default.

## Nested structures

As we have explained in the previous sections, in general, we have two kinds of data types in C. There are the types that are primitive to the language and there are types which are defined by the programmers using the `struct` keyword. The former types are PDTs, and the latter are UDTs.

So far, our structure examples have been about UDTs (structures) made up of only PDTs. But in this section, we are going to give an example of UDTs (structures) that are made from other UDTs (structures). These are called *complex data types*, which are the result of nesting a few structures.

Let's begin with the example, *example 1.23*:

```cpp
typedef struct {
  int x;
  int y;
} point_t;
typedef struct {
  point_t center;
  int radius;
} circle_t;
typedef struct {
  point_t start;
  point_t end;
} line_t;
```

Code Box 1-36 [ExtremeC_examples_chapter1_23.c]: Declaring some nested structures

In the preceding code box, we have three structures; `point_t`, `circle_t`, and `line_t`. The `point_t` structure is a simple UDT because it is made up of only PDTs, but other structures contain a variable of the `point_t` type, which makes them complex UDTs.

The size of a complex structure is calculated exactly the same as a simple structure, by summing up the sizes of all its fields. We should be still careful about the alignment, of course, because it can affect the size of a complex structure. So, `sizeof(point_t)` would be 8 bytes if `sizeof(int)` is 4 bytes. Then, `sizeof(circle_t)` is 12 bytes and `sizeof(line_t)` is 16 bytes.

It is common to call structure variables objects. They are exactly analogous to objects in object-oriented programming, and we will see that they can encapsulate both values and functions. So, it is not wrong at all to call them C objects.

## Structure pointers

Like pointers to PDTs, we can have pointers to UDTs as well. They work exactly the same as PDT pointers. They point to an address in memory, and you can do arithmetic on them just like with the PDT pointers. UDT pointers also have arithmetic step sizes equivalent to the size of the UDT. If you don't know anything about the pointers or the allowed arithmetic operations on them, please go to the *Pointers* section and give it a read.

It is important to know that a structure variable points to the address of the first field of the structure variable. In the previous example, *example 1.23*, a pointer of type `point_t` would point to the address of its first field, `x`. This is also true for the type, `circle_t`. A pointer of type `circle_t` would point to its first field, `center`, and since it is actually a `point_t` object, it would point to the address of the first field, `x`, in the `point_t` type. Therefore, we can have 3 different pointers addressing the same cell in the memory. The following code will demonstrate this:

```cpp
#include <stdio.h>
typedef struct {
  int x;
  int y;
} point_t;
typedef struct {
  point_t center;
  int radius;
} circle_t;
int main(int argc, char** argv) {
  circle_t c;
  circle_t* p1 = &c;
  point_t*  p2 = (point_t*)&c;
  int*      p3 = (int*)&c;
  printf("p1: %p\n", (void*)p1);
  printf("p2: %p\n", (void*)p2);
  printf("p3: %p\n", (void*)p3);
  return 0;
}
```

Code Box 1-37 [ExtremeC_examples_chapter1_24.c]: Having three different pointers from three different types addressing the same byte in memory

And this is the output:

```cpp
$ clang ExtremeC_examples_chapter1_24.c
$ ./a.out
p1: 0x7ffee846c8e0
p2: 0x7ffee846c8e0
p3: 0x7ffee846c8e0
$
```

Shell Box 1-15: Output of example 1.24

As you see, all of the pointers are addressing the same byte, but their types are different. This is usually used to extend structures coming from other libraries by adding more fields. This is also the way we implement *inheritance* in C. We will discuss this in *Chapter 8*, *Inheritance and Polymorphism*.

This was the last section in this chapter. In the upcoming chapter, we will dive into the C compilation pipeline and how to properly compile and link a C project.

# Summary

In this chapter, we revisited some of the important features of the C programming language. We tried to go further and show the design aspects of these features and the concepts behind them. Of course, the proper use of a feature requires a deeper insight into the different aspects of that feature. As part of this chapter, we discussed the following:

*   We talked about the C preprocessing phase and how various directives can influence the preprocessor to act differently or generate a specific C code for us.
*   Macros and the macro expansion mechanism allow us to generate C code before passing the translation unit to the compilation phase.
*   Conditional directives allow us to alter the preprocessed code based on certain conditions and allow us to have different code for different situations.
*   We also looked at variable pointers, and how they are employed in C.
*   We introduced generic pointers and how we can have a function that accepts any kind of pointer.
*   We discussed some issues such as segmentation faults and dangling pointers to show a few disastrous situations that can arise from misusing pointers.
*   Functions were discussed next, and we reviewed their syntax.
*   We explored functions' design aspects and how they contribute to a nicely shaped procedural C program.
*   We also explained the function call mechanism and how arguments are passed to a function using stack frames.
*   Function pointers were explored in this chapter. The powerful syntax of function pointers allows us to store logics in variable-like entities and use them later. They are, in fact, the fundamental mechanism that every single program uses today to be loaded and operate.
*   Structures together with function pointers gave rise to encapsulation in C. We speak more about this in the third part of the book, *Object Orientation*.
*   We tried to explain the design aspects of structures and their effect on the way we design programs in C.
*   We also discussed the memory layout of structure variables and how they are placed inside memory to maximize CPU utilization.
*   Nested structures were also discussed. We also took a look inside the complex structure variables and discussed how their memory layout should look.
*   As the final section in this chapter, we talked about structure pointers.

The next chapter will be our first step in building a C project. The C compilation pipeline and linking mechanism will be discussed as part of the next chapter. Reading it thoroughly will be essential to continue with the book and proceed to further chapters.