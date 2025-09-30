# Chapter 21

# Integration with Other Languages

Knowing how to write a C program or library can be more valuable than you might expect. Due to the important role of C in developing operating systems, C is not limited to its own world. C libraries have the potential to be loaded and used in other programming languages as well. While you are reaping the benefits of writing code in a higher-level programming language, you can have the rocket power of C as a loaded library inside your language environment.

In this chapter we are going to talk more about this, and demonstrate how C shared libraries can be integrated with some well-known programming languages.

In this chapter, we will cover the following key topics:

*   We discuss why integration is possible in the first place. The discussion is important because it gives you the basic idea of how integration works.
*   We design a C stack library. We build it as a shared object file. This shared object file is going to be used by a number of other programming languages.
*   We go through C++, Java, Python, and Golang and see how the stack library can be loaded first and then used.

As a general note in this chapter, since we are going to work on five different subprojects, each having different programming languages, we only present the builds for Linux in order to prevent any issues regarding the builds and executions. Of course, we give enough information about the macOS system, but our focus is to build and run sources on Linux. Further scripts are available in the book's GitHub repository that help you to build the sources for macOS.

The first section talks about the integration itself. We see why the integration with other programming languages is possible and it makes a basis for expanding our discussion within other environments rather than C.

# Why integration is possible?

As we have explained in *Chapter 10*, *Unix – History and Architecture*, C revolutionized the way we were developing operating systems. That's not the only magic of C; it also gave us the power to build other general-purpose programming languages on top of it. Nowadays, we call them higher-level programming languages. The compilers of these languages are mostly written in C and if not, they've been developed by other tools and compilers written in C.

A general-purpose programming language that is not able to use or provide the functionalities of a system is not doing anything at all. You can write things with it, but you cannot execute it on any system. While there could be usages for such a programming language from a theoretical point of view, certainly it is not plausible from an industrial point of view. Therefore, the programming language, especially through its compiler, should be able to produce programs that work. As you know, the functionalities of a system are exposed through the operating system. Regardless of the operating system itself, a programming language should be able to provide those functionalities, and the programs written in that language, and being run on that system, should be able to use them.

This is where C comes in. In Unix-like operating systems, the C standard library provides the API to use the available functionalities of the system. If a compiler wants to create a working program, it should be able to allow the compiled program to use the C standard library in an indirect fashion. No matter what the programming language is and whether it offers some specific and native standard library, like Java, which offers **Java Standard Edition** (**Java SE**), any request for a specific functionality made by the written program (such as opening a file) should be passed down to the C standard library and from there, it can reach the kernel and get performed.

As an example, let's talk a bit more about Java. Java programs are compiled to an intermediate language called *bytecode*. In order to execute a Java bytecode, one needs to have **Java Runtime Environment** (**JRE**) installed. JRE has a virtual machine at its heart that loads the Java bytecode and runs it within itself. This virtual machine must be able to simulate the functionalities and services exposed by the C standard library and provide them to the program running within. Since every platform can be different in terms of the C standard library and its compliance with POSIX and SUS standards, we need to have some virtual machines built specifically for each platform.

As a final note about the libraries that can be loaded in other languages, we can only load shared object files and it is not possible to load and use static libraries. Static libraries can only be linked to an executable or a shared object file. Shared object files have the `.so` extension in most Unix-like systems but they have the `.dylib` extension in macOS.

In this section, despite its short length, I tried to give you a basic idea of why we are able to load C libraries, shared libraries specifically, and how most programming languages are already using C libraries, since the ability to load a shared object library and use it exists in most of them.

The next step would be writing a C library and then loading it in various programming languages in order to use it. That's exactly what we want to do soon but before that you need to know how to get the chapter material and how to run the commands seen in the shell boxes.

# Obtaining the necessary materials

Since this chapter is full of sources from five different programming languages, and my hope is to have you all able to build and run the examples, I dedicated this section to going through some basic notes that you should be aware of regarding building the source code.

First of all, you need to obtain the chapter material. As you should know by now, the book has a repository in which this chapter has a specific directory named `ch21-integration-with-other-languages`. The following commands show you how to clone the repository and change to the chapter's root directory:

```cpp
$ git clone https://github.com/PacktPublishing/Extreme-C.git
...
$ cd Extreme-C/ch21-integration-with-other-languages
$
```

Shell Code 21-1: Cloning the book's GitHub repository and changing to the chapter's root directory

Regarding the shell boxes in this chapter, we assume that before executing the commands in a shell box, we are located in the root of the chapter, in the `ch21-integration-with-other-languages` folder. If we needed to change to other directories, we provide the required commands for that, but everything is happening inside the chapter's directory.

In addition, in order to be able to build source code, you need to have **Java Development Kit** (**JDK**), Python, and Golang installed on your machine. Depending on whether you're using Linux or macOS, and on your Linux distribution, the installation commands can be different.

As the final note, the source code written in other languages than C should be able to use the C stack library that we discuss in the upcoming section. Building those sources requires that you've already built the C library. Therefore, make sure that you read the following section first and have its shared object library built before moving on to the next sections. Now that you know how to obtain the chapter's material, we can proceed to discuss our target C library.

# Stack library

In this section, we are going to write a small library that is going to be loaded and used by programs written in other programming languages. The library is about a Stack class that offers some basic operations like *push* or *pop* on stack objects. Stack objects are created and destroyed by the library itself and there is a constructor function, as well as a destructor function, to fulfill this purpose.

Next, you can find the library's public interface, which exists as part of the `cstack.h` header file:

```cpp
#ifndef _CSTACK_H_
#define _CSTACK_H_
#include <unistd.h>
#ifdef __cplusplus
extern "C" {
#endif
#define TRUE 1
#define FALSE 0
typedef int bool_t;
typedef struct {
  char* data;
  size_t len;
} value_t;
typedef struct cstack_type cstack_t;
typedef void (*deleter_t)(value_t* value);
value_t make_value(char* data, size_t len);
value_t copy_value(char* data, size_t len);
void free_value(value_t* value);
cstack_t* cstack_new();
void cstack_delete(cstack_t*);
// Behavior functions
void cstack_ctor(cstack_t*, size_t);
void cstack_dtor(cstack_t*, deleter_t);
size_t cstack_size(const cstack_t*);
bool_t cstack_push(cstack_t*, value_t value);
bool_t cstack_pop(cstack_t*, value_t* value);
void cstack_clear(cstack_t*, deleter_t);
#ifdef __cplusplus
}
#endif
#endif
```

Code Box 21-1 [cstack.h]: The public interface of the Stack library

As we have explained in *Chapter 6*, *OOP and Encapsulation*, the preceding declarations introduce the public interface of the Stack class. As you see, the companion attribute structure of the class is `cstack_t`. We have used `cstack_t` instead of `stack_t` because the latter is used in the C standard library and I prefer to avoid any ambiguity in this code. By the preceding declarations, the attribute structure is forward declared and has no fields in it. Instead, the details will come in the source file that does the actual implementation. The class also has a constructor, a destructor, and some other behaviors such as push and pop. As you can see, all of them accept a pointer of type `cstack_t` as their first argument that indicates the object they should act on. The way we wrote the Stack class is explained as part of *implicit encapsulation* in *Chapter 6*, *OOP and Encapsulation*.

*Code Box 21-2* contains the implementation of the stack class. It also contains the actual definition for the `cstack_t` attribute structure:

```cpp
#include <stdlib.h>
#include <assert.h>
#include "cstack.h"
struct cstack_type {
  size_t top;
  size_t max_size;
  value_t* values;
};
value_t copy_value(char* data, size_t len) {
  char* buf = (char*)malloc(len * sizeof(char));
  for (size_t i = 0; i < len; i++) {
    buf[i] = data[i];
  }
  return make_value(buf, len);
}
value_t make_value(char* data, size_t len) {
  value_t value;
  value.data = data;
  value.len = len;
  return value;
}
void free_value(value_t* value) {
  if (value) {
    if (value->data) {
      free(value->data);
      value->data = NULL;
    }
  }
}
cstack_t* cstack_new() {
  return (cstack_t*)malloc(sizeof(cstack_t));
}
void cstack_delete(cstack_t* stack) {
  free(stack);
}
void cstack_ctor(cstack_t* cstack, size_t max_size) {
  cstack->top = 0;
  cstack->max_size = max_size;
  cstack->values = (value_t*)malloc(max_size * sizeof(value_t));
}
void cstack_dtor(cstack_t* cstack, deleter_t deleter) {
  cstack_clear(cstack, deleter);
  free(cstack->values);
}
size_t cstack_size(const cstack_t* cstack) {
  return cstack->top;
}
bool_t cstack_push(cstack_t* cstack, value_t value) {
  if (cstack->top < cstack->max_size) {
    cstack->values[cstack->top++] = value;
    return TRUE;
  }
  return FALSE;
}
bool_t cstack_pop(cstack_t* cstack, value_t* value) {
  if (cstack->top > 0) {
    *value = cstack->values[--cstack->top];
    return TRUE;
  }
  return FALSE;
}
void cstack_clear(cstack_t* cstack, deleter_t deleter) {
  value_t value;
  while (cstack_size(cstack) > 0) {
    bool_t popped = cstack_pop(cstack, &value);
    assert(popped);
    if (deleter) {
      deleter(&value);
    }
  }
}
```

Code Box 21-2 [cstack.c]: The definition of the stack class

As you see, the definition implies that every stack object is backed with an array, and more than that, we can store any value in the stack. Let's build the library and produce a shared object library out of it. This would be the library file that is going to be loaded by other programming languages in the upcoming sections.

The following shell box shows how to create a shared object library using the existing source files. The commands found in the text box work in Linux and they should be slightly changed in order to work in macOS. Note that before running the build commands, you should be in this chapter's root directory as explained before:

```cpp
$ gcc -c -g -fPIC cstack.c -o cstack.o
$ gcc -shared cstack.o -o libcstack.so
$
```

Shell Box 21-2: Building the stack library and producing the shared object library file in Linux

As a side note, in macOS, we can run the preceding exact commands if the `gcc` is a known command and it is pointing to the `clang` compiler. Otherwise, we can use the following commands to build the library on macOS. Note that the extension of shared object files is .`dylib` in macOS:

```cpp
$ clang -c -g -fPIC cstack.c -o cstack.o
$ clang -dynamiclib cstack.o -o libcstack.dylib
$
```

Shell Box 21-3: Building the stack library and producing the shared object library file in macOS

We now have the shared object library file, and we can write programs in other languages that can load it. Before giving our demonstration on how the preceding library can be loaded and used in other environments, we need to write some tests in order to verify its functionality. The following code creates a stack and performs some of the available operations and checks the results against the expectations:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cstack.h"
value_t make_int(int int_value) {
  value_t value;
  int* int_ptr = (int*)malloc(sizeof(int));
  *int_ptr = int_value;
  value.data = (char*)int_ptr;
  value.len = sizeof(int);
  return value;
}
int extract_int(value_t* value) {
  return *((int*)value->data);
}
void deleter(value_t* value) {
  if (value->data) {
    free(value->data);
  }
  value->data = NULL;
}
int main(int argc, char** argv) {
  cstack_t* cstack = cstack_new();
  cstack_ctor(cstack, 100);
  assert(cstack_size(cstack) == 0);
  int int_values[] = {5, 10, 20, 30};
  for (size_t i = 0; i < 4; i++) {
    cstack_push(cstack, make_int(int_values[i]));
  }
  assert(cstack_size(cstack) == 4);
  int counter = 3;
  value_t value;
  while (cstack_size(cstack) > 0) {
    bool_t popped = cstack_pop(cstack, &value);
    assert(popped);
    assert(extract_int(&value) == int_values[counter--]);
    deleter(&value);
  }
  assert(counter == -1);
  assert(cstack_size(cstack) == 0);
  cstack_push(cstack, make_int(10));
  cstack_push(cstack, make_int(20));
  assert(cstack_size(cstack) == 2);
  cstack_clear(cstack, deleter);
  assert(cstack_size(cstack) == 0);
   // In order to have something in the stack while
  // calling destructor.
  cstack_push(cstack, make_int(20));
  cstack_dtor(cstack, deleter);
  cstack_delete(cstack);
  printf("All tests were OK.\n");
  return 0;
}
```

Code Box 21-3 [cstack_tests.c]: The code testing the functionality of the Stack class

As you can see, we have used assertions to check the returned values. The following is the output of the preceding code after being built and executed in Linux. Again, note that we are in the chapter's root directory:

```cpp
$ gcc -c -g cstack_tests.c -o tests.o
$ gcc tests.o -L$PWD -lcstack -o cstack_tests.out
$ LD_LIBRARY_PATH=$PWD ./cstack_tests.out
All tests were OK.
$
```

Shell Box 21-4: Building and running the library tests

Note that in the preceding shell box, when running the final executable file `cstack_tests.out`, we have to set the environment variable `LD_LIBRARY_PATH` to point to the directory that contains the `libcstack.so`, because the executed program needs to find the shared object libraries and load them.

As you see in *Shell Box 21-4*, all tests have passed successfully. This means that from the functional point of view, our library is performing correctly. It would be nice to check the library against a non-functional requirement like memory usage or having no memory leaks.

The following command shows how to use `valgrind` to check the execution of the tests for any possible memory leaks:

```cpp
$ LD_LIBRARY_PATH=$PWD valgrind --leak-check=full ./cstack_tests.out
==31291== Memcheck, a memory error detector
==31291== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==31291== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==31291== Command: ./cstack_tests.out
==31291==
All tests were OK.
==31291==
==31291== HEAP SUMMARY:
==31291==     in use at exit: 0 bytes in 0 blocks
==31291==   total heap usage: 10 allocs, 10 frees, 2,676 bytes allocated
==31291==
==31291== All heap blocks were freed -- no leaks are possible
==31291==
==31291== For counts of detected and suppressed errors, rerun with: -v
==31291== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
$
```

Shell Box 21-5: Running the tests using valgrind

As you can see, we don't have any memory leaks, and this gives us more trust in the library that we have written. Therefore, if we see any memory issue in another environment, the root cause should be investigated there first.

In the following chapter, we will cover unit testing in C. As a proper replacement for the `assert` statements seen in *Code Box 21-3*, we could write unit tests and use a unit testing framework like CMocka to execute them.

In the following sections, we are going to integrate the stack library in programs written by four programming languages. We'll start with C++.

# Integration with C++

Integration with C++ can be assumed as the easiest. C++ can be thought of as an object-oriented extension to C. A C++ compiler produces similar object files to those that a C compiler produces. Therefore, a C++ program can load and use a C shared object library easier than any other programming language. In other words, it doesn't matter whether a shared object file is the output of a C or C++ project; both can be consumed by a C++ program. The only thing that can be problematic in some cases is the C++ *name mangling* feature that is described in *Chapter 2*, *Compilation and Linking*. As a reminder, we'll briefly review it in the following section.

## Name mangling in C++

To elaborate more on this, we should say that symbol names corresponding to functions (both global and member functions in classes) are mangled in C++. Name mangling is mainly there to support *namespaces* and *function overloading*, which are missing in C. Name mangling is enabled by default, therefore if C code gets compiled using a C++ compiler, we expect to see mangled symbol names. Look at the following example in *Code Box 21-4*:

```cpp
int add(int a, int b) {
  return a + b;
}
```

Code Box 21-4 [test.c]: A simple function in C

If we compile the preceding file using a C compiler, in this case `clang`, we see the following symbols in the generated object file, shown in *Shell Box 21-6*. Note that the file `test.c` doesn't exist in the book's GitHub repository:

```cpp
$ clang -c test.c -o test.o
$ nm test.o
0000000000000000 T _add
$
```

Shell Box 21-6: Compiling test.c with a C compiler

As you see, we have a symbol named `_add` that refers to the function `add` defined above. Now, let's compile the file with a C++ compiler, in this case `clang++`:

```cpp
$ clang++ -c test.c -o test.o
clang: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated [-Wdeprecated]
$ nm test.o
0000000000000000 T __Z3addii
$
```

Shell Box 21-7: Compiling test.c with a C++ compiler

As you can see, `clang++` has generated a warning that says that in the near future, the support for compiling C code as C++ code will be dropped. But since this behavior is not removed yet (and it is just deprecated), we see that the symbol name generated for the preceding function is mangled and is different from the one generated by `clang`. This can definitely lead to problems in the linking phase when looking for a specific symbol.

To eliminate this issue, one needs to wrap the C code inside a special scope that prevents a C++ compiler from mangling the symbol names. Then, compiling it with `clang` and `clang++` produces the same symbol names. Look at the following code in *Code Box 21-5,* which is a changed version of the code introduced in *Code Box 21-4*:

```cpp
#ifdef __cplusplus
extern "C" {
#endif
int add(int a, int b) {
  return a + b;
}
#ifdef __cplusplus
}
#endif
```

Code Box 21-5 [test.c]: Putting the function declaration into the special C scope

The preceding function is put in the scope `extern "C" { ... }` only if the macro `__cplusplus` is already defined. Having the macro `__cplusplus` is a sign that the code is being compiled by a C++ compiler. Let's compile the preceding code with `clang++` again:

```cpp
$ clang++ -c test.c -o test.o
clang: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated [-Wdeprecated]
$ nm test.o
0000000000000000 T _add
$
```

Shell Box 21-8: Compiling the new version of test.c with clang++

As you see, the generated symbol is not mangled anymore. Regarding our stack library, based on what we explained so far, we need to put all declarations in the scope `extern "C" { … }` and this is exactly the reason behind having that scope in *Code Box 21-1*. Therefore, when linking a C++ program with the stack library, the symbols can be found inside `libcstack.so` (or `libcstack.dylib`).

**Note**:

`extern "C"` is a *linkage specification*. More information can be found via the following links:

https://isocpp.org/wiki/faq/mixing-c-and-cpp

https://stackoverflow.com/questions/1041866/what-is-the-effect-of-extern-c-in-c.

Now, it's time to write the C++ code that uses our stack library. As you'll see shortly, it's an easy integration.

## C++ code

Now that we know how to disable name mangling when bringing C code into a C++ project, we can proceed by writing a C++ program that uses the stack library. We start by wrapping the stack library in a C++ class, which is the main building block of an object-oriented C++ program. It is more appropriate to expose the stack functionality in an object-oriented fashion instead of having the stack library's C functions be called directly.

*Code Box 21-6* contains the class that wraps the stack functionality derived from the stack library:

```cpp
#include <string.h>
#include <iostream>
#include <string>
#include "cstack.h"
template<typename T>
value_t CreateValue(const T& pValue);
template<typename T>
T ExtractValue(const value_t& value);
template<typename T>
class Stack {
public:
  // Constructor
  Stack(int pMaxSize) {
    mStack = cstack_new();
    cstack_ctor(mStack, pMaxSize);
  }
  // Destructor
  ~Stack() {
    cstack_dtor(mStack, free_value);
    cstack_delete(mStack);
  }
  size_t Size() {
    return cstack_size(mStack);
  }
  void Push(const T& pItem) {
    value_t value = CreateValue(pItem);
    if (!cstack_push(mStack, value)) {
      throw "Stack is full!";
    }
  }
  const T Pop() {
    value_t value;
    if (!cstack_pop(mStack, &value)) {
      throw "Stack is empty!";
    }
    return ExtractValue<T>(value);
  }
  void Clear() {
    cstack_clear(mStack, free_value);
  }
private:
  cstack_t* mStack;
};
```

Code Box 21-6 [c++/Stack.cpp]: A C++ class that wraps the functionalities exposed by the stack library

Regarding the preceding class, we can point out the following important notes:

*   The preceding class keeps a private pointer to a `cstack_t` variable. This pointer addresses the object created by the static library's `cstack_new` function. This pointer can be thought of as a *handle* to an object that exists at the C level, created and managed by a separate C library. The pointer `mStack` is analogous to a file descriptor (or file handle) that refers to a file.
*   The class wraps all behavior functions exposed by the stack library. This is not essentially true for any object-oriented wrapper around a C library, and usually a limited set of functionalities is exposed.
*   The preceding class is a template class. This means that it can operate on a variety of data types. As you can see, we have declared two template functions for serializing and deserializing objects with various types: `CreateValue` and `ExtractValue`. The preceding class uses these functions to create a byte array from a C++ object (serialization) and to create a C++ object from a byte array (deserialization) respectively.
*   We define a specialized template function for the type `std::string`. Therefore, we can use the preceding class to store values with the `std::string` type. Note that `std::string` is the standard type in C++ for having a string variable.
*   As part of the stack library, you can have multiple values from different types pushed into a single stack instance. The value can be converted to/from a character array. Look at the `value_t` structure in *Code Box 21-1*. It only needs a `char` pointer and that's all. Unlike the stack library, the preceding C++ class is *type-safe* and every instance of it can operate only on a specific data type.
*   In C++, every class has at least one constructor and one destructor. Therefore, it would be easy to initialize the underlying stack object as part of the constructor and finalize it in the destructor. That's exactly what you see in the preceding code.

We want our C++ class to be able to operate on string values. Therefore, we need to write proper serializer and deserializer functions that can be used within the class. The following code contains the function definitions that convert a C char array to an `std::string` object and vice versa:

```cpp
template<>
value_t CreateValue(const std::string& pValue) {
  value_t value;
  value.len = pValue.size() + 1;
  value.data = new char[value.len];
  strcpy(value.data, pValue.c_str());
  return value;
}
template<>
std::string ExtractValue(const value_t& value) {
  return std::string(value.data, value.len);
}
```

Code Box 21-7 [c++/Stack.cpp]: Specialized template functions meant for serialization/deserialization of the std::string type. These functions are used as part of the C++ class.

The preceding functions are `std::string` *specialization* for the declared template function used in the class. As you can see, it defines how a `std::string` object should be converted to a C char array, and conversely how a C char array can be turned into an `std::string` object.

*Code Box 21-8* contains the `main` method that uses the C++ class:

```cpp
int main(int argc, char** argv) {
  Stack<std::string> stringStack(100);
  stringStack.Push("Hello");
  stringStack.Push("World");
  stringStack.Push("!");
  std::cout << "Stack size: " << stringStack.Size() << std::endl;
  while (stringStack.Size() > 0) {
    std::cout << "Popped > " << stringStack.Pop() << std::endl;
  }
  std::cout << "Stack size after pops: " <<
      stringStack.Size() << std::endl;
  stringStack.Push("Bye");
  stringStack.Push("Bye");
  std::cout << "Stack size before clear: " <<
      stringStack.Size() << std::endl;
  stringStack.Clear();
  std::cout << "Stack size after clear: " <<
      stringStack.Size() << std::endl;
  return 0;
}
```

Code Box 21-8 [c++/Stack.cpp]: The main function using the C++ stack class

The preceding scenario covers all the functions exposed by the stack library. We execute a number of operations and we check their results. Note that the preceding code uses a `Stack<std::string>` object for testing functionality. Therefore, one can only push/pop `std::string` values into/from the stack.

The following shell box shows how to build and run the preceding code. Note that all the C++ code that you've seen in this section is written using C++11, hence it should be compiled using a compliant compiler. Like we said before, we are running the following commands when we are in the chapter's root directory:

```cpp
$ cd c++
$ g++ -c -g -std=c++11 -I$PWD/.. Stack.cpp -o Stack.o
$ g++ -L$PWD/.. Stack.o -lcstack -o cstack_cpp.out 
$ LD_LIBRARY_PATH=$PWD/.. ./cstack_cpp.out
Stack size: 3
Popped > !
Popped > World
Popped > Hello
Stack size after pops: 0
Stack size before clear: 2
Stack size after clear: 0
$
```

Shell Box 21-9: Building and running the C++ code

As you can see, we have indicated that we are going to use a C++11 compiler by passing the `-std=c++11` option. Note the `-I` and `-L` options, which are used for specifying custom include and library directories respectively. The option `-lcstack` asks the linker to link the C++ code with the library file `libcstack.so`. Note that on macOS systems, the shared object libraries have the `.dylib` extension, and therefore you might find `libcstack.dylib` instead of `libcstack.so`.

For running the `cstack_cpp.out` executable file, the loader needs to find `libcstack.so`. Note that this is different from building the executable. Here we want to run it, and the library file must be located before having the executable run. Therefore, by changing the environment variable `LD_LIBRARY_PATH`, we let the loader know where it should look for the shared objects. We have discussed more regarding this in *Chapter 2*, *Compilation and Linking*.

The C++ code should also be tested against memory leaks. `valgrind` helps us to see the memory leaks and we use it to analyze the resulting executable. The following shell box shows the output of `valgrind` running the `cstack_cpp.out` executable file:

```cpp
$ cd c++
$ LD_LIBRARY_PATH=$PWD/.. valgrind --leak-check=full ./cstack_cpp.out
==15061== Memcheck, a memory error detector
==15061== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==15061== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==15061== Command: ./cstack_cpp.out
==15061==
Stack size: 3
Popped > !
Popped > World
Popped > Hello
Stack size after pops: 0
Stack size before clear: 2
Stack size after clear: 0
==15061==
==15061== HEAP SUMMARY:
==15061==     in use at exit: 0 bytes in 0 blocks
==15061==   total heap usage: 9 allocs, 9 frees, 75,374 bytes allocated
==15061==
==15061== All heap blocks were freed -- no leaks are possible
==15061==
==15061== For counts of detected and suppressed errors, rerun with: -v
==15061== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0) 
$
```

Shell Box 21-10: Building and running the C++ code using valgrind

As is clear from the preceding output, we don't have any leaks in the code. Note that having 1081 bytes in the `still reachable` section doesn't mean that you have had a leak in your code. You can find more about this in `valgrind`'s manual.

In this section, we explained how to write a C++ wrapper around our C stack library. While mixing C and C++ code seems to be easy, some extra care about name mangling rules in C++ should be taken. In the next section, we are going to briefly talk about the Java programming language and the way that we are going to load our C library in a program written in Java.

# Integration with Java

Java programs are compiled by a Java compiler into Java bytecode. Java bytecode is analogous to the object file format specified in the **Application Binary Interface** (**ABI**). Files containing Java bytecode cannot be executed like ordinary executable files, and they need a special environment to be run.

Java bytecode can only be run within a **Java Virtual Machine** (**JVM**). The JVM is itself a process that simulates a working environment for the Java bytecode. It is usually written in C or C++ and has the power to load and use the C standard library and the functionalities exposed in that layer.

The Java programming language is not the only language that can be compiled into Java bytecode. Scala, Kotlin, and Groovy are among programming languages that can be compiled to Java bytecode hence they can be run within a JVM. They are usually called *JVM languages*.

In this section, we are going to load our already built stack library into a Java program. For those who have no prior knowledge of Java, the steps we take may seem complicated and hard to grasp. Therefore, it is strongly recommended that readers come into this section with some basic knowledge about Java programming.

## Writing the Java part

Suppose that we have a C project that it is built into a shared object library. We want to bring it into Java and use its functions. Fortunately, we can write and compile the Java part without having any C (or native) code. They are well separated by the *native methods* in Java. Obviously, you cannot run the Java program with just the Java part, and have the C functions called, without the shared object library file being loaded. We give the necessary steps and source code to make this happen and run a Java program that loads a shared object library and invokes its functions successfully.

The JVM uses **Java Native Interface** (**JNI**) to load shared object libraries. Note that JNI is not part of the Java programming language; rather, it is part of the JVM specification, therefore an imported shared object library can be used in all JVM languages such as Scala.

In the following paragraphs, we show how to use JNI to load our target shared object library file.

As we said before, JNI uses native methods. Native methods don't have any definition in Java; their actual definitions are written using C or C++ and they reside in external shared libraries. In other words, native methods are ports for the Java programs to communicate to the world outside of the JVM. The following code shows a class that contains a number of static native methods and it is supposed to expose the functionalities provided by our stack library:

```cpp
package com.packt.extreme_c.ch21.ex1;
class NativeStack {
  static {
    System.loadLibrary("NativeStack");
  }
  public static native long newStack();
  public static native void deleteStack(long stackHandler);
  public static native void ctor(long stackHandler, int maxSize);
  public static native void dtor(long stackHandler);
  public static native int size(long stackHandler);
  public static native void push(long stackHandler, byte[] item);
  public static native byte[] pop(long stackHandler);
  public static native void clear(long stackHandler);
}
```

Code Box 21-9 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: The NativeStack class

As the method signatures imply, they correspond to the functions we have in the C stack library. Note that the first operand is a `long` variable. It contains a native address read from the native library and acts as a pointer that should be passed to other methods to denote the stack instance. Note that, for writing the preceding class, we don't need to have a fully working shared object file beforehand. The only thing we need is the list of required declarations to define the stack API.

The preceding class has also a *static constructor*. The constructor loads a shared object library file located on the filesystem and tries to match the native methods with the symbols found in that shared object library. Note that the preceding shared object library is not `libcstack.so`. In other words, this is not the shared object file that we produced for our stack library. JNI has a very precise recipe for finding symbols that correspond to native methods. Therefore, we cannot use our symbols defined in `libcstack.so`; instead we need to create the symbols that JNI is looking for and then use our stack library from there.

This might be a bit unclear at the moment, but in the following section, we clarify this and you'll see how this can be done. Let's continue with the Java part. We still need to add some more Java code.

The following is a generic Java class named `Stack<T>` that wraps the native methods exposed by JNI. Generic Java classes can be regarded as twin concepts for the template classes that we had in C++. They are used to specify some generic types that can operate on other types.

As you see in the `Stack<T>` class, there is a *marshaller* object, from the type `Marshaller<T>`, that is used to serialize and deserialize the methods' input arguments (from type `T`) in order to put them into, or retrieve them from, the underlying C stack:

```cpp
interface Marshaller<T> {
  byte[] marshal(T obj);
  T unmarshal(byte[] data);
}
class Stack<T> implements AutoCloseable {
  private Marshaller<T> marshaller;
  private long stackHandler;
  public Stack(Marshaller<T> marshaller) {
    this.marshaller = marshaller;
    this.stackHandler = NativeStack.newStack();
    NativeStack.ctor(stackHandler, 100);
  }
  @Override
  public void close() {
    NativeStack.dtor(stackHandler);
    NativeStack.deleteStack(stackHandler);
  }
  public int size() {
    return NativeStack.size(stackHandler);
  }
  public void push(T item) {
    NativeStack.push(stackHandler, marshaller.marshal(item));
  }
  public T pop() {
    return marshaller.unmarshal(NativeStack.pop(stackHandler));
  }
  public void clear() {
    NativeStack.clear(stackHandler);
  }
}
```

Code Box 21-10 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: The Stack<T> class and the Marshaller<T> interface

The following points seem to be noticeable regarding the preceding code:

*   The class `Stack<T>` is a generic class. It means that its different instances can operate on various classes like `String`, `Integer`, `Point`, and so on, but every instance can operate only on the type specified upon instantiation.
*   The ability to store any data type in the underlying stack requires the stack to use an external marshaller to perform serialization and deserialization of the objects. The C stack library is able to store byte arrays in a stack data structure and higher-level languages willing to use its functionalities should be able to provide that byte array through serialization of the input objects. You will see shortly the implementation of the `Marshaller` interface for the `String` class.
*   We inject the `Marshaller` instance using the constructor. This means that we should have an already created marshaller instance that is compatible with the *generic* type of the class `T`.
*   The `Stack<T>` class implements the `AutoCloseable` interface. This simply means that it has some native resources that should be freed upon destruction. Note that the actual stack is created in the native code and not in the Java code. Therefore, the JVM's *garbage collector* cannot free the stack when it is not needed anymore. `AutoCloseable` objects can be used as resources which have a specific scope and when they are not needed anymore, their `close` method is called automatically. Shortly, you will see how we use the preceding class in a test scenario.
*   As you see, we have the constructor method and we have initialized the underlying stack using the native methods. We keep a handler to the stack as a `long` field in the class. Note that unlike in C++, we don't have any destructors in the class. Therefore, it is possible not to have the underlying stack freed and for it eventually to become a memory leak. That's why we have marked the class as an `AutoCloseable`. When an `AutoCloseable` object is not needed anymore, its `close` method is called and as you see in the preceding code, we call the destructor function from the C stack library to release the resources allocated by the C stack.

Generally, you cannot trust the garbage collector mechanism to call *finalizer methods* on Java objects and using the `AutoCloseable` resources is the correct way to manage native resources.

The following is the implementation of `StringMarshaller`. The implementation is very straightforward thanks to the great support of the `String` class in working with byte arrays:

```cpp
class StringMarshaller implements Marshaller<String> {
  @Override
  public byte[] marshal(String obj) {
    return obj.getBytes();
  }
  @Override
  public String unmarshal(byte[] data) {
    return new String(data);
  }
}
```

Code Box 21-11 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: The StringMarshaller class

The following code is our `Main` class that contains the test scenario for demonstration of C stack functionalities through Java code:

```cpp
public class Main {
  public static void main(String[] args) {
    try (Stack<String> stack = new Stack<>(new StringMarshaller())) {
      stack.push("Hello");
      stack.push("World");
      stack.push("!");
      System.out.println("Size after pushes: " + stack.size());
      while (stack.size() > 0) {
        System.out.println(stack.pop());
      }
      System.out.println("Size after pops: " + stack.size());
      stack.push("Ba");
      stack.push("Bye!");
      System.out.println("Size after before clear: " + stack.size());
      stack.clear();
      System.out.println("Size after clear: " + stack.size());
    }
  }
}
```

Code Box 21-12 [java/src/com/packt/extreme_c/ch21/ex1/Main.java]: The Main class that contains the test scenario to check the functionalities of the C stack library

As you see, the reference variable `stack` is being created and used inside a `try` block. This syntax is usually called *try-with-resources* and it has been introduced as part of Java 7\. When the `try` block is finished, the method `close` is called on the resource object and the underlying stack becomes freed. The test scenario is the same as the scenario we wrote for C++ in the previous section, but this time in Java.

In this section, we covered the Java part and all the Java code that we need to import the native part. All the sources above can be compiled but you cannot run them because you need the native part as well. Only together can they lead to an executable program. In the next section, we talk about the steps we should take to write the native part.

## Writing the native part

The most important thing we introduced in the previous section was the idea of native methods. Native methods are declared within Java, but their definitions reside outside of the JVM in a shared object library. But how does the JVM find the definition of a native method in the loaded shared object files? The answer is simple: by looking up certain symbol names in the shared object files. The JVM extracts a symbol name for every native method based on its various properties like the package, the containing class, and its name. Then, it looks for that symbol in the loaded shared object libraries and if it cannot find it, it gives you an error.

Based on what we established in the previous section, the JVM forces us to use specific symbol names for the functions we write as part of the loaded shared object file. But we didn't use any specific convention while creating the stack library. So, the JVM won't be able to find our exposed functions from the stack library and we must come up with another way. Generally, C libraries are written without any assumption about being used in a JVM environment.

*Figure 21-1* shows how we can use an intermediate C or C++ library to act as a glue between the Java part and the native part. We give the JVM the symbols it wants, and we delegate the function calls made to the functions representing those symbols to the correct function inside the C library. This is basically how JNI works.

We'll explain this with an imaginary example. Suppose that we want to make a call to a C function, `func`, from Java, and the definition of the function can be found in the `libfunc.so` shared object file. We also have a class `Clazz` in the Java part with a native function called `doFunc`. We know that the JVM would be looking for the symbol `Java_Clazz_doFunc` while trying to find the definition of the native function `doFunc`. We create an intermediate shared object library `libNativeLibrary.so` that contains a function with exactly the same symbol that the JVM is looking for. Then, inside that function, we make a call to the `func` function. We can say that the function `Java_Clazz_doFunc` acts as a relay and delegates the call to the underlying C library and eventually the `func` function.

![fig10-1](img/B11046_21_01.png)

Figure 21-1: The intermediate shared object libNativeStack.so which is used to delegate function calls from Java to the actual underlying C stack library, libcstack.so.

In order to stay aligned with JVM symbol names, the Java compiler usually generates a C header file out of the native methods found in a Java code. This way, you only need to write the definitions of those functions found in the header file. This prevents us from making any mistakes in the symbol names that the JVM eventually would be looking for.

The following commands demonstrate how to compile a Java source file and how to ask the compiler to generate a header file for the found native methods in it. Here, we are going to compile our only Java file, `Main.java`, which contains all the Java code introduced in the previous code boxes. Note that we should be in the chapter's root directory when running the following commands:

```cpp
$ cd java
$ mkdir -p build/headers
$ mkdir -p build/classes
$ javac -cp src -h build/headers -d build/classes \
src/com/packt/extreme_c/ch21/ex1/Main.java
$ tree build
build
├── classes
│   └── com
│       └── packt
│           └── extreme_c
│               └── ch21
│                   └── ex1
│                       ├── Main.class
│                       ├── Marshaller.class
│                       ├── NativeStack.class
│                       ├── Stack.class
│                       └── StringMarshaller.class
└── headers
    └── com_packt_extreme_c_ch21_ex1_NativeStack.h
7 directories, 6 files
$
```

Shell Box 21-11: Compiling the Main.java while generating a header for native methods found in the file

As shown in the preceding shell box, we have passed the option `-h` to `javac,` which is the Java compiler. We have also specified a directory that all headers should go to. The `tree` utility shows the content of the `build` directory in a tree-like format. Note the `.class` files. They contain the Java bytecode which will be used when loading these classes into a JVM instance.

In addition to class files we see a header file, `com_packt_extreme_c_ch21_ex1_NativeStack.h`, that contains the corresponding C function declarations for the native methods found in the `NativeStack` class.

If you open the header file, you will see something like *Code Box 21-13*. It has a number of function declarations with long and strange names each of which being made up of the package name, the class name, and the name of the corresponding native method:

```cpp
/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_packt_extreme_c_ch21_ex1_NativeStack */
#ifndef _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#define _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_packt_extreme_c_ch21_ex1_NativeStack
 * Method:    newStack
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_packt_extreme_1c_ch21_ex1_NativeStack_newStack
  (JNIEnv *, jclass);
/*
 * Class:     com_packt_extreme_c_ch21_ex1_NativeStack
 * Method:    deleteStack
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_packt_extreme_1c_ch21_ex1_NativeStack_deleteStack
  (JNIEnv *, jclass, jlong);

...
...
...
#ifdef __cplusplus
}
#endif
#endif
```

Code Box 21-13: The (incomplete) content of the generated JNI header file

The functions declared in the preceding header file carry the symbol names that the JVM would be looking for when loading the corresponding C function for a native method. We have modified the preceding header file and used macros to make it compact in order to have all the function declarations in a smaller area. You can see it in *Code Box 21-14*:

```cpp
// Filename: NativeStack.h
// Description: Modified JNI generated header file
#include <jni.h>
#ifndef _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#define _Included_com_packt_extreme_c_ch21_ex1_NativeStack
#define JNI_FUNC(n) Java_com_packt_extreme_1c_ch21_ex1_NativeStack_##
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT jlong JNICALL JNI_FUNC(newStack)(JNIEnv* , jclass);
JNIEXPORT void JNICALL JNI_FUNC(deleteStack)(JNIEnv* , jclass, jlong);
JNIEXPORT void JNICALL JNI_FUNC(ctor)(JNIEnv* , jclass, jlong, jint);
JNIEXPORT void JNICALL JNI_FUNC(dtor)(JNIEnv* , jclass, jlong);
JNIEXPORT jint JNICALL JNI_FUNC(size)(JNIEnv* , jclass, jlong);
JNIEXPORT void JNICALL JNI_FUNC(push)(JNIEnv* , jclass, jlong, jbyteArray);
JNIEXPORT jbyteArray JNICALL JNI_FUNC(pop)(JNIEnv* , jclass, jlong);
JNIEXPORT void JNICALL JNI_FUNC(clear)(JNIEnv* , jclass, jlong);
#ifdef __cplusplus
}
#endif
#endif
```

Code Box 21-14 [java/native/NativeStack.h]: The modified version of the generated JNI header file

As you see, we have created a new macro `JNI_FUNC` that factors out a big portion of the function name that is common for all of the declarations. We have also removed the comments in order to make the header file even more compact.

We will be using the macro `JNI_FUNC` in both the header file and the following source file, which are shown as part of *Code Box 21-15*.

**Note**:

It is not an accepted behavior to modify the generated header file. We did it because of educational purposes. In real build environments, it is desired to use the generated files directly without any modification.

In *Code Box 21-15*, you will find the definitions of the preceding functions. As you see, the definitions only relay the calls to the underlying C functions included from the C stack library:

```cpp
#include <stdlib.h>
#include "NativeStack.h"
#include "cstack.h"
void defaultDeleter(value_t* value) {
  free_value(value);
}
void extractFromJByteArray(JNIEnv* env,
                           jbyteArray byteArray,
                           value_t* value) {
  jboolean isCopy = false;
  jbyte* buffer = env->GetByteArrayElements(byteArray, &isCopy);
  value->len = env->GetArrayLength(byteArray);
  value->data = (char*)malloc(value->len * sizeof(char));
  for (size_t i = 0; i < value->len; i++) {
    value->data[i] = buffer[i];
  }
  env->ReleaseByteArrayElements(byteArray, buffer, 0);
}
JNIEXPORT jlong JNICALL JNI_FUNC(newStack)(JNIEnv* env,
                                           jclass clazz) {
  return (long)cstack_new();
}
JNIEXPORT void JNICALL JNI_FUNC(deleteStack)(JNIEnv* env,
                                            jclass clazz,
                                            jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_delete(cstack);
}
JNIEXPORT void JNICALL JNI_FUNC(ctor)(JNIEnv *env,
                                      jclass clazz,
                                      jlong stackPtr,
                                      jint maxSize) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_ctor(cstack, maxSize);
}
JNIEXPORT void JNICALL JNI_FUNC(dtor)(JNIEnv* env,
                                      jclass clazz,
                                      jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_dtor(cstack, defaultDeleter);
}
JNIEXPORT jint JNICALL JNI_FUNC(size)(JNIEnv* env,
                                      jclass clazz,
                                      jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  return cstack_size(cstack);
}
JNIEXPORT void JNICALL JNI_FUNC(push)(JNIEnv* env,
                                      jclass clazz,
                                      jlong stackPtr,
                                      jbyteArray item) {
  value_t value;
  extractFromJByteArray(env, item, &value);
  cstack_t* cstack = (cstack_t*)stackPtr;
  bool_t pushed = cstack_push(cstack, value);
  if (!pushed) {
    jclass Exception = env->FindClass("java/lang/Exception");
    env->ThrowNew(Exception, "Stack is full!");
  }
}
JNIEXPORT jbyteArray JNICALL JNI_FUNC(pop)(JNIEnv* env,
                                           jclass clazz,
                                           jlong stackPtr) {
  value_t value;
  cstack_t* cstack = (cstack_t*)stackPtr;
  bool_t popped = cstack_pop(cstack, &value);
  if (!popped) {
    jclass Exception = env->FindClass("java/lang/Exception");
    env->ThrowNew(Exception, "Stack is empty!");
  }
  jbyteArray result = env->NewByteArray(value.len);
  env->SetByteArrayRegion(result, 0,
          value.len, (jbyte*)value.data);
  defaultDeleter(&value);
  return result;
}
JNIEXPORT void JNICALL JNI_FUNC(clear)(JNIEnv* env,
                                       jclass clazz,
                                       jlong stackPtr) {
  cstack_t* cstack = (cstack_t*)stackPtr;
  cstack_clear(cstack, defaultDeleter);
}
```

Code Box 21-15 [java/native/NativeStack.cpp]: The definitions of the functions declared in the JNI header file

The preceding code is written in C++. It is possible to write the definitions in C as well. The only thing demanding attention is the conversion from C byte arrays into Java byte arrays happening in push and pop functions. The function `extractFromJByteArray` has been added to create a C byte array based on a Java byte array received from the Java part.

The following commands create the intermediate shared object `libNativeStack.so` in Linux, which is going to be loaded and used by the JVM. Note that you need to set the environment variable `JAVA_HOME` before running the following commands:

```cpp
$ cd java/native
$ g++ -c -fPIC -I$PWD/../.. -I$JAVA_HOME/include \
 -I$JAVA_HOME/include/linux NativeStack.cpp -o NativeStack.o
$ g++ -shared -L$PWD/../.. NativeStack.o -lcstack -o libNativeStack.so
$
```

Shell Box 21-12: Building the intermediate shared object library libNativeStack.so

As you see, the final shared object file is linked against the C stack library's shared object file `libcstack.so` which simply means the `libNativeStack.so` has to load `libcstack.so` in order to work. Therefore, the JVM loads the `libNativeStack.so` library and then it loads `libcstack.so` library, and eventually the Java part and the native part can cooperate and let the Java program be executed.

The following commands run the test scenario shown in *Code Box 21-12*:

```cpp
$ cd java
$ LD_LIBRARY_PATH=$PWD/.. java -Djava.library.path=$PWD/native \
  -cp build/classes com.packt.extreme_c.ch21.ex1.Main
Size after pushes: 3
!
World
Hello
Size after pops: 0
Size after before clear: 2
Size after clear: 0
$
```

Shell Box 21-13: Running the Java test scenario

As you see, we have passed the option `-Djava.library.path=...` to the JVM. It specifies the place where shared object libraries can be found. As you see, we have specified the directory which should contain the `libNativeStack.so` shared object library.

In this section, we showed how to load a native C library into the JVM and use it together with other Java source code. The same mechanism can be applied for loading bigger and multi-part native libraries.

Now, it's time to go through the Python integration and see how the C stack library can be used from Python code.

# Integration with Python

Python is an *interpreted* programming language. This means that the Python code is read and run by an intermediate program that is called an *interpreter*. If we are going to use an external native shared library, it is the interpreter that loads the shared library and makes it available to the Python code. Python has a special framework for loading external shared libraries. It is called *ctypes* and we are going to use it in this section.

Loading the shared libraries using `ctypes` is very straightforward. It only requires loading the library and defining the inputs and output of the functions that are going to be used. The following class wraps the ctypes-related logic and makes it available to our main `Stack` class, shown in the upcoming code boxes:

```cpp
from ctypes import *
class value_t(Structure):
  _fields_ = [("data", c_char_p), ("len", c_int)]
class _NativeStack:
  def __init__(self):
    self.stackLib = cdll.LoadLibrary(
            "libcstack.dylib" if platform.system() == 'Darwin'
            else "libcstack.so")
    # value_t make_value(char*, size_t)
    self._makevalue_ = self.stackLib.make_value
    self._makevalue_.argtypes = [c_char_p, c_int]
    self._makevalue_.restype = value_t
    # value_t copy_value(char*, size_t)
    self._copyvalue_ = self.stackLib.copy_value
    self._copyvalue_.argtypes = [c_char_p, c_int]
    self._copyvalue_.restype = value_t
    # void free_value(value_t*)
    self._freevalue_ = self.stackLib.free_value
    self._freevalue_.argtypes = [POINTER(value_t)]
    # cstack_t* cstack_new()
    self._new_ = self.stackLib.cstack_new
    self._new_.argtypes = []
    self._new_.restype = c_void_p
    # void cstack_delete(cstack_t*)
    self._delete_ = self.stackLib.cstack_delete
    self._delete_.argtypes = [c_void_p]
    # void cstack_ctor(cstack_t*, int)
    self._ctor_ = self.stackLib.cstack_ctor
    self._ctor_.argtypes = [c_void_p, c_int]
    # void cstack_dtor(cstack_t*, deleter_t)
    self._dtor_ = self.stackLib.cstack_dtor
    self._dtor_.argtypes = [c_void_p, c_void_p]
    # size_t cstack_size(cstack_t*)
    self._size_ = self.stackLib.cstack_size
    self._size_.argtypes = [c_void_p]
    self._size_.restype = c_int
    # bool_t cstack_push(cstack_t*, value_t)
    self._push_ = self.stackLib.cstack_push
    self._push_.argtypes = [c_void_p, value_t]
    self._push_.restype = c_int
    # bool_t cstack_pop(cstack_t*, value_t*)
    self._pop_ = self.stackLib.cstack_pop
    self._pop_.argtypes = [c_void_p, POINTER(value_t)]
    self._pop_.restype = c_int
    # void cstack_clear(cstack_t*, deleter_t)
    self._clear_ = self.stackLib.cstack_clear
    self._clear_.argtypes = [c_void_p, c_void_p]
```

Code Box 21-17 [python/stack.py]: The ctypes-related code that makes the stack library's C functions available to the rest of Python

As you can see, all the functions required to be used in our Python code are put in the class definition. The handles to the C functions are stored as private fields in the class instance (private fields have `_` on both sides) and they can be used to call the underlying C function. Note that in the above code, we have loaded the `libcstack.dylib`, as we are in a macOS system. And for Linux systems, we need to load `libcstack.so`.

The following class is the main Python component that uses the above wrapper class. All other Python code uses this class to have the stack functionality:

```cpp
class Stack:
  def __enter__(self):
    self._nativeApi_ = _NativeStack()
    self._handler_ = self._nativeApi_._new_()
    self._nativeApi_._ctor_(self._handler_, 100)
    return self
  def __exit__(self, type, value, traceback):
    self._nativeApi_._dtor_(self._handler_, self._nativeApi_._freevalue_)
    self._nativeApi_._delete_(self._handler_)
  def size(self):
    return self._nativeApi_._size_(self._handler_)
  def push(self, item):
    result = self._nativeApi_._push_(self._handler_,
            self._nativeApi_._copyvalue_(item.encode('utf-8'), len(item)));
    if result != 1:
      raise Exception("Stack is full!")
  def pop(self):
    value = value_t()
    result = self._nativeApi_._pop_(self._handler_, byref(value))
    if result != 1:
      raise Exception("Stack is empty!")
    item = string_at(value.data, value.len)
    self._nativeApi_._freevalue_(value)
    return item
  def clear(self):
    self._nativeApi_._clear_(self._handler_, self._nativeApi_._freevalue_)
```

Code Box 21-16 [python/stack.py]: The Stack class in Python that uses the loaded C functions from the stack library

As you see, the Stack class keeps a reference to the `_NativeStack` class in order to be able to call the underlying C functions. Note that the preceding class overrides `__enter__` and `__exit__` functions. This allows the class to be used as a resource class and be consumed in the `with` syntax in Python. You will see shortly what the syntax looks like. Please note that the preceding Stack class only operates on string items.

The following is the test scenario, which is very similar to the Java and C++ test scenarios:

```cpp
if __name__ == "__main__":
  with Stack() as stack:
    stack.push("Hello")
    stack.push("World")
    stack.push("!")
    print("Size after pushes:" + str(stack.size()))
    while stack.size() > 0:
      print(stack.pop())
    print("Size after pops:" + str(stack.size()))
    stack.push("Ba");
    stack.push("Bye!");
    print("Size before clear:" + str(stack.size()))
    stack.clear()
    print("Size after clear:" + str(stack.size()))
```

Code Box 21-18 [python/stack.py]: The test scenario written in Python and using the Stack class

In the preceding code, you can see Python's `with` statement.

Upon entering the `with` block, the `__enter__` function is called and an instance of the `Stack` class is referenced by the `stack` variable. When leaving the `with` block, the `__exit__` function is called. This gives us the opportunity to free the underlying native resources, the C stack object in this case, when they are not needed anymore.

Next, you can see how to run the preceding code. Note that all the Python code boxes exist within the same file named `stack.py`. Before running the following commands, you need to be in the chapter's root directory:

```cpp
$ cd python
$ LD_LIBRARY_PATH=$PWD/.. python stack.py
Size after pushes:3
!
World
Hello
Size after pops:0
Size before clear:2
Size after clear:0
$
```

Shell Box 21-14: Running the Python test scenario

Note that the interpreter should be able to find and load the C stack shared library; therefore, we set the `LD_LIBRARY_PATH` environment variable to point to the directory that contains the actual shared library file.

In the following section, we show how to load and use the C stack library in the Go language.

# Integration with Go

The Go programming language (or simply Golang) has an easy integration with native shared libraries. It can be considered as the next generation of the C and C++ programming languages and it calls itself a system programming language. Therefore, we expect to load and use the native libraries easily when using Golang.

In Golang, we use a built-in package called *cgo* to call C code and load the shared object files. In the following Go code, you see how to use the `cgo` package and use it to call the C functions loaded from the C stack library file. It also defines a new class, `Stack`, which is used by other Go code to use the C stack functionalities:

```cpp
package main
/*
#cgo CFLAGS: -I..
#cgo LDFLAGS: -L.. -lcstack
#include "cstack.h"
*/
import "C"
import (
  "fmt"
)
type Stack struct {
  handler *C.cstack_t
}
func NewStack() *Stack {
  s := new(Stack)
  s.handler = C.cstack_new()
  C.cstack_ctor(s.handler, 100)
  return s
}
func (s *Stack) Destroy() {
  C.cstack_dtor(s.handler, C.deleter_t(C.free_value))
  C.cstack_delete(s.handler)
}
func (s *Stack) Size() int {
  return int(C.cstack_size(s.handler))
}
func (s *Stack) Push(item string) bool {
  value := C.make_value(C.CString(item), C.ulong(len(item) + 1))
  pushed := C.cstack_push(s.handler, value)
  return pushed == 1
}
func (s *Stack) Pop() (bool, string) {
  value := C.make_value(nil, 0)
  popped := C.cstack_pop(s.handler, &value)
  str := C.GoString(value.data)
  defer C.free_value(&value)
  return popped == 1, str
}
func (s *Stack) Clear() {
  C.cstack_clear(s.handler, C.deleter_t(C.free_value))
}
```

Code Box 21-19 [go/stack.go]: The Stack class using the loaded libcstack.so shared object file

In order to use the cgo package, one needs to import the `C` package. It loads the shared object libraries specified in the pseudo `#cgo` directives. As you see, we have specified the `libcstack.so` library to be loaded as part of the directive `#cgo LDFLAGS: -L.. -lcstack`. Note that the `CFLAGS` and `LDFLAGS` contain the flags that are directly passed to the C compiler and to the linker respectively.

We have also indicated the path that should be searched for the shared object file. After that, we can use the `C` struct to call the loaded native functions. For example, we have used `C.cstack_new()` to call the corresponding function from the stack library. It is pretty easy with cgo. Note that the preceding `Stack` class only works on string items.

The following code shows the test scenario written in Golang. Note that we have to call the `Destroy` function on the `stack` object when quitting the `main` function:

```cpp
func main() {
  var stack = NewStack()
  stack.Push("Hello")
  stack.Push("World")
  stack.Push("!")
  fmt.Println("Stack size:", stack.Size())
  for stack.Size() > 0 {
    _, str := stack.Pop()
    fmt.Println("Popped >", str)
  }
  fmt.Println("Stack size after pops:", stack.Size())
  stack.Push("Bye")
  stack.Push("Bye")
  fmt.Println("Stack size before clear:", stack.Size())
  stack.Clear()
  fmt.Println("Stack size after clear:", stack.Size())
  stack.Destroy()
}
```

Code Box 21-20 [go/stack.go]: The test scenario written in Go and using the Stack class

The following shell box demonstrates how to build and run the test scenario:

```cpp
$ cd go
$ go build -o stack.out stack.go
$ LD_LIBRARY_PATH=$PWD/.. ./stack.out
Stack size: 3
Popped > !
Popped > World
Popped > Hello
Stack size after pops: 0
Stack size before clear: 2
Stack size after clear: 0
$
```

Shell Box 21-15: Running the Go test scenario

As you see in Golang, unlike Python, you need to compile your program first, and then run it. In addition, we still need to set the `LD_LIBRARY_PATH` environment variable in order to allow the executable to locate the `libcstack.so` library and load it.

In this section, we showed how to use the `cgo` package in Golang to load and use shared object libraries. Since Golang behaves like a thin wrapper around C code, it has been easier than using Python and Java to load an external shared object library and use it.

# Summary

In this chapter, we went through the integration of C within other programming languages. As part of this chapter:

*   We designed a C library that was exposing some stack functionality such as push, pop, and so on. We built the library and as the final output we generated a shared object library to be used by other languages.
*   We discussed the name mangling feature in C++, and how we should avoid it in C when using a C++ compiler.
*   We wrote a C++ wrapper around the stack library that could load the library's shared object file and execute the loaded functionalities within C++.
*   We continued by writing a JNI wrapper around the C library. We used native methods to achieve that.
*   We showed how to write native code in JNI and connect the native part and Java part together, and finally run a Java program that uses the C stack library.
*   We managed to write Python code that was using the ctypes package to load and use the library's shared object file.
*   As the final section, we wrote a program in Golang that could load the library's shared object file with help from the `cgo` package.

The next chapter is about unit testing and debugging in C. We will introduce some C libraries meant for writing unit tests. More than that, we talk about debugging in C, and some of the existing tools that could be used to debug or monitor a program.