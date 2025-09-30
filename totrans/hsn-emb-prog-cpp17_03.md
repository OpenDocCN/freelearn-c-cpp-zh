# C++ as an Embedded Language

When it comes to embedded development on resource-restricted systems, it is still common to consider only C and ASM as viable choices, accompanied by the thought that C++ has a larger footprint than C, or adds a significant amount of complexity. In this chapter, we will look at all of these issues in detail and consider the merits of C++ as an embedded programming language:

*   C++ relative to C
*   Advantages of C++ as a multi-paradigm language
*   Compatibility with existing C and ASM
*   Changes with C++11, C++14, and C++17

# C++ relative to C

The lineages of C and C++ both trace their lineage back to the ALGOL programming language, which saw its first version in 1958 (ALGOL 58), followed by updates in 1960 and 1968\. ALGOL introduced the concept of imperative programming—a programming style in which statements explicitly tell the machine how to make changes to data for output and control flow.

A paradigm that emerges rather naturally from imperative programming is the use of procedures. We will start with an example, to introduce the terminology. Procedures are synonymous to sub-routines and functions. They identify the groups of statements and make them self-contained, which has the effects of confining the reach of these statements to the limited scope of the section they are contained within, creating hierarchy and consequentially introducing these procedures as new, more abstract statements. Heavy use of this procedural programming style finds its place in so-called structured programming, alongside loop and branching control structures.

Over time, structured and modular programming styles were introduced as techniques to improve the development, quality and maintainability of application code. The C language is an imperative, structured programming language due to its use of statements, control structures and functions.

Take, for example, the standard Hello World example in C:

```cpp
#include <stdio.h> 
int main(void) 
{ 
    printf("hello, world"); 
    return 0; 
} 
```

The entry point of any C (and C++) application is the `main()` function (procedure). In the first statement line of this function, we call another procedure (`printf()`), which contains its own statements and possibly calls other blocks of statements in the form of additional functions.

This way we have already made use of procedural programming by implementing a `main()` logical block (the `main()` function), which is called as needed. While the `main()` function will just be called once, the procedural style is found again in the `printf()` statement, which calls the statements elsewhere in the application without having to copy them explicitly. Applying procedural programming makes it much easier to maintain the resulting code, and create libraries of code that we can use across a number of applications, while maintaining only a single code base.

In 1979, Bjarne Stroustrup started work on *C with Classes*, for which he took the existing programming paradigms of C and added elements from other languages, in particular Simula (object-oriented programming: both imperative and structured) and ML (generic programming, in the form of templates). It would also offer the speed of the **Basic Combined Programming Language** (**BCPL**), without restricting the developer to its restrictive low-level focus.

The resulting multi-paradigm language was renamed to **C++** in 1983, while adding additional features not found in C, including operator and function overloading, virtual functions, references, and starting the development of a standalone compiler for this C++ language.

The essential goal of C++ has remained to provide practical solutions to real-world issues. Additionally, it has always been the intention for C++ to be a better C, hence the name. Stroustrup himself defines a number of rules (as noted in *Evolving C++ 1991-2006*) that drive the development of C++ to this day, including the following:

*   C++'s evolution must be driven by real problems
*   Every feature must have a reasonably obvious implementation
*   C++ is a language, not a complete system
*   Don't try to force people to use a specific programming style
*   No implicit violations of the static type system
*   Provide as good support for user-defined types as for built-in types
*   Leave no room for a lower-level language below C++ (except assembler)
*   What you don't use, you don't pay for (zero-overhead rule)
*   If in doubt, provide means for manual control

The differences relative to C obviously goes beyond object-oriented programming. Despite the lingering impression that C++ is just a set of extensions to C, it has for a long time been its own language, adding a strict type system (compared to C's weak type system at that time), more powerful programming paradigms, and features not found in C. Its compatibility with C can therefore be seen more as coincidence, with C being the right language at the right time to be used as a foundation.

The problem with Simula at the time was that it was too slow for general use, and BCPL was too low-level. C, being a relatively new language at the time, provided the right middle ground between features and performance.

# C++ as an embedded language

Around 1983 when C++ had just been conceived and got its name, popular personal computer systems for a general audience, as well as businesses, had specifications like ones listed in the following table:

| **System** | **CPU** | **Clock speed (MHz)** | **RAM (KB)** | **ROM (KB)** | **Storage (KB)** |
| BBC Micro | 6502 (B+ 6512A) | 2 | 16-128 | 32-128 | Max 1,280 (ADFS floppy)Max 20 MB (hard drive) |
| MSX | Zilog Z80 | 3.58 | 8-128 | 32 | 720 (floppy) |
| Commodore 64 | 6510 | ~1 | 64 | 20 | 1,000 (tape)170 (floppy) |
| Sinclair ZX81 | Zilog Z80 | 3.58 | 1 | 8 | 15 (cartridge) |
| IBM PC | Intel 8080 | 4.77 | 16-256 | 8 | 360 (floppy) |

Now compare these computer systems to a recent 8-bit **microcontroller** (**MCU**) such as the AVR ATMega 2560 with the following specifications:

*   16 MHz clock speed
*   8 KB RAM
*   256 KB ROM (program)
*   4 KB ROM (data)

The ATMega 2560 was launched in 2005 and is among the more powerful 8-bit MCUs available nowadays. Its features stack up favorably against the 1980s computer systems, but on top of that the MCU does not rely on any external memory components.

The MCU core clock speed is significantly faster these days thanks to improved silicon IC manufacturing processes which also provide smaller chip sizes, high throughput, and thus lower cost and what's more, 1980s architectures commonly took 2 to 5 clock cycles to retrieve, decode, execute an instruction and store the result as opposed to the single-cycle execution performance of the AVR.

Current MCU (Static) RAM limitations are mostly due to cost and power constraints yet can be easily circumvented for most MCUs using external RAM chips, along with adding low-cost flash-based or other mass storage devices.

Systems like the **Commodore 64** (**C64**) were routinely programmed in C, in addition to the built-in BASIC interpreter (in a built-in ROM). A well-known C development environment for the Commodore 64 was Power C published by Spinnaker:

![](img/68f27296-883d-413d-9ab1-22a01b44e154.png)

Power C was one brand of productivity software aimed at C developers. It came on a single, double-sided floppy disk and allowed you to write C code in an editor, then compile it with the included compiler, linker, header files, and libraries to produce executables for the system.

Many more of such compiler collections existed back then, targeting a variety of systems, showing the rich ecosystem that existed for software development. Among these, C++ was of course a newcomer. The first edition of Stroustrup's *The C++ Programming Language* was only being published in 1985, yet initially without a solid implementation of the language to go with it.

Commercial support for C++ however began to appear rapidly, with major development environments such as Borland C++ 1.0 being released in 1987 and updated to 2.0 in 1991\. Development environments like these got used in particular on the IBM PC and its myriad of clones where no preferred development language such as BASIC existed.

While C++ began its life as an unofficial standard in 1985, it wasn't until 1989 and the release of the *The C++ Programming Language* in its second edition as an authoritative work that C++ reached roughly the level of features equal to what would first be then standardized by an ISO working group as ISO/IEC 14882:1998, commonly known as C++98\. Still it can be said that C++ saw significant development and adoption before the advent of the Motorola 68040 in 1990 and Intel 486DX in 1992, which bumped processing power above the 20 MIPS mark.

Now that we have considered early hardware specifications and the evolution of C++ alongside C and other languages of the time intended to be used on the relatively limited systems that existed back then, it seems plausible that C++ is more than capable of running on such hardware, and by extension on modern-day microcontrollers. However, it also seems necessary to ask to what extent the complexity added to C++ since then has impacted memory or computing performance requirements.

# C++ language features

We previously took a look at the explicit nature of changes to data and system state that defines imperative programming as opposed to declarative programming, where instead of manipulating data in a loop such functionality could be declared as mapping an operator to some data, thus spelling out the functionality, not the specific order of operations. But why should programming languages necessarily be a choice between imperative and declarative paradigms?

In fact, one of the main distinguishing features of C++ is its multi-paradigm nature making use of both imperative and declarative paradigms. With the inclusion of object-oriented, generic, and functional programming into C++ in addition to C's procedural programming, it would seem natural to assume that this would all have to come at a cost, whether in terms of higher CPU usage or more RAM and/or ROM consumed.

However, as we learned earlier in this chapter, C++ language features are ultimately built upon the C language, and as such there should in turn be little or no overhead relative to implementing a similar constructs in plain C. To resolve this conundrum and to investigate the validity of the low-overhead hypothesis, we'll now take a detailed look at a number of C++ language features, and how they are ultimately implemented, with their corresponding cost in binary and memory size.

Some of the examples that focus specifically on C++ as a low-level embedded language are taken with permission from Rud Merriam's Code Craft series, as published on Hackaday: [https://hackaday.io/project/8238-embedding-c](https://hackaday.io/project/8238-embedding-c).

# Namespaces

Namespaces are a way to introduce additional levels of scope into an application. As we saw in the earlier section on classes, these are a compiler-level concept.

The main use lies in modularizing code, dividing it into logical segments in cases where classes are not the most obvious solution, or where you want to explicitly sort classes into a particular category using a namespace. This way, you can also avoid name and type collisions between similarly named classes, types, and enumerations.

# Strongly typed

Type information is necessary to test for proper access to and interpretation of data. A big feature in C++ that's relative to C is the inclusion of a strong type system. This means that many type checks performed by the compiler are significantly more strict than what would be allowed with C, which is a weakly typed language.

This is mostly apparent when looking at this legal C code, which will generate an error when compiled as C++:

```cpp
void* pointer; 
int* number = pointer; 
```

Alternatively, they can also be written in the following way:

```cpp
int* number = malloc(sizeof(int) * 5); 
```

C++ forbids implicit casts, requiring these examples to be written as follows:

```cpp
void* pointer; 
int* number = (int*) pointer; 
```

They can also be written in the following way:

```cpp
int* number = (int*) malloc(sizeof(int) * 5); 
```

As we explicitly specify the type we are casting to, we can rest assured that during compile time any type casts do what we expect them to do.

Similarly, the compiler will also complain and throw an error if we were to try to assign to a variable with a `const` qualifier from a reference without this qualifier:

```cpp
const int constNumber = 42; 
int number = &constNumber; // Error: invalid initialization of reference. 
```

To work around this, you are required to explicitly cast the following conversion:

```cpp
const int constNumber = 42; 
int number = const_cast<int&>(constNumber); 
```

Performing an explicit cast like this is definitely possible and valid. It may also cause immense issues and headaches later on when using this reference to modify the contents of the supposedly constant value. By the time you find yourself writing code like the preceding, however, it can reasonably be assumed that you are aware of the implications.

Such enforcement of explicit types has the significant benefit of making static analysis far more useful and effective than it is in a weakly typed language. This, in turn, benefits run-time safety, as any conversions and assignments are most likely to be safe and without unexpected side effects.

As a type system is predominantly a feature of the compiler rather than any kind of run-time code, with (optional) run-time type information as an exception. The overhead of having a strongly typed type system in C++ is noticed only at compile time, as more strict checks have to be performed on each variable assignment, operation, and conversion.

# Type conversions

A type conversion occurs whenever a value is assigned to a compatible variable, which is not the exact same type as the value. Whenever a rule for conversion exists, this conversion can be done implicitly, otherwise an explicit hint (cast) can be provided to the compiler to invoke a specific rule where ambiguity exists.

Whereas C only has implicit and explicit type casting, C++ expands on this with a number of template-based functions, allowing you to cast both regular types and objects (classes) in a variety of ways:

*   `dynamic_cast <new_type>` (expression)
*   `reinterpret_cast <new_type>` (expression)
*   `static_cast <new_type>` (expression)
*   `const_cast <new_type>` (expression)

Here, `dynamic_cast` guarantees that the resulting object is valid, relying on **runtime type information** (**RTTI**) (see the later section on it) for this. A `static_cast` is similar, but does not validate the resulting object.

Next, `reinterpret_cast` can cast anything to anything, even unrelated classes. Whether this conversion makes sense is left to the developer, much like with a regular explicit conversion.

Finally, a `const_cast` is interesting in that it either sets or removes the `const` status of a value, which can be useful when you need a non-`const` version of a value for just one function. This does, however, also circumvent the type safety system and should be used very cautiously.

# Classes

**Object-oriented programming** (**OOP**) has been around since the days of Simula, which was known for being a slow language. This led Bjarne Stroustrup to base his OOP implementation on the fast and efficient C programming language.

C++ uses C-style language constructs to implement objects. This becomes obvious when we take a look at C++ code and its corresponding C code.

When looking at a C++ class, we see its typical structure:

```cpp
namespace had { 
using uint8_t = unsigned char; 
const uint8_t bufferSize = 16;  
    class RingBuffer { 
        uint8_t data[bufferSize]; 
        uint8_t newest_index; 
        uint8_t oldest_index;  
        public: 
        enum BufferStatus { 
            OK, EMPTY, FULL 
        };  
        RingBuffer();  
        BufferStatus bufferWrite(const uint8_t byte); 
        enum BufferStatus bufferRead(uint8_t& byte); 
    }; 
} 
```

This class is also inside of a namespace (which we will look at in more detail in a later section), a redefinition of the `unsigned char` type, a namespace-global variable definition, and finally the class definition itself, including a private and public section.

This C++ code defines a number of different scopes, starting with the namespace and ending with the class. The class itself adds scopes in the sense of its public, protected, and private access levels.

The same code can be implemented in regular C as follows:

```cpp
typedef unsigned char uint8_t; 
enum BufferStatus {BUFFER_OK, BUFFER_EMPTY, BUFFER_FULL}; 
#define BUFFER_SIZE 16 
struct RingBuffer { 
   uint8_t data[BUFFER_SIZE]; 
   uint8_t newest_index; 
   uint8_t oldest_index; 
};  
void initBuffer(struct RingBuffer* buffer); 
enum BufferStatus bufferWrite(struct RingBuffer* buffer, uint8_t byte); 
enum BufferStatus bufferRead(struct RingBuffer* buffer, uint8_t *byte); 
```

The `using` keyword is similar to `typedef`, making for a direct mapping there. We use a `const` instead of a `#define`. An `enum` is essentially the same between C and C++, only that C++'s compiler doesn't require the explicit marking of an `enum` when used as a type. The same is true for structs when it comes to simplifying the C++ code.

The C++ class itself is implemented in C as a `struct` containing the class variables. When the class instance is created, it essentially means that an instance of this `struct` is initialized. A pointer to this `struct` instance is then passed with each call of a function that is part of the C++ class.

What these basic examples show us is that there is no runtime overhead for any of the C++ features we used compared to the C-based code. The namespace, class access levels (public, private, and protected), and similar are only used by the compiler to validate the code that is being compiled.

A nice feature of the C++ code is that, despite the identical performance, it requires less code, while also allowing you to define strict interface access levels and have a destructor class method that gets called when the class is destroyed, allowing you to automatically clean up allocated resources.

Using the C++ class follows this pattern:

```cpp
had::RingBuffer r_buffer;  
int main() { 
    uint8_t tempCharStorage;     
    // Fill the buffer. 
    for (int i = 0; r_buffer.bufferWrite('A' + i) == 
    had::RingBuffer::OK; i++)    { 
        // 
    } 
    // Read the buffer. 
    while (r_buffer.bufferRead(tempCharStorage) == had::RingBuffer::OK) 
    { 
         // 
    } 
} 
```

This compares to the C version like this:

```cpp
struct RingBuffer buffer;  
int main() { 
    initBuffer(&buffer); 
    uint8_t tempCharStorage;  
    // Fill the buffer. 
    uint8_t i = 0; 
    for (; bufferWrite(&buffer, 'A' + i) == BUFFER_OK; i++) {          
        // 
    }  
    // Read the buffer. 
    while (bufferRead(&buffer, &tempCharStorage) == BUFFER_OK) { // 
    } 
} 
```

Using the C++ class isn't very different from using the C-style method. Not having to do the manual passing of the allocated `struct` instance for each functional call, but instead calling a class method, is probably the biggest difference. This instance is still available in the form of the `this` pointer, which points to the class instance.

While the C++ example uses a namespace and embedded enumeration in the `RingBuffer` class, these are just optional features. One can still use global enumerations, or in the scope of a namespace, or have many layers of namespaces. This is very much determined by the requirements of the application.

As for the cost of using classes, versions of the examples in this section were compiled for the aforementioned Code Craft series for both the Arduino UNO (ATMega328 MCU) and Arduino Due (AT91SAM3X8E MCU) development boards, giving the following file sizes for the compiled code:

|  | **Uno** | **Due** |  |  |
| **C** | **C++** | **C** | **C++** |  |
| **Global scope data** | 614 | 652 | 11,184 | 11,196 |
| **Main scope data** | 664 | 664 | 11,200 | 11,200 |
| **Four instances** | 638 | 676 | 11,224 | 11,228 |

Optimization settings for these code file sizes were set to `-O2`.

Here, we can see that C++ code is identical to C code once compiled, except when we perform initialization of the global class instance, on account of the added code to perform this initialization for us, amounting to 38 bytes for the Uno.

Since only one instance of this code has to exist, this is a constant cost we only have to pay once: in the first and last line, we have one and four class instances or their equivalent, respectively, yet there is only an additional 38 bytes in the Uno firmware. For the Due firmware, we can see something similar, though not as clearly defined. This difference is likely affected by some other settings or optimizations.

What this tells us is that sometimes we don't want to have the compiler initialize a class for us, but we should do it ourselves if we need those last few bytes of ROM or RAM. Most of the time this will not be an issue, however.

# Inheritance

In addition to allowing you to organize code into objects, classes also allow for classes to serve as a template for other classes through the use of polymorphism. In C++, we can combine the properties of any number of classes into a new class, giving it custom properties and methods as well.

This is a very effective way to create **user-defined types** (**UDTs**), especially when combined with operator overloading to use common operators to define operations for addition, subtraction, and so on for the UDT.

Inheritance in C++ follows the following pattern:

```cpp
class B : public A { // Private members. public: // Additional public members. }; 
```

Here, we declare a class, `B`, which derives from class `A`. This allows us to use any public methods defined in class A on an instance of class B, as if they were defined in the latter to begin with.

All of this seems fairly easy to understand, even if things can get a bit confusing the moment we start deriving from more than one base class. However, with proper planning and design, polymorphism can be a very powerful tool.

Unfortunately, none of this answers the question of how much overhead the use of polymorphism adds to our code. We saw earlier that C++ classes by themselves add no overhead during runtime, yet by deriving from one or more base classes, the resulting code would be expected to be significantly more complex.

Fortunately, this is not the case. Much like with simple classes, the resulting derived classes are simple amalgamations of the base structs that underlie the class implementations. The inheritance process itself, along with the validation that comes with it, is primarily a compiler-time issue, bringing with it various benefits for the developer.

# Virtual base classes

At times, it doesn't make a lot of sense for a base class to have an implementation for a class method, yet at the same time we wish to force any derived classes to implement that method. The answer to this problem is virtual methods.

Take the following class definition:

```cpp
class A { 
public: 
   virtual bool methodA() = 0; 
   virtual bool methodB() = 0; 
}; 
```

If we try to derive from this class, we must implement these two class methods or get a compiler error. Since both of the methods in the base class are virtual, the entire base class is referred to as a virtual base class. This is particularly useful for when you wish to define an interface that can be implemented by a range of different classes, yet keep the convenience of having just one user-defined type to refer to.

Internally, virtual methods like these are implemented using `vtables`, which is short for *virtual table*. This is a data structure containing, for each virtual method, a memory address (pointer) pointing to an implementation of that method:

```cpp
VirtualClass* → vtable_ptr → vtable[0] → methodA() 
```

We can compare the performance impact of this level of indirection relative to C-style code and classes with direct method calls. The Code Craft article on the timing of virtual functions ([https://hackaday.com/2015/11/13/code-craft-embedding-c-timing-virtual-functions/](https://hackaday.com/2015/11/13/code-craft-embedding-c-timing-virtual-functions/)) describes such an approach, with interesting findings:

|  | `Uno` | `Due` |  |  |
| `Os` | `O2` | `Os` | `O2` |  |
| **C function call** | 10.4 | 10.2 | 3.7 | 3.6 |
| **C++ direct call** | 10.4 | 10.3 | 3.8 | 3.8 |
| **C++ virtual call** | 11.1 | 10.9 | 3.9 | 3.8 |
| **Multiple C calls** | 110.4 | 106.3 | 39.4 | 35.5 |
| **C function pointer calls** | 105.7 | 102.9 | 38.6 | 34.9 |
| **C++ virtual calls** | 103.2 | 100.4 | 39.5 | 35.2 |

All times listed here are in microseconds.

The same two Arduino development boards are used for this test as for the one comparing compile output size between C code and C++ classes. Two different optimization levels are used to compare the impact of such compiler settings: -Os optimizes for the size of the resulting binary in terms of bytes, where as the `-O2` setting optimizes for speed in a more aggressive manner than the `-O1` optimization level.

From these timings, we can say for sure that the level of indirection introduced by the virtual methods is measurable, although not dramatic, adding a whole 0.7 microseconds on the ATMega328 of the Arduino Uno development board, and about 0.1 microseconds on the faster ARM-based board.

Even in absolute terms, the use of virtual class methods does not carry enough of a performance penalty to truly reconsider its use unless performance is paramount, and this is primarily the case on slower MCUs. The faster the MCU's CPU, the less severe the impact of its use will be.

# Function inlining

The inline keyword in C++ is a hint to the compiler to let it know that we would like each call to a function whose name is preceded by this keyword to result in that function's implementation instead of being copied to the location of the call, thus skipping the overhead of a function call.

This is a compile-time optimization, which only adds the size of the function implementation to the compiler output, once for each distinct call to the inline function.

# Runtime type information

The main purpose of RTTI is to allow the use of safe typecasting, like with the `dynamic_cast<>` operator. As RTTI involves storing additional information for each polymorphic class, it has a certain amount of overhead.

This is a runtime feature, as the name gives away, and thus can be disabled if you don't need the features it provides. Disabling RTTI is common practice on some embedded platforms, especially as it is rarely used on low-resource platforms, such as 8-bit MCUs.

# Exception handling

Exceptions are commonly used on desktop platforms, providing a way to generate exceptions for error conditions, which can be caught and handled in try/catch blocks.

While exception support isn't expensive by itself, an exception being generated is relatively expensive, requiring a significant amount of CPU time and RAM to prepare and handle the exception. You have to also make sure to catch every exception, or risk having the application terminate without clear cause.

Exceptions versus the checking of return code for a method being called is something that has to be decided on a case-by-case basis, and can also be a matter of personal preference. It requires a quite different programming style, which may not work for everyone.

# Templates

It's often thought that templates in C++ are very heavy, and carry a severe penalty for using them. This completely misses the point of templates, which is that templates are merely meant to be used as a shorthand method for automating the generation of nearly identical code from a single template – hence the name.

What this effectively means is that for any function or class template we define, the compiler will generate an inline implementation of the template each time the template is referenced.

This is a pattern we commonly see in the C++ **standard template library** (**STL**), which, as the name suggests, makes heavy use of templates. Take, for example, a data structure like a humble map:

```cpp
std::map<std::string, int> myMap; 
```

What happens here is that the singular template for an `std::map` is taken by the compiler, along with the template parameters we provide within the sharp brackets, filling in the template and writing an inline implementation in its spot.

Effectively, we get the same implementation as if we had written the entire data structure implementation by hand just for those two types. Since the alternative would be to write every implementation by hand for every conceivable built-in type and additional user-defined type, the use of a generic template saves us a lot of time, without sacrificing performance.

# The standard template library

The standard library for C++ (STL) contains a comprehensive and ever-growing collection of functions, classes, and more that allows for common tasks to be performed without having to rely on external libraries. The STL string class is very popular, and allows you to safely handle strings without having to deal with null terminators and anything similar.

Most embedded platforms support all or at least a significant part of the STL, barring limitations on available RAM and the like that prevent the implementation of full hash tables and other complex data structures. Many embedded STL implementations contain optimizations for the target platform, minimizing RAM and CPU usage.

# Maintainability

In the preceding sections, we have seen a number of features that C++ offers, and the viability of using them on a resource-limited platform. A big advantage of using C++ is the reduction in code size you can accomplish through the use of templates, along with the organization and modularization of a code base using classes, namespaces, and the like.

By striving for a more modular approach in your code, with clear interfaces between modules, it becomes more feasible to reuse code between projects. It also simplifies the maintenance of code by making the function of a particular section of code clearer and providing clear targets for unit and integration testing.

# Summary

In this chapter, we tackled the big question of why you would wish to use C++ for embedded development. We saw that, due to the courtesy of C++'s development, it is highly optimized for resource-constrained platforms, while providing a large number of features essential to project management and organization.

The reader should, at this point, be able to describe C++'s main features and provide concrete examples of each. When writing C++ code, the reader will have a clear idea of the cost of a particular language feature, being able to reason why one implementation of a section of code is preferable to another implementation, based on both space and RAM constraints.

In the next chapter, we will take a look at the development process for embedded Linux and similar systems, based on **single-board computers** (**SBCs**) and similar.