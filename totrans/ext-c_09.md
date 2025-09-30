# Chapter 09

# Abstraction and OOP in C++

This is the final chapter on OOP in C. In this chapter, we are going to cover the remaining topics and introduce you to a new programming paradigm. In addition, we explore C++ and look at how it implements object-oriented concepts behind the scenes.

As part of this chapter, we will cover the following topics:

*   Firstly, we discuss the *Abstraction*. This continues our discussion regarding inheritance and polymorphism and will be the last topic that we cover as part of OOP in C. We show how abstraction helps us in designing object models that have the maximum extendibility and the minimum dependencies between its various components.
*   We talk about how object-oriented concepts have been implemented in a famous C++ compiler, `g++` in this case. As part of this, we see that how close the approaches that we have discussed so far are in accordance with the approaches that `g++` has taken to provide the same concepts.

Let's start the chapter by talking about abstraction.

# Abstraction

Abstraction can have a very general meaning in various fields of science and engineering. But in programming, and especially in OOP, abstraction essentially deals with *abstract data types*. In class-based object orientation, abstract data types are the same as *abstract classes*. Abstract classes are special classes that we cannot create an object from; they are not ready or complete enough to be used for object creation. So, why do we need to have such classes or data types? This is because when we work with abstract and general data types, we avoid creating strong dependencies between various parts of code.

As an example, we can have the following relationships between the *Human* and *Apple* classes:

*An object of the Human class eats an object of the Apple class.*

*An object of the Human class eats an object of the Orange class.*

If the classes that an object from the *Human* class can eat were expanded to more than just *Apple* and *Orange*, we would need to add more relations to the *Human* class. Instead, though, we could create an abstract class called *Fruit* that is the parent of both *Apple* and *Orange* classes, and we could set the relation to be between *Human* and *Fruit* only. Therefore, we can turn our preceding two statements into one:

*An object of the Human class eats an object from a subtype of the Fruit class.*

The *Fruit* class is abstract because it lacks information about shape, taste, smell, color, and many more attributes of a specific fruit. Only when we have an apple or an orange do we know the exact values of the different attributes. The *Apple* and *Orange* classes are said to be *concrete types*.

We can even add more abstraction. The *Human* class can eat *Salad* or *Chocolate* as well. So, we can say:

*An object of the Human type eats an object from a subtype of the Eatable class.*

As you can see, the abstraction level of *Eatable* is even higher than that of *Fruit*. Abstraction is a great technique for designing an object model that has minimal dependency on concrete types and allows the maximum future extension to the object model when more concrete types are introduced to the system.

Regarding the preceding example, we could also add further abstraction by using the fact that *Human* is an *Eater*. Then, we could make our statement even more abstract:

*An object from a subtype of the Eater class eats an object from a subtype of the Eatable class.*

We can continue to abstract everything in an object model and find abstract data types that are more abstract than the level we need to solve our problem. This is usually called *over-abstraction*. It happens when you try to create abstract data types that have no real application, either for your current or your future needs. This should be avoided at all costs because abstraction can cause problems, despite all the benefits it provides.

A general guide regarding the amount of abstraction that we need can be found as part of the *abstraction principle*. I got the following quote from its Wikipedia [page, https://en.wikipedia.org/wiki/Abstraction_principle_(computer_program](https://en.wikipedia.org/wiki/Abstraction_principle_(computer_programming))ming). It simply states:

*Each significant piece of functionality in a program should be implemented in just one place in the source code. Where similar functions are carried out by distinct pieces of code, it is generally beneficial to combine them into one by abstracting out the varying parts.*

While at first glance you may not see any sign of object orientation or inheritance in this statement, by giving some further thought to it you will notice that what we did with inheritance was based on this principle. Therefore, as a general rule, whenever you don't expect to have variations in a specific logic, there is no need to introduce abstraction at that point.

In a programming language, inheritance and polymorphism are two capabilities that are required in order to create abstraction. An abstract class such as *Eatable* is a supertype in relation to its concrete classes, such as *Apple*, and this is accomplished by inheritance.

Polymorphism also plays an important role. There are behaviors in an abstract type that *cannot* have default implementation at that abstraction level. For example, *taste* as an attribute implemented using a behavior function such as `eatable_get_taste` as part of the *Eatable* class cannot have an exact value when we are talking about an *Eatable* object. In other words, we cannot create an object directly from the *Eatable* class if we don't know how to define the `eatable_get_taste` behavior function.

The preceding function can only be defined when the child class is concrete enough. For example, we know that *Apple* objects should return *sweet* for their taste (we've assumed here that all apples are sweet). This is where polymorphism helps. It allows a child class to override its parent's behaviors and return the proper taste, for example.

If you remember from the previous chapter, the behavior functions that can be overridden by child classes are called *virtual functions*. Note that it is possible that a virtual function doesn't have any definition at all. Of course, this makes the owner class abstract.

By adding more and more abstraction, at a certain level, we reach classes that have no attributes and contain only virtual functions with no default definitions. These classes are called *interfaces*. In other words, they expose functionalities but they don't offer any implementation at all, and they are usually used to create dependencies between various components in a software project. As an example, in our preceding examples, the *Eater* and *Eatable* classes are interfaces. Note that, just like abstract classes, you must not create an object from an interface. The following code shows why this cannot be done in a C code.

The following code box is the equivalent code written for the preceding interface *Eatable* in C using the techniques we introduced in the previous chapter to implement inheritance and polymorphism:

```cpp
typedef enum {SWEET, SOUR} taste_t;
// Function pointer type
typedef taste_t (*get_taste_func_t)(void*);
typedef struct {
  // Pointer to the definition of the virtual function
  get_taste_func_t get_taste_func;
} eatable_t;
eatable_t* eatable_new() { ... }
void eatable_ctor(eatable_t* eatable) {
  // We don't have any default definition for the virtual function
  eatable->get_taste_func = NULL;
}
// Virtual behavior function
taste_t eatable_get_taste(eatable_t* eatable) {
  return eatable->get_taste_func(eatable);
}
```

Code Box 9-1: The Eatable interface in C

As you can see, in the constructor function we have set the `get_taste_func` pointer to `NULL`. So, calling the `eatable_get_taste` virtual function can lead to a segmentation fault. From the coding perspective, that's basically why that we must not create an object from the *Eatable* interface other than the reasons we know from the definition of the interface and the design point of view.

The following code box demonstrates how creating an object from the *Eatable* interface, which is totally possible and allowed from a C point of view, can lead to a crash and must not be done:

```cpp
eatable_t *eatable = eatable_new();
eatable_ctor(eatable);
taste_t taste = eatable_get_taste(eatable); // Segmentation fault!
free(eatable);
```

Code Box 9-2: Segmentation fault when creating an object from the Eatable interface and calling a pure virtual function from it

To prevent ourselves from creating an object from an abstract type, we can remove the *allocator function* from the class's public interface. If you remember the approaches that we took in the previous chapter to implement inheritance in C, by removing the allocator function, only child classes are able to create objects from the parent's attribute structure.

External codes are then no longer able to do so. For instance, in the preceding example, we do not want any external code to be able to create any object from the structure `eatable_t`. In order to do that, we need to have the attribute structure forward declared and make it an incomplete type. Then, we need to remove the public memory allocator `eatable_new` from the class.

To summarize what we need to do to have an abstract class in C, you need to nullify the virtual function pointers that are not meant to have a default definition at that abstraction level. At an extremely high level of abstraction, we have an interface whose all function pointers are null. To prevent any external code from creating objects from abstract types, we should remove the allocator function from the public interface.

In the following section, we are going to compare similar object-oriented features in C and C++. This gives us an idea how C++ has been developed from pure C.

# Object-oriented constructs in C++

In this section, we are going to compare what we did in C and the underlying mechanisms employed in a famous C++ compiler, `g++` in this case, for supporting encapsulation, inheritance, polymorphism, and abstraction.

We want to show that there is a close accordance between the methods by which object-oriented concepts are implemented in C and C++. Note that, from now on, whenever we refer to C++, we are actually referring to the implementation of `g++` as one of the C++ compilers, and not the C++ standard. Of course, the underlying implementations can be different for various compilers, but we don't expect to see a lot of differences. We will also be using `g++` in a 64-bit Linux setup.

We are going to use the previously discussed techniques to write an object-oriented code in C, and then we write the same program in C++, before jumping to the final conclusion.

## Encapsulation

It is difficult to go deep into a C++ compiler and see how it uses the techniques that we've been exploring so far to produce the final executable, but there is one clever trick that we can use to actually see this. The way to do this is to compare the assembly instructions generated for two similar C and C++ programs.

This is exactly what we are going to do to demonstrate that the C++ compiler ends up generating the same assembly instructions as a C program that uses the OOP techniques that we've been discussing in the previous chapters.

*Example 9.1* is about two C and C++ programs addressing the same simple object-oriented logic. There is a `Rectangle` class in this example, which has a behavior function for calculating its area. We want to see and compare the generated assembly codes for the same behavior function in both programs. The following code box demonstrates the C version:

```cpp
#include <stdio.h>
typedef struct {
  int width;
  int length;
} rect_t;
int rect_area(rect_t* rect) {
  return rect->width * rect->length;
}
int main(int argc, char** argv) {
  rect_t r;
  r.width = 10;
  r.length = 25;
  int area = rect_area(&r);
  printf("Area: %d\n", area);
  return 0;
}
```

Code Box 9-3 [ExtremeC_examples_chapter9_1.c]: Encapsulation example in C

And the following code box shows the C++ version of the preceding program:

```cpp
#include <iostream>
class Rect {
public:
  int Area() {
    return width * length;
  }
  int width;
  int length;
};
int main(int argc, char** argv) {
  Rect r;
  r.width = 10;
  r.length = 25;
  int area = r.Area();
  std::cout << "Area: " << area << std::endl;
  return 0;
}
```

Code Box 9-4 [ExtremeC_examples_chapter9_1.cpp]: Encapsulation example in C++

So, let's generate the assembly codes for the preceding C and C++ programs:

```cpp
$ gcc -S ExtremeC_examples_chapter9_1.c -o ex9_1_c.s
$ g++ -S ExtremeC_examples_chapter9_1.cpp -o ex9_1_cpp.s
$
```

Shell Box 9-1: Generating the assembly outputs for the C and C++ codes

Now, let's dump the `ex9_1_c.s` and `ex9_1_cpp.s` files and look for the definition of the behavior functions. In `ex9_1_c.s`, we should look for the `rect_area` symbol, and in `ex9_1_cpp.s`, we should look for the `_ZN4Rect4AreaEv` symbol. Note that C++ mangles the symbol names, and that's why you need to search for this strange symbol. Name mangling in C++ has been discussed in *Chapter 2*, *Compilation and Linking*.

For the C program, the following is the generated assembly for the `rect_area` function:

```cpp
$ cat ex9_1_c.s
...
rect_area:
.LFB0:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset 6, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register 6
 movq    %rdi, -8(%rbp)
 movq    -8(%rbp), %rax
 movl    (%rax), %edx
 movq    -8(%rbp), %rax
 movl    4(%rax), %eax
    imull   %edx, %eax
    popq    %rbp
    .cfi_def_cfa 7, 8
    Ret
    .cfi_endproc
...
$
```

Shell Box 9-2: The generated assembly code of the rect_area function

The following is the generated assembly instructions for the `Rect::Area` function:

```cpp
$ cat ex9_1_cpp.s
...
_ZN4Rect4AreaEv:
.LFB1493:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset 6, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register 6
 movq    %rdi, -8(%rbp)
 movq    -8(%rbp), %rax
 movl    (%rax), %edx
 movq    -8(%rbp), %rax
 movl    4(%rax), %eax
    imull   %edx, %eax
    popq    %rbp
    .cfi_def_cfa 7, 8
    Ret
    .cfi_endproc
...
$
```

Shell Box 9-3: The generated assembly code of the Rect::Area function

Unbelievably, they are exactly the same! I'm not sure how the C++ code turns into the preceding assembly code, but I'm sure that the assembly code generated for the preceding C function is almost, to high degree of accuracy, equivalent to the assembly code generated for the C++ function.

We can conclude from this that the C++ compiler has used a similar approach to that which we used in C, introduced as *implicit encapsulation* as part of *Chapter 6*, *OOP and Encapsulation*, to implement the encapsulation. Like what we did with implicit encapsulation, you can see in *Code Box 9-3* that a pointer to the attribute structure is passed to the `rect_area` function as the first argument.

As part of the boldened assembly instructions in both shell boxes, the `width` and `length` variables are being read by adding to the memory address passed as the first argument. The first pointer argument can be found in the `%rdi` register according to *System V ABI*. So, we can infer that C++ has changed the `Area` function to accept a pointer argument as its first argument, which points to the object itself.

As a final word on encapsulation, we saw how C and C++ are closely related regarding encapsulation, at least in this simple example. Let's see if the same is true regarding inheritance as well.

## Inheritance

Investigating inheritance is easier than encapsulation. In C++, the pointers from a child class can be assigned to the pointers from the parent class. Also, the child class should have access to the private definition of the parent class.

Both of these behaviors imply that C++ is using our first approach to implementing inheritance, which was discussed in the previous chapter, *Chapter 8, Inheritance and Polymorphism*, along with the second approach. Please refer back to the previous chapter if you need to remind yourself of the two approaches.

However, C++ inheritance seems more complex because C++ supports multiple inheritances that we can't support in our first approach. In this section, we will check the memory layouts of two objects instantiated from two similar classes in C and C++, as demonstrated in *example 9.2*.

*Example 9.2* is about a simple class inheriting from another simple class, both of which have no behavior functions. The C version is as follows:

```cpp
#include <string.h>
typedef struct {
  char c;
  char d;
} a_t;
typedef struct {
  a_t parent;
  char str[5];
} b_t;
int main(int argc, char** argv) {
  b_t b;
  b.parent.c = 'A';
  b.parent.d = 'B';
  strcpy(b.str, "1234");
  // We need to set a break point at this line to see the memory layout.
  return 0;
}
```

Code Box 9-5 [ExtremeC_examples_chapter9_2.c]: Inheritance example in C

And the C++ version comes within the following code box:

```cpp
#include <string.h>
class A {
public:
  char c;
  char d;
};
class B : public A {
public:
  char str[5];
};
int main(int argc, char** argv) {
  B b;
  b.c = 'A';
  b.d = 'B';
  strcpy(b.str, "1234");
  // We need to set a break point at this line to see the memory layout.
  return 0;
}
```

Code Box 9-6 [ExtremeC_examples_chapter9_2.cpp]: Inheritance example in C++

Firstly, we need to compile the C program and use `gdb` to set a breakpoint on the last line of the `main` function. When the execution pauses, we can examine the memory layout as well as the existing values:

```cpp
$ gcc -g ExtremeC_examples_chapter9_2.c -o ex9_2_c.out
$ gdb ./ex9_2_c.out
...
(gdb) b ExtremeC_examples_chapter9_2.c:19
Breakpoint 1 at 0x69e: file ExtremeC_examples_chapter9_2.c, line 19.
(gdb) r
Starting program: .../ex9_2_c.out
Breakpoint 1, main (argc=1, argv=0x7fffffffe358) at ExtremeC_examples_chapter9_2.c:20
20    return 0;
(gdb) x/7c &b
0x7fffffffe261: 65 'A'  66 'B'  49 '1'  50 '2'  51 '3'  52 '4'  0 '\000'
(qdb) c
[Inferior 1 (process 3759) exited normally]
(qdb) q
$
```

Shell Box 9-4: Running the C version of example 9.2 in gdb

As you can see, we have printed seven characters, starting from the address of `b` object, which are as follows: `'A'`, `'B'`, `'1'`, `'2'`, `'3'`, `'4'`, `'\0'`. Let's do the same for the C++ code:

```cpp
$ g++ -g ExtremeC_examples_chapter9_2.cpp -o ex9_2_cpp.out
$ gdb ./ex9_2_cpp.out
...
(gdb) b ExtremeC_examples_chapter9_2.cpp:20
Breakpoint 1 at 0x69b: file ExtremeC_examples_chapter9_2.cpp, line 20.
(gdb) r
Starting program: .../ex9_2_cpp.out
Breakpoint 1, main (argc=1, argv=0x7fffffffe358) at ExtremeC_examples_chapter9_2.cpp:21
21    return 0;
(gdb) x/7c &b
0x7fffffffe251: 65 'A'  66 'B'  49 '1'  50 '2'  51 '3'  52 '4'  0 '\000'
(qdb) c
[Inferior 1 (process 3804) exited normally]
(qdb) q
$
```

Shell Box 9-5: Running the C++ version of example 9.2 in gdb

As you can see in the preceding two shell boxes, the memory layout and the values stored in the attributes are the same. You shouldn't get confused by having the behavior functions and attributes together in a class in C++; they are going to be treated separately from the class. In C++, the attributes, no matter where you put them in a class, are always collected within the same memory block regarding a specific object, and functions will always be independent of the attributes, just as we saw when looking at *implicit encapsulation* as part of *Chapter 6*, *OOP and Encapsulation*.

The previous example demonstrates *single inheritance*. So, what about *multiple inheritance*? In the previous chapter, we explained why our first approach to implementing inheritance in C could not support multiple inheritance. We again demonstrate the reason in the following code box:

```cpp
typedef struct { ... } a_t;
typedef struct { ... } b_t;
typedef struct {
  a_t a;
  b_t b;
  ...
} c_t;
c_t c_obj;
a_t* a_ptr = (a_ptr*)&c_obj;
b_t* b_ptr = (b_ptr*)&c_obj;
c_t* c_ptr = &c_obj;
```

Code Box 9-7: Demonstration of why multiple inheritance cannot work with our proposed first approach for implementing inheritance in C

In the preceding code box, the `c_t` class desires to inherit both `a_t` and `b_t` classes. After declaring the classes, we create the `c_obj` object. In the following lines of preceding code, we create different pointers.

An important note here is that *all of these pointers must be pointing to the same address*. The `a_ptr` and `c_ptr` pointers can be used safely with any behavior function from the `a_t` and `c_t` classes, but the `b_ptr` pointer is dangerous to use because it is pointing to the a field in the `c_t` class, which is an `a_t` object. Trying to access the fields inside `b_t` through `b_ptr` results in an undefined behavior.

The following code is the correct version of the preceding code, where all pointers can be used safely:

```cpp
c_t c_obj;
a_t* a_ptr = (a_ptr*)&c_obj;
b_t* b_ptr = (b_ptr*)(&c_obj + sizeof(a_t));
c_t* c_ptr = &c_obj;
```

Code Box 9-8: Demonstration of how casts should be updated to point to the correct fields

As you can see on the third line in *Code Box 9-8*, we have added the size of an `a_t` object to the address of `c_obj`; this eventually results in a pointer pointing to the `b` field in `c_t`. Note that casting in C does not do any magic; it is there to convert types and it doesn't modify the transferring value, the memory address in the preceding case. Eventually, after the assignment, the address from the right-hand side would be copied to the left-hand side.

For now, let's see the same example in C++ with a look at *example 9.3*. Suppose that we have a `D` class that inherits from three different classes, `A`, `B`, and `C`. The following is the code written for *example 9.3*:

```cpp
#include <string.h>
class A {
public:
  char a;
  char b[4];
};
class B {
public:
  char c;
  char d;
};
class C {
public:
  char e;
  char f;
};
class D : public A, public B, public C {
public:
  char str[5];
};
int main(int argc, char** argv) {
  D d;
  d.a = 'A';
  strcpy(d.b, "BBB");
  d.c = 'C';
  d.d = 'D';
  d.e = 'E';
  d.f = 'F';
  strcpy(d.str, "1234");
  A* ap = &d;
  B* bp = &d;
  C* cp = &d;
  D* dp = &d;
  // We need to set a break point at this line.
  return 0;
}
```

Code Box 9-9 [ExtremeC_examples_chapter9_3.cpp]: Multiple inheritance in C++

Let's compile the example and run it with `gdb`:

```cpp
$ g++ -g ExtremeC_examples_chapter9_3.cpp -o ex9_3.out
$ gdb ./ex9_3.out
...
(gdb) b ExtremeC_examples_chapter9_3.cpp:40
Breakpoint 1 at 0x100000f78: file ExtremeC_examples_chapter9_3.cpp, line 40.
(gdb) r
Starting program: .../ex9_3.out
Breakpoint 1, main (argc=1, argv=0x7fffffffe358) at ExtremeC_examples_chapter9_3.cpp:41
41    return 0;
(gdb) x/14c &d
0x7fffffffe25a: 65 'A'  66 'B'  66 'B'  66 'B'  0 '\000'    67 'C'  68 'D'  69 'E'
0x7fffffffe262: 70 'F'  49 '1'  50 '2'  51 '3'  52 '4'  0 '\000'
(gdb)
$
```

Shell Box 9-6: Compiling and running example 9.3 in gdb

As you can see, the attributes are placed adjacent to each other. This shows that multiple objects of the parent classes are being kept inside the same memory layout of the `d` object. What about the `ap`, `bp`, `cp`, and `dp` pointers? As you can see, in C++, we can cast implicitly when assigning a child pointer to a parent pointer (upcasting).

Let's examine the values of these pointers in the current execution:

```cpp
(gdb) print ap
$1 = (A *) 0x7fffffffe25a
(gdb) print bp
$2 = (B *) 0x7fffffffe25f
(gdb) print cp
$3 = (C *) 0x7fffffffe261
(gdb) print dp
$4 = (D *) 0x7fffffffe25a
(gdb)
```

Shell Box 9-7: Printing the addresses stored in the pointers as part of example 9.3

The preceding shell box shows that the starting address of the d object, shown as `$4`, is the same as the address being pointed to by ap, shown as `$1`. So, this clearly shows that C++ puts an object of the type *A* as the first field in the corresponding attribute structure of the *D* class. Based on the addresses in the pointers and the result we got from the `x` command, an object of the *B* type and then an object of the *C* type, are put into the same memory layout belonging to object `d`.

In addition, the preceding addresses show that the cast in C++ is not a passive operation, and it can perform some pointer arithmetic on the transferring address while converting the types. For example, in *Code Box 9-9*, while assigning the `bp` pointer in the `main` function, five bytes or `sizeof(A)`, are added to the address of `d`. This is done in order to overcome the problem we found in implementing multiple inheritance in C. Now, these pointers can easily be used in all behavior functions without needing to do the arithmetic yourself. As an important note, C casts and C++ casts are different, and you may see different behavior if you assume that C++ casts are as passive as C casts.

Now it's time to look at the similarities between C and C++ in the case of polymorphism.

## Polymorphism

Comparing the underlying techniques for having polymorphism in C and C++ is not an easy task. In the previous chapter, we came up with a simple method for having a polymorphic behavior function in C, but C++ uses a much more sophisticated mechanism to bring about polymorphism, though the basic underlying idea is still the same. If we want to generalize our approach for implementing polymorphism in C, we can do it as the pseudo-code that can be seen in the following code box:

```cpp
// Typedefing function pointer types
typedef void* (*func_1_t)(void*, ...);
typedef void* (*func_2_t)(void*, ...);
...
typedef void* (*func_n_t)(void*, ...);
// Attribute structure of the parent class
typedef struct {
  // Attributes
  ...
  // Pointers to functions
  func_1_t func_1;
  func_2_t func_2;
  ...
  func_n_t func_t;
} parent_t;
// Default private definitions for the
// virtual behavior functions
void* __default_func_1(void* parent, ...) {  // Default definition }
void* __default_func_2(void* parent, ...) {  // Default definition }
...
void* __default_func_n(void* parent, ...) {  // Default definition }
// Constructor
void parent_ctor(parent_t *parent) {
  // Initializing attributes
  ...
  // Setting default definitions for virtual
  // behavior functions
  parent->func_1 = __default_func_1;
  parent->func_2 = __default_func_2;
  ...
  parent->func_n = __default_func_n;
}
// Public and non-virtual behavior functions
void* parent_non_virt_func_1(parent_t* parent, ...) { // Code }
void* parent_non_virt_func_2(parent_t* parent, ...) { // Code }
...
void* parent_non_virt_func_m(parent_t* parent, ...) { // Code }
// Actual public virtual behavior functions
void* parent_func_1(parent_t* parent, ...) {
  return parent->func_1(parent, ...); 
}
void* parent_func_2(parent_t* parent, ...) {
  return parent->func_2(parent, ...); 
}
...
void* parent_func_n(parent_t* parent, ...) { 
  return parent->func_n(parent, ...); 
}
```

Code Box 9-10: Pseudo-code demonstrating how virtual functions can be declared and defined in a C code

As you can see in the preceding pseudo-code, the parent class has to maintain a list of function pointers in its attribute structure. These function pointers (in the parent class) either point to the default definitions for the virtual functions, or they are null. The pseudo-class defined as part of *Code Box 9-10* has `m` non-virtual behavior functions and `n` virtual behavior functions.

**Note**:

Not all behavior functions are polymorphic. Polymorphic behavior functions are called virtual behavior functions or simply virtual functions. In some languages, such as Java, they are called *virtual methods*.

Non-virtual functions are not polymorphic, and you never get various behaviors by calling them. In other words, a call to a non-virtual function is a simple function call and it just performs the logic inside the definition and doesn't relay the call to another function. However, virtual functions need to redirect the call to a proper function, set by either the parent or the child constructor. If a child class wants to override some of the inherited virtual functions, it should update the virtual function pointers.

**Note**:

The `void*` type for the output variables can be replaced by any other pointer type. I used a generic pointer to show that anything can be returned from the functions in the pseudo-code.

The following pseudo-code shows how a child class overrides a few of the virtual functions found in *Code Box 9-10*:

```cpp
Include everything related to parent class ...
typedef struct {
  parent_t parent;
  // Child attributes
  ...
} child_t;
void* __child_func_4(void* parent, ...) { // Overriding definition }
void* __child_func_7(void* parent, ...) { // Overriding definition }
void child_ctor(child_t* child) {
  parent_ctor((parent_t*)child);
  // Initialize child attributes
  ...
  // Update pointers to functions
  child->parent.func_4 = __child_func_4;
  child->parent.func_7 = __child_func_7;
}
// Child's behavior functions
...
```

Code Box 9-11: Pseudo-code in C demonstrating how a child class can override some virtual functions inherited from the parent class

As you can see in *Code Box 9-11*, the child class needs only to update a few pointers in the parent's attribute structure. C++ takes a similar approach. When you declare a behavior function as virtual (using the `virtual` keyword), C++ creates an array of function pointers, pretty similar to the way we did in *Code Box 9-10*.

As you can see, we added one function pointer attribute for each virtual function, but C++ has a smarter way of keeping these pointers. It just uses an array called a *virtual table* or *vtable*. The virtual table is created when an object is about to be created. It is first populated while calling the constructor of the base class, and then as part of the constructor of the child class, just as we've shown in *Code Boxes 9-10* and *9-11*.

Since the virtual table is only populated in the constructors, calling a polymorphic method in a constructor, either in the parent or in the child class, should be avoided, as its pointer may have not been updated yet and it might be pointing to an incorrect definition.

As our last discussion regarding the underlying mechanisms used for having various object-oriented concepts in C and C++, we are going to talk about abstraction.

## Abstract classes

Abstraction in C++ is possible using *pure virtual* functions. In C++ if you define a member function as a virtual function and set it to zero, you have declared a pure virtual function. Look at the following example:

```cpp
enum class Taste { Sweet, Sour };
// This is an interface
class Eatable {
public:
  virtual Taste GetTaste() = 0;
};
```

Code Box 9-12: The Eatable interface in C++

Inside the class `Eatable`, we have a `GetTaste` virtual function that is set to zero. `GetTaste` is a pure virtual function and makes the whole class abstract. You can no longer create objects from the *Eatable* type, and C++ doesn't allow this. In addition, *Eatable* is an interface, because all of its member functions are purely virtual. This function can be overridden in a child class.

The following shows a class that is overriding the `GetTaste` function:

```cpp
enum class Taste { Sweet, Sour };
// This is an interface
class Eatable {
public:
  virtual Taste GetTaste() = 0;
};
class Apple : public Eatable {
public:
  Taste GetTaste() override {
    return Taste::Sweet;
  }
};
```

Code Box 9-13: Two child classes implementing the Eatable interface

Pure virtual functions are remarkably similar to virtual functions. The addresses to the actual definitions are being kept in the virtual table in the same way as virtual functions, but with one difference. The initial values for the pointers of pure virtual functions are null, unlike the pointers of normal virtual functions, which need to point to a default definition while the construction is in progress.

Unlike a C compiler, which doesn't know anything about abstract types, a C++ compiler is aware of abstract types and generates a compilation error if you try to create an object from an abstract type.

In this section, we took various object-oriented concepts and compared them in C, using the techniques introduced in the past three chapters, and in C++, using the `g++` compiler. We showed that, in most cases, the approaches we employed are in accordance with the techniques that a compiler like `g++` uses.

# Summary

In this chapter, we concluded our exploration of topics in OOP, picking up from abstraction and moving on by showing the similarities between C and C++ regarding object-oriented concepts.

The following topics were discussed as part of this chapter:

*   Abstract classes and interfaces were initially discussed. Using them, we can have an interface or a partially abstract class, which could be used to create concrete child classes with polymorphic and different behaviors.
*   We then compared the output of the techniques we used in C to bring in some OOP features, with the output of what `g++` produces. This was to demonstrate how similar the results are. We concluded that the techniques that we employed can be very similar in their outcomes.
*   We discussed virtual tables in greater depth.
*   We showed how pure virtual functions (which is a C++ concept but does have a C counterpart) can be used to declare virtual behaviors that have no default definition.

The next chapter is about Unix and its correspondence to C. It will review the history of Unix and the invention of C. It will also explain the layered architecture of a Unix system.