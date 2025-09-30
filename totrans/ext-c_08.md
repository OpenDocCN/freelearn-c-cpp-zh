# Chapter 08

# Inheritance and Polymorphism

This chapter is a continuation of the previous two chapters, where we introduced how you can do OOP in C and reached the concepts of composition and aggregation. This chapter mainly continues the discussion regarding relationships between objects and their corresponding classes and covers inheritance and polymorphism. As part of this chapter, we conclude this topic and we continue with *Abstraction* in the following chapter.

This chapter is heavily dependent on the theory explained in the previous two chapters, where we were discussing the possible relationships between classes. We explained *composition* and *aggregation* relationships, and now we are going to talk about the *extension* or *inheritance* relationship in this chapter, along with a few other topics.

The following are the topics that will be explained throughout this chapter:

*   As explained earlier, the inheritance relationship is the first topic that we discuss. The methods for implementing the inheritance relationship in C will be covered, and we will conduct a comparison between them.
*   The next big topic is *polymorphism*. Polymorphism allows us to have different versions of the same behavior in the child classes, in the case of having an inheritance relationship between those classes. We will discuss the methods for having a polymorphic function in C; this will be the first step in our understanding of how C++ offers polymorphism.

Let's start our discussion with the inheritance relationship.

# Inheritance

We closed the previous chapter by talking about *to-have* relationships, which eventually led us to composition and aggregation relationships. In this section, we are going to talk about *to-be* or *is-a* relationships. The inheritance relationship is a to-be relationship.

An inheritance relationship can also be called an *extension relationship* because it only adds extra attributes and behaviors to an existing object or class. In the following sections, we'll explain what inheritance means and how it can be implemented in C.

There are situations when an object needs to have the same attributes that exist in another object. In other words, the new object is an extension to the other object.

For example, a student has all the attributes of a person, but may also have extra attributes. See *Code Box 8-1*:

```cpp
typedef struct {
  char first_name[32];
  char last_name[32];
  unsigned int birth_year;
} person_t;
typedef struct {
  char first_name[32];
  char last_name[32];
  unsigned int birth_year;
 char student_number[16]; // Extra attribute
 unsigned int passed_credits; // Extra attribute
} student_t;
```

Code Box 8-1: The attribute structures of the Person class and the Student class

This example clearly shows how `student_t` extends the attributes of `person_t` with new attributes, `student_number` and `passed_credits`, which are student-specific attributes.

As we have pointed out before, inheritance (or extension) is a to-be relationship, unlike composition and aggregation, which are to-have relationships. Therefore, for the preceding example, we can say that "a student is a person," which seems to be correct in the domain of educational software. Whenever a to-be relationship exists in a domain, it is probably an inheritance relationship. In the preceding example, `person_t` is usually called the *supertype*, or the *base* type, or simply the *parent* type, and `student_t` is usually called the *child* type or the *inherited subtype*.

## The nature of inheritance

If you were to dig deeper and see what an inheritance relationship really is, you would find out that it is really a composition relationship in its nature. For example, we can say that a student has a person's nature inside of them. In other words, we can suppose that there is a private person object inside the `Student` class's attribute structure. That is, an inheritance relationship can be equivalent to a one-to-one composition relationship.

So, the structures in *Code Box 8-1* can be written as:

```cpp
typedef struct {
  char first_name[32];
  char last_name[32];
  unsigned int birth_year;
} person_t;
typedef struct {
 person_t person;
  char student_number[16]; // Extra attribute
  unsigned int passed_credits; // Extra attribute
} student_t;
```

Code Box 8-2: The attribute structures of the Person and Student classes but nested this time

This syntax is totally valid in C, and in fact nesting structures by using structure variables (not pointers) is a powerful setup. It allows you to have a structure variable inside your new structure that is really an extension to the former.

With the preceding setup, necessarily having a field of type `person_t` as the first field, a `student_t` pointer can be easily cast to a `person_t` pointer, and both of them can point to the same address in memory.

This is called *upcasting*. In other words, casting a pointer of the type of the child's attribute structure to the type of the parent's attribute structure is upcasting. Note that with structure variables, you cannot have this feature.

*Example 8.1* demonstrates this as follows:

```cpp
#include <stdio.h>
typedef struct {
  char first_name[32];
  char last_name[32];
  unsigned int birth_year;
} person_t;
typedef struct {
  person_t person;
  char student_number[16]; // Extra attribute
  unsigned int passed_credits; // Extra attribute
} student_t;
int main(int argc, char** argv) {
  student_t s;
 student_t* s_ptr = &s;
 person_t* p_ptr = (person_t*)&s;
  printf("Student pointer points to %p\n", (void*)s_ptr);
  printf("Person pointer points to %p\n", (void*)p_ptr);
  return 0;
}
```

Code Box 8-3 [ExtremeC_examples_chapter8_1.c]: Example 8.1, showing upcasting between Student and Person object pointers

As you can see, we expect that the `s_ptr` and `p_ptr` pointers are pointing to the same address in memory. The following is the output after building and running *example 8.1*:

```cpp
$ gcc ExtremeC_examples_chapter8_1.c -o ex8_1.out
$ ./ex8_1.out
Student pointer points to 0x7ffeecd41810
Person pointer points to 0x7ffeecd41810
$
```

Shell Box 8-1: The output of example 8.1

And yes, they are pointing to the same address. Note that the shown addresses can be different in each run, but the point is that the pointers are referring to the same address. This means that a structure variable of the type `student_t` is really inheriting the `person_t` structure in its memory layout. This implies that we can use the function behaviors of the `Person` class with a pointer that is pointing to a `student` object. In other words, the `Person` class's behavior functions can be reused for the `student` objects, which is a great achievement.

Note that the following is wrong, and the code won't compile:

```cpp
struct person_t;
typedef struct {
 struct person_t person; // Generates an error! 
  char student_number[16]; // Extra attribute
  unsigned int passed_credits; // Extra attribute
} student_t;
```

Code Box 8-4: Establishing an inheritance relationship which doesn't compile!

The line declaring the `person` field generates an error because you cannot create a variable from an *incomplete type*. You should remember that the forward declaration of a structure (similar to the first line in *Code Box 8-4*) results in the declaration of an incomplete type. You can have only pointers of incomplete types, *not* variables. As you've seen before, you cannot even allocate Heap memory for an incomplete type.

So, what does this mean? It means that if you're going to use nested structure variables in order to implement inheritance, the `student_t` structure should see the actual definition of `person_t`, which, based on what we learned about encapsulation, should be private and not visible to any other class.

Therefore, you have two approaches for implementing the inheritance relationship:

*   Make it so that the child class has access to the private implementation (actual definition) of the base class.
*   Make it so that the child class only has access to the public interface of the base class.

### The first approach for having inheritance in C

We'll demonstrate the first approach in the following example, *example 8.2*, and the second approach in *example 8.3*, which will come up in the next section. Both of them represent the same classes, `Student` and `Person`, with some behavior functions, having some objects playing in a simple scenario in the `main` function.

We'll start with *example 8.2*, in which the `Student` class needs to have access to the actual private definition of the `Person` class's attribute structure. The following code boxes present the headers and the sources for the `Student` and `Person` classes together with the `main` function. Let's start with the header file declaring the `Person` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_2_PERSON_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_2_PERSON_H
// Forward declaration
struct person_t;
// Memory allocator
struct person_t* person_new();
// Constructor
void person_ctor(struct person_t*,
                 const char*  /* first name */,
                 const char*  /* last name */,
                 unsigned int /* birth year */);
// Destructor
void person_dtor(struct person_t*);
// Behavior functions
void person_get_first_name(struct person_t*, char*);
void person_get_last_name(struct person_t*, char*);
unsigned int person_get_birth_year(struct person_t*);
#endif
```

Code Box 8-5 [ExtremeC_examples_chapter8_2_person.h]: Example 8.2, the public interface of the Person class

Look at the constructor function in *Code Box 8-5*. It accepts all the values required for creating a `person` object: `first_name`, `second_name`, and `birth_year`. As you see, the attribute structure `person_t` is incomplete, hence the `Student` class cannot use the preceding header file for establishing an inheritance relationship, similar to what we demonstrated in the previous section.

On the other hand, the preceding header file must not contain the actual definition of the attribute structure `person_t`, since the preceding header file is going to be used by other parts of the code which should not know anything about the `Person` internals. So, what should we do? We want a certain part of the logic to know about a structure definition that other parts of the code must not know about. That's where *private header files* jump in.

A private header file is an ordinary header file that is supposed to be included and used by a certain part of code or a certain class that actually needs it. Regarding *example 8.2*, the actual definition of `person_t` should be part of a private header. In the following code box, you will see an example of a private header file:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_2_PERSON_P_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_2_PERSON_P_H
// Private definition
typedef struct {
  char first_name[32];
  char last_name[32];
  unsigned int birth_year;
} person_t;
#endif
```

Code Box 8-6 [ExtremeC_examples_chapter8_2_person_p.h]: The private header file which contains the actual definition of person_t

As you see, it only contains the definition of the `person_t` structure and nothing more than that. This is the part of the `Person` class which should stay private, but it needs to become public to the `Student` class. We are going to need this definition for defining the `student_t` attribute structure. The next code box demonstrates the private implementation of the `Person` class:

```cpp
#include <stdlib.h>
#include <string.h>
// person_t is defined in the following header file.
#include "ExtremeC_examples_chapter8_2_person_p.h"
// Memory allocator
person_t* person_new() {
  return (person_t*)malloc(sizeof(person_t));
}
// Constructor
void person_ctor(person_t* person,
                 const char* first_name,
                 const char* last_name,
                 unsigned int birth_year) {
  strcpy(person->first_name, first_name);
  strcpy(person->last_name, last_name);
  person->birth_year = birth_year;
}
// Destructor
void person_dtor(person_t* person) {
  // Nothing to do
}
// Behavior functions
void person_get_first_name(person_t* person, char* buffer) {
  strcpy(buffer, person->first_name);
}
void person_get_last_name(person_t* person, char* buffer) {
  strcpy(buffer, person->last_name);
}
unsigned int person_get_birth_year(person_t* person) {
  return person->birth_year;
}
```

Code Box 8-7 [ExtremeC_examples_chapter8_2_person.c]: The definition of the Person class

There is nothing special about the definition of the `Person` class and it is like all previous examples. The following code box shows the public interface of the `Student` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_2_STUDENT_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_2_STUDENT_H
//Forward declaration
struct student_t;
// Memory allocator
struct student_t* student_new();
// Constructor
void student_ctor(struct student_t*,
                  const char*  /* first name */,
                  const char*  /* last name */,
                  unsigned int /* birth year */,
                  const char*  /* student number */,
                  unsigned int /* passed credits */);
// Destructor
void student_dtor(struct student_t*);
// Behavior functions
void student_get_student_number(struct student_t*, char*);
unsigned int student_get_passed_credits(struct student_t*);
#endif
```

Code Box 8-8 [ExtremeC_examples_chapter8_2_student.h]: The public interface of the Student class

As you can see, the constructor of the class accepts similar arguments to the `Person` class's constructor. That's because a `student` object actually contains a `person` object and it needs those values for populating its composed `person` object.

This implies that the `student` constructor needs to set the attributes for the `person` part of the `student`.

Note that we have only two additional behavior functions as part of the `Student` class, and that's because we can use the `Person` class's behavior functions for `student` objects as well.

The next code box contains the private implementation of the `Student` class:

```cpp
#include <stdlib.h>
#include <string.h>
#include "ExtremeC_examples_chapter8_2_person.h"
// person_t is defined in the following header
// file and we need it here.
#include "ExtremeC_examples_chapter8_2_person_p.h"
//Forward declaration
typedef struct {
  // Here, we inherit all attributes from the person class and
  // also we can use all of its behavior functions because of
  // this nesting.
 person_t person;
  char* student_number;
  unsigned int passed_credits;
} student_t;
// Memory allocator
student_t* student_new() {
  return (student_t*)malloc(sizeof(student_t));
}
// Constructor
void student_ctor(student_t* student,
                  const char* first_name,
                  const char* last_name,
                  unsigned int birth_year,
                  const char* student_number,
                  unsigned int passed_credits) {
  // Call the constructor of the parent class
 person_ctor((struct person_t*)student,
 first_name, last_name, birth_year);
  student->student_number = (char*)malloc(16 * sizeof(char));
  strcpy(student->student_number, student_number);
  student->passed_credits = passed_credits;
}
// Destructor
void student_dtor(student_t* student) {
  // We need to destruct the child object first.
  free(student->student_number);
  // Then, we need to call the destructor function
  // of the parent class
 person_dtor((struct person_t*)student);
}
// Behavior functions
void student_get_student_number(student_t* student,
                                char* buffer) {
  strcpy(buffer, student->student_number);
}
unsigned int student_get_passed_credits(student_t* student) {
  return student->passed_credits;
}
```

Code Box 8-9 [ExtremeC_examples_chapter8_2_student.c]: The private definition of the Student class

The preceding code box contains the most important code regarding the inheritance relationship. Firstly, we needed to include the private header of the `Person` class because as part of defining `student_t`, we want to have the first field from the `person_t` type. And, since that field is an actual variable and not a pointer, it requires that we have `person_t` already defined. Note that this variable must be the *first field* in the structure. Otherwise, we lose the possibility of using the `Person` class's behavior functions.

Again, in the preceding code box, as part of the `Student` class's constructor, we call the parent's constructor to initialize the attributes of the parent (composed) object. Look at how we cast the `student_t` pointer to a `person_t` pointer when passing it to the `person_ctor` function. This is possible just because the `person` field is the first member of `student_t`.

Similarly, as part of the `Student` class's destructor, we called the parent's destructor. This destruction should happen first at the child level and then the parent level, in the opposite order of construction. The next code box contains *example 8.2*'s main scenario, which is going to use the `Student` class and create an object of type `Student`:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "ExtremeC_examples_chapter8_2_person.h"
#include "ExtremeC_examples_chapter8_2_student.h"
int main(int argc, char** argv) {
  // Create and construct the student object
  struct student_t* student = student_new();
  student_ctor(student, "John", "Doe",
          1987, "TA5667", 134);
  // Now, we use person's behavior functions to
  // read person's attributes from the student object
  char buffer[32];
  // Upcasting to a pointer of parent type
 struct person_t* person_ptr = (struct person_t*)student;
  person_get_first_name(person_ptr, buffer);
  printf("First name: %s\n", buffer);
  person_get_last_name(person_ptr, buffer);
  printf("Last name: %s\n", buffer);
  printf("Birth year: %d\n", person_get_birth_year(person_ptr)); 
  // Now, we read the attributes specific to the student object.
  student_get_student_number(student, buffer);
  printf("Student number: %s\n", buffer);
  printf("Passed credits: %d\n",
          student_get_passed_credits(student));
  // Destruct and free the student object
  student_dtor(student);
  free(student);
  return 0;
}
```

Code Box 8-10 [ExtremeC_examples_chapter8_2_main.c]: The main scenario of example 8.2

As you see in the main scenario, we have included the public interfaces of both the `Person` and `Student` classes (not the private header file), but we have only created one `student` object. As you can see, the `student` object has inherited all attributes from its internal `person` object, and they can be read via the `Person` class's behavior functions.

The following shell box shows how to compile and run *example 8.2*:

```cpp
$ gcc -c ExtremeC_examples_chapter8_2_person.c -o person.o
$ gcc -c ExtremeC_examples_chapter8_2_student.c -o student.o
$ gcc -c ExtremeC_examples_chapter8_2_main.c -o main.o
$ gcc person.o student.o main.o -o ex8_2.out
$ ./ex8_2.out
First name: John
Last name: Doe
Birth year: 1987
Student number: TA5667
Passed credits: 134
$
```

Shell Box 8-2: Building and running example 8.2

The following example, *example 8.3*, will address the second approach to implementing inheritance relationships in C. The output should be very similar to *example 8.2*.

### The second approach to inheritance in C

Using the first approach, we kept a structure variable as the first field in the child's attribute structure. Now, using the second approach, we'll keep a pointer to the parent's structure variable. This way, the child class can be independent of the implementation of the parent class, which is a good thing, considering information-hiding concerns.

We gain some advantages, and we lose some by choosing the second approach. After demonstrating *example 8.3* we will conduct a comparison between the two approaches, and you will see the advantages and disadvantages of using each of these techniques.

*Example 8.3*, below, is remarkably similar to *example 8.2*, especially in terms of the output and the final results. However, the main difference is that as part of this example, the `Student` class only relies on the public interface of the `Person` class, and not its private definition. This is great because it decouples the classes and allows us to easily change the implementation of the parent class without altering the implementation of the child class.

In the preceding example, the `Student` class didn't strictly violate information-hiding principles, but it could have done that because it had access to the actual definition of `person_t` and its fields. As a result, it could read or modify the fields without using `Person`'s behavior functions.

As noted, *example 8.3* is remarkably similar to *example 8.2*, but it has some fundamental differences. The `Person` class has the same public interface as part of the new example. But this is not true regarding the `Student` class and its public interface has to be changed. The following code box shows the `Student` class's new public interface:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_3_STUDENT_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_3_STUDENT_H
//Forward declaration
struct student_t;
// Memory allocator
struct student_t* student_new();
// Constructor
void student_ctor(struct student_t*,
                  const char*  /* first name */,
                  const char*  /* last name */,
                  unsigned int /* birth year */,
                  const char*  /* student number */,
                  unsigned int /* passed credits */);
// Destructor
void student_dtor(struct student_t*);
// Behavior functions
void student_get_first_name(struct student_t*, char*);
void student_get_last_name(struct student_t*, char*);
unsigned int student_get_birth_year(struct student_t*);
void student_get_student_number(struct student_t*, char*);
unsigned int student_get_passed_credits(struct student_t*);
#endif
```

Code Box 8-11 [ExtremeC_examples_chapter8_3_student.h]: The new public interface of the Student class

For reasons you will realize shortly, the `Student` class has to repeat all the behavior functions declared as part of the `Person` class. That's because of the fact that we can no longer cast a `student_t` pointer to a `person_t` pointer. In other words, upcasting doesn't work anymore regarding `Student` and `Person` pointers.

While the public interface of the `Person` class is not changed from *example 8.2*, its implementation has changed. The following code box demonstrates the implementation of the `Person` class as part of *example 8.3*:

```cpp
#include <stdlib.h>
#include <string.h>
// Private definition
typedef struct {
  char first_name[32];
  char last_name[32];
  unsigned int birth_year;
} person_t;
// Memory allocator
person_t* person_new() {
  return (person_t*)malloc(sizeof(person_t));
}
// Constructor
void person_ctor(person_t* person,
                 const char* first_name,
                 const char* last_name,
                 unsigned int birth_year) {
  strcpy(person->first_name, first_name);
  strcpy(person->last_name, last_name);
  person->birth_year = birth_year;
}
// Destructor
void person_dtor(person_t* person) {
  // Nothing to do
}
// Behavior functions
void person_get_first_name(person_t* person, char* buffer) {
  strcpy(buffer, person->first_name);
}
void person_get_last_name(person_t* person, char* buffer) {
  strcpy(buffer, person->last_name);
}
unsigned int person_get_birth_year(person_t* person) {
  return person->birth_year;
}
```

Code Box 8-12 [ExtremeC_examples_chapter8_3_person.c]: The new implementation of the Person class

As you see, the private definition of `person_t` is placed inside the source file and we are not using a private header anymore. This means that we are not going to share the definition with other classes such as the `Student` class at all. We want to conduct a complete encapsulation of the `Person` class and hide all its implementation details.

The following is the private implementation of the `Student` class:

```cpp
#include <stdlib.h>
#include <string.h>
// Public interface of the person class
#include "ExtremeC_examples_chapter8_3_person.h"
//Forward declaration
typedef struct {
  char* student_number;
  unsigned int passed_credits;
  // We have to have a pointer here since the type
  // person_t is incomplete.
 struct person_t* person;
} student_t;
// Memory allocator
student_t* student_new() {
  return (student_t*)malloc(sizeof(student_t));
}
// Constructor
void student_ctor(student_t* student,
                  const char* first_name,
                  const char* last_name,
                  unsigned int birth_year,
                  const char* student_number,
                  unsigned int passed_credits) {
  // Allocate memory for the parent object
 student->person = person_new();
 person_ctor(student->person, first_name,
 last_name, birth_year);
  student->student_number = (char*)malloc(16 * sizeof(char));
  strcpy(student->student_number, student_number);
  student->passed_credits = passed_credits;
}
// Destructor
void student_dtor(student_t* student) {
  // We need to destruct the child object first.
  free(student->student_number);
  // Then, we need to call the destructor function
  // of the parent class
 person_dtor(student->person);
  // And we need to free the parent object's allocated memory
 free(student->person);
}
// Behavior functions
void student_get_first_name(student_t* student, char* buffer) {
  // We have to use person's behavior function
  person_get_first_name(student->person, buffer);
}
void student_get_last_name(student_t* student, char* buffer) {
  // We have to use person's behavior function
  person_get_last_name(student->person, buffer);
}
unsigned int student_get_birth_year(student_t* student) {
  // We have to use person's behavior function
  return person_get_birth_year(student->person);
}
void student_get_student_number(student_t* student,
                                char* buffer) {
  strcpy(buffer, student->student_number);
}
unsigned int student_get_passed_credits(student_t* student) {
  return student->passed_credits;
}
```

Code Box 8-13 [ExtremeC_examples_chapter8_3_student.c]: The new implementation of the Student class

As demonstrated in the preceding code box, we've used the `Person` class's public interface by including its header file. In addition, as part of the definition of `student_t`, we've added a pointer field, which points to the parent `Person` object. This should remind you of the implementation of a composition relationship done as part of the previous chapter.

Note that there is no need for this pointer field to be the first item in the attribute structure. This is in contrast to what we saw in the first approach. The pointers of the types `student_t` and `person_t` are no longer interchangeable, and they are pointing to different addresses in the memory that are not necessarily adjacent. This is again in contrast to what we did in the previous approach.

Note that, as part of the `Student` class's constructor, we instantiate the parent object. Then, we construct it by calling the `Person` class's constructor and passing the required parameters. That's the same for destructors as well and we destruct the parent object lastly in the `Student` class's destructor.

Since we cannot use the behaviors of the `Person` class to read the inherited attributes, the `Student` class is required to offer its set of behavior functions to expose those inherited and private attributes.

In other words, the `Student` class has to provide some wrapper functions to expose the private attributes of its inner parent `person` object. Note that the `Student` object itself doesn't know anything about the private attributes of the `Person` object, and this is in contrast with what we saw in the first approach.

The main scenario is also very similar to how it was as part of *example 8.2*. The following code box demonstrates that:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "ExtremeC_examples_chapter8_3_student.h"
int main(int argc, char** argv) {
  // Create and construct the student object
  struct student_t* student = student_new();
  student_ctor(student, "John", "Doe",
          1987, "TA5667", 134);
  // We have to use student's behavior functions because the
  // student pointer is not a person pointer and we cannot
  // access to private parent pointer in the student object.
  char buffer[32];
  student_get_first_name(student, buffer);
  printf("First name: %s\n", buffer);
  student_get_last_name(student, buffer);
  printf("Last name: %s\n", buffer);
  printf("Birth year: %d\n", student_get_birth_year(student));
  student_get_student_number(student, buffer);
  printf("Student number: %s\n", buffer);
  printf("Passed credits: %d\n",
          student_get_passed_credits(student));
  // Destruct and free the student object
  student_dtor(student);
  free(student);
  return 0;
}
```

Code Box 8-14 [ExtremeC_examples_chapter8_3_main.c]: The main scenario of example 8.3

In comparison to the main function in *example 8.2*, we have not included the public interface of the `Person` class. We have also needed to use the `Student` class's behavior functions because the `student_t` and `person_t` pointers are not interchangeable anymore.

The following shell box demonstrates how to compile and run *example 8.3*. As you might have guessed, the outputs are identical:

```cpp
$ gcc -c ExtremeC_examples_chapter8_3_person.c -o person.o
$ gcc -c ExtremeC_examples_chapter8_3_student.c -o student.o
$ gcc -c ExtremeC_examples_chapter8_3_main.c -o main.o
$ gcc person.o student.o main.o -o ex8_3.out
$ ./ex8_3.out
First name: John
Last name: Doe
Birth year: 1987
Student number: TA5667
Passed credits: 134
$
```

Shell Box 8-3: Building and running example 8.3

In the following section, we're going to compare the aforementioned approaches to implement an inheritance relationship in C.

### Comparison of two approaches

Now that you've seen two different approaches that we can take to implement inheritance in C, we can compare them. The following bullet points outline the similarities and differences between the two approaches:

*   Both approaches intrinsically show composition relationships.
*   The first approach keeps a structure variable in the child's attribute structure and relies on having access to the private implementation of the parent class. However, the second approach keeps a structure pointer from the incomplete type of the parent's attribute structure, and hence, it doesn't rely on the private implementation of the parent class.
*   In the first approach, the parent and child types are strongly dependent. In the second approach, the classes are independent of each other, and everything inside the parent implementation is hidden from the child.
*   In the first approach, you can have only one parent. In other words, it is a way to implement *single inheritance* in C. However, in the second approach, you can have as many parents as you like, thereby demonstrating the concept of *multiple inheritance*.
*   In the first approach, the parent's structure variable must be the first field in the attribute structure of the child class, but in the second approach, the pointers to parent objects can be put anywhere in the structure.
*   In the first approach, there were no two separate parent and child objects. The parent object was included in the child object, and a pointer to the child object was actually a pointer to the parent object.
*   In the first approach, we could use the behavior functions of the parent class, but in the second approach, we needed to forward the parent's behavior functions through new behavior functions in the child class.

So far, we have only talked about inheritance itself and we haven't gone through its usages. One of the most important usages of inheritance is to have *polymorphism* in your object model. In the following section, we're going to talk about polymorphism and how it can be implemented in C.

# Polymorphism

Polymorphism is not really a relationship between two classes. It is mostly a technique for keeping the same code while having different behaviors. It allows us to extend code or add functionalities without having to recompile the whole code base.

In this section, we try to cover what polymorphism is and how we can have it in C. This also gives us a better view of how modern programming languages such as C++ implement polymorphism. We'll start by defining polymorphism.

## What is polymorphism?

Polymorphism simply means to have different behaviors by just using the same public interface (or set of behavior functions).

Suppose that we have two classes, `Cat` and `Duck`, and they each have a behavior function, `sound`, which makes them print their specific sound. Explaining polymorphism is not an easy task to do and we'll try to take a top-down approach in explaining it. First, we'll try to give you an idea of how polymorphic code looks and how it behaves, and then we'll dive into implementing it in C. Once you get the idea, it will be easier to move into the implementation. In the following code boxes, we first create some objects, and then we see how we would expect a polymorphic function to behave if polymorphism was in place. First, let's create three objects. We have already assumed that both the `Cat` and `Duck` classes are children of the `Animal` class:

```cpp
struct animal_t* animal = animal_malloc();
animal_ctor(animal);
struct cat_t* cat = cat_malloc();
cat_ctor(cat);
struct duck_t* duck = duck_malloc();
duck_ctor(duck);
```

Code Box 8-15: Creating three objects of types Animal, Cat, and Duck

*Without* polymorphism, we would have called the `sound` behavior function for each object as follows:

```cpp
// This is not a polymorphism
animal_sound(animal);
cat_sound(cat);
duck_sound(duck);
```

Code Box 8-16: Calling the sound behavior function on the created objects

And the output would be as follows:

```cpp
Animal: Beeeep
Cat: Meow
Duck: Quack
```

Shell Box 8-4: The output of the function calls

The preceding code box is not demonstrating polymorphism because it uses different functions, `cat_sound` and `duck_sound`, to call specific behaviors from the `Cat` and `Duck` objects. However, the following code box shows how we expect a polymorphic function to behave. The following code box contains a perfect example of polymorphism:

```cpp
// This is a polymorphism
animal_sound(animal);
animal_sound((struct animal_t*)cat);
animal_sound((struct animal_t*)duck);
```

Code Box 8-17: Calling the same sound behavior function on all three objects

Despite calling the same function three times, we expect to see different behaviors. It seems that passing different object pointers changes the actual behavior behind `animal_sound`. The following shell box would be the output of *Code Box 8-17* if `animal_sound` was polymorphic:

```cpp
Animal: Beeeep
Cat: Meow
Duck: Quake
```

Shell Box 8-5: The output of the function calls

As you see in *Code Box 8-17*, we have used the same function, `animal_sound`, but with different pointers, and as a result, different functions have been invoked behind the scenes.

**CAUTION**:

Please don't move forward if you're having trouble understanding the preceding code; if you are, please recap the previous section.

The preceding polymorphic code implies that there should be an inheritance relationship between the `Cat` and `Duck` classes with a third class, `Animal`, because we want to be able to cast the `duck_t` and `cat_t` pointers to an `animal_t` pointer. This also implies something else: we have to use the first approach of implementing inheritance in C in order to benefit from the polymorphism mechanism we introduced before.

You may recall that in the first approach to implementing inheritance, the child class had access to the private implementation of the parent class, and here a structure variable from the `animal_t` type should have been put as the first field in the definitions of the `duck_t` and `cat_t` attribute structures. The following code shows the relationship between these three classes:

```cpp
typedef struct {
  ...} animal_t;
typedef struct {
  animal_t animal;
  ...
} cat_t;
typedef struct {
  animal_t animal;
  ...
} duck_t;
```

Code Box 8-18: The definitions of the attribute structures of classes Animal, Cat, and Duck

With this setup, we can cast the `duck_t` and `cat_t` pointers to the `animal_t` pointers, and then we can use the same behavior functions for both child classes.

So far, we have shown how a polymorphic function is expected to behave and how an inheritance relationship should be defined between the classes. What we haven't shown is how this polymorphic behavior is fulfilled. In other words, we haven't talked about the actual mechanism behind the polymorphism.

Suppose that the behavior function `animal_sound` is defined as it can be seen in code box 8-19\. No matter the pointer you send inside as the argument, we will have always one behavior and the function calls won't be polymorphic without the underlying mechanism. The mechanism will be explained as part of *example 8.4* which you will see shortly:

```cpp
void animal_sound(animal_t* ptr) {
  printf("Animal: Beeeep");
}
// This could be a polymorphism, but it is NOT!
animal_sound(animal);
animal_sound((struct animal_t*)cat);
animal_sound((struct animal_t*)duck);
```

Code Box 8-19: The function animal_sound is not polymorphic yet!

As you see next, calling the behavior function `animal_sound` with various pointers won't change the logic of the behavior function; in other words, it is not polymorphic. We will make this function polymorphic as part of the next example, *example 8.4*:

```cpp
Animal: Beeeep
Animal: Beeeep
Animal: Beeeep
```

Shell Box 8-6: The output of the functional calls in Code Box 8-19

So, what is the underlying mechanism that enables polymorphic behavior functions? We answer that question in the upcoming sections, but before that we need to know why we want to have polymorphism in the first place.

## Why do we need polymorphism?

Before talking further about the way in which we're going to implement polymorphism in C, we should spend some time talking about the reasons behind the need for polymorphism. The main reason why polymorphism is needed is that we want to keep a piece of code "as is," even when using it with various subtypes of a base type. You are going to see some demonstration of this shortly in the examples.

We don't want to modify the current logic very often when we add new subtypes to the system, or when the behavior of one subtype is being changed. It's just not realistic to have zero changes when a new feature is added – there will always be some changes – but using polymorphism, we can significantly reduce the number of changes that are needed.

Another motivation for having polymorphism is due to the concept of *abstraction*. When we have abstract types (or classes), they usually have some vague or unimplemented behavior functions that need to be *overridden* in child classes and polymorphism is the key way to do this.

Since we want to use abstract types to write our logic, we need a way to call the proper implementation when dealing with pointers of very abstract types. This is another place where polymorphism comes in. No matter what the language is, we need a way to have polymorphic behaviors, otherwise the cost of maintaining a big project can grow quickly, for instance when we are going to add a new subtype to our code.

Now that we've established the importance of polymorphism, it's time to explain how we can have it in C.

## How to have polymorphic behavior in C

If we want to have polymorphism in C, we need to use the first approach we explored to implementing inheritance in C. To achieve polymorphic behavior, we can utilize *function pointers*. However, this time, these function pointers need to be kept as some fields in the attribute structure. Let's implement the animal sound example to illustrate this.

We have three classes, `Animal`, `Cat`, and `Duck`, and `Cat` and `Duck` are subtypes of `Animal`. Each class has one header and one source. The `Animal` class has an extra private header file that contains the actual definition of its attribute structure. This private header is required since we are taking the first approach to implement inheritance. The private header is going to be used by the `Cat` and `Duck` classes.

The following code box shows the public interface of the `Animal` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_4_ANIMAL_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_4_ANIMAL_H
// Forward declaration
struct animal_t;
// Memory allocator
struct animal_t* animal_new();
// Constructor
void animal_ctor(struct animal_t*);
// Destructor
void animal_dtor(struct animal_t*);
// Behavior functions
void animal_get_name(struct animal_t*, char*);
void animal_sound(struct animal_t*);
#endif
```

Code Box 8-20 [ExtremeC_examples_chapter8_4_animal.h]: The public interface of the Animal class

The `Animal` class has two behavior functions. The `animal_sound` function is supposed to be polymorphic and can be overridden by the child classes, while the other behavior function, `animal_get_name`, is not polymorphic, and children cannot override it.

The following is the private definition of the `animal_t` attribute structure:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_4_ANIMAL_P_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_4_ANIMAL_P_H
// The function pointer type needed to point to
// different morphs of animal_sound
typedef void (*sound_func_t)(void*);
// Forward declaration
typedef struct {
  char* name;
  // This member is a pointer to the function which
  // performs the actual sound behavior
 sound_func_t sound_func;
} animal_t;
#endif
```

Code Box 8-21 [ExtremeC_examples_chapter8_4_animal_p.h]: The private header of the Animal class

In polymorphism, every child class can provide its own version of the `animal_sound` function. In other words, every child class can override the function inherited from its parent class. Therefore, we need to have a different function for each child class that wants to override it. This means, if the child class has overridden the `animal_sound`, its own overridden function should be called.

That's why we are using function pointers here. Each instance of `animal_t` will have a function pointer dedicated to the behavior `animal_sound`, and that pointer is pointing to the actual definition of the polymorphic function inside the class.

For each polymorphic behavior function, we have a dedicated function pointer. Here, you will see how we use this function pointer to do the correct function call in each subclass. In other words, we show how the polymorphism actually works.

The following code box shows the definition of the `Animal` class:

```cpp
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ExtremeC_examples_chapter8_4_animal_p.h"
// Default definition of the animal_sound at the parent level
void __animal_sound(void* this_ptr) {
 animal_t* animal = (animal_t*)this_ptr;
 printf("%s: Beeeep\n", animal->name);
}
// Memory allocator
animal_t* animal_new() {
  return (animal_t*)malloc(sizeof(animal_t));
}
// Constructor
void animal_ctor(animal_t* animal) {
  animal->name = (char*)malloc(10 * sizeof(char));
  strcpy(animal->name, "Animal");
  // Set the function pointer to point to the default definition
 animal->sound_func = __animal_sound;
}
// Destructor
void animal_dtor(animal_t* animal) {
  free(animal->name);
}
// Behavior functions
void animal_get_name(animal_t* animal, char* buffer) {
  strcpy(buffer, animal->name);
}
void animal_sound(animal_t* animal) {
  // Call the function which is pointed by the function pointer.
 animal->sound_func(animal);
}
```

Code Box 8-22 [ExtremeC_examples_chapter8_4_animal.c]: The definition of the Animal class

The actual polymorphic behavior is happening in *Code Box 8-22*, inside the function `animal_sound`. The private function `__animal_sound` is supposed to be the default behavior of the `animal_sound` function when the subclasses decide not to override it. You will see in the next chapter that polymorphic behavior functions have a default definition which will get inherited and used if the subclass doesn't provide the overridden version.

Moving on, inside the constructor `animal_ctor`, we store the address of `__animal_sound` into the `sound_func` field of the `animal` object. Remember that `sound_func` is a function pointer. In this setup, every child object inherits this function pointer, which points to the default definition `__animal_sound`.

And the final step, inside the behavior function `animal_sound`, we just call the function that is being pointed to by the `sound_func` field. Again, `sound_func` is the function pointer field pointing to the actual definition of the sound behavior which in the preceding case is `__animal_sound`. Note that the `animal_sound` function behaves more like a relay to the actual behavior function.

Using this setup, if the `sound_func` field was pointing to another function, then that function would have been called if `animal_sound` was invoked. That's the trick we are going to use in the `Cat` and `Duck` classes to override the default definition of the `sound` behavior.

Now, it's time to show the `Cat` and `Duck` classes. The following code boxes will show the `Cat` class's public interface and private implementation. First, we show the `Cat` class's public interface:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_4_CAT_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_4_CAT_H
// Forward declaration
struct cat_t;
// Memory allocator
struct cat_t* cat_new();
// Constructor
void cat_ctor(struct cat_t*);
// Destructor
void cat_dtor(struct cat_t*);
// All behavior functions are inherited from the animal class.
#endif
```

Code Box 8-23 [ExtremeC_examples_chapter8_4_cat.h]: The public interface of the Cat class

As you will see shortly, it will inherit the `sound` behavior from its parent class, the `Animal` class.

The following code box shows the definition of the `Cat` class:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ExtremeC_examples_chapter8_4_animal.h"
#include "ExtremeC_examples_chapter8_4_animal_p.h"
typedef struct {
  animal_t animal;
} cat_t;
// Define a new behavior for the cat's sound
void __cat_sound(void* ptr) {
 animal_t* animal = (animal_t*)ptr;
 printf("%s: Meow\n", animal->name);
}
// Memory allocator
cat_t* cat_new() {
  return (cat_t*)malloc(sizeof(cat_t));
}
// Constructor
void cat_ctor(cat_t* cat) {
  animal_ctor((struct animal_t*)cat);
  strcpy(cat->animal.name, "Cat");
  // Point to the new behavior function. Overriding
  // is actually happening here.
 cat->animal.sound_func = __cat_sound;
}
// Destructor
void cat_dtor(cat_t* cat) {
  animal_dtor((struct animal_t*)cat);
}
```

Code Box 8-24 [ExtremeC_examples_chapter8_4_cat.c]: The private implementation of the Cat class

As you see in the previous code box, we have defined a new function for the cat's sound, `__cat_sound`. Then inside the constructor, we make the `sound_func` pointer point to this function.

Now, overriding is happening, and from now on, all `cat` objects will actually call `__cat_sound` instead of `__animal_sound`. The same technique is used for the `Duck` class.

The following code box shows the public interface of the `Duck` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_8_4_DUCK_H
#define EXTREME_C_EXAMPLES_CHAPTER_8_4_DUCK_H
// Forward declaration
struct duck_t;
// Memory allocator
struct duck_t* duck_new();
// Constructor
void duck_ctor(struct duck_t*);
// Destructor
void duck_dtor(struct duck_t*);
// All behavior functions are inherited from the animal class.
#endif
```

Code Box 8-25 [ExtremeC_examples_chapter8_4_duck.h]: The public interface of the Duck class

As you see, that's quite similar to the `Cat` class. Let's bring up the private definition of the `Duck` class:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ExtremeC_examples_chapter8_4_animal.h"
#include "ExtremeC_examples_chapter8_4_animal_p.h"
typedef struct {
  animal_t animal;
} duck_t;
// Define a new behavior for the duck's sound
void __duck_sound(void* ptr) {
 animal_t* animal = (animal_t*)ptr;
 printf("%s: Quacks\n", animal->name);
}
// Memory allocator
duck_t* duck_new() {
  return (duck_t*)malloc(sizeof(duck_t));
}
// Constructor
void duck_ctor(duck_t* duck) {
  animal_ctor((struct animal_t*)duck);
  strcpy(duck->animal.name, "Duck");
  // Point to the new behavior function. Overriding
  // is actually happening here.
 duck->animal.sound_func = __duck_sound;
}
// Destructor
void duck_dtor(duck_t* duck) {
  animal_dtor((struct animal_t*)duck);
}
```

Code Box 8-26 [ExtremeC_examples_chapter8_4_duck.c]: The private implementation of the Duck class

As you can see, the technique has been used to override the default definition of the `sound` behavior. A new private behavior function, `__duck_sound`, has been defined that does the duck-specific sound, and the `sound_func` pointer is updated to point to this function. This is basically the way that polymorphism is introduced to C++. We will talk more about this in the next chapter.

Finally, the following code box demonstrates the main scenario of *example 8.4*:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Only public interfaces
#include "ExtremeC_examples_chapter8_4_animal.h"
#include "ExtremeC_examples_chapter8_4_cat.h"
#include "ExtremeC_examples_chapter8_4_duck.h"
int main(int argc, char** argv) {
  struct animal_t* animal = animal_new();
  struct cat_t* cat = cat_new();
  struct duck_t* duck = duck_new();
  animal_ctor(animal);
  cat_ctor(cat);
  duck_ctor(duck);
 animal_sound(animal);
 animal_sound((struct animal_t*)cat);
 animal_sound((struct animal_t*)duck);
  animal_dtor(animal);
  cat_dtor(cat);
  duck_dtor(duck);
  free(duck);
  free(cat);
  free(animal);
  return 0;
}
```

Code Box 8-27 [ExtremeC_examples_chapter8_4_main.c]: The main scenario of example 8.4

As you see in the preceding code box, we are only using the public interfaces of the `Animal`, `Cat`, and `Duck` classes. So, the `main` function doesn't know anything about the internal implementation of the classes. Calling the `animal_sound` function with passing different pointers demonstrates how a polymorphic behavior should work. Let's look at the output of the example.

The following shell box shows how to compile and run *example 8.4*:

```cpp
$ gcc -c ExtremeC_examples_chapter8_4_animal.c -o animal.o
$ gcc -c ExtremeC_examples_chapter8_4_cat.c -o cat.o
$ gcc -c ExtremeC_examples_chapter8_4_duck.c -o duck.o
$ gcc -c ExtremeC_examples_chapter8_4_main.c -o main.o
$ gcc animal.o cat.o duck.o main.o -o ex8_4.out
$ ./ex8_4.out
Animal: Beeeep
Cat: Meow
Duck: Quake
$
```

Shell Box 8-7: The compilation, execution, and output of example 8.4

As you can see in *example 8.4*, in class-based programming languages the behavior functions which we want to be polymorphic need special care and should be treated differently. Otherwise, a simple behavior function without the underlying mechanism that we discussed as part of *example 8.4* cannot be polymorphic. That's why we have a special name for these behavior functions, and why we use specific keywords to denote a function to be polymorphic in a language such as C++. These functions are called *virtual* functions. Virtual functions are behavior functions that can be overridden by child classes. Virtual functions need to be tracked by the compiler, and proper pointers should be placed in the corresponding objects to point to the actual definitions when overridden. These pointers are used at runtime to execute the right version of the function.

In the next chapter, we'll see how C++ handles object-oriented relationships between classes. Also, we will find out how C++ implements polymorphism. We will also discuss *Abstraction* which is a direct result of polymorphism.

# Summary

In this chapter, we continued our exploration of topics in OOP, picking up from where we left off in the previous chapter. The following topics were discussed in this chapter:

*   We explained how inheritance works and looked at the two approaches that we can use to implement inheritance in C.
*   The first approach allows direct access to all the private attributes of the parent class, but the second approach has a more conservative approach, hiding the private attributes of the parent class.
*   We compared these approaches, and we saw that each of them can be suitable in some use cases.
*   Polymorphism was the next topic that we explored. To put it simply, it allows us to have different versions of the same behavior and invoke the correct behavior using the public API of an abstract supertype.
*   We saw how to write polymorphic code in C and saw how function pointers contribute to choosing the correct version of a particular behavior at runtime.

The next chapter will be our final chapter about object orientation. As part of it, we'll explore how C++ handles encapsulation, inheritance, and polymorphism. More than that, we will discuss the topic of abstraction and how it leads to a bizarre type of class which is called an *abstract class*. We cannot create objects from these classes!