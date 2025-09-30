# 第八章

# 继承和多态

本章是前两章的延续，在前两章中，我们介绍了如何在 C 中进行面向对象编程，并达到了组合和聚合的概念。本章主要继续讨论对象与其对应类之间的关系，并涵盖继承和多态。作为本章的一部分，我们总结了这个主题，并在下一章继续讨论*抽象*。

本章在很大程度上依赖于前两章中解释的理论，在前两章中，我们讨论了类之间可能存在的关系。我们解释了*组合*和*聚合*关系，现在我们将在本章中讨论*扩展*或*继承*关系，以及一些其他主题。

本章将解释以下主题：

+   如前所述，继承关系是我们首先讨论的主题。我们将介绍在 C 中实现继承关系的方法，并进行比较。

+   下一个重要主题是*多态性*。多态性允许我们在子类中拥有相同行为的不同版本，在那些类之间存在继承关系的情况下。我们将讨论在 C 中实现多态函数的方法；这将是我们理解 C++如何提供多态性的第一步。

让我们从继承关系开始讨论。

# 继承

我们在前一章结束时讨论了*拥有*关系，这最终引导我们到了组合和聚合关系。在本节中，我们将讨论*是*或*属于*关系。继承关系是一种是关系。

继承关系也可以称为*扩展关系*，因为它只向现有的对象或类添加额外的属性和行为。在接下来的几节中，我们将解释继承的含义以及如何在 C 中实现它。

有时一个对象需要拥有存在于另一个对象中的相同属性。换句话说，新对象是另一个对象的扩展。

例如，一个学生具有人的所有属性，但也可能有额外的属性。参见*代码框 8-1*：

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

代码框 8-1：Person 类和 Student 类的属性结构

这个例子清楚地展示了`student_t`如何通过新的属性`student_number`和`passed_credits`扩展了`person_t`的属性，这些属性是特定于学生的。

正如我们之前指出的，继承（或扩展）是一种将要成为的关系，与组合和聚合不同，它们是拥有关系。因此，对于前面的例子，我们可以说“一个学生是一个人”，这在教育软件的领域中似乎是正确的。每当一个将要成为的关系存在于一个领域中，它可能就是一个继承关系。在前面的例子中，`person_t` 通常被称为 *超类型*，或 *基类型*，或简单地称为 *父类型*，而 `student_t` 通常被称为 *子类型* 或 *继承子类型*。

## 继承的本质

如果你深入挖掘并了解继承关系真正是什么，你会发现它本质上实际上是一种组合关系。例如，我们可以说一个学生体内有人的本质。换句话说，我们可以假设在 `Student` 类的属性结构内部有一个私有的 `person` 对象。也就是说，继承关系可以等同于一对一的组合关系。

因此，*代码框 8-1* 中的结构可以写成如下：

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

代码框 8-2：Person 和 Student 类的属性结构，但这次是嵌套的

这种语法在 C 语言中是完全有效的，实际上通过使用结构变量（而非指针）嵌套结构是一种强大的设置。它允许你在新的结构中拥有一个结构变量，这实际上是对之前结构的扩展。

在上述设置中，必然有一个 `person_t` 类型的字段作为第一个字段，一个 `student_t` 指针可以轻松地转换为 `person_t` 指针，并且它们都可以指向内存中的相同地址。

这被称为 *向上转型*。换句话说，将子属性结构的指针类型转换为父属性结构类型的类型是向上转型。请注意，使用结构变量时，您无法拥有此功能。

*示例 8.1* 如下演示了这一点：

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

代码框 8-3 [ExtremeC_examples_chapter8_1.c]：示例 8.1，展示了 Student 和 Person 对象指针之间的向上转型

如您所见，我们预计 `s_ptr` 和 `p_ptr` 指针都指向内存中的相同地址。以下是在构建和运行 *示例 8.1* 后的输出：

```cpp
$ gcc ExtremeC_examples_chapter8_1.c -o ex8_1.out
$ ./ex8_1.out
Student pointer points to 0x7ffeecd41810
Person pointer points to 0x7ffeecd41810
$
```

Shell 框 8-1：示例 8.1 的输出

是的，它们指向的是同一个地址。请注意，显示的地址在每次运行中可能不同，但重点是这些指针正在引用相同的地址。这意味着 `student_t` 类型的结构变量实际上在内存布局中继承了 `person_t` 结构。这暗示我们可以使用指向 `student` 对象的指针使用 `Person` 类的函数行为。换句话说，`Person` 类的行为函数可以用于 `student` 对象，这是一个巨大的成就。

注意以下内容是错误的，代码无法编译：

```cpp
struct person_t;
typedef struct {
 struct person_t person; // Generates an error! 
  char student_number[16]; // Extra attribute
  unsigned int passed_credits; // Extra attribute
} student_t;
```

代码框 8-4：无法编译的继承关系建立！

声明`person`字段的行会生成错误，因为你不能从一个*不完整类型*创建变量。你应该记住，结构的向前声明（类似于*代码框 8-4*中的第一行）会导致不完整类型的声明。你可以只有不完整类型的指针，*不能*有变量。正如你之前看到的，你甚至不能为不完整类型分配堆内存。

那么，这意味着什么呢？这意味着，如果你打算使用嵌套结构变量来实现继承，`student_t`结构应该看到`person_t`的实际定义，根据我们关于封装所学的知识，它应该是私有的，并且对任何其他类不可见。

因此，你有两种实现继承关系的方法：

+   让子类能够访问基类的私有实现（实际定义）。

+   让子类只能访问基类的公共接口。

### C 语言中实现继承的第一种方法

我们将在以下示例中演示第一种方法，即*示例 8.2*，第二种方法将在下一节中的*示例 8.3*中展示。它们都代表了相同的类，`Student`和`Person`，具有一些行为函数，在`main`函数中的一些对象在一个简单场景中发挥作用。

我们将从*示例 8.2*开始，其中`Student`类需要访问`Person`类属性结构的实际私有定义。以下代码框展示了`Student`和`Person`类的头部文件和源代码，以及`main`函数。让我们从声明`Person`类的头部文件开始：

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

代码框 8-5 [ExtremeC_examples_chapter8_2_person.h]: 示例 8.2，`Person`类的公共接口

看一下*代码框 8-5*中的构造函数。它接受创建`person`对象所需的所有值：`first_name`、`second_name`和`birth_year`。正如你所看到的，属性结构`person_t`是不完整的，因此`Student`类不能使用前面章节中展示的头部文件来建立继承关系。

另一方面，前面的头部文件不应该包含属性结构`person_t`的实际定义，因为前面的头部文件将被代码的其他部分使用，这些部分不应该了解`Person`的内部情况。那么我们应该怎么做呢？我们希望逻辑的一部分了解结构定义，而代码的其他部分不应该了解这个定义。这就是*私有头部文件*介入的地方。

私有头文件是一个普通头文件，它应该被包含并用于代码的某个部分或某个实际需要它的类。关于 *示例 8.2*，`person_t` 的实际定义应该是私有头文件的一部分。在下面的代码框中，您将看到一个私有头文件的示例：

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

代码框 8-6 [ExtremeC_ 示例 _ 第八章 _2_person_p.h]: 包含 `person_t` 实际定义的私有头文件

正如您所见，它只包含 `person_t` 结构的定义，没有其他内容。这是 `Person` 类应该保持私有的部分，但它需要成为 `Student` 类的公共部分。我们将需要这个定义来定义 `student_t` 属性结构。下一个代码框演示了 `Person` 类的私有实现：

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

代码框 8-7 [ExtremeC_ 示例 _ 第八章 _2_person.c]: Person 类的定义

`Person` 类的定义没有特别之处，它和所有之前的示例类似。下面的代码框显示了 `Student` 类的公共接口：

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

代码框 8-8 [ExtremeC_ 示例 _ 第八章 _2_student.h]: Student 类的公共接口

如您所见，类的构造函数接受与 `Person` 类构造函数相似的参数。这是因为 `student` 对象实际上包含一个 `person` 对象，并且它需要这些值来填充其组成的 `person` 对象。

这意味着 `student` 构造函数需要设置 `student` 的 `person` 部分的属性。

注意，我们作为 `Student` 类的一部分只添加了两个额外的行为函数，这是因为我们可以使用 `Person` 类的行为函数来处理 `student` 对象。

下一个代码框包含了 `Student` 类的私有实现：

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

代码框 8-9 [ExtremeC_ 示例 _ 第八章 _2_student.c]: Student 类的私有定义

前面的代码框包含了关于继承关系最重要的代码。首先，我们需要包含 `Person` 类的私有头文件，因为作为定义 `student_t` 的一部分，我们希望有 `person_t` 类型的第一个字段。而且，由于该字段是一个实际变量而不是指针，这就要求我们已经有 `person_t` 定义。请注意，这个变量必须是结构的 *第一个字段*。否则，我们将失去使用 `Person` 类的行为函数的可能性。

再次，在前面的代码框中，作为 `Student` 类构造函数的一部分，我们调用父构造函数来初始化父（组合）对象的属性。看看我们如何将 `student_t` 指针转换为 `person_t` 指针，当传递给 `person_ctor` 函数时。这仅仅是因为 `person` 字段是 `student_t` 的第一个成员。

同样，作为 `Student` 类析构函数的一部分，我们调用了父类的析构函数。这种销毁应该首先在子级发生，然后在父级发生，与构建的顺序相反。下一个代码框包含 *示例 8.2* 的主场景，它将使用 `Student` 类并创建一个 `Student` 类型的对象：

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

代码框 8-10 [ExtremeC_examples_chapter8_2_main.c]: 示例 8.2 的主场景

正如你在主场景中看到的那样，我们包含了 `Person` 和 `Student` 类的公共接口（不是私有头文件），但我们只创建了一个 `student` 对象。正如你所看到的，`student` 对象从其内部的 `person` 对象继承了所有属性，并且可以通过 `Person` 类的行为函数来读取。

下面的 Shell 框展示了如何编译和运行 *示例 8.2*：

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

Shell 框 8-2：构建和运行示例 8.2

下面的示例，*示例 8.3*，将介绍在 C 语言中实现继承关系的第二种方法。输出应该与 *示例 8.2* 非常相似。

### C 语言中的继承的第二种方法

使用第一种方法，我们将结构变量作为子属性结构中的第一个字段。现在，使用第二种方法，我们将保留对父结构变量的指针。这样，子类就可以独立于父类的实现，这在考虑信息隐藏问题时是个好事。

通过选择第二种方法，我们获得了一些优势，也失去了一些。在演示 *示例 8.3* 之后，我们将对两种方法进行比较，你将看到使用这些技术各自的优缺点。

下面的 *示例 8.3* 与 *示例 8.2* 非常相似，尤其是在输出和最终结果方面。然而，主要区别在于，在这个示例中，`Student` 类只依赖于 `Person` 类的公共接口，而不是其私有定义。这很好，因为它解耦了类，使我们能够轻松地更改父类的实现，而不会更改子类的实现。

在前面的示例中，`Student` 类并没有严格违反信息隐藏原则，但它可以这样做，因为它可以访问 `person_t` 的实际定义及其字段。因此，它可以读取或修改字段，而无需使用 `Person` 的行为函数。

正如所述，*示例 8.3* 与 *示例 8.2* 非常相似，但有一些基本的不同之处。`Person` 类在新示例中具有相同的公共接口。但这一点并不适用于 `Student` 类，其公共接口需要更改。下面的代码框展示了 `Student` 类的新公共接口：

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

代码框 8-11 [ExtremeC_examples_chapter8_3_student.h]: 学生类的新公共接口

由于你很快就会意识到的原因，`Student` 类必须重复所有作为 `Person` 类一部分声明的行为函数。这是因为我们不能再将 `student_t` 指针转换为 `person_t` 指针。换句话说，关于 `Student` 和 `Person` 指针，向上转换不再起作用。

虽然 `Person` 类的公共接口与 *示例 8.2* 中没有变化，但其实现已经改变。以下代码框展示了 *示例 8.3* 中 `Person` 类的实现：

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

代码框 8-12 [ExtremeC_examples_chapter8_3_person.c]：`Person` 类的新实现

如你所见，`person_t` 的私有定义放置在源文件中，我们不再使用私有头文件。这意味着我们根本不会与其他类，如 `Student` 类共享定义。我们希望对 `Person` 类进行完全封装，并隐藏其所有实现细节。

下面的内容是 `Student` 类的私有实现：

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

代码框 8-13 [ExtremeC_examples_chapter8_3_student.c]：`Student` 类的新实现

如前一个代码框所示，我们通过包含其头文件使用了 `Person` 类的公共接口。此外，作为 `student_t` 定义的一部分，我们添加了一个指针字段，该字段指向父 `Person` 对象。这应该会让你想起上一章中作为组合关系实现的部分。

注意，这个指针字段不需要作为属性结构中的第一个项目。这与我们在第一种方法中看到的情况不同。`student_t` 和 `person_t` 类型的指针不再可以互换，它们指向内存中的不同地址，这些地址不一定相邻。这又与我们在之前的方法中做的不一样。

注意，作为 `Student` 类构造函数的一部分，我们实例化了父对象。然后，我们通过调用 `Person` 类的构造函数并传递所需的参数来构建它。这与析构函数相同，我们最后在 `Student` 类的析构函数中销毁父对象。

由于我们无法使用 `Person` 类的行为来读取继承的属性，`Student` 类需要提供其行为函数集来暴露那些继承的私有属性。

换句话说，`Student` 类必须提供一些包装函数来暴露其内部父 `person` 对象的私有属性。请注意，`Student` 对象本身对 `Person` 对象的私有属性一无所知，这与我们在第一种方法中看到的情况形成对比。

主要场景也与 *示例 8.2* 中的情况非常相似。以下代码框展示了这一点：

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

代码框 8-14 [ExtremeC_examples_chapter8_3_main.c]：示例 8.3 的主要场景

与*示例 8.2*中的主函数相比，我们没有包含`Person`类的公共接口。我们还必须使用`Student`类的行为函数，因为`student_t`和`person_t`指针不再可以互换。

以下 shell 框演示了如何编译和运行*示例 8.3*。正如你可能猜到的，输出是相同的：

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

Shell Box 8-3：构建和运行示例 8.3

在下一节中，我们将比较上述方法来实现 C 语言中的继承关系。

### 两种方法的比较

现在你已经看到了我们可以采取的两种不同的方法来实现 C 语言中的继承，我们可以比较它们。以下要点概述了两种方法之间的相似之处和不同之处：

+   两种方法本质上都显示了组合关系。

+   第一种方法在子类的属性结构中保留一个结构变量，并依赖于对父类私有实现的访问。然而，第二种方法保留指向父类属性结构不完整类型的结构指针，因此它不依赖于父类的私有实现。

+   在第一种方法中，父类和子类之间有很强的依赖性。在第二种方法中，类之间相互独立，父类实现中的所有内容对子类都是隐藏的。

+   在第一种方法中，你只能有一个父类。换句话说，这是一种在 C 语言中实现*单继承*的方法。然而，在第二种方法中，你可以有任意多的父类，从而演示了*多继承*的概念。

+   在第一种方法中，父类的结构变量必须是子类属性结构中的第一个字段，但在第二种方法中，指向父对象指针可以放在结构中的任何位置。

+   在第一种方法中，没有两个独立的父类和子类对象。父类对象包含在子类对象中，指向子类对象的指针实际上是指向父类对象的指针。

+   在第一种方法中，我们可以使用父类的行为函数，但在第二种方法中，我们需要通过子类中的新行为函数来转发父类的行为函数。

到目前为止，我们只讨论了继承本身，还没有讨论其用法。继承最重要的用法之一是在你的对象模型中实现*多态性*。在下一节中，我们将讨论多态性以及如何在 C 语言中实现它。

# 多态性

多态性实际上并不是两个类之间的关系。它主要是一种在保持相同代码的同时实现不同行为的技术。它允许我们在不重新编译整个代码库的情况下扩展代码或添加功能。

在本节中，我们试图解释多态是什么以及我们如何在 C 语言中实现它。这也让我们更好地了解现代编程语言（如 C++）是如何实现多态的。我们将从定义多态开始。

## 多态是什么？

多态简单地说就是通过使用相同的公共接口（或行为函数集）来拥有不同的行为。

假设我们有两个类，`Cat`和`Duck`，它们各自有一个行为函数`sound`，这使得它们打印出它们特定的声音。解释多态不是一个容易的任务，我们将尝试自顶向下的方法来解释它。首先，我们将尝试给你一个多态代码看起来如何以及它如何表现的概念，然后我们将深入到在 C 语言中实现它。一旦你有了这个概念，进入实现就会更容易。在以下代码框中，我们首先创建一些对象，然后看看如果多态存在，我们期望多态函数会如何表现。首先，让我们创建三个对象。我们已经假设`Cat`和`Duck`类都是`Animal`类的子类：

```cpp
struct animal_t* animal = animal_malloc();
animal_ctor(animal);
struct cat_t* cat = cat_malloc();
cat_ctor(cat);
struct duck_t* duck = duck_malloc();
duck_ctor(duck);
```

代码框 8-15：创建 Animal、Cat 和 Duck 三种类型的三个对象

**没有**多态的情况下，我们会像下面这样为每个对象调用`sound`行为函数：

```cpp
// This is not a polymorphism
animal_sound(animal);
cat_sound(cat);
duck_sound(duck);
```

代码框 8-16：在创建的对象上调用发声行为函数

输出结果如下：

```cpp
Animal: Beeeep
Cat: Meow
Duck: Quack
```

Shell 框 8-4：函数调用的输出

前面的代码框没有展示多态，因为它使用了不同的函数`cat_sound`和`duck_sound`来从`Cat`和`Duck`对象中调用特定的行为。然而，下面的代码框展示了我们期望多态函数如何表现。下面的代码框包含了一个完美的多态示例：

```cpp
// This is a polymorphism
animal_sound(animal);
animal_sound((struct animal_t*)cat);
animal_sound((struct animal_t*)duck);
```

代码框 8-17：在所有三个对象上调用相同的发声行为函数

尽管调用了三次相同的函数，但我们期望看到不同的行为。看起来传递不同的对象指针会改变`animal_sound`背后的实际行为。以下 Shell 框将显示如果`animal_sound`是多态的，则*代码框 8-17*的输出：

```cpp
Animal: Beeeep
Cat: Meow
Duck: Quake
```

Shell 框 8-5：函数调用的输出

正如你在*代码框 8-17*中看到的，我们使用了相同的函数`animal_sound`，但是使用了不同的指针，结果在幕后调用了不同的函数。

**注意**：

如果你在理解前面的代码时遇到困难，请不要继续前进；如果你遇到了，请回顾前面的章节。

之前的多态代码意味着`Cat`类和`Duck`类之间应该存在一个继承关系，并且有一个第三类`Animal`，因为我们希望能够将`duck_t`和`cat_t`指针转换为`animal_t`指针。这也意味着另一件事：我们必须使用 C 语言中实现继承的第一种方法，以便从我们之前引入的多态机制中受益。

你可能还记得，在实现继承的第一种方法中，子类可以访问父类的私有实现，在这里，`animal_t`类型的结构变量应该被放在`duck_t`和`cat_t`属性结构定义中的第一个字段。以下代码显示了这三个类之间的关系：

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

代码框 8-18：Animal、Cat 和 Duck 类属性结构的定义

在这种设置下，我们可以将`duck_t`和`cat_t`指针转换为`animal_t`指针，然后我们可以为这两个子类使用相同的行为函数。

到目前为止，我们已经展示了多态函数应该如何表现以及如何在类之间定义继承关系。我们没有展示的是如何实现这种多态行为。换句话说，我们还没有讨论多态背后的实际机制。

假设行为函数`animal_sound`的定义如代码框 8-19 所示。无论你发送什么指针作为参数，我们都会有相同的行为，并且没有底层机制，函数调用不会是多态的。这个机制将在*示例 8.4*中作为一部分进行解释，你很快就会看到：

```cpp
void animal_sound(animal_t* ptr) {
  printf("Animal: Beeeep");
}
// This could be a polymorphism, but it is NOT!
animal_sound(animal);
animal_sound((struct animal_t*)cat);
animal_sound((struct animal_t*)duck);
```

代码框 8-19：`animal_sound`函数还不是多态的！

正如你接下来看到的，使用各种指针调用行为函数`animal_sound`不会改变行为函数的逻辑；换句话说，它不是多态的。我们将在下一个示例，*示例 8.4*中使这个函数成为多态。

```cpp
Animal: Beeeep
Animal: Beeeep
Animal: Beeeep
```

Shell 框 8-6：代码框 8-19 中功能调用的输出

那么，使多态行为函数得以实现的底层机制是什么？我们将在接下来的章节中回答这个问题，但在那之前，我们需要知道为什么我们首先想要多态。

## 我们为什么需要多态？

在进一步讨论我们在 C 语言中实现多态的方法之前，我们应该花一些时间来谈谈多态需求背后的原因。多态之所以需要，主要原因是我们要保持一段代码“原样”，即使在使用它时与基类型的各种子类型一起使用。你将在接下来的示例中看到一些关于此的演示。

当我们向系统中添加新的子类型或改变一个子类型的行为时，我们不想经常修改当前的逻辑。当添加新功能时，完全没有变化是不现实的——总会有一些变化——但使用多态，我们可以显著减少所需更改的数量。

多态存在的另一个动机是由于*抽象*的概念。当我们有抽象类型（或类）时，它们通常有一些模糊或未实现的行为函数，这些函数需要在子类中*重写*，而多态是实现这一点的关键方式。

由于我们想使用抽象类型来编写我们的逻辑，我们需要一种在处理非常抽象类型的指针时调用适当实现的方法。这又是多态性发挥作用的地方。无论什么语言，我们都需要一种实现多态行为的方法，否则维护大型项目的成本可能会迅速增加，例如当我们准备向代码中添加新的子类型时。

既然我们已经确立了多态性的重要性，现在是时候解释如何在 C 语言中实现它了。

## 如何在 C 语言中实现多态行为

如果我们想在 C 语言中实现多态性，我们需要使用我们之前探索的 C 语言实现继承的第一种方法。为了实现多态行为，我们可以利用函数指针。然而，这次，这些函数指针需要作为属性结构中的某些字段来保留。让我们通过实现动物声音示例来说明这一点。

我们有三个类，`Animal`、`Cat`和`Duck`，其中`Cat`和`Duck`是`Animal`的子类型。每个类都有一个头文件和一个源文件。`Animal`类有一个额外的私有头文件，其中包含其实际的属性结构定义。由于我们正在采用第一种方法实现继承，这个私有头文件是必需的。私有头文件将被`Cat`和`Duck`类使用。

以下代码框展示了`Animal`类的公共接口：

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

代码框 8-20 [ExtremeC_examples_chapter8_4_animal.h]：`Animal`类的公共接口

`Animal`类有两个行为函数。`animal_sound`函数应该是多态的，可以被子类覆盖，而另一个行为函数`animal_get_name`不是多态的，子类不能覆盖它。

以下是对`animal_t`属性结构的私有定义：

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

代码框 8-21 [ExtremeC_examples_chapter8_4_animal_p.h]：`Animal`类的私有头文件

在多态性中，每个子类都可以提供自己的`animal_sound`函数版本。换句话说，每个子类都可以覆盖从其父类继承的函数。因此，我们需要为每个想要覆盖它的子类提供一个不同的函数。这意味着，如果子类覆盖了`animal_sound`，则应该调用其覆盖的函数。

正是因为这个原因，我们在这里使用函数指针。每个`animal_t`实例都将有一个专用于行为`animal_sound`的函数指针，而这个指针指向类内部的多态函数的实际定义。

对于每个多态行为函数，我们都有一个专用的函数指针。在这里，您将看到我们如何使用这个函数指针在各个子类中进行正确的函数调用。换句话说，我们展示了多态性实际上是如何工作的。

以下代码框展示了`Animal`类的定义：

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

代码框 8-22 [ExtremeC_examples_chapter8_4_animal.c]：`Animal`类的定义

实际的多态行为发生在*代码框 8-22*中，在`animal_sound`函数内部。私有函数`__animal_sound`是当子类决定不覆盖它时`animal_sound`函数的默认行为。你将在下一章中看到，多态行为函数有一个默认定义，如果子类没有提供覆盖版本，它将被继承并使用。

接下来，在构造函数`animal_ctor`中，我们将`__animal_sound`的地址存储到`animal`对象的`sound_func`字段中。记住，`sound_func`是一个函数指针。在这个设置中，每个子对象都继承了这个函数指针，它指向默认定义`__animal_sound`。

最后一步，在行为函数`animal_sound`内部，我们只是调用由`sound_func`字段指向的函数。再次强调，`sound_func`是函数指针字段，指向实际的声音行为定义，在前面的例子中是`__animal_sound`。请注意，`animal_sound`函数更像是一个将实际行为函数作为中继的行为。

使用这种设置，如果`sound_func`字段指向另一个函数，那么在调用`animal_sound`时，就会调用那个函数。这就是我们在`Cat`和`Duck`类中用来覆盖默认`sound`行为定义的技巧。

现在，是时候展示`Cat`和`Duck`类了。以下代码框将展示`Cat`类的公共接口和私有实现。首先，我们展示`Cat`类的公共接口：

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

代码框 8-23 [ExtremeC_examples_chapter8_4_cat.h]: `Cat`类的公共接口

如你很快就会看到的，它将从其父类`Animal`类继承`sound`行为。

以下代码框展示了`Cat`类的定义：

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

代码框 8-24 [ExtremeC_examples_chapter8_4_cat.c]: `Cat`类的私有实现

如你在前面的代码框中所见，我们为猫的声音定义了一个新函数`__cat_sound`。然后在构造函数中，我们将`sound_func`指针指向这个函数。

现在，覆盖正在发生，从现在起，所有`cat`对象实际上都会调用`__cat_sound`而不是`__animal_sound`。同样的技术也用于`Duck`类。

以下代码框展示了`Duck`类的公共接口：

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

代码框 8-25 [ExtremeC_examples_chapter8_4_duck.h]: `Duck`类的公共接口

正如你所见，这与`Cat`类非常相似。让我们来看看`Duck`类的私有定义：

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

代码框 8-26 [ExtremeC_examples_chapter8_4_duck.c]: `Duck`类的私有实现

正如你所看到的，该技术已被用来覆盖 `sound` 行为的默认定义。定义了一个新的私有行为函数 `__duck_sound`，它执行鸭特有的声音，并且 `sound_func` 指针被更新以指向这个函数。这基本上是将多态引入 C++ 的方式。我们将在下一章中更多地讨论这一点。

最后，以下代码框演示了*示例 8.4*的主要场景：

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

代码框 8-27 [ExtremeC_examples_chapter8_4_main.c]：示例 8.4 的主要场景

正如你在前面的代码框中看到的，我们只使用了 `Animal`、`Cat` 和 `Duck` 类的公共接口。因此，`main` 函数对类的内部实现一无所知。通过传递不同的指针调用 `animal_sound` 函数，展示了多态行为应该如何工作。让我们看看示例的输出。

以下 shell 框展示了如何编译和运行*示例 8.4*：

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

Shell 框 8-7：示例 8.4 的编译、执行和输出

正如你在*示例 8.4*中看到的，在基于类的编程语言中，我们想要实现多态的行为函数需要特别注意，并且应该以不同的方式处理。否则，没有我们作为*示例 8.4*一部分讨论的底层机制的简单行为函数不能实现多态。这就是为什么我们为这些行为函数取了特殊名称，以及为什么我们在像 C++ 这样的语言中使用特定的关键字来表示函数是多态的。这些函数被称为*虚函数*。虚函数是可以被子类覆盖的行为函数。虚函数需要被编译器跟踪，并且应该在相应的对象中放置适当的指针，以便在覆盖时指向实际的定义。这些指针在运行时用于执行函数的正确版本。

在下一章中，我们将看到 C++ 如何处理类之间的面向对象关系。我们还将了解 C++ 如何实现多态。我们还将讨论*抽象*，这是多态的直接结果。

# 摘要

在本章中，我们继续探索面向对象编程（OOP）中的主题，从上一章结束的地方继续。本章讨论了以下主题：

+   我们解释了继承是如何工作的，并查看了我们可以在 C 中实现继承的两种方法。

+   第一种方法允许直接访问父类的所有私有属性，但第二种方法采取了一种更为保守的方法，隐藏了父类的私有属性。

+   我们比较了这些方法，并看到它们中的每一个在某些用例中可能是合适的。

+   多态是我们接下来探索的主题。简单来说，它允许我们拥有同一行为的不同版本，并使用抽象超类型的公共 API 调用正确的行为。

+   我们看到了如何在 C 语言中编写多态代码，并了解了函数指针如何有助于在运行时选择特定行为的正确版本。

下一章将是关于面向对象编程的最后一章。作为其中的一部分，我们将探讨 C++如何处理封装、继承和多态。不仅如此，我们还将讨论抽象这个主题，以及它如何导致一种被称为*抽象类*的奇特类型的类。我们不能从这些类中创建对象！
