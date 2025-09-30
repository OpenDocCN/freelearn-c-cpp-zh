# Chapter 07

# Composition and Aggregation

In the previous chapter, we talked about encapsulation and information hiding. In this chapter, we continue with object orientation in C and we'll discuss the various relationships that can exist between two classes. Eventually, this will allow us to expand our object model and express the relations between objects as part of the upcoming chapters.

As part of this chapter, we discuss:

*   Types of relations that can exist between two objects and their corresponding classes: We will talk about *to-have* and *to-be* relationships, but our focus will be on to-have relations in this chapter.
*   *Composition* as our first to-have relation: An example will be given to demonstrate a real composition relationship between two classes. Using the given example, we explore the memory structure which we usually have in case of composition.
*   *Aggregation* as the second to-have relation: It is similar to composition since both of them address a to-have relationship. But they are different. We will give a separate complete example to cover an aggregation case. The difference among aggregation and composition will shine over the memory layout associated with these relationships.

This is the second of the four chapters covering OOP in C. The to-be relationship, which is also called *inheritance*, will be covered in the next chapter.

# Relations between classes

An object model is a set of related objects. The number of relations can be many, but there are a few relationship types that can exist between two objects. Generally, there are two categories of relationships found between objects (or their corresponding classes): to-have relationships and to-be relationships.

We'll explore to-have relationships in depth in this chapter, and we'll cover to-be relationships in the next chapter. In addition, we will also see how the relationships between various objects can lead to relationships between their corresponding classes. Before dealing with that, we need to be able to distinguish between a class and an object.

# Object versus class

If you remember from the previous chapter, we have two approaches for constructing objects. One approach is *prototype-based* and the other is *class-based*.

In the prototype-based approach, we construct an object either empty (without any attribute or behavior), or we clone it from an existing object. In this context, *instance* and *object* mean the same thing. So, the prototype-based approach can be read as the object-based approach; an approach that begins from empty objects instead of classes.

In the class-based approach, we cannot construct an object without having a blueprint that is often called a *class*. So, we should start from a class. And then, we can instantiate an object from it. In the previous chapter, we explained the implicit encapsulation technique that defines a class as a set of declarations put in a header file. We also gave some examples showing how this works in C.

Now, as part of this section, we want to talk more about the differences between a class and an object. While the differences seem to be trivial, we want to dive deeper and study them carefully. We begin by giving an example.

Suppose that we define a class, `Person`. It has the following attributes: `name`, `surname`, and `age`. We won't talk about the behaviors because the differences usually come from the attributes, and not the behaviors.

In C, we can write the `Person` class with public attributes as follows:

```cpp
typedef struct {
  char name[32];
  char surname[32];
  unsigned int age;
} person_t;
```

Code Box 7-1: The Person attribute structure in C

And in C++:

```cpp
class Person {
public:
  std::string name;
  std::string family;
  uint32_t age;
};
```

Code Box 7-2: The Person class's class in C++

The preceding code boxes are identical. In fact, the current discussion can be applied to both C and C++, and even other OOP languages such as Java. A class (or an object template) is a blueprint that only determines the attributes required to be present in every object, and *not* the values that these attributes might have in one specific object. In fact, each object has its own specific set of values for the same attributes that exist in other objects instantiated from the same class.

When an object is created based on a class, its memory is allocated first. This allocated memory will be a placeholder for the attribute values. After that, we need to initialize the attribute values with some values. This is an important step, otherwise, the object might have an invalid state after being created. As you've already seen, this step is called *construction*.

There is usually a dedicated function that performs the construction step, which is called the *constructor*. The functions `list_init` and `car_construct` in the examples, found in the previous chapter, were constructor functions. It is quite possible that as part of constructing an object, we need to allocate even more memory for resources such as other objects, buffers, arrays, streams, and so on required by that object. The resources owned by the object must have been released before having the owner object freed.

We also have another function, similar to the constructor, which is responsible for freeing any allocated resources. It is called the *destructor*. Similarly, the functions `list_destroy` and `car_destruct` in the examples found in the previous chapter were destructors. After destructing an object, its allocated memory is freed, but before that, all the owned resources and their corresponding memories must be freed.

Before moving on, let's sum up what we've explained so far:

*   A class is a blueprint that is used as a map for creating objects.
*   Many objects can be made from the same class.
*   A class determines which attributes should be present in every future object created based on that class. It doesn't say anything about the possible values they can have.
*   A class itself does not consume any memory (except in some programming languages other than C and C++) and only exists at the source level and at compile time. But objects exist at runtime and consume memory.
*   When creating an object, memory allocation happens first. In addition, memory deallocation is the last operation for an object.
*   When creating an object, it should be constructed right after memory allocation. It should be also destructed right before deallocation.
*   An object might be owning some resources such as streams, buffers, arrays, and so on, that must be released before having the object destroyed.

Now that you know the differences between a class and an object, we can move on and explain the different relationships that can exist between two objects and their corresponding classes. We'll start with composition.

# Composition

As the term "composition" implies, when an object contains or possesses another object – in other words, it is composed of another object – we say that there is a composition relationship between them.

As an example, a car has an engine; a car is an object that contains an engine object. Therefore, the car and engine objects have a composition relationship. There is an important condition that a composition relationship must have: *the lifetime of the contained object is bound to the lifetime of the container object*.

As long as the container object exists, the contained object must exist. But when the container object is about to get destroyed, the contained object must have been destructed first. This condition implies that the contained object is often internal and private to the container.

Some parts of the contained object may be still accessible through the public interface (or behavior functions) of the container class, but the lifetime of the contained object must be managed internally by the container object. If a piece of code can destruct the contained object without destructing the container object, it is a breach of the composition relationship and the relationship is no longer a composition.

The following example, *example 7.1*, demonstrates the composition relationship between a car object and an engine object.

It is composed of five files: two header files, which declare the public interfaces of the `Car` and `Engine` classes; two source files, which contain the implementation of the `Car` and `Engine` classes; and finally, a source file, which contains the `main` function and executes a simple scenario using a car and its engine object.

Note that, in some domains, we can have engine objects outside of the car objects; for example, in mechanical engineering CAD software. So, the type of relationships between the various objects is determined by the problem domain. For the sake of our example, imagine a domain in which engine objects could not exist outside of car objects.

The following code box shows the header file for the `Car` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_1_CAR_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_1_CAR_H
struct car_t;
// Memory allocator
struct car_t* car_new();
// Constructor
void car_ctor(struct car_t*);
// Destructor
void car_dtor(struct car_t*);
// Behavior functions
void car_start(struct car_t*);
void car_stop(struct car_t*);
double car_get_engine_temperature(struct car_t*);
#endif
```

Code Box 7-3 [ExtremeC_examples_chapter7_1_car.h]: The public interface of the Car class

As you see, the preceding declarations have been made in a similar way to what we did for the `List` class in the last example of the previous chapter, *example 6.3*. One of the differences is that we have chosen a new suffix for the constructor function; `car_new` instead of `car_construct`. The other difference is that we have only declared the attribute structure `car_t`. We have not defined its fields, and this is called a *forward declaration*. The definition for the structure `car_t` will be in the source file which comes in the code box 7-5\. Note that in the preceding header file, the type `car_t` is considered an incomplete type which is not defined yet.

The following code box contains the header file for the `Engine` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_1_ENGINE_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_1_ENGINE_H
struct engine_t;
// Memory allocator
struct engine_t* engine_new();
// Constructor
void engine_ctor(struct engine_t*);
// Destructor
void engine_dtor(struct engine_t*);
// Behavior functions
void engine_turn_on(struct engine_t*);
void engine_turn_off(struct engine_t*);
double engine_get_temperature(struct engine_t*);
#endif
```

Code Box 7-4 [ExtremeC_examples_chapter7_1_engine.h]: The public interface of the Engine class

The following code boxes contains the implementations done for the `Car` and `Engine` classes. We begin with the `Car` class:

```cpp
#include <stdlib.h>
// Car is only able to work with the public interface of Engine
#include "ExtremeC_examples_chapter7_1_engine.h"
typedef struct {
  // Composition happens because of this attribute
  struct engine_t* engine;
} car_t;
car_t* car_new() {
  return (car_t*)malloc(sizeof(car_t));
}
void car_ctor(car_t* car) {
  // Allocate memory for the engine object
  car->engine = engine_new();
  // Construct the engine object
  engine_ctor(car->engine);
}
void car_dtor(car_t* car) {
  // Destruct the engine object
  engine_dtor(car->engine);
  // Free the memory allocated for the engine object
  free(car->engine);
}
void car_start(car_t* car) {
  engine_turn_on(car->engine);
}
void car_stop(car_t* car) {
  engine_turn_off(car->engine);
}
double car_get_engine_temperature(car_t* car) {
  return engine_get_temperature(car->engine);
}
```

Code Box 7-5 [ExtremeC_examples_chapter7_1_car.c]: The definition of the Car class

The preceding code box shows how the car has contained the engine. As you see, we have a new attribute as part of the `car_t` attribute structure, and it is of the `struct engine_t*` type. Composition happens because of this attribute.

Though the type `struct engine_t*` is still incomplete inside this source file, it can point to an object from a complete `engine_t` type at runtime. This attribute will point to an object that is going to be constructed as part of the `Car` class's constructor, and it will be freed inside the destructor. At both places, the car object exists, and this means that the engine's lifetime is included in the car's lifetime.

The `engine` pointer is private, and no pointer is leaking from the implementation. That's an important note. When you are implementing a composition relationship, no pointer should be leaked out otherwise it causes external code to be able to change the state of the contained object. Just like encapsulation, no pointer should be leaked out when it gives direct access to the private parts of an object. Private parts should always be accessed indirectly via behavior functions.

The `car_get_engine_temperature` function in the code box gives access to the `temperature` attribute of the engine. However, there is an important note regarding this function. It uses the public interface of the engine. If you pay attention, you'll see that the *car's private implementation* is consuming the *engine's public interface*.

This means that the car itself doesn't know anything about the implementation details of the engine. This is the way that it should be.

*Two objects that are not of the same type, in most cases, must not know about each other's implementation details*. This is what information hiding dictates. Remember that the car's behaviors are considered external to the engine.

This way, we can replace the implementation of the engine with an alternative one, and it should work, as long as the new implementation provides definitions for the same public functions declared in the engine's header file.

Now, let's look at the implementation of the `Engine` class:

```cpp
#include <stdlib.h>
typedef enum {
  ON,
  OFF
} state_t;
typedef struct {
  state_t state;
  double temperature;
} engine_t;
// Memory allocator
engine_t* engine_new() {
  return (engine_t*)malloc(sizeof(engine_t));
}
// Constructor
void engine_ctor(engine_t* engine) {
  engine->state = OFF;
  engine->temperature = 15;
}
// Destructor
void engine_dtor(engine_t* engine) {
  // Nothing to do
}
// Behavior functions
void engine_turn_on(engine_t* engine) {
  if (engine->state == ON) {
    return;
  }
  engine->state = ON;
  engine->temperature = 75;
}
void engine_turn_off(engine_t* engine) {
  if (engine->state == OFF) {
    return;
  }
  engine->state = OFF;
  engine->temperature = 15;
}
double engine_get_temperature(engine_t* engine) {
  return engine->temperature;
}
```

Code Box 7-6 [ExtremeC_examples_chapter7_1_engine.c]: The definition of the Engine class

The preceding code is just using the implicit encapsulation approach for its private implementation, and it is very similar to previous examples. But there is one thing to note about this. As you see, the `engine` object doesn't know that an external object is going to contain it in a composition relationship. This is like the real world. When a company is building engines, it is not clear which engine will go into which car. Of course, we could have kept a pointer to the container `car` object, but in this example, we didn't need to.

The following code box demonstrates the scenario in which we create a `car` object and invoke some of its public API to extract information about the car's engine:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "ExtremeC_examples_chapter7_1_car.h"
int main(int argc, char** argv) {
  // Allocate memory for the car object
  struct car_t *car = car_new();
  // Construct the car object
  car_ctor(car);
  printf("Engine temperature before starting the car: %f\n",
          car_get_engine_temperature(car));
  car_start(car);
  printf("Engine temperature after starting the car: %f\n",
          car_get_engine_temperature(car));
  car_stop(car);
  printf("Engine temperature after stopping the car: %f\n",
          car_get_engine_temperature(car));
  // Destruct the car object
  car_dtor(car);
  // Free the memory allocated for the car object
  free(car);
  return 0;
}
```

Code Box 7-7 [ExtremeC_examples_chapter7_1_main.c]: The main function of example 7.1

To build the preceding example, firstly we need to compile the previous three source files. Then, we need to link them together to generate the final executable object file. Note that the main source file (the source file that contains the `main` function) only depends on the car's public interface. So, when linking, it only needs the private implementation of the `car` object. However, the private implementation of the `car` object relies on the public interface of the engine interface; then, while linking, we need to provide the private implementation of the `engine` object. Therefore, we need to link all three object files in order to have the final executable.

The following commands show how to build the example and run the final executable:

```cpp
$ gcc -c ExtremeC_examples_chapter7_1_engine.c -o engine.o
$ gcc -c ExtremeC_examples_chapter7_1_car.c -o car.o
$ gcc -c ExtremeC_examples_chapter7_1_main.c -o main.o
$ gcc engine.o car.o main.o -o ex7_1.out
$ ./ex7_1.out
Engine temperature before starting the car: 15.000000
Engine temperature after starting the car: 75.000000
Engine temperature after stopping the car: 15.000000
$
```

Shell Box 7-1: The compilation, linking, and execution of example 7.1

In this section, we explained one type of relationship that can exist between two objects. In the next section, we'll talk about the next relationship. It shares a similar concept to the composition relationship, but there are some significant differences.

# Aggregation

Aggregation also involves a container object that contains another object. The main difference is that in aggregation, the lifetime of the contained object is independent of the lifetime of the container object.

In aggregation, the contained object could be constructed even before the container object is constructed. This is opposite to composition, in which the contained object should have a lifetime shorter than or equal to the container object.

The following example, *example 7.2*, demonstrates an aggregation relationship. It describes a very simple game scenario in which a player picks up a gun, fires multiple times, and drops the gun.

The `player` object would be a container object for a while, and the `gun` object would be a contained object as long as the player object holds it. The lifetime of the gun object is independent of the lifetime of the player object.

The following code box shows the header file of the `Gun` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_2_GUN_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_2_GUN_H
typedef int bool_t;
// Type forward declarations
struct gun_t;
// Memory allocator
struct gun_t* gun_new();
// Constructor
void gun_ctor(struct gun_t*, int);
// Destructor
void gun_dtor(struct gun_t*);
// Behavior functions
bool_t gun_has_bullets(struct gun_t*);
void gun_trigger(struct gun_t*);
void gun_refill(struct gun_t*);
#endif
```

Code Box 7-8 [ExtremeC_examples_chapter7_2_gun.h]: The public interface of the Gun class

As you see, we have only declared the `gun_t` attribute structure as we have not defined its fields. As we have explained before, this is called a forward declaration and it results in an incomplete type which cannot be instantiated.

The following code box shows the header file of the `Player` class:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_2_PLAYER_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_2_PLAYER_H
// Type forward declarations
struct player_t;
struct gun_t;
// Memory allocator
struct player_t* player_new();
// Constructor
void player_ctor(struct player_t*, const char*);
// Destructor
void player_dtor(struct player_t*);
// Behavior functions
void player_pickup_gun(struct player_t*, struct gun_t*);
void player_shoot(struct player_t*);
void player_drop_gun(struct player_t*);
#endif
```

Code Box 7-9 [ExtremeC_examples_chapter7_2_player.h]: The public interface of the Player class

The preceding code box defines the public interface of all player objects. In other words, it defines the public interface of the `Player` class.

Again, we have to forward the declaration of the `gun_t` and `player_t` structures. We need to have the `gun_t` type declared since some behavior functions of the `Player` class have arguments of this type.

The implementation of the `Player` class is as follows:

```cpp
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ExtremeC_examples_chapter7_2_gun.h"
// Attribute structure
typedef struct {
  char* name;
  struct gun_t* gun;
} player_t;
// Memory allocator
player_t* player_new() {
  return (player_t*)malloc(sizeof(player_t));
}
// Constructor
void player_ctor(player_t* player, const char* name) {
  player->name =
      (char*)malloc((strlen(name) + 1) * sizeof(char));
  strcpy(player->name, name);
  // This is important. We need to nullify aggregation pointers
  // if they are not meant to be set in constructor.
  player->gun = NULL;
}
// Destructor
void player_dtor(player_t* player) {
  free(player->name);
}
// Behavior functions
void player_pickup_gun(player_t* player, struct gun_t* gun) {
  // After the following line the aggregation relation begins.
  player->gun = gun;
}
void player_shoot(player_t* player) {
  // We need to check if the player has picked up the gun
  // otherwise, shooting is meaningless
  if (player->gun) {
    gun_trigger(player->gun);
  } else {
    printf("Player wants to shoot but he doesn't have a gun!");
    exit(1);
  }
}
void player_drop_gun(player_t* player) {
  // After the following line the aggregation relation
  // ends between two objects. Note that the object gun
  // should not be freed since this object is not its
  // owner like composition.
  player->gun = NULL;
}
```

Code Box 7-10 [ExtremeC_examples_chapter7_2_player.c]: The definition of the Player class

Inside the `player_t` structure, we declare the pointer attribute `gun` that is going to point to a `gun` object soon. We need to nullify this in the constructor because unlike composition, this attribute is not meant to be set as part of the constructor.

If an aggregation pointer is required to be set upon construction, the address of the target object should be passed as an argument to the constructor. Then, this situation is called a *mandatory aggregation*.

If the aggregation pointer can be left as null in the constructor, then it is an *optional aggregation*, as in the preceding code. It is important to nullify the optional aggregation pointers in the constructor.

In the function `player_pickup_gun`, the aggregation relationship begins, and it ends in the function `player_drop_gun` when the player drops the gun.

Note that we need to nullify the pointer `gun` after dropping the aggregation relationship. Unlike in composition, the container object is not the *owner* of the contained object. So, it has no control over its lifetime. Therefore, we should not free the gun object in any place inside the player's implementation code.

In optional aggregation relations, we may not have set the contained object at some point in the program. Therefore, we should be careful while using the aggregation pointer since any access to a pointer that is not set, or a pointer that is `null`, can lead to a segmentation fault. That's basically why in the function `player_shoot`, we check the `gun` pointer is valid. If the aggregation pointer is null, it means that the code using the player object is misusing it. If that's the case, we abort the execution by returning 1 as the *exit* code of the process.

The following code is the implementation of the `Gun` class:

```cpp
#include <stdlib.h>
typedef int bool_t;
// Attribute structure
typedef struct {
  int bullets;
} gun_t;
// Memory allocator
gun_t* gun_new() {
  return (gun_t*)malloc(sizeof(gun_t));
}
// Constructor
void gun_ctor(gun_t* gun, int initial_bullets) {
  gun->bullets = 0;
  if (initial_bullets > 0) {
    gun->bullets = initial_bullets;
  }
}
// Destructor
void gun_dtor(gun_t* gun) {
  // Nothing to do
}
// Behavior functions
bool_t gun_has_bullets(gun_t* gun) {
  return (gun->bullets > 0);
}
void gun_trigger(gun_t* gun) {
  gun->bullets--;
}
void gun_refill(gun_t* gun) {
  gun->bullets = 7;
}
```

Code Box 7-11 [ExtremeC_examples_chapter7_2_gun.c]: The definition of the Gun class

The preceding code is straightforward, and it is written in a way that a gun object doesn't know that it will be contained in any object.

Finally, the following code box demonstrates a short scenario that creates a `player` object and a `gun` object. Then, the player picks up the gun and fires with it until no ammo is left. After that, the player refills the gun and does the same. Finally, they drop the gun:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "ExtremeC_examples_chapter7_2_player.h"
#include "ExtremeC_examples_chapter7_2_gun.h"
int main(int argc, char** argv) {
  // Create and constructor the gun object
  struct gun_t* gun = gun_new();
  gun_ctor(gun, 3);
  // Create and construct the player object
  struct player_t* player = player_new();
  player_ctor(player, "Billy");
  // Begin the aggregation relation.
  player_pickup_gun(player, gun);
  // Shoot until no bullet is left.
  while (gun_has_bullets(gun)) {
    player_shoot(player);
  }
  // Refill the gun
  gun_refill(gun);
  // Shoot until no bullet is left.
  while (gun_has_bullets(gun)) {
    player_shoot(player);
  }
  // End the aggregation relation.
  player_drop_gun(player);
  // Destruct and free the player object
  player_dtor(player);
  free(player);
  // Destruct and free the gun object
  gun_dtor(gun);
  free(gun);
  return 0;
}
```

Code Box 7-12 [ExtremeC_examples_chapter7_2_main.c]: The main function of example 7.2

As you see here, the `gun` and `player` objects are independent of each other. The responsible logic for creating and destroying these objects is the `main` function. At some point in the execution, they form an aggregation relationship and perform their roles, then at another point, they become separated. The important thing in aggregation is that the container object shouldn't alter the lifetime of the contained object, and as long as this rule is followed, no memory issues should arise.

The following shell box shows how to build the example and run the resulting executable file. As you see, the `main` function in *Code Box 7-12* doesn't produce any output:

```cpp
$ gcc -c ExtremeC_examples_chapter7_2_gun.c -o gun.o $ gcc -c ExtremeC_examples_chapter7_2_player.c -o player.o $ gcc -c ExtremeC_examples_chapter7_2_main.c -o main.o $ gcc gun.o player.o main.o -o ex7_2.out $ ./ex7_2.out $
```

Shell Box 7-2: The compilation, linking, and execution of example 7.2

In an object model created for a real project, the amount of aggregation relationships is usually greater than the number of composition relationships. Also, aggregation relationships are more visible externally because, in order to make an aggregation relationship, some dedicated behavior functions are required, at least in the public interface of the container object, to set and reset the contained object.

As you see in the preceding example, the `gun` and `player` objects are separated from the start. They become related for a short period of time, and then they become separated again. This means that the aggregation relationship is temporary, unlike the composition relationship, which is permanent. This shows that composition is a stronger form of *possession* (to-have) relationship between objects, while aggregation exhibits a weaker relationship.

Now, a question comes to mind. If an aggregation relationship is temporary between two objects, is it temporary between their corresponding classes? The answer is no. The aggregation relationship is permanent between the types. If there is a small chance that in the future, two objects from two different types become related based on an aggregation relationship, their types should be in the aggregation relationship permanently. This holds for composition as well.

Even a low chance of there being an aggregation relationship should cause us to declare some pointers in the attribute structure of the container object, and this means that the attribute structure is changed permanently. Of course, this is only true for class-based programming languages.

Composition and aggregation both describe the possession of some objects. In other words, these relationships describe a "to-have" or "has-a" situation; a player **has** a gun, or a car **has** an engine. Every time you feel that an object possesses another one, it means there should either be a composition relationship or an aggregation relationship between them (and their corresponding classes).

In the next chapter, we'll continue our discussion regarding relationship types by looking at the *inheritance* or *extension* relationship.

# Summary

In this chapter, the following topics have been discussed:

*   The possible relationship types between classes and objects.
*   The differences and similarities between a class, an object, an instance, and a reference.
*   Composition, which entails that a contained object is totally dependent on its container object.
*   Aggregation, in which the contained object can live freely without any dependency on its container object.
*   The fact that aggregation can be temporary between objects, but it is defined permanently between their types (or classes).

In the next chapter, we continue to explore OOP, primarily addressing the two further pillars upon which it is based: inheritance and polymorphism.