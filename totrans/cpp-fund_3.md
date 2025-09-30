# *Chapter 3*

# Classes

## Lesson Objectives

By the end of this chapter, you will be able to:

*   Declare and define a class
*   Access the members of a class using objects
*   Apply access modifiers to encapsulate data
*   Use the static modifier on data members and member functions
*   Implement a nested class
*   Utilize the friend specifier to access private and protected members
*   Use constructors, copy constructors, assignment operators, and destructors
*   Overload operators
*   Implement functors

In this chapter, we will be learning about classes and objects in C++.

## Introduction

In the previous chapter, we saw how we can use functions to combine basic operations into units with a clear meaning. Additionally, in the first chapter, we saw how, in C++, we can store data in basic types, such as integers, chars, and floats.

In this chapter, we will be covering how to define and declare classes and how to access member functions of a class. We will explore what `member` and `friend` functions are and how to use each in a program. Later in the chapter, we will look at how constructors and destructors work. At the end of the chapter, we will explore functors and how you can use them in your programs.

## Declaring and Defining a Class

A **class** is a way to combine data and operations together to create new types that can be used to represent complex concepts.

Basic types can be composed to create more meaningful abstractions. For example, *location data* is composed of latitude and longitude coordinates, which are represented as `float` values. With such a representation, when our code needs to operate on a location, we would have to provide both the latitude and longitude as separate variables. This is error-prone, as we might forget to pass one of the two variables, or we could provide them in the wrong order.

Additionally, computing the distance between two coordinates is a complex task and we don't want to write the same code again and again. It becomes even more difficult when we use more complex objects.

Continuing our example on Coordinates, instead of using operations on two `float` types, we can define a type, which stores the location and provides the necessary operations to interact with it.

### The Advantages of Using Classes

Classes provide several benefits, such as abstraction, information hiding, and encapsulation. Let's explore each of these in depth:

*   `float` variables, but this does not represent the concept that we want to use. The programmer needs to remember that the two variables have a different meaning and should be used together. Classes allow us to explicitly define a concept, composed by data and operations on that data, and assign a *name* to it.

    In our example, we can create a class to represent GPS coordinates. The data will be the two `float` variables to describe `float` variables that are used to represent it.

*   **Information hiding**: The process of exposing a set of functionalities to the user of the class while hiding the details of how they are implemented in the class.

    This approach reduces the complexity of interacting with the class and makes it easier to update the class implementation in the future:

![Figure 2.1: The class exposes functionality that the user code uses directly, hiding the fact that it is implemented with two floats](img/Image70122.jpg)

###### Figure 2.1: The class exposes functionality that the user code uses directly, hiding the fact that it is implemented with two floats

We discussed the fact that we can represent GPS coordinates as latitude and longitude. Later, we might decide to represent a coordinate as the distance from the **North Pole**. Thanks to information hiding, we can change how a class is implemented and the users of the class will not be impacted, since we do not change the functionality offered by the class:

![Figure 2.2: The implementation of the class changes, but since it is hidden from the user and the functionality was not changed, the user does not have to change how their code interacts with the class](img/Image70130.jpg)

###### Figure 2.2: The implementation of the class changes, but since it is hidden from the user and the functionality was not changed, the user does not have to change how their code interacts with it

The set of functionalities the class exposes to the users is normally referred to as the **public interface**.

#### Note

Changing the implementation of a class is generally more convenient than to changing the interface of a class, which requires you to change all the users of the class to adapt to the new interface. Getting the design of the public interface of a class right is the first step to creating a class that is easy to use and requires low maintenance.

*   **Encapsulation**: This is the principle of grouping the data and the operations we can perform on it together. Since the data is hidden in the class, the user cannot access or operate on it. The class must provide functionality to interact with it. C++ enables encapsulation by letting the user put the operations to interact with a class and the data that is used to implement such operations in the same unit: **class**.

Let's explore the structure of a class in C++ and the information associated with it. The following is the basic structure of a class:

```cpp
class ClassName { 
  // class body
};
```

#### Note

It is common to forget the last `semicolon` after closing curly brackets. Always make sure that you add it.

### C++ Data Members and Access Specifiers

Inside the body of a class, we can define the following class members:

*   **Data members**: These are variables that live inside a class, which look like a **variable declaration**, but are inside the class body. They are also called **fields**.
*   **Member functions**: These are functions that can access the variables inside a class. They look like a **function declaration** but are inside the class body. They are also called **methods**.

As we mentioned before, classes support information hiding by denying users of the class to access information. The programmer uses **access specifiers** to specify which parts of the class are available for the user to be accessed.

There are the following three access specifiers in C++:

*   `private` can only be accessed by the functions inside the class and are not allowed to be accessed directly outside the class
*   `protected` can only be accessed by the functions inside the class and the derived classes. We will learn more about in the last chapter of this book
*   `public` can be accessed from anywhere in the program

Access specifiers followed by a colon delimit an area in the class, and any member defined in that area has the access specifier that precedes it. Here's the syntax:

```cpp
class ClassName {
  private:
    int privateDataMember;
    int privateMemberFunction();
  protected:
    float protectedDataMember;
    float protectedMemberFunction();
  public:
    double publicDataMember;
    double publicMemberFunction();
};
```

#### Note

By default, class members have the `private` access modifier.

In C++, we can also use the `struct` keyword to define a class. A `struct` is identical to a class, with the only exception that, by default, the access modifier is `public`, while for the class it is `private`.

The following side-by-side code snippets are equivalent:

![](img/C11557_03_03.jpg)

###### Figure 3.2: The difference between the code snippets of class and struct

Whether to use `struct` or `class` depends on convention used: usually, we use `structs` when we want a collection of data members that should be accessible from anywhere in the code; on the other hand, we use classes when we are modelling a more complex concept.

We have learned how to define a class. Now, let's understand how to use one in a program.

A class defines a blueprint or the design of an object. Like a blueprint, we can create multiple objects from the same class. These objects are called **instances**.

We can create an instance in the same way that we create any basic type: define the type of the variable followed by the name of the variable. Let's explore the following example.

```cpp
class Coordinates {
  public:
    float latitude;
    float longitude;
    float distance(const Coordinates& other_coordinate);
};
```

Here's an example that shows a class that has multiple instances:

```cpp
Coordinates newYorkPosition;
Coordinates tokyoPosition;
```

Here, we have two instances of the `Coordinates` class, each with their `latitude` and `longitude`, which can change independently. Once we have an instance, we can access its members.

When we declare a class, we create a new scope called the `class` or a `struct` from a scope outside the class is the dot (`.`) operator.

For the previously defined variables, we can access their `latitude` using the following code:

```cpp
float newYorkLatitude = newYorkPosition.latitude;
```

If we want to call a member function instead, we can invoke it like this:

```cpp
float distance = newYorkPosition.distance(tokyoPosition);
```

On the other hand, when we are writing the body of a `class` method, we are inside the class's scope. This means that we can access the other members of the class by using their names directly, without having to use the *dot* operator. The members of the current instance are going to be used.

Let's assume that the `distance` method is implemented as follows:

```cpp
float Coordinates::distance(const Coordinates& other_coordinate) {
  return pythagorean_distance(latitude, longitude, other_coodinate.latitude, other_coordinate.longitude);
}
```

When we call `newYorkPosition.distance(tokyoPosition);`, the `distance` method is called on the `newYorkPosition` instance. This means that `latitude` and `longitude` in the `distance` method refer to `newYorkPosition.latitude` and `newYorkPosition.longitude`, while `other_coordinate.latitude` refers to `tokyoPosition.latitude`.

If we had called `tokyoPosition.distance(newYorkPosition);` instead, the current instance would have been `tokyoPosition`, and `latitude` and `longitude` would have referred to the `tokyoPosition`, and `other_coordinate` to `newYorkPosition`.

### Static Members

In the previous section, we learned that a class defines the fields and methods that compose an object. It is like a blueprint, specifying what the object looks like, but it does not actually build it. *An instance is the object that's built from the blueprint that's defined by the class*. Instances contain data and we can operate on instances.

Imagine the blueprint of a car. It specifies the engine of the car and that the car will have four wheels. The blueprint is the class of the car, but we cannot turn on and drive a blueprint. A car that's built by following the blueprint is an instance of the class. The built car has four wheels and an engine, and we can drive it. In the same way, an instance of a class contains the fields that are specified by the class.

This means that the value of each field is connected to a specific instance of a class and evolves independently from the fields of all the other instances. At the same time, it also means that a field cannot exist without the associated instance: there would be no object that would be able to provide the storage (the space in memory) to store the value of the field!

However, sometimes, we want to share the same value across all instances. In those cases, we can associate the field with the class instead of the instance by creating a `static` field. Let's examine the following syntax:

```cpp
class ClassName {
  static Type memberName;
};
```

There will be only one `memberName` field, which is shared across all instances. Like any variable in C++, `memberName` needs to be stored in memory. We cannot use the storage of the instance object, since `memberName` is not associated with any specific instance. `memberName` is stored in a similar way to a *global variable*.

Outside of the class in which the static variable is declared, in a `.cpp` file, we can define the value of the `static` variable. The syntax to initialize the value is as follows:

```cpp
Type ClassName::memberName = value;
```

#### Note

Note that we do not repeat the `static` keyword.

It is important to define the values of the `static` variables in the `.cpp` file. If we define them inside the **header** file, the definition will be included anywhere inside the header, which will create multiple definitions, and the **linker** will complain.

A class static variable's lifetime lasts for the complete duration of the program, like global variables.

Let's see an example of how a static field in a class can be defined in the header and how to assign a value to it in the `.cpp` file:

```cpp
// In the .h file
class Coordinates {
  // Data member
  float latitude_ = 0;
  // Data member
  float longitude_ = 0; 
public:
  // Static data member declaration
  static const Coordinates hearthCenter;
  // Member function declaration
  float distanceFrom(Coordinates other);
  // Member function definition
  float distanceFromCenter() {
    return distanceFrom(hearthCenter);
  }
};
// In the .cpp file 
// Static data member definition
const Coordinates Coordinates::hearthCenter = Coordinates(0, 0);
```

When accessing the members of an instance, we learned to use the dot operator.

When accessing a static member, we might not have an instance to use the dot operator on. C++ gives us the ability to access the static members of a class by using the `::`), after the class name.

#### Note

Always use `const` when declaring a static field. Any instance can access the static fields of its class; if they are **mutable**, it becomes extremely hard to track down which instances is modifying the value. In programs that use multiple threads, it is common to create bugs by modifying the static fields from different threads at the same time.

Let's examine the following exercise to understand how static variables work.

### Exercise 7: Working with Static Variables

Let's write a program to print and find the square of numbers from 1 to 10:

1.  Include the required header files.
2.  Write the `squares()` function and the following logic:

    ```cpp
    void squares() 
    {  
        static int count = 1; 
        int x = count * count;
        x = count * count;
        std::cout << count << "*" << count;
        std::cout << ": " << x <<std::endl;
        count++;
    }
    ```

3.  Now, in the `main` function, add the following code:

    ```cpp
    int main() 
    { 
        for (int i=1; i<11; i++)     
            squares(); 
        return 0; 
    }
    ```

    The output is as follows:

    ```cpp
    1*1: 1
    2*2: 4
    3*3: 9
    4*4: 16
    5*5: 25
    6*6: 36
    7*7: 49
    8*8: 64
    9*9: 81
    10*10: 100
    ```

In addition to static fields, classes can also have static methods.

A static method is associated with a class; it can be invoked without an instance. Since the fields and members of a class are associated with an instance, while static methods are not, static methods cannot invoke them. Static methods can be invoked using the scope resolution operator: `ClassName::staticMethodName();`.

#### Note

Static methods can only call other static methods and static fields inside a class.

## Member Functions

**Member functions** are functions that are used to manipulate the data members of a class, and they define the properties and behavior of the objects of the class.

Declaring a member function is just a matter of declaring a function inside the body of a class. Let's examine the following syntax:

```cpp
class Car
{
  public:
  void turnOn() {}
};
```

Member functions, like the data members of a class, can be accessed using the dot (`.`) operator that's applied on the object:

```cpp
Car car;
car.turnOn();
```

Let's understand how to declare a member function outside the class scope.

### Declaring a Member Function

Member functions, like data members, must be declared inside the class. However, a member function's implementation can be placed either inside or outside the class, body.

The following is a definition of a member function outside of the class, scope. This is done by using the scope resolution operator (`::`) to declare that the function that's being referred to is a member of the class. In the class, body, the function is declared with its prototype:

```cpp
class Car
{
  public:
  void turnOn();
};
void Car::turnOn() {}
```

### Using const Member Functions

The member functions of a class can be qualified as `const`, which means that the function limits its access to be read-only. Moreover, a member function is required to be `const` when it accesses `const` member data. So, `const` member functions are not allowed to modify the state of an object or call another function that does so.

To declare a member function as `const`, we use the `const` keyword in the function declaration after the function name and before its body:

```cpp
const std::string& getColor() const
{
  // Function body
}
```

In addition to the overloading rules that we learned in the previous chapter that member functions can be overloaded in their const-ness, which means that two functions can have identical signatures except for one being `const` and the other not. The `const` member function will be called when an object is declared `const`; otherwise, the non-const function is called. Let's examine the following code:

```cpp
class Car
{
  std::string& getColor() {}
  const std::string& getColor() const {}
};
Car car;
// Call std::string& getColor()
car.getColor();
const Car constCar; 
// Call const Color& getColor() const
constCar.getColor();
```

#### Note

It is important to distinguish between a `const` function and a function returning a `const` type. Both make use of the same `const` keyword, but in different places in the function prototype. They express a different concept and are independent.

The following examples show three versions of the `const` function:

*   The first one is a `const` member function
*   The second returns a `const` reference
*   The third one is a `const` function that returns a `const` reference:

    ```cpp
    type& function() const {}
    const type& function() {}
    const type& function() const {}
    ```

### The this Keyword

When the `this` keyword is used in the `class` context, it represents a pointer whose value is the address of the object on which the member function is called. It can appear within the body of any non-static member function.

In the following example, `setColorToRed()` and `setColorToBlue()` perform the same action. Both set a data member, but the former uses the `this` keyword to refer to the current object:

```cpp
class Car
{
  std::string color;
  void setColorToRed()
  {
    this->color = "Red";
    // explicit use of this
  }
  void setColorToBlue()
  {
    color = "Blue";
    // same as this->color = "Blue";
  }
};
```

#### Note

`pointer->member` is a convenient way to access the member of the `struct` pointed by `pointer`. It is equivalent to `(*pointer).member`.

### Exercise 8: Creating a Program Using the this Keyword to Greet New Users

Let's write a program that asks users for their names and greets them with a welcoming message:

1.  First, include the required header files.
2.  Then, add the following functions to print the required output:

    ```cpp
    class PrintName {
        std::string name;

    };
    ```

3.  Now, let's complete the program with a closing message using the `this` keyword. Define the following methods inside the previous class:

    ```cpp
    public:
       void set_name(const std::string &name){
           this->name = name;
       }    
       void print_name() {
           std::cout << this->name << "! Welcome to the C++ community :)" << std::endl;
       }
    ```

4.  Write the `main` function, as follows:

    ```cpp
    int main()
    {
    PrintName object;
    object.set_name("Marco");
    object.print_name();
    }
    ```

    The output is as follows:

    ```cpp
    Marco! Welcome to the C++ community :)
    ```

    #### Note

    A function argument that has the same name as a data member of a class can shadow its visibility. In this case, the `this` keyword is required for disambiguation.

### Non-Member Class-Related Functions

Defined as functions or operations that conceptually belong to the interface of a class, non-member class-related functions are not part of a class itself. Let's examine the following example:

```cpp
class Circle{
  public:
    int radius;
};
ostream& print(ostream& os, const Circle& circle) {
  os << "Circle's radius: " << circle.radius;
  return os;
}
```

The print function writes the radius of the circle on the given stream, which is most commonly the standard output.

### Activity 7: Information Hiding Through Getters and Setters

In this activity, you are being asked to define a class named `Coordinates`, which contains two data members, and `latitude` and `longitude`, both of type `float` and not publicly accessible.

There are four operations that are associated with the `Coordinates` class: `set_latitude`, `set_longitude`, `get_latitude`, and `get_longitude`.

#### Note

The `set_latitude` and `set_longitude` operations are used to `x` and `y` coordinates (also referred to as `get_latitude` and `get_longitude` are used to **retrieve** them (sometimes called **getters**).

Performing encapsulation using the member functions through getter and setters.

To perform this activity, follow these steps:

1.  Define a class with the name `Coordinates`, with its members under a `private` access specifier.
2.  Add the four operations previously specified and make them publicly accessible by preceding their declaration by the `public` access specifier.
3.  The setters (`set_latitude` and `set_longitude`) should take a float as a parameter and return `void`, while the getters do not take any parameters and return a `float`.
4.  The four methods should now be implemented. The setters assign the given value to the corresponding member they are supposed to set; the getters return the values that are stored.

    #### Note

    The solution for this activity can be found on page 288.

## Constructors and Destructors

Up until now, we have learned how to declare data members, how to use them in functions with a `public` specifier, and how to access them. Now, let's explore how to set a value to them.

In the following example, we'll declare a `struct` by the name of `Rectangle`, and set a value to it as follows:

```cpp
struct Rectangle {
  int height;
  int width;
};
Rectangle rectangle;
// What will the following print function print?
std::cout << "Height: " << rectangle.height << std::endl;
```

This line will print a random value because we never set the value of `int`. The C++ rule for the initialization of basic types is that they get non-specified values.

#### Note

In some situations, the values of variables are set to `0` when they are not initialized. This might happen because of some details in the implementation of the operating system, the standard library, or the compiler, and the C++ standard does not guarantee it. A program will have strange bugs when it relies on this behavior, since it is unpredictable when variables are initialized to `0`. Always explicitly initialize variables with basic types.

### Constructors

The way to initialize data members is by using a **constructor**. A constructor is a special member function that has the *same name* as the class and *no return type*, and it is called automatically by the compiler when a new object of the class is created.

Like any other function, a constructor can accept parameters and has a function body. We can invoke a constructor by adding a parameter list after the name of the variable:

```cpp
Rectangle rectangle(parameter1, paramter2, ..., parameterN);
```

When there are no parameters, we can avoid using parentheses, which is what we did in the previous example.

An example of a constructor with no parameters for the `Rectangle` struct would look as follows:

```cpp
struct Rectangle {
  int height, width;
  Rectangle() {
    height = 0;
    width = 0;
  }
};
```

#### Note

When the only operation the constructor does is initialize the data members, opt for using the initialization list, which we will show you later in this chapter.

In addition to assigning values to data members, a constructor can also execute code, similar to a normal function body. This is important for the concept of the *class invariant*.

A key advantage of hiding the implementation of a class in *private* members and only exposing *public* methods to interact with the concept represented by the class is the ability to enforce a class invariant.

A class invariant is a property or a set of properties of a class that should be `true` for any given instance of the class, at any point. It is called `true`.

Let's look at an example of a class that requires a class invariant. Imagine that we want to create a class that represents a date. The date would contain a year, month, and day, all represented as integers.

Implement it as a `struct` with all the fields as `public`. Refer to the following code:

```cpp
struct Date {
  int day;
  int month;
  int year;
};
```

Now, the user could easily do the following:

```cpp
Date date;
date.month = 153;
```

The previous code does not make any sense, as there are only 12 months in the Gregorian calendar.

A class invariant for the date would be that the month is always between 1 and 12, and that the day is always between 1 and 31, and depending on the month, even less.

Independently of any change the user performs on the `Date` object, the invariant must always hold.

A class can hide the detail that the date is stored as three integers and expose the functions to interact with the `Date` object. Functions can expect to find the dates to always be in a valid state (the invariant is satisfied at the start of the function), and they need to make sure to leave the class in a valid state (the invariant is satisfied at the end of the function).

The constructor does not only initialize the data members but also ensure that the class respects the invariant. After the constructor is executed, the invariant must be `true`.

#### Note

The concept of an invariant is not specific to the C++ language, and there is no dedicated facility to specify the invariant of a class. A best practice is to document the expected invariant of the class together with the class code so that the developers working with the class can easily check what the expected invariant is and make sure they respect it.

Using assertions in code also helps in identifying when the invariant is not respected. This probably means there is a bug in the code.

### Overloading Constructor

Similar to other functions, we can overload the constructor by accepting different parameters. This is useful when an object can be created in several ways, since the user can create the object by providing the expected parameter, and the correct constructor is going to be called.

We showed an example of a default constructor for the `Rectangle` class earlier in this chapter. If we want to add a constructor that creates a rectangle from a square, we could add the following constructor to the `Rectangle` class:

```cpp
class Rectangle {
  public: 
    Rectangle(); // as before
    Rectangle (Square square);
}; 
```

The second constructor is an overloaded constructor and will be invoked according to the way the class object is initialized.

In the following example, the first line will call the constructor with empty parameters, while the second line will call the overloaded constructor:

```cpp
Rectangle obj; // Calls the first constructor
Rectangle obj(square); // Calls the second overloaded constructor
```

#### Note

A constructor with a single non-default parameter is also called a **converting constructor**. This kind of constructor specifies an implicit conversion, from the type of the argument to the class type.

The following conversion is possible according to the previous definitions:

```cpp
Square square;
Rectangle rectangle(square);
```

The constructor is initialized, and it converts from type `Square` to `Rectangle`.

Similarly, the compiler can create implicit conversions when calling functions, as shown in the following example:

```cpp
void use_rectangle(Rectangle rectangle);
int main() {
  Square square;
  use_rectangle(square);
}
```

When calling `use_rectangle`, the compiler creates a new object of type `Rectangle` by calling the conversion constructor, which accepts a `Square`.

One way to avoid this is to use the `explicit` specifier before the constructor definition:

```cpp
explicit class_name(type arg) {}
```

Let's look at a different implementation of `Rectangle`, which has an explicit constructor:

```cpp
class ExplicitRectangle {
  public: 
    explicit ExplicitRectangle(Square square);
}; 
```

When we use try to use `Square` to call a function that takes `ExplicitRectangle`, we get an error:

```cpp
void use_explicit_rectangle(ExplicitRectangle rectangle);
int main() {
    Square square;
    use_explicit_rectangle(square); // Error!
}
```

### Constructor Member Initialization

Constructors, as we've seen already, are used to initialize members. Up until now, we have initialized the members inside the body of the function by assigning values to members directly. C++ provides a feature to initialize the values of fields of the class in a more ergonomic way: initialization lists. Initialization lists allow you to call the constructor of the data members of class before the constructor body is executed. To write an initializer list, insert a colon (`:`) and a comma-separated list of initializations for class members before the constructor's body.

Let's look at the following example:

```cpp
class Rectangle {
  public:
    Rectangle(): width(0), height(0) { } //Empty function body, as the variables have already been initialized
  private:
    int width;
    int height;
};
```

Note how, in this last case, the constructor does nothing other than initialize its members. Hence, it has an empty function body.

Now, if we try to print the width and the height of the `Rectangle` object, we will notice that they are correctly initialized to `0`:

```cpp
Rectangle rectangle; 
std::cout << "Width: " << rectangle.width << std::endl;  // 0
std::cout << "Height: " << rectangle.height << std::endl; // 0
```

Initializer lists are the recommended way to initialize member variables in C++, and they are necessary when a data member is `const`.

When using an initializer list, the order in which the members are constructed is the one in which they are declared inside the class; not the one in which they appear in the initializer list. Let's look at the following example:

```cpp
class Example {
    Example() : second(0), first(0) {}
    int first;
    int second;
};
```

When calling the default constructor of the `Example` class, the `first` method will be initialized first, and the `second` method after it, even if they appear in a different order in the initializer list.

#### Note

You should always write the members in the initializer list in the same order as they are declared; compilers will help you by warning you when the order differs from the expected one.

### Aggregate Classes Initialization

Classes or structs with no user-declared constructors, no private or protected specifiers non-static data members, no base classes, and no virtual functions are considered aggregate.

#### Note

We will talk about base classes and virtual functions in chapter 6.

These types of classes can be initialized, even though they do not have a constructor, by using a brace-enclosed comma-separated list of initializer-clauses, as shown here:

```cpp
struct Rectangle {
  int length;
  int width;
};
Rectangle rectangle = {10, 15};
std::cout << rectangle.length << "," << rectangle.width;
// Prints: 10, 15
```

### Destructors

A *destructor* function is called automatically when the object goes out of scope and is used to destroy objects of its class type.

Destructors have the same name as the class preceded by a tilde (`~`) and do not take any argument nor return any value (not even void). Let's examine the following example:

```cpp
class class_name {
  public:
    class_name() {} // constructor
    ~class_name() {} // destructor
};
```

After executing the body of the destructor and destroying any automatic objects allocated within the body, a destructor for a class calls the destructors for all the direct members of the class. Data members are destroyed in reverse order of their construction.

### Exercise 9: Creating a Simple Coordinate Program to Demonstrate the Use of Constructors and Destructors

Let's write a simple program to demonstrate the use of constructors and destructors:

1.  First, include the required header files.
2.  Now, add the following code to the `Coordinates` class:

    ```cpp
    class Coordinates {
        public:
        Coordinates(){
            std::cout << "Constructor called!" << std::endl;
        }

        ~Coordinates(){
            std::cout << "Destructor called!" << std::endl;
        }
    };
    ```

3.  In the `main` function, add the following code:

    ```cpp
    int main() 
    { 

      Coordinates c;  
      // Constructor called!
      // Destructor called!

    }
    ```

    The output is as follows:

    ```cpp
    Constructor called!
    Destructor called!
    ```

### Default Constructor and Destructor

All the classes needs constructor and destructor functions. When the programmer does not define these, the compiler automatically creates an implicitly defined constructor and destructor.

#### Note

The default constructor might not initialize data members. Classes that have members of a built-in or compound type should ordinarily either initialize those members inside the class or define their version of the default constructor.

### Activity 8: Representing Positions in a 2D Map

Alice is building a program to show 2D maps of the world. Users need to be able to save locations, such as their house, a restaurant, or their workplace. To enable this functionality, Alice needs to be able to represent a position in the world.

Create a class named `Coordinates` whose data members are the 2D coordinates of a point. To ensure that the object is always properly initialized, implement a constructor to initialize the data members of the class.

Let's perform the following steps:

1.  The first step is to create a class named `Coordinates` containing the coordinates as data members.
2.  Now, there are two floating-point values, `_latitude` and `_longitude`, which identify the coordinates on a geographic coordinate system. Additionally, these data members are defined with the `private` access specifier.
3.  Extend the class with a `public` constructor that takes two arguments, `latitude` and `longitude`, which are used to initialize the data members of the class.
4.  Alice can now use this `Coordinates` class to represent 2D positions on the map.

    #### Note

    The solution for this activity can be found on page 289.

## Resource Acquisition Is Initialization

**Resource Acquisition Is Initialization**, or just **RAII**, is a programming idiom that is used to manage the life cycle of a resource automatically by binding it to the lifetime of an object.

Through the smart use of the constructor and destructor of an object, you can achieve RAII. The former acquires the resource, while the latter takes care of realizing it. The constructor is allowed to throw an exception, when a resource cannot be acquired, while the destructor must never throw exceptions.

Typically, it is a good practice to operate on a resource via an instance of a RAII class when its usage involves `open()`/`close()`, `lock()`/`unlock()`, `start()`/`stop()`, `init()`/`destroy()`, or similar function calls.

The following is a way to open and close a file using an RAII-style mechanism.

#### Note

C++, like many languages, represents input/output operations as streams, where data can be written to or read from.

The constructor of the class opens the file into a provided stream, while the destructor closes it:

```cpp
class file_handle {
  public:
    file_handle(ofstream& stream, const char* filepath) : _stream(stream) {
      _stream.open(filepath);
    }
    ~file_handle {
      _stream.close();
    }
  private:
    ofstream& _stream;
};
```

To open the file, it is sufficient to provide the file path to the `file_handle` class. Then, for the entire lifetime of the `file_handle` object, the file will not be closed. Once the object reaches the end of the scope, the file is closed:

```cpp
ofstream stream;
{
  file_handle myfile(stream, "Some path"); // file is opened
  do_something_with_file(stream);
}                                          // file is closed here 
```

This is used instead of the following code:

```cpp
ofstream stream;
{
  stream.open("Some path");    // file is opened
  do_something_with_file(stream);
  stream.close();              // file is closed here
}
```

Even though the benefit provided by applying the RAII idiom seems to be just to reduce code, the real improvement is having safer code. It is common for a programmer to write a function that correctly opens a file but never closes it or allocates memory that never gets destroyed.

RAII makes sure that these operations cannot be forgotten, as it automatically handles them.

### Activity 9: Storing Multiple Coordinates of Different Positions on a Map

In the 2D map program, the user can save multiple positions on the map. We need to be able to store multiple coordinates to keep track of the positions saved by the user. To do so, we need a way to create an array that can store them:

1.  Using the RAII programming idiom, write a class that manages memory allocation and the deletion of an array of values . The class has an array of integers as member data, which will be used to store the values .
2.  The constructor takes the size of the array as a parameter.
3.  The constructor also takes care of allocating memory, which is used to store the coordinates.
4.  To allocate the memory use the function `allocate_memory` (number of elements) which returns a pointer to an array of Coordinates of the requested size. To release the memory, call `release_memory` (array) which takes an array of Coordinates and releases the memory.
5.  Finally, define a destructor and make sure to free the previously allocated array in its implementation:

    #### Note

    The solution for this activity can be found on page 290.

## Nested Class Declarations

Inside the scope of a class, we can declare more than just data members and member functions; we can declare a class inside another class. These classes are called **nested classes**.

Since a nested class declaration happens inside the *outer class*, it has access to all the declared names as if it were part of the outer class: it can access even **private declarations**.

On the other hand, a nested class is not associated with any instance, so it can only access *static members*.

To access a nested class, we can use the double colon (`::`), similar to accessing static members of the outer class. Let's examine the following example:

```cpp
// Declaration
class Coordinate {
...
  struct CoordinateDistance {
    float x = 0;
    float y = 0;
    static float walkingDistance(CoordinateDistance distance);
  }
};
// Create an instance of the nested class CoordinateDistance
Coordinate::CoordinateDistance distance;
/* Invoke the static method walkingDistance declared inside the nested class CoordinateDistance */
Coordinate::CoordinateDistance::walkingDistance(distance);
```

Nested classes are useful for two main reasons:

*   When implementing a class, we need an object that manages some of the logic of the class. In such cases, the nested class is usually **private**, and is not exposed through the **public interface** of the class. It is mostly used to ease the implementation of the class.
*   When designing the functionality of a class, we want to provide a different class, closely related to the original one, which provides part of that functionality. In that case, the class is accessible by the users of the class and is usually an important part of the interaction with the class.

Imagine a list – a sequence of objects. We would like the user to be able to iterate over the items contained in the list. To do so, we need to keep track of which items the user has already iterated over and which are remaining. This is typically done with an `List` class.

We will look at iterators more in detail in *Lesson 5*, *Standard Library Containers and Algorithms*.

## Friend Specifier

As we have already seen, private and protected members of a class are not accessible from within other functions and classes. A class can declare another function or class as a friend: this function or class will have access to the private and protected members of the class which declares the **friend relationship**.

The user has to specify the `friend` declaration within the body of the class.

### Friend Functions

Friend functions are non-member functions that are entitled to access the private and protected members of a class. The way to declare a function as a `friend` function is by adding its declaration within the class and preceding it by the `friend` keyword. Let's examine the following code:

```cpp
class class_name {
  type_1 member_1;
  type_2 member_2;
  public:
    friend void print(const class_name &obj);
};
friend void print(const class_name &obj){
  std::cout << obj.member_1 << " " << member_2 << std::endl;
}
```

In the previous example, the function declared outside of the class scope has the right to access the class data members because it is declared as a `friend` function.

### Friend Classes

Similarly, like a `friend` function, a class can also be made a friend of another class by using the `friend` keyword.

Declaring a class as a `friend` is like declaring all of its methods as friend functions.

#### Note

Friendship is not mutual. If a class is a friend of another, then the opposite is not automatically true.

The following code demonstrates the concept of how friendship is not mutual:

```cpp
class A {
  friend class B;
  int a = 0;
};
class B {
  friend class C;
  int b = 0;
};
class C {
  int c = 0;
  void access_a(const A& object) {
    object.a;
    // Error! A.a is private, and C is not a friend of A.
  }
};
```

Friendship is not transitive; so, in the previous example, class `C` is not a friend of class `A,` and the methods of class `C` cannot access the protected or private members of class `A`. Additionally, `A` cannot access B's private members, since `B` is a friend of `A`, but friendship is not mutual.

### Exercise 10: Creating a Program to Print the User's Height

Let's write a program that collects height input from the user in inches and, after performing a calculation, prints the height of the user in feet:

1.  First, let's add all the required header files to the program.
2.  Now, create the `Height` class with one `public` method, as illustrated:

    ```cpp
    class Height {
        double inches;
        public:
            Height(double value): inches(value) { }
            friend void print_feet(Height);
    };
    ```

3.  As you can see, in the previous code, we used a friend function named `print_feet`. Now, let's declare it:

    ```cpp
    void print_feet(Height h){
        std::cout << "Your height in inches is: " << h.inches<< std::endl;
        std::cout << "Your height in feet is: " << h.inches * 0.083 << std::endl;
    }
    ```

4.  Invoke the class in the `main` function, as shown here:

    ```cpp
    int main(){
        IHeight h(83);
        print_feet(h);
    }
    ```

    The output is as follows:

    ```cpp
    Your height in inches is: 83
    Your height in feet is: 6.889
    ```

### Activity 10: The AppleTree Class, which Creates an Apple Instance

Sometimes, we would like to prevent the creation of an object of a specific type except for a limited number of classes. This usually happens when the classes are strictly related.

Create an `Apple` class that does not provide a `public` constructor and an `AppleTree` class that is in charge of creating the former object.

Let's execute the following steps:

1.  First, we need to create a class with a `private` constructor. In this way, the object cannot be constructed, because the constructor is not publicly accessible:

    ```cpp
    class Apple 
    {
      private:
        Apple() {}
        // do nothing
    };
    ```

2.  The `AppleTree` class is defined and contains a method called `createFruit`, which is in charge of creating an `Apple` and returning it:

    ```cpp
    class AppleTree
    {
      public:
        Apple createApple(){
          Apple apple;
          return apple;
        }
    };
    ```

3.  If we compile this code, we will get an error. At this point, the `Apple` constructor is `private`, so the `AppleTree` class cannot access it. We need to declare the `AppleTree` class as a `friend` of `Apple` to allow `AppleTree` to access the private methods of `Apple`:

    ```cpp
    class Apple
    {
      friend class AppleTree;
      private:
        Apple() {}
        // do nothing
    }
    ```

4.  The `Apple` object can now be constructed using the following code:

    ```cpp
    AppleTree tree;
    Apple apple = tree.createFruit();
    ```

    #### Note

    The solution for this activity can be found on page 291.

## Copy Constructors and Assignment Operators

One special type of constructor is the `const` qualified.

The following code refers to a class with a user-defined copy constructor, which copies the data member of the other object into the current one:

```cpp
class class_name {
  public:
    class_name(const class_name& other) : member(other.member){}
  private:
    type member;
};
```

A copy constructor is declared *implicitly* by the compiler when the class definition does not explicitly declare a copy constructor and all the data members have a copy constructor. This implicit copy constructor performs a copy of the class members in the *same order* of initialization.

Let's look at an example:

```cpp
struct A {
  A() {}
  A(const A& a) {
    std::cout << "Copy construct A" << std::endl;
  }
};
struct B {
  B() {}
  B(const B& a) {
    std::cout << "Copy construct B" << std::endl;
  }
};
class C {
  A a;
  B b;
  // The copy constructor is implicitly generated
};
int main() {
  C first;
  C second(first);
  // Prints: "Copy construct A", "Copy construct B"
}
```

When `C` is copy constructed, the members are copied in order: first, `a` is copied and then `b` is copied. To copy `A` and `B`, the compiler calls the copy constructor defined in those classes.

#### Note

When a pointer is copied, we are not copying the object pointed to, but simply the address at which the object is located.

This means that when a class contains a `pointer` as a data member, the implicit copy constructor only copies the pointer and not the pointed object, so the copied object and the original one will share the object that's pointed to by the pointer. This is sometimes called a **shallow copy**.

### The copy Assignment Operator

An alternative way to copy an object is by using the **copy assignment operator**, which, contrary to the construct operator, is called when the object has been already initialized.

The assignment operator signature and implementation look quite similar to the copy constructor, with the only difference being that the former is an overload of the `=` operator and it generally returns a reference to `*this`, although it's not required.

Here's an example of the use of the copy assignment operator:

```cpp
class class_name {
  public:
    class_name& operator= (const class_name & other) {
      member = other.member;
    }
  private:
    type member;
};
```

Also, for the copy assignment operator, the compiler generates an *implicit* one when it is not explicitly declared. As for the copy constructor, the members are copied in the same order of initialization.

In the following example, the copy constructor and the copy assignment operator will output a sentence when they are called:

```cpp
class class_name {
  public:
    class_name(const class_name& other) : member(other.member){
      std::cout << "Copy constructor called!" << std::endl;
    }
    class_name& operator= (const class_name & other) {
      member = other.member;
      std::cout << "Copy assignment operator called!" << std::endl;
    }
  private:
    type member;
};
```

The following code shows two ways of copying an object. The former uses the copy constructor, while the latter uses the copy assignment operator. The two implementations will print a sentence when they are called:

```cpp
class_name obj;
class_name other_obj1(obj);
\\ prints "Copy constructor called!"
class_name other_obj2 = obj;
\\ prints "Copy assignment operator called!"
```

### The move-constructor and move-assignment Operator

Like copying, moving also allows you to set the data members of an object to be equal to those of another data member. The only difference with copying lies in the fact that the content is transferred from one object to another, removing it from the source.

The move-constructor and move-assignment are members that take a parameter of type `rvalue` reference to the `class` itself:

```cpp
class_name (class_name && other);
// move-constructor
class_name& operator= (class_name && other);
// move-assignment
```

#### Note

For clarity, we can briefly describe an `rvalue` reference (formed by placing an `&&` operator after the type of the function argument) as a value that does not have a memory address and does not persist beyond a single expression, for example, a **temporary object**.

A move constructor and a move assignment operator enable the resources owned by an `rvalue` object to be moved into an `lvalue` without copying.

When we move a construct or assign a source object to a destination object, we transfer the content of the source object into the destination object, but the source object needs to remain valid. To do so, when implementing such methods, it is fundamental to reset the data members of the source object to a valid value. This is necessary to prevent the destructor from freeing the resources (such as memory) of the class multiple times.

Let's assume that there is a `Resource` that can be acquired, released, reset, and checked if it's reset.

Here is an example of a `WrongMove` constructor:

```cpp
class WrongMove {
  public:
    WrongMove() : _resource(acquire_resource()) {}
    WrongMove(WrongMove&& other) {
      _resource = other._resource;
      // Wrong: we never reset other._resource
    }
    ~WrongMove() {
      if (not is_reset_resource(_resource)) {
        release_resource(_resource);
      }
    }
  private:
    Resource _resource;
}
```

The move-constructor of the `WrongMove` class will release the resource twice:

```cpp
{
  WrongMove first;
  // Acquires the resource
  {
  /* Call the move constructor: we copy the resource to second, but we are not resetting it in first */
    WrongMove second(std::move(first)); 
  }
  /* Second is destroyed: second._resource is released here. Since we copied the resource, now first._resource has been released as well. */
}
// First is destroyed: the same resource is released again! Error!
```

Instead, the move constructor should have reset the `_resource` member of other, so that the destructor would not call `release_resource` again:

```cpp
WrongMove(WrongMove&& other) {
  _resource = other._resource;
  other._resource = resetted_resource();
}
```

The move constructor and move assignment operator can be implicitly generated by the compiler if no user-defined ones are provided and there are no user-declared destructors, copy constructors, or copy or move assignment operators:

```cpp
struct MovableClass {
  MovableClass(MovableClass&& other) {
    std::cout << "Move construct" << std::endl;
  }
  MovableClass& operator=(MovableClass&& other) {
    std::cout << "Move assign" << std::endl;
  }
};
MovableClass first;
// Move construct
MovableClass second = std::move(first);
// Or: MovableClass second(std::move(first));
MovableClass third;
// Move assignment
second = std::move(third);
```

### Preventing Implicit Constructors and Assignment Operators

The compiler will implicitly generate the copy constructor, copy assignment, move constructor, and move assignment if our class respects all the required conditions.

For cases in which our class should not be copied or moved, we can prevent that.

To prevent the generation of implicit constructors and operators, we can write the declaration of the constructor or operator and add `= delete`; at the end of the declaration.

Let's examine the following example:

```cpp
class Rectangle {
  int length;
  int width;
  // Prevent generating the implicit move constructor
  Rectangle(Rectangle&& other) = delete;
  // Prevent generating the implicit move assignment
  Rectangle& operator=(Rectangle&& other) = delete;
};
```

## Operator Overloading

C++ classes represent user-defined types. So, the need arises to be able to operate with these types in a different way. Some operator functions may have a different meaning when operating on different types. **Operator overloading** lets you define the meaning of an operator when applied to a class type object.

For example, the `+` operator applied to numerical types is different than when it is applied to the following `Point` class, which is constituted of coordinates. The language cannot specify what the `+` operator should do for user-defined types such as `Point`, as it is not in control of such types and does not know what the expected behavior is. Because of that, the language does not define the operators for user-defined types.

However, C++ allows the user to specify the behavior of most operators for user-defined types, including classes.

Here is an example of the `+` operator, defined for the `Point` class:

```cpp
class Point
{
  Point operator+(const Point &other) 
  {
    Point new_point;
    new_point.x = x + other.x; 
    new_point.y = y + other.y;
    return new_point;
  }
  private:
    int x;
    int y;
}
```

Here is a list of all the operators that can and cannot be overloaded:

*   The following are the operators that can be overloaded:

![Figure 3.4: Operators that can be overloaded](img/C11557_03_04.jpg)

###### Figure 3.4: Operators that can be overloaded

*   The following are the operators that cannot be overloaded:

![Figure 3.5: Operators that cannot be overloaded](img/C11557_03_05.jpg)

###### Figure 3.5: Operators that cannot be overloaded

Operators that expect two operands are called `+`, `-`, `*`, and `/`.

A method overloading a binary operator needs to accept a single parameter. When the compiler encounters the use of the operator, it will call the method on the variable on the left-hand side of the operator, while the variable on the right-hand side will be passed as parameter to the method.

We saw in the previous example that `Point` defines the `+` operator, which takes a parameter. When using the addition operation on a `Point`, the code would look like this:

```cpp
Point first;
Point second;
Point sum = first + second;
```

The last line from the code example is equivalent to writing the following:

```cpp
Point sum = first.operator+(second);
```

The compiler automatically rewrites the first expression to the second one.

Operators that expect only one operand are called `--`, `++`, and `!`.

A method overloading a unary operator must not accept any parameters. When the compiler encounters the use of the operator, it will call the method on the variable to which the operator is assigned.

As an example, let's say we are given an object that's defined as follows:

```cpp
class ClassOverloadingNotOperator {
  public:
    bool condition = false;

    ClassOverloadingNotOperator& operator!() {
      condition = !condition;
    }
};
```

We would write the following:

```cpp
ClassOverloadingNotOperator object;
!object;
```

The code is therefore rewritten as follows:

```cpp
ClassOverloadingNotOperator object;
object.operator!();
```

#### Note

Operator overloading is possible in two ways: either as a member function or as a non-member function. The two end up producing the same effect.

### Activity 11: Ordering Point Objects

In the 2D map application, we want to be able to display the locations that have been saved by the user in order: from South-West to North-East. To be able to show the locations in order, we need to be able to sort the location points representing the locations in such an order.

Remember that the `x` coordinate represents the location along the West-East axis and the `y` coordinate represents the location along the North-South axis.

In a real-life scenario, to compare two points, we need to compare their `x` and `y` coordinates. To do so in code, we need to overload the `<` operator for the `Point` class. This new function we're defining returns a `bool`, either `true` or `false`, according to the order of `p_1` and `p_2`.

The `p_1` point comes before than `p_2` in the order if the `x` coordinate of `p_1` is less than the `x` coordinate of `p_2`. If they are equal, then we need to compare their `y` coordinates.

Let's perform the following steps:

1.  We need to add an overload for the `<` operator to the `Point` class that we previously defined, which takes another object of type `Point` as an argument and returns a `bool` indicating whether the object is less than the one provided as a parameter, using the previous definition for how to compare two points:
2.  At this point, we are able to compare two `Point` objects:
3.  Since, in our example, `p_1.x` is initialized to `1` and `p_2.x` to `2`, the result of the comparison will be `true`, which indicates that `p_1` comes earlier than `p_2` in the order.

    #### Note

    The solution for this activity can be found on page 293.

## Introducing Functors

A `operator()` function is also known as the **function call operator**.

The syntax that's used to define a `functor` is as follows:

```cpp
class class_name {
  public:
    type operator()(type arg) {} 
};
```

The function call operator has a return type and takes any number of arguments of any type. To invoke the call operator of an object, we can write the name of the object, followed by parentheses containing the arguments to pass to the operator. You can imagine that an object that provides a call operator can be used in the same way as you would use a function. Here's an example of a `functor`:

```cpp
class_name obj;
type t;
/* obj is an instance of a class with the call operator: it can be used as if it was a function */
obj(t);
```

They are particularly useful in places where you can pass a function object to an algorithmic template that accepts an object with `operator()` defined. This exploits code reusability and testability. We will see more on this in chapter 5 when we talk about **lambda**.

The following is a simple example of a `functor` that prints a string before appending a new line at the end of it:

```cpp
class logger{
  public:
    void operator()(const std::string &s) {
       std::cout << s << std::endl;
    }
};
logger log;
log ("Hello world!");
log("Keep learning C++");
```

### Activity 12: Implementing Functors

Write a function object that takes a number when constructed and defines an operator call that takes another number and returns the sum of the two.

Let's perform the following steps to achieve the desired output:

1.  Define a class by the name of `AddX`, constituted by a `private` data member of type `int`, and a constructor that is used to initialize it.
2.  Extend it with the call operator, `operator()`, which takes an `int` as a parameter and returns an `int`. The implementation in the function body should return the addition of the previously defined `x` value and the parameter of the function named `y`.
3.  Instantiate an object of the class we just defined and invoke the calling operator:

    ```cpp
    class AddX {
       public:
          explicit AddX(int v) : value(v) {}
          int operator()(int other_value) {
       Indent it to the right, same as above
    }
       private:
         int value;
    };
    AddX add_five(5);
    std::cout << add_five(4) << std::endl; // prints 9
    ```

    #### Note

    The solution for this activity can be found on page 294.

## Summary

In this chapter, we saw how the concept of classes can be used in C++. We started by delineating the advantages of using classes, describing how they can help us to create powerful abstractions.

We outlined the access modifiers a class can use to control who has access to class fields and methods.

We continued by exploring the conceptual differences between a class and its instances, along with the implications this has when implementing static fields and static methods.

We saw how constructors are used to initialize classes and their members, while destructors are used to clean up the resources that are managed by a class.

We then explored how constructors and destructors can be combined to implement the fundamental paradigm C++ is famous for: RAII. We showed how RAII makes it easy to create classes that handle resources and make programs safer and easier to work with.

Finally, we introduced the concept of operator overloading and how it can be used to create classes that are as easy to use as built-in types.

In the next chapter, we'll focus on templates. We'll primarily look at how to implement template functions and classes, and write code that works for multiple types.