# *Chapter 6*

# Object-Oriented Programming

## Lesson Objectives

By the end of this chapter, you will be able to:

*   Compose classes that inherit properties from other classes
*   Implement polymorphism in C++ programs
*   Implement interfaces
*   Use best practices to manage dynamic memory

In this chapter, you will learn how to use the advanced features of C++ to create dynamic programs.

## Introduction

In earlier chapters, we learned about templates that are used to create functions and classes that work with arbitrary types. This avoids duplication of work. However, using templates is not applicable in all cases, or may not be the best approach. The limitation of templates is that their types need to be known when the code is compiled.

In real-world cases, this is not always possible. A typical example would be a program that determines what logging infrastructure to use depending on the value of a configuration file.

Consider the following problems:

*   While developing the application and executing tests, the application would use a logger that prints detailed information.
*   On the other hand, when the application is deployed to the PCs of its users, the application would use a logger that prints **error summaries** and notifies the developers if there are any errors.

We can solve these problems using the concept of inheritance in C++.

## Inheritance

Inheritance allows the combination of one or more classes. Let's look at an example of inheritance:

```cpp
class Vehicle {
  public:
    TankLevel getTankLevel() const;
    void turnOn();
};
class Car : public Vehicle {
  public:
    bool isTrunkOpen();
};
```

In this example, the `Car` class inherits from the `Vehicle` class, or, we can say `Car` derives from `Vehicle`. In C++ terminology, `Vehicle` is the *base* class, and `Car` is the *derived* class.

When defining a class, we can specify the classes it derives from by appending `:`, followed by one or more classes, separated by a comma:

```cpp
class Car : public Vehicle, public Transport {
}
```

When specifying the list of classes to derive from, we can also specify the visibility of the inheritance – `private`, `protected`, or `public`.

The visibility modifier specifies who can know about the inheritance relationship between the classes.

The methods of the base class can be accessed as methods of the derived class based on the following rules:

```cpp
Car car;
car.turnOn();
```

When the inheritance is `public`, the code external to the class knows that `Car` derives from `Vehicle`. All the public methods of the base class are accessible as *public* method of the derived class by the code in the program. The protected methods of the base class can be accessed as *protected* by the methods of the derived class. When inheritance is `protected`, all the public and protected members are accessible as *protected* by the derived class. Only the derived class and classes that derive from it know about inheritance; external code sees the two classes as unrelated.

Finally, when deriving with a `private` modifier, all the `public` and `protected` methods and fields of the base class are accessible by the derived class as `private`.

The private methods and fields of a class are *never accessible* outside of that class.

Accessing the fields of the base class follows the same rules.

Let's see a summary:

![](img/C11557_06_01.jpg)

###### Figure 6.1: Base class methods and the access level they provide

Inheritance creates a hierarchy of derived and base classes.

The `Orange` class can be derived from a `Citrus` class, which is in turn derived from a `Fruit` class. Here is how it can be written:

```cpp
class Fruit {
};
class Citrus: public Fruit {
};
class Orange: public Citrus {
};
```

The class `Citrus` can access the public and protected methods of class `Fruit`, whereas class `Orange` will be able to access both `Citrus`' and `Fruit`'s public and protected methods (`Fruit`'s public methods are accessible through `Citrus`).

### Exercise 20: Creating a Program to Illustrate Inheritance in C++

Let's perform the following exercise to create a derived class that inherits from multiple base classes:

1.  Add the header file at the start of the program:

    ```cpp
    #include <iostream>
    ```

2.  Add the first base class, named `Vehicle`:

    ```cpp
    // first base class 
    class Vehicle { 
      public: 
        int getTankCapacity(){
          const int tankLiters = 10;
          std::cout << "The current tank capacity for your car is " << tankLiters << " Liters."<<std::endl;
          return tankLiters;
        }
    };
    ```

3.  Now add the second base class, named `CollectorItem`:

    ```cpp
    // second base class 
    class CollectorItem { 
      public: 
        float getValue() {
          return 100;
        }
    }; 
    ```

4.  Add the derived class, named `Ferrari250GT`, as illustrated here:

    ```cpp
    // Subclass derived from two base classes
    class Ferrari250GT: protected Vehicle, public CollectorItem { 
      public:
        Ferrari250GT() {
          std::cout << "Thank you for buying the Ferrari 250 GT with tank capacity " << getTankCapacity() << std::endl;
          return 0;
        }
    }; 
    ```

5.  Now, in the `main` function, instantiate the `Ferrari250GT` class and call the `getValue()` method:

    ```cpp
    int main()
    {
      Ferrari250GT ferrari;
      std::cout << "The value of the Ferrari is " << ferrari.getValue() << std::endl;
      /* Cannot call ferrari.getTankCapacity() because Ferrari250GT inherits from Vehicle with the protected specifier */
      return 0;
    }
    ```

    The output will be as follows:

    ```cpp
    Output:
    The current tank capacity for your car is 10 Liters.
    Thank you for buying the Ferrari 250 GT with tank capacity 10
    The value of the Ferrari is 100
    ```

The specifier is not mandatory. If it is omitted, it defaults to *public for structs* and to *private for classes*.

#### Note

If you use inheritance to group together some functionality when implementing a class, it is often correct to use **private inheritance**, as that is a detail of how you are implementing the class, and it is not part the **public interface** of the class. If, instead, you want to write a derived class that can be used in place of the base class, use public inheritance.

When inheriting from a class, the base class gets **embedded** into the derived class. This means that all the data of the base class also becomes part of the derived class in its **memory representation**:

![Figure 6.2: Representation of the derived class and the base class](img/C11557_06_02.jpg)

###### Figure 6.2: Representation of the derived class and the base class

A question might come up at this point – we are embedding the base class inside the derived class. This means that we need to initialize the base class when we initialize the derived class, otherwise, part of the class would be left uninitialized. When do we initialize the base class?

When writing the constructor of the derived class, the compiler will implicitly call the default constructor of the base class before any initialization takes place.

If the base class does not have a default constructor but has a constructor that accepts parameters, then the derived class constructor can explicitly call it in the initialization list. Otherwise, there will be an error.

In a similar way to how the compiler calls the constructor of the base class when the derived class is constructed, the compiler takes care of always calling the destructor of the base class after the destructor of the derived class has run:

```cpp
class A {
  public:
    A(const std::string& name);
};
class B: public A {
  public:
    B(int number) : A("A's name"), d_number(number) {}
  private:
    int d_number;
};

}
```

When `B`'s constructor is called, the `A` needs to be initialized. Since `A` doesn't have a default constructor, the compiler cannot initialize it for us: we have to call `A`'s constructor explicitly.

The **copy constructor** and the **assignment operator** generated by the compiler take care of calling the constructor and operator of the base class.

When, instead, we write our implementation of the copy constructor and the assignment operators, we need to take care of calling the copy constructor and assignment operator.

#### Note

In many compilers, you can enable additional warnings that notify you if you forget to add the calls to the base constructor.

It is important to understand that inheritance needs to model an `A` to inherit from another class, `B`, you are saying that `A` `B`.

To understand this, a vehicle is a good example: a car is a vehicle, a bus is a vehicle, and a truck is also a vehicle. A bad example would be for a car to inherit from an engine. While the engine might have similar functionality to a car, such as a `start` method, it is wrong to say that a car is an engine. The relationship, in this case, is **has a**: the car has an engine; this relationship represents composition.

#### Note

Using an **is a** test to understand whether a relationship can use inheritance can fail in some cases: for example, a **square** inheriting from a **rectangle**. When the width of the rectangle is doubled, the area of the rectangle doubles, but the area of the square quadruples. This means that code that expects to interact with rectangles might get surprising results when using a square, even if the square, mathematically, is a rectangle.

A more general rule is to use the `A` class inherits from `B`, we could replace the `A` class anywhere the `B` class is used, and the code would still behave correctly.

Up to now, we have seen examples of single inheritance: a derived class has a single base class. C++ supports multiple inheritance: a class can derive from multiple classes. Let's look at an example:

```cpp
struct A {
};
struct B {
};
struct C : A, B {
};
```

In this example, the `C` struct derives both from `A` and from `B`.

The rules on how inheritance works are the same for single and multiple inheritance: the methods of all the derived classes are visible based on the visibility access specified, and we need to make sure to call the appropriate constructors and assign an operator for all of the base classes.

#### Note

It is usually best to have a shallow inheritance hierarchy: there should not be many levels of derived classes.

When using a multi-level inheritance hierarchy or multiple inheritance, it's more likely that you'll encounter some problems, such as **ambiguous calls**.

A call is ambiguous when the compiler cannot clearly understand which method to call. Let's explore the following example:

```cpp
struct A {
  void foo() {}
};
struct B {
  void foo() {}
};
struct C: A, B {
  void bar() { foo(); }
};
```

In this example, it is not clear which `foo()` to call, `A`'s or `B`'s. We can disambiguate that by prepending the name of the class followed by two columns: `A::foo()`.

### Exercise 21: Using Multiple Inheritance to Create a "Welcome to the Community" Message Application

Let's use multiple inheritance to create an application to print a "welcome to the community" message:

1.  First, add the required header files in the program, as illustrated:

    ```cpp
    #include <iostream>
    ```

2.  Now, add the required classes, `DataScienceDev` and `FutureCppDev`, with the required print statement:

    ```cpp
    class DataScienceDev {
    public:
        DataScienceDev(){
            std::cout << "Welcome to the Data Science Developer Community." << std::endl;
        }
    };
    class FutureCppDev {
    public:
        FutureCppDev(){
            std::cout << "Welcome to the C++ Developer Community." << std::endl;
          }
    };
    ```

3.  Now, add the `Student` class as illustrated here:

    ```cpp
    class Student : public DataScienceDev, public FutureCppDev {
        public:
        Student(){
            std::cout << "Student is a Data Developer and C++ Developer." << std::endl;
        }
    };
    ```

4.  Now, invoke the `Student` class in the `main` function:

    ```cpp
    int main(){
        Student S1;
        return 0;
    }
    ```

    The output will be as follows:

    ```cpp
    Welcome to the Data Science Developer Community.
    Welcome to the C++ Developer Community.
    Student is a Data Developer and C++ Developer.
    ```

### Activity 23: Creating Game Characters

We want to write a new game, and in that game, create two types of characters – a hero and enemies. Enemies can swing their swords, and the hero can cast a spell.

Here is how you can achieve the task:

1.  Create a `Character` class that has a public method, `moveTo`, that prints `Moved to position`.
2.  Create a `Position` struct:

    ```cpp
    struct Position {
        std::string positionIdentifier;
    };
    ```

3.  Create two classes, `Hero` and `Enemy`, that are derived from the Character class:

    ```cpp
    class Hero : public Character {
    };
    class Enemy : public Character {
    };
    ```

4.  Create a `Spell` class with the constructor that takes the name of the spell:

    ```cpp
    class Spell {
    public:
        Spell(std::string name) : d_name(name) {}
        std::string name() const {
            return d_name;
        }
    private:
        std::string d_name;
    }
    ```

5.  The `Hero` class should have a `public` method to cast a spell. Use the value from the `Spell` class.
6.  The `Enemy` class should have a `public` method to swing a sword, which prints `Swinging sword`.
7.  Implement the main method, which calls these methods in various classes:

    ```cpp
    int main()
    {
        Position position{"Enemy castle"};
        Hero hero;
        Enemy enemy;
    }
    ```

    The output will be as follows:

    ```cpp
    Moved to position Enemy castle
    Moved to position Enemy castle
    Casting spell fireball
    Swinging sword
    ```

    #### Note

    The solution for this activity can be found on page 309.

## Polymorphism

In the previous section, we mentioned that inheritance is a solution that allows you to change the behavior of code while a program is running. This is because inheritance enables polymorphism in C++.

**Polymorphism** means *many forms* and represents the ability of objects to behave in different ways.

We mentioned earlier that templates are a way to write code that works with many different types at compilation time and, depending on the types used to instantiate the template, the behavior will change.

This kind of pattern is called **static polymorphism** – static because it is known during compilation time. C++ also supports **dynamic polymorphism** – having the behavior of methods change while the program is running. This is powerful because we can react to information we obtain only after we have compiled our program, such as user input, values in configurations, or the kind of hardware the code is running on. This is possible thanks to two features – **dynamic binding** and **dynamic dispatch**.

### Dynamic Binding

**Dynamic binding** is the ability for a reference or a pointer of a base type to point to an object of a derived type at runtime. Let's explore the following example:

```cpp
struct A {
};
struct B: A{
};
struct C: A {
};
//We can write
B b;
C c;
A& ref1 = b;
A& ref2 = c;
A* ptr = nullptr;
if (runtime_condition()) {
  ptr = &b;
} else {
  ptr = &c;
}
```

#### Note

To allow dynamic binding, the code must *know* that the derived class derives from the base class.

If the inheritance's visibility is `private`, then only code inside the derived class will be able to bind the object to a *pointer* or *reference* of the base class.

If the inheritance is `protected`, then the derived class and every class deriving from it will be able to perform dynamic binding. Finally, if the inheritance is `public`, the dynamic binding will always be *allowed*.

This creates the distinction between the `static` type and the `dynamic` (or run-time) type. The static type is the type we can see in the source code. In this case, we can see that `ref1` has a static type of a reference to the `A` struct.

The dynamic type is the real type of the object: the type that has been constructed in the object's memory location at runtime. For example, the static type of both `ref1` and `ref2` is a reference to the `A` struct, but the `ref1` dynamic type is `B`, since `ref1` refers to a memory location in which an object of type `B` has been created, and the `ref2` dynamic type is `C` for the same reason.

As said, the dynamic type can change at runtime. While the static type of a variable is always the same, its dynamic type can change: `ptr` has a static type, which is a pointer to `A`, but its dynamic type could change during the execution of the program:

```cpp
A* ptr = &b; // ptr dynamic type is B
ptr = &c; // ptr dynamic type is now C
```

It is important to understand that only references and pointers can be assigned values from a derived class safely. If we were to assign an object to a value type, we would get a surprising result – the object would get sliced.

We said earlier that a base class is **embedded** inside a derived class. Say, for example, we were to try and assign to a value, like so:

```cpp
B b;
A a = b;
```

The code would compile, but only the embedded part of `A` inside of `B` would be copied – when we declare a variable of type `A`, the compiler dedicates an area of the memory big enough to contain an object of type `A`, so there cannot be enough space for `B`. When this happens, we say that we sliced the object, as we took only a part of the object when assigning or copying.

#### Note

It is not the intended behavior to slice the object. Be mindful of this interaction and try to avoid it.

This behavior happens because C++ uses *static dispatch* by default for function and method calls: when the compiler sees a method call, it will check the static type of the variable on which the method is called, and it will execute the `A` is called, and it only copies the part of `A` inside `B`, ignoring the remaining fields.

As said before, C++ supports dynamic dispatch. This is done by marking a method with a special keyword: **virtual**.

If a method is marked with the `virtual` keyword, when the method is called on a *reference* or a *pointer*, the compiler will execute the implementation of the dynamic type instead of the static type.

These two features enable *polymorphism* – we can write a function that accepts a reference to a base class, call methods on this base class, and the methods of the derived classes will be executed:

```cpp
void safeTurnOn(Vehicle& vehicle) {
  if (vehicle.getFuelInTank() > 0.1 && vehicle.batteryHasEnergy()) {
    vehicle.turnOn();
  }
}
```

We can then call the function with many different types of vehicles, and the appropriate methods will be executed:

```cpp
Car myCar;
Truck truck;
safeTurnOn(myCar);
safeTurnOn(truck);
```

A typical pattern is to create an interface that only specifies the methods that are required for some functionality.

Classes that need to be used with such functionality must derive the interface and implement all the required methods.

## Virtual Methods

We've learned the advantages of dynamic dispatch in C++ and how it can enable us to execute the methods of a derived class by calling a method on a reference or pointer to a base class.

In this section, we will take an in-depth look at how to tell the compiler to perform dynamic dispatch on a method. The way to specify that we want to use dynamic dispatch for a method is to use the `virtual` keyword.

The `virtual` keyword is used in front of a method when declaring it:

```cpp
class Vehicle {
  public:
    virtual void turnOn();
};
```

We need to remember that the compiler decides how to perform method dispatch based on the static type of the variable that is used when calling the method.

This means that we need to apply the virtual keyword to the type we are using in the code. Let's examine the following exercise to explore the virtual keyword.

### Exercise 22: Exploring the Virtual Method

Let's create a program using the concept of inheritance using the virtual keyword:

1.  First, make sure to add the required header file and namespace to compile the program.
2.  Now, add the `Vehicle` class as illustrated:

    ```cpp
    class Vehicle {
      public:
        void turnOn() {
          std::cout << "Vehicle: turn on" << std::endl;
        }
    };
    ```

3.  In the `Car` class, add the `virtual` keyword as illustrated :

    ```cpp
    class Car : public Vehicle {
      public:
        virtual void turnOn()  {
          std::cout << "Car: turn on" << std::endl;
        }
    };
    void myTurnOn(Vehicle& vehicle) {
      std::cout << "Calling turnOn() on the vehicle reference" << std::endl;
      vehicle.turnOn();
    }
    ```

4.  Now, in the main function, invoke the `Car` class and pass the `car` object in the `myTurnOn()` function:

    ```cpp
    int main() {
      Car car;
      myTurnOn(car);
    }
    ```

    The output will be as follows:

    ```cpp
    Calling turnOn() on the vehicle reference
    Vehicle: turn on
    ```

Here, the call will not be dynamically dispatched, and the call to the implementation of `Vehicle::turnOn()` will be executed. The reason is that the static type of the variable is `Vehicle`, and we did not mark the method as `virtual`, so the compiler uses static dispatch.

The fact that we wrote a `Car` class that declares the method virtual is not important, since the compiler only sees the `Vehicle` class being used in `myTurnOn()`. When a method is declared `virtual`, we can override it in a derived class.

To override a method, we need to declare it with the same signature as the parent class: the same return type, name, parameters (including `const`-ness and `ref`-ness), `const` qualifier, and the other attributes.

If the signature does not match, we will create an overload for the function. The overload will be callable from the derived class, but it will never be executed with a dynamic dispatch from a base class, for example:

```cpp
struct Base {
  virtual void foo(int) = 0;
};
struct Derived: Base {
  /* This is an override: we are redefining a virtual method of the base class, using the same signature. */
  void foo(int) { }
  /* This is an overload: we are defining a method with the same name of a method of the base class, but the signature is different. The rules regarding virtual do not apply between Base::foo(int) and Derived:foo(float). */
  void foo(float) {}
};
```

When a class overrides a virtual method of the base class, the method of the *most derived class* will be executed when the method is called on a base class. This is `true` even if the method is called from inside the base class, for example:

```cpp
struct A {
  virtual void foo() {
    std::cout << "A's foo" << std::endl;
  }
};
struct B: A {
  virtual void foo() override {
    std::cout << "B's foo" << std::endl;
  }
};
struct C: B {
  virtual void foo() override {
    std::cout << "C's foo" << std::endl;
  }
};
int main() {
  B b;
  C c;
  A* a = &b;
  a->foo();  // B::foo() is executed
  a = &c;
  a->foo();
  /* C::foo() is executed, because it's the most derived Class overriding foo(). */
}
```

We can see a new keyword in the preceding example: the `override` keyword.

C++11 introduced this keyword to enable us to specify that we are overriding a method explicitly. This allows the compiler to give us an error message if we use the `override` keyword, but the signature does not match any base class' virtual method.

#### Note

Always use the `override` keyword when you are overriding a method. It is easy to change the signature of the base class and forget to update all the locations where we overrode the method. If we do not update them, they will become a new overload instead of an override!

In the example, we also used the `virtual` keyword for each function. This is not necessary, since a virtual method on a base class makes every method with the same signature in the derived classes virtual as well.

It is good to be explicit `virtual` keyword, but if we are already using the `override` keyword, it might be redundant – in these cases, the best way is to follow the coding standard of the project you are working on.

The `virtual` keyword can be applied to any method. Since the constructor is not a method, the constructor cannot be marked as virtual. Additionally, dynamic dispatch is disabled inside constructors and destructors.

The reason is that when constructing a hierarchy of derived classes, the constructor of the base class is executed before the constructor of the derived class. This means that if we were to call the virtual method on the derived class when constructing the base class, the derived class would not be initialized yet.

Similarly, when calling the destructor, the destructors of the whole hierarchy are executed in reverse order; first the derived and then the base class. Calling a `virtual` method in the destructor would call the method on a derived class that has already been destructed, which is an error.

While the constructor cannot be marked as virtual, the destructor can. If a class defines a virtual method, then it should also declare a virtual destructor.

Declaring a destructor virtual is extremely important when classes are created on dynamic memory, or the heap. We are going to see later in this chapter how to manage dynamic memory with classes, but for now, it is important to know that if a destructor is not declared virtual, then an object might be only partially destructed.

#### Note

If a method is marked virtual, then the destructor should also be marked virtual.

### Activity 24: Calculating Employee Salaries

We are writing a system to compute the paycheques for the employees of a company. Each employee has a base salary plus a bonus.

For employee who are not managers, the bonus is computed from the performance of the department: they get 10% of the base salary if the department reached its goal.

The company also has managers, for whom the bonus is computed in a different way: they get 20% of the base salary if the department reached its goal, plus 1% of the difference between the achieved result of the department and the expected one.

We want to create a function that takes an employee and computes their total salary, summing the base salary and the bonus, regardless of whether they are a manager or not.

Perform the following steps:

1.  The Department class accepts the expected earning and the effective earning when constructed, and stores them in two fields:

    ```cpp
    class Department {
    public:
        Department(int expectedEarning, int effectiveEarning)
        : d_expectedEarning(expectedEarning), d_effectiveEarning(effectiveEarning)
        {}
        bool hasReachedTarget() const {return d_effectiveEarning >= d_expectedEarning;}
        int expectedEarning() const {return d_expectedEarning;}
        int effectiveEarning() const {return d_effectiveEarning;}
    private:
        int d_expectedEarning;
        int d_effectiveEarning;
    };
    ```

2.  Define an `Employee` class with two `virtual` functions, `getBaseSalary()`, and `getBonus()`. Within it, implement the logic for employee bonus calculation if the department goal is met:

    ```cpp
    class Employee {
    public:
        virtual int getBaseSalary() const { return 100; }
        virtual int getBonus(const Department& dep) const {
            if (dep.hasReachedTarget()) {
                return int(0.1 * getBaseSalary());
            }
            return 0;
        }
    };
    ```

3.  Create another function that provides the total compensation:

    ```cpp
        int getTotalComp(const Department& dep) {
                return getBaseSalary() + getBonus(dep);
        }
    ```

4.  Create a `Manager` class that derives from `Employee`. Again, create the same virtual functions, `getBaseSalary()` and `getBonus()`. Within it, implement the logic for a `Manager` bonus calculation if the department goal is met:

    ```cpp
    class Manager : public Employee {
    public:
        virtual int getBaseSalary() const override { return 150; }
        virtual int getBonus(const Department& dep) const override {
            if (dep.hasReachedTarget()) {
                int additionalDeparmentEarnings = dep.effectiveEarning() - dep.expectedEarning();
                return int(0.2 * getBaseSalary() + 0.01 * additionalDeparmentEarnings);
            }
            return 0;
        }
    };
    ```

5.  Implement the `main` program, and run the program:

    The output will be as follows:

    ```cpp
    Employee: 110\. Manager: 181
    ```

    #### Note

    The solution for this activity can be found on page 311.

## Interfaces in C++

In the previous section, we saw how to define a method that is virtual, and how the compiler will do dynamic dispatch when calling it.

We have also talked about interfaces throughout the chapter, but we never specified what an interface is.

An interface is a way for the code to specify a contract that the caller needs to provide to be able to call some functionality. We looked at an informal definition when talking about the templates and the requirements they impose on the types used with them.

Functions and methods which accepts parameters as interface are a way of saying: in order to perform my actions, I need these functionalities; it's up to you to provide them.

To specify an interface in C++, we can use an **Abstract Base Class** (**ABC**).

Let's dive into the name; the class is:

*   **Abstract**: This means that it cannot be instantiated
*   **Base**: This means it is designed to be derived from

Any class that defines a pure virtual method is `abstract`. A pure virtual method is a virtual method that ends with `= 0`, for example:

```cpp
class Vehicle {
  public:
    virtual void turnOn() = 0;
};
```

A pure virtual method is a method that does not have to be defined. Nowhere in the previous code have we specified the implementation of `Vehicle::turnOn()`. Because of this, the `Vehicle` class cannot be instantiated, as we do not have any code to call for its pure virtual methods.

We can instead derive from the class and override the pure virtual method. If a class derives from an abstract base class, it can be either of the following:

*   Another abstract base class if it declares an additional pure virtual method, or if it does not override all the pure virtual methods of the base class
*   A regular class if it overrides all the pure virtual methods of the base class

Let's continue with the previous example:

```cpp
class GasolineVehicle: public Vehicle {
  public:
    virtual void fillTank() = 0;
};
class Car : public GasolineVehicle {
  virtual void turnOn() override {}
  virtual void fillTank() override {}
};
```

In this example, `Vehicle` is an abstract base class and `GasolineVehicle` is too, since it does not override all the pure virtual methods of `Vehicle`. It also defines an additional virtual method, which the `Car` class overrides together with the `Vehicle::turnOn()` method. This makes `Car` the only concrete class, a class that can be instantiated.

The same concept applies when a class is deriving from multiple abstract base classes: all the pure virtual methods of all the classes that need to be overridden in order to make the class concrete and thus instantiable.

While abstract base classes cannot be instantiated, we can define references and pointers to them.

#### Note

If you try to instantiate an abstract base class, the compiler will give an error specifying which methods are still pure virtual, thus making the class abstract.

Functions and methods that require specific methods can accept references and pointers to abstract base classes, and instances of concrete classes that derive from them can be bound to such references.

#### Note

It is good practice for the consumer of the interface to define the interface.

A function, method, or class that requires some functionality to perform its actions should define the interface. Classes that should be used with such entities should implement the interface.

Since C++ does not provide a specialized keyword for defining interfaces and interfaces are simply abstract base classes, there are some guidelines that it's best practice to follow when designing an interface in C++:

*   An abstract base class should *not* have any data members or fields.

    The reason for this is that an interface specifies behavior, which should be independent of the data representation. It derives that abstract base classes should only have a default constructor.

*   An abstract base class should always define a `virtual ~Interface() = default`. We are going to see why it is important for the destructor to be virtual later.
*   All the methods of an abstract base class should be pure virtual.

    The interface represents an expected functionality that needs to be implemented; a method which is not pure is an implementation. The implementation should be separate from the interface.

*   All of the methods of an abstract base class should be `public`.

    Similar to the previous point, we are defining a set of methods that we expect to call. We should not limit which classes can call the method only to classes deriving from the interface.

*   All the methods of an abstract base class should be regarding a single functionality.

    If our code requires multiple functionalities, separate interfaces can be created, and the class can derive from all of them. This allows us to compose interfaces more easily.

Consider disabling the copy and move constructors and assignment operators on the interface. Allowing the interface to be copied can cause the slicing problem we were describing before:

```cpp
Car redCar;
Car blueCar;
Vehicle& redVehicle = redCar;
Vehicle& redVehicle = blueCar;
redVehicle = blueVehicle;
// Problem: object slicing!
```

With the last assignment, we only copied the `Vehicle` part, since the copy constructor of the `Vehicle` class has been called. The copy constructor is not virtual, so the implementation in `Vehicle` is called, and since it only knows about the data members of the `Vehicle` class (which should be none), the ones defined inside `Car` have not been copied! This results in problems that are very hard to identify.

A possible solution is to disable the interface copy and move construct and assign operator: `Interface(const Interface&) = delete`; and similar. This has the drawback of disabling the compiler from creating the copy constructor and assign operators of the derived classes.

An alternative is to declare copy/move constructor/assignment protected so that only derived classes can call them, and we don't risk assigning interfaces while using them.

### Activity 25: Retrieving User Information

We are writing an application to allow users to buy and sell items. When a user logs in, we need to retrieve several pieces of information to populate their profile, such as the URL for the profile picture and the full name.

Our service is running in many data centers around the world, to always be close to its customers. Because of that, sometimes we want to retrieve information for the user from a cache, but sometimes we want to retrieve it from our main database.

Perform the following:

1.  Let's write the code, which can be independent of where the data is coming from, so we create an abstract `UserProfileStorage` class to retrieve the `CustomerProfile` from `UserId`:

    ```cpp
    struct UserProfile {};
    struct UserId {};
    class UserProfileStorage {
      public:
        virtual UserProfile getUserProfile(const UserId& id) const = 0;

        virtual ~UserProfileStorage() = default;
      protected:
        UserProfileStorage() = default;
        UserProfileStorage(const UserProfileStorage&) = default;
        UserProfileStorage& operator=(const UserProfileStorage&) = default;
    };
    ```

2.  Now, write the `UserProfileCache` class, which inherits from `UserProfileStorage`:

    ```cpp
    class UserProfileCache : public UserProfileStorage {
    public:
        UserProfile getUserProfile(const UserId& id) const override {
            std::cout << "Getting the user profile from the cache" << std::endl;
            return UserProfile();
        }
    };
    void exampleOfUsage(const UserProfileStorage& storage) {
        UserId user;
        std::cout << "About to retrieve the user profile from the storage" << std::endl;
        UserProfile userProfile = storage.getUserProfile(user);
    }
    ```

3.  In the `main` function, instantiate the `UserProfileCache` class and the call `exampleOfUsage` function as illustrated:

    ```cpp
    int main()
    {
      UserProfileCache cache;
      exampleOfUsage (cache);
    }
    ```

The output is as follows:

```cpp
About to retrieve the user profile from the storage
Getting the user profile from the cache
```

#### Note

The solution for this activity can be found at page 312.

## Dynamic Memory

In this chapter, we have come across the term dynamic memory. Now let's understand in more detail what dynamic memory is, what problems it solves, and when to use it.

**Dynamic memory** is the part of the memory that the program can use to store objects, for which the program is responsible for maintaining the correct lifetime.

It is usually also called the **heap** and is often the alternative to the stack, which instead is handled automatically by the program. Dynamic memory can usually store much larger objects than the stack, which usually has a limit.

A program can interact with the operating system to get pieces of dynamic memory that it can use to store objects, and later it must take care to return such memory to the operating system.

Historically, developers would make sure they called the appropriate functions to get and return memory, but modern C++ automates most of this, so it is much easier to write correct programs nowadays.

In this section, we are going to show how and when it is recommended to use dynamic memory in a program.

Let's start with an example: we want to write a function that will create a logger. When we execute tests, we create a logger specifically for the test called `TestLogger`, and when we run our program for users, we want to use a different logger, called `ReleaseLogger`.

We can see a good fit for interfaces here – we can write a logger abstract base class that defines all the methods needed for logging and have `TestLogger` and `ReleaseLogger` derive from it.

All our code will then use a reference to the logger when logging.

How can we write such a function?

As we learned in *Chapter 2*, *Functions*, we cannot create the logger inside the function and then return a reference to it, since it would be an automatic variable and it would be destructed just after the return, leaving us with a dangling reference.

We cannot create the logger before calling the function and let the function initialize it either, since the types are different, and the function knows which type should be created.

We would need some storage that is valid until we need the logger, to put the logger in it.

Given only an interface, we cannot know the size of the classes implementing it, since multiple classes could implement it and they could have different sizes. This prevents us from reserving some space in memory and passing a pointer to such space to the function, so that it could store the logger in it.

Since classes can have different sizes, the storage not only needs to remain valid longer than the function, but it also needs to be variable. That is **dynamic memory**!

In C++, there are two keywords to interact with dynamic memory – **new** and **free**.

The `new` expression is used to create a new object in dynamic memory – it is composed by the `new` keyword, followed by the type of the object to create and the parameters to pass to the constructor, and returns a pointer to the requested type:

```cpp
Car* myCar = new myCar();
```

The `new` expression requests a piece of dynamic memory big enough to hold the object created and instantiates an object in that memory. It then returns a pointer to such an instance.

The program can now use the object pointed to by `myCar` until it decides to delete it. To delete a pointer, we can use the delete expression: it is composed by the `delete` keyword followed by a variable, which is a pointer:

```cpp
delete myCar;
```

The `delete` keyword calls the destructor of the object pointed to by the pointer provided to it, and then gives the memory we initially requested back to the operating system.

Deleting pointers to automatic variables lead to an error as follows:

```cpp
Car myCar; // automatic variable
delete &myCar; // This is an error and will likely crash the program
```

It is of absolute importance that, for each new expression, we call the `delete` expression only once, with the same returned pointer.

If we forget to call the `delete` function on an object returned by calling the `new` function, we will have two major problems:

*   The memory will not be returned to the operating system when we do not need it anymore. This is known as a **memory leak**. If this repeatedly happens during the execution of the program, our program will take more and more memory, until it consumes all the memory it can get.
*   The destructor of the object will not be called.

We saw in previous chapters that, in C++, we should make use of RAII and get the resources we need in the constructor and return them in the destructor.

If we do not call the destructor, we might not return some resources. For example, a connection to the database would be kept open, and our database would struggle due to too many connections being open, even if we are using only one.

The problem that arises if we call `delete` multiple times on the same pointer is that all the calls after the first one will access memory they should not be accessing.

The result can range from our program crashing to deleting other resources our program is currently using, resulting in incorrect behavior.

We can now see why it is extremely important to define a virtual destructor in the base class if we derive from it: we need to make sure that the destructor of the runtime type is called when calling the `delete` function on the base object. If we call `delete` on a pointer to the base class while the runtime type is the derived class, we will only call the destructor of the base class and not fully destruct the derived class.

Making the destructor of the base class virtual will ensure that we are going to call the derived destructor, since we are using dynamic dispatch when calling it.

#### Note

For every call to the `new` operator, there must be exactly one call to `delete` with the pointer returned by `new`!

This error is extremely common and leads to many errors.

Like single objects, we can also use dynamic memory to create arrays of objects. For such use cases, we can use the `new[]` and `delete[]` expressions:

```cpp
int n = 15;
Car* cars = new Car[n];
delete[] cars;
```

The `new[]` expression will create enough space for `n Car` instances and will initialize them, returning a pointer to the first element created. Here, we are not providing the arguments to the constructor, so the class must have a default constructor.

With `new[]`, we can specify how many elements we want it to initialize. This is different from `std::array` and the built-in array we saw earlier because `n` can be decided at runtime.

We need to call `delete[]` on the pointer returned by `new[]` when we do not need the objects anymore.

#### Note

For every call to `new[]`, there must be exactly one call to `delete[]` with the pointer returned by `new[]`.

The `new` operator and `new[]` function calls, and `delete` and `delete[]` function calls, cannot be intermixed. Always pair the ones for an array or the ones for single elements!

Now that we have seen how to use dynamic memory, we can write the function to create our logger.

The function will call the new expression in its body to create an instance of the correct class, and it will then return a pointer to the base class so that the code calling it does not need to know about the type of logger created:

```cpp
Logger* createLogger() {
  if (are_tests_running()) {
    TestLogger* logger = new TestLogger();
    return logger;
  } else {
    ReleaseLogger logger = new ReleaseLogger("Release logger");
    return logger;
  }
}
```

There are two things to note in this function:

*   Even if we wrote the `new` expression twice, `new` will be called only once per function call.

    This shows us that it is not enough to make sure we type `new` and `delete` an equal number of times; we need to understand how our code is executed.

*   There is no call to `delete`! This means that the code calling the `createLogger` function needs to make sure to call `delete`.

From these two points, we can can see why it is error prone to manage memory manually, and why it should be avoided whenever possible.

Let's look at an example of how to call the function correctly:

```cpp
Logger* logger = createLogger();
myOperation(logger, argument1, argument2);
delete logger;
```

If `myOperation` does not call `delete` on the logger, this is a *correct* use of dynamic memory. Dynamic memory is a powerful tool, but doing it manually is risky, error prone, and easy to get wrong.

Fortunately, modern C++ provides some facilities to make all this much easier to do. It is possible to write entire programs without ever using `new` and `delete` directly.

We will see how in the next section.

## Safe and Easy Dynamic Memory

In the previous section, we learned how dynamic memory could be useful when working with interfaces, especially when creating new instances of derived classes.

We also saw how working with dynamic memory can be hard – we need to make sure to call new and delete in pairs, and failing to do so always has negative effects on our program. Fortunately for us, since C++11, there are tools in the standard library to help us overcome such limitations – **smart pointers**.

Smart pointers are types that behave like pointers, which are called **raw pointers** in this context, but have additional functionality.

We are going to look at two smart pointers from the standard library: `std::unique_ptr` and `std::shared_ptr` (read as `delete` appropriately.

They represent different ownership models. The owner of an object is the code that determines the lifetime of the object – the part of the code that decides when to create and when to destroy the object.

Usually, ownership is associated with the scope a function or method, since the lifetime of automatic variables is controlled by it:

```cpp
void foo() {
  int number;
  do_action(number);
}
```

In this case, the scope of `foo()` owns the `number` object, and it will make sure it is destroyed when the scope exits.

Alternatively, classes might own objects when they are declared as value types between the data members of the class. In that case, the lifetime of the object will be the same as the lifetime of the class:

```cpp
class A {
  int number;
};
```

`number` will be constructed when the `A` class is constructed and will be destroyed when the `A` class is destroyed. This is automatically done because the field `number` is embedded inside the class and the constructor and destructor of the class will automatically initialize `number`.

When managing objects in dynamic memory, ownership is not enforced by the compiler anymore, but it is helpful to apply the concept of ownership to the dynamic memory as well – the owner is who decides when to delete the object.

A function could be the owner of an object when the object is allocated with the new call inside the function, as in the following example:

```cpp
void foo() {
  int* number = new number();
  do_action(number);
  delete number;
}
```

Or a class might own it, by calling `new` in the constructor and storing the pointer in its fields, and calling `delete` on it in the destructor:

```cpp
class A {
    A() : number(new int(0)) {
    }
    ~A() {
        delete number;
    }
    int* number;
};
```

But the ownership of dynamic objects can also be passed around.

We looked at an example earlier with the `createLogger` function. The function creates an instance of **Logger** and then passes the ownership to the parent scope. Now, the parent scope is in charge of making sure the object is valid until it is accessed in the program and deleted afterward.

Smart pointers allow us to specify the ownership in the type of the pointer and make sure it is respected so that we do not have to keep track of it manually anymore.

#### Note

Always use smart pointers to represent the ownership of objects.

In a code base, smart pointers should be the pointers that control the lifetime of objects, and raw pointers, or regular pointers, are used only to reference objects.

### A Single Owner Using std::unique_ptr

`unique_ptr` is the pointer type that's used by default. The unique pointer points to an object that has a single owner; there is a single place in the program that decides when to delete the object.

An example is the logger from before: there is a single place in the program that determines when to delete the object. Since we want the logger to be available as long as the program is running, to always be able to log information, we will destroy the logger only at the end of the program.

The unique pointer guarantees the uniqueness of ownership: the unique pointer cannot be copied. This means that once we have created a unique pointer for an object, there can be only one.

Additionally, when the unique pointer is destroyed, it deletes the object it owns. This way, we have a concrete object that tells us the ownership of the created object, and we do not have to manually make sure that only one place is calling `delete` for the object.

A unique pointer is a template that can take one argument: the *type of the object*.

We could rewrite the previous example as follows:

```cpp
std::unique_ptr<Logger> logger = createLogger();
```

While this code would compile, we would not be respecting the guideline we mentioned previously regarding always using smart pointers for ownership: `createLogger` returns a raw pointer, but it passes ownership to the parent scope.

We can update the signature of `createLogger` to return a smart pointer:

```cpp
std::unique_ptr<Logger>createLogger();
```

Now, the signature expresses our intention, and we can update the implementation to make use of smart pointers.

As we mentioned earlier, with the use of smart pointers, code bases should not use `new` and `delete` anywhere. This is possible because the standard library, since C++14, offers a convenient function: `std::make_unique. make_unique` is a template function that takes the type of the object to create, and creates it in dynamic memory, passing its arguments to the object's constructor and returning a unique pointer to it:

```cpp
std::unique_ptr<Logger>createLogger() {
  if (are_tests_running()) {
    std::unique_ptr<TestLogger> logger = std::make_unique<TestLogger>();
     return logger; // logger is implicitly moved
  } else {
    std::unique_ptr<ReleaseLogger> logger = std::make_unique<ReleaseLogger>("Release logger");
    return logger; // logger is implicitly moved
  }
}
```

There are three important points regarding this function:

*   There is no longer a new expression in the body; it has been replaced with `make_unique`. The `make_unique` function is simple to call because we can provide all the arguments we would pass to the constructor of the type and have it created automatically.
*   We are creating a `unique_ptr` to a derived class, but we are returning a `unique_ptr` to a base class.

    Indeed, `unique_ptr` emulates the ability of raw pointers to convert pointers to derived classes to pointers to base classes. This makes using `unique_ptr` as simple as using **raw pointers**.

*   We are using the move on the `unique_ptr`. As we said earlier, we cannot copy `unique_ptr`, but we are returning from a function, so we must use a value; otherwise, a reference would become invalid after the function returns, as we saw in *Chapter 2*, *Functions*.

    While it cannot be copied, `unique_ptr` can be moved. When we move `unique_ptr`, we are transferring the ownership of the object to which it points to the recipient of the value. In this case, we are returning value, so we are transferring the ownership to the caller of the function.

Let's now see how we can rewrite the class that owns the number we showed before:

```cpp
class A {
  A(): number(std::make_unique<int>()) {}
  std::unique_ptr<int> number;
};
```

Thanks to the fact that `unique_ptr` deletes the object automatically when it is destroyed, we did not have to write the destructor for the class, making our code even easier.

If we need to pass a pointer to the object, without transferring ownership, we can use the `get()` method on the raw pointer. Remember that raw pointers should not be used for ownership, and code accepting the raw pointer should never call `delete` on it.

Thanks to these features, `unique_ptr` should be the default choice to keep track of the ownership of an object.

### Shared Ownership Using std::shared_ptr

`shared_ptr` represents an object that has multiple owners: one out of several objects will delete the owned object.

An example could make a TCP connection, which is established by multiple threads to send data. Each thread uses the TCP connection to send data and then terminates.

We want to delete the TCP connection when the last thread has finished executing, but it is not always the same thread that terminates last; it could be any of the threads.

Alternatively, if we are modeling a graph of connected nodes, we might want to delete a node when every connection to it is removed from the graph. `unique_ptr` does not solve these cases, since there is not a single owner for the object.

`shared_ptr` can be used in such situations: `shared_ptr` can be copied many times, and the object pointed to by the pointer will remain alive until the last `shared_ptr` is destroyed. We guarantee that the object remains valid as long as there is at least one `shared_ptr` instance pointing to it.

Let's look at an example making use of it:

```cpp
class Node {
  public:
    void addConnectedNode(std::shared_ptr<Node> node);
    void removeConnectedNode(std::shared_ptr<Node> node);
  private:
    std::vector<std::shared_ptr<Node>>d_connections;
};
```

Here, we can see that we are holding many `shared_ptr` instance to nodes. If we have a `shared_ptr` instance to a node, we want to be sure that the node exists, but when we remove the shared pointer, we do not care about the node anymore: it might be deleted, or it might be kept alive if there is another node connected to it.

Similar to the `unique_ptr` counterpart, when we want to create a new node, we can use the `std::make_shared` function, which takes the type of the object to construct as the template argument and the arguments to pass to the constructor of the object and returns `shared_ptr` to the object.

You might notice that there might be a problem in the example we showed: what happens if node `A` is connected to node `B` and node `B` is connected to node `A`?

Both nodes have a `shared_ptr` instance to the other, and even if no other node has a connection to them, they will remain alive because a `shared_ptr` instance to them exists. This is an example of circular dependency.

When using shared pointers, we must pay attention to these cases. The standard library offers a different kind of pointer to handle these situations: `std::weak_ptr` (read as **weak pointer**).

`weak_ptr` is a smart pointer that can be used in conjunction with `shared_ptr` to solve the circular dependencies that might happen in our programs.

Generally, `shared_ptr` is enough to model most cases where `unique_ptr` does not work, and together they cover the majority of the uses of dynamic memory in a code base.

Lastly, we are not helpless if we want to use dynamic memory for arrays of which we know the size only at runtime. `unique_ptr` can be used with array types, and `shared_ptr` can be used with array types starting from C++17:

```cpp
std::unique_ptr<int[]>ints = std::make_unique<int[]>();
std::shared_ptr<float[]>floats = std::make_shared<float[]>();
```

### Activity 26: Creating a Factory for UserProfileStorage

Our code needs to create new instances of the `UserProfileStorage` interface we wrote during *Activity 25: Retrieving User Information*:

1.  Write a new `UserProfileStorageFactory` class. Now create a new `create` method which returns a `UserProfileStorage`:
2.  In the `UserProfileStorageFactory` class, return `unique_ptr` so that it manages the lifetime of the interface:

    ```cpp
    class UserProfileStorageFactory {
    public:
        std::unique_ptr<UserProfileStorage> create() const {
            // Create the storage and return it
        }
    };
    ```

3.  Now, in the `main` function, call the `UserProfileStorageFactory` class.

    #### Note

    The solution for this activity can be found at page 313.

### Activity 27: Using a Database Connection for Multiple Operations

In our online store, after a user has paid for a purchase, we want to update their order list so that it is displayed on their profile. At the same time, we also need to schedule the processing of the order.

To do so, we need to update the records in our database.

We don't want to wait for one operation to perform the other, so we process the updates in parallel:

1.  Let's create a `DatabaseConnection` class that can be used in parallel. We want to reuse this as much as possible, and we know we can use `std::async` to start a new parallel task.
2.  Assuming that there are two functions, `updateOrderList(DatabaseConnection&)` and `scheduleOrderProcessing(DatabaseConnection&)`, write two functions, `updateWithConnection()` and `scheduleWithConnection()` which take a shared pointer to `DatabaseConnection` and call the respective function defined above:

    ```cpp
    void updateWithConnection(std::shared_ptr<DatabaseConnection> connection) {
        updateOrderList(*connection);
    }
    void scheduleWithConnection(std::shared_ptr<DatabaseConnection> connection) {
        scheduleOrderProcessing(*connection);
    }
    ```

3.  Use `shared_ptr` and keep a copy of `shared_ptr` in order to make sure that the connection remains valid.
4.  Now let's write the `main` function, where we create a shared pointer to the connection and then we call `std::async` with the two functions we defined above, as illustrated:

    ```cpp
    int main()
    {
        std::shared_ptr<DatabaseConnection> connection = std::make_shared<DatabaseConnection>();
        std::async(std::launch::async, updateWithConnection, connection);
        std::async(std::launch::async, scheduleWithConnection, connection);
    }
    ```

    The output is as follows:

    ```cpp
    Updating order and scheduling order processing in parallel
    Schedule order processing
    Updating order list
    ```

    #### Note

    The solution for this activity can be found at page 314.

## Summary

In this chapter, we saw how inheritance can be used to combine classes in C++. We saw what a base class is and what a derived class is, how to write a class that derives from another, and how to control the visibility modifier. We talked about how to initialize a base class in a derived one by calling the base class constructor.

We then explained polymorphism and the ability of C++ to dynamically bind a pointer or reference of a derived class to a pointer or reference of the base class. We explained what dispatch for functions is, how it works statically by default, and how we can make it dynamic with the use of the virtual keyword. Following that, we explored how to properly write virtual functions and how we can override them, making sure to mark such overrode functions with the `override` keyword.

Next, we showed how to define interfaces with abstract base classes and how to use pure virtual methods. We also provided guidelines on how to correctly define interfaces.

Lastly, we delved into dynamic memory and what problems it solves, but we also saw how easy it is to use it incorrectly.

We concluded the chapter by showing how modern C++ makes using dynamic memory painless by providing smart pointers that handle complex details for us: `unique_ptr` to manage objects with a single owner, and `shared_ptr` for objects owned by multiple objects.

All these tools can be effective at writing solid programs that can be effectively evolved and maintained, while retaining the performance C++ is famous for.