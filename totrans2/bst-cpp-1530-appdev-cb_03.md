# Chapter 3. Managing Resources

In this chapter we will cover:

*   Managing pointers to classes that do not leave scope
*   Reference counting of pointers to classes used across methods
*   Managing pointers to arrays that do not leave scope
*   Reference counting pointers to arrays used across methods
*   Storing any functional objects in a variable
*   Passing a function pointer in a variable
*   Passing C++11 lambda functions in a variable
*   Containers of pointers
*   Doing something at scope exit
*   Initializing the base class by a member of the derived class

# Introduction

In this chapter, we'll continue to deal with datatypes, introduced by the Boost libraries, mostly focusing on working with pointers. We'll see how to easily manage resources, and how to use a datatype capable of storing any functional objects, functions, and lambda expressions. After reading this chapter, your code will become more reliable, and memory leaks will become history.

# Managing pointers to classes that do not leave scope

There are situations where we are required to dynamically allocate memory and construct a class in that memory. And, that's where the troubles start. Have a look at the following code:

[PRE0]

This code looks correct at first glance. But, what if `some_function1()` or `some_function2()` throws an exception? In that case, `p` won't be deleted. Let's fix it in the following way:

[PRE1]

Now the code is ugly and hard to read but is correct. Maybe we can do better than this.

## Getting ready

Basic knowledge of C++ and code behavior during exceptions is required.

## How to do it...

Let's take a look at the `Boost.SmartPtr` library. There is a `boost::scoped_ptr` class that may help you out:

[PRE2]

Now, there is no chance that the resource will leak, and the source code is much clearer.

### Note

If you have control over `some_function1()` and `some_function2()`, you may wish to rewrite them so they will take a reference to `scoped_ptr<foo_class>` (or just a reference) instead of a pointer to `foo_class`. Such an interface will be more intuitive.

## How it works...

In the destructor, `boost::scoped_ptr<T>` will call `delete` for a pointer that it stores. When an exception is thrown, the stack is unwound, and the destructor of `scoped_ptr` is called.

The `scoped_ptr<T>` class template is not copyable; it stores only a pointer to the class and does not require `T` to be of a complete type (it can be forward declared). Some compilers do not warn when an incomplete type is being deleted, which may lead to errors that are hard to detect, but `scoped_ptr` (and all the classes in `Boost.SmartPtr`) has a specific compile-time assert for such cases. That makes `scoped_ptr` perfect for implementing the `Pimpl` idiom.

The `boost::scoped_ptr<T>` function is equal to `const std::auto_ptr<T>`, but it also has the `reset()` function.

## There's more...

This class is extremely fast. In most cases, the compiler will optimize the code that uses `scoped_ptr` to the machine code, which is close to our handwritten version (and sometimes even better if the compiler detects that some functions do not throw exceptions).

## See also

*   The documentation of the `Boost.SmartPtr` library contains lots of examples and other useful information about all the smart pointers' classes. You can read about it at [http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm).

# Reference counting of pointers to classes used across methods

Imagine that you have some dynamically allocated structure containing data, and you want to process it in different execution threads. The code to do this is as follows:

[PRE3]

We cannot deallocate `p` at the end of the `while` loop because it can still be used by threads that run process functions. Process functions cannot delete `p` because they do not know that other threads are not using it anymore.

## Getting ready

This recipe uses the `Boost.Thread` library, which is not a header-only library, so your program will need to link against the `libboost_thread` and `libboost_system` libraries. Make sure that you understand the concept of threads before reading further. Refer to the *See also* section for references on recipes that use threads.

You'll also need some basic knowledge on `boost::bind` or `std::bind`, which is almost the same.

## How to do it...

As you may have guessed, there is a class in Boost (and C++11) that will help you to deal with it. It is called `boost::shared_ptr`, and it can be used as:

[PRE4]

Another example of this is as follows:

[PRE5]

## How it works...

The `shared_ptr` class has an atomic reference counter inside. When you copy it, the reference counter is incremented, and when its destructor is called, the reference counter is decremented. When the reference counter equals zero, `delete` is called for the object pointed by `shared_ptr`.

Now, let's find out what's happening in the case of `boost::thread` (`boost::bind(&process_sp1, p)`). The function `process_sp1` takes a parameter as a reference, so why is it not deallocated when we get out of the `while` loop? The answer is simple. The functional object returned by `bind()` contains a copy of the shared pointer, and that means that the data pointed by `p` won't be deallocated until the functional object is destroyed.

Getting back to `boost::make_shared`, let's take a look at `shared_ptr<std::string> ps(new int(0))`. In this case, we have two calls to `new`: firstly while constructing a pointer to an integer, and secondly when constructing a `shared_ptr` class (it allocates an atomic counter on heap using call `new`). But, when we construct `shared_ptr` using `make_shared`, only one call to `new` will be made. It will allocate a single piece of memory and will construct an atomic counter and the `int` object in that piece.

## There's more...

The atomic reference counter guarantees the correct behavior of `shared_ptr` across the threads, but you must remember that atomic operations are not as fast as nonatomic. On C++11 compatible compilers, you may reduce the atomic operations' count using `std::move` (move the constructor of the shared pointer in such a way that the atomic counter is neither incremented nor decremented).

The `shared_ptr` and `make_shared` classes are part of C++11, and they are declared in the header `<memory>` in `std::` namespace.

## See also

*   Refer to [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, for more information about `Boost.Thread` and atomic operations.
*   Refer to the *Reordering the parameters of function* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, for more information about `Boost.Bind`.
*   Refer to the *Binding a value as a function parameter* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, for more information about `Boost.Bind`.
*   The documentation of the `Boost.SmartPtr` library contains lots of examples and other useful information about all the smart pointers' classes. You can read about it at [http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm).

# Managing pointers to arrays that do not leave scope

We already saw how to manage pointers to a resource in the *Managing pointers to classes that do not leave scope* recipe. But, when we deal with arrays, we need to call `delete[]` instead of a simple `delete`, otherwise there will be a memory leak. Have a look at the following code:

[PRE6]

## Getting ready

Knowledge of C++ exceptions and templates are required for this recipe.

## How to do it...

The `Boost.SmartPointer` library has not only the `scoped_ptr<>` class but also a `scoped_array<>` class.

[PRE7]

## How it works...

It works just like a `scoped_ptr<>` class but calls `delete[]` instead of `delete` in the destructor.

## There's more...

The `scoped_array<>` class has the same guarantees and design as `scoped_ptr<>`. It has neither additional memory allocations nor virtual functions' call. It cannot be copied and is not a part of C++11.

## See also

*   The documentation of the `Boost.SmartPtr` library contains lots of examples and other useful information about all the smart pointers' classes. You can read about it at [http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm).

# Reference counting pointers to arrays used across methods

We continue coping with pointers, and our next task is to reference count an array. Let's take a look at a program that gets some data from the stream and processes it in different threads. The code to do this is as follows:

[PRE8]

Just the same problem that occurred in the *Reference counting of pointers to classes used across methods* recipe.

## Getting ready

This recipe uses the `Boost.Thread` library, which is not a header-only library, so your program will need to link against the `libboost_thread` and `libboost_system` libraries. Make sure that you understand the concept of threads before reading further.

You'll also need some basic knowledge on `boost::bind` or `std::bind`, which is almost the same.

## How to do it...

There are three solutions. The main difference between them is of type and construction of the `data_cpy` variable. Each of these solutions does exactly the same things that are described in the beginning of this recipe but without memory leaks. The solutions are:

*   The first solution:

    [PRE9]

*   The second solution:

    Since Boost 1.53 `shared_ptr` itself can take care of arrays:

    [PRE10]

*   The third solution:

    [PRE11]

## How it works...

In each of these examples, shared classes count references and call `delete[]` for a pointer when the reference count becomes equal to zero. The first and second examples are trivial. In the third example, we provide a `deleter` object for a shared pointer. The `deleter` object will be called instead of the default call to `delete`. This `deleter` is the same as used in C++11 in `std::unique_ptr` and `std::shared_ptr`.

## There's more...

The first solution is traditional to Boost; prior to Boost 1.53, the functionality of the second solution was not implemented in `shared_ptr`.

The second solution is the fastest one (it uses fewer calls to `new`), but it can be used only with Boost 1.53 and higher.

The third solution is the most portable one. It can be used with older versions of Boost and with C++11 STL's `shared_ptr<>` (just don't forget to change `boost::checked_array_deleter<T>()` to `std::default_delete<T[]>()`).

## See also

*   The documentation of the `Boost.SmartPtr` library contains lots of examples and other useful information about all the smart pointers' classes. You can read about it at [http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm](http://www.boost.org/doc/libs/1_53_0/libs/smart_ptr/smart_ptr.htm).

# Storing any functional objects in a variable

C++ has a syntax to work with pointers to functions and member functions' pointers. And, that is good! However, this mechanism is hard to use with functional objects. Consider the situation when you are developing a library that has its API declared in the header files and implementation in the source files. This library shall have a function that accepts any functional objects. How would you pass a functional object to it? Have a look at the following code:

[PRE12]

## Getting ready

Reading the *Storing any value in a container/variable* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, is recommended before starting this recipe.

You'll also need some basic knowledge on `boost::bind` or `std::bind`, which is almost the same.

## How to do it...

Let's see how to fix the example and make `process_integers` accept functional objects:

1.  There is a solution, and it is called a `Boost.Function` library. It allows you to store any function, a member function, or a functional object if its signature is a match to the one described in a template argument:

    [PRE13]

    The `boost::function` class has a default constructor and has an empty state.

2.  Checking for an empty/default constructed state can be done like this:

    [PRE14]

## How it works...

The `fobject_t` method stores in itself data from functional objects and erases their exact type. It is safe to use the `boost::function` objects such as the following code:

[PRE15]

Does it remind you of the `boost::any` class? It uses the same technique—type erasure for storing any function objects.

## There's more...

The `Boost.Function` library has an insane amount of optimizations; it may store small functional objects without additional memory allocations and has optimized move assignment operators. It is accepted as a part of C++11 STL library and is defined in the `<functional>` header in the `std::` namespace.

But, remember that `boost::function` implies an optimization barrier for the compiler. It means that:

[PRE16]

will be better optimized by the compiler than

[PRE17]

This is why you should try to avoid using `Boost.Function` when its usage is not really required. In some cases, the C++11 `auto` keyword can be handy instead:

[PRE18]

## See also

*   The official documentation of `Boost.Function` contains more examples, performance measures, and class reference documentation. You can read about it at [http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html).
*   The *Passing a function pointer in a variable* recipe.
*   The *Passing C++11 lambda functions in a variable* recipe.

# Passing a function pointer in a variable

We are continuing with the previous example, and now we want to pass a pointer to a function in our `process_integeres()` method. Shall we add an overload for just function pointers, or is there a more elegant way?

## Getting ready

This recipe is continuing the previous one. You must read the previous recipe first.

## How to do it...

Nothing needs to be done as `boost::function<>` is also constructible from the function pointers:

[PRE19]

## How it works...

A pointer to `my_ints_function` will be stored inside the `boost::function` class, and calls to `boost::function` will be forwarded to the stored pointer.

## There's more...

The `Boost.Function` library provides good performance for pointers to functions, and it will not allocate memory on heap. However, whatever you store in `boost::function`, it will use an RTTI. If you disable RTTI, it will continue to work but will dramatically increase the size of a compiled binary.

## See also

*   The official documentation of `Boost.Function` contains more examples, performance measures, and class reference documentation. You can read about it at [http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html).
*   The *Passing C++11 lambda functions in a variable* recipe.

# Passing C++11 lambda functions in a variable

We are continuing with the previous example, and now we want to use a lambda function with our `process_integers()` method.

## Getting ready

This recipe is continuing the series of the previous two. You must read them first. You will also need a C++11 compatible compiler or at least a compiler with C++11 lambda support.

## How to do it...

Nothing needs to be done as `boost::function<>` is also usable with lambda functions of any difficulty:

[PRE20]

## There's more...

Performance of the lambda function storage in `Boost.Functional` is the same as in other cases. While the functional object produced by the lambda expression is small enough to fit in an instance of `boost::function`, no dynamic memory allocation will be performed. Calling an object stored in `boost::function` is close to the speed of calling a function by a pointer. Copying of an object is close to the speed of constructing `boost::function` and will exactly use a dynamic memory allocation in similar cases. Moving objects won't allocate and deallocate memory.

## See also

*   Additional information about performance and `Boost.Function` can be found on the official documentation page at [http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html)

# Containers of pointers

There are such cases when we need to store pointers in the container. The examples are: storing polymorphic data in containers, forcing fast copy of data in containers, and strict exception requirements for operations with data in containers. In such cases, the C++ programmer has the following choices:

*   Store pointers in containers and take care of their destructions using the operator `delete`:

    [PRE21]

    Such an approach is error prone and requires a lot of writing

*   Store smart pointers in containers:

    For the C++03 version:

    [PRE22]

    The `std::auto_ptr` method is deprecated, and it is not recommended to use it in containers. Moreover, this example will not compile with C++11.

    For the C++11 version:

    [PRE23]

    This solution is a good one, but it cannot be used in C++03, and you still need to write a comparator functional object

*   Use `Boost.SmartPtr` in the container:

    [PRE24]

    This solution is portable, but you still need to write comparators, and it adds performance penalties (an atomic counter requires additional memory, and its increments/decrements are not as fast as nonatomic operations)

## Getting ready

Knowledge of STL containers is required for better understanding of this recipe.

## How to do it...

The `Boost.PointerContainer` library provides a good and portable solution:

[PRE25]

## How it works...

The `Boost.PointerContainer` library has classes `ptr_array`, `ptr_vector`, `ptr_set`, `ptr_multimap`, and others. All these containers simplify your life. When dealing with pointers, they will be deallocating pointers in destructors and simplifying access to data pointed by the pointer (no need for additional dereference in `assert(*s.begin() == 0);`).

## There's more...

Previous examples were not cloning pointer data, but when we want to clone some data, all we need to do is to just define a freestanding function such as `new_clone()` in the namespace of the object to be cloned. Moreover, you may use the default `T* new_clone( const T& r )` implementation if you include the header file `<boost/ptr_container/clone_allocator.hpp>` as shown in the following code:

[PRE26]

## See also

*   The official documentation contains detailed reference for each class, and you may read about it at [http://www.boost.org/doc/libs/1_53_0/libs/ptr_container/doc/ptr_container.html](http://www.boost.org/doc/libs/1_53_0/libs/ptr_container/doc/ptr_container.html)
*   The first four recipes of this chapter will give you some examples of smart pointers' usage

# Doing something at scope exit

If you were dealing with languages such as Java, C#, or Delphi, you were obviously using the `try{} finally{}` construction or `scope(exit)` in the D programming language. Let me briefly describe to you what do these language constructions do.

When a program leaves the current scope via return or exception, code in the `finally` or `scope(exit)` blocks is executed. This mechanism is perfect for implementing the **RAII** pattern as shown in the following code snippet:

[PRE27]

Is there a way to do such a thing in C++?

## Getting ready

Basic C++ knowledge is required for this recipe. Knowledge of code behavior during thrown exceptions will be useful.

## How to do it...

The `Boost.ScopeExit` library was designed to solve such problems:

[PRE28]

## How it works...

The variable `f` is passed by value via `BOOST_SCOPE_EXIT(f)`. When the program leaves the scope of execution, the code between `BOOST_SCOPE_EXIT(f) {` and `} BOOST_SCOPE_EXIT_END` will be executed. If we wish to pass the value by reference, use the `&` symbol in the `BOOST_SCOPE_EXIT` macro. If we wish to pass multiple values, just separate them using a comma.

### Note

Passing references to a pointer does not work well on some compilers. The `BOOST_SCOPE_EXIT(&f)` macro cannot be compiled there, which is why we do not capture it by reference in the example.

## There's more...

To capture this inside a member function, we use a special symbol `this_`:

[PRE29]

The `Boost.ScopeExit` library allocates no additional memory on heap and does not use virtual functions. Use the default syntax and do not define `BOOST_SCOPE_EXIT_CONFIG_USE_LAMBDAS` because otherwise scope exit will be implemented using `boost::function`, which may allocate additional memory and imply the optimization barrier.

## See also

*   The official documentation contains more examples and use cases. You can read about it at [http://www.boost.org/doc/libs/1_53_0/libs/scope_exit/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/scope_exit/doc/html/index.html).

# Initializing the base class by a member of the derived class

Let's take a look at the following example. We have some base class that has virtual functions and must be initialized with reference to the `std::ostream` object:

[PRE30]

We also have a derived class that has a `std::ostream` object and implements the `do_process()` function:

[PRE31]

This is not a very common case in programming, but when such mistakes happen, it is not always simple to get the idea of bypassing it. Some people try to bypass it by changing the order of `logger_` and the base type initialization:

[PRE32]

It won't work as they expect because direct base classes are initialized before nonstatic data members, regardless of the order of the member initializers.

## Getting ready

Basic knowledge of C++ is required for this recipe.

## How to do it...

The `Boost.Utility` library provides a quick solution for such cases; it is called the `boost::base_from_member` template. To use it, you need to carry out the following steps:

1.  Include the `base_from_member.hpp` header:

    [PRE33]

2.  Derive your class from `boost::base_from_member<T>` where `T` is a type that must be initialized before the base (take care about the order of the base classes; `boost::base_from_member<T>` must be placed before the class that uses `T`):

    [PRE34]

3.  Correctly write the constructor as follows:

    [PRE35]

## How it works...

If direct base classes are initialized before nonstatic data members, and if direct base classes would be initialized in declaration order as they appear in the base-specifier-list, we need to somehow make a base class our nonstatic data member. Or make a base class that has a member field with the required member:

[PRE36]

## There's more...

As you may see, `base_from_member` has an integer as a second template argument. This is done for cases when we need multiple `base_from_member` classes of the same type:

[PRE37]

The `boost::base_from_member` class neither applies additional dynamic memory allocations nor has virtual functions. The current implementation does not support C++11 features (such as perfect forwarding and variadic templates), but in Boost's trunk branch, there is an implementation that can use all the benefits of C++11\. It possibly will be merged to release a branch in the nearest future.

## See also

*   The `Boost.Utility` library contains many helpful classes and methods; documentation for getting more information about it is at [http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm)
*   The *Making a noncopyable class* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, contains more examples of classes from `Boost.Utility`
*   Also, the *Using the C++11 move emulation* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, contains more examples of classes from `Boost.Utility`