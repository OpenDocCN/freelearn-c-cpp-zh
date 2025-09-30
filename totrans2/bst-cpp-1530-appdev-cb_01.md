# Chapter 1. Starting to Write Your Application

In this chapter we will cover:

*   Getting configuration options
*   Storing any value in a container/variable
*   Storing multiple chosen types in a container/variable
*   Using a safer way to work with a container that stores multiple chosen types
*   Returning a value or flag where there is no value
*   Returning an array from a function
*   Combining multiple values into one
*   Reordering the parameters of a function
*   Binding a value as a function parameter
*   Using the C++11 move emulation
*   Making a noncopyable class
*   Making a noncopyable but movable class

# Introduction

Boost is a collection of C++ libraries. Each library has been reviewed by many professional programmers before being accepted to Boost. Libraries are tested on multiple platforms using many compilers and the C++ standard library implementations. While using Boost, you can be sure that you are using one of the most portable, fast, and reliable solutions that is distributed under a license suitable for commercial and open source projects.

Many parts of Boost have been included in C++11, and even more parts are going to be included in the next standard of C++. You will find C++11-specific notes in each recipe of this book.

Without a long introduction, let's get started!

In this chapter we will see some recipes for everyday use. We'll see how to get configuration options from different sources and what can be cooked up using some of the datatypes introduced by Boost library authors.

# Getting configuration options

Take a look at some of the console programs, such as `cp` in Linux. They all have a fancy help, their input parameters do not depend on any position, and have a human readable syntax, for example:

[PRE0]

You can implement the same functionality for your program in 10 minutes. And all you need is the `Boost.ProgramOptions` library.

## Getting ready

Basic knowledge of C++ is all you need for this recipe. Remember that this library is not a header-only, so your program will need to link against the `libboost_program_options` library.

## How to do it...

Let's start with a simple program that accepts the number of apples and oranges as input and counts the total number of fruits. We want to achieve the following result:

[PRE1]

Perform the following steps:

1.  First of all, we need to include the `program_options` header and make an alias for the `boost::program_options` namespace (it is too long to type it!). We would also need an `<iostream>` header:

    [PRE2]

2.  Now we are ready to describe our options:

    [PRE3]

3.  We'll see how to use a third parameter a little bit later, after which we'll deal with parsing the command line and outputting the result:

    [PRE4]

    That was simple, wasn't it?

4.  Let's add the `--help` parameter to our option's description:

    [PRE5]

5.  Now add the following lines after `opt::notify(vm);`, and you'll get a fully functional help for your program:

    [PRE6]

    Now, if we call our program with the `--help` parameter, we'll get the following output:

    [PRE7]

    As you can see, we do not provide a type for the option's value, because we do not expect any values to be passed to it.

6.  Once we have got through all the basics, let's add short names for some of the options, set the default value for apples, add some string input, and get the missing options from the configuration file:

    [PRE8]

    ### Note

    When using a configuration file, we need to remember that its syntax differs from the command-line syntax. We do not need to place minuses before the options. So our `apples_oranges.cfg` option must look like this:

    `oranges=20`

## How it works...

This example is pretty trivial to understand from code and comments. Much more interesting is what output we get on execution:

[PRE9]

## There's more...

The C++11 standard adopted many Boost libraries; however, you won't find `Boost.ProgramOptions` in it.

## See also

*   Boost's official documentation contains many more examples and shows more advanced features of `Boost.ProgramOptions`, such as position-dependent options, nonconventional syntax, and more. This is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/program_options.html](http://www.boost.org/doc/libs/1_53_0/doc/html/program_options.html)

### Tip

**Downloading the example code**

You can download the example code files for all Packt books that you have purchased from your account at [http://www.PacktPub.com](http://www.PacktPub.com). If you purchased this book elsewhere, you can visit [http://www.PacktPub.com/support](http://www.PacktPub.com/support) and register to have the files e-mailed directly to you.

# Storing any value in a container/variable

If you have been programming in Java, C#, or Delphi, you will definitely miss the ability to create containers with the `Object` value type in C++. The `Object` class in those languages is a basic class for almost all types, so you are able to assign (almost) any value to it at any time. Just imagine how great it would be to have such a feature in C++:

[PRE10]

## Getting ready

We'll be working with the header-only library. Basic knowledge of C++ is all you need for this recipe.

## How to do it...

In such cases, Boost offers a solution, the `Boost.Any` library, which has an even better syntax:

[PRE11]

Great, isn't it? By the way, it has an empty state, which could be checked using the `empty()` member function (just as in STL containers).

You can get the value from `boost::any` using two approaches:

[PRE12]

## How it works...

The `boost::any` class just stores any value in it. To achieve this it uses the **type erasure** technique (close to what Java or C# does with all of its types). To use this library, you do not really need to know its internal implementation, so let's just have a quick glance at the type erasure technique. `Boost.Any`, on assignment of some variable of type `T`, constructs a type (let's call it `holder<T>`) that may store a value of the specified type `T`, and is derived from some internal base-type placeholder. A placeholder has virtual functions for getting `std::type_info` of a stored type and for cloning a stored type. When `any_cast<T>()` is used, `boost::any` checks that `std::type_info` of a stored value is equal to `typeid(T)` (the overloaded placeholder's function is used for getting `std::type_info`).

## There's more...

Such flexibility never comes without a cost. Copy constructing, value constructing, copy assigning, and assigning values to instances of `boost::any` will call a dynamic memory allocation function; all of the type casts need to get **runtime type information** (**RTTI**); `boost::any` uses virtual functions a lot. If you are keen on performance, see the next recipe, which will give you an idea of how to achieve almost the same results without dynamic allocations and RTTI usage.

Another disadvantage of `Boost.Any` is that it cannot be used with RTTI disabled. There is a possibility to make this library usable even with RTTI disabled, but it is not currently implemented.

### Note

Almost all exceptions in Boost derive from the `std::exception` class or from its derivatives, for example, `boost::bad_any_cast` is derived from `std::bad_cast`. It means that you can catch almost all Boost exceptions using `catch (const std::exception& e)`.

## See also

*   Boost's official documentation may give you some more examples, and it can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/any.html](http://www.boost.org/doc/libs/1_53_0/doc/html/any.html)
*   The *Using a safer way to work with a container that stores multiple chosen types* recipe for more info on the topic

# Storing multiple chosen types in a variable/container

Are you aware of the concept of unrestricted unions in C++11? Let me tell you about it in short. **C++03 unions** can only hold extremely simple types of data called POD (plain old data). So in C++03, you cannot, for example, store `std::string` or `std::vector` in a union. C++11 relaxes this requirement, but you'll have to manage the construction and destruction of such types by yourself, call in-place construction/destruction, and remember what type is stored in a union. A huge amount of work, isn't it?

## Getting ready

We'll be working with the header-only library, which is simple to use. Basic knowledge of C++ is all you need for this recipe.

## How to do it...

Let me introduce the `Boost.Variant` library to you.

1.  The `Boost.Variant` library can store any of the types specified at compile time; it also manages in-place construction/destruction and doesn't even require the C++11 standard:

    [PRE13]

    Great, isn't it?

2.  `Boost.Variant` has no empty state, but has an `empty()` function, which always returns `false`. If you do need to represent an empty state, just add some trivial type at the first position of the types supported by the `Boost.Variant` library. When `Boost.Variant` contains that type, interpret it as an empty state. Here is an example in which we will use a `boost::blank` type to represent an empty state:

    [PRE14]

3.  You can get a value from a variant using two approaches:

    [PRE15]

## How it works...

The `boost::variant` class holds an array of characters and stores values in that array. Size of the array is determined at compile time using `sizeof()` and functions to get alignment. On assignment or construction of `boost::variant`, the previous values are in-place destroyed, and new values are constructed on top of the character array using the new placement.

## There's more...

The `Boost.Variant` variables usually do not allocate memory in a heap, and they do not require RTTI to be enabled. `Boost.Variant` is extremely fast and used widely by other Boost libraries. To achieve maximum performance, make sure that there is a trivial type in the list of supported types, and that this type is at the first position.

### Note

`Boost.Variant` is not a part of the C++11 standard.

## See also

*   The *Using a safer way to work with a container that stores multiple chosen types* recipe
*   Boost's official documentation contains more examples and descriptions of some other features of `Boost.Variant`, and can be found at:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/variant.html](http://www.boost.org/doc/libs/1_53_0/doc/html/variant.html)

# Using a safer way to work with a container that stores multiple chosen types

Imagine that you are creating a wrapper around some SQL database interface. You decided that `boost::any` will perfectly match the requirements for a single cell of the database table. Some other programmer will be using your classes, and his task would be to get a row from the database and count the sum of the arithmetic types in a row.

Here's how the code would look:

[PRE16]

If you compile and run this example, it will output a correct answer:

[PRE17]

Do you remember what your thoughts were when reading the implementation of `operator()`? I guess they were, "And what about double, long, short, unsigned, and other types?". The same thoughts will come to the mind of a programmer who will use your interface. So you'll need to carefully document values stored by your `cell_t`, or read the more elegant solution described in the following sections.

## Getting ready

Reading the previous two recipes is highly recommended if you are not already familiar with the `Boost.Variant` and `Boost.Any` libraries.

## How to do it...

The `Boost.Variant` library implements a visitor programming pattern for accessing the stored data, which is much safer than getting values via `boost::get<>`. This pattern forces the programmer to take care of each variant type, otherwise the code will fail to compile. You can use this pattern via the `boost::apply_visitor` function, which takes a visitor functional object as the first parameter and a variant as the second parameter. Visitor functional objects must derive from the `boost::static_visitor<T>` class, where `T` is a type being returned by a visitor. A visitor object must have overloads of `operator()` for each type stored by a variant.

Let's change the `cell_t` type to `boost::variant<int, float, string>` and modify our example:

[PRE18]

## How it works...

The `Boost.Variant` library will generate a big `switch` statement at compile time, each case of `which` will call a visitor for a single type from the variant's list of types. At runtime, the index of the stored type can be retrieved using `which()`, and a jump to the correct case in the switch will be made. Something like this will be generated for `boost::variant<int, float, std::string>`:

[PRE19]

Here, the `address()` function returns a pointer to the internal storage of `boost::variant<int, float, std::string>`.

## There's more...

If we compare this example with the first example in this recipe, we'll see the following advantages of `boost::variant`:

*   We know what types a variable can store
*   If a library writer of the SQL interface adds or modifies a type held by a variant, we'll get a compile-time error instead of incorrect behavior

## See also

*   After reading some recipes from [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, you'll be able to make the visitor object so generic that it will be able to work correctly even if the underlying types change
*   Boost's official documentation contains more examples and a description of some other features of `Boost.Variant`, and is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/variant.html](http://www.boost.org/doc/libs/1_53_0/doc/html/variant.html)

# Returning a value or flag where there is no value

Imagine that we have a function that does not throw an exception and returns a value or indicates that an error has occurred. In Java or C# programming languages, such cases are handled by comparing a return value from a function value with a null pointer; if it is null then an error has occurred. In C++, returning a pointer from a function confuses library users and usually requires dynamic memory allocation (which is slow).

## Getting ready

Only basic knowledge of C++ is required for this recipe.

## How to do it...

Ladies and gentlemen, let me introduce you to the `Boost.Optional` library using the following example:

The `try_lock_device()` function tries to acquire a lock for a device, and may succeed or not depending on different conditions (in our example it depends on the `rand()` function call). The function returns an optional variable that can be converted to a Boolean variable. If the returned value is equal to Boolean `true`, then the lock is acquired, and an instance of a class to work with the device can be obtained by dereferencing the returned optional variable:

[PRE20]

This program will output the following:

[PRE21]

### Note

The default constructed `optional` variable is convertible to a Boolean variable holding `false` and must not be dereferenced, because it does not have an underlying type constructed.

## How it works...

The `Boost.Optional` class is very close to the `boost::variant` class but for only one type, `boost::optional<T>` has an array of `chars`, where the object of type `T` can be an in-place constructor. It also has a Boolean variable to remember the state of the object (is it constructed or not).

## There's more...

The `Boost.Optional` class does not use dynamic allocation, and it does not require a default constructor for the underlying type. It is fast and considered for inclusion in the next standard of C++. The current `boost::optional` implementation cannot work with C++11 **rvalue** references; however, there are some patches proposed to fix that.

The C++11 standard does not include the `Boost.Optional` class; however, it is currently being reviewed for inclusion in the next C++ standard or in C++14.

## See also

*   Boost's official documentation contains more examples and describes advanced features of `Boost.Optional` (like in-place construction using the factory functions). The documentation is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/optional/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/optional/doc/html/index.html)

# Returning an array from a function

Let's play a guessing game! What can you tell about the following function?

[PRE22]

Should return values be deallocated by the programmer or not? Does the function attempt to deallocate the input parameter? Should the input parameter be zero-terminated, or should the function assume that the input parameter has a specified width?

And now, let's make the task harder! Take a look at the following line:

[PRE23]

Please do not worry; I've also been scratching my head for half an hour before getting an idea of what is happening here. `vector_advance` is a function that accepts and returns an array of four elements. Is there a way to write such a function clearly?

## Getting ready

Only basic knowledge of C++ is required for this recipe.

## How to do it...

We can rewrite the function like this:

[PRE24]

Here, `boost::array<char, 4>` is just a simple wrapper around an array of four char elements.

This code answers all of the questions from our first example and is much more readable than the second example.

## How it works...

The first template parameter of `boost::array` is the element type, and the second one is the size of an array. `boost::array` is a fixed-size array; if you need to change the array size at runtime, use `std::vector` or `boost::container::vector` instead.

The `Boost.Array` library just contains an array in it. That is all. Simple and efficient. The `boost::array<>` class has no handwritten constructors and all of its members are public, so the compiler will think of it as a POD type.

![How it works...](img/4880OS_01_new.jpg)

## There's more...

Let's see some more examples of the usage of `boost::array`:

[PRE25]

One of the biggest advantages of `boost::array` is that it provides exactly the same performance as a normal C array. People from the C++ standard committee also liked it, so it was accepted to the C++11 standard. There is a chance that your STL library already has it (you may try to include the `<array>` header and check for the availability of `std::array<>`).

## See also

*   Boost's official documentation gives a complete list of the `Boost.Array` methods with a description of the method's complexity and throw behavior, and is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/boost/array.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost/array.html)

*   The `boost::array` function is widely used across recipes; for example, refer to the *Binding a value as a function parameter* recipe

# Combining multiple values into one

There is a very nice present for those who like `std::pair`. Boost has a library called `Boost.Tuple`, and it is just like `std::pair`, but it can also work with triples, quads, and even bigger collections of types.

## Getting ready

Only basic knowledge of C++ and STL is required for this recipe.

## How to do it...

Perform the following steps to combine multiple values in to one:

1.  To start working with tuples, you need to include a proper header and declare a variable:

    [PRE26]

2.  Getting a specific value is implemented via the `boost::get<N>()` function, where `N` is a zero-based index of a required value:

    [PRE27]

    The `boost::get<>` function has many overloads and is used widely across Boost. We have already seen how it can be used with other libraries in the *Storing multiple chosen types in a container/variable* recipe.

3.  You can construct tuples using the `boost::make_tuple()` function, which is shorter to write, because you do not need to fully qualify the tuple type:

    [PRE28]

4.  Another function that makes life easy is `boost::tie()`. It works almost as `make_tuple`, but adds a nonconst reference for each of the passed types. Such a tuple can be used to get values to a variable from another tuple. It can be better understood from the following example:

    [PRE29]

## How it works...

Some readers may wonder why we need a tuple when we can always write our own structures with better names, for example, instead of writing `boost::tuple<int, std::string>`, we can create a structure:

[PRE30]

Well, this structure is definitely more clear than `boost::tuple<int, std::string>`. But what if this structure is used only twice in the code?

The main idea behind the tuple's library is to simplify template programming.

![How it works...](img/4880OS_01_new.jpg)

## There's more...

A tuple works as fast as `std::pair` (it does not allocate memory on a heap and has no virtual functions). The C++ committee found this class to be very useful and it was included in STL; you can find it in a C++11-compatible STL implementation in the header file `<tuple>` (don't forget to replace all the `boost::` namespaces with `std::`).

The current Boost implementation of a tuple does not use variadic templates; it is just a set of classes generated by a script. There is an experimental version that uses C++11 rvalues and an emulation of them on C++03 compilers, so there is a chance that Boost 1.54 will be shipped with faster implementation of tuples.

## See also

*   The experimental version of tuples can be found at the following link:

    [http://svn.boost.org/svn/boost/sandbox/tuple-move/](http://svn.boost.org/svn/boost/sandbox/tuple-move/)

*   Boost's official documentation contains more examples, information about performance, and abilities of `Boost.Tuple`. It is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/tuple/doc/tuple_users_guide.html](http://www.boost.org/doc/libs/1_53_0/libs/tuple/doc/tuple_users_guide.html)

*   The *Converting all tuple elements to strings* recipe in [Chapter 8](ch08.html "Chapter 8. Metaprogramming"), *Metaprogramming,* shows some advanced usages of tuples

# Reordering the parameters of function

This recipe and the next one are devoted to a very interesting library, whose functionality at first glance looks like some kind of magic. This library is called `Boost.Bind` and it allows you to easily create new functional objects from functions, member functions, and functional objects, also allowing the reordering of the initial function's input parameters and binding some values or references as function parameters.

## Getting ready

Knowledge of C++, STL algorithms, and functional objects is required for this recipe.

## How to do it...

1.  Let's start with an example. You are working with a vector of integral types provided by some other programmer. That integral type has only one operator, `+`, but your task is to multiply a value by two. Without `bind` this can be achieved with the use of a functional object:

    [PRE31]

    With `Boost.Bind`, it would be as follows:

    [PRE32]

2.  By the way, we can easily make this function more generic:

    [PRE33]

## How it works...

Let's take a closer look at the `mul_2` function. We provide a vector of values to it, and for each value it applies a functional object returned by the `bind()` function. The `bind()` function takes in three parameters; the first parameter is an instance of the `std::plus<Number>` class (which is a functional object). The second and third parameters are placeholders. The placeholder `_1` substitutes the argument with the first input argument of the resulting functional object. As you might guess, there are many placeholders; placeholder `_2` means substituting the argument with the second input argument of the resulting functional object, and the same also applies to placeholder `_3`. Well, seems you've got the idea.

## There's more...

Just to make sure that you've got the whole idea and know where bind can be used, let's take a look at another example.

We have two classes, which work with some sensor devices. The devices and classes are from different vendors, so they provide different APIs. Both classes have only one public method `watch`, which accepts a functional object:

[PRE34]

The `Device1::watch` and `Device2::watch` functions pass values to a functional object in a different order.

Some other libraries provide a function, which is used to detect storms, and throws an exception when the risk of a storm is high enough:

[PRE35]

Your task is to provide a storm-detecting function to both of the devices. Here is how it can be achieved using the `bind` function:

[PRE36]

The `Boost.Bind` library provides good performance because it does not use dynamic allocations and virtual functions. It is useful even when the C++11 lambda functions are not usable:

[PRE37]

Bind is a part of the C++11 standard. It is defined in the `<functional>` header and may slightly differ from the `Boost.Bind` implementation (however, it will be at least as effective as Boost's implementation).

## See also

*   The *Binding a value as a function parameter* recipe says more about the features of `Boost.Bind`
*   Boost's official documentation contains many more examples and descriptions of advanced features. It is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html](http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html)

# Binding a value as a function parameter

If you work with the STL library a lot and use the `<algorithm>` header, you will definitely write a lot of functional objects. You can construct them using a set of STL adapter functions such as `bind1st`, `bind2nd`, `ptr_fun`, `mem_fun`, and `mem_fun_ref`, or you can write them by hand (because adapter functions look scary). Here is some good news: `Boost.Bind` can be used instead of all of those functions and it provides a more human-readable syntax.

## Getting ready

Read the previous recipe to get an idea of placeholders, or just make sure that you are familiar with C++11 placeholders. Knowledge of STL functions and algorithms is welcomed.

## How to do it...

Let's see some examples of the usage of `Boost.Bind` along with traditional STL classes:

1.  Count values greater than or equal to 5 as shown in the following code:

    [PRE38]

2.  This is how we could count empty strings:

    [PRE39]

3.  Now let's count strings with a length less than `5`:

    [PRE40]

4.  Compare the strings:

    [PRE41]

## How it works...

The `boost::bind` function returns a functional object that stores a copy of the bound values and a copy of the original functional object. When the actual call to `operator()` is performed, the stored parameters are passed to the original functional object along with the parameters passed at the time of call.

## There's more...

Take a look at the previous examples. When we are binding values, we copy a value into a functional object. For some classes this operation is expensive. Is there a way to bypass copying?

Yes, there is! And the `Boost.Ref` library will help us here! It contains two functions, `boost::ref()` and `boost::cref()`, the first of which allows us to pass a parameter as a reference, and the second one passes the parameter as a constant reference. The `ref()` and `cref()` functions just construct an object of type `reference_wrapper<T>` or `reference_wrapper<const T>`, which is implicitly convertible to a reference type. Let's change our previous examples:

[PRE42]

Just one more example to show you how `boost::ref` can be used to concatenate strings:

[PRE43]

The functions `ref` and `cref` (and `bind`) are accepted to the C++11 standard and defined in the `<functional>` header in the `std::` namespace. None of these functions dynamically allocate memory in the heap and they do not use virtual functions. The objects returned by them are easy to optimize and they do not apply any optimization barriers for good compilers.

STL implementations of those functions may have additional optimizations to reduce compilation time or just compiler-specific optimizations, but unfortunately, some STL implementations miss the functionality of Boost versions. You may use the STL version of those functions with any Boost library, or even mix Boost and STL versions.

## See also

*   The `Boost.Bind` library is used widely across this book; see [Chapter 6](ch06.html "Chapter 6. Manipulating Tasks"), *Manipulating Tasks*, and [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, for more examples
*   The official documentation contains many more examples and a description of advanced features at [http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html](http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html)

# Using the C++11 move emulation

One of the greatest features of the C++11 standard is rvalue references. This feature allows us to modify temporary objects, "stealing" resources from them. As you can guess, the C++03 standard has no rvalue references, but using the `Boost.Move` library you can write some portable code that uses them, and even more, you actually get started with the emulation of move semantics.

## Getting ready

It is highly recommended to at least know the basics of C++11 rvalue references.

## How to do it...

Now, let's take a look at the following examples:

1.  Imagine that you have a class with multiple fields, some of which are STL containers.

    [PRE44]

2.  It is time to add the move assignment and move constructors to it! Just remember that in C++03, STL containers have neither move operators nor move constructors.
3.  The correct implementation of the move assignment is the same as `swap` and `clear` (if an empty state is allowed). The correct implementation of the move constructor is close to the default construct and `swap`. So, let's start with the `swap` member function:

    [PRE45]

4.  Now put the following macro in the `private` section:

    [PRE46]

5.  Write a copy constructor.
6.  Write a copy assignment, taking the parameter as `BOOST_COPY_ASSIGN_REF(classname)`.
7.  Write a move constructor and a move assignment, taking the parameter as `BOOST_RV_REF(classname)`:

    [PRE47]

8.  Now we have a portable, fast implementation of the move assignment and move construction operators of the `person_info` class.

## How it works...

Here is an example of how the move assignment can be used:

[PRE48]

The `Boost.Move` library is implemented in a very efficient way. When the C++11 compiler is used, all the macros for rvalues emulation will be expanded to C++11-specific features, otherwise (on C++03 compilers) rvalues will be emulated using specific datatypes and functions that never copy passed values nor called any dynamic memory allocations or virtual functions.

## There's more...

Have you noticed the `boost::swap` call? It is a really helpful utility function, which will first search for a `swap` function in the namespace of a variable (the namespace `other::`), and if there is no swap function for the `characteristics` class, it will use the STL implementation of swap.

## See also

*   More information about emulation implementation can be found on the Boost website and in the sources of the `Boost.Move` library at [http://www.boost.org/doc/libs/1_53_0/doc/html/move.html](http://www.boost.org/doc/libs/1_53_0/doc/html/move.html).
*   The `Boost.Utility` library is the one that contains `boost::utility`, and it has many useful functions and classes. Refer to its documentation at [http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm).
*   The *Initializing the base class by the member of the derived* recipe in [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*.
*   The *Making a noncopyable class* recipe.
*   In the *Making a noncopyable but movable class* recipe, there is more info about `Boost.Move` and some examples on how we can use the movable objects in containers in a portable and efficient way.

# Making a noncopyable class

You must have almost certainly encountered situations where providing a copy constructor and move assignment operator for a class will require too much work, or where a class owns some resources that must not be copied for technical reasons:

[PRE49]

The C++ compiler, in the case of the previous example, will generate a copy constructor and an assignment operator, so the potential user of the `descriptor_owner` class will be able to create the following awful things:

[PRE50]

## Getting ready

Only very basic knowledge of C++ is required for this recipe.

## How to do it...

To avoid such situations, the `boost::noncopyable` class was invented. If you derive your own class from it, the copy constructor and assignment operator won't be generated by the C++ compiler:

[PRE51]

Now the user won't be able to do bad things:

[PRE52]

## How it works...

A sophisticated reader will tell me that we can achieve exactly the same result by making a copy constructor and an assignment operator of `descriptor_owning_fixed` private, or just by defining them without actual implementation. Yes, you are correct. Moreover, this is the current implementation of the `boost::noncopyable` class. But `boost::noncopyable` also serves as good documentation for your class. It never raises questions such as "Is the copy constructor body defined elsewhere?" or "Does it have a nonstandard copy constructor (with a nonconst referenced parameter)?".

## See also

*   The *Making a noncopyable but movable class* recipe will give you ideas on how to allow unique ownership of a resource in C++03 by moving it
*   You may find a lot of helpful functions and classes in the `Boost.Utility` library's official documentation at [http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm)
*   The *Initializing the base class by the member of the derived* recipe in [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*
*   The *Using the C++11 move emulation* recipe

# Making a noncopyable but movable class

Now imagine the following situation: we have a resource that cannot be copied, which should be correctly freed in a destructor, and we want to return it from a function:

[PRE53]

Actually, you can work around such situations using the `swap` method:

[PRE54]

But such a workaround won't allow us to use `descriptor_owner` in STL or Boost containers. And by the way, it looks awful!

## Getting ready

It is highly recommended to know at least the basics of C++11 rvalue references. Reading the *Using the C++11 move emulation* recipe is also recommended.

## How to do it...

Those readers who use C++11 already know about the move-only classes (like `std::unique_ptr` or `std::thread`). Using such an approach, we can make a move-only `descriptor_owner` class:

[PRE55]

This will work only on C++11 compatible compilers. That is the right moment for `Boost.Move`! Let's modify our example so it can be used on C++03 compilers.

According to the documentation, to write a movable but noncopyable type in portable syntax, we need to follow these simple steps:

1.  Put the `BOOST_MOVABLE_BUT_NOT_COPYABLE(classname)` macro in the `private` section:

    [PRE56]

2.  Write a move constructor and a move assignment, taking the parameter as `BOOST_RV_REF(classname)`:

    [PRE57]

## How it works...

Now we have a movable but noncopyable class that can be used even on C++03 compilers and in `Boost.Containers`:

[PRE58]

But unfortunately, C++03 STL containers still won't be able to use it (that is why we used a vector from `Boost.Containers` in the previous example).

## There's more...

If you want to use `Boost.Containers` on C++03 compilers and STL containers, on C++11 compilers you can use the following simple trick. Add the header file to your project with the following content:

[PRE59]

Now you can include `<your_project/vector.hpp>` and use a vector from the namespace `your_project_namespace`:

[PRE60]

But beware of compiler- and STL-implementation-specific issues! For example, this code will compile on GCC 4.7 in C++11 mode only if you mark the move constructor, destructor, and move assignment operators with `noexcept`.

## See also

*   The *Reducing code size and increasing performance of user-defined types (UDTs) in C++11* recipe in [Chapter 10](ch10.html "Chapter 10. Gathering Platform and Compiler Information"), *Gathering Platform and Compiler Information*, for more info on `noexcept`
*   More information about `Boost.Move` can be found on Boost's website [http://www.boost.org/doc/libs/1_53_0/doc/html/move.html](http://www.boost.org/doc/libs/1_53_0/doc/html/move.html)