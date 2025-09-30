# Chapter 10. Gathering Platform and Compiler Information

In this chapter we will cover:

*   Detecting int128 support
*   Detecting RTTI support
*   Speeding up compilation using C++11 extern templates
*   Writing metafunctions using simpler methods
*   Reducing code size and increasing performance of user-defined types (UDTs) in C++11
*   The portable way to export and import functions and classes
*   Detecting the Boost version and getting latest features

# Introduction

Different projects and companies have different coding requirements. Some of them forbid exceptions or RTTI and some forbid C++11\. If you are willing to write portable code that can be used by a wide range of projects, this chapter is for you.

Want to make your code as fast as possible and use the latest C++ features? You'll definitely need a tool for detecting compiler features.

Some compilers have unique features that may greatly simplify your life. If you are targeting a single compiler, you can save many hours and use those features. No need to implement their analogues from scratch!

This chapter is devoted to different helper macros used to detect compiler, platform, and Boost features. Those macro are widely used across Boost libraries and are essential for writing portable code that is able to work with any compiler flags.

# Detecting int128 support

Some compilers have support for extended arithmetic types such as 128-bit floats or integers. Let's take a quick glance at how to use them using Boost. We'll be creating a method that accepts three parameters and returns the multiplied value of those methods.

## Getting ready

Only basic knowledge of C++ is required.

## How to do it...

What do we need to work with 128-bit integers? Macros that show that they are available and a few typedefs to have portable type names across platforms.

1.  We'll need only a single header:

    [PRE0]

2.  Now we need to detect int128 support:

    [PRE1]

3.  Add some typedefs and implement the method as follows:

    [PRE2]

4.  For compilers that do not support the int128 type, we may require support of the int64 type:

    [PRE3]

5.  Now we need to provide some implementation for compilers without int128 support using int64:

    [PRE4]

## How it works...

The header `<boost/config.hpp>` contains a lot of macros to describe compiler and platform features. In this example, we used `BOOST_HAS_INT128` to detect support of 128-bit integers and `BOOST_NO_LONG_LONG` to detect support of 64-bit integers.

As we may see from the example, Boost has typedefs for 64-bit signed and unsigned integers:

[PRE5]

It also has typedefs for 128-bit signed and unsigned integers:

[PRE6]

## There's more...

C++11 has support of 64-bit types via the `long long int` and `unsigned long long int` built-in types. Unfortunately, not all compilers support C++11, so `BOOST_NO_LONG_LONG` will be useful for you. 128-bit integers are not a part of C++11, so typedefs and macros from Boost are the only way to write portable code.

## See also

*   Read the recipe *Detecting RTTI support* for more information about `Boost.Config`.
*   Read the official documentation of `Boost.Config` for more information about its abilities at [http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html).
*   There is a library in Boost that allows constructing types of unlimited precision. Take a look at the `Boost.Multiprecision` library at [http://www.boost.org/doc/libs/1_53_0/libs/multiprecision/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/multiprecision/doc/html/index.html).

# Detecting RTTI support

Some companies and libraries have specific requirements for their C++ code, such as successful compilation without **Runtime type information** (**RTTI**). In this small recipe, we'll take a look at how we can detect disabled RTTI, how to store information about types, and compare types at runtime, even without `typeid`.

## Getting ready

Basic knowledge of C++ RTTI usage is required for this recipe.

## How to do it...

Detecting disabled RTTI, storing information about types, and comparing types at runtime are tricks that are widely used across Boost libraries. The examples are `Boost.Exception` and `Boost.Function`.

1.  To do this, we first need to include the following header:

    [PRE7]

2.  Let's first look at the situation where RTTI is enabled and the C++11 `std::type_index` class is available:

    [PRE8]

3.  Otherwise, we need to construct our own `type_index` class:

    [PRE9]

4.  The final step is to define the `type_id` function:

    [PRE10]

5.  Now we can compare types:

    [PRE11]

## How it works...

The macro `BOOST_NO_RTTI` will be defined if RTTI is disabled, and the macro `BOOST_NO_CXX11_HDR_TYPEINDEX` will be defined when the compiler has no `<typeindex>` header and no `std::type_index` class.

The handwritten `type_index` structure from step 3 of the previous section only holds the pointer to some string; nothing really interesting here.

Take a look at the `BOOST_CURRENT_FUNCTION` macro. It returns the full name of the current function, including template parameters, arguments, and the return type. For example, `type_id<double>()` will be represented as follows:

[PRE12]

So, for any other type, `BOOST_CURRENT_FUNCTION` will return a different string, and that's why the `type_index` variable from the example won't compare equal-to it.

## There's more...

Different compilers have different macros for getting the full function name and RTTI. Using macros from Boost is the most portable solution. The `BOOST_CURRENT_FUNCTION` macro returns the name at compile time, so it implies minimal runtime penalty.

## See also

*   Read the upcoming recipes for more information on `Boost.Config`
*   Browse to [https://github.com/apolukhin/type_index](https://github.com/apolukhin/type_index) and refer to the library there, which uses all the tricks from this recipe to implement `type_index`
*   Read the official documentation of `Boost.Config` at [http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)

# Speeding up compilation using C++11 extern templates

Remember some situations where you were using some complicated template class declared in the header file? Examples of such classes would be `boost::variant`, containers from `Boost.Container`, or `Boost.Spirit` parsers. When we use such classes or methods, they are usually compiled (instantiated) separately in each source file that is using them, and duplicates are thrown away during linking. On some compilers, that may lead to slow compilation speed.

If only there was some way to tell the compiler in which source file to instantiate it!

## Getting ready

Basic knowledge of templates is required for this recipe.

## How to do it...

This method is widely used in modern C++ standard libraries for compilers that do support it. For example, the STL library, which is shipped with GCC, uses this technique to instantiate `std::basic_string<char>` and `std::basic_fstream<char>`.

1.  To do it by ourselves, we need to include the following header:

    [PRE13]

2.  We also need to include a header file that contains a template class whose instantiation count we wish to reduce:

    [PRE14]

3.  The following is the code for compilers with support for C++11 extern templates:

    [PRE15]

4.  Now we need to add the following code to the source file where we wish the template to be instantiated:

    [PRE16]

## How it works...

The C++11 keyword `extern template` just tells the compiler not to instantiate the template without an explicit request to do that.

The code in step 4 is an explicit request to instantiate the template in this source file.

The `BOOST_NO_CXX11_EXTERN_TEMPLATE` macro is defined when the compiler has support of C++11 extern templates.

## There's more...

Extern templates do not affect the runtime performance of your program, but can significantly reduce the compilation time of some template classes. Do not overuse them; they are nearly useless for small template classes.

## See also

*   Read the other recipes of this chapter to get more information about `Boost.Config`
*   Read the official documentation of `Boost.Config` for information about macros that was not covered in this chapter, at [http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)

# Writing metafunctions using simpler methods

[Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, and [Chapter 8](ch08.html "Chapter 8. Metaprogramming"), *Metaprogramming*, were devoted to metaprogramming. If you were trying to use techniques from those chapters, you may have noticed that writing a metafunction can take a lot of time. So it may be a good idea to experiment with metafunctions using more user-friendly methods, such as C++11 `constexpr`, before writing a portable implementation.

In this recipe, we'll take a look at how to detect `constexpr` support.

## Getting ready

The `constexpr` functions are functions that can be evaluated at compile time. That is all we need to know for this recipe.

## How to do it...

Currently, not many compilers support the `constexpr` feature, so a good new compiler may be required for experiments. Let's see how we can detect compiler support for the `constexpr` feature:

1.  Just like in other recipes from this chapter, we start with the following header:

    [PRE17]

2.  Now we will work with `constexpr`:

    [PRE18]

3.  Let's print an error if C++11 features are missing:

    [PRE19]

4.  That's it; now we are free to write code such as the following:

    [PRE20]

## How it works...

The `BOOST_NO_CXX11_CONSTEXPR` macro is defined when C++11 `constexpr` is available.

The `constexpr` keyword tells the compiler that the function can be evaluated at compile time if all the inputs for that function are compile-time constants. C++11 imposes a lot of limitations on what a `constexpr` function can do. C++14 will remove some of the limitations.

The `BOOST_NO_CXX11_HDR_ARRAY` macro is defined when the C++11 `std::array` class and the `<array>` header are available.

## There's more...

However, there are other usable and interesting macros for `constexpr` too, as follows:

*   The `BOOST_CONSTEXPR` macro expands to `constexpr` or does not expand
*   The `BOOST_CONSTEXPR_OR_CONST` macro expands to `constexpr` or `const`
*   The `BOOST_STATIC_CONSTEXPR` macro is the same as `static BOOST_CONSTEXPR_OR_CONST`

Using those macros, it is possible to write code that takes advantage of C++11 constant expression features if they are available:

[PRE21]

Now, we can use `integral_constant` as shown in the following code:

[PRE22]

In the example, `BOOST_CONSTEXPR operator T()` will be called to get the array size.

The C++11 constant expressions may improve compilation speed and diagnostic information in case of error. It's a good feature to use.

## See also

*   More information about `constexpr` usage can be read at [http://en.cppreference.com/w/cpp/language/constexpr](http://en.cppreference.com/w/cpp/language/constexpr)
*   Read the official documentation of `Boost.Config` for more information about macros at [http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)

# Reducing code size and increasing performance of user-defined types (UDTs) in C++11

C++11 has very specific logic when **user-defined types** (**UDTs**) are used in STL containers. Containers will use move assignment and move construction only if the move constructor does not throw exceptions or there is no copy constructor.

Let's see how we can ensure the `move_nothrow` assignment operator and `move_nothrow` constructor of our type do not throw exceptions.

## Getting ready

Basic knowledge of C++11 rvalue references is required for this recipe. Knowledge of STL containers will also serve you well.

## How to do it...

Let's take a look at how we can improve our C++ classes using Boost.

1.  All we need to do is mark the `move_nothrow` assignment operator and `move_nothrow` constructor with the `BOOST_NOEXCEPT` macro:

    [PRE23]

2.  Now we may use the class with `std::vector` in C++11 without any modifications:

    [PRE24]

3.  If we remove `BOOST_NOEXCEPT` from the move constructor, we'll get the following error for GCC-4.7 and later compilers:

    [PRE25]

## How it works...

The `BOOST_NOEXCEPT` macro expands to `noexcept` on compilers that support it. The STL containers use type traits to detect if the constructor throws an exception or not. Type traits make their decision mainly based on `noexcept` specifiers.

Why do we get an error without `BOOST_NOEXCEPT`? GCC's type traits return the move constructor that `move_nothrow` throws, so `std::vector` will try to use the copy constructor of `move_nothrow`, which is not defined.

## There's more...

The `BOOST_NOEXCEPT` macro also reduces binary size irrespective of whether the definition of the `noexcept` function or method is in a separate source file or not.

[PRE26]

That's because in the latter case, the compiler knows that the function will not throw exceptions and so there is no need to generate code that handles them.

### Note

If a function marked as `noexcept` does throw an exception, your program will terminate without calling destructors for the constructed objects.

## See also

*   A document describing why move constructors are allowed to throw exceptions and how containers must move objects is available at [http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html)
*   Read the official documentation of `Boost.Config` for more examples of `noexcept` macros existing in Boost, at [http://www.boost.org/doc/libs/1_53_0/libs/conf](http://www.boost.org/doc/libs/1_53_0/libs/conf)[ig/doc/html/index.html](http://ig/doc/html/index.html)

# The portable way to export and import functions and classes

Almost all modern languages have the ability to make libraries, which is a collection of classes and methods that have a well-defined interface. C++ is no exception to this rule. We have two types of libraries: runtime (also called shared or dynamic load) and static. But writing libraries is not a trivial task in C++. Different platforms have different methods for describing which symbols must be exported from the shared library.

Let's have a look at how to manage symbol visibility in a portable way using Boost.

## Getting ready

Experience in creating dynamic and static libraries will be useful in this recipe.

## How to do it...

The code for this recipe consists of two parts. The first part is the library itself. The second part is the code that uses that library. Both parts use the same header, in which the library methods are declared. Managing symbol visibility in a portable way using Boost is simple and can be done using the following steps:

1.  In the header file, we'll need definitions from the following `include` header:

    [PRE27]

2.  The following code must also be added to the header file:

    [PRE28]

3.  Now all the declarations must use the `MY_LIBRARY_API` macro:

    [PRE29]

4.  Exceptions must be declared with `BOOST_SYMBOL_VISIBLE`, otherwise they can be caught only using `catch(...)` in the code that will use the library:

    [PRE30]

5.  Library source files must include the header file:

    [PRE31]

6.  Definitions of methods must also be in the source files of the library:

    [PRE32]

7.  Now we can use the library as shown in the following code:

    [PRE33]

## How it works...

All the work is done in step 2\. There we are defining the macro `MY_LIBRARY_API`, which will be applied to classes and methods that we wish to export from our library. In step 2, we check for `MY_LIBRARY_LINK_DYNAMIC`; if it is not defined, we are building a static library and there is no need to define `MY_LIBRARY_API`.

### Note

The developer must take care of `MY_LIBRARY_LINK_DYNAMIC`! It will not define itself. So we need to make our build system to define it, if we are making a dynamic library.

If `MY_LIBRARY_LINK_DYNAMIC` is defined, we are building a runtime library, and that's where the workarounds start. You, as the developer, must tell the compiler that we are now exporting these methods to the user. The user must tell the compiler that he/she is importing methods from the library. To have a single header file for both library import and export, we use the following code:

[PRE34]

When exporting the library (or, in other words, compiling it), we must define `MY_LIBRARY_COMPILATION`. This leads to `MY_LIBRARY_API` being defined to `BOOST_SYMBOL_EXPORT`. For example, see step 5, where we defined `MY_LIBRARY_COMPILATION` before including `my_library.hpp`. If `MY_LIBRARY_COMPILATION` is not defined, the header is included by the user, who doesn't know anything about that macro. And, if the header is included by the user, the symbols must be imported from the library.

The `BOOST_SYMBOL_VISIBLE` macro must be used only for those classes that are not exported and are used by RTTI. Examples of such classes are exceptions and classes being cast using `dynamic_cast`.

## There's more...

Some compilers export all the symbols by default but provide flags to disable such behavior. For example, GCC provides `-fvisibility=hidden`. It is highly recommended to use those flags because it leads to smaller binary size, faster loading of dynamic libraries, and better logical structuring of binary input. Some inter-procedural optimizations can perform better when fewer symbols are exported.

C++11 has generalized attributes that someday may be used to provide a portable way to work with visibilities, but until then we have to use macros from Boost.

## See also

*   Read this chapter from the beginning to get more examples of `Boost.Config` usage
*   Consider reading the official documentation of `Boost.Config` for the full list of the `Boost.Config` macro and their description at [http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.htm)

# Detecting the Boost version and getting latest features

Boost is being actively developed, so each release contains new features and libraries. Some people wish to have libraries that compile for different versions of Boost and also want to use some of the features of the new versions.

Let's take a look at the `boost::lexical_cast` change log. According to it, Boost 1.53 has a `lexical_cast(const CharType* chars, std::size_t count)` function overload. Our task for this recipe will be to use that function overload for new versions of Boost, and work around that missing function overload for older versions.

## Getting ready

Only basic knowledge of C++ and the `Boost.Lexical` library is required.

## How to do it...

Well, all we need to do is get a version of Boost and use it to write optimal code. This can be done as shown in the following steps:

1.  We need to include headers containing the Boost version and `boost::lexical_cast`:

    [PRE35]

2.  We will use the new feature of `Boost.LexicalCast` if it is available:

    [PRE36]

3.  Otherwise, we are required to copy data to `std::string` first:

    [PRE37]

4.  Now we can use the code as shown here:

    [PRE38]

## How it works...

The `BOOST_VERSION` macro contains the Boost version written in the following format: a single number for the major version, followed by three numbers for the minor version, and then two numbers for the patch level. For example, Boost 1.46.1 will contain the `104601` number in the `BOOST_VERSION` macro.

So, we will check the Boost version in step 2 and choose the correct implementation of the `to_int` function according to the abilities of `Boost.LexicalCast`.

## There's more...

Having a version macro is a common practice for big libraries. Some of the Boost libraries allow you to specify the version of the library to use; see `Boost.Thread` and its `BOOST_THREAD_VERSION` macro for an example.

## See also

*   Read the recipe *Creating an execution thread* in [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, for more information about `BOOST_THREAD_VERSION` and how it affects the `Boost.Thread` library, or read the documentation at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread/changes.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread/changes.html)
*   Read this chapter from the beginning or consider reading the official documentation of `Boost.Config` at [http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/config/doc/html/index.html)