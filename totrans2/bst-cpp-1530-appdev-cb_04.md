# Chapter 4. Compile-time Tricks

In this chapter we will cover:

*   Checking sizes at compile time
*   Enabling the usage of templated functions for integral types
*   Disabling templated functions' usage for real types
*   Creating a type from number
*   Implementing a type trait
*   Selecting an optimal operator for a template parameter
*   Getting a type of expression in C++03

# Introduction

In this chapter we'll see some basic examples on how the Boost libraries can be used in compile-time checking, for tuning algorithms, and in other metaprogramming tasks.

Some readers may ask, "Why shall we care about compile-time things?". That's because the released version of the program is compiled once, and runs multiple times. The more we do at compile time, the less work remains for runtime, resulting in much faster and reliable programs. Runtime checks are executed only if a part of the code with check is executed. Compile-time checks won't give you to compile a program with error.

This chapter is possibly one of the most important. Understanding Boost sources and other Boost-like libraries is impossible without it.

# Checking sizes at compile time

Let's imagine that we are writing some serialization function that stores values in buffer of a specified size:

[PRE0]

This code has the following troubles:

*   The size of the buffer is not checked, so it may overflow
*   This function can be used with non-plain old data (POD) types, which would lead to incorrect behavior

We may partially fix it by adding some asserts, for example:

[PRE1]

But, this is a bad solution. The `BufSizeV` and `sizeof(value)` values are known at compile time, so we can potentially make this code to fail compilation if the buffer is too small, instead of having a runtime assert (which may not trigger during debug, if function was not called, and may even be optimized out in release mode, so very bad things may happen).

## Getting ready

This recipe requires some knowledge of C++ templates and the `Boost.Array` library.

## How to do it...

Let's use the `Boost.StaticAssert` and `Boost.TypeTraits` libraries to correct the solutions, and the output will be as follows:

[PRE2]

## How it works...

The `BOOST_STATIC_ASSERT` macro can be used only if an assert expression can be evaluated at compile time and implicitly convertible to `bool`. It means that you may only use `sizeof()`, static constants, and other constant expressions in it. If assert expression will evaluate to `false`, `BOOST_STATIC_ASSERT` will stop our program compilation. In case of `serialization()` function, if first static assertion fails, it means that someone used that function for a very small buffer and that code must be fixed by the programmer. The C++11 standard has a `static_assert` keyword that is equivalent to Boost's version.

Here are some more examples:

[PRE3]

### Note

If the `BOOST_STATIC_ASSERT` macro's assert expression has a comma sign in it, we must wrap the whole expression in additional brackets.

The last example is very close to what we can see on the second line of the `serialize()` function. So now it is time to know more about the `Boost.TypeTraits` library. This library provides a large number of compile-time metafunctions that allow us to get information about types and modify types. The metafunctions usages look like `boost::function_name<parameters>::value` or `boost::function_name<parameters>::type`. The metafunction `boost::is_pod<T>::value` will return `true`, only if `T` is a POD type.

Let's take a look at some more examples:

[PRE4]

### Note

Some compilers may compile this code even without the `typename` keyword, but such behavior violates the C++ standard, so it is highly recommended to write `typename`.

## There's more...

The `BOOST_STATIC_ASSSERT` macro has a more verbose variant called `BOOST_STATIC_ASSSERT_MSG` that will output an error message in the compiler log (or in the IDE window) if assertion fails. Take a look at the following code:

[PRE5]

The preceding code will give the following result during compilation on the g++ compiler in the C++11 mode:

[PRE6]

Neither `BOOST_STATIC_ASSSERT`, nor `BOOST_STATIC_ASSSERT_MSG`, nor any of the type traits library imply runtime penalty. All those functions are executed at compile time, and won't add a single assembly instruction in binary file.

The `Boost.TypeTraits` library was partially accepted into the C++11 standard; you may thus find traits in the `<type_traits>` header in the `std::` namespace. C++11 `<type_traits>` has some functions that do not exist in `Boost.TypeTraits`, but some metafunctions exist only in Boost. When there is a similar function in Boost and STL, the STL version (in rare cases) may work slightly better because of compiler-specific intrinsics usage.

As we have already mentioned earlier, the `BOOST_STATIC_ASSERT_MSG` macro was also accepted into C++11 (and even into C11) as the keyword `static_assert(expression, message)`.

Use the Boost version of those libraries if you need portability across compilers or metafunctions that does not exist in STLs `<type_traits>`.

## See also

*   The next recipes in this chapter will give you more examples and ideas on how static asserts and type traits may be used
*   Read the official documentation of `Boost.StaticAssert` for more examples at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_sta](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_sta)[ticassert.html](http://ticassert.html)

# Enabling the usage of templated functions for integral types

It's a common situation, when we have a templated class that implements some functionality. Have a look at the following code snippet:

[PRE7]

After execution of the preceding code, we have additional two optimized versions of that class, one for integral, and another for real types:

[PRE8]

Now the question, how to make the compiler to automatically choose the correct class for a specified type, arises.

## Getting ready

This recipe requires the knowledge of C++ templates.

## How to do it...

We'll be using `Boost.Utility` and `Boost.TypeTraits` to resolve this problem:

1.  Let's start with including headers:

    [PRE9]

2.  Let's add an additional template parameter with default value to our generic implementation:

    [PRE10]

3.  Modify optimized versions in the following way, so that now they will be treated by the compiler as template partial specializations:

    [PRE11]

4.  And, that's it! Now the compiler will automatically choose the correct class:

    [PRE12]

## How it works...

The `boost::enable_if_c` template is a tricky one. It makes use of the **SFINAE** (**Substitution Failure Is Not An Error**) principle, which is used during template instantiation. Here is how the principle works: if an invalid argument or return type is formed during the instantiation of a function or class template, the instantiation is removed from the overload resolution set and does not cause a compilation error. Now let's get back to the solution, and we'll see how it works with different types passed to the `data_processor` class as the `T` parameter.

If we pass an `int` as `T` type, first the compiler will try to instantiate template partial specializations, before using our nonspecialized (generic) version. When it tries to instantiate a `float` version, the `boost::is_float<T>::value` metafunction will return `false`. The `boost::enable_if_c<false>::type` metafunction cannot be correctly instantiated (because `boost::enable_if_c<false>` has no `::type`), and that is the place where SFINAE will act. Because class template cannot be instantiated, and this must be interpreted as not an error, compiler will skip this template specialization. Next, partial specialization is the one that is optimized for integral types. The `boost::is_integral<T>::value` metafunction will return `true`, and `boost::enable_if_c<true>::type` can be instantiated, which makes it possible to instantiate the whole `data_processor` specialization. The compiler found a matching partial specialization, so it does not need to try to instantiate the nonspecialized method.

Now, let's try to pass some nonarithmetic type (for example, `const char *`), and let's see what the compiler will do. First the compiler will try to instantiate template partial specializations. The specializations with `is_float<T>::value` and `is_integral<T>::value` will fail to instantiate, so the compiler will try to instantiate our generic version, and will succeed.

Without `boost::enable_if_c<>`, all the partial specialized versions may be instantiated at the same time for any type, which leads to ambiguity and failed compilation.

### Note

If you are using templates and compiler reports that cannot choose between two template classes of methods, you probably need `boost::enable_if_c<>`.

## There's more...

Another version of this method is called `boost::enable_if` (without `_c` at the end). Difference between them is that `enable_if_c` accepts constant as a template parameter; however, the short version accepts an object that has a `value` static member. For example, `boost::enable_if_c<boost::is_integral<T>::value >::type` is equal to `boost::enable_if<boost::is_integral<T> >::type>`.

C++11 has an `std::enable_if` defined in the `<type_traits>` header, which behaves exactly like `boost::enable_if_c`. No difference between them exists, except that Boost's version will work on non C++11 compilers too, providing better portability.

All the enabling functions are executed only at compile time and do not add a performance overhead at runtime. However, adding an additional template parameter may produce a bigger class name in `typeid(T).name()`, and add an extremely tiny performance overhead when comparing two `typeid()` results on some platforms.

## See also

*   Next recipes will give you more examples on `enable_if` usage.
*   You may also consult the official documentation of `Boost.Utility`. It contains many examples and a lot of useful classes (which are used widely in this book). Read about it at [http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm).
*   You may also read some articles about template partial specializations at [http://msdn.microsoft.com/en-us/library/3967w96f%28v=vs.110%29.aspx](http://msdn.microsoft.com/en-us/library/3967w96f%28v=vs.110%29.aspx).

# Disabling templated functions' usage for real types

We continue working with Boost metaprogramming libraries. In the previous recipe, we saw how to use `enable_if_c` with classes, now it is time to take a look at its usage in template functions. Consider the following example.

Initially, we had a template function that works with all the available types:

[PRE13]

Now that we write code using `process_data` function, we use an optimized `process_data` version for types that do have an `operator +=` function:

[PRE14]

But, we do not want to change the already written code; instead whenever it is possible, we want to force the compiler to automatically use optimized function in place of the default one.

## Getting ready

Read the previous recipe to get an idea of what `boost::enable_if_c` does, and for understanding the concept of SFINAE. However, the knowledge of templates is still required.

## How to do it...

Template magic can be done using the Boost libraries. Let's see how to do it:

1.  We will need the `boost::has_plus_assign<T>` metafunction and the `<boost/enable_if.hpp>` header:

    [PRE15]

2.  Now we will disable default implementation for types with plus assign operator:

    [PRE16]

3.  Enable optimized version for types with plus assign operator:

    [PRE17]

4.  Now, users won't feel the difference, but the optimized version will be used wherever possible:

    [PRE18]

## How it works...

The `boost::disable_if_c<bool_value>::type` metafunction disables method, if `bool_value` equals to `true` (works just like `boost::enable_if_c<!bool_value>::type`).

If we pass a class as the second parameter for `boost::enable_if_c` or `boost::disable_if_c`, it will be returned via `::type` in case of successful evaluation.

Let's go through the instantiation of templates step-by-step. If we pass `int` as `T` type, first the compiler will search for function overload with required signature. Because there is no such function, the next step will be to instantiate a template version of this function. For example, the compiler started from our second (optimized) version; in that case, it will successfully evaluate the `typename boost::enable_if_c<boost::has_plus_assign<T>::value, T>::type` expression, and will get the `T` return type. But, the compiler won't stop; it will continue instantiation attempts. It'll try to instantiate our first version of function, but will get a failure during evaluation of `typename boost::disable_if_c<boost::has_plus_assign<T>::value`. This failure won't be treated as an error (refer SFINAE). As you can see, without `enable_if_c` and `disable_if_c`, there will be ambiguity.

## There's more...

As in case of `enable_if_c` and `enable_if`, there is a `disable_if` version of the disabling function:

[PRE19]

C++11 has neither `disable_if_c`, nor `disable_if` (you may use `std::enable_if<!bool_value>::type` instead).

As it was mentioned in the previous recipe, all the enabling and disabling functions are executed only at compile time, and do not add performance overhead at runtime.

## See also

*   Read this chapter from the beginning to get more examples of compile-time tricks.
*   Consider reading the `Boost.TypeTraits` official documentation for more examples and full list of metafunctions at [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html).
*   The `Boost.Utility` library may provide you more examples of `boost::enable_if` usage. Read about it at [http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm](http://www.boost.org/doc/libs/1_53_0/libs/utility/utility.htm).

# Creating a type from number

We have now seen examples of how we can choose between functions without `boost::enable_if_c` usage. Let's consider the following example, where we have a generic method for processing POD datatypes:

[PRE20]

And, we have the same function optimized for sizes 1, 4, and 8 bytes. How do we rewrite process function, so that it can dispatch calls to optimized versions?

## Getting ready

Reading at least the first recipe from this chapter is highly recommended, so that you will not be confused by all the things that are happening here. Templates and metaprogramming shall not scare you (or just get ready to see a lot of them).

## How to do it...

We are going to see how the size of a template type can be converted to a variable of some type, and how that variable can be used for deducing the right function overload.

1.  Let's define our generic and optimized versions of `process_impl` function:

    [PRE21]

2.  Now we are ready to write process function:

    [PRE22]

## How it works...

The most interesting part here is `boost::mpl::int_<sizeof(T)>(). sizeof(T)` executes at compile time, so its output can be used as a template parameter. The class `boost::mpl::int_<>` is just an empty class that holds a compile-time value of integral type (in the `Boost.MPL` library, such classes are called Integral Constants). It can be implemented as shown in the following code:

[PRE23]

We need an instance of this class, that is why we have a round parentheses at the end of `boost::mpl::int_<sizeof(T)>()`.

Now, let's take a closer look at how the compiler will decide which `process_impl` function to use. First of all, the compiler will try to match functions that have a second parameter and not a template. If `sizeof(T)` is 4, the compiler will try to search the function with signatures like `process_impl(T, boost::mpl::int_<8>)`, and will find our 4 bytes optimized version from the `detail` namespace. If `sizeof(T)` is 34, compiler won't find the function with signature like `process_impl(T, boost::mpl::int_<34>)`,and will use a templated variant `process_impl(const T& val, Tag /*ignore*/)`.

## There's more...

The `Boost.MPL` library has several data structures for metaprogramming. In this recipe, we only scratched a top of the iceberg. You may find the following Integral Constant classes from MPL useful:

*   `bool_`
*   `int_`
*   `long_`
*   `size_t`
*   `char_`

All the `Boost.MPL` functions (except the `for_each` runtime function) are executed at compile time and won't add runtime overhead. The `Boost.MPL` library is not a part of C++11, but many STL libraries implement functions from it for their own needs.

## See also

*   The recipes from [Chapter 8](ch08.html "Chapter 8. Metaprogramming"), *Metaprogramming*, will give you more examples of the `Boost.MPL` library usage. If you feel confident, you may also try to read its documentation at [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html).
*   Read more examples of tags usage at [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/fill.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/fill.html) and [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/copy.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/boost_typetraits/examples/copy.html).

# Implementing a type trait

We need to implement a type trait that returns true if the `std::vector` type is passed to it as a template parameter.

## Getting ready

Some basic knowledge of the `Boost.TypeTrait` or STL type traits is required.

## How to do it...

Let's see how to implement a type trait:

[PRE24]

## How it works...

Almost all the work is done by the `boost::true_type` and `boost::false_type` classes. The `boost::true_type` class has a boolean `::value` static constant in it that equals to `true`, the `boost::false_type` class has a boolean `::value` static constant in it that equals to `false`. They also have some typedefs, and are usually derived from `boost::mpl::integral_c`, which makes it easy to use types derived from `true_type/false_type` with `Boost.MPL`.

Our first `is_stdvector` structure is a generic structure that will be used always when template specialized version of such structure is not found. Our second `is_stdvector` structure is a template specialization for the `std::vector` types (note that it is derived from `true_type`!). So, when we pass vector type to the `is_stdvector` structure, template specialized version will be used, otherwise generic version will be used, which is derived from `false_type`.

### Note

3 lines There is no public keyword before `boost::false_type` and `boost::true_type` in our trait because we use `struct` keyword, and by default it uses public inheritance.

## There's more...

Those readers who use the C++11 compatible compilers may use the `true_type` and `false_type` types declared in the `<type_traits>` header from the `std::` namespace for creating their own type traits.

As usual, the Boost version is more portable because it can be used on C++03 compilers.

## See also

*   Almost all the recipes from this chapter use type traits. Refer to the `Boost.TypeTraits` documentation for more examples and information at [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/i](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/i)[ndex.html](http://ndex.html).

# Selecting an optimal operator for a template parameter

Imagine that we are working with classes from different vendors that implement different amounts of arithmetic operations and have constructors from integers. And, we do want to make a function that increments by one when any class is passed to it. Also, we want this function to be effective! Take a look at the following code:

[PRE25]

## Getting ready

Some basic knowledge of the C++ templates, and the `Boost.TypeTrait` or STL type traits is required.

## How to do it...

All the selecting can be done at compile time. This can be achieved using the `Boost.TypeTraits` library, as shown in the following steps:

1.  Let's start from making correct functional objects:

    [PRE26]

2.  After that we will need a bunch of type traits:

    [PRE27]

3.  And, we are ready to deduce correct functor and use it:

    [PRE28]

## How it works...

All the magic is done via the `conditional<bool Condition, class T1, class T2>` metafunction. When this metafunction accepts `true` as a first parameter, it returns `T1` via the `::type` typedef. When the `boost::conditional` metafunction accepts `false` as a first parameter, it returns `T2` via the `::type` typedef. It acts like some kind of compile-time `if` statement.

So, `step0_t` holds a `detail::plus_functor` metafunction and `step1_t` will hold `step0_t` or `detail::plus_assignable_functor`. The `step2_t` type will hold `step1_t` or `detail::post_inc_functor`. The `step3_t` type will hold `step2_t` or `detail::pre_inc_functor`. What each `step*_t` typedef holds is deduced using type trait.

## There's more...

There is a C++11 version of this function, which can be found in the `<type_traits>` header in the `std::` namespace. Boost has multiple versions of this function in different libraries, for example, `Boost.MPL` has function `boost::mpl::if_c`, which acts exactly like `boost::conditional`. It also has a version `boost::mpl::if_` (without `c` at the end), which will call `::type` for its first template argument; and if it is derived from `boost::true_type` (or is a `boost::true_type` type), it will return its second argument during the `::type` call, otherwise it will return the last template parameter. We can rewrite our `inc()` function to use `Boost.MPL`, as shown in the following code:

[PRE29]

## See also

*   The recipe *Enabling the usage of templated functions for integral types*
*   The recipe *Disabling templated functions' usage for real types*
*   The `Boost.TypeTraits` documentation has a full list of available metafunctions. Read about it at [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html).
*   The recipes from [Chapter 8](ch08.html "Chapter 8. Metaprogramming"), *Metaprogramming*, will give you more examples of the `Boost.MPL` library usage. If you feel confident, you may also try to read its documentation at [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html).
*   There is a proposal to add type switch to C++, and you may find it interesting. Read about it at [http://www.stroustrup.com/OOPSLA-ty](http://www.stroustrup.com/OOPSLA-ty)[peswitch-draft.pdf](http://peswitch-draft.pdf).

# Getting a type of expression in C++03

In the previous recipes, we saw some examples on `boost::bind` usage. It is a good and useful tool with a small drawback; it is hard to store `boost::bind` metafunction's functor as a variable in C++03.

[PRE30]

In C++11, we can use `auto` keyword instead of `???`, and that will work. Is there a way to do it in C++03?

## Getting ready

The knowledge of the C++11 `auto` and `decltype` keywords may help you to understand this recipe.

## How to do it...

We will need a `Boost.Typeof` library for getting return type of expression:

[PRE31]

## How it works...

It just creates a variable with the name `var`, and the value of the expression is passed as a second argument. Type of `var` is detected from the type of expression.

## There's more...

An experienced C++11 reader will note that there are more keywords in the new standard for detecting the types of expression. Maybe `Boost.Typeof` has macro for them too. Let's take a look at the following C++11 code:

[PRE32]

Using `Boost.Typeof`, the preceding code can be written like the following code:

[PRE33]

C++11 version's `decltype(expr)` deduces and returns the type of `expr`.

[PRE34]

Using `Boost.Typeof`, the preceding code can be written like the following code:

[PRE35]

### Note

C++11 has a special syntax for specifying return type at the end of the function declaration. Unfortunately, this cannot be emulated in C++03, so we cannot use `t1` and `t2` variables in macro.

You can freely use the results of the `BOOST_TYPEOF()` functions in templates and in any other compile-time expressions:

[PRE36]

But unfortunately, this magic does not always work without help. For example, user-defined classes are not always detected, so the following code may fail on some compilers:

[PRE37]

In such situations, you may give `Boost.Typeof` a helping hand and register a template:

[PRE38]

However, three most popular compilers correctly detected type even without `BOOST_TYPEOF_REGISTER_TEMPLATE` and without C++11.

## See also

*   The official documentation of `Boost.Typeof` has more examples. Read about it at [http://www.boost.org/doc/libs/1_53_0/doc/html/typeof.html](http://www.boost.org/doc/libs/1_53_0/doc/html/typeof.html).
*   *Bjarne Stroustrup* may introduce some of the C++11 features to you. Read about it at [http://www.stroustrup.com/C++11FAQ.html](http://www.stroustrup.com/C++11FAQ.html).