# Chapter 8. Metaprogramming

In this chapter we will cover:

*   Using type "vector of types"
*   Manipulating a vector of types
*   Getting a function's result type at compile time
*   Making a higher-order metafunction
*   Evaluating metafunctions lazily
*   Converting all the tuple elements to strings
*   Splitting tuples

# Introduction

This chapter is devoted to some cool and hard to understand metaprogramming methods. These methods are not for everyday use, but they will be a real help in the development of generic libraries.

[Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, already covered the basics of metaprogramming. Reading it is recommended for better understanding. In this chapter we'll go deeper and see how multiple types can be packed into a single tuple like type. We'll make functions for manipulating collections of types, we'll see how types of compile-time collections can be changed, and how compile-time tricks can be mixed with runtime. All this is metaprogramming.

Fasten your seat belts and get ready, here we go!

# Using type "vector of types"

There are situations when it would be great to work with all the template parameters as if they were in a container. Imagine that we are writing something such as `Boost.Variant`:

[PRE0]

And the preceding code is where all the following interesting tasks start to happen:

*   How can we remove constant and volatile qualifiers from all the types?
*   How can we remove duplicate types?
*   How can we get the sizes of all the types?
*   How can we get the maximum size of the input parameters?

All these tasks can be easily solved using `Boost.MPL`.

## Getting ready

A basic knowledge of [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, is required for this recipe. Gather your courage before reading—there will be a lot of metaprogramming in this recipe.

## How to do it…

We have already seen how a type can be manipulated at compile time. Why can't we go further and combine multiple types in an array and perform operations for each element of that array?

1.  First of all, let's pack all the types in one of the `Boost.MPL` types containers:

    [PRE1]

2.  Let's make our example less abstract and see how it will work if we specify types:

    [PRE2]

3.  We can check everything at compile time. Let's assert that types is not empty:

    [PRE3]

4.  We can also check that, for example, the `non_defined` types is still at the index `4` position:

    [PRE4]

5.  And that the last type is still `std::string`:

    [PRE5]

6.  Now, when we are sure that types really contain all the types passed to our variant structure, we can do some transformations. We'll start with removing constant and volatile qualifiers:

    [PRE6]

7.  Now we remove the duplicate types:

    [PRE7]

8.  We can now check that the vector contains only `5` types:

    [PRE8]

9.  The next step is to compute sizes:

    [PRE9]

10.  The final step is getting the maximum size:

    [PRE10]

    We can assert that the maximum size of the type is equal to the declared size of the structure, which must be the largest one in our example:

    [PRE11]

## How it works...

The `boost::mpl::vector` class is a compile-time container that holds types. To be more precise, it is a type that holds types. We don't make instances of it; instead we are just using it in typedefs.

Unlike the STL containers, the `Boost.MPL` containers have no member methods. Instead, methods are declared in a separate header. So to use some methods we need to:

*   Include the correct header
*   Call that method, usually by specifying the container as the first parameter

Here is another example:

[PRE12]

These methods should be familiar to you. We have already seen metafunctions in [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*. By the way, we are also using some metafunctions (such as `boost::is_same`) from the familiar `Boost.TypeTraits` library.

So, in step 3, step 4, and step 5 we are just calling metafunctions for our container type.

The hardest part is coming up!

Remember, placeholders are widely used with the `boost::bind` and `Boost.Asio` libraries. `Boost.MPL` has them too and they are required for combining the metafunctions:

[PRE13]

Here, `boost::mpl::_1` is a placeholder and the whole expression means "for each type in types, do `boost::remove_cv<>::type` and push back that type to the resulting vector. Return the resulting vector via `::type`".

Let's move to step 7\. Here, we specify a comparison metafunction for `boost::mpl::unique` using the `boost::is_same<boost::mpl::_1, boost::mpl::_2>` template parameter, where `boost::mpl::_1` and `boost::mpl::_2` are placeholders. You may find it similar to `boost::bind(std::equal_to(), _1, _2)`, and the whole expression in step 7 is similar to the following pseudo code:

[PRE14]

There is something interesting, which is required for better understanding, in step 9\. In the preceding code `sizes_types` is not a vector of values, but rather a vector of integral constants—types representing numbers. The `sizes_types` typedef is actually the following type:

[PRE15]

The final step should be clear now. It just gets the maximum element from the `sizes_types` typedef.

### Note

We can use the `Boost.MPL` metafunctions in any place where typedefs are allowed.

## There's more...

The `Boost.MPL` library usage results in longer compilation time, but gives you the ability to do everything you want with types. It does not add runtime overhead and won't add even a single instruction to the binary. C++11 has no `Boost.MPL` classes, and `Boost.MPL` does not use features of C++11, such as the variadic templates. This makes the `Boost.MPL` compilation time longer on C++11 compilers, but makes it usable on C++03 compilers.

## See also

*   See [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks,* for the basics of metaprogramming
*   The *Manipulating a vector of types* recipe will give you even more information on metaprogramming and the `Boost.MPL` library
*   See the official `Boost.MPL` documentation for more examples and full references at [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/in](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/in)[dex.html](http://dex.html)

# Manipulating a vector of types

The task of this recipe will be to modify the content of one `boost::mpl::vector` function depending on the content of a second `boost::mpl::vector` function. We'll be calling the second vector as the vector of modifiers and each of those modifiers can have the following type:

[PRE16]

So where shall we start?

## Getting ready

A basic knowledge of `Boost.MPL` is required. Reading the *Using type "vector of types"* recipe and [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks,* may help.

## How to do it...

This recipe is similar to the previous one, but it also uses conditional compile-time statements. Get ready, it won't be easy!

1.  We shall start with headers:

    [PRE17]

2.  Now, let's put all the metaprogramming magic inside the structure, for simpler re-use:

    [PRE18]

3.  It is a good idea to check that the passed vectors have the same size:

    [PRE19]

4.  Now let's take care of modifying the metafunction:

    [PRE20]

5.  And the final step:

    [PRE21]

    We can now run some tests and make sure that our metafunction works correctly:

    [PRE22]

## How it works...

In step 3 we assert that the sizes are equal, but we do it in an unusual way. The `boost::mpl::size<Types>::type` metafunction actually returns the integral constant `struct boost::mpl::long_<4>`, so in a static assertion we actually compare two types, not two numbers. This can be rewritten in a more familiar way:

[PRE23]

### Note

Notice the `typename` keyword we use. Without it the compiler won't be able to decide if `::type` is actually a type or some variable. Previous recipes did not require it, because parameters for the metafunction were fully known at the point where we were using them. But in this recipe, the parameter for the metafunction is a template.

We'll take a look at step 5, before taking care of step 4\. In step 5, we provide the `Types`, `Modifiers`, and `binary_operator_t` parameters from step 4 to the `boost::mpl::transform` metafunction. This metafunction is rather simple—for each passed vector it takes an element and passes it to a third parameter—a binary metafunction. If we rewrite it in pseudo code, it will look like the following:

[PRE24]

Step 4 may make someone's head hurt. At this step we are writing a metafunction that will be called for each pair of types from the `Types` and `Modifiers` vectors (see the preceding pseudo code). As we already know, `boost::mpl::_2` and `boost::mpl::_1` are placeholders. In this recipe, `_1` is a placeholder for a type from the `Types` vector and `_2` is a placeholder for a type from the `Modifiers` vector.

So the whole metafunction works like this:

*   Compares the second parameter passed to it (via `_2`) with an `unsigned` type
*   If the types are equal, makes the first parameter passed to it (via `_1`) `unsigned` and returns that type
*   Otherwise, compares the second parameter passed to it (via `_2`) with a constant type
*   If the types are equal, makes the first parameter passed to it (via `_1`) constant and returns that type
*   Otherwise, returns the first parameter passed to it (via `_1`)

We need to be very careful while constructing this metafunction. Additional care should be taken so as to not call `::type` at the end of it:

[PRE25]

If we call `::type`, the compiler will attempt to evaluate the binary operator at this point and this will lead to a compilation error. In pseudo code, such an attempt would look like this:

[PRE26]

## There's more...

Working with metafunctions requires some practice. Even your humble servant cannot write some functions correctly at the first attempt (second and third attempts are also not good though). Do not be afraid to experiment!

The `Boost.MPL` library is not a part of C++11 and does not use C++11 features, but it can be used with C++11 variadic templates:

[PRE27]

As always, metafunctions won't add a single instruction to the resulting binary file and do not make performance worse. However, by using them you can make your code more tuned to a specific situation.

## See also

*   Read this chapter from the beginning to get more simple examples of `Boost.MPL` usage
*   See [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, especially the *Selecting an optimal operator for a template parameter* recipe, which contains code similar to the `binary_operator_t` metafunction
*   The official documentation for `Boost.MPL` has more examples and a full table of contents at [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html)

# Getting a function's result type at compile time

Many features were added to C++11 to simplify the metaprogramming. One such feature is the alternative function syntax. It allows deducing the result type of a template function. Here is an example:

[PRE28]

It allows us to write generic functions more easily and work in difficult situations:

[PRE29]

But Boost has a lot of functions like these and it does not require C++11 to work.

How is that possible and how can we make a C++03 version of the `my_function_cpp11` function?

## Getting ready

A basic knowledge of C++ and templates is required for this recipe.

## How to do it...

C++11 greatly simplifies metaprogramming. A lot of code must be written in C++03 to make something close to the alternative functions syntax.

1.  We'll need to include the following header:

    [PRE30]

2.  Now we need to make a metafunction in the `result_of` namespace for any types:

    [PRE31]

3.  And specialize it for types `s1`, and `s2`:

    [PRE32]

4.  Now we are ready to write the `my_function_cpp03` function:

    [PRE33]

    That's it! Now we can use this function almost like a C++11 one:

    [PRE34]

## How it works...

The main idea of this recipe is that we can make a special metafunction that will deduce the resulting type. Such a technique can be seen all through the Boost libraries, for example, in the `Boost.Variants` implementation of `boost::get<>` or in almost any function from `Boost.Fusion`.

Now, let's move through this step by step. The `result_of` namespace is just a kind of tradition, but you can use your own and it won't matter. The `boost::common_type<>` metafunction deduces a type common to several types, so we use it as a general case. We also added two template specializations of the `my_function_cpp03` structures for the `s1` and `s2` types.

### Note

The disadvantage of writing metafunctions in C++03 is that sometimes we are required to write a lot of code. Compare the amount of code for `my_function_cpp11` and `my_function_cpp03` including the `result_of` namespace to see the difference.

When the metafunction is ready, we can deduce the resulting type without C++11, so writing `my_function_cpp03` will be as easy as a pie:

[PRE35]

## There's more...

This technique does not add runtime overhead but it may slow down compilation a little bit. You can use it with C++11 compilers as well.

## See also

*   The recipes *Enabling the usage of templated functions for integral types*, *Disabling templated functions' usage for real types*, and *Selecting an optimal operator for a template parameter* from [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, will give you much more information about `Boost.TypeTraits` and metaprogramming.
*   Consider the official documentation of `Boost.Typetraits` for more information about ready metafunctions at [http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/type_traits/doc/html/index.html)

# Making a higher-order metafunction

Functions that accept other functions as an input parameter or functions that return other functions are called higher-order functions. For example, the following functions are higher-order:

[PRE36]

We have already seen higher-order metafunctions in the recipes *Using type "vector of types"* and *Manipulating a vector of types* from this chapter, where we used `boost::transform`.

In this recipe, we'll try to make our own higher-order metafunction named `coalesce`, which accepts two types and two metafunctions. The `coalesce` metafunction applies the first type-parameter to the first metafunction and compares the resulting type with the `boost::mpl::false_ type` metafunction. If the resulting type is the `boost::mpl::false_ type` metafunction, it returns the result of applying the second type-parameter to the second metafunction, otherwise, it returns the first result type:

[PRE37]

## Getting ready

This recipe (and chapter) is a tricky one. Reading this chapter from the beginning is highly recommended.

## How to do it...

The `Boost.MPL` metafunctions are actually structures, which can be easily passed as a template parameter. The hard part is to do it correctly.

1.  We'll need the following headers to write a higher-order metafunction:

    [PRE38]

2.  The next step is to evaluate our functions:

    [PRE39]

3.  Now we need to choose the correct result type:

    [PRE40]

    That's it! we have completed a higher-order metafunction! Now we can use it, just like that:

    [PRE41]

## How it works...

The main problem with writing the higher-order metafunctions is taking care of the placeholders. That's why we should not call `Func1<Param1>::type` directly. Instead, we shall use the `boost::apply` metafunction, which accepts one function and up to five parameters that will be passed to this function.

### Note

You can configure `boost::mpl::apply` to accept even more parameters, defining the `BOOST_MPL_LIMIT_METAFUNCTION_ARITY` macro to the required amount of parameters, for example, to 6.

## There's more...

C++11 has nothing close to the `Boost.MPL` library to apply a metafunction.

## See also

*   See the official documentation, especially the *Tutorial* section, for more information about `Boost.MPL` at [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html)

# Evaluating metafunctions lazily

Lazy evaluation means that the function won't be called until we really need its result. Knowledge of this recipe is highly recommended for writing good metafunctions. The importance of lazy evaluation will be shown in the following example.

Imagine that we are writing a metafunction that accepts a function, a parameter, and a condition. The resulting type of that function must be a `fallback` type if the condition is `false` otherwise the result will be as follows:

[PRE42]

And the preceding code is the place where we cannot live without lazy evaluation.

## Getting ready

Reading [Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, is highly recommended. However, a good knowledge of metaprogramming should be enough.

## How to do it...

We will see how this recipe is essential for writing good metafunctions:

1.  We'll need the following headers:

    [PRE43]

2.  The beginning of the function is simple:

    [PRE44]

3.  But we should be careful here:

    [PRE45]

4.  Additional care must be taken when evaluating an expression:

    [PRE46]

    That's it! Now we are free to use it like this:

    [PRE47]

## How it works...

The main idea of this recipe is that we should not execute the metafunction if the condition is `false`. Because when the condition is `false`, there is a chance that the metafunction for that type won't work:

[PRE48]

So, how do we evaluate a metafunction lazily?

The compiler won't look inside the metafunction if there is no access to the metafunction's internal types or values. In other words, the compiler will try to compile the metafunction when we try to get one of its members via `::`. This can be a call to `::type` or `::value`. That is what an incorrect version of `apply_if` looks like:

[PRE49]

This differs from our example, where we did not call `::type` at step 3 and implemented step 4 using `eval_if_c`, which calls `::type` only for one of its parameters. The `boost::mpl::eval_if_c` metafunction is implemented like this:

[PRE50]

Because `boost::mpl::eval_if_c` calls `::type` for a success condition and `fallback` may have no `::type`, we were required to wrap `fallback` into the `boost::mpl::identity`. `boost::mpl::identity` class. This class is a very simple but useful structure that returns its template parameter via a `::type` call:

[PRE51]

## There's more...

As we previously mentioned, C++11 has no classes of `Boost.MPL`, but we can use `std::common_type<T>` with a single argument just like `boost::mpl::identity<T>`.

Just as always, metafunctions do not add a single line to the output binary file. So you can use metafunctions as many times as you want. The more you do at compile-time, the less will remain for runtime.

## See also

*   The `boost::mpl::identity` type can be used to disable **Argument Dependent Lookup** (**ADL**) for template functions. See the sources of `boost::implicit_cast` in the `<boost/implicit_cast.hpp>` header.
*   Reading this chapter from the beginning and the official documentation for `Boost.MPL` may help: [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html)

# Converting all the tuple elements to strings

This recipe and the next one are devoted to a mix of compile time and runtime features. We'll be using the `Boost.Fusion` library to see what it can do.

Remember that we were talking about tuples and arrays in the first chapter. Now we want to write a single function that can stream elements of tuples and arrays to strings.

![Converting all the tuple elements to strings](img/4880OS_08_01.jpg)

## Getting ready

You should be aware of the `boost::tuple` and `boost::array` classes and of the `boost::lexical_cast` function.

## How to do it...

We already know almost all the functions and classes that will be used in this recipe. We just need to gather all of them together.

1.  We need to write a functor that converts any type to a string:

    [PRE52]

2.  And this is the tricky part of the code:

    [PRE53]

3.  That's all! Now we can convert anything we want to a string:

    [PRE54]

    The preceding example will output the following:

    [PRE55]

## How it works...

The main problem with the `stringize` function is that neither `boost::tuple` nor `std::pair` have `begin()` or `end()` methods, so we cannot call `std::for_each`. And this is where `Boost.Fusion` steps in.

The `Boost.Fusion` library contains lots of terrific algorithms that can manipulate structures at compile time.

The `boost::fusion::for_each` function iterates through elements in sequence and applies a functor to each of the elements.

Note that we have included:

[PRE56]

This is required because, by default, `Boost.Fusion` works only with its own classes.`Boost.Fusion` has its own tuple class, `boost::fusion::vector`, which is quite close to `boost::tuple`:

[PRE57]

But `boost::fusion::vector` is not as simple as `boost::tuple`. We'll see the difference in the *Splitting tuples* recipe.

## There's more...

There is one fundamental difference between `boost::fusion::for_each` and `std::for_each`. The `std::for_each` function contains a loop inside it and determinates at runtime, how many iterations will be done. However, `boost::fusion::for_each` knows the iteration count at compile time and fully unrolls the loop, generating the following code for `stringize(tup2)`:

[PRE58]

C++11 contains no `Boost.Fusion` classes. All the methods of `Boost.Fusion` are very effective. They do as much as possible at compile time and have some very advanced optimizations.

## See also

*   The *Splitting tuples* recipe will give more information about the true power of `Boost.Fusion`
*   The official documentation for `Boost.Fusion` contains some interesting examples and full references which can be found at [http://www.boost.org/doc/libs/1_53_0/libs/fusion/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/fusion/doc/html/index.html)

# Splitting tuples

This recipe will show a tiny piece of the `Boost.Fusion` library's abilities. We'll be splitting a single tuple into two tuples, one with arithmetic types and the other with all the other types.

![Splitting tuples](img/4880OS_08_02.jpg)

## Getting ready

This recipe requires knowledge of `Boost.MPL`, placeholders, and `Boost.Tuple`. Read the following recipes from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, *Combining multiple values into one* for more information about tuples and *Reordering parameters of a function* for information about placeholders. Reading this chapter from the beginning is recommended.

## How to do it...

This is possibly one of the hardest recipes in this chapter. Result types will be determined at compile time and values for those types will be filled at runtime.

1.  To implement that mix, we'll need the following headers:

    [PRE59]

2.  Now we are ready to make a function that returns non-arithmetic types:

    [PRE60]

3.  And a function that returns arithmetic types:

    [PRE61]

That's it! Now we are capable of doing the following tasks:

[PRE62]

## How it works...

The idea behind `Boost.Fusion` is that the compiler knows the structure layout at compile time and whatever the compiler knows at compile time, we can change at the same time. `Boost.Fusion` allows us to modify different sequences, add and remove fields, and change field types. This is what we did in step 2 and step 3; we removed the non-required fields from the tuple.

Now let's take a very close look at `get_arithmetics`. First of all its result type is deduced using the following construction:

[PRE63]

This should be familiar to us. We saw something like this in the *Getting a function's result type at compile time* recipe in this chapter. The `Boost.MPL` placeholder `boost::mpl::_1` should also be familiar.

Now let's move inside the function and we'll see the following code:

[PRE64]

Remember that the compiler knows all the types of `seq` at compile time. This means that `Boost.Fusion` can apply metafunctions to different elements of `seq` and get the metafunction results for them. This also means that `Boost.Fusion` will be capable of copying required fields from the old structure to the new one.

### Note

However, `Boost.Fusion` tries not to copy fields if at all possible.

The code in step 3 is very similar to the code in step 2, but it has a negated predicate for removing non-required types.

Our functions can be used with any type supported by `Boost.Fusion` and not just with `boost::fusion::vector`.

## There's more...

You can use `Boost.MPL` functions for the `Boost.Fusion` containers. You just need to include `#include <boost/fusion/include/mpl.hpp>`:

[PRE65]

### Note

We have used `boost::fusion::result_of::value_at_c` instead of `boost::fusion::result_of::at_c` because `boost::fusion::result_of::at_c` returns the exact type that will be used as a return type in the `boost::fusion::at_c` call, which is a reference. `boost::fusion::result_of::value_at_c` returns type without a reference.

The `Boost.Fusion` and `Boost.MPL` libraries are not a part of C++11\. `Boost.Fusion` is extremely fast. It has many optimizations. All the metafunctions that you use with it will be evaluated at compile time.

It is worth mentioning that we saw only a tiny part of the `Boost.Fusion` abilities. A separate book could be written about it.

## See also

*   Good tutorials and full documentation for `Boost.Fusion` is available at the Boost site [http://www.boost.org/doc/libs/1_53_0/libs/fusion/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/fusion/doc/html/index.html)
*   You may also wish to see the official documentation for `Boost.MPL` at [http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/mpl/doc/index.html)